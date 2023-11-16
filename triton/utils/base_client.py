import sys
import cv2
import numpy as np
from .utils import read_config
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

"""
Base client to process image via Trition Server
"""


class BaseClient():

    def __init__(self, config_file):
        self.config = read_config(config_file)

    def __call__(self, ):
        raise NotImplementedError

    def preprocess(self, images):
        return NotImplementedError

    def postprocess(self, raw_results):
        return NotImplementedError

    def vis(self, img, bboxes, scores, clas):
        for i, cls in enumerate(clas.tolist()):
            if cls == 0:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cla_name = self.class_map[cls]
            bbox = bboxes[i].astype(np.int32).tolist()
            scr = scores[i]
            x1, y1, x2, y2 = bbox

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(img, "{}: {:.03f}".format(cla_name, scr), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=1)

        return img
    
    def infer(self, inputs):
        """
        Process inputs via Triton Server
        """
        triton_client = grpcclient.InferenceServerClient(
            url=self.config.server.url,
            verbose=self.config.server.verbose,
            ssl=self.config.server.ssl,
            root_certificates=self.config.server.root_certificates,
            private_key=self.config.server.private_key,
            certificate_chain=self.config.server.certificate_chain
        )

        buffer_inputs = []
        for name, value in inputs.items():
            buffer_inputs.append(
                grpcclient.InferInput(
                    name,
                    value.shape,
                    'INT64' if name=='timestep' else 'FP32'
                )
            )
            buffer_inputs[-1].set_data_from_numpy(value)

        buffer_outputs = []
        output_names = self.verify_model(inputs, triton_client)
        for name in output_names:
            buffer_outputs.append(grpcclient.InferRequestedOutput(name))

        # get infer result
        results = triton_client.infer(
            model_name=self.config.model.name,
            inputs=buffer_inputs,
            outputs=buffer_outputs,
            client_timeout=self.config.server.client_timeout,
            headers={'test': '1'}
        )

        outputs = {}
        for name in output_names:
            outputs[name] = results.as_numpy(name)
        return outputs

    def verify_model(self, inputs, triton_client):
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=self.config.model.name,
                model_version=self.config.model.version
            )
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        # model_inputs = model_metadata.inputs
        # for i in range(len(model_inputs)):
        #     input_name = model_inputs[i].name
        #     print(type(model_inputs[i].shape))
        #     if inputs[input_name].shape != set(model_inputs[i].shape):
        #         raise Exception('input {} expects shape {}, but receives shape {}'\
        #             .format(input_name, inputs[input_name].shape, model_inputs[i].shape))

        output_names = [output.name for output in model_metadata.outputs]
        return output_names
