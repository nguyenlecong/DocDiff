import torch

from src.trainer import Trainer
from src.config import load_config


class Converter():
    def __init__(self, config='conf.yml'):
        # Model definition 
        config = load_config(config)
        trainer = Trainer(config)
        network = trainer.network
        self.device = trainer.device

        # Initialize model with the pretrained weights
        network.init_predictor.load_state_dict(torch.load(trainer.TEST_INITIAL_PREDICTOR_WEIGHT_PATH))
        network.denoiser.load_state_dict(torch.load(trainer.TEST_DENOISER_WEIGHT_PATH))
        
        # Set the model to inference mode
        network.eval()

        self.init_predictor = network.init_predictor
        self.denoiser = network.denoiser

    def convert(self, torch_model, save_path, channels):
        # Input to the model
        x = torch.randn(6, channels, 128, 128, requires_grad=True)
        
        # Export the model
        torch.onnx.export(torch_model,       # model being run
                  x.to(self.device),                         # model input (or a tuple for multiple inputs)
                  f"{save_path}.onnx",       # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input', 'timestep'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
        
        print(f'Done - {save_path}')
    
    def run(self):
        self.convert(self.init_predictor, 'checksave_onnx/model_init_best', channels=3)
        self.convert(self.denoiser, 'checksave_onnx/model_denoiser_best', channels=6)

if __name__ == '__main__':
    converter = Converter()
    converter.run()