import cv2
import torch
import onnxruntime
import numpy as np
from torchvision.transforms import ToTensor

from src.config import load_config
from schedule.schedule import Schedule
from schedule.diffusionSample import GaussianDiffusion

class Inferer():
    def __init__(self, config='conf.yml'):
        config = load_config(config)

        self.pre_ori = config.PRE_ORI
        self.device = torch.device(f"cuda:{config.CUDA_VISIBLE_DEVICES}" if torch.cuda.is_available() else "cpu")

        providers = ['CUDAExecutionProvider']
        self.init_predictor = onnxruntime.InferenceSession(config.ONNX_TEST_INITIAL_PREDICTOR_WEIGHT_PATH, providers=providers)
        denoiser = onnxruntime.InferenceSession(config.ONNX_TEST_DENOISER_WEIGHT_PATH, providers=providers)
        
        schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        diffusion = GaussianDiffusion(denoiser, config.TIMESTEPS, schedule).to(self.device)
        self.sampler = diffusion

    @staticmethod     
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    @staticmethod
    def from_numpy(x, device):
        return torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x

    @staticmethod
    def transform_image(x):
            x = ToTensor()(x)
            x = x[None, :, :, :]
            return x

    @staticmethod
    def crop_concat(img, size=128):
                shape = img.shape
                correct_shape = (size*(shape[2]//size+1), size*(shape[3]//size+1))
                one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]))
                one[:, :, :shape[2], :shape[3]] = img

                for i in range(shape[2]//size+1):
                    for j in range(shape[3]//size+1):
                        if i == 0 and j == 0:
                            crop = one[:, :, i*size:(i+1)*size, j*size:(j+1)*size]
                        else:
                            crop = torch.cat((crop, one[:, :, i*size:(i+1)*size, j*size:(j+1)*size]), dim=0)
                return crop

    @staticmethod
    def crop_concat_back(img, prediction, size=128):
            shape = img.shape
            for i in range(shape[2]//size+1):
                for j in range(shape[3]//size+1):
                    if j == 0:
                        crop = prediction[(i*(shape[3]//size+1)+j)*shape[0]:(i*(shape[3]//size+1)+j+1)*shape[0], :, :, :]
                    else:
                        crop = torch.cat((crop, prediction[(i*(shape[3]//size+1)+j)*shape[0]:(i*(shape[3]//size+1)+j+1)*shape[0], :, :, :]), dim=3)
                if i == 0:
                    crop_concat = crop
                else:
                    crop_concat = torch.cat((crop_concat, crop), dim=2)
            return crop_concat[:, :, :shape[2], :shape[3]]
        
    def infer(self, img_path):
        img = cv2.imread(img_path)
        img = self.transform_image(img)
        
        temp = img
        img = self.crop_concat(img)
        
        ort_inputs = {self.init_predictor.get_inputs()[0].name: self.to_numpy(img),}
        init_predict = self.init_predictor.run(None, ort_inputs)[0]
        init_predict = self.from_numpy(init_predict, self.device)

        noisyImage = torch.randn_like(img).to(self.device)
        sampledImgs = self.sampler(noisyImage, init_predict, self.pre_ori)

        finalImgs = (sampledImgs + init_predict)
        finalImgs = self.crop_concat_back(temp, finalImgs)
        finalImgs = finalImgs.cpu()[0]
        finalImgs = finalImgs.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return finalImgs

if __name__ == '__main__':
    inferer = Inferer()

    img_path = 'demo/input.png'
    output = inferer.infer(img_path)
    cv2.imwrite('demo/onnx_output.png', output)