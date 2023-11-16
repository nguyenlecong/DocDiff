import torch
import numpy as np
np.bool = np.bool_

from torchvision.transforms import ToTensor

from .utils.utils import read_config
from .utils.base_client import BaseClient

from schedule.schedule import Schedule
from schedule.diffusionSample import GaussianDiffusion


class Processor():
    def __init__(self, config_file, init_predictor_config_file, denoiser_config_file):
        config = read_config(config_file)

        self.pre_ori = config.PRE_ORI

        device = config.CUDA_VISIBLE_DEVICES
        if device.isdigit() and torch.cuda.is_available():
            device = f"cuda:{device}"
        self.device = torch.device(device)

        self.init_predictor = BaseClient(init_predictor_config_file)
        denoiser = BaseClient(denoiser_config_file)

        schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        diffusion = GaussianDiffusion(denoiser, config.TIMESTEPS, schedule).to(self.device)
        self.sampler = diffusion

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

    def __call__(self, img):
        img = self.transform_image(img)
                
        temp = img
        img = self.crop_concat(img)

        inputs = {'input': img.cpu().numpy()}
        results = self.init_predictor.infer(inputs)
        results = results.get('output')
        results = np.array(results)
        init_predict = torch.from_numpy(results).to(self.device)

        noisyImage = torch.randn_like(img).to(self.device)
        sampledImgs = self.sampler(noisyImage, init_predict, self.pre_ori)

        finalImgs = (sampledImgs + init_predict)
        finalImgs = self.crop_concat_back(temp, finalImgs)
        finalImgs = finalImgs.cpu()[0]
        finalImgs = finalImgs.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return finalImgs
 