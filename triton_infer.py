from triton.processor import Processor as SealRemoval


class Inferer():
    def __init__(self):
        config_file = 'conf.yml'
        self.inferer = SealRemoval(config_file)

    def infer(self, img):
        output = self.inferer(img)
        return output
    
if __name__ == '__main__':
    import cv2

    inferer = Inferer()

    img_path = 'demo/input.png'
    img = cv2.imread(img_path)
    
    output = inferer.infer(img)
    cv2.imwrite('demo/triton_output.png', output)