import os
import torch
import torchvision
import cv2
import numpy as np


class Resnet50(object):
    def __init__(self):
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
    
    def infer(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0).float()
        image.div_(255).sub_(0.5).div_(0.5)
        if torch.cuda.is_available():
            image = image.cuda()
        
        print(image.shape)

        with torch.no_grad():
            outputs = self.model(image)
            print(outputs.shape)
            outputs = outputs.cpu().numpy()[0]
        print('------------------------')
        for i in range(10):
            print(outputs[i])
    
    def to_onnx(self, onnx_path):
        inputs = torch.rand(1, 3, 224, 224)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        torch.onnx.export(self.model, inputs, onnx_path, 
            verbose=True,
            opset_version=11, 
            input_names=['input'], 
            output_names=['output'], 
            dynamic_axes={
                'input': {0: 'batch'}, 
                'output': {0: 'batch'}
            }
        )


if __name__ == '__main__':
    resnet = Resnet50()
    # single image infer
    image_path = '../../data/29bb3ece3180_11.jpg'
    resnet.infer(image_path)

    # onnx export
    onnx_path = './resnet50.onnx'
    if not os.path.exists(onnx_path):
        resnet.to_onnx(onnx_path)
