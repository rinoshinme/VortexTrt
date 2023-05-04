"""
face feature extraction
"""
import os
import cv2
import numpy as np
import torch
from backbones import get_model


class Demo(object):
    def __init__(self, name, weight):
        self.net = get_model(name, fp16=False)
        state_dict = torch.load(weight, map_location='cpu')
        self.net.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()
    
    def extract(self, image, normalize=True):
        if isinstance(image, str):
            image = cv2.imread(image)
        # resize
        image = cv2.resize(image, (112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0).float()
        image.div_(255).sub_(0.5).div_(0.5)

        # extract feature
        with torch.no_grad():
            feat = self.net(image)
            feat = feat.numpy()[0]
        
        if normalize:
            # do normalization
            length = np.linalg.norm(feat)
            feat = feat / length
        return feat
    
    def to_onnx(self, onnx_path):
        dummy_inputs = torch.randn(1, 3, 112, 112)
        input_names = ['input']
        output_names = ['output']
        torch.onnx.export(
            self.net, 
            dummy_inputs, 
            onnx_path, 
            verbose=True, 
            input_names=input_names, 
            output_names=output_names
        )


if __name__ == '__main__':
    demo = Demo('r50', 'weights/ms1mv3_arcface_r50_fp16/backbone.pth')
    # convert to onnx
    onnx_path = './arcface_resnet50.onnx'
    if not os.path.exists(onnx_path):
        demo.to_onnx(onnx_path)

    # run inference
    image_path = '../../sample/sample1.jpg'
    feat = demo.extract(image_path, normalize=False)
    print(feat[:10])
    """
    sample output: 
        [-0.5209716  -0.11782981  0.0165088  -0.71297413  0.92646235  0.2421217
         -0.35743704 -0.23166685  0.3014495   0.47779232]
    """
