import argparse
import logging
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torchvision import transforms

from unet_model import UNet


class UNetInfer(object):
    """
    only support scale-factor=0.5 model
    """
    def __init__(self, model_path):
        # self.scale_factor = 0.5
        self.scale_factor = None
        self.num_classes = 2
        self.bilinear = False
        self.threshold = 0.5
        self.net, self.device = self.load_model(model_path)
    
    def load_model(self, model_path):
        net = UNet(n_channels=3, n_classes=self.num_classes, bilinear=self.bilinear)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
        net.eval()
        return net, device
    
    def preprocess(self, pil_img, scale=None):
        w, h = pil_img.size
        if scale is None:
            newW, newH = 512, 512
        else:
            newW, newH = int(scale * w), int(scale * h)
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        return img 

    def infer(self, image_path):
        image = Image.open(image_path)
        inputs = self.preprocess(image, self.scale_factor)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device, dtype=torch.float32)
        print(inputs)

        with torch.no_grad():
            output = self.net(inputs).cpu()
            print(output)
            output = F.interpolate(output, (image.size[1], image.size[0]), mode='bilinear')
            if self.net.n_classes > 1:
                mask = output.argmax(dim=1)
            else:
                mask = torch.sigmoid(output) > self.threshold
        return mask[0].long().squeeze().numpy()
    
    def display(self, image, mask):
        classes = mask.max() + 1
        fig, ax = plt.subplots(1, classes + 1)
        ax[0].set_title('Input image')
        ax[0].imshow(img)
        for i in range(classes):
            ax[i + 1].set_title(f'Mask (class {i + 1})')
            ax[i + 1].imshow(mask == i)
        plt.xticks([]), plt.yticks([])
        plt.show()
    
    def to_onnx(self, onnx_path):
        input_width = 512
        input_height = 512
        dummy_inputs = torch.randn(1, 3, input_width, input_height)
        input_names = ['input']
        output_names = ['output']
        torch.onnx.export(
            self.net, 
            dummy_inputs, 
            onnx_path, 
            verbose=True, 
            opset_version=11,
            input_names=input_names, 
            output_names=output_names
        )


if __name__ == '__main__':
    demo = UNetInfer('./weights/unet_carvana_scale0.5_epoch2.pth')
    image_path = '../../sample/29bb3ece3180_11.jpg'
    mask = demo.infer(image_path)
    print(mask.shape)

    onnx_path = './unet.onnx'
    if not os.path.exists(onnx_path):
        demo.to_onnx(onnx_path)
