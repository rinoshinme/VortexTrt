import torch 
import torchvision
import struct
import cv2
import numpy as np


def save_path():
    model = torchvision.models.alexnet(pretrained=True)
    model.eval()
    torch.save(model, 'alexnet.pth')


def export_onnx():
    model = torchvision.models.alexnet(pretrained=True)
    model = model.eval()

    # export
    inputs = torch.rand(1, 3, 224, 224)
    torch.onnx.export(model, inputs, 'alexnet.onnx', 
        verbose=True,
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={
            'input': {0: 'batch'}, 
            'output': {0: 'batch'}
        }
    )


def alexnet_infer():
    model = torchvision.models.alexnet(pretrained=True)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    image_path = '../../data/29bb3ece3180_11.jpg'
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
        outputs = model(image)
        print(outputs.shape)
        outputs = outputs.cpu().numpy()[0]
    print(outputs)


def save_weights(weights_path):
    """
    export binary weights for c++ loading.
    """
    model = torch.load('alexnet.pth')
    model = model.eval()

    state_dict = model.state_dict()
    with open('alexnet.wts', 'w') as f:
        # write number of weight tensors
        f.write('{}\n'.format(len(state_dict.keys())))
        for k, v in state_dict.items():
            print('key: ', k)
            print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {}'.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')


if __name__ == '__main__':
    # save_weights('./alexnet.pth')
    # export_onnx()
    alexnet_infer()
