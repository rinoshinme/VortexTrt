import torch 
import torch.nn as nn
import torchvision
import struct


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
        input_names=['image'], 
        output_names=['output'], 
        dynamic_axes={
            'image': {0: 'batch'}, 
            'output': {0: 'batch'}
        }
    )


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
    export_onnx()
