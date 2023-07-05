"""
a simple multiple io models for testing.
"""
import torch
import torch.nn as nn


class SimpleMimo(nn.Module):
    def __init__(self, dropout=0.5, num_classes=1000):
        super().__init__()
        # build simple network, adapted from AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x1, x2):
        feat1 = self.features(x1)
        feat2 = self.features(x2)
        feat = torch.cat([feat1, feat2], dim=1)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        out = self.classifier(feat)
        return out, feat
    
    def init_weights(self):
        pass
    

def to_onnx():
    model = SimpleMimo()
    input1 = torch.rand(1, 3, 224, 224)
    input2 = torch.rand(1, 3, 224, 224)
    inputs = (input1, input2)

    torch.onnx.export(model, inputs, 'mimo.onnx', 
        verbose=True,
        opset_version=11, 
        input_names=['x1', 'x2'], 
        output_names=['output', 'feature'], 
        dynamic_axes={
            'x1': {0: 'batch'},
            'x2': {0: 'batch'}, 
            'output': {0: 'batch'},
            'feature': {0: 'batch'}
        }
    )


if __name__ == '__main__':
    # model = SimpleMimo()
    # x1 = torch.rand(1, 3, 224, 224)
    # x2 = torch.rand(1, 3, 224, 224)
    # out, feat = model(x1, x2)
    # print(out.shape)
    # print(feat.shape)
    to_onnx()
