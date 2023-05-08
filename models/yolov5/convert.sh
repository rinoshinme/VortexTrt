#!/bin/bash


# convert model to onnx
# change line of code in yolo.py
# '''
#         return x if self.training else (torch.cat(z, 1), x)
#         # return x if self.training else torch.cat(z, 1)
# '''
python export.py --weights yolov5s.pt --include onnx

# convert tensorrt
trtexec --onnx=./yolov5s.onnx \
        --saveEngine=./yolov5s.engine \
        --minShapes=input.1:1x3x640x640 \
        --optShapes=input.1:16x3x640x640 \
        --maxShapes=input.1:16x3x640x640 \
        --workspace=2048
