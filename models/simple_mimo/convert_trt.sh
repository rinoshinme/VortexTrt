#!/bin/bash

trtexec --onnx=./mimo.onnx \
        --saveEngine=./mimo.engine \
        --minShapes=x1:1x3x224x224,x2:1x3x224x224 \
        --optShapes=x1:16x3x224x224,x2:16x3x224x224 \
        --maxShapes=x1:16x3x224x224,x2:16x3x224x224 \
        --workspace=1024
