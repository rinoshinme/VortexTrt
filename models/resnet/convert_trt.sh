#!/bin/bash

trtexec --onnx=./resnet50.onnx \
        --saveEngine=./resnet50.engine \
        --minShapes=input.1:1x3x224x224 \
        --optShapes=input.1:16x3x224x224 \
        --maxShapes=input.1:16x3x224x224 \
        --workspace=1024
