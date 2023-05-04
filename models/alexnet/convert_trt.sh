#!/bin/bash

trtexec --onnx=./alexnet.onnx \
        --saveEngine=./alexnet.engine \
        --minShapes=input.1:1x3x224x224 \
        --optShapes=input.1:16x3x224x224 \
        --maxShapes=input.1:16x3x224x224 \
        --workspace=1024
