#!/bin/bash

trtexec --onnx=./arcface_resnet50.onnx \
        --saveEngine=./arcface_resnet50.engine \
        --minShapes=input.1:1x3x112x112 \
        --optShapes=input.1:16x3x112x112 \
        --maxShapes=input.1:16x3x112x112 \
        --workspace=1024
