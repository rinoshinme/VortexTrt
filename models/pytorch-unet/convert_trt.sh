#!/bin/bash

trtexec --onnx=./unet.onnx \
        --saveEngine=./unet.engine \
        --minShapes=input.1:1x3x512x512 \
        --optShapes=input.1:16x3x512x512 \
        --maxShapes=input.1:16x3x512x512 \
        --workspace=1024
