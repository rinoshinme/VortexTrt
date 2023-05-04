# VortexTRT
主要支持转onnx以及onnx模型转tensorrt  
暂不支持从零构建tensorrt引擎


##  支持模型
- alexnet
- resnet


## TODO:
- (*)onnx model convertion
- (*)tensorrt inference
- engine builder (in replace of trtexec)
- int8 engine builder
- python binding for vortex-trt engine


## 注意事项
- 核函数不能传递引用，也不能传递RAII的对象。
