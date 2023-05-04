# VortexTRT
支持torch转onnx
支持onnx转tensorrt
支持tensorrt引擎推理
暂不支持int8量化  
不支持从零构建tensorrt引擎
不支持动态尺寸
不支持batching


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
