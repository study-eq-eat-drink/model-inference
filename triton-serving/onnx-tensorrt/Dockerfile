FROM nvcr.io/nvidia/tritonserver:22.08-py3

RUN mkdir -p /models/nsmc-onnx-tensorrt/0

COPY model/nsmc/onnx/model.onnx /models/nsmc-onnx-tensorrt/0/model.onnx
COPY triton-serving/onnx-tensorrt/config.pbtxt /models/nsmc-onnx-tensorrt/config.pbtxt

#tritonserver --model-repository=/models
