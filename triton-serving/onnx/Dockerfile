FROM nvcr.io/nvidia/tritonserver:22.08-py3

RUN mkdir -p /models/nsmc-onnx/0

COPY model/nsmc/onnx/model.onnx /models/nsmc-onnx/0/model.onnx
COPY triton-serving/onnx/config.pbtxt /models/nsmc-onnx/config.pbtxt

#tritonserver --model-repository=/models
