#$ docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /blar_blar/test_models:/models nvcr.io/nvidia/tritonserver:21.04-py3 tritonserver --model-repository=/models
FROM nvcr.io/nvidia/tritonserver:21.04-py3

RUN mkdir -p /models/nsmc-onnx/0_1

COPY model/onnx/model.onnx /models/nsmc-onnx/0_1/model.onnx
COPY triton-serving/onnx/config.pbtxt /models/nsmc-onnx/0_1/config.pbtxt

RUN tritonserver --model-repository=/models
