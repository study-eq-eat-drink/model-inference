# model-serving
모델 서빙에 가장 적합한 도구를 찾자

# 테스트 조합

1. triton + tensorflow
2. triton + tensorRT
3. triton + tensorflow-onnx
4. triton + pytorch-onnx
5. bentoML + tensorflow
6. bentoML + tensorflow-onnx
7. bentoML + pytorch
8. bentoML + pytorch-onnx
9. akka + tensorflow
10. akka + triton

# Triton build & run
## Build
```bash
docker build -t benjamin/nsmc/triton-onnx:0.0.1 -f triton-serving/onnx/Dockerfile .
docker build -t benjamin/nsmc/triton-onnx-tensorrt:0.0.1 -f triton-serving/onnx-tensorrt/Dockerfile .
```
## Run
```shell
docker run --gpus=all -p 9000:8000 -p 9001:8001 -p 9002:8002 --rm benjamin/nsmc/triton-onnx:0.0.1 tritonserver --model-repository=/models
docker run --gpus=all -p 9000:8000 -p 9001:8001 -p 9002:8002 --rm benjamin/nsmc/triton-onnx-tensorrt:0.0.1 tritonserver --model-repository=/models
```


