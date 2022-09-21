# 참조
- https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html

# 서론
공식 문서에 따르면 max_workspace_size_bytes는 tensorrt의 작업 메모리를 제한한다.
너무 적으면 tensorrt의 특정 알고리즘을 사용할 수 없어 성능이 저하 될 수 있다고 한다.(log를 잘 확인)
근데 여기서 triton은 gpu 메모리를 제한 할 수 없기 때문에 max_batch를 너무 높여 버리면 tensorrt의 작업 공간이 없어지게 되고 이에 따라 API는 죽어버린다.

# onnx-tensorrt
## case1
max_batch : 8
max_workspace_size_bytes : 1g
```text
=================================
user count : 1
total request count : 480
total request time : 65.26229 sec
dps : 7.3549365989751045 dps
=================================
user count : 2
total request count : 480
total request time : 1.17978 sec
dps : 406.85526656412765 dps
=================================
user count : 3
total request count : 480
total request time : 1.16563 sec
dps : 411.79377031978834 dps
=================================
user count : 4
total request count : 480
total request time : 1.16771 sec
dps : 411.059431587858 dps
=================================
user count : 5
total request count : 480
total request time : 1.16494 sec
dps : 412.0376716008466 dps
=================================
user count : 6
total request count : 480
total request time : 1.16697 sec
dps : 411.320362142931 dps
=================================
user count : 7
total request count : 480
total request time : 1.16353 sec
dps : 412.5391906570678 dps
=================================
user count : 8
total request count : 480
total request time : 1.16162 sec
dps : 413.21427985012133 dps
=================================
user count : 9
total request count : 480
total request time : 1.16741 sec
dps : 411.16487249139334 dps
=================================
user count : 10
total request count : 480
total request time : 1.16275 sec
dps : 412.81419253505794 dps
```

## case2
max_batch : 8
max_workspace_size_bytes : 8g
```text
=================================
user count : 1
total request count : 480
total request time : 67.10373 sec
dps : 7.153104963592396 dps
=================================
user count : 2
total request count : 480
total request time : 1.16635 sec
dps : 411.540483039331 dps
=================================
user count : 3
total request count : 480
total request time : 1.15679 sec
dps : 414.94031357759707 dps
=================================
user count : 4
total request count : 480
total request time : 1.15682 sec
dps : 414.930649988644 dps
=================================
user count : 5
total request count : 480
total request time : 1.15663 sec
dps : 415.0001865502044 dps
=================================
user count : 6
total request count : 480
total request time : 1.15741 sec
dps : 414.71850656787797 dps
=================================
user count : 7
total request count : 480
total request time : 1.15232 sec
dps : 416.552284062307 dps
=================================
user count : 8
total request count : 480
total request time : 1.15067 sec
dps : 417.14851074031077 dps
=================================
user count : 9
total request count : 480
total request time : 1.15037 sec
dps : 417.25536945849564 dps
=================================
user count : 10
total request count : 480
total request time : 1.15381 sec
dps : 416.0137140344446 dps
```
## 결론
 - max_workspace_size_bytes에 따른 초기 warm up에는 변화 없음
 - dps가 소폭 상승되어 보이나 memory에 비해 큰 의미 없음
