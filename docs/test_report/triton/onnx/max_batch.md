# onnx
## 서론
아래 두가지를 테스트하고 추론
 - max_batch가 dps에 어떤 영향을 미치는가
 - onnx의 model최적화에 따라 학습시 사용했던 16batch보다 얼마나 더 많은 batch가 가능한가

## case1
- max_batch : 8
```text
=================================
user count : 1
total request count : 480
total request time : 3.68067 sec
dps : 130.4110046908725 dps
=================================
user count : 2
total request count : 480
total request time : 2.84505 sec
dps : 168.71423215330537 dps
=================================
user count : 3
total request count : 480
total request time : 2.91788 sec
dps : 164.50279041754413 dps
=================================
user count : 4
total request count : 480
total request time : 2.89072 sec
dps : 166.0484525264362 dps
=================================
user count : 5
total request count : 480
total request time : 2.95023 sec
dps : 162.69942249738023 dps
=================================
user count : 6
total request count : 480
total request time : 2.93813 sec
dps : 163.3692895754055 dps
=================================
user count : 7
total request count : 480
total request time : 2.88689 sec
dps : 166.26884028060292 dps
=================================
user count : 8
total request count : 480
total request time : 2.81294 sec
dps : 170.63983350193058 dps
=================================
user count : 9
total request count : 480
total request time : 2.81848 sec
dps : 170.30484855529 dps
=================================
user count : 10
total request count : 480
total request time : 2.82138 sec
dps : 170.1296321350257 dps
```
## case2
- max_batch : 16
```text
=================================
user count : 1
total request count : 480
total request time : 3.48065 sec
dps : 137.90515556736162 dps
=================================
user count : 2
total request count : 480
total request time : 2.85584 sec
dps : 168.07660275312014 dps
=================================
user count : 3
total request count : 480
total request time : 2.63334 sec
dps : 182.2780442275062 dps
=================================
user count : 4
total request count : 480
total request time : 2.50918 sec
dps : 191.29736835053896 dps
=================================
user count : 5
total request count : 480
total request time : 2.45144 sec
dps : 195.80314200650997 dps
=================================
user count : 6
total request count : 480
total request time : 2.47236 sec
dps : 194.14658532295243 dps
=================================
user count : 7
total request count : 480
total request time : 2.49169 sec
dps : 192.64063755724513 dps
=================================
user count : 8
total request count : 480
total request time : 2.51108 sec
dps : 191.15299020918945 dps
=================================
user count : 9
total request count : 480
total request time : 2.48021 sec
dps : 193.5318639415186 dps
=================================
user count : 10
total request count : 480
total request time : 2.47084 sec
dps : 194.265600781823 dps
```
## case3
- max_batch : 32
```text
=================================
user count : 1
total request count : 480
total request time : 3.41106 sec
dps : 140.7186874476918 dps
=================================
user count : 2
total request count : 480
total request time : 2.79509 sec
dps : 171.7295237672805 dps
=================================
user count : 3
total request count : 480
total request time : 2.60496 sec
dps : 184.26358164933202 dps
=================================
user count : 4
total request count : 480
total request time : 2.38705 sec
dps : 201.0852293991522 dps
=================================
user count : 5
total request count : 480
total request time : 2.34830 sec
dps : 204.40322851496148 dps
=================================
user count : 6
total request count : 480
total request time : 2.32663 sec
dps : 206.30696594201993 dps
=================================
user count : 7
total request count : 480
total request time : 2.26129 sec
dps : 212.26823407405126 dps
=================================
user count : 8
total request count : 480
total request time : 2.26588 sec
dps : 211.83799322376314 dps
=================================
user count : 9
total request count : 480
total request time : 2.26665 sec
dps : 211.7665561586897 dps
=================================
user count : 10
total request count : 480
total request time : 2.26645 sec
dps : 211.78511268414744 dps
```
## case4
- max_batch : 64
```text
=================================
user count : 1
total request count : 60
total request time : 3.50061 sec
dps : 137.11912293492878 dps
=================================
user count : 2
total request count : 60
total request time : 2.81003 sec
dps : 170.81658122299208 dps
=================================
user count : 3
total request count : 60
total request time : 2.58707 sec
dps : 185.53777282338473 dps
=================================
user count : 4
total request count : 60
total request time : 2.43368 sec
dps : 197.23234437920428 dps
=================================
user count : 5
total request count : 60
total request time : 2.36903 sec
dps : 202.61444996687948 dps
=================================
user count : 6
total request count : 60
total request time : 2.36046 sec
dps : 203.34988266251548 dps
=================================
user count : 7
total request count : 60
total request time : 2.27260 sec
dps : 211.21196680350093 dps
=================================
user count : 8
total request count : 60
total request time : 2.51072 sec
dps : 191.1800184108724 dps
=================================
user count : 9
total request count : 60
total request time : 2.36320 sec
dps : 203.11399515738498 dps
=================================
user count : 10
total request count : 60
total request time : 2.54497 sec
dps : 188.60731102015396 dps

```
## 결론
 - max_batch가 올라갈수록 dps는 올라감
 - 올라가는것도 어느정도 수준까지고 32이상에서는 동일함
 - 학습시 최대 batch가 16 수준이었던걸 생각했을때 64batch를 돌려도 돌아가는게 onnx의 model 최적화 덕분인지 triton 최적화 덕분이지 확인이 필요함 하지만 놀라움