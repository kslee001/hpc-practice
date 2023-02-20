# 2023-winter-school-project
2023년 2월 겨울학교 프로젝트
참고 git : https://github.com/milesial/Pytorch-UNet

# 실행
## 모델 가중치 준비
스크립트를 통해 weights.bin 다운로드
```
sh download_weights.sh
```

## 이미지를 바이너리로 변환 (img2bin)
jpg이미지를 바이너리 파일로 생성
```
python tools/img2bin.py img/inData1.jpg input1.bin
```
## C코드 컴파일 및 실행
```
make clean && make -j
./main weights.bin input1.bin output1.bin
```

## 바이너리 결과를 이미지로 변환 (bin2img)
바이너리로 저장된 추론 결과를 png이미지로 변환
```
python tools/bin2img.py output1.bin output1.png
```

# ETC
## Python 환경 준비
python실행에 필요한 numpy, pillow, pytorch env 설치
```
conda env create -f ./tools/env.yaml
```
