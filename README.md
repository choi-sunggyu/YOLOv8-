# YOLOv8-기반 모바일 객체 인식 웹 서비스 구축

Python 및 javascript, React를 활용한 모바일 객체 인식 웹 서비스입니다.

## 애플리케이션 구조
```text
YoloCamera
├─runs
│  └─detect
│      └─predict
└─www
    ├─static
    │  ├─css
    │  └─js
    └─templates
```

## 사전 준비사항
pip install ultralytics fastapi uvicorn python-multipart pyngrok
pip install -U opencv-python pillow numpy
yolo task=detect mode=predict model=yolov8n.pt

## 시작하기
# ngrok 인증토큰 생성
1. ngrok 웹사이트에 가입합니다 (아직 계정이 없는 경우).
2. 가입 후 대시보드에서 인증 토큰을 확인합니다.
3. 아래 코드를 새로운 셀에 붙여넣고, YOUR_AUTH_TOKEN 부분을 본인의 토큰으로 바꿔 실행합니다:

# 로컬 실행
1. 아래 명령어 실행
python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_AUTH_TOKEN")
print("ngrok 인증 토큰이 설정되었습니다.")
exit()
 > `{{YOUR_AUTH_TOKEN}}`은 앞서 생성한 ngrok토큰 값
2. 아래 명령어 실행
python run_server.py
3. 터미널 창에 생긴 주소로 모바일 접속


