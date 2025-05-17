import subprocess
import time
from pyngrok import ngrok

# FastAPI 서버 시작
server_process = subprocess.Popen(["python", "app.py"])
print("FastAPI 서버가 시작되었습니다.")

# 서버가 시작될 때까지 잠시 대기
time.sleep(5)

# ngrok 터널 생성
http_tunnel = ngrok.connect(3000)
print(f"ngrok 터널이 생성되었습니다: {http_tunnel.public_url}")

try:
    # 앱이 계속 실행되도록 대기
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # 종료 시 프로세스 정리
    server_process.terminate()
    ngrok.kill()

# 이 코드는 Colab이 계속 실행 중일 때만 작동합니다