from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from ultralytics import YOLO
import os
import base64
from PIL import Image
from io import BytesIO
import pathlib

# 디렉토리 구조 확인
current_dir = pathlib.Path.cwd()
print(f"현재 작업 디렉토리: {current_dir}")
if not os.path.exists("www"):
    os.makedirs("www")
if not os.path.exists("www/templates"):
    os.makedirs("www/templates")

# FastAPI 앱 생성
app = FastAPI()

# CORS 미들웨어 설정 (모든 오리진 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="www/static"), name="static")
templates = Jinja2Templates(directory="www/templates")

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/info")
async def api_info():
    return {"message": "YOLOv8 Object Detection API"}

@app.post("/api/detect")
async def detect_objects(image: UploadFile = File(...)):
    # 이미지 읽기
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # YOLOv8로 객체 감지
    results = model(img)
    result = results[0]
    
    # 결과 처리
    detected_objects = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        
        detected_objects.append({
            "class": cls_name,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        })
    
    # 바운딩 박스가 있는 이미지 생성
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cls_name = obj["class"]
        conf = obj["confidence"]
        
        # 바운딩 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 레이블 텍스트
        label = f"{cls_name} {conf:.2f}"
        
        # 텍스트 배경
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # 텍스트 추가
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 이미지를 base64로 인코딩
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "detected_objects": detected_objects,
        "processed_image": f"data:image/jpeg;base64,{img_base64}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)