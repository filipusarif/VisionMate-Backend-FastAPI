from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import cv2
import numpy as np
import io

app = FastAPI()

# Tambahkan middleware CORS
origins = [
    "https://blindsens.vercel.app/",  # Tambahkan URL frontend React kamu di sini
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(image)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return JSONResponse(content=detections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
