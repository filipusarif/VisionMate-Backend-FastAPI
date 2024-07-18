from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import cv2
import numpy as np
import io
import random

app = FastAPI()

# Development
origins = [
    "http://localhost:5173",  # Tambahkan URL frontend React kamu di sini
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

def calculate_distance(bbox):
    # Placeholder function to calculate distance from the camera
    # Assume we have some method to calculate distance
    return random.randint(50, 150)  # Example random distance for all objects

def detections_to_story(detections):
    introduction_templates = [
        "Di depan Anda,",
        "Anda dapat melihat",
        "Terdapat",
    ]
    object_templates = [
        "{count} {label} pada jarak {distance} cm",
        "{count} {label} dengan jarak {distance} cm",
        "{count} {label} yang berada {distance} cm dari Anda",
    ]
    conjunctions = [
        "dan",
        "serta",
        "juga",
    ]

    if not detections:
        return "Tidak ada objek yang terdeteksi."

    # Group detections by label and find the closest distance
    detection_summary = {}
    for detection in detections:
        label = detection['name']
        distance = calculate_distance(detection)
        if label in detection_summary:
            detection_summary[label]['count'] += 1
            detection_summary[label]['closest_distance'] = min(detection_summary[label]['closest_distance'], distance)
        else:
            detection_summary[label] = {'count': 1, 'closest_distance': distance}

    stories = [random.choice(introduction_templates)]
    for i, (label, summary) in enumerate(detection_summary.items()):
        count = summary['count']
        closest_distance = summary['closest_distance']
        stories.append(random.choice(object_templates).format(count=count, label=label, distance=closest_distance))
        if i < len(detection_summary) - 1:
            stories.append(random.choice(conjunctions))

    return " ".join(stories) + "."

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(image)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    story = detections_to_story(detections)
    return JSONResponse(content={"detections": detections, "story": story})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
