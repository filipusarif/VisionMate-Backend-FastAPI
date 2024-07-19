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
    "http://localhost:5173",  
]

# Production
# origins = [
#     "https://blindsens.vercel.app",  
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def calculate_distance(bbox, image_height):
    # Placeholder function to calculate distance from the camera using vertical position
    x1, y1, x2, y2 = bbox
    object_height = y2 - y1
    vertical_position = y1 + (object_height / 2)
    
    # Normalize vertical position to a value between 0 (top) and 1 (bottom)
    normalized_vertical_position = vertical_position / image_height
    
    # Assume a simple linear relationship between vertical position and distance
    # This is just an example, the actual calculation would depend on the camera setup and calibration
    distance = 100 * (1 - normalized_vertical_position)
    return distance

def detections_to_story(detections):
    introduction_templates = [
        "Di depan Anda,",
        "Anda dapat melihat",
        "Terdapat",
    ]
    object_templates = [
        "{count} {label} pada jarak {distance:.2f} sentimeter",
        "{count} {label} dengan jarak {distance:.2f} sentimeter",
        "{count} {label} yang berada {distance:.2f} sentimeter dari Anda",
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
        distance = detection['distance']
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
    image_height, image_width, _ = image.shape
    results = model(image)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    
    # Calculate distance for each detection and add to the dictionary
    for detection in detections:
        bbox = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
        detection['distance'] = calculate_distance(bbox, image_height)
    
    story = detections_to_story(detections)
    return JSONResponse(content={"detections": detections, "story": story})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
