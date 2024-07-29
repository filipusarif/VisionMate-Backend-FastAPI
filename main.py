from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import cv2
import numpy as np
import io
import random
import requests
from bs4 import BeautifulSoup
import re

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
    x1, y1, x2, y2 = bbox
    object_height = y2 - y1
    vertical_position = y1 + (object_height / 2)
    normalized_vertical_position = vertical_position / image_height
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
    
    for detection in detections:
        bbox = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
        detection['distance'] = calculate_distance(bbox, image_height)
    
    story = detections_to_story(detections)
    return JSONResponse(content={"detections": detections, "story": story})

@app.get("/search/")
async def search(query: str):
    URL = "https://www.google.com/search?q=" + query

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    }

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    result = ''
    potential_selectors = [
        'div.BNeawe.iBp4i.AP7Wnd',  # Primary result
        'div.BNeawe.tAd8D.AP7Wnd',  # Alternative result
        'div.Z0LcW.t2b5Cf',         # Another possible result
        'div.Z0LcW t2b5Cf',         # Another possible result
        'div.vk_gbt',               # Another possible result
        'div.vk_gbt.TDf9v',         # Another possible result
        'div.ayqGOc.kno-fb-ctx.kpd-lv.kpd-le.KBXm4e',         # Another possible result
        'div.vk_h.yvOIec',         # Another possible result
        'span.hgKElc',         # Another possible result
        'div.kno-rdesc span',       # Knowledge panel description
    ]
    
    for selector in potential_selectors:
        element = soup.select_one(selector)
        if element:
            result = element.get_text()
            break

    # Additional check using regex
    # if not result:
    #     regex_patterns = [
    #         r'\b\d{1,2}\s\w+\s\d{4}\b',  # Date format like "5 February 1985"
    #     ]
    #     for pattern in regex_patterns:
    #         match = re.search(pattern, soup.text)
    #         if match:
    #             result = match.group(0)
    #             break

    if result:
        return JSONResponse(content={"answer": result})
    else:
        raise HTTPException(status_code=404, detail="No results found")
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
