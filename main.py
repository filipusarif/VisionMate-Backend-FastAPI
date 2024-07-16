from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import imageio
import numpy as np
import io
import onnxruntime as ort

app = FastAPI()

# Tambahkan middleware CORS
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

# Load ONNX model
ort_session = ort.InferenceSession('yolov5s.onnx')

def preprocess_image(image):
    # Resize image to 640x640
    image_resized = imageio.imresize(image, (640, 640))
    # Normalize image
    image_normalized = image_resized / 255.0
    # Add batch dimension
    input_data = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    # Transpose dimensions to match model input
    input_data = np.transpose(input_data, (0, 3, 1, 2))
    return input_data

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image_np = np.array(image)

    # Preprocess the image to match input size requirements
    input_data = preprocess_image(image_np)
    
    # Run inference
    inputs = {ort_session.get_inputs()[0].name: input_data}
    outputs = ort_session.run(None, inputs)
    
    # Extract detection results
    boxes = outputs[0][0]
    scores = outputs[1][0]
    classes = outputs[2][0]
    
    # Format detections for JSON response
    detections = [
        {"box": box.tolist(), "class": int(cls), "score": float(score)}
        for box, cls, score in zip(boxes, classes, scores)
        if score > 0.5  # Confidence threshold
    ]
    
    return JSONResponse(content=detections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
