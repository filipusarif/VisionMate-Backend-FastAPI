from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    return "JSONResponse(content=detections)"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
