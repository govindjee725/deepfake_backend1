from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

from app.routes.detect_route import router as detect_router

app = FastAPI(title="Deepfake Detection API")

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Include detection route
app.include_router(detect_router)

# Serve heatmap images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
def home():
    return {
        "message": "Deepfake Detection API Running"
    }