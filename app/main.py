from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
import shutil

from app.routes.detect_route import router as detect_router

app = FastAPI(title="Deepfake Detection API")

# Temporary folders
TEMP_VIDEO = "temp_video"
TEMP_FRAMES = "temp_frames"
OUTPUTS = "outputs"

# Create folders if not exist
os.makedirs(TEMP_VIDEO, exist_ok=True)
os.makedirs(TEMP_FRAMES, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# Function to clean temporary files
def clean_temp():

    if os.path.exists(TEMP_VIDEO):
        shutil.rmtree(TEMP_VIDEO)
        os.makedirs(TEMP_VIDEO)

    if os.path.exists(TEMP_FRAMES):
        shutil.rmtree(TEMP_FRAMES)
        os.makedirs(TEMP_FRAMES)

# Clean at startup
clean_temp()

# Include detection route
app.include_router(detect_router)

# Serve heatmap images if needed
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")


@app.get("/")
def home():
    return {
        "message": "Deepfake Detection API Running"
    }