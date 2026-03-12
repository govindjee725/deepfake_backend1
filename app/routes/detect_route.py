from fastapi import APIRouter, UploadFile, File
import shutil
from app.services.detect_service import detect_deepfake

router = APIRouter()

UPLOAD_FOLDER = "uploads/"

@router.post("/detect")
async def detect(video: UploadFile = File(...)):

    video_path = UPLOAD_FOLDER + video.filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    result = detect_deepfake(video_path)

    return result