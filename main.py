import cv2
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import uuid
from fastapi import FastAPI, File, UploadFile, Header, Response
import shutil
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.notification import sent_notification
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:8080",
    "https://d290-2800-4b0-9902-9df6-15d7-abf-40c6-1813.ngrok.io",
    "*",
]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = None


@app.get("/")
def home():
    return {"message": "Welcome to vision api"}


@app.post("/analyze-image")
def read_item(file: UploadFile):
    if file.content_type not in ["image/jpg", "image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    filepath = _save_image(upload_file=file)
    sent_notification(channel="humanState", event_name="dreamState", data={"status": "sleepy", "filename": f"/{filepath}", "live": "/video-live", "video": "/video"})
    return {"message": "sleep", "filename": file.filename}

def _save_image(upload_file: UploadFile)->str:
    content_type = upload_file.content_type.split("/")
    filepath = f"static/images/{uuid.uuid4().hex}.{content_type[-1]}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return filepath

@app.get("/video")
async def video_endpoint(range: str = Header(None)):
    CHUNK_SIZE = 1024*1024
    video_path = Path("videos/pensive.mp4")
    start, end = range.replace("bytes=", "").split("-")
    start = int(start)
    end = int(end) if end else start + CHUNK_SIZE

    with open(video_path, "rb") as video:
        video.seek(start)
        data = video.read(end - start)
        filesize = str(video_path.stat().st_size)
        headers = {
            'Content-Range': f'bytes {str(start)}-{str(end)}/{filesize}',
            'Accept-Ranges': 'bytes'
        }
        return Response(data, status_code=206, headers=headers, media_type="video/mp4")

camera = cv2.VideoCapture("/dev/video2")

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/video-live")
async def video_live():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
