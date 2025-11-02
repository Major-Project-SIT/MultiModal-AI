from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from Deployment.inference import process_local_video
import tempfile
import shutil
import os
import uvicorn  # 👈 required to run the server

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend (on localhost:5500) to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            shutil.copyfileobj(file.file, temp_video)
            temp_video_path = temp_video.name

        # Run your existing process_local_video function
        result = process_local_video(temp_video_path)
        print(f"Analysis result: {result}")

        # Cleanup
        os.remove(temp_video_path)

        return JSONResponse(content={"status": "success", "result": result})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# 👇 Add this section to actually run the FastAPI server
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)