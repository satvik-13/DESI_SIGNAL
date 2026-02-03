import os
import base64
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from detector import analyze_voice

# 1. Initialize App & Environment
load_dotenv()
app = FastAPI()

# 2. CORS & Middleware (Crucial for browser/portal testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_ngrok_skip_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# 3. Models
class VoiceRequest(BaseModel):
    audioBase64: Optional[str] = None
    language: str = "english"

SECRET_API_KEY = "sk_desi_9988776655"

# 4. The Unified Endpoint
@app.post("/api/voice-detection")
async def detect_voice(
    request_data: Optional[VoiceRequest] = None,
    file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form(None),
    x_api_key: str = Header(None)
):
    # Security Validation
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    temp_path = "temp_eval_audio.mp3"
    selected_lang = "english"

    try:
        # PATH A: Direct File Upload (Postman/cURL -F)
        if file:
            selected_lang = language if language else "english"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # PATH B: Base64 JSON (Automated Evaluation/Portal)
        elif request_data and request_data.audioBase64:
            selected_lang = request_data.language
            audio_data = base64.b64decode(request_data.audioBase64)
            with open(temp_path, "wb") as f:
                f.write(audio_data)
        
        else:
            return {"status": "error", "message": "No audio data provided (Expected File or Base64)"}

        # Run Analysis
        prediction, confidence, explanation = analyze_voice(temp_path, selected_lang)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "status": "success",
            "classification": prediction,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"status": "error", "message": f"Processing failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
