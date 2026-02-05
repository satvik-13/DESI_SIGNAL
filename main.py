import os
import base64
import shutil
from fastapi import FastAPI, Header, HTTPException, Request, File, UploadFile, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# This imports the AI logic from your detector.py file
from detector import analyze_voice 

# 1. Initialization
load_dotenv()
app = FastAPI()

# 2. Ngrok Warning Bypass Middleware
@app.middleware("http")
async def add_ngrok_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# 3. CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fixed Security Key
SECRET_API_KEY = "sk_desi_9988776655"

@app.post("/api/voice-detection")
async def detect_voice(
    request: Request, 
    file: Optional[UploadFile] = File(None), 
    language: Optional[str] = Form(None),
    x_api_key: str = Header(None)
):
    # 4. Security Check
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    temp_path = ""
    try:
        # 5. Unified Input Handling
        # Path A: Multipart File Upload (Commonly used by manual testing)
        if file:
            audio_binary = await file.read()
            target_lang = language if language else "english"
        
        # Path B: JSON Body (Commonly used by automated evaluation systems)
        else:
            try:
                body = await request.json()
                if "audioBase64" in body:
                    audio_binary = base64.b64decode(body["audioBase64"])
                    target_lang = body.get("language", "english")
                else:
                    return {"status": "error", "message": "No audio data provided"}
            except Exception:
                return {"status": "error", "message": "Invalid request format"}

        # 6. Save temporary file with a unique name to avoid conflicts
        temp_path = f"eval_{os.urandom(4).hex()}.mp3"
        with open(temp_path, "wb") as f:
            f.write(audio_binary)

        # 7. Process voice using your 'detector.py' logic
        classification, confidence, explanation = analyze_voice(temp_path, target_lang)

        # 8. Return Response in official Level 2 format
        return {
            "status": "success",
            "classification": classification,
            "confidenceScore": float(confidence),
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": f"Server processing error: {str(e)}"}
    finally:
        # 9. Guaranteed Cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # Use environment PORT for Render compatibility
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
