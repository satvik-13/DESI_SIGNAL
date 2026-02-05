import os
import base64
import shutil
import gc  # Added for memory management
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

@app.get("/")
async def root():
    print("DEBUG: Root endpoint hit - Service is Awake", flush=True)
    return {"message": "Service is running"}

@app.post("/api/voice-detection")
async def detect_voice(
    request: Request, 
    file: Optional[UploadFile] = File(None), 
    language: Optional[str] = Form(None),
    x_api_key: str = Header(None)
):
    # 4. Security Check
    if x_api_key != SECRET_API_KEY:
        print(f"DEBUG: Auth Failed. Received Key: {x_api_key}", flush=True)
        raise HTTPException(status_code=401, detail="Invalid API Key")

    temp_path = ""
    try:
        # 5. Unified Input Handling
        print("DEBUG: Request received. Processing input...", flush=True)
        
        if file:
            print(f"DEBUG: Mode: Multipart. File: {file.filename}", flush=True)
            audio_binary = await file.read()
            target_lang = language if language else "english"
        else:
            print("DEBUG: Mode: JSON/Base64", flush=True)
            try:
                body = await request.json()
                if "audioBase64" in body:
                    audio_binary = base64.b64decode(body["audioBase64"])
                    target_lang = body.get("language", "english")
                else:
                    return {"status": "error", "message": "No audio data provided"}
            except Exception as e:
                print(f"ERROR: JSON Parsing failed: {str(e)}", flush=True)
                return {"status": "error", "message": "Invalid request format"}

        # 6. Save temporary file
        temp_path = f"eval_{os.urandom(4).hex()}.mp3"
        with open(temp_path, "wb") as f:
            f.write(audio_binary)
        print(f"DEBUG: File saved to {temp_path}. Size: {len(audio_binary)} bytes", flush=True)

        # 7. Process voice using your 'detector.py' logic
        print(f"DEBUG: Starting analyze_voice for lang: {target_lang}...", flush=True)
        classification, confidence, explanation = analyze_voice(temp_path, target_lang)
        print("DEBUG: analyze_voice completed successfully", flush=True)

        # 8. Return Response
        return {
            "status": "success",
            "classification": classification,
            "confidenceScore": float(confidence),
            "explanation": explanation
        }

    except Exception as e:
        print(f"CRITICAL ERROR during processing: {str(e)}", flush=True)
        return {"status": "error", "message": f"Server processing error: {str(e)}"}
    
    finally:
        # 9. Guaranteed Cleanup & RAM recovery
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"DEBUG: Deleted {temp_path}", flush=True)
        
        # This forces Python to free up RAM immediately - Crucial for Render 512MB
        gc.collect() 
        print("DEBUG: RAM cleared", flush=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
