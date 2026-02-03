import os
import base64
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# This imports the AI logic from your detector.py file
from detector import analyze_voice 

# 1. Initialization
load_dotenv()
app = FastAPI()

# 2. CORS Policy (Essential for the judges' tester)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Request Model (As per Problem Statement)
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# Fixed Security Key
SECRET_API_KEY = "sk_desi_9988776655"

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # Log the incoming request so you can watch it live
    print(f"--- Incoming Evaluation Request | Language: {request.language} ---")
    # 4. Security Check
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 5. Decode Base64 and save as temporary file
        audio_binary = base64.b64decode(request.audioBase64)
        file_path = f"eval_{request.language}_{os.urandom(4).hex()}.mp3" # Unique filename for multi-request handling
        
        with open(file_path, "wb") as f:
            f.write(audio_binary)

        # 6. Process voice using the 'detector.py' logic
        # It returns: classification (str), confidence (float), explanation (str)
        classification, confidence, explanation = analyze_voice(file_path, request.language)

        # 7. Final Response (Level 2 Required Format)
        return {
            "status": "success",
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Server processing error: {str(e)}"
        }
    finally:
        # Cleanup temp file to keep your desktop clean
        if os.path.exists("temp_eval_audio.mp3"):
            os.remove("temp_eval_audio.mp3")

if __name__ == "__main__":
    import uvicorn
    # This matches the port 8000 you used in the ngrok command
    uvicorn.run(app, host="0.0.0.0", port=8000)