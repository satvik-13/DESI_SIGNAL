import joblib
import librosa
import numpy as np
import whisper
import os

# Load models once for optimized response time
voice_model = joblib.load("voice_detector_model.pkl")
lang_model = whisper.load_model("tiny")

def detect_language_autonomously(file_path):
    """Verifies the spoken language matches the request"""
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(lang_model.device)
    
    _, probs = lang_model.detect_language(mel)
    code = max(probs, key=probs.get)
    
    mapping = {"ta": "Tamil", "en": "English", "hi": "Hindi", "ml": "Malayalam", "te": "Telugu"}
    return mapping.get(code, "Unknown")

def extract_features(file_path):
    """Generates a 13-dimensional DNA fingerprint using MFCCs"""
    y, sr = librosa.load(file_path, duration=3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def analyze_voice(file_path, user_selected_lang):
    mapping = {"ta": "Tamil", "hi": "Hindi", "en": "English", "ml": "Malayalam", "te": "Telugu"}
    
    try:
        # 1. LOAD & PREPROCESS (Even for MP3, this is key)
        audio = whisper.load_audio(file_path)
        
        # We manually pad to 30s so the 'Tiny' model doesn't get confused
        audio = whisper.pad_or_trim(audio) 
        mel = whisper.log_mel_spectrogram(audio).to(lang_model.device)

        # 2. LANGUAGE DETECTION
        _, probs = lang_model.detect_language(mel)
        detected_code = max(probs, key=probs.get)
        
        # If the AI is very unsure (<15%), we trust your UI selection
        if probs[detected_code] < 0.15:
            actual_lang = user_selected_lang
        else:
            actual_lang = mapping.get(detected_code, "Unknown")

        # 3. AI VS HUMAN PREDICTION
        dna = extract_features(file_path).reshape(1, -1)
        prediction = voice_model.predict(dna)[0]
        confidence = max(voice_model.predict_proba(dna)[0])

        # 4. DYNAMIC EXPLANATION BUILDER
        explanation = f"Vocal patterns analyzed for {actual_lang} phonetics. "
        
        # Mismatch Alert (The Shield)
        if actual_lang != user_selected_lang and actual_lang != "Unknown":
            explanation = (f"⚠️ ALERT: Language Mismatch! Expected {user_selected_lang} "
                           f"but detected {actual_lang}. Potential spoofing.")
            return prediction, 0.40, explanation

        # Technical Resonance Details
        if prediction == "HUMAN":
            explanation += "Matches biological resonance and natural phonetic decay."
        else:
            explanation += "Detected synthetic texture and unnatural pitch consistency."

        return prediction, round(float(confidence), 2), explanation

    except Exception as e:
        return "HUMAN", 0.50, f"System Error: {str(e)}"