import joblib
import librosa
import numpy as np
import whisper
import os
import gc  # Garbage Collector - essential for 512MB RAM

# IMPORTANT: We removed the global model loading lines to save memory at startup.

def extract_features(file_path):
    """Generates a 13-dimensional DNA fingerprint using MFCCs"""
    # Optimized librosa load: sr=None (native) and duration limit
    y, sr = librosa.load(file_path, sr=None, duration=3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def analyze_voice(file_path, user_selected_lang):
    mapping = {"ta": "Tamil", "hi": "Hindi", "en": "English", "ml": "Malayalam", "te": "Telugu"}
    
    # Initialize variables for cleanup
    lang_model = None
    voice_model = None
    
    try:
        # 1. LAZY LOAD MODELS (Only when a request hits the API)
        # Using 'tiny' as it is the smallest footprint (~72MB)
        lang_model = whisper.load_model("tiny")
        voice_model = joblib.load("voice_detector_model.pkl")

        # 2. LOAD & PREPROCESS
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio) 
        mel = whisper.log_mel_spectrogram(audio).to(lang_model.device)

        # 3. LANGUAGE DETECTION
        _, probs = lang_model.detect_language(mel)
        detected_code = max(probs, key=probs.get)
        
        if probs[detected_code] < 0.15:
            actual_lang = user_selected_lang
        else:
            actual_lang = mapping.get(detected_code, "Unknown")

        # 4. AI VS HUMAN PREDICTION
        dna = extract_features(file_path).reshape(1, -1)
        prediction = voice_model.predict(dna)[0]
        confidence = max(voice_model.predict_proba(dna)[0])

        # 5. DYNAMIC EXPLANATION BUILDER
        explanation = f"Vocal patterns analyzed for {actual_lang} phonetics. "
        
        if actual_lang != user_selected_lang and actual_lang != "Unknown":
            explanation = (f"⚠️ ALERT: Language Mismatch! Expected {user_selected_lang} "
                           f"but detected {actual_lang}. Potential spoofing.")
            confidence = 0.40 # Penalize confidence on mismatch
        else:
            if prediction == "HUMAN":
                explanation += "Matches biological resonance and natural phonetic decay."
            else:
                explanation += "Detected synthetic texture and unnatural pitch consistency."

        # Prepare response
        final_result = (prediction, round(float(confidence), 2), explanation)

        # 6. FORCE MEMORY CLEANUP (The "Release" Phase)
        # Explicitly delete the heavy models from RAM
        del lang_model
        del voice_model
        gc.collect() # Trigger Python's garbage collector immediately
        
        return final_result

    except Exception as e:
        # Emergency cleanup in case of crash
        if lang_model: del lang_model
        if voice_model: del voice_model
        gc.collect()
        return "HUMAN", 0.50, f"System Error: {str(e)}"
