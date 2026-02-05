import os
import numpy as np
import gc

def extract_features(file_path):
    # Local import to keep RAM free during startup
    import librosa
    
    # Duration limited to 2 seconds to prevent memory spikes
    y, sr = librosa.load(file_path, sr=None, duration=2)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    feature_vector = np.mean(mfccs.T, axis=0)
    
    # Manual cleanup of audio arrays
    del y, mfccs
    return feature_vector

def analyze_voice(file_path, user_selected_lang):
    # Heavy imports moved inside to save ~200MB of "idle" RAM
    import whisper
    import joblib
    
    # Mapping for the 5 languages required
    mapping = {
        "ta": "Tamil", 
        "hi": "Hindi", 
        "en": "English", 
        "ml": "Malayalam", 
        "te": "Telugu"
    }
    
    actual_lang = "Unknown"
    
    try:
        # --- PHASE 1: WHISPER (Multilingual) ---
        # Using "tiny" to support Tamil, Hindi, Malayalam, and Telugu
        lang_model = whisper.load_model("tiny")
        
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(lang_model.device)
        
        # Detect language
        _, probs = lang_model.detect_language(mel)
        detected_code = max(probs, key=probs.get)
        
        # Logic to confirm if detected language is confident
        if probs[detected_code] > 0.15:
            actual_lang = mapping.get(detected_code, "Unknown")
        else:
            actual_lang = user_selected_lang

        # IMMEDIATE CLEANUP: Delete Whisper before loading the next model
        del lang_model, audio, mel
        gc.collect()

        # --- PHASE 2: CLASSIFIER
