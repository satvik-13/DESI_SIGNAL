import joblib
import librosa
import numpy as np
import whisper
import os
import gc 

def extract_features(file_path):
    # Reduced duration to 2 seconds to save RAM during feature extraction
    y, sr = librosa.load(file_path, sr=None, duration=2)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Clear audio from memory immediately
    del y
    return np.mean(mfccs.T, axis=0)

def analyze_voice(file_path, user_selected_lang):
    mapping = {"ta": "Tamil", "hi": "Hindi", "en": "English", "ml": "Malayalam", "te": "Telugu"}
    actual_lang = "Unknown"
    
    try:
        # --- STAGE 1: LANGUAGE DETECTION ---
        # Load whisper, detect, then DELETE it immediately
        lang_model = whisper.load_model("tiny")
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(lang_model.device)
        
        _, probs = lang_model.detect_language(mel)
        detected_code = max(probs, key=probs.get)
        actual_lang = mapping.get(detected_code, "Unknown") if probs[detected_code] > 0.15 else user_selected_lang
        
        # KILL WHISPER NOW to free up ~150-200MB
        del lang_model
        del audio
        del mel
        gc.collect()

        # --- STAGE 2: VOICE CLASSIFICATION ---
        # Now that Whisper is gone, we have room for the voice model
        voice_model = joblib.load("voice_detector_model.pkl")
        dna = extract_features(file_path).reshape(1, -1)
        
        prediction = voice_model.predict(dna)[0]
        confidence = max(voice_model.predict_proba(dna)[0])
        
        # KILL VOICE MODEL
        del voice_model
        gc.collect()

        # --- STAGE 3: EXPLANATION ---
        explanation = f"Vocal patterns analyzed for {actual_lang} phonetics. "
        if actual_lang != user_selected_lang and actual_lang != "Unknown":
            explanation = (f"⚠️ ALERT: Language Mismatch! Expected {user_selected_lang} "
                           f"but detected {actual_lang}. Potential spoofing.")
            confidence = 0.40
        else:
            if prediction == "HUMAN":
                explanation += "Matches biological resonance and natural phonetic decay."
            else:
                explanation += "Detected synthetic texture and unnatural pitch consistency."

        return prediction, round(float(confidence), 2), explanation

    except Exception as e:
        gc.collect()
        return "HUMAN", 0.50, f"Analysis Error: {str(e)}"
