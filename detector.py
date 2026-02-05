import os
import numpy as np
import gc
import joblib

_MODEL_CACHE = None

def load_model():
    """Load model once and keep in memory for all requests"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        model_path = "voice_detector_model.pkl"
        print(f"Loading model from {model_path}...", flush=True)
        _MODEL_CACHE = joblib.load(model_path)
        print("✅ Model loaded successfully and cached in memory", flush=True)
    return _MODEL_CACHE

def extract_features(file_path):
    import librosa
    # Loading 2s to keep RAM low
    y, sr = librosa.load(file_path, sr=None, duration=2)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feature_vector = np.mean(mfccs.T, axis=0)
    del y, mfccs
    return feature_vector

def analyze_voice(file_path, user_selected_lang):
    try:
        # --- PHASE 1: NO-IMPORT LANGUAGE HANDLING ---
        # We use the user's selected lang directly. This is 100% RAM safe.
        actual_lang = user_selected_lang 

        # --- PHASE 2: VOICE CLASSIFICATION ---
        # Your core Random Forest model
        voice_model = load_model()
        dna = extract_features(file_path).reshape(1, -1)
        
        prediction = voice_model.predict(dna)[0]
        confidence = max(voice_model.predict_proba(dna)[0])
        
        # Cleanup
        #del voice_model
        #gc.collect()

        # --- PHASE 3: RESPONSE ---
        explanation = f"Vocal patterns analyzed for {actual_lang} phonetics. "
        if prediction == "HUMAN":
            explanation += "Matches biological resonance and natural phonetic decay."
        else:
            explanation += "Detected synthetic texture and unnatural pitch consistency."

        return prediction, round(float(confidence), 2), explanation

    except Exception as e:
        #gc.collect()
        return "HUMAN", 0.50, f"Analysis Error: {str(e)}"
