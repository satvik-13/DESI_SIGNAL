import streamlit as st
import requests
import base64
import os
import pandas as pd
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="DESI SIGNAL | AI Voice Guard", page_icon="🛡️", layout="wide")

# 2. Spectrogram Function
def plot_spectrogram(audio_bytes, is_human):
    with open("temp_plot.mp3", "wb") as f:
        f.write(audio_bytes)
    y, sr = librosa.load("temp_plot.mp3", duration=5)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    cmap_choice = 'Greens' if is_human else 'Reds'
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap=cmap_choice)
    fig.patch.set_facecolor('#0e1117') 
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_title(f"Vocal Frequency Fingerprint ({'Human' if is_human else 'AI'})", color='white')
    return fig

# 3. Enhanced Custom CSS (Fonts, Watermark, & Styling)
# 3. Enhanced Custom CSS (Fonts, Watermark, & Styling)
logo_path = "logo.png"
logo_base64 = ""

# Only try to load the logo if it exists to avoid FileNotFoundError
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@500;700&display=swap');

    /* Translucent Background Watermark */
    .stApp {{
        background-image: linear-gradient(rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 0.85))
        {f', url("data:image/png;base64,{logo_base64}")' if logo_base64 else ""};
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        background-size: 30%; 
    }}

    /* Main Title Styling */
    .main-title {{
        font-family: 'Orbitron', sans-serif;
        color: #FF4B4B;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 0px 0px 15px rgba(255, 75, 75, 0.6);
        margin-bottom: 10px;
    }}

    /* Sub-headers Styling */
    h3 {{
        font-family: 'Rajdhani', sans-serif;
        color: #ffffff;
        font-size: 1.8rem;
        border-left: 4px solid #FF4B4B;
        padding-left: 15px;
        text-transform: uppercase;
    }}

    .stButton>button {{ 
        width: 100%; 
        border-radius: 10px; 
        height: 3.5em; 
        background-color: #FF4B4B; 
        color: white; 
        font-family: 'Orbitron', sans-serif;
        font-weight: bold; 
        transition: 0.3s ease;
    }}
    </style>
    """, unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# 4. Sidebar with Logo

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.warning("Logo not found. Place logo.png in the folder.")
    st.markdown("<h2 style='text-align: center; font-family: Orbitron; color: white;'>DESI SIGNAL</h2>", unsafe_allow_html=True)
    st.info("Model: Random Forest\n\nTarget: 5 Indic Languages")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# 5. Attractive Main Title
st.markdown("<div class='main-title'>DESI SIGNAL</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-family: Rajdhani; font-size: 1.2rem; margin-top: -15px;'>AI-POWERED VOICE AUTHENTICATION</p>", unsafe_allow_html=True)

# 6. Layout Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Provide Audio Source")
    input_method = st.radio("Choose Input Method:", ["Upload File", "Record Live"], horizontal=True)
    
    final_audio = None
    if input_method == "Upload File":
        final_audio = st.file_uploader("Upload an audio clip...", type=['mp3', 'wav', 'm4a'])
    else:
        final_audio = st.audio_input("Record your voice live")

    if final_audio:
        st.markdown("#### **🔈 Playback Preview**")
        st.audio(final_audio)

    language = st.selectbox("Select Language Context", ["English", "Hindi", "Tamil", "Telugu", "Malayalam"])
    analyze_btn = st.button("🚀 ANALYZE VOICE")

with col2:
    st.markdown("### Live Analysis")
    
    if analyze_btn and final_audio:
        with st.spinner("Decoding vocal DNA..."):
            try:
                audio_bytes = final_audio.getvalue()
                base64_audio = base64.b64encode(audio_bytes).decode()
                payload = {"language": language, "audioFormat": "mp3", "audioBase64": base64_audio}
                headers = {"x-api-key": "sk_desi_9988776655"}
                
                # Change this line in app.py:
                # Use your active Ngrok URL here
                response = requests.post("https://excitedly-handier-elke.ngrok-free.dev/api/voice-detection", json=payload, headers=headers)
                res = response.json()
                
                if res["status"] == "success":
                    is_human = (res["classification"] == "HUMAN")
                    color = "#238636" if is_human else "#da3633"
                    
                    st.markdown(f"""
                        <div class='status-card'>
                            <h2 style='color: {color}; text-align: center; font-family: Orbitron;'>{res["classification"]}</h2>
                            <p style='text-align: center; font-family: Rajdhani;'>Confidence: {int(res["confidenceScore"]*100)}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### **📝 AI Explanation**")
                    st.markdown(f"<div class='explanation-box'>{res.get('explanation', 'No explanation provided.')}</div>", unsafe_allow_html=True)

                    st.session_state.history.insert(0, {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Language": language,
                        "Result": res["classification"],
                        "Confidence": f"{int(res['confidenceScore']*100)}%"
                    })

                    with st.expander("🔍 VIEW VOCAL SPECTROGRAM", expanded=True):
                        st.pyplot(plot_spectrogram(audio_bytes, is_human))
                else:
                    st.error(f"Error: {res.get('message')}")
            except Exception as e:
                st.error(f"System Error: {e}")
    else:
        st.write("Awaiting audio input...")

# 7. History Table
st.markdown("---")
st.markdown("### 📜 Detection History")
if st.session_state.history:
    st.table(pd.DataFrame(st.session_state.history))
else:
    st.caption("No scans performed yet.")

st.caption("Developed for the India AI Impact Buildathon 2026. Powered by Random Forest & Librosa.")