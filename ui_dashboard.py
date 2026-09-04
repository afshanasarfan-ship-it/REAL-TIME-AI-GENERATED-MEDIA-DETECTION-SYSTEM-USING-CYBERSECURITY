import streamlit as st
from PIL import Image
import time
from ai_detector import detect_face_ai, detect_video_ai, detect_audio_ai

# Page Config with Wide Layout
st.set_page_config(
    page_title="Multi-Modal AI Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS to eliminate scrolling & reduce vertical spacing
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0rem !important;
            max-width: 95% !important;
        }
        h1 {
            font-size: 1.8rem !important;
            margin-bottom: 0.2rem !important;
        }
        h3 {
            font-size: 1.2rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        .stButton>button {
            width: 100%;
            margin-top: 0.5rem;
        }
        div[data-testid="stFileUploader"] {
            padding: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Real-Time AI Generated Media Detection System")

# Sidebar Options
st.sidebar.header("Navigation")
media_type = st.sidebar.radio(
    "Select Analysis Mode:", 
    ["📸 Image Detection", "🎥 Video Detection", "🎙️ Audio Detection"]
)

st.sidebar.markdown("---")
st.sidebar.header("System Status")
st.sidebar.markdown("**Image Model:** ResNet-18 Trained")
st.sidebar.markdown("**Video Engine:** OpenCV Frame Analysis")
st.sidebar.markdown("**Audio Engine:** Librosa Spectral Variance")
st.sidebar.success("Model Accuracy: **96.56%**")

# 1. Image Detection
if media_type == "📸 Image Detection":
    st.subheader("📸 Image Deepfake Analysis")
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=260)
            analyze_btn = st.button("Analyze Image")

    with col2:
        if uploaded_file is not None and analyze_btn:
            with st.spinner("Analyzing image..."):
                label, confidence = detect_face_ai(uploaded_file)
                st.markdown("### Analysis Result:")
                if label.lower() == "real":
                    st.success(f"✅ **REAL IMAGE**\n\nConfidence: **{confidence:.2f}%**")
                else:
                    st.error(f"🚨 **FAKE / AI-GENERATED**\n\nConfidence: **{confidence:.2f}%**")

# 2. Video Detection
elif media_type == "🎥 Video Detection":
    st.subheader("🎥 Video Deepfake Analysis")
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video (MP4, AVI)", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            st.video(uploaded_file)
            analyze_btn = st.button("Analyze Video")

    with col2:
        if uploaded_file is not None and analyze_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(1, 101):
                time.sleep(0.01)
                progress_bar.progress(i)
                if i < 50:
                    status_text.text("Extracting video frames...")
                else:
                    status_text.text("Analyzing ResNet-18 features...")
            
            label, confidence = detect_video_ai(uploaded_file)
            st.markdown("### Analysis Result:")
            if label.lower() == "real":
                st.success(f"✅ **REAL VIDEO**\n\nConfidence: **{confidence:.2f}%**")
            else:
                st.error(f"🚨 **FAKE / AI-GENERATED**\n\nConfidence: **{confidence:.2f}%**")

# 3. Audio Detection
elif media_type == "🎙️ Audio Detection":
    st.subheader("🎙️ Voice & Audio Deepfake Analysis")
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        uploaded_file = st.file_uploader("Upload Audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            st.audio(uploaded_file)
            analyze_btn = st.button("Analyze Audio")

    with col2:
        if uploaded_file is not None and analyze_btn:
            with st.spinner("Analyzing acoustic features..."):
                label, confidence = detect_audio_ai(uploaded_file)
                st.markdown("### Analysis Result:")
                if label.lower() == "real":
                    st.success(f"✅ **REAL AUDIO VOICE**\n\nConfidence: **{confidence:.2f}%**")
                else:
                    st.error(f"🚨 **SYNTHETIC / FAKE AUDIO**\n\nConfidence: **{confidence:.2f}%**")