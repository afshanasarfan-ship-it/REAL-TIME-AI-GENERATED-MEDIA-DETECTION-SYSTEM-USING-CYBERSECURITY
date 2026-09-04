import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt

def process_media(file_path, media_type):
    if media_type == "image":
        return process_image_face(file_path)
    elif media_type == "video":
        return process_video_frames(file_path)
    elif media_type == "audio":
        return process_audio_spectrogram(file_path)
    return []

def process_image_face(file_path):
    img = cv2.imread(file_path)
    if img is None:
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop Face using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        face = img_rgb[y:y+h, x:x+w]
        cropped_faces.append(cv2.resize(face, (224, 224)))
        
    if not cropped_faces:
        return [cv2.resize(img_rgb, (224, 224))]
    return cropped_faces

def process_video_frames(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < 8:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, (224, 224)))
        frame_count += 1
    cap.release()
    return frames

def process_audio_spectrogram(file_path):
    """Converts Audio Waveform into a Mel-Spectrogram Image"""
    try:
        y, sr = librosa.load(file_path, duration=5.0)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Save temporary spectrogram plot
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(S_dB, aspect='auto', origin='lower', cmap='magma')
        ax.axis('off')
        
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        return [cv2.resize(rgb, (224, 224))]
    except Exception:
        return [np.zeros((224, 224, 3), dtype=np.uint8)]