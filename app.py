import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import math
import threading
import os
from dotenv import load_dotenv
import pandas as pd
import urllib.request
from datetime import datetime
from gtts import gTTS
import pygame
import streamlit as st
from openai import OpenAI
from collections import deque

load_dotenv()

st.set_page_config(page_title="DriveSafe AI", page_icon="🚘", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #040C14;
    --bg-secondary: #071525;
    --bg-card: rgba(5, 18, 35, 0.85);
    --accent: #00E5FF;
    --accent-dim: rgba(0, 229, 255, 0.15);
    --accent-glow: rgba(0, 229, 255, 0.4);
    --amber: #FFB300;
    --amber-dim: rgba(255, 179, 0, 0.15);
    --danger: #FF2B4E;
    --danger-dim: rgba(255, 43, 78, 0.15);
    --success: #00E676;
    --success-dim: rgba(0, 230, 118, 0.15);
    --text-primary: #E8F4FB;
    --text-secondary: #7A9BB5;
    --border: rgba(0, 229, 255, 0.18);
    --border-bright: rgba(0, 229, 255, 0.5);
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 100, 160, 0.15), transparent),
        linear-gradient(180deg, var(--bg-primary) 0%, #020810 100%);
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0, 229, 255, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 229, 255, 0.025) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030D1A 0%, #040F1F 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.6) !important;
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: var(--text-secondary) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    padding: 6px 0 !important;
    transition: color 0.2s ease !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: var(--accent) !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] h1,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] h2,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] h3 {
    font-family: 'Orbitron', monospace !important;
    color: var(--accent) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 1.6rem !important;
    margin-bottom: 0.2rem !important;
    text-shadow: 0 0 20px var(--accent-glow), 0 0 40px rgba(0, 229, 255, 0.15) !important;
}

h2 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.05em !important;
    font-size: 1.05rem !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 8px !important;
    margin-bottom: 16px !important;
}

h3 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
}

.stButton > button {
    width: 100% !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 10px 20px !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button[kind="primary"],
.stButton > button:first-child {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.15), inset 0 0 12px rgba(0, 229, 255, 0.04) !important;
}

.stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
    box-shadow: 0 0 20px var(--accent-glow), 0 0 40px rgba(0, 229, 255, 0.1) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

[data-testid="stMetricContainer"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
    position: relative !important;
    overflow: hidden !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
}

[data-testid="stMetricContainer"]::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent, var(--accent), transparent) !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    text-shadow: 0 0 16px var(--accent-glow) !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}

.stSuccess {
    background: linear-gradient(135deg, var(--success-dim), rgba(0, 0, 0, 0.3)) !important;
    border: 1px solid rgba(0, 230, 118, 0.4) !important;
    border-radius: 8px !important;
    color: var(--success) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(0, 230, 118, 0.1) !important;
}

.stWarning {
    background: linear-gradient(135deg, var(--amber-dim), rgba(0, 0, 0, 0.3)) !important;
    border: 1px solid rgba(255, 179, 0, 0.4) !important;
    border-radius: 8px !important;
    color: var(--amber) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(255, 179, 0, 0.1) !important;
}

.stError {
    background: linear-gradient(135deg, var(--danger-dim), rgba(0, 0, 0, 0.3)) !important;
    border: 1px solid rgba(255, 43, 78, 0.4) !important;
    border-radius: 8px !important;
    color: var(--danger) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(255, 43, 78, 0.1) !important;
    animation: pulseRed 2s ease-in-out infinite !important;
}

@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 43, 78, 0.1); }
    50% { box-shadow: 0 0 30px rgba(255, 43, 78, 0.25); }
}

.stInfo {
    background: linear-gradient(135deg, var(--accent-dim), rgba(0, 0, 0, 0.3)) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
    color: var(--accent) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    font-weight: 500 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] table {
    background: var(--bg-card) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stDataFrame"] th {
    background: rgba(0, 229, 255, 0.08) !important;
    color: var(--accent) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border-bright) !important;
    padding: 12px 16px !important;
}

[data-testid="stDataFrame"] td {
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    border-bottom: 1px solid rgba(0, 229, 255, 0.06) !important;
    padding: 10px 16px !important;
}

.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(0, 230, 118, 0.5) !important;
    color: var(--success) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    border-radius: 4px !important;
    transition: all 0.25s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(0, 230, 118, 0.1) !important;
    box-shadow: 0 0 20px rgba(0, 230, 118, 0.2) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stChatInput"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg-card) !important;
    font-family: 'Rajdhani', sans-serif !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
}

[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    margin: 6px 0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
}

[data-testid="stImage"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 16px 0 !important;
}

::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 229, 255, 0.25);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 229, 255, 0.5);
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: transparent !important;
}

.stMarkdown hr {
    border-color: var(--border) !important;
}

@keyframes scanline {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}

[data-testid="stBarChart"], [data-testid="stAreaChart"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4) !important;
}

[data-baseweb="radio"] span {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}

div[data-testid="stSidebarNav"] {
    display: none;
}
</style>

<style>
.header-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(0,229,255,0.04));
    border: 1px solid rgba(0,229,255,0.35);
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(0,229,255,0.9);
    margin-bottom: 8px;
}
.page-header-rule {
    height: 1px;
    background: linear-gradient(90deg, rgba(0,229,255,0.6), rgba(0,229,255,0.1), transparent);
    margin: 4px 0 24px 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)

ALERT_CONFIG = {
    7: {"l1": "CRITICAL: EYES CLOSED > 15s", "l2": "STOP THE VEHICLE NOW!", "audio": "extreme_eyes_15.mp3", "mode": "extreme"},
    6: {"l1": "CRITICAL: MULTIPLE EYE CLOSURES", "l2": "PLEASE PULL OVER!", "audio": "extreme_eyes_mult.mp3", "mode": "extreme"},
    4: {"l1": "CRITICAL: HEAD TILT > 15s", "l2": "STOP THE VEHICLE NOW!", "audio": "extreme_tilt_15.mp3", "mode": "extreme"},
    3: {"l1": "CRITICAL: MULTIPLE HEAD TILTS", "l2": "PLEASE PULL OVER!", "audio": "extreme_tilt_mult.mp3", "mode": "extreme"},
    5: {"text": "!!! WAKE UP !!!", "audio": "eyes_emergency.mp3", "mode": "standard"},
    2: {"text": "!!! HEAD UP !!!", "audio": "tilt_emergency.mp3", "mode": "standard"},
}

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 0, 17]

if 'running' not in st.session_state: st.session_state['running'] = False
if 'yawn_count' not in st.session_state: st.session_state['yawn_count'] = 0
if 'eyes_closed_count' not in st.session_state: st.session_state['eyes_closed_count'] = 0
if 'tilt_count' not in st.session_state: st.session_state['tilt_count'] = 0
if 'history_time' not in st.session_state: st.session_state['history_time'] = []
if 'history_ear' not in st.session_state: st.session_state['history_ear'] = []
if 'history_mar' not in st.session_state: st.session_state['history_mar'] = []
if 'log_data' not in st.session_state: 
    st.session_state['log_data'] = pd.DataFrame(columns=["Timestamp", "Event Type", "Duration (sec)", "AI State"])
if 'messages' not in st.session_state: st.session_state.messages = []

class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        threading.Thread(target=self.get, args=(), daemon=True).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()

pygame.mixer.init()

def generate_tts_file(text, filename):
    if not os.path.exists(filename):
        tts = gTTS(text=text, lang='en', tld='co.uk')
        tts.save(filename)

if not os.path.exists("alarm.mp3"):
    try:
        urllib.request.urlretrieve("https://www.soundjay.com/buttons_c2026/sounds/beep-01a.mp3", "alarm.mp3")
    except Exception as e:
        print(f"Failed to download custom alarm: {e}")

generate_tts_file("Drive Safe A.I. initialized. Monitoring active.", "boot.mp3")
generate_tts_file("Warning. Fatigue detected. Please stay alert.", "warning.mp3")
generate_tts_file("WAKE UP! Eyes on the road!", "eyes_emergency.mp3")
generate_tts_file("Warning! Head Up and Stay Alert!", "tilt_emergency.mp3")
generate_tts_file("Emergency! Multiple lapses detected. PULL OVER NOW!", "extreme_eyes_mult.mp3")
generate_tts_file("DANGER! 15 second eye lapse! PULL OVER!", "extreme_eyes_15.mp3")
generate_tts_file("Emergency! Repeated head drops. STOP THE VEHICLE!.", "extreme_tilt_mult.mp3")
generate_tts_file("DANGER! 15 second head tilt! PULL OVER!", "extreme_tilt_15.mp3")

def play_audio_file(filename):
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

def play_alarm():
    for _ in range(3):
        if os.path.exists("alarm.mp3"):
            try:
                alarm_sound = pygame.mixer.Sound("alarm.mp3")
                alarm_sound.play()
                time.sleep(alarm_sound.get_length())
            except:
                pass
        time.sleep(0.1)

def play_warning_twice():
    play_audio_file("warning.mp3")
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    play_audio_file("warning.mp3")

class LiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = sorted(self.interpreter.get_output_details(), key=lambda x: x['index'])

    def predict(self, input_data):
        input_data = np.array(input_data, dtype=np.float32)
        if input_data.shape[0] != self.input_details[0]['shape'][0]:
            self.interpreter.resize_tensor_input(self.input_details[0]['index'], input_data.shape)
            self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(detail['index']) for detail in self.output_details]

@st.cache_resource
def load_cnn_model():
    cnn_model = LiteModel('drivesafe_model.tflite')
    return cnn_model

cnn_model = load_cnn_model()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False, min_detection_confidence=0.5)

clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(4,4))
DARKNESS_THRESHOLD = 80.0 

def process_feature(frame, landmarks, indices, w, h, draw=True, crop=False, scale=1.2, color=(0,255,0), padding=8):
    x_coords = [landmarks[i].x for i in indices]
    y_coords = [landmarks[i].y for i in indices]
    
    x_min_d, x_max_d = int(min(x_coords) * w) - padding, int(max(x_coords) * w) + padding
    y_min_d, y_max_d = int(min(y_coords) * h) - padding, int(max(y_coords) * h) + padding
    
    if draw:
        cv2.rectangle(frame, (x_min_d, y_min_d), (x_max_d, y_max_d), color, 2)
        
    crop_tensor = None
    if crop:
        cx = int(sum(x_coords) / len(x_coords) * w)
        cy = int(sum(y_coords) / len(y_coords) * h)
        width = (max(x_coords) - min(x_coords)) * w
        size = int(width * scale)
        x_min_c, x_max_c = cx - size // 2, cx + size // 2
        y_min_c, y_max_c = cy - size // 2, cy + size // 2
        if not (x_min_c < 0 or y_min_c < 0 or x_max_c > w or y_max_c > h):
            cropped = frame[y_min_c:y_max_c, x_min_c:x_max_c]
            try:
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                crop_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                crop_resized = cv2.resize(crop_rgb, (64, 64))
                crop_tensor = np.expand_dims(crop_resized, axis=0).astype(np.float32) 
            except Exception:
                pass
                
    return crop_tensor, (x_min_d, y_min_d, x_max_d, y_max_d)

def draw_extreme_alert(frame, w, h, line1, line2):
    color_red = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (255, 255, 255)
    text_size1 = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 4)[0]
    text_size2 = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 4)[0]
    cv2.putText(frame, line1, ((w - text_size1[0]) // 2, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_red, 4)
    cv2.putText(frame, line2, ((w - text_size2[0]) // 2, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_red, 4)

def draw_standard_alert(frame, w, h, text):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 6)[0]
    cv2.putText(frame, text, ((w - text_size[0]) // 2, (h + text_size[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 6) 

def draw_live_graph(frame, history_ear, history_mar):
    if len(history_ear) < 2: return
    h, w, _ = frame.shape
    gw, gh = 150, 60
    margin = 10
    x_off, y_off = w - gw - margin, h - gh - margin
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_off, y_off), (x_off+gw, y_off+gh), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    pts_ear = []
    pts_mar = []
    
    for i in range(len(history_ear)):
        x = x_off + int((i / max(1, len(history_ear)-1)) * gw) 
        ear = min(max(history_ear[i], 0), 1.0)
        mar = min(max(history_mar[i], 0), 1.0)
        y_e = y_off + gh - int(ear * gh)
        y_m = y_off + gh - int(mar * gh)
        pts_ear.append([x, y_e])
        pts_mar.append([x, y_m])
        
    pts_ear = np.array(pts_ear, np.int32).reshape((-1, 1, 2))
    pts_mar = np.array(pts_mar, np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(frame, [pts_ear], isClosed=False, color=(0, 255, 255), thickness=2) 
    cv2.polylines(frame, [pts_mar], isClosed=False, color=(255, 165, 0), thickness=2) 
    
    cv2.putText(frame, "EYE", (x_off + 5, y_off + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, "MOUTH", (x_off + 5, y_off + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

def log_incident(event_type, duration, state):
    new_row = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Event Type": event_type,
        "Duration (sec)": round(duration, 1),
        "AI State": state
    }
    st.session_state['log_data'] = pd.concat([st.session_state['log_data'], pd.DataFrame([new_row])], ignore_index=True)

st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 20px 0 24px 0;">
        <img src="https://cdn-icons-png.flaticon.com/512/5664/5664359.png" width="72"
             style="filter: drop-shadow(0 0 12px rgba(0,229,255,0.6)) drop-shadow(0 0 24px rgba(0,229,255,0.2));">
        <div style="margin-top: 10px; font-family: 'Orbitron', monospace; font-size: 1rem;
                    font-weight: 900; letter-spacing: 0.15em; color: #00E5FF;
                    text-shadow: 0 0 16px rgba(0,229,255,0.5);">DRIVESAFE</div>
        <div style="font-family: 'Rajdhani', sans-serif; font-size: 0.7rem; letter-spacing: 0.25em;
                    color: rgba(0,229,255,0.5); text-transform: uppercase; margin-top: 2px;">AI Monitoring System</div>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<div style='font-family: Orbitron, monospace; font-size: 0.6rem; font-weight: 700; "
    "letter-spacing: 0.2em; color: rgba(0,229,255,0.4); text-transform: uppercase; "
    "padding: 0 4px 8px 4px;'>Navigation</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio("", [
    "📊 Live Monitor", 
    "📋 Incident Analytics",
    "🤖 AI Safety Consultant" 
], label_visibility="collapsed")

st.sidebar.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='height: 1px; background: linear-gradient(90deg, transparent, rgba(0,229,255,0.3), transparent); margin: 8px 0 16px 0;'></div>",
    unsafe_allow_html=True
)

if st.sidebar.button("⬡  NEW TRIP  ⬡", use_container_width=True):
    st.session_state['running'] = False
    st.session_state['yawn_count'] = 0
    st.session_state['eyes_closed_count'] = 0
    st.session_state['tilt_count'] = 0
    st.session_state['history_time'] = []
    st.session_state['history_ear'] = []
    st.session_state['history_mar'] = []
    st.session_state['log_data'] = pd.DataFrame(columns=["Timestamp", "Event Type", "Duration (sec)", "AI State"])
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='font-family: Rajdhani, sans-serif; font-size: 0.7rem; color: rgba(0,229,255,0.3); "
    "text-align: center; letter-spacing: 0.1em;'>v2.0 · Neural Vision Core</div>",
    unsafe_allow_html=True
)

if page == "📊 Live Monitor":
    st.markdown('<div class="header-badge">⬡ Real-Time Monitoring</div>', unsafe_allow_html=True)
    st.title("DRIVESAFE  AI  //  LIVE MONITOR")
    st.markdown('<div class="page-header-rule"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        frame_placeholder = st.empty()
        
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("▶  START MONITORING", use_container_width=True):
            st.session_state['running'] = True
            st.rerun()
        if btn_col2.button("■  STOP MONITORING", use_container_width=True):
            st.session_state['running'] = False
            st.rerun()

    with col2:
        st.subheader("Driver Status")
        state_placeholder = st.empty()
        st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)
        reasoning_placeholder = st.empty()

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family: Orbitron, monospace; font-size: 0.6rem; font-weight: 700; "
            "letter-spacing: 0.2em; color: rgba(0,229,255,0.4); text-transform: uppercase; "
            "padding-bottom: 10px; border-bottom: 1px solid rgba(0,229,255,0.1); margin-bottom: 12px;'>"
            "Trip Metrics</div>",
            unsafe_allow_html=True
        )
        
        m1, m2 = st.columns(2)
        yawn_metric = m1.empty()
        eye_metric = m2.empty()
        tilt_metric = st.empty() 
        
    yawn_metric.metric("Yawns", st.session_state['yawn_count'])
    eye_metric.metric("Eye Closures", st.session_state['eyes_closed_count'])
    tilt_metric.metric("Head Tilts", st.session_state['tilt_count'])
    
    if not st.session_state['running']:
        frame_placeholder.info("⬡  Camera offline — press START MONITORING to begin.")
        state_placeholder.success("⬡  SYSTEM STANDBY")

    if st.session_state['running']:
        video_getter = VideoGet(0).start()
        
        DROWSY_TIME_THRESHOLD = 2.0
        drowsy_start_time = 0
        last_alarm_time = 0
        last_vocal_warning_time = 0
        VOCAL_WARNING_COOLDOWN = 10.0 
        
        is_yawning = False
        is_tilting = False
        is_eyes_closed = False
        
        yawn_duration = 0.0
        eyes_closed_duration = 0.0
        tilt_duration = 0.0
        
        eyes_closed_start_time = 0
        tilt_start_time = 0
        yawn_start_time = 0        
        
        YAWN_TIME_THRESHOLD = 1.2  
        TILT_ROLL_THRESHOLD = 22.0 
        TILT_CRITICAL_TIME = 3.0 
        
        frame_counter = 0            
        
        EAR_CONSECUTIVE_FRAMES = 4 
        ear_closed_counter = 0
        ear_open_counter = 0

        eyes_counted = False
        tilt_counted = False

        active_extreme_priority = 0
        extreme_play_count = 0
        handled_eyes_count = st.session_state['eyes_closed_count']
        handled_tilt_count = st.session_state['tilt_count']
        
        start_session_time = time.time()
        prev_frame_time = 0
        new_frame_time = 0
        last_ui_update_time = time.time()
        UI_UPDATE_INTERVAL = 0.1 
        
        state = "ALERT"
        color = (0, 255, 0)
        reasoning = "Driver is attentive."

        HISTORY_LENGTH = 5
        l_pred_history = deque(maxlen=HISTORY_LENGTH)
        r_pred_history = deque(maxlen=HISTORY_LENGTH)
        m_pred_history = deque(maxlen=HISTORY_LENGTH)
        smooth_eye_avg = 1.0
        smooth_m_avg = 0.0
        CNN_RUN_EVERY = 2

        while not video_getter.stopped and st.session_state['running']:
            frame = video_getter.frame
            if frame is None:
                continue
            
            orig_h, orig_w = frame.shape[:2]
            new_h = 360
            new_w = int(orig_w * (new_h / orig_h))
            frame = cv2.resize(frame, (new_w, new_h))

            new_frame_time = time.time()
            frame_counter += 1
            h, w, _ = frame.shape
            
            rgb_clean = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = rgb_clean.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_dark = np.mean(gray) < DARKNESS_THRESHOLD
            
            if is_dark:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                lab = cv2.merge((clahe.apply(l), a, b))
                logic_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                display_frame = cv2.cvtColor(logic_frame, cv2.COLOR_BGR2RGB)
                cv2.putText(display_frame, "NIGHT VISION", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                logic_frame = frame  

            if frame_counter < 10:
                cv2.putText(display_frame, "Initializing Camera...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                bgr_display = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', bgr_display, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                frame_placeholder.image(buffer.tobytes(), use_container_width=True)
                continue

            if frame_counter == 10:
                threading.Thread(target=play_audio_file, args=("boot.mp3",), daemon=True).start()

            results = face_mesh.process(cv2.cvtColor(logic_frame, cv2.COLOR_BGR2RGB))

            if results and results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lms = face_landmarks.landmark

                    left_eye_crop, left_coords = process_feature(logic_frame, lms, LEFT_EYE_INDICES, w, h, draw=False, crop=True)
                    right_eye_crop, right_coords = process_feature(logic_frame, lms, RIGHT_EYE_INDICES, w, h, draw=False, crop=True)
                    mouth_crop, mouth_coords = process_feature(logic_frame, lms, MOUTH_INDICES, w, h, draw=False, crop=True, scale=1.5)

                    if frame_counter % CNN_RUN_EVERY == 0:
                        crops_to_process = []
                        has_eyes = False
                        has_mouth = False
                        
                        if left_eye_crop is not None and right_eye_crop is not None:
                            crops_to_process.extend([left_eye_crop[0], right_eye_crop[0]])
                            has_eyes = True
                            
                        if mouth_crop is not None:
                            crops_to_process.append(mouth_crop[0])
                            has_mouth = True
                            
                        if crops_to_process:
                            batch = np.array(crops_to_process, dtype=np.float32)
                            preds = cnn_model.predict(batch)
                            
                            eye_out = preds[0]
                            yawn_out = preds[1]
                            
                            if has_eyes:
                                l_pred_history.append(float(eye_out[0][0]))
                                r_pred_history.append(float(eye_out[1][0]))
                                
                            if has_mouth:
                                m_idx = 2 if has_eyes else 0
                                m_pred = float(yawn_out[m_idx][0])
                                m_pred_history.append(m_pred)

                    if len(l_pred_history) > 0 and len(r_pred_history) > 0:
                        smooth_eye_avg = (sum(l_pred_history) + sum(r_pred_history)) / (len(l_pred_history) + len(r_pred_history))
                    if len(m_pred_history) > 0:
                        smooth_m_avg = sum(m_pred_history) / len(m_pred_history)

                    current_t = time.time() - start_session_time
                    st.session_state['history_time'].append(current_t)
                    st.session_state['history_ear'].append(smooth_eye_avg)
                    st.session_state['history_mar'].append(smooth_m_avg)
                    if len(st.session_state['history_time']) > 100: 
                        st.session_state['history_time'].pop(0)
                        st.session_state['history_ear'].pop(0)
                        st.session_state['history_mar'].pop(0)

                    dy = lms[263].y - lms[33].y
                    dx = lms[263].x - lms[33].x
                    roll_angle = abs(math.degrees(math.atan2(dy, dx)))
                    
                    if smooth_m_avg > 0.5:
                        if yawn_start_time == 0: yawn_start_time = time.time()
                        yawn_duration = time.time() - yawn_start_time
                        if yawn_duration > YAWN_TIME_THRESHOLD:
                            if not is_yawning:
                                st.session_state['yawn_count'] += 1
                                is_yawning = True
                    else:
                        if is_yawning: log_incident("Sustained Yawn", yawn_duration, "SLIGHTLY DROWSY")
                        yawn_start_time = 0; yawn_duration = 0.0; is_yawning = False

                    if roll_angle > TILT_ROLL_THRESHOLD:
                        if tilt_start_time == 0: 
                            tilt_start_time = time.time()
                            tilt_counted = False
                        tilt_duration = time.time() - tilt_start_time
                        
                        if not is_tilting:
                            is_tilting = True
                            
                        if tilt_duration > 2.0 and not tilt_counted:
                            st.session_state['tilt_count'] += 1
                            tilt_counted = True
                    else:
                        if is_tilting and tilt_duration > 2.0: 
                            log_incident("Head Tilt", tilt_duration, "DROWSY" if tilt_duration > TILT_CRITICAL_TIME else "SLIGHTLY DROWSY")
                        tilt_start_time = 0; tilt_duration = 0.0; is_tilting = False; tilt_counted = False

                    if smooth_eye_avg < 0.5:
                        ear_closed_counter += 1
                        ear_open_counter = 0 
                        if ear_closed_counter >= EAR_CONSECUTIVE_FRAMES:
                            if not is_eyes_closed:
                                is_eyes_closed = True
                                eyes_closed_start_time = time.time() 
                                eyes_counted = False
                            eyes_closed_duration = time.time() - eyes_closed_start_time
                            
                            if eyes_closed_duration > 2.0 and not eyes_counted:
                                st.session_state['eyes_closed_count'] += 1
                                eyes_counted = True
                    else:
                        ear_open_counter += 1
                        ear_closed_counter = 0 
                        if ear_open_counter >= EAR_CONSECUTIVE_FRAMES: 
                            if is_eyes_closed and eyes_closed_duration > 2.0: 
                                log_incident("Eyes Closed", eyes_closed_duration, state)
                            is_eyes_closed = False
                            eyes_closed_start_time = 0; eyes_closed_duration = 0.0; eyes_counted = False 

                    current_priority = 0
                    cond_eyes_15 = (eyes_closed_duration >= 15.0)
                    cond_eyes_mult = (st.session_state['eyes_closed_count'] - handled_eyes_count >= 3)
                    cond_eyes_norm = (is_eyes_closed and eyes_closed_duration >= 1.0)
                    
                    cond_tilt_15 = (tilt_duration >= 15.0)
                    cond_tilt_mult = (st.session_state['tilt_count'] - handled_tilt_count >= 3)
                    cond_tilt_norm = (is_tilting and tilt_duration > TILT_CRITICAL_TIME)
                    
                    cond_yawn = is_yawning
                    cond_tilt_slight = (is_tilting and tilt_duration <= TILT_CRITICAL_TIME)

                    if cond_eyes_15: current_priority = 7
                    elif cond_eyes_mult: current_priority = 6
                    elif cond_eyes_norm: current_priority = 5
                    elif cond_tilt_15: current_priority = 4
                    elif cond_tilt_mult: current_priority = 3
                    elif cond_tilt_norm: current_priority = 2
                    elif cond_yawn or cond_tilt_slight: current_priority = 1

                    if current_priority in [3, 4, 6, 7]:
                        if current_priority > active_extreme_priority:
                            active_extreme_priority = current_priority
                            extreme_play_count = 0

                    priority = max(current_priority, active_extreme_priority)

                    if priority >= 5:
                        state = "DROWSY"
                        color = (255, 0, 0) 
                        if priority == 7: reasoning = f"Critical: Eyes shut for 15+s"
                        elif priority == 6: reasoning = f"Critical: Multiple eye closures"
                        else: reasoning = f"Critical: Eyes shut for {eyes_closed_duration:.1f}s"
                    elif priority >= 2:
                        state = "DROWSY"
                        color = (255, 0, 0)
                        if priority == 4: reasoning = f"Critical: Head tilted for 15+s"
                        elif priority == 3: reasoning = f"Critical: Multiple head tilts"
                        else: reasoning = f"Critical: Head tilted for {tilt_duration:.1f}s"
                    elif priority == 1:
                        state = "SLIGHTLY DROWSY"
                        color = (255, 255, 0) 
                        if cond_yawn: reasoning = f"Fatigue: Sustained yawn ({yawn_duration:.1f}s)"
                        else: reasoning = f"Fatigue: Head roll detected ({tilt_duration:.1f}s)"
                    else:
                        state = "ALERT"
                        color = (0, 255, 0) 
                        reasoning = "Driver is attentive."

                    cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1) 
                    cv2.putText(display_frame, f"STATE: {state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if priority != 5 and priority != 2:
                        drowsy_start_time = 0

                    config = ALERT_CONFIG.get(priority)
                    if config:
                        if config["mode"] == "extreme":
                            draw_extreme_alert(display_frame, w, h, config["l1"], config["l2"])
                            
                            if not pygame.mixer.music.get_busy():
                                if extreme_play_count < 2:
                                    play_audio_file(config["audio"])
                                    threading.Thread(target=play_alarm, daemon=True).start()
                                    extreme_play_count += 1
                                else:
                                    handled_eyes_count = st.session_state['eyes_closed_count']
                                    handled_tilt_count = st.session_state['tilt_count']
                                    active_extreme_priority = 0
                                    extreme_play_count = 0
                                    
                        elif config["mode"] == "standard":
                            if priority == 2:
                                draw_standard_alert(display_frame, w, h, config["text"])
                                if time.time() - last_alarm_time > 3.0:
                                    threading.Thread(target=play_audio_file, args=(config["audio"],), daemon=True).start()
                                    threading.Thread(target=play_alarm, daemon=True).start()
                                    last_alarm_time = time.time()
                            else:
                                if drowsy_start_time == 0:
                                    drowsy_start_time = time.time()
                                elif time.time() - drowsy_start_time > DROWSY_TIME_THRESHOLD:
                                    draw_standard_alert(display_frame, w, h, config["text"])
                                    if time.time() - last_alarm_time > 3.0:
                                        threading.Thread(target=play_audio_file, args=(config["audio"],), daemon=True).start()
                                        threading.Thread(target=play_alarm, daemon=True).start()
                                        last_alarm_time = time.time()
                                    
                    elif priority == 1:
                        if cond_yawn and time.time() - last_vocal_warning_time > VOCAL_WARNING_COOLDOWN:
                            if not pygame.mixer.music.get_busy():
                                threading.Thread(target=play_warning_twice, daemon=True).start()
                                last_vocal_warning_time = time.time()

                    cv2.rectangle(display_frame, (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3]), color, 2)
                    cv2.rectangle(display_frame, (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3]), color, 2)
                    cv2.rectangle(display_frame, (mouth_coords[0], mouth_coords[1]), (mouth_coords[2], mouth_coords[3]), color, 2)
            
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
        
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            draw_live_graph(display_frame, st.session_state['history_ear'], st.session_state['history_mar'])
            
            bgr_display = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', bgr_display, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame_placeholder.image(buffer.tobytes(), use_container_width=True)
            
            current_ui_time = time.time()
            if current_ui_time - last_ui_update_time > UI_UPDATE_INTERVAL:
                if state == "ALERT": state_placeholder.success(f"⬡  STATE: {state}")
                elif state == "SLIGHTLY DROWSY": state_placeholder.warning(f"⬡  STATE: {state}")
                else: state_placeholder.error(f"⬡  STATE: {state}")
                reasoning_placeholder.info(f"⬡  AI Logic: {reasoning}")
                yawn_metric.metric("Yawns", st.session_state['yawn_count'], delta=f"{yawn_duration:.1f}s live" if is_yawning else None, delta_color="inverse")
                eye_metric.metric("Eye Closures", st.session_state['eyes_closed_count'], delta=f"{eyes_closed_duration:.1f}s live" if (is_eyes_closed and eyes_closed_duration >= 1.0) else None, delta_color="inverse")
                tilt_metric.metric("Head Tilts", st.session_state['tilt_count'], delta=f"{tilt_duration:.1f}s live" if is_tilting else None, delta_color="inverse")
                
                last_ui_update_time = current_ui_time

        video_getter.stop()

elif page == "📋 Incident Analytics":
    st.markdown('<div class="header-badge">⬡ Post-Trip Analysis</div>', unsafe_allow_html=True)
    st.title("INCIDENT LOG  //  SESSION ANALYTICS")
    st.markdown('<div class="page-header-rule"></div>', unsafe_allow_html=True)

    df = st.session_state['log_data']
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇  EXPORT TRIP LOG  //  CSV",
        data=csv,
        file_name=f'drivesafe_log_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv'
    )
    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=250)
    if not df.empty:
        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Event Frequency")
            st.bar_chart(df['Event Type'].value_counts(), color="#00E5FF")
        with c2:
            st.subheader("Event Duration Over Time")
            st.area_chart(df.set_index('Timestamp')['Duration (sec)'], color="#FF2B4E")
    else:
        st.info("⬡  No incidents recorded yet. Drive safely!")

elif page == "🤖 AI Safety Consultant":
    st.markdown('<div class="header-badge">⬡ Neural Language Interface</div>', unsafe_allow_html=True)
    st.title("AI SAFETY  CONSULTANT")
    st.markdown('<div class="page-header-rule"></div>', unsafe_allow_html=True)
    st.info("⬡  Powered by Llama 3 via Groq — ultra-fast safety intelligence.")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1", 
        api_key=os.getenv("GROQ_API_KEY")
    )

    df = st.session_state['log_data']
    log_context = df.tail(15).to_string(index=False) if not df.empty else "No incidents logged yet."
    system_prompt = f"You are the 'DriveSafe AI Assistant'. Trip Data: Eyes Closed: {st.session_state['eyes_closed_count']}, Yawns: {st.session_state['yawn_count']}, Tilts: {st.session_state['tilt_count']}. Log: {log_context}"
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about your trip safety data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=api_messages)
                full_res = res.choices[0].message.content
                st.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
            except Exception as e:
                st.error(f"⬡  Connection error. Verify your Groq API key. Details: {e}")