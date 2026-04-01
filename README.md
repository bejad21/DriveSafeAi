# 🚘 DriveSafe AI

> **Real-time driver drowsiness and fatigue detection powered by deep learning, computer vision, and a large language model safety consultant — built on Streamlit.**

---

## Overview

DriveSafe AI is a real-time driver monitoring system that uses your webcam, a custom-trained TensorFlow Lite CNN, and Google's MediaPipe Face Mesh to detect signs of drowsiness, fatigue, and distraction while driving. When danger is detected, the system responds immediately with on-screen visual alerts, audio warnings via text-to-speech, and an alarm sound — scaled to the severity of the situation.

The dashboard is built entirely with **Streamlit**, giving it a web-based, zero-installation UI that runs locally on any machine with a webcam.

---

## Features

### 🧠 Neural Vision Core
- Custom **TensorFlow Lite CNN** (`drivesafe_model.tflite`) runs inference on cropped eye and mouth regions every 2 frames for efficiency.
- **MediaPipe Face Mesh** provides 468 facial landmarks for precise feature localisation without needing a GPU.
- **Temporal smoothing** with a rolling prediction history prevents single-frame false positives.

### 👁️ Eye Closure Detection
- Detects sustained eye closure using CNN confidence scores averaged across both eyes.
- Escalates through a tiered priority system based on duration and frequency.

### 😮 Yawn Detection
- Detects sustained mouth-open events exceeding a configurable time threshold.
- Triggers a vocal warning (repeated twice) after each confirmed yawn.

### 🤕 Head Tilt Detection
- Computes facial roll angle from landmark geometry in real time.
- Detects lateral head drops toward the shoulder — a key indicator of microsleep.
- Alert fires immediately when the tilt exceeds the critical threshold (3 seconds).

### 🌙 Night Vision Mode
- Automatically detects low-light conditions using frame brightness analysis.
- Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) in the LAB colour space for improved accuracy in the dark.

### 🚨 7-Level Priority Alert System
| Priority | Condition | Mode |
|----------|-----------|------|
| 7 | Eyes closed ≥ 15 seconds | Extreme |
| 6 | 3+ eye closure incidents | Extreme |
| 5 | Eyes closed ≥ 1 second | Standard |
| 4 | Head tilted ≥ 15 seconds | Extreme |
| 3 | 3+ head tilt incidents | Extreme |
| 2 | Head tilt ≥ 3 seconds | Standard |
| 1 | Yawn or slight head roll | Vocal warning |

- **Extreme alerts** flash critical text on the camera feed and play a TTS announcement + alarm sound twice before auto-resetting.
- **Standard alerts** display a large overlay warning and repeat the alarm every 3 seconds while the condition persists.
- Once a higher-priority alert completes, all lower-priority conditions are fully cleared — no replaying of old alerts.

### 📊 Live Dashboard
- Real-time camera feed with bounding boxes drawn around detected facial features, colour-coded by driver state.
- Live **EAR/MAR mini-graph** overlaid directly on the video feed.
- Trip metrics panel showing live counts of yawns, eye closures, and head tilts with live duration deltas.
- Driver state indicator: **ALERT** / **SLIGHTLY DROWSY** / **DROWSY** with matching colour and glow.

### 📋 Incident Analytics
- Full incident log table with timestamp, event type, duration, and AI state at time of event.
- Bar chart of event frequency breakdown.
- Area chart of event durations over the session timeline.
- One-click **CSV export** of the complete trip log.

### 🤖 AI Safety Consultant
- Conversational AI chatbot powered by **Llama 3.3 70B** via the Groq API.
- Automatically given context about the current trip (incident counts and full log) so it can give personalised safety advice.
- Ask questions like *"Was my trip safe?"*, *"How many times was I drowsy?"*, or *"What should I do after this kind of trip?"*

---

## Project Structure

```
drivesafe-ai/
├── app.py                    # Main Streamlit application
├── drivesafe_model.tflite    # Trained TensorFlow Lite CNN model
├── requirements.txt          # Python dependencies
├── .env                      # Your API key
└── README.md
```

> **Note:** The `.mp3` audio files (`boot.mp3`, `warning.mp3`, etc.) are **auto-generated** at first run using gTTS and saved locally.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/bejad21/DriveSafeAi.git
cd drivesafe-ai
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
In your project's root directory, create a new file named `.env` and add the following line and replace the placeholder:
```
GROQ_API_KEY=your_actual_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run the app
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Requirements

- Python 3.9 – 3.11
- A working webcam
- A [Groq API key](https://console.groq.com) (free tier is sufficient)
- `drivesafe_model.tflite` model file

---

## Built With

| Technology | Role |
|------------|------|
| [Streamlit](https://streamlit.io) | Web dashboard and UI framework |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | On-device CNN inference |
| [MediaPipe](https://mediapipe.dev) | Real-time facial landmark detection |
| [OpenCV](https://opencv.org) | Video capture and frame processing |
| [gTTS](https://gtts.readthedocs.io) | Text-to-speech audio generation |
| [Pygame](https://www.pygame.org) | Audio playback |
| [Groq + Llama 3](https://groq.com) | AI safety consultant chatbot |
| [Pandas](https://pandas.pydata.org) | Incident logging and analytics |

## License

This project is for educational and personal use.
