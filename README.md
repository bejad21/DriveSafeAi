# DriveSafe AI

A real-time driver drowsiness and fatigue detector: point a webcam at the driver, and the app watches for closed eyes, yawning, and head tilting, then warns them before it becomes dangerous.

## What it does

DriveSafe AI runs a custom-trained TensorFlow Lite CNN and Google's MediaPipe Face Mesh on a live webcam feed to detect signs of drowsiness, fatigue, and distraction while driving. When it detects danger, it responds with on-screen alerts, spoken warnings, and an alarm, scaled to how serious the situation is.

The whole dashboard runs in Streamlit, so there's nothing to install beyond the Python dependencies. It runs locally on any machine with a webcam.

## Features

**Detection**
- Runs CNN inference on cropped eye and mouth regions every 2 frames, using MediaPipe's 468 facial landmarks to locate them without needing a GPU.
- Rolling prediction history smooths out single-frame false positives.
- Detects sustained eye closure, yawning, and head tilt (a common sign of microsleep), each with its own duration threshold.
- Automatically switches to a low-light mode using CLAHE contrast enhancement when the frame is dark.

**Alerts**

The alert system has 7 priority levels, from a simple spoken warning up to a flashing, repeating alarm:

| Priority | Condition | Mode |
|----------|-----------|------|
| 7 | Eyes closed 15+ seconds | Extreme |
| 6 | 3+ eye closure incidents | Extreme |
| 5 | Eyes closed 1+ second | Standard |
| 4 | Head tilted 15+ seconds | Extreme |
| 3 | 3+ head tilt incidents | Extreme |
| 2 | Head tilt 3+ seconds | Standard |
| 1 | Yawn or slight head roll | Vocal warning |

Extreme alerts flash on the video feed and play a spoken announcement plus an alarm sound twice before resetting. Standard alerts show a large overlay and repeat the alarm every 3 seconds until the condition clears. Once a higher-priority alert finishes, lower-priority ones are cleared too, so nothing replays.

**Dashboard**
- Live camera feed with bounding boxes on the detected face, color-coded by driver state (alert, slightly drowsy, drowsy).
- A live EAR/MAR graph overlaid on the video.
- A trip metrics panel tracking yawns, eye closures, and head tilts as they happen.
- A full incident log with charts, and a one-click CSV export.

**AI safety consultant**

A chatbot (Llama 3.3 70B via the Groq API) that has access to the current trip's incident log, so you can ask it things like "was my trip safe?" or "how many times was I drowsy?" and get an answer based on what actually happened.

## Getting started

```bash
git clone https://github.com/bejad21/DriveSafeAi.git
cd DriveSafeAi
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. The `.mp3` alert sounds are generated automatically on first run with gTTS.

## Requirements

- Python 3.9-3.11
- A webcam
- A free [Groq API key](https://console.groq.com)

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
