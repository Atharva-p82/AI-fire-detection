# 🔥 Fire Guard AI – Real-Time Fire Detection & Monitoring System

**Developed and maintained by Atharva Pawar**
B.Tech Computer Science & Engineering | IIIT Raichur

Fire Guard AI is an AI-powered camera monitoring and fire detection platform designed to identify fire incidents through real-time computer vision. It combines YOLO-based object detection, multi-camera monitoring, automated alerts, and an interactive dashboard to help users monitor environments and respond to potential fire hazards.

> **Model limitation:** The currently bundled model detects Fire only. Smoke detection requires a separately validated model containing a Smoke class.

## 🌐 Live Demo

[**Launch Fire Guard AI Dashboard**](https://ai-fire-detection-drab.vercel.app/)



## ✨ Key Features

### 1. Real-Time Fire Detection

- YOLO-based fire detection through live camera feeds.
- Real-time dashboard updates using WebSockets.
- Configurable detection confidence thresholds.
- Consecutive-frame confirmation to reduce false alarms.

### 2. Multi-Camera Monitoring

- Monitor multiple webcam and IP camera streams.
- Persistent camera configurations using SQLite.
- Enable or disable individual cameras.
- Automatic reconnection when camera connections are interrupted.
- Live camera connection status indicators.

### 3. Intelligent Detection Configuration

- Dynamically adjust AI confidence thresholds.
- Configure alarm cooldown periods.
- Set detection persistence requirements.
- Configure IoU thresholds and class-specific confidence overrides.
- Define polygon-based detection regions for individual cameras.

### 4. Image Analysis

- Upload JPEG, PNG, WebP, and BMP images.
- View annotated detection results.
- Review detection confidence scores.
- Supports confidence and IoU overrides.
- Upload limits: 10 MB and 16 megapixels.

### 5. Detection History & Analytics

- Interactive weekly statistics.
- Historical fire detection records.
- Search and filter events by camera, date, type, and confidence.
- View captured incident snapshots.
- Export filtered detection history as CSV.

### 6. Automated Notifications

- Telegram notifications with incident photographs.
- Email-based incident reports.
- Per-camera notification-attempt cooldown configurable through `ai_cooldown_seconds` (research branch); event creation is separate.
- Integrated siren alerts.

## 🛠️ Technology Stack

| Component       | Technologies                 |
| --------------- | ---------------------------- |
| Frontend        | Next.js, React, Tailwind CSS |
| Backend         | Python, FastAPI              |
| Computer Vision | OpenCV                       |
| AI Model        | YOLOv8, Ultralytics          |
| Database        | SQLite                       |
| Communication   | REST API, WebSockets         |
| Visualization   | Recharts                     |
| Notifications   | Telegram, SMTP Email         |

## 🏗️ System Architecture

```text
       Camera / IP Camera
               |
               v
       Camera Management
               |
               v
       OpenCV Frame Processing
               |
               v
        YOLO Fire Detection
               |
               v
      Detection Verification
               |
               v
      Region & Cooldown Check
               |
        +------+------+
        |             |
        v             v
   Save Event     Notifications
        |          Telegram
        |          Email
        v
      SQLite
        |
        v
    FastAPI Backend
        |
     WebSockets
        |
        v
   Next.js Dashboard
```

## 💻 Installation & Setup

### Prerequisites

- Python 3.10 or above (Python 3.12 validated)
- Node.js 20.9 or above
- Git
- Webcam or accessible IP camera

### 1. Clone the Repository

```bash
git clone https://github.com/Atharva-p82/AI-fire-detection.git
cd AI-fire-detection
```

### 2. Backend Setup

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# Install CPU-compatible PyTorch
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" "torchvision==0.20.1+cpu"

# Install dependencies
pip install -r requirements.txt

# Start backend
python -m uvicorn main:app --reload --port 8000
```

Backend: [http://localhost:8000](http://localhost:8000)

### 3. Frontend Setup

Open another terminal:

```bash
cd frontend

npm ci

npm run dev
```

Frontend: [http://localhost:3000](http://localhost:3000)

### 4. Windows Quick Start

Alternatively, run:

```bash
start.bat
```

This starts the backend and frontend using the provided Windows startup script.

## 🚀 How to Use

1. Start the backend and frontend.
2. Open [http://localhost:3000](http://localhost:3000).
3. Navigate to the camera dashboard.
4. Click **Add Camera**.
5. Add a webcam using index `0` or configure an IP camera using an RTSP stream.
6. Configure AI thresholds and notification settings.
7. Monitor live camera feeds and review detection events.

## ⚙️ Configuration

For custom model loading, configure:

```powershell
$env:FIRE_MODEL_PATH="path\to\your\model.pt"
```

Check the `/model` endpoint or Settings screen to verify the classes supported by the loaded model.

For remote frontend deployments, configure:

```env
NEXT_PUBLIC_API_URL=
NEXT_PUBLIC_WS_URL=
```

Configure matching CORS origins on the backend.

## ⚡ Fast Pilot Evaluation

A short event-level pilot can be run on the public FURG fire-video dataset. See [docs/FAST_PILOT_EVALUATION.md](docs/FAST_PILOT_EVALUATION.md) for the download, XML-to-event conversion and benchmark commands.

The helper `evaluation/prepare_furg_events.py` converts the dataset's per-frame XML flame boxes into event intervals for `evaluation/run_video_benchmark.py`. The benchmark reports event-level metrics and false alerts per video-hour, not detector mAP. Because the bundled checkpoint's training-data provenance is unknown, overlap with FURG cannot be ruled out; describe results as exploratory pilot measurements, not independent generalization results.

## 🔬 Development & Testing

Backend tests:

```bash
cd backend
python -m unittest discover -s tests -v
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## 📌 Future Improvements

- Integration of a validated fire and smoke detection model.
- Improved detection accuracy in challenging lighting conditions.
- Cloud-based camera monitoring.
- Mobile notifications and remote monitoring.
- Advanced analytics and incident reporting.
- Further optimization for low-resource devices.

## 👨‍💻 Developer

**Atharva Pawar, Sagar Kumar , Dheeraj Alanka&#x20;**

B.Tech Computer Science & Engineering
Indian Institute of Information Technology, Raichur

- GitHub: [@Atharva-p82](https://github.com/Atharva-p82)
- Project Repository: [AI-fire-detection](https://github.com/Atharva-p82/AI-fire-detection)
- Live Demo: [Fire Guard ](https://ai-fire-detection-drab.vercel.app/)


## 🧪 Research Evaluation Branch

The `research-evaluation-kit` branch contains an offline experimental IoU-linked temporal verifier and a reproducible video event benchmark. The verifier is a research comparator; it is **not enabled in live camera alerting**.

- [Research evaluation protocol](docs/RESEARCH_EVALUATION_PROTOCOL.md)
- [Offline benchmark script](evaluation/run_video_benchmark.py)
- [Experimental temporal verifier](backend/experimental_tracking.py)

The benchmark compares the existing class-level persistence rule with an IoU-linked alternative on the same sampled detector outputs. It can calculate event precision, event recall, event F1, false alerts per camera-hour and detection delay when a ground-truth event CSV is provided. It does not calculate detector mAP.

No benchmark performance result is claimed until a labelled held-out video set is evaluated. The bundled checkpoint currently supports Fire only; do not claim smoke detection with it.
