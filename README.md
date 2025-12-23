# Traffic Violation Detection System

A real-time computer vision system for detecting traffic violations, tracking vehicles, and capturing evidence including license plates. Built with YOLOv12 for detection, SORT/ByteTrack for tracking, and MinIO for evidence storage.

## Features

- **Vehicle Detection** — Detects cars, motorcycles, buses, and trucks using YOLO
- **Object Tracking** — SORT and ByteTrack algorithms for robust real-time tracking
- **Red Light Violation Detection** — Detects vehicles crossing stop lines during red phases
- **Traffic Light Detection** — Automatic detection of Red/Green/Yellow states
- **License Plate Recognition** — YOLO detection + FastOCR for plate reading with voting system
- **Evidence Management** — Captures image crops and video clips of violations
- **MinIO Integration** — S3-compatible storage for proofs and retraining data
- **Web Dashboard** — Gradio-based UI for configuration and monitoring

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone <repository_url>
cd Object-Tracking

# 2. Start all services (MinIO + Traffic Monitor)
docker-compose up -d

# 3. Enter the container
docker exec -it traffic-monitor /bin/bash

# 4. Run the Web UI (inside container)
python app.py

# 5. Run the CLI (inside container)
python3 main.py --data_path data/traffic_video.avi --tracker bytetrack --save True
```

**Access points:**
- Web Dashboard: http://localhost:7860
- MinIO Console: http://localhost:9001 (login: `minioadmin` / `minioadmin`)

> **GPU Support:** Uncomment the `deploy` section in `docker-compose.yml` for NVIDIA GPU acceleration.

---

### Option 2: Local Installation

#### Prerequisites
- Python 3.11+
- CUDA 12.x (for GPU acceleration)
- MinIO server running locally (optional)

#### Steps

```bash
# 1. Clone the repository
git clone <repository_url>
cd Object-Tracking

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (optional, for MinIO)
cp .env.example .env
# Edit .env with your MinIO credentials

# 5. Start MinIO (if using Docker for MinIO only)
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# 6. Run the application
python app.py          # Web Dashboard
# OR
python main.py         # CLI mode
```

---

## Usage

### Web Dashboard (Gradio)

```bash
python app.py
```

Access at http://localhost:7860.

| Tab | Description |
|-----|-------------|
| **Dashboard** | View MinIO connection status and violation counts |
| **Visualization** | Start/Stop live video processing feed |
| **Zone Drawing** | Interactively draw ROI polygons and violation lines |
| **Settings** | Configure model paths, thresholds, and FPS |

### CLI Mode

```bash
python main.py --data_path data/traffic_video.avi --tracker bytetrack --save True
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Video file or RTSP URL | `data/test_video.mp4` |
| `--vehicle_model` | Path to vehicle detection weights | `models/detect_gtvn.pt` |
| `--license_model` | Path to LP detection weights | `models/lp_yolo11s.pt` |
| `--tracker` | Tracking algorithm: `sort` or `bytetrack` | `bytetrack` |
| `--light_detect` | Enable traffic light detection: `True`/`False` | `False` |
| `--save` | Save output video and CSV: `True`/`False` | `False` |
| `--device` | Device to run on: `cuda` or `cpu` | `cuda` |

---

## Configuration

### `config.yaml`

```yaml
system:
  data_path: data/test_video.mp4    # Input video/RTSP
  vehicle_model: models/detect_gtvn.pt
  license_model: models/lp_yolo11s.pt
  tracker: bytetrack                # sort or bytetrack
  device: cuda                      # cuda or cpu

detections:
  conf_threshold: 0.25
  iou_threshold: 0.5
  classes: [0, 1, 2, 3, 4]          # Vehicle classes

violation:
  fps: 60
  video_proof_duration: 3           # Seconds of video proof
  padding: 30                       # Crop padding in pixels
```

### `zones.json`

Defines detection zones. Create interactively via **Zone Drawing** tab or manually:

```json
{
  "polygon": [[x1,y1], [x2,y2], ...],           // ROI polygon
  "lines_config": {
    "violation_lines": [[x1,y1], [x2,y2]],     // Stop lines
    "left_exception_lines": [...],              // Legal left turns
    "right_exception_lines": [...]              // Legal right turns
  },
  "light_zones": {
    "straight": [[x1,y1], [x2,y2]],            // Traffic light regions
    "left": [],
    "right": []
  }
}
```

---

## Project Structure

```
Object-Tracking/
├── app.py                  # Gradio Web Dashboard
├── main.py                 # CLI entry point
├── config.yaml             # System configuration
├── zones.json              # Zone definitions
├── docker-compose.yml      # Docker services
├── core/                   # Core logic
│   ├── vehicle.py          # Vehicle class with LP voting
│   ├── violation.py        # Violation detection logic
│   ├── violation_manager.py # Centralized violation handling
│   └── license_plate_recognizer.py
├── detect/                 # YOLO inference utilities
├── track/                  # SORT and ByteTrack implementations
├── utils/                  # Helper functions (drawing, storage, etc.)
├── models/                 # YOLO weights (not included)
└── tests/                  # Unit tests
```

---

## Models

Download or train these models and place them in the `models/` directory:

| Model | Purpose | Filename |
|-------|---------|----------|
| Vehicle Detection | Detect vehicles in frame | `detect_gtvn.pt` |
| License Plate Detection | Locate plates on vehicles | `lp_yolo11s.pt` |
| Character Recognition | Read plate text (auto-downloaded) | FastOCR |

---

## Output

| Location | Content |
|----------|---------|
| `output/csv/` | Tracking data logs (local) |
| `output/video/` | Annotated output videos (local) |
| MinIO `proofs/` | Violation images and video clips |
| MinIO `retraining-data/` | Vehicle crops for model retraining |

---

## Troubleshooting

### Common Issues

**MinIO connection failed:**
```bash
# Verify MinIO is running
curl http://localhost:9000/minio/health/live

# Check environment variables
echo $MINIO_ENDPOINT  # Should be http://localhost:9000 or http://minio:9000 in Docker
```

**CUDA out of memory:**
- Reduce `imgsz` in `config.yaml` (e.g., 640 → 480)
- Use `--device cpu` for CPU-only inference

**X11 display errors in Docker:**
```bash
# Allow X11 forwarding
xhost +local:docker

# Run with display
docker-compose up -d
```

**License plate not being recognized:**
- The system uses a voting system (3 votes needed by default)
- LP detection runs every 5 frames for performance
- Check if the plate is visible in the vehicle crop

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_vehicle.py -v
```

---

## License

MIT License
