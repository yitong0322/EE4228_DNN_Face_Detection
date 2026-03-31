# EE4228 Face Detection & Recognition System

A real-time face detection and recognition system built with deep neural networks. It provides both a command-line pipeline and a web-based UI for building a face database and performing live recognition via webcam.

## Features

### Web UI (app.py)

- **Live Camera Feed** - Real-time video stream with bounding boxes and identity labels overlaid on detected faces
- **Image Upload** - Upload photos of a person to build or extend the face embedding database
- **Camera Scan** - Timed 5-second scan that captures 100+ face embeddings from the webcam; an on-screen prompt guides the user to move their head for multi-angle coverage
- **Dashboard** - Two-panel view showing known people (in database) and unknown people (scanned but unnamed); unknown entries can be named and confirmed to move them into the database
- **Database Management** - Add or remove identities through the web interface

### Command-Line Scripts

- `build_database.py` - Offline batch tool that walks a folder of labelled images and produces a pickle file of face embeddings
- `recognize.py` - Standalone real-time recognition loop using OpenCV windows (no web server)
- `utils.py` - Shared utilities: distance metrics (Euclidean, cosine), database I/O, OpenCV drawing helpers

## Tech Stack

| Layer | Technology |
|---|---|
| Face Detection | **MTCNN** (Multi-task Cascaded Convolutional Networks) via `facenet-pytorch` |
| Face Embedding | **InceptionResnetV1** (FaceNet) pretrained on VGGFace2, produces 128-dim vectors |
| Deep Learning Framework | **PyTorch** |
| Image Processing | **OpenCV**, **Pillow** |
| Identity Matching | Nearest-neighbour search with Euclidean (L2) distance |
| Web Server | **Flask** with MJPEG streaming |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Serialisation | Python `pickle` for the embedding database |

## Project Structure

```
.
├── app.py               # Flask web application (main entry point)
├── build_database.py    # Offline database builder (CLI)
├── recognize.py         # Standalone real-time recognition (CLI + OpenCV window)
├── utils.py             # Shared utilities
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Web UI template
├── database/
│   └── faces.pkl        # Saved face embedding database
└── dataset/             # Training images (one sub-folder per person)
    ├── Alice/
    │   ├── img1.jpg
    │   └── ...
    └── Bob/
        └── ...
```

## Requirements

- **Python** 3.10+
- A webcam (built-in or USB)
- (Optional) NVIDIA GPU with CUDA for faster inference

## Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd EE4228_DNN_Face_Detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install flask
   ```

   Key packages:

   - `facenet-pytorch` - MTCNN detector + FaceNet encoder
   - `torch` / `torchvision` - PyTorch runtime
   - `opencv-python` - Video capture and image processing
   - `flask` - Web server
   - `Pillow` - Image loading
   - `numpy` - Numerical operations

## Usage

### Web UI (recommended)

```bash
python app.py --camera 0 --port 5001
```

Then open **http://localhost:5001** in your browser.

| Argument | Default | Description |
|---|---|---|
| `--camera` | `1` | OpenCV camera index (`0` = built-in, `1` = first USB cam) |
| `--port` | `5001` | HTTP server port |
| `--threshold` | `1.0` | L2 distance threshold for recognition (lower = stricter) |
| `--database` | `database/faces.pkl` | Path to the embedding database file |

#### Workflow

1. **Upload Images** - Enter a person's name, select their photos, and click Upload. The system detects the face in each image, computes its embedding, and saves it to the database.
2. **Camera Scan** - Click Start Scan to run a 5-second capture session. Move your head slowly during the scan. Detected unknown faces appear in the Unknown People panel. Type a name and click Confirm to add them to the database.
3. **Live Recognition** - All known faces are recognised in the camera feed in real time with bounding boxes and distance scores.

### CLI: Build Database

```bash
python build_database.py --dataset dataset/ --output database/faces.pkl
```

Place at least 10 images per person in `dataset/<Name>/`.

### CLI: Real-Time Recognition

```bash
python recognize.py --database database/faces.pkl --camera 0
```

Press `q` to quit, `s` to save a screenshot.
