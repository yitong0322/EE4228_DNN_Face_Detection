"""
app.py
------
Flask web application integrating face database building and real-time recognition.

Usage:
    python app.py [--camera 1] [--threshold 1.0] [--port 5001]
"""

import os
import io
import time
import uuid
import base64
import argparse
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, render_template, Response, request, jsonify

from utils import (
    euclidean_distance,
    load_database,
    save_database,
    draw_face_box,
    draw_fps,
)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models (initialised once at startup)
mtcnn_single = None   # keep_all=False, for enrolment (single face per image)
mtcnn_multi = None    # keep_all=True,  for live recognition
facenet = None

# Face database: {name: [emb1, emb2, ...]}
database = {}
DATABASE_PATH = "database/faces.pkl"
THRESHOLD = 1.0

# Unknown people buffer: {uid: {"embedding": np.array, "thumbnail": base64_str, "name": ""}}
unknown_people = {}
unknown_lock = threading.Lock()

# Camera
camera = None
camera_index = 0
camera_lock = threading.Lock()

# Scanning mode
scanning_active = False
scan_lock = threading.Lock()
scan_start_time = 0
scan_frame_count = 0
SCAN_DURATION = 5          # seconds
SCAN_MIN_FRAMES = 100      # minimum embeddings to collect

# Frame skip for performance
SKIP_FRAMES = 2
frame_counter = 0
last_results = []


def init_models():
    """Load MTCNN and FaceNet models."""
    global mtcnn_single, mtcnn_multi, facenet

    mtcnn_single = MTCNN(
        image_size=160, margin=20, min_face_size=40,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        keep_all=False, post_process=True, device=device
    )

    mtcnn_multi = MTCNN(
        image_size=160, margin=20, min_face_size=40,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        keep_all=True, post_process=True, device=device
    )

    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print(f"[app] Models loaded on {device}")


def get_camera():
    """Get or open the camera."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def release_camera():
    """Release the camera resource."""
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None


# ---------------------------------------------------------------------------
# Face processing helpers
# ---------------------------------------------------------------------------

def compute_embedding_from_pil(pil_image, single=True):
    """Detect face(s) and compute embedding(s) from a PIL image.

    Args:
        pil_image: PIL RGB image.
        single: If True, return one embedding (for enrolment).
                If False, return list of (box, embedding) tuples.

    Returns:
        Single mode: np.ndarray (128,) or None
        Multi mode:  list of (box_tuple, np.ndarray) or []
    """
    if single:
        face_tensor = mtcnn_single(pil_image)
        if face_tensor is None:
            return None
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = facenet(face_tensor).squeeze().cpu().numpy()
        return emb
    else:
        boxes, _ = mtcnn_multi.detect(pil_image, landmarks=False)
        face_tensors = mtcnn_multi(pil_image)
        if boxes is None or face_tensors is None:
            return []
        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)
        face_tensors = face_tensors.to(device)
        with torch.no_grad():
            embeddings = facenet(face_tensors).cpu().numpy()
        results = []
        for i in range(len(boxes)):
            results.append((tuple(boxes[i]), embeddings[i]))
        return results


def identify_face(query_embedding, threshold):
    """Find the closest identity in the database."""
    best_name = "Unknown"
    best_distance = float("inf")
    for person_name, stored_embeddings in database.items():
        distances = [euclidean_distance(query_embedding, s) for s in stored_embeddings]
        closest = min(distances)
        if closest < best_distance:
            best_distance = closest
            best_name = person_name
    if best_distance > threshold:
        best_name = "Unknown"
    return best_name, best_distance


def crop_face_thumbnail(frame_rgb, box, size=80):
    """Crop a face region from the frame and return as base64 JPEG."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    # Add some padding
    pad = 15
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size))
    # Convert RGB to BGR for encoding
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", crop_bgr)
    return base64.b64encode(buf).decode("utf-8")


def find_matching_unknown(embedding, threshold=0.7):
    """Find an existing unknown entry that matches this embedding. Returns uid or None."""
    for uid, info in unknown_people.items():
        # Compare against the first embedding as the anchor
        dist = euclidean_distance(embedding, info["embeddings"][0])
        if dist < threshold:
            return uid
    return None


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

def generate_frames():
    """Generator that yields MJPEG frames for the video stream."""
    global frame_counter, last_results, scan_frame_count, scanning_active

    fps_start = time.time()
    fps_counter = 0
    fps = 0.0

    while True:
        with camera_lock:
            cap = get_camera()
            ret, frame = cap.read()

        if not ret:
            time.sleep(0.03)
            continue

        frame_counter += 1
        fps_counter += 1

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # Run detection every N frames
        if frame_counter % SKIP_FRAMES == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            boxes, _ = mtcnn_multi.detect(pil_image, landmarks=False)
            face_tensors = mtcnn_multi(pil_image)

            results = []
            if boxes is not None and face_tensors is not None:
                if face_tensors.ndim == 3:
                    face_tensors = face_tensors.unsqueeze(0)
                face_tensors = face_tensors.to(device)
                with torch.no_grad():
                    embeddings = facenet(face_tensors).cpu().numpy()

                for i in range(len(boxes)):
                    name, dist = identify_face(embeddings[i], THRESHOLD)
                    results.append({
                        "box": tuple(boxes[i]),
                        "name": name,
                        "distance": dist,
                        "embedding": embeddings[i],
                    })

                    # If scanning and face is unknown, collect embedding
                    if scanning_active and name == "Unknown":
                        with unknown_lock:
                            match_uid = find_matching_unknown(embeddings[i])
                            if match_uid:
                                # Add embedding to existing unknown entry
                                unknown_people[match_uid]["embeddings"].append(embeddings[i])
                            else:
                                # Create new unknown entry
                                uid = str(uuid.uuid4())[:8]
                                thumb = crop_face_thumbnail(frame_rgb, boxes[i])
                                if thumb:
                                    unknown_people[uid] = {
                                        "embeddings": [embeddings[i]],
                                        "thumbnail": thumb,
                                        "name": "",
                                    }

                        # Increment global scan frame count
                        with scan_lock:
                            scan_frame_count += 1

            # Auto-stop: after duration AND min frames met, or hard cap at 2x duration
            if scanning_active:
                elapsed_scan = time.time() - scan_start_time
                met_goal = elapsed_scan >= SCAN_DURATION and scan_frame_count >= SCAN_MIN_FRAMES
                hard_cap = elapsed_scan >= SCAN_DURATION * 2
                if met_goal or hard_cap:
                    with scan_lock:
                        scanning_active = False

            last_results = results

        # Draw results
        for r in last_results:
            draw_face_box(frame, r["box"], r["name"], r["distance"], THRESHOLD)
        draw_fps(frame, fps)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/database", methods=["GET"])
def get_database():
    """Return list of known people with their embedding counts."""
    result = []
    for name, embs in database.items():
        result.append({"name": name, "count": len(embs)})
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def upload_images():
    """Upload images for a person to add to the database.

    Expects multipart form with:
        - name: person's name
        - files: one or more image files
    """
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    added = 0
    for f in files:
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            emb = compute_embedding_from_pil(img, single=True)
            if emb is not None:
                if name not in database:
                    database[name] = []
                database[name].append(emb)
                added += 1
        except Exception as e:
            print(f"[upload] Error processing file {f.filename}: {e}")

    if added > 0:
        save_database(database, DATABASE_PATH)

    return jsonify({"name": name, "added": added, "total": len(database.get(name, []))})


@app.route("/api/scan/start", methods=["POST"])
def start_scan():
    """Start a timed scan for unknown faces from the camera."""
    global scanning_active, scan_start_time, scan_frame_count
    with scan_lock:
        scanning_active = True
        scan_start_time = time.time()
        scan_frame_count = 0
    return jsonify({"scanning": True, "duration": SCAN_DURATION, "min_frames": SCAN_MIN_FRAMES})


@app.route("/api/scan/stop", methods=["POST"])
def stop_scan():
    """Stop scanning for unknown faces."""
    global scanning_active
    with scan_lock:
        scanning_active = False
    return jsonify({"scanning": False})


@app.route("/api/scan/status", methods=["GET"])
def scan_status():
    elapsed = time.time() - scan_start_time if scanning_active else 0
    return jsonify({
        "scanning": scanning_active,
        "frame_count": scan_frame_count,
        "elapsed": round(elapsed, 1),
        "duration": SCAN_DURATION,
        "min_frames": SCAN_MIN_FRAMES,
    })


@app.route("/api/unknown", methods=["GET"])
def get_unknown():
    """Return list of unknown people with thumbnails."""
    with unknown_lock:
        result = []
        for uid, info in unknown_people.items():
            result.append({
                "uid": uid,
                "thumbnail": info["thumbnail"],
                "name": info["name"],
                "count": len(info["embeddings"]),
            })
    return jsonify(result)


@app.route("/api/unknown/<uid>/name", methods=["POST"])
def set_unknown_name(uid):
    """Assign a name to an unknown person and move them to the database."""
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    with unknown_lock:
        if uid not in unknown_people:
            return jsonify({"error": "Unknown person not found"}), 404

        embs = unknown_people[uid]["embeddings"]
        if name not in database:
            database[name] = []
        database[name].extend(embs)
        del unknown_people[uid]

    save_database(database, DATABASE_PATH)
    return jsonify({"success": True, "name": name})


@app.route("/api/unknown/<uid>", methods=["DELETE"])
def delete_unknown(uid):
    """Remove an unknown person entry."""
    with unknown_lock:
        if uid in unknown_people:
            del unknown_people[uid]
    return jsonify({"success": True})


@app.route("/api/database/<name>", methods=["DELETE"])
def delete_person(name):
    """Remove a person from the database."""
    if name in database:
        del database[name]
        save_database(database, DATABASE_PATH)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global DATABASE_PATH, THRESHOLD, camera_index, database

    parser = argparse.ArgumentParser(description="Face Recognition Web UI")
    parser.add_argument("--camera", type=int, default=1, help="Camera index")
    parser.add_argument("--threshold", type=float, default=1.0, help="Recognition threshold")
    parser.add_argument("--database", type=str, default="database/faces.pkl", help="Database path")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    args = parser.parse_args()

    camera_index = args.camera
    DATABASE_PATH = args.database
    THRESHOLD = args.threshold

    print("[app] Initialising models...")
    init_models()

    # Load existing database if available
    if os.path.exists(DATABASE_PATH):
        database = load_database(DATABASE_PATH)
    else:
        print("[app] No existing database found. Starting fresh.")

    print(f"[app] Starting server on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
