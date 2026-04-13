"""
recognize.py
------------
Phase 2 (live demo): Real-time face detection and recognition using a webcam.

Usage:
    python recognize.py --database database/faces.pkl

What this script does every frame:
    1. Read one frame from the webcam with OpenCV.
    2. Convert the BGR frame to RGB (MTCNN expects RGB).
    3. Run MTCNN on the full frame → list of bounding boxes + aligned face crops.
    4. For each detected face, compute its embedding with FaceNet.
    5. Compare the embedding against every entry in the database.
    6. If the closest match is within the threshold → display the person's name.
       Otherwise → display "Unknown".
    7. Overlay bounding boxes and labels with OpenCV, then show the frame.
    8. Repeat until the user presses 'q'.

Press 'q' to quit.
Press 's' to save a screenshot of the current frame.
"""

import os
import time
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import (
    euclidean_distance,
    load_database,
    draw_face_box,
    draw_fps,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time face recognition using MTCNN + FaceNet."
    )
    parser.add_argument(
        "--database", type=str, default="database/faces.pkl",
        help="Path to the embedding database built by build_database.py."
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help=(
            "Euclidean distance threshold for identity decision. "
            "Faces with distance < threshold are labelled with the matched name; "
            "faces above the threshold are labelled 'Unknown'. "
            "Typical range: 0.8 (strict) to 1.2 (lenient)."
        )
    )
    parser.add_argument(
        "--camera", type=int, default=1,
        help="OpenCV camera index (0 = default built-in webcam, 1 = first USB cam)."
    )
    parser.add_argument(
        "--skip-frames", type=int, default=3,
        help=(
            "Run MTCNN+FaceNet only every N frames to reduce CPU/GPU load. "
            "Between inference frames the last detection result is re-used. "
            "Set to 1 to run on every frame (slower but more responsive)."
        )
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model initialisation (shared with build_database.py)
# ---------------------------------------------------------------------------

def load_models(device: torch.device, min_face_size: int = 80):
    """
    Initialise MTCNN and FaceNet for inference.

    In the live recognition phase MTCNN is configured with keep_all=True so
    that it returns *all* faces visible in the frame — we want to detect and
    label every person present, not just the most confident one.

    Args:
        device:        torch.device (CPU or CUDA).
        min_face_size: Smallest face dimension (px) that MTCNN will detect.

    Returns:
        Tuple (mtcnn, facenet).
    """
    # keep_all=True: return every face found, not just the highest-confidence one
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=min_face_size,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.85,            # larger factor → fewer pyramid levels → faster
        keep_all=True,          # ← differs from build_database.py
        post_process=True,
        device=device
    )

    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    return mtcnn, facenet


# ---------------------------------------------------------------------------
# Identity matching
# ---------------------------------------------------------------------------

def identify_face(
    query_embedding: np.ndarray,
    database: dict,
    threshold: float
) -> tuple[str, float]:
    """
    Compare a single query embedding against all stored embeddings and return
    the best matching identity.

    Strategy — nearest neighbour search:
        For each person in the database, compute the Euclidean distance between
        the query embedding and every one of that person's stored embeddings,
        then take the *minimum* (closest match) as the representative distance
        for that person.

        The person with the overall lowest distance wins — provided that
        distance is below `threshold`. If every distance exceeds the threshold
        the face is declared "Unknown".

    Why minimum and not mean?
        Using the minimum (best-case match) is more permissive and handles the
        situation where most enrolled photos are a poor angle but one happens to
        match the live feed well. Averaging would dilute that good match with
        noise from bad angles.

    Args:
        query_embedding: 128-dim numpy array of the face to identify.
        database:        {name: [emb_1, emb_2, ...]} dictionary.
        threshold:       Maximum Euclidean distance to still call a match.

    Returns:
        Tuple (predicted_name, best_distance)
        predicted_name is "Unknown" when no stored embedding is close enough.
    """
    best_name     = "Unknown"
    best_distance = float("inf")   # start at infinity; lower is better

    for person_name, stored_embeddings in database.items():
        if not stored_embeddings:
            continue

        # Compute L2 distance to every stored embedding for this person
        distances = [
            euclidean_distance(query_embedding, stored_emb)
            for stored_emb in stored_embeddings
        ]

        # Take the closest (minimum) distance as this person's score
        closest = min(distances)

        if closest < best_distance:
            best_distance = closest
            best_name = person_name

    # Apply the threshold: if even the best match is too far away, say "Unknown"
    if best_distance > threshold:
        best_name = "Unknown"

    return best_name, best_distance


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(
    frame_bgr: np.ndarray,
    mtcnn: MTCNN,
    facenet: InceptionResnetV1,
    database: dict,
    threshold: float,
    device: torch.device
) -> list[dict]:
    """
    Run the full detection → embedding → matching pipeline on one video frame.

    Args:
        frame_bgr: Raw BGR image from OpenCV (H × W × 3).
        mtcnn:     Initialised MTCNN model (keep_all=True).
        facenet:   Initialised FaceNet model.
        database:  Loaded face embedding database.
        threshold: Identity decision boundary.
        device:    Compute device.

    Returns:
        List of result dictionaries, one per detected face:
            {
                "box":      (x1, y1, x2, y2),
                "name":     "Alice" | "Unknown",
                "distance": float
            }
    """
    results = []

    # ── Step 1: Convert BGR (OpenCV) → RGB (PIL/MTCNN expected format) ──────
    # OpenCV stores channels as Blue-Green-Red; most deep learning models
    # expect Red-Green-Blue. Forgetting this swap is a common bug that reduces
    # accuracy because the colour channels are misinterpreted.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # ── Step 2: MTCNN — detect all faces in the frame ───────────────────────
    # boxes  : tensor of shape (N, 4) with (x1, y1, x2, y2) in pixel coords
    # probs  : tensor of shape (N,)   with detection confidence scores [0, 1]
    # faces  : tensor of shape (N, 3, 160, 160) — aligned face crops, normalised
    #
    # All three tensors share the same index: boxes[i] and faces[i] correspond
    # to the same detected face.
   
    # Detect bounding boxes, then extract aligned face crops from those boxes.
    # Calling detect() once then extract() avoids running the detection network
    # twice (which is what separate mtcnn.detect() + mtcnn() calls would do).
    boxes, probs = mtcnn.detect(pil_image, landmarks=False)

    # If no faces were detected, return an empty list immediately
    if boxes is None:
        return results

    face_tensors = mtcnn.extract(pil_image, boxes, save_path=None)

    if face_tensors is None:
        return results

    # Ensure face_tensors has a batch dimension even when N=1
    # (MTCNN sometimes returns a 3-D tensor for a single detection)
    if face_tensors.ndim == 3:
        face_tensors = face_tensors.unsqueeze(0)

    # ── Step 3: FaceNet — encode all detected faces in one batch ─────────────
    # Processing all faces as a single batch is more efficient than a Python
    # loop because the GPU can parallelise the computation across faces.
    face_tensors = face_tensors.to(device)

    with torch.no_grad():
        # embeddings shape: (N, 128)
        # Each row is the 128-dimensional face representation for one detection
        embeddings = facenet(face_tensors).cpu().numpy()

    # ── Step 4: Match each embedding against the database ────────────────────
    for i in range(min(len(boxes), len(embeddings))):
        box = boxes[i]            # bounding box [x1, y1, x2, y2]
        embedding = embeddings[i] # 512-dim vector for this face

        # Find the closest identity in the database
        name, distance = identify_face(embedding, database, threshold)

        results.append({
            "box":      tuple(box),
            "name":     name,
            "distance": distance,
            "embedding": embedding,
        })

    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Device selection ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[recognize] Using device: {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("[recognize] Loading MTCNN and FaceNet …")
    mtcnn, facenet = load_models(device)

    # ── Load the pre-built face database ─────────────────────────────────────
    print(f"[recognize] Loading database from '{args.database}' …")
    database = load_database(args.database)

    # ── Open webcam ──────────────────────────────────────────────────────────
    print(f"[recognize] Opening camera (index {args.camera}) …")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera with index {args.camera}. "
            "Try a different index (e.g. --camera 0) or check your camera connection."
        )

    # Lower resolution reduces the image area MTCNN must process, which is the
    # main speed bottleneck. 640×480 gives a good balance of speed vs. accuracy.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[recognize] Camera ready. Press 'q' to quit, 's' to save screenshot.")

    # ── Threshold trackbar ───────────────────────────────────────────────────
    WIN_NAME = "Face Recognition — press q to quit"
    cv2.namedWindow(WIN_NAME)
    # Trackbar range: 0–200 → threshold 0.00–2.00 (divided by 100)
    cv2.createTrackbar(
        "Threshold x100", WIN_NAME,
        int(args.threshold * 100), 200,
        lambda _: None  # no-op; value is read each frame
    )

    # ── State for frame-skipping optimisation ────────────────────────────────
    # Running the full neural network pipeline on every single frame can be
    # slow on CPU. We re-use the detection result from the previous inference
    # frame for the next (skip_frames - 1) display frames.
    last_results   = []   # results from the most recent inference frame
    frame_counter  = 0    # counts total frames captured

    # FPS measurement
    fps_start_time = time.time()
    fps            = 0.0
    fps_counter    = 0

    # ── Main capture loop ────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[recognize] Failed to read frame — end of stream or camera error.")
            break

        frame_counter += 1
        fps_counter   += 1

        # Update FPS display once per second
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps            = fps_counter / elapsed
            fps_counter    = 0
            fps_start_time = time.time()

        # ── Read live threshold from trackbar ────────────────────────────────
        threshold = cv2.getTrackbarPos("Threshold x100", WIN_NAME) / 100.0

        # ── Run inference only on every Nth frame ────────────────────────────
        # This is a common real-time trick: the bounding boxes move slightly
        # between skipped frames but the identity labels are stable enough that
        # the user does not notice the staleness.
        if frame_counter % args.skip_frames == 0:
            last_results = process_frame(
                frame, mtcnn, facenet, database, threshold, device
            )

        # ── Draw results on the current frame ────────────────────────────────
        for result in last_results:
            draw_face_box(
                frame,
                result["box"],
                result["name"],
                result["distance"],
                threshold,
            )

        # Overlay the FPS counter
        draw_fps(frame, fps)

        # Display the annotated frame in a window titled "Face Recognition"
        cv2.imshow(WIN_NAME, frame)

        # ── Keyboard controls ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # 'q' → quit the application
            print("[recognize] 'q' pressed — exiting.")
            break

        elif key == ord("s"):
            # 's' → save the current annotated frame as a PNG screenshot
            screenshot_path = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(screenshot_path, frame)
            print(f"[recognize] Screenshot saved to '{screenshot_path}'")

    # ── Clean up ─────────────────────────────────────────────────────────────
    cap.release()           # release the webcam resource
    cv2.destroyAllWindows() # close all OpenCV windows

    print("[recognize] Done.")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
