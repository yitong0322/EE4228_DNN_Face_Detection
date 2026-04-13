"""
utils.py
--------
Shared utility functions used by both build_database.py and recognize.py.

Responsibilities:
  - Compute the distance between two face embeddings
  - Draw bounding boxes and name labels on video frames
  - Load the saved face database from disk
"""

import numpy as np
import cv2
import os
import pickle


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------

def euclidean_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute the Euclidean (L2) distance between two 512-dim face embeddings.

    FaceNet is trained with a triplet loss that pushes same-identity embeddings
    close together and different-identity embeddings far apart in L2 space.
    A smaller distance means the two faces are more likely to be the same person.

    Args:
        embedding_a: 1-D numpy array of shape (512,)
        embedding_b: 1-D numpy array of shape (512,)

    Returns:
        Scalar float representing the L2 distance.
    """
    return float(np.linalg.norm(embedding_a - embedding_b))


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two face embeddings.

    Cosine similarity measures the angle between two vectors regardless of their
    magnitude. A value of 1.0 means identical direction (same person), 0.0 means
    orthogonal (unrelated). It is sometimes more stable than Euclidean distance
    when embeddings are not L2-normalised.

    Args:
        embedding_a: 1-D numpy array of shape (512,)
        embedding_b: 1-D numpy array of shape (512,)

    Returns:
        Scalar float in [-1, 1]. Higher is more similar.
    """
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    # Guard against division by zero (all-zero vector should never happen in practice)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Database I/O
# ---------------------------------------------------------------------------

def load_database(database_path: str) -> dict:
    """
    Load the pre-built face embedding database from disk.

    The database is a Python dictionary serialised with pickle:
        {
            "Alice": [embedding_1, embedding_2, ...],   # list of np.ndarray
            "Bob":   [embedding_1, ...],
            ...
        }

    Each entry stores *all* embeddings collected for that person so we can
    average or compare against each individually during recognition.

    Args:
        database_path: Path to the .pkl file created by build_database.py

    Returns:
        Dictionary mapping person name -> list of 512-dim numpy arrays.

    Raises:
        FileNotFoundError: If the database file does not exist yet.
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError(
            f"Database not found at '{database_path}'. "
            "Run build_database.py first to generate it."
        )

    with open(database_path, "rb") as f:
        database = pickle.load(f)

    print(f"[utils] Loaded database with {len(database)} identities: "
          f"{list(database.keys())}")
    return database


def save_database(database: dict, database_path: str) -> None:
    """
    Serialise the face embedding dictionary to disk using pickle.

    Args:
        database:      Dictionary mapping name -> list of embeddings.
        database_path: Destination file path (e.g. 'database/faces.pkl').
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(database_path), exist_ok=True)

    with open(database_path, "wb") as f:
        pickle.dump(database, f)

    print(f"[utils] Database saved to '{database_path}'")


# ---------------------------------------------------------------------------
# OpenCV drawing helpers
# ---------------------------------------------------------------------------

# Colour palette (BGR format, as required by OpenCV)
COLOUR_KNOWN   = (50, 205, 50)    # lime green  — face is in the database
COLOUR_UNKNOWN = (0, 0, 220)      # red         — face is NOT in the database
COLOUR_TEXT_BG = (0, 0, 0)        # black label background for readability


def draw_face_box(
    frame: np.ndarray,
    box: tuple,
    name: str,
    distance: float,
    threshold: float
) -> None:
    """
    Draw a bounding box and identity label on a BGR video frame (in-place).

    The box is green when the identity is known (distance < threshold)
    and red when the face is not in the database.

    Args:
        frame:     OpenCV BGR image array (modified in-place).
        box:       (x1, y1, x2, y2) pixel coordinates of the detected face.
        name:      Predicted identity string, or "Unknown".
        distance:  Euclidean distance to the closest database embedding.
        threshold: Decision boundary — below this value the face is "known".
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # Choose colour based on whether the face was recognised
    colour = COLOUR_KNOWN if name != "Unknown" else COLOUR_UNKNOWN

    # Draw the bounding rectangle around the detected face
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness=2)

    # Build the label string: show the name and the raw distance score
    label = f"{name}  ({distance:.2f})"

    # Measure how much space the text needs so we can draw a background pill
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 1
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw a filled rectangle behind the text so it remains readable on any background
    cv2.rectangle(
        frame,
        (x1, y1 - label_h - baseline - 6),   # top-left of background rect
        (x1 + label_w + 4, y1),               # bottom-right of background rect
        colour,
        thickness=cv2.FILLED
    )

    # Draw the label text in white on top of the coloured background
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - baseline - 2),          # position: just above the box
        font, font_scale,
        (255, 255, 255),                        # white text
        thickness,
        lineType=cv2.LINE_AA
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Overlay a frames-per-second counter in the top-left corner of the frame.

    This is useful for benchmarking how fast the full detection + recognition
    pipeline runs in real time.

    Args:
        frame: OpenCV BGR image array (modified in-place).
        fps:   Current frames-per-second value to display.
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),   # light grey — unobtrusive
        1,
        cv2.LINE_AA
    )
