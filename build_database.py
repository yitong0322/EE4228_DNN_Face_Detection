"""
build_database.py
-----------------
Phase 1 (offline): Build the face embedding database from a folder of photos.

Usage:
    python build_database.py --dataset dataset/ --output database/faces.pkl

Expected dataset folder layout:
    dataset/
        Alice/
            img1.jpg
            img2.jpg
            ...          (at least 10 images recommended)
        Bob/
            img1.jpg
            ...
        <Name>/
            ...

What this script does:
    1. Iterate over every person's image folder.
    2. For each image, use MTCNN to detect and align the face.
    3. Feed the aligned face crop into FaceNet to get a 128-dim embedding.
    4. Collect all embeddings per person and save them as a pickle file.

The resulting .pkl file is loaded by recognize.py at startup.
"""

import os
import argparse
import pickle

import numpy as np
from PIL import Image                      # Pillow — image loading
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1  # FaceNet ecosystem

from utils import save_database


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a face embedding database from a folder of labelled images."
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset",
        help="Root folder containing one sub-folder per person."
    )
    parser.add_argument(
        "--output", type=str, default="database/faces.pkl",
        help="Destination path for the serialised embedding dictionary."
    )
    parser.add_argument(
        "--min-face-size", type=int, default=40,
        help="Smallest face (in pixels) that MTCNN will attempt to detect."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def load_models(device: torch.device):
    """
    Initialise MTCNN (detector) and FaceNet (encoder) and move them to the
    specified compute device (CPU or CUDA GPU).

    MTCNN — Multi-task Cascaded Convolutional Networks
        A three-stage cascade (P-Net → R-Net → O-Net) that progressively
        refines face candidate regions. It also predicts five facial landmarks
        (eyes, nose, mouth corners) which are used for geometric alignment
        before passing the crop to FaceNet.

        Key parameters:
          image_size  : output crop size expected by FaceNet (160 × 160)
          margin      : extra pixels added around the detected bounding box
                        so that the face is not clipped at the edges
          keep_all    : False → return only the highest-confidence face per image
                        (suitable for the enrolment phase where each photo
                        should contain exactly one face)
          post_process: True → normalise pixel values to [-1, 1] before returning

    InceptionResnetV1 — FaceNet backbone
        A deep Inception-ResNet-v1 architecture pretrained on the VGGFace2
        dataset. Given a 160 × 160 aligned face crop, it produces a 512-dim
        internal representation which is then L2-normalised to a 128-dim
        embedding vector. Points that are close in this 128-D space correspond
        to the same identity.

        pretrained='vggface2' loads weights trained on ~3.3 M images of
        ~9 000 identities — a strong starting point for our small custom dataset.

    Args:
        device: torch.device('cuda') or torch.device('cpu')

    Returns:
        Tuple (mtcnn, facenet) — both models set to eval mode.
    """
    # MTCNN: face detector + aligner
    mtcnn = MTCNN(
        image_size=160,       # FaceNet expects 160 × 160 crops
        margin=20,            # pad 20 px around the face to preserve context
        min_face_size=40,     # ignore detections smaller than 40 × 40 px
        thresholds=[0.6, 0.7, 0.7],  # confidence thresholds for P/R/O-Net stages
        factor=0.709,         # image pyramid scale factor
        keep_all=False,       # keep only the most confident face per image
        post_process=True,    # normalise to [-1, 1]
        device=device
    )

    # FaceNet: face encoder (pretrained on VGGFace2)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    return mtcnn, facenet


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------

def embed_image(
    image_path: str,
    mtcnn: MTCNN,
    facenet: InceptionResnetV1,
    device: torch.device
) -> np.ndarray | None:
    """
    Detect the face in a single image, align it, and compute its embedding.

    Pipeline for one image:
        Raw JPEG/PNG  →  MTCNN  →  160×160 aligned crop  →  FaceNet  →  128-dim vector

    Args:
        image_path: Path to the image file on disk.
        mtcnn:      Initialised MTCNN model.
        facenet:    Initialised FaceNet model.
        device:     Compute device (CPU / CUDA).

    Returns:
        128-dim numpy array if a face was detected, otherwise None.
    """
    # Load the image as a PIL RGB image (MTCNN expects RGB, not BGR like OpenCV)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [skip] Could not open '{image_path}': {e}")
        return None

    # Run MTCNN — returns a (1, 3, 160, 160) float tensor if a face is found,
    # or None if no face exceeds the confidence thresholds
    face_tensor = mtcnn(img)

    if face_tensor is None:
        print(f"  [skip] No face detected in '{image_path}'")
        return None

    # Add a batch dimension: (3, 160, 160) → (1, 3, 160, 160)
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # Compute the 128-dim embedding with gradient tracking disabled (we are not training)
    with torch.no_grad():
        embedding = facenet(face_tensor)  # shape: (1, 128)

    # Detach from the computation graph, move to CPU, convert to numpy 1-D array
    return embedding.squeeze().cpu().numpy()   # shape: (128,)


# ---------------------------------------------------------------------------
# Database builder
# ---------------------------------------------------------------------------

def build_database(dataset_dir: str, mtcnn: MTCNN, facenet: InceptionResnetV1,
                   device: torch.device) -> dict:
    """
    Walk the dataset directory tree and build a dictionary of embeddings.

    Expected directory structure:
        dataset_dir/
            PersonName1/  ← each sub-folder is one identity
                img001.jpg
                img002.png
                ...
            PersonName2/
                ...

    For each person we collect all their embeddings into a list. During
    recognition (recognize.py) we compare the query embedding against every
    stored embedding and take the closest match.

    Args:
        dataset_dir: Root path of the labelled image dataset.
        mtcnn:       Initialised MTCNN model.
        facenet:     Initialised FaceNet model.
        device:      Compute device.

    Returns:
        Dictionary  { "PersonName": [emb_1, emb_2, ...], ... }
        where each emb_i is a numpy array of shape (128,).
    """
    database = {}

    # Supported image file extensions
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Each sub-folder in the dataset root represents one person
    person_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not person_dirs:
        raise ValueError(
            f"No person sub-folders found in '{dataset_dir}'. "
            "Create one folder per person and place their photos inside."
        )

    for person_name in person_dirs:
        person_folder = os.path.join(dataset_dir, person_name)
        embeddings = []

        print(f"\n[build] Processing identity: '{person_name}'")

        # Collect all image files for this person
        image_files = [
            f for f in sorted(os.listdir(person_folder))
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]

        if len(image_files) < 10:
            print(f"  [warn] Only {len(image_files)} images found. "
                  "The assignment requires at least 10 per person.")

        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)
            print(f"  → {image_file}", end=" ")

            embedding = embed_image(image_path, mtcnn, facenet, device)

            if embedding is not None:
                embeddings.append(embedding)
                print(f"✓  (vector norm: {np.linalg.norm(embedding):.3f})")
            # If embedding is None, embed_image already printed a skip message

        if embeddings:
            database[person_name] = embeddings
            print(f"  Stored {len(embeddings)} embeddings for '{person_name}'.")
        else:
            print(f"  [warn] No valid embeddings for '{person_name}' — skipping.")

    return database


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Select the best available compute device
    # CUDA GPU is ~10× faster than CPU for the FaceNet forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build] Using device: {device}")

    # Load the two models
    print("[build] Loading MTCNN and FaceNet models …")
    mtcnn, facenet = load_models(device)

    # Build the embedding database by processing all images
    print(f"\n[build] Reading images from '{args.dataset}' …")
    database = build_database(args.dataset, mtcnn, facenet, device)

    # Persist the database to disk so recognize.py can load it at startup
    save_database(database, args.output)

    # Summary report
    print("\n[build] ─── Summary ───────────────────────────────")
    total_embeddings = sum(len(v) for v in database.values())
    for name, embs in database.items():
        print(f"  {name:20s} : {len(embs):3d} embeddings")
    print(f"  {'TOTAL':20s} : {total_embeddings:3d} embeddings")
    print(f"[build] Database saved to '{args.output}'")
    print("[build] Run  python recognize.py  to start live recognition.")
