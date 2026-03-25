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
from torchvision import transforms         # Data augmentation pipeline
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
    parser.add_argument(
        "--augmentations", type=int, default=10,
        help=(
            "Number of augmented variants to generate per original image. "
            "Set to 1 to disable augmentation (use raw image only). "
            "Higher values produce a larger, more diverse embedding database."
        )
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
# Data augmentation pipeline
# ---------------------------------------------------------------------------

# Each transform simulates a real-world capture condition that the live camera
# will encounter during the demo. By exposing FaceNet to these variations at
# enrolment time we increase the chance that at least one stored embedding
# closely matches whatever angle/lighting the camera sees at recognition time.
#
# Why these six transforms specifically:
#   RandomHorizontalFlip    — person turns their head left or right
#   ColorJitter             — varying room lighting, warm vs cool white bulbs
#   RandomRotation          — slight head tilt (nodding, looking at phone)
#   GaussianBlur            — camera defocus or subject moving quickly
#   RandomPerspective       — camera held at an angle, not perfectly frontal
#   RandomGrayscale         — very low light causes the camera to drop to mono
#
# NOTE: transforms are applied to PIL images BEFORE MTCNN, so MTCNN still
# performs its own normalisation step on the output crop.
AUGMENTER = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.1),
])

def embed_pil_image(
    base_img: Image.Image,
    mtcnn: MTCNN,
    facenet: InceptionResnetV1,
    device: torch.device,
    num_augmentations: int = 10,
) -> list[np.ndarray]:
    """
    Apply augmentation to a PIL image and compute multiple face embeddings.

    Pipeline for one image:
        PIL RGB image
            └─ AUGMENTER × num_augmentations
                  └─ MTCNN (detect + align) → 160×160 crop
                        └─ FaceNet → 512-dim embedding
                              └─ list of numpy arrays

    Why generate multiple embeddings per photo?
        FaceNet's recognition quality depends on how well the stored embeddings
        cover the range of poses and lighting conditions seen at query time.
        With only a handful of photos per person, the raw images alone may not
        capture enough variation. Augmentation artificially widens this coverage
        without requiring additional photos.

        Full-embedding matching (taking the minimum distance at query time)
        benefits most from this strategy: the larger and more diverse the set
        of stored embeddings, the more likely one of them will closely match
        whatever angle the live camera captures.

        We deliberately chose full-embedding matching over centroid matching
        for this reason — averaging all embeddings into one centroid would
        destroy the pose diversity that augmentation worked to create.

    Args:
        base_img:           PIL RGB image.
        mtcnn:              Initialised MTCNN model.
        facenet:            Initialised FaceNet model.
        device:             Compute device (CPU / CUDA).
        num_augmentations:  How many augmented variants to generate per image.
                            Set to 1 to use the raw image only (no augmentation).

    Returns:
        List of 512-dim numpy arrays, one per successful face detection.
        Returns an empty list if MTCNN finds no face in any variant.
    """
    embeddings = []

    for i in range(num_augmentations):
        # Apply the augmentation pipeline to produce one variant of the image.
        # When num_augmentations == 1 we skip augmentation and use the raw
        # image so the function degrades gracefully to the original behaviour.
        aug_img = AUGMENTER(base_img) if num_augmentations > 1 else base_img

        # Run MTCNN — returns a (3, 160, 160) float tensor if a face is found,
        # or None if no face exceeds the confidence thresholds.
        # keep_all=False ensures we get exactly one crop per augmented image.
        face_tensor = mtcnn(aug_img)

        if face_tensor is None:
            # The augmentation may have distorted the face beyond MTCNN's
            # detection threshold — skip this variant silently.
            continue

        # Add a batch dimension: (3, 160, 160) → (1, 3, 160, 160)
        face_tensor = face_tensor.unsqueeze(0).to(device)

        # Compute the embedding with gradient tracking disabled (inference only)
        with torch.no_grad():
            embedding = facenet(face_tensor)  # shape: (1, 512)

        # Detach, move to CPU, flatten to 1-D numpy array of shape (512,)
        embeddings.append(embedding.squeeze().cpu().numpy())

    return embeddings


def embed_image(
    image_path: str,
    mtcnn: MTCNN,
    facenet: InceptionResnetV1,
    device: torch.device,
    num_augmentations: int = 10,
) -> list[np.ndarray]:
    """Load an image from disk and compute augmented face embeddings."""
    try:
        base_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [skip] Could not open '{image_path}': {e}")
        return []
    return embed_pil_image(base_img, mtcnn, facenet, device, num_augmentations)


# ---------------------------------------------------------------------------
# Database builder
# ---------------------------------------------------------------------------

def build_database(dataset_dir: str, mtcnn: MTCNN, facenet: InceptionResnetV1,
                   device: torch.device, num_augmentations: int = 10) -> dict:
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

    For each image we generate num_augmentations variants via AUGMENTER, then
    embed each variant with FaceNet. All resulting embeddings are stored in a
    flat list per person.

    During recognition (recognize.py) we compare the query embedding against
    every stored embedding and take the minimum distance — so a larger, more
    diverse embedding list directly improves recognition robustness.

    Why not use centroid (average embedding) instead?
        Averaging all embeddings into one centroid collapses pose diversity:
        a centroid computed from frontal + left-profile + right-profile
        embeddings lands in a region that does not correspond to any real face.
        Full-embedding matching avoids this problem entirely — it only requires
        one stored embedding to match the current camera angle.
        This matters even more when live camera enrolment (multiple angles) is
        added in a later phase.

    Args:
        dataset_dir:        Root path of the labelled image dataset.
        mtcnn:              Initialised MTCNN model.
        facenet:            Initialised FaceNet model.
        device:             Compute device.
        num_augmentations:  Augmented variants to generate per source image.

    Returns:
        Dictionary  { "PersonName": [emb_1, emb_2, ...], ... }
        where each emb_i is a numpy array of shape (512,).
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
            print(f"  → {image_file} ", end="")

            # embed_image now returns a list of embeddings (one per augmentation)
            new_embeddings = embed_image(
                image_path, mtcnn, facenet, device, num_augmentations
            )

            embeddings.extend(new_embeddings)   # flatten into the person's list
            print(f"✓  +{len(new_embeddings)} embeddings "
                  f"(total so far: {len(embeddings)})")

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
    print(f"[build] Augmentations per image: {args.augmentations}")
    database = build_database(
        args.dataset, mtcnn, facenet, device,
        num_augmentations=args.augmentations
    )

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