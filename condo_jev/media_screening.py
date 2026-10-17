"""
Media screening service for resident photo and video uploads.
Validates uploaded attachments to ensure they are related to condominium maintenance
and do not contain human faces or personal identifying images.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger("condo_jev.media_screening")

# Load OpenCV cascades for face detection
_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def _get_cascade(name: str) -> cv2.CascadeClassifier | None:
    if name not in _CASCADES:
        try:
            path = cv2.data.haarcascades + name
            classifier = cv2.CascadeClassifier(path)
            if not classifier.empty():
                _CASCADES[name] = classifier
            else:
                _CASCADES[name] = None
        except Exception as exc:
            logger.warning(f"Failed to load cascade {name}: {exc}")
            _CASCADES[name] = None
    return _CASCADES[name]


class MediaScreeningResult(NamedTuple):
    is_acceptable: bool
    rejection_reason: str | None
    has_face: bool = False


def detect_face_in_frame(frame: np.ndarray) -> bool:
    """
    Check if a single image frame contains a human face using an ensemble of Haar cascades.
    Returns True if a face is detected.
    """
    if frame is None or frame.size == 0:
        return False

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Resize if very small to help feature extraction
    h, w = gray.shape[:2]
    if max(h, w) < 180:
        scale = 200.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # Equalize histogram for lighting variation
    try:
        gray = cv2.equalizeHist(gray)
    except Exception:
        pass

    # 1. Frontal face alt2 (balanced precision/recall)
    alt2 = _get_cascade("haarcascade_frontalface_alt2.xml")
    if alt2:
        faces = alt2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(24, 24))
        if len(faces) > 0:
            return True

    # 2. Frontal face default
    default_face = _get_cascade("haarcascade_frontalface_default.xml")
    if default_face:
        faces = default_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
        if len(faces) > 0:
            return True

    # 3. Frontal face alt
    alt = _get_cascade("haarcascade_frontalface_alt.xml")
    if alt:
        faces = alt.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(24, 24))
        if len(faces) > 0:
            return True

    # 4. Profile face
    profile = _get_cascade("haarcascade_profileface.xml")
    if profile:
        profiles = profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
        if len(profiles) > 0:
            return True

    # 5. Eye pair verification
    eye_cascade = _get_cascade("haarcascade_eye.xml")
    if eye_cascade:
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        if len(eyes) >= 2:
            return True

    return False


def screen_image_file(file_path: Path | str) -> MediaScreeningResult:
    """Screen an image file for human faces and basic image validity."""
    p = Path(file_path)
    if not p.is_file():
        return MediaScreeningResult(False, "Media file was not found.")

    img = cv2.imread(str(p))
    if img is None:
        return MediaScreeningResult(
            False,
            "Unable to process the image format. Please send a standard photo (JPG, PNG) showing the maintenance issue.",
        )

    # Check for empty / corrupt image
    h, w = img.shape[:2]
    if h < 40 or w < 40:
        return MediaScreeningResult(
            False,
            "The image resolution is too low. Please send a clear photo showing the condominium maintenance issue.",
        )

    # Check for completely blank / solid color images
    std_dev = float(np.std(img))
    if std_dev < 3.0:
        return MediaScreeningResult(
            False,
            "The provided image appears blank. Please send a photo showing the maintenance issue.",
        )

    # Detect human face
    if detect_face_in_frame(img):
        return MediaScreeningResult(
            False,
            "This image cannot be accepted as it appears to contain a person's face.\n\n"
            "Please kindly send only photos or videos showing the condominium maintenance issue "
            "(such as water leaks, damaged fixtures, or building defects) without people's faces "
            "for resident privacy and operational assessment.",
            has_face=True,
        )

    return MediaScreeningResult(True, None)


def screen_video_file(file_path: Path | str) -> MediaScreeningResult:
    """Sample frames from a video file and check for human faces."""
    p = Path(file_path)
    if not p.is_file():
        return MediaScreeningResult(False, "Video file was not found.")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return MediaScreeningResult(
            False,
            "Unable to open the video file. Please send a standard MP4 or MOV video showing the maintenance issue.",
        )

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 30

        # Sample 5 frame positions
        sample_positions = [
            int(total_frames * ratio) for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]

        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if detect_face_in_frame(frame):
                return MediaScreeningResult(
                    False,
                    "This video cannot be accepted as it appears to contain a person's face.\n\n"
                    "Please kindly send only photos or videos showing the condominium maintenance issue "
                    "(such as water leaks, damaged fixtures, or building defects) without people's faces "
                    "for resident privacy and operational assessment.",
                    has_face=True,
                )

        return MediaScreeningResult(True, None)
    finally:
        cap.release()


def screen_media_file(file_path: Path | str) -> MediaScreeningResult:
    """
    Main entry point for media screening. Automatically detects whether the file
    is an image or video based on suffix and runs the appropriate verification.
    """
    p = Path(file_path)
    suffix = p.suffix.lower()

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if suffix in video_exts:
        return screen_video_file(p)
    else:
        return screen_image_file(p)
