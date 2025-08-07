import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import HTTPException

_base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
_hand_options = vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_hand_landmarker = vision.HandLandmarker.create_from_options(_hand_options)


def select_hand(
    img_bgr: np.ndarray,
    box_size: tuple[int, int],
) -> np.ndarray:
    h, w, _ = img_bgr.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    )

    detection_result = _hand_landmarker.detect(mp_image)

    if not detection_result.hand_landmarks:
        raise HTTPException(status_code=400, detail="No hand detected")

    hand_landmarks = detection_result.hand_landmarks[0]
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    half = box_size[0] // 2
    sx = max(cx - half, 0)
    sy = max(cy - half, 0)
    ex = min(sx + box_size[0], w)
    ey = min(sy + box_size[1], h)

    if ex - sx < box_size[0]:
        sx = max(ex - box_size[0], 0)
    if ey - sy < box_size[1]:
        sy = max(ey - box_size[1], 0)

    img_out = img_bgr.copy()
    cv2.rectangle(
        img_out, (sx, sy), (sx + box_size[0], sy + box_size[1]), (0, 255, 0), 2
    )
    crop = img_bgr[sy : sy + box_size[1], sx : sx + box_size[0]]

    return crop
