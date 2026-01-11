import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import torch

import numpy as np
import supervision as sv

from pathlib import Path
from PIL import Image
from typing import Optional
from IPython.display import Video

from sam3.sam3.model_builder import build_sam3_video_predictor

device = torch.device("mps")
predictor = build_sam3_video_predictor(gpus_to_use=[device])

SOURCE_VIDEO = "jets.mp4"
SOURCE_FRAMES = "jetsframes"

def load_frame(directory: str, index: int): 
    directory_path = Path(directory)
    frame_path = directory_path / f"{index:05d}.jpg"  # 05d is formatting: index = 1 -> 00001.jpg

    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Failed to read frame: {frame_path}")

    return frame  # Numpy array


response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=Path(SOURCE_VIDEO).as_posix(),
    )
)
session_id = response["session_id"]

_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)

frame_idx = 0
text = "jet"

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=text,
    )
)

result = response["outputs"]  # Dict of numpy arrays

def from_sam(result: dict) -> sv.Detections:
    return sv.Detections(
        xyxy=sv.mask_to_xyxy(result["out_binary_masks"]),  # Returns tightest possible rectangle around object
        mask=result["out_binary_masks"],  # Returns mask of object (pixel accurate)
        confidence=result["out_probs"],  # Probability of correct object
        tracker_id=result["out_objs_ids"],  # Assigns unique id to object
    )


COLOR = sv.ColorPalette.from_hex(["#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"])


def annotate(image: np.ndarray, detections: sv.Detections, text = None) -> np.ndarray:
    h, w, _ = image.shape  # image.shape returns height, width, # of color channels (usually 3, BGR
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))

    mask_annotator = sv.MaskAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.TRACK,
        opacity=0.6,
    )

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)

    if text:
        label_annotator = sv.LabelAnnotator(
            color=COLOR,
            color_lookup=sv.ColorLookup.TRACK,
            text_scale=text_scale,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.TOP_CENTER,
            text_offset=(0,-30)
        )
        labels = [
            f"#{tracker_id} {text}"
            for tracker_id in detections.tracker_id
        ]
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    return annotated_image


def propagate_in_video(predictor, session_id):
    frame_outputs = {}
    for response in predictor.handle_stream_request(
        request = dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_outputs[response["frame_index"]] = response["outputs"]

    return frame_outputs


frame_outputs = propagate_in_video(predictor, session_id)

TARGET_VIDEO = f"{Path(SOURCE_VIDEO).stem}-result{Path(SOURCE_VIDEO).suffix}"


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    output = frame_outputs[index]
    detections = from_sam(output)
    return annotate(frame, detections, text)


sv.process_video(source_path=SOURCE_VIDEO, target_path=TARGET_VIDEO, callback=callback)

