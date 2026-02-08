"""
Process a video with MediaPipe FaceLandmarker and dump 478 face landmarks to CSV.

Output CSV has 1434 columns: face_0000_x, face_0000_y, face_0000_z, ... face_0477_x, face_0477_y, face_0477_z
One row per frame. Frames with no face detected get all zeros.

Usage:
    python landm_face.py
    python landm_face.py --video path/to/video.mp4
    python landm_face.py --video path/to/video.mp4 --output landmarks.csv
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Defaults ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "face_landmarker.task")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "mediapipe_face_3d_xyz.csv")
NUM_LANDMARKS = 478


def build_header():
    cols = []
    for i in range(NUM_LANDMARKS):
        cols.extend([f"face_{i:04d}_x", f"face_{i:04d}_y", f"face_{i:04d}_z"])
    return cols


def process_video(video_path, model_path, output_path):
    if not os.path.isfile(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}")
    print(f"  {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # Create FaceLandmarker
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    header = build_header()
    frame_idx = 0
    faces_detected = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps) if fps > 0 else frame_idx * 33

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks and len(result.face_landmarks) > 0:
                lm_list = result.face_landmarks[0]
                row = []
                for lm in lm_list:
                    row.extend([lm.x, lm.y, lm.z])
                # Pad if fewer than 478 landmarks (shouldn't happen, but safe)
                while len(row) < NUM_LANDMARKS * 3:
                    row.extend([0.0, 0.0, 0.0])
                writer.writerow(row)
                faces_detected += 1
            else:
                writer.writerow([0.0] * (NUM_LANDMARKS * 3))

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{total_frames}")

    cap.release()
    landmarker.close()

    print(f"\nDone. {frame_idx} frames processed, {faces_detected} with face detected.")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract face landmarks from video to CSV")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="FaceLandmarker .task model path")
    args = parser.parse_args()

    process_video(args.video, args.model, args.output)


if __name__ == "__main__":
    main()
