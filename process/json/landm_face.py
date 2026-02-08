"""
Process a video with MediaPipe FaceLandmarker:
  1. Dump 478 face landmarks (x,y,z) per frame to CSV
  2. Dump blendshapes per frame to JSON (same format as blendshapes_output.json)
  3. Print min/max ranges for each blendshape

Usage:
    python landm_face.py --video path/to/video.mp4
    python landm_face.py --video path/to/video.mp4 --output landmarks.csv --blendshapes bs.json
"""

import argparse
import csv
import json
import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Defaults ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "face_landmarker.task")
DEFAULT_CSV_OUTPUT = os.path.join(SCRIPT_DIR, "mediapipe_face_3d_xyz_new.csv")
DEFAULT_BS_OUTPUT = os.path.join(SCRIPT_DIR, "blendshapes_output.json")
NUM_LANDMARKS = 478


def build_header():
    cols = []
    for i in range(NUM_LANDMARKS):
        cols.extend([f"face_{i:04d}_x", f"face_{i:04d}_y", f"face_{i:04d}_z"])
    return cols


def process_video(video_path, model_path, csv_path, bs_path):
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

    # Create FaceLandmarker with blendshapes enabled
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    header = build_header()
    frame_idx = 0
    faces_detected = 0
    bs_results = []
    bs_min = {}
    bs_max = {}

    with open(csv_path, "w", newline="") as f:
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
                # -- Landmarks to CSV --
                lm_list = result.face_landmarks[0]
                row = []
                for lm in lm_list:
                    row.extend([lm.x, lm.y, lm.z])
                while len(row) < NUM_LANDMARKS * 3:
                    row.extend([0.0, 0.0, 0.0])
                writer.writerow(row)
                faces_detected += 1

                # -- Blendshapes to dict --
                if result.face_blendshapes and len(result.face_blendshapes) > 0:
                    bs = {}
                    for cat in result.face_blendshapes[0]:
                        name = cat.category_name
                        if name == "_neutral":
                            continue
                        val = round(cat.score, 6)
                        bs[name] = val
                        if name not in bs_min or val < bs_min[name]:
                            bs_min[name] = val
                        if name not in bs_max or val > bs_max[name]:
                            bs_max[name] = val
                    bs_results.append(bs)
                else:
                    bs_results.append(None)
            else:
                writer.writerow([0.0] * (NUM_LANDMARKS * 3))
                bs_results.append(None)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{total_frames}")

    cap.release()
    landmarker.close()

    # Write blendshapes JSON
    with open(bs_path, "w") as f:
        json.dump(bs_results, f, indent=2)

    # Summary
    valid_bs = sum(1 for r in bs_results if r is not None)
    print(f"\nDone. {frame_idx} frames processed, {faces_detected} with face detected.")
    print(f"Landmarks CSV: {csv_path}")
    print(f"Blendshapes JSON: {bs_path}  ({valid_bs} valid frames)")

    # Print min/max per blendshape
    if bs_min:
        print(f"\n{'Blendshape':<25} {'Min':>10} {'Max':>10}")
        print("-" * 47)
        for name in sorted(bs_min.keys()):
            print(f"{name:<25} {bs_min[name]:>10.6f} {bs_max[name]:>10.6f}")


def main():
    parser = argparse.ArgumentParser(description="Extract face landmarks + blendshapes from video")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default=DEFAULT_CSV_OUTPUT, help="Output landmarks CSV path")
    parser.add_argument("--blendshapes", "-b", default=DEFAULT_BS_OUTPUT, help="Output blendshapes JSON path")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="FaceLandmarker .task model path")
    args = parser.parse_args()

    process_video(args.video, args.model, args.output, args.blendshapes)


if __name__ == "__main__":
    main()
