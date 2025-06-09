# Holistic Landmark System in MediaPipe

Based on the workspace information, I'll analyze how the MediaPipe holistic landmark system works, focusing specifically on hand detection, landmarks, and how they relate to the overall body pose.

## Overview of the Holistic Landmark System

The holistic landmark system integrates multiple landmark detection components:
- Pose landmarks
- Left/right hand landmarks
- Face landmarks

## Hand Tracking and Detection Process

### 1. Hand Landmark Detection Pipeline

The hand landmark detection in the holistic system follows this flow:

1. **Initial Hand Location from Pose**: 
   - The system first detects pose landmarks
   - It uses specific pose landmarks to derive initial hand locations:
   ```cc
   // Extracts left-hand-related landmarks from the pose landmarks
   ranges: { begin: 15 end: 16 }  // Left wrist
   ranges: { begin: 17 end: 18 }  // Left pinky
   ranges: { begin: 19 end: 20 }  // Left index
   
   // Extracts right-hand-related landmarks from the pose landmarks
   ranges: { begin: 16 end: 17 }  // Right wrist
   ranges: { begin: 18 end: 19 }  // Right pinky
   ranges: { begin: 20 end: 21 }  // Right index
   ```

2. **Hand ROI Extraction**:
   - Uses `HandLandmarksFromPoseToRecropRoi` to get ROI from pose landmarks
   - ROI serves as a search region for the hand re-crop model

3. **Hand ROI Refinement**:
   - Uses `HandRecropByRoiGpu`/`HandRecropByRoiCpu` to refine hand crop rectangle
   - The re-cropping model (`hand_recrop.tflite`) provides a more accurate hand ROI

4. **Hand Tracking**:
   - The `HandTracking` subgraph decides what ROI to use for landmark prediction
   - It either uses:
     - Previous frame landmarks ROI (if hand is still tracked)
     - Current frame re-crop ROI (if tracking is lost)

5. **Hand Landmark Detection**:
   - `HandLandmarkGpu`/`HandLandmarkCpu` predicts hand landmarks from the tracking rectangle
   - Uses the `hand_landmark_full.tflite` model

### 2. Key Components for Hand Detection and Tracking

#### Hand Visibility Check

```pbtxt
# Gets hand visibility
node {
  calculator: "HandVisibilityFromHandLandmarksFromPose"
  input_stream: "HAND_LANDMARKS_FROM_POSE:hand_landmarks_from_pose"
  output_stream: "VISIBILITY:hand_visibility"
}

# Drops hand-related pose landmarks if pose wrist is not visible
node {
  calculator: "GateCalculator"
  input_stream: "hand_landmarks_from_pose"
  input_stream: "ALLOW:hand_visibility"
  output_stream: "ensured_hand_landmarks_from_pose"
}
```

This ensures hand detection is only attempted when the wrist is visible.

#### Hand Tracking Logic

```pbtxt
node {
  calculator: "RoiTrackingCalculator"
  input_stream: "PREV_LANDMARKS:prev_hand_landmarks"
  input_stream: "PREV_LANDMARKS_RECT:prev_hand_landmarks_roi"
  input_stream: "RECROP_RECT:hand_roi_from_recrop"
  input_stream: "IMAGE_SIZE:image_size"
  output_stream: "TRACKING_RECT:hand_tracking_roi"
  options: {
    [mediapipe.RoiTrackingCalculatorOptions.ext] {
      rect_requirements: {
        rotation_degrees: 40.0
        translation: 0.2
        scale: 0.4
      }
      landmarks_requirements: {
        recrop_rect_margin: -0.1
      }
    }
  }
}
```

This calculator makes the decision about whether to use previous frame data or new detection based on tracking quality.

### 3. Hand World Landmarks and Alignment to Pose

A key feature is the alignment of hand world landmarks with pose:

```cc
// Align hand world landmarks with pose world landmarks
hand_world_landmarks = AlignHandToPoseInWorldCalculator(
    hand_world_landmarks.value(), pose_world_landmarks,
    pose_indices.wrist_idx, graph);
```

This ensures that the hand landmarks are properly aligned with the wrist position in the pose landmarks, creating a cohesive skeleton.

## Left and Right Hand Detection

The system handles left and right hands separately:

```pbtxt
# Predicts left hand landmarks
node {
  calculator: "HandLandmarksFromPoseGpu"
  input_stream: "IMAGE:input_video"
  input_stream: "HAND_LANDMARKS_FROM_POSE:left_hand_landmarks_from_pose"
  output_stream: "HAND_LANDMARKS:left_hand_landmarks"
  ...
}

# Predicts right hand landmarks
node {
  calculator: "HandLandmarksFromPoseGpu" 
  input_stream: "IMAGE:input_video"
  input_stream: "HAND_LANDMARKS_FROM_POSE:right_hand_landmarks_from_pose"
  output_stream: "HAND_LANDMARKS:right_hand_landmarks"
  ...
}
```

## Key Insights About the Hand Detection System

1. **Pose-Guided Detection**: 
   - Uses pose landmarks as guidance for hand detection
   - This makes the system more efficient as it narrows down search areas

2. **Re-Cropping Technique**:
   - Uses a two-step approach: approximate location from pose, then refined ROI from specialized model
   - `hand_recrop.tflite` is specifically trained to refine hand regions

3. **Temporal Consistency**:
   - Leverages previous frame information for tracking
   - `HandTracking` module provides this temporal consistency
   - The system maintains tracking as long as hand movement is within reasonable bounds

4. **Alignment with Pose**:
   - Hand world landmarks are aligned with pose world landmarks at the wrist
   - This creates a consistent 3D representation where hands correctly connect to arms

5. **Visibility-Based Processing**:
   - Only processes hands when corresponding wrist landmarks are visible
   - This optimizes computation and prevents erroneous detections

## HolisticLandmarkerResult Structure

The output result contains both normalized and world landmarks:

```py
@dataclasses.dataclass
class HolisticLandmarkerResult:
  face_landmarks: List[landmark_module.NormalizedLandmark]
  pose_landmarks: List[landmark_module.NormalizedLandmark]
  pose_world_landmarks: List[landmark_module.Landmark]
  left_hand_landmarks: List[landmark_module.NormalizedLandmark]
  left_hand_world_landmarks: List[landmark_module.Landmark]
  right_hand_landmarks: List[landmark_module.NormalizedLandmark]
  right_hand_world_landmarks: List[landmark_module.Landmark]
  face_blendshapes: Optional[...]
  segmentation_mask: Optional[...]
```

- **Normalized landmarks** are in image coordinates (0.0-1.0)
- **World landmarks** represent 3D positions relative to the body

## Conclusion

The MediaPipe holistic landmark system uses a sophisticated approach for hand tracking and landmark detection. It leverages pose information to guide hand detection, uses specialized models for refinement, maintains temporal consistency through tracking, and ensures anatomical consistency by aligning hand landmarks with the pose skeleton. This integrated approach enables more accurate and stable hand landmark detection as part of a complete body pose estimation system.


## Detailed Model Specifications

### TensorFlow Lite Models Used

#### 1. Pose Detection Model (`pose_detection.tflite`)
- **Input**: RGB image (192×192)
- **Output**: 
  - Detection score
  - Bounding box for the person
  - Keypoints used for determining the center of the person
- **Architecture**: Based on BlazePose detector, a lightweight convolutional neural network

#### 2. Pose Landmark Full Model (`pose_landmark_full.tflite`)
- **Input**: RGB image (256×256) cropped around the person
- **Output**: 
  - 33 pose landmarks with (x, y, z, visibility) for each point
  - Auxiliary 3D landmarks in world coordinates
  - Segmentation mask
- **Properties**:
  - Higher accuracy but slower than the "lite" variant
  - Better accuracy for subtle movements and extremities

#### 3. Hand Recrop Model (`hand_recrop.tflite`)
- **Input**: Image region determined by pose landmarks (wrist, pinky, index)
- **Output**: Refined bounding box for hand landmark detection
- **Purpose**: Improve the initial region-of-interest estimation based on pose landmarks

#### 4. Hand Landmark Full Model (`hand_landmark_full.tflite`)
- **Input**: RGB image (224×224) cropped around the hand
- **Output**: 
  - 21 hand landmarks with (x, y, z) coordinates
  - Handedness probability (left/right)
- **Properties**:
  - Higher accuracy compared to "lite" variant
  - Better for detecting subtle finger movements

#### 5. Face Landmark Model (`face_landmark.tflite`)
- **Input**: RGB image (192×192) cropped around the face
- **Output**: 478 face landmarks with (x, y, z) coordinates
- **Features**: Accurately tracks facial features including eyes, eyebrows, lips, and facial contours

### Coordinate Systems

The holistic system utilizes two different coordinate systems:

1. **Normalized Image Coordinates**:
   - (x,y) coordinates normalized to [0.0, 1.0] range within the image
   - Origin (0,0) is at the top-left corner of the image
   - z-coordinate represents depth relative to the reference point (usually wrist for hands)
   - Used in `NormalizedLandmark` outputs

2. **World Coordinates**:
   - Real-world 3D coordinates in meters
   - Origin at approximately the center of the hips
   - Y-axis points up (height)
   - X-axis points to the person's right
   - Z-axis points forward from the person
   - Used in `Landmark` outputs with "_world" suffix
   - Enables consistent scaling regardless of camera distance

### Data Flow Between Models

The system processes data in this sequence:

1. **Pose Detection → Pose Landmark**:
   - Pose detection provides ROI for pose landmark model
   - Pose landmark model outputs 33 body keypoints

2. **Pose Landmark → Hand Landmark**:
   - Specific pose landmarks (wrists, fingers) define initial hand regions
   - These regions feed into hand recrop model
   - Refined ROIs are passed to hand landmark model

3. **Pose Landmark → Face Landmark**:
   - Face ROI determined from pose landmarks (eyes, nose, ears)
   - ROI is adjusted and passed to face landmark model

4. **Cross-component Alignment**:
   - Hand world landmarks are aligned with pose world landmarks at wrists
   - This ensures anatomical consistency in the full body representation
   - Alignment uses specific transformations to maintain proper connections

### Performance-Accuracy Tradeoffs

The models offer different performance profiles:

| Model Type | Variant | Landmarks | Accuracy | Latency (Snapdragon 865) |
|------------|---------|-----------|----------|--------------------------|
| Pose       | Full    | 33        | High     | ~30ms                    |
| Hand       | Full    | 21 per hand | High   | ~12ms per hand           |
| Face       | -       | 478       | High     | ~10ms                    |

The holistic tracking pipeline uses multiple strategies to maintain real-time performance:
- Temporal ROI tracking to avoid full detection on every frame
- Pipeline parallelism where possible
- Optional ML acceleration via GPU, DSP, or Edge TPU