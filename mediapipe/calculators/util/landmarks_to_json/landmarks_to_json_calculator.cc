#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/deps/json_to_proto.h"
#include "mediapipe/framework/port/map_util.h"

namespace mediapipe {

// A calculator to convert various landmark formats to JSON string.
//
// Input streams:
//   IMAGE_SIZE - Size of the image where the landmarks were detected.
//   POSE_LANDMARKS - NormalizedLandmarkList containing pose landmarks.
//   LEFT_HAND_LANDMARKS - NormalizedLandmarkList containing left hand landmarks.
//   RIGHT_HAND_LANDMARKS - NormalizedLandmarkList containing right hand landmarks.
//   FACE_LANDMARKS - NormalizedLandmarkList containing face landmarks.
//
// Output stream:
//   LANDMARKS_JSON - string containing JSON representation of all landmarks.
//
class LandmarksToJsonCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  std::string ConvertLandmarksToJson(const NormalizedLandmarkList& landmarks);
};

absl::Status LandmarksToJsonCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag("IMAGE_SIZE").Set<std::pair<int, int>>();
  cc->Inputs().Tag("POSE_LANDMARKS").Set<NormalizedLandmarkList>();
  cc->Inputs().Tag("LEFT_HAND_LANDMARKS").Set<NormalizedLandmarkList>();
  cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").Set<NormalizedLandmarkList>();
  cc->Inputs().Tag("FACE_LANDMARKS").Set<NormalizedLandmarkList>();
  
  cc->Outputs().Tag("LANDMARKS_JSON").Set<std::string>();
  
  return absl::OkStatus();
}

absl::Status LandmarksToJsonCalculator::Open(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status LandmarksToJsonCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag("IMAGE_SIZE").IsEmpty() ||
      (cc->Inputs().Tag("POSE_LANDMARKS").IsEmpty() &&
       cc->Inputs().Tag("LEFT_HAND_LANDMARKS").IsEmpty() &&
       cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").IsEmpty() &&
       cc->Inputs().Tag("FACE_LANDMARKS").IsEmpty())) {
    return absl::OkStatus();
  }
  
  auto image_size = cc->Inputs().Tag("IMAGE_SIZE").Get<std::pair<int, int>>();
  
  // Build JSON structure
  std::string json = "{";
  
  // Add image dimensions
  absl::StrAppend(&json, "\"image_width\":", image_size.first, 
                 ",\"image_height\":", image_size.second);
  
  // Process pose landmarks
  if (!cc->Inputs().Tag("POSE_LANDMARKS").IsEmpty()) {
    const auto& landmarks = cc->Inputs().Tag("POSE_LANDMARKS").Get<NormalizedLandmarkList>();
    if (!landmarks.landmark().empty()) {
      std::string landmarks_json = ConvertLandmarksToJson(landmarks);
      absl::StrAppend(&json, ",\"pose_landmarks\":", landmarks_json);
    }
  }
  
  // Process left hand landmarks
  if (!cc->Inputs().Tag("LEFT_HAND_LANDMARKS").IsEmpty()) {
    const auto& landmarks = cc->Inputs().Tag("LEFT_HAND_LANDMARKS").Get<NormalizedLandmarkList>();
    if (!landmarks.landmark().empty()) {
      std::string landmarks_json = ConvertLandmarksToJson(landmarks);
      absl::StrAppend(&json, ",\"left_hand_landmarks\":", landmarks_json);
    }
  }
  
  // Process right hand landmarks
  if (!cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").IsEmpty()) {
    const auto& landmarks = cc->Inputs().Tag("RIGHT_HAND_LANDMARKS").Get<NormalizedLandmarkList>();
    if (!landmarks.landmark().empty()) {
      std::string landmarks_json = ConvertLandmarksToJson(landmarks);
      absl::StrAppend(&json, ",\"right_hand_landmarks\":", landmarks_json);
    }
  }
  
  // Process face landmarks
  if (!cc->Inputs().Tag("FACE_LANDMARKS").IsEmpty()) {
    const auto& landmarks = cc->Inputs().Tag("FACE_LANDMARKS").Get<NormalizedLandmarkList>();
    if (!landmarks.landmark().empty()) {
      std::string landmarks_json = ConvertLandmarksToJson(landmarks);
      absl::StrAppend(&json, ",\"face_landmarks\":", landmarks_json);
    }
  }
  
  absl::StrAppend(&json, "}");
  
  cc->Outputs().Tag("LANDMARKS_JSON").Add(new std::string(json), cc->InputTimestamp());
  
  return absl::OkStatus();
}

std::string LandmarksToJsonCalculator::ConvertLandmarksToJson(
    const NormalizedLandmarkList& landmarks) {
  std::vector<std::string> landmark_strings;
  
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);
    std::string landmark_json = absl::StrCat(
        "{\"x\":", landmark.x(),
        ",\"y\":", landmark.y(),
        ",\"z\":", landmark.z(),
        ",\"visibility\":", landmark.has_visibility() ? absl::StrCat(landmark.visibility()) : "null",
        ",\"presence\":", landmark.has_presence() ? absl::StrCat(landmark.presence()) : "null",
        "}");
    landmark_strings.push_back(landmark_json);
  }
  
  return absl::StrCat("[", absl::StrJoin(landmark_strings, ","), "]");
}

REGISTER_CALCULATOR(LandmarksToJsonCalculator);

}  // namespace mediapipe