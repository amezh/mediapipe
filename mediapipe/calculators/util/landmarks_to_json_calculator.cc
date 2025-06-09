#include <fstream>
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
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/calculators/util/landmarks_to_json/landmarks_to_json_calculator.pb.h"

namespace mediapipe {

// A calculator to convert various landmark formats to JSON string and optionally 
// write to a file.
//
// EXAMPLE:
// node {
//   calculator: "LandmarksToJsonCalculator"
//   input_stream: "IMAGE_SIZE:image_size"
//   input_stream: "POSE_LANDMARKS:pose_landmarks"
//   input_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
//   input_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
//   input_stream: "FACE_LANDMARKS:face_landmarks"
//   output_stream: "LANDMARKS_JSON:landmarks_json"
//   options: {
//     [mediapipe.LandmarksToJsonCalculatorOptions.ext] {
//       output_file: "/path/to/output.json"
//       append_mode: false  # Only save the latest frame
//     }
//   }
// }
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
  absl::Status Close(CalculatorContext* cc) override;

 private:
  std::string ConvertLandmarksToJson(const NormalizedLandmarkList& landmarks);
  absl::Status WriteJsonToFile(const std::string& json);
  
  std::string output_file_;
  bool append_mode_ = false;
  bool initialized_ = false;
  std::vector<std::string> frame_jsons_; // Store all frames if in append mode
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
  // Get options from the calculator options
  const auto& options = cc->Options<LandmarksToJsonCalculatorOptions>();
  if (options.has_output_file()) {
    output_file_ = options.output_file();
    // If not appending, clear the file content
    if (!options.append_mode()) {
      std::ofstream output_file(output_file_, std::ios::out | std::ios::trunc);
      if (!output_file.is_open()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Could not open output file: ", output_file_));
      }
      output_file.close();
    }
  }
  
  append_mode_ = options.append_mode();
  initialized_ = true;
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
  
  // Add frame timestamp
  absl::StrAppend(&json, "\"timestamp_us\":", cc->InputTimestamp().Microseconds());
  
  // Add image dimensions
  absl::StrAppend(&json, ",\"image_width\":", image_size.first, 
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
  
  // Output to stream
  cc->Outputs().Tag("LANDMARKS_JSON").Add(new std::string(json), cc->InputTimestamp());
  
  // Handle file output if configured
  if (!output_file_.empty()) {
    if (append_mode_) {
      // Store for writing all frames at the end
      frame_jsons_.push_back(json);
    } else {
      // Write current frame immediately, overwriting previous
      WriteJsonToFile(json);
    }
  }
  
  return absl::OkStatus();
}

absl::Status LandmarksToJsonCalculator::Close(CalculatorContext* cc) {
  // If in append mode, write all frames at once when closing
  if (!output_file_.empty() && append_mode_ && !frame_jsons_.empty()) {
    std::string all_frames_json = absl::StrCat("[", 
                                               absl::StrJoin(frame_jsons_, ","), 
                                               "]");
    WriteJsonToFile(all_frames_json);
  }
  
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

absl::Status LandmarksToJsonCalculator::WriteJsonToFile(const std::string& json) {
  if (output_file_.empty()) {
    return absl::OkStatus();
  }
  
  std::ofstream output_file(output_file_, std::ios::out | std::ios::trunc);
  if (!output_file.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not open output file: ", output_file_));
  }
  
  output_file << json;
  output_file.close();
  
  return absl::OkStatus();
}

REGISTER_CALCULATOR(LandmarksToJsonCalculator);

}  // namespace mediapipe