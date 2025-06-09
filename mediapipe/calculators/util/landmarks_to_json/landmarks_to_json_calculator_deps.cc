#include "mediapipe/framework/calculator_registry.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

// This file ensures that the LandmarksToJsonCalculator is properly linked when using
// the calculator in a MediaPipe graph.
namespace mediapipe {
namespace {

// Forces the calculator to be included in the registry.
extern "C" {
// This is a workaround for the fact that mediapipe doesn't expose the calculator registration.
// It ensures the calculator is linked in if this file is linked in.
int LandmarksToJsonCalculatorDeps() { return 0; }
}  // extern "C"

}  // namespace
}  // namespace mediapipe