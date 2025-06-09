licenses(["notice"])

package(default_visibility = ["//visibility:public"])

load("//mediapipe/framework:mediapipe_proto.bzl", "mediapipe_proto_library")

mediapipe_proto_library(
    name = "landmarks_to_json_calculator_proto",
    srcs = ["landmarks_to_json_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "landmarks_to_json_calculator",
    srcs = ["landmarks_to_json_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":landmarks_to_json_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/deps:json_to_proto",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:file_helpers",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)