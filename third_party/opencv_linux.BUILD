cc_library(
    name = "opencv",
    hdrs = glob(["include/opencv4/opencv2/**/*.h*"]),
    includes = ["include/opencv4"],
    linkopts = [
        "-Wl,-rpath,/usr/local/lib",
        "-L/usr/local/lib",
        "-lopencv_core",
        "-lopencv_highgui",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_videoio",      # <-- missing
        "-lopencv_video",        # optional if needed
    ],
    visibility = ["//visibility:public"],
)