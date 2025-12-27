# Testing 4-Channel Format Support in VideoWriter

This document describes how to test the implementation of 4-channel format support (BGRA, RGBA, BGRX, RGBX) in OpenCV's VideoWriter with GStreamer backend.

## Changes Made

1. **Format Detection**: Added code to detect 4-channel formats from appsrc caps in manual pipelines
2. **Write Function**: Added support for accepting CV_8UC4 frames when format is BGRA/RGBA/BGRX/RGBX
3. **Tests**: Added comprehensive tests to verify the functionality

## Building OpenCV with GStreamer Support

Before testing, ensure OpenCV is built with GStreamer support:

```bash
# Configure CMake with GStreamer
cmake -D WITH_GSTREAMER=ON \
      -D BUILD_TESTS=ON \
      -D BUILD_PERF_TESTS=ON \
      ..

# Build
make -j$(nproc)
```

## Running C++ Unit Tests

The C++ unit tests are located in `modules/videoio/test/test_gstreamer.cpp`:

```bash
# Run all GStreamer tests
./bin/opencv_test_videoio --gtest_filter="*gstreamer*"

# Run only the 4-channel format tests
./bin/opencv_test_videoio --gtest_filter="*videoio_gstreamer_4channel*"
```

The test `videoio_gstreamer_4channel.write_4channel` will test all four formats:
- BGRA
- RGBA  
- BGRX
- RGBX

## Running Python Test Script

A Python test script is provided for manual testing:

```bash
# Ensure OpenCV Python bindings are installed
python3 test_4channel_videowriter.py
```

The script will:
1. Check if GStreamer backend is available
2. Test writing frames in all 4-channel formats
3. Verify output files are created correctly

## Manual Testing Example

Here's a simple Python example to test the functionality:

```python
import cv2
import numpy as np

# Create a 4-channel BGRA frame
width, height = 320, 240
frame = np.zeros((height, width, 4), dtype=np.uint8)
frame[:, :, 0] = 255  # B
frame[:, :, 1] = 128  # G
frame[:, :, 2] = 64   # R
frame[:, :, 3] = 255  # A

# Create a GStreamer pipeline for writing
pipeline = (
    f"appsrc ! "
    f"video/x-raw, format=BGRA, width={width}, height={height}, framerate=30/1 ! "
    "videoconvert ! "
    "x264enc ! "
    "mp4mux ! "
    "filesink location=test_output.mp4"
)

# Open VideoWriter
writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (width, height))

if writer.isOpened():
    # Write the frame
    writer.write(frame)
    writer.release()
    print("Success! Frame written successfully.")
else:
    print("Error: Failed to open VideoWriter")
```

## Expected Behavior

### Before the Fix
- Writing CV_8UC4 frames would fail with: `"write frame skipped - expected CV_8UC3"`
- Only 3-channel BGR frames were accepted

### After the Fix
- Writing CV_8UC4 frames succeeds when format is BGRA/RGBA/BGRX/RGBX
- Format is automatically detected from pipeline caps
- Frames are written directly without conversion

## Troubleshooting

1. **GStreamer not found**: Ensure GStreamer development libraries are installed
2. **Pipeline errors**: Check that required GStreamer plugins are available (x264enc, mp4mux, etc.)
3. **Format not detected**: Verify the format is specified correctly in the pipeline string

## Related Issue

This implementation fixes GitHub issue #28296: "VideoWriter() support 4 channel format (ie BGRA, BGRX etc...)"

