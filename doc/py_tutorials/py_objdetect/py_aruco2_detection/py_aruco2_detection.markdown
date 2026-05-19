Detection of ArUco2 Markers in Python {#tutorial_py_aruco2_detection}
===========================

@prev_tutorial{tutorial_aruco2_java_detection}
@next_tutorial{tutorial_js_aruco2_detection}

Goals
-----

In this tutorial you will learn:
- What ArUco2 markers are and why they are useful.
- How to generate ArUco2 markers with Python.
- How to detect ArUco2 markers in an image.
- How to handle multiple dictionaries in a single pass.
- How to detect Boards, Diamonds and Fractal markers.
- How to estimate camera pose.

Introduction
------------

The `aruco2` module is a modern, high-performance replacement for the legacy `aruco` module in OpenCV 5. It is designed to be faster, more robust, and easier to use.

Key benefits of `aruco2`:
- **Speed**: Up to 6.5x faster detection than legacy ArUco.
- **Robustness**: Better error correction and lower false positive rates.
- **Modern API**: Simplified functions that return easy-to-use objects.
- **Multi-dictionary support**: Detect markers from different families (e.g., ArUco and AprilTag) simultaneously.

Marker Creation
---------------

Before detection, you need to generate and print markers. Use `cv.aruco2.getFiducialMarkerImage()` for this.

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py marker_creation

Marker Detection
----------------

Detection is done with a single call to `cv.aruco2.detectFiducialMarkers()`.

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py marker_detection

Multi-Dictionary Detection
--------------------------

One of the most powerful features of `aruco2` is detecting markers from multiple dictionaries at once. Just pass a list of dictionaries.

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py multi_dict

Detection Parameters
--------------------

You can tune the detection process using `cv.aruco2.DetectionParameters`.

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py detection_params

Pose Estimation
---------------

To estimate the pose of a marker, you need the camera calibration parameters (camera matrix and distortion coefficients).

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py pose_estimation

Advanced Markers: Boards, Diamonds and Fractals
-----------------------------------------------

`aruco2` supports more complex marker arrangements that provide better pose estimation accuracy and robustness to occlusion.

### Grid Boards

A grid board is a rectangular arrangement of markers. It allows pose estimation even if some markers are occluded.

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py grid_board

### Diamond Markers

A diamond marker is a 2x2 arrangement of markers that behaves like a single high-resolution marker.

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py diamonds

### Fractal Markers

Fractal markers are nested markers that can be detected at very different distances.

@snippet samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py fractals

Full Source Code
----------------

You can find the full source code for this tutorial in `samples/python/tutorial_code/objdetect/aruco2/py_aruco2_detection.py`.

