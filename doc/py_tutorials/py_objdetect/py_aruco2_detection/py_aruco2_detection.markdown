Detection of ArUco2 Markers {#tutorial_py_aruco2_detection}
===========================

Goals
-----

In this tutorial you will learn:
- What ArUco2 markers are and why they are useful.
- How to generate ArUco2 markers with Python.
- How to detect ArUco2 markers in an image.
- How to handle multiple dictionaries in a single pass.

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

Before detection, you need to generate and print markers. Use `cv.aruco2.getFiducialMarker()` for this.

@code{.py}
import cv2 as cv

# Select a dictionary
dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12

# Generate a marker (ID 42, 200x200 pixels)
# bitSize=20 means each bit in the 6x6 grid will be 20x20 pixels
marker_img = cv.aruco2.getFiducialMarker(dictionary, 42, bitSize=20)

cv.imwrite("marker42.png", marker_img)
cv.imshow("Marker 42", marker_img)
cv.waitKey(0)
@endcode

Marker Detection
----------------

Detection is done with a single call to `cv.aruco2.detectFiducialMarkers()`.

@code{.py}
import cv2 as cv

# Load image
img = cv.imread("scene.jpg")

# Select the same dictionary used for creation
dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12

# Detect markers
markers = cv.aruco2.detectFiducialMarkers(img, dictionary)

# Iterate over detected markers
for m in markers:
    print(f"Detected marker ID: {m.id}")
    print(f"Corners: {m.corners}")

# Visualize results
cv.aruco2.drawFiducialMarkers(img, markers)
cv.imshow("Detection", img)
cv.waitKey(0)
@endcode

Multi-Dictionary Detection
--------------------------

One of the most powerful features of `aruco2` is detecting markers from multiple dictionaries at once. Just pass a list of dictionaries.

@code{.py}
# Detect both ArUco and AprilTag markers
dictionaries = [cv.aruco2.DICT_ARUCO_MIP_36h12, cv.aruco2.DICT_APRILTAG_36h11]
markers = cv.aruco2.detectFiducialMarkers(img, dictionaries)

for m in markers:
    dict_name = "ArUco" if m.dict == cv.aruco2.DICT_ARUCO_MIP_36h12 else "AprilTag"
    print(f"Found {dict_name} marker ID: {m.id}")
@endcode

Advanced: Detection Parameters
------------------------------

You can tune the detection process using `cv.aruco2.DetectionParameters`.

@code{.py}
params = cv.aruco2.DetectionParameters()
params.boxFilterSize = 15
params.thres = 3

markers = cv.aruco2.detectFiducialMarkers(img, dictionary, detectorParams=params)
@endcode
