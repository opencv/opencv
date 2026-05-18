Detection of ArUco2 Markers in Java {#tutorial_aruco2_java_detection}
===================================

@prev_tutorial{tutorial_aruco2_pose}

|    |    |
| -: | :- |
| Original author | Rafael Muñoz-Salinas |
| Compatibility    | OpenCV >= 5.0.0 |

Goals
-----

In this tutorial you will learn:
- How to generate ArUco2 markers with Java.
- How to detect ArUco2 markers in an image using Java.
- How to handle multiple dictionaries in a single pass.
- How to configure detection parameters.

Introduction
------------

The `aruco2` module is a modern, high-performance replacement for the legacy `aruco` module in OpenCV 5. It is designed to be faster, more robust, and easier to use.

Key benefits of `aruco2`:
- **Speed**: Up to 6.5x faster detection than legacy ArUco.
- **Robustness**: Better error correction and lower false positive rates.
- **Modern API**: Simplified functions that return easy-to-use objects.
- **Multi-dictionary support**: Detect markers from different families (e.g., ArUco and AprilTag) simultaneously.

The Java API for `aruco2` is provided through the `org.opencv.objdetect.Aruco2` class. All functions are static and operate on OpenCV `Mat` objects.

Prerequisites
-------------

Make sure you have the OpenCV Java library loaded in your application:

@code{.java}
System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
@endcode

Source Code
-----------

Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/java/tutorial_code/objdetect/aruco2/Aruco2Detection.java).

@include java/tutorial_code/objdetect/aruco2/Aruco2Detection.java

Marker Creation
---------------

Before detection, you need to generate and print markers. In Java, use `Aruco2.getFiducialMarker()` for this. The output image is passed as an empty `Mat` object.

@snippet java/tutorial_code/objdetect/aruco2/Aruco2Detection.java generate_marker

The parameters are:
- The output image (`Mat`).
- The dictionary type (e.g., `Aruco2.DICT_ARUCO_MIP_36h12`).
- The marker ID (must be valid for the chosen dictionary).
- The size of each bit in pixels (`bitSize`).
- Whether to add an external white border (`externalBorder`).

Marker Detection
----------------

Detection is done with a single call to `Aruco2.detectFiducialMarkers()`. It returns a `List<FiducialMarker>` containing all detected markers.

@snippet java/tutorial_code/objdetect/aruco2/Aruco2Detection.java detect_single

Each `FiducialMarker` in the list provides:
- `get_id()`: the marker identifier.
- `get_corners()`: the four corner points in the image.
- `get_dict()`: the dictionary the marker was found in.

Drawing Detected Markers
------------------------

To visualize the detection results, convert the grayscale image to BGR and call `Aruco2.drawFiducialMarkers()`. The image is modified in-place.

@snippet java/tutorial_code/objdetect/aruco2/Aruco2Detection.java draw_markers

Multi-Dictionary Detection
--------------------------

One of the most powerful features of `aruco2` is detecting markers from multiple dictionaries at once. In Java, pass a `MatOfInt` containing the dictionary constants.

@snippet java/tutorial_code/objdetect/aruco2/Aruco2Detection.java multi_dict

Advanced: Detection Parameters
------------------------------

You can tune the detection process using `DetectionParameters`. Create an instance, adjust the fields, and pass it to `detectFiducialMarkers()`.

@snippet java/tutorial_code/objdetect/aruco2/Aruco2Detection.java params

Key parameters include:
- `boxFilterSize`: size of the adaptive thresholding kernel (must be odd).
- `thres`: threshold offset applied after box filtering.
- `errorCorrectionRate`: fraction of error-correction capacity to use (0 = no errors tolerated).
- `detectInvertedMarker`: set to `true` to detect white-on-black markers.

Grid Board Detection
--------------------

`aruco2` also supports grid boards for more robust pose estimation. Generate a board image with `getGridBoard()`, then detect it with `detectGridBoard()`.

@snippet java/tutorial_code/objdetect/aruco2/Aruco2Detection.java grid_board

`detectGridBoard()` returns a `boolean` indicating whether any board marker was found. The `Aruco2_GridBoard` object is populated with the detected markers and can be accessed via `get_markers()`.

For pose estimation, use `Aruco2.getSolvePnpPoints(board, objPoints, imgPoints, markerSize)` to obtain 3D-2D correspondences suitable for `Calib3d.solvePnP()`.
