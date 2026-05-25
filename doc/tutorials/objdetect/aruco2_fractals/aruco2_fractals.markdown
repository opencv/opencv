Detection of ArUco2 Fractal Markers {#tutorial_aruco2_fractals}
===================================

@prev_tutorial{tutorial_aruco2_diamonds}
@next_tutorial{tutorial_aruco2_pose}

|    |    |
| -: | :- |
| Original author | Rafael Muñoz-Salinas |
| Compatibility    | OpenCV >= 5.0.0 |

Fractal markers @cite romero2019fractal : are nested ArUco-like markers designed for extreme robustness and high-precision pose estimation. An outer marker contains one or more smaller markers at increasing scales.

![Fractal Marker](aruco2_fractal.jpg)

Benefits of Fractal Markers
---------------------------

- **Occlusion Robustness:** Fractal markers are designed to be detectable even when significant portions of the marker are covered or outside the camera's field of view.
- **Dynamic Range:** They can be detected at both long range (outer marker) and very close range (inner markers).
- **Precision:** The nested design provides many more image-to-3D correspondences than a standard marker by utilizing **inner corners**, significantly improving pose accuracy.

Occlusion and Inner Corners
---------------------------

One of the most powerful features of ArUco2 fractal markers is their ability to provide a pose even when partially occluded. While a standard ArUco marker requires all four corners to be visible for pose estimation, a fractal marker can use any combination of its nested markers.

Furthermore, `aruco2` detects all the **inner corners** of the fractal grid. Instead of having just 4 points for the `solvePnP` algorithm, a fractal marker can provide dozens or even hundreds of points, depending on the fractal level.

The following image illustrates a fractal marker that is partially outside the image frame. Despite the missing outer corners, the system detects the visible inner markers and their inner corners, allowing for a stable pose estimation (indicated by the coordinate axes):

![Occluded Fractal Marker with Inner Corners and Axis](aruco2_fractal_detected.jpg)

Fractal Types
-------------

`aruco2` supports several fractal configurations via the `cv::aruco2::FractalType` enum:
- `FRACTAL_2L_6`: 2 levels (outer + 1 inner marker).
- `FRACTAL_3L_6`: 3 levels.
- `FRACTAL_4L_6`: 4 levels.
- `FRACTAL_5L_6`: 5 levels (highest density of corners).

Fractal Creation
----------------

Generate a fractal marker image using `cv::aruco2::getFractalMarkerImage()`.

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_fractals.cpp create_fractal

The parameters are:
- The output image (`cv::Mat`).
- The fractal type.
- Optional: `bitSize` (default 20), the size of each bit in pixels.

Fractal Detection
-----------------

Detection is handled by the `cv::aruco2::detectFractals()`.

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_fractals.cpp detect_fractals

Each `cv::aruco2::FractalMarker` object contains:
- `corners`: The 4 corners of the outer marker.
- `type`: The fractal configuration used.
- `id`: The ID of the outer marker.

Drawing Detected Fractals
-------------------------

Visualize the detection with `cv::aruco2::drawFractals()`.

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_fractals.cpp draw_fractals

By default, this draws the outer border, the ID, and all matched image points (inner corners) as small circles. You can disable the inner points by passing `false` as the fourth parameter.
