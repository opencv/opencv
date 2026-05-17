Pose Estimation with ArUco2 {#tutorial_aruco2_pose}
============================

@prev_tutorial{tutorial_aruco2_fractals}

|    |    |
| -: | :- |
| Original author | Rafael Muñoz-Salinas |
| Compatibility    | OpenCV >= 5.0.0 |

Pose estimation is the process of determining the 3D position and orientation (the "pose") of a camera relative to a target. In ArUco2, this process is unified across all target types (markers, boards, diamonds, and fractals) using a consistent API pattern.

Unified Pose Estimation Pattern
-------------------------------

The `aruco2` module provides overloaded versions of `cv::aruco2::getSolvePnpPoints()` for each target type. This function extracts the 2D image points and their corresponding 3D object points, which are then passed to the standard `cv::solvePnP()` function.

The general workflow is:
1. Detect the target.
2. Use `getSolvePnpPoints()` to obtain `objPoints` and `imgPoints`.
3. Call `cv::solvePnP()` with your camera calibration data.
4. Draw the resulting pose using `cv::aruco2::drawAxis()`.

### Pose Estimation for a Single Marker

@code{.cpp}
cv::Mat cameraMatrix, distCoeffs; // Load from calibration file
float markerSize = 0.05f;        // Physical side length in meters (e.g., 5 cm)

for (const auto &m : cv::aruco2::detectMarkers(image)) {
    cv::Mat objPoints, imgPoints, rvec, tvec;
    
    // 1. Extract points
    cv::aruco2::getSolvePnpPoints(m, objPoints, imgPoints, markerSize);
    
    // 2. Estimate pose
    cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
    
    // 3. Draw XYZ axes (Red=X, Green=Y, Blue=Z)
    cv::aruco2::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, markerSize);
}
@endcode

The following image shows a single marker with its estimated pose visualized as a 3D coordinate system:

![Pose Estimation for a Single Marker](marker_axis.jpg)

### Pose Estimation for Boards, Diamonds, and Fractals

The same pattern applies to more complex targets. The only difference is the target object passed to `getSolvePnpPoints()`.

**For a Board:**
@code{.cpp}
cv::aruco2::getSolvePnpPoints(board, objPoints, imgPoints, markerSize);
@endcode

**For a Diamond:**
@code{.cpp}
cv::aruco2::getSolvePnpPoints(diamond, objPoints, imgPoints, markerSize);
@endcode

**For a Fractal Marker:**
@code{.cpp}
cv::aruco2::getSolvePnpPoints(fractal, objPoints, imgPoints, markerSize);
@endcode

Note on Calibration
-------------------

Accurate pose estimation requires the camera's intrinsic parameters (`cameraMatrix` and `distCoeffs`). These are obtained through the @ref tutorial_aruco2_calibration process. If the image is already undistorted, `distCoeffs` can be passed as an empty `cv::Mat()`.

Visualizing the Pose
--------------------

The `cv::aruco2::drawAxis()` function projects the origin and the three axis tips onto the image plane. By convention:
- **X-axis** is Red.
- **Y-axis** is Green.
- **Z-axis** is Blue.

@code{.cpp}
cv::aruco2::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, axisLength);
@endcode
The `axisLength` parameter should be in the same units as your `markerSize` (e.g., meters).
