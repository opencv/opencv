Calibration with ArUco and ChArUco {#tutorial_aruco_calibration}
==================================

@prev_tutorial{tutorial_charuco_diamond_detection}
@next_tutorial{tutorial_aruco_faq}

The ArUco module can also be used to calibrate a camera. Camera calibration consists in obtaining the
camera intrinsic parameters and distortion coefficients. This parameters remain fixed unless the camera
optic is modified, thus camera calibration only need to be done once.

Camera calibration is usually performed using the OpenCV `cv::calibrateCamera()` function. This function
requires some correspondences between environment points and their projection in the camera image from
different viewpoints. In general, these correspondences are obtained from the corners of chessboard
patterns. See `cv::calibrateCamera()` function documentation or the OpenCV calibration tutorial for
more detailed information.

Using the ArUco module, calibration can be performed based on ArUco markers corners or ChArUco corners.
Calibrating using ArUco is much more versatile than using traditional chessboard patterns, since it
allows occlusions or partial views.

As it can be stated, calibration can be done using both, marker corners or ChArUco corners. However,
it is highly recommended using the ChArUco corners approach since the provided corners are much
more accurate in comparison to the marker corners. Calibration using a standard Board should only be
employed in those scenarios where the ChArUco boards cannot be employed because of any kind of restriction.

Calibration with ChArUco Boards
-------------------------------

To calibrate using a ChArUco board, it is necessary to detect the board from different viewpoints, in the
same way that the standard calibration does with the traditional chessboard pattern. However, due to the
benefits of using ChArUco, occlusions and partial views are allowed, and not all the corners need to be
visible in all the viewpoints.

![ChArUco calibration viewpoints](images/charucocalibration.jpg)

The example of using `cv::calibrateCamera()` for cv::aruco::CharucoBoard:

@snippet samples/cpp/tutorial_code/objectDetection/calibrate_camera_charuco.cpp CalibrationWithCharucoBoard1
@snippet samples/cpp/tutorial_code/objectDetection/calibrate_camera_charuco.cpp CalibrationWithCharucoBoard2
@snippet samples/cpp/tutorial_code/objectDetection/calibrate_camera_charuco.cpp CalibrationWithCharucoBoard3

The ChArUco corners and ChArUco identifiers captured on each viewpoint are stored in the vectors
`allCharucoCorners` and `allCharucoIds`, one element per viewpoint.

The `calibrateCamera()` function will fill the `cameraMatrix` and `distCoeffs` arrays with the
camera calibration parameters. It will return the reprojection error obtained from the calibration.
The elements in `rvecs` and `tvecs` will be filled with the estimated pose of the camera
(respect to the ChArUco board) in each of the viewpoints.

Finally, the `calibrationFlags` parameter determines some of the options for the calibration.

A full working example is included in the `calibrate_camera_charuco.cpp` inside the
`samples/cpp/tutorial_code/objectDetection` folder.

The samples now take input via commandline via the `cv::CommandLineParser`. For this file the example
parameters will look like:
@code{.cpp}
    "camera_calib.txt" -w=5 -h=7 -sl=0.04 -ml=0.02 -d=10
    -v=path/img_%02d.jpg
@endcode

The camera calibration parameters from `opencv/samples/cpp/tutorial_code/objectDetection/tutorial_camera_charuco.yml`
were obtained by the `img_00.jpg-img_03.jpg` placed from this
[folder](https://github.com/opencv/opencv_contrib/tree/4.6.0/modules/aruco/tutorials/aruco_calibration/images).

Calibration with ArUco Boards
-----------------------------

As it has been stated, it is recommended the use of ChAruco boards instead of ArUco boards for camera
calibration, since ChArUco corners are more accurate than marker corners. However, in some special cases
it must be required to use calibration based on ArUco boards. As in the previous case, it requires
the detections of an ArUco board from different viewpoints.

![ArUco calibration viewpoints](images/arucocalibration.jpg)

The example of using `cv::calibrateCamera()` for cv::aruco::GridBoard:

@snippet samples/cpp/tutorial_code/objectDetection/calibrate_camera.cpp CalibrationWithArucoBoard1
@snippet samples/cpp/tutorial_code/objectDetection/calibrate_camera.cpp CalibrationWithArucoBoard2
@snippet samples/cpp/tutorial_code/objectDetection/calibrate_camera.cpp CalibrationWithArucoBoard3

A full working example is included in the `calibrate_camera.cpp` inside the `samples/cpp/tutorial_code/objectDetection` folder.

The samples now take input via commandline via the `cv::CommandLineParser`. For this file the example
parameters will look like:
@code{.cpp}
    "camera_calib.txt" -w=5 -h=7 -l=100 -s=10 -d=10 -v=path/aruco_videos_or_images
@endcode
