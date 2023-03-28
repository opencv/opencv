Calibration with ArUco and ChArUco {#tutorial_aruco_calibration}
==============================

@prev_tutorial{tutorial_charuco_diamond_detection}
@next_tutorial{tutorial_aruco_faq}

The ArUco module can also be used to calibrate a camera. Camera calibration consists in obtaining the
camera intrinsic parameters and distortion coefficients. This parameters remain fixed unless the camera
optic is modified, thus camera calibration only need to be done once.

Camera calibration is usually performed using the OpenCV ```calibrateCamera()``` function. This function
requires some correspondences between environment points and their projection in the camera image from
different viewpoints. In general, these correspondences are obtained from the corners of chessboard
patterns. See ```calibrateCamera()``` function documentation or the OpenCV calibration tutorial for
more detailed information.

Using the ArUco module, calibration can be performed based on ArUco markers corners or ChArUco corners.
Calibrating using ArUco is much more versatile than using traditional chessboard patterns, since it
allows occlusions or partial views.

As it can be stated, calibration can be done using both, marker corners or ChArUco corners. However,
it is highly recommended using the ChArUco corners approach since the provided corners are much
more accurate in comparison to the marker corners. Calibration using a standard Board should only be
employed in those scenarios where the ChArUco boards cannot be employed because of any kind of restriction.

Calibration with ChArUco Boards
------

To calibrate using a ChArUco board, it is necessary to detect the board from different viewpoints, in the
same way that the standard calibration does with the traditional chessboard pattern. However, due to the
benefits of using ChArUco, occlusions and partial views are allowed, and not all the corners need to be
visible in all the viewpoints.

![ChArUco calibration viewpoints](images/charucocalibration.png)

The function to calibrate is ```calibrateCameraCharuco()```. Example:

@code{.cpp}
    cv::Ptr<aruco::CharucoBoard> board = ... // create charuco board
    cv::Size imgSize = ... // camera image size

    std::vector<std::vector<cv::Point2f>> allCharucoCorners;
    std::vector<std::vector<int>> allCharucoIds;
    // Detect charuco board from several viewpoints and fill allCharucoCorners and allCharucoIds
    ...
    ...

    // After capturing in several viewpoints, start calibration
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    int calibrationFlags = ... // Set calibration flags (same than in calibrateCamera() function)

    double repError = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);
@endcode

The ChArUco corners and ChArUco identifiers captured on each viewpoint are stored in the vectors ```allCharucoCorners``` and ```allCharucoIds```, one element per viewpoint.

The ```calibrateCameraCharuco()``` function will fill the ```cameraMatrix``` and ```distCoeffs``` arrays with the camera calibration parameters. It will return the reprojection
error obtained from the calibration. The elements in ```rvecs``` and ```tvecs``` will be filled with the estimated pose of the camera (respect to the ChArUco board)
in each of the viewpoints.

Finally, the ```calibrationFlags``` parameter determines some of the options for the calibration. Its format is equivalent to the flags parameter in the OpenCV
```calibrateCamera()``` function.

A full working example is included in the `calibrate_camera_charuco.cpp` inside the `modules/aruco/samples/`.

Note: The samples now take input via commandline via the [OpenCV Commandline Parser](http://docs.opencv.org/trunk/d0/d2e/classcv_1_1CommandLineParser.html#gsc.tab=0). For this file the example parameters will look like
@code{.cpp}
    "output_path/camera_calib.txt" -w=5 -h=7 -sl=0.04 -ml=0.02 -d=10
    -v="path_aruco/tutorials/aruco_calibration/images/img_%02d.jpg
    -c=path_aruco/samples/tutorial_camera_params.yml
@endcode

The camera calibration parameters from `samples/tutorial_camera_charuco.yml` were obtained by `aruco_calibration/images/img_00.jpg-img_03.jpg`.

Calibration with ArUco Boards
------

As it has been stated, it is recommended the use of ChAruco boards instead of ArUco boards for camera calibration, since
ChArUco corners are more accurate than marker corners. However, in some special cases it must be required to use calibration
based on ArUco boards. For these cases, the ```calibrateCameraAruco()``` function is provided. As in the previous case, it
requires the detections of an ArUco board from different viewpoints.

![ArUco calibration viewpoints](images/arucocalibration.png)

Example of ```calibrateCameraAruco()``` use:

@code{.cpp}
    cv::Ptr<aruco::Board> board = ... // create aruco board
    cv::Size imgSize = ... // camera image size

    std::vector<std::vector<cv::Point2f>> allCornersConcatenated;
    std::vector<int> allIdsConcatenated;
    std::vector<int> markerCounterPerFrame;
    // Detect aruco board from several viewpoints and fill allCornersConcatenated, allIdsConcatenated and markerCounterPerFrame
    ...
    ...

    // After capturing in several viewpoints, start calibration
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    int calibrationFlags = ... // Set calibration flags (same than in calibrateCamera() function)

    double repError = cv::aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated, markerCounterPerFrame, board, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);
@endcode

In this case, and contrary to the ```calibrateCameraCharuco()``` function, the detected markers on each viewpoint are concatenated in the arrays ```allCornersConcatenated``` and
```allCornersConcatenated``` (the first two parameters). The third parameter, the array ```markerCounterPerFrame```, indicates the number of marker detected on each viewpoint.
The rest of parameters are the same than in ```calibrateCameraCharuco()```, except the board layout object which does not need to be a ```CharucoBoard``` object, it can be
any ```Board``` object.

A full working example is included in the `calibrate_camera.cpp` inside the `modules/aruco/samples/`.

Note: The samples now take input via commandline via the [OpenCV Commandline Parser](http://docs.opencv.org/trunk/d0/d2e/classcv_1_1CommandLineParser.html#gsc.tab=0). For this file the example parameters will look like
@code{.cpp}
    "_path_/calib.txt" -w=5 -h=7 -l=100 -s=10 -d=10
@endcode
