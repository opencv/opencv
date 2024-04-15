Detection of ChArUco Boards {#tutorial_charuco_detection}
===========================

@prev_tutorial{tutorial_aruco_board_detection}
@next_tutorial{tutorial_charuco_diamond_detection}

ArUco markers and boards are very useful due to their fast detection and their versatility.
However, one of the problems of ArUco markers is that the accuracy of their corner positions is not
too high, even after applying subpixel refinement.

On the contrary, the corners of chessboard patterns can be refined more accurately since each corner
is surrounded by two black squares. However, finding a chessboard pattern is not as versatile as
finding an ArUco board: it has to be completely visible and occlusions are not permitted.

A ChArUco board tries to combine the benefits of these two approaches:

![Charuco definition](images/charucodefinition.png)

The ArUco part is used to interpolate the position of the chessboard corners, so that it has the
versatility of marker boards, since it allows occlusions or partial views. Moreover, since the
interpolated corners belong to a chessboard, they are very accurate in terms of subpixel accuracy.

When high precision is necessary, such as in camera calibration, Charuco boards are a better option
than standard ArUco boards.

Goal
----

In this tutorial you will learn:

- How to create a charuco board ?
- How to detect the charuco corners without performing camera calibration ?
- How to detect the charuco corners with camera calibration and pose estimation ?

Source code
-----------

You can find this code in `samples/cpp/tutorial_code/objectDetection/detect_board_charuco.cpp`

Here's a sample code of how to achieve all the stuff enumerated at the goal list.

@snippet samples/cpp/tutorial_code/objectDetection/detect_board_charuco.cpp charuco_detect_board_full_sample

ChArUco Board Creation
----------------------

The aruco module provides the `cv::aruco::CharucoBoard` class that represents a Charuco Board and
which inherits from the `cv::aruco::Board` class.

This class, as the rest of ChArUco functionalities, are defined in:

@snippet samples/cpp/tutorial_code/objectDetection/detect_board_charuco.cpp charucohdr

To define a `cv::aruco::CharucoBoard`, it is necessary:

- Number of chessboard squares in X and Y directions.
- Length of square side.
- Length of marker side.
- The dictionary of the markers.
- Ids of all the markers.

As for the `cv::aruco::GridBoard` objects, the aruco module provides to create `cv::aruco::CharucoBoard`
easily. This object can be easily created from these parameters using the `cv::aruco::CharucoBoard`
constructor:

@snippet samples/cpp/tutorial_code/objectDetection/create_board_charuco.cpp create_charucoBoard

- The first parameter is the number of squares in X and Y direction respectively.
- The second and third parameters are the length of the squares and the markers respectively. They can
  be provided in any unit, having in mind that the estimated pose for this board would be measured
  in the same units (usually meters are used).
- Finally, the dictionary of the markers is provided.

The ids of each of the markers are assigned by default in ascending order and starting on 0, like in
`cv::aruco::GridBoard` constructor. This can be easily customized by accessing to the ids vector
through `board.ids`, like in the `cv::aruco::Board` parent class.

Once we have our `cv::aruco::CharucoBoard` object, we can create an image to print it. There are
two ways to do this:
1. By using the script `doc/patter_tools/gen_pattern.py `, see @subpage tutorial_camera_calibration_pattern.
2. By using the function `cv::aruco::CharucoBoard::generateImage()`.

The function `cv::aruco::CharucoBoard::generateImage()` is provided in cv::aruco::CharucoBoard class
and can be called by using the following code:
@snippet samples/cpp/tutorial_code/objectDetection/create_board_charuco.cpp generate_charucoBoard

- The first parameter is the size of the output image in pixels. If this is not proportional
to the board dimensions, it will be centered on the image.
- The second parameter is the output image with the charuco board.
- The third parameter is the (optional) margin in pixels, so none of the markers are touching the
  image border.
- Finally, the size of the marker border, similarly to `cv::aruco::generateImageMarker()` function.
  The default value is 1.

The output image will be something like this:

![](images/charucoboard.png)

A full working example is included in the `create_board_charuco.cpp` inside the `samples/cpp/tutorial_code/objectDetection/`.

The samples `create_board_charuco.cpp` now take input via commandline via the `cv::CommandLineParser`.
For this file the example
parameters will look like:
@code{.cpp}
    "_output_path_/chboard.png" -w=5 -h=7 -sl=100 -ml=60 -d=10
@endcode


ChArUco Board Detection
-----------------------

When you detect a ChArUco board, what you are actually detecting is each of the chessboard corners
of the board.

Each corner on a ChArUco board has a unique identifier (id) assigned. These ids go from 0 to the total
number of corners in the board.
The steps of charuco board detection can be broken down to the following steps:

- **Taking input Image**

@snippet samples/cpp/tutorial_code/objectDetection/detect_board_charuco.cpp inputImg

The original image where the markers are to be detected. The image is necessary to perform subpixel
refinement in the ChArUco corners.

- **Reading the camera calibration Parameters(only for detection with camera calibration)**

@snippet samples/cpp/tutorial_code/objectDetection/aruco_samples_utility.hpp camDistCoeffs

The parameters of `readCameraParameters` are:
- The first parameter is the path to the camera intrinsic matrix and distortion coefficients.
- The second and third parameters are cameraMatrix and distCoeffs.

This function takes these parameters as input and returns a boolean value of whether the camera
calibration parameters are valid or not. For detection of charuco corners without calibration,
this step is not required.

- **Detecting the markers and interpolation of charuco corners from markers**

The detection of the ChArUco corners is based on the previous detected markers.
So that, first markers are detected, and then ChArUco corners are interpolated from markers.
The method that detect the ChArUco corners is `cv::aruco::CharucoDetector::detectBoard()`.

@snippet samples/cpp/tutorial_code/objectDetection/detect_board_charuco.cpp interpolateCornersCharuco

The parameters of detectBoard are:
- `image` - Input image.
- `charucoCorners` - output list of image positions of the detected corners.
- `charucoIds` - output ids for each of the detected corners in `charucoCorners`.
- `markerCorners` - input/output vector of detected marker corners.
- `markerIds` - input/output vector of identifiers of the detected markers

If markerCorners and markerIds are empty, the function will detect aruco markers and ids.

If calibration parameters are provided, the ChArUco corners are interpolated by, first, estimating
a rough pose from the ArUco markers and, then, reprojecting the ChArUco corners back to the image.

On the other hand, if calibration parameters are not provided, the ChArUco corners are interpolated
by calculating the corresponding homography between the ChArUco plane and the ChArUco image projection.

The main problem of using homography is that the interpolation is more sensible to image distortion.
Actually, the homography is only performed using the closest markers of each ChArUco corner to reduce
the effect of distortion.

When detecting markers for ChArUco boards, and specially when using homography, it is recommended to
disable the corner refinement of markers. The reason of this is that, due to the proximity of the
chessboard squares, the subpixel process can produce important deviations in the corner positions and
these deviations are propagated to the ChArUco corner interpolation, producing poor results.

@note To avoid deviations, the margin between chessboard square and aruco marker should be greater
than 70% of one marker module.

Furthermore, only those corners whose two surrounding markers have be found are returned. If any of
the two surrounding markers has not been detected, this usually means that there is some occlusion
or the image quality is not good in that zone. In any case, it is preferable not to consider that
corner, since what we want is to be sure that the interpolated ChArUco corners are very accurate.

After the ChArUco corners have been interpolated, a subpixel refinement is performed.

Once we have interpolated the ChArUco corners, we would probably want to draw them to see if their
detections are correct. This can be easily done using the `cv::aruco::drawDetectedCornersCharuco()`
function:

@snippet samples/cpp/tutorial_code/objectDetection/detect_board_charuco.cpp drawDetectedCornersCharuco

- `imageCopy` is the image where the corners will be drawn (it will normally be the same image where
   the corners were detected).
- The `outputImage` will be a clone of `inputImage` with the corners drawn.
- `charucoCorners` and `charucoIds` are the detected Charuco corners from the `cv::aruco::CharucoDetector::detectBoard()`
  function.
- Finally, the last parameter is the (optional) color we want to draw the corners with, of type `cv::Scalar`.

For this image:

![Image with Charuco board](images/choriginal.jpg)

The result will be:

![Charuco board detected](images/chcorners.jpg)

In the presence of occlusion. like in the following image, although some corners are clearly visible,
not all their surrounding markers have been detected due occlusion and, thus, they are not interpolated:

![Charuco detection with occlusion](images/chocclusion.jpg)

Sample video:

@youtube{Nj44m_N_9FY}

A full working example is included in the `detect_board_charuco.cpp` inside the
`samples/cpp/tutorial_code/objectDetection/`.

The samples `detect_board_charuco.cpp` now take input via commandline via the `cv::CommandLineParser`.
For this file the example parameters will look like:
@code{.cpp}
    -w=5 -h=7 -sl=0.04 -ml=0.02 -d=10 -v=/path_to_opencv/opencv/doc/tutorials/objdetect/charuco_detection/images/choriginal.jpg
@endcode

ChArUco Pose Estimation
-----------------------

The final goal of the ChArUco boards is finding corners very accurately for a high precision calibration
or pose estimation.

The aruco module provides a function to perform ChArUco pose estimation easily. As in the
`cv::aruco::GridBoard`, the coordinate system of the `cv::aruco::CharucoBoard` is placed in
the board plane with the Z axis pointing in, and centered in the bottom left corner of the board.

@note After OpenCV 4.6.0, there was an incompatible change in the coordinate systems of the boards,
now the coordinate systems are placed in the boards plane with the Z axis pointing in the plane
(previously the axis pointed out the plane).
`objPoints` in CW order correspond to the Z-axis pointing in the plane.
`objPoints` in CCW order correspond to the Z-axis pointing out the plane.
See PR https://github.com/opencv/opencv_contrib/pull/3174


To perform pose estimation for charuco boards, you should use `cv::aruco::CharucoBoard::matchImagePoints()`
and `cv::solvePnP()`:

@snippet samples/cpp/tutorial_code/objectDetection/detect_board_charuco.cpp poseCharuco

- The `charucoCorners` and `charucoIds` parameters are the detected charuco corners from the
  `cv::aruco::CharucoDetector::detectBoard()` function.
- The `cameraMatrix` and `distCoeffs` are the camera calibration parameters which are necessary
  for pose estimation.
- Finally, the `rvec` and `tvec` parameters are the output pose of the Charuco Board.
- `cv::solvePnP()` returns true if the pose was correctly estimated and false otherwise.
  The main reason of failing is that there are not enough corners for pose estimation or
  they are in the same line.

The axis can be drawn using `cv::drawFrameAxes()` to check the pose is correctly estimated.
The result would be: (X:red, Y:green, Z:blue)

![Charuco Board Axis](images/chaxis.jpg)

A full working example is included in the `detect_board_charuco.cpp` inside the
`samples/cpp/tutorial_code/objectDetection/`.

The samples `detect_board_charuco.cpp` now take input via commandline via the `cv::CommandLineParser`.
For this file the example parameters will look like:
@code{.cpp}
    -w=5 -h=7 -sl=0.04 -ml=0.02 -d=10
    -v=/path_to_opencv/opencv/doc/tutorials/objdetect/charuco_detection/images/choriginal.jpg
    -c=/path_to_opencv/opencv/samples/cpp/tutorial_code/objectDetection/tutorial_camera_charuco.yml
@endcode
