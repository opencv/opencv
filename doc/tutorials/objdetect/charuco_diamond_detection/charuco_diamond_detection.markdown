Detection of Diamond Markers {#tutorial_charuco_diamond_detection}
==============================

@prev_tutorial{tutorial_charuco_detection}
@next_tutorial{tutorial_aruco_calibration}

A ChArUco diamond marker (or simply diamond marker) is a chessboard composed by 3x3 squares and 4 ArUco markers inside the white squares.
It is similar to a ChArUco board in appearance, however they are conceptually different.

![Diamond marker examples](images/diamondmarkers.jpg)

In both, ChArUco board and Diamond markers, their detection is based on the previous detected ArUco
markers. In the ChArUco case, the used markers are selected by directly looking their identifiers. This means
that if a marker (included in the board) is found on a image, it will be automatically assumed to belong to the board. Furthermore,
if a marker board is found more than once in the image, it will produce an ambiguity since the system wont
be able to know which one should be used for the Board.

On the other hand, the detection of Diamond marker is not based on the identifiers. Instead, their detection
is based on the relative position of the markers. As a consequence, marker identifiers can be repeated in the
same diamond or among different diamonds, and they can be detected simultaneously without ambiguity. However,
due to the complexity of finding marker based on their relative position, the diamond markers are limited to
a size of 3x3 squares and 4 markers.

As in a single ArUco marker, each Diamond marker is composed by 4 corners and a identifier. The four corners
correspond to the 4 chessboard corners in the marker and the identifier is actually an array of 4 numbers, which are
the identifiers of the four ArUco markers inside the diamond.

Diamond markers are useful in those scenarios where repeated markers should be allowed. For instance:

- To increase the number of identifiers of single markers by using diamond marker for labeling. They would allow
up to N^4 different ids, being N the number of markers in the used dictionary.

- Give to each of the four markers a conceptual meaning. For instance, one of the four marker ids could be
used to indicate the scale of the marker (i.e. the size of the square), so that the same diamond can be found
in the environment with different sizes just by changing one of the four markers and the user does not need
to manually indicate the scale of each of them. This case is included in the `detect_diamonds.cpp` file inside
the samples folder of the module.

Furthermore, as its corners are chessboard corners, they can be used for accurate pose estimation.

The diamond functionalities are included in `<opencv2/objdetect/charuco_detector.hpp>`


ChArUco Diamond Creation
------

The image of a diamond marker can be easily created using the `cv::aruco::CharucoBoard::generateImage()` function.
For instance:

@snippet samples/cpp/tutorial_code/objectDetection/create_diamond.cpp generate_diamond

This will create a diamond marker image with a square size of 200 pixels and a marker size of 120 pixels.
The marker ids are given in the second parameter as a `cv::Vec4i` object. The order of the marker ids
in the diamond layout are the same as in a standard ChArUco board, i.e. top, left, right and bottom.

The image produced will be:

![Diamond marker](images/diamondmarker.png)

A full working example is included in the `create_diamond.cpp` inside the `samples/cpp/tutorial_code/objectDetection/`.

The samples `create_diamond.cpp` now take input via commandline via the `cv::CommandLineParser`. For this file the example
parameters will look like:
@code{.cpp}
    "_path_/mydiamond.png" -sl=200 -ml=120 -d=10 -ids=0,1,2,3
@endcode

ChArUco Diamond Detection
------

As in most cases, the detection of diamond markers requires a previous detection of ArUco markers.
After detecting markers, diamond are detected using the `cv::aruco::CharucoDetector::detectDiamonds()` function:

@snippet samples/cpp/tutorial_code/objectDetection/detect_diamonds.cpp detect_diamonds

The `cv::aruco::CharucoDetector::detectDiamonds()` function receives the original image and the previous detected marker corners and ids.
If markerCorners and markerIds are empty, the function will detect aruco markers and ids.
The input image is necessary to perform subpixel refinement in the ChArUco corners.
It also receives the rate between the square size and the marker sizes which is required for both, detecting the diamond
from the relative positions of the markers and interpolating the ChArUco corners.

The function returns the detected diamonds in two parameters. The first parameter, `diamondCorners`, is an array containing
all the four corners of each detected diamond. Its format is similar to the detected corners by the `cv::aruco::ArucoDetector::detectMarkers()`
function and, for each diamond, the corners are represented in the same order than in the ArUco markers, i.e. clockwise order
starting with the top-left corner. The second returned parameter, `diamondIds`, contains all the ids of the returned
diamond corners in `diamondCorners`. Each id is actually an array of 4 integers that can be represented with `cv::Vec4i`.

The detected diamond can be visualized using the function `cv::aruco::drawDetectedDiamonds()` which simply receives the image and the diamond
corners and ids:

@snippet samples/cpp/tutorial_code/objectDetection/detect_diamonds.cpp draw_diamonds

The result is the same that the one produced by `cv::aruco::drawDetectedMarkers()`, but printing the four ids of the diamond:

![Detected diamond markers](images/detecteddiamonds.jpg)

A full working example is included in the `detect_diamonds.cpp` inside the `samples/cpp/tutorial_code/objectDetection/`.

The samples `detect_diamonds.cpp` now take input via commandline via the `cv::CommandLineParser`. For this file the example
parameters will look like:
@code{.cpp}
    -dp=path_to_opencv/opencv/samples/cpp/tutorial_code/objectDetection/detector_params.yml -sl=0.4 -ml=0.25 -refine=3
    -v=path_to_opencv/opencv/doc/tutorials/objdetect/charuco_diamond_detection/images/diamondmarkers.jpg
    -cd=path_to_opencv/opencv/samples/cpp/tutorial_code/objectDetection/tutorial_dict.yml
@endcode

ChArUco Diamond Pose Estimation
------

Since a ChArUco diamond is represented by its four corners, its pose can be estimated in the same way than in a single ArUco marker,
i.e. using the `cv::solvePnP()` function. For instance:

@snippet samples/cpp/tutorial_code/objectDetection/detect_diamonds.cpp diamond_pose_estimation
@snippet samples/cpp/tutorial_code/objectDetection/detect_diamonds.cpp draw_diamond_pose_estimation

The function will obtain the rotation and translation vector for each of the diamond marker and store them
in `rvecs` and `tvecs`. Note that the diamond corners are a chessboard square corners and thus, the square length
has to be provided for pose estimation, and not the marker length. Camera calibration parameters are also required.

Finally, an axis can be drawn to check the estimated pose is correct using `drawFrameAxes()`:

![Detected diamond axis](images/diamondsaxis.jpg)

The coordinate system of the diamond pose will be in the center of the marker with the Z axis pointing out,
as in a simple ArUco marker pose estimation.

Sample video:

@youtube{OqKpBnglH7k}

Also ChArUco diamond pose can be estimated as ChArUco board:
@snippet samples/cpp/tutorial_code/objectDetection/detect_diamonds.cpp diamond_pose_estimation_as_charuco

A full working example is included in the `detect_diamonds.cpp` inside the `samples/cpp/tutorial_code/objectDetection/`.

The samples `detect_diamonds.cpp` now take input via commandline via the `cv::CommandLineParser`. For this file the example
parameters will look like:
@code{.cpp}
    -dp=path_to_opencv/opencv/samples/cpp/tutorial_code/objectDetection/detector_params.yml -sl=0.4 -ml=0.25 -refine=3
    -v=path_to_opencv/opencv/doc/tutorials/objdetect/charuco_diamond_detection/images/diamondmarkers.jpg
    -cd=path_to_opencv/opencv/samples/cpp/tutorial_code/objectDetection/tutorial_dict.yml
    -c=path_to_opencv/opencv/samples/cpp/tutorial_code/objectDetection/tutorial_camera_params.yml
@endcode
