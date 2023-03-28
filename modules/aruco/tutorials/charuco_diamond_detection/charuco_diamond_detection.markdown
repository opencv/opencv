Detection of Diamond Markers {#tutorial_charuco_diamond_detection}
==============================

@prev_tutorial{tutorial_charuco_detection}
@next_tutorial{tutorial_aruco_calibration}

A ChArUco diamond marker (or simply diamond marker) is a chessboard composed by 3x3 squares and 4 ArUco markers inside the white squares.
It is similar to a ChArUco board in appearance, however they are conceptually different.

![Diamond marker examples](images/diamondmarkers.png)

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
to manually indicate the scale of each of them. This case is included in the ```diamond_detector.cpp``` file inside
the samples folder of the module.

Furthermore, as its corners are chessboard corners, they can be used for accurate pose estimation.

The diamond functionalities are included in ```<opencv2/aruco/charuco.hpp>```


ChArUco Diamond Creation
------

The image of a diamond marker can be easily created using the ```drawCharucoDiamond()``` function.
For instance:

@code{.cpp}
    cv::Mat diamondImage;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::drawCharucoDiamond(dictionary, cv::Vec4i(45,68,28,74), 200, 120, markerImage);
@endcode

This will create a diamond marker image with a square size of 200 pixels and a marker size of 120 pixels.
The marker ids are given in the second parameter as a ```Vec4i``` object. The order of the marker ids
in the diamond layout are the same as in a standard ChArUco board, i.e. top, left, right and bottom.

The image produced will be:

![Diamond marker](images/diamondmarker.png)

A full working example is included in the `create_diamond.cpp` inside the `modules/aruco/samples/`.

Note: The samples now take input via commandline via the [OpenCV Commandline Parser](http://docs.opencv.org/trunk/d0/d2e/classcv_1_1CommandLineParser.html#gsc.tab=0). For this file the example parameters will look like
@code{.cpp}
    "_path_/mydiamond.png" -sl=200 -ml=120 -d=10 -ids=45,68,28,74
@endcode

ChArUco Diamond Detection
------

As in most cases, the detection of diamond markers requires a previous detection of ArUco markers.
After detecting markers, diamond are detected using the ```detectCharucoDiamond()``` function:

@code{.cpp}
    cv::Mat inputImage;
    float squareLength = 0.40;
    float markerLength = 0.25;
    ...


    std::vector<int> markerIds;
    std::vector<std::vector< cv::Point2f>> markerCorners;

    // detect ArUco markers
    cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds);

    std::vector<cv::Vec4i> diamondIds;
    std::vector<std::vector<cv::Point2f>> diamondCorners;

    // detect diamon diamonds
    cv::aruco::detectCharucoDiamond(inputImage, markerCorners, markerIds, squareLength / markerLength, diamondCorners, diamondIds);
@endcode

The ```detectCharucoDiamond()``` function receives the original image and the previous detected marker corners and ids.
The input image is necessary to perform subpixel refinement in the ChArUco corners.
It also receives the rate between the square size and the marker sizes which is required for both, detecting the diamond
from the relative positions of the markers and interpolating the ChArUco corners.

The function returns the detected diamonds in two parameters. The first parameter, ```diamondCorners```, is an array containing
all the four corners of each detected diamond. Its format is similar to the detected corners by the ```detectMarkers()```
function and, for each diamond, the corners are represented in the same order than in the ArUco markers, i.e. clockwise order
starting with the top-left corner. The second returned parameter, ```diamondIds```, contains all the ids of the returned
diamond corners in ```diamondCorners```. Each id is actually an array of 4 integers that can be represented with ```Vec4i```.

The detected diamond can be visualized using the function ```drawDetectedDiamonds()``` which simply receives the image and the diamond
corners and ids:

@code{.cpp}
    ...
    std::vector<cv::Vec4i> diamondIds;
    std::vector<std::vector<cv::Point2f>> diamondCorners;
    cv::aruco::detectCharucoDiamond(inputImage, markerCorners, markerIds, squareLength / markerLength, diamondCorners, diamondIds);

    cv::aruco::drawDetectedDiamonds(inputImage, diamondCorners, diamondIds);
@endcode

The result is the same that the one produced by ```drawDetectedMarkers()```, but printing the four ids of the diamond:

![Detected diamond markers](images/detecteddiamonds.png)

A full working example is included in the `detect_diamonds.cpp` inside the `modules/aruco/samples/`.

Note: The samples now take input via commandline via the [OpenCV Commandline Parser](http://docs.opencv.org/trunk/d0/d2e/classcv_1_1CommandLineParser.html#gsc.tab=0). For this file the example parameters will look like
@code{.cpp}
    -dp="path_aruco/samples/detector_params.yml" -sl=0.04 -ml=0.012 -refine=3
    -v="path_aruco/tutorials/charuco_diamond_detection/images/diamondmarkers.png"
    -cd="path_aruco/samples/tutorial_dict.yml
@endcode

ChArUco Diamond Pose Estimation
------

Since a ChArUco diamond is represented by its four corners, its pose can be estimated in the same way than in a single ArUco marker,
i.e. using the ```estimatePoseSingleMarkers()``` function. For instance:

@code{.cpp}
    ...

    std::vector<cv::Vec4i> diamondIds;
    std::vector<std::vector<cv::Point2f>> diamondCorners;

    // detect diamon diamonds
    cv::aruco::detectCharucoDiamond(inputImage, markerCorners, markerIds, squareLength / markerLength, diamondCorners, diamondIds);

    // estimate poses
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, camMatrix, distCoeffs, rvecs, tvecs);

    // draw axis
    for(unsigned int i=0; i<rvecs.size(); i++)
        cv::drawFrameAxes(inputImage, camMatrix, distCoeffs, rvecs[i], tvecs[i], axisLength);
@endcode

The function will obtain the rotation and translation vector for each of the diamond marker and store them
in ```rvecs``` and ```tvecs```. Note that the diamond corners are a chessboard square corners and thus, the square length
has to be provided for pose estimation, and not the marker length. Camera calibration parameters are also required.

Finally, an axis can be drawn to check the estimated pose is correct using ```drawFrameAxes()```:

![Detected diamond axis](images/diamondsaxis.jpg)

The coordinate system of the diamond pose will be in the center of the marker with the Z axis pointing out,
as in a simple ArUco marker pose estimation.

Sample video:

@htmlonly
<iframe width="420" height="315" src="https://www.youtube.com/embed/OqKpBnglH7k" frameborder="0" allowfullscreen></iframe>
@endhtmlonly

A full working example is included in the `detect_diamonds.cpp` inside the `modules/aruco/samples/`.

Note: The samples now take input via commandline via the [OpenCV Commandline Parser](http://docs.opencv.org/trunk/d0/d2e/classcv_1_1CommandLineParser.html#gsc.tab=0). For this file the example parameters will look like
@code{.cpp}
    -dp="path_aruco/samples/detector_params.yml" -sl=0.04 -ml=0.012 -refine=3
    -v="path_aruco/tutorials/charuco_diamond_detection/images/diamondmarkers.png"
    -cd="path_aruco/samples/tutorial_dict.yml
    -c="path_aruco/samples/tutorial_camera_params.yml"
@endcode
