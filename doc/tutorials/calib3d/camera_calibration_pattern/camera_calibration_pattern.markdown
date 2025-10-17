Create Calibration Pattern {#tutorial_camera_calibration_pattern}
==========================

@tableofcontents

@next_tutorial{tutorial_camera_calibration_square_chess}

|    |    |
| -: | :- |
| Authors | Laurent Berger, Alexander Panov, Alexander Smorkalov |
| Compatibility   | OpenCV > 4.12  |


The tutorial describes all pattern supported by OpenCV for camera(s) calibration and pose estimation
with their strength, pitfalls and practical recommendations.

What is calibration pattern? why I need it?
-------------------------------------------

The flat printable pattern may be used:

1. For camera intrinsics (internal parameters) calibration. See @ref tutorial_camera_calibration.
2. For stereo or multi-camera system extrinsics (external parameters: rotation and translation
   of each camera) calibration. See cv::stereoCalibrate for details.
3. Camera pose registration relative to well known point in 3d world. See multiview calibration
   tutorial in OpenCV 5.x.

Pattern Types
-------------

**Chessboard**. Classic calibration pattern of black and white squares. The all calibration algorithms
use internal chessboard corners as features. See cv::findChessboardCorners and cv::cornerSubPix to
detect the board and refine corners coordinates with sub-pixel accuracy. The board size is defined
as amount of internal corners, but not amount of black or white squares. Also pay attention, that
the board with even size is symmetric. If board has even amount of corners by one of direction then
its pose is defined up to 180 degrees (2 solutions). It the board is square with size N x N then its
pose is defined up to 90 degrees (4 solutions). The last two cases are not suitable for calibration.
Example code to generate features coordinates for calibration (object points):
```
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints.push_back(Point3f(j*squareSize, i*squareSize, 0));
        }
    }
```
Printable chessboard pattern: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
(9x6 chessboard, page width: 210 mm, page height: 297 mm (A4))

**Circles Grid**. The circles grid is symmetric or asymmetric (each even row shifted) grid of black
circles on a white background or vice verse. See cv::findCirclesGrid function to detect the board
with OpenCV. The detector produces sub-pixel coordinates of the circle centers and does not require
additional refinement. The board size is defined as amount of circles in grid by x and y axis.
In case of asymmetric grid the shifted rows are taken into account too. The board is suitable for
intrinsics calibration. Symmetric grids suffer from the same issue as chessboard pattern with even
size. It's pose is defined up to 180 degrees.
Example code to generate features coordinates for calibration with symmetric grid (object points):
```
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints.push_back(Point3f(j*squareSize, i*squareSize, 0));
        }
    }
```
Example code to generate features corrdinates for calibration with asymmetic grid (object points):
```
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objectPoints.push_back(Point3f((2 * j + i % 2)*squareSize, i*squareSize, 0));
        }
    }
```
Printable asymmetric circles grid pattern: https://github.com/opencv/opencv/blob/4.x/doc/acircles_pattern.png
(11x4 asymmetric circles grid, page width: 210 mm, page height: 297 mm (A4))

**ChAruco board**. Chessboard unreached with ArUco markers. Each internal corner of the board is
described by 2 neighborhood ArUco markers that makes it unique. The board size is defined in number
of units, but not internal corners. ChAruco board of size N x M is equivalent to chessboard pattern
of size  N-1 x M-1. OpenCV provides `cv::aruco::CharucoDetector` class for the board detection.
The detector algorithm finds ArUco markers first and them "assembles" the board using knowledge
about ArUco pairs. In opposite to the previous pattern partially occluded board may be used as all
corners are labeled. The board is rotation invariant, but set of ArUco markers and their order
should be known to detector apriori. It cannot detect ChAruco board with predefined size and random
set of markers.
Example code to generate features corrdinates for calibration (object points) for board size in units:
```
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < boardSize.height-1; ++i) {
        for (int j = 0; j < boardSize.width-1; ++j) {
            objectPoints.push_back(Point3f(j*squareSize, i*squareSize, 0));
        }
    }
```
Printable ChAruco board pattern: https://github.com/opencv/opencv/blob/4.x/doc/charuco_board_pattern.png
(7X5 ChAruco board, square size: 30 mm, marker size: 15 mm, ArUco dict: DICT_5X5_100, page width:
210 mm, page height: 297 mm (A4))

Create Your Own Pattern
-----------------------

In case if ready pattern does not satisfy your requirements, you can generate your own. OpenCV
provides generate_pattern.py tool in `apps/pattern-tools` of source repository or your binary
distribution. The only requirement is Python 3.

Examples:

create a checkerboard pattern in file chessboard.svg with 9 rows, 6 columns and a square size of 20mm:

        python generate_pattern.py -o chessboard.svg --rows 9 --columns 6 --type checkerboard --square_size 20

create a circle board pattern in file circleboard.svg with 7 rows, 5 columns and a radius of 15 mm:

        python generate_pattern.py -o circleboard.svg --rows 7 --columns 5 --type circles --square_size 15

create a circle board pattern in file acircleboard.svg with 7 rows, 5 columns and a square size of
10mm and less spacing between circle:

        python generate_pattern.py -o acircleboard.svg --rows 7 --columns 5 --type acircles --square_size 10 --radius_rate 2

create a radon checkerboard for findChessboardCornersSB() with markers in (7 4), (7 5), (8 5) cells:

        python generate_pattern.py -o radon_checkerboard.svg --rows 10 --columns 15 --type radon_checkerboard -s 12.1 -m 7 4 7 5 8 5

create a ChAruco board pattern in charuco_board.svg with 7 rows, 5 columns, square size 30 mm, aruco
marker size 15 mm and using DICT_5X5_100 as dictionary for aruco markers (it contains in DICT_ARUCO.json file):

        python generate_pattern.py -o charuco_board.svg --rows 7 --columns 5 -T charuco_board --square_size 30 --marker_size 15 -f DICT_5X5_100.json.gz

If you want to change the measurement units, use the -u option (e.g. mm, inches, px, m)

If you want to change the page size, use the -w (width) and -h (height) options

If you want to use your own dictionary for the ChAruco board, specify the name of your dictionary
file. For example:

        python generate_pattern.py -o charuco_board.svg --rows 7 --columns 5 -T charuco_board -f my_dictionary.json

You can generate your dictionary in the file my_dictionary.json with 30 markers and a marker size of
5 bits using the utility provided in `samples/cpp/aruco_dict_utils.cpp`.

        bin/example_cpp_aruco_dict_utils.exe my_dict.json -nMarkers=30 -markerSize=5

Pattern Size
------------

Pattern is defined by it's physical board size, element (square or circle) physical size and amount
of elements. Factors that affect calibration quality:

- **Amount of features**. Most of OpenCV functions that work with detected patterns use optimization
or some random consensus strategies inside. More features on board means more points for optimization
and better estimation quality. Calibration process requires several images. It means that in most
of cases lower amount of pattern features may be compensated by higher amount frames.

- **Element size**. The physical size of elements depends on the distance and size in pixels.
Each detector defines some minimal size for reliable detection. For circles grid it's circle
radius, for chessboard it's square size, for ChAruco board it's ArUco marker element size.
General recommendation: larger elements (in frame pixels) reduces detection uncertainty.

- **Board size**. The board should be fully visible, sharp and reliably detected by OpenCV algorithms.
So, the board size should satisfy previous items, if it's used with typical target distance.
Usually larger board is better, but smaller boards allow to calibrate corners better.

Generic Recommendations
-----------------------

1. The final pattern should be as flat as possible. It improves calibration accuracy.
2. Glance pattern is worse than matte. Blinks and shadows on glance surface degrades board detection
significantly.
3. Most of detection algorithms expect white (black) border around the markers. Please do not cut
them or cover them.
