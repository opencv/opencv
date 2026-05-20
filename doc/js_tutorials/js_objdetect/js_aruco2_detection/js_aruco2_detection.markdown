Detection of ArUco2 Markers in JavaScript {#tutorial_js_aruco2_detection}
=========================================

@prev_tutorial{tutorial_py_aruco2_detection}
@next_tutorial{tutorial_barcode_detect_and_decode}

Goals
-----

In this tutorial you will learn:
- What ArUco2 markers are and why they are useful.
- How to generate ArUco2 markers with OpenCV.js.
- How to detect ArUco2 markers in an image using OpenCV.js.
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

In OpenCV.js, the `aruco2` functions are exposed as flat functions on the `cv` object and enums live under `cv.aruco2_DictionaryType`.

Prerequisites
-------------

Make sure OpenCV.js is loaded in your page or Node.js script:

@code{.js}
// Browser
let cv = await cvPromise;

// Node.js
const cv = await require('./opencv.js');
@endcode

@note Always delete OpenCV objects (`cv.Mat`, `cv.DictionaryTypeVector`, etc.) when you are done with them to avoid memory leaks.

Marker Creation
---------------

Before detection, you need to generate and print markers. Use `cv.aruco2_getFiducialMarkerImage()` for this.

@code{.js}
let markerImg = new cv.Mat();
let DICT = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;

// Generate marker ID 42 with 20x20 pixels per bit and a white border
cv.aruco2_getFiducialMarkerImage(markerImg, DICT, 42, 20, true);

// markerImg now contains the generated marker
console.log("Marker size:", markerImg.cols, "x", markerImg.rows);

markerImg.delete();
@endcode

The parameters are:
- The output image (`cv.Mat`).
- The dictionary type (e.g., `cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12`).
- The marker ID (must be valid for the chosen dictionary).
- The size of each bit in pixels (`bitSize`).
- Whether to add an external white border (`externalBorder`).

Marker Detection
----------------

Detection is done with a single call to `cv.aruco2_detectFiducialMarkers()`. It returns a vector-like object containing all detected markers.

@code{.js}
let img = new cv.Mat();
cv.aruco2_getFiducialMarkerImage(img, DICT, 42, 20, true);

let markers = cv.aruco2_detectFiducialMarkers(img, DICT);

console.log("Detected", markers.size(), "marker(s)");
for (let i = 0; i < markers.size(); i++) {
    let m = markers.get(i);
    console.log("  Marker ID:", m.id);
    console.log("  Corners:", m.corners.size());
}

img.delete();
markers.delete();
@endcode

Each marker object provides:
- `id`: the marker identifier.
- `corners`: the four corner points in the image (a `cv.Point2fVector` with `size()` = 4).
- `dictionary`: the dictionary the marker was found in.

Drawing Detected Markers
------------------------

To visualize the detection results, convert the grayscale image to BGR and call `cv.aruco2_drawFiducialMarkers()`. The image is modified in-place.

@code{.js}
let colorImg = new cv.Mat();
cv.cvtColor(img, colorImg, cv.COLOR_GRAY2BGR);

cv.aruco2_drawFiducialMarkers(colorImg, markers);
// colorImg now contains the drawn markers

colorImg.delete();
@endcode

Multi-Dictionary Detection
--------------------------

One of the most powerful features of `aruco2` is detecting markers from multiple dictionaries at once. In JavaScript, pass a `cv.DictionaryTypeVector` containing the dictionary constants.

@code{.js}
let DICT1 = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;
let DICT2 = cv.aruco2_DictionaryType.DICT_APRILTAG_36h11;

let dictionaries = new cv.DictionaryTypeVector();
dictionaries.push_back(DICT1);
dictionaries.push_back(DICT2);

let multiMarkers = cv.aruco2_detectFiducialMarkers1(img, dictionaries);

console.log("Found", multiMarkers.size(), "marker(s)");
for (let i = 0; i < multiMarkers.size(); i++) {
    let m = multiMarkers.get(i);
    let name = (m.dictionary === DICT1) ? "ArUco" : "AprilTag";
    console.log(name, "marker ID:", m.id);
}

dictionaries.delete();
multiMarkers.delete();
@endcode

Advanced: Detection Parameters
------------------------------

You can tune the detection process using `cv.aruco2_DetectionParameters`. Create an instance, adjust the fields, and pass it to `cv.aruco2_detectFiducialMarkers()`.

@code{.js}
let params = new cv.aruco2_DetectionParameters();
params.boxFilterSize = 15;
params.thres = 3;
params.errorCorrectionRate = 0.0;

let markers = cv.aruco2_detectFiducialMarkers(img, DICT, params);

params.delete();
markers.delete();
@endcode

Key parameters include:
- `boxFilterSize`: size of the adaptive thresholding kernel (must be odd).
- `thres`: threshold offset applied after box filtering.
- `errorCorrectionRate`: fraction of error-correction capacity to use (0 = no errors tolerated).
- `detectInvertedMarker`: set to `true` to detect white-on-black markers.

Grid Board Detection
--------------------

`aruco2` also supports grid boards for more robust pose estimation. Generate a board image with `cv.aruco2_getGridBoardImage()`, then detect it with `cv.aruco2_detectGridBoard()`.

@code{.js}
let boardImg = new cv.Mat();
let gridSize = new cv.Size(3, 2);
cv.aruco2_getGridBoardImage(boardImg, gridSize, DICT, 20);

let boardScene = new cv.Mat(boardImg.rows + 100, boardImg.cols + 100, cv.CV_8UC1);
boardScene.setTo(new cv.Scalar(255));
let roi = new cv.Rect(50, 50, boardImg.cols, boardImg.rows);
let sub = boardScene.roi(roi);
boardImg.copyTo(sub);
sub.delete();

let board = new cv.aruco2_GridBoard();
let found = cv.aruco2_detectGridBoard(boardScene, gridSize, DICT, board);

if (found) {
    console.log("Board detected with", board.markers.size(), "markers");
}

let colorScene = new cv.Mat();
cv.cvtColor(boardScene, colorScene, cv.COLOR_GRAY2BGR);
cv.aruco2_drawGridBoard(colorScene, board);

boardImg.delete(); boardScene.delete(); board.delete(); colorScene.delete();
@endcode

`cv.aruco2_detectGridBoard()` returns a boolean indicating whether any board marker was found. The `cv.aruco2_GridBoard` object is populated with the detected markers and can be accessed via `board.markers`.

For pose estimation, use `cv.aruco2_getSolvePnpPoints1(board, objPoints, imgPoints)` to obtain 3D-2D correspondences suitable for `cv.solvePnP()`.
