Detection of ArUco2 Boards {#tutorial_aruco2_boards}
==========================

@prev_tutorial{tutorial_aruco2_detection}
@next_tutorial{tutorial_aruco2_calibration}

|    |    |
| -: | :- |
| Original author | Rafael Muñoz-Salinas |
| Compatibility    | OpenCV >= 5.0.0 |

An ArUco board is a set of markers that acts as a single target, providing a unified pose for the camera. Boards are more robust to occlusion than single markers because the pose can be estimated as long as at least one marker is visible.

ArUco2 introduces a new board design based on **ChArUco2** @cite MunozSalinas2026ChArUco2. In this design, markers are placed on every square of a grid, alternating between standard markers on black squares and inverted markers on white squares.

Benefits of ArUco2 Boards
-------------------------

- **Higher Density:** Compared to legacy boards where markers occupied only half the squares, ArUco2 boards double the marker density.
- **Occlusion Robustness:** With more markers in the same area, the board remains detectable even when significantly occluded.
- **Precision:** More markers mean more corner correspondences, leading to more accurate pose estimation.

<img src="aruco2_board.png" alt="Example of ArUco2 Board" width="50%"/>


Board Creation
--------------

You can generate a board image ready for printing using `cv::aruco2::generateBoardImage()`.

@code{.cpp}
//Generate a 9x5 board using DICT_ARUCO_MIP_36h12 markers
cv::Mat boardImage;
cv::aruco2::generateBoardImage(boardImage, cv::Size(9, 5), cv::aruco2::DICT_ARUCO_MIP_36h12);
cv::imwrite("board.png", boardImage);
@endcode

The parameters are:
- The output image (`cv::Mat`).
- The grid size as `cv::Size(columns, rows)`.
- The dictionary type.
- Optional: `bitSize` (default 25), the size of each bit in pixels.
- Optional: `ids`, a `std::vector<int>` of custom marker IDs. If empty, IDs 0 to (N-1) are used.

Board Detection
---------------

Detecting an ArUco2 board is handled by the `cv::aruco2::detectBoard()` function.

@code{.cpp}
cv::Mat image = cv::imread("board_scene.jpg");
cv::aruco2::GridBoard board;
bool found = cv::aruco2::detectBoard(image, cv::Size(4, 3), cv::aruco2::DICT_ARUCO_MIP_36h12, board);

if (found) {
    std::cout << "Detected " << board.markers.size() << " markers on the board." << std::endl;
}
@endcode

The `cv::aruco2::GridBoard` structure populated by the function contains:
- `gridSize`: The dimensions of the board.
- `dict`: The dictionary used.
- `markers`: A `std::vector<Marker>` containing only the markers that were successfully detected in the current frame.

Drawing Detected Boards
-----------------------

To visualize the board detection, use the overloaded `cv::aruco2::drawDetected()` function for boards. It draws a circle and the ID for each detected board corner.

@code{.cpp}
cv::aruco2::drawDetected(image, board);
cv::imshow("Detected Board", image);
cv::waitKey(0);
@endcode

You can also pass `true` as an optional fourth parameter to draw the IDs of the individual markers on the board.


Camera Calibration with Boards
------------------------------

ArUco2 boards are particularly useful for camera calibration. Unlike traditional chessboards, they do not require the entire board to be visible to be useful.

Key advantages for calibration:
- **Partial Visibility:** You can calibrate your camera even if only a part of the board is in the frame. This is extremely valuable for capturing data at the edges and corners of the image sensor, where lens distortion is typically most significant.
- **Ambiguity Removal:** Traditional chessboards can suffer from "phase" ambiguity (the calibration algorithm might misidentify which corner is which if the board is symmetric). Since each square in an ArUco2 board contains a unique marker ID, the identity and position of every corner are always known unequivocally.
- **Robustness:** Calibration can proceed even in the presence of significant occlusions or shadows that would cause traditional chessboard detection to fail.
