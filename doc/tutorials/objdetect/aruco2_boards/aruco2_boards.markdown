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

You can generate a board image ready for printing using `cv::aruco2::getGridBoard()`.

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_boards.cpp create_board

The parameters are:
- The output image (`cv::Mat`).
- The grid size as `cv::Size(columns, rows)`.
- The dictionary type.
- Optional: `bitSize` (default 25), the size of each bit in pixels.
- Optional: `ids`, a `std::vector<int>` of custom marker IDs. If empty, IDs 0 to (N-1) are used.

Board Detection
---------------

Detecting an ArUco2 board is handled by the `cv::aruco2::detectGridBoard()` function.

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_boards.cpp detect_board

The `cv::aruco2::GridBoard` structure populated by the function contains:
- `gridSize`: The dimensions of the board.
- `dict`: The dictionary used.
- `markers`: A `std::vector<FiducialMarker>` containing only the markers that were successfully detected in the current frame.

Drawing Detected Boards
-----------------------

To visualize the board detection, use the `cv::aruco2::drawGridBoard()` function. It draws a circle and the ID for each detected board corner.

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_boards.cpp draw_board

You can also pass `true` as an optional fourth parameter to draw the IDs of the individual markers on the board.


Camera Calibration with Boards
------------------------------

ArUco2 boards are particularly useful for camera calibration. Unlike traditional chessboards, they do not require the entire board to be visible to be useful.

Key advantages for calibration:
- **Partial Visibility:** You can calibrate your camera even if only a part of the board is in the frame. This is extremely valuable for capturing data at the edges and corners of the image sensor, where lens distortion is typically most significant.
- **Ambiguity Removal:** Traditional chessboards can suffer from "phase" ambiguity (the calibration algorithm might misidentify which corner is which if the board is symmetric). Since each square in an ArUco2 board contains a unique marker ID, the identity and position of every corner are always known unequivocally.
- **Robustness:** Calibration can proceed even in the presence of significant occlusions or shadows that would cause traditional chessboard detection to fail.
