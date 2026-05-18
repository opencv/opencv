Detection of ArUco2 Diamonds {#tutorial_aruco2_diamonds}
============================

@prev_tutorial{tutorial_aruco2_calibration}
@next_tutorial{tutorial_aruco2_fractals}

|    |    |
| -: | :- |
| Original author | Rafael Muñoz-Salinas |
| Compatibility    | OpenCV >= 5.0.0 |

An ArUco2 diamond marker is a 2×2 block of ArUco markers following the ChArUco2 design @cite MunozSalinas2026ChArUco2. It consists of standard markers on black squares and inverted markers on white squares.

Diamonds are conceptually different from boards. While a board's identity is fixed, a diamond's identity is the combination of the IDs of its four constituent markers. This allows for:
- **Increased ID Space:** Up to $N^4$ unique combinations, where $N$ is the dictionary size.
- **Conceptual Meaning:** Each of the four markers can represent different information (e.g., one marker ID could indicate the physical scale).
- **Ambiguity Resolution:** Multiple diamonds with the same or different IDs can be detected simultaneously without ambiguity.

![Example of ArUco2 Diamond](aruco2_diamond.png)

Diamond Creation
----------------

You can generate a diamond image for printing using `cv::aruco2::getDiamondImage()`.

@code{.cpp}
cv::Mat diamondImage;
cv::aruco2::DictionaryType dict = cv::aruco2::DICT_ARUCO_MIP_36h12;
cv::Vec4i ids(10, 11, 12, 13); // IDs clockwise from top-left
cv::aruco2::getDiamondImage(diamondImage, dict, ids);
cv::imwrite("diamond.png", diamondImage);
@endcode

The parameters are:
- The output image (`cv::Mat`).
- The dictionary type.
- The four marker IDs as a `cv::Vec4i` (top-left, top-right, bottom-right, bottom-left).
- Optional: `bitSize` (default 20), the size of each bit in pixels.

Diamond Detection
-----------------

Detection is handled by the `cv::aruco2::detectDiamonds()` function.

@code{.cpp}
cv::Mat image = cv::imread("diamond_scene.jpg");
auto diamonds = cv::aruco2::detectDiamonds(image, cv::aruco2::DICT_ARUCO_MIP_36h12);

for (const auto &d : diamonds) {
    std::cout << "Detected diamond with IDs: " << d.id << std::endl;
}
@endcode

Each `cv::aruco2::Diamond` object in the returned vector contains:
- `id`: A `cv::Vec4i` with the IDs of the four markers.
- `dict`: The dictionary used.
- `markers`: A `std::vector<FiducialMarker>` containing the four individual markers forming the diamond.

Drawing Detected Diamonds
-------------------------

Visualize the results using the `cv::aruco2::drawDiamonds()` function.

@code{.cpp}
cv::aruco2::drawDiamonds(image, diamonds);
cv::imshow("Detected Diamonds", image);
cv::waitKey(0);
@endcode

This function draws the diamond's outer boundary, small squares at each of the 9 grid corners, and the `Vec4i` ID at the centroid. You can also pass `true` as an optional fourth parameter to draw the individual IDs of the four constituent markers.

Pose Estimation with Diamonds
-----------------------------

Diamonds are excellent for high-precision pose estimation. To estimate the pose, use `cv::aruco2::getSolvePnpPoints()` to obtain the image-to-3D correspondences and then call `cv::solvePnP()`.

@code{.cpp}
float markerSize = 0.05f; // Physical side length of a single marker in meters
for (const auto &d : diamonds) {
    cv::Mat objPoints, imgPoints, rvec, tvec;
    cv::aruco2::getSolvePnpPoints(d, objPoints, imgPoints, markerSize);
    cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
    cv::aruco2::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, markerSize);
}
@endcode
