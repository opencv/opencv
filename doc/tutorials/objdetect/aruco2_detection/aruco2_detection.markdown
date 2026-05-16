Detection of ArUco2 Markers {#tutorial_aruco2_detection}
===========================

@next_tutorial{tutorial_aruco2_boards}

|    |    |
| -: | :- |
| Original author | Rafael Muñoz-Salinas |
| Compatibility    | OpenCV >= 5.0.0 |

Pose estimation is fundamental in many computer vision applications, such as robot navigation and augmented reality. This process involves finding correspondences between real-world 3D points and their 2D image projections. Using synthetic or fiducial markers significantly simplifies this task.

The `aruco2` module is a proposed replacement for the legacy `aruco` module in OpenCV 5, designed by the original ArUco author to be faster, simpler, and more robust.

The implementation is based on the ArUco Library @cite Aruco2014 @cite romero2018speeded @cite GARRIDOJURADO2026102690.

Benefits of ArUco2
------------------

Compared to the legacy `aruco` module, `aruco2` offers several significant improvements:

- **Performance:** The detection engine is up to **6.5× faster** (based on ArUco Nano), and dictionary identification is up to **2.7× faster** thanks to O(1) hash-map lookups.
- **Simpler API:** Detection is now a single function call returning a `std::vector<Marker>`, where each marker contains its ID, corners, and dictionary info. No more managing parallel vectors!
- **Ease of Use:** A single public header `#include <opencv2/objdetect/aruco2.hpp>` provides everything you need.
- **Robustness:** Safer default parameters (e.g., `errorCorrectionRate=0`) help prevent false positives in cluttered scenes.
- **Advanced Features:** Native support for multi-dictionary detection in a single pass, and enhanced designs for boards, diamonds, and fractal markers.

The `aruco2` functionalities are included in:
@code{.cpp}
#include <opencv2/objdetect/aruco2.hpp>
@endcode


Markers and Dictionaries
------------------------

An ArUco marker is a synthetic square marker with a wide black border and an inner binary matrix that determines its identifier (ID). The black border allows for fast detection, while the binary codification enables identification and error correction.

Example of an ArUco2 marker:

![Example of ArUco2 marker](aruco2_marker.png)

It must be noted that a marker can be found rotated in the environment, however, the detection
process needs to be able to determine its original rotation, so that each corner is identified
unequivocally. This is also done based on the binary codification.

A dictionary is a set of markers used in an application. It defines the marker size (number of bits) and the number of markers it contains. `aruco2` includes many predefined dictionaries, such as `DICT_4X4_50`, `DICT_ARUCO_MIP_36h12`, and `DICT_APRILTAG_36h11`.

### Selecting a Dictionary

For most applications, it is **highly recommended** to use `DICT_ARUCO_MIP_36h12`. This dictionary contains 250 markers and offers the highest intermarker distance @cite garrido2016generation, which significantly reduces the probability of false positives and improves detection robustness.

If your application requires more than 250 markers, consider using the AprilTag dictionaries @cite wang2016iros :
- `DICT_APRILTAG_36h11`: 587 markers.
- `DICT_APRILTAG_36h10`: 2320 markers.

More information on selecting a dictionary can be found in the "Selecting a dictionary" section of the legacy documentation.


Marker Creation
---------------

Markers must be printed before they can be detected. You can generate marker images using the `cv::aruco2::generateMarkerImage()` function.

Example:
@code{.cpp}
cv::Mat markerImage;
cv::aruco2::generateMarkerImage(markerImage, cv::aruco2::DICT_ARUCO_MIP_36h12, 42);
cv::imwrite("marker42.png", markerImage);
@endcode

The parameters are:
- The output image (`cv::Mat`).
- The dictionary type (e.g., `cv::aruco2::DICT_ARUCO_MIP_36h12`).
- The marker ID (must be valid for the chosen dictionary).
- Optional: `bitSize` (default 20), which is the size of each bit in pixels.
- Optional: `externalBorder` (default true), which adds a white border around the marker.


Marker Detection
----------------

Detecting markers in `aruco2` is straightforward. The `cv::aruco2::detectMarkers()` function performs the entire pipeline: thresholding, contour tracing, quadrilateral fitting, bit extraction, and dictionary lookup.

### Basic Detection (Single Dictionary)

@code{.cpp}
cv::Mat image = cv::imread("scene.jpg");
auto markers = cv::aruco2::detectMarkers(image, cv::aruco2::DICT_ARUCO_MIP_36h12);

for (const auto &m : markers) {
    std::cout << "Detected marker ID: " << m.id << " at " << m.corners[0] << std::endl;
}
@endcode

### Multi-Dictionary Detection

One of the new features of `aruco2` is the ability to search for markers from multiple dictionaries in a single pass:

@code{.cpp}
using namespace cv::aruco2;
auto markers = detectMarkers(image, {DICT_ARUCO_MIP_36h12, DICT_APRILTAG_36h11});

for (const auto &m : markers) {
    std::string dictName = (m.dict == DICT_ARUCO_MIP_36h12) ? "ArUco" : "AprilTag";
    std::cout << "Found " << dictName << " marker ID: " << m.id << std::endl;
}
@endcode


Drawing Detected Markers
------------------------

To visualize the detection results, use `cv::aruco2::drawDetected()`. It draws a colored outline around each marker, a dot at the first corner to show orientation, and the marker ID.

@code{.cpp}
cv::aruco2::drawDetected(image, markers);
cv::imshow("Detected Markers", image);
cv::waitKey(0);
@endcode

The default color is green (`cv::Scalar(0, 255, 0)`), but you can specify a custom color as the third parameter.
