Detection of ArUco2 Markers {#tutorial_aruco2_detection}
===========================

@prev_tutorial{tutorial_table_of_content_aruco2}
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

- **Performance:** The detection engine is up to **6.5× more efficient** (based on ArUco Nano), and dictionary identification is up to **2.7× faster** thanks to O(1) hash-map lookups.
- **Simpler API:** Detection is now a single function call returning a `std::vector<FiducialMarker>`, where each marker contains its ID, corners, and dictionary info. No more managing parallel vectors!
- **Ease of Use:** A single public header `#include <opencv2/objdetect/aruco2.hpp>` provides everything you need.
- **Robustness:** Safer default parameters (e.g., `errorCorrectionRate=0`) help prevent false positives in cluttered scenes.
- **Advanced Features:** Native support for multi-dictionary detection in a single pass, and enhanced designs for boards, diamonds, and fractal markers.

The `aruco2` functionalities are included in:
@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_detection.cpp aruco2hdr


Markers and Dictionaries
------------------------

An ArUco marker is a synthetic square marker with a wide black border and an inner binary matrix that determines its identifier (ID). The black border allows for fast detection, while the binary codification enables identification and error correction.

Example of an ArUco2 marker:

![Example of ArUco2 marker](aruco2_marker.png)

It must be noted that a marker can be found rotated in the environment, however, the detection
process needs to be able to determine its original rotation, so that each corner is identified
unequivocally. This is also done based on the binary codification.

A dictionary is a set of markers used in an application. It defines the marker size (number of bits) and the number of markers it contains. `aruco2` includes many predefined dictionaries, such as `DICT_6X6_250`, `DICT_ARUCO_MIP_36h12`, and `DICT_APRILTAG_36h11`.

### Selecting a Dictionary

The choice of dictionary depends on the number of markers needed and the required detection robustness. Two main factors influence this:
- **Marker Size**: The number of bits in the matrix (e.g., 6x6, 7x7). **It is strongly recommended to use at least 6x6 bits** for reliable detection. Larger bit matrices (like 7x7) provide higher inter-marker distance and better error correction.
- **Dictionary Size**: The total number of unique markers in the set.

#### Inter-marker Distance

The most critical parameter of a dictionary is the **inter-marker distance** (minimum Hamming distance). It represents the minimum number of bit differences between any two markers in the dictionary.
- A higher inter-marker distance allows for better **error correction** and reduces **false positives** (misidentifying a marker or detecting one where there is none).
- For a fixed marker size, increasing the dictionary size decreases the inter-marker distance.
- For a fixed dictionary size, increasing the marker size (more bits) increases the inter-marker distance.

#### Recommendations

- **Use the smallest dictionary possible**: Always choose the smallest dictionary that fits your application's needs. For example, if you only need 250 markers, `DICT_6X6_250` is much more robust than `DICT_6X6_1000`.
- **Prefer `DICT_ARUCO_MIP_36h12`**: For most general applications requiring up to 250 markers, `DICT_ARUCO_MIP_36h12` is highly recommended. It uses a 6x6 grid and is specifically optimized for maximum inter-marker distance (distance = 12) @cite garrido2016generation .
- **AprilTag Dictionaries**: If you need more than 250 markers, the AprilTag dictionaries (`DICT_APRILTAG_36h11` or `DICT_APRILTAG_36h10`) are excellent alternatives @cite wang2016iros .


Marker Creation
---------------

Markers must be printed before they can be detected. You can generate marker images using the `cv::aruco2::getFiducialMarker()` function.

Example:
@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_detection.cpp generate_marker

The parameters are:
- The output image (`cv::Mat`).
- The dictionary type (e.g., `cv::aruco2::DICT_ARUCO_MIP_36h12`).
- The marker ID (must be valid for the chosen dictionary).
- Optional: `bitSize` (default 20), which is the size of each bit in pixels.
- Optional: `externalBorder` (default true), whether to add a white border.
...
Marker Detection
----------------

Detecting markers in `aruco2` is straightforward. The `cv::aruco2::detectFiducialMarkers()` function performs the entire pipeline: thresholding, contour tracing, quadrilateral fitting, bit extraction, and dictionary lookup.

### Basic Detection (Single Dictionary)

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_detection.cpp detect_single

### Multi-Dictionary Detection

One of the new features of `aruco2` is the ability to search for markers from multiple dictionaries in a single pass:

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_detection.cpp multi_dict


Drawing Detected Markers
------------------------

To visualize the detection results, use `cv::aruco2::drawFiducialMarkers()`. It draws a colored outline around each marker, a dot at the first corner to show orientation, and the marker ID.

@snippet samples/cpp/tutorial_code/objdetect/aruco2/aruco2_detection.cpp draw_markers

The default color is green (`cv::Scalar(0, 255, 0)`), but you can specify a custom color as the third parameter.
