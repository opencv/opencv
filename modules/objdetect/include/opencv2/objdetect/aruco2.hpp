// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#pragma once
#include <opencv2/core.hpp>

namespace cv {
namespace aruco2 {

//! @addtogroup objdetect_aruco2
//! @{

/** @brief Predefined markers dictionaries/sets
 *
 * Each dictionary indicates the number of bits and the number of markers contained
 * - DICT_ARUCO_ORIGINAL: standard ArUco Library Markers. 1024 markers, 5x5 bits, 0 minimum
                          distance
 */
enum DictionaryType {
    DICT_4X4_50 = 0,        ///< 4x4 bits, minimum hamming distance between any two codes = 4, 50 codes
    DICT_4X4_100,           ///< 4x4 bits, minimum hamming distance between any two codes = 3, 100 codes
    DICT_4X4_250,           ///< 4x4 bits, minimum hamming distance between any two codes = 3, 250 codes
    DICT_4X4_1000,          ///< 4x4 bits, minimum hamming distance between any two codes = 2, 1000 codes
    DICT_5X5_50,            ///< 5x5 bits, minimum hamming distance between any two codes = 8, 50 codes
    DICT_5X5_100,           ///< 5x5 bits, minimum hamming distance between any two codes = 7, 100 codes
    DICT_5X5_250,           ///< 5x5 bits, minimum hamming distance between any two codes = 6, 250 codes
    DICT_5X5_1000,          ///< 5x5 bits, minimum hamming distance between any two codes = 5, 1000 codes
    DICT_6X6_50,            ///< 6x6 bits, minimum hamming distance between any two codes = 13, 50 codes
    DICT_6X6_100,           ///< 6x6 bits, minimum hamming distance between any two codes = 12, 100 codes
    DICT_6X6_250,           ///< 6x6 bits, minimum hamming distance between any two codes = 11, 250 codes
    DICT_6X6_1000,          ///< 6x6 bits, minimum hamming distance between any two codes = 9, 1000 codes
    DICT_7X7_50,            ///< 7x7 bits, minimum hamming distance between any two codes = 19, 50 codes
    DICT_7X7_100,           ///< 7x7 bits, minimum hamming distance between any two codes = 18, 100 codes
    DICT_7X7_250,           ///< 7x7 bits, minimum hamming distance between any two codes = 17, 250 codes
    DICT_7X7_1000,          ///< 7x7 bits, minimum hamming distance between any two codes = 14, 1000 codes
    DICT_ARUCO_ORIGINAL,    ///< 6x6 bits, minimum hamming distance between any two codes = 3, 1024 codes
    /** @brief AprilTag dictionaries. See @cite wang2016iros for details. */
    DICT_APRILTAG_16h5,     ///< 4x4 bits, minimum hamming distance between any two codes = 5, 30 codes
    DICT_APRILTAG_25h9,     ///< 5x5 bits, minimum hamming distance between any two codes = 9, 35 codes
    DICT_APRILTAG_36h10,    ///< 6x6 bits, minimum hamming distance between any two codes = 10, 2320 codes
    DICT_APRILTAG_36h11,     ///< 6x6 bits, minimum hamming distance between any two codes = 11, 587 codes
    /** @brief 6x6 bits, minimum hamming distance between any two codes = 12, 250 codes.
     * See @cite garrido2016generation for details.
     */
    DICT_ARUCO_MIP_36h12
};

/** @brief Detection parameters for detectMarkers() and detectGridBoard().
 *
 * All parameters have defaults that work well for standard printed markers under normal lighting.
 * Tune only when detection fails or produces false positives in your specific setup.
 *
 * The implementation is based on the ArUco Library @cite Aruco2014 @cite romero2018speeded @cite GARRIDOJURADO2026102690.
 */
struct CV_EXPORTS_W_SIMPLE DetectionParameters {

    /** @brief Size of the box filter kernel used for adaptive thresholding (pixels, must be odd).
     *
     * Larger values tolerate more uneven lighting but may merge nearby markers.
     * Default: 15.
     */
    CV_PROP_RW int boxFilterSize = 15;

    /** @brief Threshold offset applied after the box filter subtraction.
     *
     * A pixel is considered foreground if `boxFilter(p) - p > thres`.
     * Increase to suppress noise; decrease to detect faint borders.
     * Default: 3.
     */
    CV_PROP_RW int thres = 3;

    /** @brief Minimum side length (pixels) for a contour to be considered a marker candidate.
     *
     * Contour sides shorter than this are discarded early.
     * Default: 10.
     */
    CV_PROP_RW int minSize = 10;

    /** @brief Number of attempts to identify a candidate by slightly perturbing its corners.
     *
     * On each attempt after the first, Gaussian noise (σ=0.75 px) is added to the corners
     * before re-sampling the bits.  Improves robustness near perspective extremes.
     * Default: 5.
     */
    CV_PROP_RW int maxAttemptsPerCandidate = 5;

    /** @brief Controls how aggressively the contour tracer prunes revisited paths.
     *
     * Expressed as a fraction of the total contour length [0, 1].  A contour is discarded
     * if the number of already-visited pixels exceeds `maxTimesRevisited * contourLength`.
     * - 0.05 (default): filters most noise and thin structures efficiently.
     * - 1.0 : behaves like a standard Moore neighbour tracer (no pruning).
     *
     * Lower values speed up detection and reduce false candidates at the cost of occasionally
     * missing very distorted markers.
     */
    CV_PROP_RW float maxTimesRevisited = 0.05f;

    /** @brief Width of the mandatory black border around each marker, in bits.
     *
     * Almost all standard dictionaries use 1 border bit.  Kept for compatibility with
     * custom dictionaries that deviate from this convention.
     * Default: 1.
     */
    CV_PROP_RW int markerBorderBits = 1;

    /** @brief Fraction of `maxCorrectionBits` to use when matching a candidate against the dictionary.
     *
     * A candidate is accepted if its Hamming distance to the nearest dictionary entry is at most
     * `floor(maxCorrectionBits * errorCorrectionRate)`.
     * - 0 (default): no bit errors tolerated — lowest false-positive rate.
     * - 1.0: use the full error-correction capacity of the dictionary.
     *
     * @warning The legacy OpenCV ArUco default of 0.6 produces many false positives in cluttered
     * scenes.  Raise this only if you need tolerance against printing or lighting artefacts and
     * accept the trade-off.
     */
    CV_PROP_RW double errorCorrectionRate = 0;

    /** @brief Maximum fraction of border bits allowed to be wrong before rejecting a candidate.
     *
     * Set to 0 (default) to require a perfect black border.  Small non-zero values (e.g. 0.05)
     * add tolerance for border damage or ink bleed.
     */
    CV_PROP_RW double maxErroneousBitsInBorderRate = 0;

    /** @brief Set to true to detect markers printed white-on-black (inverted polarity).
     *
     * Default: false (standard black-on-white markers).
     */
    CV_PROP_RW bool detectInvertedMarker = false;
};


/** @brief A single detected ArUco marker.
 *
 * `corners` holds the four image-plane corner points in clockwise order starting from the
 * top-left corner.  `id` is the marker identifier within its `dict` dictionary.
 *
 * Corner order (viewed from front, standard orientation):
 * @code
 *   corners[0] ---- corners[1]
 *       |                |
 *   corners[3] ---- corners[2]
 * @endcode
 *
 * @sa detectMarkers, drawDetected, getSolvePnpPoints
 */
struct CV_EXPORTS_W_SIMPLE Marker {
    CV_PROP_RW std::vector<cv::Point2f> corners; ///< four corner points in clockwise order
    CV_PROP_RW int id = -1;                      ///< marker id; -1 if unidentified
    CV_PROP_RW cv::aruco2::DictionaryType dict = cv::aruco2::DictionaryType(-1); ///< dictionary this marker belongs to
    CV_WRAP Point2f getCorner(int i) const { return corners[i]; }
    cv::Point2f operator[](size_t i) const { return corners[i]; }
    size_t size() const { return corners.size(); }
};


/** @brief Generate a canonical marker image ready for printing.
 *
 * @param img         output grayscale image (CV_8UC1)
 * @param dictionary  predefined dictionary the marker belongs to
 * @param id          marker identifier; must be a valid index in the chosen dictionary
 * @param bitSize  output size in pixels of each marker bit
 * @param externalBorder indicates whether to add a white border around the marker
 * @code
 * cv::Mat markerImg;
 * cv::aruco2::generateMarkerImage(markerImg,DICT_ARUCO_MIP_36h12, 42);
 * cv::imwrite("marker_42.png", markerImg);
 * @endcode
 */
CV_EXPORTS_W void generateMarkerImage(OutputArray img, cv::aruco2::DictionaryType dictionary, int id, int bitSize=20,bool externalBorder=true);


/** @brief Detect ArUco markers in an image using a single dictionary.
 *
 * @param image        input image (grayscale or BGR)
 * @param dict         dictionary to search; default is DICT_ARUCO_MIP_36h12
 * @param detectorParams  detection tuning parameters
 * @return             vector of detected Marker objects; empty if none found
 *
 * Performs the full detection pipeline: adaptive thresholding → contour tracing →
 * quadrilateral fitting → bit extraction → dictionary lookup → subpixel corner refinement.
 *
 * @note Lens distortion is not corrected internally.  For accurate pose estimation,
 * undistort the image first with the known camera model.
 * @sa undistort, detectMarkers(InputArray, const std::vector<DictionaryType>&, const DetectionParameters&)
 */
CV_EXPORTS_W std::vector<cv::aruco2::Marker> detectMarkers(InputArray image, cv::aruco2::DictionaryType dict = cv::aruco2::DICT_ARUCO_MIP_36h12,
                                          const cv::aruco2::DetectionParameters &detectorParams = {});

/** @brief Detect ArUco markers in an image searching across multiple dictionaries in one pass.
 *
 * @param image        input image (grayscale or BGR)
 * @param dicts        list of dictionaries to search simultaneously
 * @param detectorParams  detection tuning parameters
 * @return             vector of detected Marker objects; each carries the dictionary it was found in
 *
 * Each marker candidate is tested against all dictionaries in `dicts`.  Once identified in one
 * dictionary it is removed from the candidate pool, so the same region is never matched twice.
 *
 * The implementation is based on the ArUco Library @cite Aruco2014 @cite romero2018speeded @cite GARRIDOJURADO2026102690.
 *
 * @sa Marker::dict
 */
CV_EXPORTS_W std::vector<cv::aruco2::Marker> detectMarkers(InputArray image, const std::vector<cv::aruco2::DictionaryType> &dicts,
                                          const cv::aruco2::DetectionParameters &detectorParams = {});


/** @brief Draw detected markers onto an image.
 *
 * @param image        input/output image (1 or 3 channels); modified in place
 * @param markers      markers returned by detectMarkers()
 * @param borderColor  color used to draw the marker outline (default: green)
 *
 * For each marker the function draws:
 * - a quadrilateral outline in `borderColor`
 * - a filled circle on corner[0] in a contrasting color to indicate orientation
 * - the marker id as text at the marker centroid
 *
 * Useful for visualisation and debugging.
 */
CV_EXPORTS_AS(drawDetectedMarkers) CV_EXPORTS_W void drawDetected(InputOutputArray image, const std::vector<cv::aruco2::Marker> &markers,
                                 Scalar borderColor = Scalar(0, 255, 0));


/** @brief Draw the XYZ coordinate frame of a pose estimate onto an image.
 *
 * Projects four points (origin + axis tips) with cv::projectPoints and draws three coloured
 * segments from the marker origin: X red, Y green, Z blue.
 *
 * @param image         input/output BGR image; modified in place
 * @param cameraMatrix  3×3 camera intrinsic matrix (from calibrateCamera)
 * @param distCoeffs    distortion coefficients (from calibrateCamera)
 * @param rvec          rotation vector (from solvePnP)
 * @param tvec          translation vector (from solvePnP)
 * @param length        axis length in the same unit as @p tvec (e.g. metres)
 */
CV_EXPORTS_W void drawAxis(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                      InputArray rvec, InputArray tvec, float length);


/** @brief Compute object and image points for a single marker to pass to solvePnP().
 *
 * @param marker     a detected marker
 * @param objPoints  output 4×1 array of the corresponding 3-D object points in marker
 *                   coordinates (CV_32FC3), with the marker centre at the origin and
 *                   half-unit side length (i.e. corners at ±0.5 in X and Y, Z=0)
 * @param imgPoints  output 4×1 array of the marker's corner pixel coordinates (CV_32FC2)
 *
 * @param markerSize  physical side length of the marker (e.g. metres); objPoints are scaled
 *                    by this value.  Default 1.f returns unit-size points.
 *
 * @code
 * cv::Mat rvec, tvec, imgPts, objPts;
 * cv::aruco2::getSolvePnpPoints(marker, objPts, imgPts, 0.05f); // 5 cm marker
 * cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
 * @endcode
 */
CV_EXPORTS_W void getSolvePnpPoints(const cv::aruco2::Marker &marker, OutputArray objPoints, OutputArray imgPoints, float markerSize = 1.f);


/** @brief Result of detecting a ChArUco2-style grid board.
 *
 * Follows the ChArUco2 design @cite MunozSalinas2026ChArUco2 : every square carries an ArUco marker (standard markers on black
 * squares, inverted markers on white squares), yielding N×M markers on an N×M board and
 * (N+1)×(M+1) observable intersection corners including the board border.
 *
 * `markers` holds the detected ArUco markers (a subset of all board markers when the board is
 * partially occluded).  Use getSolvePnpPoints() to obtain the corresponding object and image
 * point arrays for solvePnP().
 *
 * @sa detectGridBoard, getSolvePnpPoints(const GridBoard, OutputArray, OutputArray)
 */
struct CV_EXPORTS_W_SIMPLE GridBoard {
    CV_PROP_RW cv::Size gridSize;              ///< board dimensions: width × height in markers
    CV_PROP_RW cv::aruco2::DictionaryType dict;            ///< dictionary used for all markers on the board
    CV_PROP_RW std::vector<cv::aruco2::Marker> markers;    ///< detected markers (subset of the full board)
private:
    std::vector<std::pair<int,cv::Point2f>> detectedBoardCorners;
    friend bool detectGridBoard(InputArray image, cv::Size gridSize, cv::aruco2::DictionaryType dict, GridBoard &board, InputArray ids);
    friend void getSolvePnpPoints(const GridBoard& board, OutputArray objPoints, OutputArray imgPoints, float markerSize);
    friend void drawDetected(InputOutputArray image, const GridBoard &board, Scalar color, bool drawMarkerIds);
};


/** @brief Generate a grid board image ready for printing.
 *
 * @param img          output grayscale image (CV_8UC1) containing the full board
 * @param boardSize    board layout as columns × rows (e.g. `cv::Size(4, 3)`)
 * @param dict         dictionary used for the markers
 * @param bitSize     size of each marker bit in pixels (default 25)
 * @param ids          optional custom marker id list in row-major order;
 *                     if empty, ids 0…(cols*rows−1) are used
 *
 * Markers are laid out in row-major order with no gap between them.
 * Pass the same `boardSize`, `dict`, and `ids` to detectGridBoard() for detection.
 */
CV_EXPORTS_W void generateGridBoardImage(OutputArray img, Size boardSize, cv::aruco2::DictionaryType dict,
                                int bitSize = 25, InputArray ids = noArray());


/** @brief Detect a rectangular grid board of ArUco markers.
 *
 * @param image        input image (grayscale or BGR)
 * @param gridSize     board layout as columns × rows (e.g. `cv::Size(4, 3)` for a 4×3 grid)
 * @param dict         dictionary used to print the board
 * @param board        output GridBoard populated with the detected markers
 * @param ids          optional custom marker id list in row-major order;
 *                     if empty, ids 0…(cols*rows−1) are assumed
 * @return             true if at least one board marker was detected
 *
 * In Python the return value and output parameter are combined:
 * @code{.py}
 * found, board = cv.aruco2.detectGridBoard(image, (4, 3), cv.aruco2.DICT_ARUCO_MIP_36h12)
 * @endcode
 */
CV_EXPORTS_W bool detectGridBoard(InputArray image, cv::Size gridSize, cv::aruco2::DictionaryType dict,
                         CV_OUT cv::aruco2::GridBoard &board, InputArray ids = noArray());

/** @brief Draw detected board corners and optionally marker ids onto an image.
 *
 * @param image          input/output image (1 or 3 channels); modified in place
 * @param board          board returned by detectGridBoard()
 * @param color          color used to draw corner markers and text (default: green)
 * @param drawMarkerIds  if true, draws the id of each detected marker at its centroid
 *
 * For each detected board corner a filled circle is drawn together with its global corner id.
 * Useful for verifying that the board detection and corner assignment are correct.
 */
CV_EXPORTS_AS(drawDetectedGridBoard) CV_EXPORTS_W void drawDetected(InputOutputArray image, const cv::aruco2::GridBoard &board,
                               Scalar color = Scalar(0, 255, 0),bool drawMarkerIds=false);


/** @brief Compute object and image points for a detected board to pass to solvePnP().
 *
 * @param board      a detected board returned by detectGridBoard()
 * @param objPoints  output array of corresponding 3-D object points in board coordinates (CV_32FC3).
 *                   The board origin is at the top-left marker corner; X points right, Y points
 *                   down, Z=0.
 * @param imgPoints  output array of marker corner image coordinates for all detected markers (CV_32FC2)
 * @param markerSize  physical side length of one marker (e.g. metres); objPoints are scaled
 *                    by this value.  Default 1.f returns unit-size points.
 *
 * Only detected markers are included, so `imgPoints` and `objPoints` are always the same length
 * even when the board is partially occluded.
 *
 * @sa getSolvePnpPoints(const Marker, OutputArray, OutputArray)
 */
CV_EXPORTS_W void getSolvePnpPoints(const cv::aruco2::GridBoard &board, OutputArray objPoints, OutputArray imgPoints, float markerSize = 1.f);

/** @brief A detected ChArUco2-style diamond marker.
 *
 * A diamond is a 2×2 block of ArUco markers (standard on black squares, inverted on white) follwing the ChArUco2 design @cite MunozSalinas2026ChArUco2.
 *
 * Its identity is the combination of the four constituent marker ids, accessible via `id`
 * (as a `Vec4i` convenience field) or individually through each `markers[i].id`.
 *
 * @sa detectDiamonds, getSolvePnpPoints(const GridBoard, OutputArray, OutputArray)
 */
 struct CV_EXPORTS_W_SIMPLE Diamond {
    CV_PROP_RW cv::Vec4i id;                   ///< ids of the 4 constituent markers (clockwise from top-left)
    CV_PROP_RW cv::aruco2::DictionaryType dict;            ///< dictionary used for the 4 markers
    CV_PROP_RW std::vector<cv::aruco2::Marker> markers;    ///< the 4 detected markers forming the diamond
private:
    std::vector<cv::Point2f> corners;
    friend std::vector<Diamond> detectDiamonds(InputArray image, cv::aruco2::DictionaryType dict);
    friend void getSolvePnpPoints(const Diamond& diamond, OutputArray objPoints, OutputArray imgPoints, float markerSize);
    friend void drawDetected(InputOutputArray image, const std::vector<Diamond> &diamonds, Scalar color, bool drawMarkerIds);
};

/** @brief Generate a ChArUco2-style diamond image ready for printing.
 *
 * A diamond is a 2×2 block of ArUco markers following the ChArUco2 design: standard markers
 * on black squares and inverted markers on white squares @cite MunozSalinas2026ChArUco2.  The four marker ids are arranged
 * in clockwise order from the top-left, matching the `Diamond::id` field returned by
 * detectDiamonds().
 *
 * @param img           output grayscale image (CV_8UC1)
 * @param dictionary    predefined dictionary for all four markers
 * @param ids           ids of the 4 constituent markers in clockwise order from top-left
 * @param bitSize      size of each marker bit in pixels (default 20)
 *
 * @code
 * cv::Mat diamondImg;
 * cv::aruco2::generateDiamondImage(diamondImg,DICT_ARUCO_MIP_36h12, {10, 11, 12, 13} );
 * cv::imwrite("diamond.png", diamondImg);
 * @endcode
 *
 * Pass the same `dictionary` and `ids` to detectDiamonds() for detection.
 */
CV_EXPORTS_W void generateDiamondImage(OutputArray img,const cv::aruco2::DictionaryType &dictionary, const cv::Vec4i &ids,
                                  int bitSize=20);

/** @brief Detect ChArUco2-style diamond markers in an image.
 *
 * A diamond is a 2×2 block of ArUco markers (standard on black squares, inverted on white).
 * Each detected Diamond carries the 4 constituent Marker objects and their combined id as a
 * `Vec4i` for convenient access.
 *
 * @param image        input image (grayscale or BGR)
 * @param dict         dictionary used to print the diamond markers
 * @return             vector of detected Diamond objects; empty if none found
 */
CV_EXPORTS_W std::vector<cv::aruco2::Diamond> detectDiamonds(InputArray image, cv::aruco2::DictionaryType dict);

/** @brief Draw detected diamond outlines and optionally constituent marker ids onto an image.
 *
 * @param image          input/output image (1 or 3 channels); modified in place
 * @param diamonds       diamonds returned by detectDiamonds()
 * @param color          color used to draw the diamond outline and corner squares (default: green)
 * @param drawMarkerIds  if true, draws the id of each constituent marker at its centroid
 *
 * For each diamond the function draws:
 * - a quadrilateral outline connecting the 4 outer corners
 * - a small filled square at each of the 9 grid corners
 * - the Vec4i diamond id as text at the diamond centroid
 */
CV_EXPORTS_AS(drawDetectedDiamonds) CV_EXPORTS_W void drawDetected(InputOutputArray image, const std::vector<cv::aruco2::Diamond> &diamonds,
                               Scalar color = Scalar(0, 255, 0),bool drawMarkerIds=false);


/** @brief Compute object and image points for a detected diamond to pass to solvePnP().
 *
 * Returns all 9 points of the diamond's 3×3 corner grid: the 4 outer corners, the 4
 * edge mid-corners, and the central intersection point.
 *
 * @param diamond    a detected diamond returned by detectDiamonds()
 * @param objPoints  output 9×1 array of corresponding 3-D object points in diamond coordinates
 *                   (CV_32FC3).  The origin is at the top-left corner; X points right,
 *                   Y points down, Z=0.  Adjacent grid points are spaced markerSize apart.
 * @param imgPoints  output 9×1 array of image coordinates of the diamond's 3×3 corner grid (CV_32FC2)
 * @param markerSize  physical side length of one marker (e.g. metres); objPoints are scaled
 *                    by this value.  Default 1.f returns unit-size points.
 *
 * @sa getSolvePnpPoints(const Marker, OutputArray, OutputArray),
 *     getSolvePnpPoints(const GridBoard, OutputArray, OutputArray)
 */
CV_EXPORTS_W void getSolvePnpPoints(const cv::aruco2::Diamond &diamond, OutputArray objPoints, OutputArray imgPoints, float markerSize = 1.f);


/** @brief Fractal marker type — selects the nested-marker configuration.
 *
 * Each variant is a 6×6-bit design with a different number of nesting levels:
 * - FRACTAL_2L_6: 2 levels (outer + 1 inner marker)
 * - FRACTAL_3L_6: 3 levels
 * - FRACTAL_4L_6: 4 levels
 * - FRACTAL_5L_6: 5 levels — most corners, most robust pose estimation
 *
 * More levels give more visible corners (and therefore better pose accuracy), but the
 * inner markers become smaller and harder to detect at long range.
 */
enum FractalType {
    FRACTAL_2L_6=0,
    FRACTAL_3L_6,
    FRACTAL_4L_6,
    FRACTAL_5L_6
};

/** @brief A detected fractal marker.
 *
 * Fractal markers @cite romero2019fractal are nested ArUco-like markers: an outer 6×6 marker contains one or more
 * smaller markers at increasing scales.  The nesting provides many more image-to-3D
 * correspondences than a plain marker, improving pose accuracy and partial-occlusion
 * robustness.
 *
 * 3-D coordinates use a normalised frame where the outer marker spans [-1, +1] on both
 * axes (total extent = 2 units).  Pass @p markerSize to getSolvePnpPoints() to scale to
 * physical units.
 */
struct CV_EXPORTS_W_SIMPLE FractalMarker {
    CV_PROP_RW std::vector<cv::Point2f> corners; ///< 4 outer corners, clockwise from top-left
    CV_PROP_RW cv::aruco2::FractalType type;                 ///< fractal configuration used for detection
    CV_PROP_RW int id = -1;                      ///< id of the outer (external) marker
    CV_WRAP Point2f getCorner(int i) const { return corners[i]; }
private:
    std::vector<cv::Point2f> imgPoints; ///< all 2-D correspondences (set by detectFractals)
    std::vector<cv::Point3f> objPoints; ///< matching 3-D model points in normalised space
    friend std::vector<cv::aruco2::FractalMarker> detectFractals(InputArray image, cv::aruco2::FractalType ftype);
    friend void getSolvePnpPoints(const cv::aruco2::FractalMarker &fractal, OutputArray objPoints, OutputArray imgPoints, float markerSize);
    friend void drawDetected(InputOutputArray image, const std::vector<cv::aruco2::FractalMarker> &fractals, Scalar color, bool drawAllImagePoints);
};

/** @brief Render a fractal marker to a grayscale image.
 *
 * Generates a white-background image containing the full nested marker pattern.
 * The image is square with side length @p (nBits + 2) * bitSize pixels, where
 * @p nBits = 6 for all current configurations.
 *
 * @param img      output CV_8UC1 image
 * @param ftype    fractal configuration (FRACTAL_2L_6 … FRACTAL_5L_6)
 * @param bitSize  side length of one bit cell in pixels (default 20)
 */
CV_EXPORTS_W void generateFractalImage(OutputArray img, cv::aruco2::FractalType ftype, int bitSize=20);

/** @brief Detect fractal markers in an image.
 *
 * Returns one FractalMarker per detected instance.  Each result carries the 4 outer
 * corners and, when only one marker is present in the scene, all inner-corner
 * correspondences needed for full pose estimation.
 *
 * @param image  input image (BGR or grayscale)
 * @param ftype  fractal configuration to search for
 * @return vector of detected fractal markers; empty if none found
 */
CV_EXPORTS_W std::vector<cv::aruco2::FractalMarker> detectFractals(InputArray image, cv::aruco2::FractalType ftype);

/** @brief Draw detected fractal markers on an image.
 *
 * Draws a coloured quadrilateral around each marker, a red dot on @p corners[0] to
 * indicate orientation, and the marker id at the centroid.
 *
 * @param image            input/output BGR or grayscale image
 * @param fractals         vector returned by detectFractals()
 * @param color            border and label colour (default green)
 * @param drawAllImagePoints  if true, draw a small circle at every matched image point
 *                         stored inside each FractalMarker (default true)
 */
CV_EXPORTS_AS(drawDetectedFractals) CV_EXPORTS_W void drawDetected(InputOutputArray image, const std::vector<cv::aruco2::FractalMarker> &fractals,
                                  Scalar color = Scalar(0, 255, 0), bool drawAllImagePoints = true);

/** @brief Extract solvePnP inputs for a detected fractal marker.
 *
 * Copies the image-to-3D correspondences stored inside @p fractal into flat output
 * arrays ready for cv::solvePnP().  When only one marker was present in the scene,
 * @p imgPoints and @p objPoints contain all visible corners (outer and inner); otherwise
 * only the 4 outer corners are returned.
 *
 * 3-D coordinates are in normalised space where the outer marker spans [-1, +1] on both
 * axes.  @p markerSize scales them to physical units (e.g. metres).
 *
 * @param fractal     a detected fractal marker returned by detectFractals()
 * @param objPoints   output N×1 array of 3-D object points (CV_32FC3)
 * @param imgPoints   output N×1 array of 2-D image points (CV_32FC2)
 * @param markerSize  physical side length of the outer marker; objPoints are scaled by
 *                    @p markerSize / 2.  Default 1.f returns normalised-space points.
 *
 * @sa getSolvePnpPoints(const Marker &, OutputArray, OutputArray, float),
 *     getSolvePnpPoints(const GridBoard &, OutputArray, OutputArray, float),
 *     getSolvePnpPoints(const Diamond &, OutputArray, OutputArray, float)
 */
CV_EXPORTS_W void getSolvePnpPoints(const cv::aruco2::FractalMarker &fractal, OutputArray objPoints,
                               OutputArray imgPoints, float markerSize = 1.f);

//! @}

} // namespace aruco2
} // namespace cv
