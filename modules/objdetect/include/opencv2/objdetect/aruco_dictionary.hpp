// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_DICTIONARY_HPP
#define OPENCV_OBJDETECT_DICTIONARY_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace aruco {

//! @addtogroup objdetect_aruco
//! @{


/** @brief Encoding used to store the markers of a Dictionary in `Dictionary::bytesList`
 */
enum DictionaryEncoding {
    /// binary cells (black=0 / white=1) packed in bytes (default)
    DICT_ENCODING_BINARY = 0,
    /// each cell stores its white pixel ratio in percent [0,100], one byte per cell.
    /// This allows non-binary cells, e.g. a cell holding a nested marker
    DICT_ENCODING_CELL_RATIO = 1
};

/** @brief Dictionary is a set of unique ArUco markers of the same size
 *
 * If `dictEncoding` is DICT_ENCODING_BINARY:
 * `bytesList` storing as 2-dimensions Mat with 4-th channels (CV_8UC4 type was used) and contains the marker codewords where:
 * - bytesList.rows is the dictionary size
 * - each marker is encoded using `nbytes = ceil(markerSize*markerSize/8.)` bytes
 * - each row contains all 4 rotations of the marker, so its length is `4*nbytes`
 * - the byte order in the bytesList[i] row:
 * `//bytes without rotation/bytes with rotation 1/bytes with rotation 2/bytes with rotation 3//`
 * So `bytesList.ptr(i)[k*nbytes + j]` is the j-th byte of i-th marker, in its k-th rotation.
 * @note Python bindings generate matrix with shape of bytesList `dictionary_size x nbytes x 4`,
 * but it should be indexed like C++ version. Python example for j-th byte of i-th marker, in its k-th rotation:
 * `aruco_dict.bytesList[id].ravel()[k*nbytes + j]`
 *
 * If `dictEncoding` is DICT_ENCODING_CELL_RATIO:
 * `bytesList` stores one byte per marker cell instead of one bit:
 * - bytesList.rows is the dictionary size
 * - each row is a CV_8UC4 Mat with `markerSize*markerSize` columns and contains all 4 rotations of the
 *   marker, so its length is `4*markerSize*markerSize` bytes:
 * `//cells without rotation/cells with rotation 1/cells with rotation 2/cells with rotation 3//`
 * - each cell value is the ratio of white pixels in the cell in percent, between 0 (black cell) and
 *   100 (white cell), e.g. 75 describes a cell with 75% white and 25% black pixels.
 *
 * Non-binary cells allow markers to be nested inside the cells of larger markers
 * (see DetectorParameters::detectNestedMarkers).
 */
class CV_EXPORTS_W_SIMPLE Dictionary {

    public:
    CV_PROP_RW Mat bytesList;         ///< marker code information. See class description for more details
    CV_PROP_RW int markerSize;        ///< number of bits per dimension
    CV_PROP_RW int maxCorrectionBits; ///< maximum number of bits that can be corrected
    CV_PROP_RW int dictEncoding;      ///< encoding of bytesList, see DictionaryEncoding (default DICT_ENCODING_BINARY)

    CV_WRAP Dictionary();

    /** @brief Basic ArUco dictionary constructor
     *
     * @param bytesList bits for all ArUco markers in dictionary see memory layout in the class description
     * @param _markerSize ArUco marker size in units
     * @param maxcorr maximum number of bits that can be corrected
     * @param dictEncoding encoding of bytesList, see DictionaryEncoding
     */
    CV_WRAP Dictionary(const Mat &bytesList, int _markerSize, int maxcorr = 0, int dictEncoding = (int)DICT_ENCODING_BINARY);

    /** @brief Read a new dictionary from FileNode.
     *
     * Dictionary example in YAML format:\n
     * nmarkers: 35\n
     * markersize: 6\n
     * maxCorrectionBits: 5\n
     * marker_0: "101011111011111001001001101100000000"\n
     * ...\n
     * marker_34: "011111010000111011111110110101100101"
     *
     * Dictionaries with DICT_ENCODING_CELL_RATIO encoding add a `dictEncoding` entry and store each
     * marker as the list of its cell ratios in percent:\n
     * nmarkers: 35\n
     * markersize: 6\n
     * maxCorrectionBits: 5\n
     * dictEncoding: 1\n
     * marker_0: [100, 0, 75, ...]\n
     * If the `dictEncoding` entry is missing, DICT_ENCODING_BINARY is assumed.
     */
    CV_WRAP bool readDictionary(const cv::FileNode& fn);

    /** @brief Write a dictionary to FileStorage, format is the same as in readDictionary().
     */
    CV_WRAP void writeDictionary(FileStorage& fs, const String& name = String());

    /** @brief Given a matrix of bits. Returns whether if marker is identified or not.
     *
     * Returns reference to the marker id in the dictionary (if any) and its rotation.
     */
    CV_WRAP bool identify(const Mat &onlyBits, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate) const;

    /** @brief Given a matrix of pixel ratio ranging from 0 to 1. Returns whether the marker is identified or not.
     *
     * Returns reference to the marker id in the dictionary (if any) and its rotation.
     */
    CV_WRAP bool identify(const Mat &onlyCellPixelRatio, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate, float validBitIdThreshold) const;

    /** @brief Returns Hamming distance of the input bits to the specific id.
     *
     * If `allRotations` flag is set, the four possible marker rotations are considered
     */
    CV_WRAP int getDistanceToId(InputArray bits, int id, bool allRotations = true) const;

    /** @brief Returns number of cells that differ from the specific id.
     *
     * For each cell, the distance is increased when the difference between the detected
     * cell pixel ratio and the dictionary bit value is greater than `validBitIdThreshold`.
     * If `allRotations` is set, the four possible marker rotations are considered.
     *
     * @param onlyCellPixelRatio markerSize x markerSize matrix (CV_32FC1) holding, for each cell,
     * the ratio of white pixels ranging from 0 to 1
     * @param id marker id in the dictionary to compute the distance to
     * @param allRotations if set, the four possible marker rotations are considered and the
     * smallest distance is returned
     * @param validBitIdThreshold maximum allowed difference between a cell pixel ratio and the
     * dictionary bit value; cells exceeding it are counted as differing
     */
    CV_WRAP int getDistanceToId(InputArray onlyCellPixelRatio, int id, bool allRotations, float validBitIdThreshold) const;

    /** @brief Generate a canonical marker image
     *
     * For `DICT_ENCODING_CELL_RATIO` dictionaries, this function renders each cell as a constant
     * grayscale intensity proportional to the expected ratio (0..255). It does not synthesize a
     * binary pattern with the requested white-pixel ratio, so non-binary cells are primarily intended
     * for visualization and may not be directly detectable by the ArUco detector.
     */
    CV_WRAP void generateImageMarker(int id, int sidePixels, OutputArray _img, int borderBits = 1) const;


    /** @brief Transform matrix of bits to list of bytes with 4 marker rotations
      */
    CV_WRAP static Mat getByteListFromBits(const Mat &bits);


    /** @brief Transform list of bytes to matrix of bits
      */
    CV_WRAP static Mat getBitsFromByteList(const Mat &byteList, int markerSize, int rotationId = 0);


    /** @brief Transform matrix of cell ratios to list of cell ratios with 4 marker rotations
      *
      * @param cellRatios marker cells as `markerSize x markerSize` CV_8UC1 Mat, each cell holding its
      * white pixel ratio in percent [0,100]
      *
      * Returns one `bytesList` row for a dictionary with DICT_ENCODING_CELL_RATIO encoding.
      */
    CV_WRAP static Mat getRatioListFromCellRatios(const Mat &cellRatios);


    /** @brief Transform list of cell ratios to matrix of cell ratios, see getRatioListFromCellRatios()
      */
    CV_WRAP static Mat getCellRatiosFromRatioList(const Mat &ratioList, int markerSize, int rotationId = 0);


    /** @brief Compute the white pixel ratio of each marker cell from a marker image
      *
      * @param markerImage grayscale image of the marker, including its border
      * (e.g. the output of generateImageMarker() or generateImageMarkerNested())
      * @param markerSize number of cells per dimension, without the border
      * @param borderBits width of the marker border in cells
      *
      * Pixels above 127 are counted as white. Returns a `markerSize x markerSize` CV_8UC1 Mat
      * holding the white pixel ratio of each cell in percent [0,100], the format expected by
      * getRatioListFromCellRatios(). Useful to build custom cell-ratio dictionaries from composed
      * marker images, e.g. markers hosting nested markers.
      */
    CV_WRAP static Mat getCellRatiosFromImage(InputArray markerImage, int markerSize, int borderBits = 1);


    /** @brief Create a cell-ratio encoded copy of this binary dictionary
      *
      * The returned dictionary preserves the marker ids, marker size, rotations, and maximum
      * correction bits, but stores each binary cell as a ratio value: 0 for black and 100 for white.
      * This is useful as a starting point for custom cell-ratio dictionaries, for example when
      * replacing some marker cells with non-binary ratios that represent nested markers.
      *
      * The source dictionary must use DICT_ENCODING_BINARY.
      */
    CV_WRAP Dictionary convertToCellRatioDictionary() const;


    /** @brief Get the expected cell values of a marker as a `markerSize x markerSize` CV_32FC1 Mat
      *
      * Each cell holds its expected white pixel ratio in [0,1]. For DICT_ENCODING_BINARY dictionaries
      * the values are exactly 0 or 1.
      */
    CV_WRAP Mat getMarkerBits(int markerId, int rotationId = 0) const;
};




/** @brief Predefined markers dictionaries/sets
 *
 * Each dictionary indicates the number of bits and the number of markers contained
 * - DICT_ARUCO_ORIGINAL: standard ArUco Library Markers. 1024 markers, 5x5 bits, 0 minimum
                          distance
 */
enum PredefinedDictionaryType {
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
    DICT_APRILTAG_16h5,     ///< 4x4 bits, minimum hamming distance between any two codes = 5, 30 codes
    DICT_APRILTAG_25h9,     ///< 5x5 bits, minimum hamming distance between any two codes = 9, 35 codes
    DICT_APRILTAG_36h10,    ///< 6x6 bits, minimum hamming distance between any two codes = 10, 2320 codes
    DICT_APRILTAG_36h11,     ///< 6x6 bits, minimum hamming distance between any two codes = 11, 587 codes
    DICT_ARUCO_MIP_36h12,    ///< 6x6 bits, minimum hamming distance between any two codes = 12, 250 codes
    DICT_4X4_NESTED_5,       ///< 4x4 cells, 5 nested marker pairs (10 ids), minimum separation distance = 4
    DICT_4X4_NESTED_10,      ///< 4x4 cells, 10 nested marker pairs (20 ids), minimum separation distance = 3
    DICT_4X4_NESTED_24       ///< 4x4 cells, 24 nested marker pairs (48 ids), minimum separation distance = 2
};


/** @brief Returns one of the predefined dictionaries defined in PredefinedDictionaryType
  */
CV_EXPORTS Dictionary getPredefinedDictionary(PredefinedDictionaryType name);


/** @brief Returns one of the predefined dictionaries referenced by DICT_*.
  */
CV_EXPORTS_W Dictionary getPredefinedDictionary(int dict);

/** @brief Extend base dictionary by new nMarkers
  *
  * @param nMarkers number of markers in the dictionary
  * @param markerSize number of bits per dimension of each markers
  * @param baseDictionary Include the markers in this dictionary at the beginning (optional)
  * @param randomSeed a user supplied seed for theRNG()
  *
  * This function creates a new dictionary composed by nMarkers markers and each markers composed
  * by markerSize x markerSize bits. If baseDictionary is provided, its markers are directly
  * included and the rest are generated based on them. If the size of baseDictionary is higher
  * than nMarkers, only the first nMarkers in baseDictionary are taken and no new marker is added.
  */
CV_EXPORTS_W Dictionary extendDictionary(int nMarkers, int markerSize, const Dictionary &baseDictionary = Dictionary(),
                                         int randomSeed=0);


/** @brief Generate a dictionary of nested marker pairs
  *
  * @param nPairs number of marker pairs; the dictionary holds 2*nPairs entries
  * @param markerSize number of cells per dimension of each marker
  * @param minDistance minimum separation distance between any two entries and between an entry
  * and its own rotations. Two cell values are separating when no single observation can match
  * both of them under the default DetectorParameters::validBitIdThreshold
  * @param innerHalfDiagonal half diagonal (center to corner) of the inner marker square, in
  * outer cell units. The inner marker is rotated 45 degrees, so its diagonals align with the
  * cell grid and it covers a triangle of area `innerHalfDiagonal^2 / 2` in each of the 4 host
  * cells. Must be in [0.2, sqrt(0.5)]: the upper bound keeps that coverage at or below a
  * quarter of the cell, so host ratios stay within 25 percent of their color; the lower
  * bound keeps them measurably different from plain black or white. It must be passed
  * unchanged to generateImageMarkerNested()
  * @param randomSeed user supplied seed for theRNG()
  *
  * A nested marker pair is one printed marker showing two ids: an outer marker (even id `2k`)
  * hosting a smaller inner marker (odd id `2k + 1`) that is rotated 45 degrees and centered on
  * the corner shared by the outer marker's unique white 2x2 cell block.
  *
  * The returned dictionary uses DICT_ENCODING_CELL_RATIO and requires
  * DetectorParameters::detectNestedMarkers to detect both markers of a pair.
  * Generation is deterministic for a given randomSeed. Each pattern keeps at least 4 cells of
  * each color, so plain bright or dark quads never resemble a marker.
  */
CV_EXPORTS_W Dictionary generateNestedDictionary(int nPairs, int markerSize, int minDistance = 3,
                                                 float innerHalfDiagonal = 0.7f, int randomSeed = 0);


/** @brief Generate the printable image of a nested marker pair
  *
  * @param dictionary dictionary of nested marker pairs, see generateNestedDictionary()
  * @param outerId even id of the outer marker of the pair; the inner marker id is outerId + 1
  * @param sidePixels size of the output image in pixels
  * @param img output image with the outer marker hosting its rotated inner marker
  * @param borderBits width of the outer marker border in cells
  * @param innerHalfDiagonal half diagonal of the inner marker square in outer cell units, see
  * generateNestedDictionary(); must match the value used when the dictionary was generated
  *
  * The host block is located automatically from the dictionary entry (the 2x2 group of cells
  * with a non-binary expected ratio). Prefer sidePixels that give the inner marker enough
  * resolution, e.g. at least 60 pixels per outer cell.
  */
CV_EXPORTS_W void generateImageMarkerNested(const Dictionary &dictionary, int outerId,
                                            int sidePixels, OutputArray img, int borderBits = 1,
                                            float innerHalfDiagonal = 0.7f);


/** @brief Corner positions of both markers of a nested pair, for building a Board
  *
  * @param dictionary dictionary of nested marker pairs, see generateNestedDictionary()
  * @param outerId even id of the outer marker of the pair
  * @param outerSideLength printed side length of the outer marker, including its border
  * @param outerCorners 4x3 CV_32F, corners of the outer marker (id outerId)
  * @param innerCorners 4x3 CV_32F, corners of the inner marker (id outerId + 1)
  * @param borderBits width of the outer marker border in cells, as printed
  * @param innerHalfDiagonal half diagonal of the inner marker square in outer cell units, see
  * generateNestedDictionary(); must match the value used when the dictionary was generated
  *
  * Both corner arrays are in the same planar frame: origin at the outer marker top left corner,
  * x right, y down, z = 0, in the top-left, top-right, bottom-right, bottom-left order used by
  * the detector. Can be passed to a cv::aruco::Board with ids {outerId, outerId + 1} so
  * Board::matchImagePoints() fuses detections into one pose estimate.
  */
CV_EXPORTS_W void getNestedMarkerObjectPoints(const Dictionary &dictionary, int outerId,
                                              float outerSideLength, OutputArray outerCorners,
                                              OutputArray innerCorners, int borderBits = 1,
                                              float innerHalfDiagonal = 0.7f);


//! @}
}
}

#endif
