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


/** @brief Dictionary/Set of markers, it contains the inner codification
 *
 * BytesList contains the marker codewords where:
 * - bytesList.rows is the dictionary size
 * - each marker is encoded using `nbytes = ceil(markerSize*markerSize/8.)`
 * - each row contains all 4 rotations of the marker, so its length is `4*nbytes`
 *
 * `bytesList.ptr(i)[k*nbytes + j]` is then the j-th byte of i-th marker, in its k-th rotation.
 */
class CV_EXPORTS_W_SIMPLE Dictionary {

    public:
    CV_PROP_RW Mat bytesList;         // marker code information
    CV_PROP_RW int markerSize;        // number of bits per dimension
    CV_PROP_RW int maxCorrectionBits; // maximum number of bits that can be corrected


    CV_WRAP Dictionary();

    CV_WRAP Dictionary(const Mat &bytesList, int _markerSize, int maxcorr = 0);



    /** @brief Read a new dictionary from FileNode.
     *
     * Dictionary format:\n
     * nmarkers: 35\n
     * markersize: 6\n
     * maxCorrectionBits: 5\n
     * marker_0: "101011111011111001001001101100000000"\n
     * ...\n
     * marker_34: "011111010000111011111110110101100101"
     */
    CV_WRAP bool readDictionary(const cv::FileNode& fn);

    /** @brief Write a dictionary to FileStorage, format is the same as in readDictionary().
     */
    CV_WRAP void writeDictionary(FileStorage& fs, const String& name = String());

    /** @brief Given a matrix of bits. Returns whether if marker is identified or not.
     *
     * It returns by reference the correct id (if any) and the correct rotation
     */
    CV_WRAP bool identify(const Mat &onlyBits, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate) const;

    /** @brief Returns the distance of the input bits to the specific id.
     *
     * If allRotations is true, the four posible bits rotation are considered
     */
    CV_WRAP int getDistanceToId(InputArray bits, int id, bool allRotations = true) const;


    /** @brief Generate a canonical marker image
     */
    CV_WRAP void generateImageMarker(int id, int sidePixels, OutputArray _img, int borderBits = 1) const;


    /** @brief Transform matrix of bits to list of bytes in the 4 rotations
      */
    CV_WRAP static Mat getByteListFromBits(const Mat &bits);


    /** @brief Transform list of bytes to matrix of bits
      */
    CV_WRAP static Mat getBitsFromByteList(const Mat &byteList, int markerSize);
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
    DICT_APRILTAG_36h11     ///< 6x6 bits, minimum hamming distance between any two codes = 11, 587 codes
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



//! @}
}
}

#endif
