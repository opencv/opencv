// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#pragma once

#include <opencv2/core.hpp>
#include "opencv2/objdetect/aruco2.hpp"
#include <unordered_map>
namespace cv {
namespace aruco2 {

//! @addtogroup objdetect_aruco
//! @{


/** @brief Dictionary is a set of unique ArUco markers of the same size
  *
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
  */
class CV_EXPORTS_W_SIMPLE Dictionary {

    public:
    CV_PROP_RW Mat bytesList;         ///< marker code information. See class description for more details
    CV_PROP_RW int markerSize;        ///< number of bits per dimension
    CV_PROP_RW int maxCorrectionBits; ///< maximum number of bits that can be corrected
    CV_PROP_RW std::unordered_map<uint64_t, std::pair<int, int>> bits_id; ///< a map with all marker bytes and its associated (id, rotation)


    CV_WRAP Dictionary();

    /** @brief Basic ArUco dictionary constructor
     *
     * @param bytesList bits for all ArUco markers in dictionary see memory layout in the class description
     * @param _markerSize ArUco marker size in units
     * @param maxcorr maximum number of bits that can be corrected
     */
    CV_WRAP Dictionary(const Mat &bytesList, int _markerSize, int maxcorr = 0);

    /**
     * Returns the number of markers in this dictionary
     */
    CV_WRAP size_t size()const;
    /** @brief Read a new dictionary from FileNode.
     *
     * Dictionary example in YAML format:\n
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
     * Returns reference to the marker id in the dictionary (if any) and its rotation.
     */
    CV_WRAP bool identify(const Mat &onlyBits, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate) const;

    /** @brief Returns Hamming distance of the input bits to the specific id.
     *
     * If `allRotations` flag is set, the four possible marker rotations are considered
     */
    CV_WRAP int getDistanceToId(InputArray bits, int id, bool allRotations = true) const;

    /** @brief Generate a canonical marker image
     */
    CV_WRAP void generateImageMarker(int id, int sidePixels, OutputArray _img, int borderBits = 1) const;


    /** @brief Transform matrix of bits to list of bytes with 4 marker rotations
      */
    CV_WRAP static Mat getByteListFromBits(const Mat &bits);


    /** @brief Transform list of bytes to matrix of bits
      */
    CV_WRAP static Mat getBitsFromByteList(const Mat &byteList, int markerSize, int rotationId = 0);

    /** @brief Get ground truth bits float
      */
     CV_WRAP Mat getMarkerBits(int markerId, int rotationId = 0) const;
};







/** @brief Returns one of the predefined dictionaries defined in PredefinedDictionaryType
  */
CV_EXPORTS Dictionary getPredefinedDictionary(DictionaryType name);


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
} // namespace aruco2
} // namespace cv

