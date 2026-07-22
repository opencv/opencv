// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "opencv2/core/hal/hal.hpp"

#include "aruco_utils.hpp"
#include "predefined_dictionaries.hpp"
#include "apriltag/predefined_dictionaries_apriltag.hpp"
#include <opencv2/objdetect/aruco_dictionary.hpp>

namespace cv {
namespace aruco {

using namespace std;

struct CellBitMasks {
    CellBitMasks(const Mat &onlyCellPixelRatio, int markerSize, float validBitIdThreshold)
        : s((markerSize * markerSize + 8 - 1) / 8),
          totalCells(markerSize * markerSize),
          temp(4 * s),
          not0(temp.data()), not1(not0 + s), notXor(not1 + s), temp0(temp.data() + 3 * s) {
        uint8_t* not0Writable = temp.data();
        uint8_t* not1Writable = not0Writable + s;
        uint8_t* notXorWritable = not1Writable + s;

        // Fill bit masks of cells that are not black (not0) and not white (not1).
        unsigned char not0Byte = 0, not1Byte = 0;
        int currentByte = 0, currentBit = 0;
        for(int j = 0; j < markerSize; j++) {
            const float* cellPixelRatioRow = onlyCellPixelRatio.ptr<float>(j);
            for(int i = 0; i < markerSize; i++) {
                not0Byte <<= 1; not1Byte <<= 1;
                if(cellPixelRatioRow[i] > validBitIdThreshold) not0Byte |= 1;
                if(cellPixelRatioRow[i] < 1 - validBitIdThreshold) not1Byte |= 1;
                ++currentBit;
                if(currentBit == 8) {
                    not0Writable[currentByte] = not0Byte;
                    not1Writable[currentByte] = not1Byte;
                    not0Byte = not1Byte = 0;
                    ++currentByte;
                    currentBit = 0;
                }
            }
        }
        if(currentBit != 0) {
            not0Writable[currentByte] = not0Byte;
            not1Writable[currentByte] = not1Byte;
        }

        // Computing: notXor = not0 ^ not1
        hal::xor8u(not0, s, not1, s, notXorWritable, s, s, 1, nullptr);
    }

    CellBitMasks(const CellBitMasks&) = delete;
    CellBitMasks& operator=(const CellBitMasks&) = delete;

    // Smallest Hamming distance between these cell masks and dictionary marker `id`,
    // searching the tested rotations; `rotation` returns the best one.
    // Mutates the internal buffer (temp0).
    int hammingDistanceToId(const Mat& bytesList, int id, bool allRotations, int& rotation) {
        CV_Assert(id >= 0 && id < bytesList.rows);

        const unsigned int nRotations = allRotations ? 4u : 1u;
        int currentMinDistance = totalCells + 1;
        rotation = -1;

        const uchar* bytesRot = bytesList.ptr(id);
        for(unsigned int r = 0; r < nRotations; r++, bytesRot += s) {
            // Error if (marker is 0 and input is not 0) or (marker is 1 and input is not 1)
            // i.e.: (!bytesRot && not0) || (bytesRot && not1)
            // This is equivalent to: not0 ^ ((not0 ^ not1) & bytesRot)
            // Computing: temp0 = (not0 ^ not1) & bytesRot
            hal::and8u(notXor, s, bytesRot, s, temp0, s, s, 1, nullptr);
            // Computing the final result (xor is performed internally).
            int currentHamming = cv::hal::normHamming(not0, temp0, s);

            if(currentHamming < currentMinDistance) {
                currentMinDistance = currentHamming;
                rotation = static_cast<int>(r);
                // Break for perfect distance.
                if(currentMinDistance == 0) break;
            }
        }

        return currentMinDistance;
    }

    const int s; // bytes per rotation
    const int totalCells;
    std::vector<uint8_t> temp;
    const uint8_t *not0, *not1, *notXor;
    uint8_t *temp0;  // internal scratch workspace
};


Dictionary::Dictionary(): markerSize(0), maxCorrectionBits(0) {}


Dictionary::Dictionary(const Mat &_bytesList, int _markerSize, int _maxcorr) {
    markerSize = _markerSize;
    maxCorrectionBits = _maxcorr;
    bytesList = _bytesList;
}


bool Dictionary::readDictionary(const cv::FileNode& fn) {
    int nMarkers = 0, _markerSize = 0;
    if (fn.empty() || !readParameter("nmarkers", nMarkers, fn) || !readParameter("markersize", _markerSize, fn))
        return false;
    Mat bytes(0, 0, CV_8UC1), marker(_markerSize, _markerSize, CV_8UC1);
    std::string markerString;
    for (int i = 0; i < nMarkers; i++) {
        std::ostringstream ostr;
        ostr << i;
        if (!readParameter("marker_" + ostr.str(), markerString, fn))
            return false;
        for (int j = 0; j < (int) markerString.size(); j++)
            marker.at<unsigned char>(j) = (markerString[j] == '0') ? 0 : 1;
        bytes.push_back(Dictionary::getByteListFromBits(marker));
    }
    int _maxCorrectionBits = 0;
    readParameter("maxCorrectionBits", _maxCorrectionBits, fn);
    *this = Dictionary(bytes, _markerSize, _maxCorrectionBits);
    return true;
}


void Dictionary::writeDictionary(FileStorage& fs, const String &name)
{
    CV_Assert(fs.isOpened());

    if (!name.empty())
        fs << name << "{";

    fs << "nmarkers" << bytesList.rows;
    fs << "markersize" << markerSize;
    fs << "maxCorrectionBits" << maxCorrectionBits;
    for (int i = 0; i < bytesList.rows; i++) {
        Mat row = bytesList.row(i);;
        Mat bitMarker = getBitsFromByteList(row, markerSize);
        std::ostringstream ostr;
        ostr << i;
        string markerName = "marker_" + ostr.str();
        string marker;
        for (int j = 0; j < markerSize * markerSize; j++)
            marker.push_back(bitMarker.at<uint8_t>(j) + '0');
        fs << markerName << marker;
    }

    if (!name.empty())
        fs << "}";
}


bool Dictionary::identify(const Mat &onlyCellPixelRatio, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate, float validBitIdThreshold) const {
    CV_Assert(onlyCellPixelRatio.rows == markerSize && onlyCellPixelRatio.cols == markerSize);
    CV_Assert(onlyCellPixelRatio.type() == CV_32FC1);

    CellBitMasks cellBitMasks(onlyCellPixelRatio, markerSize, validBitIdThreshold);

    int maxCorrectionRecalculed = int(double(maxCorrectionBits) * maxCorrectionRate);

    idx = -1; // by default, not found

    // search closest marker in dict
    for(int m = 0; m < bytesList.rows; m++) {
        int currentRotation = -1;
        int currentMinDistance = cellBitMasks.hammingDistanceToId(bytesList, m, true, currentRotation);

        // if maxCorrection is fulfilled, return this one
        if(currentMinDistance <= maxCorrectionRecalculed) {
            idx = m;
            rotation = currentRotation;
            break;
        }
    }

    return idx != -1;
}


bool Dictionary::identify(const Mat &onlyBits, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate) const {
    CV_Assert(onlyBits.rows == markerSize && onlyBits.cols == markerSize);

    Mat candidateBitRatio;
    Mat(onlyBits > 0).convertTo(candidateBitRatio, CV_32F, 1.0 / 255.0);
    return identify(candidateBitRatio, idx, rotation, maxCorrectionRate, DEFAULT_VALID_BIT_ID_THRESHOLD);
}


int Dictionary::getDistanceToId(InputArray bits, int id, bool allRotations) const {

    CV_Assert(id >= 0 && id < bytesList.rows);

    unsigned int nRotations = 4;
    if(!allRotations) nRotations = 1;

    Mat candidateBytes = getByteListFromBits(bits.getMat());
    int currentMinDistance = int(bits.total() * bits.total());
    for(unsigned int r = 0; r < nRotations; r++) {
        int currentHamming = cv::hal::normHamming(
                bytesList.ptr(id) + r*candidateBytes.cols,
                candidateBytes.ptr(),
                candidateBytes.cols);

        if(currentHamming < currentMinDistance) {
            currentMinDistance = currentHamming;
        }
    }
    return currentMinDistance;
}


int Dictionary::getDistanceToId(InputArray onlyCellPixelRatio, int id, bool allRotations, float validBitIdThreshold) const {

    Mat onlyCellPixelRatioMat = onlyCellPixelRatio.getMat();
    CV_Assert(onlyCellPixelRatioMat.rows == markerSize && onlyCellPixelRatioMat.cols == markerSize);
    CV_Assert(onlyCellPixelRatioMat.type() == CV_32FC1);
    CV_Assert(id >= 0 && id < bytesList.rows);

    int rotation = -1;
    CellBitMasks cellBitMasks(onlyCellPixelRatioMat, markerSize, validBitIdThreshold);
    return cellBitMasks.hammingDistanceToId(bytesList, id, allRotations, rotation);
}


void Dictionary::generateImageMarker(int id, int sidePixels, OutputArray _img, int borderBits) const {
    CV_Assert(sidePixels >= (markerSize + 2*borderBits));
    CV_Assert(id < bytesList.rows);
    CV_Assert(borderBits > 0);

    _img.create(sidePixels, sidePixels, CV_8UC1);

    // create small marker with 1 pixel per bin
    Mat tinyMarker(markerSize + 2 * borderBits, markerSize + 2 * borderBits, CV_8UC1,
                   Scalar::all(0));
    Mat innerRegion = tinyMarker.rowRange(borderBits, tinyMarker.rows - borderBits)
                          .colRange(borderBits, tinyMarker.cols - borderBits);
    // put inner bits
    Mat bits = getMarkerBits(id);
    bits.convertTo(bits, CV_8U, 255.0);
    CV_Assert(innerRegion.total() == bits.total());
    bits.copyTo(innerRegion);

    // resize tiny marker to output size
    cv::resize(tinyMarker, _img.getMat(), _img.getMat().size(), 0, 0, INTER_NEAREST);
}


Mat Dictionary::getByteListFromBits(const Mat &bits) {
    // integer ceil
    int nbytes = (bits.cols * bits.rows + 8 - 1) / 8;

    Mat candidateByteList(1, nbytes, CV_8UC4, Scalar::all(0));
    unsigned char currentBit = 0;
    int currentByte = 0;

    // the 4 rotations
    uchar* rot0 = candidateByteList.ptr();
    uchar* rot1 = candidateByteList.ptr() + 1*nbytes;
    uchar* rot2 = candidateByteList.ptr() + 2*nbytes;
    uchar* rot3 = candidateByteList.ptr() + 3*nbytes;

    for(int row = 0; row < bits.rows; row++) {
        for(int col = 0; col < bits.cols; col++) {
            // circular shift
            rot0[currentByte] <<= 1;
            rot1[currentByte] <<= 1;
            rot2[currentByte] <<= 1;
            rot3[currentByte] <<= 1;
            // set bit
            rot0[currentByte] |= bits.at<uchar>(row, col);
            rot1[currentByte] |= bits.at<uchar>(col, bits.cols - 1 - row);
            rot2[currentByte] |= bits.at<uchar>(bits.rows - 1 - row, bits.cols - 1 - col);
            rot3[currentByte] |= bits.at<uchar>(bits.rows - 1 - col, row);
            currentBit++;
            if(currentBit == 8) {
                // next byte
                currentBit = 0;
                currentByte++;
            }
        }
    }
    return candidateByteList;
}


Mat Dictionary::getMarkerBits(int markerId, int rotationId) const {

    const int nbRotations = 4;
    CV_Assert(markerId < bytesList.rows);
    CV_Assert(rotationId < nbRotations);

    Mat bits(markerSize, markerSize, CV_32F, Scalar::all(0));
    Mat bitsUints = getBitsFromByteList(bytesList.rowRange(markerId, markerId + 1), markerSize, rotationId);
    bitsUints.convertTo(bits, CV_32F);

    CV_Assert(bits.rows == markerSize && bits.cols == markerSize);
    return bits;
}


Mat Dictionary::getBitsFromByteList(const Mat &byteList, int markerSize, int rotationId) {
    CV_Assert(byteList.total() > 0 &&
              byteList.total() >= (unsigned int)markerSize * markerSize / 8 &&
              byteList.total() <= (unsigned int)markerSize * markerSize / 8 + 1);

    CV_Assert(byteList.channels() >= 4);
    CV_Assert(rotationId >= 0 && rotationId < 4);

    Mat bits = Mat::zeros(markerSize, markerSize, CV_8UC1);
    unsigned char *bitsPtr = bits.ptr();

    // Use a base offset for the selected rotation
    int nbytes = (bits.cols * bits.rows + 8 - 1) / 8; // integer ceil
    int base = rotationId * nbytes;
    const unsigned char *currentBytePtr = byteList.ptr() + base;
    const unsigned char *currentBytePtrEnd = currentBytePtr + bits.total() / 8;

    for(;currentBytePtr < currentBytePtrEnd; ++currentBytePtr) {
        unsigned char currentByte = *currentBytePtr;
        for(int mask = 1 << 7; mask != 0; mask >>= 1) {
            if (currentByte & mask) *bitsPtr = 1;
            ++bitsPtr;
        }
    }
    // if not enough bits for one more byte, we are in the end
    // update bit position accordingly
    if (bits.total() % 8 != 0) {
        unsigned char currentByte = *currentBytePtrEnd;
        int mask = 1 << ((bits.total() % 8) - 1);
        for(; mask != 0; mask >>= 1) {
            if (currentByte & mask) *bitsPtr = 1;
            ++bitsPtr;
        }
    }
    return bits;
}


Dictionary getPredefinedDictionary(PredefinedDictionaryType name) {
    // The maximum number of bits that can be corrected is theoretically (d-1)/2,
    // where d is the minimum Hamming distance between any two codes in the dictionary.
    // However, we use a more conservative limit (d/2)-1 to reduce the probability
    // of false positives during detection. This formula is equivalent to the
    // theoretical limit for even distances and stricter for odd distances.

    // DictionaryData constructors calls
    //    moved out of globals so construted on first use, which allows lazy-loading of opencv dll
    static const Dictionary DICT_ARUCO_DATA = Dictionary(Mat(1024, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_ARUCO_BYTES), 5, (3-1)/2);

    static const Dictionary DICT_4X4_50_DATA = Dictionary(Mat(50, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (4-1)/2);
    static const Dictionary DICT_4X4_100_DATA = Dictionary(Mat(100, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (3-1)/2);
    static const Dictionary DICT_4X4_250_DATA = Dictionary(Mat(250, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (3-1)/2);
    static const Dictionary DICT_4X4_1000_DATA = Dictionary(Mat(1000, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (2-1)/2);

    static const Dictionary DICT_5X5_50_DATA = Dictionary(Mat(50, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (8-1)/2);
    static const Dictionary DICT_5X5_100_DATA = Dictionary(Mat(100, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (7-1)/2);
    static const Dictionary DICT_5X5_250_DATA = Dictionary(Mat(250, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (6-1)/2);
    static const Dictionary DICT_5X5_1000_DATA = Dictionary(Mat(1000, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (5-1)/2);

    static const Dictionary DICT_6X6_50_DATA = Dictionary(Mat(50, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (13-1)/2);
    static const Dictionary DICT_6X6_100_DATA = Dictionary(Mat(100, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (12-1)/2);
    static const Dictionary DICT_6X6_250_DATA = Dictionary(Mat(250, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (11-1)/2);
    static const Dictionary DICT_6X6_1000_DATA = Dictionary(Mat(1000, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (9-1)/2);

    static const Dictionary DICT_7X7_50_DATA = Dictionary(Mat(50, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (19-1)/2);
    static const Dictionary DICT_7X7_100_DATA = Dictionary(Mat(100, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (18-1)/2);
    static const Dictionary DICT_7X7_250_DATA = Dictionary(Mat(250, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (17-1)/2);
    static const Dictionary DICT_7X7_1000_DATA = Dictionary(Mat(1000, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (14-1)/2);

    static const Dictionary DICT_APRILTAG_16h5_DATA = Dictionary(Mat(30, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_16h5_BYTES), 4, (5-1)/2);
    static const Dictionary DICT_APRILTAG_25h9_DATA = Dictionary(Mat(35, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_25h9_BYTES), 5, (9-1)/2);
    static const Dictionary DICT_APRILTAG_36h10_DATA = Dictionary(Mat(2320, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_36h10_BYTES), 6, (10-1)/2);
    static const Dictionary DICT_APRILTAG_36h11_DATA = Dictionary(Mat(587, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_36h11_BYTES), 6, (11-1)/2);

    static const Dictionary DICT_ARUCO_MIP_36h12_DATA = Dictionary(Mat(250, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_ARUCO_MIP_36h12_BYTES), 6, (12-1)/2);

    switch(name) {

    case DICT_ARUCO_ORIGINAL:
        return Dictionary(DICT_ARUCO_DATA);

    case DICT_4X4_50:
        return Dictionary(DICT_4X4_50_DATA);
    case DICT_4X4_100:
        return Dictionary(DICT_4X4_100_DATA);
    case DICT_4X4_250:
        return Dictionary(DICT_4X4_250_DATA);
    case DICT_4X4_1000:
        return Dictionary(DICT_4X4_1000_DATA);

    case DICT_5X5_50:
        return Dictionary(DICT_5X5_50_DATA);
    case DICT_5X5_100:
        return Dictionary(DICT_5X5_100_DATA);
    case DICT_5X5_250:
        return Dictionary(DICT_5X5_250_DATA);
    case DICT_5X5_1000:
        return Dictionary(DICT_5X5_1000_DATA);

    case DICT_6X6_50:
        return Dictionary(DICT_6X6_50_DATA);
    case DICT_6X6_100:
        return Dictionary(DICT_6X6_100_DATA);
    case DICT_6X6_250:
        return Dictionary(DICT_6X6_250_DATA);
    case DICT_6X6_1000:
        return Dictionary(DICT_6X6_1000_DATA);

    case DICT_7X7_50:
        return Dictionary(DICT_7X7_50_DATA);
    case DICT_7X7_100:
        return Dictionary(DICT_7X7_100_DATA);
    case DICT_7X7_250:
        return Dictionary(DICT_7X7_250_DATA);
    case DICT_7X7_1000:
        return Dictionary(DICT_7X7_1000_DATA);

    case DICT_APRILTAG_16h5:
        return Dictionary(DICT_APRILTAG_16h5_DATA);
    case DICT_APRILTAG_25h9:
        return Dictionary(DICT_APRILTAG_25h9_DATA);
    case DICT_APRILTAG_36h10:
        return Dictionary(DICT_APRILTAG_36h10_DATA);
    case DICT_APRILTAG_36h11:
        return Dictionary(DICT_APRILTAG_36h11_DATA);

    case DICT_ARUCO_MIP_36h12:
        return Dictionary(DICT_ARUCO_MIP_36h12_DATA);
    }
    return Dictionary(DICT_4X4_50_DATA);
}


Dictionary getPredefinedDictionary(int dict) {
    return getPredefinedDictionary(PredefinedDictionaryType(dict));
}


/**
 * @brief Generates a random marker Mat of size markerSize x markerSize
 */
static Mat _generateRandomMarker(int markerSize, RNG &rng) {
    Mat marker(markerSize, markerSize, CV_8UC1, Scalar::all(0));
    for(int i = 0; i < markerSize; i++) {
        for(int j = 0; j < markerSize; j++) {
            unsigned char bit = (unsigned char) (rng.uniform(0,2));
            marker.at<unsigned char>(i, j) = bit;
        }
    }
    return marker;
}


/**
 * @brief Calculate selfDistance of the codification of a marker Mat. Self distance is the Hamming
 * distance of the marker to itself in the other rotations.
 * See S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
 * "Automatic generation and detection of highly reliable fiducial markers under occlusion".
 * Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005
 */
static int _getSelfDistance(const Mat &marker) {
    Mat bytes = Dictionary::getByteListFromBits(marker);
    int minHamming = (int)marker.total() + 1;
    for(int r = 1; r < 4; r++) {
        int currentHamming = cv::hal::normHamming(bytes.ptr(), bytes.ptr() + bytes.cols*r, bytes.cols);
        if(currentHamming < minHamming) minHamming = currentHamming;
    }
    return minHamming;
}


Dictionary extendDictionary(int nMarkers, int markerSize, const Dictionary &baseDictionary, int randomSeed) {
    CV_Assert(nMarkers > 0);
    RNG rng((uint64)(randomSeed));

    Dictionary out = Dictionary(Mat(), markerSize);
    out.markerSize = markerSize;

    // theoretical maximum intermarker distance
    // See S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
    // "Automatic generation and detection of highly reliable fiducial markers under occlusion".
    // Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005
    int C = (int)std::floor(float(markerSize * markerSize) / 4.f);
    int tau = 2 * (int)std::floor(float(C) * 4.f / 3.f);

    // if baseDictionary is provided, calculate its intermarker distance
    if(baseDictionary.bytesList.rows > 0) {
        CV_Assert(baseDictionary.markerSize == markerSize);
        out.bytesList = baseDictionary.bytesList.rowRange(0, min(nMarkers, baseDictionary.bytesList.rows)).clone();

        int minDistance = markerSize * markerSize + 1;
        for(int i = 0; i < out.bytesList.rows; i++) {
            Mat markerBytes = out.bytesList.rowRange(i, i + 1);
            Mat markerBits = Dictionary::getBitsFromByteList(markerBytes, markerSize);
            minDistance = min(minDistance, _getSelfDistance(markerBits));
            for(int j = i + 1; j < out.bytesList.rows; j++) {
                minDistance = min(minDistance, out.getDistanceToId(markerBits, j));
            }
        }
        tau = minDistance;
    }

    // current best option
    int bestTau = 0;
    Mat bestMarker;

    // after these number of unproductive iterations, the best option is accepted
    const int maxUnproductiveIterations = 5000;
    int unproductiveIterations = 0;

    while(out.bytesList.rows < nMarkers) {
        Mat currentMarker = _generateRandomMarker(markerSize, rng);

        int selfDistance = _getSelfDistance(currentMarker);
        int minDistance = selfDistance;

        // if self distance is better or equal than current best option, calculate distance
        // to previous accepted markers
        if(selfDistance >= bestTau) {
            for(int i = 0; i < out.bytesList.rows; i++) {
                int currentDistance = out.getDistanceToId(currentMarker, i);
                minDistance = min(currentDistance, minDistance);
                if(minDistance <= bestTau) {
                    break;
                }
            }
        }

        // if distance is high enough, accept the marker
        if(minDistance >= tau) {
            unproductiveIterations = 0;
            bestTau = 0;
            Mat bytes = Dictionary::getByteListFromBits(currentMarker);
            out.bytesList.push_back(bytes);
        } else {
            unproductiveIterations++;

            // if distance is not enough, but is better than the current best option
            if(minDistance > bestTau) {
                bestTau = minDistance;
                bestMarker = currentMarker;
            }

            // if number of unproductive iterarions has been reached, accept the current best option
            if(unproductiveIterations == maxUnproductiveIterations) {
                unproductiveIterations = 0;
                tau = bestTau;
                bestTau = 0;
                Mat bytes = Dictionary::getByteListFromBits(bestMarker);
                out.bytesList.push_back(bytes);
            }
        }
    }

    // update the maximum number of correction bits for the generated dictionary
    out.maxCorrectionBits = (tau - 1) / 2;

    return out;
}

}
}
