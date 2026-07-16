// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "opencv2/core/hal/hal.hpp"

#include "aruco_utils.hpp"
#include "predefined_dictionaries.hpp"
#include "predefined_dictionaries_nested.hpp"
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

// Equivalent of CellBitMasks for dictionaries with DICT_ENCODING_CELL_RATIO encoding.
// A cell counts as an error when |observedRatio - expectedRatio| > validBitIdThreshold.
// As the expected ratios are integers in [0,100], the test is precomputed once per candidate as an
// inclusive [lower, upper] range of valid expected values per cell:
//   |observed - expected/100| <= threshold  <=>  ceil(100*(observed - threshold)) <= expected &&
//                                                expected <= floor(100*(observed + threshold))
// Each rotation is then a branch-free integer pass over markerSize*markerSize bytes.
struct CellRatioDistance {
    CellRatioDistance(const Mat &onlyCellPixelRatio, int markerSize, float validBitIdThreshold)
        : totalCells(markerSize * markerSize),
          bounds(2 * totalCells),
          lower(bounds.data()), width(bounds.data() + totalCells) {
        uint8_t* lowerWritable = bounds.data();
        uint8_t* widthWritable = lowerWritable + totalCells;

        const float threshold = 100.f * validBitIdThreshold;
        int cell = 0;
        for(int j = 0; j < markerSize; j++) {
            const float* cellPixelRatioRow = onlyCellPixelRatio.ptr<float>(j);
            for(int i = 0; i < markerSize; i++, cell++) {
                const float observed = 100.f * cellPixelRatioRow[i];
                const int lo = saturate_cast<uint8_t>(std::ceil(observed - threshold));
                const int hi = saturate_cast<uint8_t>(std::floor(observed + threshold));
                lowerWritable[cell] = static_cast<uint8_t>(lo);
                widthWritable[cell] = static_cast<uint8_t>(hi - lo); // hi >= lo
            }
        }
    }

    // Smallest number of erroneous cells between the candidate and dictionary marker `id`,
    // searching the tested rotations; `rotation` returns the best one.
    int distanceToId(const Mat& bytesList, int id, bool allRotations, int& rotation) const {
        CV_Assert(id >= 0 && id < bytesList.rows);

        const unsigned int nRotations = allRotations ? 4u : 1u;
        int currentMinDistance = totalCells + 1;
        rotation = -1;

        const uchar* expectedRatios = bytesList.ptr(id);
        for(unsigned int r = 0; r < nRotations; r++, expectedRatios += totalCells) {
            // A cell is an error when the expected ratio is outside [lower, lower+width]. The
            // unsigned wrap turns the two-sided range test into a single compare (all values are in [0,255]):
            // values below 'lower' underflow to numbers (>=156), which exceeds the max width (<=100).
            int currentDistance = 0;
            for(int i = 0; i < totalCells; i++)
                currentDistance += static_cast<uint8_t>(expectedRatios[i] - lower[i]) > width[i];

            if(currentDistance < currentMinDistance) {
                currentMinDistance = currentDistance;
                rotation = static_cast<int>(r);
                // Break for perfect distance.
                if(currentMinDistance == 0) break;
            }
        }

        return currentMinDistance;
    }

    const int totalCells;
    std::vector<uint8_t> bounds;   // [lower | width]: cell ok when expected in [lower, lower+width]
    const uint8_t *lower, *width;
};

Dictionary::Dictionary(): markerSize(0), maxCorrectionBits(0), dictEncoding(DICT_ENCODING_BINARY) {}


Dictionary::Dictionary(const Mat &_bytesList, int _markerSize, int _maxcorr, int _dictEncoding) {
    CV_Assert(_dictEncoding == DICT_ENCODING_BINARY || _dictEncoding == DICT_ENCODING_CELL_RATIO);
    if(_dictEncoding == DICT_ENCODING_CELL_RATIO && !_bytesList.empty())
        CV_Assert(_bytesList.cols * _bytesList.channels() == 4 * _markerSize * _markerSize);
    markerSize = _markerSize;
    maxCorrectionBits = _maxcorr;
    bytesList = _bytesList;
    dictEncoding = _dictEncoding;
}


bool Dictionary::readDictionary(const cv::FileNode& fn) {
    int nMarkers = 0, _markerSize = 0;
    if (fn.empty() || !readParameter("nmarkers", nMarkers, fn) || !readParameter("markersize", _markerSize, fn))
        return false;
    int _dictEncoding = (int)DICT_ENCODING_BINARY;
    readParameter("dictEncoding", _dictEncoding, fn);
    int _maxCorrectionBits = 0;
    readParameter("maxCorrectionBits", _maxCorrectionBits, fn);

    if (_dictEncoding == DICT_ENCODING_CELL_RATIO) {
        // markers are stored as the list of their cell ratios in percent, only the 0 deg rotation
        Mat ratioList(0, 0, CV_8UC4), cellRatios(_markerSize, _markerSize, CV_8UC1);
        std::vector<int> markerValues;
        for (int i = 0; i < nMarkers; i++) {
            std::ostringstream ostr;
            ostr << i;
            if (!readParameter("marker_" + ostr.str(), markerValues, fn))
                return false;
            if (markerValues.size() != (size_t)(_markerSize * _markerSize))
                return false;
            for (int j = 0; j < (int)markerValues.size(); j++) {
                if (markerValues[j] < 0 || markerValues[j] > 100)
                    return false;
                cellRatios.at<unsigned char>(j) = (unsigned char)markerValues[j];
            }
            ratioList.push_back(Dictionary::getRatioListFromCellRatios(cellRatios));
        }
        *this = Dictionary(ratioList, _markerSize, _maxCorrectionBits, DICT_ENCODING_CELL_RATIO);
        return true;
    }

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
    if (dictEncoding == DICT_ENCODING_CELL_RATIO)
        fs << "dictEncoding" << dictEncoding;
    for (int i = 0; i < bytesList.rows; i++) {
        Mat row = bytesList.row(i);
        std::ostringstream ostr;
        ostr << i;
        string markerName = "marker_" + ostr.str();
        if (dictEncoding == DICT_ENCODING_CELL_RATIO) {
            Mat cellRatios = getCellRatiosFromRatioList(row, markerSize);
            std::vector<int> markerValues(markerSize * markerSize);
            for (int j = 0; j < (int)markerValues.size(); j++)
                markerValues[j] = cellRatios.at<uint8_t>(j);
            fs << markerName << markerValues;
        } else {
            Mat bitMarker = getBitsFromByteList(row, markerSize);
            string marker;
            for (int j = 0; j < markerSize * markerSize; j++)
                marker.push_back(bitMarker.at<uint8_t>(j) + '0');
            fs << markerName << marker;
        }
    }

    if (!name.empty())
        fs << "}";
}


bool Dictionary::identify(const Mat &onlyCellPixelRatio, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate, float validBitIdThreshold) const {
    CV_Assert(onlyCellPixelRatio.rows == markerSize && onlyCellPixelRatio.cols == markerSize);
    CV_Assert(onlyCellPixelRatio.type() == CV_32FC1);

    int maxCorrectionRecalculed = int(double(maxCorrectionBits) * maxCorrectionRate);

    idx = -1; // by default, not found

    // search closest marker in dict
    if(dictEncoding == DICT_ENCODING_CELL_RATIO) {
        CellRatioDistance cellRatioDistance(onlyCellPixelRatio, markerSize, validBitIdThreshold);
        for(int m = 0; m < bytesList.rows; m++) {
            int currentRotation = -1;
            int currentMinDistance = cellRatioDistance.distanceToId(bytesList, m, true, currentRotation);

            // if maxCorrection is fulfilled, return this one
            if(currentMinDistance <= maxCorrectionRecalculed) {
                idx = m;
                rotation = currentRotation;
                break;
            }
        }
    } else {
        CellBitMasks cellBitMasks(onlyCellPixelRatio, markerSize, validBitIdThreshold);
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

    if(dictEncoding == DICT_ENCODING_CELL_RATIO) {
        Mat candidateBitRatio;
        Mat(bits.getMat() > 0).convertTo(candidateBitRatio, CV_32F, 1.0 / 255.0);
        return getDistanceToId(candidateBitRatio, id, allRotations, DEFAULT_VALID_BIT_ID_THRESHOLD);
    }

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
    if(dictEncoding == DICT_ENCODING_CELL_RATIO) {
        CellRatioDistance cellRatioDistance(onlyCellPixelRatioMat, markerSize, validBitIdThreshold);
        return cellRatioDistance.distanceToId(bytesList, id, allRotations, rotation);
    }
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

    CV_Assert(markerId >= 0 && markerId < bytesList.rows);
    CV_Assert(rotationId >= 0 && rotationId < 4);

    Mat bits(markerSize, markerSize, CV_32F, Scalar::all(0));
    if(dictEncoding == DICT_ENCODING_CELL_RATIO) {
        Mat cellRatios = getCellRatiosFromRatioList(bytesList.rowRange(markerId, markerId + 1), markerSize, rotationId);
        cellRatios.convertTo(bits, CV_32F, 1.0 / 100.0);
    } else {
        Mat bitsUints = getBitsFromByteList(bytesList.rowRange(markerId, markerId + 1), markerSize, rotationId);
        bitsUints.convertTo(bits, CV_32F);
    }

    CV_Assert(bits.rows == markerSize && bits.cols == markerSize);
    return bits;
}


Mat Dictionary::getRatioListFromCellRatios(const Mat &cellRatios) {
    CV_Assert(cellRatios.type() == CV_8UC1);
    CV_Assert(cellRatios.rows == cellRatios.cols && cellRatios.rows > 0);

    const int markerSize = cellRatios.rows;
    const int totalCells = markerSize * markerSize;

    // the 4 rotations, stored with the same order as in getByteListFromBits()
    Mat ratioList(1, totalCells, CV_8UC4);
    uchar* rot0 = ratioList.ptr();
    uchar* rot1 = rot0 + totalCells;
    uchar* rot2 = rot1 + totalCells;
    uchar* rot3 = rot2 + totalCells;

    for(int row = 0; row < markerSize; row++) {
        for(int col = 0; col < markerSize; col++) {
            const int cell = row * markerSize + col;
            rot0[cell] = cellRatios.at<uchar>(row, col);
            rot1[cell] = cellRatios.at<uchar>(col, markerSize - 1 - row);
            rot2[cell] = cellRatios.at<uchar>(markerSize - 1 - row, markerSize - 1 - col);
            rot3[cell] = cellRatios.at<uchar>(markerSize - 1 - col, row);
        }
    }
    return ratioList;
}


Mat Dictionary::getCellRatiosFromRatioList(const Mat &ratioList, int markerSize, int rotationId) {
    const int totalCells = markerSize * markerSize;
    CV_Assert(markerSize > 0 && ratioList.rows == 1);
    CV_Assert(ratioList.total() * ratioList.channels() == (size_t)(4 * totalCells));
    CV_Assert(rotationId >= 0 && rotationId < 4);

    return Mat(markerSize, markerSize, CV_8UC1, (void*)(ratioList.ptr() + rotationId * totalCells)).clone();
}


Mat Dictionary::getCellRatiosFromImage(InputArray markerImage, int markerSize, int borderBits) {
    Mat img = markerImage.getMat();
    CV_Assert(!img.empty() && img.channels() == 1 && img.depth() == CV_8U);
    CV_Assert(markerSize > 0 && borderBits > 0);

    const int totalCells = markerSize + 2 * borderBits;
    CV_Assert(img.rows == img.cols && img.rows >= totalCells);

    Mat cellRatios(markerSize, markerSize, CV_8UC1);
    for(int y = 0; y < markerSize; y++) {
        const int y0 = (borderBits + y) * img.rows / totalCells;
        const int y1 = (borderBits + y + 1) * img.rows / totalCells;
        for(int x = 0; x < markerSize; x++) {
            const int x0 = (borderBits + x) * img.cols / totalCells;
            const int x1 = (borderBits + x + 1) * img.cols / totalCells;
            Mat cell = img(Range(y0, y1), Range(x0, x1));
            const int white = countNonZero(cell > 127);
            cellRatios.at<uchar>(y, x) = (uchar)cvRound(100.0 * white / (double)cell.total());
        }
    }
    return cellRatios;
}


Dictionary Dictionary::convertToCellRatioDictionary() const {
    CV_Assert(dictEncoding == DICT_ENCODING_BINARY);

    Mat ratioList(0, 0, CV_8UC4), cellRatios;
    for(int id = 0; id < bytesList.rows; id++) {
        getMarkerBits(id).convertTo(cellRatios, CV_8U, 100.0);
        ratioList.push_back(getRatioListFromCellRatios(cellRatios));
    }

    return Dictionary(ratioList, markerSize, maxCorrectionBits, DICT_ENCODING_CELL_RATIO);
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

    // nested marker pair dictionaries store one byte per cell (DICT_ENCODING_CELL_RATIO)
    static const Dictionary DICT_4X4_NESTED_5_DATA = Dictionary(Mat(10, 4 * 4, CV_8UC4, (uchar*)DICT_4X4_NESTED_5_BYTES), 4, (4-1)/2, DICT_ENCODING_CELL_RATIO);
    static const Dictionary DICT_4X4_NESTED_10_DATA = Dictionary(Mat(20, 4 * 4, CV_8UC4, (uchar*)DICT_4X4_NESTED_10_BYTES), 4, (3-1)/2, DICT_ENCODING_CELL_RATIO);
    static const Dictionary DICT_4X4_NESTED_24_DATA = Dictionary(Mat(48, 4 * 4, CV_8UC4, (uchar*)DICT_4X4_NESTED_24_BYTES), 4, (2-1)/2, DICT_ENCODING_CELL_RATIO);

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

    case DICT_4X4_NESTED_5:
        return Dictionary(DICT_4X4_NESTED_5_DATA);
    case DICT_4X4_NESTED_10:
        return Dictionary(DICT_4X4_NESTED_10_DATA);
    case DICT_4X4_NESTED_24:
        return Dictionary(DICT_4X4_NESTED_24_DATA);
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
        // random marker generation is only implemented for binary dictionaries
        CV_Assert(baseDictionary.dictEncoding == DICT_ENCODING_BINARY);
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


// A candidate cell matches a dictionary entry when |observedRatio - expectedRatio| is at most
// validBitIdThreshold. Two entries A and B are "separating" at a cell when their expected
// ratios differ by more than 2 * validBitIdThreshold: an observation matching A is
// always more than validBitIdThreshold away from B, so no observation can match both entries
// at that cell. Ratios are stored in percent, hence the factor 200.
static const int SEPARATING_PERCENT = (int)(200 * DEFAULT_VALID_BIT_ID_THRESHOLD);

static int _countSeparatingCells(const uchar* a, const uchar* b, int totalCells) {
    int count = 0;
    for(int i = 0; i < totalCells; i++)
        if(std::abs((int)a[i] - (int)b[i]) > SEPARATING_PERCENT) count++;
    return count;
}

/**
 * @brief Separation distance between a candidate (rotation 0) and an entry, minimum over the 4
 * relative rotations. cellRatios is markerSize x markerSize CV_8UC1 in percent, ratioList one
 * bytesList row as built by Dictionary::getRatioListFromCellRatios()
 */
static int _ratioDistance(const Mat &cellRatios, const Mat &ratioList) {
    const int totalCells = (int)cellRatios.total();
    Mat candidate = cellRatios.isContinuous() ? cellRatios : cellRatios.clone();
    int minDist = totalCells + 1;
    for(int r = 0; r < 4; r++)
        minDist = min(minDist, _countSeparatingCells(candidate.ptr(), ratioList.ptr() + r * totalCells, totalCells));
    return minDist;
}

static int _ratioSelfDistance(const Mat &ratioList, int totalCells) {
    int minDist = totalCells + 1;
    for(int r = 1; r < 4; r++)
        minDist = min(minDist, _countSeparatingCells(ratioList.ptr(), ratioList.ptr() + r * totalCells, totalCells));
    return minDist;
}

/**
 * @brief Find the uniform 2x2 cell block of a binary marker. Returns true only when the marker
 * has exactly one uniform block and it is white, the layout rule of nested marker pairs.
 */
static bool _findSingleWhiteBlock(const Mat &bits, Point &block) {
    int count = 0;
    bool white = false;
    for(int y = 0; y + 1 < bits.rows; y++) {
        for(int x = 0; x + 1 < bits.cols; x++) {
            const uchar v = bits.at<uchar>(y, x);
            if(bits.at<uchar>(y, x + 1) == v && bits.at<uchar>(y + 1, x) == v &&
               bits.at<uchar>(y + 1, x + 1) == v) {
                count++;
                block = Point(x, y);
                white = v != 0;
            }
        }
    }
    return count == 1 && white;
}

/**
 * @brief Expected cell ratios in percent of an outer marker hosting its rotated inner marker.
 * The 4 host cells are rasterized at a fixed resolution with integer arithmetic only, so the
 * ratios are identical on every platform. innerHalfDiagonal is the half diagonal of the inner
 * marker square in outer cell units. The diagonals align with the outer cell grid because the
 * inner marker was rotated 45 degrees.
 */
static Mat _nestedCellRatios(const Mat &outerBits, const Mat &innerBits, Point block,
                             float innerHalfDiagonal) {
    const int K = 120;                                   // rasterization pixels per cell
    const int R2 = 2 * cvRound(innerHalfDiagonal * K);   // doubled half diagonal in pixels
    const int innerTotal = innerBits.rows + 2;    // inner marker plus its 1 cell border
    const bool hostWhite = outerBits.at<uchar>(block.y, block.x) != 0;

    Mat ratios(outerBits.rows, outerBits.cols, CV_8UC1);
    for(int y = 0; y < outerBits.rows; y++)
        for(int x = 0; x < outerBits.cols; x++)
            ratios.at<uchar>(y, x) = outerBits.at<uchar>(y, x) ? 100 : 0;

    int white[2][2] = {{0, 0}, {0, 0}};
    for(int y = 0; y < 2 * K; y++) {
        for(int x = 0; x < 2 * K; x++) {
            // doubled pixel center coordinates relative to the block center
            const int du = 2 * (x - K) + 1;
            const int dv = 2 * (y - K) + 1;
            bool isWhite = hostWhite;
            if(std::abs(du) + std::abs(dv) <= R2) {
                // rotate 45 degrees into the inner marker frame; the sqrt(2) factors cancel
                const int col = min(innerTotal * (du - dv + R2) / (2 * R2), innerTotal - 1);
                const int row = min(innerTotal * (du + dv + R2) / (2 * R2), innerTotal - 1);
                const bool border = row == 0 || col == 0 || row == innerTotal - 1 || col == innerTotal - 1;
                isWhite = !border && innerBits.at<uchar>(row - 1, col - 1) != 0;
                if(!hostWhite) isWhite = !isWhite;  // inverted inner marker in a black block
            }
            if(isWhite) white[y / K][x / K]++;
        }
    }
    for(int cy = 0; cy < 2; cy++)
        for(int cx = 0; cx < 2; cx++)
            ratios.at<uchar>(block.y + cy, block.x + cx) = (uchar)((200 * white[cy][cx] + K * K) / (2 * K * K));
    return ratios;
}

/**
 * @brief Locate the host block of an outer entry: the 2x2 group of cells with a non binary
 * expected ratio. Returns the cell ratios of the entry through `ratios`.
 */
static Point _findHostBlock(const Dictionary &dictionary, int outerId, Mat &ratios) {
    const int markerSize = dictionary.markerSize;
    ratios = Dictionary::getCellRatiosFromRatioList(
        dictionary.bytesList.rowRange(outerId, outerId + 1), markerSize);

    Point block(markerSize, markerSize);
    int bandCells = 0;
    for(int y = 0; y < markerSize; y++) {
        for(int x = 0; x < markerSize; x++) {
            const uchar v = ratios.at<uchar>(y, x);
            if(v != 0 && v != 100) {
                bandCells++;
                block.x = min(block.x, x);
                block.y = min(block.y, y);
            }
        }
    }
    bool valid = bandCells == 4 && block.x + 1 < markerSize && block.y + 1 < markerSize;
    for(int cy = 0; valid && cy < 2; cy++)
        for(int cx = 0; valid && cx < 2; cx++) {
            const uchar v = ratios.at<uchar>(block.y + cy, block.x + cx);
            valid = v != 0 && v != 100;
        }
    if(!valid)
        CV_Error(Error::StsBadArg, "the dictionary entry is not the outer marker of a nested pair");
    return block;
}


Dictionary generateNestedDictionary(int nPairs, int markerSize, int minDistance,
                                    float innerHalfDiagonal, int randomSeed) {
    CV_Assert(nPairs > 0 && markerSize >= 3 && minDistance >= 1);
    // the inner marker covers a triangle of area innerHalfDiagonal^2 / 2 in each host cell.
    // upper bound sqrt(0.5): that triangle may cover at most a quarter of the cell, so host
    // ratios stay within 25 percent of their color.
    // lower bound 0.2: keep the host ratios non binary after rounding to percent
    CV_Assert(innerHalfDiagonal >= 0.2f &&
              innerHalfDiagonal * innerHalfDiagonal <= 0.5f + FLT_EPSILON);

    RNG rng((uint64)randomSeed);
    const int totalCells = markerSize * markerSize;
    const int minColorCells = 4;
    const int maxAttempts = 2000000;

    Mat bytesList(0, totalCells, CV_8UC4);
    std::vector<Mat> accepted;  // ratio list of every accepted entry, host cells as 50 percent
    int attempts = 0;
    while(bytesList.rows < 2 * nPairs) {
        if(++attempts > maxAttempts)
            CV_Error(Error::StsError,
                     "generateNestedDictionary: cannot generate the requested number of pairs, "
                     "reduce nPairs or minDistance");

        Mat outer = _generateRandomMarker(markerSize, rng);
        Point block;
        if(!_findSingleWhiteBlock(outer, block)) continue;
        int whiteCells = countNonZero(outer);
        if(min(whiteCells, totalCells - whiteCells) < minColorCells) continue;

        // search on a proxy where the 4 host cells are set to 50 percent. The exact host
        // ratios are not known before rasterization, but any of their possible values keeps
        // the cell at least as separating as 50 percent does, so distances measured on the
        // proxy are a lower bound of the exact ones and the minDistance guarantee holds
        Mat outerProxy(markerSize, markerSize, CV_8UC1);
        for(int y = 0; y < markerSize; y++)
            for(int x = 0; x < markerSize; x++)
                outerProxy.at<uchar>(y, x) = outer.at<uchar>(y, x) ? 100 : 0;
        outerProxy(Rect(block.x, block.y, 2, 2)).setTo(50);
        Mat outerProxyList = Dictionary::getRatioListFromCellRatios(outerProxy);
        if(_ratioSelfDistance(outerProxyList, totalCells) < minDistance) continue;

        Mat inner = _generateRandomMarker(markerSize, rng);
        whiteCells = countNonZero(inner);
        if(min(whiteCells, totalCells - whiteCells) < minColorCells) continue;
        Mat innerRatios;
        inner.convertTo(innerRatios, CV_8U, 100.0);
        Mat innerList = Dictionary::getRatioListFromCellRatios(innerRatios);
        if(_ratioSelfDistance(innerList, totalCells) < minDistance) continue;
        if(_ratioDistance(outerProxy, innerList) < minDistance) continue;

        bool farEnough = true;
        for(size_t i = 0; farEnough && i < accepted.size(); i++)
            farEnough = _ratioDistance(outerProxy, accepted[i]) >= minDistance &&
                        _ratioDistance(innerRatios, accepted[i]) >= minDistance;
        if(!farEnough) continue;

        bytesList.push_back(Dictionary::getRatioListFromCellRatios(
            _nestedCellRatios(outer, inner, block, innerHalfDiagonal)));
        bytesList.push_back(innerList);
        accepted.push_back(outerProxyList);
        accepted.push_back(innerList);
    }
    return Dictionary(bytesList, markerSize, (minDistance - 1) / 2, DICT_ENCODING_CELL_RATIO);
}


void generateImageMarkerNested(const Dictionary &dictionary, int outerId, int sidePixels,
                               OutputArray _img, int borderBits, float innerHalfDiagonal) {
    CV_Assert(dictionary.dictEncoding == DICT_ENCODING_CELL_RATIO);
    CV_Assert(outerId >= 0 && outerId % 2 == 0 && outerId + 1 < dictionary.bytesList.rows);
    CV_Assert(borderBits > 0);
    CV_Assert(innerHalfDiagonal >= 0.2f &&
              innerHalfDiagonal * innerHalfDiagonal <= 0.5f + FLT_EPSILON);

    const int markerSize = dictionary.markerSize;
    const int totalCells = markerSize + 2 * borderBits;
    CV_Assert(sidePixels >= totalCells);

    Mat outerRatios;
    const Point block = _findHostBlock(dictionary, outerId, outerRatios);
    const bool hostWhite = outerRatios.at<uchar>(block.y, block.x) >= 50;
    Mat innerRatios = Dictionary::getCellRatiosFromRatioList(
        dictionary.bytesList.rowRange(outerId + 1, outerId + 2), markerSize);

    const double cellPixels = sidePixels / (double)totalCells;
    const double centerX = (borderBits + block.x + 1) * cellPixels;
    const double centerY = (borderBits + block.y + 1) * cellPixels;
    const double halfDiag = innerHalfDiagonal * cellPixels;
    const int innerTotal = markerSize + 2;  // inner marker plus its 1 cell border

    _img.create(sidePixels, sidePixels, CV_8UC1);
    Mat img = _img.getMat();
    for(int y = 0; y < sidePixels; y++) {
        for(int x = 0; x < sidePixels; x++) {
            const int cellX = x * totalCells / sidePixels;
            const int cellY = y * totalCells / sidePixels;
            bool isWhite = false;
            if(cellX >= borderBits && cellX < totalCells - borderBits &&
               cellY >= borderBits && cellY < totalCells - borderBits)
                isWhite = outerRatios.at<uchar>(cellY - borderBits, cellX - borderBits) >= 50;

            const double u = x + 0.5 - centerX;
            const double v = y + 0.5 - centerY;
            if(std::abs(u) + std::abs(v) <= halfDiag) {
                // rotate 45 degrees into the inner marker frame; the sqrt(2) factors cancel
                const int col = min((int)(innerTotal * (u - v + halfDiag) / (2 * halfDiag)), innerTotal - 1);
                const int row = min((int)(innerTotal * (u + v + halfDiag) / (2 * halfDiag)), innerTotal - 1);
                const bool border = row == 0 || col == 0 || row == innerTotal - 1 || col == innerTotal - 1;
                isWhite = !border && innerRatios.at<uchar>(row - 1, col - 1) >= 50;
                if(!hostWhite) isWhite = !isWhite;
            }
            img.at<uchar>(y, x) = isWhite ? 255 : 0;
        }
    }
}


void getNestedMarkerObjectPoints(const Dictionary &dictionary, int outerId, float outerSideLength,
                                 OutputArray outerCorners, OutputArray innerCorners, int borderBits,
                                 float innerHalfDiagonal) {
    CV_Assert(dictionary.dictEncoding == DICT_ENCODING_CELL_RATIO);
    CV_Assert(outerId >= 0 && outerId % 2 == 0 && outerId + 1 < dictionary.bytesList.rows);
    CV_Assert(outerSideLength > 0.f && borderBits > 0);

    Mat outerRatios;
    const Point block = _findHostBlock(dictionary, outerId, outerRatios);

    const float side = outerSideLength;
    Mat(Matx43f(0.f, 0.f, 0.f,
                side, 0.f, 0.f,
                side, side, 0.f,
                0.f, side, 0.f)).copyTo(outerCorners);

    // the inner marker is centered on the corner shared by the 4 host cells; its canonical
    // corners map to the left, top, right and bottom vertices of the rotated square
    const float cell = side / (float)(dictionary.markerSize + 2 * borderBits);
    const float cx = (borderBits + block.x + 1) * cell;
    const float cy = (borderBits + block.y + 1) * cell;
    const float r = innerHalfDiagonal * cell;
    Mat(Matx43f(cx - r, cy, 0.f,
                cx, cy - r, 0.f,
                cx + r, cy, 0.f,
                cx, cy + r, 0.f)).copyTo(innerCorners);
}

}
}
