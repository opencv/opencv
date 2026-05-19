// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "../precomp.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/imgproc.hpp"
#include "aruco2_dictionary.hpp"
#include "aruco2_predefined_dictionaries.hpp"
#include "aruco2_predefined_dictionaries_apriltag.hpp"
namespace cv {
namespace aruco2 {

using namespace std;


template<typename T>
inline bool readParameter(const std::string& name, T& parameter, const FileNode& node)
{
    if (!node.empty() && !node[name].empty()) {
        node[name] >> parameter;
        return true;
    }
    return false;
}

template<typename T>
inline bool readWriteParameter(const std::string& name, T& parameter, const FileNode* readNode, FileStorage* writeStorage)
{
    if (readNode)
        return readParameter(name, parameter, *readNode);
    CV_Assert(writeStorage);
    *writeStorage << name << parameter;
    return true;
}


Dictionary::Dictionary(): markerSize(0), maxCorrectionBits(0) {}


Dictionary::Dictionary(const Mat &_bytesList, int _markerSize, int _maxcorr) {
    markerSize = _markerSize;
    maxCorrectionBits = _maxcorr;
    bytesList = _bytesList;

    if (bytesList.empty() || markerSize == 0) return;

    // nbytes is the number of bytes for each rotation
    int nbytes = bytesList.cols; // Since bytesList is CV_8UC4, cols == nbytes

    for (int m = 0; m < bytesList.rows; m++) {
        for (int r = 0; r < 4; r++) {
            // skip nbytes for each rotation r
            const uchar* bytesPtr = bytesList.ptr(m) + r * nbytes;
            uint64_t val = 0;

            // we can only pack into uint64_t if nbytes <= 8 (up to 8x8 bits)
            if (nbytes <= 8) {
                for (int j = 0; j < nbytes; j++) {
                    val <<= 8;
                    val |= bytesPtr[j];
                }
                // save the rotation in the map pointing to the marker ID and rotation
                bits_id[val] = std::make_pair(m, r);
            }
        }
    }
}
size_t Dictionary::size()const{
    return bits_id.size();
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


bool Dictionary::identify(const Mat &onlyBits, CV_OUT int &idx, CV_OUT int &rotation, double maxCorrectionRate) const {
    CV_Assert(onlyBits.rows == markerSize && onlyBits.cols == markerSize);

    int maxCorrectionRecalculed = int(double(maxCorrectionBits) * maxCorrectionRate);

    //slow path, when maxCorrectionRate > 0, we need to compute hamming distance for all markers in the dictionary
    if(maxCorrectionRecalculed>0){
        Mat candidateBytes = getByteListFromBits(onlyBits);
        int nbytes = candidateBytes.cols;
        const uchar* candPtr = candidateBytes.ptr();
        idx = -1; // by default, not found

        // search closest marker in dict
        for(int m = 0; m < bytesList.rows; m++) {
            int currentMinDistance = markerSize * markerSize + 1;
            int currentRotation = -1;
            const uchar* dictPtr = bytesList.ptr(m);
            for(int r = 0; r < 4; r++) {
                int currentHamming = cv::hal::normHamming(dictPtr + r * nbytes, candPtr, nbytes);

                if(currentHamming < currentMinDistance) {
                    currentMinDistance = currentHamming;
                    currentRotation = r;
                }
            }

            // if maxCorrection is fulfilled, return this one
            if(currentMinDistance <= maxCorrectionRecalculed) {
                idx = m;
                rotation = currentRotation;
                break;
            }
        }

        return idx != -1;
    }
    else{//ultra fast path, only exact matches
        Mat candidateBytes = getByteListFromBits(onlyBits);
        int nbytes = candidateBytes.cols;
        const uchar* candPtr = candidateBytes.ptr();
        uint64_t val = 0;

        // we can only pack into uint64_t if nbytes <= 8 (up to 8x8 bits)
        if (nbytes <= 8) {
            for (int j = 0; j < nbytes; j++) {
                val <<= 8;
                val |= candPtr[j];
            }
            auto it = bits_id.find(val);//exact search in time O(logn)
            if (it != bits_id.end()) {
                idx = it->second.first;
                rotation = it->second.second;
                return true;
            }
        }
        return false;

    }
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


Dictionary getPredefinedDictionary(DictionaryType name) {
    // DictionaryData constructors calls
    //    moved out of globals so construted on first use, which allows lazy-loading of opencv dll


    switch(name) {

    case DICT_ARUCO_ORIGINAL:{
        static const Dictionary DICT_ARUCO_DATA = Dictionary(Mat(1024, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_ARUCO_BYTES), 5, (3/2)-1);
        return Dictionary(DICT_ARUCO_DATA);
    }break;

    case DICT_4X4_50:{
        static const Dictionary DICT_4X4_50_DATA = Dictionary(Mat(50, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (4/2)-1);
        return Dictionary(DICT_4X4_50_DATA);
    }break;

    case DICT_4X4_250:{
        static const Dictionary DICT_4X4_250_DATA = Dictionary(Mat(250, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (3/2)-1);
        return Dictionary(DICT_4X4_250_DATA);
    }break;
    case DICT_4X4_1000:{
        static const Dictionary DICT_4X4_1000_DATA = Dictionary(Mat(1000, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (2/2)-1);
        return Dictionary(DICT_4X4_1000_DATA);
    }break;
    case DICT_5X5_50:{
        static const Dictionary DICT_5X5_50_DATA = Dictionary(Mat(50, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (8/2)-1);
        return Dictionary(DICT_5X5_50_DATA);
    }break;
    case DICT_5X5_100:{
        static const Dictionary DICT_5X5_100_DATA = Dictionary(Mat(100, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (7/2)-1);
        return Dictionary(DICT_5X5_100_DATA);
    }break;
    case DICT_5X5_250:{
        static const Dictionary DICT_5X5_250_DATA = Dictionary(Mat(250, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (6/2)-1);
        return Dictionary(DICT_5X5_250_DATA);
    }break;
    case DICT_5X5_1000:{
        static const Dictionary DICT_5X5_1000_DATA = Dictionary(Mat(1000, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_5X5_1000_BYTES), 5, (5/2)-1);
        return Dictionary(DICT_5X5_1000_DATA);
    }break;
    case DICT_6X6_50:{
        static const Dictionary DICT_6X6_50_DATA = Dictionary(Mat(50, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (13/2)-1);
        return Dictionary(DICT_6X6_50_DATA);
    }break;
    case DICT_6X6_100:{
        static const Dictionary DICT_6X6_100_DATA = Dictionary(Mat(100, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (12/2)-1);
        return Dictionary(DICT_6X6_100_DATA);
    }break;
    case DICT_6X6_250:{
        static const Dictionary DICT_6X6_250_DATA = Dictionary(Mat(250, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (11/2)-1);
        return Dictionary(DICT_6X6_250_DATA);
    }break;
    case DICT_6X6_1000:{
        static const Dictionary DICT_6X6_1000_DATA = Dictionary(Mat(1000, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES), 6, (9/2)-1);
        return Dictionary(DICT_6X6_1000_DATA);
    }break;
    case DICT_7X7_50:{
        static const Dictionary DICT_7X7_50_DATA = Dictionary(Mat(50, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (19/2)-1);
        return Dictionary(DICT_7X7_50_DATA);
    }break;
    case DICT_7X7_100:{
        static const Dictionary DICT_7X7_100_DATA = Dictionary(Mat(100, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (18/2)-1);
        return Dictionary(DICT_7X7_100_DATA);
    }break;
    case DICT_7X7_250:{
        static const Dictionary DICT_7X7_250_DATA = Dictionary(Mat(250, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (17/2)-1);
        return Dictionary(DICT_7X7_250_DATA);
    }break;
    case DICT_7X7_1000:{
        static const Dictionary DICT_7X7_1000_DATA = Dictionary(Mat(1000, (7 * 7 + 7) / 8, CV_8UC4, (uchar*)DICT_7X7_1000_BYTES), 7, (14/2)-1);
        return Dictionary(DICT_7X7_1000_DATA);
    }break;
    case DICT_APRILTAG_16h5:{
        static const Dictionary DICT_APRILTAG_16h5_DATA = Dictionary(Mat(30, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_16h5_BYTES), 4, (5/2)-1);
        return Dictionary(DICT_APRILTAG_16h5_DATA);
    }break;
    case DICT_APRILTAG_25h9:{
        static const Dictionary DICT_APRILTAG_25h9_DATA = Dictionary(Mat(35, (5 * 5 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_25h9_BYTES), 5, (9/2)-1);
        return Dictionary(DICT_APRILTAG_25h9_DATA);
    }break;
    case DICT_APRILTAG_36h10:{
        static const Dictionary DICT_APRILTAG_36h10_DATA = Dictionary(Mat(2320, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_36h10_BYTES), 6, (10/2)-1);
        return Dictionary(DICT_APRILTAG_36h10_DATA);
    }break;
    case DICT_APRILTAG_36h11:{
        static const Dictionary DICT_APRILTAG_36h11_DATA = Dictionary(Mat(587, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_APRILTAG_36h11_BYTES), 6, (11/2)-1);
        return Dictionary(DICT_APRILTAG_36h11_DATA);
    }break;
    case DICT_ARUCO_MIP_36h12:{
        static const Dictionary DICT_ARUCO_MIP_36h12_DATA = Dictionary(Mat(250, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_ARUCO_MIP_36h12_BYTES), 6, (12/2)-1);
        return Dictionary(DICT_ARUCO_MIP_36h12_DATA);
    }break;
    default:{
        static const Dictionary DICT_4X4_100_DATA = Dictionary(Mat(100, (4 * 4 + 7) / 8, CV_8UC4, (uchar*)DICT_4X4_1000_BYTES), 4, (3/2)-1);
        return Dictionary(DICT_4X4_100_DATA);
    }
    }
    //    return Dictionary(DICT_4X4_50_DATA);
}


Dictionary getPredefinedDictionary(int dictionary) {
    return getPredefinedDictionary(DictionaryType(dictionary));
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


    //finally, recalculate bits_id
    out.bits_id.clear();
    // nbytes is the number of bytes for each rotation
    int nbytes = out.bytesList.cols; // Since bytesList is CV_8UC4, cols == nbytes

    for (int m = 0; m < out.bytesList.rows; m++) {
        for (int r = 0; r < 4; r++) {
            // skip nbytes for each rotation r
            const uchar* bytesPtr = out.bytesList.ptr(m) + r * nbytes;
            uint64_t val = 0;

            // we can only pack into uint64_t if nbytes <= 8 (up to 8x8 bits)
            if (nbytes <= 8) {
                for (int j = 0; j < nbytes; j++) {
                    val <<= 8;
                    val |= bytesPtr[j];
                }
                // save the rotation in the map pointing to the marker ID and rotation
                out.bits_id[val] = std::make_pair(m, r);
            }
        }
    }

    return out;
}

} // namespace aruco2
} // namespace cv
