#include <opencv2/core/hal/hal.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>

using namespace cv;
using namespace std;

static int _getSelfDistance(const Mat &marker) {
    Mat bytes = aruco::Dictionary::getByteListFromBits(marker);
    int minHamming = (int)marker.total() + 1;
    for(int r = 1; r < 4; r++) {
        int currentHamming = cv::hal::normHamming(bytes.ptr(), bytes.ptr() + bytes.cols*r, bytes.cols);
        if(currentHamming < minHamming) minHamming = currentHamming;
    }
    Mat b;
    flip(marker, b, 0);
    Mat flipBytes = aruco::Dictionary::getByteListFromBits(b);
    for(int r = 0; r < 4; r++) {
        int currentHamming = cv::hal::normHamming(flipBytes.ptr(), bytes.ptr() + bytes.cols*r, bytes.cols);
        if(currentHamming < minHamming) minHamming = currentHamming;
    }
    flip(marker, b, 1);
    flipBytes = aruco::Dictionary::getByteListFromBits(b);
    for(int r = 0; r < 4; r++) {
        int currentHamming = cv::hal::normHamming(flipBytes.ptr(), bytes.ptr() + bytes.cols*r, bytes.cols);
        if(currentHamming < minHamming) minHamming = currentHamming;
    }
    return minHamming;
}

static inline int getFlipDistanceToId(const aruco::Dictionary& dict, InputArray bits, int id, bool allRotations = true) {
    Mat bytesList = dict.bytesList;
    CV_Assert(id >= 0 && id < bytesList.rows);

    unsigned int nRotations = 4;
    if(!allRotations) nRotations = 1;

    Mat candidateBytes = aruco::Dictionary::getByteListFromBits(bits.getMat());
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
    Mat b;
    flip(bits.getMat(), b, 0);
    candidateBytes = aruco::Dictionary::getByteListFromBits(b);
    for(unsigned int r = 0; r < nRotations; r++) {
        int currentHamming = cv::hal::normHamming(
                bytesList.ptr(id) + r * candidateBytes.cols,
                candidateBytes.ptr(),
                candidateBytes.cols);
        if (currentHamming < currentMinDistance) {
            currentMinDistance = currentHamming;
        }
    }

    flip(bits.getMat(), b, 1);
    candidateBytes = aruco::Dictionary::getByteListFromBits(b);
    for(unsigned int r = 0; r < nRotations; r++) {
        int currentHamming = cv::hal::normHamming(
                bytesList.ptr(id) + r * candidateBytes.cols,
                candidateBytes.ptr(),
                candidateBytes.cols);
        if (currentHamming < currentMinDistance) {
            currentMinDistance = currentHamming;
        }
    }
    return currentMinDistance;
}

static inline aruco::Dictionary generateCustomAsymmetricDictionary(int nMarkers, int markerSize,
                                                                        const aruco::Dictionary &baseDictionary,
                                                                        int randomSeed) {
    RNG rng((uint64)(randomSeed));

    aruco::Dictionary out;
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
        out.bytesList = baseDictionary.bytesList.clone();

        int minDistance = markerSize * markerSize + 1;
        for(int i = 0; i < out.bytesList.rows; i++) {
            Mat markerBytes = out.bytesList.rowRange(i, i + 1);
            Mat markerBits = aruco::Dictionary::getBitsFromByteList(markerBytes, markerSize);
            minDistance = min(minDistance, _getSelfDistance(markerBits));
            for(int j = i + 1; j < out.bytesList.rows; j++) {
                minDistance = min(minDistance, getFlipDistanceToId(out, markerBits, j));
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
        Mat currentMarker(markerSize, markerSize, CV_8UC1, Scalar::all(0));
        rng.fill(currentMarker, RNG::UNIFORM, 0, 2);

        int selfDistance = _getSelfDistance(currentMarker);
        int minDistance = selfDistance;

        // if self distance is better or equal than current best option, calculate distance
        // to previous accepted markers
        if(selfDistance >= bestTau) {
            for(int i = 0; i < out.bytesList.rows; i++) {
                int currentDistance = getFlipDistanceToId(out, currentMarker, i);
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
            Mat bytes = aruco::Dictionary::getByteListFromBits(currentMarker);
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
                Mat bytes = aruco::Dictionary::getByteListFromBits(bestMarker);
                out.bytesList.push_back(bytes);
            }
        }
    }

    // update the maximum number of correction bits for the generated dictionary
    out.maxCorrectionBits = (tau - 1) / 2;

    return out;
}

static inline int getMinDistForDict(const aruco::Dictionary& dict) {
    const int dict_size = dict.bytesList.rows;
    const int marker_size = dict.markerSize;
    int minDist = marker_size * marker_size;
    for (int i = 0; i < dict_size; i++) {
        Mat row = dict.bytesList.row(i);
        Mat marker = dict.getBitsFromByteList(row, marker_size);
        for (int j = 0; j < dict_size; j++) {
            if (j != i) {
                minDist = min(dict.getDistanceToId(marker, j), minDist);
            }
        }
    }
    return minDist;
}

static inline int getMinAsymDistForDict(const aruco::Dictionary& dict) {
    const int dict_size = dict.bytesList.rows;
    const int marker_size = dict.markerSize;
    int minDist = marker_size * marker_size;
    for (int i = 0; i < dict_size; i++)
    {
        Mat row = dict.bytesList.row(i);
        Mat marker = dict.getBitsFromByteList(row, marker_size);
        for (int j = 0; j < dict_size; j++)
        {
            if (j != i)
            {
                minDist = min(getFlipDistanceToId(dict, marker, j), minDist);
            }
        }
    }
    return minDist;
}

const char* keys  =
        "{@outfile   |<none> | Output file with custom dict }"
        "{r          | false | Calculate the metric considering flipped markers }"
        "{d          |       | Dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16,"
        "DICT_APRILTAG_16h5 = 17, DICT_APRILTAG_25h9 = 18, DICT_APRILTAG_36h10 = 19,"
        "DICT_APRILTAG_36h11 = 20 }"
        "{nMarkers   |       | Number of markers in the dictionary }"
        "{markerSize |       | Marker size }"
        "{cd         |       | Input file with custom dictionary }";

const char* about =
        "This program can be used to calculate the ArUco dictionary metric.\n"
        "To calculate the metric considering flipped markers use -'r' flag.\n"
        "This program can be used to create and write the custom ArUco dictionary.\n";

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }
    string outputFile = parser.get<String>(0);
    int nMarkers = parser.get<int>("nMarkers");
    int markerSize = parser.get<int>("markerSize");
    bool checkFlippedMarkers = parser.get<bool>("r");

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(0);
    if (parser.has("d")) {
        int dictionaryId = parser.get<int>("d");
        dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(dictionaryId));
    }
    else if (parser.has("cd")) {
        FileStorage fs(parser.get<std::string>("cd"), FileStorage::READ);
        bool readOk = dictionary.aruco::Dictionary::readDictionary(fs.root());
        if(!readOk) {
            cerr << "Invalid dictionary file" << endl;
            return 0;
        }
    }
    else if (outputFile.empty() || nMarkers == 0 || markerSize == 0) {
        cerr << "Dictionary not specified" << endl;
        return 0;
    }

    if (!outputFile.empty() && nMarkers > 0 && markerSize > 0)
    {
        FileStorage fs(outputFile, FileStorage::WRITE);
        if (checkFlippedMarkers)
            dictionary = generateCustomAsymmetricDictionary(nMarkers, markerSize, aruco::Dictionary(), 0);
        else
            dictionary = aruco::extendDictionary(nMarkers, markerSize, aruco::Dictionary(), 0);
        dictionary.writeDictionary(fs);
    }

    if (checkFlippedMarkers) {
        cout << "Hamming distance: " << getMinAsymDistForDict(dictionary) << endl;
    }
    else {
        cout << "Hamming distance: " << getMinDistForDict(dictionary) << endl;
    }
    return 0;
}
