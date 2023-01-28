// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/calib3d.hpp"

namespace cv {
    namespace aruco {
        bool operator==(const Dictionary& d1, const Dictionary& d2);
        bool operator==(const Dictionary& d1, const Dictionary& d2) {
            return d1.markerSize == d2.markerSize
                && std::equal(d1.bytesList.begin<Vec<uint8_t, 4>>(), d1.bytesList.end<Vec<uint8_t, 4>>(), d2.bytesList.begin<Vec<uint8_t, 4>>())
                && std::equal(d2.bytesList.begin<Vec<uint8_t, 4>>(), d2.bytesList.end<Vec<uint8_t, 4>>(), d1.bytesList.begin<Vec<uint8_t, 4>>())
                && d1.maxCorrectionBits == d2.maxCorrectionBits;
        };
    }
}

namespace opencv_test { namespace {

/**
 * @brief Draw 2D synthetic markers and detect them
 */
class CV_ArucoDetectionSimple : public cvtest::BaseTest {
    public:
    CV_ArucoDetectionSimple();

    protected:
    void run(int);
};


CV_ArucoDetectionSimple::CV_ArucoDetectionSimple() {}


void CV_ArucoDetectionSimple::run(int) {
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250));

    // 20 images
    for(int i = 0; i < 20; i++) {

        const int markerSidePixels = 100;
        int imageSize = markerSidePixels * 2 + 3 * (markerSidePixels / 2);

        // draw synthetic image and store marker corners and ids
        vector<vector<Point2f> > groundTruthCorners;
        vector<int> groundTruthIds;
        Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
        for(int y = 0; y < 2; y++) {
            for(int x = 0; x < 2; x++) {
                Mat marker;
                int id = i * 4 + y * 2 + x;
                aruco::generateImageMarker(detector.getDictionary(), id, markerSidePixels, marker);
                Point2f firstCorner =
                    Point2f(markerSidePixels / 2.f + x * (1.5f * markerSidePixels),
                            markerSidePixels / 2.f + y * (1.5f * markerSidePixels));
                Mat aux = img.colRange((int)firstCorner.x, (int)firstCorner.x + markerSidePixels)
                              .rowRange((int)firstCorner.y, (int)firstCorner.y + markerSidePixels);
                marker.copyTo(aux);
                groundTruthIds.push_back(id);
                groundTruthCorners.push_back(vector<Point2f>());
                groundTruthCorners.back().push_back(firstCorner);
                groundTruthCorners.back().push_back(firstCorner + Point2f(markerSidePixels - 1, 0));
                groundTruthCorners.back().push_back(
                    firstCorner + Point2f(markerSidePixels - 1, markerSidePixels - 1));
                groundTruthCorners.back().push_back(firstCorner + Point2f(0, markerSidePixels - 1));
            }
        }
        if(i % 2 == 1) img.convertTo(img, CV_8UC3);

        // detect markers
        vector<vector<Point2f> > corners;
        vector<int> ids;

        detector.detectMarkers(img, corners, ids);

        // check detection results
        for(unsigned int m = 0; m < groundTruthIds.size(); m++) {
            int idx = -1;
            for(unsigned int k = 0; k < ids.size(); k++) {
                if(groundTruthIds[m] == ids[k]) {
                    idx = (int)k;
                    break;
                }
            }
            if(idx == -1) {
                ts->printf(cvtest::TS::LOG, "Marker not detected");
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }

            for(int c = 0; c < 4; c++) {
                double dist = cv::norm(groundTruthCorners[m][c] - corners[idx][c]);  // TODO cvtest
                if(dist > 0.001) {
                    ts->printf(cvtest::TS::LOG, "Incorrect marker corners position");
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }
        }
    }
}


static double deg2rad(double deg) { return deg * CV_PI / 180.; }

/**
 * @brief Get rvec and tvec from yaw, pitch and distance
 */
static void getSyntheticRT(double yaw, double pitch, double distance, Mat &rvec, Mat &tvec) {

    rvec = Mat(3, 1, CV_64FC1);
    tvec = Mat(3, 1, CV_64FC1);

    // Rvec
    // first put the Z axis aiming to -X (like the camera axis system)
    Mat rotZ(3, 1, CV_64FC1);
    rotZ.ptr<double>(0)[0] = 0;
    rotZ.ptr<double>(0)[1] = 0;
    rotZ.ptr<double>(0)[2] = -0.5 * CV_PI;

    Mat rotX(3, 1, CV_64FC1);
    rotX.ptr<double>(0)[0] = 0.5 * CV_PI;
    rotX.ptr<double>(0)[1] = 0;
    rotX.ptr<double>(0)[2] = 0;

    Mat camRvec, camTvec;
    composeRT(rotZ, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotX, Mat(3, 1, CV_64FC1, Scalar::all(0)),
              camRvec, camTvec);

    // now pitch and yaw angles
    Mat rotPitch(3, 1, CV_64FC1);
    rotPitch.ptr<double>(0)[0] = 0;
    rotPitch.ptr<double>(0)[1] = pitch;
    rotPitch.ptr<double>(0)[2] = 0;

    Mat rotYaw(3, 1, CV_64FC1);
    rotYaw.ptr<double>(0)[0] = yaw;
    rotYaw.ptr<double>(0)[1] = 0;
    rotYaw.ptr<double>(0)[2] = 0;

    composeRT(rotPitch, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotYaw,
              Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // compose both rotations
    composeRT(camRvec, Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec,
              Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // Tvec, just move in z (camera) direction the specific distance
    tvec.ptr<double>(0)[0] = 0.;
    tvec.ptr<double>(0)[1] = 0.;
    tvec.ptr<double>(0)[2] = distance;
}

/**
 * @brief Create a synthetic image of a marker with perspective
 */
static Mat projectMarker(const aruco::Dictionary &dictionary, int id, Mat cameraMatrix, double yaw,
                         double pitch, double distance, Size imageSize, int markerBorder,
                         vector<Point2f> &corners, int encloseMarker=0) {

    // canonical image
    Mat marker, markerImg;
    const int markerSizePixels = 100;

    aruco::generateImageMarker(dictionary, id, markerSizePixels, marker, markerBorder);
    marker.copyTo(markerImg);

    if(encloseMarker){ //to enclose the marker
        int enclose = int(marker.rows/4);
        markerImg = Mat::zeros(marker.rows+(2*enclose), marker.cols+(enclose*2), CV_8UC1);

        Mat field= markerImg.rowRange(int(enclose), int(markerImg.rows-enclose))
                            .colRange(int(0), int(markerImg.cols));
        field.setTo(255);
        field= markerImg.rowRange(int(0), int(markerImg.rows))
                            .colRange(int(enclose), int(markerImg.cols-enclose));
        field.setTo(255);

        field = markerImg(Rect(enclose,enclose,marker.rows,marker.cols));
        marker.copyTo(field);
    }

    // get rvec and tvec for the perspective
    Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    const float markerLength = 0.05f;
    vector<Point3f> markerObjPoints;
    markerObjPoints.push_back(Point3f(-markerLength / 2.f, +markerLength / 2.f, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, 0, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, -markerLength, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(0, -markerLength, 0));

    // project markers and draw them
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    vector<Point2f> originalCorners;
    originalCorners.push_back(Point2f(0+float(encloseMarker*markerSizePixels/4), 0+float(encloseMarker*markerSizePixels/4)));
    originalCorners.push_back(originalCorners[0]+Point2f((float)markerSizePixels, 0));
    originalCorners.push_back(originalCorners[0]+Point2f((float)markerSizePixels, (float)markerSizePixels));
    originalCorners.push_back(originalCorners[0]+Point2f(0, (float)markerSizePixels));

    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    Mat img(imageSize, CV_8UC1, Scalar::all(255));
    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, imageSize, INTER_NEAREST, BORDER_CONSTANT,
                    Scalar::all(borderValue));

    // copy only not-border pixels
    for(int y = 0; y < aux.rows; y++) {
        for(int x = 0; x < aux.cols; x++) {
            if(aux.at<unsigned char>(y, x) == borderValue) continue;
            img.at<unsigned char>(y, x) = aux.at<unsigned char>(y, x);
        }
    }

    return img;
}

enum class ArucoAlgParams
{
    USE_DEFAULT = 0,
    USE_APRILTAG=1,             /// Detect marker candidates :: using AprilTag
    DETECT_INVERTED_MARKER,     /// Check if there is a white marker
    USE_ARUCO3                  /// Check if aruco3 should be used
};


/**
 * @brief Draws markers in perspective and detect them
 */
class CV_ArucoDetectionPerspective : public cvtest::BaseTest {
    public:
    CV_ArucoDetectionPerspective(ArucoAlgParams arucoAlgParam) : arucoAlgParams(arucoAlgParam) {}

    protected:
    void run(int);
    ArucoAlgParams arucoAlgParams;
};


void CV_ArucoDetectionPerspective::run(int) {

    int iter = 0;
    int szEnclosed = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = 650;
    cameraMatrix.at<double>(0, 2) = imgSize.width / 2;
    cameraMatrix.at<double>(1, 2) = imgSize.height / 2;
    aruco::DetectorParameters params;
    params.minDistanceToBorder = 1;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250), params);

    // detect from different positions
    for(double distance : {0.1, 0.3, 0.5, 0.7}) {
        for(int pitch = 0; pitch < 360; pitch += (distance == 0.1? 60:180)) {
            for(int yaw = 70; yaw <= 120; yaw += 40){
                int currentId = iter % 250;
                int markerBorder = iter % 2 + 1;
                iter++;
                vector<Point2f> groundTruthCorners;
                aruco::DetectorParameters detectorParameters = params;
                detectorParameters.markerBorderBits = markerBorder;

                /// create synthetic image
                Mat img=
                    projectMarker(detector.getDictionary(), currentId, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                      distance, imgSize, markerBorder, groundTruthCorners, szEnclosed);
                // marker :: Inverted
                if(ArucoAlgParams::DETECT_INVERTED_MARKER == arucoAlgParams){
                    img = ~img;
                    detectorParameters.detectInvertedMarker = true;
                }

                if(ArucoAlgParams::USE_APRILTAG == arucoAlgParams){
                    detectorParameters.cornerRefinementMethod = (int)aruco::CORNER_REFINE_APRILTAG;
                }

                if (ArucoAlgParams::USE_ARUCO3 == arucoAlgParams) {
                    detectorParameters.useAruco3Detection = true;
                    detectorParameters.cornerRefinementMethod = (int)aruco::CORNER_REFINE_SUBPIX;
                }
                detector.setDetectorParameters(detectorParameters);

                // detect markers
                vector<vector<Point2f> > corners;
                vector<int> ids;
                detector.detectMarkers(img, corners, ids);

                // check results
                if(ids.size() != 1 || (ids.size() == 1 && ids[0] != currentId)) {
                    if(ids.size() != 1)
                        ts->printf(cvtest::TS::LOG, "Incorrect number of detected markers");
                    else
                        ts->printf(cvtest::TS::LOG, "Incorrect marker id");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }
                for(int c = 0; c < 4; c++) {
                    double dist = cv::norm(groundTruthCorners[c] - corners[0][c]);  // TODO cvtest
                    if(dist > 5) {
                            ts->printf(cvtest::TS::LOG, "Incorrect marker corners position");
                            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                            return;
                    }
                }
            }
        }
        // change the state :: to detect an enclosed inverted marker
        if(ArucoAlgParams::DETECT_INVERTED_MARKER == arucoAlgParams && distance == 0.1){
            distance -= 0.1;
            szEnclosed++;
        }
    }
}


/**
 * @brief Draw 2D synthetic markers, temper with some pixels, detect them and compute their uncertainty.
 */
class CV_ArucoDetectionUnc : public cvtest::BaseTest {
    public:
    CV_ArucoDetectionUnc(ArucoAlgParams arucoAlgParam) : arucoAlgParams(arucoAlgParam) {}

    protected:
    void run(int);
    ArucoAlgParams arucoAlgParams;
};


void CV_ArucoDetectionUnc::run(int) {

    aruco::DetectorParameters params;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250), params);

    // Params to test
    float ingnoreMarginPerCell[3] = {0.0, 0.1, 0.2};
    int borderBitsTest[3] = {1,2,3};

    const int markerSidePixels = 150;
    const int imageSize = (markerSidePixels * 2) + 3 * (markerSidePixels / 2);

    // 25 images containing 4 markers.
    for(int i = 0; i < 25; i++) {

        // Modify default params
        params.perspectiveRemovePixelPerCell = 6 + i;
        params.perspectiveRemoveIgnoredMarginPerCell = ingnoreMarginPerCell[i % 3];
        params.markerBorderBits = borderBitsTest[i % 3];

        // draw synthetic image
        vector<float > groundTruthUncs;
        vector<int> groundTruthIds;
        Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));

        // Invert the pixel value of a % of each cell [0%, 2%, 4%, ..., 48%]
        float invertPixelPercent = 2 * i / 100.f;
        int markerSizeWithBorders = 6 + 2 * params.markerBorderBits;
        int cellSidePixelsSize = markerSidePixels / markerSizeWithBorders;
        int cellSidePixelsInvert = int(sqrt(invertPixelPercent) * cellSidePixelsSize);
        int cellMarginPixels = (cellSidePixelsSize - cellSidePixelsInvert) / 2; // Invert center of the cell

        float groundTruthUnc;

        // Generate 4 markers
        for(int y = 0; y < 2; y++) {
            for(int x = 0; x < 2; x++) {
                Mat marker;
                int id = i * 4 + y * 2 + x;
                groundTruthIds.push_back(id);

                // Generate marker
                aruco::generateImageMarker(detector.getDictionary(), id, markerSidePixels, marker, params.markerBorderBits);

                // Test all 4 rotations: [0, 90, 180, 270]
                if(y == 0 && x == 0){
                    // Rotate 90 deg CCW
                    cv::transpose(marker, marker);
                    cv::flip(marker, marker,0);
                } else if (y == 0 && x == 1){
                    // Rotate 90 deg CW
                    cv::transpose(marker, marker);
                    cv::flip(marker, marker,1);
                } else if (y == 1 && x == 0){
                    // Rotate 180 deg CCW
                    cv::flip(marker, marker,-1);
                }

                // Invert the pixel value of a % of each cell [0%, 2%, 4%, ..., 48%]
                if(cellSidePixelsInvert > 0){
                    // loop over each cell
                    for(int k = 0; k < markerSizeWithBorders; k++) {
                        for(int p = 0; p < markerSizeWithBorders; p++) {
                            int Xstart = p * (cellSidePixelsSize) + cellMarginPixels;
                            int Ystart = k * (cellSidePixelsSize) + cellMarginPixels;
                            Mat square(marker, Rect(Xstart, Ystart, cellSidePixelsInvert, cellSidePixelsInvert));
                            square = ~square;
                        }
                    }
                }

                // Assume a perfect marker detection and thus a ground truth equal to the percentage of inverted pixels.
                groundTruthUnc = markerSizeWithBorders * markerSizeWithBorders * cellSidePixelsInvert * cellSidePixelsInvert / (float)(markerSidePixels * markerSidePixels);
                groundTruthUncs.push_back(groundTruthUnc);

                // Make sure that the marker is still detected when it was highly tempered.
                if(groundTruthUnc >= 0.2) params.perspectiveRemoveIgnoredMarginPerCell = 0;

                // Copy marker into full image
                Point2f firstCorner =
                    Point2f(markerSidePixels / 2.f + x * (1.5f * markerSidePixels),
                            markerSidePixels / 2.f + y * (1.5f * markerSidePixels));
                Mat aux = img.colRange((int)firstCorner.x, (int)firstCorner.x + markerSidePixels)
                              .rowRange((int)firstCorner.y, (int)firstCorner.y + markerSidePixels);

                marker.copyTo(aux);
            }
        }

        // Test inverted markers
        if(ArucoAlgParams::DETECT_INVERTED_MARKER == arucoAlgParams){
            img = ~img;
            params.detectInvertedMarker = true;
        }

        detector.setDetectorParameters(params);

        // detect markers and compute uncertainty
        vector<vector<Point2f> > corners, rejected;
        vector<int> ids;
        vector<float> markerUnc;

        detector.detectMarkersWithUnc(img, corners, ids, rejected, markerUnc);

        // check detection results
        for(unsigned int m = 0; m < groundTruthIds.size(); m++) {
            int idx = -1;
            for(unsigned int k = 0; k < ids.size(); k++) {
                if(groundTruthIds[m] == ids[k]) {
                    idx = (int)k;
                    break;
                }
            }
            if(idx == -1) {
                ts->printf(cvtest::TS::LOG, "Marker not detected");
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }
            double dist = (double)cv::abs(groundTruthUncs[m] - markerUnc[idx]);  // TODO cvtest
            if(dist > 0.05) {
                ts->printf(cvtest::TS::LOG, "Marker: %d is incorrect: uncertainty: %.2f (GT: %.2f) ", m, markerUnc[idx], groundTruthUncs[m]);
                ts->printf(cvtest::TS::LOG, "");
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
        }
    }
}

/**
 * @brief Check max and min size in marker detection parameters
 */
class CV_ArucoDetectionMarkerSize : public cvtest::BaseTest {
    public:
    CV_ArucoDetectionMarkerSize();

    protected:
    void run(int);
};


CV_ArucoDetectionMarkerSize::CV_ArucoDetectionMarkerSize() {}


void CV_ArucoDetectionMarkerSize::run(int) {
    aruco::DetectorParameters params;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250), params);
    int markerSide = 20;
    int imageSize = 200;

    // 10 cases
    for(int i = 0; i < 10; i++) {
        Mat marker;
        int id = 10 + i * 20;

        // create synthetic image
        Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
        aruco::generateImageMarker(detector.getDictionary(), id, markerSide, marker);
        Mat aux = img.colRange(30, 30 + markerSide).rowRange(50, 50 + markerSide);
        marker.copyTo(aux);

        vector<vector<Point2f> > corners;
        vector<int> ids;

        // set a invalid minMarkerPerimeterRate
        aruco::DetectorParameters detectorParameters = params;
        detectorParameters.minMarkerPerimeterRate = min(4., (4. * markerSide) / float(imageSize) + 0.1);
        detector.setDetectorParameters(detectorParameters);
        detector.detectMarkers(img, corners, ids);
        if(corners.size() != 0) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::minMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        // set an valid minMarkerPerimeterRate
        detectorParameters = params;
        detectorParameters.minMarkerPerimeterRate = max(0., (4. * markerSide) / float(imageSize) - 0.1);
        detector.setDetectorParameters(detectorParameters);
        detector.detectMarkers(img, corners, ids);
        if(corners.size() != 1 || (corners.size() == 1 && ids[0] != id)) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::minMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        // set a invalid maxMarkerPerimeterRate
        detectorParameters = params;
        detectorParameters.maxMarkerPerimeterRate = min(4., (4. * markerSide) / float(imageSize) - 0.1);
        detector.setDetectorParameters(detectorParameters);
        detector.detectMarkers(img, corners, ids);
        if(corners.size() != 0) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::maxMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        // set an valid maxMarkerPerimeterRate
        detectorParameters = params;
        detectorParameters.maxMarkerPerimeterRate = max(0., (4. * markerSide) / float(imageSize) + 0.1);
        detector.setDetectorParameters(detectorParameters);
        detector.detectMarkers(img, corners, ids);
        if(corners.size() != 1 || (corners.size() == 1 && ids[0] != id)) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::maxMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }
    }
}


/**
 * @brief Check error correction in marker bits
 */
class CV_ArucoBitCorrection : public cvtest::BaseTest {
    public:
    CV_ArucoBitCorrection();

    protected:
    void run(int);
};


CV_ArucoBitCorrection::CV_ArucoBitCorrection() {}


void CV_ArucoBitCorrection::run(int) {

    aruco::Dictionary dictionary1 = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::Dictionary dictionary2 = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::DetectorParameters params;
    aruco::ArucoDetector detector1(dictionary1, params);
    int markerSide = 50;
    int imageSize = 150;

    // 10 markers
    for(int l = 0; l < 10; l++) {
        Mat marker;
        int id = 10 + l * 20;

        Mat currentCodeBytes = dictionary1.bytesList.rowRange(id, id + 1);
        aruco::DetectorParameters detectorParameters = detector1.getDetectorParameters();
        // 5 valid cases
        for(int i = 0; i < 5; i++) {
            // how many bit errors (the error is low enough so it can be corrected)
            detectorParameters.errorCorrectionRate = 0.2 + i * 0.1;
            detector1.setDetectorParameters(detectorParameters);
            int errors =
                (int)std::floor(dictionary1.maxCorrectionBits * detector1.getDetectorParameters().errorCorrectionRate - 1.);

            // create erroneous marker in currentCodeBits
            Mat currentCodeBits =
                aruco::Dictionary::getBitsFromByteList(currentCodeBytes, dictionary1.markerSize);
            for(int e = 0; e < errors; e++) {
                currentCodeBits.ptr<unsigned char>()[2 * e] =
                    !currentCodeBits.ptr<unsigned char>()[2 * e];
            }

            // add erroneous marker to dictionary2 in order to create the erroneous marker image
            Mat currentCodeBytesError = aruco::Dictionary::getByteListFromBits(currentCodeBits);
            currentCodeBytesError.copyTo(dictionary2.bytesList.rowRange(id, id + 1));
            Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
            dictionary2.generateImageMarker(id, markerSide, marker);
            Mat aux = img.colRange(30, 30 + markerSide).rowRange(50, 50 + markerSide);
            marker.copyTo(aux);

            // try to detect using original dictionary
            vector<vector<Point2f> > corners;
            vector<int> ids;
            detector1.detectMarkers(img, corners, ids);
            if(corners.size() != 1 || (corners.size() == 1 && ids[0] != id)) {
                ts->printf(cvtest::TS::LOG, "Error in bit correction");
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
        }

        // 5 invalid cases
        for(int i = 0; i < 5; i++) {
            // how many bit errors (the error is too high to be corrected)
            detectorParameters.errorCorrectionRate = 0.2 + i * 0.1;
            detector1.setDetectorParameters(detectorParameters);
            int errors =
                (int)std::floor(dictionary1.maxCorrectionBits * detector1.getDetectorParameters().errorCorrectionRate + 1.);

            // create erroneous marker in currentCodeBits
            Mat currentCodeBits =
                aruco::Dictionary::getBitsFromByteList(currentCodeBytes, dictionary1.markerSize);
            for(int e = 0; e < errors; e++) {
                currentCodeBits.ptr<unsigned char>()[2 * e] =
                    !currentCodeBits.ptr<unsigned char>()[2 * e];
            }

            // dictionary3 is only composed by the modified marker (in its original form)
            aruco::Dictionary _dictionary3 = aruco::Dictionary(
                    dictionary2.bytesList.rowRange(id, id + 1).clone(),
                    dictionary1.markerSize,
                    dictionary1.maxCorrectionBits);
            aruco::ArucoDetector detector3(_dictionary3, detector1.getDetectorParameters());
            // add erroneous marker to dictionary2 in order to create the erroneous marker image
            Mat currentCodeBytesError = aruco::Dictionary::getByteListFromBits(currentCodeBits);
            currentCodeBytesError.copyTo(dictionary2.bytesList.rowRange(id, id + 1));
            Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
            dictionary2.generateImageMarker(id, markerSide, marker);
            Mat aux = img.colRange(30, 30 + markerSide).rowRange(50, 50 + markerSide);
            marker.copyTo(aux);

            // try to detect using dictionary3, it should fail
            vector<vector<Point2f> > corners;
            vector<int> ids;
            detector3.detectMarkers(img, corners, ids);
            if(corners.size() != 0) {
                ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::errorCorrectionRate");
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
        }
    }
}

typedef CV_ArucoDetectionPerspective CV_AprilTagDetectionPerspective;
typedef CV_ArucoDetectionPerspective CV_InvertedArucoDetectionPerspective;
typedef CV_ArucoDetectionPerspective CV_Aruco3DetectionPerspective;

TEST(CV_InvertedArucoDetectionPerspective, algorithmic) {
    CV_InvertedArucoDetectionPerspective test(ArucoAlgParams::DETECT_INVERTED_MARKER);
    test.safe_run();
}

TEST(CV_AprilTagDetectionPerspective, algorithmic) {
    CV_AprilTagDetectionPerspective test(ArucoAlgParams::USE_APRILTAG);
    test.safe_run();
}

TEST(CV_Aruco3DetectionPerspective, algorithmic) {
    CV_Aruco3DetectionPerspective test(ArucoAlgParams::USE_ARUCO3);
    test.safe_run();
}

TEST(CV_ArucoDetectionSimple, algorithmic) {
    CV_ArucoDetectionSimple test;
    test.safe_run();
}

TEST(CV_ArucoDetectionPerspective, algorithmic) {
    CV_ArucoDetectionPerspective test(ArucoAlgParams::USE_DEFAULT);
    test.safe_run();
}

TEST(CV_ArucoDetectionMarkerSize, algorithmic) {
    CV_ArucoDetectionMarkerSize test;
    test.safe_run();
}

TEST(CV_ArucoBitCorrection, algorithmic) {
    CV_ArucoBitCorrection test;
    test.safe_run();
}

typedef CV_ArucoDetectionUnc CV_InvertedArucoDetectionUnc;

TEST(CV_ArucoDetectionUnc, algorithmic) {
    CV_ArucoDetectionUnc test(ArucoAlgParams::USE_DEFAULT);
    test.safe_run();
}

TEST(CV_InvertedArucoDetectionUnc, algorithmic) {
    CV_InvertedArucoDetectionUnc test(ArucoAlgParams::DETECT_INVERTED_MARKER);
    test.safe_run();
}

TEST(CV_ArucoDetectMarkers, regression_3192)
{
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_4X4_50));
    vector<int> markerIds;
    vector<vector<Point2f> > markerCorners;
    string imgPath = cvtest::findDataFile("aruco/regression_3192.png");
    Mat image = imread(imgPath);
    const size_t N = 2ull;
    const int goldCorners[N][8] = { {345,120, 520,120, 520,295, 345,295}, {101,114, 270,112, 276,287, 101,287} };
    const int goldCornersIds[N] = { 6, 4 };
    map<int, const int*> mapGoldCorners;
    for (size_t i = 0; i < N; i++)
        mapGoldCorners[goldCornersIds[i]] = goldCorners[i];

    detector.detectMarkers(image, markerCorners, markerIds);

    ASSERT_EQ(N, markerIds.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = markerIds[i];
        ASSERT_EQ(4ull, markerCorners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2]), markerCorners[i][j].x, 1.f);
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2 + 1]), markerCorners[i][j].y, 1.f);
        }
    }
}

TEST(CV_ArucoDetectMarkers, regression_2492)
{
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_5X5_50));
    aruco::DetectorParameters detectorParameters = detector.getDetectorParameters();
    detectorParameters.minMarkerDistanceRate = 0.026;
    detector.setDetectorParameters(detectorParameters);
    vector<int> markerIds;
    vector<vector<Point2f> > markerCorners;
    string imgPath = cvtest::findDataFile("aruco/regression_2492.png");
    Mat image = imread(imgPath);
    const size_t N = 8ull;
    const int goldCorners[N][8] = { {179,139, 179,95, 223,95, 223,139}, {99,139, 99,95, 143,95, 143,139},
                                    {19,139, 19,95, 63,95, 63,139},     {256,140, 256,93, 303,93, 303,140},
                                    {256,62, 259,21, 300,23, 297,64},   {99,21, 143,17, 147,60, 103,64},
                                    {69,61, 28,61, 14,21, 58,17},       {174,62, 182,13, 230,19, 223,68} };
    const int goldCornersIds[N] = {13, 13, 13, 13, 1, 15, 14, 4};
    map<int, vector<const int*> > mapGoldCorners;
    for (size_t i = 0; i < N; i++)
        mapGoldCorners[goldCornersIds[i]].push_back(goldCorners[i]);

    detector.detectMarkers(image, markerCorners, markerIds);

    ASSERT_EQ(N, markerIds.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = markerIds[i];
        ASSERT_EQ(4ull, markerCorners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        float totalDist = 8.f;
        for (size_t k = 0ull; k < mapGoldCorners[arucoId].size(); k++)
        {
            float dist = 0.f;
            for (int j = 0; j < 4; j++) // total distance up to 4 points
            {
                dist += abs(mapGoldCorners[arucoId][k][j * 2] - markerCorners[i][j].x);
                dist += abs(mapGoldCorners[arucoId][k][j * 2 + 1] - markerCorners[i][j].y);
            }
            totalDist = min(totalDist, dist);
        }
        EXPECT_LT(totalDist, 8.f);
    }
}


TEST(CV_ArucoDetectMarkers, regression_contour_24220)
{
    aruco::ArucoDetector detector;
    vector<int> markerIds;
    vector<vector<Point2f> > markerCorners;
    string imgPath = cvtest::findDataFile("aruco/failmask9.png");
    Mat image = imread(imgPath);

    const size_t N = 1ull;
    const int goldCorners[8] = {392,175, 99,257, 117,109, 365,44};
    const int goldCornersId = 0;

    detector.detectMarkers(image, markerCorners, markerIds);

    ASSERT_EQ(N, markerIds.size());
    ASSERT_EQ(4ull, markerCorners[0].size());
    ASSERT_EQ(goldCornersId, markerIds[0]);
    for (int j = 0; j < 4; j++)
    {
        EXPECT_NEAR(static_cast<float>(goldCorners[j * 2]), markerCorners[0][j].x, 1.f);
        EXPECT_NEAR(static_cast<float>(goldCorners[j * 2 + 1]), markerCorners[0][j].y, 1.f);
    }
}

TEST(CV_ArucoMultiDict, setGetDictionaries)
{
    vector<aruco::Dictionary> dictionaries = {aruco::getPredefinedDictionary(aruco::DICT_4X4_50), aruco::getPredefinedDictionary(aruco::DICT_5X5_100)};
    aruco::ArucoDetector detector(dictionaries);
    vector<aruco::Dictionary> dicts = detector.getDictionaries();
    ASSERT_EQ(dicts.size(), 2ul);
    EXPECT_EQ(dicts[0].markerSize, 4);
    EXPECT_EQ(dicts[1].markerSize, 5);
    dictionaries.clear();
    dictionaries.push_back(aruco::getPredefinedDictionary(aruco::DICT_6X6_100));
    dictionaries.push_back(aruco::getPredefinedDictionary(aruco::DICT_7X7_250));
    dictionaries.push_back(aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_25h9));
    detector.setDictionaries(dictionaries);
    dicts = detector.getDictionaries();
    ASSERT_EQ(dicts.size(), 3ul);
    EXPECT_EQ(dicts[0].markerSize, 6);
    EXPECT_EQ(dicts[1].markerSize, 7);
    EXPECT_EQ(dicts[2].markerSize, 5);
    auto dict = detector.getDictionary();
    EXPECT_EQ(dict.markerSize, 6);
    detector.setDictionary(aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_16h5));
    dicts = detector.getDictionaries();
    ASSERT_EQ(dicts.size(), 3ul);
    EXPECT_EQ(dicts[0].markerSize, 4);
    EXPECT_EQ(dicts[1].markerSize, 7);
    EXPECT_EQ(dicts[2].markerSize, 5);
}


TEST(CV_ArucoMultiDict, noDict)
{
    aruco::ArucoDetector detector;
    EXPECT_THROW({
        detector.setDictionaries({});
    }, Exception);
}


TEST(CV_ArucoMultiDict, multiMarkerDetection)
{
    const int markerSidePixels = 100;
    const int imageSize = markerSidePixels * 2 + 3 * (markerSidePixels / 2);
    vector<aruco::Dictionary> usedDictionaries;

    // draw synthetic image
    Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
    for(int y = 0; y < 2; y++) {
        for(int x = 0; x < 2; x++) {
            Mat marker;
            int id = y * 2 + x;
            int dictId = x * 4 + y * 8;
            auto dict = aruco::getPredefinedDictionary(dictId);
            usedDictionaries.push_back(dict);
            aruco::generateImageMarker(dict, id, markerSidePixels, marker);
            Point2f firstCorner(markerSidePixels / 2.f + x * (1.5f * markerSidePixels),
                        markerSidePixels / 2.f + y * (1.5f * markerSidePixels));
            Mat aux = img(Rect((int)firstCorner.x, (int)firstCorner.y, markerSidePixels, markerSidePixels));
            marker.copyTo(aux);
        }
    }
    img.convertTo(img, CV_8UC3);

    aruco::ArucoDetector detector(usedDictionaries);

    vector<vector<Point2f> > markerCorners;
    vector<int> markerIds;
    vector<vector<Point2f> > rejectedImgPts;
    vector<int> dictIds;
    detector.detectMarkersMultiDict(img, markerCorners, markerIds, rejectedImgPts, dictIds);
    ASSERT_EQ(markerIds.size(), 4u);
    ASSERT_EQ(dictIds.size(), 4u);
    for (size_t i = 0; i < dictIds.size(); ++i) {
        EXPECT_EQ(dictIds[i], (int)i);
    }
}


TEST(CV_ArucoMultiDict, multiMarkerDoubleDetection)
{
    const int markerSidePixels = 100;
    const int imageWidth = 2 * markerSidePixels + 3 * (markerSidePixels / 2);
    const int imageHeight = markerSidePixels + 2 * (markerSidePixels / 2);
    vector<aruco::Dictionary> usedDictionaries = {
        aruco::getPredefinedDictionary(aruco::DICT_5X5_50),
        aruco::getPredefinedDictionary(aruco::DICT_5X5_100)
    };

    // draw synthetic image
    Mat img = Mat(imageHeight, imageWidth, CV_8UC1, Scalar::all(255));
    for(int y = 0; y < 2; y++) {
        Mat marker;
        int id = 49 + y;
        auto dict = aruco::getPredefinedDictionary(aruco::DICT_5X5_100);
        aruco::generateImageMarker(dict, id, markerSidePixels, marker);
        Point2f firstCorner(markerSidePixels / 2.f + y * (1.5f * markerSidePixels),
                    markerSidePixels / 2.f);
        Mat aux = img(Rect((int)firstCorner.x, (int)firstCorner.y, markerSidePixels, markerSidePixels));
        marker.copyTo(aux);
    }
    img.convertTo(img, CV_8UC3);

    aruco::ArucoDetector detector(usedDictionaries);

    vector<vector<Point2f> > markerCorners;
    vector<int> markerIds;
    vector<vector<Point2f> > rejectedImgPts;
    vector<int> dictIds;
    detector.detectMarkersMultiDict(img, markerCorners, markerIds, rejectedImgPts, dictIds);
    ASSERT_EQ(markerIds.size(), 3u);
    ASSERT_EQ(dictIds.size(), 3u);
    EXPECT_EQ(dictIds[0], 0); // 5X5_50
    EXPECT_EQ(dictIds[1], 1); // 5X5_100
    EXPECT_EQ(dictIds[2], 1); // 5X5_100
}


TEST(CV_ArucoMultiDict, serialization)
{
    aruco::ArucoDetector detector;
    {
        FileStorage fs_out(".json", FileStorage::WRITE + FileStorage::MEMORY);
        ASSERT_TRUE(fs_out.isOpened());
        detector.write(fs_out);
        std::string serialized_string = fs_out.releaseAndGetString();
        FileStorage test_fs(serialized_string, FileStorage::Mode::READ + FileStorage::MEMORY);
        ASSERT_TRUE(test_fs.isOpened());
        aruco::ArucoDetector test_detector;
        test_detector.read(test_fs.root());
        // compare default constructor result
        EXPECT_EQ(aruco::getPredefinedDictionary(aruco::DICT_4X4_50), test_detector.getDictionary());
    }
    detector.setDictionaries({aruco::getPredefinedDictionary(aruco::DICT_4X4_50), aruco::getPredefinedDictionary(aruco::DICT_5X5_100)});
    {
        FileStorage fs_out(".json", FileStorage::WRITE + FileStorage::MEMORY);
        ASSERT_TRUE(fs_out.isOpened());
        detector.write(fs_out);
        std::string serialized_string = fs_out.releaseAndGetString();
        FileStorage test_fs(serialized_string, FileStorage::Mode::READ + FileStorage::MEMORY);
        ASSERT_TRUE(test_fs.isOpened());
        aruco::ArucoDetector test_detector;
        test_detector.read(test_fs.root());
        // check for one additional dictionary
        auto dicts = test_detector.getDictionaries();
        ASSERT_EQ(2ul, dicts.size());
        EXPECT_EQ(aruco::getPredefinedDictionary(aruco::DICT_4X4_50), dicts[0]);
        EXPECT_EQ(aruco::getPredefinedDictionary(aruco::DICT_5X5_100), dicts[1]);
    }
}


struct ArucoThreading: public testing::TestWithParam<aruco::CornerRefineMethod>
{
    struct NumThreadsSetter {
        NumThreadsSetter(const int num_threads)
        : original_num_threads_(getNumThreads()) {
            setNumThreads(num_threads);
        }

        ~NumThreadsSetter() {
            setNumThreads(original_num_threads_);
        }
     private:
        int original_num_threads_;
    };
};

TEST_P(ArucoThreading, number_of_threads_does_not_change_results)
{
    // We are not testing against different dictionaries
    // As we are interested mostly in small images, smaller
    // markers is better -> 4x4
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_4X4_50));

    // Height of the test image can be chosen quite freely
    // We aim to test against small images as in those the
    // number of threads has most effect
    const int height_img = 20;
    // Just to get nice white boarder
    const int shift = height_img > 10 ? 5 : 1;
    const int height_marker = height_img-2*shift;

    // Create a test image
    Mat img_marker;
    aruco::generateImageMarker(detector.getDictionary(), 23, height_marker, img_marker, 1);

    // Copy to bigger image to get a white border
    Mat img(height_img, height_img, CV_8UC1, Scalar(255));
    img_marker.copyTo(img(Rect(shift, shift, height_marker, height_marker)));

    aruco::DetectorParameters detectorParameters = detector.getDetectorParameters();
    detectorParameters.cornerRefinementMethod = (int)GetParam();
    detector.setDetectorParameters(detectorParameters);

    vector<vector<Point2f> > original_corners;
    vector<int> original_ids;
    {
        NumThreadsSetter thread_num_setter(1);
        detector.detectMarkers(img, original_corners, original_ids);
    }

    ASSERT_EQ(original_ids.size(), 1ull);
    ASSERT_EQ(original_corners.size(), 1ull);

    int num_threads_to_test[] = { 2, 8, 16, 32, height_img-1, height_img, height_img+1};

    for (size_t i_num_threads = 0; i_num_threads < sizeof(num_threads_to_test)/sizeof(int); ++i_num_threads) {
        NumThreadsSetter thread_num_setter(num_threads_to_test[i_num_threads]);

        vector<vector<Point2f> > corners;
        vector<int> ids;
        detector.detectMarkers(img, corners, ids);

        // If we don't find any markers, the test is broken
        ASSERT_EQ(ids.size(), 1ull);

        // Make sure we got the same result as the first time
        ASSERT_EQ(corners.size(), original_corners.size());
        ASSERT_EQ(ids.size(), original_ids.size());
        ASSERT_EQ(ids.size(), corners.size());
        for (size_t i = 0; i < corners.size(); ++i) {
            EXPECT_EQ(ids[i], original_ids[i]);
            for (size_t j = 0; j < corners[i].size(); ++j) {
                EXPECT_NEAR(corners[i][j].x, original_corners[i][j].x, 0.1f);
                EXPECT_NEAR(corners[i][j].y, original_corners[i][j].y, 0.1f);
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(
        CV_ArucoDetectMarkers, ArucoThreading,
        ::testing::Values(
            aruco::CORNER_REFINE_NONE,
            aruco::CORNER_REFINE_SUBPIX,
            aruco::CORNER_REFINE_CONTOUR,
            aruco::CORNER_REFINE_APRILTAG
        ));

}} // namespace
