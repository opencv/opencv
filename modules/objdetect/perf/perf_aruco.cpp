// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"
#include "opencv2/calib3d.hpp"

namespace opencv_test {
using namespace perf;

typedef tuple<bool, int> UseArucoParams;
typedef TestBaseWithParam<UseArucoParams> EstimateAruco;
#define ESTIMATE_PARAMS Combine(Values(false, true), Values(-1))

static double deg2rad(double deg) { return deg * CV_PI / 180.; }

class MarkerPainter
{
private:
    int imgMarkerSize = 0;
    Mat cameraMatrix;
public:
    MarkerPainter(const int size) {
        setImgMarkerSize(size);
    }

    void setImgMarkerSize(const int size) {
        imgMarkerSize = size;
        cameraMatrix = Mat::eye(3, 3, CV_64FC1);
        cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = imgMarkerSize;
        cameraMatrix.at<double>(0, 2) = imgMarkerSize / 2.0;
        cameraMatrix.at<double>(1, 2) = imgMarkerSize / 2.0;
    }

    static std::pair<Mat, Mat> getSyntheticRT(double yaw, double pitch, double distance) {
        auto rvec_tvec = std::make_pair(Mat(3, 1, CV_64FC1), Mat(3, 1, CV_64FC1));
        Mat& rvec = rvec_tvec.first;
        Mat& tvec = rvec_tvec.second;

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
        return rvec_tvec;
    }

    std::pair<Mat, vector<Point2f> > getProjectMarker(int id, double yaw, double pitch,
                                                      const aruco::DetectorParameters& parameters,
                                                      const aruco::Dictionary& dictionary) {
        auto marker_corners = std::make_pair(Mat(imgMarkerSize, imgMarkerSize, CV_8UC1, Scalar::all(255)), vector<Point2f>());
        Mat& img = marker_corners.first;
        vector<Point2f>& corners = marker_corners.second;

        // canonical image
        const int markerSizePixels = static_cast<int>(imgMarkerSize/sqrt(2.f));
        aruco::generateImageMarker(dictionary, id, markerSizePixels, img, parameters.markerBorderBits);

        // get rvec and tvec for the perspective
        const double distance = 0.1;
        auto rvec_tvec = MarkerPainter::getSyntheticRT(yaw, pitch, distance);
        Mat& rvec = rvec_tvec.first;
        Mat& tvec = rvec_tvec.second;

        const float markerLength = 0.05f;
        vector<Point3f> markerObjPoints;
        markerObjPoints.emplace_back(Point3f(-markerLength / 2.f, +markerLength / 2.f, 0));
        markerObjPoints.emplace_back(markerObjPoints[0] + Point3f(markerLength, 0, 0));
        markerObjPoints.emplace_back(markerObjPoints[0] + Point3f(markerLength, -markerLength, 0));
        markerObjPoints.emplace_back(markerObjPoints[0] + Point3f(0, -markerLength, 0));

        // project markers and draw them
        Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
        projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

        vector<Point2f> originalCorners;
        originalCorners.emplace_back(Point2f(0.f, 0.f));
        originalCorners.emplace_back(originalCorners[0]+Point2f((float)markerSizePixels, 0));
        originalCorners.emplace_back(originalCorners[0]+Point2f((float)markerSizePixels, (float)markerSizePixels));
        originalCorners.emplace_back(originalCorners[0]+Point2f(0, (float)markerSizePixels));

        Mat transformation = getPerspectiveTransform(originalCorners, corners);

        warpPerspective(img, img, transformation, Size(imgMarkerSize, imgMarkerSize), INTER_NEAREST, BORDER_CONSTANT,
                        Scalar::all(255));
        return marker_corners;
    }

    std::pair<Mat, map<int, vector<Point2f> > > getProjectMarkersTile(const int numMarkers,
                                                                      const aruco::DetectorParameters& params,
                                                                      const aruco::Dictionary& dictionary) {
        Mat tileImage(imgMarkerSize*numMarkers, imgMarkerSize*numMarkers, CV_8UC1, Scalar::all(255));
        map<int, vector<Point2f> > idCorners;

        int iter = 0, pitch = 0, yaw = 0;
        for (int i = 0; i < numMarkers; i++) {
            for (int j = 0; j < numMarkers; j++) {
                int currentId = iter;
                auto marker_corners = getProjectMarker(currentId, deg2rad(70+yaw), deg2rad(pitch), params, dictionary);
                Point2i startPoint(j*imgMarkerSize, i*imgMarkerSize);
                Mat tmp_roi = tileImage(Rect(startPoint.x, startPoint.y, imgMarkerSize, imgMarkerSize));
                marker_corners.first.copyTo(tmp_roi);

                for (Point2f& point: marker_corners.second)
                    point += static_cast<Point2f>(startPoint);
                idCorners[currentId] = marker_corners.second;
                auto test = idCorners[currentId];
                yaw = (yaw + 10) % 51; // 70+yaw >= 70 && 70+yaw <= 120
                iter++;
            }
            pitch = (pitch + 60) % 360;
        }
        return std::make_pair(tileImage, idCorners);
    }
};

static inline double getMaxDistance(map<int, vector<Point2f> > &golds, const vector<int>& ids,
                                    const vector<vector<Point2f> >& corners) {
    std::map<int, double> mapDist;
    for (const auto& el : golds)
        mapDist[el.first] = std::numeric_limits<double>::max();
    for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        const auto gold_corners = golds.find(id);
        if (gold_corners != golds.end()) {
            double distance = 0.;
            for (int c = 0; c < 4; c++)
                distance = std::max(distance, cv::norm(gold_corners->second[c] - corners[i][c]));
            mapDist[id] = distance;
        }
    }
    return std::max_element(std::begin(mapDist), std::end(mapDist),
           [](const pair<int, double>& p1, const pair<int, double>& p2){return p1.second < p2.second;})->second;
}

PERF_TEST_P(EstimateAruco, ArucoFirst, ESTIMATE_PARAMS) {
    UseArucoParams testParams = GetParam();
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::DetectorParameters detectorParams;
    detectorParams.minDistanceToBorder = 1;
    detectorParams.markerBorderBits = 1;
    detectorParams.cornerRefinementMethod = (int)cv::aruco::CORNER_REFINE_SUBPIX;

    const int markerSize = 100;
    const int numMarkersInRow = 9;
    //USE_ARUCO3
    detectorParams.useAruco3Detection = get<0>(testParams);
    if (detectorParams.useAruco3Detection) {
        detectorParams.minSideLengthCanonicalImg = 32;
        detectorParams.minMarkerLengthRatioOriginalImg = 0.04f / numMarkersInRow;
    }
    aruco::ArucoDetector detector(dictionary, detectorParams);
    MarkerPainter painter(markerSize);
    auto image_map = painter.getProjectMarkersTile(numMarkersInRow, detectorParams, dictionary);

    // detect markers
    vector<vector<Point2f> > corners;
    vector<int> ids;
    TEST_CYCLE() {
        detector.detectMarkers(image_map.first, corners, ids);
    }
    ASSERT_EQ(numMarkersInRow*numMarkersInRow, static_cast<int>(ids.size()));
    double maxDistance = getMaxDistance(image_map.second, ids, corners);
    ASSERT_LT(maxDistance, 3.);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(EstimateAruco, ArucoSecond, ESTIMATE_PARAMS) {
    UseArucoParams testParams = GetParam();
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::DetectorParameters detectorParams;
    detectorParams.minDistanceToBorder = 1;
    detectorParams.markerBorderBits = 1;
    detectorParams.cornerRefinementMethod = (int)cv::aruco::CORNER_REFINE_SUBPIX;

    //USE_ARUCO3
    detectorParams.useAruco3Detection = get<0>(testParams);
    if (detectorParams.useAruco3Detection) {
        detectorParams.minSideLengthCanonicalImg = 64;
        detectorParams.minMarkerLengthRatioOriginalImg = 0.f;
    }
    aruco::ArucoDetector detector(dictionary, detectorParams);
    const int markerSize = 200;
    const int numMarkersInRow = 11;
    MarkerPainter painter(markerSize);
    auto image_map = painter.getProjectMarkersTile(numMarkersInRow, detectorParams, dictionary);

    // detect markers
    vector<vector<Point2f> > corners;
    vector<int> ids;
    TEST_CYCLE() {
        detector.detectMarkers(image_map.first, corners, ids);
    }
    ASSERT_EQ(numMarkersInRow*numMarkersInRow, static_cast<int>(ids.size()));
    double maxDistance = getMaxDistance(image_map.second, ids, corners);
    ASSERT_LT(maxDistance, 3.);
    SANITY_CHECK_NOTHING();
}

struct Aruco3Params {
    bool useAruco3Detection = false;
    float minMarkerLengthRatioOriginalImg = 0.f;
    int minSideLengthCanonicalImg = 0;

    Aruco3Params(bool useAruco3, float minMarkerLen, int minSideLen): useAruco3Detection(useAruco3),
                                                                      minMarkerLengthRatioOriginalImg(minMarkerLen),
                                                                      minSideLengthCanonicalImg(minSideLen) {}
    friend std::ostream& operator<<(std::ostream& os, const Aruco3Params& d) {
        os << d.useAruco3Detection << " " << d.minMarkerLengthRatioOriginalImg << " " << d.minSideLengthCanonicalImg;
        return os;
    }
};
typedef tuple<Aruco3Params, pair<int, int>> ArucoTestParams;

typedef TestBaseWithParam<ArucoTestParams> EstimateLargeAruco;
#define ESTIMATE_FHD_PARAMS Combine(Values(Aruco3Params(false, 0.f, 0), Aruco3Params(true, 0.f, 32), \
Aruco3Params(true, 0.015f, 32), Aruco3Params(true, 0.f, 16), Aruco3Params(true, 0.0069f, 16)),       \
Values(std::make_pair(1440, 1), std::make_pair(480, 3), std::make_pair(144, 10)))

PERF_TEST_P(EstimateLargeAruco, ArucoFHD, ESTIMATE_FHD_PARAMS) {
    ArucoTestParams testParams = GetParam();
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::DetectorParameters detectorParams;
    detectorParams.minDistanceToBorder = 1;
    detectorParams.markerBorderBits = 1;
    detectorParams.cornerRefinementMethod = (int)cv::aruco::CORNER_REFINE_SUBPIX;

    //USE_ARUCO3
    detectorParams.useAruco3Detection = get<0>(testParams).useAruco3Detection;
    if (detectorParams.useAruco3Detection) {
        detectorParams.minSideLengthCanonicalImg = get<0>(testParams).minSideLengthCanonicalImg;
        detectorParams.minMarkerLengthRatioOriginalImg = get<0>(testParams).minMarkerLengthRatioOriginalImg;
    }
    aruco::ArucoDetector detector(dictionary, detectorParams);
    const int markerSize = get<1>(testParams).first;       // 1440 or 480 or 144
    const int numMarkersInRow = get<1>(testParams).second; // 1 or 3 or 144
    MarkerPainter painter(markerSize);                     // num pixels is 1440x1440 as in FHD 1920x1080
    auto image_map = painter.getProjectMarkersTile(numMarkersInRow, detectorParams, dictionary);

    // detect markers
    vector<vector<Point2f> > corners;
    vector<int> ids;
    TEST_CYCLE()
    {
        detector.detectMarkers(image_map.first, corners, ids);
    }
    ASSERT_EQ(numMarkersInRow*numMarkersInRow, static_cast<int>(ids.size()));
    double maxDistance = getMaxDistance(image_map.second, ids, corners);
    ASSERT_LT(maxDistance, 3.);
    SANITY_CHECK_NOTHING();
}

}
