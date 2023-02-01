// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#include "test_aruco_utils.hpp"

namespace opencv_test {

vector<Point2f> getAxis(InputArray _cameraMatrix, InputArray _distCoeffs, InputArray _rvec,
                        InputArray _tvec, float length, const float offset) {
    vector<Point3f> axis;
    axis.push_back(Point3f(offset, offset, 0.f));
    axis.push_back(Point3f(length+offset, offset, 0.f));
    axis.push_back(Point3f(offset, length+offset, 0.f));
    axis.push_back(Point3f(offset, offset, length));
    vector<Point2f> axis_to_img;
    projectPoints(axis, _rvec, _tvec, _cameraMatrix, _distCoeffs, axis_to_img);
    return axis_to_img;
}

vector<Point2f> getMarkerById(int id, const vector<vector<Point2f> >& corners, const vector<int>& ids) {
    for (size_t i = 0ull; i < ids.size(); i++)
        if (ids[i] == id)
            return corners[i];
    return vector<Point2f>();
}

void getSyntheticRT(double yaw, double pitch, double distance, Mat& rvec, Mat& tvec) {
    rvec = Mat::zeros(3, 1, CV_64FC1);
    tvec = Mat::zeros(3, 1, CV_64FC1);

    // rotate "scene" in pitch axis (x-axis)
    Mat rotPitch(3, 1, CV_64FC1);
    rotPitch.at<double>(0) = -pitch;
    rotPitch.at<double>(1) = 0;
    rotPitch.at<double>(2) = 0;

    // rotate "scene" in yaw (y-axis)
    Mat rotYaw(3, 1, CV_64FC1);
    rotYaw.at<double>(0) = 0;
    rotYaw.at<double>(1) = yaw;
    rotYaw.at<double>(2) = 0;

    // compose both rotations
    composeRT(rotPitch, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotYaw,
        Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // Tvec, just move in z (camera) direction the specific distance
    tvec.at<double>(0) = 0.;
    tvec.at<double>(1) = 0.;
    tvec.at<double>(2) = distance;
}

void projectMarker(Mat& img, const aruco::Board& board, int markerIndex, Mat cameraMatrix, Mat rvec, Mat tvec,
                   int markerBorder) {
    // canonical image
    Mat markerImg;
    const int markerSizePixels = 100;
    aruco::generateImageMarker(board.getDictionary(), board.getIds()[markerIndex], markerSizePixels, markerImg, markerBorder);

    // projected corners
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    vector<Point2f> corners;

    // get max coordinate of board
    Point3f maxCoord = board.getRightBottomCorner();
    // copy objPoints
    vector<Point3f> objPoints = board.getObjPoints()[markerIndex];
    // move the marker to the origin
    for (size_t i = 0; i < objPoints.size(); i++)
        objPoints[i] -= (maxCoord / 2.f);

    projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    // get perspective transform
    vector<Point2f> originalCorners;
    originalCorners.push_back(Point2f(0, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, (float)markerSizePixels));
    originalCorners.push_back(Point2f(0, (float)markerSizePixels));
    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    // apply transformation
    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, img.size(), INTER_NEAREST, BORDER_CONSTANT,
        Scalar::all(borderValue));

    // copy only not-border pixels
    for (int y = 0; y < aux.rows; y++) {
        for (int x = 0; x < aux.cols; x++) {
            if (aux.at< unsigned char >(y, x) == borderValue) continue;
            img.at< unsigned char >(y, x) = aux.at< unsigned char >(y, x);
        }
    }
}

Mat projectBoard(const aruco::GridBoard& board, Mat cameraMatrix, double yaw, double pitch, double distance,
                 Size imageSize, int markerBorder) {
    Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    Mat img = Mat(imageSize, CV_8UC1, Scalar::all(255));
    for (unsigned int index = 0; index < board.getIds().size(); index++)
        projectMarker(img, board, index, cameraMatrix, rvec, tvec, markerBorder);
    return img;
}

/** Check if a set of 3d points are enough for calibration. Z coordinate is ignored.
 * Only axis parallel lines are considered */
static bool _arePointsEnoughForPoseEstimation(const std::vector<Point3f> &points) {
    if(points.size() < 4) return false;

    std::vector<double> sameXValue; // different x values in points
    std::vector<int> sameXCounter;  // number of points with the x value in sameXValue
    for(unsigned int i = 0; i < points.size(); i++) {
        bool found = false;
        for(unsigned int j = 0; j < sameXValue.size(); j++) {
            if(sameXValue[j] == points[i].x) {
                found = true;
                sameXCounter[j]++;
            }
        }
        if(!found) {
            sameXValue.push_back(points[i].x);
            sameXCounter.push_back(1);
        }
    }

    // count how many x values has more than 2 points
    int moreThan2 = 0;
    for(unsigned int i = 0; i < sameXCounter.size(); i++) {
        if(sameXCounter[i] >= 2) moreThan2++;
    }

    // if we have more than 1 two xvalues with more than 2 points, calibration is ok
    if(moreThan2 > 1)
        return true;
    return false;
}

bool getCharucoBoardPose(InputArray charucoCorners, InputArray charucoIds,  const aruco::CharucoBoard &board,
                         InputArray cameraMatrix, InputArray distCoeffs, InputOutputArray rvec, InputOutputArray tvec,
                         bool useExtrinsicGuess) {
    CV_Assert((charucoCorners.getMat().total() == charucoIds.getMat().total()));
    if(charucoIds.getMat().total() < 4) return false; // need, at least, 4 corners

    std::vector<Point3f> objPoints;
    objPoints.reserve(charucoIds.getMat().total());
    for(unsigned int i = 0; i < charucoIds.getMat().total(); i++) {
        int currId = charucoIds.getMat().at< int >(i);
        CV_Assert(currId >= 0 && currId < (int)board.getChessboardCorners().size());
        objPoints.push_back(board.getChessboardCorners()[currId]);
    }

    // points need to be in different lines, check if detected points are enough
    if(!_arePointsEnoughForPoseEstimation(objPoints)) return false;

    solvePnP(objPoints, charucoCorners, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess);
    return true;
}

/**
  * @brief Return object points for the system centered in a middle (by default) or in a top left corner of single
  * marker, given the marker length
  */
static Mat _getSingleMarkerObjectPoints(float markerLength, bool use_aruco_ccw_center) {
    CV_Assert(markerLength > 0);
    Mat objPoints(4, 1, CV_32FC3);
    // set coordinate system in the top-left corner of the marker, with Z pointing out
    if (use_aruco_ccw_center) {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);
    }
    else {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(0.f, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength, markerLength, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(0.f, markerLength, 0);
    }
    return objPoints;
}

void getMarkersPoses(InputArrayOfArrays corners, float markerLength, InputArray cameraMatrix, InputArray distCoeffs,
                     OutputArray _rvecs, OutputArray _tvecs, OutputArray objPoints,
                     bool use_aruco_ccw_center, SolvePnPMethod solvePnPMethod) {
    CV_Assert(markerLength > 0);
    Mat markerObjPoints = _getSingleMarkerObjectPoints(markerLength, use_aruco_ccw_center);
    int nMarkers = (int)corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);

    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();
    for (int i = 0; i < nMarkers; i++)
        solvePnP(markerObjPoints, corners.getMat(i), cameraMatrix, distCoeffs, rvecs.at<Vec3d>(i), tvecs.at<Vec3d>(i),
                false, solvePnPMethod);

    if(objPoints.needed())
        markerObjPoints.convertTo(objPoints, -1);
}

}
