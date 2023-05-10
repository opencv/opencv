// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/calib3d.hpp"

namespace opencv_test {

static inline double deg2rad(double deg) { return deg * CV_PI / 180.; }

vector<Point2f> getAxis(InputArray _cameraMatrix, InputArray _distCoeffs, InputArray _rvec, InputArray _tvec,
                        float length, const Point2f offset = Point2f(0, 0));

vector<Point2f> getMarkerById(int id, const vector<vector<Point2f> >& corners, const vector<int>& ids);

/**
 * @brief Get rvec and tvec from yaw, pitch and distance
 */
void getSyntheticRT(double yaw, double pitch, double distance, Mat& rvec, Mat& tvec);

/**
 * @brief Project a synthetic marker
 */
void projectMarker(Mat& img, const aruco::Board& board, int markerIndex, Mat cameraMatrix, Mat rvec, Mat tvec,
                   int markerBorder);

/**
 * @brief Get a synthetic image of GridBoard in perspective
 */
Mat projectBoard(const aruco::GridBoard& board, Mat cameraMatrix, double yaw, double pitch, double distance,
                 Size imageSize, int markerBorder);

bool getCharucoBoardPose(InputArray charucoCorners, InputArray charucoIds,  const aruco::CharucoBoard &board,
                         InputArray cameraMatrix, InputArray distCoeffs, InputOutputArray rvec,
                         InputOutputArray tvec, bool useExtrinsicGuess = false);

void getMarkersPoses(InputArrayOfArrays corners, float markerLength, InputArray cameraMatrix, InputArray distCoeffs,
                     OutputArray _rvecs, OutputArray _tvecs, OutputArray objPoints = noArray(),
                     bool use_aruco_ccw_center = true, SolvePnPMethod solvePnPMethod = SolvePnPMethod::SOLVEPNP_ITERATIVE);

}
