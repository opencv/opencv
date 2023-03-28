// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_ARUCO_HPP
#define OPENCV_ARUCO_HPP

#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/aruco/aruco_calib.hpp"

namespace cv {
namespace aruco {

/**
 * @defgroup aruco Aruco markers, module functionality was moved to objdetect module
 * @{
 * ArUco Marker Detection, module functionality was moved to objdetect module
 * @sa ArucoDetector, CharucoDetector, Board, GridBoard, CharucoBoard
 * @}
 */

//! @addtogroup aruco
//! @{

/** @brief detect markers
@deprecated Use class ArucoDetector::detectMarkers
*/
CV_EXPORTS_W void detectMarkers(InputArray image, const Ptr<Dictionary> &dictionary, OutputArrayOfArrays corners,
                                OutputArray ids, const Ptr<DetectorParameters> &parameters = makePtr<DetectorParameters>(),
                                OutputArrayOfArrays rejectedImgPoints = noArray());

/** @brief refine detected markers
@deprecated Use class ArucoDetector::refineDetectedMarkers
*/
CV_EXPORTS_W void refineDetectedMarkers(InputArray image,const  Ptr<Board> &board,
                                        InputOutputArrayOfArrays detectedCorners,
                                        InputOutputArray detectedIds, InputOutputArrayOfArrays rejectedCorners,
                                        InputArray cameraMatrix = noArray(), InputArray distCoeffs = noArray(),
                                        float minRepDistance = 10.f, float errorCorrectionRate = 3.f,
                                        bool checkAllOrders = true, OutputArray recoveredIdxs = noArray(),
                                        const Ptr<DetectorParameters> &parameters = makePtr<DetectorParameters>());

/** @brief draw planar board
@deprecated Use Board::generateImage
*/
CV_EXPORTS_W void drawPlanarBoard(const Ptr<Board> &board, Size outSize, OutputArray img, int marginSize,
                                  int borderBits);

/** @brief get board object and image points
@deprecated Use Board::matchImagePoints
*/
CV_EXPORTS_W void getBoardObjectAndImagePoints(const Ptr<Board> &board, InputArrayOfArrays detectedCorners,
                                               InputArray detectedIds, OutputArray objPoints, OutputArray imgPoints);


/** @deprecated Use cv::solvePnP
 */
CV_EXPORTS_W int estimatePoseBoard(InputArrayOfArrays corners, InputArray ids, const Ptr<Board> &board,
                                   InputArray cameraMatrix, InputArray distCoeffs, InputOutputArray rvec,
                                   InputOutputArray tvec, bool useExtrinsicGuess = false);

/**
 * @brief Pose estimation for a ChArUco board given some of their corners
 * @param charucoCorners vector of detected charuco corners
 * @param charucoIds list of identifiers for each corner in charucoCorners
 * @param board layout of ChArUco board.
 * @param cameraMatrix input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvec Output vector (e.g. cv::Mat) corresponding to the rotation vector of the board
 * (see cv::Rodrigues).
 * @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
 * @param useExtrinsicGuess defines whether initial guess for \b rvec and \b tvec will be used or not.
 *
 * This function estimates a Charuco board pose from some detected corners.
 * The function checks if the input corners are enough and valid to perform pose estimation.
 * If pose estimation is valid, returns true, else returns false.
 * @sa use cv::drawFrameAxes to get world coordinate system axis for object points
 */
CV_EXPORTS_W bool estimatePoseCharucoBoard(InputArray charucoCorners, InputArray charucoIds,
                                           const Ptr<CharucoBoard> &board, InputArray cameraMatrix,
                                           InputArray distCoeffs, InputOutputArray rvec,
                                           InputOutputArray tvec, bool useExtrinsicGuess = false);

/** @deprecated Use cv::solvePnP
 */
CV_EXPORTS_W void estimatePoseSingleMarkers(InputArrayOfArrays corners, float markerLength,
                                            InputArray cameraMatrix, InputArray distCoeffs,
                                            OutputArray rvecs, OutputArray tvecs, OutputArray objPoints = noArray(),
                                            const Ptr<EstimateParameters>& estimateParameters = makePtr<EstimateParameters>());


/** @deprecated Use CharucoBoard::checkCharucoCornersCollinear
 */
CV_EXPORTS_W bool testCharucoCornersCollinear(const Ptr<CharucoBoard> &board, InputArray charucoIds);

//! @}

}
}

#endif
