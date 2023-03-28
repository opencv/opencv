// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_ARUCO_CALIB_POSE_HPP
#define OPENCV_ARUCO_CALIB_POSE_HPP
#include <opencv2/objdetect/aruco_board.hpp>

namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{

/** @brief rvec/tvec define the right handed coordinate system of the marker.
 *
 * PatternPositionType defines center this system and axes direction.
 * Axis X (red color) - first coordinate, axis Y (green color) - second coordinate,
 * axis Z (blue color) - third coordinate.
 * @sa estimatePoseSingleMarkers(), check tutorial_aruco_detection in aruco contrib
 */
enum PatternPositionType {
    /** @brief The marker coordinate system is centered on the middle of the marker.
     *
     * The coordinates of the four corners (CCW order) of the marker in its own coordinate system are:
     * (-markerLength/2, markerLength/2, 0), (markerLength/2, markerLength/2, 0),
     * (markerLength/2, -markerLength/2, 0), (-markerLength/2, -markerLength/2, 0).
     *
     * These pattern points define this coordinate system:
     * ![Image with axes drawn](tutorials/images/singlemarkersaxes.jpg)
     */
    ARUCO_CCW_CENTER,
    /** @brief The marker coordinate system is centered on the top-left corner of the marker.
     *
     * The coordinates of the four corners (CW order) of the marker in its own coordinate system are:
     * (0, 0, 0), (markerLength, 0, 0),
     * (markerLength, markerLength, 0), (0, markerLength, 0).
     *
     * These pattern points define this coordinate system:
     * ![Image with axes drawn](tutorials/images/singlemarkersaxes2.jpg)
     *
     * These pattern dots are convenient to use with a chessboard/ChArUco board.
     */
    ARUCO_CW_TOP_LEFT_CORNER
};

/** @brief Pose estimation parameters
 *
 * @param pattern Defines center this system and axes direction (default PatternPositionType::ARUCO_CCW_CENTER).
 * @param useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses the provided
 * rvec and tvec values as initial approximations of the rotation and translation vectors, respectively, and further
 * optimizes them (default false).
 * @param solvePnPMethod Method for solving a PnP problem: see @ref calib3d_solvePnP_flags (default SOLVEPNP_ITERATIVE).
 * @sa PatternPositionType, solvePnP(), check tutorial_aruco_detection in aruco contrib
 */
struct CV_EXPORTS_W_SIMPLE EstimateParameters {
    CV_PROP_RW PatternPositionType pattern;
    CV_PROP_RW bool useExtrinsicGuess;
    CV_PROP_RW int solvePnPMethod;

    CV_WRAP EstimateParameters();
};

/**
 * @brief Calibrate a camera using aruco markers
 *
 * @param corners vector of detected marker corners in all frames.
 * The corners should have the same format returned by detectMarkers (see #detectMarkers).
 * @param ids list of identifiers for each marker in corners
 * @param counter number of markers in each frame so that corners and ids can be split
 * @param board Marker Board layout
 * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
 * @param cameraMatrix Output 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
 * and/or CV_CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
 * initialized before calling the function.
 * @param distCoeffs Output vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each board view
 * (e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
 * k-th translation vector (see the next output parameter description) brings the board pattern
 * from the model coordinate space (in which object points are specified) to the world coordinate
 * space, that is, a real position of the board pattern in the k-th pattern view (k=0.. *M* -1).
 * @param tvecs Output vector of translation vectors estimated for each pattern view.
 * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
 * Order of deviations values:
 * \f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 * s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
 * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
 * Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
 * \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 * @param perViewErrors Output vector of average re-projection errors estimated for each pattern view.
 * @param flags flags Different flags  for the calibration process (see #calibrateCamera for details).
 * @param criteria Termination criteria for the iterative optimization algorithm.
 *
 * This function calibrates a camera using an Aruco Board. The function receives a list of
 * detected markers from several views of the Board. The process is similar to the chessboard
 * calibration in calibrateCamera(). The function returns the final re-projection error.
 */
CV_EXPORTS_AS(calibrateCameraArucoExtended)
double calibrateCameraAruco(InputArrayOfArrays corners, InputArray ids, InputArray counter, const Ptr<Board> &board,
                            Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                            OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, OutputArray stdDeviationsIntrinsics,
                            OutputArray stdDeviationsExtrinsics, OutputArray perViewErrors, int flags = 0,
                            const TermCriteria& criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));

/** @overload
 * @brief It's the same function as #calibrateCameraAruco but without calibration error estimation.
 */
CV_EXPORTS_W double calibrateCameraAruco(InputArrayOfArrays corners, InputArray ids, InputArray counter,
                                         const Ptr<Board> &board, Size imageSize, InputOutputArray cameraMatrix,
                                         InputOutputArray distCoeffs, OutputArrayOfArrays rvecs = noArray(),
                                         OutputArrayOfArrays tvecs = noArray(), int flags = 0,
                                         const TermCriteria& criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                                                                                     30, DBL_EPSILON));


/**
 * @brief Calibrate a camera using Charuco corners
 *
 * @param charucoCorners vector of detected charuco corners per frame
 * @param charucoIds list of identifiers for each corner in charucoCorners per frame
 * @param board Marker Board layout
 * @param imageSize input image size
 * @param cameraMatrix Output 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
 * and/or CV_CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
 * initialized before calling the function.
 * @param distCoeffs Output vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each board view
 * (e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
 * k-th translation vector (see the next output parameter description) brings the board pattern
 * from the model coordinate space (in which object points are specified) to the world coordinate
 * space, that is, a real position of the board pattern in the k-th pattern view (k=0.. *M* -1).
 * @param tvecs Output vector of translation vectors estimated for each pattern view.
 * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
 * Order of deviations values:
 * \f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 * s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
 * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
 * Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
 * \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 * @param perViewErrors Output vector of average re-projection errors estimated for each pattern view.
 * @param flags flags Different flags  for the calibration process (see #calibrateCamera for details).
 * @param criteria Termination criteria for the iterative optimization algorithm.
 *
 * This function calibrates a camera using a set of corners of a  Charuco Board. The function
 * receives a list of detected corners and its identifiers from several views of the Board.
 * The function returns the final re-projection error.
 */
CV_EXPORTS_AS(calibrateCameraCharucoExtended)
double calibrateCameraCharuco(InputArrayOfArrays charucoCorners, InputArrayOfArrays charucoIds,
                              const Ptr<CharucoBoard> &board, Size imageSize, InputOutputArray cameraMatrix,
                              InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                              OutputArray stdDeviationsIntrinsics, OutputArray stdDeviationsExtrinsics,
                              OutputArray perViewErrors, int flags = 0, const TermCriteria& criteria = TermCriteria(
                              TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));

/**
 * @brief It's the same function as #calibrateCameraCharuco but without calibration error estimation.
 */
CV_EXPORTS_W double calibrateCameraCharuco(InputArrayOfArrays charucoCorners, InputArrayOfArrays charucoIds,
                                           const Ptr<CharucoBoard> &board, Size imageSize,
                                           InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                           OutputArrayOfArrays rvecs = noArray(),
                                           OutputArrayOfArrays tvecs = noArray(), int flags = 0,
                                           const TermCriteria& criteria=TermCriteria(TermCriteria::COUNT +
                                                                 TermCriteria::EPS, 30, DBL_EPSILON));
//! @}

}
}
#endif
