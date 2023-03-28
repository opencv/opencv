// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_CHARUCO_HPP
#define OPENCV_CHARUCO_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/aruco/aruco_calib.hpp>


namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{

/**
 * @brief Interpolate position of ChArUco board corners
 * @param markerCorners vector of already detected markers corners. For each marker, its four
 * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
 * dimensions of this array should be Nx4. The order of the corners should be clockwise.
 * @param markerIds list of identifiers for each marker in corners
 * @param image input image necesary for corner refinement. Note that markers are not detected and
 * should be sent in corners and ids parameters.
 * @param board layout of ChArUco board.
 * @param charucoCorners interpolated chessboard corners
 * @param charucoIds interpolated chessboard corners identifiers
 * @param cameraMatrix optional 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs optional vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param minMarkers number of adjacent markers that must be detected to return a charuco corner
 *
 * This function receives the detected markers and returns the 2D position of the chessboard corners
 * from a ChArUco board using the detected Aruco markers. If camera parameters are provided,
 * the process is based in an approximated pose estimation, else it is based on local homography.
 * Only visible corners are returned. For each corner, its corresponding identifier is
 * also returned in charucoIds.
 * The function returns the number of interpolated corners.
 *
 * @deprecated Use CharucoDetector::detectBoard
 */
CV_EXPORTS_W int interpolateCornersCharuco(InputArrayOfArrays markerCorners, InputArray markerIds,
                                           InputArray image, const Ptr<CharucoBoard> &board,
                                           OutputArray charucoCorners, OutputArray charucoIds,
                                           InputArray cameraMatrix = noArray(),
                                           InputArray distCoeffs = noArray(), int minMarkers = 2);

/**
 * @brief Detect ChArUco Diamond markers
 *
 * @param image input image necessary for corner subpixel.
 * @param markerCorners list of detected marker corners from detectMarkers function.
 * @param markerIds list of marker ids in markerCorners.
 * @param squareMarkerLengthRate rate between square and marker length:
 * squareMarkerLengthRate = squareLength/markerLength. The real units are not necessary.
 * @param diamondCorners output list of detected diamond corners (4 corners per diamond). The order
 * is the same than in marker corners: top left, top right, bottom right and bottom left. Similar
 * format than the corners returned by detectMarkers (e.g std::vector<std::vector<cv::Point2f> > ).
 * @param diamondIds ids of the diamonds in diamondCorners. The id of each diamond is in fact of
 * type Vec4i, so each diamond has 4 ids, which are the ids of the aruco markers composing the
 * diamond.
 * @param cameraMatrix Optional camera calibration matrix.
 * @param distCoeffs Optional camera distortion coefficients.
 * @param dictionary dictionary of markers indicating the type of markers.
 *
 * This function detects Diamond markers from the previous detected ArUco markers. The diamonds
 * are returned in the diamondCorners and diamondIds parameters. If camera calibration parameters
 * are provided, the diamond search is based on reprojection. If not, diamond search is based on
 * homography. Homography is faster than reprojection, but less accurate.
 *
 * @deprecated Use CharucoDetector::detectDiamonds
 */
CV_EXPORTS_W void detectCharucoDiamond(InputArray image, InputArrayOfArrays markerCorners,
                                       InputArray markerIds, float squareMarkerLengthRate,
                                       OutputArrayOfArrays diamondCorners, OutputArray diamondIds,
                                       InputArray cameraMatrix = noArray(),
                                       InputArray distCoeffs = noArray(),
                                       Ptr<Dictionary> dictionary = makePtr<Dictionary>
                                               (getPredefinedDictionary(PredefinedDictionaryType::DICT_4X4_50)));


/**
 * @brief Draw a ChArUco Diamond marker
 *
 * @param dictionary dictionary of markers indicating the type of markers.
 * @param ids list of 4 ids for each ArUco marker in the ChArUco marker.
 * @param squareLength size of the chessboard squares in pixels.
 * @param markerLength size of the markers in pixels.
 * @param img output image with the marker. The size of this image will be
 * 3*squareLength + 2*marginSize,.
 * @param marginSize minimum margins (in pixels) of the marker in the output image
 * @param borderBits width of the marker borders.
 *
 * This function return the image of a ChArUco marker, ready to be printed.
 */
CV_EXPORTS_W void drawCharucoDiamond(const Ptr<Dictionary> &dictionary, Vec4i ids, int squareLength,
                                     int markerLength, OutputArray img, int marginSize = 0,
                                     int borderBits = 1);

//! @}
}
}

#endif
