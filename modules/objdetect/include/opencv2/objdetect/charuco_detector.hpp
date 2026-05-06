// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_CHARUCO_DETECTOR_HPP
#define OPENCV_OBJDETECT_CHARUCO_DETECTOR_HPP

#include "opencv2/objdetect/aruco_detector.hpp"

namespace cv {
namespace aruco {

//! @addtogroup objdetect_aruco
//! @{

struct CV_EXPORTS_W_SIMPLE CharucoParameters {
    CV_WRAP CharucoParameters() {
        minMarkers = 2;
        tryRefineMarkers = false;
        checkMarkers = true;
    }
    /// cameraMatrix optional 3x3 floating-point camera matrix
    CV_PROP_RW Mat cameraMatrix;

    /// distCoeffs optional vector of distortion coefficients
    CV_PROP_RW Mat distCoeffs;

    /// minMarkers number of adjacent markers that must be detected to return a charuco corner, default = 2
    CV_PROP_RW int minMarkers;

    /// try to use refine board, default false
    CV_PROP_RW bool tryRefineMarkers;

    /// run check to verify that markers belong to the same board, default true
    CV_PROP_RW bool checkMarkers;
};

class CV_EXPORTS_W CharucoDetector : public Algorithm {
public:
    /** @brief Basic CharucoDetector constructor
     *
     * @param board ChAruco board
     * @param charucoParams charuco detection parameters (used only for CHARUCO_1; ignored for CHARUCO_2)
     * @param detectorParams marker detection parameters (used only for CHARUCO_1; ignored for CHARUCO_2)
     * @param refineParams marker refine detection parameters (used only for CHARUCO_1; ignored for CHARUCO_2)
     */
    CV_WRAP CharucoDetector(const CharucoBoard& board,
                            const CharucoParameters& charucoParams = CharucoParameters(),
                            const DetectorParameters &detectorParams = DetectorParameters(),
                            const RefineParameters& refineParams = RefineParameters());

    CV_WRAP const CharucoBoard& getBoard() const;
    CV_WRAP void setBoard(const CharucoBoard& board);

    CV_WRAP const CharucoParameters& getCharucoParameters() const;
    CV_WRAP void setCharucoParameters(CharucoParameters& charucoParameters);

    CV_WRAP const DetectorParameters& getDetectorParameters() const;
    CV_WRAP void setDetectorParameters(const DetectorParameters& detectorParameters);

    CV_WRAP const RefineParameters& getRefineParameters() const;
    CV_WRAP void setRefineParameters(const RefineParameters& refineParameters);

    /**
     * @brief detect ArUco markers and interpolate position of ChArUco board corners
     * @param image input image.
     * @param charucoCorners interpolated chessboard corners.
     * @param charucoIds interpolated chessboard corners identifiers.
     * @param markerCorners vector of already detected markers corners. For each marker, its four
     * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
     * dimensions of this array should be Nx4. The order of the corners should be clockwise.
     * Used only for CHARUCO_1; for CHARUCO_2 all markers are always detected internally and this
     * parameter is ignored. If empty (CHARUCO_1 only), the function detects aruco markers automatically.
     * @param markerIds list of identifiers for each marker in corners.
     * Used only for CHARUCO_1; ignored for CHARUCO_2.
     * If empty (CHARUCO_1 only), the function detects aruco markers automatically.
     *
     * This function returns the 2D position of the chessboard corners from a ChArUco board.
     * For CHARUCO_1, detection can optionally be seeded with pre-detected marker corners; if camera
     * parameters are provided the process uses an approximated pose estimation, otherwise local
     * homography. For CHARUCO_2, markers are always detected internally from the image.
     * Only visible corners are returned. For each corner, its corresponding identifier is also returned in charucoIds.
     * @sa findChessboardCorners
     * @note After OpenCV 4.6.0, there was an incompatible change in the ChArUco pattern generation algorithm for even row counts.
     * Use cv::aruco::CharucoBoard::setLegacyPattern() to ensure compatibility with patterns created using OpenCV versions prior to 4.6.0.
     * For more information, see the issue: https://github.com/opencv/opencv/issues/23152
     */
    CV_WRAP void detectBoard(InputArray image, OutputArray charucoCorners, OutputArray charucoIds,
                             InputOutputArrayOfArrays markerCorners = noArray(),
                             InputOutputArray markerIds = noArray()) const;

    /**
     * @brief Detect ChArUco Diamond markers
     *
     * @param image input image necessary for corner subpixel.
     * @param diamondCorners output list of detected diamond corners per diamond. The format depends
     * on the board type:
     * - CHARUCO_1: 4 corners per diamond (the interior chessboard intersections), ordered
     *   top-left, top-right, bottom-right, bottom-left.
     * - CHARUCO_2: 9 corners per diamond (the full 3x3 grid of corner intersections of the
     *   2x2 marker board), in row-major order from top-left to bottom-right.
     * @param diamondIds ids of the diamonds in diamondCorners. The id of each diamond is of
     * type Vec4i, containing the ids of the four ArUco markers composing the diamond.
     * @param markerCorners list of detected marker corners. Used only for CHARUCO_1; for CHARUCO_2
     * all markers are always detected internally and this parameter is ignored.
     * If empty (CHARUCO_1 only), the function detects aruco markers automatically.
     * @param markerIds list of marker ids in markerCorners. Used only for CHARUCO_1; ignored for CHARUCO_2.
     * If empty (CHARUCO_1 only), the function detects aruco markers automatically.
     *
     * This function detects Diamond markers. For CHARUCO_1, the detector board must be of size (3,3)
     * and detection can be seeded with pre-detected marker corners. For CHARUCO_2, a 2x2 board is
     * used and all markers are detected internally from the image.
     * The two types produce different corner counts and coordinate origins; see CharucoBoardType.
     */
    CV_WRAP void detectDiamonds(InputArray image, OutputArrayOfArrays diamondCorners, OutputArray diamondIds,
                                InputOutputArrayOfArrays markerCorners = noArray(),
                                InputOutputArray markerIds = noArray()) const;
protected:
    struct CharucoBaseDetectorImpl;
    Ptr<CharucoBaseDetectorImpl> charucoDetectorImpl;
    friend struct Charuco1DetectorImpl;
    friend struct Charuco2DetectorImpl;

};

/**
 * @brief Draws a set of Charuco corners
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
 * altered.
 * @param charucoCorners vector of detected charuco corners
 * @param charucoIds list of identifiers for each corner in charucoCorners
 * @param cornerColor color of the square surrounding each corner
 *
 * This function draws a set of detected Charuco corners. If identifiers vector is provided, it also
 * draws the id of each corner.
 */
CV_EXPORTS_W void drawDetectedCornersCharuco(InputOutputArray image, InputArray charucoCorners,
                                             InputArray charucoIds = noArray(), Scalar cornerColor = Scalar(255, 0, 0));

/**
 * @brief Draw a set of detected ChArUco Diamond markers
 *
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
 * altered.
 * @param diamondCorners positions of diamond corners in the same format returned by
 * detectDiamonds(). (e.g std::vector<std::vector<cv::Point2f> > ). For N detected diamonds,
 * each entry has 4 corners (CHARUCO_1) or 9 corners (CHARUCO_2).
 * @param diamondIds vector of identifiers for diamonds in diamondCorners, in the same format
 * returned by detectCharucoDiamond() (e.g. std::vector<Vec4i>).
 * Optional, if not provided, ids are not painted.
 * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
 * are calculated based on this one.
 *
 * Given an array of detected diamonds, this functions draws them in the image. The marker borders
 * are painted and the markers identifiers if provided.
 * Useful for debugging purposes.
 */
CV_EXPORTS_W void drawDetectedDiamonds(InputOutputArray image, InputArrayOfArrays diamondCorners,
                                       InputArray diamondIds = noArray(),
                                       Scalar borderColor = Scalar(0, 0, 255));

//! @}

}
}

#endif
