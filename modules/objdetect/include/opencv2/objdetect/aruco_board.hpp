// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_ARUCO_BOARD_HPP
#define OPENCV_OBJDETECT_ARUCO_BOARD_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace aruco {
//! @addtogroup objdetect_aruco
//! @{

class Dictionary;

/** @brief Board of ArUco markers
 *
 * A board is a set of markers in the 3D space with a common coordinate system.
 * The common form of a board of marker is a planar (2D) board, however any 3D layout can be used.
 * A Board object is composed by:
 * - The object points of the marker corners, i.e. their coordinates respect to the board system.
 * - The dictionary which indicates the type of markers of the board
 * - The identifier of all the markers in the board.
 */
class CV_EXPORTS_W_SIMPLE Board {
public:
    /** @brief Common Board constructor
     *
     * @param objPoints array of object points of all the marker corners in the board
     * @param dictionary the dictionary of markers employed for this board
     * @param ids vector of the identifiers of the markers in the board
     */
    CV_WRAP Board(InputArrayOfArrays objPoints, const Dictionary& dictionary, InputArray ids);

    /** @brief return the Dictionary of markers employed for this board
     */
    CV_WRAP const Dictionary& getDictionary() const;

    /** @brief return array of object points of all the marker corners in the board.
     *
     * Each marker include its 4 corners in this order:
     * -   objPoints[i][0] - left-top point of i-th marker
     * -   objPoints[i][1] - right-top point of i-th marker
     * -   objPoints[i][2] - right-bottom point of i-th marker
     * -   objPoints[i][3] - left-bottom point of i-th marker
     *
     * Markers are placed in a certain order - row by row, left to right in every row. For M markers, the size is Mx4.
     */
    CV_WRAP const std::vector<std::vector<Point3f> >& getObjPoints() const;

    /** @brief vector of the identifiers of the markers in the board (should be the same size as objPoints)
     * @return vector of the identifiers of the markers
     */
    CV_WRAP const std::vector<int>& getIds() const;

    /** @brief get coordinate of the bottom right corner of the board, is set when calling the function create()
     */
    CV_WRAP const Point3f& getRightBottomCorner() const;

    /** @brief Given a board configuration and a set of detected markers, returns the corresponding
     * image points and object points, can be used in solvePnP()
     *
     * @param detectedCorners List of detected marker corners of the board.
     * For cv::Board and cv::GridBoard the method expects std::vector<std::vector<Point2f>> or std::vector<Mat> with Aruco marker corners.
     * For cv::CharucoBoard the method expects std::vector<Point2f> or Mat with ChAruco corners (chess board corners matched with Aruco markers).
     *
     * @param detectedIds List of identifiers for each marker or charuco corner.
     * For any Board class the method expects std::vector<int> or Mat.
     *
     * @param objPoints Vector of marker points in the board coordinate space.
     * For any Board class the method expects std::vector<cv::Point3f> objectPoints or cv::Mat
     *
     * @param imgPoints Vector of marker points in the image coordinate space.
     * For any Board class the method expects std::vector<cv::Point2f> objectPoints or cv::Mat
     *
     * @sa solvePnP
     */
    CV_WRAP void matchImagePoints(InputArrayOfArrays detectedCorners, InputArray detectedIds,
                                  OutputArray objPoints, OutputArray imgPoints) const;

     /** @brief Draw a planar board
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the board, ready to be printed.
     */
    CV_WRAP void generateImage(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1) const;

    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    Board();

    struct Impl;
protected:
    Board(const Ptr<Impl>& impl);
    Ptr<Impl> impl;
};

/** @brief Planar board with grid arrangement of markers
 *
 * More common type of board. All markers are placed in the same plane in a grid arrangement.
 * The board image can be drawn using generateImage() method.
 */
class CV_EXPORTS_W_SIMPLE GridBoard : public Board {
public:
    /**
     * @brief GridBoard constructor
     *
     * @param size number of markers in x and y directions
     * @param markerLength marker side length (normally in meters)
     * @param markerSeparation separation between two markers (same unit as markerLength)
     * @param dictionary dictionary of markers indicating the type of markers
     * @param ids set of marker ids in dictionary to use on board.
     */
    CV_WRAP GridBoard(const Size& size, float markerLength, float markerSeparation,
                      const Dictionary &dictionary, InputArray ids = noArray());

    CV_WRAP Size getGridSize() const;
    CV_WRAP float getMarkerLength() const;
    CV_WRAP float getMarkerSeparation() const;

    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    GridBoard();
};

/**
 * @brief ChArUco board is a planar chessboard where the markers are placed inside the white squares of a chessboard.
 *
 * The benefits of ChArUco boards is that they provide both, ArUco markers versatility and chessboard corner precision,
 * which is important for calibration and pose estimation. The board image can be drawn using generateImage() method.
 */
class CV_EXPORTS_W_SIMPLE CharucoBoard : public Board {
public:
    /** @brief CharucoBoard constructor
     *
     * @param size number of chessboard squares in x and y directions
     * @param squareLength squareLength chessboard square side length (normally in meters)
     * @param markerLength marker side length (same unit than squareLength)
     * @param dictionary dictionary of markers indicating the type of markers
     * @param ids array of id used markers
     * The first markers in the dictionary are used to fill the white chessboard squares.
     */
    CV_WRAP CharucoBoard(const Size& size, float squareLength, float markerLength,
                         const Dictionary &dictionary, InputArray ids = noArray());

    /** @brief set legacy chessboard pattern.
     *
     * In OpenCV 4.6.0 the ChArUco pattern generation algorithm changed in an incompatible way for
     * boards with an even number of rows. Enable this setting to interpret a board that was created
     * with an OpenCV version older than 4.6.0, or with a third-party generator that follows the same
     * convention; boards from calib.io, for example, use the legacy layout.
     *
     * The legacy layout starts the chessboard with a white box in the upper left corner when the row
     * count is even, whereas the current layout starts with a black box. The two layouts differ only
     * in that case: when `size.height` is odd they are identical, so this setting has no effect on
     * boards with an odd number of rows.
     *
     * @note Getting this wrong fails quietly. The ArUco markers are still detected normally, but the
     * corners they imply do not match the board, so CharucoDetector::detectBoard() returns zero
     * ChArUco corners while reporting a full set of marker detections. The only trace is a
     * `CV_LOG_DEBUG` message, which is not shown at the default log level. If you see detected
     * markers and no corners on a board with an even row count, try toggling this setting before
     * looking anywhere else. See https://github.com/opencv/opencv/issues/29636 for an example.
     *
     * See https://github.com/opencv/opencv/issues/23152.
     *
     * Default value: false.
     */
    CV_WRAP void setLegacyPattern(bool legacyPattern);
    CV_WRAP bool getLegacyPattern() const;

    CV_WRAP Size getChessboardSize() const;
    CV_WRAP float getSquareLength() const;
    CV_WRAP float getMarkerLength() const;

    /** @brief get CharucoBoard::chessboardCorners
     */
    CV_WRAP std::vector<Point3f> getChessboardCorners() const;

    /** @brief get CharucoBoard::nearestMarkerIdx, for each charuco corner, nearest marker index in ids array
     */
    CV_PROP std::vector<std::vector<int> > getNearestMarkerIdx() const;

    /** @brief get CharucoBoard::nearestMarkerCorners, for each charuco corner, nearest marker corner id of each marker
     */
    CV_PROP std::vector<std::vector<int> > getNearestMarkerCorners() const;

    /** @brief check whether the ChArUco markers are collinear
     *
     * @param charucoIds list of identifiers for each corner in charucoCorners per frame.
     * @return bool value, 1 (true) if detected corners form a line, 0 (false) if they do not.
     * solvePnP, calibration functions will fail if the corners are collinear (true).
     *
     * The number of ids in charucoIDs should be <= the number of chessboard corners in the board.
     * This functions checks whether the charuco corners are on a straight line (returns true, if so), or not (false).
     * Axis parallel, as well as diagonal and other straight lines detected.  Degenerate cases:
     * for number of charucoIDs <= 2,the function returns true.
     */
    CV_WRAP bool checkCharucoCornersCollinear(InputArray charucoIds) const;

    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    CharucoBoard();
};

//! @}

}
}

#endif
