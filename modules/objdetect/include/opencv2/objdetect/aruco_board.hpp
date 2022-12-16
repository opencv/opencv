// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_ARUCO_BOARD_HPP
#define OPENCV_OBJDETECT_ARUCO_BOARD_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace aruco {
//! @addtogroup aruco
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
class CV_EXPORTS_W Board {
protected:
    Board(); // use ::create()
public:
    /** @brief Draw a planar board
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the GridBoard, ready to be printed.
     */
    CV_WRAP virtual void generateImage(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1) const;

    /** @brief Provide way to create Board by passing necessary data. Specially needed in Python.
     *
     * @param objPoints array of object points of all the marker corners in the board
     * @param dictionary the dictionary of markers employed for this board
     * @param ids vector of the identifiers of the markers in the board
     */
    CV_WRAP static Ptr<Board> create(InputArrayOfArrays objPoints, const Dictionary &dictionary, InputArray ids);

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
     * image points and object points to call solvePnP
     *
     * @param detectedCorners List of detected marker corners of the board.
     * @param detectedIds List of identifiers for each marker.
     * @param objPoints Vector of vectors of board marker points in the board coordinate space.
     * @param imgPoints Vector of vectors of the projections of board marker corner points.
     */
    CV_WRAP void matchImagePoints(InputArrayOfArrays detectedCorners, InputArray detectedIds,
                                  OutputArray objPoints, OutputArray imgPoints) const;
    virtual ~Board();
protected:
    struct BoardImpl;
    Ptr<BoardImpl> boardImpl;
};

/** @brief Planar board with grid arrangement of markers
 *
 * More common type of board. All markers are placed in the same plane in a grid arrangement.
 * The board image can be drawn using generateImage() method.
 */
class CV_EXPORTS_W GridBoard : public Board {
protected:
    GridBoard();
public:
    /** @brief Draw a GridBoard
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the GridBoard, ready to be printed.
     */
    CV_WRAP void generateImage(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1) const CV_OVERRIDE;

    /**
     * @brief Create a GridBoard object
     *
     * @param markersX number of markers in X direction
     * @param markersY number of markers in Y direction
     * @param markerLength marker side length (normally in meters)
     * @param markerSeparation separation between two markers (same unit as markerLength)
     * @param dictionary dictionary of markers indicating the type of markers
     * @param ids set marker ids in dictionary to use on board.
     * @return the output GridBoard object
     *
     * This functions creates a GridBoard object given the number of markers in each direction and
     * the marker size and marker separation.
     */
    CV_WRAP static Ptr<GridBoard> create(int markersX, int markersY, float markerLength, float markerSeparation,
                                         const Dictionary &dictionary, InputArray ids);

    /**
     * @overload
     * @brief Create a GridBoard object
     *
     * @param markersX number of markers in X direction
     * @param markersY number of markers in Y direction
     * @param markerLength marker side length (normally in meters)
     * @param markerSeparation separation between two markers (same unit as markerLength)
     * @param dictionary dictionary of markers indicating the type of markers
     * @param firstMarker id of first marker in dictionary to use on board.
     * @return the output GridBoard object
     */
    CV_WRAP static Ptr<GridBoard> create(int markersX, int markersY, float markerLength, float markerSeparation,
                                         const Dictionary &dictionary, int firstMarker = 0);

    CV_WRAP Size getGridSize() const;
    CV_WRAP float getMarkerLength() const;
    CV_WRAP float getMarkerSeparation() const;

protected:
    struct GridImpl;
    Ptr<GridImpl> gridImpl;
    friend class CharucoBoard;
};

/**
 * @brief ChArUco board is a planar chessboard where the markers are placed inside the white squares of a chessboard.
 *
 * The benefits of ChArUco boards is that they provide both, ArUco markers versatility and chessboard corner precision,
 * which is important for calibration and pose estimation. The board image can be drawn using generateImage() method.
 */
class CV_EXPORTS_W CharucoBoard : public Board {
protected:
    CharucoBoard();
public:

    /** @brief Draw a ChArUco board
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the ChArUco board, ready to be printed.
     */
    CV_WRAP void generateImage(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1) const CV_OVERRIDE;


    /** @brief Create a CharucoBoard object
     *
     * @param squaresX number of chessboard squares in X direction
     * @param squaresY number of chessboard squares in Y direction
     * @param squareLength chessboard square side length (normally in meters)
     * @param markerLength marker side length (same unit than squareLength)
     * @param dictionary dictionary of markers indicating the type of markers.
     * @param ids array of id used markers
     * The first markers in the dictionary are used to fill the white chessboard squares.
     * @return the output CharucoBoard object
     *
     * This functions creates a CharucoBoard object given the number of squares in each direction
     * and the size of the markers and chessboard squares.
     */
    CV_WRAP static Ptr<CharucoBoard> create(int squaresX, int squaresY, float squareLength, float markerLength,
                                            const Dictionary &dictionary, InputArray ids = noArray());

    CV_WRAP Size getChessboardSize() const;
    CV_WRAP float getSquareLength() const;
    CV_WRAP float getMarkerLength() const;

    /** @brief get CharucoBoard::chessboardCorners
     */
    CV_WRAP std::vector<Point3f> getChessboardCorners() const;

    /** @brief get CharucoBoard::nearestMarkerIdx
     */
    CV_PROP std::vector<std::vector<int> > getNearestMarkerIdx() const;

    /** @brief get CharucoBoard::nearestMarkerCorners
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

protected:
    struct CharucoImpl;
    friend struct CharucoImpl;
    Ptr<CharucoImpl> charucoImpl;
};

//! @}

}
}

#endif
