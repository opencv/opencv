// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "opencv2/objdetect/aruco_board.hpp"

#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <numeric>

namespace cv {
namespace aruco {
using namespace std;

struct Board::Impl {
    Dictionary dictionary;
    std::vector<int> ids;
    std::vector<std::vector<Point3f> > objPoints;
    Point3f rightBottomBorder;

    explicit Impl(const Dictionary& _dictionary):
        dictionary(_dictionary)
    {}

    virtual ~Impl() {}

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    virtual void matchImagePoints(InputArray detectedCorners, InputArray detectedIds, OutputArray _objPoints,
                                  OutputArray imgPoints) const;

    virtual void generateImage(Size outSize, OutputArray img, int marginSize, int borderBits) const;
};

void Board::Impl::matchImagePoints(InputArray detectedCorners, InputArray detectedIds, OutputArray _objPoints,
                                        OutputArray imgPoints) const {

    CV_Assert(ids.size() == objPoints.size());
    CV_Assert(detectedIds.total() == detectedCorners.total());

    size_t nDetectedMarkers = detectedIds.total();

    vector<Point3f> objPnts;
    objPnts.reserve(nDetectedMarkers);

    vector<Point2f> imgPnts;
    imgPnts.reserve(nDetectedMarkers);

    // look for detected markers that belong to the board and get their information
    for(unsigned int i = 0; i < nDetectedMarkers; i++) {
        int currentId = detectedIds.getMat().ptr< int >(0)[i];
        for(unsigned int j = 0; j < ids.size(); j++) {
            if(currentId == ids[j]) {
                for(int p = 0; p < 4; p++) {
                    objPnts.push_back(objPoints[j][p]);
                    imgPnts.push_back(detectedCorners.getMat(i).ptr<Point2f>(0)[p]);
                }
            }
        }
    }

    // create output
    Mat(objPnts).copyTo(_objPoints);
    Mat(imgPnts).copyTo(imgPoints);
}

void Board::Impl::generateImage(Size outSize, OutputArray img, int marginSize, int borderBits) const {
    CV_Assert(!outSize.empty());
    CV_Assert(marginSize >= 0);

    img.create(outSize, CV_8UC1);
    Mat out = img.getMat();
    out.setTo(Scalar::all(255));
    out.adjustROI(-marginSize, -marginSize, -marginSize, -marginSize);

    // calculate max and min values in XY plane
    CV_Assert(objPoints.size() > 0);
    float minX, maxX, minY, maxY;
    minX = maxX = objPoints[0][0].x;
    minY = maxY = objPoints[0][0].y;

    for(unsigned int i = 0; i < objPoints.size(); i++) {
        for(int j = 0; j < 4; j++) {
            minX = min(minX, objPoints[i][j].x);
            maxX = max(maxX, objPoints[i][j].x);
            minY = min(minY, objPoints[i][j].y);
            maxY = max(maxY, objPoints[i][j].y);
        }
    }

    float sizeX = maxX - minX;
    float sizeY = maxY - minY;

    // proportion transformations
    float xReduction = sizeX / float(out.cols);
    float yReduction = sizeY / float(out.rows);

    // determine the zone where the markers are placed
    if(xReduction > yReduction) {
        int nRows = int(sizeY / xReduction);
        int rowsMargins = (out.rows - nRows) / 2;
        out.adjustROI(-rowsMargins, -rowsMargins, 0, 0);
    } else {
        int nCols = int(sizeX / yReduction);
        int colsMargins = (out.cols - nCols) / 2;
        out.adjustROI(0, 0, -colsMargins, -colsMargins);
    }

    // now paint each marker
    Mat marker;
    Point2f outCorners[3];
    Point2f inCorners[3];
    for(unsigned int m = 0; m < objPoints.size(); m++) {
        // transform corners to markerZone coordinates
        for(int j = 0; j < 3; j++) {
            Point2f pf = Point2f(objPoints[m][j].x, objPoints[m][j].y);
            // move top left to 0, 0
            pf -= Point2f(minX, minY);
            pf.x = pf.x / sizeX * float(out.cols);
            pf.y = pf.y / sizeY * float(out.rows);
            outCorners[j] = pf;
        }

        // get marker
        Size dst_sz(outCorners[2] - outCorners[0]); // assuming CCW order
        dst_sz.width = dst_sz.height = std::min(dst_sz.width, dst_sz.height); //marker should be square
        dictionary.generateImageMarker(ids[m], dst_sz.width, marker, borderBits);

        if((outCorners[0].y == outCorners[1].y) && (outCorners[1].x == outCorners[2].x)) {
            // marker is aligned to image axes
            marker.copyTo(out(Rect(outCorners[0], dst_sz)));
            continue;
        }

        // interpolate tiny marker to marker position in markerZone
        inCorners[0] = Point2f(-0.5f, -0.5f);
        inCorners[1] = Point2f(marker.cols - 0.5f, -0.5f);
        inCorners[2] = Point2f(marker.cols - 0.5f, marker.rows - 0.5f);

        // remove perspective
        Mat transformation = getAffineTransform(inCorners, outCorners);
        warpAffine(marker, out, transformation, out.size(), INTER_LINEAR,
                        BORDER_TRANSPARENT);
    }
}

Board::Board(const Ptr<Impl>& _impl):
    impl(_impl)
{
    CV_Assert(impl);
}

Board::Board():
    impl(nullptr)
{}

Board::Board(InputArrayOfArrays objPoints, const Dictionary &dictionary, InputArray ids):
    Board(new Board::Impl(dictionary)) {
    CV_Assert(ids.size() == objPoints.size());
    CV_Assert(objPoints.total() == ids.total());
    CV_Assert(objPoints.type() == CV_32FC3 || objPoints.type() == CV_32FC1);

    vector<vector<Point3f> > obj_points_vector;
    Point3f rightBottomBorder = Point3f(0.f, 0.f, 0.f);
    for (unsigned int i = 0; i < objPoints.total(); i++) {
        vector<Point3f> corners;
        Mat corners_mat = objPoints.getMat(i);

        if (corners_mat.type() == CV_32FC1)
            corners_mat = corners_mat.reshape(3);
        CV_Assert(corners_mat.total() == 4);

        for (int j = 0; j < 4; j++) {
            const Point3f &corner = corners_mat.at<Point3f>(j);
            corners.push_back(corner);
            rightBottomBorder.x = std::max(rightBottomBorder.x, corner.x);
            rightBottomBorder.y = std::max(rightBottomBorder.y, corner.y);
            rightBottomBorder.z = std::max(rightBottomBorder.z, corner.z);
        }
        obj_points_vector.push_back(corners);
    }

    ids.copyTo(impl->ids);
    impl->objPoints = obj_points_vector;
    impl->rightBottomBorder = rightBottomBorder;
}

const Dictionary& Board::getDictionary() const {
    CV_Assert(this->impl);
    return this->impl->dictionary;
}

const vector<vector<Point3f> >& Board::getObjPoints() const {
    CV_Assert(this->impl);
    return this->impl->objPoints;
}

const Point3f& Board::getRightBottomCorner() const {
    CV_Assert(this->impl);
    return this->impl->rightBottomBorder;
}

const vector<int>& Board::getIds() const {
    CV_Assert(this->impl);
    return this->impl->ids;
}

/** @brief Implementation of draw planar board that accepts a raw Board pointer.
 */
void Board::generateImage(Size outSize, OutputArray img, int marginSize, int borderBits) const {
    CV_Assert(this->impl);
    impl->generateImage(outSize, img, marginSize, borderBits);
}

void Board::matchImagePoints(InputArray detectedCorners, InputArray detectedIds, OutputArray objPoints,
                             OutputArray imgPoints) const {
    CV_Assert(this->impl);
    impl->matchImagePoints(detectedCorners, detectedIds, objPoints, imgPoints);
}

struct GridBoardImpl : public Board::Impl {
    GridBoardImpl(const Dictionary& _dictionary, const Size& _size, float _markerLength, float _markerSeparation):
        Board::Impl(_dictionary),
        size(_size),
        markerLength(_markerLength),
        markerSeparation(_markerSeparation)
    {
        CV_Assert(size.width*size.height > 0 && markerLength > 0 && markerSeparation > 0);
    }

    // number of markers in X and Y directions
    const Size size;
    // marker side length (normally in meters)
    float markerLength;
    // separation between markers in the grid
    float markerSeparation;
};

GridBoard::GridBoard() {}

GridBoard::GridBoard(const Size& size, float markerLength, float markerSeparation,
                     const Dictionary &dictionary, InputArray ids):
    Board(new GridBoardImpl(dictionary, size, markerLength, markerSeparation)) {

    size_t totalMarkers = (size_t) size.width*size.height;
    CV_Assert(ids.empty() || totalMarkers == ids.total());
    vector<vector<Point3f> > objPoints;
    objPoints.reserve(totalMarkers);

    if(!ids.empty()) {
        ids.copyTo(impl->ids);
    } else {
        impl->ids = std::vector<int>(totalMarkers);
        std::iota(impl->ids.begin(), impl->ids.end(), 0);
    }

    // calculate Board objPoints
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            vector <Point3f> corners(4);
            corners[0] = Point3f(x * (markerLength + markerSeparation),
                                 y * (markerLength + markerSeparation), 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, markerLength, 0);
            corners[3] = corners[0] + Point3f(0, markerLength, 0);
            objPoints.push_back(corners);
        }
    }
    impl->objPoints = objPoints;
    impl->rightBottomBorder = Point3f(size.width * markerLength + markerSeparation * (size.width - 1),
                                           size.height * markerLength + markerSeparation * (size.height - 1), 0.f);
}

Size GridBoard::getGridSize() const {
    CV_Assert(impl);
    return static_pointer_cast<GridBoardImpl>(impl)->size;
}

float GridBoard::getMarkerLength() const {
    CV_Assert(impl);
    return static_pointer_cast<GridBoardImpl>(impl)->markerLength;
}

float GridBoard::getMarkerSeparation() const {
    CV_Assert(impl);
    return static_pointer_cast<GridBoardImpl>(impl)->markerSeparation;
}

struct CharucoBoardImpl : Board::Impl {
    CharucoBoardImpl(const Dictionary& _dictionary, const Size& _size, float _squareLength, float _markerLength):
        Board::Impl(_dictionary),
        size(_size),
        squareLength(_squareLength),
        markerLength(_markerLength)
    {}

    // chessboard size
    Size size;

    // Physical size of chessboard squares side (normally in meters)
    float squareLength;

    // Physical marker side length (normally in meters)
    float markerLength;

    // vector of chessboard 3D corners precalculated
    std::vector<Point3f> chessboardCorners;

    // for each charuco corner, nearest marker id and nearest marker corner id of each marker
    std::vector<std::vector<int> > nearestMarkerIdx;
    std::vector<std::vector<int> > nearestMarkerCorners;

    void calcNearestMarkerCorners();

    void matchImagePoints(InputArrayOfArrays detectedCorners, InputArray detectedIds,
                          OutputArray objPoints, OutputArray imgPoints) const override;

    void generateImage(Size outSize, OutputArray img, int marginSize, int borderBits) const override;
};

/** Fill nearestMarkerIdx and nearestMarkerCorners arrays */
void CharucoBoardImpl::calcNearestMarkerCorners() {
    nearestMarkerIdx.resize(chessboardCorners.size());
    nearestMarkerCorners.resize(chessboardCorners.size());
    unsigned int nMarkers = (unsigned int)objPoints.size();
    unsigned int nCharucoCorners = (unsigned int)chessboardCorners.size();
    for(unsigned int i = 0; i < nCharucoCorners; i++) {
        double minDist = -1; // distance of closest markers
        Point3f charucoCorner = chessboardCorners[i];
        for(unsigned int j = 0; j < nMarkers; j++) {
            // calculate distance from marker center to charuco corner
            Point3f center = Point3f(0, 0, 0);
            for(unsigned int k = 0; k < 4; k++)
                center += objPoints[j][k];
            center /= 4.;
            double sqDistance;
            Point3f distVector = charucoCorner - center;
            sqDistance = distVector.x * distVector.x + distVector.y * distVector.y;
            if(j == 0 || fabs(sqDistance - minDist) < cv::pow(0.01 * squareLength, 2)) {
                // if same minimum distance (or first iteration), add to nearestMarkerIdx vector
                nearestMarkerIdx[i].push_back(j);
                minDist = sqDistance;
            } else if(sqDistance < minDist) {
                // if finding a closest marker to the charuco corner
                nearestMarkerIdx[i].clear(); // remove any previous added marker
                nearestMarkerIdx[i].push_back(j); // add the new closest marker index
                minDist = sqDistance;
            }
        }
        // for each of the closest markers, search the marker corner index closer
        // to the charuco corner
        for(unsigned int j = 0; j < nearestMarkerIdx[i].size(); j++) {
            nearestMarkerCorners[i].resize(nearestMarkerIdx[i].size());
            double minDistCorner = -1;
            for(unsigned int k = 0; k < 4; k++) {
                double sqDistance;
                Point3f distVector = charucoCorner - objPoints[nearestMarkerIdx[i][j]][k];
                sqDistance = distVector.x * distVector.x + distVector.y * distVector.y;
                if(k == 0 || sqDistance < minDistCorner) {
                    // if this corner is closer to the charuco corner, assing its index
                    // to nearestMarkerCorners
                    minDistCorner = sqDistance;
                    nearestMarkerCorners[i][j] = k;
                }
            }
        }
    }
}

void CharucoBoardImpl::matchImagePoints(InputArrayOfArrays detectedCorners, InputArray detectedIds,
                                   OutputArray _objPoints, OutputArray imgPoints) const {
    if (detectedCorners.kind() == _InputArray::STD_VECTOR_VECTOR ||
        detectedCorners.isMatVector() || detectedCorners.isUMatVector())
        Board::Impl::matchImagePoints(detectedCorners, detectedIds, _objPoints, imgPoints);
    else {
        CV_Assert(detectedCorners.isMat() || detectedCorners.isVector());
        size_t nDetected = detectedCorners.total();
        vector<Point3f> objPnts(nDetected);
        vector<Point2f> imgPnts(nDetected);
        for(size_t i = 0ull; i < nDetected; i++) {
            int pointId = detectedIds.getMat().at<int>((int)i);
            CV_Assert(pointId >= 0 && pointId < (int)chessboardCorners.size());
            objPnts[i] = chessboardCorners[pointId];
            imgPnts[i] = detectedCorners.getMat().at<Point2f>((int)i);
        }
        Mat(objPnts).copyTo(_objPoints);
        Mat(imgPnts).copyTo(imgPoints);
    }
}

void CharucoBoardImpl::generateImage(Size outSize, OutputArray img, int marginSize, int borderBits) const {
    CV_Assert(!outSize.empty());
    CV_Assert(marginSize >= 0);

    img.create(outSize, CV_8UC1);
    img.setTo(255);
    Mat out = img.getMat();
    Mat noMarginsImg =
        out.colRange(marginSize, out.cols - marginSize).rowRange(marginSize, out.rows - marginSize);

    double totalLengthX, totalLengthY;
    totalLengthX = squareLength * size.width;
    totalLengthY = squareLength * size.height;

    // proportional transformation
    double xReduction = totalLengthX / double(noMarginsImg.cols);
    double yReduction = totalLengthY / double(noMarginsImg.rows);

    // determine the zone where the chessboard is placed
    Mat chessboardZoneImg;
    if(xReduction > yReduction) {
        int nRows = int(totalLengthY / xReduction);
        int rowsMargins = (noMarginsImg.rows - nRows) / 2;
        chessboardZoneImg = noMarginsImg.rowRange(rowsMargins, noMarginsImg.rows - rowsMargins);
    } else {
        int nCols = int(totalLengthX / yReduction);
        int colsMargins = (noMarginsImg.cols - nCols) / 2;
        chessboardZoneImg = noMarginsImg.colRange(colsMargins, noMarginsImg.cols - colsMargins);
    }

    // determine the margins to draw only the markers
    // take the minimum just to be sure
    double squareSizePixels = min(double(chessboardZoneImg.cols) / double(size.width),
                                  double(chessboardZoneImg.rows) / double(size.height));

    double diffSquareMarkerLength = (squareLength - markerLength) / 2;
    int diffSquareMarkerLengthPixels =
        int(diffSquareMarkerLength * squareSizePixels / squareLength);

    // draw markers
    Mat markersImg;
    Board::Impl::generateImage(chessboardZoneImg.size(), markersImg, diffSquareMarkerLengthPixels, borderBits);
    markersImg.copyTo(chessboardZoneImg);

    // now draw black squares
    for(int y = 0; y < size.height; y++) {
        for(int x = 0; x < size.width; x++) {

            if(y % 2 != x % 2) continue; // white corner, dont do anything

            double startX, startY;
            startX = squareSizePixels * double(x);
            startY = squareSizePixels * double(y);

            Mat squareZone = chessboardZoneImg.rowRange(int(startY), int(startY + squareSizePixels))
                                 .colRange(int(startX), int(startX + squareSizePixels));

            squareZone.setTo(0);
        }
    }
}

CharucoBoard::CharucoBoard(){}

CharucoBoard::CharucoBoard(const Size& size, float squareLength, float markerLength,
                           const Dictionary &dictionary, InputArray ids):
    Board(new CharucoBoardImpl(dictionary, size, squareLength, markerLength)) {

    CV_Assert(size.width > 1 && size.height > 1 && markerLength > 0 && squareLength > markerLength);

    vector<vector<Point3f> > objPoints;
    float diffSquareMarkerLength = (squareLength - markerLength) / 2;
    int totalMarkers = (int)(ids.total());
    ids.copyTo(impl->ids);
    // calculate Board objPoints
    int nextId = 0;
    for(int y = 0; y < size.height; y++) {
        for(int x = 0; x < size.width; x++) {

            if(y % 2 == x % 2) continue; // black corner, no marker here

            vector<Point3f> corners(4);
            corners[0] = Point3f(x * squareLength + diffSquareMarkerLength,
                                 y * squareLength + diffSquareMarkerLength, 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, markerLength, 0);
            corners[3] = corners[0] + Point3f(0, markerLength, 0);
            objPoints.push_back(corners);
            // first ids in dictionary
            if (totalMarkers == 0)
                impl->ids.push_back(nextId);
            nextId++;
        }
    }
    if (totalMarkers > 0 && nextId != totalMarkers)
        CV_Error(cv::Error::StsBadSize, "Size of ids must be equal to the number of markers: "+std::to_string(nextId));
    impl->objPoints = objPoints;

    // now fill chessboardCorners
    std::vector<Point3f> & c = static_pointer_cast<CharucoBoardImpl>(impl)->chessboardCorners;
    for(int y = 0; y < size.height - 1; y++) {
        for(int x = 0; x < size.width - 1; x++) {
            Point3f corner;
            corner.x = (x + 1) * squareLength;
            corner.y = (y + 1) * squareLength;
            corner.z = 0;
            c.push_back(corner);
        }
    }
    impl->rightBottomBorder = Point3f(size.width * squareLength, size.height * squareLength, 0.f);
    static_pointer_cast<CharucoBoardImpl>(impl)->calcNearestMarkerCorners();
}

Size CharucoBoard::getChessboardSize() const {
    CV_Assert(impl);
    return static_pointer_cast<CharucoBoardImpl>(impl)->size;
}

float CharucoBoard::getSquareLength() const {
    CV_Assert(impl);
    return static_pointer_cast<CharucoBoardImpl>(impl)->squareLength;
}

float CharucoBoard::getMarkerLength() const {
    CV_Assert(impl);
    return static_pointer_cast<CharucoBoardImpl>(impl)->markerLength;
}

bool CharucoBoard::checkCharucoCornersCollinear(InputArray charucoIds) const {
    CV_Assert(impl);

    unsigned int nCharucoCorners = (unsigned int)charucoIds.getMat().total();
    if (nCharucoCorners <= 2)
        return true;

    // only test if there are 3 or more corners
    auto board = static_pointer_cast<CharucoBoardImpl>(impl);
    CV_Assert(board->chessboardCorners.size() >= charucoIds.getMat().total());

    Vec<double, 3> point0(board->chessboardCorners[charucoIds.getMat().at<int>(0)].x,
                          board->chessboardCorners[charucoIds.getMat().at<int>(0)].y, 1);

    Vec<double, 3> point1(board->chessboardCorners[charucoIds.getMat().at<int>(1)].x,
                          board->chessboardCorners[charucoIds.getMat().at<int>(1)].y, 1);

    // create a line from the first two points.
    Vec<double, 3> testLine = point0.cross(point1);
    Vec<double, 3> testPoint(0, 0, 1);

    double divisor = sqrt(testLine[0]*testLine[0] + testLine[1]*testLine[1]);
    CV_Assert(divisor != 0.0);

    // normalize the line with normal
    testLine /= divisor;

    double dotProduct;
    for (unsigned int i = 2; i < nCharucoCorners; i++){
        testPoint(0) = board->chessboardCorners[charucoIds.getMat().at<int>(i)].x;
        testPoint(1) = board->chessboardCorners[charucoIds.getMat().at<int>(i)].y;

        // if testPoint is on testLine, dotProduct will be zero (or very, very close)
        dotProduct = testPoint.dot(testLine);

        if (std::abs(dotProduct) > 1e-6){
            return false;
        }
    }
    // no points found that were off of testLine, return true that all points collinear.
    return true;
}

std::vector<Point3f> CharucoBoard::getChessboardCorners() const {
    CV_Assert(impl);
    return static_pointer_cast<CharucoBoardImpl>(impl)->chessboardCorners;
}

std::vector<std::vector<int> > CharucoBoard::getNearestMarkerIdx() const {
    CV_Assert(impl);
    return static_pointer_cast<CharucoBoardImpl>(impl)->nearestMarkerIdx;
}

std::vector<std::vector<int> > CharucoBoard::getNearestMarkerCorners() const {
    CV_Assert(impl);
    return static_pointer_cast<CharucoBoardImpl>(impl)->nearestMarkerCorners;
}

}
}
