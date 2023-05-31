// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef CV_CHESSBOARDGENERATOR_H143KJTVYM389YTNHKFDHJ89NYVMO3VLMEJNTBGUEIYVCM203P
#define CV_CHESSBOARDGENERATOR_H143KJTVYM389YTNHKFDHJ89NYVMO3VLMEJNTBGUEIYVCM203P

namespace cv
{

using std::vector;

class ChessBoardGenerator
{
public:
    double sensorWidth;
    double sensorHeight;
    size_t squareEdgePointsNum;
    double min_cos;
    mutable double cov;
    Size patternSize;
    int rendererResolutionMultiplier;

    ChessBoardGenerator(const Size& patternSize = Size(8, 6));
    Mat operator()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, std::vector<Point2f>& corners) const;
    Mat operator()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, const Size2f& squareSize, std::vector<Point2f>& corners) const;
    Mat operator()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, const Size2f& squareSize, const Point3f& pos, std::vector<Point2f>& corners) const;
    Size cornersSize() const;

    mutable std::vector<Point3f> corners3d;
private:
    void generateEdge(const Point3f& p1, const Point3f& p2, std::vector<Point3f>& out) const;
    Mat generateChessBoard(const Mat& bg, const Mat& camMat, const Mat& distCoeffs,
        const Point3f& zero, const Point3f& pb1, const Point3f& pb2,
        float sqWidth, float sqHeight, const std::vector<Point3f>& whole, std::vector<Point2f>& corners) const;
    void generateBasis(Point3f& pb1, Point3f& pb2) const;

    Mat rvec, tvec;
};

}


#endif
