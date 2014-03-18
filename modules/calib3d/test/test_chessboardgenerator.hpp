#ifndef CV_CHESSBOARDGENERATOR_H143KJTVYM389YTNHKFDHJ89NYVMO3VLMEJNTBGUEIYVCM203P
#define CV_CHESSBOARDGENERATOR_H143KJTVYM389YTNHKFDHJ89NYVMO3VLMEJNTBGUEIYVCM203P

#include "opencv2/calib3d/calib3d.hpp"

namespace cv
{

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
    Mat operator()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, vector<Point2f>& corners) const;
    Mat operator()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, const Size2f& squareSize, vector<Point2f>& corners) const;
    Mat operator()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, const Size2f& squareSize, const Point3f& pos, vector<Point2f>& corners) const;
    Size cornersSize() const;

    mutable vector<Point3f> corners3d;
private:
    void generateEdge(const Point3f& p1, const Point3f& p2, vector<Point3f>& out) const;
    Mat generateChessBoard(const Mat& bg, const Mat& camMat, const Mat& distCoeffs,
        const Point3f& zero, const Point3f& pb1, const Point3f& pb2,
        float sqWidth, float sqHeight, const vector<Point3f>& whole, vector<Point2f>& corners) const;
    void generateBasis(Point3f& pb1, Point3f& pb2) const;

    Mat rvec, tvec;
};

}


#endif
