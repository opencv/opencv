#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <stdio.h>

using namespace cv;
using namespace std;

static void help()
{
    printf( "\nThis code generates an artificial camera and artificial chessboard images,\n"
            "and then calibrates. It is basically test code for calibration that shows\n"
            "how to package calibration points and then calibrate the camera.\n"
            "Usage:\n"
            "./calibration_artificial\n\n");
}
namespace cv
{

/* copy of class defines int tests/cv/chessboardgenerator.h */
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
    Size cornersSize() const;
private:
    void generateEdge(const Point3f& p1, const Point3f& p2, vector<Point3f>& out) const;
    Mat generageChessBoard(const Mat& bg, const Mat& camMat, const Mat& distCoeffs,
        const Point3f& zero, const Point3f& pb1, const Point3f& pb2,
        float sqWidth, float sqHeight, const vector<Point3f>& whole, vector<Point2f>& corners) const;
    void generateBasis(Point3f& pb1, Point3f& pb2) const;
    Point3f generateChessBoardCenter(const Mat& camMat, const Size& imgSize) const;
    Mat rvec, tvec;
};
};



const Size imgSize(800, 600);
const Size brdSize(8, 7);
const size_t brds_num = 20;

template<class T> ostream& operator<<(ostream& out, const Mat_<T>& mat)
{
    for(int j = 0; j < mat.rows; ++j)
        for(int i = 0; i < mat.cols; ++i)
            out << mat(j, i) << " ";
    return out;
}



int main()
{
    help();
    cout << "Initializing background...";
    Mat background(imgSize, CV_8UC3);
    randu(background, Scalar::all(32), Scalar::all(255));
    GaussianBlur(background, background, Size(5, 5), 2);
    cout << "Done" << endl;

    cout << "Initializing chess board generator...";
    ChessBoardGenerator cbg(brdSize);
    cbg.rendererResolutionMultiplier = 4;
    cout << "Done" << endl;

    /* camera params */
    Mat_<double> camMat(3, 3);
    camMat << 300., 0., background.cols/2., 0, 300., background.rows/2., 0., 0., 1.;

    Mat_<double> distCoeffs(1, 5);
    distCoeffs << 1.2, 0.2, 0., 0., 0.;

    cout << "Generating chessboards...";
    vector<Mat> boards(brds_num);
    vector<Point2f> tmp;
    for(size_t i = 0; i < brds_num; ++i)
        cout << (boards[i] = cbg(background, camMat, distCoeffs, tmp), i) << " ";
    cout << "Done" << endl;

    vector<Point3f> chessboard3D;
    for(int j = 0; j < cbg.cornersSize().height; ++j)
        for(int i = 0; i < cbg.cornersSize().width; ++i)
            chessboard3D.push_back(Point3i(i, j, 0));

    /* init points */
    vector< vector<Point3f> > objectPoints;
    vector< vector<Point2f> > imagePoints;

    cout << endl << "Finding chessboards' corners...";
    for(size_t i = 0; i < brds_num; ++i)
    {
        cout << i;
        namedWindow("Current chessboard"); imshow("Current chessboard", boards[i]); waitKey(100);
        bool found = findChessboardCorners(boards[i], cbg.cornersSize(), tmp);
        if (found)
        {
            imagePoints.push_back(tmp);
            objectPoints.push_back(chessboard3D);
            cout<< "-found ";
        }
        else
            cout<< "-not-found ";

        drawChessboardCorners(boards[i], cbg.cornersSize(), Mat(tmp), found);
        imshow("Current chessboard", boards[i]); waitKey(1000);
    }
    cout << "Done" << endl;
    cvDestroyAllWindows();

    Mat camMat_est;
    Mat distCoeffs_est;
    vector<Mat> rvecs, tvecs;

    cout << "Calibrating...";
    double rep_err = calibrateCamera(objectPoints, imagePoints, imgSize, camMat_est, distCoeffs_est, rvecs, tvecs);
    cout << "Done" << endl;

    cout << endl << "Average Reprojection error: " << rep_err/brds_num/cbg.cornersSize().area() << endl;
    cout << "==================================" << endl;
    cout << "Original camera matrix:\n" << camMat << endl;
    cout << "Original distCoeffs:\n" << distCoeffs << endl;
    cout << "==================================" << endl;
    cout << "Estimated camera matrix:\n" << (Mat_<double>&)camMat_est << endl;
    cout << "Estimated distCoeffs:\n" << (Mat_<double>&)distCoeffs_est << endl;

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

// Copy of  tests/cv/src/chessboardgenerator code. Just do not want to add dependency.


ChessBoardGenerator::ChessBoardGenerator(const Size& _patternSize) : sensorWidth(32), sensorHeight(24),
    squareEdgePointsNum(200), min_cos(sqrt(2.f)*0.5f), cov(0.5),
    patternSize(_patternSize), rendererResolutionMultiplier(4), tvec(Mat::zeros(1, 3, CV_32F))
{
    Rodrigues(Mat::eye(3, 3, CV_32F), rvec);
}

void cv::ChessBoardGenerator::generateEdge(const Point3f& p1, const Point3f& p2, vector<Point3f>& out) const
{
    Point3f step = (p2 - p1) * (1.f/squareEdgePointsNum);
    for(size_t n = 0; n < squareEdgePointsNum; ++n)
        out.push_back( p1 + step * (float)n);
}

Size cv::ChessBoardGenerator::cornersSize() const
{
    return Size(patternSize.width-1, patternSize.height-1);
}

struct Mult
{
    float m;
    Mult(int mult) : m((float)mult) {}
    Point2f operator()(const Point2f& p)const { return p * m; }
};

void cv::ChessBoardGenerator::generateBasis(Point3f& pb1, Point3f& pb2) const
{
    RNG& rng = theRNG();

    Vec3f n;
    for(;;)
    {
        n[0] = rng.uniform(-1.f, 1.f);
        n[1] = rng.uniform(-1.f, 1.f);
        n[2] = rng.uniform(-1.f, 1.f);
        float len = (float)norm(n);
        n[0]/=len;
        n[1]/=len;
        n[2]/=len;

        if (fabs(n[2]) > min_cos)
            break;
    }

    Vec3f n_temp = n; n_temp[0] += 100;
    Vec3f b1 = n.cross(n_temp);
    Vec3f b2 = n.cross(b1);
    float len_b1 = (float)norm(b1);
    float len_b2 = (float)norm(b2);

    pb1 = Point3f(b1[0]/len_b1, b1[1]/len_b1, b1[2]/len_b1);
    pb2 = Point3f(b2[0]/len_b1, b2[1]/len_b2, b2[2]/len_b2);
}

Mat cv::ChessBoardGenerator::generageChessBoard(const Mat& bg, const Mat& camMat, const Mat& distCoeffs,
                                                const Point3f& zero, const Point3f& pb1, const Point3f& pb2,
                                                float sqWidth, float sqHeight, const vector<Point3f>& whole,
                                                vector<Point2f>& corners) const
{
    vector< vector<Point> > squares_black;
    for(int i = 0; i < patternSize.width; ++i)
        for(int j = 0; j < patternSize.height; ++j)
            if ( (i % 2 == 0 && j % 2 == 0) || (i % 2 != 0 && j % 2 != 0) )
            {
                vector<Point3f> pts_square3d;
                vector<Point2f> pts_square2d;

                Point3f p1 = zero + (i + 0) * sqWidth * pb1 + (j + 0) * sqHeight * pb2;
                Point3f p2 = zero + (i + 1) * sqWidth * pb1 + (j + 0) * sqHeight * pb2;
                Point3f p3 = zero + (i + 1) * sqWidth * pb1 + (j + 1) * sqHeight * pb2;
                Point3f p4 = zero + (i + 0) * sqWidth * pb1 + (j + 1) * sqHeight * pb2;
                generateEdge(p1, p2, pts_square3d);
                generateEdge(p2, p3, pts_square3d);
                generateEdge(p3, p4, pts_square3d);
                generateEdge(p4, p1, pts_square3d);

                projectPoints( Mat(pts_square3d), rvec, tvec, camMat, distCoeffs, pts_square2d);
                squares_black.resize(squares_black.size() + 1);
                vector<Point2f> temp;
                approxPolyDP(Mat(pts_square2d), temp, 1.0, true);
                transform(temp.begin(), temp.end(), back_inserter(squares_black.back()), Mult(rendererResolutionMultiplier));
            }

    /* calculate corners */
    vector<Point3f> corners3d;
    for(int j = 0; j < patternSize.height - 1; ++j)
        for(int i = 0; i < patternSize.width - 1; ++i)
            corners3d.push_back(zero + (i + 1) * sqWidth * pb1 + (j + 1) * sqHeight * pb2);
    corners.clear();
    projectPoints( Mat(corners3d), rvec, tvec, camMat, distCoeffs, corners);

    vector<Point3f> whole3d;
    vector<Point2f> whole2d;
    generateEdge(whole[0], whole[1], whole3d);
    generateEdge(whole[1], whole[2], whole3d);
    generateEdge(whole[2], whole[3], whole3d);
    generateEdge(whole[3], whole[0], whole3d);
    projectPoints( Mat(whole3d), rvec, tvec, camMat, distCoeffs, whole2d);
    vector<Point2f> temp_whole2d;
    approxPolyDP(Mat(whole2d), temp_whole2d, 1.0, true);

    vector< vector<Point > > whole_contour(1);
    transform(temp_whole2d.begin(), temp_whole2d.end(),
        back_inserter(whole_contour.front()), Mult(rendererResolutionMultiplier));

    Mat result;
    if (rendererResolutionMultiplier == 1)
    {
        result = bg.clone();
        drawContours(result, whole_contour, -1, Scalar::all(255), CV_FILLED, CV_AA);
        drawContours(result, squares_black, -1, Scalar::all(0), CV_FILLED, CV_AA);
    }
    else
    {
        Mat tmp;
        resize(bg, tmp, bg.size() * rendererResolutionMultiplier);
        drawContours(tmp, whole_contour, -1, Scalar::all(255), CV_FILLED, CV_AA);
        drawContours(tmp, squares_black, -1, Scalar::all(0), CV_FILLED, CV_AA);
        resize(tmp, result, bg.size(), 0, 0, INTER_AREA);
    }
    return result;
}

Mat cv::ChessBoardGenerator::operator ()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, vector<Point2f>& corners) const
{
    cov = min(cov, 0.8);
    double fovx, fovy, focalLen;
    Point2d principalPoint;
    double aspect;
    calibrationMatrixValues( camMat, bg.size(), sensorWidth, sensorHeight,
        fovx, fovy, focalLen, principalPoint, aspect);

    RNG& rng = theRNG();

    float d1 = static_cast<float>(rng.uniform(0.1, 10.0));
    float ah = static_cast<float>(rng.uniform(-fovx/2 * cov, fovx/2 * cov) * CV_PI / 180);
    float av = static_cast<float>(rng.uniform(-fovy/2 * cov, fovy/2 * cov) * CV_PI / 180);

    Point3f p;
    p.z = cos(ah) * d1;
    p.x = sin(ah) * d1;
    p.y = p.z * tan(av);

    Point3f pb1, pb2;
    generateBasis(pb1, pb2);

    float cbHalfWidth = static_cast<float>(norm(p) * sin( min(fovx, fovy) * 0.5 * CV_PI / 180));
    float cbHalfHeight = cbHalfWidth * patternSize.height / patternSize.width;

    vector<Point3f> pts3d(4);
    vector<Point2f> pts2d(4);
    for(;;)
    {
        pts3d[0] = p + pb1 * cbHalfWidth + cbHalfHeight * pb2;
        pts3d[1] = p + pb1 * cbHalfWidth - cbHalfHeight * pb2;
        pts3d[2] = p - pb1 * cbHalfWidth - cbHalfHeight * pb2;
        pts3d[3] = p - pb1 * cbHalfWidth + cbHalfHeight * pb2;

        /* can remake with better perf */
        projectPoints( Mat(pts3d), rvec, tvec, camMat, distCoeffs, pts2d);

        bool inrect1 = pts2d[0].x < bg.cols && pts2d[0].y < bg.rows && pts2d[0].x > 0 && pts2d[0].y > 0;
        bool inrect2 = pts2d[1].x < bg.cols && pts2d[1].y < bg.rows && pts2d[1].x > 0 && pts2d[1].y > 0;
        bool inrect3 = pts2d[2].x < bg.cols && pts2d[2].y < bg.rows && pts2d[2].x > 0 && pts2d[2].y > 0;
        bool inrect4 = pts2d[3].x < bg.cols && pts2d[3].y < bg.rows && pts2d[3].x > 0 && pts2d[3].y > 0;

        if ( inrect1 && inrect2 && inrect3 && inrect4)
            break;

        cbHalfWidth*=0.8f;
        cbHalfHeight = cbHalfWidth * patternSize.height / patternSize.width;
    }

    cbHalfWidth  *= static_cast<float>(patternSize.width)/(patternSize.width + 1);
    cbHalfHeight *= static_cast<float>(patternSize.height)/(patternSize.height + 1);

    Point3f zero = p - pb1 * cbHalfWidth - cbHalfHeight * pb2;
    float sqWidth  = 2 * cbHalfWidth/patternSize.width;
    float sqHeight = 2 * cbHalfHeight/patternSize.height;

    return generageChessBoard(bg, camMat, distCoeffs, zero, pb1, pb2, sqWidth, sqHeight,  pts3d, corners);
}

