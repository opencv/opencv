#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef tr1::tuple<std::string, int, int, tr1::tuple<int,int>, int> Path_Idx_Cn_NPoints_WSize_t;
typedef TestBaseWithParam<Path_Idx_Cn_NPoints_WSize_t> Path_Idx_Cn_NPoints_WSize;

void FormTrackingPointsArray(vector<Point2f>& points, int width, int height, int nPointsX, int nPointsY)
{
    int stepX = width / nPointsX;
    int stepY = height / nPointsY;
    if (stepX < 1 || stepY < 1) FAIL() << "Specified points number is too big";

    points.clear();
    points.reserve(nPointsX * nPointsY);

    for( int x = stepX / 2; x < width; x += stepX )
    {
        for( int y = stepY / 2; y < height; y += stepY )
        {
            Point2f pt(static_cast<float>(x), static_cast<float>(y));
            points.push_back(pt);
        }
    }
}

PERF_TEST_P(Path_Idx_Cn_NPoints_WSize, OpticalFlowPyrLK, testing::Combine(
                testing::Values<std::string>("cv/optflow/frames/VGA_%02d.png", "cv/optflow/frames/720p_%02d.jpg"),
                testing::Range(0, 3),
                testing::Values(1, 3, 4),
                testing::Values(make_tuple(9, 9), make_tuple(15, 15)),
                testing::Values(7, 11, 21, 25)
                )
            )
{
    string filename1 = getDataPath(cv::format(get<0>(GetParam()).c_str(), get<1>(GetParam())));
    string filename2 = getDataPath(cv::format(get<0>(GetParam()).c_str(), get<1>(GetParam()) + 1));
    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);
    if (img1.empty()) FAIL() << "Unable to load source image " << filename1;
    if (img2.empty()) FAIL() << "Unable to load source image " << filename2;

    int cn = get<2>(GetParam());
    int nPointsX = min(get<0>(get<3>(GetParam())), img1.cols);
    int nPointsY = min(get<1>(get<3>(GetParam())), img1.rows);
    int winSize = get<4>(GetParam());
    int maxLevel = 2;
    TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 7, 0.001);
    int flags = 0;
    double minEigThreshold = 1e-4;

    Mat frame1, frame2;
    switch(cn)
    {
    case 1:
        cvtColor(img1, frame1, COLOR_BGR2GRAY, cn);
        cvtColor(img2, frame2, COLOR_BGR2GRAY, cn);
        break;
    case 3:
        frame1 = img1;
        frame2 = img2;
        break;
    case 4:
        cvtColor(img1, frame1, COLOR_BGR2BGRA, cn);
        cvtColor(img2, frame2, COLOR_BGR2BGRA, cn);
        break;
    default:
        FAIL() << "Unexpected number of channels: " << cn;
    }

    vector<Point2f> inPoints;
    vector<Point2f> outPoints;
    vector<uchar> status;
    vector<float> err;

    FormTrackingPointsArray(inPoints, frame1.cols, frame1.rows, nPointsX, nPointsY);
    outPoints.resize(inPoints.size());
    status.resize(inPoints.size());
    err.resize(inPoints.size());

    declare.in(frame1, frame2, inPoints).out(outPoints);

    TEST_CYCLE_N(30)
    {
        calcOpticalFlowPyrLK(frame1, frame2, inPoints, outPoints, status, err,
                             Size(winSize, winSize), maxLevel, criteria,
                             flags, minEigThreshold);
    }
}
