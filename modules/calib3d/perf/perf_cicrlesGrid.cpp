#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;


typedef std::tr1::tuple<std::string, cv::Size> String_Size_t;
typedef perf::TestBaseWithParam<String_Size_t> String_Size;

PERF_TEST_P(String_Size, asymm_circles_grid, testing::Values(
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles1.jpg", Size(7,13)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles2.jpg", Size(7,13)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles3.jpg", Size(7,13)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles4.jpg", Size(5,5)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles5.jpg", Size(5,5)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles6.jpg", Size(5,5)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles7.jpg", Size(3,9)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles8.jpg", Size(3,9)),
                 String_Size_t("cv/cameracalibration/asymmetric_circles/acircles9.jpg", Size(3,9))
                 )
             )
{
    String filename = getDataPath(std::tr1::get<0>(GetParam()));
    Size gridSize = std::tr1::get<1>(GetParam());

    Mat frame = imread(filename);
    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;
    vector<Point2f> ptvec;
    ptvec.resize(gridSize.area());

    cvtColor(frame, frame, COLOR_BGR2GRAY);

    declare.in(frame).out(ptvec);

    TEST_CYCLE(100)
    {
        ASSERT_TRUE(findCirclesGrid(frame, gridSize, ptvec, CALIB_CB_CLUSTERING | CALIB_CB_ASYMMETRIC_GRID));
    }
}
