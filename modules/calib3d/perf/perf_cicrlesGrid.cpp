#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<std::string, cv::Size> String_Size_t;
typedef perf::TestBaseWithParam<String_Size_t> String_Size;

PERF_TEST_P(String_Size, asymm_circles_grid, testing::Values(
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles1.png", Size(7,13)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles2.png", Size(7,13)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles3.png", Size(7,13)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles4.png", Size(5,5)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles5.png", Size(5,5)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles6.png", Size(5,5)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles7.png", Size(3,9)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles8.png", Size(3,9)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles9.png", Size(3,9))
                )
            )
{
    string filename = getDataPath(get<0>(GetParam()));
    Size gridSize = get<1>(GetParam());

    Mat frame = imread(filename);
    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    vector<Point2f> ptvec;
    ptvec.resize(gridSize.area());

    cvtColor(frame, frame, COLOR_BGR2GRAY);

    declare.in(frame).out(ptvec);

    TEST_CYCLE() ASSERT_TRUE(findCirclesGrid(frame, gridSize, ptvec, CALIB_CB_CLUSTERING | CALIB_CB_ASYMMETRIC_GRID));

    SANITY_CHECK(ptvec, 2);
}
