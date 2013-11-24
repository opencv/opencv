#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(BorderType, BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT_101)

typedef std::tr1::tuple<string, int, int, double, BorderType> Img_BlockSize_ApertureSize_k_BorderType_t;
typedef perf::TestBaseWithParam<Img_BlockSize_ApertureSize_k_BorderType_t> Img_BlockSize_ApertureSize_k_BorderType;

PERF_TEST_P(Img_BlockSize_ApertureSize_k_BorderType, cornerHarris,
            testing::Combine(
                testing::Values( "stitching/a1.png", "cv/shared/pic5.png"),
                testing::Values( 3, 5 ),
                testing::Values( 3, 5 ),
                testing::Values( 0.04, 0.1 ),
                BorderType::all()
                )
          )
{
    string filename = getDataPath(get<0>(GetParam()));
    int blockSize = get<1>(GetParam());
    int apertureSize = get<2>(GetParam());
    double k = get<3>(GetParam());
    BorderType borderType = get<4>(GetParam());

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    if (src.empty())
        FAIL() << "Unable to load source image" << filename;

    Mat dst;

    TEST_CYCLE() cornerHarris(src, dst, blockSize, apertureSize, k, borderType);

    SANITY_CHECK(dst, 2e-5);
}
