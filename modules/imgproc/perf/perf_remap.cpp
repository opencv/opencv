#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4)

typedef TestBaseWithParam< tr1::tuple<Size, MatType, MatType, InterType> > TestRemap;

PERF_TEST_P( TestRemap, Remap,
             Combine(
                Values( szVGA, sz1080p ),
                Values( CV_16UC1, CV_16SC1, CV_32FC1 ),
                Values( CV_16SC2, CV_32FC1, CV_32FC2 ),
                InterType::all()
             )
)
{
    Size sz;
    int src_type, map1_type, inter_type;

    sz         = get<0>(GetParam());
    src_type   = get<1>(GetParam());
    map1_type  = get<2>(GetParam());
    inter_type = get<3>(GetParam());

    Mat src(sz, src_type), dst(sz, src_type), map1(sz, map1_type), map2;
    if (map1_type == CV_32FC1)
        map2.create(sz, CV_32FC1);
    else if (inter_type != INTER_NEAREST && map1_type == CV_16SC2)
    {
        map2.create(sz, CV_16UC1);
        map2 = Scalar::all(0);
    }

    RNG rng;
    rng.fill(src, RNG::UNIFORM, 0, 256);

    for (int j = 0; j < map1.rows; ++j)
        for (int i = 0; i < map1.cols; ++i)
            switch (map1_type)
            {
                case CV_32FC1:
                    map1.at<float>(j, i) = static_cast<float>(src.cols - i - 1);
                    map2.at<float>(j, i) = static_cast<float>(j);
                    break;
                case CV_32FC2:
                    map1.at<Vec2f>(j, i)[0] = static_cast<float>(src.cols - i - 1);
                    map1.at<Vec2f>(j, i)[1] = static_cast<float>(j);
                    break;
                case CV_16SC2:
                    map1.at<Vec2s>(j, i)[0] = static_cast<short>(src.cols - i - 1);
                    map1.at<Vec2s>(j, i)[1] = static_cast<short>(j);
                    break;
                default:
                    CV_Assert(0);
            }


    declare.in(src, WARMUP_RNG).out(dst).time(20);

    int runs = (sz.width <= 640) ? 3 : 1;
    TEST_CYCLE_MULTIRUN(runs) remap(src, dst, map1, map2, inter_type);

    SANITY_CHECK(dst);
}
