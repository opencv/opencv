#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

template <typename T>
static void CvRoundMat(const cv::Mat & src, cv::Mat & dst)
{
    for (int y = 0; y < dst.rows; ++y)
    {
        const T * sptr = src.ptr<T>(y);
        int * dptr = dst.ptr<int>(y);

        for (int x = 0; x < dst.cols; ++x)
            dptr[x] = cvRound(sptr[x]);
    }
}

PERF_TEST_P(Size_MatType, CvRound_Float,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(CV_32FC1, CV_64FC1)))
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam()), depth = CV_MAT_DEPTH(type);

    cv::Mat src(size, type), dst(size, CV_32SC1);

    declare.in(src, WARMUP_RNG).out(dst);

    if (depth == CV_32F)
    {
        TEST_CYCLE()
            CvRoundMat<float>(src, dst);
    }
    else if (depth == CV_64F)
    {
        TEST_CYCLE()
            CvRoundMat<double>(src, dst);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
