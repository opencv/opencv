#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {

namespace ocl {

CV_ENUM(MethodType, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)

typedef std::tr1::tuple<Size, Size, MethodType> ImgSize_TmplSize_Method_t;
typedef TestBaseWithParam<ImgSize_TmplSize_Method_t> ImgSize_TmplSize_Method;

OCL_PERF_TEST_P(ImgSize_TmplSize_Method, MatchTemplate,
        ::testing::Combine(
            testing::Values(szSmall128, cv::Size(320, 240),
                            cv::Size(640, 480), cv::Size(800, 600),
                            cv::Size(1024, 768), cv::Size(1280, 1024)),
            testing::Values(cv::Size(12, 12), cv::Size(28, 9),
                            cv::Size(8, 30), cv::Size(16, 16)),
            MethodType::all()
            )
        )
{
    const ImgSize_TmplSize_Method_t params = GetParam();
    const Size imgSz = get<0>(params), tmplSz = get<1>(params);
    const int method = get<2>(params);

    UMat img(imgSz, CV_8UC1), tmpl(tmplSz, CV_8UC1);
    UMat result(imgSz - tmplSz + Size(1, 1), CV_32F);

    declare.in(img, tmpl, WARMUP_RNG).out(result);

    OCL_TEST_CYCLE() matchTemplate(img, tmpl, result, method);

    bool isNormed =
        method == TM_CCORR_NORMED ||
        method == TM_SQDIFF_NORMED ||
        method == TM_CCOEFF_NORMED;
    double eps = isNormed ? 3e-2
        : 255 * 255 * tmpl.total() * 1e-4;

    if (isNormed)
        SANITY_CHECK(result, eps, ERROR_RELATIVE);
    else
        SANITY_CHECK(result, eps);
}


/////////// matchTemplate (performance tests from 2.4) ////////////////////////

typedef Size_MatType CV_TM_CCORRFixture;

OCL_PERF_TEST_P(CV_TM_CCORRFixture, matchTemplate,
                ::testing::Combine(::testing::Values(Size(1000, 1000), Size(2000, 2000)),
                               OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params), templSize(5, 5);
    const int type = get<1>(params);

    UMat src(srcSize, type), templ(templSize, type);
    const Size dstSize(src.cols - templ.cols + 1, src.rows - templ.rows + 1);
    UMat dst(dstSize, CV_32F);

    declare.in(src, templ, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::matchTemplate(src, templ, dst, CV_TM_CCORR);

    SANITY_CHECK(dst, 1e-4);
}

typedef TestBaseWithParam<Size> CV_TM_CCORR_NORMEDFixture;

OCL_PERF_TEST_P(CV_TM_CCORR_NORMEDFixture, matchTemplate,
                ::testing::Values(Size(1000, 1000), Size(2000, 2000), Size(4000, 4000)))
{
    const Size srcSize = GetParam(), templSize(5, 5);

    UMat src(srcSize, CV_8UC1), templ(templSize, CV_8UC1);
    const Size dstSize(src.cols - templ.cols + 1, src.rows - templ.rows + 1);
    UMat dst(dstSize, CV_8UC1);

    declare.in(src, templ, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::matchTemplate(src, templ, dst, CV_TM_CCORR_NORMED);

    SANITY_CHECK(dst, 3e-2);
}

} }

#endif // HAVE_OPENCL
