// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<Size, MatType> Size_Source_t;
typedef TestBaseWithParam<Size_Source_t> Size_Source;
typedef TestBaseWithParam<Size> TestMatSize;

static const float rangeHight = 256.0f;
static const float rangeLow = 0.0f;

PERF_TEST_P(Size_Source, calcHist1d,
            testing::Combine(testing::Values(sz3MP, sz5MP),
                             testing::Values(CV_8U, CV_16U, CV_32F) )
            )
{
    Size size = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    Mat source(size.height, size.width, type);
    Mat hist;
    int channels [] = {0};
    int histSize [] = {256};
    int dims = 1;
    int numberOfImages = 1;

    const float range[] = {rangeLow, rangeHight};
    const float* ranges[] = {range};

    randu(source, rangeLow, rangeHight);

    declare.in(source);

    TEST_CYCLE_MULTIRUN(3)
    {
        calcHist(&source, numberOfImages, channels, Mat(), hist, dims, histSize, ranges);
    }

    SANITY_CHECK(hist);
}

PERF_TEST_P(Size_Source, calcHist2d,
            testing::Combine(testing::Values(sz3MP, sz5MP),
                             testing::Values(CV_8UC2, CV_16UC2, CV_32FC2) )
            )
{
    Size size = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    Mat source(size.height, size.width, type);
    Mat hist;
    int channels [] = {0, 1};
    int histSize [] = {256, 256};
    int dims = 2;
    int numberOfImages = 1;

    const float r[] = {rangeLow, rangeHight};
    const float* ranges[] = {r, r};

    randu(source, rangeLow, rangeHight);

    declare.in(source);
    TEST_CYCLE()
    {
        calcHist(&source, numberOfImages, channels, Mat(), hist, dims, histSize, ranges);
    }

    SANITY_CHECK(hist);
}

PERF_TEST_P(Size_Source, calcHist3d,
            testing::Combine(testing::Values(sz3MP, sz5MP),
                             testing::Values(CV_8UC3, CV_16UC3, CV_32FC3) )
            )
{
    Size size = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    Mat hist;
    int channels [] = {0, 1, 2};
    int histSize [] = {32, 32, 32};
    int dims = 3;
    int numberOfImages = 1;
    Mat source(size.height, size.width, type);

    const float r[] = {rangeLow, rangeHight};
    const float* ranges[] = {r, r, r};

    randu(source, rangeLow, rangeHight);

    declare.in(source);
    TEST_CYCLE()
    {
        calcHist(&source, numberOfImages, channels, Mat(), hist, dims, histSize, ranges);
    }

    SANITY_CHECK(hist);
}

#define MatSize TestMatSize
PERF_TEST_P(MatSize, equalizeHist,
            testing::Values(TYPICAL_MAT_SIZES)
            )
{
    Size size = GetParam();
    Mat source(size.height, size.width, CV_8U);
    Mat destination;
    declare.in(source, WARMUP_RNG);

    TEST_CYCLE()
    {
        equalizeHist(source, destination);
    }

    SANITY_CHECK(destination);
}
#undef MatSize

typedef TestBaseWithParam< tuple<int, int> > Dim_Cmpmethod;
PERF_TEST_P(Dim_Cmpmethod, compareHist,
            testing::Combine(testing::Values(1, 3),
                             testing::Values(HISTCMP_CORREL, HISTCMP_CHISQR, HISTCMP_INTERSECT, HISTCMP_BHATTACHARYYA, HISTCMP_CHISQR_ALT, HISTCMP_KL_DIV))
            )
{
    int dims = get<0>(GetParam());
    int method = get<1>(GetParam());
    int histSize[] = { 2048, 128, 64 };

    Mat hist1(dims, histSize, CV_32FC1);
    Mat hist2(dims, histSize, CV_32FC1);
    randu(hist1, 0, 256);
    randu(hist2, 0, 256);

    declare.in(hist1.reshape(1, 256), hist2.reshape(1, 256));

    TEST_CYCLE()
    {
        compareHist(hist1, hist2, method);
    }

    SANITY_CHECK_NOTHING();
}

typedef tuple<Size, double, MatType> Sz_ClipLimit_t;
typedef TestBaseWithParam<Sz_ClipLimit_t> Sz_ClipLimit;

PERF_TEST_P(Sz_ClipLimit, CLAHE,
            testing::Combine(testing::Values(::perf::szVGA, ::perf::sz720p, ::perf::sz1080p),
                             testing::Values(0.0, 40.0),
                             testing::Values(MatType(CV_8UC1), MatType(CV_16UC1)))
            )
{
    const Size size = get<0>(GetParam());
    const double clipLimit = get<1>(GetParam());
    const int type = get<2>(GetParam());

    Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    Ptr<CLAHE> clahe = createCLAHE(clipLimit);
    Mat dst;

    TEST_CYCLE() clahe->apply(src, dst);

    SANITY_CHECK(dst);
}

} // namespace
