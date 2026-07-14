// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(BorderType, BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT_101)

typedef tuple<string, int, int, double, BorderType> Img_BlockSize_ApertureSize_k_BorderType_t;
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
    ASSERT_FALSE(src.empty()) << "Unable to load source image: " << filename;

    Mat dst;

    TEST_CYCLE() cornerHarris(src, dst, blockSize, apertureSize, k, borderType);

    SANITY_CHECK(dst, 2e-5, ERROR_RELATIVE);
}

typedef tuple<string, int, int, BorderType> Img_BlockSize_ApertureSize_BorderType_t;
typedef perf::TestBaseWithParam<Img_BlockSize_ApertureSize_BorderType_t> Img_BlockSize_ApertureSize_BorderType;

PERF_TEST_P(Img_BlockSize_ApertureSize_BorderType, cornerEigenValsAndVecs,
            testing::Combine(
                testing::Values( "stitching/a1.png", "cv/shared/pic5.png"),
                testing::Values( 3, 5 ),
                testing::Values( 3, 5 ),
                BorderType::all()
            )
          )
{
    string filename = getDataPath(get<0>(GetParam()));
    int blockSize = get<1>(GetParam());
    int apertureSize = get<2>(GetParam());
    BorderType borderType = get<3>(GetParam());

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Unable to load source image: " << filename;

    Mat dst;

    TEST_CYCLE() cornerEigenValsAndVecs(src, dst, blockSize, apertureSize, borderType);

    Mat l1;
    extractChannel(dst, l1, 0);

    SANITY_CHECK(l1, 2e-5, ERROR_RELATIVE);
}

PERF_TEST_P(Img_BlockSize_ApertureSize_BorderType, cornerMinEigenVal,
            testing::Combine(
                testing::Values( "stitching/a1.png", "cv/shared/pic5.png"),
                testing::Values( 3, 5 ),
                testing::Values( 3, 5 ),
                BorderType::all()
            )
          )
{
    string filename = getDataPath(get<0>(GetParam()));
    int blockSize = get<1>(GetParam());
    int apertureSize = get<2>(GetParam());
    BorderType borderType = get<3>(GetParam());

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Unable to load source image: " << filename;

    Mat dst;

    TEST_CYCLE() cornerMinEigenVal(src, dst, blockSize, apertureSize, borderType);

    SANITY_CHECK(dst, 2e-5, ERROR_RELATIVE);
}

} // namespace
