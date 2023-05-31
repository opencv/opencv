// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(DistanceType, DIST_L1, DIST_L2 , DIST_C)
CV_ENUM(MaskSize, DIST_MASK_3, DIST_MASK_5, DIST_MASK_PRECISE)
CV_ENUM(DstType, CV_8U, CV_32F)
CV_ENUM(LabelType, DIST_LABEL_CCOMP, DIST_LABEL_PIXEL)

typedef tuple<Size, DistanceType, MaskSize, DstType> SrcSize_DistType_MaskSize_DstType;
typedef tuple<Size, DistanceType, MaskSize, LabelType> SrcSize_DistType_MaskSize_LabelType;
typedef perf::TestBaseWithParam<SrcSize_DistType_MaskSize_DstType> DistanceTransform_Test;
typedef perf::TestBaseWithParam<SrcSize_DistType_MaskSize_LabelType> DistanceTransform_NeedLabels_Test;

PERF_TEST_P(DistanceTransform_Test, distanceTransform,
            testing::Combine(
                testing::Values(cv::Size(640, 480), cv::Size(800, 600), cv::Size(1024, 768), cv::Size(1280, 1024)),
                DistanceType::all(),
                MaskSize::all(),
                DstType::all()
                )
            )
{
    Size srcSize = get<0>(GetParam());
    int distanceType = get<1>(GetParam());
    int maskSize = get<2>(GetParam());
    int dstType = get<3>(GetParam());

    Mat src(srcSize, CV_8U);
    Mat dst(srcSize, dstType);

    declare
        .in(src, WARMUP_RNG)
        .out(dst, WARMUP_RNG)
        .time(30);

    TEST_CYCLE() distanceTransform( src, dst, distanceType, maskSize, dstType);

    double eps = 2e-4;

    SANITY_CHECK(dst, eps);
}

PERF_TEST_P(DistanceTransform_NeedLabels_Test, distanceTransform_NeedLabels,
            testing::Combine(
                testing::Values(cv::Size(640, 480), cv::Size(800, 600), cv::Size(1024, 768), cv::Size(1280, 1024)),
                DistanceType::all(),
                MaskSize::all(),
                LabelType::all()
                )
    )
{
    Size srcSize = get<0>(GetParam());
    int distanceType = get<1>(GetParam());
    int maskSize = get<2>(GetParam());
    int labelType = get<3>(GetParam());

    Mat src(srcSize, CV_8U);
    Mat label(srcSize, CV_32S);
    Mat dst(srcSize, CV_32F);

    declare
        .in(src, WARMUP_RNG)
        .out(label, WARMUP_RNG)
        .out(dst, WARMUP_RNG)
        .time(30);

    TEST_CYCLE() distanceTransform( src, dst, label, distanceType, maskSize, labelType);

    double eps = 2e-4;

    SANITY_CHECK(label, eps);
    SANITY_CHECK(dst, eps);
}

} // namespace
