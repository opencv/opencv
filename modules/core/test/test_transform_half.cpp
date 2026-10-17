// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using HalfTransformParams = std::tuple<int, int, bool, int>;
typedef testing::TestWithParam<HalfTransformParams> Core_HalfTransform;

TEST_P(Core_HalfTransform, accuracy_and_storage)
{
    const int scn = std::get<0>(GetParam()), dcn = std::get<1>(GetParam());
    const bool bias = std::get<2>(GetParam());
    const int matrixDepth = std::get<3>(GetParam());
    RNG rng(0x30361);
    for (int width : {1, 7, 17, 257})
    {
        for (int diagonal = 0; diagonal <= (scn == dcn); ++diagonal)
        {
            SCOPED_TRACE(format("width=%d diagonal=%d", width, diagonal));
            Mat matrixParent(dcn + 2, scn + 3, CV_32F);
            rng.fill(matrixParent, RNG::UNIFORM, -1.f, 1.f);
            Mat matrix = matrixParent(Rect(1, 1, scn + int(bias), dcn));
            if (diagonal)
                for (int y = 0; y < dcn; ++y)
                    for (int x = 0; x < scn; ++x)
                        if (x != y) matrix.at<float>(y, x) = 0;
            Mat typedMatrix;
            matrixParent.convertTo(typedMatrix, matrixDepth);
            matrix = typedMatrix(Rect(1, 1, scn + int(bias), dcn));

            Mat parent(7, width + 4, CV_MAKETYPE(CV_32F, scn)), half;
            rng.fill(parent, RNG::UNIFORM, -2.f, 2.f);
            parent.convertTo(half, CV_16F);
            Rect roi(1, 1, width, 5);
            Mat src = half(roi), original = half.clone(), represented;
            src.convertTo(represented, CV_32F);
            Mat referenceFloat, reference;
            transform(represented, referenceFloat, matrix);
            referenceFloat.convertTo(reference, CV_16F);

            Mat destinationParent(half.size(), CV_MAKETYPE(CV_16F, dcn));
            destinationParent.reshape(1).setTo(Scalar::all(-7));
            Mat guard = destinationParent.clone();
            Mat dst = destinationParent(roi);
            uchar* storage = dst.data;
            transform(src, dst, matrix);
            EXPECT_EQ(storage, dst.data);
            ASSERT_EQ(reference.type(), dst.type());
            Mat actualFloat, expectedFloat;
            dst.convertTo(actualFloat, CV_32F);
            reference.convertTo(expectedFloat, CV_32F);
            for (int y = 0; y < dst.rows; ++y)
                for (int x = 0; x < dst.cols * dcn; ++x)
                {
                    float expected = expectedFloat.ptr<float>(y)[x];
                    EXPECT_LE(std::abs(actualFloat.ptr<float>(y)[x] - expected),
                              std::max(std::abs(expected) * 0.001f, 2e-6f));
                }
            EXPECT_EQ(0, cv::norm(half, original, NORM_INF));
            Mat mask(half.size(), CV_8U, Scalar::all(255));
            mask(roi).setTo(0);
            EXPECT_EQ(0, cv::norm(destinationParent, guard, NORM_INF, mask));
            if (scn == dcn)
            {
                transform(src, src, matrix);
                EXPECT_EQ(0, cv::norm(src, dst, NORM_INF));
                EXPECT_EQ(0, cv::norm(half, original, NORM_INF, mask));
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(Core, Core_HalfTransform,
    testing::Combine(testing::Values(1, 2, 3, 4, 5), testing::Values(1, 2, 3, 4, 5),
                     testing::Bool(), testing::Values(CV_32F, CV_64F)));

TEST(Core_HalfTransform, nonfinite_and_subnormal)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Mat src = Mat_<Vec3f>({1, 4}, {Vec3f(0, -0.f, 1.f / 16777216), Vec3f(65504, -65504, 1),
        Vec3f(inf, -inf, 0), Vec3f(nan, nan, nan)});
    src.convertTo(src, CV_16F);
    Mat matrix = Mat_<float>({3, 4}, {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -1, 0});
    Mat represented, expected, actual;
    src.convertTo(represented, CV_32F);
    transform(represented, expected, matrix);
    transform(src, actual, matrix);
    actual.convertTo(actual, CV_32F);
    expected.convertTo(expected, CV_16F);
    expected.convertTo(expected, CV_32F);
    for (int x = 0; x < src.cols * src.channels(); ++x)
    {
        float a = actual.ptr<float>()[x], e = expected.ptr<float>()[x];
        if (cvIsNaN(e)) EXPECT_TRUE(cvIsNaN(a));
        else EXPECT_EQ(e, a);
    }
}

}} // namespace
