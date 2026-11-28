// Regression tests: 1D Mats must stay 1D through output-producing operations.
// The defect class was confirmed by issues #30703/#30705/#30713/#30777 and the
// merged fix "core: keep 1D Mat 1D in cv::flip" (#30778); these tests extend
// the same contract to transpose, repeat and reduce on the 5.x branch.
#include "test_precomp.hpp"
#include <vector>

namespace opencv_test { namespace {

using namespace cv;

// A true 1D Mat with external data (create() normalizes dims=1 to 2D, so the
// external-data constructor is the only way to obtain one).
static Mat make1D(int n, std::vector<float>& storage)
{
    storage.resize(n);
    for (int i = 0; i < n; i++)
        storage[i] = float(i + 1);
    return Mat(1, &n, CV_32F, storage.data());
}

TEST(Core_1DMat, transposeKeeps1D)
{
    std::vector<float> storage;
    Mat src = make1D(6, storage);
    ASSERT_EQ(1, src.dims);

    Mat dst;
    cv::transpose(src, dst);
    EXPECT_EQ(1, dst.dims);
    EXPECT_EQ(6, dst.total());
    // Transposing a 1D array is the identity on data.
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(float(i + 1), dst.at<float>(i));
}

TEST(Core_1DMat, repeatKeeps1D)
{
    std::vector<float> storage;
    Mat src = make1D(3, storage);
    ASSERT_EQ(1, src.dims);

    Mat dst;
    cv::repeat(src, 2, 3, dst);
    EXPECT_EQ(1, dst.dims);
    EXPECT_EQ(18, dst.total());
    // ny*nx concatenated copies of [1, 2, 3].
    const float expected[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3};
    for (int i = 0; i < 18; i++)
        EXPECT_EQ(expected[i], dst.at<float>(i)) << "at index " << i;
}

TEST(Core_1DMat, reduceKeeps1D)
{
    std::vector<float> storage;
    Mat src = make1D(4, storage);
    ASSERT_EQ(1, src.dims);

    Mat dst;
    cv::reduce(src, dst, 0, REDUCE_SUM, CV_32F);
    EXPECT_EQ(1, dst.dims);
    EXPECT_EQ(1, dst.total());
    EXPECT_EQ(10.f, dst.at<float>(0));
}

TEST(Core_1DMat, reduceKeeps1DNonSum)
{
    std::vector<float> storage;
    Mat src = make1D(4, storage);
    Mat dst;
    cv::reduce(src, dst, 0, REDUCE_MAX, CV_32F);
    EXPECT_EQ(1, dst.dims);
    EXPECT_EQ(4.f, dst.at<float>(0));
}

TEST(Core_1DMat, repeatMatches2DContentSemantics)
{
    // The 1D result of repeat(1D, 1, nx) must hold the same values as the
    // 2D result of repeat on the equivalent 1-row Mat.
    std::vector<float> storage;
    Mat src1d = make1D(3, storage);
    Mat src2d;
    src1d.copyTo(src2d);
    src2d = src2d.reshape(1, 1);

    Mat d1, d2;
    cv::repeat(src1d, 1, 3, d1);
    cv::repeat(src2d, 1, 3, d2);
    ASSERT_EQ(1, d1.dims);
    ASSERT_EQ(2, d2.dims);
    ASSERT_EQ(d1.total(), d2.total());
    for (int i = 0; i < d1.total(); i++)
        EXPECT_EQ(d2.at<float>(i), d1.at<float>(i)) << "at index " << i;
}

}} // namespace opencv_test::
