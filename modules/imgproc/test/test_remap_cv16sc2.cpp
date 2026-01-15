#include "test_precomp.hpp"

namespace opencv_test {

TEST(Imgproc_Remap, Linear_Interpolation_CV16SC2)
{
    // Two-pixel source gradient
    Mat src(1, 2, CV_8UC1);
    src.at<uchar>(0,0) = 0;
    src.at<uchar>(0,1) = 32;

    // CV_16SC2 map uses 11.5 fixed-point format
    // 16 = 0.5 fractional offset (1 << 4)
    Mat map(1, 1, CV_16SC2);
    map.at<Vec2s>(0,0)[0] = 16;
    map.at<Vec2s>(0,0)[1] = 0;

    Mat dst;
    remap(src, dst, map, Mat(), INTER_LINEAR);

    // Reference map in floating-point coordinates
    Mat fmap(1, 1, CV_32FC2, Scalar(0.5f, 0.f));
    Mat ref;
    remap(src, ref, fmap, Mat(), INTER_LINEAR);

    ASSERT_EQ(dst.at<uchar>(0,0), ref.at<uchar>(0,0));
}

}