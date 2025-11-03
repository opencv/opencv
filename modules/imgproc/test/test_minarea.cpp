// GSoC 2026 FIX #27667: minAreaRect angle in [-90, 0)
#include <opencv2/imgproc.hpp>
#include <opencv2/ts.hpp>

namespace opencv_test {  // ✅ This is the correct namespace

using namespace cv;

TEST(Imgproc_MinAreaRect, angle_range_27667)
{
    // Test 1: Horizontal rectangle → angle ≈ 0.0
    const float p1[][2] = {
        {8,112},{8,123},{7,124},{4,124},{4,179},{7,179},{8,180},{8,219},
        {215,219},{215,172},{216,171},{219,171},{219,148},{216,148},{215,147},
        {215,112},{128,112},{128,115},{127,116},{64,116},{63,115},{63,112}
    };
    Mat m1(22, 1, CV_32FC2, (void*)p1);
    RotatedRect r1 = minAreaRect(m1);
    EXPECT_GE(r1.angle, -90.0);
    EXPECT_LT(r1.angle,   0.0);
    EXPECT_GE(r1.size.width, r1.size.height);

    // Test 2: Was 90.0 → now 0.0
    const float p2[][2] = {
        {60,4},{60,7},{59,8},{8,8},{8,71},{7,72},{4,72},{4,111},
        {215,111},{215,84},{216,83},{219,83},{219,52},{216,52},{215,51},
        {215,12},{216,11},{219,11},{219,8},{216,8},{215,7},{215,4}
    };
    Mat m2(22, 1, CV_32FC2, (void*)p2);
    RotatedRect r2 = minAreaRect(m2);
    EXPECT_GE(r2.angle, -90.0);
    EXPECT_LT(r2.angle,   0.0);
    EXPECT_GE(r2.size.width, r2.size.height);

    // Test 3: Vertical line → -90.0
    std::vector<Point2f> v;
    for (int y = 0; y < 100; y += 10) v.emplace_back(50, y);
    RotatedRect rv = minAreaRect(v);
    EXPECT_NEAR(rv.angle, -90.0, 1e-3);
}

} // namespace opencv_test
