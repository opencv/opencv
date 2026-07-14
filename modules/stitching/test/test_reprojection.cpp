// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/stitching/warpers.hpp"

namespace opencv_test { namespace {
class ReprojectionTest : public ::testing::Test {

protected:
    const size_t TEST_COUNT = 15;
    Mat K, R;
    RNG rng = RNG(0);
    ReprojectionTest()
    {
        K = Mat::eye(3, 3, CV_32FC1);
        float angle = (float)(30.0 * CV_PI / 180.0);
        float rotationMatrix[9] = {
                (float)cos(angle), (float)sin(angle), 0,
                (float)-sin(angle), (float)cos(angle), 0,
                0, 0, 1
        };
        Mat(3, 3, CV_32FC1, rotationMatrix).copyTo(R);
    }
    void TestReprojection(Ptr<detail::RotationWarper> warper, Point2f pt) {
        Point2f projected_pt = warper->warpPoint(pt, K, R);
        Point2f reprojected_pt = warper->warpPointBackward(projected_pt, K, R);
        EXPECT_NEAR(pt.x, reprojected_pt.x, float( 1e-5));
        EXPECT_NEAR(pt.y, reprojected_pt.y, float( 1e-5));
    }
};


TEST_F(ReprojectionTest, PlaneWarper)
{
    Ptr<WarperCreator> creator = makePtr<PlaneWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, AffineWarper)
{
    Ptr<WarperCreator> creator = makePtr<AffineWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, CylindricalWarper)
{
    Ptr<WarperCreator> creator = makePtr<CylindricalWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, SphericalWarper)
{
    Ptr<WarperCreator> creator = makePtr<SphericalWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, FisheyeWarper)
{
    Ptr<WarperCreator> creator = makePtr<FisheyeWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, StereographicWarper)
{
    Ptr<WarperCreator> creator = makePtr<StereographicWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, CompressedRectilinearWarper)
{
    Ptr<WarperCreator> creator = makePtr<CompressedRectilinearWarper>(1.5f, 1.0f);
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, CompressedRectilinearPortraitWarper)
{
    Ptr<WarperCreator> creator = makePtr<CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, PaniniWarper)
{
    Ptr<WarperCreator> creator = makePtr<PaniniWarper>(1.5f, 1.0f);
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, PaniniPortraitWarper)
{
    Ptr<WarperCreator> creator = makePtr<PaniniPortraitWarper>(1.5f, 1.0f);
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, MercatorWarper)
{
    Ptr<WarperCreator> creator = makePtr<MercatorWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

TEST_F(ReprojectionTest, TransverseMercatorWarper)
{
    Ptr<WarperCreator> creator = makePtr<TransverseMercatorWarper>();
    for (size_t i = 0; i < TEST_COUNT; ++i) {
        TestReprojection(creator->create(1), Point2f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f)));
    }
}

}} // namespace
