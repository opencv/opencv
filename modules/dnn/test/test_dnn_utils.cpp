#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::dnn;

// This test verifies that blobRectsToImageRects correctly maps
// bounding boxes back to the original image when letterbox padding
// was applied during blob creation.
TEST(DNN_utils, blobRectsToImageRects_letterbox_case)
{
    // Original image size (non-square)
    Size imgSize(640, 480);

    // Blob size used during preprocessing (square → padding applied)
    Size blobSize(640, 640);

    // Simulate letterbox padding: image height 480 -> padded to 640
    // padding = (640-480)/2 = 80 on top & bottom
    float pad = 80.0f;

    // Bounding box inside blob coordinates
    // (x, y, width, height)
    std::vector<Rect2f> blobRects = {
        Rect2f(200, 200, 100, 100)
    };

    std::vector<Rect2f> imageRects;

    Image2BlobParams p;
    p.scale = 1.0;         // no scaling
    p.pad_t = pad;
    p.pad_l = 0;
    p.pad_b = pad;
    p.pad_r = 0;

    blobRectsToImageRects(
        blobRects,
        imageRects,
        p
    );

    ASSERT_EQ(imageRects.size(), 1u);

    // Expected: just remove padding effect in Y direction
    Rect2f expected(200, 200 - pad, 100, 100);

    EXPECT_NEAR(imageRects[0].x, expected.x, 1e-4);
    EXPECT_NEAR(imageRects[0].y, expected.y, 1e-4);
    EXPECT_NEAR(imageRects[0].width, expected.width, 1e-4);
    EXPECT_NEAR(imageRects[0].height, expected.height, 1e-4);
}

}}