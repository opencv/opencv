#include "test_precomp.hpp"
#include <opencv2/imgcodecs.hpp>

TEST(Imgcodecs_Imread, HandlesLargeImagesGracefully)
{
    cv::Mat img = cv::imread("large_test_image.jpg");

    // Ensure that if the image is too large, it returns an empty matrix
    if (img.total() > 100000000)  // Example threshold
    {
        ASSERT_TRUE(img.empty());
    }
}

