#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;


const char* window_name1 = "Original Image";
const char* window_name2 = "Reconverted image";

static void help(const char** argv)
{
    printf("\nThis sample demonstrates conversion from RGB to YUV422 (UYVY) and back.\n"
           "Call:\n"
           "    %s [image_name -- Default is fruits.jpg]\n\n", argv[0]);
}

const char* keys =
{
    "{help h||}{@image |fruits.jpg|input image name}"
};

int main( int argc, const char** argv )
{
    help(argv);
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    Mat image, image_yuv, image_yuv2bgr;

    Mat orig_img = imread(samples::findFile(filename), IMREAD_COLOR);
    if(orig_img.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help(argv);
        return -1;
    }

    image = orig_img.clone();

    // Forward conversion to UYVY
    cvtColor(image, image_yuv, COLOR_BGR2YUV_UYVY);

    // Backward conversion from UYVY to BGR (gives the reconverted image)
    cvtColor(image_yuv, image_yuv2bgr, COLOR_YUV2BGR_UYVY);

    // Create a window
    namedWindow(window_name1, 1);
    namedWindow(window_name2, 1);

    // Display the original and reconverted images.
    imshow(window_name1, orig_img);
    imshow(window_name2, image_yuv2bgr);

    // Wait for a key stroke; the same function arranges events processing
    waitKey(0);

    return 0;
}
