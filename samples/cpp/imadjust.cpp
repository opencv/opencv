#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;

int lowFract = 1;
int highFract = 99;

Mat image, gray, adjusted;

// define a trackbar callback
static void onTrackbar(int, void*)
{
    try {
        double lowIn, highIn;
        stretchlim(gray, &lowIn, &highIn, lowFract/100.0, highFract/100.0);
        imadjust(gray, adjusted, lowIn, highIn);
    } catch (Exception& e) {
    }
    imshow("imadjust", adjusted);
}

static void help()
{
    printf("\nThis sample demonstrates Canny edge detection\n"
           "Call:\n"
           "    /.edge [image_name -- Default is fruits.jpg]\n\n");
}

const char* keys =
{
    "{1| |fruits.jpg|input image name}"
};

int main( int argc, const char** argv )
{
    help();

    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>("1");

    image = imread(filename, 1);
    if(image.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }
    adjusted.create(image.size(), CV_8U);
    cvtColor(image, gray, CV_BGR2GRAY);

    // Create a window
    namedWindow("imadjust", 1);

    // create a toolbar
    createTrackbar("Low fraction", "imadjust", &lowFract, 100, onTrackbar);
    createTrackbar("High fraction", "imadjust", &highFract, 100, onTrackbar);

    // Show the image
    onTrackbar(0, 0);

    // Wait for a key stroke; the same function arranges events processing
    waitKey(0);

    return 0;
}
