#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;

double timeStepSize = 1.0;
double k = 0.02;
int noOfTimeSteps_Exp=10;
int noOfTimeSteps_InvQuad=10;

Mat src, src_gray, dst_Exp, dst_InvQuad;

const char* window_name1 = "Anisodiff : Exponential Flux";
const char* window_name2 = "Anisodiff : Inverse Quadratic Flux"; 

// define a trackbar callback
static void onTrackbar(int, void*)
{
    PeronaMalik(src_gray, dst_Exp, timeStepSize, k, noOfTimeSteps_Exp, CV_PERONA_MALIK_EXPONENTIAL);
    imshow(window_name1, dst_Exp);

    PeronaMalik(src_gray, dst_InvQuad, timeStepSize, k, noOfTimeSteps_InvQuad, CV_PERONA_MALIK_INVERSE_QUADRATIC);
    imshow(window_name2, dst_InvQuad);
}

static void help()
{
    printf("\nThis sample demonstrates perona-malik anisotropic diffusion \n"
           "Call:\n"
           "    /.anisodiff [image_name -- Default is ../data/fruits.jpg]\n\n");
}

const char* keys =
{
    "{help h||}{@image |../data/fruits.jpg|input image name}"
};

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>(0);

    src = imread(filename, 1);
    if(src.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }
    cvtColor( src, src_gray, CV_BGR2GRAY );
    src_gray.convertTo(src_gray, CV_64FC1, 1/255.0);
    dst_Exp.create(src_gray.size(), src_gray.type());
    dst_InvQuad.create(src_gray.size(), src_gray.type());

    // Create a window
    namedWindow(window_name1, 1);
    namedWindow(window_name2, 1);

    // create a toolbar
    createTrackbar("No. of time steps", window_name1, &noOfTimeSteps_Exp, 30, onTrackbar);
    createTrackbar("No. of time steps", window_name2, &noOfTimeSteps_InvQuad, 30, onTrackbar);

    // Show the image
    onTrackbar(0, 0);

    // Wait for a key stroke; the same function arranges events processing
    waitKey(0);

    return 0;
}
