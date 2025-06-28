#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>

using namespace cv;

static void help(char** argv)
{

printf("\nShow off image morphology: erosion, dialation, open and close\n"
    "Call:\n   %s [image]\n"
    "This program also shows use of rect, ellipse, cross and diamond kernels\n\n", argv[0]);
printf( "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tr - use rectangle structuring element\n"
    "\te - use elliptic structuring element\n"
    "\tc - use cross-shaped structuring element\n"
    "\td - use diamond-shaped structuring element\n"
    "\tSPACE - loop through all the options\n" );
}

Mat src, dst;

int element_shape = MORPH_RECT;

//the address of variable which receives trackbar position update
int max_iters = 10;
int open_close_pos = 0;
int erode_dilate_pos = 0;

// callback function for open/close trackbar
static void OpenClose(int, void*)
{
    int n = open_close_pos;
    int an = abs(n);
    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an) );
    if( n < 0 )
        morphologyEx(src, dst, MORPH_OPEN, element);
    else
        morphologyEx(src, dst, MORPH_CLOSE, element);
    imshow("Open/Close",dst);
}

// callback function for erode/dilate trackbar
static void ErodeDilate(int, void*)
{
    int n = erode_dilate_pos;
    int an = abs(n);
    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an) );
    if( n < 0 )
        erode(src, dst, element);
    else
        dilate(src, dst, element);
    imshow("Erode/Dilate",dst);
}


int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, "{help h||}{ @image | baboon.jpg | }");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    std::string filename = samples::findFile(parser.get<std::string>("@image"));
    if( (src = imread(filename,IMREAD_COLOR)).empty() )
    {
        help(argv);
        return -1;
    }

    //create windows for output images
    namedWindow("Open/Close",1);
    namedWindow("Erode/Dilate",1);

    open_close_pos = erode_dilate_pos = max_iters;
    createTrackbar("iterations", "Open/Close",&open_close_pos,max_iters*2+1,OpenClose);
    setTrackbarMin("iterations", "Open/Close", -max_iters);
    setTrackbarMax("iterations", "Open/Close", max_iters);
    setTrackbarPos("iterations", "Open/Close", 0);

    createTrackbar("iterations", "Erode/Dilate",&erode_dilate_pos,max_iters*2+1,ErodeDilate);
    setTrackbarMin("iterations", "Erode/Dilate", -max_iters);
    setTrackbarMax("iterations", "Erode/Dilate", max_iters);
    setTrackbarPos("iterations", "Erode/Dilate", 0);

    for(;;)
    {
        OpenClose(open_close_pos, 0);
        ErodeDilate(erode_dilate_pos, 0);
        char c = (char)waitKey(0);

        if( c == 27 )
            break;
        if( c == 'e' )
            element_shape = MORPH_ELLIPSE;
        else if( c == 'r' )
            element_shape = MORPH_RECT;
        else if( c == 'c' )
            element_shape = MORPH_CROSS;
        else if( c == 'd' )
            element_shape = MORPH_DIAMOND;
        else if( c == ' ' )
            element_shape = (element_shape + 1) % 4;
    }

    return 0;
}
