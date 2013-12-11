#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

static void help()
{

printf("\nShow off image morphology: erosion, dialation, open and close\n"
    "Call:\n   morphology2 [image]\n"
    "This program also shows use of rect, elipse and cross kernels\n\n");
printf( "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tr - use rectangle structuring element\n"
    "\te - use elliptic structuring element\n"
    "\tc - use cross-shaped structuring element\n"
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
    int n = open_close_pos - max_iters;
    int an = n > 0 ? n : -n;
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
    int n = erode_dilate_pos - max_iters;
    int an = n > 0 ? n : -n;
    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an) );
    if( n < 0 )
        erode(src, dst, element);
    else
        dilate(src, dst, element);
    imshow("Erode/Dilate",dst);
}


int main( int argc, char** argv )
{
    char* filename = argc == 2 ? argv[1] : (char*)"baboon.jpg";
    if( (src = imread(filename,1)).data == 0 )
        return -1;

    help();

    //create windows for output images
    namedWindow("Open/Close",1);
    namedWindow("Erode/Dilate",1);

    open_close_pos = erode_dilate_pos = max_iters;
    createTrackbar("iterations", "Open/Close",&open_close_pos,max_iters*2+1,OpenClose);
    createTrackbar("iterations", "Erode/Dilate",&erode_dilate_pos,max_iters*2+1,ErodeDilate);

    for(;;)
    {
        int c;

        OpenClose(open_close_pos, 0);
        ErodeDilate(erode_dilate_pos, 0);
        c = waitKey(0);

        if( (char)c == 27 )
            break;
        if( (char)c == 'e' )
            element_shape = MORPH_ELLIPSE;
        else if( (char)c == 'r' )
            element_shape = MORPH_RECT;
        else if( (char)c == 'c' )
            element_shape = MORPH_CROSS;
        else if( (char)c == ' ' )
            element_shape = (element_shape + 1) % 3;
    }

    return 0;
}
