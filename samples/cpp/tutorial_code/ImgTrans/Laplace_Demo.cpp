/**
 * @file Laplace_Demo.cpp
 * @brief Sample code showing how to detect edges using the Laplace operator
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

/**
 * @function main
 */
int main( int argc, char** argv )
{
    //![variables]
    // Declare the variables we are going to use
    Mat src, src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    const char* window_name = "Laplace Demo";
    //![variables]

    //![load]
    const char* imageName = argc >=2 ? argv[1] : "../data/lena.jpg";

    src = imread( imageName, IMREAD_COLOR ); // Load an image

    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default ../data/lena.jpg] \n");
        return -1;
    }
    //![load]

    //![reduce_noise]
    // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur( src, src, Size(3, 3), 0, 0, BORDER_DEFAULT );
    //![reduce_noise]

    //![convert_to_gray]
    cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to grayscale
    //![convert_to_gray]

    /// Apply Laplace function
    Mat abs_dst;
    //![laplacian]
    Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    //![laplacian]

    //![convert]
    // converting back to CV_8U
    convertScaleAbs( dst, abs_dst );
    //![convert]

    //![display]
    imshow( window_name, abs_dst );
    waitKey(0);
    //![display]

    return 0;
}
