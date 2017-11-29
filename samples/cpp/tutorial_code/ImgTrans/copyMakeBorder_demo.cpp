/**
 * @file copyMakeBorder_demo.cpp
 * @brief Sample code that shows the functionality of copyMakeBorder
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

//![variables]
// Declare the variables
Mat src, dst;
int top, bottom, left, right;
int borderType = BORDER_CONSTANT;
const char* window_name = "copyMakeBorder Demo";
RNG rng(12345);
//![variables]

/**
 * @function main
 */
int main( int argc, char** argv )
{
    //![load]
    const char* imageName = argc >=2 ? argv[1] : "../data/lena.jpg";

    // Loads an image
    src = imread( imageName, IMREAD_COLOR ); // Load an image

    // Check if image is loaded fine
    if( src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default ../data/lena.jpg] \n");
        return -1;
    }
    //![load]

    // Brief how-to for this program
    printf( "\n \t copyMakeBorder Demo: \n" );
    printf( "\t -------------------- \n" );
    printf( " ** Press 'c' to set the border to a random constant value \n");
    printf( " ** Press 'r' to set the border to be replicated \n");
    printf( " ** Press 'ESC' to exit the program \n");

    //![create_window]
    namedWindow( window_name, WINDOW_AUTOSIZE );
    //![create_window]

    //![init_arguments]
    // Initialize arguments for the filter
    top = (int) (0.05*src.rows); bottom = top;
    left = (int) (0.05*src.cols); right = left;
    //![init_arguments]

    for(;;)
    {
        //![update_value]
        Scalar value( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
        //![update_value]

        //![copymakeborder]
        copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
        //![copymakeborder]

        //![display]
        imshow( window_name, dst );
        //![display]

        //![check_keypress]
        char c = (char)waitKey(500);
        if( c == 27 )
        { break; }
        else if( c == 'c' )
        { borderType = BORDER_CONSTANT; }
        else if( c == 'r' )
        { borderType = BORDER_REPLICATE; }
        //![check_keypress]
    }

    return 0;
}
