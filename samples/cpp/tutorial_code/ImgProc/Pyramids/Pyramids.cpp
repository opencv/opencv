/**
 * @file Pyramids.cpp
 * @brief Sample code of image pyramids (pyrDown and pyrUp)
 * @author OpenCV team
 */

#include "iostream"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

const char* window_name = "Pyramids Demo";

/**
 * @function main
 */
int main( int argc, char** argv )
{
    /// General instructions
    cout << "\n Zoom In-Out demo \n "
            "------------------  \n"
            " * [i] -> Zoom in   \n"
            " * [o] -> Zoom out  \n"
            " * [ESC] -> Close program \n" << endl;

    //![load]
    const char* filename = argc >=2 ? argv[1] : "chicky_512.png";

    // Loads an image
    Mat src = imread( samples::findFile( filename ) );

    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default chicky_512.png] \n");
        return EXIT_FAILURE;
    }
    //![load]

    //![loop]
    for(;;)
    {
        //![show_image]
        imshow( window_name, src );
        //![show_image]
        char c = (char)waitKey(0);

        if( c == 27 )
        { break; }
        //![pyrup]
        else if( c == 'i' )
        { pyrUp( src, src, Size( src.cols*2, src.rows*2 ) );
            printf( "** Zoom In: Image x 2 \n" );
        }
        //![pyrup]
        //![pyrdown]
        else if( c == 'o' )
        { pyrDown( src, src, Size( src.cols/2, src.rows/2 ) );
            printf( "** Zoom Out: Image / 2 \n" );
        }
        //![pyrdown]
    }
    //![loop]

    return EXIT_SUCCESS;
}
