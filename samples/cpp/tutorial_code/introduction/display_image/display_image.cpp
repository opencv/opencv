//! [includes]
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
//! [includes]

//! [namespace]
using namespace cv;
using namespace std;
//! [namespace]


int main( int argc, char** argv )
{
    //! [load]
    String imageName( "HappyFish.jpg" ); // by default
    if( argc > 1)
    {
        imageName = argv[1];
    }
    //! [load]

    //! [mat]
    Mat image;
    //! [mat]

    //! [imread]
    image = imread( samples::findFile( imageName ), IMREAD_COLOR ); // Read the file
    //! [imread]

    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    //! [window]
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    //! [window]

    //! [imshow]
    imshow( "Display window", image );                // Show our image inside it.
    //! [imshow]

    //! [wait]
    waitKey(0); // Wait for a keystroke in the window
    //! [wait]
    return 0;
}
