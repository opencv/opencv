//! [includes]
#include <opencv2/opencv.hpp>
//! [includes]

//! [namespace]
using namespace cv;
//! [namespace]


int main( int argc, char** argv )
{
    //! [load]
    char* imageName = argv[1];
    //! [load]

    //! [mat]
    Mat image;
    //! [mat]

    //! [imread]
    image = imread( imageName , IMREAD_COLOR ); // Read the file
    //! [imread]

    if( argc != 2 || !image.data )
    {
        printf( " No image data \n " );
        return -1;
    }

    //! [mat]
    Mat gray_image;
    //! [mat]

    cvtColor( image, gray_image, COLOR_BGR2GRAY );  // Convert image from BGR to Grayscale format

    //! [imwrite]
    imwrite( "../../images/Gray_Image.jpg", gray_image );
    //! [imwrite]

    //! [window]
    namedWindow( imageName, WINDOW_AUTOSIZE );
    namedWindow( "Gray image", WINDOW_AUTOSIZE );
    //! [window]

    //! [imshow]
    imshow( imageName, image );
    imshow( "Gray image", gray_image );
    //! [imshow]

    //! [wait]
    waitKey(0);
    //! [wait]

    return 0;
}
