/**
 * @function Geometric_Transforms_Demo.cpp
 * @brief Demo code for Geometric Transforms
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function main
 */
int main( int argc, char** argv )
{
    //! [Load the image]
    CommandLineParser parser( argc, argv, "{@input | ../data/lena.jpg | input image}" );
    Mat src = imread( parser.get<String>( "@input" ) );
    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    //! [Load the image]

    //! [Set your 3 points to calculate the  Affine Transform]
    Point2f srcTri[3];
    srcTri[0] = Point2f( 0.f, 0.f );
    srcTri[1] = Point2f( src.cols - 1.f, 0.f );
    srcTri[2] = Point2f( 0.f, src.rows - 1.f );

    Point2f dstTri[3];
    dstTri[0] = Point2f( 0.f, src.rows*0.33f );
    dstTri[1] = Point2f( src.cols*0.85f, src.rows*0.25f );
    dstTri[2] = Point2f( src.cols*0.15f, src.rows*0.7f );
    //! [Set your 3 points to calculate the  Affine Transform]

    //! [Get the Affine Transform]
    Mat warp_mat = getAffineTransform( srcTri, dstTri );
    //! [Get the Affine Transform]

    //! [Apply the Affine Transform just found to the src image]
    /// Set the dst image the same type and size as src
    Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );

    warpAffine( src, warp_dst, warp_mat, warp_dst.size() );
    //! [Apply the Affine Transform just found to the src image]

    /** Rotating the image after Warp */

    //! [Compute a rotation matrix with respect to the center of the image]
    Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
    double angle = -50.0;
    double scale = 0.6;
    //! [Compute a rotation matrix with respect to the center of the image]

    //! [Get the rotation matrix with the specifications above]
    Mat rot_mat = getRotationMatrix2D( center, angle, scale );
    //! [Get the rotation matrix with the specifications above]

    //! [Rotate the warped image]
    Mat warp_rotate_dst;
    warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );
    //! [Rotate the warped image]

    //! [Show what you got]
    imshow( "Source image", src );
    imshow( "Warp", warp_dst );
    imshow( "Warp + Rotate", warp_rotate_dst );
    //! [Show what you got]

    //! [Wait until user exits the program]
    waitKey();
    //! [Wait until user exits the program]

    return 0;
}
