/**
 * @function pointPolygonTest_demo.cpp
 * @brief Demo code to use the pointPolygonTest function...fairly easy
 * @author OpenCV team
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function main
 */
int main( void )
{
    /// Create an image
    const int r = 100;
    Mat src = Mat::zeros( Size( 4*r, 4*r ), CV_8U );

    /// Create a sequence of points to make a contour
    vector<Point2f> vert(6);
    vert[0] = Point( 3*r/2, static_cast<int>(1.34*r) );
    vert[1] = Point( 1*r, 2*r );
    vert[2] = Point( 3*r/2, static_cast<int>(2.866*r) );
    vert[3] = Point( 5*r/2, static_cast<int>(2.866*r) );
    vert[4] = Point( 3*r, 2*r );
    vert[5] = Point( 5*r/2, static_cast<int>(1.34*r) );

    /// Draw it in src
    for( int i = 0; i < 6; i++ )
    {
        line( src, vert[i],  vert[(i+1)%6], Scalar( 255 ), 3 );
    }

    /// Get the contours
    vector<vector<Point> > contours;
    findContours( src, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    /// Calculate the distances to the contour
    Mat raw_dist( src.size(), CV_32F );
    for( int i = 0; i < src.rows; i++ )
    {
        for( int j = 0; j < src.cols; j++ )
        {
            raw_dist.at<float>(i,j) = (float)pointPolygonTest( contours[0], Point2f((float)j, (float)i), true );
        }
    }

    double minVal, maxVal;
    Point maxDistPt; // inscribed circle center
    minMaxLoc(raw_dist, &minVal, &maxVal, NULL, &maxDistPt);
    minVal = abs(minVal);
    maxVal = abs(maxVal);

    /// Depicting the  distances graphically
    Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
    for( int i = 0; i < src.rows; i++ )
    {
        for( int j = 0; j < src.cols; j++ )
        {
            if( raw_dist.at<float>(i,j) < 0 )
            {
                drawing.at<Vec3b>(i,j)[0] = (uchar)(255 - abs(raw_dist.at<float>(i,j)) * 255 / minVal);
            }
            else if( raw_dist.at<float>(i,j) > 0 )
            {
                drawing.at<Vec3b>(i,j)[2] = (uchar)(255 - raw_dist.at<float>(i,j) * 255 / maxVal);
            }
            else
            {
                drawing.at<Vec3b>(i,j)[0] = 255;
                drawing.at<Vec3b>(i,j)[1] = 255;
                drawing.at<Vec3b>(i,j)[2] = 255;
            }
        }
    }
    circle(drawing, maxDistPt, (int)maxVal, Scalar(255,255,255));

    /// Show your results
    imshow( "Source", src );
    imshow( "Distance and inscribed circle", drawing );

    waitKey();
    return 0;
}
