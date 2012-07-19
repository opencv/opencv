//Example 12-2. Example of 2D line fitting.

/* License:
   July 20, 2011
   Standard BSD

   BOOK: It would be nice if you cited it:
   Learning OpenCV 2: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   Main OpenCV site
   http://opencv.willowgarage.com/wiki/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/

#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>

using namespace cv;

void  help()
{
	std::cout << "Demonstrate line fitting:\n Usage: ./ch12_ex12_4n" << std::endl;
}

int main( int argc, char** argv )
{
    Mat img(500, 500, CV_8UC3);
    RNG rng(-1);
    help();
    for(;;)
    {
        char key;
        int i, count = rng.uniform(0,100) + 3, outliers = count/5;
        float a = (float)rng.uniform(0., 200.);
        float b = (float)rng.uniform(0., 40.);
        float angle = (float)rng.uniform(0., CV_PI);
        float cos_a = cos(angle), sin_a = sin(angle);
        Point pt1, pt2;
        vector<Point> points(count);
        Vec4f line;
        float d, t;

        b = MIN(a*0.3f, b);

        // generate some points that are close to the line
        for( i = 0; i < count - outliers; i++ )
        {
            float x = (float)rng.uniform(-1.,1.)*a;
            float y = (float)rng.uniform(-1.,1.)*b;
            points[i].x = cvRound(x*cos_a - y*sin_a + img.cols/2);
            points[i].y = cvRound(x*sin_a + y*cos_a + img.rows/2);
        }

        // generate "completely off" points
        for( ; i < count; i++ )
        {
            points[i].x = rng.uniform(0, img.cols);
            points[i].y = rng.uniform(0, img.rows);
        }

        // find the optimal line
        fitLine( points, line, CV_DIST_L1, 1, 0.001, 0.001);

        // draw the points
        img = Scalar::all(0);
        for( i = 0; i < count; i++ )
            circle( img, points[i], 2, i < count - outliers ? Scalar(0, 0, 255) :
                Scalar(0,255,255), CV_FILLED, CV_AA, 0 );

        // ... and the long enough line to cross the whole image
        d = sqrt((double)line[0]*line[0] + (double)line[1]*line[1]);
        line[0] /= d;
        line[1] /= d;
        t = (float)(img.cols + img.rows);
        pt1.x = cvRound(line[2] - line[0]*t);
        pt1.y = cvRound(line[3] - line[1]*t);
        pt2.x = cvRound(line[2] + line[0]*t);
        pt2.y = cvRound(line[3] + line[1]*t);
        cv::line( img, pt1, pt2, Scalar(0,255,0), 3, CV_AA, 0 );

        imshow( "Fit Line", img );

        key = (char)waitKey(0);
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
    return 0;
}
