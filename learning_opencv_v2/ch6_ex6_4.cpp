//
// Example 6-4. Log-polar transform example
// logPolar.cpp : Defines the entry point for the console applicatio
// 
//
//   input to second cvLogPolar does not need "CV_WARP_FILL_OUTLIERS": "M, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP );" --> "M, CV_INTER_LINEAR+CV_WARP_INVERSE_MAP );"
//
//
// ./ch6_ex6_4 image m
//     Where m is the scale factor, which should be set so that the 
//     features of interest dominate the available image area.
//
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

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if(argc != 3) { cout << "LogPolar\nUsage: ch6_ex6_4 <imagename> <M value>\n<M value>~30 is usually good enough\n"; return -1; }
    
    Mat src = imread(argv[1],1);
    
    if( src.empty() ) { cout << "Can not load " << argv[1] << endl; return -1; } 
    
    double M = atof(argv[2]);
    Mat dst(src.size(), src.type()), src2(src.size(), src.type());
    
    IplImage c_src = src, c_dst = dst, c_src2 = src2;

    cvLogPolar( &c_src,  &c_dst, Point2f(src.cols*0.5f, src.rows*0.5f),
                M, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS );
    cvLogPolar( &c_dst, &c_src2, Point2f(src.cols*0.5f, src.rows*0.5f),
                M, CV_INTER_LINEAR+CV_WARP_INVERSE_MAP );
    imshow( "log-polar", dst );
    imshow( "inverse log-polar", src2 );
    waitKey();
    return 0;
}
