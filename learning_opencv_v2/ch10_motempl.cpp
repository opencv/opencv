// This is the motion template code motempl.c from the OpenCV samples/c directory.  
//   It's just too fun not to also include here and is described in detail in Chapter 10.
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
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <iostream>

using namespace cv;
using namespace std;

void help()
{
cout << "\n\n"
"This one uses a video camera to demonstrate motion templates in chapter 10.\n"
"  \n"
" 1 Point the camera away from you\n"
" 2 Start the program\n"
" 3 Wait a few seconds until the video window goes black\n"
" 4 Move in front of the camera and watch the templates being drawn along with\n"
"   their segmented motion vectors.\n"
"   \n"
"Ussage: \n"
"./ch10_motempl\n"
"\n";
}


// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)
const int N = 4;

// ring image buffer
vector<Mat> buf(N);
int last=0;

Mat mhi; // MHI

// parameters:
//  img - input video frame
//  dst - resultant motion picture
//  args - optional parameters
void  update_mhi( const Mat& img, Mat& dst, int diff_threshold )
{
    double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds
    int idx1 = last, idx2;
    Mat silh, orient, mask, segmask;

    cvtColor( img, buf[last], CV_BGR2GRAY ); // convert frame to grayscale

    idx2 = (last + 1) % N; // index of (last - (N-1))th frame
    last = idx2;

    if( buf[idx1].size() != buf[idx2].size() )
        silh = Mat::ones(img.size(), CV_8U)*255;
    else
        absdiff(buf[idx1], buf[idx2], silh); // get difference between frames
    
    threshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it
    if( mhi.empty() )
        mhi = Mat::zeros(silh.size(), CV_32F);
    updateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // update MHI

    // convert MHI to blue 8u image
    mhi.convertTo(mask, CV_8U, 255./MHI_DURATION,
                (MHI_DURATION - timestamp)*255./MHI_DURATION);
    dst = Mat::zeros(mask.size(), CV_8UC3);
    insertChannel(mask, dst, 0);

    // calculate motion gradient orientation and valid orientation mask
    calcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
    
    // segment motion: get sequence of motion components
    // segmask is marked motion components map. It is not used further
    vector<Rect> brects;
    segmentMotion(mhi, segmask, brects, timestamp, MAX_TIME_DELTA );

    // iterate through the motion components,
    // One more iteration (i == -1) corresponds to the whole image (global motion)
    for( int i = -1; i < (int)brects.size(); i++ ) {
        Rect roi; Scalar color; double magnitude;
        Mat maski = mask;
        if( i < 0 ) { // case of the whole image
            roi = Rect(0, 0, img.cols, img.rows);
            color = Scalar::all(255);
            magnitude = 100;
        }
        else { // i-th motion component
            roi = brects[i];
            if( roi.area() < 3000 ) // reject very small components
                continue;
            color = Scalar(0, 0, 255);
            magnitude = 30;
            maski = mask(roi);
        }
        
        // calculate orientation
        double angle = calcGlobalOrientation( orient(roi), maski, mhi(roi), timestamp, MHI_DURATION);
        angle = 360.0 - angle;  // adjust for images with top-left origin

        int count = norm( silh, NORM_L1 ); // calculate number of points within silhouette ROI
        // check for the case of little motion
        if( count < roi.area() * 0.05 )
            continue;

        // draw a clock with arrow indicating the direction
        Point center( roi.x + roi.width/2, roi.y + roi.height/2 );
        circle( dst, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );
        line( dst, center, Point( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
                cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );
    }
}


int main(int argc, char** argv)
{
    Mat motion;
    VideoCapture capture;
    
    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        capture.open(argc > 1 ? argv[1][0] - '0' : 0);
    else if( argc == 2 )
        capture.open(argv[1]);

    if( !capture.isOpened() )
    {
        cout << "\nFailed to open camera or file\n";
        return -1;
    }
        
    help();
    
    for(;;)
    {
        Mat image;
        capture >> image;

        if( image.empty() )
            break;

        update_mhi( image, motion, 30 );
        imshow( "Motion", motion );

        if( waitKey(10) >= 0 )
            break;
    }

    return 0;
}
