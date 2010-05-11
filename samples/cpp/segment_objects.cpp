#include "cvaux.h"
#include "highgui.h"
#include <stdio.h>
#include <string>

using namespace cv;


void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
    int niters = 3;
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    Mat temp;
    
    dilate(mask, temp, Mat(), Point(-1,-1), niters);
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    dilate(temp, temp, Mat(), Point(-1,-1), niters);
    
    findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	
	dst = Mat::zeros(img.size(), CV_8UC3);
    
    if( contours.size() == 0 )
        return;
        
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0, largestComp = 0;
    double maxArea = 0;
    
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;
        }
    }
    Scalar color( 0, 0, 255 );
    drawContours( dst, contours, largestComp, color, CV_FILLED, 8, hierarchy );
}


int main(int argc, char** argv)
{
    VideoCapture cap;
    bool update_bg_model = true;
    
    if( argc < 2 )
        cap.open(0);
    else
        cap.open(std::string(argv[1]));
    
    if( !cap.isOpened() )
    {
        printf("can not open camera or video file\n");
        return -1;
    }
    
    Mat tmp_frame, bgmask, out_frame;
    
    cap >> tmp_frame;
    if(!tmp_frame.data)
    {
        printf("can not read data from the video source\n");
        return -1;
    }
    
    namedWindow("video", 1);
    namedWindow("segmented", 1);
    
    BackgroundSubtractorMOG bgsubtractor;
    bgsubtractor.noiseSigma = 10;
    
    for(;;)
    {
        cap >> tmp_frame;
        if( !tmp_frame.data )
            break;
        bgsubtractor(tmp_frame, bgmask, update_bg_model ? -1 : 0);
        //CvMat _bgmask = bgmask;
        //cvSegmentFGMask(&_bgmask);
        refineSegments(tmp_frame, bgmask, out_frame);
        imshow("video", tmp_frame);
        imshow("segmented", out_frame);
        char keycode = waitKey(30);
        if( keycode == 27 )
            break;
        if( keycode == ' ' )
            update_bg_model = !update_bg_model;
    }
    
    return 0;
}
