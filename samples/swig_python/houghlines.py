#!/usr/bin/python
# This is a standalone program. Pass an image name as a first parameter of the program.

import sys
from math import sin,cos,sqrt
from opencv.cv import *
from opencv.highgui import *

# toggle between CV_HOUGH_STANDARD and CV_HOUGH_PROBILISTIC
USE_STANDARD=0

if __name__ == "__main__":
    filename = "../../docs/ref/pics/building.jpg"
    if len(sys.argv)>1:
        filename = sys.argv[1]

    src=cvLoadImage(filename, 0);
    if not src:
        print "Error opening image %s" % filename
        sys.exit(-1)

    dst = cvCreateImage( cvGetSize(src), 8, 1 );
    color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
    storage = cvCreateMemStorage(0);
    lines = 0;
    cvCanny( src, dst, 50, 200, 3 );
    cvCvtColor( dst, color_dst, CV_GRAY2BGR );

    if USE_STANDARD:
        lines = cvHoughLines2( dst, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 100, 0, 0 );

        for i in range(min(lines.total, 100)):
            line = lines[i]
            rho = line[0];
            theta = line[1];
            pt1 = CvPoint();
            pt2 = CvPoint();
            a = cos(theta);
            b = sin(theta);
            x0 = a*rho 
            y0 = b*rho
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            cvLine( color_dst, pt1, pt2, CV_RGB(255,0,0), 3, 8 );

    else:
        lines = cvHoughLines2( dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 50, 50, 10 );
        for line in lines:
            cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );

    cvNamedWindow( "Source", 1 );
    cvShowImage( "Source", src );

    cvNamedWindow( "Hough", 1 );
    cvShowImage( "Hough", color_dst );

    cvWaitKey(0);
