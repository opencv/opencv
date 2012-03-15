/* This sample code was originally provided by Liu Liu
 * Copyright (C) 2009, Liu Liu All rights reserved.
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void help()
{
	cout << "\nThis program demonstrates the Maximal Extremal Region interest point detector.\n"
    "It finds the most stable (in size) dark and white regions as a threshold is increased.\n"
    "\nCall:\n"
    "./mser_sample <path_and_image_filename, Default is 'puzzle.png'>\n\n";
}

static const Vec3b bcolors[] = 
{
    Vec3b(0,0,255),
    Vec3b(0,128,255),
    Vec3b(0,255,255),
    Vec3b(0,255,0),
    Vec3b(255,128,0),
    Vec3b(255,255,0),
    Vec3b(255,0,0),
    Vec3b(255,0,255),
    Vec3b(255,255,255)
};

int main( int argc, char** argv )
{
	string path;
	Mat img0, img, yuv, gray, ellipses;
	help();
    
    img0 = imread( argc != 2 ? "puzzle.png" : argv[1], 1 );
    if( img0.empty() )
    {
        if( argc != 2 )
            cout << "\nUsage: mser_sample <path_to_image>\n";
        else
            cout << "Unable to load image " << argv[1] << endl;
        return 0;
    }
    
	cvtColor(img0, yuv, COLOR_BGR2YCrCb);
    cvtColor(img0, gray, COLOR_BGR2GRAY);
    cvtColor(gray, img, COLOR_GRAY2BGR);
    img.copyTo(ellipses);
    
    vector<vector<Point> > contours;
	double t = (double)getTickCount();
    MSER()(yuv, contours);
	t = (double)getTickCount() - t;
	printf( "MSER extracted %d contours in %g ms.\n", (int)contours.size(),
           t*1000./getTickFrequency() );
    
	// draw mser's with different colors
	for( int i = (int)contours.size()-1; i >= 0; i-- )
	{
		const vector<Point>& r = contours[i];
		for ( int j = 0; j < (int)r.size(); j++ )
		{
			Point pt = r[j];
			img.at<Vec3b>(r[j]) = bcolors[i%9];
		}
        
        // find ellipse (it seems cvfitellipse2 have error or sth?)
        RotatedRect box = fitEllipse( r );
        
        box.angle=(float)CV_PI/2-box.angle;
        ellipse( ellipses, box, Scalar(196,255,255), 2 );
	}
    
	imshow( "original", img0 );
	imshow( "response", img );
	imshow( "ellipses", ellipses );
    
	waitKey(0);
}
