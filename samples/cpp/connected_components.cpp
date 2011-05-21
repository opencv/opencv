#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat img;
int threshval = 100;

void help()
{
	cout <<
		   "\n This program demonstrates connected components and use of the trackbar\n"
		<< "\n"
		<< "Usage: ./connected_components <image>\n"
		<< "\n"
		<< "The image is converted to grayscale and displayed, another image has a trackbar\n"
		<< "that controls thresholding and thereby the extracted contours which are drawn in color\n"
		<< endl;
}


void on_trackbar(int, void*)
{
	Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    findContours( bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	
	Mat dst = Mat::zeros(img.size(), CV_8UC3);

    if( !contours.empty() && !hierarchy.empty() )
    {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            Scalar color( (rand()&255), (rand()&255), (rand()&255) );
            drawContours( dst, contours, idx, color, CV_FILLED, 8, hierarchy );
        }
    }

	imshow( "Connected Components", dst );
}

int main( int argc, char** argv )
{
    // the first command line parameter
	if(argc != 2)
	{
		help();
		return -1;
	}
    if( !(img=imread(argc == 2 ? argv[1] : "stuff.jpg", 0)).data) //The ending 0 in imread converts the image to grayscale.
        return -1;

    namedWindow( "Image", 1 );
    imshow( "Image", img );

	namedWindow( "Connected Components", 1 );
	createTrackbar( "Threshold", "Connected Components", &threshval, 255, on_trackbar );
	on_trackbar(threshval, 0);

    waitKey(0);
    return 0;
}
