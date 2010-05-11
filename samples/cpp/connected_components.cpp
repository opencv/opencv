#include "cv.h"
#include "highgui.h"

using namespace cv;

Mat img;
int threshval = 100;

void on_trackbar(int, void*)
{
	Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    findContours( bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	
	Mat dst = Mat::zeros(img.size(), CV_8UC3);

    if( contours.size() > 0 )
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
    // the first command line parameter must be file name of binary 
    // (black-n-white) image
    if( !(img=imread(argc == 2 ? argv[1] : "stuff.jpg", 0)).data)
        return -1;

    namedWindow( "Image", 1 );
    imshow( "Image", img );

	namedWindow( "Connected Components", 1 );
	createTrackbar( "Threshold", "Connected Components", &threshval, 255, on_trackbar );
	on_trackbar(threshval, 0);

    waitKey(0);
    return 0;
}
