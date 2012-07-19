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
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int g_slider_position = 0;
int run = 1, dontset = 0; //start out in single step mode
VideoCapture cap;


void onTrackbarSlide(int pos, void *) {
	cap.set(CV_CAP_PROP_POS_FRAMES,pos);
	if(!dontset)
		run = 1;
	dontset = 0;
}

void help()
{
	cout << "\n./ch2_ex2_3 tree.avi\n" << endl;
    cout << "'s' to single step\n'r' to run.\nTrack bar causes single step mode.\n'h'help\nESC to quit\n" << endl;
}
int main( int argc, char** argv ) {
	namedWindow( "Example2_3", CV_WINDOW_AUTOSIZE );
    if(argc > 1) cap.open(string(argv[1]));
    else {cap.open(0); cerr << "Call: ./ch2_ex2_3 tree.avi" << endl; return -1;}
    help();
    int frames = (int) cap.get(CV_CAP_PROP_FRAME_COUNT);
    int tmpw = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int tmph = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "Video has " << frames << " frames of dimensions(" << tmpw << ", " << tmph << ")." << std::endl;

    createTrackbar(
        "Position",
        "Example2_3",
        &g_slider_position,
        frames,
        onTrackbarSlide
    );
    Mat frame;
    frames = 0;
    int single_step = 0; //Start out free run
    while(1) {
    	if(run != 0)
    	{
    		cap >> frame;
    		if(!frame.data) break;
    		frames = (int)cap.get(CV_CAP_PROP_POS_FRAMES);
    		dontset = 1;
    		setTrackbarPos("Position","Example2_3",frames); //This causes a call to onTrackbarSlide()
    		imshow( "Example2_3", frame );
    		run-=1;
    	}
    	char c = (char)waitKey(10);
    	if(c == 'h') help();
    	if(c == 's') //single step
    		{run = 1; cout << "Single step, run = " << run << endl;}
    	if(c == 'r') //run mode
    		{run = -1; cout << "Run mode, run = " << run <<endl;}
    	if( c == 27 )
    		break;
    }
    return(0);
}
