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

void help()
{
	cout << "Call: ./ch2_ex2_20 tree.avi [output_file_name.avi] \n" << endl;
	cout << "Convert a video to grayscale" << endl;
	cout << "argv[1]: input video file" << endl;
	cout << "if argv[2], name of new output file\n" << endl;
}



//#define NOWRITE 1;   //Turn this on (removed the first comment out "//" if you can't write on linux

int NOWRITE = 1;  //Don't write unless we get an output file name

int main( int argc, char* argv[] )
{
	if(argc > 2)
		NOWRITE = 0;
	cout << "nowrite = " << NOWRITE << endl;
	namedWindow( "Example2_10", CV_WINDOW_AUTOSIZE );
	namedWindow( "Log_Polar", CV_WINDOW_AUTOSIZE );
	Mat bgr_frame;
	VideoCapture capture;
	if( argc < 2 || !capture.open( argv[1] ) ){
		help();
		cout << "Failed to open " << argv[1] << "\n" << endl;
		return -1;
	}

	double fps = capture.get(CV_CAP_PROP_FPS);
	cout << "fps = " << fps << endl;
	Size size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),
			(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	cout << " frame (w, h) = (" << size.width << ", " << size.height << ")" <<endl;
	VideoWriter writer;
	if(! NOWRITE)
	{ writer.open(  // On linux Will only work if you've installed ffmpeg development files correctly,
			argv[2],                               // otherwise segmentation fault.  Windows probably better.
			CV_FOURCC('M','J','P','G'),
			fps,
			size
	);
	}
	Mat logpolar_frame(size,CV_8UC3);
	Mat gray_frame(size,CV_8UC1);

	for(;;) {
		capture >> bgr_frame;
		if( bgr_frame.empty() ) break;
		imshow( "Example2_10", bgr_frame );
		cvtColor(   //We never make use of this gray image
				bgr_frame, gray_frame, CV_BGR2GRAY);
		IplImage lp = logpolar_frame;
		IplImage bgrf = bgr_frame;
		cvLogPolar( &bgrf, &lp,  //This is just a fun conversion the mimic's the human visual system
				cvPoint2D32f(bgr_frame.cols/2,
						bgr_frame.rows/2),
						40,
						CV_WARP_FILL_OUTLIERS );
		imshow( "Log_Polar", logpolar_frame );
		//Sigh, on linux, depending on your ffmpeg, this often won't work ...
		if(! NOWRITE)
			writer << logpolar_frame;
		char c = waitKey(10);
		if( c == 27 ) break;
	}

	capture.release();
}
