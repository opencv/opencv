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

using namespace cv;
using namespace std;


void example2_4( Mat & image )
{
    // Create some windows to show the input
    // and output images in.
    //
    namedWindow( "Example2_4-in", CV_WINDOW_AUTOSIZE );
    namedWindow( "Example2_4-out", CV_WINDOW_AUTOSIZE );
    
    // Create a window to show our input image
    //
    imshow( "Example2_4-in", image );
    
    // Create an image to hold the smoothed output
    Mat out;
    
    // Do the smoothing
    //  Could use GaussianBlur(), blur(), medianBlur() or bilateralFilter().
    GaussianBlur(image, out, Size(5,5),3,3);
    GaussianBlur(out,out,Size(5,5),3,3);
    
    // Show the smoothed image in the output window
    //
    imshow( "Example2_4-out", out );

    // Wait for the user to hit a key, windows will self destruct
    //
    waitKey( 0 );
}

void help()
{
	cout << "Call: ./ch2_ex2_4 faceScene.jpg" << endl;
}

int main( int argc, char** argv )
{
	help();
	cv::Mat img = imread(argv[1],-1);
	if(img.empty())
	{
		std::cerr << "Couldn't open the image " << argv[1] << std::endl;
		return -1;
	}
	example2_4( img );
}

