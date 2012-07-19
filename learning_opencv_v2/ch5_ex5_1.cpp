//
//Example 5-1. Doing something with each element in the sequence of connected components returned
//             by cvPyrSegmentation(
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
#include <opencv2/legacy/legacy.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void f( 
  const Mat& src,
  Mat& dst
) {
  Ptr<CvMemStorage> storage = cvCreateMemStorage(0);
  Seq<CvConnectedComp> comp;

  dst.create(src.size(), src.type());
  IplImage c_src = src, c_dst = dst;
  cvPyrSegmentation( &c_src, &c_dst, storage, &comp.seq, 4, 200, 50 );
  int n_comp = comp.size();

  for( int i=0; i<n_comp; i++ ) {
	CvConnectedComp cc = comp[i];
    // do_something_with( cc );
	//rectangle(dst, cc.rect, Scalar(128, 255, 255), 1);  
  }
}

void help()
{
	cout << "Call: ./ch5_ex5_1 stuff.jpg" << endl;
	cout << "Shows pyramid segmentation." << endl;
}

int main(int argc, char** argv)
{
	help();
	if(argc < 2) { cout << "specify input image" << endl; return -1; }
	// Create a named window with a the name of the file.
	namedWindow( argv[1], 1 );
	// Load the image from the given file name.
	Mat src = imread(argv[1]), dst;
	namedWindow("Raw",1);
	imshow("Raw",src);
	if(src.empty()) { cout << "Couldn't seem to open " << argv[1] << ", sorry\n"; return -1;}
	f( src, dst);

	// Show the image in the named window
	imshow( argv[1], dst );

	// Idle until the user hits any key.
	waitKey();
	return 0;
}
