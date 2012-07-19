// ch7_ex7_5_HistBackProj  OK, OK, this isn't in the book, its' "extra"
//     We cut the source code for actually doing back project in the book
//     but here it is no extra charge.
//   Gary Bradski July, 28, 2011
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
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

void help(){
cout << "\nDemonstrate histogram back projection\nUsage:\n" <<
  "  ./ch7_ex7_5_HistBackProj modelImage testImage\n\n" <<
  "     Example call: ./ch7_ex7_5 BlueCup.jpg adrian.jpg\n" <<
  "       (notice that the blue color and flesh color are picked up\n" <<
  "     Projection is done using calcBackProject()\n\n" << endl;
 }

int main( int argc, char** argv ) {

    Mat src[2],dst; //dst is what to display backproject on
	int i;
    if( argc == 3){
		//Load 2 images, first on is to build histogram of, 2nd is to run on
		for(i = 1; i<3; ++i){
			src[i-1] = imread(argv[i],1);
			if(src[i-1].empty()) {
				cerr << "Error on reading image "<<i<<": " << argv[i] << "\n" << endl;;
				return(-1);
			}
		}
        // Compute the HSV image, and decompose it into separate planes.
        //
		Mat hsv[2], hist[2], hist_img[2];
	    int h_bins = 16, s_bins = 16;
	    int ch[] = {0, 1};
        int    hist_size[] = { h_bins, s_bins };
        float  h_ranges[]  = { 0, 180 };          // hue is [0,180]
        float  s_ranges[]  = { 0, 255 }; 
        const float* ranges[]    = { h_ranges, s_ranges };
		int scale = 10;
#define patchx 61
#define patchy 61
//		Mat dst(src[1].size(),CV_8UC1,Scalar(0,0,0)); //One way of initialization
//		Mat dst = Mat::zeros(src[1].size(),CV_8UC1); //Another way of initialization
		Mat dst(src[1].size(),CV_8UC1);
		dst.setTo(Scalar(0,0,0)); //Another way of zero'ing a matrix

 		for(i = 0; i<2; ++i){ 
        	cvtColor( src[i], hsv[i], CV_BGR2HSV );
    		calcHist(&hsv[i],1,ch,noArray(),hist[i],2,hist_size,ranges,true);
    		normalize( hist[i], hist[i], 0, 255, NORM_MINMAX);
			// Create an image to use to visualize our histogram.
  	        //
    		hist_img[i] = Mat::zeros(Size(h_bins * scale, s_bins * scale),CV_8UC3);
    		//Draw our histogram
    		for( int h = 0; h < hist_size[0]; h++ )
    			for( int s = 0; s < hist_size[1]; s++ )
    			{
    				float hval = hist[i].at<float>(h, s);
    				rectangle(hist_img[i], Rect(h*scale, s*scale, scale, scale),
    						Scalar::all(hval), -1);
    			}
		}//For the 2 images

		//DO THE BACK PROJECTION
 		calcBackProject(&(hsv[1]),1,ch,hist[0],dst,ranges,scale);

        //DISPLAY
		namedWindow( "Model Image", 0 );
        imshow(   "Model Image", src[0] );
        namedWindow( "Model H-S Histogram", 0 );
        imshow(   "Model H-S Histogram", hist_img[0] );

		namedWindow( "Test Image", 0 );
        imshow(   "Test Image", src[1] );
        namedWindow( "Test H-S Histogram", 0 );
        imshow(   "Test H-S Histogram", hist_img[1] );

		namedWindow( "Back Projection",0);
		imshow(   "Back Projection", dst );
        waitKey(0);
    }
	else { cerr << "Error: Wrong number of arguments\n" << endl; help(); return -1;}
}
