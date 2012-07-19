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

void help(){
cout << "\nCall is:\n" <<
		"./ch7_ex7_3_expanded modelImage0 testImage1 testImage2 badImage3\n\n" <<
		"for example: ./ch7_ex7_3_expanded HandIndoorColor.jpg HandOutdoorColor.jpg HandOutdoorSunColor.jpg fruits.jpg\n\n" <<
"  Note that the model image is split in half.  Top half(0) makes model.  It's then tested\n" <<
"  against its lower half(0), testImages 1 and 2 in different lighting and different object 3\n\n" << endl;
}
//Compare 3 images histograms together, 
// the first is divided in half along y to test its other half
// Call is: 
//    ./ch7_ex7_3_expanded modelImage0 testImage1 testImage2 badImage3
// Note that the model image is split in half.  Top half(0) makes model.  It's then tested
// against its lower half(0), testImages 1 and 2 in different lighting and different object 3
// 
int main( int argc, char** argv ) {

	vector<Mat> src(5);
	Mat tmp;
	int i;
	if(argc != 5) { help(); return -1;}

	tmp = imread(argv[1], 1);
	if(tmp.empty()){ //We're going to split this one in half
		cerr << "Error on reading image 1," << argv[1] << "\n" << endl;
		help();
		return(-1);
	}

	//Parse the first image into two image halves divided halfway on y
	cout << "Getting size [[" << tmp.cols << "] [" << tmp.rows << "]]\n" << endl;
	Size size = tmp.size();
	cout << "Got size (w,h): (" << size.width << "," << size.height << ")" << endl;
	int width = size.width;
	int height = size.height;
	int halfheight = height >> 1;
	src[0] = Mat(Size(width,halfheight), CV_8UC3);
	src[1] = Mat(Size(width,halfheight), CV_8UC3);


	//Divide the first image into top and bottom halfs in src[0] and src[1] respectively
	Mat_<Vec3b>::iterator tmpit = tmp.begin<Vec3b>();
	Mat_<Vec3b>::iterator s0it = src[0].begin<Vec3b>();

	for(i = 0; i < width*halfheight; ++i, ++tmpit, ++s0it) //top half
		*s0it = *tmpit;
	Mat_<Vec3b>::iterator s1it = src[1].begin<Vec3b>();
	for(i = 0; i < width*halfheight; ++i, ++tmpit, ++s1it) //Bottom half
		*s1it = *tmpit;

	//LOAD THE OTHER THREE IMAGES
	for(i = 2; i<5; ++i){
		src[i] = imread(argv[i], 1);
		if(src[i].empty())
		{
			cerr << "Error on reading image " << i << ": " << argv[i] << "\n" << endl;
			help();
			return(-1);
		}
	}


	// Compute the HSV image, and decompose it into separate planes.
	//



	vector<Mat> hsv(5),hist(5),hist_img(5);
	int h_bins = 8, s_bins = 8;
	int    hist_size[] = { h_bins, s_bins }, ch[] = {0, 1};
	float  h_ranges[]  = { 0, 180 };          // hue is [0,180]
	float  s_ranges[]  = { 0, 255 };
	const float* ranges[]    = { h_ranges, s_ranges };
	int scale = 10;

	for(i = 0; i<5; ++i){
		cvtColor( src[i], hsv[i], CV_BGR2HSV );
		calcHist(&hsv[i],1,ch,noArray(),hist[i],2,hist_size,ranges,true);
		normalize( hist[i], hist[i], 0, 255, NORM_MINMAX);
		hist_img[i] = Mat::zeros(hist_size[0]*scale,hist_size[1]*scale,CV_8UC3);
		//Draw our histogram
		for( int h = 0; h < hist_size[0]; h++ )
			for( int s = 0; s < hist_size[1]; s++ )
			{
				float hval = hist[i].at<float>(h, s);
				rectangle(hist_img[i], Rect(h*scale, s*scale, scale, scale),
						Scalar::all(hval), -1);
			}
	}//For the 5 images

	//DISPLAY
	namedWindow( "Source0", 1 );
	imshow(   "Source0", src[0] );
	namedWindow( "H-S Histogram0", 1 );
	imshow(   "H-S Histogram0", hist_img[0] );

	namedWindow( "Source1", 1 );
	imshow(   "Source1", src[1] );
	namedWindow( "H-S Histogram1", 1 );
	imshow(   "H-S Histogram1", hist_img[1] );

	namedWindow( "Source2", 1 );
	imshow(   "Source2", src[2] );
	namedWindow( "H-S Histogram2", 1 );
	imshow(   "H-S Histogram2", hist_img[2] );

	namedWindow( "Source3", 1 );
	imshow(   "Source3", src[3] );
	namedWindow( "H-S Histogram3", 1 );
	imshow(   "H-S Histogram3", hist_img[3] );

	namedWindow( "Source4", 1 );
	imshow(   "Source4", src[4] );
	namedWindow( "H-S Histogram4", 1 );
	imshow(   "H-S Histogram4", hist_img[4] );

	//Compare the histogram src0 vs 1, vs 2, vs 3, vs 4
	cout << "Comparison:\nCorr                 Chi                 Intersect          Bhat\n" << endl;
	for(i=1; i<5; ++i){//For histogram
		cout << "Hist[0] vs Hist["<<i<<"]: " << endl;;
		for(int j=0; j<4; ++j) { //For comparision type
			cout << "method["<<j<<"]: " << compareHist(hist[0],hist[i],j) << "  ";
		}
		cout << endl;
	}
    
    //Do EMD AND REPORT
    vector<Mat> sig(5);
    cout << "\nEMD: " << endl;
    for( i=0; i<5; ++i) {
        //Oi Vey, parse histogram to earth movers signatures
        vector<Vec3f> sigv;
        // renormalize histogram to make the bin weights sum to 1.
        normalize(hist[i], hist[i], 1, 0, NORM_L1);
        for( int h = 0; h < h_bins; h++ )
            for( int s = 0; s < s_bins; s++ ) {
                float bin_val = hist[i].at<float>(h, s);
                if( bin_val != 0 )
                    sigv.push_back(Vec3f(bin_val, (float)h, (float)s));
            }
        // make Nx3 32fC1 matrix, where N is the number of non-zero histogram bins
        sig[i] = Mat(sigv).clone().reshape(1);
        if( i > 0 )
            cout << "Hist[0] vs Hist[" << i << "]: " <<
                EMD(sig[0], sig[i], CV_DIST_L2) << endl;
    }

	waitKey(0);
}
