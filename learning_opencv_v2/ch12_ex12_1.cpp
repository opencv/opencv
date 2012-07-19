// Given a view of a checkerboard on a plane, view that image and a 
// list of others frontal parallel to that plane
//
// This presumes that you have previously callibrated your camera and stored
// an instrinics and distortion model for your camera.
//
// console application.
// Gary Bradski Oct 3, 2008
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
#include <iostream>

using namespace cv;
using namespace std;

void help(){
	cout << "Birds eye view\n\n"
			"  birdseye board_w board_h intrinsics.xml checker_image \n\n"
			"Where: board_{w,h}    are the # of internal corners in the checkerboard\n"
			"       intrinsic      file with camera matrix and distortion coefficients\n"
			"       checker_image  is the path/name of image of checkerboard on the plane \n"
			"                      Frontal view of this will be shown.\n\n"
			" ADJUST VIEW HEIGHT using keys 'u' up, 'd' down. ESC to quit.\n\n";
}

int main(int argc, char* argv[]) {
	if(argc != 5){
		cout << "\nERROR: too few parameters\n";
		help();
		return -1;
	}
	help();
	//INPUT PARAMETERS:
	int board_w = atoi(argv[1]);
	int board_h = atoi(argv[2]);
	int board_n  = board_w * board_h;
	Size board_sz( board_w, board_h );
    FileStorage fs(argv[3], FileStorage::READ);
    Mat intrinsic, distortion;
    fs["camera_matrix"] >> intrinsic;
    fs["distortion_coefficients"] >> distortion;
    if( !fs.isOpened() || intrinsic.empty() || distortion.empty() )
    {
        cout << "Error: Couldn't load intrinsic parameters from " << argv[3] << endl;
		return -1;
	}
	fs.release();
    Mat gray_image, image, image0 = imread(argv[4], 1);
    if( image0.empty() )
    {
        cout << "Error: Couldn't load image " << argv[4] << endl;
		return -1;
	}
    
	//UNDISTORT OUR IMAGE
    undistort(image0, image, intrinsic, distortion, intrinsic);
    cvtColor(image, gray_image, CV_BGR2GRAY);

	//GET THE CHECKERBOARD ON THE PLANE
    vector<Point2f> corners;
    bool found = findChessboardCorners(image, board_sz, corners,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
	if(!found) {
		cout << "Couldn't aquire checkerboard on " << argv[4] <<
            ", only found " << corners.size() << " of " << board_n << " corners\n";
		return -1;
	}
	//Get Subpixel accuracy on those corners
	cornerSubPix(gray_image, corners,
			  Size(11,11),Size(-1,-1), 
			  TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

	//GET THE IMAGE AND OBJECT POINTS:
	//Object points are at (r,c): (0,0), (board_w-1,0), (0,board_h-1), (board_w-1,board_h-1)
	//That means corners are at: corners[r*board_w + c]
	Point2f objPts[4], imgPts[4];
	objPts[0].x = 0;         objPts[0].y = 0; 
	objPts[1].x = board_w-1; objPts[1].y = 0; 
	objPts[2].x = 0;         objPts[2].y = board_h-1;
	objPts[3].x = board_w-1; objPts[3].y = board_h-1; 
	imgPts[0] = corners[0];
	imgPts[1] = corners[board_w-1];
	imgPts[2] = corners[(board_h-1)*board_w];
	imgPts[3] = corners[(board_h-1)*board_w + board_w-1];

	//DRAW THE POINTS in order: B,G,R,YELLOW
	circle(image,imgPts[0],9,Scalar(255,0,0),3);
	circle(image,imgPts[1],9,Scalar(0,255,0),3);
	circle(image,imgPts[2],9,Scalar(0,0,255),3);
	circle(image,imgPts[3],9,Scalar(0,255,255),3);

	//DRAW THE FOUND CHECKERBOARD
	drawChessboardCorners(image, board_sz, corners, found);
    imshow( "Checkers", image );

	//FIND THE HOMOGRAPHY
	Mat H = getPerspectiveTransform(objPts, imgPts);

	//LET THE USER ADJUST THE Z HEIGHT OF THE VIEW
	double Z = 25;
	Mat birds_image;
    for(;;) {//escape key stops
	   H.at<double>(2, 2) = Z;
	   //USE HOMOGRAPHY TO REMAP THE VIEW
	   warpPerspective(image, birds_image, H, image.size(), WARP_INVERSE_MAP + INTER_LINEAR,
                       BORDER_CONSTANT, Scalar::all(0));
	   imshow("Birds_Eye", birds_image);
	   int key = waitKey() & 255;
	   if(key == 'u') Z += 0.5;
	   if(key == 'd') Z -= 0.5;       
       if(key == 27) break;
	}

	//SHOW ROTATION AND TRANSLATION VECTORS
	vector<Point2f> image_points;
	vector<Point3f> object_points;
	for(int i=0;i<4;++i){
		image_points.push_back(imgPts[i]);
        object_points.push_back(Point3f(objPts[i].x, objPts[i].y, 0));
	}

	Mat rvec, tvec, rmat;
    solvePnP(object_points, image_points, intrinsic,
             Mat(), // since we corrected distortion in the beginning,
                    // now we have zero distortion coefficients
             rvec, tvec);
    Rodrigues(rvec, rmat);

	// PRINT AND EXIT
    cout << "rotation matrix: " << rmat << endl;
    cout << "translation vector: " << tvec << endl;
    cout << "homography matrix: " << H << endl;
    cout << "inverted homography matrix: " << H.inv() << endl;
	return 0;
}
