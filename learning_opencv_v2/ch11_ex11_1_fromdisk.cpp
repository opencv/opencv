// Calibrate from a list of images
//
// console application.
// Gary Bradski April'08, converted to C++ by Vadim Pisarevsky June'11.
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
#include <stdio.h>

using namespace cv;
using namespace std;

void help(){
	printf("Calibration from disk. Call convention:\n\n"
			"  ch11_ex11_1_fromdisk board_w board_h image_list\n\n"
			"Where: board_{w,h} are the # of internal corners in the checkerboard\n"
			"       width (board_w) and height (board_h)\n"
			"       image_list is space separated list of path/filename of checkerboard\n"
			"       images\n\n"
			"Hit 'p' to pause/unpause, ESC to quit.  After calibration, press any other key to step through the images\n\n");
}

int main(int argc, char* argv[]) {
    
    int board_w=0;
    int board_h=0;
    
    if(argc != 4){
        cout << "\nERROR: Wrong number of input parameters";
        help();
        return -1;
    }
    board_w  = atoi(argv[1]);
    board_h  = atoi(argv[2]);
    
    int board_n  = board_w * board_h;
    Size board_sz = Size( board_w, board_h );
    
    FILE* f = fopen(argv[3], "rt");
    if(!f) { cout << "\nCouldn't read the file list " << argv[3] << endl; help(); return -1;}
    
    vector<string> filelist;
    for(;;)
    {
        char buf[1000];
        if(!fgets(buf, (int)sizeof(buf)-2, f))
            break;
        if(buf[0] == '#' || buf[0] == '\n') continue;
        int l = (int)strlen(buf);
        if(buf[l-1] == '\n') buf[l-1]='\0';
        filelist.push_back(buf);
    }
    
    //ALLOCATE STORAGE
    vector<vector<Point2f> > image_points;
    vector<vector<Point3f> > object_points;
    
    // CAPTURE CORNER VIEWS LOOP UNTIL WE’VE GOT n_boards 
    // SUCCESSFUL CAPTURES (ALL CORNERS ON THE BOARD ARE FOUND)
    //
    Size image_size;
    for(size_t i = 0; i < filelist.size(); i++) {
        Mat image = imread(filelist[i]);
        if(image.empty())
            continue;
        if(image_size == Size())
            image_size = image.size();
        else if(image_size != image.size())
        {
            cout << "\nImages " << filelist[0] << " and " << filelist[i] << " have different sizes\n";
            return -1;
        }
        
        //Find the board
        vector<Point2f> corners;
        bool found = findChessboardCorners(image, board_sz, corners);
        
        //Draw it
        drawChessboardCorners(image, board_sz, corners, found);
        
        // If we got a good board, add it to our data
        if( found ) {
            image_points.push_back(corners);
            object_points.push_back(vector<Point3f>());
            vector<Point3f>& opts = object_points.back();
            opts.resize(board_n);
            for( int j=0; j<board_n; j++ ) {
                opts[j] = Point3f((float)(j/board_w), (float)(j%board_w), 0.f);
            }
        }
        imshow( "Preparing for Calibration", image ); //show in color if we did collect the image
        if((waitKey(500) & 255) == 27)
            return -1;
    } //END COLLECTION WHILE LOOP.
    destroyWindow("Preparing for Calibration");
    printf("\n\n*** CALLIBRATING THE CAMERA...");
    
    //CALIBRATE THE CAMERA!
    Mat intrinsic_matrix, distortion_coeffs;
    double err = calibrateCamera(object_points, image_points, image_size, intrinsic_matrix,
                                 distortion_coeffs, noArray(), noArray(),
                                 CALIB_ZERO_TANGENT_DIST+CALIB_FIX_PRINCIPAL_POINT);
    
    // SAVE THE INTRINSICS AND DISTORTIONS
    cout << " *** DONE!\n\nReprojection error is " << err <<
    "\nStoring Intrinsics.xml and Distortions.xml files\n\n";
    FileStorage fs("intrinsics.xml", FileStorage::WRITE);
    
    fs << "image_width" << image_size.width << "image_height" << image_size.height <<
    "camera_matrix" << intrinsic_matrix << "distortion_coefficients" << distortion_coeffs;
    fs.release();
    
    // EXAMPLE OF LOADING THESE MATRICES BACK IN:
    fs.open("intrinsics.xml", FileStorage::READ);
    cout << "\nimage width: " << (int)fs["image_width"];
    cout << "\nimage height: " << (int)fs["image_height"];
    Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
    cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
    cout << "\ndistortion coefficients: " << distortion_coeffs_loaded << endl;
    
    // Build the undistort map which we will use for all 
    // subsequent frames.
    //
    Mat map1, map2;
    initUndistortRectifyMap(intrinsic_matrix_loaded, distortion_coeffs_loaded, Mat(),
                            intrinsic_matrix_loaded, image_size, CV_16SC2,
                            map1, map2);
    // Just run the camera to the screen, now showing the raw and
    // the undistorted image.
    //
    for(size_t i = 0; i < filelist.size(); i++) {
        Mat image = imread(filelist[i]), dst;
        if(image.empty())
            continue;
        remap(image, dst, map1, map2, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        imshow("Original", image);
        imshow("Undistorted", dst);
        if((waitKey(500) & 255) == 27)
            break;
    }
    
    return 0;
}
