// Uses Calibration example from ch 11 as input to Example 12-2 at the bottom
//   Example 12-2.  Computing the fundamental matrix using RANSAC.

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warrenty, support or any guarentee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   OTHER OPENCV SITES:
   * The source code is on sourceforge at:
     http://sourceforge.net/projects/opencvlibrary/
   * The OpenCV wiki page (As of Oct 1, 2008 this is down for changing over servers, but should come back):
     http://opencvlibrary.sourceforge.net/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
   ************************************************** */

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void help(){
    cout << "\n\n"
    " Calling convention:\n"
    " ch11_ex11_1 board_w  board_h  number_of_boards [delay [image_scale_factor]]\n"
    "\n"
    "   WHERE:\n"
    "     board_w, board_h   -- are the number of corners along the row and columns respectively\n"
    "     number_of_boards   -- are the number of chessboard views to collect before calibration\n"
    "     delay (=1)         -- minimal delay in seconds between captured boards.\n"
    "                           This allows you time to move the chessboard.  \n"
    "                           Move it to many different locations and angles so that calibration \n"
    "                           space will be well covered. \n"
    "     image_scale_factor (=1)  -- scale captured images by the specified factor before detecting corners.\n"       
    "\n";
}
//
//

int main(int argc, char* argv[]) {
    
    int n_boards = 0; //Will be set by input list
    float image_sf = 0.5f;
    float delay=1.f;
    int board_w=0;
    int board_h=0;
    
    if(argc < 4 || argc > 6){
        cout << "\nERROR: Wrong number of input parameters";
        help();
        return -1;
    }
    board_w  = atoi(argv[1]);
    board_h  = atoi(argv[2]);
    n_boards = atoi(argv[3]);
    if( argc > 4 )
        delay = atof(argv[4]);
    if( argc > 5 )
        image_sf = atof(argv[5]);
    
    int board_n  = board_w * board_h;
    Size board_sz = Size( board_w, board_h );
    VideoCapture capture(0);
    if(!capture.isOpened()) { cout << "\nCouldn't open the camera\n"; help(); return -1;}
    
    //ALLOCATE STORAGE
    vector<vector<Point2f> > image_points;
    vector<vector<Point3f> > object_points;
    
    // CAPTURE CORNER VIEWS LOOP UNTIL WE’VE GOT n_boards 
    // SUCCESSFUL CAPTURES (ALL CORNERS ON THE BOARD ARE FOUND)
    //
    double last_captured_timestamp = 0;
    Size image_size;
    while(image_points.size() < (size_t)n_boards) {
        Mat image0, image;
        capture >> image0;
        image_size = image0.size();
        resize(image0, image, Size(), image_sf, image_sf, INTER_LINEAR);
        
        //Find the board
        vector<Point2f> corners;
        bool found = findChessboardCorners(image, board_sz, corners);
        
        //Draw it
        drawChessboardCorners(image, board_sz, corners, found);
        
        // If we got a good board, add it to our data
        double timestamp = (double)clock()/CLOCKS_PER_SEC;
        if( found && timestamp - last_captured_timestamp > 1 ) {
            last_captured_timestamp = timestamp;
            image ^= Scalar::all(255);
            
            Mat mcorners(corners); // do not copy the data
            mcorners *= (1./image_sf); // scale the corner coordinates
            image_points.push_back(corners);
            object_points.push_back(vector<Point3f>());
            vector<Point3f>& opts = object_points.back();
            opts.resize(board_n);
            for( int j=0; j<board_n; j++ ) {
                opts[j] = Point3f((float)(j/board_w), (float)(j%board_w), 0.f);
            }
            cout << "Collected our " <<  (int)image_points.size() <<
            " of " << n_boards << " needed chessboard images\n" << endl;
        }
        imshow( "Calibration", image ); //show in color if we did collect the image
        if((waitKey(30) & 255) == 27)
            return -1;
    } //END COLLECTION WHILE LOOP.
    destroyWindow("Calibration");
    cout << "\n\n*** CALLIBRATING THE CAMERA...\n" << endl;
    
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
    
    // Compute Fundamental Matrix Between the first and the second frames:
    undistortPoints(image_points[0], image_points[0], intrinsic_matrix,
                    distortion_coeffs, Mat(), intrinsic_matrix);
    undistortPoints(image_points[1], image_points[1], intrinsic_matrix,
                    distortion_coeffs, Mat(), intrinsic_matrix);
    // Since all the found chessboard corners are inliers, i.e. they must satisfy
    // epipolar constraints, here we are using the fastest,
    // and the most accurate (in this case) 8-point algorithm.
    Mat F = findFundamentalMat(image_points[0], image_points[1], FM_8POINT);
    cout << "Fundamental matrix: " << F << endl;
    
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
    for(;;)
    {
        Mat image, image0;
        capture >> image0;
        if( image0.empty() )
            break;
        remap(image0, image, map1, map2, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        imshow("Undistorted", image);
        if((waitKey(30) & 255) == 27)
            break;
    }
    
    return 0;
}

