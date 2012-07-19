#pragma warning( disable: 4996 )
//
// Given a list of chessboard images, the number of corners (nx, ny)
// on the chessboards, and a flag: useUncalibrated for calibrated (0) or
// uncalibrated (1: use stereoCalibrate(), 2: compute fundamental
// matrix separately) stereo. Calibrate the cameras and display the
// rectified results along with the computed disparity images.
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
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

void help()
{
	cout << "Demonstrate the use of stereoCalibrate\n" <<
			"Usage: ./ch12_ex12_3\n" <<
			"  This is equivalent to ./ch12_ex12_3 ch12_list.txt 9 6\n\n" <<
			"  The list is a path/file list of stereo images and 9 is the width of the pattern and 6 is the height\n" <<
			"  You can put in your own list of calibration images with different width and height\n" << endl;
}
static void
StereoCalib(const char* imageList, int nx, int ny, bool useUncalibrated)
{
    bool displayCorners = false;
    bool showUndistorted = true;
    bool isVerticalStereo = false; // OpenCV can handle left-right
                                   // or up-down camera arrangements
    const int maxScale = 1;
    const float squareSize = 1.f; //Set this to your actual square size
    FILE* f = fopen(imageList, "rt");
    int i, j, lr, N = nx*ny;
    vector<string> imageNames[2];
    vector<Point3f> boardModel;
    vector<vector<Point3f> > objectPoints;
    vector<vector<Point2f> > points[2];
    vector<Point2f> corners[2];
    bool found[2]={false, false};
    Size imageSize;

    // READ IN THE LIST OF CHESSBOARDS:
    if( !f )
    {
        cout << "Can not open file " << imageList << endl;
        return;
    }
    
    for( i = 0; i < ny; i++ )
        for( j = 0; j < nx; j++ )
            boardModel.push_back(Point3f((float)(i*squareSize), (float)(j*squareSize), 0.f)); 
    
    i = 0;
    for(;;)
    {
        char buf[1024];
        lr = i % 2;
        if( lr == 0 )
            found[0] = found[1] = false;
        
        if( !fgets( buf, sizeof(buf)-3, f ))
            break;
        size_t len = strlen(buf);
        while( len > 0 && isspace(buf[len-1]))
            buf[--len] = '\0';
        if( buf[0] == '#')
            continue;
        Mat img = imread( buf, 0 );
        if( img.empty() )
            break;
        imageSize = img.size();
        imageNames[lr].push_back(buf);
        
        i++;
        // if we did not find board on the left image,
        // it does not make sense to find it on the right
        if( lr == 1 && !found[0] )
            continue;
        
    //FIND CHESSBOARDS AND CORNERS THEREIN:
        for( int s = 1; s <= maxScale; s++ )
        {
            Mat timg = img;
            if( s > 1 )
                resize(img, timg, Size(), s, s, INTER_CUBIC);
            found[lr] = findChessboardCorners( timg, Size(nx, ny),
                corners[lr], CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
            if( found[lr] || s == maxScale )
            {
                Mat mcorners(corners[lr]);
                mcorners *= (1./s);
            }
            if( found[lr] )
                break;
        }
        if( displayCorners )
        {
            cout << buf << endl;
            Mat cimg;
            cvtColor( img, cimg, CV_GRAY2BGR );
            drawChessboardCorners( cimg, Size(nx, ny), corners[lr], found[lr] );
            imshow( "Corners", cimg );
            if( (waitKey(0)&255) == 27 ) //Allow ESC to quit
                exit(-1);
        }
        else
            cout << '.';

        if( found[lr] )
        {
            // Calibration will suffer without subpixel interpolation
            cornerSubPix( img, corners[lr],
                Size(11, 11), Size(-1,-1),
                TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                30, 0.01) );
        }
        
        if( lr == 1 && found[0] && found[1] )
        {
            objectPoints.push_back(boardModel);
            points[0].push_back(corners[0]);
            points[1].push_back(corners[1]);
        }
    }
    fclose(f);
    
    // CALIBRATE THE STEREO CAMERAS
    Mat M1 = Mat::eye(3, 3, CV_64F), M2 = Mat::eye(3, 3, CV_64F);
    Mat D1, D2, R, T, E, F;
    cout << "\nRunning stereo calibration ...\n";
    stereoCalibrate(objectPoints, points[0], points[1],
                    M1, D1, M2, D2,
                    imageSize, R, T, E, F,
                    TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_SAME_FOCAL_LENGTH );
    cout << "Done\n\n";
    
// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    vector<Point3f> lines[2];
    
    double avgErr = 0;
    int nframes = (int)objectPoints.size();
    
    for( i = 0; i < nframes; i++ )
    {
        vector<Point2f>& pt0 = points[0][i];
        vector<Point2f>& pt1 = points[1][i];
        
        undistortPoints(pt0, pt0, M1, D1, Mat(), M1);
        undistortPoints(pt1, pt1, M2, D2, Mat(), M2);
        computeCorrespondEpilines(pt0, 1, F, lines[0]);
        computeCorrespondEpilines(pt1, 2, F, lines[1]);
        
        for( j = 0; j < N; j++ )
        {
            double err = fabs(pt0[j].x*lines[1][j].x +
                pt0[j].y*lines[1][j].y + lines[1][j].z)
                + fabs(pt1[j].x*lines[0][j].x +
                pt1[j].y*lines[0][j].y + lines[0][j].z);
            avgErr += err;
        }
    }
    
    cout << "avg err = " << avgErr/(nframes*N) << endl;
        
//COMPUTE AND DISPLAY RECTIFICATION
    if( showUndistorted )
    {
        Mat R1, R2, P1, P2, map11, map12, map21, map22;
        
// IF BY CALIBRATED (BOUGUET'S METHOD)
        if( !useUncalibrated )
        {
            stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, noArray(), 0);
            isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    //Precompute maps for cvRemap()
            initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, map11, map12);
            initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, map21, map22);
        }
//OR ELSE HARTLEY'S METHOD
        else
     // use intrinsic parameters of each camera, but
     // compute the rectification transformation directly
     // from the fundamental matrix
        {
            vector<Point2f> allpoints[2];
            for( i = 0; i < nframes; i++ )
            {
                copy(points[0][i].begin(), points[0][i].end(), back_inserter(allpoints[0]));
                copy(points[1][i].begin(), points[1][i].end(), back_inserter(allpoints[1]));
            }
            Mat F = findFundamentalMat(allpoints[0], allpoints[1], FM_8POINT);
            Mat H1, H2;
            stereoRectifyUncalibrated(allpoints[0], allpoints[1], F, imageSize, H1, H2, 3);
            
            R1 = M1.inv()*H1*M1;
            R2 = M2.inv()*H2*M2;
    //Precompute map for cvRemap()
            initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, map11, map12);
            initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, map21, map22);
        }

// RECTIFY THE IMAGES AND FIND DISPARITY MAPS
        Mat pair;
        if( !isVerticalStereo )
            pair.create( imageSize.height, imageSize.width*2, CV_8UC3 );
        else
            pair.create( imageSize.height*2, imageSize.width, CV_8UC3 );
//Setup for finding stereo corrrespondences
        /*StereoBM stereo(StereoBM::BASIC_PRESET, 128, 31);
        stereo.state->preFilterSize=41;
        stereo.state->preFilterCap=31;
        stereo.state->minDisparity=-64;
        stereo.state->textureThreshold=10;
        stereo.state->uniquenessRatio=15;*/
        StereoSGBM stereo(-64, 128, 11, 100, 1000, 32, 0, 15, 1000, 16, true);

        for( i = 0; i < nframes; i++ )
        {
            Mat img1 = imread(imageNames[0][i].c_str(),0);
            Mat img2 = imread(imageNames[1][i].c_str(),0);
            Mat img1r, img2r, disp, vdisp;
            
            if( img1.empty() || img2.empty() )
                continue;

            remap( img1, img1r, map11, map12, INTER_LINEAR);
            remap( img2, img2r, map21, map22, INTER_LINEAR);
            if( !isVerticalStereo || !useUncalibrated )
            {
          // When the stereo camera is oriented vertically,
          // Hartley method does not transpose the
          // image, so the epipolar lines in the rectified
          // images are vertical. Stereo correspondence
          // function does not support such a case.
                stereo(img1r, img2r, disp);
                normalize( disp, vdisp, 0, 256, NORM_MINMAX, CV_8U );
                imshow( "disparity", vdisp );
            }
            if( !isVerticalStereo )
            {
                Mat part = pair.colRange(0, imageSize.width);
                cvtColor(img1r, part, CV_GRAY2BGR);
                part = pair.colRange(imageSize.width, imageSize.width*2);
                cvtColor(img2r, part, CV_GRAY2BGR);
                
                for( j = 0; j < imageSize.height; j += 16 )
                    line( pair, Point(0,j), Point(imageSize.width*2,j), Scalar(0,255,0));
            }
            else
            {
                Mat part = pair.rowRange(0, imageSize.height);
                cvtColor(img1r, part, CV_GRAY2BGR);
                part = pair.rowRange(imageSize.height, imageSize.height*2);
                cvtColor(img2r, part, CV_GRAY2BGR);
                
                for( j = 0; j < imageSize.width; j += 16 )
                    line( pair, Point(j,0), Point(j,imageSize.height*2), Scalar(0,255,0));
            }
            imshow( "rectified", pair );
            if( (waitKey()&255) == 27 )
                break;
        }
    }
}

int main(int argc, char** argv)
{
	help();
    int board_w = 9, board_h = 6;
    const char* board_list = "ch12_list.txt";
    if( argc == 4 )
    {
        board_list = argv[1];
        board_w = atoi(argv[2]);
        board_h = atoi(argv[3]);
    }
    StereoCalib(board_list, board_w, board_h, true);
    return 0;
}
