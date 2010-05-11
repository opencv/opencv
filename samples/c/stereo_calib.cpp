/* This is sample from the OpenCV book. The copyright notice is below */

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

#undef _GLIBCXX_DEBUG

#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>

using namespace std;

//
// Given a list of chessboard images, the number of corners (nx, ny)
// on the chessboards, and a flag: useCalibrated for calibrated (0) or
// uncalibrated (1: use cvStereoCalibrate(), 2: compute fundamental
// matrix separately) stereo. Calibrate the cameras and display the
// rectified results along with the computed disparity images.
//
static void
StereoCalib(const char* path, const char* imageList, int useUncalibrated)
{
    CvRect roi1, roi2;
    int nx = 0, ny = 0;
    int displayCorners = 1;
    int showUndistorted = 1;
    bool isVerticalStereo = false;//OpenCV can handle left-right
                                      //or up-down camera arrangements
    const int maxScale = 1;
    const float squareSize = 1.f; //Set this to your actual square size
    FILE* f = fopen(imageList, "rt");
    int i, j, lr, nframes = 0, n, N = 0;
    vector<string> imageNames[2];
    vector<CvPoint3D32f> objectPoints;
    vector<CvPoint2D32f> points[2];
    vector<CvPoint2D32f> temp_points[2];
    vector<int> npoints;
//    vector<uchar> active[2];
    int is_found[2] = {0, 0};
    vector<CvPoint2D32f> temp;
    CvSize imageSize = {0,0};
    // ARRAY AND VECTOR STORAGE:
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    double Q[4][4];
    CvMat _M1 = cvMat(3, 3, CV_64F, M1 );
    CvMat _M2 = cvMat(3, 3, CV_64F, M2 );
    CvMat _D1 = cvMat(1, 5, CV_64F, D1 );
    CvMat _D2 = cvMat(1, 5, CV_64F, D2 );
    CvMat matR = cvMat(3, 3, CV_64F, R );
    CvMat matT = cvMat(3, 1, CV_64F, T );
    CvMat matE = cvMat(3, 3, CV_64F, E );
    CvMat matF = cvMat(3, 3, CV_64F, F );
    
    CvMat matQ = cvMat(4, 4, CV_64FC1, Q);

    char buf[1024];
    
    if( displayCorners )
        cvNamedWindow( "corners", 1 );
// READ IN THE LIST OF CHESSBOARDS:
    if( !f )
    {
        fprintf(stderr, "can not open file %s\n", imageList );
        return;
    }
    
    if( !fgets(buf, sizeof(buf)-3, f) || sscanf(buf, "%d%d", &nx, &ny) != 2 )
        return;
    n = nx*ny;
    temp.resize(n);
    temp_points[0].resize(n);
    temp_points[1].resize(n);
    
    for(i=0;;i++)
    {
        int count = 0, result=0;
        lr = i % 2;
        vector<CvPoint2D32f>& pts = temp_points[lr];//points[lr];
        if( !fgets( buf, sizeof(buf)-3, f ))
            break;
        size_t len = strlen(buf);
        while( len > 0 && isspace(buf[len-1]))
            buf[--len] = '\0';
        if( buf[0] == '#')
            continue;
	char fullpath[1024];
	sprintf(fullpath, "%s/%s", path, buf);
        IplImage* img = cvLoadImage( fullpath, 0 );
        if( !img )
        {
            printf("Cannot read file %s\n", fullpath);
            return;
        }
        imageSize = cvGetSize(img);
        imageNames[lr].push_back(buf);
    //FIND CHESSBOARDS AND CORNERS THEREIN:
        for( int s = 1; s <= maxScale; s++ )
        {
            IplImage* timg = img;
            if( s > 1 )
            {
                timg = cvCreateImage(cvSize(img->width*s,img->height*s),
                    img->depth, img->nChannels );
                cvResize( img, timg, CV_INTER_CUBIC );
            }
            result = cvFindChessboardCorners( timg, cvSize(nx, ny),
                &temp[0], &count,
                CV_CALIB_CB_ADAPTIVE_THRESH |
                CV_CALIB_CB_NORMALIZE_IMAGE);
            if( timg != img )
                cvReleaseImage( &timg );
            if( result || s == maxScale )
                for( j = 0; j < count; j++ )
            {
                temp[j].x /= s;
                temp[j].y /= s;
            }
            if( result )
                break;
        }
        if( displayCorners )
        {
            printf("%s\n", buf);
            IplImage* cimg = cvCreateImage( imageSize, 8, 3 );
            cvCvtColor( img, cimg, CV_GRAY2BGR );
            cvDrawChessboardCorners( cimg, cvSize(nx, ny), &temp[0],
                count, result );
            IplImage* cimg1 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
            cvResize(cimg, cimg1);
            cvShowImage( "corners", cimg1 );
            cvReleaseImage( &cimg );
            cvReleaseImage( &cimg1 );
            int c = cvWaitKey(1000);
            if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                exit(-1);
        }
        else
            putchar('.');
        //N = pts.size();
        //pts.resize(N + n, cvPoint2D32f(0,0));
        //active[lr].push_back((uchar)result);
        is_found[lr] = result > 0 ? 1 : 0;
    //assert( result != 0 );
        if( result )
        {
         //Calibration will suffer without subpixel interpolation
            cvFindCornerSubPix( img, &temp[0], count,
                cvSize(11, 11), cvSize(-1,-1),
                cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                30, 0.01) );
            copy( temp.begin(), temp.end(), pts.begin() );
        }
        cvReleaseImage( &img );
        
        if(lr)
        {
            if(is_found[0] == 1 && is_found[1] == 1)
            {
                assert(temp_points[0].size() == temp_points[1].size());
                int current_size = points[0].size();
                
                points[0].resize(current_size + temp_points[0].size(), cvPoint2D32f(0.0, 0.0));
                points[1].resize(current_size + temp_points[1].size(), cvPoint2D32f(0.0, 0.0));
                
                copy(temp_points[0].begin(), temp_points[0].end(), points[0].begin() + current_size);
                copy(temp_points[1].begin(), temp_points[1].end(), points[1].begin() + current_size);
                
                nframes++;

                printf("Pair successfully detected...\n");
            }
            
            is_found[0] = 0;
            is_found[1] = 0;
            
        }
    }
    fclose(f);
    printf("\n");
// HARVEST CHESSBOARD 3D OBJECT POINT LIST:
    objectPoints.resize(nframes*n);
    for( i = 0; i < ny; i++ )
        for( j = 0; j < nx; j++ )
        objectPoints[i*nx + j] =
        cvPoint3D32f(i*squareSize, j*squareSize, 0);
    for( i = 1; i < nframes; i++ )
        copy( objectPoints.begin(), objectPoints.begin() + n,
        objectPoints.begin() + i*n );
    npoints.resize(nframes,n);
    N = nframes*n;
    CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
    CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
    cvSetIdentity(&_M1);
    cvSetIdentity(&_M2);
    cvZero(&_D1);
    cvZero(&_D2);
    
// CALIBRATE THE STEREO CAMERAS
    printf("Running stereo calibration ...");
    fflush(stdout);
    cvStereoCalibrate( &_objectPoints, &_imagePoints1,
        &_imagePoints2, &_npoints,
        &_M1, &_D1, &_M2, &_D2,
        imageSize, &matR, &matT, &matE, &matF,
        cvTermCriteria(CV_TERMCRIT_ITER+
        CV_TERMCRIT_EPS, 100, 1e-5),
        CV_CALIB_FIX_ASPECT_RATIO +
        CV_CALIB_ZERO_TANGENT_DIST +
        CV_CALIB_SAME_FOCAL_LENGTH +
        CV_CALIB_FIX_K3);
    printf(" done\n");
    
// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    vector<CvPoint3D32f> lines[2];
    points[0].resize(N);
    points[1].resize(N);
    _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    lines[0].resize(N);
    lines[1].resize(N);
    CvMat _L1 = cvMat(1, N, CV_32FC3, &lines[0][0]);
    CvMat _L2 = cvMat(1, N, CV_32FC3, &lines[1][0]);
//Always work in undistorted space
    cvUndistortPoints( &_imagePoints1, &_imagePoints1,
        &_M1, &_D1, 0, &_M1 );
    cvUndistortPoints( &_imagePoints2, &_imagePoints2,
        &_M2, &_D2, 0, &_M2 );
    cvComputeCorrespondEpilines( &_imagePoints1, 1, &matF, &_L1 );
    cvComputeCorrespondEpilines( &_imagePoints2, 2, &matF, &_L2 );
    double avgErr = 0;
    for( i = 0; i < N; i++ )
    {
        double err = fabs(points[0][i].x*lines[1][i].x +
            points[0][i].y*lines[1][i].y + lines[1][i].z)
            + fabs(points[1][i].x*lines[0][i].x +
            points[1][i].y*lines[0][i].y + lines[0][i].z);
        avgErr += err;
    }
    printf( "avg err = %g\n", avgErr/(nframes*n) );
    
    // save intrinsic parameters
    CvFileStorage* fstorage = cvOpenFileStorage("intrinsics.yml", NULL, CV_STORAGE_WRITE);
    cvWrite(fstorage, "M1", &_M1);
    cvWrite(fstorage, "D1", &_D1);
    cvWrite(fstorage, "M2", &_M2);
    cvWrite(fstorage, "D2", &_D2);
    cvReleaseFileStorage(&fstorage);
    
//COMPUTE AND DISPLAY RECTIFICATION
    if( showUndistorted )
    {
        CvMat* mx1 = cvCreateMat( imageSize.height,
            imageSize.width, CV_32F );
        CvMat* my1 = cvCreateMat( imageSize.height,
            imageSize.width, CV_32F );
        CvMat* mx2 = cvCreateMat( imageSize.height,
            imageSize.width, CV_32F );
        CvMat* my2 = cvCreateMat( imageSize.height,
            imageSize.width, CV_32F );
        CvMat* img1r = cvCreateMat( imageSize.height,
            imageSize.width, CV_8U );
        CvMat* img2r = cvCreateMat( imageSize.height,
            imageSize.width, CV_8U );
        CvMat* disp = cvCreateMat( imageSize.height,
            imageSize.width, CV_16S );
        double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
        CvMat _R1 = cvMat(3, 3, CV_64F, R1);
        CvMat _R2 = cvMat(3, 3, CV_64F, R2);
// IF BY CALIBRATED (BOUGUET'S METHOD)
        if( useUncalibrated == 0 )
        {
            CvMat _P1 = cvMat(3, 4, CV_64F, P1);
            CvMat _P2 = cvMat(3, 4, CV_64F, P2);

            cvStereoRectify( &_M1, &_M2, &_D1, &_D2, imageSize,
                &matR, &matT,
                &_R1, &_R2, &_P1, &_P2, &matQ,
                CV_CALIB_ZERO_DISPARITY,
                1, imageSize, &roi1, &roi2);
            
            CvFileStorage* file = cvOpenFileStorage("extrinsics.yml", NULL, CV_STORAGE_WRITE);
            cvWrite(file, "R", &matR);
            cvWrite(file, "T", &matT);    
            cvWrite(file, "R1", &_R1);
            cvWrite(file, "R2", &_R2);
            cvWrite(file, "P1", &_P1);    
            cvWrite(file, "P2", &_P2);    
            cvWrite(file, "Q", &matQ);
            cvReleaseFileStorage(&file);
                        
            isVerticalStereo = fabs(P2[1][3]) > fabs(P2[0][3]);
            if(!isVerticalStereo)
                roi2.x += imageSize.width;
            else
                roi2.y += imageSize.height;
    //Precompute maps for cvRemap()
            cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_P1,mx1,my1);
            cvInitUndistortRectifyMap(&_M2,&_D2,&_R2,&_P2,mx2,my2);
        }
//OR ELSE HARTLEY'S METHOD
        else if( useUncalibrated == 1 || useUncalibrated == 2 )
     // use intrinsic parameters of each camera, but
     // compute the rectification transformation directly
     // from the fundamental matrix
        {
            double H1[3][3], H2[3][3], iM[3][3];
            CvMat _H1 = cvMat(3, 3, CV_64F, H1);
            CvMat _H2 = cvMat(3, 3, CV_64F, H2);
            CvMat _iM = cvMat(3, 3, CV_64F, iM);
    //Just to show you could have independently used F
            if( useUncalibrated == 2 )
                cvFindFundamentalMat( &_imagePoints1,
                &_imagePoints2, &matF);
            cvStereoRectifyUncalibrated( &_imagePoints1,
                &_imagePoints2, &matF,
                imageSize,
                &_H1, &_H2, 3);
            cvInvert(&_M1, &_iM);
            cvMatMul(&_H1, &_M1, &_R1);
            cvMatMul(&_iM, &_R1, &_R1);
            cvInvert(&_M2, &_iM);
            cvMatMul(&_H2, &_M2, &_R2);
            cvMatMul(&_iM, &_R2, &_R2);
    //Precompute map for cvRemap()
            cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_M1,mx1,my1);

            cvInitUndistortRectifyMap(&_M2,&_D1,&_R2,&_M2,mx2,my2);
        }
        else
            assert(0);
        
        
        cvReleaseMat( &mx1 );
        cvReleaseMat( &my1 );
        cvReleaseMat( &mx2 );
        cvReleaseMat( &my2 );
        cvReleaseMat( &img1r );
        cvReleaseMat( &img2r );
        cvReleaseMat( &disp );
    }
}

int main(int argc, char** argv)
{
    if(argc > 1 && !strcmp(argv[1], "--help"))
    {
        printf("Usage:\n ./stereo_calib <path to images> <file wtih image list>\n");
        return 0;
    }	

    StereoCalib(argc > 1 ? argv[1] : ".", argc > 2 ? argv[2] : "stereo_calib.txt", 0);
    return 0;
}

