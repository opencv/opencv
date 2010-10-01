#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <string.h>
#include <time.h>

using namespace cv;
using namespace std;

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.resize(0);
    
    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(Point3f(float(j*squareSize),
                                      float(i*squareSize), 0));
}


static bool run3Calibration( vector<vector<Point2f> > imagePoints1,
                            vector<vector<Point2f> > imagePoints2,
                            vector<vector<Point2f> > imagePoints3,                            
                            Size imageSize, Size boardSize,
                            float squareSize, float aspectRatio,
                            int flags,
                            Mat& cameraMatrix1, Mat& distCoeffs1,
                            Mat& cameraMatrix2, Mat& distCoeffs2,
                            Mat& cameraMatrix3, Mat& distCoeffs3,
                            Mat& R12, Mat& T12, Mat& R13, Mat& T13)
{
    int c, i;
    
    // step 1: calibrate each camera individually
    vector<vector<Point3f> > objpt(1);
    vector<vector<Point2f> > imgpt;
    calcChessboardCorners(boardSize, squareSize, objpt[0]);
    vector<Mat> rvecs, tvecs;
    
    for( c = 1; c <= 3; c++ )
    {
        const vector<vector<Point2f> >& imgpt0 = c == 1 ? imagePoints1 : c == 2 ? imagePoints2 : imagePoints3;
        imgpt.clear();
        int N = 0;
        for( i = 0; i < (int)imgpt0.size(); i++ )
            if( !imgpt0[i].empty() )
            {
                imgpt.push_back(imgpt0[i]);
                N += (int)imgpt0[i].size();
            }
        
        if( imgpt.size() < 3 )
        {
            printf("Error: not enough views for camera %d\n", c);
            return false;
        }

        objpt.resize(imgpt.size(),objpt[0]);
            
        Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
        if( flags & CV_CALIB_FIX_ASPECT_RATIO )
            cameraMatrix.at<double>(0,0) = aspectRatio;
        
        Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
        
        double err = calibrateCamera(objpt, imgpt, imageSize, cameraMatrix,
                        distCoeffs, rvecs, tvecs,
                        flags|CV_CALIB_FIX_K3/*|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_FIX_K6*/);
        bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
        if(!ok)
        {
            printf("Error: camera %d was not calibrated\n", c);
            return false;
        }
        printf("Camera %d calibration reprojection error = %g\n", c, sqrt(err/N));
        
        if( c == 1 )
            cameraMatrix1 = cameraMatrix, distCoeffs1 = distCoeffs;
        else if( c == 2 )
            cameraMatrix2 = cameraMatrix, distCoeffs2 = distCoeffs;
        else
            cameraMatrix3 = cameraMatrix, distCoeffs3 = distCoeffs;
    }
    
    vector<vector<Point2f> > imgpt_right;
    
    // step 2: calibrate (1,2) and (3,2) pairs
    for( c = 2; c <= 3; c++ )
    {
        const vector<vector<Point2f> >& imgpt0 = c == 2 ? imagePoints2 : imagePoints3;
        
        imgpt.clear();
        imgpt_right.clear();
        int N = 0;
        
        for( i = 0; i < (int)std::min(imagePoints1.size(), imgpt0.size()); i++ )
            if( !imagePoints1.empty() && !imgpt0[i].empty() )
            {
                imgpt.push_back(imagePoints1[i]);
                imgpt_right.push_back(imgpt0[i]);
                N += (int)imgpt0[i].size();
            }
        
        if( imgpt.size() < 3 )
        {
            printf("Error: not enough shared views for cameras 1 and %d\n", c);
            return false;
        }
        
        objpt.resize(imgpt.size(),objpt[0]);
        Mat cameraMatrix = c == 2 ? cameraMatrix2 : cameraMatrix3;
        Mat distCoeffs = c == 2 ? distCoeffs2 : distCoeffs3;
        Mat R, T, E, F;
        double err = stereoCalibrate(objpt, imgpt, imgpt_right, cameraMatrix1, distCoeffs1,
                                     cameraMatrix, distCoeffs,
                                     imageSize, R, T, E, F,
                                     TermCriteria(TermCriteria::COUNT, 30, 0),
                                     CV_CALIB_FIX_INTRINSIC);
        printf("Pair (1,%d) calibration reprojection error = %g\n", c, sqrt(err/(N*2)));
        if( c == 2 )
        {
            cameraMatrix2 = cameraMatrix;
            distCoeffs2 = distCoeffs;
            R12 = R; T12 = T;
        }
        else
        {
            R13 = R; T13 = T;
        }
    }
    
    return true;
}

static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}


int main( int argc, char** argv )
{
    int i, k;
    int flags = 0;
    Size boardSize, imageSize;
    float squareSize = 1.f, aspectRatio = 1.f;
    const char* outputFilename = "out_camera_data.yml";
    const char* inputFilename = 0;
    
    vector<vector<Point2f> > imgpt[3];
    vector<string> imageList;
    
    if( argc < 2 )
    {
        printf( "This is a camera calibration sample.\n"
               "Usage: calibration\n"
               "     -w <board_width>         # the number of inner corners per one of board dimension\n"
               "     -h <board_height>        # the number of inner corners per another board dimension\n"
               "     [-s <squareSize>]       # square size in some user-defined units (1 by default)\n"
               "     [-o <out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
               "     [-zt]                    # assume zero tangential distortion\n"
               "     [-a <aspectRatio>]      # fix aspect ratio (fx/fy)\n"
               "     [-p]                     # fix the principal point at the center\n"
               "     [input_data]             # input data - text file with a list of the images of the board\n"
               "\n" );
        return 0;
    }
    
    for( i = 1; i < argc; i++ )
    {
        const char* s = argv[i];
        if( strcmp( s, "-w" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &boardSize.width ) != 1 || boardSize.width <= 0 )
                return fprintf( stderr, "Invalid board width\n" ), -1;
        }
        else if( strcmp( s, "-h" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &boardSize.height ) != 1 || boardSize.height <= 0 )
                return fprintf( stderr, "Invalid board height\n" ), -1;
        }
        else if( strcmp( s, "-s" ) == 0 )
        {
            if( sscanf( argv[++i], "%f", &squareSize ) != 1 || squareSize <= 0 )
                return fprintf( stderr, "Invalid board square width\n" ), -1;
        }
        else if( strcmp( s, "-a" ) == 0 )
        {
            if( sscanf( argv[++i], "%f", &aspectRatio ) != 1 || aspectRatio <= 0 )
                return printf("Invalid aspect ratio\n" ), -1;
            flags |= CV_CALIB_FIX_ASPECT_RATIO;
        }
        else if( strcmp( s, "-zt" ) == 0 )
        {
            flags |= CV_CALIB_ZERO_TANGENT_DIST;
        }
        else if( strcmp( s, "-p" ) == 0 )
        {
            flags |= CV_CALIB_FIX_PRINCIPAL_POINT;
        }
        else if( strcmp( s, "-o" ) == 0 )
        {
            outputFilename = argv[++i];
        }
        else if( s[0] != '-' )
        {
            inputFilename = s;
		}
        else
            return fprintf( stderr, "Unknown option %s", s ), -1;
    }
    
    if( !inputFilename ||
       !readStringList(inputFilename, imageList) ||
       imageList.size() == 0 || imageList.size() % 3 != 0 )
    {
        printf("Error: the input image list is not specified, or can not be read, or the number of files is not divisible by 3\n");
        return -1;
    }
    
    Mat view, viewGray;
    Mat cameraMatrix[3], distCoeffs[3], R[3], P[3], R12, T12;
    for( k = 0; k < 3; k++ )
    {
        cameraMatrix[k] = Mat_<double>::eye(3,3);
        cameraMatrix[k].at<double>(0,0) = aspectRatio;
        cameraMatrix[k].at<double>(1,1) = 1;
        distCoeffs[k] = Mat_<double>::zeros(5,1);
    }
    Mat R13=Mat_<double>::eye(3,3), T13=Mat_<double>::zeros(3,1);
    
    FileStorage fs;
    namedWindow( "Image View", 0 );
    
    for( k = 0; k < 3; k++ )
        imgpt[k].resize(imageList.size()/3);
    
    for( i = 0; i < (int)(imageList.size()/3); i++ )
    {
        for( k = 0; k < 3; k++ )
        {
            int k1 = k == 0 ? 2 : k == 1 ? 0 : 1;
            printf("%s\n", imageList[i*3+k].c_str());
            view = imread(imageList[i*3+k], 1);
            
            if(view.data)
            {
                vector<Point2f> ptvec;
                imageSize = view.size();
                cvtColor(view, viewGray, CV_BGR2GRAY);
                bool found = findChessboardCorners( view, boardSize, ptvec, CV_CALIB_CB_ADAPTIVE_THRESH );
            
                drawChessboardCorners( view, boardSize, Mat(ptvec), found );
                if( found )
                {
                    imgpt[k1][i].resize(ptvec.size());
                    std::copy(ptvec.begin(), ptvec.end(), imgpt[k1][i].begin());
                }
                //imshow("view", view);
                //int c = waitKey(0) & 255;
                //if( c == 27 || c == 'q' || c == 'Q' )
                //    return -1;
            }
        }
    }
    
    printf("Running calibration ...\n");
    
    run3Calibration(imgpt[0], imgpt[1], imgpt[2], imageSize,
                    boardSize, squareSize, aspectRatio, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5,
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    cameraMatrix[2], distCoeffs[2],
                    R12, T12, R13, T13);
        
    fs.open(outputFilename, CV_STORAGE_WRITE);
    
    fs << "cameraMatrix1" << cameraMatrix[0];
    fs << "cameraMatrix2" << cameraMatrix[1];
    fs << "cameraMatrix3" << cameraMatrix[2];
    
    fs << "distCoeffs1" << distCoeffs[0];
    fs << "distCoeffs2" << distCoeffs[1];
    fs << "distCoeffs3" << distCoeffs[2];
    
    fs << "R12" << R12;
    fs << "T12" << T12;
    fs << "R13" << R13;
    fs << "T13" << T13;
    
    fs << "imageWidth" << imageSize.width;
    fs << "imageHeight" << imageSize.height;
    fs.release();
    
    Mat Q;
    
    // step 3: find rectification transforms
    rectify3(cameraMatrix[0], distCoeffs[0], cameraMatrix[1],
             distCoeffs[1], cameraMatrix[2], distCoeffs[2],
             imgpt[0], imgpt[2],
             imageSize, R12, T12, R13, T13,
             R[0], R[1], R[2], P[0], P[1], P[2], Q, -1.,
             imageSize, 0, 0, CV_CALIB_ZERO_DISPARITY);
    Mat map1[3], map2[3];
    
    for( k = 0; k < 3; k++ )
        initUndistortRectifyMap(cameraMatrix[k], distCoeffs[k], R[k], P[k], imageSize, CV_16SC2, map1[k], map2[k]);
    
    Mat canvas(imageSize.height, imageSize.width*3, CV_8UC3), small_canvas;
    destroyWindow("view");
    canvas = Scalar::all(0);
    
    for( i = 0; i < (int)(imageList.size()/3); i++ )
    {
        canvas = Scalar::all(0);
        for( k = 0; k < 3; k++ )
        {
            int k1 = k == 0 ? 2 : k == 1 ? 0 : 1;
            int k2 = k == 0 ? 1 : k == 1 ? 0 : 2;
            view = imread(imageList[i*3+k], 1);
            
            if(!view.data)
                continue;
            
            Mat rview = canvas.colRange(k2*imageSize.width, (k2+1)*imageSize.width);
            remap(view, rview, map1[k1], map2[k1], CV_INTER_LINEAR);
        }
        printf("%s %s %s\n", imageList[i*3].c_str(), imageList[i*3+1].c_str(), imageList[i*3+2].c_str());
        resize( canvas, small_canvas, Size(1500, 1500/3) );
        for( k = 0; k < small_canvas.rows; k += 16 )
            line(small_canvas, Point(0, k), Point(small_canvas.cols, k), Scalar(0,255,0), 1);
        imshow("rectified", small_canvas);
        int c = waitKey(0);
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
    
    return 0;
}
