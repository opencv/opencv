#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <string.h>
#include <time.h>

using namespace cv;
using namespace std;

/* 
 example command line for calibration from a live feed.
   calibration  -w 4 -h 5 -s 0.025 -o camera.yml -op -oe
 
 example command line for calibration from a list of stored images:
   calibration -w 4 -h 5 -s 0.025 -o camera.yml -op -oe image_list.xml
 where image_list.xml is the standard OpenCV XML/YAML
 file consisting of the list of strings, e.g.:
 
<?xml version="1.0"?>
<opencv_storage>
<images>
"view000.png"
"view001.png"
<!-- view002.png -->
"view003.png"
"view010.png"
"one_extra_view.jpg"
</images>
</opencv_storage>

 */

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());
    
    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L1 );
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)(err/n);
        totalErr += err;
        totalPoints += n;
    }
    
    return totalErr/totalPoints;
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.resize(0);
    
    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(Point3f(float(j*squareSize),
                                      float(i*squareSize), 0));
}

static bool runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize,
                    float squareSize, float aspectRatio,
                    int flags, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;
    
    distCoeffs = Mat::zeros(5, 1, CV_64F);
    
    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0]);

	objectPoints.resize(imagePoints.size(),objectPoints[0]);
    
    calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                    distCoeffs, rvecs, tvecs, flags);
    
    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
    
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );
    
    time_t t;
    time( &t );
    struct tm *t2 = localtime( &t );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;
    
    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;
    
    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        sprintf( buf, "flags: %s%s%s%s",
            flags & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CV_CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );
    }
    
    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);
    
    if( !rvecs.empty() && !tvecs.empty() )
    {
        Mat bigmat(rvecs.size(), 6, CV_32F);
        for( size_t i = 0; i < rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));
            rvecs[i].copyTo(r);
            tvecs[i].copyTo(t);
        }
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }
    
    if( !imagePoints.empty() )
    {
        Mat imagePtMat(imagePoints.size(), imagePoints[0].size(), CV_32FC2);
        for( size_t i = 0; i < imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
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


bool runAndSave(const string& outputFilename,
                const vector<vector<Point2f> >& imagePoints,
                Size imageSize, Size boardSize, float squareSize,
                float aspectRatio, int flags, Mat& cameraMatrix,
                Mat& distCoeffs, bool writeExtrinsics, bool writePoints )
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    
    bool ok = runCalibration(imagePoints, imageSize, boardSize, squareSize,
                   aspectRatio, flags, cameraMatrix, distCoeffs,
                   rvecs, tvecs, reprojErrs, totalAvgErr);
    printf("%s. avg reprojection error = %.2f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);
    
    if( ok )
        saveCameraParams( outputFilename, imageSize,
                         boardSize, squareSize, aspectRatio,
                         flags, cameraMatrix, distCoeffs,
                         writeExtrinsics ? rvecs : vector<Mat>(),
                         writeExtrinsics ? tvecs : vector<Mat>(),
                         writeExtrinsics ? reprojErrs : vector<float>(),
                         writePoints ? imagePoints : vector<vector<Point2f> >(),
                         totalAvgErr );
    return ok;
}


int main( int argc, char** argv )
{
    Size boardSize, imageSize;
    float squareSize = 1.f, aspectRatio = 1.f;
    Mat cameraMatrix, distCoeffs;
    const char* outputFilename = "out_camera_data.yml";
    const char* inputFilename = 0;
    
    int i, nframes = 10;
    bool writeExtrinsics = false, writePoints = false;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;
    bool flipVertical = false;
    int delay = 1000;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f> > imagePoints;
    vector<string> imageList;

    const char* liveCaptureHelp = 
        "When the live video from camera is used as input, the following hot-keys may be used:\n"
            "  <ESC>, 'q' - quit the program\n"
            "  'g' - start capturing images\n"
            "  'u' - switch undistortion on/off\n";

    if( argc < 2 )
    {
        printf( "This is a camera calibration sample.\n"
            "Usage: calibration\n"
            "     -w <board_width>         # the number of inner corners per one of board dimension\n"
            "     -h <board_height>        # the number of inner corners per another board dimension\n"
            "     [-n <number_of_frames>]  # the number of frames to use for calibration\n"
            "                              # (if not specified, it will be set to the number\n"
            "                              #  of board views actually available)\n"
            "     [-d <delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
            "                              # (used only for video capturing)\n"
            "     [-s <squareSize>]       # square size in some user-defined units (1 by default)\n"
            "     [-o <out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
            "     [-op]                    # write detected feature points\n"
            "     [-oe]                    # write extrinsic parameters\n"
            "     [-zt]                    # assume zero tangential distortion\n"
            "     [-a <aspectRatio>]      # fix aspect ratio (fx/fy)\n"
            "     [-p]                     # fix the principal point at the center\n"
            "     [-v]                     # flip the captured images around the horizontal axis\n"
            "     [input_data]             # input data, one of the following:\n"
            "                              #  - text file with a list of the images of the board\n"
            "                              #  - name of video file with a video of the board\n"
            "                              # if input_data not specified, a live view from the camera is used\n"
            "\n" );
        printf( "%s", liveCaptureHelp );
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
        else if( strcmp( s, "-n" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &nframes ) != 1 || nframes <= 3 )
                return printf("Invalid number of images\n" ), -1;
        }
        else if( strcmp( s, "-a" ) == 0 )
        {
            if( sscanf( argv[++i], "%f", &aspectRatio ) != 1 || aspectRatio <= 0 )
                return printf("Invalid aspect ratio\n" ), -1;
            flags |= CV_CALIB_FIX_ASPECT_RATIO;
        }
        else if( strcmp( s, "-d" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &delay ) != 1 || delay <= 0 )
                return printf("Invalid delay\n" ), -1;
        }
        else if( strcmp( s, "-op" ) == 0 )
        {
            writePoints = 1;
        }
        else if( strcmp( s, "-oe" ) == 0 )
        {
            writeExtrinsics = 1;
        }
        else if( strcmp( s, "-zt" ) == 0 )
        {
            flags |= CV_CALIB_ZERO_TANGENT_DIST;
        }
        else if( strcmp( s, "-p" ) == 0 )
        {
            flags |= CV_CALIB_FIX_PRINCIPAL_POINT;
        }
        else if( strcmp( s, "-v" ) == 0 )
        {
            flipVertical = 1;
        }
        else if( strcmp( s, "-o" ) == 0 )
        {
            outputFilename = argv[++i];
        }
        else if( s[0] != '-' )
        {
			if( isdigit(s[0]) )
				sscanf(s, "%d", &cameraId);
			else
				inputFilename = s;
		}
        else
            return fprintf( stderr, "Unknown option %s", s ), -1;
    }

    if( inputFilename )
    {
        if( readStringList(inputFilename, imageList) )
            mode = CAPTURING;
        else    
            capture.open(inputFilename);
    }
    else
        capture.open(cameraId);

    if( !capture.isOpened() && imageList.empty() )
        return fprintf( stderr, "Could not initialize video capture\n" ), -2;
    
    if( !imageList.empty() )
        nframes = (int)imageList.size();

    if( capture.isOpened() )
        printf( "%s", liveCaptureHelp );

    namedWindow( "Image View", 1 );

    for(i = 0;;i++)
    {
        Mat view, viewGray;
        bool blink = false;
        
        if( capture.isOpened() )
        {
            Mat view0;
            capture >> view0;
            view0.copyTo(view);
        }
        else if( i < (int)imageList.size() )
            view = imread(imageList[i], 1);
        
        if(!view.data)
        {
            if( imagePoints.size() > 0 )
                runAndSave(outputFilename, imagePoints, imageSize,
                           boardSize, squareSize, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints);
            break;
        }
        
        imageSize = view.size();

        if( flipVertical )
            flip( view, view, 0 );

        vector<Point2f> pointbuf;
        bool found = findChessboardCorners( view, boardSize, pointbuf, CV_CALIB_CB_ADAPTIVE_THRESH );

        // improve the found corners' coordinate accuracy
        cvtColor(view, viewGray, CV_BGR2GRAY);
        if(found) cornerSubPix( viewGray, pointbuf, Size(11,11),
            Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

        if( mode == CAPTURING && found &&
           (!capture.isOpened() || clock() - prevTimestamp > delay*1e-3*CLOCKS_PER_SEC) )
        {
            imagePoints.push_back(pointbuf);
            prevTimestamp = clock();
            blink = capture.isOpened();
        }

        if(found) drawChessboardCorners( view, boardSize, Mat(pointbuf), found );

        string msg = mode == CAPTURING ? "100/100" :
            mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);        
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
			if(undistortImage)
            	msg = format( "%d/%d Undist", (int)imagePoints.size(), nframes );
            else
            	msg = format( "%d/%d", (int)imagePoints.size(), nframes );
		}

        putText( view, msg, textOrigin, 1, 1,
                 mode != CALIBRATED ? Scalar(0,0,255) : Scalar(0,255,0));

        if( blink )
            bitwise_not(view, view);

        if( mode == CALIBRATED && undistortImage )
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }

        imshow("Image View", view);
        int key = waitKey(capture.isOpened() ? 50 : 500);

        if( (key & 255) == 27 )
            break;
        
        if( key == 'u' && mode == CALIBRATED )
            undistortImage = !undistortImage;

        if( capture.isOpened() && key == 'g' )
        {
            mode = CAPTURING;
            imagePoints.clear();
        }

        if( mode == CAPTURING && imagePoints.size() >= (unsigned)nframes )
        {
            if( runAndSave(outputFilename, imagePoints, imageSize,
                       boardSize, squareSize, aspectRatio,
                       flags, cameraMatrix, distCoeffs,
                       writeExtrinsics, writePoints))
                mode = CALIBRATED;
            else
                mode = DETECTION;
            if( !capture.isOpened() )
                break;
        }
    }
    return 0;
}
