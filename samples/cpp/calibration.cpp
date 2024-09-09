#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
//#include <opencv2/objdetect/charuco_detector.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace std;

const char * usage =
" \nexample command line for calibration from a live feed.\n"
"   calibration  -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe\n"
" \n"
" example command line for calibration from a list of stored images:\n"
"   imagelist_creator image_list.xml *.png\n"
"   calibration -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe image_list.xml\n"
" where image_list.xml is the standard OpenCV XML/YAML\n"
" use imagelist_creator to create the xml or yaml list\n"
" file consisting of the list of strings, e.g.:\n"
" \n"
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<images>\n"
"view000.png\n"
"view001.png\n"
"<!-- view002.png -->\n"
"view003.png\n"
"view010.png\n"
"one_extra_view.jpg\n"
"</images>\n"
"</opencv_storage>\n";

const char* liveCaptureHelp =
    "When the live video from camera is used as input, the following hot-keys may be used:\n"
        "  <ESC>, 'q' - quit the program\n"
        "  'g' - start capturing images\n"
        "  'u' - switch undistortion on/off\n";

static void help(char** argv)
{
    printf( "This is a camera calibration sample.\n"
        "Usage: %s\n"
        "     -w=<board_width>         # the calibration board horizontal size in inner corners "
        "for chessboard and in squares or circles for others like ChArUco or circles grid\n"
        "     -h=<board_height>        # the calibration board verical size in inner corners "
        "for chessboard and in squares or circles for others like ChArUco or circles grid\n"
        "     [-pt=<pattern>]          # the type of pattern: chessboard, charuco, circles, acircles\n"
        "     [-n=<number_of_frames>]  # the number of frames to use for calibration\n"
        "                              # (if not specified, it will be set to the number\n"
        "                              #  of board views actually available)\n"
        "     [-d=<delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
        "                              # (used only for video capturing)\n"
        "     [-s=<squareSize>]        # square size in some user-defined units (1 by default)\n"
        "     [-ms=<markerSize>]       # marker size in some user-defined units (0.5 by default)\n"
        "     [-ad=<arucoDict>]        # Aruco dictionary name for ChArUco board. "
        "Available ArUco dictionaries: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, "
        "DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, "
        "DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, "
        "DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, "
        "DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11\n"
        "     [-adf=<dictFilename>]    # Custom aruco dictionary file for ChArUco board\n"
        "     [-o=<out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
        "     [-op]                    # write detected feature points\n"
        "     [-oe]                    # write extrinsic parameters\n"
        "     [-oo]                    # write refined 3D object points\n"
        "     [-zt]                    # assume zero tangential distortion\n"
        "     [-a=<aspectRatio>]       # fix aspect ratio (fx/fy)\n"
        "     [-p]                     # fix the principal point at the center\n"
        "     [-v]                     # flip the captured images around the horizontal axis\n"
        "     [-V]                     # use a video file, and not an image list, uses\n"
        "                              # [input_data] string for the video file name\n"
        "     [-su]                    # show undistorted images after calibration\n"
        "     [-ws=<number_of_pixel>]  # half of search window for cornerSubPix (11 by default)\n"
        "     [-fx=<X focal length>]   # focal length in X-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-fy=<Y focal length>]   # focal length in Y-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-cx=<X center point>]   # camera center point in X-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-cy=<Y center point>]   # camera center point in Y-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-imshow-scale           # image resize scaling factor when displaying the results (must be >= 1)\n"
        "     [-enable-k3=<0/1>        # to enable (1) or disable (0) K3 coefficient for the distortion model\n"
        "     [-dt=<distance>]         # actual distance between top-left and top-right corners of\n"
        "                              # the calibration grid. If this parameter is specified, a more\n"
        "                              # accurate calibration method will be used which may be better\n"
        "                              # with inaccurate, roughly planar target.\n"
        "     [input_data]             # input data, one of the following:\n"
        "                              #  - text file with a list of the images of the board\n"
        "                              #    the text file can be generated with imagelist_creator\n"
        "                              #  - name of video file with a video of the board\n"
        "                              # if input_data not specified, a live view from the camera is used\n"
        "\n", argv[0] );
    printf("\n%s",usage);
    printf( "\n%s", liveCaptureHelp );
}

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID, CHARUCOBOARD};

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
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}
// 删除 calcChessboardCorners 函数中与 CharucoBoard 相关的部分

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch(patternType)
    {
      case CHESSBOARD:
      case CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float(j*squareSize),
                                          float(i*squareSize), 0));
        break;

      case ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize),
                                          float(i*squareSize), 0));
        break;

    //   case CHARUCOBOARD:
    //     for( int i = 0; i < boardSize.height-1; i++ )
    //         for( int j = 0; j < boardSize.width-1; j++ )
    //             corners.push_back(Point3f(float(j*squareSize),
    //                                       float(i*squareSize), 0));
    //     break;
      default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

static bool runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize, Pattern patternType,
                    float squareSize, float aspectRatio,
                    float grid_width, bool release_object,
                    int flags, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    vector<Point3f>& newObjPoints,
                    double& totalAvgErr)
{
    if( flags & CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
    int offset = patternType != CHARUCOBOARD ? boardSize.width - 1: boardSize.width - 2;
    objectPoints[0][offset].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms;
    int iFixedPoint = -1;
    if (release_object)
        iFixedPoint = boardSize.width - 1;
    rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
                            cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
                            flags | CALIB_USE_LU);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    if (release_object) {
        cout << "New board corners: " << endl;
        cout << newObjPoints[0] << endl;
        cout << newObjPoints[boardSize.width - 1] << endl;
        cout << newObjPoints[boardSize.width * (boardSize.height - 1)] << endl;
        cout << newObjPoints.back() << endl;
    }

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

static void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       const vector<Point3f>& newObjPoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
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

    if( flags & CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        snprintf( buf, sizeof(buf), "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        //cvWriteComment( *fs, buf, 0);
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }

    if( !newObjPoints.empty() )
    {
        fs << "grid_points" << newObjPoints;
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
    {
        string fname = (string)*it;
        l.push_back(fname);
    }
    return true;
}


static bool runAndSave(const string& outputFilename,
                const vector<vector<Point2f> >& imagePoints,
                Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                float grid_width, bool release_object,
                float aspectRatio, int flags, Mat& cameraMatrix,
                Mat& distCoeffs, bool writeExtrinsics, bool writePoints, bool writeGrid )
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    vector<Point3f> newObjPoints;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
                   aspectRatio, grid_width, release_object, flags, cameraMatrix, distCoeffs,
                   rvecs, tvecs, reprojErrs, newObjPoints, totalAvgErr);
    printf("%s. avg reprojection error = %.7f\n",
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
                         writeGrid ? newObjPoints : vector<Point3f>(),
                         totalAvgErr );
    return ok;
}

int main( int argc, char** argv )
{
    Size boardSize, imageSize;
    float squareSize, markerSize, aspectRatio = 1;
    Mat cameraMatrix, distCoeffs;
    string outputFilename;
    string inputFilename = "";
    int arucoDict;
    string dictFilename;

    int i, nframes;
    bool writeExtrinsics, writePoints;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;
    bool flipVertical;
    bool showUndistorted;
    bool videofile;
    int delay;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f>> imagePoints;
    vector<string> imageList;
    Pattern pattern = CHESSBOARD;

    cv::CommandLineParser parser(argc, argv,
        "{help ||}{w||}{h||}{pt|chessboard|}{n|10|}{d|1000|}{s|1|}{ms|0.5|}{ad|DICT_4X4_50|}{adf|None|}{o|out_camera_data.yml|}"
        "{op||}{oe||}{zt||}{a||}{p||}{v||}{V||}{su||}"
        "{oo||}{ws|11|}{dt||}"
        "{fx||}{fy||}{cx||}{cy||}"
        "{imshow-scale|1|}{enable-k3|0|}"
        "{@input_data|0|}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    boardSize.width = parser.get<int>( "w" );
    boardSize.height = parser.get<int>( "h" );
    if ( parser.has("pt") )
    {
        string val = parser.get<string>("pt");
        if( val == "circles" )
            pattern = CIRCLES_GRID;
        else if( val == "acircles" )
            pattern = ASYMMETRIC_CIRCLES_GRID;
        else if( val == "chessboard" )
            pattern = CHESSBOARD;
        else if( val == "charuco" )
            pattern = CHARUCOBOARD;
        else
            return fprintf( stderr, "Invalid pattern type: must be chessboard or circles\n" ), -1;
    }

    squareSize = parser.get<float>("s");
    markerSize = parser.get<float>("ms");

    // string arucoDictName = parser.get<string>("ad");
    // if (arucoDictName == "DICT_4X4_50") { arucoDict = cv::aruco::DICT_4X4_50; }
    // else if (arucoDictName == "DICT_4X4_100") { arucoDict = cv::aruco::DICT_4X4_100; }
    // else if (arucoDictName == "DICT_4X4_250") { arucoDict = cv::aruco::DICT_4X4_250; }
    // else if (arucoDictName == "DICT_4X4_1000") { arucoDict = cv::aruco::DICT_4X4_1000; }
    // else if (arucoDictName == "DICT_5X5_50") { arucoDict = cv::aruco::DICT_5X5_50; }
    // else if (arucoDictName == "DICT_5X5_100") { arucoDict = cv::aruco::DICT_5X5_100; }
    // else if (arucoDictName == "DICT_5X5_250") { arucoDict = cv::aruco::DICT_5X5_250; }
    // else if (arucoDictName == "DICT_5X5_1000") { arucoDict = cv::aruco::DICT_5X5_1000; }
    // else if (arucoDictName == "DICT_6X6_50") { arucoDict = cv::aruco::DICT_6X6_50; }
    // else if (arucoDictName == "DICT_6X6_100") { arucoDict = cv::aruco::DICT_6X6_100; }
    // else if (arucoDictName == "DICT_6X6_250") { arucoDict = cv::aruco::DICT_6X6_250; }
    // else if (arucoDictName == "DICT_6X6_1000") { arucoDict = cv::aruco::DICT_6X6_1000; }
    // else if (arucoDictName == "DICT_7X7_50") { arucoDict = cv::aruco::DICT_7X7_50; }
    // else if (arucoDictName == "DICT_7X7_100") { arucoDict = cv::aruco::DICT_7X7_100; }
    // else if (arucoDictName == "DICT_7X7_250") { arucoDict = cv::aruco::DICT_7X7_250; }
    // else if (arucoDictName == "DICT_7X7_1000") { arucoDict = cv::aruco::DICT_7X7_1000; }
    // else if (arucoDictName == "DICT_ARUCO_ORIGINAL") { arucoDict = cv::aruco::DICT_ARUCO_ORIGINAL; }
    // else if (arucoDictName == "DICT_APRILTAG_16h5") { arucoDict = cv::aruco::DICT_APRILTAG_16h5; }
    // else if (arucoDictName == "DICT_APRILTAG_25h9") { arucoDict = cv::aruco::DICT_APRILTAG_25h9; }
    // else if (arucoDictName == "DICT_APRILTAG_36h10") { arucoDict = cv::aruco::DICT_APRILTAG_36h10; }
    // else if (arucoDictName == "DICT_APRILTAG_36h11") { arucoDict = cv::aruco::DICT_APRILTAG_36h11; }
    // else {
    //     cout << "Incorrect Aruco dictionary name " <<  arucoDictName << std::endl;
    //     return 1;
    // }

    dictFilename = parser.get<std::string>("adf");
    nframes = parser.get<int>("n");
    delay = parser.get<int>("d");
    writePoints = parser.has("op");
    writeExtrinsics = parser.has("oe");
    bool writeGrid = parser.has("oo");
    if (parser.has("a")) {
        flags |= CALIB_FIX_ASPECT_RATIO;
        aspectRatio = parser.get<float>("a");
    }
    if ( parser.has("zt") )
        flags |= CALIB_ZERO_TANGENT_DIST;
    if ( parser.has("p") )
        flags |= CALIB_FIX_PRINCIPAL_POINT;
    flipVertical = parser.has("v");
    videofile = parser.has("V");
    if ( parser.has("o") )
        outputFilename = parser.get<string>("o");
    showUndistorted = parser.has("su");
    if ( isdigit(parser.get<string>("@input_data")[0]) )
        cameraId = parser.get<int>("@input_data");
    else
        inputFilename = parser.get<string>("@input_data");
    int winSize = parser.get<int>("ws");
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (parser.has("fx") && parser.has("fy") && parser.has("cx") && parser.has("cy"))
    {
        cameraMatrix.at<double>(0,0) = parser.get<double>("fx");
        cameraMatrix.at<double>(0,2) = parser.get<double>("cx");
        cameraMatrix.at<double>(1,1) = parser.get<double>("fy");
        cameraMatrix.at<double>(1,2) = parser.get<double>("cy");
        flags |= CALIB_USE_INTRINSIC_GUESS;
        std::cout << "Use the following camera matrix as an initial guess:\n" << cameraMatrix << std::endl;
    }
    int viewScaleFactor = parser.get<int>("imshow-scale");
    bool useK3 = parser.get<bool>("enable-k3");
    std::cout << "Use K3 distortion coefficient? " << useK3 << std::endl;
    if (!useK3)
    {
        flags |= CALIB_FIX_K3;
    }

    float grid_width = squareSize *(pattern != CHARUCOBOARD ? (boardSize.width - 1): (boardSize.width - 2) );
    bool release_object = false;
    if (parser.has("dt")) {
        grid_width = parser.get<float>("dt");
        release_object = true;
    }
    if (!parser.check())
    {
        help(argv);
        parser.printErrors();
        return -1;
    }
    if ( squareSize <= 0 )
        return fprintf( stderr, "Invalid board square width\n" ), -1;
    if ( nframes <= 3 )
        return printf("Invalid number of images\n" ), -1;
    if ( aspectRatio <= 0 )
        return printf( "Invalid aspect ratio\n" ), -1;
    if ( delay <= 0 )
        return printf( "Invalid delay\n" ), -1;
    if ( boardSize.width <= 0 )
        return fprintf( stderr, "Invalid board width\n" ), -1;
    if ( boardSize.height <= 0 )
        return fprintf( stderr, "Invalid board height\n" ), -1;

    // 删除与 CharucoBoard 相关的内容    
    // cv::aruco::Dictionary dictionary;
    // if (dictFilename == "None") {
    //     std::cout << "Using predefined dictionary with id: " << arucoDict << std::endl;
    //     dictionary = aruco::getPredefinedDictionary(arucoDict);
    // }
    // else {
    //     std::cout << "Using custom dictionary from file: " << dictFilename << std::endl;
    //     cv::FileStorage dict_file(dictFilename, cv::FileStorage::Mode::READ);
    //     cv::FileNode fn(dict_file.root());
    //     dictionary.readDictionary(fn);
    // }

    // cv::aruco::CharucoBoard ch_board(boardSize, squareSize, markerSize, dictionary);
    // std::vector<int> markerIds;
    // cv::aruco::CharucoDetector ch_detector(ch_board);

    // if( !inputFilename.empty() )
    // {
    //     if( !videofile && readStringList(samples::findFile(inputFilename), imageList) )
    //         mode = CAPTURING;
    //     else
    //         capture.open(samples::findFileOrKeep(inputFilename));
    // }
    // else
    //     capture.open(cameraId);

    // if( !capture.isOpened() && imageList.empty() )
    //     return fprintf( stderr, "Could not initialize video (%d) capture\n", cameraId ), -2;

    // if( !imageList.empty() )
    //     nframes = (int)imageList.size();

    // if( capture.isOpened() )
    //     printf( "%s", liveCaptureHelp );

    // namedWindow( "Image View", 1 );  // 注释掉.

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
            view = imread(imageList[i], IMREAD_COLOR);

        if(view.empty())
        {
            if( imagePoints.size() > 0 )
                runAndSave(outputFilename, imagePoints, imageSize,
                           boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints, writeGrid);
            break;
        }

        imageSize = view.size();

        if( flipVertical )
            flip( view, view, 0 );

        vector<Point2f> pointbuf;
        cvtColor(view, viewGray, COLOR_BGR2GRAY);

        bool found;
        // 删除 switch 语句中与 CharucoBoard 相关的部分
        switch( pattern )
        {
            case CHESSBOARD:
                found = findChessboardCorners( view, boardSize, pointbuf,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                break;
            case CIRCLES_GRID:
                found = findCirclesGrid( view, boardSize, pointbuf );
                break;
            case ASYMMETRIC_CIRCLES_GRID:
                found = findCirclesGrid( view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID );
                break;
            // case CHARUCOBOARD:
            // {
            //     ch_detector.detectBoard(view, pointbuf, markerIds);
            //     found = pointbuf.size() == (size_t)(boardSize.width-1)*(boardSize.height-1);
            //     break;
            // }
            default:
                return fprintf( stderr, "Unknown pattern type\n" ), -1;
        }

       // improve the found corners' coordinate accuracy
        if( pattern == CHESSBOARD && found) cornerSubPix( viewGray, pointbuf, Size(winSize,winSize),
            Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));

        if (mode == CAPTURING && found &&
   (!capture.isOpened() || clock() - prevTimestamp > delay*1e-3*CLOCKS_PER_SEC) )
{
    cout << "Found corners, capturing..." << endl;
    imagePoints.push_back(pointbuf);
    prevTimestamp = clock();
    blink = capture.isOpened();
}

        if(found)
        {
            if(pattern != CHARUCOBOARD)
                drawChessboardCorners( view, boardSize, Mat(pointbuf), found );
            else
                drawChessboardCorners( view, Size(boardSize.width-1, boardSize.height-1), Mat(pointbuf), found );
        }

        string msg = mode == CAPTURING ? "100/100" :
            mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(undistortImage)
                msg = cv::format( "%d/%d Undist", (int)imagePoints.size(), nframes );
            else
                msg = cv::format( "%d/%d", (int)imagePoints.size(), nframes );
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

        /*
        if (viewScaleFactor > 1)
        {
            Mat viewScale;
            resize(view, viewScale, Size(), 1.0/viewScaleFactor, 1.0/viewScaleFactor, INTER_AREA);
            imshow("Image View", viewScale);
        }
        else
        {
            imshow("Image View", view);
        }

        char key = (char)waitKey(capture.isOpened() ? 50 : 500);

        if( key == 27 )
            break;

        if( key == 'u' && mode == CALIBRATED )
            undistortImage = !undistortImage;

        if( capture.isOpened() && key == 'g' )
        {
            mode = CAPTURING;
            imagePoints.clear();
        }
        */

        if (mode == CAPTURING && imagePoints.size() >= (unsigned)nframes )
{
    if (runAndSave(outputFilename, imagePoints, imageSize, boardSize, pattern, squareSize, grid_width, release_object, aspectRatio, flags, cameraMatrix, distCoeffs, writeExtrinsics, writePoints, writeGrid))
        mode = CALIBRATED;
    else
        mode = DETECTION;
    if (!capture.isOpened())
        break;
}
    }

    if( !capture.isOpened() && showUndistorted )
    {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                imageSize, CV_16SC2, map1, map2);

        for( i = 0; i < (int)imageList.size(); i++ )
        {
            view = imread(imageList[i], IMREAD_COLOR);
            if(view.empty())
                continue;
            remap(view, rview, map1, map2, INTER_LINEAR);
            /*
            if (viewScaleFactor > 1)
            {
                Mat rviewScale;
                resize(rview, rviewScale, Size(), 1.0/viewScaleFactor, 1.0/viewScaleFactor, INTER_AREA);
                imshow("Image View", rviewScale);
            }
            else
            {
                imshow("Image View", rview);
            }
            char c = (char)waitKey();
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
            */
            imwrite("undistorted_" + imageList[i], rview);
        }
    }

    return 0;
}

