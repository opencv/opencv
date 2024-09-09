#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

static int print_help(char** argv)
{
    cout <<
            " Given a list of chessboard or ChArUco images, the number of corners (nx, ny)\n"
            " on the chessboards and the number of squares (nx, ny) on ChArUco,\n"
            " and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use stereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n" << endl;
    cout << "Usage:\n " << argv[0] << " -w=<board_width default=9> -h=<board_height default=6>"
        <<" -t=<pattern type: chessboard or charucoboard default=chessboard> -s=<square_size default=1.0> -ms=<marker size default=0.5>"
        <<" -ad=<predefined aruco dictionary name default=DICT_4X4_50> -adf=<aruco dictionary file default=None>"
        <<" <image list XML/YML file default=stereo_calib.xml>\n" << endl;
    cout << "Available Aruco dictionaries: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, "
        << "DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, "
        << "DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, "
        << "DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, "
        << "DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11\n";

    return 0;
}

static void
StereoCalib(const vector<string>& imagelist, Size inputBoardSize, string type, float squareSize, float markerSize, cv::aruco::PredefinedDictionaryType arucoDict, string arucoDictFile, bool displayCorners = false, bool useCalibrated=true, bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2;
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    Size boardSizeInnerCorners, boardSizeUnits;
    if (type == "chessboard") {
        //chess board pattern boardSize is given in inner corners
        boardSizeInnerCorners = inputBoardSize;
        boardSizeUnits.height = inputBoardSize.height+1;
        boardSizeUnits.width = inputBoardSize.width+1;
    }
    else if (type == "charucoboard") {
        //ChArUco board pattern boardSize is given in squares units
        boardSizeUnits = inputBoardSize;
        boardSizeInnerCorners.width = inputBoardSize.width - 1;
        boardSizeInnerCorners.height = inputBoardSize.height - 1;
    }
    else {
        std::cout << "unknown pattern type " << type << "\n";
        return;
    }

    cv::aruco::Dictionary dictionary;
    if (arucoDictFile == "None") {
        dictionary = cv::aruco::getPredefinedDictionary(arucoDict);
    }
    else {
        cv::FileStorage dict_file(arucoDictFile, cv::FileStorage::Mode::READ);
        cv::FileNode fn(dict_file.root());
        dictionary.readDictionary(fn);
    }
    cv::aruco::CharucoBoard ch_board(boardSizeUnits, squareSize, markerSize, dictionary);
    cv::aruco::CharucoDetector ch_detector(ch_board);
    std::vector<int> markerIds;

    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            cout << "Reading file: " << filename << endl; // 添加调试信息
            Mat img = imread(filename, IMREAD_GRAYSCALE);
            if(img.empty())
            {
                cout << "Error: Couldn't load image " << filename << endl;
                break;
            }
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);

                if (type == "chessboard") {
                    found = findChessboardCorners(timg, boardSizeInnerCorners, corners,
                        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                }
                else if (type == "charucoboard") {
                    ch_detector.detectBoard(timg, corners, markerIds);
                    found = corners.size() == (size_t) (boardSizeInnerCorners.height*boardSizeInnerCorners.width);
                }
                else {
                    cout << "Error: unknown pattern " << type << "\n";
                    return;
                }
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSizeInnerCorners, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
                // imshow("corners", cimg1); // 注释掉imshow
                // char c = (char)waitKey(500); // 注释掉waitKey
                // if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                //    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
            if (type == "chessboard") {
                cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
                    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                        30, 0.01));
            }
        }
        if( k == 2 )
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSizeInnerCorners.height; j++ )
            for( k = 0; k < boardSizeInnerCorners.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_USE_INTRINSIC_GUESS +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " <<  err/npoints << endl;

    // save intrinsic parameters
    string outputDir = "stereo_calib";
    mkdir(outputDir.c_str(), 0777);

    FileStorage fs(outputDir + "/intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
        cout << "Intrinsic parameters saved to " << outputDir + "/intrinsics.yml" << endl;
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open(outputDir + "/extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
        cout << "Extrinsic parameters saved to " << outputDir + "/extrinsics.yml" << endl;
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    for( i = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], IMREAD_GRAYSCALE), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
        }

        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        // imshow("rectified", canvas); // 注释掉imshow
        // char c = (char)waitKey(); // 注释掉waitKey
        // if( c == 27 || c == 'q' || c == 'Q' )
        //    break;

        // Save rectified images
        string rectifiedFilename = outputDir + "/rectified_" + to_string(i) + ".png";
        imwrite(rectifiedFilename, canvas);
        cout << "Rectified image saved to " << rectifiedFilename << endl;
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

int main(int argc, char** argv)
{
    Size inputBoardSize;
    string imagelistfn;
    bool showRectified;
    cv::CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{t|chessboard|}{s|1.0|}{ms|0.5|}{ad|DICT_4X4_50|}{adf|None|}{nr||}{help||}{@input|stereo_calib.xml|}");
    if (parser.has("help"))
        return print_help(argv);
    showRectified = !parser.has("nr");
    imagelistfn = samples::findFile(parser.get<string>("@input"));
    inputBoardSize.width = parser.get<int>("w");
    inputBoardSize.height = parser.get<int>("h");
    string type = parser.get<string>("t");
    float squareSize = parser.get<float>("s");
    float markerSize = parser.get<float>("ms");
    string arucoDictName = parser.get<string>("ad");
    string arucoDictFile = parser.get<string>("adf");

    cv::aruco::PredefinedDictionaryType arucoDict;
    if (arucoDictName == "DICT_4X4_50") { arucoDict = cv::aruco::DICT_4X4_50; }
    else if (arucoDictName == "DICT_4X4_100") { arucoDict = cv::aruco::DICT_4X4_100; }
    else if (arucoDictName == "DICT_4X4_250") { arucoDict = cv::aruco::DICT_4X4_250; }
    else if (arucoDictName == "DICT_4X4_1000") { arucoDict = cv::aruco::DICT_4X4_1000; }
    else if (arucoDictName == "DICT_5X5_50") { arucoDict = cv::aruco::DICT_5X5_50; }
    else if (arucoDictName == "DICT_5X5_100") { arucoDict = cv::aruco::DICT_5X5_100; }
    else if (arucoDictName == "DICT_5X5_250") { arucoDict = cv::aruco::DICT_5X5_250; }
    else if (arucoDictName == "DICT_5X5_1000") { arucoDict = cv::aruco::DICT_5X5_1000; }
    else if (arucoDictName == "DICT_6X6_50") { arucoDict = cv::aruco::DICT_6X6_50; }
    else if (arucoDictName == "DICT_6X6_100") { arucoDict = cv::aruco::DICT_6X6_100; }
    else if (arucoDictName == "DICT_6X6_250") { arucoDict = cv::aruco::DICT_6X6_250; }
    else if (arucoDictName == "DICT_6X6_1000") { arucoDict = cv::aruco::DICT_6X6_1000; }
    else if (arucoDictName == "DICT_7X7_50") { arucoDict = cv::aruco::DICT_7X7_50; }
    else if (arucoDictName == "DICT_7X7_100") { arucoDict = cv::aruco::DICT_7X7_100; }
    else if (arucoDictName == "DICT_7X7_250") { arucoDict = cv::aruco::DICT_7X7_250; }
    else if (arucoDictName == "DICT_7X7_1000") { arucoDict = cv::aruco::DICT_7X7_1000; }
    else if (arucoDictName == "DICT_ARUCO_ORIGINAL") { arucoDict = cv::aruco::DICT_ARUCO_ORIGINAL; }
    else if (arucoDictName == "DICT_APRILTAG_16h5") { arucoDict = cv::aruco::DICT_APRILTAG_16h5; }
    else if (arucoDictName == "DICT_APRILTAG_25h9") { arucoDict = cv::aruco::DICT_APRILTAG_25h9; }
    else if (arucoDictName == "DICT_APRILTAG_36h10") { arucoDict = cv::aruco::DICT_APRILTAG_36h10; }
    else if (arucoDictName == "DICT_APRILTAG_36h11") { arucoDict = cv::aruco::DICT_APRILTAG_36h11; }
    else {
        cout << "incorrect name of aruco dictionary \n";
        return 1;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help(argv);
    }

    StereoCalib(imagelist, inputBoardSize, type, squareSize, markerSize, arucoDict, arucoDictFile, false, true, showRectified);
    return 0;
}

