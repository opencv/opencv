#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;
using namespace cv;

static string helphelp(char** argv)
{
    return  string("\nThis program's purpose is to collect data sets of an object and its segmentation mask.\n")
        +   "\n"
            "It shows how to use a calibrated camera together with a calibration pattern to\n"
            "compute the homography of the plane the calibration pattern is on. It also shows grabCut\n"
            "segmentation etc.\n"
            "\n"
        +   argv[0]
        +   " -w=<board_width> -h=<board_height> [-s=<square_size>]\n"
            "           -i=<camera_intrinsics_filename> -o=<output_prefix>\n"
            "\n"
            " -w=<board_width>          Number of chessboard corners wide\n"
            " -h=<board_height>         Number of chessboard corners height\n"
            " [-s=<square_size>]        Optional measure of chessboard squares in meters\n"
            " -i=<camera_intrinsics_filename> Camera matrix .yml file from calibration.cpp\n"
            " -o=<output_prefix>        Prefix the output segmentation images with this\n"
            " [video_filename/cameraId] If present, read from that video file or that ID\n"
            "\n"
            "Using a camera's intrinsics (from calibrating a camera -- see calibration.cpp) and an\n"
            "image of the object sitting on a planar surface with a calibration pattern of\n"
            "(board_width x board_height) on the surface, we draw a 3D box around the object. From\n"
            "then on, we can move a camera and as long as it sees the chessboard calibration pattern,\n"
            "it will store a mask of where the object is. We get successive images using <output_prefix>\n"
            "of the segmentation mask containing the object. This makes creating training sets easy.\n"
            "It is best if the chessboard is odd x even in dimensions to avoid ambiguous poses.\n"
            "\n"
            "The actions one can use while the program is running are:\n"
            "\n"
            "  Select object as 3D box with the mouse.\n"
            "   First draw one line on the plane to outline the projection of that object on the plane\n"
            "    Then extend that line into a box to encompass the projection of that object onto the plane\n"
            "    The use the mouse again to extend the box upwards from the plane to encase the object.\n"
            "  Then use the following commands\n"
            "    ESC   - Reset the selection\n"
            "    SPACE - Skip the frame; move to the next frame (not in video mode)\n"
            "    ENTER - Confirm the selection. Grab next object in video mode.\n"
            "    q     - Exit the program\n"
            "\n\n";
}

static bool readCameraMatrix(const string& filename,
                             Mat& cameraMatrix, Mat& distCoeffs,
                             Size& calibratedImageSize )
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: Could not open the file " << filename << endl;
        return false;
    }
    fs["image_width"] >> calibratedImageSize.width;
    fs["image_height"] >> calibratedImageSize.height;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["cameraMatrix1"] >> cameraMatrix; // 或 cameraMatrix2, cameraMatrix3
    fs["distCoeffs1"] >> distCoeffs; // 或 distCoeffs2, distCoeffs3

    cout << "Loaded camera matrix: " << cameraMatrix << endl;
    cout << "Loaded distortion coefficients: " << distCoeffs << endl;

    if( distCoeffs.type() != CV_64F )
        distCoeffs = Mat_<double>(distCoeffs);
    if( cameraMatrix.type() != CV_64F )
        cameraMatrix = Mat_<double>(cameraMatrix);

    return true;
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.resize(0);

    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(Point3f(float(j*squareSize),
                                      float(i*squareSize), 0));
}


static Rect extract3DBox(const Mat& frame, Mat& selectedObjFrame,
                         const Mat& cameraMatrix, const Mat& rvec, const Mat& tvec,
                         const vector<Point3f>& box, int nobjpt, bool runExtraSegmentation)
{
    selectedObjFrame = Mat::zeros(frame.size(), frame.type());
    if( nobjpt == 0 )
        return Rect();
    vector<Point3f> objpt;
    vector<Point2f> imgpt;

    objpt.push_back(box[0]);
    if( nobjpt > 1 )
        objpt.push_back(box[1]);
    if( nobjpt > 2 )
    {
        objpt.push_back(box[2]);
        objpt.push_back(objpt[2] - objpt[1] + objpt[0]);
    }
    if( nobjpt > 3 )
        for( int i = 0; i < 4; i++ )
            objpt.push_back(Point3f(objpt[i].x, objpt[i].y, box[3].z));

    projectPoints(Mat(objpt), rvec, tvec, cameraMatrix, Mat(), imgpt);

    if( nobjpt <= 2 )
        return Rect();
    vector<Point> hull;
    convexHull(Mat_<Point>(Mat(imgpt)), hull);
    Mat selectedObjMask = Mat::zeros(frame.size(), CV_8U);
    fillConvexPoly(selectedObjMask, &hull[0], (int)hull.size(), Scalar::all(255), 8, 0);
    Rect roi = boundingRect(Mat(hull)) & Rect(Point(), frame.size());

    if( runExtraSegmentation )
    {
        selectedObjMask = Scalar::all(GC_BGD);
        fillConvexPoly(selectedObjMask, &hull[0], (int)hull.size(), Scalar::all(GC_PR_FGD), 8, 0);
        Mat bgdModel, fgdModel;
        grabCut(frame, selectedObjMask, roi, bgdModel, fgdModel,
                3, GC_INIT_WITH_RECT + GC_INIT_WITH_MASK);
        bitwise_and(selectedObjMask, Scalar::all(1), selectedObjMask);
    }

    frame.copyTo(selectedObjFrame, selectedObjMask);
    return roi;
}

int main(int argc, char** argv)
{
    string help = string("Usage: ") + argv[0] + " -w=<board_width> -h=<board_height> [-s=<square_size>]\n" +
           "\t-i=<intrinsics_filename> -o=<output_prefix> [video_filename/cameraId]\n";

    cv::CommandLineParser parser(argc, argv, "{help h||}{w||}{h||}{s|1|}{i||}{o||}{@input|0|}");
    if (parser.has("help"))
    {
        puts(helphelp(argv).c_str());
        puts(help.c_str());
        return 0;
    }

    string intrinsicsFilename = parser.get<string>("i");
    string outprefix = parser.get<string>("o");
    string inputName = parser.get<string>("@input");
    int cameraId = 0;
    Size boardSize;
    double squareSize;

    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    squareSize = parser.get<double>("s");

    if (inputName.size() == 1 && isdigit(inputName[0]))
        cameraId = parser.get<int>("@input");
    else
        inputName = samples::findFileOrKeep(inputName);

    if (!parser.check())
    {
        puts(help.c_str());
        parser.printErrors();
        return 0;
    }

    if (boardSize.width <= 0)
    {
        printf("Incorrect -w parameter (must be a positive integer)\n");
        puts(help.c_str());
        return 0;
    }

    if (boardSize.height <= 0)
    {
        printf("Incorrect -h parameter (must be a positive integer)\n");
        puts(help.c_str());
        return 0;
    }

    if (squareSize <= 0)
    {
        printf("Incorrect -s parameter (must be a positive real number)\n");
        puts(help.c_str());
        return 0;
    }

    Mat cameraMatrix, distCoeffs;
    Size calibratedImageSize;
    bool success = readCameraMatrix("camera_params.yml", cameraMatrix, distCoeffs, calibratedImageSize);


    VideoCapture capture;
    if (!inputName.empty())
    {
        if (!capture.open(inputName))
        {
            fprintf(stderr, "The input file could not be opened\n");
            return -1;
        }
    }
    else
        capture.open(cameraId);

    if (!capture.isOpened())
    {
        fprintf(stderr, "Could not initialize video capture\n");
        return -2;
    }

    // 创建输出目录
    system(("mkdir -p " + outprefix).c_str());

    Mat frame, selectedObjFrame, mapxy;
    vector<Point3f> box, boardPoints;

    calcChessboardCorners(boardSize, (float)squareSize, boardPoints);
    int frameIdx = 0;
    bool boardFound = false;

    for (int i = 0;; i++)
    {
        Mat frame0;
        capture >> frame0;
        if (frame0.empty())
            break;

        if (frame.empty())
        {
            if (frame0.size() != calibratedImageSize)
            {
                double sx = (double)frame0.cols / calibratedImageSize.width;
                double sy = (double)frame0.rows / calibratedImageSize.height;

                // adjust the camera matrix for the new resolution
                cameraMatrix.at<double>(0, 0) *= sx;
                cameraMatrix.at<double>(0, 2) *= sx;
                cameraMatrix.at<double>(1, 1) *= sy;
                cameraMatrix.at<double>(1, 2) *= sy;
            }
            Mat dummy;
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                    cameraMatrix, frame0.size(),
                                    CV_32FC2, mapxy, dummy);
            distCoeffs = Mat::zeros(5, 1, CV_64F);
        }
        remap(frame0, frame, mapxy, Mat(), INTER_LINEAR);
        vector<Point2f> foundBoardCorners;
        boardFound = findChessboardCorners(frame, boardSize, foundBoardCorners);

        Mat rvec, tvec;
        if (boardFound)
            solvePnP(Mat(boardPoints), Mat(foundBoardCorners), cameraMatrix,
                     distCoeffs, rvec, tvec, false);

        selectedObjFrame = Mat::zeros(frame.size(), frame.type());

        if (boardFound && !box.empty())
        {
            Rect r = extract3DBox(frame, selectedObjFrame,
                                  cameraMatrix, rvec, tvec, box, 4, true);
            if (!r.empty())
            {
                string path = format("%s/%04d.jpg", outprefix.c_str(), frameIdx++);
                imwrite(path, selectedObjFrame(r));
                cout << "Saved: " << path << endl;
            }
        }
    }

    return 0;
}
