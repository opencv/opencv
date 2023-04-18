/*
 *
 * select3obj.cpp With a calibration chessboard on a table, mark an object in a 3D box and
 *                track that object in all subsequent frames as long as the camera can see
 *                the chessboard. Also segments the object using the box projection. This
 *                program is useful for collecting large datasets of many views of an object
 *                on a table.
 *
 */

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
            " -h=<board_height>         Number of chessboard corners width\n"
            " [-s=<square_size>]            Optional measure of chessboard squares in meters\n"
            " -i=<camera_intrinsics_filename> Camera matrix .yml file from calibration.cpp\n"
            " -o=<output_prefix>        Prefix the output segmentation images with this\n"
            " [video_filename/cameraId]  If present, read from that video file or that ID\n"
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
// static void help()
// {
//     puts(helphelp);
// }


struct MouseEvent
{
    MouseEvent() { event = -1; buttonState = 0; }
    Point pt;
    int event;
    int buttonState;
};

static void onMouse(int event, int x, int y, int flags, void* userdata)
{
    MouseEvent* data = (MouseEvent*)userdata;
    data->event = event;
    data->pt = Point(x,y);
    data->buttonState = flags;
}

static bool readCameraMatrix(const string& filename,
                             Mat& cameraMatrix, Mat& distCoeffs,
                             Size& calibratedImageSize )
{
    FileStorage fs(filename, FileStorage::READ);
    fs["image_width"] >> calibratedImageSize.width;
    fs["image_height"] >> calibratedImageSize.height;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;

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


static Point3f image2plane(Point2f imgpt, const Mat& R, const Mat& tvec,
                           const Mat& cameraMatrix, double Z)
{
    Mat R1 = R.clone();
    R1.col(2) = R1.col(2)*Z + tvec;
    Mat_<double> v = (cameraMatrix*R1).inv()*(Mat_<double>(3,1) << imgpt.x, imgpt.y, 1);
    double iw = fabs(v(2,0)) > DBL_EPSILON ? 1./v(2,0) : 0;
    return Point3f((float)(v(0,0)*iw), (float)(v(1,0)*iw), (float)Z);
}


static Rect extract3DBox(const Mat& frame, Mat& shownFrame, Mat& selectedObjFrame,
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

    if( !shownFrame.empty() )
    {
        if( nobjpt == 1 )
            circle(shownFrame, imgpt[0], 3, Scalar(0,255,0), -1, LINE_AA);
        else if( nobjpt == 2 )
        {
            circle(shownFrame, imgpt[0], 3, Scalar(0,255,0), -1, LINE_AA);
            circle(shownFrame, imgpt[1], 3, Scalar(0,255,0), -1, LINE_AA);
            line(shownFrame, imgpt[0], imgpt[1], Scalar(0,255,0), 3, LINE_AA);
        }
        else if( nobjpt == 3 )
            for( int i = 0; i < 4; i++ )
            {
                circle(shownFrame, imgpt[i], 3, Scalar(0,255,0), -1, LINE_AA);
                line(shownFrame, imgpt[i], imgpt[(i+1)%4], Scalar(0,255,0), 3, LINE_AA);
            }
        else
            for( int i = 0; i < 8; i++ )
            {
                circle(shownFrame, imgpt[i], 3, Scalar(0,255,0), -1, LINE_AA);
                line(shownFrame, imgpt[i], imgpt[(i+1)%4 + (i/4)*4], Scalar(0,255,0), 3, LINE_AA);
                line(shownFrame, imgpt[i], imgpt[i%4], Scalar(0,255,0), 3, LINE_AA);
            }
    }

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


static int select3DBox(const string& windowname, const string& selWinName, const Mat& frame,
                       const Mat& cameraMatrix, const Mat& rvec, const Mat& tvec,
                       vector<Point3f>& box)
{
    const float eps = 1e-3f;
    MouseEvent mouse;

    setMouseCallback(windowname, onMouse, &mouse);
    vector<Point3f> tempobj(8);
    vector<Point2f> imgpt(4), tempimg(8);
    vector<Point> temphull;
    int nobjpt = 0;
    Mat R, selectedObjMask, selectedObjFrame, shownFrame;
    Rodrigues(rvec, R);
    box.resize(4);

    for(;;)
    {
        float Z = 0.f;
        bool dragging = (mouse.buttonState & EVENT_FLAG_LBUTTON) != 0;
        int npt = nobjpt;

        if( (mouse.event == EVENT_LBUTTONDOWN ||
             mouse.event == EVENT_LBUTTONUP ||
             dragging) && nobjpt < 4 )
        {
            Point2f m = mouse.pt;

            if( nobjpt < 2 )
                imgpt[npt] = m;
            else
            {
                tempobj.resize(1);
                int nearestIdx = npt-1;
                if( nobjpt == 3 )
                {
                    nearestIdx = 0;
                    for( int i = 1; i < npt; i++ )
                        if( norm(m - imgpt[i]) < norm(m - imgpt[nearestIdx]) )
                            nearestIdx = i;
                }

                if( npt == 2 )
                {
                    float dx = box[1].x - box[0].x, dy = box[1].y - box[0].y;
                    float len = 1.f/std::sqrt(dx*dx+dy*dy);
                    tempobj[0] = Point3f(dy*len + box[nearestIdx].x,
                                         -dx*len + box[nearestIdx].y, 0.f);
                }
                else
                    tempobj[0] = Point3f(box[nearestIdx].x, box[nearestIdx].y, 1.f);

                projectPoints(Mat(tempobj), rvec, tvec, cameraMatrix, Mat(), tempimg);

                Point2f a = imgpt[nearestIdx], b = tempimg[0], d1 = b - a, d2 = m - a;
                float n1 = (float)norm(d1), n2 = (float)norm(d2);
                if( n1*n2 < eps )
                    imgpt[npt] = a;
                else
                {
                    Z = d1.dot(d2)/(n1*n1);
                    imgpt[npt] = d1*Z + a;
                }
            }
            box[npt] = image2plane(imgpt[npt], R, tvec, cameraMatrix, npt<3 ? 0 : Z);

            if( (npt == 0 && mouse.event == EVENT_LBUTTONDOWN) ||
               (npt > 0 && norm(box[npt] - box[npt-1]) > eps &&
                mouse.event == EVENT_LBUTTONUP) )
            {
                nobjpt++;
                if( nobjpt < 4 )
                {
                    imgpt[nobjpt] = imgpt[nobjpt-1];
                    box[nobjpt] = box[nobjpt-1];
                }
            }

            // reset the event
            mouse.event = -1;
            //mouse.buttonState = 0;
            npt++;
        }

        frame.copyTo(shownFrame);
        extract3DBox(frame, shownFrame, selectedObjFrame,
                     cameraMatrix, rvec, tvec, box, npt, false);
        imshow(windowname, shownFrame);
        imshow(selWinName, selectedObjFrame);

        char c = (char)waitKey(30);
        if( c == 27 )
        {
            nobjpt = 0;
        }
        if( c == 'q' || c == 'Q' || c == ' ' )
        {
            box.clear();
            return c == ' ' ? -1 : -100;
        }
        if( (c == '\r' || c == '\n') && nobjpt == 4 && box[3].z != 0 )
            return 1;
    }
}


static bool readModelViews( const string& filename, vector<Point3f>& box,
                            vector<string>& imagelist,
                            vector<Rect>& roiList, vector<Vec6f>& poseList )
{
    imagelist.resize(0);
    roiList.resize(0);
    poseList.resize(0);
    box.resize(0);

    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    fs["box"] >> box;

    FileNode all = fs["views"];
    if( all.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = all.begin(), it_end = all.end();

    for(; it != it_end; ++it)
    {
        FileNode n = *it;
        imagelist.push_back((string)n["image"]);
        FileNode nr = n["rect"];
        roiList.push_back(Rect((int)nr[0], (int)nr[1], (int)nr[2], (int)nr[3]));
        FileNode np = n["pose"];
        poseList.push_back(Vec6f((float)np[0], (float)np[1], (float)np[2],
                                 (float)np[3], (float)np[4], (float)np[5]));
    }

    return true;
}


static bool writeModelViews(const string& filename, const vector<Point3f>& box,
                            const vector<string>& imagelist,
                            const vector<Rect>& roiList,
                            const vector<Vec6f>& poseList)
{
    FileStorage fs(filename, FileStorage::WRITE);
    if( !fs.isOpened() )
        return false;

    fs << "box" << "[:";
    fs << box << "]" << "views" << "[";

    size_t i, nviews = imagelist.size();

    CV_Assert( nviews == roiList.size() && nviews == poseList.size() );

    for( i = 0; i < nviews; i++ )
    {
        Rect r = roiList[i];
        Vec6f p = poseList[i];

        fs << "{" << "image" << imagelist[i] <<
            "roi" << "[:" << r.x << r.y << r.width << r.height << "]" <<
            "pose" << "[:" << p[0] << p[1] << p[2] << p[3] << p[4] << p[5] << "]" << "}";
    }
    fs << "]";

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


int main(int argc, char** argv)
{
    string help = string("Usage: ") + argv[0] + " -w=<board_width> -h=<board_height> [-s=<square_size>]\n" +
           "\t-i=<intrinsics_filename> -o=<output_prefix> [video_filename/cameraId]\n";
    const char* screen_help =
    "Actions: \n"
    "\tSelect object as 3D box with the mouse. That's it\n"
    "\tESC - Reset the selection\n"
    "\tSPACE - Skip the frame; move to the next frame (not in video mode)\n"
    "\tENTER - Confirm the selection. Grab next object in video mode.\n"
    "\tq - Exit the program\n";

    cv::CommandLineParser parser(argc, argv, "{help h||}{w||}{h||}{s|1|}{i||}{o||}{@input|0|}");
    if (parser.has("help"))
    {
        puts(helphelp(argv).c_str());
        puts(help.c_str());
        return 0;
    }
    string intrinsicsFilename;
    string outprefix = "";
    string inputName = "";
    int cameraId = 0;
    Size boardSize;
    double squareSize;
    vector<string> imageList;
    intrinsicsFilename = parser.get<string>("i");
    outprefix = parser.get<string>("o");
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    squareSize = parser.get<double>("s");
    if ( parser.get<string>("@input").size() == 1 && isdigit(parser.get<string>("@input")[0]) )
        cameraId = parser.get<int>("@input");
    else
        inputName = samples::findFileOrKeep(parser.get<string>("@input"));
    if (!parser.check())
    {
        puts(help.c_str());
        parser.printErrors();
        return 0;
    }
    if ( boardSize.width <= 0 )
    {
        printf("Incorrect -w parameter (must be a positive integer)\n");
        puts(help.c_str());
        return 0;
    }
    if ( boardSize.height <= 0 )
    {
        printf("Incorrect -h parameter (must be a positive integer)\n");
        puts(help.c_str());
        return 0;
    }
    if ( squareSize <= 0 )
    {
        printf("Incorrect -s parameter (must be a positive real number)\n");
        puts(help.c_str());
        return 0;
    }
    Mat cameraMatrix, distCoeffs;
    Size calibratedImageSize;
    readCameraMatrix(intrinsicsFilename, cameraMatrix, distCoeffs, calibratedImageSize );

    VideoCapture capture;
    if( !inputName.empty() )
    {
        if( !readStringList(inputName, imageList) &&
            !capture.open(inputName))
        {
            fprintf( stderr, "The input file could not be opened\n" );
            return -1;
        }
    }
    else
        capture.open(cameraId);

    if( !capture.isOpened() && imageList.empty() )
        return fprintf( stderr, "Could not initialize video capture\n" ), -2;

    const char* outbarename = 0;
    {
        outbarename = strrchr(outprefix.c_str(), '/');
        const char* tmp = strrchr(outprefix.c_str(), '\\');
        char cmd[1000];
        snprintf(cmd, sizeof(cmd), "mkdir %s", outprefix.c_str());
        if( tmp && tmp > outbarename )
            outbarename = tmp;
        if( outbarename )
        {
            cmd[6 + outbarename - outprefix.c_str()] = '\0';
            int result = system(cmd);
            CV_Assert(result == 0);
            outbarename++;
        }
        else
            outbarename = outprefix.c_str();
    }

    Mat frame, shownFrame, selectedObjFrame, mapxy;

    namedWindow("View", 1);
    namedWindow("Selected Object", 1);
    setMouseCallback("View", onMouse, 0);
    bool boardFound = false;

    string indexFilename = cv::format("%s_index.yml", outprefix.c_str());

    vector<string> capturedImgList;
    vector<Rect> roiList;
    vector<Vec6f> poseList;
    vector<Point3f> box, boardPoints;

    readModelViews(indexFilename, box, capturedImgList, roiList, poseList);
    calcChessboardCorners(boardSize, (float)squareSize, boardPoints);
    int frameIdx = 0;
    bool grabNext = !imageList.empty();

    puts(screen_help);

    for(int i = 0;;i++)
    {
        Mat frame0;
        if( !imageList.empty() )
        {
            if( i < (int)imageList.size() )
                frame0 = imread(string(imageList[i]), 1);
        }
        else
            capture >> frame0;
        if( frame0.empty() )
            break;
        if( frame.empty() )
        {
            if( frame0.size() != calibratedImageSize )
            {
                double sx = (double)frame0.cols/calibratedImageSize.width;
                double sy = (double)frame0.rows/calibratedImageSize.height;

                // adjust the camera matrix for the new resolution
                cameraMatrix.at<double>(0,0) *= sx;
                cameraMatrix.at<double>(0,2) *= sx;
                cameraMatrix.at<double>(1,1) *= sy;
                cameraMatrix.at<double>(1,2) *= sy;
            }
            Mat dummy;
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                    cameraMatrix, frame0.size(),
                                    CV_32FC2, mapxy, dummy );
            distCoeffs = Mat::zeros(5, 1, CV_64F);
        }
        remap(frame0, frame, mapxy, Mat(), INTER_LINEAR);
        vector<Point2f> foundBoardCorners;
        boardFound = findChessboardCorners(frame, boardSize, foundBoardCorners);

        Mat rvec, tvec;
        if( boardFound )
            solvePnP(Mat(boardPoints), Mat(foundBoardCorners), cameraMatrix,
                     distCoeffs, rvec, tvec, false);

        frame.copyTo(shownFrame);
        drawChessboardCorners(shownFrame, boardSize, Mat(foundBoardCorners), boardFound);
        selectedObjFrame = Mat::zeros(frame.size(), frame.type());

        if( boardFound && grabNext )
        {
            if( box.empty() )
            {
                int code = select3DBox("View", "Selected Object", frame,
                                        cameraMatrix, rvec, tvec, box);
                if( code == -100 )
                    break;
            }

            if( !box.empty() )
            {
                Rect r = extract3DBox(frame, shownFrame, selectedObjFrame,
                                      cameraMatrix, rvec, tvec, box, 4, true);
                if( !r.empty() )
                {
                    const int maxFrameIdx = 10000;
                    char path[1000];
                    for(;frameIdx < maxFrameIdx;frameIdx++)
                    {
                        snprintf(path, sizeof(path), "%s%04d.jpg", outprefix.c_str(), frameIdx);
                        FILE* f = fopen(path, "rb");
                        if( !f )
                            break;
                        fclose(f);
                    }
                    if( frameIdx == maxFrameIdx )
                    {
                        printf("Can not save the image as %s<...>.jpg", outprefix.c_str());
                        break;
                    }
                    imwrite(path, selectedObjFrame(r));

                    capturedImgList.push_back(string(path));
                    roiList.push_back(r);

                    float p[6];
                    Mat RV(3, 1, CV_32F, p), TV(3, 1, CV_32F, p+3);
                    rvec.convertTo(RV, RV.type());
                    tvec.convertTo(TV, TV.type());
                    poseList.push_back(Vec6f(p[0], p[1], p[2], p[3], p[4], p[5]));
                }
            }
            grabNext = !imageList.empty();
        }

        imshow("View", shownFrame);
        imshow("Selected Object", selectedObjFrame);
        char c = (char)waitKey(imageList.empty() && !box.empty() ? 30 : 300);
        if( c == 'q' || c == 'Q' )
            break;
        if( c == '\r' || c == '\n' )
            grabNext = true;
    }

    writeModelViews(indexFilename, box, capturedImgList, roiList, poseList);
    return 0;
}
