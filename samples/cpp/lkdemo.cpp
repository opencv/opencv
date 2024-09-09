#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include <cstdlib>  // 为 system() 函数包含的头文件

using namespace cv;
using namespace std;

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses a video file as input, which should be provided as a command-line argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}

int main( int argc, char** argv )
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;

    help();

    string input = argv[1];
    cap.open(input);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    // 创建子目录 lkdemo
    string outputDir = "lkdemo";
    system(("mkdir -p " + outputDir).c_str());  // 使用 system() 创建目录

    // 创建保存文件的路径
    string outputFilePath = outputDir + "/output.avi";
    cout << "Processed video will be saved at: " << outputFilePath << endl;

    // 设置视频写入器
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter writer(outputFilePath, codec, cap.get(CAP_PROP_FPS), 
                       Size((int)cap.get(CAP_PROP_FRAME_WIDTH), 
                            (int)cap.get(CAP_PROP_FRAME_HEIGHT)));

    Mat gray, prevGray, image, frame;
    vector<Point2f> points[2];

    for(;;)
    {
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        if( needToInit )
        {
            // automatic initialization
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            addRemovePt = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( addRemovePt )
                {
                    if( norm(point - points[1][i]) <= 5 )
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if( !status[i] )
                    continue;

                points[1][k++] = points[1][i];
                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
        }

        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        needToInit = false;

        // 注释掉图形显示部分
        // imshow("LK Demo", image);

        // 将处理后的帧写入视频文件
        writer.write(image);

        // char c = (char)waitKey(10);
        // if( c == 27 )
        //     break;
        // switch( c )
        // {
        // case 'r':
        //     needToInit = true;
        //     break;
        // case 'c':
        //     points[0].clear();
        //     points[1].clear();
        //     break;
        // case 'n':
        //     nightMode = !nightMode;
        //     break;
        // }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }

    return 0;
}

