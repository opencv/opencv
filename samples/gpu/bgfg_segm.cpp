#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

enum Method
{
    MOG,
    MOG2,
};

int main(int argc, const char** argv)
{
    cv::CommandLineParser cmd(argc, argv,
        "{ c camera |                    | use camera }"
        "{ f file   | ../data/vtest.avi  | input video file }"
        "{ m method | mog                | method (mog, mog2) }"
        "{ h help   |                    | print help message }");

    if (cmd.has("help") || !cmd.check())
    {
        cmd.printMessage();
        cmd.printErrors();
        return 0;
    }

    bool useCamera = cmd.has("camera");
    string file = cmd.get<string>("file");
    string method = cmd.get<string>("method");

    if (method != "mog"
        && method != "mog2")
    {
        cerr << "Incorrect method" << endl;
        return -1;
    }

    Method m = method == "mog" ? MOG :
               method == "mog2" ? MOG2 :
                                  (Method)-1;
    CV_Assert(m != (Method)-1);

    VideoCapture cap;

    if (useCamera)
        cap.open(0);
    else
        cap.open(file);

    if (!cap.isOpened())
    {
        cerr << "can not open camera or video file" << endl;
        return -1;
    }

    Mat frame;
    cap >> frame;

    GpuMat d_frame(frame);

    Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG();
    Ptr<BackgroundSubtractor> mog2 = cuda::createBackgroundSubtractorMOG2();

    GpuMat d_fgmask;
    GpuMat d_fgimg;
    GpuMat d_bgimg;

    Mat fgmask;
    Mat fgimg;
    Mat bgimg;

    switch (m)
    {
    case MOG:
        mog->apply(d_frame, d_fgmask, 0.01);
        break;

    case MOG2:
        mog2->apply(d_frame, d_fgmask);
        break;
    }

    namedWindow("image", WINDOW_NORMAL);
    namedWindow("foreground mask", WINDOW_NORMAL);
    namedWindow("foreground image", WINDOW_NORMAL);
    namedWindow("mean background image", WINDOW_NORMAL);

    for(;;)
    {
        cap >> frame;
        if (frame.empty())
            break;
        d_frame.upload(frame);

        int64 start = cv::getTickCount();

        //update the model
        switch (m)
        {
        case MOG:
            mog->apply(d_frame, d_fgmask, 0.01);
            mog->getBackgroundImage(d_bgimg);
            break;

        case MOG2:
            mog2->apply(d_frame, d_fgmask);
            mog2->getBackgroundImage(d_bgimg);
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;

        d_fgimg.create(d_frame.size(), d_frame.type());
        d_fgimg.setTo(Scalar::all(0));
        d_frame.copyTo(d_fgimg, d_fgmask);

        d_fgmask.download(fgmask);
        d_fgimg.download(fgimg);
        if (!d_bgimg.empty())
            d_bgimg.download(bgimg);

        imshow("image", frame);
        imshow("foreground mask", fgmask);
        imshow("foreground image", fgimg);
        if (!bgimg.empty())
            imshow("mean background image", bgimg);

        char key = (char)waitKey(30);
        if (key == 27)
            break;
    }

    return 0;
}
