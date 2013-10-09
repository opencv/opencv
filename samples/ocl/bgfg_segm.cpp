#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ocl.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

#define M_MOG  1
#define M_MOG2 2

int main(int argc, const char** argv)
{

    cv::CommandLineParser cmd(argc, argv,
        "{ c camera | false       | use camera }"
        "{ f file   | 768x576.avi | input video file }"
        "{ m method | mog         | method (mog, mog2) }"
        "{ h help   | false       | print help message }");

    if (cmd.get<bool>("help"))
    {
        cout << "Usage : bgfg_segm [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return 0;
    }

    bool useCamera = cmd.get<bool>("camera");
    string file = cmd.get<string>("file");
    string method = cmd.get<string>("method");

    if (method != "mog" && method != "mog2")
    {
        cerr << "Incorrect method" << endl;
        return -1;
    }

    int m = method == "mog" ? M_MOG : M_MOG2;

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

    oclMat d_frame(frame);

    cv::ocl::MOG mog;
    cv::ocl::MOG2 mog2;

    oclMat d_fgmask;
    oclMat d_fgimg;
    oclMat d_bgimg;

    d_fgimg.create(d_frame.size(), d_frame.type());

    Mat fgmask;
    Mat fgimg;
    Mat bgimg;

    switch (m)
    {
    case M_MOG:
        mog(d_frame, d_fgmask, 0.01f);
        break;

    case M_MOG2:
        mog2(d_frame, d_fgmask);
        break;
    }

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
        case M_MOG:
            mog(d_frame, d_fgmask, 0.01f);
            mog.getBackgroundImage(d_bgimg);
            break;

        case M_MOG2:
            mog2(d_frame, d_fgmask);
            mog2.getBackgroundImage(d_bgimg);
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;

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

        int key = waitKey(30);
        if (key == 27)
            break;
    }

    return 0;
}
