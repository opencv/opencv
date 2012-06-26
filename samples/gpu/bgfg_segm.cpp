#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

enum Method
{
    FGD_STAT,
    MOG,
    MOG2,
    VIBE
};

int main(int argc, const char** argv)
{
    cv::CommandLineParser cmd(argc, argv,
        "{ c | camera | false       | use camera }"
        "{ f | file   | 768x576.avi | input video file }"
        "{ m | method | mog         | method (fgd_stat, mog, mog2, vibe) }"
        "{ h | help   | false       | print help message }");

    if (cmd.get<bool>("help"))
    {
        cout << "Usage : bgfg_segm [options]" << endl;
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    bool useCamera = cmd.get<bool>("camera");
    string file = cmd.get<string>("file");
    string method = cmd.get<string>("method");

    if (method != "fgd_stat" && method != "mog" && method != "mog2" && method != "vibe")
    {
        cerr << "Incorrect method" << endl;
        return -1;
    }

    Method m = method == "fgd_stat" ? FGD_STAT : method == "mog" ? MOG : method == "mog2" ? MOG2 : VIBE;

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

    FGDStatModel fgd_stat;
    MOG_GPU mog;
    MOG2_GPU mog2;
    VIBE_GPU vibe;

    GpuMat d_fgmask;
    GpuMat d_fgimg;
    GpuMat d_bgimg;

    Mat fgmask;
    Mat fgimg;
    Mat bgimg;

    switch (m)
    {
    case FGD_STAT:
        fgd_stat.create(d_frame);
        break;

    case MOG:
        mog(d_frame, d_fgmask, 0.01);
        break;

    case MOG2:
        mog2(d_frame, d_fgmask);
        break;

    case VIBE:
        vibe.initialize(d_frame);
        break;
    }

    namedWindow("image", WINDOW_NORMAL);
    namedWindow("foreground mask", WINDOW_NORMAL);
    namedWindow("foreground image", WINDOW_NORMAL);
    if (m != VIBE)
        namedWindow("mean background image", WINDOW_NORMAL);

    for(;;)
    {
        cap >> frame;
        if (frame.empty())
            break;
        d_frame.upload(frame);

        //update the model
        switch (m)
        {
        case FGD_STAT:
            fgd_stat.update(d_frame);
            d_fgmask = fgd_stat.foreground;
            d_bgimg = fgd_stat.background;
            break;

        case MOG:
            mog(d_frame, d_fgmask, 0.01);
            mog.getBackgroundImage(d_bgimg);
            break;

        case MOG2:
            mog2(d_frame, d_fgmask);
            mog2.getBackgroundImage(d_bgimg);
            break;

        case VIBE:
            vibe(d_frame, d_fgmask);
            break;
        }

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

        char key = waitKey(30);
        if (key == 27)
            break;
    }

    return 0;
}
