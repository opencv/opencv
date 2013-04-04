#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/gpu.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

enum Method
{
    FGD_STAT,
    MOG,
    MOG2,
#ifdef HAVE_OPENCV_NONFREE
    VIBE,
#endif
    GMG
};

int main(int argc, const char** argv)
{
    cv::CommandLineParser cmd(argc, argv,
        "{ c camera |             | use camera }"
        "{ f file   | 768x576.avi | input video file }"
        "{ m method | mog         | method (fgd, mog, mog2, vibe, gmg) }"
        "{ h help   |             | print help message }");

    if (cmd.has("help") || !cmd.check())
    {
        cmd.printMessage();
        cmd.printErrors();
        return 0;
    }

    bool useCamera = cmd.has("camera");
    string file = cmd.get<string>("file");
    string method = cmd.get<string>("method");

    if (method != "fgd"
        && method != "mog"
        && method != "mog2"
    #ifdef HAVE_OPENCV_NONFREE
        && method != "vibe"
    #endif
        && method != "gmg")
    {
        cerr << "Incorrect method" << endl;
        return -1;
    }

    Method m = method == "fgd" ? FGD_STAT :
               method == "mog" ? MOG :
               method == "mog2" ? MOG2 :
            #ifdef HAVE_OPENCV_NONFREE
               method == "vibe" ? VIBE :
            #endif
                                  GMG;

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
#ifdef HAVE_OPENCV_NONFREE
    VIBE_GPU vibe;
#endif
    GMG_GPU gmg;
    gmg.numInitializationFrames = 40;

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
        mog(d_frame, d_fgmask, 0.01f);
        break;

    case MOG2:
        mog2(d_frame, d_fgmask);
        break;

#ifdef HAVE_OPENCV_NONFREE
    case VIBE:
        vibe.initialize(d_frame);
        break;
#endif

    case GMG:
        gmg.initialize(d_frame.size());
        break;
    }

    namedWindow("image", WINDOW_NORMAL);
    namedWindow("foreground mask", WINDOW_NORMAL);
    namedWindow("foreground image", WINDOW_NORMAL);
    if (m != GMG
    #ifdef HAVE_OPENCV_NONFREE
        && m != VIBE
    #endif
        )
    {
        namedWindow("mean background image", WINDOW_NORMAL);
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
        case FGD_STAT:
            fgd_stat.update(d_frame);
            d_fgmask = fgd_stat.foreground;
            d_bgimg = fgd_stat.background;
            break;

        case MOG:
            mog(d_frame, d_fgmask, 0.01f);
            mog.getBackgroundImage(d_bgimg);
            break;

        case MOG2:
            mog2(d_frame, d_fgmask);
            mog2.getBackgroundImage(d_bgimg);
            break;

#ifdef HAVE_OPENCV_NONFREE
        case VIBE:
            vibe(d_frame, d_fgmask);
            break;
#endif

        case GMG:
            gmg(d_frame, d_fgmask);
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
