#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct Timer
{
    Timer(const string& msg_)
    {
        msg = msg_;

        tm.reset();
        tm.start();
    }

    ~Timer()
    {
        tm.stop();
        cout << msg << " " << tm.getTimeMilli() << " ms\n";
    }

    string msg;
    TickMeter tm;
};

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " image" << endl;
        return -1;
    }

    try
    {
        bool haveCuda = getCudaEnabledDeviceCount() > 0;

        namedWindow("OpenGL Mat", WINDOW_OPENGL | WINDOW_AUTOSIZE);
        namedWindow("OpenGL GlBuffer", WINDOW_OPENGL | WINDOW_AUTOSIZE);
        namedWindow("OpenGL GlTexture", WINDOW_OPENGL | WINDOW_AUTOSIZE);
        if (haveCuda)
            namedWindow("OpenGL GpuMat", WINDOW_OPENGL | WINDOW_AUTOSIZE);
        namedWindow("Mat", WINDOW_AUTOSIZE);

        Mat img = imread(argv[1]);
        
        if (haveCuda)
            setGlDevice();

        setOpenGlContext("OpenGL GlBuffer");
        GlBuffer buf(img, GlBuffer::TEXTURE_BUFFER);

        setOpenGlContext("OpenGL GlTexture");
        GlTexture tex(img);
        
        GpuMat d_img;
        if (haveCuda)
            d_img.upload(img);

        cout << "=== First call\n\n";

        {
            Timer t("OpenGL Mat      ");
            imshow("OpenGL Mat", img);
        }
        {
            Timer t("OpenGL GlBuffer ");
            imshow("OpenGL GlBuffer", buf);
        }
        {
            Timer t("OpenGL GlTexture");
            imshow("OpenGL GlTexture", tex);
        }
        if (haveCuda)
        {
            Timer t("OpenGL GpuMat   ");
            imshow("OpenGL GpuMat", d_img);
        }
        {
            Timer t("Mat             ");
            imshow("Mat", img);
        }

        cout << "\n=== Second call\n\n";   

        {
            Timer t("OpenGL Mat      ");
            imshow("OpenGL Mat", img);
        }
        {
            Timer t("OpenGL GlBuffer ");
            imshow("OpenGL GlBuffer", buf);
        }
        {
            Timer t("OpenGL GlTexture");
            imshow("OpenGL GlTexture", tex);
        }
        if (haveCuda)
        {
            Timer t("OpenGL GpuMat   ");
            imshow("OpenGL GpuMat", d_img);
        }
        {
            Timer t("Mat             ");
            imshow("Mat", img);
        }

        cout << "\n";

        waitKey();
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;
    }

    return 0;
}