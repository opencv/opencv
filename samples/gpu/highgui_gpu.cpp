#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/core/opengl_interop.hpp"
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

        const string openGlMatWnd = "OpenGL Mat";
        const string openGlBufferWnd = "OpenGL GlBuffer";
        const string openGlTextureWnd = "OpenGL GlTexture";
        const string openGlGpuMatWnd = "OpenGL GpuMat";
        const string matWnd = "Mat";

        namedWindow(openGlMatWnd, WINDOW_OPENGL | WINDOW_AUTOSIZE);
        namedWindow(openGlBufferWnd, WINDOW_OPENGL | WINDOW_AUTOSIZE);
        namedWindow(openGlTextureWnd, WINDOW_OPENGL | WINDOW_AUTOSIZE);
        if (haveCuda)
            namedWindow(openGlGpuMatWnd, WINDOW_OPENGL | WINDOW_AUTOSIZE);
        namedWindow("Mat", WINDOW_AUTOSIZE);

        Mat img = imread(argv[1]);
        
        if (haveCuda)
            setGlDevice();

        setOpenGlContext(openGlBufferWnd);
        GlBuffer buf(img, GlBuffer::TEXTURE_BUFFER);

        setOpenGlContext(openGlTextureWnd);
        GlTexture tex(img);
        
        GpuMat d_img;
        if (haveCuda)
            d_img.upload(img);
            
        cout << "=== First call\n\n";

        {
            Timer t("OpenGL Mat      ");
            imshow(openGlMatWnd, img);
        }
        {
            Timer t("OpenGL GlBuffer ");
            imshow(openGlBufferWnd, buf);
        }
        {
            Timer t("OpenGL GlTexture");
            imshow(openGlTextureWnd, tex);
        }
        if (haveCuda)
        {
            Timer t("OpenGL GpuMat   ");
            imshow(openGlGpuMatWnd, d_img);
        }
        {
            Timer t("Mat             ");
            imshow(matWnd, img);
        }

        waitKey();

        cout << "\n=== Second call\n\n";   

        {
            Timer t("OpenGL Mat      ");
            imshow(openGlMatWnd, img);
        }
        {
            Timer t("OpenGL GlBuffer ");
            imshow(openGlBufferWnd, buf);
        }
        {
            Timer t("OpenGL GlTexture");
            imshow(openGlTextureWnd, tex);
        }
        if (haveCuda)
        {
            Timer t("OpenGL GpuMat   ");
            imshow(openGlGpuMatWnd, d_img);
        }
        {
            Timer t("Mat             ");
            imshow(matWnd, img);
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
