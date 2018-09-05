/*
// Sample demonstrating interoperability of OpenCV UMat with Direct X surface
// Base class for Direct X application
*/
#include <string>
#include <iostream>
#include <queue>

#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "winapp.hpp"

#define SAFE_RELEASE(p) if (p) { p->Release(); p = NULL; }


class D3DSample : public WinApp
{
public:
    enum MODE
    {
        MODE_CPU,
        MODE_GPU_RGBA,
        MODE_GPU_NV12
    };

    D3DSample(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        WinApp(width, height, window_name)
    {
        m_shutdown          = false;
        m_mode              = MODE_CPU;
        m_modeStr[0]        = cv::String("Processing on CPU");
        m_modeStr[1]        = cv::String("Processing on GPU RGBA");
        m_modeStr[2]        = cv::String("Processing on GPU NV12");
        m_demo_processing   = false;
        m_cap               = cap;
    }

    ~D3DSample() {}

    virtual int create() { return WinApp::create(); }
    virtual int render() = 0;
    virtual int cleanup()
    {
        m_shutdown = true;
        return WinApp::cleanup();
    }

protected:
    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        switch (message)
        {
        case WM_CHAR:
            if (wParam == '1')
            {
                m_mode = MODE_CPU;
                return EXIT_SUCCESS;
            }
            if (wParam == '2')
            {
                m_mode = MODE_GPU_RGBA;
                return EXIT_SUCCESS;
            }
            if (wParam == '3')
            {
                m_mode = MODE_GPU_NV12;
                return EXIT_SUCCESS;
            }
            else if (wParam == VK_SPACE)
            {
                m_demo_processing = !m_demo_processing;
                return EXIT_SUCCESS;
            }
            else if (wParam == VK_ESCAPE)
            {
                return cleanup();
            }
            break;

        case WM_CLOSE:
            return cleanup();

        case WM_DESTROY:
            ::PostQuitMessage(0);
            return EXIT_SUCCESS;
        }

        return ::DefWindowProc(hWnd, message, wParam, lParam);
    }

    // do render at idle
    virtual int idle() { return render(); }

protected:
    bool               m_shutdown;
    bool               m_demo_processing;
    MODE               m_mode;
    cv::String         m_modeStr[3];
    cv::VideoCapture   m_cap;
    cv::Mat            m_frame_bgr;
    cv::Mat            m_frame_rgba;
    cv::TickMeter      m_timer;
};


static const char* keys =
{
    "{c camera | 0     | camera id  }"
    "{f file   |       | movie file name  }"
};


template <typename TApp>
int d3d_app(int argc, char** argv, std::string& title)
{
    cv::CommandLineParser parser(argc, argv, keys);
    std::string file = parser.get<std::string>("file");
    int    camera_id = parser.get<int>("camera");

    parser.about(
        "\nA sample program demonstrating interoperability of DirectX and OpenCL with OpenCV.\n\n"
        "Hot keys: \n"
        "  SPACE - turn processing on/off\n"
        "    1   - process DX surface through OpenCV on CPU\n"
        "    2   - process DX RGBA surface through OpenCV on GPU (via OpenCL)\n"
        "    3   - process DX NV12 surface through OpenCV on GPU (via OpenCL)\n"
        "   ESC  - exit\n\n");

    parser.printMessage();

    cv::VideoCapture cap;

    if (file.empty())
        cap.open(camera_id);
    else
        cap.open(file.c_str());

    if (!cap.isOpened())
    {
        printf("can not open camera or video file\n");
        return EXIT_FAILURE;
    }

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::string wndname = title;

    TApp app(width, height, wndname, cap);

    try
    {
        app.create();
        return app.run();
    }

    catch (cv::Exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 10;
    }

    catch (...)
    {
        std::cerr << "FATAL ERROR: Unknown exception" << std::endl;
        return 11;
    }
}
