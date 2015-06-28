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
        MODE_NOP,
        MODE_CPU,
        MODE_GPU
    };

    D3DSample(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        WinApp(width, height, window_name)
    {
        m_shutdown          = false;
        m_mode              = MODE_NOP;
        m_modeStr[0]        = cv::String("No processing");
        m_modeStr[1]        = cv::String("Processing on CPU");
        m_modeStr[2]        = cv::String("Processing on GPU");
        m_disableProcessing = false;
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

    static float getFps()
    {
        static std::queue<int64> time_queue;

        int64 now = cv::getTickCount();
        int64 then = 0;
        time_queue.push(now);

        if (time_queue.size() >= 2)
            then = time_queue.front();

        if (time_queue.size() >= 25)
            time_queue.pop();

        size_t sz = time_queue.size();

        float fps = sz * (float)cv::getTickFrequency() / (now - then);

        return fps;
    }

protected:
    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        switch (message)
        {
        case WM_CHAR:
            if (wParam >= '0' && wParam <= '2')
            {
                m_mode = static_cast<MODE>((char)wParam - '0');
                return 0;
            }
            else if (wParam == VK_SPACE)
            {
                m_disableProcessing = !m_disableProcessing;
                return 0;
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
            return 0;
        }

        return ::DefWindowProc(hWnd, message, wParam, lParam);
    }

    // do render at idle
    virtual int idle() { return render(); }

protected:
    bool               m_shutdown;
    bool               m_disableProcessing;
    MODE               m_mode;
    cv::String         m_modeStr[3];
    cv::VideoCapture   m_cap;
    cv::Mat            m_frame_bgr;
    cv::Mat            m_frame_rgba;
};


static void help()
{
    printf(
        "\nSample demonstrating interoperability of DirectX and OpenCL with OpenCV.\n"
        "Hot keys: \n"
        "    0 - no processing\n"
        "    1 - blur DX surface on CPU through OpenCV\n"
        "    2 - blur DX surface on GPU through OpenCV using OpenCL\n"
        "  ESC - exit\n\n");
}


static const char* keys =
{
    "{c camera | true  | use camera or not}"
    "{f file   |       | movie file name  }"
    "{h help   | false | print help info  }"
};


template <typename TApp>
int d3d_app(int argc, char** argv, std::string& title)
{
    cv::CommandLineParser parser(argc, argv, keys); \
    bool   useCamera = parser.has("camera"); \
    string file      = parser.get<string>("file"); \
    bool   showHelp  = parser.get<bool>("help"); \

    if (showHelp)
        help();

    parser.printMessage();

    cv::VideoCapture cap;

    if (useCamera)
        cap.open(0);
    else
        cap.open(file.c_str());

    if (!cap.isOpened())
    {
        printf("can not open camera or video file\n");
        return -1;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

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
