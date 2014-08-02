//
// Don't use as a standalone file
//

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp" // cv::format
#include "opencv2/imgproc.hpp" // cvtColor
#include "opencv2/imgproc/types_c.h" // cvtColor
#include "opencv2/highgui.hpp" // imread
#include "opencv2/core/directx.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <queue>

using namespace cv;
using namespace cv::directx;
static const int fontFace = cv::FONT_HERSHEY_DUPLEX;
#if !defined(USE_D3D9) && !defined(USE_D3DEX)
const cv::Scalar frameColor(255,128,0,255);
#else
const cv::Scalar frameColor(0,128,255,255); // BGRA for D3D9
#endif

#define SAFE_RELEASE(p) if (p) { p->Release(); p = NULL; }

const int WIDTH = 1024;
const int HEIGHT = 768;

HINSTANCE hInstance;
HWND hWnd;

// external declaration
bool initDirect3D(void);
bool initDirect3DTextures(void);
void render(void);
void cleanUp (void);

#define USAGE_DESCRIPTION_0 "1 - CPU write via LockRect/Map"
#define USAGE_DESCRIPTION_1 "2* - Mat->D3D"
#define USAGE_DESCRIPTION_2 "3* - D3D->UMat / change UMat / UMat->D3D"
#define USAGE_DESCRIPTION_3 "0 - show input texture without any processing"
#define USAGE_DESCRIPTION_SPACE "SPACE - toggle frame processing (only data transfers)"

static int g_sampleType = 0;
static int g_disableProcessing = false;

// forward declaration
static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

static bool initWindow()
{
    WNDCLASSEX wcex;

    wcex.cbSize             = sizeof(WNDCLASSEX);
    wcex.style              = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc        = WndProc;
    wcex.cbClsExtra         = 0;
    wcex.cbWndExtra         = 0;
    wcex.hInstance          = hInstance;
    wcex.hIcon              = LoadIcon(0, IDI_APPLICATION);
    wcex.hCursor            = LoadCursor(0, IDC_ARROW);
    wcex.hbrBackground      = 0;
    wcex.lpszMenuName       = 0L;
    wcex.lpszClassName      = "OpenCVDirectX";
    wcex.hIconSm            = 0;

    RegisterClassEx(&wcex);

    RECT rc = {0, 0, WIDTH, HEIGHT};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, false);
    hWnd = CreateWindow("OpenCVDirectX", WINDOW_NAME,
            WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, hInstance, NULL);

    if (!hWnd)
        return false;

    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    return true;
}

static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_CHAR:
            if (wParam >= '0' && wParam <= '3')
            {
                g_sampleType = (char)wParam - '0';
                return 0;
            }
            else if (wParam == ' ')
            {
                g_disableProcessing = !g_disableProcessing;
                return 0;
            }
            else if (wParam == VK_ESCAPE)
            {
                DestroyWindow(hWnd);
                return 0;
            }
            break;
    }
    return DefWindowProc(hWnd, message, wParam, lParam);
}

static float getFps()
{
    static std::queue<int64> time_queue;

    int64 now = cv::getTickCount(), then = 0;
    time_queue.push(now);

    if (time_queue.size() >= 2)
        then = time_queue.front();

    if (time_queue.size() >= 25)
        time_queue.pop();

    return time_queue.size() * (float)cv::getTickFrequency() / (now - then);
}

static int bgColor[4] = {0, 0, 0, 0};
static cv::Mat* inputMat = NULL;

static void renderToD3DObject(void)
{
    static int frame = 0;

    const float fps = getFps();

    String deviceName = cv::ocl::useOpenCL() ? cv::ocl::Context::getDefault().device(0).name() : "No OpenCL device";

    if ((frame % std::max(1, (int)(fps / 25))) == 0)
    {
        String msg = format("%s%s: %s, Sample %d, Frame %d, fps %g (%g ms)",
                g_disableProcessing ? "(FRAME PROCESSING DISABLED) " : "",
                WINDOW_NAME, deviceName.c_str(), g_sampleType,
                frame, fps, (int(10 * 1000.0 / fps)) * 0.1);
        SetWindowText(hWnd, msg.c_str());
    }

    // 0..255
    int c[4] =
    {
        std::abs((frame & 0x1ff) - 0x100),
        std::abs(((frame * 2) & 0x1ff) - 0x100),
        std::abs(((frame / 2) & 0x1ff) - 0x100),
        0
    };

    int c1 = c[0] / 2 - 0x40 - bgColor[0];
    int c2 = c[1] / 2 - 0x40 - bgColor[1];
    int c3 = c[2] / 2 - 0x40 - bgColor[2];

    switch (g_sampleType)
    {
    case 0:
#if defined(USE_D3D9) || defined (USE_D3DEX)
        if (FAILED(dev->StretchRect(pReadOnlySurface, NULL, pBackBuffer, NULL, D3DTEXF_NONE)))
        {
            std::cerr << "Can't StretchRect()" << std::endl;
        }
#elif defined(USE_D3D10)
        dev->CopyResource(pBackBufferTexture, pInputTexture);
#elif defined(USE_D3D11)
        devcon->CopyResource(pBackBufferTexture, pInputTexture);
#else
#error "Invalid USE_D3D value"
#endif
        break;

    case 1:
    {
        int BOXSIZE = 50;
        int x = std::abs(((frame * 1) % (2 * (WIDTH - BOXSIZE))) - (WIDTH - BOXSIZE));
        int y = std::abs(((frame / 2) % (2 * (HEIGHT - BOXSIZE))) - (HEIGHT - BOXSIZE));
        cv::Rect boxRect(x, y, BOXSIZE, BOXSIZE);
#if defined(USE_D3D9) || defined (USE_D3DEX)
        D3DLOCKED_RECT memDesc = {0, NULL};
        RECT rc = {0, 0, WIDTH, HEIGHT};
        if (SUCCEEDED(pCPUWriteSurface->LockRect(&memDesc, &rc, 0)))
        {
            if (!g_disableProcessing)
            {
                Mat m(Size(WIDTH, HEIGHT), CV_8UC4, memDesc.pBits, (int)memDesc.Pitch);
                inputMat->copyTo(m);
                m(boxRect).setTo(Scalar(c[0], c[1], c[2], 255));
            }
            pCPUWriteSurface->UnlockRect();
        }
        else
        {
            std::cerr << "Can't LockRect() on surface" << std::endl;
        }
#elif defined(USE_D3D10)
        D3D10_MAPPED_TEXTURE2D mappedTex;
        if (SUCCEEDED(pCPUWriteTexture->Map( D3D10CalcSubresource(0, 0, 1), D3D10_MAP_WRITE_DISCARD, 0, &mappedTex)))
        {
            if (!g_disableProcessing)
            {
                Mat m(Size(WIDTH, HEIGHT), CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);
                inputMat->copyTo(m);
                m(boxRect).setTo(Scalar(c[0], c[1], c[2], 255));
            }
            pCPUWriteTexture->Unmap(D3D10CalcSubresource(0, 0, 1));
            dev->CopyResource(pBackBufferTexture, pCPUWriteTexture);
        }
        else
        {
            std::cerr << "Can't Map() texture" << std::endl;
        }
#elif defined(USE_D3D11)
        D3D11_MAPPED_SUBRESOURCE mappedTex;
        if (SUCCEEDED(devcon->Map(pCPUWriteTexture, D3D11CalcSubresource(0, 0, 1), D3D11_MAP_WRITE_DISCARD, 0, &mappedTex)))
        {
            if (!g_disableProcessing)
            {
                Mat m(Size(WIDTH, HEIGHT), CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);
                inputMat->copyTo(m);
                m(boxRect).setTo(Scalar(c[0], c[1], c[2], 255));
            }
            devcon->Unmap(pCPUWriteTexture, D3D11CalcSubresource(0, 0, 1));
            devcon->CopyResource(pBackBufferTexture, pCPUWriteTexture);
        }
        else
        {
            std::cerr << "Can't Map() texture" << std::endl;
        }
#else
#error "Invalid USE_D3D value"
#endif
        break;
    }
    case 2:
    {
        static Mat m;
        if (!g_disableProcessing)
        {
#if 1
            cv::add(*inputMat, Scalar(c1, c2, c3, 255), m);
#else
            inputMat->copyTo(m);
#endif
            cv::putText(m,
                    cv::format("Frame %d, fps %g (%g ms)",
                            frame, fps, (int(10 * 1000.0 / fps)) * 0.1),
                    cv::Point(8, 80), fontFace, 1, frameColor, 2);
        }
        else
        {
            m.create(Size(WIDTH, HEIGHT), CV_8UC4);
        }
        try
        {
    #if defined(USE_D3D9) || defined (USE_D3DEX)
            convertToDirect3DSurface9(m, pSurface, (void*)surfaceShared);
    #elif defined(USE_D3D10)
            convertToD3D10Texture2D(m, pBackBufferTexture);
    #elif defined(USE_D3D11)
            convertToD3D11Texture2D(m, pBackBufferTexture);
    #else
    #error "Invalid USE_D3D value"
    #endif
        }
        catch (cv::Exception& e)
        {
            std::cerr << "Can't convert to D3D object: exception: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Can't convert to D3D object" << std::endl;
        }
        break;
    }
    case 3:
    {
        static UMat tmp;
        try
        {
#if defined(USE_D3D9) || defined (USE_D3DEX)
            convertFromDirect3DSurface9(pReadOnlySurface, tmp, (void*)readOnlySurfaceShared);
#elif defined(USE_D3D10)
            convertFromD3D10Texture2D(pInputTexture, tmp);
#elif defined(USE_D3D11)
            convertFromD3D11Texture2D(pInputTexture, tmp);
#else
#error "Invalid USE_D3D value"
#endif
        }
        catch (cv::Exception& e)
        {
            std::cerr << "Can't convert from D3D object: exception: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Can't convert from D3D object" << std::endl;
        }
        static UMat res;
        if (!g_disableProcessing)
        {
            cv::add(tmp, Scalar(c1, c2, c3, 255), res);
        }
        else
        {
            res = tmp;
        }
        try
        {
#if defined(USE_D3D9) || defined (USE_D3DEX)
            convertToDirect3DSurface9(res, pSurface, (void*)surfaceShared);
#elif defined(USE_D3D10)
            convertToD3D10Texture2D(res, pBackBufferTexture);
#elif defined(USE_D3D11)
            convertToD3D11Texture2D(res, pBackBufferTexture);
#else
#error "Invalid USE_D3D value"
#endif
        }
        catch (cv::Exception& e)
        {
            std::cerr << "Can't convert to D3D object: exception: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Can't convert to D3D object" << std::endl;
        }
        break;
    }
    }
    frame++;
}


static cv::Mat getInputTexture()
{
    cv::Mat inputMat = cv::imread("input.bmp", cv::IMREAD_COLOR);

    if (inputMat.depth() != CV_8U)
    {
        inputMat.convertTo(inputMat, CV_8U);
    }
    if (inputMat.type() == CV_8UC3)
    {
        cv::cvtColor(inputMat, inputMat, CV_RGB2BGRA);
    }
    if (inputMat.type() != CV_8UC4 || inputMat.size().area() == 0)
    {
        std::cerr << "Invalid input image format. Generate other" << std::endl;
        inputMat.create(cv::Size(WIDTH, HEIGHT), CV_8UC4);
        inputMat.setTo(cv::Scalar(0, 0, 255, 255));
        bgColor[0] = -128; bgColor[1] = -128; bgColor[2] = 127; bgColor[3] = -128;
    }
    if (inputMat.size().width != WIDTH || inputMat.size().height != HEIGHT)
    {
        cv::resize(inputMat, inputMat, cv::Size(WIDTH, HEIGHT));
    }
    String deviceName = cv::ocl::useOpenCL() ? cv::ocl::Context::getDefault().device(0).name() : "No OpenCL device";
    cv::Scalar color(64, 255, 64, 255);
    cv::putText(inputMat,
            cv::format("OpenCL Device name: %s", deviceName.c_str()),
            cv::Point(8,32), fontFace, 1, color);
    cv::putText(inputMat, WINDOW_NAME, cv::Point(50, HEIGHT - 32), fontFace, 1, color);
    cv::putText(inputMat, USAGE_DESCRIPTION_0, cv::Point(30, 128), fontFace, 1, color);
    cv::putText(inputMat, USAGE_DESCRIPTION_1, cv::Point(30, 192), fontFace, 1, color);
    cv::putText(inputMat, USAGE_DESCRIPTION_2, cv::Point(30, 256), fontFace, 1, color);
    cv::putText(inputMat, USAGE_DESCRIPTION_3, cv::Point(30, 320), fontFace, 1, color);
    cv::putText(inputMat, USAGE_DESCRIPTION_SPACE, cv::Point(30, 448), fontFace, 1, color);

#if defined(USE_D3D9) || defined (USE_D3DEX)
    cv::cvtColor(inputMat, inputMat, CV_RGBA2BGRA);
    std::swap(bgColor[0], bgColor[2]);
#endif

    // Make a global copy
    ::inputMat = new cv::Mat(inputMat);

    return inputMat;
}

static int mainLoop()
{
    hInstance = GetModuleHandle(NULL);

    if (!initWindow())
        CV_Error(cv::Error::StsError, "Can't create window");

    if (!initDirect3D())
        CV_Error(cv::Error::StsError, "Can't create D3D object");

    if (cv::ocl::haveOpenCL())
    {
#if defined(USE_D3D9)
        cv::ocl::Context& ctx = cv::directx::ocl::initializeContextFromDirect3DDevice9(dev);
#elif defined (USE_D3DEX)
        cv::ocl::Context& ctx = cv::directx::ocl::initializeContextFromDirect3DDevice9Ex(dev);
#elif defined(USE_D3D10)
        cv::ocl::Context& ctx = cv::directx::ocl::initializeContextFromD3D10Device(dev);
#elif defined(USE_D3D11)
        cv::ocl::Context& ctx = cv::directx::ocl::initializeContextFromD3D11Device(dev);
#else
#error "Invalid USE_D3D value"
#endif
        std::cout << "Selected device: " << ctx.device(0).name().c_str() << std::endl;
        g_sampleType = 2;
    }
    else
    {
        std::cerr << "OpenCL is not available. DirectX - OpenCL interop will not work" << std::endl;
    }

    if (!initDirect3DTextures())
        CV_Error(cv::Error::StsError, "Can't create D3D texture object");

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));

    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            render();
        }
    }

    cleanUp();

    return static_cast<int>(msg.wParam);
}

int main(int /*argc*/, char ** /*argv*/)
{
    try
    {
        return mainLoop();
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
