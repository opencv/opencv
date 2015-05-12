#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>

#include <string>
#include <iostream>
#include <queue>

#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "winapp.hpp"

#pragma comment (lib, "d3d11.lib")


class D3D11WinApp : public WinApp
{
public:
    D3D11WinApp(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        WinApp(width, height, window_name)
    {
        m_shutdown = false;
        m_mode = 0;
        m_modeStr[0] = cv::String("No processing");
        m_modeStr[1] = cv::String("Processing on CPU");
        m_modeStr[2] = cv::String("Processing on GPU");
        m_disableProcessing = false;
        m_cap = cap;
    }

    ~D3D11WinApp() {}

    int onClose(void)
    {
        m_shutdown = true;
        cleanup();
        ::DestroyWindow(m_hWnd);
        return 0;
    }

    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        switch (message)
        {
        case WM_CHAR:
            if (wParam >= '0' && wParam <= '2')
            {
                m_mode = (char)wParam - '0';
                return 0;
            }
            else if (wParam == VK_SPACE)
            {
                m_disableProcessing = !m_disableProcessing;
                return 0;
            }
            else if (wParam == VK_ESCAPE)
            {
                return onClose();
            }
            break;

        case WM_CLOSE:
            return onClose();

        case WM_DESTROY:
            ::PostQuitMessage(0);
            return 0;
        }

        return ::DefWindowProc(hWnd, message, wParam, lParam);
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

        return time_queue.size() * (float)cv::getTickFrequency() / (now - then);
    }

    int init(void)
    {
        HRESULT r;

        DXGI_SWAP_CHAIN_DESC scd;

        ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));

        scd.BufferCount       = 1;                               // one back buffer
        scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;      // use 32-bit color
        scd.BufferDesc.Width  = m_width;                         // set the back buffer width
        scd.BufferDesc.Height = m_height;                        // set the back buffer height
        scd.BufferUsage       = DXGI_USAGE_RENDER_TARGET_OUTPUT; // how swap chain is to be used
        scd.OutputWindow      = m_hWnd;                          // the window to be used
        scd.SampleDesc.Count  = 1;                               // how many multisamples
        scd.Windowed          = TRUE;                            // windowed/full-screen mode
        scd.SwapEffect        = DXGI_SWAP_EFFECT_DISCARD;
        scd.Flags             = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH; // allow full-screen switching

        r = ::D3D11CreateDeviceAndSwapChain(
                NULL,
                D3D_DRIVER_TYPE_HARDWARE,
                NULL,
                0,
                NULL,
                0,
                D3D11_SDK_VERSION,
                &scd,
                &m_pD3D11SwapChain,
                &m_pD3D11Dev,
                NULL,
                &m_pD3D11Ctx);
        if (FAILED(r))
        {
            return -1;
        }

        r = m_pD3D11SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&m_pBackBuffer);
        if (FAILED(r))
        {
            return -1;
        }

        r = m_pD3D11Dev->CreateRenderTargetView(m_pBackBuffer, NULL, &m_pRenderTarget);
        if (FAILED(r))
        {
            return -1;
        }

        m_pD3D11Ctx->OMSetRenderTargets(1, &m_pRenderTarget, NULL);

        D3D11_VIEWPORT viewport;
        ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

        viewport.Width    = (float)m_width;
        viewport.Height   = (float)m_height;
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 0.0f;

        m_pD3D11Ctx->RSSetViewports(1, &viewport);

        D3D11_TEXTURE2D_DESC desc = { 0 };

        desc.Width            = m_width;
        desc.Height           = m_height;
        desc.MipLevels        = 1;
        desc.ArraySize        = 1;
        desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
        desc.Usage            = D3D11_USAGE_DYNAMIC;
        desc.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;

        r = m_pD3D11Dev->CreateTexture2D(&desc, NULL, &m_pSurface);
        if (FAILED(r))
        {
            std::cerr << "Can't create texture with input image" << std::endl;
            return -1;
        }

        if (cv::ocl::haveOpenCL())
        {
            m_oclCtx = cv::directx::ocl::initializeContextFromD3D11Device(m_pD3D11Dev);
        }

        m_oclDevName = cv::ocl::useOpenCL() ?
            cv::ocl::Context::getDefault().device(0).name() :
            "No OpenCL device";

        return 0;
    } // init()


    int get_surface(ID3D11Texture2D** ppSurface)
    {
        HRESULT r;

        if (!m_cap.read(m_frame_bgr))
            return -1;

        cv::cvtColor(m_frame_bgr, m_frame_rgba, CV_RGB2BGRA);

        UINT subResource = ::D3D11CalcSubresource(0, 0, 1);

        D3D11_MAPPED_SUBRESOURCE mappedTex;
        r = m_pD3D11Ctx->Map(m_pSurface, subResource, D3D11_MAP_WRITE_DISCARD, 0, &mappedTex);
        if (FAILED(r))
        {
            return r;
        }

        cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);
        // copy video frame data to surface
        m_frame_rgba.copyTo(m);

        m_pD3D11Ctx->Unmap(m_pSurface, subResource);

        *ppSurface = m_pSurface;

        return 0;
    }


    void print_info(ID3D11Texture2D* pSurface, int mode, float fps, cv::String oclDevName)
    {
        HRESULT r;

        UINT subResource = ::D3D11CalcSubresource(0, 0, 1);

        D3D11_MAPPED_SUBRESOURCE mappedTex;
        r = m_pD3D11Ctx->Map(pSurface, subResource, D3D11_MAP_WRITE_DISCARD, 0, &mappedTex);
        if (FAILED(r))
        {
            return;
        }

        cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);

        cv::String strMode = cv::format("%s", m_modeStr[mode].c_str());
        cv::String strFPS = cv::format("%2.1f", fps);
        cv::String strDevName = cv::format("%s", oclDevName.c_str());

        cv::putText(m, strMode, cv::Point(0, 16), 1, 0.8, cv::Scalar(0, 0, 0));
        cv::putText(m, strFPS, cv::Point(0, 32), 1, 0.8, cv::Scalar(0, 0, 0));
        cv::putText(m, strDevName, cv::Point(0, 48), 1, 0.8, cv::Scalar(0, 0, 0));

        m_pD3D11Ctx->Unmap(pSurface, subResource);

        return;
    }


    int render()
    {
        try
        {
            if (m_shutdown)
                return 0;

            HRESULT r;
            ID3D11Texture2D* pSurface;

            r = get_surface(&pSurface);
            if (FAILED(r))
            {
                return -1;
            }

            switch (m_mode)
            {
                case 0:
                    // no processing
                    break;

                case 1:
                {
                    // process video frame on CPU
                    UINT subResource = ::D3D11CalcSubresource(0, 0, 1);

                    D3D11_MAPPED_SUBRESOURCE mappedTex;
                    r = m_pD3D11Ctx->Map(m_pSurface, subResource, D3D11_MAP_WRITE_DISCARD, 0, &mappedTex);
                    if (FAILED(r))
                    {
                        return r;
                    }

                    cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);

                    if (!m_disableProcessing)
                    {
                        // blur D3D10 surface with OpenCV on CPU
                        cv::blur(m, m, cv::Size(15, 15), cv::Point(-7, -7));
                    }

                    m_pD3D11Ctx->Unmap(m_pSurface, subResource);

                    break;
                }

                case 2:
                {
                    // process video frame on GPU
                    cv::UMat u;

                    cv::directx::convertFromD3D11Texture2D(pSurface, u);

                    if (!m_disableProcessing)
                    {
                        // blur D3D9 surface with OpenCV on GPU with OpenCL
                        cv::blur(u, u, cv::Size(15, 15), cv::Point(-7, -7));
                    }

                    cv::directx::convertToD3D11Texture2D(u, pSurface);

                    break;
                }

            } // switch

            print_info(pSurface, m_mode, getFps(), m_oclDevName);

            // traditional DX render pipeline:
            //   BitBlt surface to backBuffer and flip backBuffer to frontBuffer
            m_pD3D11Ctx->CopyResource(m_pBackBuffer, pSurface);

            // present the back buffer contents to the display
            // switch the back buffer and the front buffer
            r = m_pD3D11SwapChain->Present(0, 0);
            if (FAILED(r))
            {
                return -1;
            }
        }

        catch (cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return 0;
    }

    int cleanup(void)
    {
        SAFE_RELEASE(m_pSurface);
        SAFE_RELEASE(m_pBackBuffer);
        SAFE_RELEASE(m_pD3D11SwapChain);
        SAFE_RELEASE(m_pRenderTarget);
        SAFE_RELEASE(m_pD3D11Dev);
        SAFE_RELEASE(m_pD3D11Ctx);
        return 0;
    }

private:
    bool                    m_shutdown;
    int                     m_mode;
    cv::String              m_modeStr[3];
    int                     m_disableProcessing;
    ID3D11Device*           m_pD3D11Dev;
    IDXGISwapChain*         m_pD3D11SwapChain;
    ID3D11DeviceContext*    m_pD3D11Ctx;
    ID3D11Texture2D*        m_pBackBuffer;
    ID3D11Texture2D*        m_pSurface;
    ID3D11RenderTargetView* m_pRenderTarget;
    cv::VideoCapture        m_cap;
    cv::Mat                 m_frame_bgr;
    cv::Mat                 m_frame_rgba;
    cv::ocl::Context        m_oclCtx;
    cv::String              m_oclPlatformName;
    cv::String              m_oclDevName;
};


using namespace cv;

int main(int argc, char** argv)
{
    cv::VideoCapture cap;

    if (argc > 1)
    {
        cap.open(argv[1]);
    }
    else
        cap.open(0);

    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    std::string wndname = "D3D11 Window";

    D3D11WinApp app(width, height, wndname, cap);

    try
    {
        app.Create();
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
