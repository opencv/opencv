/*
// Sample demonstrating interoperability of OpenCV UMat with Direct X surface
// At first, the data obtained from video file or camera and
// placed onto Direct X surface,
// following mapping of this Direct X surface to OpenCV UMat and call cv::Blur
// function. The result is mapped back to Direct X surface and rendered through
// Direct X API.
*/
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d10.h>

#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "d3dsample.hpp"

#pragma comment (lib, "d3d10.lib")


using namespace std;
using namespace cv;

class D3D10WinApp : public D3DSample
{
public:
    D3D10WinApp(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        D3DSample(width, height, window_name, cap) {}

    ~D3D10WinApp() {}


    int create(void)
    {
        // base initialization
        D3DSample::create();

        // initialize DirectX
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

        r = ::D3D10CreateDeviceAndSwapChain(
                NULL,
                D3D10_DRIVER_TYPE_HARDWARE,
                NULL,
                0,
                D3D10_SDK_VERSION,
                &scd,
                &m_pD3D10SwapChain,
                &m_pD3D10Dev);
        if (FAILED(r))
        {
            return -1;
        }

        r = m_pD3D10SwapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID*)&m_pBackBuffer);
        if (FAILED(r))
        {
            return -1;
        }

        r = m_pD3D10Dev->CreateRenderTargetView(m_pBackBuffer, NULL, &m_pRenderTarget);
        if (FAILED(r))
        {
            return -1;
        }

        m_pD3D10Dev->OMSetRenderTargets(1, &m_pRenderTarget, NULL);

        D3D10_VIEWPORT viewport;
        ZeroMemory(&viewport, sizeof(D3D10_VIEWPORT));

        viewport.Width    = m_width;
        viewport.Height   = m_height;
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 0.0f;

        m_pD3D10Dev->RSSetViewports(1, &viewport);

        D3D10_TEXTURE2D_DESC desc = { 0 };

        desc.Width            = m_width;
        desc.Height           = m_height;
        desc.MipLevels        = 1;
        desc.ArraySize        = 1;
        desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BindFlags        = D3D10_BIND_SHADER_RESOURCE;
        desc.Usage            = D3D10_USAGE_DYNAMIC;
        desc.CPUAccessFlags   = D3D10_CPU_ACCESS_WRITE;

        r = m_pD3D10Dev->CreateTexture2D(&desc, NULL, &m_pSurface);
        if (FAILED(r))
        {
            std::cerr << "Can't create texture with input image" << std::endl;
            return -1;
        }

        // initialize OpenCL context of OpenCV lib from DirectX
        if (cv::ocl::haveOpenCL())
        {
            m_oclCtx = cv::directx::ocl::initializeContextFromD3D10Device(m_pD3D10Dev);
        }

        m_oclDevName = cv::ocl::useOpenCL() ?
            cv::ocl::Context::getDefault().device(0).name() :
            "No OpenCL device";

        return 0;
    } // create()


    // get media data on DX surface for further processing
    int get_surface(ID3D10Texture2D** ppSurface)
    {
        HRESULT r;

        if (!m_cap.read(m_frame_bgr))
            return -1;

        cv::cvtColor(m_frame_bgr, m_frame_rgba, CV_RGB2BGRA);

        UINT subResource = ::D3D10CalcSubresource(0, 0, 1);

        D3D10_MAPPED_TEXTURE2D mappedTex;
        r = m_pSurface->Map(subResource, D3D10_MAP_WRITE_DISCARD, 0, &mappedTex);
        if (FAILED(r))
        {
            return r;
        }

        cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);
        // copy video frame data to surface
        m_frame_rgba.copyTo(m);

        m_pSurface->Unmap(subResource);

        *ppSurface = m_pSurface;

        return 0;
    } // get_surface()


    // process and render media data
    int render()
    {
        try
        {
            if (m_shutdown)
                return 0;

            HRESULT r;
            ID3D10Texture2D* pSurface;

            r = get_surface(&pSurface);
            if (FAILED(r))
            {
                return -1;
            }

            switch (m_mode)
            {
                case MODE_NOP:
                    // no processing
                    break;

                case MODE_CPU:
                {
                    // process video frame on CPU
                    UINT subResource = ::D3D10CalcSubresource(0, 0, 1);

                    D3D10_MAPPED_TEXTURE2D mappedTex;
                    r = m_pSurface->Map(subResource, D3D10_MAP_WRITE_DISCARD, 0, &mappedTex);
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

                    m_pSurface->Unmap(subResource);

                    break;
                }

                case MODE_GPU:
                {
                    // process video frame on GPU
                    cv::UMat u;

                    cv::directx::convertFromD3D10Texture2D(pSurface, u);

                    if (!m_disableProcessing)
                    {
                        // blur D3D9 surface with OpenCV on GPU with OpenCL
                        cv::blur(u, u, cv::Size(15, 15), cv::Point(-7, -7));
                    }

                    cv::directx::convertToD3D10Texture2D(u, pSurface);

                    break;
                }

            } // switch

            print_info(pSurface, m_mode, getFps(), m_oclDevName);

            // traditional DX render pipeline:
            //   BitBlt surface to backBuffer and flip backBuffer to frontBuffer
            m_pD3D10Dev->CopyResource(m_pBackBuffer, pSurface);

            // present the back buffer contents to the display
            // switch the back buffer and the front buffer
            r = m_pD3D10SwapChain->Present(0, 0);
            if (FAILED(r))
            {
                return -1;
            }
        } // try

        catch (cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return 0;
    } // render()


    void print_info(ID3D10Texture2D* pSurface, int mode, float fps, cv::String oclDevName)
    {
        HRESULT r;

        UINT subResource = ::D3D10CalcSubresource(0, 0, 1);

        D3D10_MAPPED_TEXTURE2D mappedTex;
        r = pSurface->Map(subResource, D3D10_MAP_WRITE_DISCARD, 0, &mappedTex);
        if (FAILED(r))
        {
            return;
        }

        cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);

        cv::String strMode    = cv::format("%s", m_modeStr[mode].c_str());
        cv::String strFPS     = cv::format("%2.1f", fps);
        cv::String strDevName = cv::format("%s", oclDevName.c_str());

        cv::putText(m, strMode, cv::Point(0, 16), 1, 0.8, cv::Scalar(0, 0, 0));
        cv::putText(m, strFPS, cv::Point(0, 32), 1, 0.8, cv::Scalar(0, 0, 0));
        cv::putText(m, strDevName, cv::Point(0, 48), 1, 0.8, cv::Scalar(0, 0, 0));

        m_pSurface->Unmap(subResource);

        return;
    } // print_info()


    int cleanup(void)
    {
        SAFE_RELEASE(m_pSurface);
        SAFE_RELEASE(m_pBackBuffer);
        SAFE_RELEASE(m_pD3D10SwapChain);
        SAFE_RELEASE(m_pRenderTarget);
        SAFE_RELEASE(m_pD3D10Dev);
        D3DSample::cleanup();
        return 0;
    } // cleanup()

private:
    ID3D10Device*           m_pD3D10Dev;
    IDXGISwapChain*         m_pD3D10SwapChain;
    ID3D10Texture2D*        m_pBackBuffer;
    ID3D10Texture2D*        m_pSurface;
    ID3D10RenderTargetView* m_pRenderTarget;
    cv::ocl::Context        m_oclCtx;
    cv::String              m_oclPlatformName;
    cv::String              m_oclDevName;
};


// main func
int main(int argc, char** argv)
{
    std::string title = "D3D10 interop sample";
    return d3d_app<D3D10WinApp>(argc, argv, title);
}
