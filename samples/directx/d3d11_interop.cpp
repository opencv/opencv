/*
// A sample program demonstrating interoperability of OpenCV cv::UMat with Direct X surface
// At first, the data obtained from video file or camera and placed onto Direct X surface,
// following mapping of this Direct X surface to OpenCV cv::UMat and call cv::Blur function.
// The result is mapped back to Direct X surface and rendered through Direct X API.
*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>

#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "d3dsample.hpp"


class D3D11WinApp : public D3DSample
{
public:
    D3D11WinApp(int width, int height, std::string& window_name, cv::VideoCapture& cap)
    : D3DSample(width, height, window_name, cap),
      m_nv12_available(false)
    {}

    ~D3D11WinApp() {}


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
            throw std::runtime_error("D3D11CreateDeviceAndSwapChain() failed!");
        }

#if defined(_WIN32_WINNT_WIN8) && _WIN32_WINNT >= _WIN32_WINNT_WIN8
        UINT fmt = 0;
        r = m_pD3D11Dev->CheckFormatSupport(DXGI_FORMAT_NV12, &fmt);
        if (SUCCEEDED(r))
        {
            m_nv12_available = true;
        }
#endif

        r = m_pD3D11SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&m_pBackBuffer);
        if (FAILED(r))
        {
            throw std::runtime_error("GetBuffer() failed!");
        }

        r = m_pD3D11Dev->CreateRenderTargetView(m_pBackBuffer, NULL, &m_pRenderTarget);
        if (FAILED(r))
        {
            throw std::runtime_error("CreateRenderTargetView() failed!");
        }

        m_pD3D11Ctx->OMSetRenderTargets(1, &m_pRenderTarget, NULL);

        D3D11_VIEWPORT viewport;
        ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

        viewport.Width    = (float)m_width;
        viewport.Height   = (float)m_height;
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 0.0f;

        m_pD3D11Ctx->RSSetViewports(1, &viewport);

        m_pSurfaceRGBA = 0;
        m_pSurfaceNV12 = 0;
        m_pSurfaceNV12_cpu_copy = 0;

        D3D11_TEXTURE2D_DESC desc_rgba;

        desc_rgba.Width              = m_width;
        desc_rgba.Height             = m_height;
        desc_rgba.MipLevels          = 1;
        desc_rgba.ArraySize          = 1;
        desc_rgba.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc_rgba.SampleDesc.Count   = 1;
        desc_rgba.SampleDesc.Quality = 0;
        desc_rgba.BindFlags          = D3D11_BIND_SHADER_RESOURCE;
        desc_rgba.Usage              = D3D11_USAGE_DYNAMIC;
        desc_rgba.CPUAccessFlags     = D3D11_CPU_ACCESS_WRITE;
        desc_rgba.MiscFlags          = 0;

        r = m_pD3D11Dev->CreateTexture2D(&desc_rgba, 0, &m_pSurfaceRGBA);
        if (FAILED(r))
        {
            throw std::runtime_error("Can't create DX texture");
        }

#if defined(_WIN32_WINNT_WIN8) && _WIN32_WINNT >= _WIN32_WINNT_WIN8
        if(m_nv12_available)
        {
            D3D11_TEXTURE2D_DESC desc_nv12;

            desc_nv12.Width              = m_width;
            desc_nv12.Height             = m_height;
            desc_nv12.MipLevels          = 1;
            desc_nv12.ArraySize          = 1;
            desc_nv12.Format             = DXGI_FORMAT_NV12;
            desc_nv12.SampleDesc.Count   = 1;
            desc_nv12.SampleDesc.Quality = 0;
            desc_nv12.BindFlags          = D3D11_BIND_SHADER_RESOURCE;
            desc_nv12.Usage              = D3D11_USAGE_DEFAULT;
            desc_nv12.CPUAccessFlags     = 0;
            desc_nv12.MiscFlags          = D3D11_RESOURCE_MISC_SHARED;

            r = m_pD3D11Dev->CreateTexture2D(&desc_nv12, 0, &m_pSurfaceNV12);
            if (FAILED(r))
            {
                throw std::runtime_error("Can't create DX NV12 texture");
            }

            D3D11_TEXTURE2D_DESC desc_nv12_cpu_copy;

            desc_nv12_cpu_copy.Width              = m_width;
            desc_nv12_cpu_copy.Height             = m_height;
            desc_nv12_cpu_copy.MipLevels          = 1;
            desc_nv12_cpu_copy.ArraySize          = 1;
            desc_nv12_cpu_copy.Format             = DXGI_FORMAT_NV12;
            desc_nv12_cpu_copy.SampleDesc.Count   = 1;
            desc_nv12_cpu_copy.SampleDesc.Quality = 0;
            desc_nv12_cpu_copy.BindFlags          = 0;
            desc_nv12_cpu_copy.Usage              = D3D11_USAGE_STAGING;
            desc_nv12_cpu_copy.CPUAccessFlags     = /*D3D11_CPU_ACCESS_WRITE | */D3D11_CPU_ACCESS_READ;
            desc_nv12_cpu_copy.MiscFlags          = 0;

            r = m_pD3D11Dev->CreateTexture2D(&desc_nv12_cpu_copy, 0, &m_pSurfaceNV12_cpu_copy);
            if (FAILED(r))
            {
                throw std::runtime_error("Can't create DX NV12 texture");
            }
        }
#endif

        // initialize OpenCL context of OpenCV lib from DirectX
        if (cv::ocl::haveOpenCL())
        {
            m_oclCtx = cv::directx::ocl::initializeContextFromD3D11Device(m_pD3D11Dev);
        }

        m_oclDevName = cv::ocl::useOpenCL() ?
            cv::ocl::Context::getDefault().device(0).name() :
            "No OpenCL device";

        return EXIT_SUCCESS;
    } // create()


    // get media data on DX surface for further processing
    int get_surface(ID3D11Texture2D** ppSurface, bool use_nv12)
    {
        HRESULT r;

        if (!m_cap.read(m_frame_bgr))
            return EXIT_FAILURE;

        if (use_nv12)
        {
            cv::cvtColor(m_frame_bgr, m_frame_i420, cv::COLOR_BGR2YUV_I420);

            convert_I420_to_NV12(m_frame_i420, m_frame_nv12, m_width, m_height);

            m_pD3D11Ctx->UpdateSubresource(m_pSurfaceNV12, 0, 0, m_frame_nv12.data, (UINT)m_frame_nv12.step[0], (UINT)m_frame_nv12.total());
        }
        else
        {
            cv::cvtColor(m_frame_bgr, m_frame_rgba, cv::COLOR_BGR2RGBA);

            // process video frame on CPU
            UINT subResource = ::D3D11CalcSubresource(0, 0, 1);

            D3D11_MAPPED_SUBRESOURCE mappedTex;
            r = m_pD3D11Ctx->Map(m_pSurfaceRGBA, subResource, D3D11_MAP_WRITE_DISCARD, 0, &mappedTex);
            if (FAILED(r))
            {
                throw std::runtime_error("surface mapping failed!");
            }

            cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, mappedTex.RowPitch);
            m_frame_rgba.copyTo(m);

            m_pD3D11Ctx->Unmap(m_pSurfaceRGBA, subResource);
        }

        *ppSurface = use_nv12 ? m_pSurfaceNV12 : m_pSurfaceRGBA;

        return EXIT_SUCCESS;
    } // get_surface()


    // process and render media data
    int render()
    {
        try
        {
            if (m_shutdown)
                return EXIT_SUCCESS;

            // capture user input once
            MODE mode = (m_mode == MODE_GPU_NV12 && !m_nv12_available) ? MODE_GPU_RGBA : m_mode;

            HRESULT r;
            ID3D11Texture2D* pSurface = 0;

            r = get_surface(&pSurface, mode == MODE_GPU_NV12);
            if (FAILED(r))
            {
                throw std::runtime_error("get_surface() failed!");
            }

            m_timer.reset();
            m_timer.start();

            switch (mode)
            {
            case MODE_CPU:
            {
                // process video frame on CPU
                UINT subResource = ::D3D11CalcSubresource(0, 0, 1);

                D3D11_MAPPED_SUBRESOURCE mappedTex;
                r = m_pD3D11Ctx->Map(pSurface, subResource, D3D11_MAP_WRITE_DISCARD, 0, &mappedTex);
                if (FAILED(r))
                {
                    throw std::runtime_error("surface mapping failed!");
                }

                cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);

                if (m_demo_processing)
                {
                    // blur data from D3D11 surface with OpenCV on CPU
                    cv::blur(m, m, cv::Size(15, 15));
                }

                m_timer.stop();

                cv::String strMode = cv::format("mode: %s", m_modeStr[MODE_CPU].c_str());
                cv::String strProcessing = m_demo_processing ? "blur frame" : "copy frame";
                cv::String strTime = cv::format("time: %4.3f msec", m_timer.getTimeMilli());
                cv::String strDevName = cv::format("OpenCL device: %s", m_oclDevName.c_str());

                cv::putText(m, strMode, cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
                cv::putText(m, strProcessing, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
                cv::putText(m, strTime, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
                cv::putText(m, strDevName, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);

                m_pD3D11Ctx->Unmap(pSurface, subResource);

                break;
            }

            case MODE_GPU_RGBA:
            case MODE_GPU_NV12:
            {
                // process video frame on GPU
                cv::UMat u;

                cv::directx::convertFromD3D11Texture2D(pSurface, u);

                if (m_demo_processing)
                {
                    // blur data from D3D11 surface with OpenCV on GPU with OpenCL
                    cv::blur(u, u, cv::Size(15, 15));
                }

                m_timer.stop();

                cv::String strMode = cv::format("mode: %s", m_modeStr[mode].c_str());
                cv::String strProcessing = m_demo_processing ? "blur frame" : "copy frame";
                cv::String strTime = cv::format("time: %4.3f msec", m_timer.getTimeMilli());
                cv::String strDevName = cv::format("OpenCL device: %s", m_oclDevName.c_str());

                cv::putText(u, strMode, cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
                cv::putText(u, strProcessing, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
                cv::putText(u, strTime, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
                cv::putText(u, strDevName, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);

                cv::directx::convertToD3D11Texture2D(u, pSurface);

                if (mode == MODE_GPU_NV12)
                {
                    // just for rendering, we need to convert NV12 to RGBA.
                    m_pD3D11Ctx->CopyResource(m_pSurfaceNV12_cpu_copy, m_pSurfaceNV12);

                    // process video frame on CPU
                    {
                        UINT subResource = ::D3D11CalcSubresource(0, 0, 1);

                        D3D11_MAPPED_SUBRESOURCE mappedTex;
                        r = m_pD3D11Ctx->Map(m_pSurfaceNV12_cpu_copy, subResource, D3D11_MAP_READ, 0, &mappedTex);
                        if (FAILED(r))
                        {
                            throw std::runtime_error("surface mapping failed!");
                        }

                        cv::Mat frame_nv12(m_height + (m_height / 2), m_width, CV_8UC1, mappedTex.pData, mappedTex.RowPitch);
                        cv::cvtColor(frame_nv12, m_frame_rgba, cv::COLOR_YUV2RGBA_NV12);

                        m_pD3D11Ctx->Unmap(m_pSurfaceNV12_cpu_copy, subResource);
                    }

                    {
                        UINT subResource = ::D3D11CalcSubresource(0, 0, 1);

                        D3D11_MAPPED_SUBRESOURCE mappedTex;
                        r = m_pD3D11Ctx->Map(m_pSurfaceRGBA, subResource, D3D11_MAP_WRITE_DISCARD, 0, &mappedTex);
                        if (FAILED(r))
                        {
                            throw std::runtime_error("surface mapping failed!");
                        }

                        cv::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, mappedTex.RowPitch);
                        m_frame_rgba.copyTo(m);

                        m_pD3D11Ctx->Unmap(m_pSurfaceRGBA, subResource);
                    }

                    pSurface = m_pSurfaceRGBA;
                }

                break;
            }

            } // switch

            // traditional DX render pipeline:
            //   BitBlt surface to backBuffer and flip backBuffer to frontBuffer
            m_pD3D11Ctx->CopyResource(m_pBackBuffer, pSurface);

            // present the back buffer contents to the display
            // switch the back buffer and the front buffer
            r = m_pD3D11SwapChain->Present(0, 0);
            if (FAILED(r))
            {
                throw std::runtime_error("switch between fronat and back buffers failed!");
            }
        } // try

        catch (const cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            cleanup();
            return 10;
        }

        catch (const std::exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            cleanup();
            return 11;
        }

        return EXIT_SUCCESS;
    } // render()


    int cleanup(void)
    {
        SAFE_RELEASE(m_pSurfaceRGBA);
        SAFE_RELEASE(m_pSurfaceNV12);
        SAFE_RELEASE(m_pSurfaceNV12_cpu_copy);
        SAFE_RELEASE(m_pBackBuffer);
        SAFE_RELEASE(m_pD3D11SwapChain);
        SAFE_RELEASE(m_pRenderTarget);
        SAFE_RELEASE(m_pD3D11Dev);
        SAFE_RELEASE(m_pD3D11Ctx);
        D3DSample::cleanup();
        return EXIT_SUCCESS;
    } // cleanup()

protected:
    void convert_I420_to_NV12(cv::Mat& i420, cv::Mat& nv12, int width, int height)
    {
        nv12.create(i420.rows, i420.cols, CV_8UC1);

        unsigned char* pSrcY = i420.data;
        unsigned char* pDstY = nv12.data;
        size_t srcStep = i420.step[0];
        size_t dstStep = nv12.step[0];

        {
            unsigned char* src;
            unsigned char* dst;

            // copy Y plane
            for (int i = 0; i < height; i++)
            {
                src = pSrcY + i*srcStep;
                dst = pDstY + i*dstStep;

                for (int j = 0; j < width; j++)
                {
                    dst[j] = src[j];
                }
            }
        }

        {
            // copy U/V planes to UV plane
            unsigned char* pSrcU;
            unsigned char* pSrcV;
            unsigned char* pDstUV;

            size_t uv_offset = height * dstStep;

            for (int i = 0; i < height / 2; i++)
            {
                pSrcU = pSrcY + height*width + i*(width / 2);
                pSrcV = pSrcY + height*width + (height / 2) * (width / 2) + i*(width / 2);

                pDstUV = pDstY + uv_offset + i*dstStep;

                for (int j = 0; j < width / 2; j++)
                {
                    pDstUV[j*2 + 0] = pSrcU[j];
                    pDstUV[j*2 + 1] = pSrcV[j];
                }
            }
        }

        return;
    }

private:
    ID3D11Device*           m_pD3D11Dev;
    IDXGISwapChain*         m_pD3D11SwapChain;
    ID3D11DeviceContext*    m_pD3D11Ctx;
    ID3D11Texture2D*        m_pBackBuffer;
    ID3D11Texture2D*        m_pSurfaceRGBA;
    ID3D11Texture2D*        m_pSurfaceNV12;
    ID3D11Texture2D*        m_pSurfaceNV12_cpu_copy;
    ID3D11RenderTargetView* m_pRenderTarget;
    cv::ocl::Context        m_oclCtx;
    cv::String              m_oclPlatformName;
    cv::String              m_oclDevName;
    bool                    m_nv12_available;
    cv::Mat                 m_frame_i420;
    cv::Mat                 m_frame_nv12;
};


// main func
int main(int argc, char** argv)
{
    std::string title = "D3D11 interop sample";
    return d3d_app<D3D11WinApp>(argc, argv, title);
}
