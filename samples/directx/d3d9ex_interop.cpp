/*
// A sample program demonstrating interoperability of OpenCV cv::UMat with Direct X surface
// At first, the data obtained from video file or camera and placed onto Direct X surface,
// following mapping of this Direct X surface to OpenCV cv::UMat and call cv::Blur function.
// The result is mapped back to Direct X surface and rendered through Direct X API.
*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>

#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "d3dsample.hpp"

#pragma comment (lib, "d3d9.lib")


class D3D9ExWinApp : public D3DSample
{
public:
    D3D9ExWinApp(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        D3DSample(width, height, window_name, cap) {}

    ~D3D9ExWinApp() {}

    int create(void)
    {
        // base initialization
        D3DSample::create();

        // initialize DirectX
        HRESULT r;

        r = ::Direct3DCreate9Ex(D3D_SDK_VERSION, &m_pD3D9Ex);
        if (FAILED(r))
        {
            return EXIT_FAILURE;
        }

        DWORD flags = D3DCREATE_HARDWARE_VERTEXPROCESSING |
                      D3DCREATE_PUREDEVICE |
                      D3DCREATE_NOWINDOWCHANGES |
                      D3DCREATE_MULTITHREADED |
                      D3DCREATE_FPU_PRESERVE;

        D3DPRESENT_PARAMETERS d3dpp;
        ::ZeroMemory(&d3dpp, sizeof(D3DPRESENT_PARAMETERS));

        d3dpp.Windowed                   = true;
        d3dpp.Flags                      = 0;
        d3dpp.BackBufferCount            = 0;
        d3dpp.BackBufferFormat           = D3DFMT_A8R8G8B8;
        d3dpp.BackBufferHeight           = m_height;
        d3dpp.BackBufferWidth            = m_width;
        d3dpp.MultiSampleType            = D3DMULTISAMPLE_NONE;
        d3dpp.SwapEffect                 = D3DSWAPEFFECT_DISCARD;
        d3dpp.hDeviceWindow              = m_hWnd;
        d3dpp.PresentationInterval       = D3DPRESENT_INTERVAL_IMMEDIATE;
        d3dpp.FullScreen_RefreshRateInHz = D3DPRESENT_RATE_DEFAULT;

        r = m_pD3D9Ex->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_hWnd, flags, &d3dpp, NULL, &m_pD3D9DevEx);
        if (FAILED(r))
        {
            return EXIT_FAILURE;
        }

        r = m_pD3D9DevEx->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &m_pBackBuffer);
        if (FAILED(r))
        {
            return EXIT_FAILURE;
        }

        r = m_pD3D9DevEx->CreateOffscreenPlainSurface(m_width, m_height, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &m_pSurface, NULL);
        if (FAILED(r))
        {
            std::cerr << "Can't create surface for result" << std::endl;
            return EXIT_FAILURE;
        }

        // initialize OpenCL context of OpenCV lib from DirectX
        if (cv::ocl::haveOpenCL())
        {
            m_oclCtx = cv::directx::ocl::initializeContextFromDirect3DDevice9(m_pD3D9DevEx);
        }

        m_oclDevName = cv::ocl::useOpenCL() ?
            cv::ocl::Context::getDefault().device(0).name() :
            "No OpenCL device";

        return EXIT_SUCCESS;
    } // create()


    // get media data on DX surface for further processing
    int get_surface(LPDIRECT3DSURFACE9* ppSurface)
    {
        HRESULT r;

        if (!m_cap.read(m_frame_bgr))
            return EXIT_FAILURE;

        cv::cvtColor(m_frame_bgr, m_frame_rgba, cv::COLOR_BGR2BGRA);

        D3DLOCKED_RECT memDesc = { 0, NULL };
        RECT rc = { 0, 0, m_width, m_height };

        r = m_pSurface->LockRect(&memDesc, &rc, 0);
        if (FAILED(r))
        {
            return r;
        }

        cv::Mat m(m_height, m_width, CV_8UC4, memDesc.pBits, memDesc.Pitch);
        // copy video frame data to surface
        m_frame_rgba.copyTo(m);

        r = m_pSurface->UnlockRect();
        if (FAILED(r))
        {
            return r;
        }

        *ppSurface = m_pSurface;

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
            MODE mode = m_mode == MODE_GPU_NV12 ? MODE_GPU_RGBA : m_mode;

            HRESULT r;
            LPDIRECT3DSURFACE9 pSurface;

            r = get_surface(&pSurface);
            if (FAILED(r))
            {
                return EXIT_FAILURE;
            }

            m_timer.reset();
            m_timer.start();

            switch (mode)
            {
                case MODE_CPU:
                {
                    // process video frame on CPU
                    D3DLOCKED_RECT memDesc = { 0, NULL };
                    RECT rc = { 0, 0, m_width, m_height };

                    r = pSurface->LockRect(&memDesc, &rc, 0);
                    if (FAILED(r))
                    {
                        return EXIT_FAILURE;
                    }

                    cv::Mat m(m_height, m_width, CV_8UC4, memDesc.pBits, memDesc.Pitch);

                    if (m_demo_processing)
                    {
                        // blur D3D9 surface with OpenCV on CPU
                        cv::blur(m, m, cv::Size(15, 15));
                    }

                    r = pSurface->UnlockRect();
                    if (FAILED(r))
                    {
                        return EXIT_FAILURE;
                    }

                    break;
                }

                case MODE_GPU_RGBA:
                {
                    // process video frame on GPU
                    cv::UMat u;

                    cv::directx::convertFromDirect3DSurface9(pSurface, u);

                    if (m_demo_processing)
                    {
                        // blur D3D9 surface with OpenCV on GPU with OpenCL
                        cv::blur(u, u, cv::Size(15, 15));
                    }

                    cv::directx::convertToDirect3DSurface9(u, pSurface);

                    break;
                }

            } // switch

            m_timer.stop();

            print_info(pSurface, m_mode, m_timer.getTimeMilli(), m_oclDevName);

            // traditional DX render pipeline:
            //   BitBlt surface to backBuffer and flip backBuffer to frontBuffer
            r = m_pD3D9DevEx->StretchRect(pSurface, NULL, m_pBackBuffer, NULL, D3DTEXF_NONE);
            if (FAILED(r))
            {
                return EXIT_FAILURE;
            }

            // present the back buffer contents to the display
            r = m_pD3D9DevEx->Present(NULL, NULL, NULL, NULL);
            if (FAILED(r))
            {
                return EXIT_FAILURE;
            }

        } // try

        catch (const cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return EXIT_SUCCESS;
    } // render()


    void print_info(LPDIRECT3DSURFACE9 pSurface, int mode, double time, cv::String oclDevName)
    {
        HDC hDC;

        HRESULT r = pSurface->GetDC(&hDC);
        if (FAILED(r))
        {
            return;
        }

        HFONT hFont = (HFONT)::GetStockObject(SYSTEM_FONT);

        HFONT hOldFont = (HFONT)::SelectObject(hDC, hFont);

        if (hOldFont)
        {
            TEXTMETRIC tm;
            ::GetTextMetrics(hDC, &tm);

            char buf[256];
            int  y = 0;

            buf[0] = 0;
            sprintf(buf, "mode: %s", m_modeStr[mode].c_str());
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf(buf, m_demo_processing ? "blur frame" : "copy frame");
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf(buf, "time: %4.1f msec", time);
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf(buf, "OpenCL device: %s", oclDevName.c_str());
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            ::SelectObject(hDC, hOldFont);
        }

        r = pSurface->ReleaseDC(hDC);

        return;
    } // print_info()


    int cleanup(void)
    {
        SAFE_RELEASE(m_pSurface);
        SAFE_RELEASE(m_pBackBuffer);
        SAFE_RELEASE(m_pD3D9DevEx);
        SAFE_RELEASE(m_pD3D9Ex);
        D3DSample::cleanup();
        return EXIT_SUCCESS;
    } // cleanup()

private:
    LPDIRECT3D9EX        m_pD3D9Ex;
    LPDIRECT3DDEVICE9EX  m_pD3D9DevEx;
    LPDIRECT3DSURFACE9   m_pBackBuffer;
    LPDIRECT3DSURFACE9   m_pSurface;
    cv::ocl::Context     m_oclCtx;
    cv::String           m_oclPlatformName;
    cv::String           m_oclDevName;
};


// main func
int main(int argc, char** argv)
{
    std::string title = "D3D9Ex interop sample";
    return d3d_app<D3D9ExWinApp>(argc, argv, title);
}
