#include <windows.h>
#include <d3d9.h>
#pragma comment (lib, "d3d9.lib")

#define USE_D3DEX
#define WINDOW_NAME "OpenCV Direct3D 9 Ex Sample"

IDirect3D9Ex *pD3D = NULL;
IDirect3DDevice9Ex *dev = NULL;
IDirect3DSurface9 *pBackBuffer = NULL;
IDirect3DSurface9 *pCPUWriteSurface = NULL; // required name
IDirect3DSurface9 *pReadOnlySurface = NULL; // required name
HANDLE readOnlySurfaceShared = 0; // required name
IDirect3DSurface9 *pSurface = NULL; // required name
HANDLE surfaceShared = 0; // required name

#include "d3d_base.inl.hpp"

bool initDirect3D(void)
{
    if (FAILED(Direct3DCreate9Ex(D3D_SDK_VERSION, &pD3D)))
    {
        return false;
    }

    D3DDISPLAYMODEEX ddm;
    ZeroMemory(&ddm, sizeof(ddm));
    ddm.Size = sizeof(D3DDISPLAYMODEEX);
    D3DDISPLAYROTATION rotation;
    if (FAILED(pD3D->GetAdapterDisplayModeEx(D3DADAPTER_DEFAULT, &ddm, &rotation)))
    {
        return false;
    }

    D3DPRESENT_PARAMETERS d3dpp;
    ZeroMemory(&d3dpp,sizeof(D3DPRESENT_PARAMETERS));

    DWORD flags = D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_PUREDEVICE | D3DCREATE_NOWINDOWCHANGES
            | D3DCREATE_MULTITHREADED;

    d3dpp.Windowed = true;
    d3dpp.Flags = 0;
    d3dpp.BackBufferCount = 0;
    d3dpp.BackBufferFormat = ddm.Format;
    d3dpp.BackBufferHeight = HEIGHT;
    d3dpp.BackBufferWidth = WIDTH;
    d3dpp.MultiSampleType = D3DMULTISAMPLE_NONE;
    d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    d3dpp.hDeviceWindow = hWnd;
    d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
    d3dpp.FullScreen_RefreshRateInHz = D3DPRESENT_RATE_DEFAULT;

    if (FAILED(pD3D->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd, flags, &d3dpp, NULL, &dev)))
    {
        return false;
    }

    if (FAILED(dev->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &pBackBuffer)))
    {
        return false;
    }

    return true;
}

bool initDirect3DTextures()
{
    if (FAILED(dev->CreateOffscreenPlainSurface(WIDTH, HEIGHT, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &pSurface, &surfaceShared)))
    {
        std::cerr << "Can't create surface for result" << std::endl;
        return false;
    }

    if (FAILED(dev->CreateOffscreenPlainSurface(WIDTH, HEIGHT, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &pReadOnlySurface, &readOnlySurfaceShared)))
    {
        std::cerr << "Can't create read only surface" << std::endl;
        return false;
    }
    else
    {
        IDirect3DSurface9* pTmpSurface;
        if (FAILED(dev->CreateOffscreenPlainSurface(WIDTH, HEIGHT, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &pTmpSurface, NULL)))
        {
            std::cerr << "Can't create temp surface for CPU write" << std::endl;
            return false;
        }

        D3DLOCKED_RECT memDesc = {0, NULL};
        RECT rc = {0, 0, WIDTH, HEIGHT};
        if (SUCCEEDED(pTmpSurface->LockRect(&memDesc, &rc, 0)))
        {
            cv::Mat m(cv::Size(WIDTH, HEIGHT), CV_8UC4, memDesc.pBits, (int)memDesc.Pitch);
            getInputTexture().copyTo(m);
            pTmpSurface->UnlockRect();
            dev->StretchRect(pTmpSurface, NULL, pReadOnlySurface, NULL, D3DTEXF_NONE);
        }
        else
        {
            std::cerr << "Can't LockRect() on surface" << std::endl;
        }
        pTmpSurface->Release();
    }

    if (FAILED(dev->CreateOffscreenPlainSurface(WIDTH, HEIGHT, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &pCPUWriteSurface, NULL)))
    {
        std::cerr << "Can't create surface for CPU write" << std::endl;
        return false;
    }

    return true;
}


void render(void)
{
    // check to make sure you have a valid Direct3D device
    CV_Assert(dev);

    renderToD3DObject();

    if (g_sampleType == 0)
    {
        // nothing
    }
    else if (g_sampleType == 1)
    {
        if (FAILED(dev->StretchRect(pCPUWriteSurface, NULL, pBackBuffer, NULL, D3DTEXF_NONE)))
        {
            std::cerr << "Can't StretchRect()" << std::endl;
        }
    }
    else
    {
        if (FAILED(dev->StretchRect(pSurface, NULL, pBackBuffer, NULL, D3DTEXF_NONE)))
        {
            std::cerr << "Can't StretchRect()" << std::endl;
        }
    }

    if (SUCCEEDED(dev -> BeginScene()))
    {
        // end the scene
        dev -> EndScene();
    }

    // present the back buffer contents to the display
    dev->Present(NULL, NULL, NULL, NULL);
}

void cleanUp (void)
{
    SAFE_RELEASE(pCPUWriteSurface);
    SAFE_RELEASE(pReadOnlySurface);
    SAFE_RELEASE(pSurface);
    SAFE_RELEASE(pBackBuffer);
    SAFE_RELEASE(dev);
    SAFE_RELEASE(pD3D);
}
