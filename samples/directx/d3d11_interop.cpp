#include <windows.h>
#include <d3d11.h>
#pragma comment (lib, "d3d11.lib")

#define USE_D3D11
#define WINDOW_NAME "OpenCV Direct3D 11 Sample"

IDXGISwapChain *swapchain = NULL;
ID3D11Device *dev = NULL;
ID3D11DeviceContext *devcon = NULL;
ID3D11Texture2D *pBackBufferTexture = NULL;
ID3D11Texture2D *pCPUWriteTexture = NULL;
ID3D11Texture2D *pInputTexture = NULL;
ID3D11RenderTargetView *backbuffer = NULL;

#include "d3d_base.inl.hpp"

bool initDirect3D()
{
    DXGI_SWAP_CHAIN_DESC scd;

    ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));

    scd.BufferCount = 1;                                    // one back buffer
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;     // use 32-bit color
    scd.BufferDesc.Width = WIDTH;                    // set the back buffer width
    scd.BufferDesc.Height = HEIGHT;                  // set the back buffer height
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;      // how swap chain is to be used
    scd.OutputWindow = hWnd;                                // the window to be used
    scd.SampleDesc.Count = 1;                               // how many multisamples
    scd.Windowed = TRUE;                                    // windowed/full-screen mode
    scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;     // allow full-screen switching

    if (FAILED(D3D11CreateDeviceAndSwapChain(
            NULL,
            D3D_DRIVER_TYPE_HARDWARE,
            NULL,
            0,
            NULL,
            0,
            D3D11_SDK_VERSION,
            &scd,
            &swapchain,
            &dev,
            NULL,
            &devcon)))
    {
        return false;
    }

    if (FAILED(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBufferTexture)))
    {
        return false;
    }

    if (FAILED(dev->CreateRenderTargetView(pBackBufferTexture, NULL, &backbuffer)))
    {
        return false;
    }

    devcon->OMSetRenderTargets(1, &backbuffer, NULL);

    D3D11_VIEWPORT viewport = { 0 };
    viewport.Width = WIDTH;
    viewport.Height = HEIGHT;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 0.0f;
    devcon->RSSetViewports(1, &viewport);

    return true;
}

bool initDirect3DTextures()
{
    { // Create texture for demo 0
        D3D11_TEXTURE2D_DESC desc = { 0 };
        desc.Width = WIDTH;
        desc.Height = HEIGHT;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(dev->CreateTexture2D(&desc, NULL, &pCPUWriteTexture)))
        {
            std::cerr << "Can't create texture for CPU write sample" << std::endl;
            return false;
        }
    }

    { // Create Read-only texture
        cv::Mat inputMat = getInputTexture();

        D3D11_TEXTURE2D_DESC desc = { 0 };
        desc.Width = inputMat.size().width;
        desc.Height = inputMat.size().height;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.CPUAccessFlags = cv::ocl::useOpenCL() ? 0 : D3D11_CPU_ACCESS_READ;

        D3D11_SUBRESOURCE_DATA srInitData;
        srInitData.pSysMem = inputMat.data;
        srInitData.SysMemPitch = (UINT)inputMat.step[0];

        if (FAILED(dev->CreateTexture2D(&desc, &srInitData, &pInputTexture)))
        {
            std::cerr << "Can't create texture with input image" << std::endl;
            return false;
        }
    }

    return true;
}

void cleanUp(void)
{
    if (swapchain) swapchain->SetFullscreenState(FALSE, NULL);    // switch to windowed mode

    SAFE_RELEASE(swapchain);
    SAFE_RELEASE(pCPUWriteTexture);
    SAFE_RELEASE(pInputTexture);
    SAFE_RELEASE(pBackBufferTexture);
    SAFE_RELEASE(backbuffer);
    SAFE_RELEASE(dev);
    SAFE_RELEASE(devcon);
}


void render(void)
{
    // check to make sure you have a valid Direct3D device
    CV_Assert(dev);

    renderToD3DObject();

    // switch the back buffer and the front buffer
    swapchain->Present(0, 0);
}
