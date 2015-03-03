#include "pch.h"
#include "Direct3DContentProvider.h"

using namespace PhoneXamlDirect3DApp1Comp;

Direct3DContentProvider::Direct3DContentProvider(Direct3DInterop^ controller) :
    m_controller(controller)
{
    m_controller->RequestAdditionalFrame += ref new RequestAdditionalFrameHandler([=] ()
        {
            if (m_host)
            {
                m_host->RequestAdditionalFrame();
            }
        });

    m_controller->RecreateSynchronizedTexture += ref new RecreateSynchronizedTextureHandler([=] ()
        {
            if (m_host)
            {
                m_host->CreateSynchronizedTexture(m_controller->GetTexture(), &m_synchronizedTexture);
            }
        });
}

// IDrawingSurfaceContentProviderNative interface
HRESULT Direct3DContentProvider::Connect(_In_ IDrawingSurfaceRuntimeHostNative* host)
{
    m_host = host;

    return m_controller->Connect(host);
}

void Direct3DContentProvider::Disconnect()
{
    m_controller->Disconnect();
    m_host = nullptr;
    m_synchronizedTexture = nullptr;
}

HRESULT Direct3DContentProvider::PrepareResources(_In_ const LARGE_INTEGER* presentTargetTime, _Out_ BOOL* contentDirty)
{
    return m_controller->PrepareResources(presentTargetTime, contentDirty);
}

HRESULT Direct3DContentProvider::GetTexture(_In_ const DrawingSurfaceSizeF* size, _Out_ IDrawingSurfaceSynchronizedTextureNative** synchronizedTexture, _Out_ DrawingSurfaceRectF* textureSubRectangle)
{
    HRESULT hr = S_OK;

    if (!m_synchronizedTexture)
    {
        hr = m_host->CreateSynchronizedTexture(m_controller->GetTexture(), &m_synchronizedTexture);
    }

    // Set output parameters.
    textureSubRectangle->left = 0.0f;
    textureSubRectangle->top = 0.0f;
    textureSubRectangle->right = static_cast<FLOAT>(size->width);
    textureSubRectangle->bottom = static_cast<FLOAT>(size->height);

    m_synchronizedTexture.CopyTo(synchronizedTexture);

    // Draw to the texture.
    if (SUCCEEDED(hr))
    {
        hr = m_synchronizedTexture->BeginDraw();

        if (SUCCEEDED(hr))
        {
            hr = m_controller->GetTexture(size, synchronizedTexture, textureSubRectangle);
        }

        m_synchronizedTexture->EndDraw();
    }

    return hr;
}