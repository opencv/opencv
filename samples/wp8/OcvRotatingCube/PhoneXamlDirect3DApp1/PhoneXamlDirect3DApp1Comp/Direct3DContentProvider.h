#pragma once

#include "pch.h"
#include <wrl/module.h>
#include <Windows.Phone.Graphics.Interop.h>
#include <DrawingSurfaceNative.h>

#include "Direct3DInterop.h"

class Direct3DContentProvider : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRtClassicComMix>,
        ABI::Windows::Phone::Graphics::Interop::IDrawingSurfaceContentProvider,
        IDrawingSurfaceContentProviderNative>
{
public:
    Direct3DContentProvider(PhoneXamlDirect3DApp1Comp::Direct3DInterop^ controller);

    void ReleaseD3DResources();

    // IDrawingSurfaceContentProviderNative
    HRESULT STDMETHODCALLTYPE Connect(_In_ IDrawingSurfaceRuntimeHostNative* host);
    void STDMETHODCALLTYPE Disconnect();

    HRESULT STDMETHODCALLTYPE PrepareResources(_In_ const LARGE_INTEGER* presentTargetTime, _Out_ BOOL* contentDirty);
    HRESULT STDMETHODCALLTYPE GetTexture(_In_ const DrawingSurfaceSizeF* size, _Out_ IDrawingSurfaceSynchronizedTextureNative** synchronizedTexture, _Out_ DrawingSurfaceRectF* textureSubRectangle);

private:
    HRESULT InitializeTexture(_In_ const DrawingSurfaceSizeF* size);

    PhoneXamlDirect3DApp1Comp::Direct3DInterop^ m_controller;
    Microsoft::WRL::ComPtr<IDrawingSurfaceRuntimeHostNative> m_host;
    Microsoft::WRL::ComPtr<IDrawingSurfaceSynchronizedTextureNative> m_synchronizedTexture;
};