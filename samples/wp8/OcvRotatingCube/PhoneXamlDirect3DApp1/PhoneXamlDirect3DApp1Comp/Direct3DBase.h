#pragma once

#include "DirectXHelper.h"

// Helper class that initializes DirectX APIs for 3D rendering.
ref class Direct3DBase abstract
{
internal:
    Direct3DBase();

public:
    virtual void Initialize();
    virtual void CreateDeviceResources();
    virtual void CreateWindowSizeDependentResources();
    virtual void UpdateForRenderResolutionChange(float width, float height);
    virtual void UpdateForWindowSizeChange(float width, float height);
    virtual void Render() = 0;

internal:
    virtual ID3D11Texture2D* GetTexture()
    {
        return m_renderTarget.Get();
    }

protected private:
    // Direct3D Objects.
    Microsoft::WRL::ComPtr<ID3D11Device1> m_d3dDevice;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext1> m_d3dContext;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_renderTarget;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_renderTargetView;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> m_depthStencilView;

    // Cached renderer properties.
    D3D_FEATURE_LEVEL m_featureLevel;
    Windows::Foundation::Size m_renderTargetSize;
    Windows::Foundation::Rect m_windowBounds;
};