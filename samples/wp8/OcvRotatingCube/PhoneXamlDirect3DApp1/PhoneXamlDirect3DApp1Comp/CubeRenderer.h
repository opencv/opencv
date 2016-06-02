#pragma once

#include "Direct3DBase.h"
#include <d3d11.h>
#include <mutex>


struct ModelViewProjectionConstantBuffer
{
    DirectX::XMFLOAT4X4 model;
    DirectX::XMFLOAT4X4 view;
    DirectX::XMFLOAT4X4 projection;
};

struct Vertex	//Overloaded Vertex Structure
{
    Vertex(){}
    Vertex(float x, float y, float z,
        float u, float v)
        : pos(x,y,z), texCoord(u, v){}

    DirectX::XMFLOAT3 pos;
    DirectX::XMFLOAT2 texCoord;
};

// This class renders a simple spinning cube.
ref class CubeRenderer sealed : public Direct3DBase
{
public:
    CubeRenderer();

    // Direct3DBase methods.
    virtual void CreateDeviceResources() override;
    virtual void CreateWindowSizeDependentResources() override;
    virtual void Render() override;

    // Method for updating time-dependent objects.
    void Update(float timeTotal, float timeDelta);

    void CreateTextureFromByte(byte  *  buffer,int width,int height);
private:
    void Render(Microsoft::WRL::ComPtr<ID3D11RenderTargetView> renderTargetView, Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depthStencilView);
    bool m_loadingComplete;

    Microsoft::WRL::ComPtr<ID3D11InputLayout>	m_inputLayout;
    Microsoft::WRL::ComPtr<ID3D11Buffer>		m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer>		m_indexBuffer;
    Microsoft::WRL::ComPtr<ID3D11VertexShader>	m_vertexShader;
    Microsoft::WRL::ComPtr<ID3D11PixelShader>	m_pixelShader;
    Microsoft::WRL::ComPtr<ID3D11Buffer>		m_constantBuffer;
    Microsoft::WRL::ComPtr<ID3D11Texture2D>		 m_texture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_SRV;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> m_cubesTexSamplerState;
    uint32 m_indexCount;
    ModelViewProjectionConstantBuffer m_constantBufferData;
    std::mutex   m_mutex;
    Microsoft::WRL::ComPtr<ID3D11BlendState> m_transparency;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_CCWcullMode;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_CWcullMode;

};
