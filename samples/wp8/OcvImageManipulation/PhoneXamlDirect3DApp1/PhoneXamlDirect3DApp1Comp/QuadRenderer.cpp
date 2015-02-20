#include "pch.h"
#include "QuadRenderer.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Windows::Foundation;
using namespace Windows::UI::Core;

QuadRenderer::QuadRenderer() :
    m_loadingComplete(false),
    m_indexCount(0)
{
}

void QuadRenderer::CreateTextureFromByte(byte* buffer,int width,int height)
{
    int pixelSize = 4;

    if (m_Texture.Get() == nullptr)
    {
        CD3D11_TEXTURE2D_DESC textureDesc(
            DXGI_FORMAT_B8G8R8A8_UNORM,		// format
            static_cast<UINT>(width),		// width
            static_cast<UINT>(height),		// height
            1,								// arraySize
            1,								// mipLevels
            D3D11_BIND_SHADER_RESOURCE,		// bindFlags
            D3D11_USAGE_DYNAMIC,			// usage
            D3D11_CPU_ACCESS_WRITE,			// cpuaccessFlags
            1,								// sampleCount
            0,								// sampleQuality
            0								// miscFlags
            );

        D3D11_SUBRESOURCE_DATA data;
        data.pSysMem = buffer;
        data.SysMemPitch = pixelSize*width;
        data.SysMemSlicePitch = pixelSize*width*height;

        DX::ThrowIfFailed(
            m_d3dDevice->CreateTexture2D(
            &textureDesc,
            &data,
            m_Texture.ReleaseAndGetAddressOf()
            )
            );

        m_d3dDevice->CreateShaderResourceView(m_Texture.Get(), NULL, m_SRV.ReleaseAndGetAddressOf());
        D3D11_SAMPLER_DESC sampDesc;
        ZeroMemory(&sampDesc, sizeof(sampDesc));
        sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sampDesc.MinLOD = 0;
        sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
        m_d3dDevice->CreateSamplerState(&sampDesc, m_QuadsTexSamplerState.ReleaseAndGetAddressOf());
    }
    else
    {
        int nRowSpan = width * pixelSize;
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        HRESULT hr = m_d3dContext->Map(m_Texture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
        BYTE* mappedData = static_cast<BYTE*>(mappedResource.pData);

        for (int i = 0; i < height; ++i)
        {
            memcpy(mappedData + (i*mappedResource.RowPitch), buffer + (i*nRowSpan), nRowSpan);
        }

        m_d3dContext->Unmap(m_Texture.Get(), 0);
    }
}

void QuadRenderer::CreateDeviceResources()
{
    Direct3DBase::CreateDeviceResources();
    D3D11_BLEND_DESC blendDesc;
    ZeroMemory( &blendDesc, sizeof(blendDesc) );

    D3D11_RENDER_TARGET_BLEND_DESC rtbd;
    ZeroMemory( &rtbd, sizeof(rtbd) );

    rtbd.BlendEnable = TRUE;
    rtbd.SrcBlend = D3D11_BLEND_SRC_ALPHA;
    rtbd.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    rtbd.BlendOp = D3D11_BLEND_OP_ADD;
    rtbd.SrcBlendAlpha = D3D11_BLEND_ONE;
    rtbd.DestBlendAlpha = D3D11_BLEND_ZERO;
    rtbd.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    rtbd.RenderTargetWriteMask = 0x0f;

    blendDesc.AlphaToCoverageEnable = false;
    blendDesc.RenderTarget[0] = rtbd;

    m_d3dDevice->CreateBlendState(&blendDesc, &m_Transparency);

    D3D11_RASTERIZER_DESC cmdesc;
    ZeroMemory(&cmdesc, sizeof(D3D11_RASTERIZER_DESC));

    cmdesc.FillMode = D3D11_FILL_SOLID;
    cmdesc.CullMode = D3D11_CULL_BACK;
    cmdesc.DepthClipEnable = TRUE;

    cmdesc.FrontCounterClockwise = true;
    m_d3dDevice->CreateRasterizerState(&cmdesc, &CCWcullMode);

    cmdesc.FrontCounterClockwise = false;
    m_d3dDevice->CreateRasterizerState(&cmdesc, &CWcullMode);

    auto loadVSTask = DX::ReadDataAsync("SimpleVertexShader.cso");
    auto loadPSTask = DX::ReadDataAsync("SimplePixelShader.cso");
    auto createVSTask = loadVSTask.then([this](Platform::Array<byte>^ fileData)
    {
        DX::ThrowIfFailed(
            m_d3dDevice->CreateVertexShader(
            fileData->Data,
            fileData->Length,
            nullptr,
            &m_vertexShader
            )
            );

        const D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
        {
            { "POSITION",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        DX::ThrowIfFailed(
            m_d3dDevice->CreateInputLayout(
            vertexDesc,
            ARRAYSIZE(vertexDesc),
            fileData->Data,
            fileData->Length,
            &m_inputLayout
            )
            );
    });

    auto createPSTask = loadPSTask.then([this](Platform::Array<byte>^ fileData)
    {
        DX::ThrowIfFailed(
            m_d3dDevice->CreatePixelShader(
            fileData->Data,
            fileData->Length,
            nullptr,
            &m_pixelShader
            )
            );

        CD3D11_BUFFER_DESC constantBufferDesc(sizeof(ModelViewProjectionConstantBuffer), D3D11_BIND_CONSTANT_BUFFER);
        DX::ThrowIfFailed(
            m_d3dDevice->CreateBuffer(
            &constantBufferDesc,
            nullptr,
            &m_constantBuffer
            )
            );
    });

    auto createCubeTask = (createPSTask && createVSTask).then([this] ()
    {
        Vertex v[] =
        {
            Vertex(-1.0f, -1.0f, 1.0f, 1.0f, 1.0f),
            Vertex(1.0f, -1.0f, 1.0f, 0.0f, 1.0f),
            Vertex(1.0f, 1.0f, 1.0f, 0.0f, 0.0f),
            Vertex(-1.0f, 1.0f, 1.0f, 1.0f, 0.0f)
        };

        D3D11_SUBRESOURCE_DATA vertexBufferData = {0};
        vertexBufferData.pSysMem = v;
        vertexBufferData.SysMemPitch = 0;
        vertexBufferData.SysMemSlicePitch = 0;
        CD3D11_BUFFER_DESC vertexBufferDesc(sizeof(v), D3D11_BIND_VERTEX_BUFFER);
        DX::ThrowIfFailed(
            m_d3dDevice->CreateBuffer(
            &vertexBufferDesc,
            &vertexBufferData,
            &m_vertexBuffer
            )
            );

        DWORD indices[] =
        {
            // Front Face
            0,  2,  1,
            0,  3,  2,

        };

        m_indexCount = ARRAYSIZE(indices);

        D3D11_SUBRESOURCE_DATA indexBufferData = {0};
        indexBufferData.pSysMem = indices;
        indexBufferData.SysMemPitch = 0;
        indexBufferData.SysMemSlicePitch = 0;
        CD3D11_BUFFER_DESC indexBufferDesc(sizeof(indices), D3D11_BIND_INDEX_BUFFER);
        DX::ThrowIfFailed(
            m_d3dDevice->CreateBuffer(
            &indexBufferDesc,
            &indexBufferData,
            &m_indexBuffer
            )
            );
    });

    createCubeTask.then([this] ()
    {
        m_loadingComplete = true;
    });
}

void QuadRenderer::CreateWindowSizeDependentResources()
{
    Direct3DBase::CreateWindowSizeDependentResources();

    float aspectRatio = m_windowBounds.Width / m_windowBounds.Height;
    float fovAngleY = 60.0f * (XM_PI / 180.0f);

    if (aspectRatio < 1.0f)
    {
        fovAngleY /= aspectRatio;
    }

    XMStoreFloat4x4(
        &m_constantBufferData.projection,
        XMMatrixTranspose(
        XMMatrixPerspectiveFovRH(
        fovAngleY,
        aspectRatio,
        0.01f,
        100.0f
        )
        )
        );
}

void QuadRenderer::Update(float timeTotal, float timeDelta)
{
    (void) timeDelta; // Unused parameter.

    XMVECTOR X = XMVectorSet(0.0f, 0.0f, .3f, 0.0f);
    XMVECTOR Y = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    XMVECTOR Z = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);

    XMStoreFloat4x4(&m_constantBufferData.view, XMMatrixTranspose(XMMatrixLookAtLH(X, Y, Z)));
    XMStoreFloat4x4(&m_constantBufferData.model, XMMatrixTranspose(XMMatrixRotationY(timeTotal * XM_PIDIV4)));
}

void QuadRenderer::Render()
{
    Render(m_renderTargetView, m_depthStencilView);
}

void QuadRenderer::Render(Microsoft::WRL::ComPtr<ID3D11RenderTargetView> renderTargetView, Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depthStencilView)
{
    const float black[] = {0, 0, 0, 1.0 };

    m_d3dContext->ClearRenderTargetView(
        renderTargetView.Get(),
        black
        );

    m_d3dContext->ClearDepthStencilView(
        depthStencilView.Get(),
        D3D11_CLEAR_DEPTH,
        1.0f,
        0
        );

    if (m_SRV && m_loadingComplete)	// Only draw the cube once it is loaded (loading is asynchronous).
    {
        m_d3dContext->OMSetRenderTargets(
            1,
            renderTargetView.GetAddressOf(),
            depthStencilView.Get()
            );

        m_d3dContext->UpdateSubresource(
            m_constantBuffer.Get(),
            0,
            NULL,
            &m_constantBufferData,
            0,
            0
            );

        UINT stride = sizeof(Vertex);
        UINT offset = 0;

        m_d3dContext->IASetVertexBuffers(
            0,
            1,
            m_vertexBuffer.GetAddressOf(),
            &stride,
            &offset
            );

        m_d3dContext->IASetIndexBuffer(
            m_indexBuffer.Get(),
            DXGI_FORMAT_R32_UINT,
            0
            );

        m_d3dContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_d3dContext->IASetInputLayout(m_inputLayout.Get());

        m_d3dContext->VSSetShader(
            m_vertexShader.Get(),
            nullptr,
            0
            );

        m_d3dContext->VSSetConstantBuffers(
            0,
            1,
            m_constantBuffer.GetAddressOf()
            );

        m_d3dContext->PSSetShader(
            m_pixelShader.Get(),
            nullptr,
            0
            );

        m_d3dContext->PSSetShaderResources(0, 1, m_SRV.GetAddressOf());
        m_d3dContext->PSSetSamplers(0, 1, m_QuadsTexSamplerState.GetAddressOf());
        m_d3dContext->OMSetBlendState(m_Transparency.Get(), nullptr, 0xffffffff);
        m_d3dContext->RSSetState(CCWcullMode.Get());

        m_d3dContext->DrawIndexed(
            m_indexCount,
            0,
            0
            );
    }
}