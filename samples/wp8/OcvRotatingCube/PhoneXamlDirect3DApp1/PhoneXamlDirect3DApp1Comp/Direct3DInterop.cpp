#include "pch.h"
#include "Direct3DInterop.h"
#include "Direct3DContentProvider.h"
#include <windows.storage.streams.h>
#include <wrl.h>
#include <robuffer.h>
#include <opencv2\imgproc\types_c.h>

using namespace Windows::Storage::Streams;
using namespace Microsoft::WRL;
using namespace Windows::Foundation;
using namespace Windows::UI::Core;
using namespace Microsoft::WRL;
using namespace Windows::Phone::Graphics::Interop;
using namespace Windows::Phone::Input::Interop;

namespace PhoneXamlDirect3DApp1Comp
{
    void Direct3DInterop::ApplyGrayFilter(const cv::Mat& image)
    {
        cv::Mat intermediateMat;
        cv::cvtColor(image, intermediateMat, COLOR_RGBA2GRAY);
        cv::cvtColor(intermediateMat, image, COLOR_GRAY2BGRA);
    }

    void Direct3DInterop::ApplyCannyFilter(const cv::Mat& image)
    {
        cv::Mat intermediateMat;
        cv::Canny(image, intermediateMat, 80, 90);
        cv::cvtColor(intermediateMat, image, COLOR_GRAY2BGRA);
    }

    void Direct3DInterop::ApplySepiaFilter(const cv::Mat& image)
    {
        const float SepiaKernelData[16] =
        {
            /* B */0.131f, 0.534f, 0.272f, 0.f,
            /* G */0.168f, 0.686f, 0.349f, 0.f,
            /* R */0.189f, 0.769f, 0.393f, 0.f,
            /* A */0.000f, 0.000f, 0.000f, 1.f
        };

        const cv::Mat SepiaKernel(4, 4, CV_32FC1, (void*)SepiaKernelData);
        cv::transform(image, image, SepiaKernel);
    }

    Direct3DInterop::Direct3DInterop() :
        m_timer(ref new BasicTimer())
    {
    }

    IDrawingSurfaceContentProvider^ Direct3DInterop::CreateContentProvider()
    {
        ComPtr<Direct3DContentProvider> provider = Make<Direct3DContentProvider>(this);
        return reinterpret_cast<IDrawingSurfaceContentProvider^>(provider.Detach());
    }

    // IDrawingSurfaceManipulationHandler
    void Direct3DInterop::SetManipulationHost(DrawingSurfaceManipulationHost^ manipulationHost)
    {
        manipulationHost->PointerPressed +=
            ref new TypedEventHandler<DrawingSurfaceManipulationHost^, PointerEventArgs^>(this, &Direct3DInterop::OnPointerPressed);

        manipulationHost->PointerMoved +=
            ref new TypedEventHandler<DrawingSurfaceManipulationHost^, PointerEventArgs^>(this, &Direct3DInterop::OnPointerMoved);

        manipulationHost->PointerReleased +=
            ref new TypedEventHandler<DrawingSurfaceManipulationHost^, PointerEventArgs^>(this, &Direct3DInterop::OnPointerReleased);
    }

    void Direct3DInterop::RenderResolution::set(Windows::Foundation::Size renderResolution)
    {
        if (renderResolution.Width  != m_renderResolution.Width ||
            renderResolution.Height != m_renderResolution.Height)
        {
            m_renderResolution = renderResolution;

            if (m_renderer)
            {
                m_renderer->UpdateForRenderResolutionChange(m_renderResolution.Width, m_renderResolution.Height);
                RecreateSynchronizedTexture();
            }
        }
    }

    // Event Handlers

    void Direct3DInterop::OnPointerPressed(DrawingSurfaceManipulationHost^ sender, PointerEventArgs^ args)
    {
        // Insert your code here.
    }

    void Direct3DInterop::OnPointerMoved(DrawingSurfaceManipulationHost^ sender, PointerEventArgs^ args)
    {
        // Insert your code here.
    }

    void Direct3DInterop::OnPointerReleased(DrawingSurfaceManipulationHost^ sender, PointerEventArgs^ args)
    {
        // Insert your code here.
    }

    // Interface With Direct3DContentProvider
    HRESULT Direct3DInterop::Connect(_In_ IDrawingSurfaceRuntimeHostNative* host)
    {
        m_renderer = ref new CubeRenderer();
        m_renderer->Initialize();
        m_renderer->UpdateForWindowSizeChange(WindowBounds.Width, WindowBounds.Height);
        m_renderer->UpdateForRenderResolutionChange(m_renderResolution.Width, m_renderResolution.Height);

        // Restart timer after renderer has finished initializing.
        m_timer->Reset();

        return S_OK;
    }

    void Direct3DInterop::Disconnect()
    {
        m_renderer = nullptr;
    }

    HRESULT Direct3DInterop::PrepareResources(_In_ const LARGE_INTEGER* presentTargetTime, _Out_ BOOL* contentDirty)
    {
        *contentDirty = true;

        return S_OK;
    }

    HRESULT Direct3DInterop::GetTexture(_In_ const DrawingSurfaceSizeF* size, _Out_ IDrawingSurfaceSynchronizedTextureNative** synchronizedTexture, _Out_ DrawingSurfaceRectF* textureSubRectangle)
    {
        m_timer->Update();
        m_renderer->Update(m_timer->Total, m_timer->Delta);
        m_renderer->Render();

        RequestAdditionalFrame();

        return S_OK;
    }

    ID3D11Texture2D* Direct3DInterop::GetTexture()
    {
        return m_renderer->GetTexture();
    }

    void Direct3DInterop::CreateTexture(const Platform::Array<int>^  buffer,int width,int height, OCVFilterType filter)
    {
        if (m_renderer)
        {
            cv::Mat Lena = cv::Mat(height, width, CV_8UC4);
            memcpy(Lena.data, buffer->Data, 4 * height*width);

            switch (filter)
            {
                case OCVFilterType::ePreview:
                    break;

                case OCVFilterType::eGray:
                    ApplyGrayFilter(Lena);
                    break;

                case OCVFilterType::eCanny:
                    ApplyCannyFilter(Lena);
                    break;

                case OCVFilterType::eSepia:
                    ApplySepiaFilter(Lena);
                    break;
            }

            m_renderer->CreateTextureFromByte(Lena.data, width, height);
        }
    }

    byte* GetPointerToPixelData( Windows::Storage::Streams::IBuffer ^ pixelBuffer)
    {
        // Query the IBufferByteAccess interface.
        ComPtr<IBufferByteAccess> bufferByteAccess;
        reinterpret_cast<IInspectable*>( pixelBuffer)->QueryInterface(IID_PPV_ARGS(&bufferByteAccess));

        // Retrieve the buffer data.
        byte* pixels = nullptr;
        bufferByteAccess->Buffer(&pixels);
        return pixels;
    }

}
