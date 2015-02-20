#pragma once

#include "pch.h"
#include "BasicTimer.h"
#include "CubeRenderer.h"
#include <DrawingSurfaceNative.h>
#include <ppltasks.h>
#include <windows.storage.streams.h>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>

namespace PhoneXamlDirect3DApp1Comp
{

public enum class OCVFilterType
{
    ePreview,
    eGray,
    eCanny,
    eSepia,
    eNumOCVFilterTypes
};

public delegate void RequestAdditionalFrameHandler();
public delegate void RecreateSynchronizedTextureHandler();

[Windows::Foundation::Metadata::WebHostHidden]
public ref class Direct3DInterop sealed : public Windows::Phone::Input::Interop::IDrawingSurfaceManipulationHandler
{
public:
    Direct3DInterop();

    Windows::Phone::Graphics::Interop::IDrawingSurfaceContentProvider^ CreateContentProvider();

    // IDrawingSurfaceManipulationHandler
    virtual void SetManipulationHost(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ manipulationHost);

    event RequestAdditionalFrameHandler^ RequestAdditionalFrame;
    event RecreateSynchronizedTextureHandler^ RecreateSynchronizedTexture;

    property Windows::Foundation::Size WindowBounds;
    property Windows::Foundation::Size NativeResolution;
    property Windows::Foundation::Size RenderResolution
    {
        Windows::Foundation::Size get(){ return m_renderResolution; }
        void set(Windows::Foundation::Size renderResolution);
    }
    void CreateTexture(const Platform::Array<int>^ buffer, int with, int height, OCVFilterType filter);


protected:
    // Event Handlers
    void OnPointerPressed(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ sender, Windows::UI::Core::PointerEventArgs^ args);
    void OnPointerMoved(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ sender, Windows::UI::Core::PointerEventArgs^ args);
    void OnPointerReleased(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ sender, Windows::UI::Core::PointerEventArgs^ args);

internal:
    HRESULT STDMETHODCALLTYPE Connect(_In_ IDrawingSurfaceRuntimeHostNative* host);
    void STDMETHODCALLTYPE Disconnect();
    HRESULT STDMETHODCALLTYPE PrepareResources(_In_ const LARGE_INTEGER* presentTargetTime, _Out_ BOOL* contentDirty);
    HRESULT STDMETHODCALLTYPE GetTexture(_In_ const DrawingSurfaceSizeF* size, _Out_ IDrawingSurfaceSynchronizedTextureNative** synchronizedTexture, _Out_ DrawingSurfaceRectF* textureSubRectangle);
    ID3D11Texture2D* GetTexture();

private:
    CubeRenderer^ m_renderer;
    BasicTimer^ m_timer;
    Windows::Foundation::Size m_renderResolution;

    void ApplyGrayFilter(const cv::Mat& image);
    void ApplyCannyFilter(const cv::Mat& image);
    void ApplySepiaFilter(const cv::Mat& image);

    void UpdateImage(const cv::Mat& image);

    cv::Mat Lena;
    unsigned int frameWidth, frameHeight;
};

}
