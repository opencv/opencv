#pragma once

#include "pch.h"
#include "BasicTimer.h"
#include "QuadRenderer.h"
#include <DrawingSurfaceNative.h>
#include <ppltasks.h>
#include <windows.storage.streams.h>
#include <memory>
#include <mutex>


#include <opencv2\imgproc\types_c.h>


namespace PhoneXamlDirect3DApp1Comp
{

public enum class OCVFilterType
{
    ePreview,
    eGray,
    eCanny,
    eBlur,
    eFindFeatures,
    eSepia,
    eNumOCVFilterTypes
};

class CameraCapturePreviewSink;
class CameraCaptureSampleSink;

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
    void SetAlgorithm(OCVFilterType type) { m_algorithm = type; };
    void UpdateFrame(byte* buffer, int width, int height);


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
    void StartCamera();
    void ProcessFrame();
    bool SwapFrames();

    QuadRenderer^ m_renderer;
    Windows::Foundation::Size m_renderResolution;
    OCVFilterType m_algorithm;
    bool m_contentDirty;
    std::shared_ptr<cv::Mat> m_backFrame;
    std::shared_ptr<cv::Mat> m_frontFrame;
    std::mutex m_mutex;

    Windows::Phone::Media::Capture::AudioVideoCaptureDevice ^pAudioVideoCaptureDevice;
    ICameraCaptureDeviceNative* pCameraCaptureDeviceNative;
    IAudioVideoCaptureDeviceNative* pAudioVideoCaptureDeviceNative;
    CameraCapturePreviewSink* pCameraCapturePreviewSink;
    CameraCaptureSampleSink* pCameraCaptureSampleSink;

    //void ApplyPreviewFilter(const cv::Mat& image);
    void ApplyGrayFilter(cv::Mat* mat);
    void ApplyCannyFilter(cv::Mat* mat);
    void ApplyBlurFilter(cv::Mat* mat);
    void ApplyFindFeaturesFilter(cv::Mat* mat);
    void ApplySepiaFilter(cv::Mat* mat);
};

class CameraCapturePreviewSink :
    public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
        ICameraCapturePreviewSink>
{
public:
    void SetDelegate(Direct3DInterop^ delegate)
    {
        m_Direct3dInterop = delegate;
    }

    IFACEMETHODIMP_(void) OnFrameAvailable(
        DXGI_FORMAT format,
        UINT width,
        UINT height,
        BYTE* pixels);

private:
    Direct3DInterop^ m_Direct3dInterop;
};

class CameraCaptureSampleSink :
    public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
        ICameraCaptureSampleSink>
{
public:
    void SetDelegate(Direct3DInterop^ delegate)
    {
        m_Direct3dInterop = delegate;
    }

    IFACEMETHODIMP_(void) OnSampleAvailable(
            ULONGLONG hnsPresentationTime,
            ULONGLONG hnsSampleDuration,
            DWORD cbSample,
            BYTE* pSample);

private:
    Direct3DInterop^ m_Direct3dInterop;
};

}
