//
// MainPage.xaml.cpp
// Implementation of the MainPage class.
//

#include "pch.h"
#include "MainPage.xaml.h"

#include <opencv2\imgproc\types_c.h>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <Robuffer.h>
#include <ppl.h>
#include <ppltasks.h>

using namespace PhoneTutorial;
using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Controls::Primitives;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::UI::Xaml::Input;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::UI::Xaml::Navigation;
using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Windows::Storage::Streams;
using namespace Microsoft::WRL;
using namespace Windows::ApplicationModel;

MainPage::MainPage()
{
    InitializeComponent();
}

/// <summary>
/// Invoked when this page is about to be displayed in a Frame.
/// </summary>
/// <param name="e">Event data that describes how this page was reached.  The Parameter
/// property is typically used to configure the page.</param>
void MainPage::OnNavigatedTo(NavigationEventArgs^ e)
{
    (void) e;	// Unused parameter
    LoadImage();
}

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw Exception::CreateException(hr);
    }
}

byte* GetPointerToPixelData(IBuffer^ buffer)
{
    // Cast to Object^, then to its underlying IInspectable interface.
    Object^ obj = buffer;
    ComPtr<IInspectable> insp(reinterpret_cast<IInspectable*>(obj));

    // Query the IBufferByteAccess interface.
    ComPtr<IBufferByteAccess> bufferByteAccess;
    ThrowIfFailed(insp.As(&bufferByteAccess));

    // Retrieve the buffer data.
    byte* pixels = nullptr;
    ThrowIfFailed(bufferByteAccess->Buffer(&pixels));
    return pixels;
}

void PhoneTutorial::MainPage::Process_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    (void) e;	// Unused parameter

    // get the pixels from the WriteableBitmap
    byte* pPixels = GetPointerToPixelData(m_bitmap->PixelBuffer);
    int height = m_bitmap->PixelHeight;
    int width = m_bitmap->PixelWidth;

    // create a matrix the size and type of the image
    cv::Mat mat(width, height, CV_8UC4);
    memcpy(mat.data, pPixels, 4 * height*width);

    // convert to grayscale
    cv::Mat intermediateMat;
    cv::cvtColor(mat, intermediateMat, CV_RGB2GRAY);

    // convert to BGRA
    cv::cvtColor(intermediateMat, mat, CV_GRAY2BGRA);

    // copy processed image back to the WriteableBitmap
    memcpy(pPixels, mat.data, 4 * height*width);

    // update the WriteableBitmap
    m_bitmap->Invalidate();
}

void PhoneTutorial::MainPage::LoadImage()
{
    Concurrency::task<Windows::Storage::StorageFile^> getFileTask(Package::Current->InstalledLocation->GetFileAsync(L"Lena.png"));

    auto getStreamTask = getFileTask.then(
        [](Windows::Storage::StorageFile ^storageFile)
    {
        return storageFile->OpenReadAsync();
    });

    getStreamTask.then(
        [this](Windows::Storage::Streams::IRandomAccessStreamWithContentType^ stream)
    {
        m_bitmap = ref new Windows::UI::Xaml::Media::Imaging::WriteableBitmap(1, 1);
        m_bitmap->SetSource(stream);
        image->Source = m_bitmap;
    });
}

void PhoneTutorial::MainPage::Reset_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    (void) e;	// Unused parameter
    LoadImage();
}
