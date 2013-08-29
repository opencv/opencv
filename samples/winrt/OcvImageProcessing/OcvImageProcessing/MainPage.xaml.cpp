//
// MainPage.xaml.cpp
// Implementation of the MainPage class.
//

#include "pch.h"
#include "MainPage.xaml.h"
#include <ppltasks.h>
#include <wrl\client.h>
#include <Robuffer.h>
using namespace OcvImageProcessing;

using namespace Microsoft::WRL;
using namespace concurrency;
using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::Storage::Streams;
using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Foundation::Collections;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Controls::Primitives;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::UI::Xaml::Input;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::UI::Xaml::Navigation;

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

Uri^ InputImageUri = ref new Uri(L"ms-appx:///Assets/Lena.png");

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=234238

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
}


void OcvImageProcessing::MainPage::Button_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
	RandomAccessStreamReference^ streamRef = RandomAccessStreamReference::CreateFromUri(InputImageUri);

    task<IRandomAccessStreamWithContentType^> (streamRef->OpenReadAsync()).
	then([](task<IRandomAccessStreamWithContentType^> thisTask)
    {
        IRandomAccessStreamWithContentType^ fileStream = thisTask.get();
        return BitmapDecoder::CreateAsync(fileStream);
    }).
    then([](task<BitmapDecoder^> thisTask)
    {
        BitmapDecoder^ decoder = thisTask.get();
        return decoder->GetFrameAsync(0);
    }).
    then([this](task<BitmapFrame^> thisTask)
    {
        BitmapFrame^ frame = thisTask.get();   

        // Save some information as fields
        frameWidth = frame->PixelWidth;
        frameHeight = frame->PixelHeight;

        return frame->GetPixelDataAsync();
    }).
    then([this](task<PixelDataProvider^> thisTask)
    {
        PixelDataProvider^ pixelProvider = thisTask.get();
		Platform::Array<byte>^ srcPixels = pixelProvider->DetachPixelData();
		
		cv::Mat inputImage(frameHeight, frameWidth, CV_8UC4, srcPixels->Data);
		unsigned char* dstPixels;
		
        // Create the WriteableBitmap 
        WriteableBitmap^ bitmap = ref new WriteableBitmap(frameWidth, frameHeight);

		// Get access to the pixels
		IBuffer^ buffer = bitmap->PixelBuffer;

		// Obtain IBufferByteAccess
		ComPtr<IBufferByteAccess> pBufferByteAccess;
		ComPtr<IUnknown> pBuffer((IUnknown*)buffer);
		pBuffer.As(&pBufferByteAccess);

		// Get pointer to pixel bytes
		pBufferByteAccess->Buffer(&dstPixels);
		cv::Mat outputImage(frameHeight, frameWidth, CV_8UC4, dstPixels);

		cv::Mat intermediateMat;
		cv::Canny(inputImage, intermediateMat, 80, 90);
		cv::cvtColor(intermediateMat, outputImage, CV_GRAY2BGRA);
		//cv::blur(inputImage, outputImage, cv::Size(3,3));

        // Set the bitmap to the Image element
		PreviewWidget->Source = bitmap;       
    });
}
