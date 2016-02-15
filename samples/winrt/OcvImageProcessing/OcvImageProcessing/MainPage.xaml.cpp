//
// MainPage.xaml.cpp
// Implementation of the MainPage class.
//

#include "pch.h"
#include "MainPage.xaml.h"
#include <ppltasks.h>
#include <wrl\client.h>
#include <Robuffer.h>
#include <vector>
#include <opencv2\imgproc\types_c.h>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\core.hpp>

#include <windows.storage.h>

using namespace OcvImageProcessing;

using namespace Microsoft::WRL;
using namespace concurrency;
using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::Storage::Streams;
using namespace Windows::Storage;
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

Uri^ InputImageUri = ref new Uri(L"ms-appx:///Assets/Lena.png");

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=234238

MainPage::MainPage()
{
    InitializeComponent();

#ifdef __OPENCV_IMGCODECS_HPP__

    // Image loading OpenCV way ... way more simple
    cv::Mat image = cv::imread("Assets/Lena.png");
    Lena = cv::Mat(image.rows, image.cols, CV_8UC4);
    cvtColor(image, Lena, CV_BGR2BGRA);
    UpdateImage(Lena);

#else

    // Image loading WinRT way
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
        Lena = cv::Mat(frameHeight, frameWidth, CV_8UC4);
        memcpy(Lena.data, srcPixels->Data, 4*frameWidth*frameHeight);
        UpdateImage(Lena);
    });

#endif
}

/// <summary>
/// Temporary file creation example. Will be created in WinRT application temporary directory
/// which usually is "C:\Users\{username}\AppData\Local\Packages\{package_id}\TempState\{random_name}.{suffix}"
/// </summary>
/// <param name="suffix">Temporary file suffix, e.g. "tmp"</param>
std::string OcvImageProcessing::MainPage::CreateTempFile(const std::string &suffix) {
    return cv::tempfile(suffix.c_str());
}

/// <summary>
/// Creating/writing a file in the application local directory
/// </summary>
/// <param name="path">Image to save</param>
bool OcvImageProcessing::MainPage::SaveImage(cv::Mat image) {
    StorageFolder^ localFolderRT = ApplicationData::Current->LocalFolder;
    cv::String localFile = ConvertPath(ApplicationData::Current->LocalFolder->Path) + "\\Lena.png";

    return cv::imwrite(localFile, image);
}

/// <summary>
/// Getting std::string from managed string via std::wstring.
/// Provides an example of three ways to do it.
/// Can't use this one: https://msdn.microsoft.com/en-us/library/bb384865.aspx, not available on WinRT.
/// </summary>
/// <param name="path">Path to be converted</param>
cv::String OcvImageProcessing::MainPage::ConvertPath(Platform::String^ path) {
    std::wstring localPathW(path->Begin());

    // Opt #1
    //std::string localPath(localPathW.begin(), localPathW.end());

    // Opt #2
    //std::string localPath(StrToWStr(localPathW));

    // Opt #3
    size_t outSize = localPathW.length() + 1;
    char* localPathC = new char[outSize];
    size_t charsConverted = 0;
    wcstombs_s(&charsConverted, localPathC, outSize, localPathW.c_str(), localPathW.length());
    cv::String localPath(localPathC);

    // Implicit conversion from std::string to cv::String
    return localPath;
}

std::string OcvImageProcessing::MainPage::StrToWStr(const std::wstring &input) {
    if (input.empty()) {
        return std::string();
    }

    int size = WideCharToMultiByte(CP_UTF8, 0, &input[0], (int)input.size(), NULL, 0, NULL, NULL);
    std::string result(size, 0);

    WideCharToMultiByte(CP_UTF8, 0, &input[0], (int)input.size(), &result[0], size, NULL, NULL);

    return result;
}

/// <summary>
/// Invoked when this page is about to be displayed in a Frame.
/// </summary>
/// <param name="e">Event data that describes how this page was reached.  The Parameter
/// property is typically used to configure the page.</param>
void MainPage::OnNavigatedTo(NavigationEventArgs^ e)
{
    (void) e;    // Unused parameter
}

void OcvImageProcessing::MainPage::UpdateImage(const cv::Mat& image)
{
    // Create the WriteableBitmap
    WriteableBitmap^ bitmap = ref new WriteableBitmap(image.cols, image.rows);

    // Get access to the pixels
    IBuffer^ buffer = bitmap->PixelBuffer;
    unsigned char* dstPixels;

    // Obtain IBufferByteAccess
    ComPtr<IBufferByteAccess> pBufferByteAccess;
    ComPtr<IInspectable> pBuffer((IInspectable*)buffer);
    pBuffer.As(&pBufferByteAccess);

    // Get pointer to pixel bytes
    pBufferByteAccess->Buffer(&dstPixels);
    memcpy(dstPixels, image.data, image.step.buf[1]*image.cols*image.rows);

    // Set the bitmap to the Image element
    PreviewWidget->Source = bitmap;
}


cv::Mat OcvImageProcessing::MainPage::ApplyGrayFilter(const cv::Mat& image)
{
    cv::Mat result;
    cv::Mat intermediateMat;
    cv::cvtColor(image, intermediateMat, CV_RGBA2GRAY);
    cv::cvtColor(intermediateMat, result, CV_GRAY2BGRA);
    return result;
}

cv::Mat OcvImageProcessing::MainPage::ApplyCannyFilter(const cv::Mat& image)
{
    cv::Mat result;
    cv::Mat intermediateMat;
    cv::Canny(image, intermediateMat, 80, 90);
    cv::cvtColor(intermediateMat, result, CV_GRAY2BGRA);
    return result;
}

cv::Mat OcvImageProcessing::MainPage::ApplyBlurFilter(const cv::Mat& image)
{
    cv::Mat result;
    cv::blur(image, result, cv::Size(3,3));
    return result;
}

cv::Mat OcvImageProcessing::MainPage::ApplyFindFeaturesFilter(const cv::Mat& image)
{
    cv::Mat result;
    cv::Mat intermediateMat;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(50);
    std::vector<cv::KeyPoint> features;

    image.copyTo(result);
    cv::cvtColor(image, intermediateMat, CV_RGBA2GRAY);
    detector->detect(intermediateMat, features);

    for( unsigned int i = 0; i < std::min(features.size(), (size_t)50); i++ )
    {
        const cv::KeyPoint& kp = features[i];
        cv::circle(result, cv::Point((int)kp.pt.x, (int)kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

    return result;
}

cv::Mat OcvImageProcessing::MainPage::ApplySepiaFilter(const cv::Mat& image)
{
    const float SepiaKernelData[16] =
    {
        /* B */0.131f, 0.534f, 0.272f, 0.f,
        /* G */0.168f, 0.686f, 0.349f, 0.f,
        /* R */0.189f, 0.769f, 0.393f, 0.f,
        /* A */0.000f, 0.000f, 0.000f, 1.f
    };
    const cv::Mat SepiaKernel(4, 4, CV_32FC1, (void*)SepiaKernelData);
    cv::Mat result;
    cv::transform(image, result, SepiaKernel);
    return result;
}

void OcvImageProcessing::MainPage::Button_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    switch(FilterTypeWidget->SelectedIndex)
    {
    case PREVIEW:
        UpdateImage(Lena);
        break;
    case GRAY:
        UpdateImage(ApplyGrayFilter(Lena));
        break;
    case CANNY:
        UpdateImage(ApplyCannyFilter(Lena));
        break;
    case BLUR:
        UpdateImage(ApplyBlurFilter(Lena));
        break;
    case FEATURES:
        UpdateImage(ApplyFindFeaturesFilter(Lena));
        break;
    case SEPIA:
        UpdateImage(ApplySepiaFilter(Lena));
        break;
    default:
        UpdateImage(Lena);
    }
}
