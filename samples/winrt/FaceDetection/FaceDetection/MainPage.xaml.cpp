//
// MainPage.xaml.cpp
// Implementation of the MainPage class.
//

#include "pch.h"
#include "MainPage.xaml.h"

#include <opencv2\imgproc\types_c.h>
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <Robuffer.h>

using namespace FaceDetection;

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


// Name of the resource classifier used to detect human faces (frontal)
cv::String face_cascade_name = "Assets/haarcascade_frontalface_alt.xml";

MainPage::MainPage()
{
    InitializeComponent();
}

void FaceDetection::MainPage::InitBtn_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    // load Image and Init recognizer
    cv::Mat image = cv::imread("Assets/group1.jpg");
    groupFaces = cv::Mat(image.rows, image.cols, CV_8UC4);
    cv::cvtColor(image, groupFaces, CV_BGR2BGRA);
    UpdateImage(groupFaces);

    if (!face_cascade.load(face_cascade_name)) {
        Windows::UI::Popups::MessageDialog("Couldn't load face detector \n").ShowAsync();
    }
}


void FaceDetection::MainPage::detectBtn_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    if (!groupFaces.empty()) {
        std::vector<cv::Rect> facesColl;
        cv::Mat frame_gray;

        cvtColor(groupFaces, frame_gray, CV_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);

        // Detect faces
        face_cascade.detectMultiScale(frame_gray, facesColl, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(1, 1));
        for (unsigned int i = 0; i < facesColl.size(); i++)
        {
            auto face = facesColl[i];
            cv::rectangle(groupFaces, face, cv::Scalar(0, 255, 255), 5);
        }

        UpdateImage(groupFaces);
    } else {
        Windows::UI::Popups::MessageDialog("Initialize image before processing \n").ShowAsync();
    }
}

void FaceDetection::MainPage::UpdateImage(const cv::Mat& image) {
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
    memcpy(dstPixels, image.data, image.step.buf[1] * image.cols*image.rows);

    // Set the bitmap to the Image element
    img1->Source = bitmap;
}