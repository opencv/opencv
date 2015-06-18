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
#include <opencv2\objdetect.hpp>
#include <Robuffer.h>

using namespace RaspberryCV;

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

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

MainPage::MainPage()
{
	InitializeComponent();
}

void  RaspberryCV::MainPage::UpdateImage(const cv::Mat& image)
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
	memcpy(dstPixels, image.data, image.step.buf[1] * image.cols*image.rows);

	// Set the bitmap to the Image element
	img1->Source = bitmap;
}

void RaspberryCV::MainPage::btn1_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
	cv::Mat image = cv::imread("Assets/grpPC1.jpg");
	Lena = cv::Mat(image.rows, image.cols, CV_8UC4);
	cv::cvtColor(image, Lena, CV_BGR2BGRA);
	UpdateImage(Lena);

}


void RaspberryCV::MainPage::btn2_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
	//Canny filter
	cv::Mat result;
	cv::Mat intermediateMat;
	cv::Canny(Lena, intermediateMat, 80, 90);
	cv::cvtColor(intermediateMat, result, CV_GRAY2BGRA);
	UpdateImage(result);
}

cv::String face_cascade_name = "Assets/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;

cv::String body_cascade_name = "Assets/haarcascade_fullbody.xml";
cv::CascadeClassifier body_cascade;

void internalDetectObjects(cv::Mat& inputImg, std::vector<cv::Rect> & objectVector, std::vector<cv::Rect> & objectVectorBodies)
{
	cv::Mat frame_gray;

	cvtColor(inputImg, frame_gray, CV_BGR2GRAY);
	cv::equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, objectVector, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
	//detect bodies
	body_cascade.detectMultiScale(frame_gray, objectVectorBodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 300));
}


void RaspberryCV::MainPage::btn3_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
	if (!face_cascade.load(face_cascade_name)) {
		printf("Couldnt load Face detector '%s'\n", face_cascade_name);
		exit(1);
	}

	if (!body_cascade.load(body_cascade_name)) {
		printf("Couldnt load Body detector '%s'\n", body_cascade_name);
		exit(1);
	}

	cv::Mat frame = cv::imread("Assets/grpPC1.jpg");

	if (frame.empty())
		return;

	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> bodies;

	internalDetectObjects(frame, faces, bodies);

	for (unsigned int i = 0; i < faces.size(); i++)
	{
		auto face = faces[i];

		cv::rectangle(Lena, face, cv::Scalar(0,0, 255, 255), 5);

	}

	for (unsigned int i = 0; i < bodies.size(); i++)
	{
		auto body = bodies[i];

		cv::rectangle(Lena, body, cv::Scalar(0,0, 0, 255), 5);

	}

	UpdateImage(Lena);


}



