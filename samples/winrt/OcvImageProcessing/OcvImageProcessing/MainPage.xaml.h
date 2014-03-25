//
// MainPage.xaml.h
// Declaration of the MainPage class.
//

#pragma once

#include "MainPage.g.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>

namespace OcvImageProcessing
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public ref class MainPage sealed
    {
    public:
        MainPage();

    protected:
        virtual void OnNavigatedTo(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;

    private:
        static const int PREVIEW  = 0;
        static const int GRAY     = 1;
        static const int CANNY    = 2;
        static const int BLUR     = 3;
        static const int FEATURES = 4;
        static const int SEPIA    = 5;

        void Button_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
        cv::Mat ApplyGrayFilter(const cv::Mat& image);
        cv::Mat ApplyCannyFilter(const cv::Mat& image);
        cv::Mat ApplyBlurFilter(const cv::Mat& image);
        cv::Mat ApplyFindFeaturesFilter(const cv::Mat& image);
        cv::Mat ApplySepiaFilter(const cv::Mat& image);

        void UpdateImage(const cv::Mat& image);

        cv::Mat Lena;
        unsigned int frameWidth, frameHeight;
    };
}
