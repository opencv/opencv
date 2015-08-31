//
// MainPage.xaml.cpp
// Implementation of the MainPage class.
//

#include "pch.h"
#include "MainPage.xaml.h"

// nb. path relative to modules/videoio/include
#include "../src/cap_winrt_bridge.hpp"
#include "../src/cap_winrt_video.hpp"
#include "opencv2/videoio/cap_winrt.hpp"

using namespace highgui_xaml;

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

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=234238

#include <ppl.h>
#include <ppltasks.h>
#include <concrt.h>
#include <agile.h>

using namespace ::concurrency;
using namespace ::Windows::Foundation;

using namespace Windows::UI::Xaml::Media::Imaging;


namespace highgui_xaml
{
using namespace cv;

    MainPage::MainPage()
    {
        InitializeComponent();

        grabberStarted = false;

        Window::Current->VisibilityChanged += ref new Windows::UI::Xaml::WindowVisibilityChangedEventHandler(this, &highgui_xaml::MainPage::OnVisibilityChanged);

        // set XAML elements
        VideoioBridge::getInstance().cvImage = cvImage;
        //VideoioBridge::getInstance().cvSlider = cvSlider;

        // handler
        //cvSlider->ValueChanged +=
        //    ref new RangeBaseValueChangedEventHandler(this, &MainPage::cvSlider_ValueChanged);

        auto asyncTask = TaskWithProgressAsync();
        asyncTask->Progress = ref new AsyncActionProgressHandler<int>([this](IAsyncActionWithProgress<int>^ act, int progress)
        {
            int action = progress;

            // these actions will be processed on the UI thread asynchronously
            switch (action)
            {
            case OPEN_CAMERA:
                {
                    int device = VideoioBridge::getInstance().getDeviceIndex();
                    int width = VideoioBridge::getInstance().getWidth();
                    int height = VideoioBridge::getInstance().getHeight();

                    // buffers must alloc'd on UI thread
                    VideoioBridge::getInstance().allocateBuffers(width, height);

                    // nb. video capture device init must be done on UI thread;
                    // code is located in the OpenCV Highgui DLL, class Video
                    if (!grabberStarted)
                    {
                        grabberStarted = true;
						Video::getInstance().initGrabber(device, width, height);
                    }
                }
                break;
            case CLOSE_CAMERA:
                Video::getInstance().closeGrabber();
                break;
            case UPDATE_IMAGE_ELEMENT:
                {
                    // copy output Mat to WBM
                     Video::getInstance().CopyOutput();

                    // set XAML image element with image WBM
                    VideoioBridge::getInstance().cvImage->Source = VideoioBridge::getInstance().backOutputBuffer;
                }
                break;
            //case SHOW_TRACKBAR:
            //    cvSlider->Visibility = Windows::UI::Xaml::Visibility::Visible;
            //    break;
            }
        });


    }

    //void MainPage::cvSlider_ValueChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs^ e)
    //{
    //    sliderChanged1(e->NewValue);
    //}
}

// nb. implemented in main.cpp
void cvMain();

// set the reporter method for the HighguiAssist singleton
// start the main OpenCV as an async thread in WinRT
IAsyncActionWithProgress<int>^ MainPage::TaskWithProgressAsync()
{
    return create_async([this](progress_reporter<int> reporter)
    {
        VideoioBridge::getInstance().setReporter(reporter);
        cvMain();
    });
}

void highgui_xaml::MainPage::OnVisibilityChanged(Platform::Object ^sender,
    Windows::UI::Core::VisibilityChangedEventArgs ^e)
{
    if (e->Visible)
    {
        // only start the grabber if the camera was opened in OpenCV
        if (VideoioBridge::getInstance().backInputPtr != nullptr)
        {
            if (grabberStarted) return;

            int device = VideoioBridge::getInstance().getDeviceIndex();
            int width = VideoioBridge::getInstance().getWidth();
            int height = VideoioBridge::getInstance().getHeight();

             Video::getInstance().initGrabber(device, width, height);
        }
    }
    else
    {
        grabberStarted = false;
		Video::getInstance().closeGrabber();
    }
}
