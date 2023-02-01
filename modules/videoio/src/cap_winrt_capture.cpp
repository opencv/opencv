// Capture support for WinRT

// Copyright (c) Microsoft Open Technologies, Inc.
// All rights reserved.
//
// (3 - clause BSD License)
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
// following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
// following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
// promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include "precomp.hpp"
#include "cap_winrt_capture.hpp"
#include "cap_winrt_bridge.hpp"
#include "cap_winrt_video.hpp"
#include <opencv2\videoio\cap_winrt.hpp>

using namespace Windows::Foundation;
using namespace Windows::Media::Capture;
using namespace Windows::Media::MediaProperties;
using namespace Windows::Devices::Enumeration;

using namespace Platform;

using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Microsoft::WRL;

using namespace ::std;

namespace cv {

    /******************************* exported API functions **************************************/

    template <typename ...Args>
    void winrt_startMessageLoop(std::function<void(Args...)>&& callback, Args... args)
    {
        auto asyncTask = ::concurrency::create_async([=](::concurrency::progress_reporter<int> reporter)
        {
            VideoioBridge::getInstance().setReporter(reporter);

            // frame reading loop
            callback(args...);
        });

        asyncTask->Progress = ref new AsyncActionProgressHandler<int>([=](IAsyncActionWithProgress<int>^ act, int progress)
        {
            int action = progress;

            // these actions will be processed on the UI thread asynchronously
            switch (action)
            {
            case OPEN_CAMERA:
                VideoioBridge::getInstance().openCamera();
                break;
            case CLOSE_CAMERA:
                Video::getInstance().closeGrabber();
                break;
            case UPDATE_IMAGE_ELEMENT:
                VideoioBridge::getInstance().updateFrameContainer();
                break;
            }
        });
    }

    template <typename ...Args>
    void winrt_startMessageLoop(void callback(Args...), Args... args)
    {
        winrt_startMessageLoop(std::function<void(Args...)>(callback), args...);
    }

    void winrt_onVisibilityChanged(bool visible)
    {
        if (visible)
        {
            VideoioBridge& bridge = VideoioBridge::getInstance();

            // only start the grabber if the camera was opened in OpenCV
            if (bridge.backInputPtr != nullptr)
            {
                if (Video::getInstance().isStarted()) return;

                int device = bridge.getDeviceIndex();
                int width = bridge.getWidth();
                int height = bridge.getHeight();

                Video::getInstance().initGrabber(device, width, height);
            }
        } else
        {
            //grabberStarted = false;
            Video::getInstance().closeGrabber();
        }
    }

    void winrt_imshow()
    {
        VideoioBridge::getInstance().imshow();
    }

    void winrt_setFrameContainer(::Windows::UI::Xaml::Controls::Image^ image)
    {
        VideoioBridge::getInstance().cvImage = image;
    }

    /********************************* VideoCapture_WinRT class ****************************/

    VideoCapture_WinRT::VideoCapture_WinRT(int device) : started(false)
    {
        VideoioBridge::getInstance().setDeviceIndex(device);
    }

    bool VideoCapture_WinRT::isOpened() const
    {
        return true; // started;
    }

    // grab a frame:
    // this will NOT block per spec
    // should be called on the image processing thread, not the UI thread
    bool VideoCapture_WinRT::grabFrame()
    {
        // if device is not started we must return true so retrieveFrame() is called to start device
        // nb. we cannot start the device here because we do not know the size of the input Mat
        if (!started) return true;

        if (VideoioBridge::getInstance().bIsFrameNew)
        {
            return true;
        }

        // nb. if blocking is to be added:
        // unique_lock<mutex> lock(VideoioBridge::getInstance().frameReadyMutex);
        // VideoioBridge::getInstance().frameReadyEvent.wait(lock);
        return false;
    }

    // should be called on the image processing thread after grabFrame
    // see VideoCapture::read
    bool VideoCapture_WinRT::retrieveFrame(int channel, cv::OutputArray outArray)
    {
        if (!started) {

            int width, height;
            width = outArray.size().width;
            height = outArray.size().height;
            if (width == 0) width = 640;
            if (height == 0) height = 480;

            VideoioBridge::getInstance().setWidth(width);
            VideoioBridge::getInstance().setHeight(height);

            // nb. Mats will be alloc'd on UI thread

            // request device init on UI thread - this does not block, and is async
            VideoioBridge::getInstance().requestForUIthreadAsync(OPEN_CAMERA);

            started = true;
            return false;
        }

        if (!started) return false;

        return VideoioBridge::getInstance().bIsFrameNew;
    }


    bool VideoCapture_WinRT::setProperty(int property_id, double value)
    {
        switch (property_id)
        {
        case CAP_PROP_FRAME_WIDTH:
            size.width = (int)value;
            break;
        case CAP_PROP_FRAME_HEIGHT:
            size.height = (int)value;
            break;
        default:
            return false;
        }
        return true;
    }

Ptr<IVideoCapture> create_WRT_capture(int device)
{
    return makePtr<VideoCapture_WinRT>(device);
}

}

// end
