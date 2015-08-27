// Copyright (c) 2015, Microsoft Open Technologies, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of Microsoft Open Technologies, Inc. nor the names
//   of its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "precomp.hpp"

#include "VideoIo/src/cap_winrt_bridge.hpp"
#include "VideoIo/src/cap_winrt_video.hpp"
#include "opencv2/VideoIo/cap_winrt.hpp"

#include "opencvrt/videoio.hpp"

// nb. path relative to modules/VideoIo/include

using namespace ::concurrency;
using namespace ::Windows::Foundation;
using namespace ::Windows::UI::Xaml::Controls;
// using namespace cv;
using namespace cvRT;

VideoIo::VideoIo()
{
}

void VideoIo::Initialize()
{
	auto asyncTask = create_async([this](progress_reporter<int> reporter)
	{
        VideoioBridge::getInstance().setReporter(reporter);
	});

	asyncTask->Progress = ref new AsyncActionProgressHandler<int>([this](IAsyncActionWithProgress<int>^ act, int progress)
	{
		int action = progress;

		// these actions will be processed on the UI thread asynchronously
		switch (action)
		{
        case cv::OPEN_CAMERA:
		{
            int device = VideoioBridge::getInstance().getDeviceIndex();
            int width = VideoioBridge::getInstance().getWidth();
            int height = VideoioBridge::getInstance().getHeight();

            // buffers must alloc'd on UI thread
            VideoioBridge::getInstance().allocateBuffers(width, height);

            // nb. video capture device init must be done on UI thread;
            // code is located in the OpenCV Highgui DLL, class Video
            if (!frameGrabberStarted)
            {
                frameGrabberStarted = true;
                Video::getInstance().initGrabber(device, width, height);
            }
		}
		break;
		case cv::CLOSE_CAMERA:
            Video::getInstance().closeGrabber();
			break;
		case cv::UPDATE_IMAGE_ELEMENT:
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

void VideoIo::SetImage(Windows::UI::Xaml::Controls::Image^ cvImage)
{
    VideoioBridge::getInstance().cvImage = cvImage;
}

void VideoIo::StartCapture()
{
	vidCap.open(0);
}


void VideoIo::StopCapture()
{
	vidCap.release();
}

void VideoIo::GetFrame(Mat^ frame)
{
	// tbd can't loop forever here.
	while (1)
	{
		vidCap >> frame->Get();

		// tbd these are from original sample.  need to look more into this.
		if (!vidCap.grab())
			continue;

		// ditto as above.
		if (frame->Get().total() == 0)
			continue;

		break;
	}

	OutputDebugString(L"Frame obtained");
}

void VideoIo::ShowFrame(Mat^ frame)
{
    VideoioBridge::getInstance().imshow();
}