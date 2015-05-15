// videoio to XAML bridge for OpenCV

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

#include "opencv2\videoio\cap_winrt.hpp"
#include "cap_winrt_capture.hpp"
#include "cap_winrt_bridge.hpp"
#include "cap_winrt_video.hpp"

using namespace Windows::Foundation;
using namespace Windows::Media::Capture;
using namespace Windows::Media::MediaProperties;
using namespace Windows::Devices::Enumeration;

using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Microsoft::WRL;

using namespace Platform;
using namespace ::Concurrency;

using namespace ::std;

/***************************** VideoioBridge class ******************************/

// non-blocking
void VideoioBridge::requestForUIthreadAsync(int action, int widthp, int heightp)
{
    reporter.report(action);
}

VideoioBridge& VideoioBridge::getInstance()
{
    static VideoioBridge instance;
    return instance;
}

void VideoioBridge::swapInputBuffers()
{
    // TODO: already locked, check validity
    // lock_guard<mutex> lock(inputBufferMutex);
    swap(backInputPtr, frontInputPtr);
    //if (currentFrame != frameCounter)
    //{
    //    currentFrame = frameCounter;
    //    swap(backInputPtr, frontInputPtr);
    //}
}

void VideoioBridge::swapOutputBuffers()
{
    lock_guard<mutex> lock(outputBufferMutex);
    swap(frontOutputBuffer, backOutputBuffer);
}

void VideoioBridge::allocateOutputBuffers()
{
    frontOutputBuffer = ref new WriteableBitmap(width, height);
    backOutputBuffer = ref new WriteableBitmap(width, height);
}

void VideoioBridge::imshow()
{
    VideoioBridge::getInstance().swapOutputBuffers();
    VideoioBridge::getInstance().requestForUIthreadAsync(cv::UPDATE_IMAGE_ELEMENT);
}

// end