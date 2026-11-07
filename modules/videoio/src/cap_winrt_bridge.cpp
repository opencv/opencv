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
void VideoioBridge::requestForUIthreadAsync(int action)
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

// performed on UI thread
void VideoioBridge::allocateBuffers(int width_, int height_)
{
    // allocate input Mats (bgra8 = CV_8UC4, RGB24 = CV_8UC3)
    frontInputMat.create(height_, width_, CV_8UC3);
    backInputMat.create(height_, width_, CV_8UC3);

    frontInputPtr = frontInputMat.ptr(0);
    backInputPtr = backInputMat.ptr(0);

    allocateOutputBuffers();
}

// performed on UI thread
bool VideoioBridge::openCamera()
{
    // buffers must alloc'd on UI thread
    allocateBuffers(width, height);

    // nb. video capture device init must be done on UI thread;
    if (!Video::getInstance().isStarted())
    {
        Video::getInstance().initGrabber(deviceIndex, width, height);
        return true;
    }

    return false;
}

// nb on UI thread
void VideoioBridge::updateFrameContainer()
{
    // copy output Mat to WBM
    Video::getInstance().CopyOutput();

    // set XAML image element with image WBM
    cvImage->Source = backOutputBuffer;
}

void VideoioBridge::imshow()
{
    swapOutputBuffers();
    requestForUIthreadAsync(cv::UPDATE_IMAGE_ELEMENT);
}

int VideoioBridge::getDeviceIndex()
{
    return deviceIndex;
}

void VideoioBridge::setDeviceIndex(int index)
{
    deviceIndex = index;
}

int VideoioBridge::getWidth()
{
    return width;
}

int VideoioBridge::getHeight()
{
    return height;
}

void VideoioBridge::setWidth(int _width)
{
    width = _width;
}

void VideoioBridge::setHeight(int _height)
{
    height = _height;
}

// end
