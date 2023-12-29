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

#pragma once

// this header is included in the XAML App, so it cannot include any
// OpenCV headers, or a static assert will be raised

#include <ppl.h>
#include <ppltasks.h>
#include <concrt.h>
#include <agile.h>
#include <opencv2/core.hpp>

#include <mutex>
#include <memory>
#include <atomic>
#include <functional>


// Class VideoioBridge (singleton) is needed because the interface for
// VideoCapture_WinRT in cap_winrt_capture.hpp is fixed by OpenCV.
class VideoioBridge
{
public:

    static VideoioBridge& getInstance();

    // call after initialization
    void    setReporter(Concurrency::progress_reporter<int> pr) { reporter = pr; }

    // to be called from cvMain via cap_winrt on bg thread - non-blocking (async)
    void    requestForUIthreadAsync(int action);

    // TODO: modify in window.cpp: void cv::imshow( const String& winname, InputArray _img )
    void    imshow(/*cv::InputArray matToShow*/);   // shows Mat in the cvImage element
    void    swapInputBuffers();
    void    allocateOutputBuffers();
    void    swapOutputBuffers();
    void    updateFrameContainer();
    bool    openCamera();
    void    allocateBuffers(int width, int height);

    int     getDeviceIndex();
    void    setDeviceIndex(int index);
    int     getWidth();
    void    setWidth(int width);
    int     getHeight();
    void    setHeight(int height);

    std::atomic<bool>           bIsFrameNew;
    std::mutex                  inputBufferMutex;   // input is double buffered
    unsigned char *             frontInputPtr;      // OpenCV reads this
    unsigned char *             backInputPtr;       // Video grabber writes this
    std::atomic<unsigned long>  frameCounter;
    unsigned long               currentFrame;

    std::mutex                  outputBufferMutex;  // output is double buffered
    Windows::UI::Xaml::Media::Imaging::WriteableBitmap^ frontOutputBuffer;  // OpenCV write this
    Windows::UI::Xaml::Media::Imaging::WriteableBitmap^ backOutputBuffer;   // XAML reads this
    Windows::UI::Xaml::Controls::Image ^cvImage;

private:

    VideoioBridge() {
        deviceIndex = 0;
        width = 640;
        height = 480;
        deviceReady = false;
        bIsFrameNew = false;
        currentFrame = 0;
        frameCounter = 0;
    };

    // singleton
    VideoioBridge(VideoioBridge const &);
    void operator=(const VideoioBridge &);

    std::atomic<bool>   deviceReady;
    Concurrency::progress_reporter<int> reporter;

    // Mats are wrapped with singleton class, we do not support more than one
    // capture device simultaneously with the design at this time
    //
    // nb. inputBufferMutex was not able to guarantee that OpenCV Mats were
    // ready to accept data in the UI thread (memory access exceptions were thrown
    // even though buffer address was good).
    // Therefore allocation of Mats is also done on the UI thread before the video
    // device is initialized.
    cv::Mat frontInputMat;
    cv::Mat backInputMat;

    int deviceIndex, width, height;
};
