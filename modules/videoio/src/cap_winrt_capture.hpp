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

#pragma once

#include "precomp.hpp"

#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>

#include <agile.h>


// nb. implemented the newer IVideoCapture C++ interface so that we can work
// directly with Mat, not the older C cv interface
// (which may have added overhead for IPL file conversion)

namespace cv {

    class VideoCapture_WinRT : public IVideoCapture
    {
    public:
        VideoCapture_WinRT() : started(false) {}
        VideoCapture_WinRT(int device);
        virtual ~VideoCapture_WinRT() {}

        // from base class IVideoCapture
        virtual double getProperty(int) { return 0; }
        virtual bool setProperty(int, double);
        virtual bool grabFrame();
        virtual bool retrieveFrame(int channel, cv::OutputArray outArray);

        virtual int getCaptureDomain() CV_OVERRIDE { return CAP_WINRT; }

        virtual bool isOpened() const;

    protected:

        bool                    started;
        CvSize                  size;
        int                     bytesPerPixel;
        unsigned long           frameCurrent;
        std::atomic<bool>       isFrameNew;
    };
}
