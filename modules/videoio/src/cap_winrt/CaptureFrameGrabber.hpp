// Copyright (c) Microsoft. All rights reserved.
//
// The MIT License (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "MFIncludes.hpp"


namespace Media {

class MediaSink;

enum class CaptureStreamType
{
    Preview = 0,
    Record
};

ref class CaptureFrameGrabber sealed
{
public:

    // IClosable
    virtual ~CaptureFrameGrabber();

    virtual void ShowCameraSettings();

internal:

    static concurrency::task<CaptureFrameGrabber^> CreateAsync(_In_ WMC::MediaCapture^ capture, _In_ WMMp::VideoEncodingProperties^ props)
    {
        return CreateAsync(capture, props, CaptureStreamType::Preview);
    }

    static concurrency::task<CaptureFrameGrabber^> CreateAsync(_In_ WMC::MediaCapture^ capture, _In_ WMMp::VideoEncodingProperties^ props, CaptureStreamType streamType);

    concurrency::task<MW::ComPtr<IMF2DBuffer2>> GetFrameAsync();
    concurrency::task<void> FinishAsync();

private:

    CaptureFrameGrabber(_In_ WMC::MediaCapture^ capture, _In_ WMMp::VideoEncodingProperties^ props, CaptureStreamType streamType);

    void ProcessSample(_In_ MediaSample^ sample);

    Platform::Agile<WMC::MediaCapture> _capture;
    ::Windows::Media::IMediaExtension^ _mediaExtension;

    MW::ComPtr<MediaSink> _mediaSink;

    CaptureStreamType _streamType;

    enum class State
    {
        Created,
        Started,
        Closing,
        Closed
    } _state;

    std::queue<concurrency::task_completion_event<MW::ComPtr<IMF2DBuffer2>>> _videoSampleRequestQueue;
    AutoMF _mf;
    MWW::SRWLock _lock;
};

}