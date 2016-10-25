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

#include "MediaStreamSink.hpp"
#include "MediaSink.hpp"
#include "CaptureFrameGrabber.hpp"

using namespace Media;
using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::Media;
using namespace Windows::Media::Capture;
using namespace Windows::Media::MediaProperties;
using namespace concurrency;
using namespace Microsoft::WRL::Details;
using namespace Microsoft::WRL;

task<Media::CaptureFrameGrabber^> Media::CaptureFrameGrabber::CreateAsync(_In_ MediaCapture^ capture, _In_ VideoEncodingProperties^ props, CaptureStreamType streamType)
{
    auto reader = ref new Media::CaptureFrameGrabber(capture, props, streamType);

    auto profile = ref new MediaEncodingProfile();
    profile->Video = props;

    task<void> task;
    if (reader->_streamType == CaptureStreamType::Preview)
    {
        task = create_task(capture->StartPreviewToCustomSinkAsync(profile, reader->_mediaExtension));
    }
    else
    {
        task = create_task(capture->StartRecordToCustomSinkAsync(profile, reader->_mediaExtension));
    }

    return task.then([reader]()
    {
        reader->_state = State::Started;
        return reader;
    });
}

Media::CaptureFrameGrabber::CaptureFrameGrabber(_In_ MediaCapture^ capture, _In_ VideoEncodingProperties^ props, CaptureStreamType streamType)
: _state(State::Created)
, _streamType(streamType)
, _capture(capture)
{
    auto videoSampleHandler = ref new MediaSampleHandler(this, &Media::CaptureFrameGrabber::ProcessSample);

    _mediaSink = Make<MediaSink>(nullptr, props, nullptr, videoSampleHandler);
    _mediaExtension = reinterpret_cast<IMediaExtension^>(static_cast<AWM::IMediaExtension*>(_mediaSink.Get()));
}

Media::CaptureFrameGrabber::~CaptureFrameGrabber()
{
    if (_state == State::Started)
    {
        if (_streamType == CaptureStreamType::Preview)
        {
            (void)_capture->StopPreviewAsync();
        }
        else
        {
            (void)_capture->StopRecordAsync();
        }
    }

    if (_mediaSink != nullptr)
    {
        (void)_mediaSink->Shutdown();
        _mediaSink = nullptr;
    }
    _mediaExtension = nullptr;
    _capture = nullptr;
}

void Media::CaptureFrameGrabber::ShowCameraSettings()
{
#if WINAPI_FAMILY!=WINAPI_FAMILY_PHONE_APP
    if (_state == State::Started)
    {
        CameraOptionsUI::Show(_capture.Get());
    }
#endif
}

task<void> Media::CaptureFrameGrabber::FinishAsync()
{
    auto lock = _lock.LockExclusive();

    if (_state != State::Started)
    {
        throw ref new COMException(E_UNEXPECTED, L"State");
    }
    _state = State::Closing;

    if (_mediaSink != nullptr)
    {
        (void)_mediaSink->Shutdown();
        _mediaSink = nullptr;
    }
    _mediaExtension = nullptr;

    task<void> task;
    if (_streamType == CaptureStreamType::Preview)
    {
        task = create_task(_capture->StopPreviewAsync());
    }
    else
    {
        task = create_task(_capture->StopRecordAsync());
    }

    return task.then([this]()
    {
        auto lock = _lock.LockExclusive();
        _state = State::Closed;
        _capture = nullptr;
    });
}

task<ComPtr<IMF2DBuffer2>> Media::CaptureFrameGrabber::GetFrameAsync()
{
    auto lock = _lock.LockExclusive();

    if (_state != State::Started)
    {
        throw ref new COMException(E_UNEXPECTED, L"State");
    }

    _mediaSink->RequestVideoSample();

    task_completion_event<ComPtr<IMF2DBuffer2>> taskEvent;
    _videoSampleRequestQueue.push(taskEvent);

    return create_task(taskEvent);
}

void Media::CaptureFrameGrabber::ProcessSample(_In_ MediaSample^ sample)
{
    task_completion_event<ComPtr<IMF2DBuffer2>> t;

    {
        auto lock = _lock.LockExclusive();

        t = _videoSampleRequestQueue.front();
        _videoSampleRequestQueue.pop();
    }

    ComPtr<IMFMediaBuffer> buffer;
    CHK(sample->Sample->ConvertToContiguousBuffer(&buffer));

    // Dispatch without the lock taken to avoid deadlocks
    t.set(As<IMF2DBuffer2>(buffer));
}