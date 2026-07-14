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
#include "MFIncludes.hpp"

using namespace Media;
using namespace Microsoft::WRL;
using namespace Platform;
using namespace Windows::Foundation;

MediaStreamSink::MediaStreamSink(
    __in const MW::ComPtr<IMFMediaSink>& sink,
    __in DWORD id,
    __in const MW::ComPtr<IMFMediaType>& mt,
    __in MediaSampleHandler^ sampleHandler
    )
    : _shutdown(false)
    , _id((DWORD)-1)
    , _width(0)
    , _height(0)
{
    CHK(MFCreateEventQueue(&_eventQueue));
    CHK(MFCreateMediaType(&_curMT));

    _UpdateMediaType(mt);

    _sink = sink;
    _id = id;
    _sampleHandler = sampleHandler;
}

HRESULT MediaStreamSink::GetMediaSink(__deref_out IMFMediaSink **sink)
{
    return ExceptionBoundary([this, sink]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(sink);
        *sink = nullptr;

        _VerifyNotShutdown();

        CHK(_sink.CopyTo(sink));
    });
}

HRESULT MediaStreamSink::GetIdentifier(__out DWORD *identifier)
{
    return ExceptionBoundary([this, identifier]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(identifier);

        _VerifyNotShutdown();

        *identifier = _id;
    });
}

HRESULT MediaStreamSink::GetMediaTypeHandler(__deref_out IMFMediaTypeHandler **handler)
{
    return ExceptionBoundary([this, handler]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(handler);
        *handler = nullptr;

        _VerifyNotShutdown();

        *handler = this;
        this->AddRef();

    });
}

void MediaStreamSink::RequestSample()
{
    auto lock = _lock.LockExclusive();

    _VerifyNotShutdown();

    CHK(_eventQueue->QueueEventParamVar(MEStreamSinkRequestSample, GUID_NULL, S_OK, nullptr));
}

HRESULT MediaStreamSink::ProcessSample(__in_opt IMFSample *sample)
{
    return ExceptionBoundary([this, sample]()
    {
        MediaSampleHandler^ sampleHandler;
        auto mediaSample = ref new MediaSample();

        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();

            if (sample == nullptr)
            {
                return;
            }

            mediaSample->Sample = sample;
            sampleHandler = _sampleHandler;
        }

        // Call back without the lock taken to avoid deadlocks
        sampleHandler(mediaSample);
    });
}

HRESULT MediaStreamSink::PlaceMarker(__in MFSTREAMSINK_MARKER_TYPE /*markerType*/, __in const PROPVARIANT * /*markerValue*/, __in const PROPVARIANT * contextValue)
{
    return ExceptionBoundary([this, contextValue]()
    {
        auto lock = _lock.LockExclusive();
        CHKNULL(contextValue);

        _VerifyNotShutdown();

        CHK(_eventQueue->QueueEventParamVar(MEStreamSinkMarker, GUID_NULL, S_OK, contextValue));
    });
}

HRESULT MediaStreamSink::Flush()
{
    return ExceptionBoundary([this]()
    {
        auto lock = _lock.LockExclusive();

        _VerifyNotShutdown();
    });
}

HRESULT MediaStreamSink::GetEvent(__in DWORD flags, __deref_out IMFMediaEvent **event)
{
    return ExceptionBoundary([this, flags, event]()
    {
        CHKNULL(event);
        *event = nullptr;

        ComPtr<IMFMediaEventQueue> eventQueue;

        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();

            eventQueue = _eventQueue;
        }

        // May block for a while
        CHK(eventQueue->GetEvent(flags, event));
    });
}

HRESULT MediaStreamSink::BeginGetEvent(__in IMFAsyncCallback *callback, __in_opt IUnknown *state)
{
    return ExceptionBoundary([this, callback, state]()
    {
        auto lock = _lock.LockExclusive();

        _VerifyNotShutdown();

        CHK(_eventQueue->BeginGetEvent(callback, state));
    });
}


HRESULT MediaStreamSink::EndGetEvent(__in IMFAsyncResult *result, __deref_out IMFMediaEvent **event)
{
    return ExceptionBoundary([this, result, event]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(event);
        *event = nullptr;

        _VerifyNotShutdown();

        CHK(_eventQueue->EndGetEvent(result, event));
    });
}

HRESULT MediaStreamSink::QueueEvent(
    __in MediaEventType met,
    __in REFGUID extendedType,
    __in HRESULT status,
    __in_opt const PROPVARIANT *value_
    )
{
    return ExceptionBoundary([this, met, extendedType, status, value_]()
    {
        auto lock = _lock.LockExclusive();

        _VerifyNotShutdown();

        CHK(_eventQueue->QueueEventParamVar(met, extendedType, status, value_));
    });
}

HRESULT MediaStreamSink::IsMediaTypeSupported(__in IMFMediaType *mediaType, __deref_out_opt  IMFMediaType **closestMediaType)
{
    bool supported = false;

    HRESULT hr = ExceptionBoundary([this, mediaType, closestMediaType, &supported]()
    {
        auto lock = _lock.LockExclusive();

        if (closestMediaType != nullptr)
        {
            *closestMediaType = nullptr;
        }

        CHKNULL(mediaType);

        _VerifyNotShutdown();

        supported = _IsMediaTypeSupported(mediaType);
    });

    // Avoid throwing an exception to return MF_E_INVALIDMEDIATYPE as this is not a exceptional case
    return FAILED(hr) ? hr : supported ? S_OK : MF_E_INVALIDMEDIATYPE;
}

HRESULT MediaStreamSink::GetMediaTypeCount(__out DWORD *typeCount)
{
    return ExceptionBoundary([this, typeCount]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(typeCount);

        _VerifyNotShutdown();

        // No media type provided by default (app needs to specify it)
        *typeCount = 0;
    });
}

HRESULT MediaStreamSink::GetMediaTypeByIndex(__in DWORD /*index*/, __deref_out  IMFMediaType **mediaType)
{
    HRESULT hr = ExceptionBoundary([this, mediaType]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(mediaType);
        *mediaType = nullptr;

        _VerifyNotShutdown();
    });

    // Avoid throwing an exception to return MF_E_NO_MORE_TYPES as this is not a exceptional case
    return FAILED(hr) ? hr : MF_E_NO_MORE_TYPES;
}

HRESULT MediaStreamSink::SetCurrentMediaType(__in IMFMediaType *mediaType)
{
    return ExceptionBoundary([this, mediaType]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(mediaType);

        _VerifyNotShutdown();

        if (!_IsMediaTypeSupported(mediaType))
        {
            CHK(MF_E_INVALIDMEDIATYPE);
        }

        _UpdateMediaType(mediaType);
    });
}

HRESULT MediaStreamSink::GetCurrentMediaType(__deref_out_opt IMFMediaType **mediaType)
{
    return ExceptionBoundary([this, mediaType]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(mediaType);
        *mediaType = nullptr;

        _VerifyNotShutdown();

        ComPtr<IMFMediaType> mt;
        CHK(MFCreateMediaType(&mt));
        CHK(_curMT->CopyAllItems(mt.Get()));
        *mediaType = mt.Detach();
    });
}

HRESULT MediaStreamSink::GetMajorType(__out GUID *majorType)
{
    return ExceptionBoundary([this, majorType]()
    {
        auto lock = _lock.LockExclusive();

        CHKNULL(majorType);

        _VerifyNotShutdown();

        *majorType = _majorType;
    });
}

void MediaStreamSink::InternalSetCurrentMediaType(__in const ComPtr<IMFMediaType>& mediaType)
{
    auto lock = _lock.LockExclusive();

    CHKNULL(mediaType);

    _VerifyNotShutdown();

    _UpdateMediaType(mediaType);
}

void MediaStreamSink::Shutdown()
{
    auto lock = _lock.LockExclusive();

    if (_shutdown)
    {
        return;
    }
    _shutdown = true;

    (void)_eventQueue->Shutdown();
    _eventQueue = nullptr;

    _curMT = nullptr;
    _sink = nullptr;
    _sampleHandler = nullptr;
}

bool MediaStreamSink::_IsMediaTypeSupported(__in const ComPtr<IMFMediaType>& mt) const
{
    GUID majorType;
    GUID subType;
    if (SUCCEEDED(mt->GetGUID(MF_MT_MAJOR_TYPE, &majorType)) &&
        SUCCEEDED(mt->GetGUID(MF_MT_SUBTYPE, &subType)) &&
        (majorType == _majorType) &&
        (subType == _subType))
    {
        return true;
    }

    return false;
}

void MediaStreamSink::_UpdateMediaType(__in const ComPtr<IMFMediaType>& mt)
{
    CHK(mt->GetGUID(MF_MT_MAJOR_TYPE, &_majorType));
    CHK(mt->GetGUID(MF_MT_SUBTYPE, &_subType));

    if (_majorType == MFMediaType_Video)
    {
        CHK(MFGetAttributeSize(mt.Get(), MF_MT_FRAME_SIZE, &_width, &_height));
    }

    CHK(mt->CopyAllItems(_curMT.Get()));
}
