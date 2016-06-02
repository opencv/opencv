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

#include "MediaStreamSink.hpp"
#include "MFIncludes.hpp"

namespace Media {

const unsigned int c_audioStreamSinkId = 0;
const unsigned int c_videoStreamSinkId = 1;

class MediaSink WrlSealed
    : public MW::RuntimeClass<
    MW::RuntimeClassFlags<
    MW::RuntimeClassType::WinRtClassicComMix>
    , AWM::IMediaExtension
    , IMFMediaSink
    , IMFClockStateSink
    , MW::FtmBase
    >
{
    InspectableClass(L"MediaSink", BaseTrust)

public:

    MediaSink(
        _In_opt_ WMMp::AudioEncodingProperties^ audioProps,
        _In_opt_ WMMp::VideoEncodingProperties^ videoProps,
        _In_opt_ MediaSampleHandler^ audioSampleHandler,
        _In_opt_ MediaSampleHandler^ videoSampleHandler
        )
        : _shutdown(false)
    {
        MW::ComPtr<IMFMediaType> audioMT;
        if (audioProps != nullptr)
        {
            CHK(MFCreateMediaTypeFromProperties(As<IUnknown>(audioProps).Get(), &audioMT));
            _audioStreamSink = MW::Make<MediaStreamSink>(
                this,
                c_audioStreamSinkId,
                audioMT,
                audioSampleHandler
                );
        }

        MW::ComPtr<IMFMediaType> videoMT;
        if (videoProps != nullptr)
        {
            CHK(MFCreateMediaTypeFromProperties(As<IUnknown>(videoProps).Get(), &videoMT));
            _videoStreamSink = MW::Make<MediaStreamSink>(
                this,
                c_videoStreamSinkId,
                videoMT,
                videoSampleHandler
                );
        }
    }

    void RequestAudioSample()
    {
        auto lock = _lock.LockExclusive();

        _VerifyNotShutdown();

        _audioStreamSink->RequestSample();
    }

    void RequestVideoSample()
    {
        auto lock = _lock.LockExclusive();

        _VerifyNotShutdown();

        _videoStreamSink->RequestSample();
    }

    void SetCurrentAudioMediaType(_In_ IMFMediaType* mt)
    {
        auto lock = _lock.LockExclusive();

        _VerifyNotShutdown();

        _audioStreamSink->InternalSetCurrentMediaType(mt);
    }

    void SetCurrentVideoMediaType(_In_ IMFMediaType* mt)
    {
        auto lock = _lock.LockExclusive();

        _VerifyNotShutdown();

        _videoStreamSink->InternalSetCurrentMediaType(mt);
    }

    //
    // IMediaExtension
    //

    IFACEMETHODIMP SetProperties(_In_ AWFC::IPropertySet * /*configuration*/)
    {
        return ExceptionBoundary([this]()
        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();
        });
    }

    //
    // IMFMediaSink
    //

    IFACEMETHODIMP GetCharacteristics(_Out_ DWORD *characteristics)
    {
        return ExceptionBoundary([this, characteristics]()
        {
            _VerifyNotShutdown();

            CHKNULL(characteristics);
            *characteristics = MEDIASINK_RATELESS | MEDIASINK_FIXED_STREAMS;
        });
    }

    IFACEMETHODIMP AddStreamSink(
        DWORD /*streamSinkIdentifier*/,
        _In_ IMFMediaType * /*mediaType*/,
        _COM_Outptr_ IMFStreamSink **streamSink
        )
    {
        return ExceptionBoundary([this, streamSink]()
        {
            _VerifyNotShutdown();

            CHKNULL(streamSink);
            *streamSink = nullptr;

            CHK(MF_E_STREAMSINKS_FIXED);
        });
    }

    IFACEMETHODIMP RemoveStreamSink(DWORD /*streamSinkIdentifier*/)
    {
        return ExceptionBoundary([this]()
        {
            _VerifyNotShutdown();

            CHK(MF_E_STREAMSINKS_FIXED);
        });
    }

    IFACEMETHODIMP GetStreamSinkCount(_Out_ DWORD *streamSinkCount)
    {
        return ExceptionBoundary([this, streamSinkCount]()
        {
            CHKNULL(streamSinkCount);

            _VerifyNotShutdown();

            *streamSinkCount = (_audioStreamSink != nullptr) + (_videoStreamSink != nullptr);
        });
    }

    IFACEMETHODIMP GetStreamSinkByIndex(DWORD index, _COM_Outptr_ IMFStreamSink **streamSink)
    {
        return ExceptionBoundary([this, index, streamSink]()
        {
            auto lock = _lock.LockExclusive();

            CHKNULL(streamSink);
            *streamSink = nullptr;

            _VerifyNotShutdown();

            switch (index)
            {
            case 0:
                if (_audioStreamSink != nullptr)
                {
                    CHK(_audioStreamSink.CopyTo(streamSink));
                }
                else
                {
                    CHK(_videoStreamSink.CopyTo(streamSink));
                }
                break;

            case 1:
                if ((_audioStreamSink != nullptr) && (_videoStreamSink != nullptr))
                {
                    CHK(_videoStreamSink.CopyTo(streamSink));
                }
                else
                {
                    CHK(E_INVALIDARG);
                }
                break;

            default:
                CHK(E_INVALIDARG);
            }
        });
    }

    IFACEMETHODIMP GetStreamSinkById(DWORD identifier, _COM_Outptr_ IMFStreamSink **streamSink)
    {
        return ExceptionBoundary([this, identifier, streamSink]()
        {
            auto lock = _lock.LockExclusive();

            CHKNULL(streamSink);
            *streamSink = nullptr;

            _VerifyNotShutdown();

            if ((identifier == 0) && (_audioStreamSink != nullptr))
            {
                CHK(_audioStreamSink.CopyTo(streamSink));
            }
            else if ((identifier == 1) && (_videoStreamSink != nullptr))
            {
                CHK(_videoStreamSink.CopyTo(streamSink));
            }
            else
            {
                CHK(E_INVALIDARG);
            }
        });
    }

    IFACEMETHODIMP SetPresentationClock(_In_ IMFPresentationClock *clock)
    {
        return ExceptionBoundary([this, clock]()
        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();

            if (_clock != nullptr)
            {
                CHK(_clock->RemoveClockStateSink(this));
                _clock = nullptr;
            }

            if (clock != nullptr)
            {
                CHK(clock->AddClockStateSink(this));
                _clock = clock;
            }
        });
    }

    IFACEMETHODIMP GetPresentationClock(_COM_Outptr_ IMFPresentationClock **clock)
    {
        return ExceptionBoundary([this, clock]()
        {
            auto lock = _lock.LockExclusive();

            CHKNULL(clock);
            *clock = nullptr;

            _VerifyNotShutdown();

            if (_clock != nullptr)
            {
                CHK(_clock.CopyTo(clock))
            }
        });
    }

    IFACEMETHODIMP Shutdown()
    {
        return ExceptionBoundary([this]()
        {
            auto lock = _lock.LockExclusive();

            if (_shutdown)
            {
                return;
            }
            _shutdown = true;

            if (_audioStreamSink != nullptr)
            {
                _audioStreamSink->Shutdown();
                _audioStreamSink = nullptr;
            }

            if (_videoStreamSink != nullptr)
            {
                _videoStreamSink->Shutdown();
                _videoStreamSink = nullptr;
            }

            if (_clock != nullptr)
            {
                (void)_clock->RemoveClockStateSink(this);
                _clock = nullptr;
            }
        });
    }

    //
    // IMFClockStateSink methods
    //

    IFACEMETHODIMP OnClockStart(MFTIME /*hnsSystemTime*/, LONGLONG /*llClockStartOffset*/)
    {
        return ExceptionBoundary([this]()
        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();
        });
    }

    IFACEMETHODIMP OnClockStop(MFTIME /*hnsSystemTime*/)
    {
        return ExceptionBoundary([this]()
        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();
        });
    }

    IFACEMETHODIMP OnClockPause(MFTIME /*hnsSystemTime*/)
    {
        return ExceptionBoundary([this]()
        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();
        });
    }

    IFACEMETHODIMP OnClockRestart(MFTIME /*hnsSystemTime*/)
    {
        return ExceptionBoundary([this]()
        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();
        });
    }

    IFACEMETHODIMP OnClockSetRate(MFTIME /*hnsSystemTime*/, float /*flRate*/)
    {
        return ExceptionBoundary([this]()
        {
            auto lock = _lock.LockExclusive();

            _VerifyNotShutdown();
        });
    }

private:

    bool _shutdown;

    void _VerifyNotShutdown()
    {
        if (_shutdown)
        {
            CHK(MF_E_SHUTDOWN);
        }
    }

    MW::ComPtr<MediaStreamSink> _audioStreamSink;
    MW::ComPtr<MediaStreamSink> _videoStreamSink;
    MW::ComPtr<IMFPresentationClock> _clock;

    MWW::SRWLock _lock;
};

}