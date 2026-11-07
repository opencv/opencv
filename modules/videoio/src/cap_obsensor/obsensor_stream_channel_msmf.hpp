// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
* Copyright(C) 2022 by ORBBEC Technology., Inc.
* Authors:
*   Huang Zhenchang <yufeng@orbbec.com>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_MSMF_HPP
#define OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_MSMF_HPP
#ifdef HAVE_OBSENSOR_MSMF

#include "obsensor_uvc_stream_channel.hpp"

#include <condition_variable>

#include <windows.h>
#include <guiddef.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfplay.h>
#include <mfobjects.h>
#include <mfreadwrite.h>
#include <tchar.h>
#include <strsafe.h>
#include <codecvt>
#include <ks.h>
#include <comdef.h>
#include <mutex>
#include <vidcap.h> //IKsTopologyInfo
#include <ksproxy.h> //IKsControl
#include <ksmedia.h>

namespace cv {
namespace obsensor {
template <class T>
class ComPtr
{
public:
    ComPtr() {}
    ComPtr(T* lp)
    {
        p = lp;
    }
    ComPtr(_In_ const ComPtr<T>& lp)
    {
        p = lp.p;
    }
    virtual ~ComPtr() {}

    void swap(_In_ ComPtr<T>& lp)
    {
        ComPtr<T> tmp(p);
        p = lp.p;
        lp.p = tmp.p;
        tmp = NULL;
    }
    T** operator&()
    {
        CV_Assert(p == NULL);
        return p.operator&();
    }
    T* operator->() const
    {
        CV_Assert(p != NULL);
        return p.operator->();
    }
    operator bool()
    {
        return p.operator!=(NULL);
    }

    T* Get() const
    {
        return p;
    }

    void Release()
    {
        if (p)
            p.Release();
    }

    // query for U interface
    template <typename U>
    HRESULT As(_Out_ ComPtr<U>& lp) const
    {
        lp.Release();
        return p->QueryInterface(__uuidof(U), reinterpret_cast<void**>((T**)&lp));
    }

private:
    _COM_SMARTPTR_TYPEDEF(T, __uuidof(T));
    TPtr p;
};

class MFContext
{
public:
    ~MFContext(void);
    static MFContext& getInstance();

    std::vector<UvcDeviceInfo> queryUvcDeviceInfoList();
    Ptr<IStreamChannel> createStreamChannel(const UvcDeviceInfo& devInfo);

private:
    MFContext(void);
};

struct FrameRate
{
    unsigned int denominator;
    unsigned int numerator;
};

class MSMFStreamChannel : public IUvcStreamChannel, public IMFSourceReaderCallback
{
public:
    MSMFStreamChannel(const UvcDeviceInfo& devInfo);
    virtual ~MSMFStreamChannel() noexcept;

    virtual void start(const StreamProfile& profile, FrameCallback frameCallback) override;
    virtual void stop() override;

private:
    virtual bool setXu(uint8_t ctrl, const uint8_t* data, uint32_t len) override;
    virtual bool getXu(uint8_t ctrl, uint8_t** data, uint32_t* len) override;

private:
    MFContext& mfContext_;

    ComPtr<IMFAttributes> deviceAttrs_ = nullptr;
    ComPtr<IMFMediaSource> deviceSource_ = nullptr;
    ComPtr<IMFAttributes> readerAttrs_ = nullptr;
    ComPtr<IMFSourceReader> streamReader_ = nullptr;
    ComPtr<IAMCameraControl> cameraControl_ = nullptr;
    ComPtr<IAMVideoProcAmp> videoProcAmp_ = nullptr;
    ComPtr<IKsTopologyInfo> xuKsTopologyInfo_ = nullptr;
    ComPtr<IUnknown> xuNodeInstance_ = nullptr;
    ComPtr<IKsControl> xuKsControl_ = nullptr;
    int xuNodeId_;

    FrameCallback frameCallback_;
    StreamProfile currentProfile_;
    int8_t currentStreamIndex_;

    StreamState streamState_;
    std::mutex streamStateMutex_;
    std::condition_variable streamStateCv_;

    std::vector<uint8_t> xuRecvBuf_;
    std::vector<uint8_t> xuSendBuf_;

public:
    STDMETHODIMP QueryInterface(REFIID iid, void** ppv) override;
    STDMETHODIMP_(ULONG)
        AddRef() override;
    STDMETHODIMP_(ULONG)
        Release() override;
    STDMETHODIMP OnReadSample(HRESULT /*hrStatus*/, DWORD dwStreamIndex, DWORD /*dwStreamFlags*/, LONGLONG /*llTimestamp*/, IMFSample* sample) override;
    STDMETHODIMP OnEvent(DWORD /*sidx*/, IMFMediaEvent* /*event*/) override;
    STDMETHODIMP OnFlush(DWORD) override;

private:
    long refCount_ = 1;
};
}} // namespace cv::obsensor::
#endif // HAVE_OBSENSOR_MSMF
#endif // OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_MSMF_HPP
