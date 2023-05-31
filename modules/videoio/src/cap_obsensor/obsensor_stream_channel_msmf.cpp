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


#ifdef HAVE_OBSENSOR_MSMF

#include "obsensor_stream_channel_msmf.hpp"

#include <shlwapi.h> // QISearch
#include <Mferror.h>

#pragma warning(disable : 4503)
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mf")
#pragma comment(lib, "mfuuid")
#pragma comment(lib, "Strmiids")
#pragma comment(lib, "Mfreadwrite")
#pragma comment(lib, "dxgi")
#pragma comment(lib, "Shlwapi")

namespace cv {
namespace obsensor {
std::string wideCharToUTF8(const WCHAR* s)
{
    auto len = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
    if (len == 0)
        return "";
    std::string buffer(len - 1, ' ');
    len = WideCharToMultiByte(CP_UTF8, 0, s, -1, &buffer[0], static_cast<int>(buffer.size()) + 1, nullptr, nullptr);
    return buffer;
}

std::string hr_to_string(HRESULT hr)
{
    _com_error err(hr);
    std::stringstream ss;
    ss << "HResult 0x" << std::hex << hr << ": \"" << err.ErrorMessage() << "\"";
    return ss.str();
}

#define HR_FAILED_RETURN(x)                                                      \
    if (x < 0)                                                                   \
    {                                                                            \
        CV_LOG_INFO(NULL, "Media Foundation error return: " << hr_to_string(x)); \
        return;                                                                  \
    }

#define HR_FAILED_LOG(x)                                                         \
    if (x < 0)                                                                   \
    {                                                                            \
        CV_LOG_INFO(NULL, "Media Foundation error return: " << hr_to_string(x)); \
    }

#define HR_FAILED_EXEC(x, statement)                                             \
    if (x < 0)                                                                   \
    {                                                                            \
        CV_LOG_INFO(NULL, "Media Foundation error return: " << hr_to_string(x)); \
        statement;                                                               \
    }

std::vector<std::string> stringSplit(std::string string, char separator)
{
    std::vector<std::string> tokens;
    std::string::size_type i1 = 0;
    while (true)
    {
        auto i2 = string.find(separator, i1);
        if (i2 == std::string::npos)
        {
            tokens.push_back(string.substr(i1));
            return tokens;
        }
        tokens.push_back(string.substr(i1, i2 - i1));
        i1 = i2 + 1;
    }
}

bool parseUvcDeviceSymbolicLink(const std::string& symbolicLink, uint16_t& vid, uint16_t& pid, uint16_t& mi, std::string& unique_id,
    std::string& device_guid)
{
    std::string lowerStr = symbolicLink;
    for (size_t i = 0; i < lowerStr.length(); i++)
    {
        lowerStr[i] = (char)tolower(lowerStr[i]);
    }
    auto tokens = stringSplit(lowerStr, '#');
    if (tokens.size() < 1 || (tokens[0] != R"(\\?\usb)" && tokens[0] != R"(\\?\hid)"))
        return false; // Not a USB device
    if (tokens.size() < 3)
    {
        return false;
    }

    auto ids = stringSplit(tokens[1], '&');
    if (ids[0].size() != 8 || ids[0].substr(0, 4) != "vid_" || !(std::istringstream(ids[0].substr(4, 4)) >> std::hex >> vid))
    {
        return false;
    }
    if (ids[1].size() != 8 || ids[1].substr(0, 4) != "pid_" || !(std::istringstream(ids[1].substr(4, 4)) >> std::hex >> pid))
    {
        return false;
    }
    if (ids.size() > 2 && (ids[2].size() != 5 || ids[2].substr(0, 3) != "mi_" || !(std::istringstream(ids[2].substr(3, 2)) >> mi)))
    {
        return false;
    }
    ids = stringSplit(tokens[2], '&');
    if (ids.size() == 0)
    {
        return false;
    }

    if (ids.size() > 2)
        unique_id = ids[1];
    else
        unique_id = "";

    if (tokens.size() >= 3)
        device_guid = tokens[3];

    return true;
}

#pragma pack(push, 1)
template <class T>
class big_endian
{
    T be_value;

public:
    operator T() const
    {
        T le_value = 0;
        for (unsigned int i = 0; i < sizeof(T); ++i)
            reinterpret_cast<char*>(&le_value)[i] = reinterpret_cast<const char*>(&be_value)[sizeof(T) - i - 1];
        return le_value;
    }
};
#pragma pack(pop)

MFContext::~MFContext(void)
{
    CoUninitialize();
}

MFContext& MFContext::getInstance()
{
    static MFContext instance;
    return instance;
}

std::vector<UvcDeviceInfo> MFContext::queryUvcDeviceInfoList()
{
    std::vector<UvcDeviceInfo> uvcDevList;
    ComPtr<IMFAttributes> pAttributes = nullptr;
    MFCreateAttributes(&pAttributes, 1);
    pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    // pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_CATEGORY, KSCATEGORY_SENSOR_CAMERA);
    IMFActivate** ppDevices;
    uint32_t numDevices;
    MFEnumDeviceSources(pAttributes.Get(), &ppDevices, &numDevices);
    for (uint32_t i = 0; i < numDevices; ++i)
    {
        ComPtr<IMFActivate> pDevice;
        pDevice = ppDevices[i];

        WCHAR* wCharStr = nullptr;
        uint32_t length;
        pDevice->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &wCharStr, &length);
        auto symbolicLink = wideCharToUTF8(wCharStr);
        CoTaskMemFree(wCharStr);

        pDevice->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &wCharStr, &length);
        auto name = wideCharToUTF8(wCharStr);
        CoTaskMemFree(wCharStr);

        uint16_t vid, pid, mi;
        std::string uid, guid;
        if (!parseUvcDeviceSymbolicLink(symbolicLink, vid, pid, mi, uid, guid))
            continue;
        uvcDevList.emplace_back(UvcDeviceInfo{ symbolicLink, name, uid, vid, pid, mi });
        CV_LOG_INFO(NULL, "UVC device found: name=" << name << ", vid=" << vid << ", pid=" << pid << ", mi=" << mi << ", uid=" << uid << ", guid=" << guid);
    }
    return uvcDevList;
}

Ptr<IStreamChannel> MFContext::createStreamChannel(const UvcDeviceInfo& devInfo)
{
    return makePtr<MSMFStreamChannel>(devInfo);
}

MFContext::MFContext()
{
    CoInitialize(0);
    CV_Assert(SUCCEEDED(MFStartup(MF_VERSION)));
}

MSMFStreamChannel::MSMFStreamChannel(const UvcDeviceInfo& devInfo) :
    IUvcStreamChannel(devInfo),
    mfContext_(MFContext::getInstance()),
    xuNodeId_(-1)
{
    HR_FAILED_RETURN(MFCreateAttributes(&deviceAttrs_, 2));
    HR_FAILED_RETURN(deviceAttrs_->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID));
    WCHAR* buffer = new wchar_t[devInfo_.id.length() + 1];
    MultiByteToWideChar(CP_UTF8, 0, devInfo_.id.c_str(), (int)devInfo_.id.length() + 1, buffer, (int)devInfo_.id.length() * sizeof(WCHAR));
    HR_FAILED_EXEC(deviceAttrs_->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, buffer), {
        delete[] buffer;
        return;
        })
        delete[] buffer;
        HR_FAILED_RETURN(MFCreateDeviceSource(deviceAttrs_.Get(), &deviceSource_));
        HR_FAILED_RETURN(deviceSource_->QueryInterface(__uuidof(IAMCameraControl), reinterpret_cast<void**>(&cameraControl_)));
        HR_FAILED_RETURN(deviceSource_->QueryInterface(__uuidof(IAMVideoProcAmp), reinterpret_cast<void**>(&videoProcAmp_)));

        HR_FAILED_RETURN(MFCreateAttributes(&readerAttrs_, 3));
        HR_FAILED_RETURN(readerAttrs_->SetUINT32(MF_SOURCE_READER_DISCONNECT_MEDIASOURCE_ON_SHUTDOWN, false));
        HR_FAILED_RETURN(readerAttrs_->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, true));
        HR_FAILED_RETURN(readerAttrs_->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, static_cast<IUnknown*>(this)));
        HR_FAILED_RETURN(MFCreateSourceReaderFromMediaSource(deviceSource_.Get(), readerAttrs_.Get(), &streamReader_));
        HR_FAILED_RETURN(streamReader_->SetStreamSelection(static_cast<DWORD>(MF_SOURCE_READER_ALL_STREAMS), true));

        HR_FAILED_RETURN(deviceSource_->QueryInterface(__uuidof(IKsTopologyInfo), reinterpret_cast<void**>(&xuKsTopologyInfo_)));
        DWORD nNodes = 0;
        HR_FAILED_RETURN(xuKsTopologyInfo_->get_NumNodes(&nNodes));
        for (DWORD i = 0; i < nNodes; i++)
        {
            GUID nodeType;
            HR_FAILED_EXEC(xuKsTopologyInfo_->get_NodeType(i, &nodeType), { continue; })
                if (nodeType == KSNODETYPE_DEV_SPECIFIC)
                {
                    xuNodeId_ = i;
                }
        }
        if (xuNodeId_ != -1)
        {
            HR_FAILED_RETURN(xuKsTopologyInfo_->CreateNodeInstance(xuNodeId_, IID_IUnknown, reinterpret_cast<LPVOID*>(&xuNodeInstance_)));
            HR_FAILED_RETURN(xuNodeInstance_->QueryInterface(__uuidof(IKsControl), reinterpret_cast<void**>(&xuKsControl_)));
        }

        if (streamType_ == OBSENSOR_STREAM_DEPTH)
        {
            initDepthFrameProcessor();
        }
}

MSMFStreamChannel::~MSMFStreamChannel()
{
    stop();
    if (cameraControl_)
    {
        cameraControl_.Release();
    }
    if (videoProcAmp_)
    {
        videoProcAmp_.Release();
    }
    if (streamReader_)
    {
        streamReader_.Release();
    }
    if (readerAttrs_)
    {
        readerAttrs_.Release();
    }
    if (deviceAttrs_)
    {
        deviceAttrs_.Release();
    }
    if (deviceSource_)
    {
        deviceSource_.Release();
    }
    if (xuKsTopologyInfo_)
    {
        xuKsTopologyInfo_.Release();
    }
    if (xuNodeInstance_)
    {
        xuNodeInstance_.Release();
    }
    if (xuKsControl_)
    {
        xuKsControl_.Release();
    }
}

void MSMFStreamChannel::start(const StreamProfile& profile, FrameCallback frameCallback)
{
    ComPtr<IMFMediaType> mediaType = nullptr;
    unsigned int width, height, fps;
    FrameRate frameRateMin, frameRateMax;
    bool quit = false;

    frameCallback_ = frameCallback;
    currentProfile_ = profile;
    currentStreamIndex_ = -1;

    for (uint8_t index = 0; index <= 5; index++)
    {
        for (uint32_t k = 0;; k++)
        {
            auto hr = streamReader_->GetNativeMediaType(index, k, &mediaType);
            if(hr == MF_E_INVALIDSTREAMNUMBER || hr == MF_E_NO_MORE_TYPES){
                break;
            }
            HR_FAILED_EXEC(hr, { continue; })
            GUID subtype;
            HR_FAILED_RETURN(mediaType->GetGUID(MF_MT_SUBTYPE, &subtype));
            HR_FAILED_RETURN(MFGetAttributeSize(mediaType.Get(), MF_MT_FRAME_SIZE, &width, &height));
            HR_FAILED_RETURN(MFGetAttributeRatio(mediaType.Get(), MF_MT_FRAME_RATE_RANGE_MIN, &frameRateMin.numerator, &frameRateMin.denominator));
            HR_FAILED_RETURN(MFGetAttributeRatio(mediaType.Get(), MF_MT_FRAME_RATE_RANGE_MAX, &frameRateMax.numerator, &frameRateMax.denominator));

            if (static_cast<float>(frameRateMax.numerator) / frameRateMax.denominator < static_cast<float>(frameRateMin.numerator) / frameRateMin.denominator)
            {
                std::swap(frameRateMax, frameRateMin);
            }

            fps = frameRateMax.numerator / frameRateMax.denominator;
            uint32_t device_fourcc = reinterpret_cast<const big_endian<uint32_t> &>(subtype.Data1);
            if (width == profile.width &&
                height == profile.height &&
                fps == profile.fps &&
                frameFourccToFormat(device_fourcc) == profile.format)
            {
                HR_FAILED_RETURN(streamReader_->SetCurrentMediaType(index, nullptr, mediaType.Get()));
                HR_FAILED_RETURN(streamReader_->SetStreamSelection(index, true));
                streamReader_->ReadSample(index, 0, nullptr, nullptr, nullptr, nullptr);

                streamState_ = STREAM_STARTING;
                currentStreamIndex_ = index;
                quit = true;

                // wait for frame
                std::unique_lock<std::mutex> lock(streamStateMutex_);
                auto success = streamStateCv_.wait_for(lock, std::chrono::milliseconds(3000), [&]() {
                    return streamState_ == STREAM_STARTED;
                });
                if (!success)
                {
                    stop();
                }
                break;
            }
            mediaType.Release();
        }
        if (quit)
        {
            break;
        }
    }
    streamState_ = quit ? streamState_ : STREAM_STOPED;
}

void MSMFStreamChannel::stop()
{
    if (streamState_ == STREAM_STARTING || streamState_ == STREAM_STARTED)
    {
        streamState_ = STREAM_STOPPING;
        streamReader_->SetStreamSelection(currentStreamIndex_, false);
        streamReader_->Flush(currentStreamIndex_);
        std::unique_lock<std::mutex> lk(streamStateMutex_);
        streamStateCv_.wait_for(lk, std::chrono::milliseconds(1000), [&]() {
            return streamState_ == STREAM_STOPED;
        });
    }
}

bool  MSMFStreamChannel::setXu(uint8_t ctrl, const uint8_t* data, uint32_t len)
{
    if (xuSendBuf_.size() < XU_MAX_DATA_LENGTH) {
        xuSendBuf_.resize(XU_MAX_DATA_LENGTH);
    }
    memcpy(xuSendBuf_.data(), data, len);

    KSP_NODE                              node;
    memset(&node, 0, sizeof(KSP_NODE));
    node.Property.Set = { 0xA55751A1, 0xF3C5, 0x4A5E, {0x8D, 0x5A, 0x68, 0x54, 0xB8, 0xFA, 0x27, 0x16} };
    node.Property.Id = ctrl;
    node.Property.Flags = KSPROPERTY_TYPE_SET | KSPROPERTY_TYPE_TOPOLOGY;
    node.NodeId = xuNodeId_;

    ULONG bytes_received = 0;
    HR_FAILED_EXEC(xuKsControl_->KsProperty(reinterpret_cast<PKSPROPERTY>(&node), sizeof(KSP_NODE), (void*)xuSendBuf_.data(), XU_MAX_DATA_LENGTH, &bytes_received), {
        return false;
    });
    return true;
}

bool  MSMFStreamChannel::getXu(uint8_t ctrl, uint8_t** data, uint32_t* len)
{
    if (xuRecvBuf_.size() < XU_MAX_DATA_LENGTH) {
        xuRecvBuf_.resize(XU_MAX_DATA_LENGTH);
    }
    KSP_NODE node;
    memset(&node, 0, sizeof(KSP_NODE));
    node.Property.Set = { 0xA55751A1, 0xF3C5, 0x4A5E, {0x8D, 0x5A, 0x68, 0x54, 0xB8, 0xFA, 0x27, 0x16} };
    node.Property.Id = ctrl;
    node.Property.Flags = KSPROPERTY_TYPE_GET | KSPROPERTY_TYPE_TOPOLOGY;
    node.NodeId = xuNodeId_;

    ULONG bytes_received = 0;
    HR_FAILED_EXEC(xuKsControl_->KsProperty(reinterpret_cast<PKSPROPERTY>(&node), sizeof(node), xuRecvBuf_.data(), XU_MAX_DATA_LENGTH, &bytes_received), {
        *len = 0;
        data = nullptr;
        return false;
    });
    *data = xuRecvBuf_.data();
    *len = bytes_received;
    return true;
}

STDMETHODIMP MSMFStreamChannel::QueryInterface(REFIID iid, void** ppv)
{
#pragma warning(push)
#pragma warning(disable : 4838)
    static const QITAB qit[] = {
        QITABENT(MSMFStreamChannel, IMFSourceReaderCallback),
        {nullptr},
    };
    return QISearch(this, qit, iid, ppv);
#pragma warning(pop)
}

STDMETHODIMP_(ULONG)
MSMFStreamChannel::AddRef()
{
    return InterlockedIncrement(&refCount_);
}

STDMETHODIMP_(ULONG)
MSMFStreamChannel::Release()
{
    ULONG count = InterlockedDecrement(&refCount_);
    if (count <= 0)
    {
        delete this;
    }
    return count;
}

STDMETHODIMP MSMFStreamChannel::OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex, DWORD /*dwStreamFlags*/, LONGLONG /*timeStamp*/, IMFSample* sample)
{
    HR_FAILED_LOG(hrStatus);

    if (streamState_ == STREAM_STARTING)
    {
        std::unique_lock<std::mutex> lock(streamStateMutex_);
        streamState_ = STREAM_STARTED;
        streamStateCv_.notify_all();
    }

    if (streamState_ != STREAM_STOPPING && streamState_ != STREAM_STOPED)
    {
        HR_FAILED_LOG(streamReader_->ReadSample(dwStreamIndex, 0, nullptr, nullptr, nullptr, nullptr));
        if (sample)
        {
            ComPtr<IMFMediaBuffer> buffer = nullptr;
            DWORD maxLength, currentLength;
            byte* byte_buffer = nullptr;

            HR_FAILED_EXEC(sample->GetBufferByIndex(0, &buffer), { return S_OK; });

            buffer->Lock(&byte_buffer, &maxLength, &currentLength);
            Frame fo = { currentProfile_.format, currentProfile_.width, currentProfile_.height, currentLength, (uint8_t*)byte_buffer };
            if (depthFrameProcessor_)
            {
                depthFrameProcessor_->process(&fo);
            }
            frameCallback_(&fo);
            buffer->Unlock();
        }
    }
    return S_OK;
}

STDMETHODIMP MSMFStreamChannel::OnEvent(DWORD /*sidx*/, IMFMediaEvent* /*event*/)
{
    return S_OK;
}

STDMETHODIMP MSMFStreamChannel::OnFlush(DWORD)
{
    if (streamState_ != STREAM_STOPED)
    {
        std::unique_lock<std::mutex> lock(streamStateMutex_);
        streamState_ = STREAM_STOPED;
        streamStateCv_.notify_all();
    }
    return S_OK;
}
}} // namespace cv::obsensor::
#endif // HAVE_OBSENSOR_MSMF
