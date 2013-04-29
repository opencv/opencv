/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"
#if (defined WIN32 || defined _WIN32) && defined HAVE_MSMF
/*
   Media Foundation-based Video Capturing module is based on
   videoInput library by Evgeny Pereguda:
   http://www.codeproject.com/Articles/559437/Capturing-of-video-from-web-camera-on-Windows-7-an
   Originaly licensed under The Code Project Open License (CPOL) 1.02:
   http://www.codeproject.com/info/cpol10.aspx
*/
#include <windows.h>
#include <guiddef.h>
#include <mfidl.h>
#include <Mfapi.h>
#include <mfplay.h>
#include <mfobjects.h>
#include "Strsafe.h"
#include <new>
#include <map>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#pragma warning(disable:4503)
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mf")
#pragma comment(lib, "mfuuid")
#pragma comment(lib, "Strmiids")
#pragma comment(lib, "MinCore_Downlevel")
struct IMFMediaType;
struct IMFActivate;
struct IMFMediaSource;
struct IMFAttributes;
namespace
{
template <class T> void SafeRelease(T **ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = NULL;
    }
}
 /// Class for printing info into consol
class DebugPrintOut
{
public:
    ~DebugPrintOut(void);
    static DebugPrintOut& getInstance();
    void printOut(const wchar_t *format, ...);
    void setVerbose(bool state);
    bool verbose;
private:
    DebugPrintOut(void);
};
// Structure for collecting info about types of video, which are supported by current video device
struct MediaType
{
    unsigned int MF_MT_FRAME_SIZE;
    unsigned int height;
    unsigned int width;
    unsigned int MF_MT_YUV_MATRIX;
    unsigned int MF_MT_VIDEO_LIGHTING;
    unsigned int MF_MT_DEFAULT_STRIDE;
    unsigned int MF_MT_VIDEO_CHROMA_SITING;
    GUID MF_MT_AM_FORMAT_TYPE;
    wchar_t *pMF_MT_AM_FORMAT_TYPEName;
    unsigned int MF_MT_FIXED_SIZE_SAMPLES;
    unsigned int MF_MT_VIDEO_NOMINAL_RANGE;
    unsigned int MF_MT_FRAME_RATE;
    unsigned int MF_MT_FRAME_RATE_low;
    unsigned int MF_MT_PIXEL_ASPECT_RATIO;
    unsigned int MF_MT_PIXEL_ASPECT_RATIO_low;
    unsigned int MF_MT_ALL_SAMPLES_INDEPENDENT;
    unsigned int MF_MT_FRAME_RATE_RANGE_MIN;
    unsigned int MF_MT_FRAME_RATE_RANGE_MIN_low;
    unsigned int MF_MT_SAMPLE_SIZE;
    unsigned int MF_MT_VIDEO_PRIMARIES;
    unsigned int MF_MT_INTERLACE_MODE;
    unsigned int MF_MT_FRAME_RATE_RANGE_MAX;
    unsigned int MF_MT_FRAME_RATE_RANGE_MAX_low;
    GUID MF_MT_MAJOR_TYPE;
    GUID MF_MT_SUBTYPE;
    wchar_t *pMF_MT_MAJOR_TYPEName;
    wchar_t *pMF_MT_SUBTYPEName;
    MediaType();
    ~MediaType();
    void Clear();
};
/// Class for parsing info from IMFMediaType into the local MediaType
class FormatReader
{
public:
    static MediaType Read(IMFMediaType *pType);
    ~FormatReader(void);
private:
    FormatReader(void);
};
DWORD WINAPI MainThreadFunction( LPVOID lpParam );
typedef void(*emergensyStopEventCallback)(int, void *);
typedef unsigned char BYTE;
class RawImage
{
public:
    ~RawImage(void);
    // Function of creation of the instance of the class
    static long CreateInstance(RawImage **ppRImage,unsigned int size);
    void setCopy(const BYTE * pSampleBuffer);
    void fastCopy(const BYTE * pSampleBuffer);
    unsigned char * getpPixels();
    bool isNew();
    unsigned int getSize();
private:
    bool ri_new;
    unsigned int ri_size;
    unsigned char *ri_pixels;
    RawImage(unsigned int size);
};
// Class for grabbing image from video stream
class ImageGrabber : public IMFSampleGrabberSinkCallback
{
public:
    ~ImageGrabber(void);
    HRESULT initImageGrabber(IMFMediaSource *pSource, GUID VideoFormat);
    HRESULT startGrabbing(void);
    void stopGrabbing();
    RawImage *getRawImage();
    // Function of creation of the instance of the class
    static HRESULT CreateInstance(ImageGrabber **ppIG,unsigned int deviceID);
private:
    bool ig_RIE;
    bool ig_Close;
    long m_cRef;
    unsigned int ig_DeviceID;
    IMFMediaSource *ig_pSource;
    IMFMediaSession *ig_pSession;
    IMFTopology *ig_pTopology;
    RawImage *ig_RIFirst;
    RawImage *ig_RISecond;
    RawImage *ig_RIOut;
    ImageGrabber(unsigned int deviceID);
    HRESULT CreateTopology(IMFMediaSource *pSource, IMFActivate *pSinkActivate, IMFTopology **ppTopo);
    HRESULT AddSourceNode(
    IMFTopology *pTopology,
    IMFMediaSource *pSource,
    IMFPresentationDescriptor *pPD,
    IMFStreamDescriptor *pSD,
    IMFTopologyNode **ppNode);
    HRESULT AddOutputNode(
    IMFTopology *pTopology,
    IMFActivate *pActivate,
    DWORD dwId,
    IMFTopologyNode **ppNode);
    // IUnknown methods
    STDMETHODIMP QueryInterface(REFIID iid, void** ppv);
    STDMETHODIMP_(ULONG) AddRef();
    STDMETHODIMP_(ULONG) Release();
    // IMFClockStateSink methods
    STDMETHODIMP OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset);
    STDMETHODIMP OnClockStop(MFTIME hnsSystemTime);
    STDMETHODIMP OnClockPause(MFTIME hnsSystemTime);
    STDMETHODIMP OnClockRestart(MFTIME hnsSystemTime);
    STDMETHODIMP OnClockSetRate(MFTIME hnsSystemTime, float flRate);
    // IMFSampleGrabberSinkCallback methods
    STDMETHODIMP OnSetPresentationClock(IMFPresentationClock* pClock);
    STDMETHODIMP OnProcessSample(REFGUID guidMajorMediaType, DWORD dwSampleFlags,
        LONGLONG llSampleTime, LONGLONG llSampleDuration, const BYTE * pSampleBuffer,
        DWORD dwSampleSize);
    STDMETHODIMP OnShutdown();
};
/// Class for controlling of thread of the grabbing raw data from video device
class ImageGrabberThread
{
    friend DWORD WINAPI MainThreadFunction( LPVOID lpParam );
public:
    ~ImageGrabberThread(void);
    static HRESULT CreateInstance(ImageGrabberThread **ppIGT, IMFMediaSource *pSource, unsigned int deviceID);
    void start();
    void stop();
    void setEmergencyStopEvent(void *userData, void(*func)(int, void *));
    ImageGrabber *getImageGrabber();
protected:
    virtual void run();
private:
    ImageGrabberThread(IMFMediaSource *pSource, unsigned int deviceID);
    HANDLE igt_Handle;
    DWORD   igt_ThreadIdArray;
    ImageGrabber *igt_pImageGrabber;
    emergensyStopEventCallback igt_func;
    void *igt_userData;
    bool igt_stop;
    unsigned int igt_DeviceID;
};
// Structure for collecting info about one parametr of current video device
struct Parametr
{
    long CurrentValue;
    long Min;
    long Max;
    long Step;
    long Default;
    long Flag;
    Parametr();
};
// Structure for collecting info about 17 parametrs of current video device
struct CamParametrs
{
        Parametr Brightness;
        Parametr Contrast;
        Parametr Hue;
        Parametr Saturation;
        Parametr Sharpness;
        Parametr Gamma;
        Parametr ColorEnable;
        Parametr WhiteBalance;
        Parametr BacklightCompensation;
        Parametr Gain;
        Parametr Pan;
        Parametr Tilt;
        Parametr Roll;
        Parametr Zoom;
        Parametr Exposure;
        Parametr Iris;
        Parametr Focus;
};
typedef std::wstring String;
typedef std::vector<int> vectorNum;
typedef std::map<String, vectorNum> SUBTYPEMap;
typedef std::map<UINT64, SUBTYPEMap> FrameRateMap;
typedef void(*emergensyStopEventCallback)(int, void *);
/// Class for controlling of video device
class videoDevice
{
public:
    videoDevice(void);
    ~videoDevice(void);
    void closeDevice();
    CamParametrs getParametrs();
    void setParametrs(CamParametrs parametrs);
    void setEmergencyStopEvent(void *userData, void(*func)(int, void *));
    long readInfoOfDevice(IMFActivate *pActivate, unsigned int Num);
    wchar_t *getName();
    int getCountFormats();
    unsigned int getWidth();
    unsigned int getHeight();
    MediaType getFormat(unsigned int id);
    bool setupDevice(unsigned int w, unsigned int h, unsigned int idealFramerate = 0);
    bool setupDevice(unsigned int id);
    bool isDeviceSetup();
    bool isDeviceMediaSource();
    bool isDeviceRawDataSource();
    bool isFrameNew();
    IMFMediaSource *getMediaSource();
    RawImage *getRawImageOut();
private:
    enum typeLock
    {
        MediaSourceLock,
        RawDataLock,
        OpenLock
    } vd_LockOut;
    wchar_t *vd_pFriendlyName;
    ImageGrabberThread *vd_pImGrTh;
    CamParametrs vd_PrevParametrs;
    unsigned int vd_Width;
    unsigned int vd_Height;
    unsigned int vd_CurrentNumber;
    bool vd_IsSetuped;
    std::map<UINT64, FrameRateMap> vd_CaptureFormats;
    std::vector<MediaType> vd_CurrentFormats;
    IMFMediaSource *vd_pSource;
    emergensyStopEventCallback vd_func;
    void *vd_userData;
    long enumerateCaptureFormats(IMFMediaSource *pSource);
    long setDeviceFormat(IMFMediaSource *pSource, unsigned long dwFormatIndex);
    void buildLibraryofTypes();
    int findType(unsigned int size, unsigned int frameRate = 0);
    long resetDevice(IMFActivate *pActivate);
    long initDevice();
    long checkDevice(IMFAttributes *pAttributes, IMFActivate **pDevice);
};
/// Class for managing of list of video devices
class videoDevices
{
public:
    ~videoDevices(void);
    long initDevices(IMFAttributes *pAttributes);
    static videoDevices& getInstance();
    videoDevice *getDevice(unsigned int i);
    unsigned int getCount();
    void clearDevices();
private:
    UINT32 count;
    std::vector<videoDevice *> vds_Devices;
    videoDevices(void);
};
// Class for creating of Media Foundation context
class Media_Foundation
{
public:
    virtual ~Media_Foundation(void);
    static Media_Foundation& getInstance();
    bool buildListOfDevices();
private:
    Media_Foundation(void);
};
/// The only visiable class for controlling of video devices in format singelton
class videoInput
{
public:
    virtual ~videoInput(void);
    // Getting of static instance of videoInput class
    static videoInput& getInstance();
    // Closing video device with deviceID
    void closeDevice(int deviceID);
    // Setting callback function for emergency events(for example: removing video device with deviceID) with userData
    void setEmergencyStopEvent(int deviceID, void *userData, void(*func)(int, void *));
    // Closing all devices
    void closeAllDevices();
    // Getting of parametrs of video device with deviceID
    CamParametrs getParametrs(int deviceID);
    // Setting of parametrs of video device with deviceID
    void setParametrs(int deviceID, CamParametrs parametrs);
    // Getting numbers of existence videodevices with listing in consol
    unsigned int listDevices(bool silent = false);
    // Getting numbers of formats, which are supported by videodevice with deviceID
    unsigned int getCountFormats(int deviceID);
    // Getting width of image, which is getting from videodevice with deviceID
    unsigned int getWidth(int deviceID);
    // Getting height of image, which is getting from videodevice with deviceID
    unsigned int getHeight(int deviceID);
    // Getting name of videodevice with deviceID
    wchar_t *getNameVideoDevice(int deviceID);
    // Getting interface MediaSource for Media Foundation from videodevice with deviceID
    IMFMediaSource *getMediaSource(int deviceID);
    // Getting format with id, which is supported by videodevice with deviceID
    MediaType getFormat(int deviceID, int unsigned id);
    // Checking of existence of the suitable video devices
    bool isDevicesAcceable();
    // Checking of using the videodevice with deviceID
    bool isDeviceSetup(int deviceID);
    // Checking of using MediaSource from videodevice with deviceID
    bool isDeviceMediaSource(int deviceID);
    // Checking of using Raw Data of pixels from videodevice with deviceID
    bool isDeviceRawDataSource(int deviceID);
    // Setting of the state of outprinting info in console
    static void setVerbose(bool state);
    // Initialization of video device with deviceID by media type with id
    bool setupDevice(int deviceID, unsigned int id = 0);
    // Initialization of video device with deviceID by wisth w, height h and fps idealFramerate
    bool setupDevice(int deviceID, unsigned int w, unsigned int h, unsigned int idealFramerate = 30);
    // Checking of recivig of new frame from video device with deviceID
    bool isFrameNew(int deviceID);
    // Writing of Raw Data pixels from video device with deviceID with correction of RedAndBlue flipping flipRedAndBlue and vertical flipping flipImage
    bool getPixels(int deviceID, unsigned char * pixels, bool flipRedAndBlue = false, bool flipImage = false);
private:
    bool accessToDevices;
    videoInput(void);
    void processPixels(unsigned char * src, unsigned char * dst, unsigned int width, unsigned int height, unsigned int bpp, bool bRGB, bool bFlip);
    void updateListOfDevices();
};
DebugPrintOut::DebugPrintOut(void):verbose(true)
{
}
DebugPrintOut::~DebugPrintOut(void)
{
}
DebugPrintOut& DebugPrintOut::getInstance()
{
    static DebugPrintOut instance;
    return instance;
}
void DebugPrintOut::printOut(const wchar_t *format, ...)
{
    if(verbose)
    {
        int i = 0;
        wchar_t *p = NULL;
        va_list args;
        va_start(args, format);
        if(wcscmp(format, L"%i"))
        {
            i = va_arg (args, int);
        }
        if(wcscmp(format, L"%s"))
        {
            p = va_arg (args, wchar_t *);
        }
        wprintf(format, i,p);
        va_end (args);
    }
}
void DebugPrintOut::setVerbose(bool state)
{
    verbose = state;
}
LPCWSTR GetGUIDNameConstNew(const GUID& guid);
HRESULT GetGUIDNameNew(const GUID& guid, WCHAR **ppwsz);
HRESULT LogAttributeValueByIndexNew(IMFAttributes *pAttr, DWORD index);
HRESULT SpecialCaseAttributeValueNew(GUID guid, const PROPVARIANT& var, MediaType &out);
unsigned int *GetParametr(GUID guid, MediaType &out)
{
    if(guid == MF_MT_YUV_MATRIX)
        return &(out.MF_MT_YUV_MATRIX);
    if(guid == MF_MT_VIDEO_LIGHTING)
        return &(out.MF_MT_VIDEO_LIGHTING);
    if(guid == MF_MT_DEFAULT_STRIDE)
        return &(out.MF_MT_DEFAULT_STRIDE);
    if(guid == MF_MT_VIDEO_CHROMA_SITING)
        return &(out.MF_MT_VIDEO_CHROMA_SITING);
    if(guid == MF_MT_VIDEO_NOMINAL_RANGE)
        return &(out.MF_MT_VIDEO_NOMINAL_RANGE);
    if(guid == MF_MT_ALL_SAMPLES_INDEPENDENT)
        return &(out.MF_MT_ALL_SAMPLES_INDEPENDENT);
    if(guid == MF_MT_FIXED_SIZE_SAMPLES)
        return &(out.MF_MT_FIXED_SIZE_SAMPLES);
    if(guid == MF_MT_SAMPLE_SIZE)
        return &(out.MF_MT_SAMPLE_SIZE);
    if(guid == MF_MT_VIDEO_PRIMARIES)
        return &(out.MF_MT_VIDEO_PRIMARIES);
    if(guid == MF_MT_INTERLACE_MODE)
        return &(out.MF_MT_INTERLACE_MODE);
    return NULL;
}
HRESULT LogAttributeValueByIndexNew(IMFAttributes *pAttr, DWORD index, MediaType &out)
{
    WCHAR *pGuidName = NULL;
    WCHAR *pGuidValName = NULL;
    GUID guid = { 0 };
    PROPVARIANT var;
    PropVariantInit(&var);
    HRESULT hr = pAttr->GetItemByIndex(index, &guid, &var);
    if (FAILED(hr))
    {
        goto done;
    }
    hr = GetGUIDNameNew(guid, &pGuidName);
    if (FAILED(hr))
    {
        goto done;
    }
    hr = SpecialCaseAttributeValueNew(guid, var, out);
    unsigned int *p;
    if (FAILED(hr))
    {
        goto done;
    }
    if (hr == S_FALSE)
    {
        switch (var.vt)
        {
        case VT_UI4:
            p = GetParametr(guid, out);
            if(p)
            {
                *p = var.ulVal;
            }
            break;
        case VT_UI8:
            break;
        case VT_R8:
            break;
        case VT_CLSID:
            if(guid == MF_MT_AM_FORMAT_TYPE)
            {
                hr = GetGUIDNameNew(*var.puuid, &pGuidValName);
                if (SUCCEEDED(hr))
                {
                    out.MF_MT_AM_FORMAT_TYPE = MF_MT_AM_FORMAT_TYPE;
                    out.pMF_MT_AM_FORMAT_TYPEName = pGuidValName;
                    pGuidValName = NULL;
                }
            }
            if(guid == MF_MT_MAJOR_TYPE)
            {
                hr = GetGUIDNameNew(*var.puuid, &pGuidValName);
                if (SUCCEEDED(hr))
                {
                    out.MF_MT_MAJOR_TYPE = MF_MT_MAJOR_TYPE;
                    out.pMF_MT_MAJOR_TYPEName = pGuidValName;
                    pGuidValName = NULL;
                }
            }
            if(guid == MF_MT_SUBTYPE)
            {
                hr = GetGUIDNameNew(*var.puuid, &pGuidValName);
                if (SUCCEEDED(hr))
                {
                    out.MF_MT_SUBTYPE = MF_MT_SUBTYPE;
                    out.pMF_MT_SUBTYPEName = pGuidValName;
                    pGuidValName = NULL;
                }
            }
            break;
        case VT_LPWSTR:
            break;
        case VT_VECTOR | VT_UI1:
            break;
        case VT_UNKNOWN:
            break;
        default:
            break;
        }
    }
done:
    CoTaskMemFree(pGuidName);
    CoTaskMemFree(pGuidValName);
    PropVariantClear(&var);
    return hr;
}
HRESULT GetGUIDNameNew(const GUID& guid, WCHAR **ppwsz)
{
    HRESULT hr = S_OK;
    WCHAR *pName = NULL;
    LPCWSTR pcwsz = GetGUIDNameConstNew(guid);
    if (pcwsz)
    {
        size_t cchLength = 0;
        hr = StringCchLengthW(pcwsz, STRSAFE_MAX_CCH, &cchLength);
        if (FAILED(hr))
        {
            goto done;
        }
        pName = (WCHAR*)CoTaskMemAlloc((cchLength + 1) * sizeof(WCHAR));
        if (pName == NULL)
        {
            hr = E_OUTOFMEMORY;
            goto done;
        }
        hr = StringCchCopyW(pName, cchLength + 1, pcwsz);
        if (FAILED(hr))
        {
            goto done;
        }
    }
    else
    {
        hr = StringFromCLSID(guid, &pName);
    }
done:
    if (FAILED(hr))
    {
        *ppwsz = NULL;
        CoTaskMemFree(pName);
    }
    else
    {
        *ppwsz = pName;
    }
    return hr;
}
void LogUINT32AsUINT64New(const PROPVARIANT& var, UINT32 &uHigh, UINT32 &uLow)
{
    Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &uHigh, &uLow);
}
float OffsetToFloatNew(const MFOffset& offset)
{
    return offset.value + (static_cast<float>(offset.fract) / 65536.0f);
}
HRESULT LogVideoAreaNew(const PROPVARIANT& var)
{
    if (var.caub.cElems < sizeof(MFVideoArea))
    {
        return S_OK;
    }
    return S_OK;
}
HRESULT SpecialCaseAttributeValueNew(GUID guid, const PROPVARIANT& var, MediaType &out)
{
    if (guid == MF_MT_FRAME_SIZE)
    {
        UINT32 uHigh = 0, uLow = 0;
        LogUINT32AsUINT64New(var, uHigh, uLow);
        out.width = uHigh;
        out.height = uLow;
        out.MF_MT_FRAME_SIZE = out.width * out.height;
    }
    else
    if (guid == MF_MT_FRAME_RATE)
    {
        UINT32 uHigh = 0, uLow = 0;
        LogUINT32AsUINT64New(var, uHigh, uLow);
        out.MF_MT_FRAME_RATE = uHigh;
        out.MF_MT_FRAME_RATE_low = uLow;
    }
    else
    if (guid == MF_MT_FRAME_RATE_RANGE_MAX)
    {
        UINT32 uHigh = 0, uLow = 0;
        LogUINT32AsUINT64New(var, uHigh, uLow);
        out.MF_MT_FRAME_RATE_RANGE_MAX = uHigh;
        out.MF_MT_FRAME_RATE_RANGE_MAX_low = uLow;
    }
    else
    if (guid == MF_MT_FRAME_RATE_RANGE_MIN)
    {
        UINT32 uHigh = 0, uLow = 0;
        LogUINT32AsUINT64New(var, uHigh, uLow);
        out.MF_MT_FRAME_RATE_RANGE_MIN = uHigh;
        out.MF_MT_FRAME_RATE_RANGE_MIN_low = uLow;
    }
    else
    if (guid == MF_MT_PIXEL_ASPECT_RATIO)
    {
        UINT32 uHigh = 0, uLow = 0;
        LogUINT32AsUINT64New(var, uHigh, uLow);
        out.MF_MT_PIXEL_ASPECT_RATIO = uHigh;
        out.MF_MT_PIXEL_ASPECT_RATIO_low = uLow;
    }
    else
    {
        return S_FALSE;
    }
    return S_OK;
}
#ifndef IF_EQUAL_RETURN
#define IF_EQUAL_RETURN(param, val) if(val == param) return L#val
#endif
LPCWSTR GetGUIDNameConstNew(const GUID& guid)
{
    IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_SUBTYPE);
    IF_EQUAL_RETURN(guid, MF_MT_ALL_SAMPLES_INDEPENDENT);
    IF_EQUAL_RETURN(guid, MF_MT_FIXED_SIZE_SAMPLES);
    IF_EQUAL_RETURN(guid, MF_MT_COMPRESSED);
    IF_EQUAL_RETURN(guid, MF_MT_SAMPLE_SIZE);
    IF_EQUAL_RETURN(guid, MF_MT_WRAPPED_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_NUM_CHANNELS);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_AVG_BYTES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BLOCK_ALIGNMENT);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BITS_PER_SAMPLE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_VALID_BITS_PER_SAMPLE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_BLOCK);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_CHANNEL_MASK);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FOLDDOWN_MATRIX);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKREF);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKTARGET);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGREF);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGTARGET);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_PREFER_WAVEFORMATEX);
    IF_EQUAL_RETURN(guid, MF_MT_AAC_PAYLOAD_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_SIZE);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MAX);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MIN);
    IF_EQUAL_RETURN(guid, MF_MT_PIXEL_ASPECT_RATIO);
    IF_EQUAL_RETURN(guid, MF_MT_DRM_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_PAD_CONTROL_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_SOURCE_CONTENT_HINT);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_CHROMA_SITING);
    IF_EQUAL_RETURN(guid, MF_MT_INTERLACE_MODE);
    IF_EQUAL_RETURN(guid, MF_MT_TRANSFER_FUNCTION);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_PRIMARIES);
    IF_EQUAL_RETURN(guid, MF_MT_CUSTOM_VIDEO_PRIMARIES);
    IF_EQUAL_RETURN(guid, MF_MT_YUV_MATRIX);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_LIGHTING);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_NOMINAL_RANGE);
    IF_EQUAL_RETURN(guid, MF_MT_GEOMETRIC_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_MINIMUM_DISPLAY_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_ENABLED);
    IF_EQUAL_RETURN(guid, MF_MT_AVG_BITRATE);
    IF_EQUAL_RETURN(guid, MF_MT_AVG_BIT_ERROR_RATE);
    IF_EQUAL_RETURN(guid, MF_MT_MAX_KEYFRAME_SPACING);
    IF_EQUAL_RETURN(guid, MF_MT_DEFAULT_STRIDE);
    IF_EQUAL_RETURN(guid, MF_MT_PALETTE);
    IF_EQUAL_RETURN(guid, MF_MT_USER_DATA);
    IF_EQUAL_RETURN(guid, MF_MT_AM_FORMAT_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG_START_TIME_CODE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_PROFILE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_LEVEL);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG_SEQUENCE_HEADER);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_0);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_0);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_1);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_1);
    IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_SRC_PACK);
    IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_CTRL_PACK);
    IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_HEADER);
    IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_FORMAT);
    IF_EQUAL_RETURN(guid, MF_MT_IMAGE_LOSS_TOLERANT);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG4_SAMPLE_DESCRIPTION);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY);
    IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_4CC);
    IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_WAVE_FORMAT_TAG);
    // Media types
    IF_EQUAL_RETURN(guid, MFMediaType_Audio);
    IF_EQUAL_RETURN(guid, MFMediaType_Video);
    IF_EQUAL_RETURN(guid, MFMediaType_Protected);
    IF_EQUAL_RETURN(guid, MFMediaType_SAMI);
    IF_EQUAL_RETURN(guid, MFMediaType_Script);
    IF_EQUAL_RETURN(guid, MFMediaType_Image);
    IF_EQUAL_RETURN(guid, MFMediaType_HTML);
    IF_EQUAL_RETURN(guid, MFMediaType_Binary);
    IF_EQUAL_RETURN(guid, MFMediaType_FileTransfer);
    IF_EQUAL_RETURN(guid, MFVideoFormat_AI44); //     FCC('AI44')
    IF_EQUAL_RETURN(guid, MFVideoFormat_ARGB32); //   D3DFMT_A8R8G8B8
    IF_EQUAL_RETURN(guid, MFVideoFormat_AYUV); //     FCC('AYUV')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DV25); //     FCC('dv25')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DV50); //     FCC('dv50')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVH1); //     FCC('dvh1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVSD); //     FCC('dvsd')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVSL); //     FCC('dvsl')
    IF_EQUAL_RETURN(guid, MFVideoFormat_H264); //     FCC('H264')
    IF_EQUAL_RETURN(guid, MFVideoFormat_I420); //     FCC('I420')
    IF_EQUAL_RETURN(guid, MFVideoFormat_IYUV); //     FCC('IYUV')
    IF_EQUAL_RETURN(guid, MFVideoFormat_M4S2); //     FCC('M4S2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MJPG);
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP43); //     FCC('MP43')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP4S); //     FCC('MP4S')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP4V); //     FCC('MP4V')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MPG1); //     FCC('MPG1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MSS1); //     FCC('MSS1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MSS2); //     FCC('MSS2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_NV11); //     FCC('NV11')
    IF_EQUAL_RETURN(guid, MFVideoFormat_NV12); //     FCC('NV12')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P010); //     FCC('P010')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P016); //     FCC('P016')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P210); //     FCC('P210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P216); //     FCC('P216')
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB24); //    D3DFMT_R8G8B8
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB32); //    D3DFMT_X8R8G8B8
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB555); //   D3DFMT_X1R5G5B5
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB565); //   D3DFMT_R5G6B5
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB8);
    IF_EQUAL_RETURN(guid, MFVideoFormat_UYVY); //     FCC('UYVY')
    IF_EQUAL_RETURN(guid, MFVideoFormat_v210); //     FCC('v210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_v410); //     FCC('v410')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV1); //     FCC('WMV1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV2); //     FCC('WMV2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV3); //     FCC('WMV3')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WVC1); //     FCC('WVC1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y210); //     FCC('Y210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y216); //     FCC('Y216')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y410); //     FCC('Y410')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y416); //     FCC('Y416')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y41P);
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y41T);
    IF_EQUAL_RETURN(guid, MFVideoFormat_YUY2); //     FCC('YUY2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_YV12); //     FCC('YV12')
    IF_EQUAL_RETURN(guid, MFVideoFormat_YVYU);
    IF_EQUAL_RETURN(guid, MFAudioFormat_PCM); //              WAVE_FORMAT_PCM
    IF_EQUAL_RETURN(guid, MFAudioFormat_Float); //            WAVE_FORMAT_IEEE_FLOAT
    IF_EQUAL_RETURN(guid, MFAudioFormat_DTS); //              WAVE_FORMAT_DTS
    IF_EQUAL_RETURN(guid, MFAudioFormat_Dolby_AC3_SPDIF); //  WAVE_FORMAT_DOLBY_AC3_SPDIF
    IF_EQUAL_RETURN(guid, MFAudioFormat_DRM); //              WAVE_FORMAT_DRM
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV8); //        WAVE_FORMAT_WMAUDIO2
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV9); //        WAVE_FORMAT_WMAUDIO3
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudio_Lossless); // WAVE_FORMAT_WMAUDIO_LOSSLESS
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMASPDIF); //         WAVE_FORMAT_WMASPDIF
    IF_EQUAL_RETURN(guid, MFAudioFormat_MSP1); //             WAVE_FORMAT_WMAVOICE9
    IF_EQUAL_RETURN(guid, MFAudioFormat_MP3); //              WAVE_FORMAT_MPEGLAYER3
    IF_EQUAL_RETURN(guid, MFAudioFormat_MPEG); //             WAVE_FORMAT_MPEG
    IF_EQUAL_RETURN(guid, MFAudioFormat_AAC); //              WAVE_FORMAT_MPEG_HEAAC
    IF_EQUAL_RETURN(guid, MFAudioFormat_ADTS); //             WAVE_FORMAT_MPEG_ADTS_AAC
    return NULL;
}
FormatReader::FormatReader(void)
{
}
MediaType FormatReader::Read(IMFMediaType *pType)
{
    UINT32 count = 0;
    HRESULT hr = S_OK;
    MediaType out;
    hr = pType->LockStore();
    if (FAILED(hr))
    {
        return out;
    }
    hr = pType->GetCount(&count);
    if (FAILED(hr))
    {
        return out;
    }
    for (UINT32 i = 0; i < count; i++)
    {
        hr = LogAttributeValueByIndexNew(pType, i, out);
        if (FAILED(hr))
        {
            break;
        }
    }
    hr = pType->UnlockStore();
    if (FAILED(hr))
    {
        return out;
    }
    return out;
}
FormatReader::~FormatReader(void)
{
}
#define CHECK_HR(x) if (FAILED(x)) { goto done; }
ImageGrabber::ImageGrabber(unsigned int deviceID): m_cRef(1), ig_DeviceID(deviceID), ig_pSource(NULL), ig_pSession(NULL), ig_pTopology(NULL), ig_RIE(true), ig_Close(false)
{
}
ImageGrabber::~ImageGrabber(void)
{
    if (ig_pSession)
    {
        ig_pSession->Shutdown();
    }
    //SafeRelease(&ig_pSession);
    //SafeRelease(&ig_pTopology);
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: Destroing instance of the ImageGrabber class \n", ig_DeviceID);
}
HRESULT ImageGrabber::initImageGrabber(IMFMediaSource *pSource, GUID VideoFormat)
{
    IMFActivate *pSinkActivate = NULL;
    IMFMediaType *pType = NULL;
    IMFPresentationDescriptor *pPD = NULL;
    IMFStreamDescriptor *pSD = NULL;
    IMFMediaTypeHandler *pHandler = NULL;
    IMFMediaType *pCurrentType = NULL;
    HRESULT hr = S_OK;
    MediaType MT;
     // Clean up.
    if (ig_pSession)
    {
        ig_pSession->Shutdown();
    }
    SafeRelease(&ig_pSession);
    SafeRelease(&ig_pTopology);
    ig_pSource = pSource;
    hr = pSource->CreatePresentationDescriptor(&pPD);
    if (FAILED(hr))
        goto err;
    BOOL fSelected;
    hr = pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD);
    if (FAILED(hr))
        goto err;
    hr = pSD->GetMediaTypeHandler(&pHandler);
    if (FAILED(hr))
        goto err;
    DWORD cTypes = 0;
    hr = pHandler->GetMediaTypeCount(&cTypes);
    if (FAILED(hr))
        goto err;
    if(cTypes > 0)
    {
        hr = pHandler->GetCurrentMediaType(&pCurrentType);
        if (FAILED(hr))
            goto err;
        MT = FormatReader::Read(pCurrentType);
    }
err:
    SafeRelease(&pPD);
    SafeRelease(&pSD);
    SafeRelease(&pHandler);
    SafeRelease(&pCurrentType);
    unsigned int sizeRawImage = 0;
    if(VideoFormat == MFVideoFormat_RGB24)
    {
        sizeRawImage = MT.MF_MT_FRAME_SIZE * 3;
    }
    else if(VideoFormat == MFVideoFormat_RGB32)
    {
        sizeRawImage = MT.MF_MT_FRAME_SIZE * 4;
    }
    CHECK_HR(hr = RawImage::CreateInstance(&ig_RIFirst, sizeRawImage));
    CHECK_HR(hr = RawImage::CreateInstance(&ig_RISecond, sizeRawImage));
    ig_RIOut = ig_RISecond;
    // Configure the media type that the Sample Grabber will receive.
    // Setting the major and subtype is usually enough for the topology loader
    // to resolve the topology.
    CHECK_HR(hr = MFCreateMediaType(&pType));
    CHECK_HR(hr = pType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video));
    CHECK_HR(hr = pType->SetGUID(MF_MT_SUBTYPE, VideoFormat));
    // Create the sample grabber sink.
    CHECK_HR(hr = MFCreateSampleGrabberSinkActivate(pType, this, &pSinkActivate));
    // To run as fast as possible, set this attribute (requires Windows 7):
    CHECK_HR(hr = pSinkActivate->SetUINT32(MF_SAMPLEGRABBERSINK_IGNORE_CLOCK, TRUE));
    // Create the Media Session.
    CHECK_HR(hr = MFCreateMediaSession(NULL, &ig_pSession));
    // Create the topology.
    CHECK_HR(hr = CreateTopology(pSource, pSinkActivate, &ig_pTopology));
done:
    // Clean up.
    if (FAILED(hr))
    {
        if (ig_pSession)
        {
            ig_pSession->Shutdown();
        }
        SafeRelease(&ig_pSession);
        SafeRelease(&ig_pTopology);
    }
    SafeRelease(&pSinkActivate);
    SafeRelease(&pType);
    return hr;
}
void ImageGrabber::stopGrabbing()
{
    if(ig_pSession)
        ig_pSession->Stop();
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: Stopping of of grabbing of images\n", ig_DeviceID);
}
HRESULT ImageGrabber::startGrabbing(void)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    IMFMediaEvent *pEvent = NULL;
    PROPVARIANT var;
    PropVariantInit(&var);
    HRESULT hr = S_OK;
    CHECK_HR(hr = ig_pSession->SetTopology(0, ig_pTopology));
    CHECK_HR(hr = ig_pSession->Start(&GUID_NULL, &var));
    DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: Start Grabbing of the images\n", ig_DeviceID);
    for(;;)
    {
        HRESULT hrStatus = S_OK;
        MediaEventType met;
        if(!ig_pSession) break;
        hr = ig_pSession->GetEvent(0, &pEvent);
        if(!SUCCEEDED(hr))
        {
            hr = S_OK;
            goto done;
        }
        hr = pEvent->GetStatus(&hrStatus);
        if(!SUCCEEDED(hr))
        {
            hr = S_OK;
            goto done;
        }
        hr = pEvent->GetType(&met);
        if(!SUCCEEDED(hr))
        {
            hr = S_OK;
            goto done;
        }
        if (met == MESessionEnded)
        {
            DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: MESessionEnded \n", ig_DeviceID);
            ig_pSession->Stop();
            break;
        }
        if (met == MESessionStopped)
        {
            DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: MESessionStopped \n", ig_DeviceID);
            break;
        }
        if (met == MEVideoCaptureDeviceRemoved)
        {
            DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: MEVideoCaptureDeviceRemoved \n", ig_DeviceID);
            break;
        }
        SafeRelease(&pEvent);
    }
    DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: Finish startGrabbing \n", ig_DeviceID);
done:
    SafeRelease(&pEvent);
    SafeRelease(&ig_pSession);
    SafeRelease(&ig_pTopology);
    return hr;
}
HRESULT ImageGrabber::CreateTopology(IMFMediaSource *pSource, IMFActivate *pSinkActivate, IMFTopology **ppTopo)
{
    IMFTopology *pTopology = NULL;
    IMFPresentationDescriptor *pPD = NULL;
    IMFStreamDescriptor *pSD = NULL;
    IMFMediaTypeHandler *pHandler = NULL;
    IMFTopologyNode *pNode1 = NULL;
    IMFTopologyNode *pNode2 = NULL;
    HRESULT hr = S_OK;
    DWORD cStreams = 0;
    CHECK_HR(hr = MFCreateTopology(&pTopology));
    CHECK_HR(hr = pSource->CreatePresentationDescriptor(&pPD));
    CHECK_HR(hr = pPD->GetStreamDescriptorCount(&cStreams));
    for (DWORD i = 0; i < cStreams; i++)
    {
        // In this example, we look for audio streams and connect them to the sink.
        BOOL fSelected = FALSE;
        GUID majorType;
        CHECK_HR(hr = pPD->GetStreamDescriptorByIndex(i, &fSelected, &pSD));
        CHECK_HR(hr = pSD->GetMediaTypeHandler(&pHandler));
        CHECK_HR(hr = pHandler->GetMajorType(&majorType));
        if (majorType == MFMediaType_Video && fSelected)
        {
            CHECK_HR(hr = AddSourceNode(pTopology, pSource, pPD, pSD, &pNode1));
            CHECK_HR(hr = AddOutputNode(pTopology, pSinkActivate, 0, &pNode2));
            CHECK_HR(hr = pNode1->ConnectOutput(0, pNode2, 0));
            break;
        }
        else
        {
            CHECK_HR(hr = pPD->DeselectStream(i));
        }
        SafeRelease(&pSD);
        SafeRelease(&pHandler);
    }
    *ppTopo = pTopology;
    (*ppTopo)->AddRef();
done:
    SafeRelease(&pTopology);
    SafeRelease(&pNode1);
    SafeRelease(&pNode2);
    SafeRelease(&pPD);
    SafeRelease(&pSD);
    SafeRelease(&pHandler);
    return hr;
}
HRESULT ImageGrabber::AddSourceNode(
    IMFTopology *pTopology,           // Topology.
    IMFMediaSource *pSource,          // Media source.
    IMFPresentationDescriptor *pPD,   // Presentation descriptor.
    IMFStreamDescriptor *pSD,         // Stream descriptor.
    IMFTopologyNode **ppNode)         // Receives the node pointer.
{
    IMFTopologyNode *pNode = NULL;
    HRESULT hr = S_OK;
    CHECK_HR(hr = MFCreateTopologyNode(MF_TOPOLOGY_SOURCESTREAM_NODE, &pNode));
    CHECK_HR(hr = pNode->SetUnknown(MF_TOPONODE_SOURCE, pSource));
    CHECK_HR(hr = pNode->SetUnknown(MF_TOPONODE_PRESENTATION_DESCRIPTOR, pPD));
    CHECK_HR(hr = pNode->SetUnknown(MF_TOPONODE_STREAM_DESCRIPTOR, pSD));
    CHECK_HR(hr = pTopology->AddNode(pNode));
    // Return the pointer to the caller.
    *ppNode = pNode;
    (*ppNode)->AddRef();
done:
    SafeRelease(&pNode);
    return hr;
}
HRESULT ImageGrabber::AddOutputNode(
    IMFTopology *pTopology,     // Topology.
    IMFActivate *pActivate,     // Media sink activation object.
    DWORD dwId,                 // Identifier of the stream sink.
    IMFTopologyNode **ppNode)   // Receives the node pointer.
{
    IMFTopologyNode *pNode = NULL;
    HRESULT hr = S_OK;
    CHECK_HR(hr = MFCreateTopologyNode(MF_TOPOLOGY_OUTPUT_NODE, &pNode));
    CHECK_HR(hr = pNode->SetObject(pActivate));
    CHECK_HR(hr = pNode->SetUINT32(MF_TOPONODE_STREAMID, dwId));
    CHECK_HR(hr = pNode->SetUINT32(MF_TOPONODE_NOSHUTDOWN_ON_REMOVE, FALSE));
    CHECK_HR(hr = pTopology->AddNode(pNode));
    // Return the pointer to the caller.
    *ppNode = pNode;
    (*ppNode)->AddRef();
done:
    SafeRelease(&pNode);
    return hr;
}
HRESULT ImageGrabber::CreateInstance(ImageGrabber **ppIG, unsigned int deviceID)
{
    *ppIG = new (std::nothrow) ImageGrabber(deviceID);
    if (ppIG == NULL)
    {
        return E_OUTOFMEMORY;
    }
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    DPO->printOut(L"IMAGEGRABBER VIDEODEVICE %i: Creating instance of ImageGrabber\n", deviceID);
    return S_OK;
}
STDMETHODIMP ImageGrabber::QueryInterface(REFIID riid, void** ppv)
{
    HRESULT hr = E_NOINTERFACE;
    *ppv = NULL;
    if(riid == IID_IUnknown || riid == IID_IMFSampleGrabberSinkCallback)
    {
        *ppv = static_cast<IMFSampleGrabberSinkCallback *>(this);
        hr = S_OK;
    }
    if(riid == IID_IMFClockStateSink)
    {
        *ppv = static_cast<IMFClockStateSink *>(this);
        hr = S_OK;
    }
    if(SUCCEEDED(hr))
    {
        reinterpret_cast<IUnknown *>(*ppv)->AddRef();
    }
    return hr;
}
STDMETHODIMP_(ULONG) ImageGrabber::AddRef()
{
    return InterlockedIncrement(&m_cRef);
}
STDMETHODIMP_(ULONG) ImageGrabber::Release()
{
    ULONG cRef = InterlockedDecrement(&m_cRef);
    if (cRef == 0)
    {
        delete this;
    }
    return cRef;
}
STDMETHODIMP ImageGrabber::OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset)
{
    (void)hnsSystemTime;
    (void)llClockStartOffset;
    return S_OK;
}
STDMETHODIMP ImageGrabber::OnClockStop(MFTIME hnsSystemTime)
{
    (void)hnsSystemTime;
    return S_OK;
}
STDMETHODIMP ImageGrabber::OnClockPause(MFTIME hnsSystemTime)
{
    (void)hnsSystemTime;
    return S_OK;
}
STDMETHODIMP ImageGrabber::OnClockRestart(MFTIME hnsSystemTime)
{
    (void)hnsSystemTime;
    return S_OK;
}
STDMETHODIMP ImageGrabber::OnClockSetRate(MFTIME hnsSystemTime, float flRate)
{
    (void)flRate;
    (void)hnsSystemTime;
    return S_OK;
}
STDMETHODIMP ImageGrabber::OnSetPresentationClock(IMFPresentationClock* pClock)
{
    (void)pClock;
    return S_OK;
}
STDMETHODIMP ImageGrabber::OnProcessSample(REFGUID guidMajorMediaType, DWORD dwSampleFlags,
    LONGLONG llSampleTime, LONGLONG llSampleDuration, const BYTE * pSampleBuffer,
    DWORD dwSampleSize)
{
    (void)guidMajorMediaType;
    (void)llSampleTime;
    (void)dwSampleFlags;
    (void)llSampleDuration;
    (void)dwSampleSize;
    if(ig_RIE)
    {
        ig_RIFirst->fastCopy(pSampleBuffer);
        ig_RIOut = ig_RIFirst;
    }
    else
    {
        ig_RISecond->fastCopy(pSampleBuffer);
        ig_RIOut = ig_RISecond;
    }
    ig_RIE = !ig_RIE;
    return S_OK;
}
STDMETHODIMP ImageGrabber::OnShutdown()
{
    return S_OK;
}
RawImage *ImageGrabber::getRawImage()
{
    return ig_RIOut;
}
DWORD WINAPI MainThreadFunction( LPVOID lpParam )
{
    ImageGrabberThread *pIGT = (ImageGrabberThread *)lpParam;
    pIGT->run();
    return 0;
}
HRESULT ImageGrabberThread::CreateInstance(ImageGrabberThread **ppIGT, IMFMediaSource *pSource, unsigned int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    *ppIGT = new (std::nothrow) ImageGrabberThread(pSource, deviceID);
    if (ppIGT == NULL)
    {
        DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: Memory cannot be allocated\n", deviceID);
        return E_OUTOFMEMORY;
    }
    else
        DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: Creating of the instance of ImageGrabberThread\n", deviceID);
    return S_OK;
}
ImageGrabberThread::ImageGrabberThread(IMFMediaSource *pSource, unsigned int deviceID): igt_Handle(NULL), igt_stop(false)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    HRESULT hr = ImageGrabber::CreateInstance(&igt_pImageGrabber, deviceID);
    igt_DeviceID = deviceID;
    if(SUCCEEDED(hr))
    {
        hr = igt_pImageGrabber->initImageGrabber(pSource, MFVideoFormat_RGB24);
        if(!SUCCEEDED(hr))
        {
            DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: There is a problem with initialization of the instance of the ImageGrabber class\n", deviceID);
        }
        else
        {
            DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: Initialization of instance of the ImageGrabber class\n", deviceID);
        }
    }
    else
    {
        DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i There is a problem with creation of the instance of the ImageGrabber class\n", deviceID);
    }
}
void ImageGrabberThread::setEmergencyStopEvent(void *userData, void(*func)(int, void *))
{
    if(func)
    {
        igt_func = func;
        igt_userData = userData;
    }
}
ImageGrabberThread::~ImageGrabberThread(void)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: Destroing ImageGrabberThread\n", igt_DeviceID);
    delete igt_pImageGrabber;
}
void ImageGrabberThread::stop()
{
    igt_stop = true;
    if(igt_pImageGrabber)
    {
        igt_pImageGrabber->stopGrabbing();
    }
}
void ImageGrabberThread::start()
{
    igt_Handle = CreateThread(
            NULL,                   // default security attributes
            0,                      // use default stack size
            MainThreadFunction,       // thread function name
            this,          // argument to thread function
            0,                      // use default creation flags
            &igt_ThreadIdArray);   // returns the thread identifier
}
void ImageGrabberThread::run()
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if(igt_pImageGrabber)
    {
        DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: Thread for grabbing images is started\n", igt_DeviceID);
        HRESULT hr = igt_pImageGrabber->startGrabbing();
        if(!SUCCEEDED(hr))
        {
            DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: There is a problem with starting the process of grabbing\n", igt_DeviceID);
        }
    }
    else
    {
        DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i The thread is finished without execution of grabbing\n", igt_DeviceID);
    }
    if(!igt_stop)
    {
        DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: Emergency Stop thread\n", igt_DeviceID);
        if(igt_func)
        {
            igt_func(igt_DeviceID, igt_userData);
        }
    }
    else
        DPO->printOut(L"IMAGEGRABBERTHREAD VIDEODEVICE %i: Finish thread\n", igt_DeviceID);
}
ImageGrabber *ImageGrabberThread::getImageGrabber()
{
    return igt_pImageGrabber;
}
Media_Foundation::Media_Foundation(void)
{
    HRESULT hr = MFStartup(MF_VERSION);
    if(!SUCCEEDED(hr))
    {
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MEDIA FOUNDATION: It cannot be created!!!\n");
    }
}
Media_Foundation::~Media_Foundation(void)
{
    HRESULT hr = MFShutdown();
    if(!SUCCEEDED(hr))
    {
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MEDIA FOUNDATION: Resources cannot be released\n");
    }
}
bool Media_Foundation::buildListOfDevices()
{
    HRESULT hr = S_OK;
    IMFAttributes *pAttributes = NULL;
    CoInitialize(NULL);
    hr = MFCreateAttributes(&pAttributes, 1);
    if (SUCCEEDED(hr))
    {
        hr = pAttributes->SetGUID(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
            );
    }
    if (SUCCEEDED(hr))
    {
        videoDevices *vDs = &videoDevices::getInstance();
        hr = vDs->initDevices(pAttributes);
    }
    else
    {
       DebugPrintOut *DPO = &DebugPrintOut::getInstance();
       DPO->printOut(L"MEDIA FOUNDATION: The access to the video cameras denied\n");
    }
    SafeRelease(&pAttributes);
    return (SUCCEEDED(hr));
}
Media_Foundation& Media_Foundation::getInstance()
{
    static Media_Foundation instance;
    return instance;
}
RawImage::RawImage(unsigned int size): ri_new(false), ri_pixels(NULL)
{
    ri_size = size;
    ri_pixels = new unsigned char[size];
    memset((void *)ri_pixels,0,ri_size);
}
bool RawImage::isNew()
{
    return ri_new;
}
unsigned int RawImage::getSize()
{
    return ri_size;
}
RawImage::~RawImage(void)
{
    delete []ri_pixels;
    ri_pixels = NULL;
}
long RawImage::CreateInstance(RawImage **ppRImage,unsigned int size)
{
    *ppRImage = new (std::nothrow) RawImage(size);
    if (ppRImage == NULL)
    {
        return E_OUTOFMEMORY;
    }
    return S_OK;
}
void RawImage::setCopy(const BYTE * pSampleBuffer)
{
    memcpy(ri_pixels, pSampleBuffer, ri_size);
    ri_new = true;
}
void RawImage::fastCopy(const BYTE * pSampleBuffer)
{
    memcpy(ri_pixels, pSampleBuffer, ri_size);
    ri_new = true;
}
unsigned char * RawImage::getpPixels()
{
    ri_new = false;
    return ri_pixels;
}
videoDevice::videoDevice(void): vd_IsSetuped(false), vd_LockOut(OpenLock), vd_pFriendlyName(NULL),
    vd_Width(0), vd_Height(0), vd_pSource(NULL), vd_func(NULL), vd_userData(NULL)
{
}
void videoDevice::setParametrs(CamParametrs parametrs)
{
    if(vd_IsSetuped)
    {
        if(vd_pSource)
        {
            Parametr *pParametr = (Parametr *)(&parametrs);
            Parametr *pPrevParametr = (Parametr *)(&vd_PrevParametrs);
            IAMVideoProcAmp *pProcAmp = NULL;
            HRESULT hr = vd_pSource->QueryInterface(IID_PPV_ARGS(&pProcAmp));
            if (SUCCEEDED(hr))
            {
                for(unsigned int i = 0; i < 10; i++)
                {
                    if(pPrevParametr[i].CurrentValue != pParametr[i].CurrentValue || pPrevParametr[i].Flag != pParametr[i].Flag)
                        hr = pProcAmp->Set(VideoProcAmp_Brightness + i, pParametr[i].CurrentValue, pParametr[i].Flag);
                }
                pProcAmp->Release();
            }
            IAMCameraControl *pProcControl = NULL;
            hr = vd_pSource->QueryInterface(IID_PPV_ARGS(&pProcControl));
            if (SUCCEEDED(hr))
            {
                for(unsigned int i = 0; i < 7; i++)
                {
                    if(pPrevParametr[10 + i].CurrentValue != pParametr[10 + i].CurrentValue || pPrevParametr[10 + i].Flag != pParametr[10 + i].Flag)
                    hr = pProcControl->Set(CameraControl_Pan+i, pParametr[10 + i].CurrentValue, pParametr[10 + i].Flag);
                }
                pProcControl->Release();
            }
            vd_PrevParametrs = parametrs;
        }
    }
}
CamParametrs videoDevice::getParametrs()
{
    CamParametrs out;
    if(vd_IsSetuped)
    {
        if(vd_pSource)
        {
            Parametr *pParametr = (Parametr *)(&out);
            IAMVideoProcAmp *pProcAmp = NULL;
            HRESULT hr = vd_pSource->QueryInterface(IID_PPV_ARGS(&pProcAmp));
            if (SUCCEEDED(hr))
            {
                for(unsigned int i = 0; i < 10; i++)
                {
                    Parametr temp;
                    hr = pProcAmp->GetRange(VideoProcAmp_Brightness+i, &temp.Min, &temp.Max, &temp.Step, &temp.Default, &temp.Flag);
                    if (SUCCEEDED(hr))
                    {
                        temp.CurrentValue = temp.Default;
                        pParametr[i] = temp;
                    }
                }
                pProcAmp->Release();
            }
            IAMCameraControl *pProcControl = NULL;
            hr = vd_pSource->QueryInterface(IID_PPV_ARGS(&pProcControl));
            if (SUCCEEDED(hr))
            {
                for(unsigned int i = 0; i < 7; i++)
                {
                    Parametr temp;
                    hr = pProcControl->GetRange(CameraControl_Pan+i, &temp.Min, &temp.Max, &temp.Step, &temp.Default, &temp.Flag);
                    if (SUCCEEDED(hr))
                    {
                        temp.CurrentValue = temp.Default;
                        pParametr[10 + i] = temp;
                    }
                }
                pProcControl->Release();
            }
        }
    }
    return out;
}
long videoDevice::resetDevice(IMFActivate *pActivate)
{
    HRESULT hr = -1;
    vd_CurrentFormats.clear();
    if(vd_pFriendlyName)
        CoTaskMemFree(vd_pFriendlyName);
    vd_pFriendlyName = NULL;
    if(pActivate)
    {
        IMFMediaSource *pSource = NULL;
        hr = pActivate->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                &vd_pFriendlyName,
                NULL
                );
        hr = pActivate->ActivateObject(
            __uuidof(IMFMediaSource),
            (void**)&pSource
            );
        enumerateCaptureFormats(pSource);
        buildLibraryofTypes();
        SafeRelease(&pSource);
        if(FAILED(hr))
        {
            vd_pFriendlyName = NULL;
            DebugPrintOut *DPO = &DebugPrintOut::getInstance();
            DPO->printOut(L"VIDEODEVICE %i: IMFMediaSource interface cannot be created \n", vd_CurrentNumber);
        }
    }
    return hr;
}
long videoDevice::readInfoOfDevice(IMFActivate *pActivate, unsigned int Num)
{
    HRESULT hr = -1;
    vd_CurrentNumber = Num;
    hr = resetDevice(pActivate);
    return hr;
}
long videoDevice::checkDevice(IMFAttributes *pAttributes, IMFActivate **pDevice)
{
    HRESULT hr = S_OK;
    IMFActivate **ppDevices = NULL;
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    UINT32 count;
    wchar_t *newFriendlyName = NULL;
    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
    if (SUCCEEDED(hr))
    {
        if(count > 0)
        {
            if(count > vd_CurrentNumber)
            {
                hr = ppDevices[vd_CurrentNumber]->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                &newFriendlyName,
                NULL
                );
                if (SUCCEEDED(hr))
                {
                    if(wcscmp(newFriendlyName, vd_pFriendlyName) != 0)
                    {
                        DPO->printOut(L"VIDEODEVICE %i: Chosen device cannot be found \n", vd_CurrentNumber);
                        hr = -1;
                        pDevice = NULL;
                    }
                    else
                    {
                        *pDevice = ppDevices[vd_CurrentNumber];
                        (*pDevice)->AddRef();
                    }
                }
                else
                {
                    DPO->printOut(L"VIDEODEVICE %i: Name of device cannot be gotten \n", vd_CurrentNumber);
                }
            }
            else
            {
                DPO->printOut(L"VIDEODEVICE %i: Number of devices more than corrent number of the device \n", vd_CurrentNumber);
                hr = -1;
            }
            for(UINT32 i = 0; i < count; i++)
            {
                SafeRelease(&ppDevices[i]);
            }
            SafeRelease(ppDevices);
        }
        else
            hr = -1;
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE %i: List of DeviceSources cannot be enumerated \n", vd_CurrentNumber);
    }
    return hr;
}
long videoDevice::initDevice()
{
    HRESULT hr = -1;
    IMFAttributes *pAttributes = NULL;
    IMFActivate * vd_pActivate= NULL;
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    CoInitialize(NULL);
    hr = MFCreateAttributes(&pAttributes, 1);
    if (SUCCEEDED(hr))
    {
        hr = pAttributes->SetGUID(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
            );
    }
    if (SUCCEEDED(hr))
    {
        hr = checkDevice(pAttributes, &vd_pActivate);
        if (SUCCEEDED(hr) && vd_pActivate)
        {
            SafeRelease(&vd_pSource);
            hr = vd_pActivate->ActivateObject(
                __uuidof(IMFMediaSource),
                (void**)&vd_pSource
                );
            if (SUCCEEDED(hr))
            {
            }
            SafeRelease(&vd_pActivate);
        }
        else
        {
            DPO->printOut(L"VIDEODEVICE %i: Device there is not \n", vd_CurrentNumber);
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE %i: The attribute of video cameras cannot be getting \n", vd_CurrentNumber);
    }
    SafeRelease(&pAttributes);
    return hr;
}
MediaType videoDevice::getFormat(unsigned int id)
{
    if(id < vd_CurrentFormats.size())
    {
        return vd_CurrentFormats[id];
    }
    else return MediaType();
}
int videoDevice::getCountFormats()
{
    return vd_CurrentFormats.size();
}
void videoDevice::setEmergencyStopEvent(void *userData, void(*func)(int, void *))
{
    vd_func = func;
    vd_userData = userData;
}
void videoDevice::closeDevice()
{
    if(vd_IsSetuped)
    {
        vd_IsSetuped = false;
        vd_pSource->Stop();
        SafeRelease(&vd_pSource);
        if(vd_LockOut == RawDataLock)
        {
            vd_pImGrTh->stop();
            Sleep(500);
            delete vd_pImGrTh;
        }
        vd_pImGrTh = NULL;
        vd_LockOut = OpenLock;
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"VIDEODEVICE %i: Device is stopped \n", vd_CurrentNumber);
    }
}
unsigned int videoDevice::getWidth()
{
    if(vd_IsSetuped)
        return vd_Width;
    else
        return 0;
}
unsigned int videoDevice::getHeight()
{
    if(vd_IsSetuped)
        return vd_Height;
    else
        return 0;
}
IMFMediaSource *videoDevice::getMediaSource()
{
    IMFMediaSource *out = NULL;
    if(vd_LockOut == OpenLock)
    {
        vd_LockOut = MediaSourceLock;
        out = vd_pSource;
    }
    return out;
}
int videoDevice::findType(unsigned int size, unsigned int frameRate)
{
    if(vd_CaptureFormats.size() == 0)
        return 0;
    FrameRateMap FRM = vd_CaptureFormats[size];
    if(FRM.size() == 0)
        return 0;
    UINT64 frameRateMax = 0;  SUBTYPEMap STMMax;
    if(frameRate == 0)
    {
        std::map<UINT64, SUBTYPEMap>::iterator f = FRM.begin();
        for(; f != FRM.end(); f++)
        {
             if((*f).first >= frameRateMax)
             {
                 frameRateMax = (*f).first;
                 STMMax = (*f).second;
             }
        }
    }
    else
    {
        std::map<UINT64, SUBTYPEMap>::iterator f = FRM.begin();
        for(; f != FRM.end(); f++)
        {
             if((*f).first >= frameRateMax)
             {
                 if(frameRate > (*f).first)
                 {
                     frameRateMax = (*f).first;
                     STMMax = (*f).second;
                 }
             }
        }
    }
    if(STMMax.size() == 0)
        return 0;
    std::map<String, vectorNum>::iterator S = STMMax.begin();
    vectorNum VN = (*S).second;
    if(VN.size() == 0)
        return 0;
    return VN[0];
}
void videoDevice::buildLibraryofTypes()
{
    unsigned int size;
    unsigned int framerate;
    std::vector<MediaType>::iterator i = vd_CurrentFormats.begin();
    int count = 0;
    for(; i != vd_CurrentFormats.end(); i++)
    {
        size = (*i).MF_MT_FRAME_SIZE;
        framerate = (*i).MF_MT_FRAME_RATE;
        FrameRateMap FRM = vd_CaptureFormats[size];
        SUBTYPEMap STM = FRM[framerate];
        String subType((*i).pMF_MT_SUBTYPEName);
        vectorNum VN = STM[subType];
        VN.push_back(count);
        STM[subType] = VN;
        FRM[framerate] = STM;
        vd_CaptureFormats[size] = FRM;
        count++;
    }
}
long videoDevice::setDeviceFormat(IMFMediaSource *pSource, unsigned long  dwFormatIndex)
{
    IMFPresentationDescriptor *pPD = NULL;
    IMFStreamDescriptor *pSD = NULL;
    IMFMediaTypeHandler *pHandler = NULL;
    IMFMediaType *pType = NULL;
    HRESULT hr = pSource->CreatePresentationDescriptor(&pPD);
    if (FAILED(hr))
    {
        goto done;
    }
    BOOL fSelected;
    hr = pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD);
    if (FAILED(hr))
    {
        goto done;
    }
    hr = pSD->GetMediaTypeHandler(&pHandler);
    if (FAILED(hr))
    {
        goto done;
    }
    hr = pHandler->GetMediaTypeByIndex((DWORD)dwFormatIndex, &pType);
    if (FAILED(hr))
    {
        goto done;
    }
    hr = pHandler->SetCurrentMediaType(pType);
done:
    SafeRelease(&pPD);
    SafeRelease(&pSD);
    SafeRelease(&pHandler);
    SafeRelease(&pType);
    return hr;
}
bool videoDevice::isDeviceSetup()
{
    return vd_IsSetuped;
}
RawImage * videoDevice::getRawImageOut()
{
    if(!vd_IsSetuped) return NULL;
    if(vd_pImGrTh)
            return vd_pImGrTh->getImageGrabber()->getRawImage();
    else
    {
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"VIDEODEVICE %i: The instance of ImageGrabberThread class does not exist  \n", vd_CurrentNumber);
    }
    return NULL;
}
bool videoDevice::isFrameNew()
{
    if(!vd_IsSetuped) return false;
    if(vd_LockOut == RawDataLock || vd_LockOut == OpenLock)
    {
        if(vd_LockOut == OpenLock)
        {
            vd_LockOut = RawDataLock;
            HRESULT hr = ImageGrabberThread::CreateInstance(&vd_pImGrTh, vd_pSource, vd_CurrentNumber);
            if(FAILED(hr))
            {
                DebugPrintOut *DPO = &DebugPrintOut::getInstance();
                DPO->printOut(L"VIDEODEVICE %i: The instance of ImageGrabberThread class cannot be created.\n", vd_CurrentNumber);
                return false;
            }
            vd_pImGrTh->setEmergencyStopEvent(vd_userData, vd_func);
            vd_pImGrTh->start();
            return true;
        }
        if(vd_pImGrTh)
            return vd_pImGrTh->getImageGrabber()->getRawImage()->isNew();
    }
    return false;
}
bool videoDevice::isDeviceMediaSource()
{
    if(vd_LockOut == MediaSourceLock) return true;
    return false;
}
bool videoDevice::isDeviceRawDataSource()
{
    if(vd_LockOut == RawDataLock) return true;
    return false;
}
bool videoDevice::setupDevice(unsigned int id)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if(!vd_IsSetuped)
    {
        HRESULT hr = -1;
        hr = initDevice();
        if(SUCCEEDED(hr))
        {
            vd_Width = vd_CurrentFormats[id].width;
            vd_Height = vd_CurrentFormats[id].height;
            hr = setDeviceFormat(vd_pSource, (DWORD) id);
            vd_IsSetuped = (SUCCEEDED(hr));
            if(vd_IsSetuped)
                DPO->printOut(L"\n\nVIDEODEVICE %i: Device is setuped \n", vd_CurrentNumber);
            vd_PrevParametrs = getParametrs();
            return vd_IsSetuped;
        }
        else
        {
            DPO->printOut(L"VIDEODEVICE %i: Interface IMFMediaSource cannot be got \n", vd_CurrentNumber);
            return false;
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE %i: Device is setuped already \n", vd_CurrentNumber);
        return false;
    }
}
bool videoDevice::setupDevice(unsigned int w, unsigned int h, unsigned int idealFramerate)
{
    unsigned int id = findType(w * h, idealFramerate);
    return setupDevice(id);
}
wchar_t *videoDevice::getName()
{
    return vd_pFriendlyName;
}
videoDevice::~videoDevice(void)
{
    closeDevice();
    SafeRelease(&vd_pSource);
    if(vd_pFriendlyName)
        CoTaskMemFree(vd_pFriendlyName);
}
long videoDevice::enumerateCaptureFormats(IMFMediaSource *pSource)
{
    IMFPresentationDescriptor *pPD = NULL;
    IMFStreamDescriptor *pSD = NULL;
    IMFMediaTypeHandler *pHandler = NULL;
    IMFMediaType *pType = NULL;
    HRESULT hr = pSource->CreatePresentationDescriptor(&pPD);
    if (FAILED(hr))
    {
        goto done;
    }
    BOOL fSelected;
    hr = pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD);
    if (FAILED(hr))
    {
        goto done;
    }
    hr = pSD->GetMediaTypeHandler(&pHandler);
    if (FAILED(hr))
    {
        goto done;
    }
    DWORD cTypes = 0;
    hr = pHandler->GetMediaTypeCount(&cTypes);
    if (FAILED(hr))
    {
        goto done;
    }
    for (DWORD i = 0; i < cTypes; i++)
    {
        hr = pHandler->GetMediaTypeByIndex(i, &pType);
        if (FAILED(hr))
        {
            goto done;
        }
        MediaType MT = FormatReader::Read(pType);
        vd_CurrentFormats.push_back(MT);
        SafeRelease(&pType);
    }
done:
    SafeRelease(&pPD);
    SafeRelease(&pSD);
    SafeRelease(&pHandler);
    SafeRelease(&pType);
    return hr;
}
videoDevices::videoDevices(void): count(0)
{}
void videoDevices::clearDevices()
{
    std::vector<videoDevice *>::iterator i = vds_Devices.begin();
    for(; i != vds_Devices.end(); ++i)
        delete (*i);
    vds_Devices.clear();
}
videoDevices::~videoDevices(void)
{
    clearDevices();
}
videoDevice * videoDevices::getDevice(unsigned int i)
{
    if(i >= vds_Devices.size())
    {
        return NULL;
    }
    if(i < 0)
    {
        return NULL;
    }
    return vds_Devices[i];
}
long videoDevices::initDevices(IMFAttributes *pAttributes)
{
    HRESULT hr = S_OK;
    IMFActivate **ppDevices = NULL;
    clearDevices();
    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
    if (SUCCEEDED(hr))
    {
        if(count > 0)
        {
            for(UINT32 i = 0; i < count; i++)
            {
                videoDevice *vd = new videoDevice;
                vd->readInfoOfDevice(ppDevices[i], i);
                vds_Devices.push_back(vd);
                SafeRelease(&ppDevices[i]);
            }
            SafeRelease(ppDevices);
        }
        else
            hr = -1;
    }
    else
    {
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"VIDEODEVICES: The instances of the videoDevice class cannot be created\n");
    }
    return hr;
}
size_t videoDevices::getCount()
{
    return vds_Devices.size();
}
videoDevices& videoDevices::getInstance()
{
    static videoDevices instance;
    return instance;
}
Parametr::Parametr()
{
    CurrentValue = 0;
    Min = 0;
    Max = 0;
    Step = 0;
    Default = 0;
    Flag = 0;
}
MediaType::MediaType()
{
    pMF_MT_AM_FORMAT_TYPEName = NULL;
    pMF_MT_MAJOR_TYPEName = NULL;
    pMF_MT_SUBTYPEName = NULL;
    Clear();
}
MediaType::~MediaType()
{
    Clear();
}
void MediaType::Clear()
{
    MF_MT_FRAME_SIZE = 0;
    height = 0;
    width = 0;
    MF_MT_YUV_MATRIX = 0;
    MF_MT_VIDEO_LIGHTING = 0;
    MF_MT_DEFAULT_STRIDE = 0;
    MF_MT_VIDEO_CHROMA_SITING = 0;
    MF_MT_FIXED_SIZE_SAMPLES = 0;
    MF_MT_VIDEO_NOMINAL_RANGE = 0;
    MF_MT_FRAME_RATE = 0;
    MF_MT_FRAME_RATE_low = 0;
    MF_MT_PIXEL_ASPECT_RATIO = 0;
    MF_MT_PIXEL_ASPECT_RATIO_low = 0;
    MF_MT_ALL_SAMPLES_INDEPENDENT = 0;
    MF_MT_FRAME_RATE_RANGE_MIN = 0;
    MF_MT_FRAME_RATE_RANGE_MIN_low = 0;
    MF_MT_SAMPLE_SIZE = 0;
    MF_MT_VIDEO_PRIMARIES = 0;
    MF_MT_INTERLACE_MODE = 0;
    MF_MT_FRAME_RATE_RANGE_MAX = 0;
    MF_MT_FRAME_RATE_RANGE_MAX_low = 0;
    memset(&MF_MT_MAJOR_TYPE, 0, sizeof(GUID));
    memset(&MF_MT_AM_FORMAT_TYPE, 0, sizeof(GUID));
    memset(&MF_MT_SUBTYPE, 0, sizeof(GUID));
}
videoInput::videoInput(void): accessToDevices(false)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    DPO->printOut(L"\n***** VIDEOINPUT LIBRARY - 2013 (Author: Evgeny Pereguda) *****\n\n");
    updateListOfDevices();
    if(!accessToDevices)
        DPO->printOut(L"INITIALIZATION: Ther is not any suitable video device\n");
}
void videoInput::updateListOfDevices()
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    Media_Foundation *MF = &Media_Foundation::getInstance();
    accessToDevices = MF->buildListOfDevices();
    if(!accessToDevices)
        DPO->printOut(L"UPDATING: Ther is not any suitable video device\n");
}
videoInput::~videoInput(void)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    DPO->printOut(L"\n***** CLOSE VIDEOINPUT LIBRARY - 2013 *****\n\n");
}
IMFMediaSource *videoInput::getMediaSource(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
        {
            IMFMediaSource *out = VD->getMediaSource();
            if(!out)
                DPO->printOut(L"VideoDevice %i: There is not any suitable IMFMediaSource interface\n", deviceID);
            return out;
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return NULL;
}
bool videoInput::setupDevice(int deviceID, unsigned int id)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0 )
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return false;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
        {
            bool out = VD->setupDevice(id);
            if(!out)
                DPO->printOut(L"VIDEODEVICE %i: This device cannot be started\n", deviceID);
            return out;
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return false;
}
bool videoInput::setupDevice(int deviceID, unsigned int w, unsigned int h, unsigned int idealFramerate)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0 )
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return false;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
        {
            bool out = VD->setupDevice(w, h, idealFramerate);
            if(!out)
                DPO->printOut(L"VIDEODEVICE %i: this device cannot be started\n", deviceID);
            return out;
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n", deviceID);
    }
    return false;
}
MediaType videoInput::getFormat(int deviceID, unsigned int id)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return MediaType();
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
            return VD->getFormat(id);
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return MediaType();
}
bool videoInput::isDeviceSetup(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return false;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
            return VD->isDeviceSetup();
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return false;
}
bool videoInput::isDeviceMediaSource(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return false;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
            return VD->isDeviceMediaSource();
    }
    else
    {
        DPO->printOut(L"Device(s): There is not any suitable video device\n");
    }
    return false;
}
bool videoInput::isDeviceRawDataSource(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return false;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
        {
            bool isRaw = VD->isDeviceRawDataSource();
            return isRaw;
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return false;
}
bool videoInput::isFrameNew(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return false;
    }
    if(accessToDevices)
    {
        if(!isDeviceSetup(deviceID))
        {
            if(isDeviceMediaSource(deviceID))
                return false;
        }
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
        {
            return VD->isFrameNew();
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return false;
}
unsigned int videoInput::getCountFormats(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return 0;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
            return VD->getCountFormats();
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return 0;
}
void videoInput::closeAllDevices()
{
    videoDevices *VDS = &videoDevices::getInstance();
    for(unsigned int i = 0; i < VDS->getCount(); i++)
        closeDevice(i);
}
void videoInput::setParametrs(int deviceID, CamParametrs parametrs)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice *VD = VDS->getDevice(deviceID);
        if(VD)
            VD->setParametrs(parametrs);
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
}
CamParametrs videoInput::getParametrs(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    CamParametrs out;
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return out;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice *VD = VDS->getDevice(deviceID);
        if(VD)
            out = VD->getParametrs();
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return out;
}
void videoInput::closeDevice(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice *VD = VDS->getDevice(deviceID);
        if(VD)
            VD->closeDevice();
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
}
unsigned int videoInput::getWidth(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return 0;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
            return VD->getWidth();
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return 0;
}
unsigned int videoInput::getHeight(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return 0;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
            return VD->getHeight();
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return 0;
}
wchar_t *videoInput::getNameVideoDevice(int deviceID)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return NULL;
    }
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        videoDevice * VD = VDS->getDevice(deviceID);
        if(VD)
            return VD->getName();
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return L"Empty";
}
unsigned int videoInput::listDevices(bool silent)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    int out = 0;
    if(accessToDevices)
    {
        videoDevices *VDS = &videoDevices::getInstance();
        out = VDS->getCount();
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        if(!silent)DPO->printOut(L"\nVIDEOINPUT SPY MODE!\n\n");
        if(!silent)DPO->printOut(L"SETUP: Looking For Capture Devices\n");
        for(int i = 0; i < out; i++)
        {
            if(!silent)DPO->printOut(L"SETUP: %i) %s \n",i, getNameVideoDevice(i));
        }
        if(!silent)DPO->printOut(L"SETUP: %i Device(s) found\n\n", out);
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return out;
}
videoInput& videoInput::getInstance()
{
    static videoInput instance;
    return instance;
}
bool videoInput::isDevicesAcceable()
{
    return accessToDevices;
}
void videoInput::setVerbose(bool state)
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    DPO->setVerbose(state);
}
void videoInput::setEmergencyStopEvent(int deviceID, void *userData, void(*func)(int, void *))
{
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return;
    }
    if(accessToDevices)
    {
        if(func)
        {
            videoDevices *VDS = &videoDevices::getInstance();
            videoDevice * VD = VDS->getDevice(deviceID);
            if(VD)
                VD->setEmergencyStopEvent(userData, func);
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
}
bool videoInput::getPixels(int deviceID, unsigned char * dstBuffer, bool flipRedAndBlue, bool flipImage)
{
    bool success = false;
    unsigned int bytes = 3;
    DebugPrintOut *DPO = &DebugPrintOut::getInstance();
    if (deviceID < 0)
    {
        DPO->printOut(L"VIDEODEVICE %i: Invalid device ID\n", deviceID);
        return success;
    }
    if(accessToDevices)
    {
        bool isRaw = isDeviceRawDataSource(deviceID);
        if(isRaw)
        {
            videoDevices *VDS = &videoDevices::getInstance();
            DebugPrintOut *DPO = &DebugPrintOut::getInstance();
            RawImage *RIOut = VDS->getDevice(deviceID)->getRawImageOut();
            if(RIOut)
            {
                unsigned int height = VDS->getDevice(deviceID)->getHeight();
                unsigned int width  = VDS->getDevice(deviceID)->getWidth();
                unsigned int size = bytes * width * height;
                if(size == RIOut->getSize())
                {
                    processPixels(RIOut->getpPixels(), dstBuffer, width, height, bytes, flipRedAndBlue, flipImage);
                    success = true;
                }
                else
                {
                    DPO->printOut(L"ERROR: GetPixels() - bufferSizes do not match!\n");
                }
            }
            else
            {
                DPO->printOut(L"ERROR: GetPixels() - Unable to grab frame for device %i\n", deviceID);
            }
        }
        else
        {
            DPO->printOut(L"ERROR: GetPixels() - Not raw data source device %i\n", deviceID);
        }
    }
    else
    {
        DPO->printOut(L"VIDEODEVICE(s): There is not any suitable video device\n");
    }
    return success;
}
void videoInput::processPixels(unsigned char * src, unsigned char * dst, unsigned int width,
                                unsigned int height, unsigned int bpp, bool bRGB, bool bFlip)
{
    unsigned int widthInBytes = width * bpp;
    unsigned int numBytes = widthInBytes * height;
    int *dstInt, *srcInt;
    if(!bRGB)
    {
        if(bFlip)
        {
            for(unsigned int y = 0; y < height; y++)
            {
                dstInt = (int *)(dst + (y * widthInBytes));
                srcInt = (int *)(src + ( (height -y -1) * widthInBytes));
                memcpy(dstInt, srcInt, widthInBytes);
            }
        }
        else
        {
            memcpy(dst, src, numBytes);
        }
    }
    else
    {
        if(bFlip)
        {
            unsigned int x = 0;
            unsigned int y = (height - 1) * widthInBytes;
            src += y;
            for(unsigned int i = 0; i < numBytes; i+=3)
            {
                if(x >= width)
                {
                    x = 0;
                    src -= widthInBytes*2;
                }
                *dst = *(src+2);
                dst++;
                *dst = *(src+1);
                dst++;
                *dst = *src;
                dst++;
                src+=3;
                x++;
            }
        }
        else
        {
            for(unsigned int i = 0; i < numBytes; i+=3)
            {
                *dst = *(src+2);
                dst++;
                *dst = *(src+1);
                dst++;
                *dst = *src;
                dst++;
                src+=3;
            }
        }
    }
}
}
/******* Capturing video from camera via Microsoft Media Foundation **********/
class CvCaptureCAM_MSMF : public CvCapture
{
public:
    CvCaptureCAM_MSMF();
    virtual ~CvCaptureCAM_MSMF();
    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain() { return CV_CAP_MSMF; } // Return the type of the capture object: CV_CAP_VFW, etc...
protected:
    void init();
    int index, width, height,fourcc;
    int widthSet, heightSet;
    IplImage* frame;
    videoInput VI;
};
struct SuppressVideoInputMessages
{
    SuppressVideoInputMessages() { videoInput::setVerbose(true); }
};
static SuppressVideoInputMessages do_it;
CvCaptureCAM_MSMF::CvCaptureCAM_MSMF():
    index(-1),
    width(-1),
    height(-1),
    fourcc(-1),
    widthSet(-1),
    heightSet(-1),
    frame(0),
    VI(videoInput::getInstance())
{
    CoInitialize(0);
}
CvCaptureCAM_MSMF::~CvCaptureCAM_MSMF()
{
    close();
    CoUninitialize();
}
void CvCaptureCAM_MSMF::close()
{
    if( index >= 0 )
    {
        VI.closeDevice(index);
        index = -1;
        cvReleaseImage(&frame);
    }
    widthSet = heightSet = width = height = -1;
}
// Initialize camera input
bool CvCaptureCAM_MSMF::open( int _index )
{
    int try_index = _index;
    int devices = 0;
    close();
    devices = VI.listDevices(true);
    if (devices == 0)
        return false;
    try_index = try_index < 0 ? 0 : (try_index > devices-1 ? devices-1 : try_index);
    VI.setupDevice(try_index);
    if( !VI.isFrameNew(try_index) )
        return false;
    index = try_index;
    return true;
}
bool CvCaptureCAM_MSMF::grabFrame()
{
    return true;
}
IplImage* CvCaptureCAM_MSMF::retrieveFrame(int)
{
    if( !frame || (int)VI.getWidth(index) != frame->width || (int)VI.getHeight(index) != frame->height )
    {
        if (frame)
            cvReleaseImage( &frame );
        unsigned int w = VI.getWidth(index), h = VI.getHeight(index);
        frame = cvCreateImage( cvSize(w,h), 8, 3 );
    }
    VI.getPixels( index, (uchar*)frame->imageData, false, true );
    return frame;
}
double CvCaptureCAM_MSMF::getProperty( int property_id )
{
    // image format proprrties
    switch( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        return VI.getWidth(index);
    case CV_CAP_PROP_FRAME_HEIGHT:
        return VI.getHeight(index);
    case CV_CAP_PROP_FOURCC:
        // FIXME: implement method in VideoInput back end
        //return VI.getFourcc(index);
        ;
    case CV_CAP_PROP_FPS:
        // FIXME: implement method in VideoInput back end
        //return VI.getFPS(index);
        ;
    }
    // video filter properties
    switch( property_id )
    {
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_HUE:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_SHARPNESS:
    case CV_CAP_PROP_GAMMA:
    case CV_CAP_PROP_MONOCROME:
    case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
    case CV_CAP_PROP_BACKLIGHT:
    case CV_CAP_PROP_GAIN:
        // FIXME: implement method in VideoInput back end
        // if ( VI.getVideoSettingFilter(index, VI.getVideoPropertyFromCV(property_id), min_value,
        //                               max_value, stepping_delta, current_value, flags,defaultValue) )
        //     return (double)current_value;
        return 0.;
    }
    // camera properties
    switch( property_id )
    {
    case CV_CAP_PROP_PAN:
    case CV_CAP_PROP_TILT:
    case CV_CAP_PROP_ROLL:
    case CV_CAP_PROP_ZOOM:
    case CV_CAP_PROP_EXPOSURE:
    case CV_CAP_PROP_IRIS:
    case CV_CAP_PROP_FOCUS:
    // FIXME: implement method in VideoInput back end
    //     if (VI.getVideoSettingCamera(index,VI.getCameraPropertyFromCV(property_id),min_value,
    //          max_value,stepping_delta,current_value,flags,defaultValue) ) return (double)current_value;
        return 0.;
    }
    // unknown parameter or value not available
    return -1;
}
bool CvCaptureCAM_MSMF::setProperty( int property_id, double value )
{
    // image capture properties
    bool handled = false;
    switch( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        width = cvRound(value);
        handled = true;
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
        height = cvRound(value);
        handled = true;
        break;
    case CV_CAP_PROP_FOURCC:
        fourcc = (int)(unsigned long)(value);
        if ( fourcc == -1 ) {
            // following cvCreateVideo usage will pop up caprturepindialog here if fourcc=-1
            // TODO - how to create a capture pin dialog
        }
        handled = true;
        break;
    case CV_CAP_PROP_FPS:
        // FIXME: implement method in VideoInput back end
        // int fps = cvRound(value);
        // if (fps != VI.getFPS(index))
        // {
        //     VI.stopDevice(index);
        //     VI.setIdealFramerate(index,fps);
        //     if (widthSet > 0 && heightSet > 0)
        //         VI.setupDevice(index, widthSet, heightSet);
        //     else
        //         VI.setupDevice(index);
        // }
        // return VI.isDeviceSetup(index);
        ;
    }
    if ( handled ) {
        // a stream setting
        if( width > 0 && height > 0 )
        {
            if( width != (int)VI.getWidth(index) || height != (int)VI.getHeight(index) )//|| fourcc != VI.getFourcc(index) )
            {
                // FIXME: implement method in VideoInput back end
                // int fps = static_cast<int>(VI.getFPS(index));
                // VI.stopDevice(index);
                // VI.setIdealFramerate(index, fps);
                // VI.setupDeviceFourcc(index, width, height, fourcc);
            }
            bool success = VI.isDeviceSetup(index);
            if (success)
            {
                widthSet = width;
                heightSet = height;
                width = height = fourcc = -1;
            }
            return success;
        }
        return true;
    }
    // show video/camera filter dialog
    // FIXME: implement method in VideoInput back end
    // if ( property_id == CV_CAP_PROP_SETTINGS ) {
    //     VI.showSettingsWindow(index);
    //     return true;
    // }
    //video Filter properties
    switch( property_id )
    {
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_HUE:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_SHARPNESS:
    case CV_CAP_PROP_GAMMA:
    case CV_CAP_PROP_MONOCROME:
    case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
    case CV_CAP_PROP_BACKLIGHT:
    case CV_CAP_PROP_GAIN:
        // FIXME: implement method in VideoInput back end
        //return VI.setVideoSettingFilter(index,VI.getVideoPropertyFromCV(property_id),(long)value);
        ;
    }
    //camera properties
    switch( property_id )
    {
    case CV_CAP_PROP_PAN:
    case CV_CAP_PROP_TILT:
    case CV_CAP_PROP_ROLL:
    case CV_CAP_PROP_ZOOM:
    case CV_CAP_PROP_EXPOSURE:
    case CV_CAP_PROP_IRIS:
    case CV_CAP_PROP_FOCUS:
        // FIXME: implement method in VideoInput back end
        //return VI.setVideoSettingCamera(index,VI.getCameraPropertyFromCV(property_id),(long)value);
        ;
    }
    return false;
}
CvCapture* cvCreateCameraCapture_MSMF( int index )
{
    CvCaptureCAM_MSMF* capture = new CvCaptureCAM_MSMF;
    try
    {
        if( capture->open( index ))
            return capture;
    }
    catch(...)
    {
        delete capture;
        throw;
    }
    delete capture;
    return 0;
}
#endif