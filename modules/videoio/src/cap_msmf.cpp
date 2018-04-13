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
#if defined _WIN32 && defined HAVE_MSMF
/*
   Media Foundation-based Video Capturing module is based on
   videoInput library by Evgeny Pereguda:
   http://www.codeproject.com/Articles/559437/Capturing-of-video-from-web-camera-on-Windows-7-an
   Originally licensed under The Code Project Open License (CPOL) 1.02:
   http://www.codeproject.com/info/cpol10.aspx
*/
//require Windows 8 for some of the formats defined otherwise could baseline on lower version
#if WINVER < _WIN32_WINNT_WIN8
#undef WINVER
#define WINVER _WIN32_WINNT_WIN8
#endif
#include <windows.h>
#include <guiddef.h>
#include <mfidl.h>
#include <Mfapi.h>
#include <mfplay.h>
#include <mfobjects.h>
#include <tchar.h>
#include <strsafe.h>
#include <Mfreadwrite.h>
#include <new>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#ifdef _MSC_VER
#pragma warning(disable:4503)
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mf")
#pragma comment(lib, "mfuuid")
#pragma comment(lib, "Strmiids")
#pragma comment(lib, "Mfreadwrite")
#if (WINVER >= 0x0602) // Available since Win 8
#pragma comment(lib, "MinCore_Downlevel")
#endif
#endif

#include <mferror.h>

#include <comdef.h>

struct IMFMediaType;
struct IMFActivate;
struct IMFMediaSource;
struct IMFAttributes;

namespace
{

#ifdef _DEBUG
void DPOprintOut(const wchar_t *format, ...)
{
    int i = 0;
    wchar_t *p = NULL;
    va_list args;
    va_start(args, format);
    if (::IsDebuggerPresent())
    {
        WCHAR szMsg[512];
        ::StringCchVPrintfW(szMsg, sizeof(szMsg) / sizeof(szMsg[0]), format, args);
        ::OutputDebugStringW(szMsg);
    }
    else
    {
        if (wcscmp(format, L"%i"))
        {
            i = va_arg(args, int);
        }
        if (wcscmp(format, L"%s"))
        {
            p = va_arg(args, wchar_t *);
        }
        wprintf(format, i, p);
    }
    va_end(args);
}
#define DebugPrintOut(...) DPOprintOut(__VA_ARGS__)
#else
#define DebugPrintOut(...) void()
#endif

template <class T>
class ComPtr
{
public:
    ComPtr() throw()
    {
    }
    ComPtr(T* lp) throw()
    {
        p = lp;
    }
    ComPtr(_In_ const ComPtr<T>& lp) throw()
    {
        p = lp.p;
    }
    virtual ~ComPtr()
    {
    }

    T** operator&() throw()
    {
        assert(p == NULL);
        return p.operator&();
    }
    T* operator->() const throw()
    {
        assert(p != NULL);
        return p.operator->();
    }
    bool operator!() const throw()
    {
        return p.operator==(NULL);
    }
    bool operator==(_In_opt_ T* pT) const throw()
    {
        return p.operator==(pT);
    }
    bool operator!=(_In_opt_ T* pT) const throw()
    {
        return p.operator!=(pT);
    }
    operator bool()
    {
        return p.operator!=(NULL);
    }

    T* const* GetAddressOf() const throw()
    {
        return &p;
    }

    T** GetAddressOf() throw()
    {
        return &p;
    }

    T** ReleaseAndGetAddressOf() throw()
    {
        p.Release();
        return &p;
    }

    T* Get() const throw()
    {
        return p;
    }

    // Attach to an existing interface (does not AddRef)
    void Attach(_In_opt_ T* p2) throw()
    {
        p.Attach(p2);
    }
    // Detach the interface (does not Release)
    T* Detach() throw()
    {
        return p.Detach();
    }
    _Check_return_ HRESULT CopyTo(_Deref_out_opt_ T** ppT) throw()
    {
        assert(ppT != NULL);
        if (ppT == NULL)
            return E_POINTER;
        *ppT = p;
        if (p != NULL)
            p->AddRef();
        return S_OK;
    }

    void Reset()
    {
        p.Release();
    }

    // query for U interface
    template<typename U>
    HRESULT As(_Inout_ U** lp) const throw()
    {
        return p->QueryInterface(__uuidof(U), reinterpret_cast<void**>(lp));
    }
    // query for U interface
    template<typename U>
    HRESULT As(_Out_ ComPtr<U>* lp) const throw()
    {
        return p->QueryInterface(__uuidof(U), reinterpret_cast<void**>(lp->ReleaseAndGetAddressOf()));
    }
private:
    _COM_SMARTPTR_TYPEDEF(T, __uuidof(T));
    TPtr p;
};

#define _ComPtr ComPtr

// Structure for collecting info about types of video, which are supported by current video device
struct MediaType
{
    unsigned int MF_MT_FRAME_SIZE;
    UINT32 height;
    UINT32 width;
    unsigned int MF_MT_YUV_MATRIX;
    unsigned int MF_MT_VIDEO_LIGHTING;
    int MF_MT_DEFAULT_STRIDE; // stride is negative if image is bottom-up
    unsigned int MF_MT_VIDEO_CHROMA_SITING;
    GUID MF_MT_AM_FORMAT_TYPE;
    LPCWSTR pMF_MT_AM_FORMAT_TYPEName;
    unsigned int MF_MT_FIXED_SIZE_SAMPLES;
    unsigned int MF_MT_VIDEO_NOMINAL_RANGE;
    UINT32 MF_MT_FRAME_RATE_NUMERATOR;
    UINT32 MF_MT_FRAME_RATE_DENOMINATOR;
    UINT32 MF_MT_PIXEL_ASPECT_RATIO;
    UINT32 MF_MT_PIXEL_ASPECT_RATIO_low;
    unsigned int MF_MT_ALL_SAMPLES_INDEPENDENT;
    UINT32 MF_MT_FRAME_RATE_RANGE_MIN;
    UINT32 MF_MT_FRAME_RATE_RANGE_MIN_low;
    unsigned int MF_MT_SAMPLE_SIZE;
    unsigned int MF_MT_VIDEO_PRIMARIES;
    unsigned int MF_MT_INTERLACE_MODE;
    UINT32 MF_MT_FRAME_RATE_RANGE_MAX;
    UINT32 MF_MT_FRAME_RATE_RANGE_MAX_low;
    GUID MF_MT_MAJOR_TYPE;
    GUID MF_MT_SUBTYPE;
    LPCWSTR pMF_MT_MAJOR_TYPEName;
    LPCWSTR pMF_MT_SUBTYPEName;
    MediaType();
    MediaType(IMFMediaType *pType);
    ~MediaType();
    void Clear();
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
    Parametr()
    {
        CurrentValue = 0;
        Min = 0;
        Max = 0;
        Step = 0;
        Default = 0;
        Flag = 0;
    }
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

CamParametrs videoDevice__getParametrs(IMFMediaSource* vd_pSource)
{
    CamParametrs out;
    if (vd_pSource)
    {
        Parametr *pParametr = (Parametr *)(&out);
        IAMVideoProcAmp *pProcAmp = NULL;
        HRESULT hr = vd_pSource->QueryInterface(IID_PPV_ARGS(&pProcAmp));
        if (SUCCEEDED(hr))
        {
            for (unsigned int i = 0; i < 10; i++)
            {
                Parametr temp;
                hr = pProcAmp->GetRange(VideoProcAmp_Brightness + i, &temp.Min, &temp.Max, &temp.Step, &temp.Default, &temp.Flag);
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
            for (unsigned int i = 0; i < 7; i++)
            {
                Parametr temp;
                hr = pProcControl->GetRange(CameraControl_Pan + i, &temp.Min, &temp.Max, &temp.Step, &temp.Default, &temp.Flag);
                if (SUCCEEDED(hr))
                {
                    temp.CurrentValue = temp.Default;
                    pParametr[10 + i] = temp;
                }
            }
            pProcControl->Release();
        }
    }
    return out;
}

void videoDevice__setParametrs(IMFMediaSource* vd_pSource, CamParametrs parametrs)
{
    if (vd_pSource)
    {
        CamParametrs vd_PrevParametrs = videoDevice__getParametrs(vd_pSource);
        Parametr *pParametr = (Parametr *)(&parametrs);
        Parametr *pPrevParametr = (Parametr *)(&vd_PrevParametrs);
        IAMVideoProcAmp *pProcAmp = NULL;
        HRESULT hr = vd_pSource->QueryInterface(IID_PPV_ARGS(&pProcAmp));
        if (SUCCEEDED(hr))
        {
            for (unsigned int i = 0; i < 10; i++)
            {
                if (pPrevParametr[i].CurrentValue != pParametr[i].CurrentValue || pPrevParametr[i].Flag != pParametr[i].Flag)
                    hr = pProcAmp->Set(VideoProcAmp_Brightness + i, pParametr[i].CurrentValue, pParametr[i].Flag);
            }
            pProcAmp->Release();
        }
        IAMCameraControl *pProcControl = NULL;
        hr = vd_pSource->QueryInterface(IID_PPV_ARGS(&pProcControl));
        if (SUCCEEDED(hr))
        {
            for (unsigned int i = 0; i < 7; i++)
            {
                if (pPrevParametr[10 + i].CurrentValue != pParametr[10 + i].CurrentValue || pPrevParametr[10 + i].Flag != pParametr[10 + i].Flag)
                    hr = pProcControl->Set(CameraControl_Pan + i, pParametr[10 + i].CurrentValue, pParametr[10 + i].Flag);
            }
            pProcControl->Release();
        }
    }
}

// Class for creating of Media Foundation context
class Media_Foundation
{
public:
    ~Media_Foundation(void) { /*CV_Assert(SUCCEEDED(MFShutdown()));*/ }
    static Media_Foundation& getInstance()
    {
        static Media_Foundation instance;
        return instance;
    }
private:
    Media_Foundation(void) { CV_Assert(SUCCEEDED(MFStartup(MF_VERSION))); }
};

#ifndef IF_GUID_EQUAL_RETURN
#define IF_GUID_EQUAL_RETURN(val) if(val == guid) return L#val
#endif
LPCWSTR GetGUIDNameConstNew(const GUID& guid)
{
    IF_GUID_EQUAL_RETURN(MF_MT_MAJOR_TYPE);
    IF_GUID_EQUAL_RETURN(MF_MT_SUBTYPE);
    IF_GUID_EQUAL_RETURN(MF_MT_ALL_SAMPLES_INDEPENDENT);
    IF_GUID_EQUAL_RETURN(MF_MT_FIXED_SIZE_SAMPLES);
    IF_GUID_EQUAL_RETURN(MF_MT_COMPRESSED);
    IF_GUID_EQUAL_RETURN(MF_MT_SAMPLE_SIZE);
    IF_GUID_EQUAL_RETURN(MF_MT_WRAPPED_TYPE);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_NUM_CHANNELS);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_SAMPLES_PER_SECOND);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_AVG_BYTES_PER_SECOND);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_BLOCK_ALIGNMENT);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_BITS_PER_SAMPLE);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_VALID_BITS_PER_SAMPLE);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_SAMPLES_PER_BLOCK);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_CHANNEL_MASK);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_FOLDDOWN_MATRIX);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_WMADRC_PEAKREF);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_WMADRC_PEAKTARGET);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_WMADRC_AVGREF);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_WMADRC_AVGTARGET);
    IF_GUID_EQUAL_RETURN(MF_MT_AUDIO_PREFER_WAVEFORMATEX);
    IF_GUID_EQUAL_RETURN(MF_MT_AAC_PAYLOAD_TYPE);
    IF_GUID_EQUAL_RETURN(MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION);
    IF_GUID_EQUAL_RETURN(MF_MT_FRAME_SIZE);
    IF_GUID_EQUAL_RETURN(MF_MT_FRAME_RATE);
    IF_GUID_EQUAL_RETURN(MF_MT_FRAME_RATE_RANGE_MAX);
    IF_GUID_EQUAL_RETURN(MF_MT_FRAME_RATE_RANGE_MIN);
    IF_GUID_EQUAL_RETURN(MF_MT_PIXEL_ASPECT_RATIO);
    IF_GUID_EQUAL_RETURN(MF_MT_DRM_FLAGS);
    IF_GUID_EQUAL_RETURN(MF_MT_PAD_CONTROL_FLAGS);
    IF_GUID_EQUAL_RETURN(MF_MT_SOURCE_CONTENT_HINT);
    IF_GUID_EQUAL_RETURN(MF_MT_VIDEO_CHROMA_SITING);
    IF_GUID_EQUAL_RETURN(MF_MT_INTERLACE_MODE);
    IF_GUID_EQUAL_RETURN(MF_MT_TRANSFER_FUNCTION);
    IF_GUID_EQUAL_RETURN(MF_MT_VIDEO_PRIMARIES);
    IF_GUID_EQUAL_RETURN(MF_MT_CUSTOM_VIDEO_PRIMARIES);
    IF_GUID_EQUAL_RETURN(MF_MT_YUV_MATRIX);
    IF_GUID_EQUAL_RETURN(MF_MT_VIDEO_LIGHTING);
    IF_GUID_EQUAL_RETURN(MF_MT_VIDEO_NOMINAL_RANGE);
    IF_GUID_EQUAL_RETURN(MF_MT_GEOMETRIC_APERTURE);
    IF_GUID_EQUAL_RETURN(MF_MT_MINIMUM_DISPLAY_APERTURE);
    IF_GUID_EQUAL_RETURN(MF_MT_PAN_SCAN_APERTURE);
    IF_GUID_EQUAL_RETURN(MF_MT_PAN_SCAN_ENABLED);
    IF_GUID_EQUAL_RETURN(MF_MT_AVG_BITRATE);
    IF_GUID_EQUAL_RETURN(MF_MT_AVG_BIT_ERROR_RATE);
    IF_GUID_EQUAL_RETURN(MF_MT_MAX_KEYFRAME_SPACING);
    IF_GUID_EQUAL_RETURN(MF_MT_DEFAULT_STRIDE);
    IF_GUID_EQUAL_RETURN(MF_MT_PALETTE);
    IF_GUID_EQUAL_RETURN(MF_MT_USER_DATA);
    IF_GUID_EQUAL_RETURN(MF_MT_AM_FORMAT_TYPE);
    IF_GUID_EQUAL_RETURN(MF_MT_MPEG_START_TIME_CODE);
    IF_GUID_EQUAL_RETURN(MF_MT_MPEG2_PROFILE);
    IF_GUID_EQUAL_RETURN(MF_MT_MPEG2_LEVEL);
    IF_GUID_EQUAL_RETURN(MF_MT_MPEG2_FLAGS);
    IF_GUID_EQUAL_RETURN(MF_MT_MPEG_SEQUENCE_HEADER);
    IF_GUID_EQUAL_RETURN(MF_MT_DV_AAUX_SRC_PACK_0);
    IF_GUID_EQUAL_RETURN(MF_MT_DV_AAUX_CTRL_PACK_0);
    IF_GUID_EQUAL_RETURN(MF_MT_DV_AAUX_SRC_PACK_1);
    IF_GUID_EQUAL_RETURN(MF_MT_DV_AAUX_CTRL_PACK_1);
    IF_GUID_EQUAL_RETURN(MF_MT_DV_VAUX_SRC_PACK);
    IF_GUID_EQUAL_RETURN(MF_MT_DV_VAUX_CTRL_PACK);
    IF_GUID_EQUAL_RETURN(MF_MT_ARBITRARY_HEADER);
    IF_GUID_EQUAL_RETURN(MF_MT_ARBITRARY_FORMAT);
    IF_GUID_EQUAL_RETURN(MF_MT_IMAGE_LOSS_TOLERANT);
    IF_GUID_EQUAL_RETURN(MF_MT_MPEG4_SAMPLE_DESCRIPTION);
    IF_GUID_EQUAL_RETURN(MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY);
    IF_GUID_EQUAL_RETURN(MF_MT_ORIGINAL_4CC);
    IF_GUID_EQUAL_RETURN(MF_MT_ORIGINAL_WAVE_FORMAT_TAG);
    // Media types
    IF_GUID_EQUAL_RETURN(MFMediaType_Audio);
    IF_GUID_EQUAL_RETURN(MFMediaType_Video);
    IF_GUID_EQUAL_RETURN(MFMediaType_Protected);
#ifdef MFMediaType_Perception
    IF_GUID_EQUAL_RETURN(MFMediaType_Perception);
#endif
    IF_GUID_EQUAL_RETURN(MFMediaType_Stream);
    IF_GUID_EQUAL_RETURN(MFMediaType_SAMI);
    IF_GUID_EQUAL_RETURN(MFMediaType_Script);
    IF_GUID_EQUAL_RETURN(MFMediaType_Image);
    IF_GUID_EQUAL_RETURN(MFMediaType_HTML);
    IF_GUID_EQUAL_RETURN(MFMediaType_Binary);
    IF_GUID_EQUAL_RETURN(MFMediaType_FileTransfer);
    IF_GUID_EQUAL_RETURN(MFVideoFormat_AI44); //     FCC('AI44')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_ARGB32); //   D3DFMT_A8R8G8B8
    IF_GUID_EQUAL_RETURN(MFVideoFormat_AYUV); //     FCC('AYUV')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_DV25); //     FCC('dv25')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_DV50); //     FCC('dv50')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_DVH1); //     FCC('dvh1')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_DVC);
    IF_GUID_EQUAL_RETURN(MFVideoFormat_DVHD);
    IF_GUID_EQUAL_RETURN(MFVideoFormat_DVSD); //     FCC('dvsd')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_DVSL); //     FCC('dvsl')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_H264); //     FCC('H264')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_I420); //     FCC('I420')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_IYUV); //     FCC('IYUV')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_M4S2); //     FCC('M4S2')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MJPG);
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MP43); //     FCC('MP43')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MP4S); //     FCC('MP4S')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MP4V); //     FCC('MP4V')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MPG1); //     FCC('MPG1')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MSS1); //     FCC('MSS1')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MSS2); //     FCC('MSS2')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_NV11); //     FCC('NV11')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_NV12); //     FCC('NV12')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_P010); //     FCC('P010')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_P016); //     FCC('P016')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_P210); //     FCC('P210')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_P216); //     FCC('P216')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_RGB24); //    D3DFMT_R8G8B8
    IF_GUID_EQUAL_RETURN(MFVideoFormat_RGB32); //    D3DFMT_X8R8G8B8
    IF_GUID_EQUAL_RETURN(MFVideoFormat_RGB555); //   D3DFMT_X1R5G5B5
    IF_GUID_EQUAL_RETURN(MFVideoFormat_RGB565); //   D3DFMT_R5G6B5
    IF_GUID_EQUAL_RETURN(MFVideoFormat_RGB8);
    IF_GUID_EQUAL_RETURN(MFVideoFormat_UYVY); //     FCC('UYVY')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_v210); //     FCC('v210')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_v410); //     FCC('v410')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_WMV1); //     FCC('WMV1')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_WMV2); //     FCC('WMV2')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_WMV3); //     FCC('WMV3')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_WVC1); //     FCC('WVC1')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_Y210); //     FCC('Y210')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_Y216); //     FCC('Y216')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_Y410); //     FCC('Y410')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_Y416); //     FCC('Y416')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_Y41P);
    IF_GUID_EQUAL_RETURN(MFVideoFormat_Y41T);
    IF_GUID_EQUAL_RETURN(MFVideoFormat_YUY2); //     FCC('YUY2')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_YV12); //     FCC('YV12')
    IF_GUID_EQUAL_RETURN(MFVideoFormat_YVYU);
#ifdef MFVideoFormat_H263
    IF_GUID_EQUAL_RETURN(MFVideoFormat_H263);
#endif
#ifdef MFVideoFormat_H265
    IF_GUID_EQUAL_RETURN(MFVideoFormat_H265);
#endif
#ifdef MFVideoFormat_H264_ES
    IF_GUID_EQUAL_RETURN(MFVideoFormat_H264_ES);
#endif
#ifdef MFVideoFormat_HEVC
    IF_GUID_EQUAL_RETURN(MFVideoFormat_HEVC);
#endif
#ifdef MFVideoFormat_HEVC_ES
    IF_GUID_EQUAL_RETURN(MFVideoFormat_HEVC_ES);
#endif
#ifdef MFVideoFormat_MPEG2
    IF_GUID_EQUAL_RETURN(MFVideoFormat_MPEG2);
#endif
#ifdef MFVideoFormat_VP80
    IF_GUID_EQUAL_RETURN(MFVideoFormat_VP80);
#endif
#ifdef MFVideoFormat_VP90
    IF_GUID_EQUAL_RETURN(MFVideoFormat_VP90);
#endif
#ifdef MFVideoFormat_420O
    IF_GUID_EQUAL_RETURN(MFVideoFormat_420O);
#endif
#ifdef MFVideoFormat_Y42T
    IF_GUID_EQUAL_RETURN(MFVideoFormat_Y42T);
#endif
#ifdef MFVideoFormat_YVU9
    IF_GUID_EQUAL_RETURN(MFVideoFormat_YVU9);
#endif
#ifdef MFVideoFormat_v216
    IF_GUID_EQUAL_RETURN(MFVideoFormat_v216);
#endif
#ifdef MFVideoFormat_L8
    IF_GUID_EQUAL_RETURN(MFVideoFormat_L8);
#endif
#ifdef MFVideoFormat_L16
    IF_GUID_EQUAL_RETURN(MFVideoFormat_L16);
#endif
#ifdef MFVideoFormat_D16
    IF_GUID_EQUAL_RETURN(MFVideoFormat_D16);
#endif
#ifdef D3DFMT_X8R8G8B8
    IF_GUID_EQUAL_RETURN(D3DFMT_X8R8G8B8);
#endif
#ifdef D3DFMT_A8R8G8B8
    IF_GUID_EQUAL_RETURN(D3DFMT_A8R8G8B8);
#endif
#ifdef D3DFMT_R8G8B8
    IF_GUID_EQUAL_RETURN(D3DFMT_R8G8B8);
#endif
#ifdef D3DFMT_X1R5G5B5
    IF_GUID_EQUAL_RETURN(D3DFMT_X1R5G5B5);
#endif
#ifdef D3DFMT_A4R4G4B4
    IF_GUID_EQUAL_RETURN(D3DFMT_A4R4G4B4);
#endif
#ifdef D3DFMT_R5G6B5
    IF_GUID_EQUAL_RETURN(D3DFMT_R5G6B5);
#endif
#ifdef D3DFMT_P8
    IF_GUID_EQUAL_RETURN(D3DFMT_P8);
#endif
#ifdef D3DFMT_A2R10G10B10
    IF_GUID_EQUAL_RETURN(D3DFMT_A2R10G10B10);
#endif
#ifdef D3DFMT_A2B10G10R10
    IF_GUID_EQUAL_RETURN(D3DFMT_A2B10G10R10);
#endif
#ifdef D3DFMT_L8
    IF_GUID_EQUAL_RETURN(D3DFMT_L8);
#endif
#ifdef D3DFMT_L16
    IF_GUID_EQUAL_RETURN(D3DFMT_L16);
#endif
#ifdef D3DFMT_D16
    IF_GUID_EQUAL_RETURN(D3DFMT_D16);
#endif
#ifdef MFVideoFormat_A2R10G10B10
    IF_GUID_EQUAL_RETURN(MFVideoFormat_A2R10G10B10);
#endif
#ifdef MFVideoFormat_A16B16G16R16F
    IF_GUID_EQUAL_RETURN(MFVideoFormat_A16B16G16R16F);
#endif
    IF_GUID_EQUAL_RETURN(MFAudioFormat_PCM); //              WAVE_FORMAT_PCM
    IF_GUID_EQUAL_RETURN(MFAudioFormat_Float); //            WAVE_FORMAT_IEEE_FLOAT
    IF_GUID_EQUAL_RETURN(MFAudioFormat_DTS); //              WAVE_FORMAT_DTS
    IF_GUID_EQUAL_RETURN(MFAudioFormat_Dolby_AC3_SPDIF); //  WAVE_FORMAT_DOLBY_AC3_SPDIF
    IF_GUID_EQUAL_RETURN(MFAudioFormat_DRM); //              WAVE_FORMAT_DRM
    IF_GUID_EQUAL_RETURN(MFAudioFormat_WMAudioV8); //        WAVE_FORMAT_WMAUDIO2
    IF_GUID_EQUAL_RETURN(MFAudioFormat_WMAudioV9); //        WAVE_FORMAT_WMAUDIO3
    IF_GUID_EQUAL_RETURN(MFAudioFormat_WMAudio_Lossless); // WAVE_FORMAT_WMAUDIO_LOSSLESS
    IF_GUID_EQUAL_RETURN(MFAudioFormat_WMASPDIF); //         WAVE_FORMAT_WMASPDIF
    IF_GUID_EQUAL_RETURN(MFAudioFormat_MSP1); //             WAVE_FORMAT_WMAVOICE9
    IF_GUID_EQUAL_RETURN(MFAudioFormat_MP3); //              WAVE_FORMAT_MPEGLAYER3
    IF_GUID_EQUAL_RETURN(MFAudioFormat_MPEG); //             WAVE_FORMAT_MPEG
    IF_GUID_EQUAL_RETURN(MFAudioFormat_AAC); //              WAVE_FORMAT_MPEG_HEAAC
    IF_GUID_EQUAL_RETURN(MFAudioFormat_ADTS); //             WAVE_FORMAT_MPEG_ADTS_AAC
#ifdef MFAudioFormat_ALAC
    IF_GUID_EQUAL_RETURN(MFAudioFormat_ALAC);
#endif
#ifdef MFAudioFormat_AMR_NB
    IF_GUID_EQUAL_RETURN(MFAudioFormat_AMR_NB);
#endif
#ifdef MFAudioFormat_AMR_WB
    IF_GUID_EQUAL_RETURN(MFAudioFormat_AMR_WB);
#endif
#ifdef MFAudioFormat_AMR_WP
    IF_GUID_EQUAL_RETURN(MFAudioFormat_AMR_WP);
#endif
#ifdef MFAudioFormat_Dolby_AC3
    IF_GUID_EQUAL_RETURN(MFAudioFormat_Dolby_AC3);
#endif
#ifdef MFAudioFormat_Dolby_DDPlus
    IF_GUID_EQUAL_RETURN(MFAudioFormat_Dolby_DDPlus);
#endif
#ifdef MFAudioFormat_FLAC
    IF_GUID_EQUAL_RETURN(MFAudioFormat_FLAC);
#endif
#ifdef MFAudioFormat_Opus
    IF_GUID_EQUAL_RETURN(MFAudioFormat_Opus);
#endif
#ifdef MEDIASUBTYPE_RAW_AAC1
    IF_GUID_EQUAL_RETURN(MEDIASUBTYPE_RAW_AAC1);
#endif
#ifdef MFAudioFormat_Float_SpatialObjects
    IF_GUID_EQUAL_RETURN(MFAudioFormat_Float_SpatialObjects);
#endif
#ifdef MFAudioFormat_QCELP
    IF_GUID_EQUAL_RETURN(MFAudioFormat_QCELP);
#endif

    return NULL;
}

bool LogAttributeValueByIndexNew(IMFAttributes *pAttr, DWORD index, MediaType &out)
{
    PROPVARIANT var;
    PropVariantInit(&var);
    GUID guid = { 0 };
    if (SUCCEEDED(pAttr->GetItemByIndex(index, &guid, &var)))
    {
        if (guid == MF_MT_DEFAULT_STRIDE && var.vt == VT_INT)
            out.MF_MT_DEFAULT_STRIDE = var.intVal;
        else if (guid == MF_MT_FRAME_RATE && var.vt == VT_UI8)
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.MF_MT_FRAME_RATE_NUMERATOR, &out.MF_MT_FRAME_RATE_DENOMINATOR);
        else if (guid == MF_MT_FRAME_RATE_RANGE_MAX && var.vt == VT_UI8)
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.MF_MT_FRAME_RATE_RANGE_MAX, &out.MF_MT_FRAME_RATE_RANGE_MAX_low);
        else if (guid == MF_MT_FRAME_RATE_RANGE_MIN && var.vt == VT_UI8)
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.MF_MT_FRAME_RATE_RANGE_MIN, &out.MF_MT_FRAME_RATE_RANGE_MIN_low);
        else if (guid == MF_MT_PIXEL_ASPECT_RATIO && var.vt == VT_UI8)
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.MF_MT_PIXEL_ASPECT_RATIO, &out.MF_MT_PIXEL_ASPECT_RATIO_low);
        else if (guid == MF_MT_YUV_MATRIX && var.vt == VT_UI4)
            out.MF_MT_YUV_MATRIX = var.ulVal;
        else if (guid == MF_MT_VIDEO_LIGHTING && var.vt == VT_UI4)
            out.MF_MT_VIDEO_LIGHTING = var.ulVal;
        else if (guid == MF_MT_DEFAULT_STRIDE && var.vt == VT_UI4)
            out.MF_MT_DEFAULT_STRIDE = (int)var.ulVal;
        else if (guid == MF_MT_VIDEO_CHROMA_SITING && var.vt == VT_UI4)
            out.MF_MT_VIDEO_CHROMA_SITING = var.ulVal;
        else if (guid == MF_MT_VIDEO_NOMINAL_RANGE && var.vt == VT_UI4)
            out.MF_MT_VIDEO_NOMINAL_RANGE = var.ulVal;
        else if (guid == MF_MT_ALL_SAMPLES_INDEPENDENT && var.vt == VT_UI4)
            out.MF_MT_ALL_SAMPLES_INDEPENDENT = var.ulVal;
        else if (guid == MF_MT_FIXED_SIZE_SAMPLES && var.vt == VT_UI4)
            out.MF_MT_FIXED_SIZE_SAMPLES = var.ulVal;
        else if (guid == MF_MT_SAMPLE_SIZE && var.vt == VT_UI4)
            out.MF_MT_SAMPLE_SIZE = var.ulVal;
        else if (guid == MF_MT_VIDEO_PRIMARIES && var.vt == VT_UI4)
            out.MF_MT_VIDEO_PRIMARIES = var.ulVal;
        else if (guid == MF_MT_INTERLACE_MODE && var.vt == VT_UI4)
            out.MF_MT_INTERLACE_MODE = var.ulVal;
        else if (guid == MF_MT_AM_FORMAT_TYPE && var.vt == VT_CLSID)
            out.MF_MT_AM_FORMAT_TYPE = *var.puuid;
        else if (guid == MF_MT_MAJOR_TYPE && var.vt == VT_CLSID)
            out.pMF_MT_MAJOR_TYPEName = GetGUIDNameConstNew(out.MF_MT_MAJOR_TYPE = *var.puuid);
        else if (guid == MF_MT_SUBTYPE && var.vt == VT_CLSID)
            out.pMF_MT_SUBTYPEName = GetGUIDNameConstNew(out.MF_MT_SUBTYPE = *var.puuid);
        else if (guid == MF_MT_FRAME_SIZE && var.vt == VT_UI8)
        {
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.width, &out.height);
            out.MF_MT_FRAME_SIZE = out.width * out.height;
        }
        PropVariantClear(&var);
        return true;
    }
    return false;
}

MediaType::MediaType()
{
    pMF_MT_MAJOR_TYPEName = NULL;
    pMF_MT_SUBTYPEName = NULL;
    Clear();
}

MediaType::MediaType(IMFMediaType *pType)
{
    UINT32 count = 0;
    if (SUCCEEDED(pType->GetCount(&count)) &&
        SUCCEEDED(pType->LockStore()))
    {
        for (UINT32 i = 0; i < count; i++)
            if (!LogAttributeValueByIndexNew(pType, i, *this))
                break;
        pType->UnlockStore();
    }
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
    MF_MT_FRAME_RATE_NUMERATOR = 0;
    MF_MT_FRAME_RATE_DENOMINATOR = 0;
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

}

/******* Capturing video from camera via Microsoft Media Foundation **********/
class CvCaptureCAM_MSMF : public CvCapture
{
public:
    CvCaptureCAM_MSMF();
    virtual ~CvCaptureCAM_MSMF();
    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual IplImage* retrieveFrame(int) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE { return CV_CAP_MSMF; } // Return the type of the capture object: CV_CAP_VFW, etc...
protected:
    double getFramerate(MediaType MT) const;
    bool configureOutput(unsigned int width, unsigned int height, unsigned int prefFramerate);
    Media_Foundation& MF;
    _ComPtr<IMFSourceReader> videoFileSource;
    DWORD dwStreamIndex;
    MediaType captureFormat;
    _ComPtr<IMFSample> videoSample;
    IplImage* frame;
    bool isOpened;
};

CvCaptureCAM_MSMF::CvCaptureCAM_MSMF():
    MF(Media_Foundation::getInstance()),
    videoFileSource(NULL),
    videoSample(NULL),
    frame(NULL),
    isOpened(false)
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
    if (isOpened)
    {
        isOpened = false;
        if (videoSample)
            videoSample.Reset();
        if (videoFileSource)
            videoFileSource.Reset();
        if (frame)
            cvReleaseImage(&frame);
    }
}

bool CvCaptureCAM_MSMF::configureOutput(unsigned int width, unsigned int height, unsigned int prefFramerate)
{
    HRESULT hr = S_OK;
    int dwStreamFallback = -1;
    MediaType MTFallback;
    int dwStreamBest = -1;
    MediaType MTBest;

    DWORD dwMediaTypeTest = 0;
    DWORD dwStreamTest = 0;
    while (SUCCEEDED(hr))
    {
        _ComPtr<IMFMediaType> pType;
        hr = videoFileSource->GetNativeMediaType(dwStreamTest, dwMediaTypeTest, &pType);
        if (hr == MF_E_NO_MORE_TYPES)
        {
            hr = S_OK;
            ++dwStreamTest;
            dwMediaTypeTest = 0;
        }
        else if (SUCCEEDED(hr))
        {
            MediaType MT(pType.Get());
            if (MT.MF_MT_MAJOR_TYPE == MFMediaType_Video)
            {
                if (dwStreamFallback < 0 ||
                    ((MT.width * MT.height) > (MTFallback.width * MTFallback.height)) ||
                    (((MT.width * MT.height) == (MTFallback.width * MTFallback.height)) && getFramerate(MT) > getFramerate(MTFallback) && (prefFramerate == 0 || getFramerate(MT) <= prefFramerate)))
                {
                    dwStreamFallback = (int)dwStreamTest;
                    MTFallback = MT;
                }
                if (MT.width == width && MT.height == height)
                {
                    if (dwStreamBest < 0 ||
                        (getFramerate(MT) > getFramerate(MTBest) && (prefFramerate == 0 || getFramerate(MT) <= prefFramerate)))
                    {
                        dwStreamBest = (int)dwStreamTest;
                        MTBest = MT;
                    }
                }
            }
            ++dwMediaTypeTest;
        }
    }
    if (dwStreamBest >= 0 || dwStreamFallback >= 0)
    {
        // Retrieved stream media type
        DWORD tryStream = (DWORD)(dwStreamBest >= 0 ? dwStreamBest : dwStreamFallback);
        MediaType tryMT = dwStreamBest >= 0 ? MTBest : MTFallback;
        _ComPtr<IMFMediaType>  mediaTypeOut;
        if (// Set the output media type.
            SUCCEEDED(MFCreateMediaType(&mediaTypeOut)) &&
            SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) &&
            SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB24)) &&
            SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)) &&
            SUCCEEDED(MFSetAttributeRatio(mediaTypeOut.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) &&
            SUCCEEDED(MFSetAttributeSize(mediaTypeOut.Get(), MF_MT_FRAME_SIZE, tryMT.width, tryMT.height)) &&
            SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_DEFAULT_STRIDE, 3 * tryMT.width)))//Assume BGR24 input
        {
            if (SUCCEEDED(videoFileSource->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, false)) &&
                SUCCEEDED(videoFileSource->SetStreamSelection(tryStream, true)) &&
                SUCCEEDED(videoFileSource->SetCurrentMediaType(tryStream, NULL, mediaTypeOut.Get()))
                )
            {
                dwStreamIndex = tryStream;
                captureFormat = tryMT;
                return true;
            }
            else
                close();
        }
    }
    return false;
}

// Initialize camera input
bool CvCaptureCAM_MSMF::open(int _index)
{
    close();

    _ComPtr<IMFAttributes> msAttr = NULL;
    if (SUCCEEDED(MFCreateAttributes(msAttr.GetAddressOf(), 1)) &&
        SUCCEEDED(msAttr->SetGUID(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
        )))
    {
        IMFActivate **ppDevices = NULL;
        UINT32 count;
        if (SUCCEEDED(MFEnumDeviceSources(msAttr.Get(), &ppDevices, &count)))
        {
            if (count > 0)
            {
                _index = std::min(std::max(0, _index), (int)count - 1);
                for (int ind = 0; ind < (int)count; ind++)
                {
                    if (ind == _index && ppDevices[ind])
                    {
                        // Set source reader parameters
                        _ComPtr<IMFMediaSource> mSrc;
                        _ComPtr<IMFAttributes> srAttr;
                        if (SUCCEEDED(ppDevices[ind]->ActivateObject(__uuidof(IMFMediaSource), (void**)&mSrc)) && mSrc &&
                            SUCCEEDED(MFCreateAttributes(&srAttr, 10)) &&
                            SUCCEEDED(srAttr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, true)) &&
                            SUCCEEDED(srAttr->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, false)) &&
                            SUCCEEDED(srAttr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, false)) &&
                            SUCCEEDED(srAttr->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, true)) &&
                            //ToDo: Enable D3D MF_SOURCE_READER_D3D_MANAGER attribute
                            SUCCEEDED(MFCreateSourceReaderFromMediaSource(mSrc.Get(), srAttr.Get(), &videoFileSource)))
                        {
                            isOpened = true;
                            configureOutput(0, 0, 0);
                        }
                    }
                    if (ppDevices[ind])
                        ppDevices[ind]->Release();
                }
            }
        }
        CoTaskMemFree(ppDevices);
    }

    return isOpened;
}
bool CvCaptureCAM_MSMF::grabFrame()
{
    if (isOpened)
    {
        DWORD streamIndex, flags;
        LONGLONG llTimeStamp;
        if (videoSample)
            videoSample.Reset();
        HRESULT hr;
        while(SUCCEEDED(hr = videoFileSource->ReadSample(
                                                            dwStreamIndex, // Stream index.
                                                            0,             // Flags.
                                                            &streamIndex,  // Receives the actual stream index.
                                                            &flags,        // Receives status flags.
                                                            &llTimeStamp,  // Receives the time stamp.
                                                            &videoSample   // Receives the sample or NULL.
                                                        )) &&
              streamIndex == dwStreamIndex && !(flags & (MF_SOURCE_READERF_ERROR|MF_SOURCE_READERF_ALLEFFECTSREMOVED|MF_SOURCE_READERF_ENDOFSTREAM)) &&
              !videoSample
             )
        {
            if (flags & MF_SOURCE_READERF_STREAMTICK)
            {
                DebugPrintOut(L"\tStream tick detected. Retrying to grab the frame\n");
            }
        }

        if (SUCCEEDED(hr))
        {
            if (streamIndex != dwStreamIndex)
            {
                DebugPrintOut(L"\tWrong stream readed. Abort capturing\n");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ERROR)
            {
                DebugPrintOut(L"\tStream reading error. Abort capturing\n");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ALLEFFECTSREMOVED)
            {
                DebugPrintOut(L"\tStream decoding error. Abort capturing\n");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
            {
                DebugPrintOut(L"\tEnd of stream detected\n");
            }
            else
            {
                if (flags & MF_SOURCE_READERF_NEWSTREAM)
                {
                    DebugPrintOut(L"\tNew stream detected\n");
                }
                if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)
                {
                    DebugPrintOut(L"\tStream native media type changed\n");
                }
                if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
                {
                    DebugPrintOut(L"\tStream current media type changed\n");
                }
                return true;
            }
        }
    }
    return false;
}

IplImage* CvCaptureCAM_MSMF::retrieveFrame(int)
{
    unsigned int width = captureFormat.width;
    unsigned int height = captureFormat.height;
    unsigned int bytes = 3; //Suppose output format is BGR24
    if (!frame || (int)width != frame->width || (int)height != frame->height)
    {
        if (frame)
            cvReleaseImage(&frame);
        frame = cvCreateImage(cvSize(width, height), 8, bytes);
    }

    unsigned int size = bytes * width * height;
    DWORD bcnt;
    if (videoSample && SUCCEEDED(videoSample->GetBufferCount(&bcnt)) && bcnt > 0)
    {
        _ComPtr<IMFMediaBuffer> buf = NULL;
        if (SUCCEEDED(videoSample->GetBufferByIndex(0, &buf)))
        {
            DWORD maxsize, cursize;
            BYTE* ptr = NULL;
            if (SUCCEEDED(buf->Lock(&ptr, &maxsize, &cursize)))
            {
                if ((unsigned int)cursize == size)
                {
                    memcpy(frame->imageData, ptr, size);
                    buf->Unlock();
                    return frame;
                }
                buf->Unlock();
            }
        }
    }

    return NULL;
}

double CvCaptureCAM_MSMF::getFramerate(MediaType MT) const
{
    if (MT.MF_MT_SUBTYPE == MFVideoFormat_MP43) //Unable to estimate FPS for MP43
        return 0;
    return MT.MF_MT_FRAME_RATE_DENOMINATOR != 0 ? ((double)MT.MF_MT_FRAME_RATE_NUMERATOR) / ((double)MT.MF_MT_FRAME_RATE_DENOMINATOR) : 0;
}

double CvCaptureCAM_MSMF::getProperty( int property_id ) const
{
    // image format properties
    if (isOpened)
        switch (property_id)
        {
        case CV_CAP_PROP_FRAME_WIDTH:
            return captureFormat.width;
        case CV_CAP_PROP_FRAME_HEIGHT:
            return captureFormat.height;
        case CV_CAP_PROP_FOURCC:
            return captureFormat.MF_MT_SUBTYPE.Data1;
        case CV_CAP_PROP_FPS:
            return getFramerate(captureFormat);
        }

    return -1;
}
bool CvCaptureCAM_MSMF::setProperty( int property_id, double value )
{
    // image capture properties
    if (isOpened)
    {
        unsigned int width = captureFormat.width;
        unsigned int height = captureFormat.height;
        unsigned int fps = getProperty(CV_CAP_PROP_FPS);
        switch (property_id)
        {
        case CV_CAP_PROP_FRAME_WIDTH:
            width = cvRound(value);
            break;
        case CV_CAP_PROP_FRAME_HEIGHT:
            height = cvRound(value);
            break;
        case CV_CAP_PROP_FPS:
            fps = cvRound(value);
            break;
        }

        if (width > 0 && height > 0 && fps >= 0)
        {
            if (width != captureFormat.width || height != captureFormat.height || fps != getFramerate(captureFormat))
                return configureOutput(width, height, fps);
            else
                return true;
        }
    }
    return false;
}

class CvCaptureFile_MSMF : public CvCapture
{
public:
    CvCaptureFile_MSMF();
    virtual ~CvCaptureFile_MSMF();

    virtual bool open( const char* filename );
    virtual void close();

    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain() { return CV_CAP_MSMF; }
protected:
    Media_Foundation& MF;
    _ComPtr<IMFSourceReader> videoFileSource;
    DWORD dwStreamIndex;
    MediaType captureFormat;
    _ComPtr<IMFSample> videoSample;
    IplImage* frame;
    bool isOpened;

    HRESULT getSourceDuration(MFTIME *pDuration) const;
};

CvCaptureFile_MSMF::CvCaptureFile_MSMF():
    MF(Media_Foundation::getInstance()),
    videoFileSource(NULL),
    videoSample(NULL),
    frame(NULL),
    isOpened(false)
{
}

CvCaptureFile_MSMF::~CvCaptureFile_MSMF()
{
    close();
}

bool CvCaptureFile_MSMF::open(const char* filename)
{
    close();
    if (!filename)
        return false;

    // Set source reader parameters
    _ComPtr<IMFAttributes> srAttr;
    if (SUCCEEDED(MFCreateAttributes(&srAttr, 10)) &&
        SUCCEEDED(srAttr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, true)) &&
        SUCCEEDED(srAttr->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, false)) &&
        SUCCEEDED(srAttr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, false)) &&
        SUCCEEDED(srAttr->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, true))
        )
    {
        //ToDo: Enable D3D MF_SOURCE_READER_D3D_MANAGER attribute
        cv::AutoBuffer<wchar_t> unicodeFileName(strlen(filename) + 1);
        MultiByteToWideChar(CP_ACP, 0, filename, -1, unicodeFileName, (int)strlen(filename) + 1);
        if (SUCCEEDED(MFCreateSourceReaderFromURL(unicodeFileName, srAttr.Get(), &videoFileSource)))
        {
            HRESULT hr = S_OK;
            DWORD dwMediaTypeIndex = 0;
            dwStreamIndex = 0;
            while (SUCCEEDED(hr))
            {
                _ComPtr<IMFMediaType> pType;
                hr = videoFileSource->GetNativeMediaType(dwStreamIndex, dwMediaTypeIndex, &pType);
                if (hr == MF_E_NO_MORE_TYPES)
                {
                    hr = S_OK;
                    ++dwStreamIndex;
                    dwMediaTypeIndex = 0;
                }
                else if (SUCCEEDED(hr))
                {
                    MediaType MT(pType.Get());
                    if (MT.MF_MT_MAJOR_TYPE == MFMediaType_Video)
                    {
                        captureFormat = MT;
                        break;
                    }
                    ++dwMediaTypeIndex;
                }
            }

            _ComPtr<IMFMediaType>  mediaTypeOut;
            if (// Retrieved stream media type
                SUCCEEDED(hr) &&
                // Set the output media type.
                SUCCEEDED(MFCreateMediaType(&mediaTypeOut)) &&
                SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) &&
                SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB24)) &&
                SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)) &&
                SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_DEFAULT_STRIDE, 3 * captureFormat.width)) && //Assume BGR24 input
                SUCCEEDED(MFSetAttributeSize(mediaTypeOut.Get(), MF_MT_FRAME_SIZE, captureFormat.width, captureFormat.height)) &&
                SUCCEEDED(MFSetAttributeRatio(mediaTypeOut.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) &&
                SUCCEEDED(videoFileSource->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, false)) &&
                SUCCEEDED(videoFileSource->SetStreamSelection(dwStreamIndex, true)) &&
                SUCCEEDED(videoFileSource->SetCurrentMediaType(dwStreamIndex, NULL, mediaTypeOut.Get()))
                )
            {
                isOpened = true;
                return true;
            }
        }
    }

    return false;
}

void CvCaptureFile_MSMF::close()
{
    if (isOpened)
    {
        isOpened = false;
        if (videoSample)
            videoSample.Reset();
        if (videoFileSource)
            videoFileSource.Reset();
        if (frame)
            cvReleaseImage(&frame);
    }
}

bool CvCaptureFile_MSMF::setProperty(int property_id, double value)
{
    // image capture properties
    // FIXME: implement method in VideoInput back end
    (void) property_id;
    (void) value;
    return false;
}

double CvCaptureFile_MSMF::getProperty(int property_id) const
{
    // image format properties
    if (isOpened)
        switch( property_id )
        {
        case CV_CAP_PROP_FRAME_WIDTH:
            return captureFormat.width;
        case CV_CAP_PROP_FRAME_HEIGHT:
            return captureFormat.height;
        case CV_CAP_PROP_FRAME_COUNT:
            {
                if(captureFormat.MF_MT_SUBTYPE == MFVideoFormat_MP43) //Unable to estimate FPS for MP43
                    return 0;
                MFTIME duration;
                getSourceDuration(&duration);
                double fps = ((double)captureFormat.MF_MT_FRAME_RATE_NUMERATOR) /
                ((double)captureFormat.MF_MT_FRAME_RATE_DENOMINATOR);
                return (double)floor(((double)duration/1e7)*fps+0.5);
            }
        case CV_CAP_PROP_FOURCC:
            return captureFormat.MF_MT_SUBTYPE.Data1;
        case CV_CAP_PROP_FPS:
            if (captureFormat.MF_MT_SUBTYPE == MFVideoFormat_MP43) //Unable to estimate FPS for MP43
                return 0;
            return ((double)captureFormat.MF_MT_FRAME_RATE_NUMERATOR) /
                ((double)captureFormat.MF_MT_FRAME_RATE_DENOMINATOR);
        }

    return -1;
}

bool CvCaptureFile_MSMF::grabFrame()
{
    if (isOpened)
    {
        DWORD streamIndex, flags;
        LONGLONG llTimeStamp;
        if (videoSample)
            videoSample.Reset();
        if (SUCCEEDED(videoFileSource->ReadSample(
                                                     dwStreamIndex, // Stream index.
                                                     0,             // Flags.
                                                     &streamIndex,  // Receives the actual stream index.
                                                     &flags,        // Receives status flags.
                                                     &llTimeStamp,  // Receives the time stamp.
                                                     &videoSample   // Receives the sample or NULL.
                                                 )))
        {
            if (streamIndex != dwStreamIndex)
            {
                DebugPrintOut(L"\tWrong stream readed. Abort capturing\n");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ERROR)
            {
                DebugPrintOut(L"\tStream reading error. Abort capturing\n");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ALLEFFECTSREMOVED)
            {
                DebugPrintOut(L"\tStream decoding error. Abort capturing\n");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
            {
                DebugPrintOut(L"\tEnd of stream detected\n");
            }
            else
            {
                if (flags & MF_SOURCE_READERF_NEWSTREAM)
                {
                    DebugPrintOut(L"\tNew stream detected\n");
                }
                if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)
                {
                    DebugPrintOut(L"\tStream native media type changed\n");
                }
                if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
                {
                    DebugPrintOut(L"\tStream current media type changed\n");
                }
                if (flags & MF_SOURCE_READERF_STREAMTICK)
                {
                    DebugPrintOut(L"\tStream tick detected\n");
                }
                return true;
            }
        }
    }
    return false;
}

IplImage* CvCaptureFile_MSMF::retrieveFrame(int)
{
    unsigned int width = captureFormat.width;
    unsigned int height = captureFormat.height;
    unsigned int bytes = 3; //Suppose output format is BGR24
    if( !frame || (int)width != frame->width || (int)height != frame->height )
    {
        if (frame)
            cvReleaseImage( &frame );
        frame = cvCreateImage( cvSize(width,height), 8, bytes );
    }

    unsigned int size = bytes * width * height;
    DWORD bcnt;
    if (videoSample && SUCCEEDED(videoSample->GetBufferCount(&bcnt)) && bcnt > 0)
    {
        _ComPtr<IMFMediaBuffer> buf = NULL;
        if (SUCCEEDED(videoSample->GetBufferByIndex(0, &buf)))
        {
            DWORD maxsize, cursize;
            BYTE* ptr = NULL;
            if (SUCCEEDED(buf->Lock(&ptr, &maxsize, &cursize)))
            {
                if ((unsigned int)cursize == size)
                {
                    memcpy(frame->imageData, ptr, size);
                    buf->Unlock();
                    return frame;
                }
                buf->Unlock();
            }
        }
    }

    return NULL;
}

HRESULT CvCaptureFile_MSMF::getSourceDuration(MFTIME *pDuration) const
{
    *pDuration = 0;

    PROPVARIANT var;
    HRESULT hr = videoFileSource->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &var);
    if (SUCCEEDED(hr) && var.vt == VT_I8)
    {
        *pDuration = var.hVal.QuadPart;
        PropVariantClear(&var);
    }
    return hr;
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

CvCapture* cvCreateFileCapture_MSMF (const char* filename)
{
    CvCaptureFile_MSMF* capture = new CvCaptureFile_MSMF;
    try
    {
        if( capture->open(filename) )
            return capture;
        else
        {
            delete capture;
            return NULL;
        }
    }
    catch(...)
    {
        delete capture;
        throw;
    }
}

//
//
// Media Foundation-based Video Writer
//
//

class CvVideoWriter_MSMF : public CvVideoWriter
{
public:
    CvVideoWriter_MSMF();
    virtual ~CvVideoWriter_MSMF();
    virtual bool open(const char* filename, int fourcc,
                       double fps, CvSize frameSize, bool isColor);
    virtual void close();
    virtual bool writeFrame(const IplImage* img);

private:
    Media_Foundation& MF;
    UINT32 videoWidth;
    UINT32 videoHeight;
    double fps;
    UINT32 bitRate;
    UINT32 frameSize;
    GUID   encodingFormat;
    GUID   inputFormat;

    DWORD  streamIndex;
    _ComPtr<IMFSinkWriter> sinkWriter;

    bool   initiated;

    LONGLONG rtStart;
    UINT64 rtDuration;

    static const GUID FourCC2GUID(int fourcc);
};

CvVideoWriter_MSMF::CvVideoWriter_MSMF():
    MF(Media_Foundation::getInstance()),
    initiated(false)
{
}

CvVideoWriter_MSMF::~CvVideoWriter_MSMF()
{
    close();
}

const GUID CvVideoWriter_MSMF::FourCC2GUID(int fourcc)
{
    switch(fourcc)
    {
        case CV_FOURCC_MACRO('d', 'v', '2', '5'):
            return MFVideoFormat_DV25; break;
        case CV_FOURCC_MACRO('d', 'v', '5', '0'):
            return MFVideoFormat_DV50; break;
        case CV_FOURCC_MACRO('d', 'v', 'c', ' '):
            return MFVideoFormat_DVC; break;
        case CV_FOURCC_MACRO('d', 'v', 'h', '1'):
            return MFVideoFormat_DVH1; break;
        case CV_FOURCC_MACRO('d', 'v', 'h', 'd'):
            return MFVideoFormat_DVHD; break;
        case CV_FOURCC_MACRO('d', 'v', 's', 'd'):
            return MFVideoFormat_DVSD; break;
        case CV_FOURCC_MACRO('d', 'v', 's', 'l'):
                return MFVideoFormat_DVSL; break;
#if (WINVER >= 0x0602)
        case CV_FOURCC_MACRO('H', '2', '6', '3'):   // Available only for Win 8 target.
                return MFVideoFormat_H263; break;
#endif
        case CV_FOURCC_MACRO('H', '2', '6', '4'):
                return MFVideoFormat_H264; break;
        case CV_FOURCC_MACRO('M', '4', 'S', '2'):
                return MFVideoFormat_M4S2; break;
        case CV_FOURCC_MACRO('M', 'J', 'P', 'G'):
                return MFVideoFormat_MJPG; break;
        case CV_FOURCC_MACRO('M', 'P', '4', '3'):
                return MFVideoFormat_MP43; break;
        case CV_FOURCC_MACRO('M', 'P', '4', 'S'):
                return MFVideoFormat_MP4S; break;
        case CV_FOURCC_MACRO('M', 'P', '4', 'V'):
                return MFVideoFormat_MP4V; break;
        case CV_FOURCC_MACRO('M', 'P', 'G', '1'):
                return MFVideoFormat_MPG1; break;
        case CV_FOURCC_MACRO('M', 'S', 'S', '1'):
                return MFVideoFormat_MSS1; break;
        case CV_FOURCC_MACRO('M', 'S', 'S', '2'):
                return MFVideoFormat_MSS2; break;
        case CV_FOURCC_MACRO('W', 'M', 'V', '1'):
                return MFVideoFormat_WMV1; break;
        case CV_FOURCC_MACRO('W', 'M', 'V', '2'):
                return MFVideoFormat_WMV2; break;
        case CV_FOURCC_MACRO('W', 'M', 'V', '3'):
                return MFVideoFormat_WMV3; break;
        case CV_FOURCC_MACRO('W', 'V', 'C', '1'):
                return MFVideoFormat_WVC1; break;
        default:
            return MFVideoFormat_H264;
    }
}

bool CvVideoWriter_MSMF::open( const char* filename, int fourcc,
                       double _fps, CvSize _frameSize, bool /*isColor*/ )
{
    if (initiated)
        close();
    videoWidth = _frameSize.width;
    videoHeight = _frameSize.height;
    fps = _fps;
    bitRate = (UINT32)fps*videoWidth*videoHeight; // 1-bit per pixel
    encodingFormat = FourCC2GUID(fourcc);
    inputFormat = MFVideoFormat_RGB32;

    _ComPtr<IMFMediaType>  mediaTypeOut;
    _ComPtr<IMFMediaType>  mediaTypeIn;
    _ComPtr<IMFAttributes> spAttr;
    if (// Set the output media type.
        SUCCEEDED(MFCreateMediaType(&mediaTypeOut)) &&
        SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) &&
        SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_SUBTYPE, encodingFormat)) &&
        SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_AVG_BITRATE, bitRate)) &&
        SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)) &&
        SUCCEEDED(MFSetAttributeSize(mediaTypeOut.Get(), MF_MT_FRAME_SIZE, videoWidth, videoHeight)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeOut.Get(), MF_MT_FRAME_RATE, (UINT32)fps, 1)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeOut.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) &&
        // Set the input media type.
        SUCCEEDED(MFCreateMediaType(&mediaTypeIn)) &&
        SUCCEEDED(mediaTypeIn->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) &&
        SUCCEEDED(mediaTypeIn->SetGUID(MF_MT_SUBTYPE, inputFormat)) &&
        SUCCEEDED(mediaTypeIn->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)) &&
        SUCCEEDED(mediaTypeIn->SetUINT32(MF_MT_DEFAULT_STRIDE, 4 * videoWidth)) && //Assume BGR32 input
        SUCCEEDED(MFSetAttributeSize(mediaTypeIn.Get(), MF_MT_FRAME_SIZE, videoWidth, videoHeight)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeIn.Get(), MF_MT_FRAME_RATE, (UINT32)fps, 1)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeIn.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) &&
        // Set sink writer parameters
        SUCCEEDED(MFCreateAttributes(&spAttr, 10)) &&
        SUCCEEDED(spAttr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, true)) &&
        SUCCEEDED(spAttr->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, true))
        )
    {
        // Create the sink writer
        cv::AutoBuffer<wchar_t> unicodeFileName(strlen(filename) + 1);
        MultiByteToWideChar(CP_ACP, 0, filename, -1, unicodeFileName, (int)strlen(filename) + 1);
        HRESULT hr = MFCreateSinkWriterFromURL(unicodeFileName, NULL, spAttr.Get(), &sinkWriter);
        if (SUCCEEDED(hr))
        {
            // Configure the sink writer and tell it start to start accepting data
            if (SUCCEEDED(sinkWriter->AddStream(mediaTypeOut.Get(), &streamIndex)) &&
                SUCCEEDED(sinkWriter->SetInputMediaType(streamIndex, mediaTypeIn.Get(), NULL)) &&
                SUCCEEDED(sinkWriter->BeginWriting()))
            {
                initiated = true;
                rtStart = 0;
                MFFrameRateToAverageTimePerFrame((UINT32)fps, 1, &rtDuration);
                return true;
            }
        }
    }

    return false;
}

void CvVideoWriter_MSMF::close()
{
    if (initiated)
    {
        initiated = false;
        sinkWriter->Finalize();
        sinkWriter.Reset();
    }
}

bool CvVideoWriter_MSMF::writeFrame(const IplImage* img)
{
    if (!img ||
        (img->nChannels != 1 && img->nChannels != 3 && img->nChannels != 4) ||
        (UINT32)img->width != videoWidth || (UINT32)img->height != videoHeight)
        return false;

    const LONG cbWidth = 4 * videoWidth;
    const DWORD cbBuffer = cbWidth * videoHeight;
    _ComPtr<IMFSample> sample;
    _ComPtr<IMFMediaBuffer> buffer;
    BYTE *pData = NULL;
    // Prepare a media sample.
    if (SUCCEEDED(MFCreateSample(&sample)) &&
        // Set sample time stamp and duration.
        SUCCEEDED(sample->SetSampleTime(rtStart)) &&
        SUCCEEDED(sample->SetSampleDuration(rtDuration)) &&
        // Create a memory buffer.
        SUCCEEDED(MFCreateMemoryBuffer(cbBuffer, &buffer)) &&
        // Set the data length of the buffer.
        SUCCEEDED(buffer->SetCurrentLength(cbBuffer)) &&
        // Add the buffer to the sample.
        SUCCEEDED(sample->AddBuffer(buffer.Get())) &&
        // Lock the buffer.
        SUCCEEDED(buffer->Lock(&pData, NULL, NULL)))
    {
        // Copy the video frame to the buffer.
        cv::cvtColor(cv::cvarrToMat(img), cv::Mat(videoHeight, videoWidth, CV_8UC4, pData, cbWidth), img->nChannels > 1 ? cv::COLOR_BGR2BGRA : cv::COLOR_GRAY2BGRA);
        buffer->Unlock();
        // Send media sample to the Sink Writer.
        if (SUCCEEDED(sinkWriter->WriteSample(streamIndex, sample.Get())))
        {
            rtStart += rtDuration;
            return true;
        }
    }

    return false;
}

CvVideoWriter* cvCreateVideoWriter_MSMF( const char* filename, int fourcc,
                                        double fps, CvSize frameSize, int isColor )
{
    CvVideoWriter_MSMF* writer = new CvVideoWriter_MSMF;
    if( writer->open( filename, fourcc, fps, frameSize, isColor != 0 ))
        return writer;
    delete writer;
    return NULL;
}

#endif
