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
#ifdef HAVE_DXVA
#include <D3D11.h>
#include <D3d11_4.h>
#endif
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
#ifdef HAVE_DXVA
#pragma comment(lib, "d3d11")
#endif
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
    ComPtr()
    {
    }
    ComPtr(T* lp)
    {
        p = lp;
    }
    ComPtr(_In_ const ComPtr<T>& lp)
    {
        p = lp.p;
    }
    virtual ~ComPtr()
    {
    }

    T** operator&()
    {
        assert(p == NULL);
        return p.operator&();
    }
    T* operator->() const
    {
        assert(p != NULL);
        return p.operator->();
    }
    bool operator!() const
    {
        return p.operator==(NULL);
    }
    bool operator==(_In_opt_ T* pT) const
    {
        return p.operator==(pT);
    }
    bool operator!=(_In_opt_ T* pT) const
    {
        return p.operator!=(pT);
    }
    operator bool()
    {
        return p.operator!=(NULL);
    }

    T* const* GetAddressOf() const
    {
        return &p;
    }

    T** GetAddressOf()
    {
        return &p;
    }

    T** ReleaseAndGetAddressOf()
    {
        p.Release();
        return &p;
    }

    T* Get() const
    {
        return p;
    }

    // Attach to an existing interface (does not AddRef)
    void Attach(_In_opt_ T* p2)
    {
        p.Attach(p2);
    }
    // Detach the interface (does not Release)
    T* Detach()
    {
        return p.Detach();
    }
    _Check_return_ HRESULT CopyTo(_Deref_out_opt_ T** ppT)
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
    HRESULT As(_Inout_ U** lp) const
    {
        return p->QueryInterface(__uuidof(U), reinterpret_cast<void**>(lp));
    }
    // query for U interface
    template<typename U>
    HRESULT As(_Out_ ComPtr<U>* lp) const
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
    unsigned int MF_MT_FIXED_SIZE_SAMPLES;
    unsigned int MF_MT_VIDEO_NOMINAL_RANGE;
    UINT32 MF_MT_FRAME_RATE_NUMERATOR;
    UINT32 MF_MT_FRAME_RATE_DENOMINATOR;
    UINT32 MF_MT_PIXEL_ASPECT_RATIO_NUMERATOR;
    UINT32 MF_MT_PIXEL_ASPECT_RATIO_DENOMINATOR;
    unsigned int MF_MT_ALL_SAMPLES_INDEPENDENT;
    UINT32 MF_MT_FRAME_RATE_RANGE_MIN_NUMERATOR;
    UINT32 MF_MT_FRAME_RATE_RANGE_MIN_DENOMINATOR;
    unsigned int MF_MT_SAMPLE_SIZE;
    unsigned int MF_MT_VIDEO_PRIMARIES;
    unsigned int MF_MT_INTERLACE_MODE;
    UINT32 MF_MT_FRAME_RATE_RANGE_MAX_NUMERATOR;
    UINT32 MF_MT_FRAME_RATE_RANGE_MAX_DENOMINATOR;
    GUID MF_MT_MAJOR_TYPE;
    GUID MF_MT_SUBTYPE;
    LPCWSTR pMF_MT_MAJOR_TYPEName;
    LPCWSTR pMF_MT_SUBTYPEName;
    MediaType();
    MediaType(IMFMediaType *pType);
    ~MediaType();
    void Clear();
};

// Class for creating of Media Foundation context
class Media_Foundation
{
public:
    ~Media_Foundation(void) { /*CV_Assert(SUCCEEDED(MFShutdown()));*/ CoUninitialize(); }
    static Media_Foundation& getInstance()
    {
        static Media_Foundation instance;
        return instance;
    }
private:
    Media_Foundation(void) { CoInitialize(0); CV_Assert(SUCCEEDED(MFStartup(MF_VERSION))); }
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
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.MF_MT_FRAME_RATE_RANGE_MAX_NUMERATOR, &out.MF_MT_FRAME_RATE_RANGE_MAX_DENOMINATOR);
        else if (guid == MF_MT_FRAME_RATE_RANGE_MIN && var.vt == VT_UI8)
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.MF_MT_FRAME_RATE_RANGE_MIN_NUMERATOR, &out.MF_MT_FRAME_RATE_RANGE_MIN_DENOMINATOR);
        else if (guid == MF_MT_PIXEL_ASPECT_RATIO && var.vt == VT_UI8)
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &out.MF_MT_PIXEL_ASPECT_RATIO_NUMERATOR, &out.MF_MT_PIXEL_ASPECT_RATIO_DENOMINATOR);
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
    pMF_MT_MAJOR_TYPEName = NULL;
    pMF_MT_SUBTYPEName = NULL;
    Clear();
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
    MF_MT_PIXEL_ASPECT_RATIO_NUMERATOR = 0;
    MF_MT_PIXEL_ASPECT_RATIO_DENOMINATOR = 0;
    MF_MT_ALL_SAMPLES_INDEPENDENT = 0;
    MF_MT_FRAME_RATE_RANGE_MIN_NUMERATOR = 0;
    MF_MT_FRAME_RATE_RANGE_MIN_DENOMINATOR = 0;
    MF_MT_SAMPLE_SIZE = 0;
    MF_MT_VIDEO_PRIMARIES = 0;
    MF_MT_INTERLACE_MODE = 0;
    MF_MT_FRAME_RATE_RANGE_MAX_NUMERATOR = 0;
    MF_MT_FRAME_RATE_RANGE_MAX_DENOMINATOR = 0;
    memset(&MF_MT_MAJOR_TYPE, 0, sizeof(GUID));
    memset(&MF_MT_AM_FORMAT_TYPE, 0, sizeof(GUID));
    memset(&MF_MT_SUBTYPE, 0, sizeof(GUID));
}

}

/******* Capturing video from camera or file via Microsoft Media Foundation **********/
class CvCapture_MSMF : public CvCapture
{
public:
    typedef enum {
        MODE_SW = 0,
        MODE_HW = 1
    } MSMFCapture_Mode;
    CvCapture_MSMF();
    virtual ~CvCapture_MSMF();
    virtual bool open(int index);
    virtual bool open(const char* filename);
    virtual void close();
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual IplImage* retrieveFrame(int) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE { return CV_CAP_MSMF; } // Return the type of the capture object: CV_CAP_VFW, etc...
protected:
    double getFramerate(MediaType MT) const;
    bool configureOutput(UINT32 width, UINT32 height, double prefFramerate, UINT32 aspectRatioN, UINT32 aspectRatioD, int outFormat, bool convertToFormat);
    bool setTime(double time, bool rough);
    bool configureHW(bool enable);

    Media_Foundation& MF;
    cv::String filename;
    int camid;
    MSMFCapture_Mode captureMode;
#ifdef HAVE_DXVA
    _ComPtr<ID3D11Device> D3DDev;
    _ComPtr<IMFDXGIDeviceManager> D3DMgr;
#endif
    _ComPtr<IMFSourceReader> videoFileSource;
    DWORD dwStreamIndex;
    MediaType nativeFormat;
    MediaType captureFormat;
    int outputFormat;
    bool convertFormat;
    UINT32 aspectN, aspectD;
    MFTIME duration;
    LONGLONG frameStep;
    _ComPtr<IMFSample> videoSample;
    LONGLONG sampleTime;
    IplImage* frame;
    bool isOpened;
};

CvCapture_MSMF::CvCapture_MSMF():
    MF(Media_Foundation::getInstance()),
    filename(""),
    camid(-1),
    captureMode(MODE_SW),
#ifdef HAVE_DXVA
    D3DDev(NULL),
    D3DMgr(NULL),
#endif
    videoFileSource(NULL),
    videoSample(NULL),
    outputFormat(CV_CAP_MODE_BGR),
    convertFormat(true),
    aspectN(1),
    aspectD(1),
    sampleTime(0),
    frame(NULL),
    isOpened(false)
{
}

CvCapture_MSMF::~CvCapture_MSMF()
{
    close();
}

void CvCapture_MSMF::close()
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
        camid = -1;
        filename = "";
    }
}

bool CvCapture_MSMF::configureHW(bool enable)
{
#ifdef HAVE_DXVA
    if ((enable && D3DMgr && D3DDev) || (!enable && !D3DMgr && !D3DDev))
        return true;

    bool reopen = isOpened;
    int prevcam = camid;
    cv::String prevfile = filename;
    close();
    if (enable)
    {
        D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_3,  D3D_FEATURE_LEVEL_9_2, D3D_FEATURE_LEVEL_9_1 };
        if (SUCCEEDED(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            levels, sizeof(levels) / sizeof(*levels), D3D11_SDK_VERSION, D3DDev.GetAddressOf(), NULL, NULL)))
        {
            // NOTE: Getting ready for multi-threaded operation
            _ComPtr<ID3D11Multithread> D3DDevMT;
            UINT mgrRToken;
            if (SUCCEEDED(D3DDev->QueryInterface(IID_PPV_ARGS(&D3DDevMT))))
            {
                D3DDevMT->SetMultithreadProtected(TRUE);
                D3DDevMT.Reset();
                if (SUCCEEDED(MFCreateDXGIDeviceManager(&mgrRToken, D3DMgr.GetAddressOf())))
                {
                    if (SUCCEEDED(D3DMgr->ResetDevice(D3DDev.Get(), mgrRToken)))
                    {
                        captureMode = MODE_HW;
                        return reopen ? camid >= 0 ? open(prevcam) : open(prevfile.c_str()) : true;
                    }
                    D3DMgr.Reset();
                }
            }
            D3DDev.Reset();
        }
        return false;
    }
    else
    {
        if (D3DMgr)
            D3DMgr.Reset();
        if (D3DDev)
            D3DDev.Reset();
        captureMode = MODE_SW;
        return reopen ? camid >= 0 ? open(prevcam) : open(prevfile.c_str()) : true;
    }
#else
    return !enable;
#endif
}

bool CvCapture_MSMF::configureOutput(UINT32 width, UINT32 height, double prefFramerate, UINT32 aspectRatioN, UINT32 aspectRatioD, int outFormat, bool convertToFormat)
{
    if (width != 0 && height != 0 &&
        width == captureFormat.width && height == captureFormat.height && prefFramerate == getFramerate(nativeFormat) &&
        aspectRatioN == aspectN && aspectRatioD == aspectD && outFormat == outputFormat && convertToFormat == convertFormat)
        return true;

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
        GUID outSubtype = GUID_NULL;
        UINT32 outStride = 0;
        UINT32 outSize = 0;
        if(convertToFormat)
            switch (outFormat)
            {
            case CV_CAP_MODE_BGR:
            case CV_CAP_MODE_RGB:
                outSubtype = captureMode == MODE_HW ? MFVideoFormat_RGB32 : MFVideoFormat_RGB24; // HW accelerated mode support only RGB32
                outStride = (captureMode == MODE_HW ? 4 : 3) * tryMT.width;
                outSize = outStride * tryMT.height;
                break;
            case CV_CAP_MODE_GRAY:
                outSubtype = MFVideoFormat_NV12;
                outStride = tryMT.width;
                outSize = outStride * tryMT.height * 3 / 2;
                break;
            case CV_CAP_MODE_YUYV:
                outSubtype = MFVideoFormat_YUY2;
                outStride = 2 * tryMT.width;
                outSize = outStride * tryMT.height;
                break;
            default:
                return false;
            }
        _ComPtr<IMFMediaType>  mediaTypeOut;
        if (// Set the output media type.
            SUCCEEDED(MFCreateMediaType(&mediaTypeOut)) &&
            SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) &&
            SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_SUBTYPE, convertToFormat ? outSubtype : tryMT.MF_MT_SUBTYPE)) &&
            SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_INTERLACE_MODE, convertToFormat ? MFVideoInterlace_Progressive : tryMT.MF_MT_INTERLACE_MODE)) &&
            SUCCEEDED(MFSetAttributeRatio(mediaTypeOut.Get(), MF_MT_PIXEL_ASPECT_RATIO, aspectRatioN, aspectRatioD)) &&
            SUCCEEDED(MFSetAttributeSize(mediaTypeOut.Get(), MF_MT_FRAME_SIZE, tryMT.width, tryMT.height)) &&
            SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_FIXED_SIZE_SAMPLES, convertToFormat ? 1 : tryMT.MF_MT_FIXED_SIZE_SAMPLES)) &&
            SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_SAMPLE_SIZE, convertToFormat ? outSize : tryMT.MF_MT_SAMPLE_SIZE)) &&
            SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_DEFAULT_STRIDE, convertToFormat ? outStride : tryMT.MF_MT_DEFAULT_STRIDE)))//Assume BGR24 input
        {
            if (SUCCEEDED(videoFileSource->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, false)) &&
                SUCCEEDED(videoFileSource->SetStreamSelection(tryStream, true)) &&
                SUCCEEDED(videoFileSource->SetCurrentMediaType(tryStream, NULL, mediaTypeOut.Get()))
                )
            {
                dwStreamIndex = tryStream;
                nativeFormat = tryMT;
                aspectN = aspectRatioN;
                aspectD = aspectRatioD;
                outputFormat = outFormat;
                convertFormat = convertToFormat;
                captureFormat = MediaType(mediaTypeOut.Get());
                return true;
            }
            close();
        }
    }
    return false;
}

// Initialize camera input
bool CvCapture_MSMF::open(int _index)
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
                            SUCCEEDED(srAttr->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, true)))
                        {
#ifdef HAVE_DXVA
                            if (D3DMgr)
                                srAttr->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, D3DMgr.Get());
#endif
                            if (SUCCEEDED(MFCreateSourceReaderFromMediaSource(mSrc.Get(), srAttr.Get(), &videoFileSource)))
                            {
                                isOpened = true;
                                duration = 0;
                                if (configureOutput(0, 0, 0, aspectN, aspectD, outputFormat, convertFormat))
                                {
                                    double fps = getFramerate(nativeFormat);
                                    frameStep = (LONGLONG)(fps > 0 ? 1e7 / fps : 0);
                                    camid = _index;
                                }
                            }
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

bool CvCapture_MSMF::open(const char* _filename)
{
    close();
    if (!_filename)
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
#ifdef HAVE_DXVA
        if(D3DMgr)
            srAttr->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, D3DMgr.Get());
#endif
        cv::AutoBuffer<wchar_t> unicodeFileName(strlen(_filename) + 1);
        MultiByteToWideChar(CP_ACP, 0, _filename, -1, unicodeFileName, (int)strlen(_filename) + 1);
        if (SUCCEEDED(MFCreateSourceReaderFromURL(unicodeFileName, srAttr.Get(), &videoFileSource)))
        {
            isOpened = true;
            sampleTime = 0;
            if (configureOutput(0, 0, 0, aspectN, aspectD, outputFormat, convertFormat))
            {
                double fps = getFramerate(nativeFormat);
                frameStep = (LONGLONG)(fps > 0 ? 1e7 / fps : 0);
                filename = _filename;
                PROPVARIANT var;
                HRESULT hr;
                if (SUCCEEDED(hr = videoFileSource->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &var)) &&
                    var.vt == VT_UI8)
                {
                    duration = var.uhVal.QuadPart;
                    PropVariantClear(&var);
                }
                else
                    duration = 0;
            }
        }
    }

    return isOpened;
}

bool CvCapture_MSMF::grabFrame()
{
    if (isOpened)
    {
        DWORD streamIndex, flags;
        if (videoSample)
            videoSample.Reset();
        HRESULT hr;
        while(SUCCEEDED(hr = videoFileSource->ReadSample(
                                                            dwStreamIndex, // Stream index.
                                                            0,             // Flags.
                                                            &streamIndex,  // Receives the actual stream index.
                                                            &flags,        // Receives status flags.
                                                            &sampleTime,  // Receives the time stamp.
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
                sampleTime += frameStep;
                DebugPrintOut(L"\tEnd of stream detected\n");
            }
            else
            {
                sampleTime += frameStep;
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

IplImage* CvCapture_MSMF::retrieveFrame(int)
{
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
                if (convertFormat)
                {
                    if ((unsigned int)cursize == captureFormat.MF_MT_SAMPLE_SIZE)
                    {
                        if (!frame || (int)captureFormat.width != frame->width || (int)captureFormat.height != frame->height)
                        {
                            cvReleaseImage(&frame);
                            unsigned int bytes = outputFormat == CV_CAP_MODE_GRAY || !convertFormat ? 1 : outputFormat == CV_CAP_MODE_YUYV ? 2 : 3;
                            frame = cvCreateImage(cvSize(captureFormat.width, captureFormat.height), 8, bytes);
                        }
                        switch (outputFormat)
                        {
                        case CV_CAP_MODE_YUYV:
                            memcpy(frame->imageData, ptr, cursize);
                            break;
                        case CV_CAP_MODE_BGR:
                            if (captureMode == MODE_HW)
                                cv::cvtColor(cv::Mat(captureFormat.height, captureFormat.width, CV_8UC4, ptr), cv::cvarrToMat(frame), cv::COLOR_BGRA2BGR);
                            else
                                memcpy(frame->imageData, ptr, cursize);
                            break;
                        case CV_CAP_MODE_RGB:
                            if (captureMode == MODE_HW)
                                cv::cvtColor(cv::Mat(captureFormat.height, captureFormat.width, CV_8UC4, ptr), cv::cvarrToMat(frame), cv::COLOR_BGRA2BGR);
                            else
                                cv::cvtColor(cv::Mat(captureFormat.height, captureFormat.width, CV_8UC3, ptr), cv::cvarrToMat(frame), cv::COLOR_BGR2RGB);
                            break;
                        case CV_CAP_MODE_GRAY:
                            memcpy(frame->imageData, ptr, captureFormat.height*captureFormat.width);
                            break;
                        default:
                            cvReleaseImage(&frame);
                            break;
                        }
                    }
                    else
                        cvReleaseImage(&frame);
                }
                else
                {
                    if (!frame || frame->width != (int)cursize || frame->height != 1)
                    {
                        cvReleaseImage(&frame);
                        frame = cvCreateImage(cvSize(cursize, 1), 8, 1);
                    }
                    memcpy(frame->imageData, ptr, cursize);
                }
                buf->Unlock();
                return frame;
            }
        }
    }

    return NULL;
}

double CvCapture_MSMF::getFramerate(MediaType MT) const
{
    if (MT.MF_MT_SUBTYPE == MFVideoFormat_MP43) //Unable to estimate FPS for MP43
        return 0;
    return MT.MF_MT_FRAME_RATE_DENOMINATOR != 0 ? ((double)MT.MF_MT_FRAME_RATE_NUMERATOR) / ((double)MT.MF_MT_FRAME_RATE_DENOMINATOR) : 0;
}

bool CvCapture_MSMF::setTime(double time, bool rough)
{
    PROPVARIANT var;
    if (SUCCEEDED(videoFileSource->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_SOURCE_READER_MEDIASOURCE_CHARACTERISTICS, &var)) &&
        var.vt == VT_UI4 && var.ulVal & MFMEDIASOURCE_CAN_SEEK)
    {
        if (videoSample)
            videoSample.Reset();
        bool useGrabbing = time > 0 && !rough && !(var.ulVal & MFMEDIASOURCE_HAS_SLOW_SEEK);
        PropVariantClear(&var);
        sampleTime = (useGrabbing && time >= frameStep) ? (LONGLONG)floor(time + 0.5) - frameStep : (LONGLONG)floor(time + 0.5);
        var.vt = VT_I8;
        var.hVal.QuadPart = sampleTime;
        bool resOK = SUCCEEDED(videoFileSource->SetCurrentPosition(GUID_NULL, var));
        PropVariantClear(&var);
        if (resOK && useGrabbing)
        {
            LONGLONG timeborder = (LONGLONG)floor(time + 0.5) - frameStep / 2;
            do { resOK = grabFrame(); videoSample.Reset(); } while (resOK && sampleTime < timeborder);
        }
        return resOK;
    }
    return false;
}

double CvCapture_MSMF::getProperty( int property_id ) const
{
    IAMVideoProcAmp *pProcAmp = NULL;
    IAMCameraControl *pProcControl = NULL;
    // image format properties
    if (property_id == CV_CAP_PROP_FORMAT)
        return outputFormat;
    else if (property_id == CV_CAP_PROP_MODE)
        return captureMode;
    else if (property_id == CV_CAP_PROP_CONVERT_RGB)
        return convertFormat ? 1 : 0;
    else if (property_id == CV_CAP_PROP_SAR_NUM)
        return aspectN;
    else if (property_id == CV_CAP_PROP_SAR_DEN)
        return aspectD;
    else if (isOpened)
        switch (property_id)
        {
        case CV_CAP_PROP_FRAME_WIDTH:
            return captureFormat.width;
        case CV_CAP_PROP_FRAME_HEIGHT:
            return captureFormat.height;
        case CV_CAP_PROP_FOURCC:
            return nativeFormat.MF_MT_SUBTYPE.Data1;
        case CV_CAP_PROP_FPS:
            return getFramerate(nativeFormat);
        case CV_CAP_PROP_FRAME_COUNT:
            if (duration != 0)
                return floor(((double)duration / 1e7)*getFramerate(nativeFormat) + 0.5);
            else
                break;
        case CV_CAP_PROP_POS_FRAMES:
            return floor(((double)sampleTime / 1e7)*getFramerate(nativeFormat) + 0.5);
        case CV_CAP_PROP_POS_MSEC:
            return (double)sampleTime / 1e4;
        case CV_CAP_PROP_POS_AVI_RATIO:
            if (duration != 0)
                return (double)sampleTime / duration;
            else
                break;
        case CV_CAP_PROP_BRIGHTNESS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_Brightness, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if(FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_Brightness, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_CONTRAST:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_Contrast, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_Contrast, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_SATURATION:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_Saturation, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_Saturation, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_HUE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_Hue, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_Hue, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_GAIN:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_Gain, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_Gain, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_SHARPNESS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_Sharpness, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_Sharpness, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_GAMMA:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_Gamma, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_Gamma, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_BACKLIGHT:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_BacklightCompensation, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_BacklightCompensation, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_MONOCHROME:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_ColorEnable, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_ColorEnable, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal == 0 ? 1 : 0;
            }
            break;
        case CV_CAP_PROP_TEMPERATURE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcAmp->Get(VideoProcAmp_WhiteBalance, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcAmp->GetRange(VideoProcAmp_WhiteBalance, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcAmp->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
        case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
        case CV_CAP_PROP_WHITE_BALANCE_RED_V:
            break;
        case CV_CAP_PROP_PAN:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Pan, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Pan, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_TILT:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Tilt, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Tilt, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_ROLL:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Roll, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Roll, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_IRIS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Iris, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Iris, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_EXPOSURE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Exposure, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Exposure, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
        case CV_CAP_PROP_AUTO_EXPOSURE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Exposure, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Exposure, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramFlag == VideoProcAmp_Flags_Auto;
            }
            break;
        case CV_CAP_PROP_ZOOM:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Zoom, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Zoom, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
            break;
        case CV_CAP_PROP_FOCUS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Focus, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Focus, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramVal;
            }
        case CV_CAP_PROP_AUTOFOCUS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal, paramFlag;
                HRESULT hr = pProcControl->Get(CameraControl_Focus, &paramVal, &paramFlag);
                long minVal, maxVal, stepVal;
                if (FAILED(hr))
                    hr = pProcControl->GetRange(CameraControl_Focus, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag);//Unable to get the property, trying to return default value
                pProcControl->Release();
                if (SUCCEEDED(hr))
                    return paramFlag == VideoProcAmp_Flags_Auto;
            }
            break;

        case CV_CAP_PROP_RECTIFICATION:
        case CV_CAP_PROP_TRIGGER:
        case CV_CAP_PROP_TRIGGER_DELAY:
        case CV_CAP_PROP_GUID:
        case CV_CAP_PROP_ISO_SPEED:
        case CV_CAP_PROP_SETTINGS:
        case CV_CAP_PROP_BUFFERSIZE:
        default:
            break;
        }

    return -1;
}

bool CvCapture_MSMF::setProperty( int property_id, double value )
{
    IAMVideoProcAmp *pProcAmp = NULL;
    IAMCameraControl *pProcControl = NULL;
    // image capture properties
    if (property_id == CV_CAP_PROP_FORMAT)
    {
        if (isOpened)
            return configureOutput(captureFormat.width, captureFormat.height, getFramerate(nativeFormat), aspectN, aspectD, (int)cvRound(value), convertFormat);
        else
            outputFormat = (int)cvRound(value);
        return true;
    }
    else if (property_id == CV_CAP_PROP_MODE)
    {
        switch ((MSMFCapture_Mode)((int)value))
        {
        case MODE_SW:
            return configureHW(false);
        case MODE_HW:
            return configureHW(true);
        default:
            return false;
        }
    }
    else if (property_id == CV_CAP_PROP_CONVERT_RGB)
    {
        if (isOpened)
            return configureOutput(captureFormat.width, captureFormat.height, getFramerate(nativeFormat), aspectN, aspectD, outputFormat, value != 0);
        else
            convertFormat = value != 0;
        return true;
    }
    else if (property_id == CV_CAP_PROP_SAR_NUM && value > 0)
    {
        if (isOpened)
            return configureOutput(captureFormat.width, captureFormat.height, getFramerate(nativeFormat), (UINT32)cvRound(value), aspectD, outputFormat, convertFormat);
        else
            aspectN = (UINT32)cvRound(value);
        return true;
    }
    else if (property_id == CV_CAP_PROP_SAR_DEN && value > 0)
    {
        if (isOpened)
            return configureOutput(captureFormat.width, captureFormat.height, getFramerate(nativeFormat), aspectN, (UINT32)cvRound(value), outputFormat, convertFormat);
        else
            aspectD = (UINT32)cvRound(value);
        return true;
    }
    else if (isOpened)
        switch (property_id)
        {
        case CV_CAP_PROP_FRAME_WIDTH:
            if (value > 0)
                return configureOutput((UINT32)cvRound(value), captureFormat.height, getFramerate(nativeFormat), aspectN, aspectD, outputFormat, convertFormat);
            break;
        case CV_CAP_PROP_FRAME_HEIGHT:
            if (value > 0)
                return configureOutput(captureFormat.width, (UINT32)cvRound(value), getFramerate(nativeFormat), aspectN, aspectD, outputFormat, convertFormat);
            break;
        case CV_CAP_PROP_FPS:
            if (value >= 0)
                return configureOutput(captureFormat.width, captureFormat.height, value, aspectN, aspectD, outputFormat, convertFormat);
            break;
        case CV_CAP_PROP_FOURCC:
            break;
        case CV_CAP_PROP_FRAME_COUNT:
            break;
        case CV_CAP_PROP_POS_AVI_RATIO:
            if (duration != 0)
                return setTime(duration * value, true);
            break;
        case CV_CAP_PROP_POS_FRAMES:
            if (getFramerate(nativeFormat) != 0)
                return setTime(value  * 1e7 / getFramerate(nativeFormat), false);
            break;
        case CV_CAP_PROP_POS_MSEC:
                return setTime(value  * 1e4, false);
        case CV_CAP_PROP_BRIGHTNESS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_Brightness, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_CONTRAST:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_Contrast, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_SATURATION:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_Saturation, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_HUE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_Hue, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_GAIN:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_Gain, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_SHARPNESS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_Sharpness, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_GAMMA:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_Gamma, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_BACKLIGHT:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_BacklightCompensation, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_MONOCHROME:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = value != 0 ? 0 : 1;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_ColorEnable, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_TEMPERATURE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcAmp))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcAmp->Set(VideoProcAmp_WhiteBalance, paramVal, VideoProcAmp_Flags_Manual);
                pProcAmp->Release();
                return SUCCEEDED(hr);
            }
        case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
        case CV_CAP_PROP_WHITE_BALANCE_RED_V:
            break;
        case CV_CAP_PROP_PAN:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcControl->Set(CameraControl_Pan, paramVal, VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_TILT:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcControl->Set(CameraControl_Tilt, paramVal, VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_ROLL:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcControl->Set(CameraControl_Roll, paramVal, VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_IRIS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcControl->Set(CameraControl_Iris, paramVal, VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_EXPOSURE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcControl->Set(CameraControl_Exposure, paramVal, VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
        case CV_CAP_PROP_AUTO_EXPOSURE:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = 0;
                HRESULT hr = pProcControl->Set(CameraControl_Exposure, paramVal, value != 0 ? VideoProcAmp_Flags_Auto : VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_ZOOM:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcControl->Set(CameraControl_Zoom, paramVal, VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
            break;
        case CV_CAP_PROP_FOCUS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = (long)value;
                HRESULT hr = pProcControl->Set(CameraControl_Focus, paramVal, VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
        case CV_CAP_PROP_AUTOFOCUS:
            if (SUCCEEDED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&pProcControl))))
            {
                long paramVal = 0;
                HRESULT hr = pProcControl->Set(CameraControl_Focus, paramVal, value != 0 ? VideoProcAmp_Flags_Auto : VideoProcAmp_Flags_Manual);
                pProcControl->Release();
                return SUCCEEDED(hr);
            }
            break;

        case CV_CAP_PROP_RECTIFICATION:
        case CV_CAP_PROP_TRIGGER:
        case CV_CAP_PROP_TRIGGER_DELAY:
        case CV_CAP_PROP_GUID:
        case CV_CAP_PROP_ISO_SPEED:
        case CV_CAP_PROP_SETTINGS:
        case CV_CAP_PROP_BUFFERSIZE:
        default:
            break;
        }

    return false;
}

CvCapture* cvCreateCameraCapture_MSMF( int index )
{
    CvCapture_MSMF* capture = new CvCapture_MSMF;
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
    CvCapture_MSMF* capture = new CvCapture_MSMF;
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
