// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef MFXHELPER_H
#define MFXHELPER_H

#include "opencv2/core.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

CV_SUPPRESS_DEPRECATED_START
#  if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable:4201)  // nonstandard extension used: nameless struct/union
#  endif
#ifdef HAVE_ONEVPL
#  include <vpl/mfxcommon.h>
#  include <vpl/mfxstructures.h>
#  include <vpl/mfxvideo++.h>
#  include <vpl/mfxvp8.h>
#  include <vpl/mfxjpeg.h>
#  include <vpl/mfxdispatcher.h>
#else
#  include <mfxcommon.h>
#  include <mfxstructures.h>
#  include <mfxvideo++.h>
#  include <mfxvp8.h>
#  include <mfxjpeg.h>
#  ifdef HAVE_MFX_PLUGIN
#    include <mfxplugin++.h>
#  endif
#endif
#  if defined(_MSC_VER)
#    pragma warning(pop)
#  endif
CV_SUPPRESS_DEPRECATED_END

//                 //
//  Debug helpers  //
//                 //

#if 0
#   define DBG(i) i
#else
#   define DBG(i)
#endif

#if 1
#   define MSG(i) i
#else
#   define MSG(i)
#endif

template <typename T>
struct HexWrap {
    HexWrap(T val_) : val(val_) {}
    T val;
};

template <typename T>
inline std::ostream & operator<<(std::ostream &out, const HexWrap<T> &wrap) {
    std::ios_base::fmtflags flags = out.flags(std::ios::hex | std::ios::showbase);
    out << wrap.val;
    out.flags(flags);
    return out;
}

template <typename T>
inline ::HexWrap<T> asHex(const T & val) {
    return ::HexWrap<T>(val);
}

struct FourCC
{
    FourCC(uint val) : val32(val) {}
    FourCC(char a, char b, char c, char d) { val8[0] = a; val8[1] = b; val8[2] = c; val8[3] = d; }
    union {
        uint val32;
        int vali32;
        uchar val8[4];
    };
};

inline std::ostream & operator<<(std::ostream &out, FourCC cc) {
    for (size_t i = 0; i < 4; out << cc.val8[i++]) {}
    out << " (" << asHex(cc.val32) << ")";
    return out;
}

inline std::string mfxStatusToString(mfxStatus s) {
    switch (s)
    {
    case MFX_ERR_NONE: return "MFX_ERR_NONE";
    case MFX_ERR_UNKNOWN: return "MFX_ERR_UNKNOWN";
    case MFX_ERR_NULL_PTR: return "MFX_ERR_NULL_PTR";
    case MFX_ERR_UNSUPPORTED: return "MFX_ERR_UNSUPPORTED";
    case MFX_ERR_MEMORY_ALLOC: return "MFX_ERR_MEMORY_ALLOC";
    case MFX_ERR_NOT_ENOUGH_BUFFER: return "MFX_ERR_NOT_ENOUGH_BUFFER";
    case MFX_ERR_INVALID_HANDLE: return "MFX_ERR_INVALID_HANDLE";
    case MFX_ERR_LOCK_MEMORY: return "MFX_ERR_LOCK_MEMORY";
    case MFX_ERR_NOT_INITIALIZED: return "MFX_ERR_NOT_INITIALIZED";
    case MFX_ERR_NOT_FOUND: return "MFX_ERR_NOT_FOUND";
    case MFX_ERR_MORE_DATA: return "MFX_ERR_MORE_DATA";
    case MFX_ERR_MORE_SURFACE: return "MFX_ERR_MORE_SURFACE";
    case MFX_ERR_ABORTED: return "MFX_ERR_ABORTED";
    case MFX_ERR_DEVICE_LOST: return "MFX_ERR_DEVICE_LOST";
    case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM: return "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM";
    case MFX_ERR_INVALID_VIDEO_PARAM: return "MFX_ERR_INVALID_VIDEO_PARAM";
    case MFX_ERR_UNDEFINED_BEHAVIOR: return "MFX_ERR_UNDEFINED_BEHAVIOR";
    case MFX_ERR_DEVICE_FAILED: return "MFX_ERR_DEVICE_FAILED";
    case MFX_ERR_MORE_BITSTREAM: return "MFX_ERR_MORE_BITSTREAM";
    case MFX_ERR_GPU_HANG: return "MFX_ERR_GPU_HANG";
    case MFX_ERR_REALLOC_SURFACE: return "MFX_ERR_REALLOC_SURFACE";
    case MFX_WRN_IN_EXECUTION: return "MFX_WRN_IN_EXECUTION";
    case MFX_WRN_DEVICE_BUSY: return "MFX_WRN_DEVICE_BUSY";
    case MFX_WRN_VIDEO_PARAM_CHANGED: return "MFX_WRN_VIDEO_PARAM_CHANGED";
    case MFX_WRN_PARTIAL_ACCELERATION: return "MFX_WRN_PARTIAL_ACCELERATION";
    case MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: return "MFX_WRN_INCOMPATIBLE_VIDEO_PARAM";
    case MFX_WRN_VALUE_NOT_CHANGED: return "MFX_WRN_VALUE_NOT_CHANGED";
    case MFX_WRN_OUT_OF_RANGE: return "MFX_WRN_OUT_OF_RANGE";
    case MFX_WRN_FILTER_SKIPPED: return "MFX_WRN_FILTER_SKIPPED";
    default: return "<Invalid or unknown mfxStatus>";
    }
}

inline std::ostream & operator<<(std::ostream &out, mfxStatus s) {
    out << mfxStatusToString(s) << " (" << (int)s << ")"; return out;
}

inline std::ostream & operator<<(std::ostream &out, const mfxInfoMFX &info) {
    out << "InfoMFX:" << std::endl
        << "| Codec: " << FourCC(info.CodecId) << " / " << info.CodecProfile << " / " << info.CodecLevel << std::endl
        << "| DecodedOrder: " << info.DecodedOrder << std::endl
        << "| TimeStampCalc: " << info.TimeStampCalc << std::endl
           ;
    return out;
}

inline std::ostream & operator<<(std::ostream & out, const mfxFrameInfo & info) {
    out << "FrameInfo: " << std::endl
        << "| FourCC: " << FourCC(info.FourCC) << std::endl
        << "| Size: " << info.Width << "x" << info.Height << std::endl
        << "| ROI: " << "(" << info.CropX << ";" << info.CropY << ") " << info.CropW << "x" << info.CropH << std::endl
        << "| BitDepth(L/C): " << info.BitDepthLuma << " / " << info.BitDepthChroma << std::endl
        << "| Shift: " << info.Shift << std::endl
        << "| TemporalID: " << info.FrameId.TemporalId << std::endl
        << "| FrameRate: " << info.FrameRateExtN << "/" << info.FrameRateExtD << std::endl
        << "| AspectRatio: " << info.AspectRatioW << "x" << info.AspectRatioH << std::endl
        << "| PicStruct: " << info.PicStruct << std::endl
        << "| ChromaFormat: " << info.ChromaFormat << std::endl
           ;
    return out;
}

inline std::ostream & operator<<(std::ostream &out, const mfxFrameData &data) {
    out << "FrameData:" << std::endl
        << "| NumExtParam: " << data.NumExtParam << std::endl
        << "| MemType: " << data.MemType << std::endl
        << "| PitchHigh: " << data.PitchHigh << std::endl
        << "| TimeStamp: " << data.TimeStamp << std::endl
        << "| FrameOrder: " << data.FrameOrder << std::endl
        << "| Locked: " << data.Locked << std::endl
        << "| Pitch: " << data.PitchHigh << ", " << data.PitchLow << std::endl
        << "| Y: " << (void*)data.Y << std::endl
        << "| U: " << (void*)data.U << std::endl
        << "| V: " << (void*)data.V << std::endl
           ;
    return out;
}

//==================================================================================================

template <typename T>
inline void cleanup(T * &ptr)
{
    if (ptr)
    {
        delete ptr;
        ptr = 0;
    }
}

//==================================================================================================

#ifdef HAVE_ONEVPL
mfxLoader getVPLLoaderInstance();
#endif

//==================================================================================================

class MFXVideoSession_WRAP : public MFXVideoSession
{
#ifdef HAVE_ONEVPL
public:
    mfxStatus CreateSession()
    {
        return MFXCreateSession(getVPLLoaderInstance(), 0, &m_session);
    }
#endif
};

//==================================================================================================

class Plugin
{
public:
    static Plugin * loadEncoderPlugin(MFXVideoSession_WRAP &session, mfxU32 codecId)
    {
#ifdef HAVE_MFX_PLUGIN
        static const mfxPluginUID hevc_enc_uid = { 0x6f, 0xad, 0xc7, 0x91, 0xa0, 0xc2, 0xeb, 0x47, 0x9a, 0xb6, 0xdc, 0xd5, 0xea, 0x9d, 0xa3, 0x47 };
        if (codecId == MFX_CODEC_HEVC)
            return new Plugin(session, hevc_enc_uid);
#else
        CV_UNUSED(session); CV_UNUSED(codecId);
#endif
        return 0;
    }
    static Plugin * loadDecoderPlugin(MFXVideoSession_WRAP &session, mfxU32 codecId)
    {
#ifdef HAVE_MFX_PLUGIN
        static const mfxPluginUID hevc_dec_uid = { 0x33, 0xa6, 0x1c, 0x0b, 0x4c, 0x27, 0x45, 0x4c, 0xa8, 0xd8, 0x5d, 0xde, 0x75, 0x7c, 0x6f, 0x8e };
        if (codecId == MFX_CODEC_HEVC)
            return new Plugin(session, hevc_dec_uid);
#else
        CV_UNUSED(session); CV_UNUSED(codecId);
#endif
        return 0;
    }
    ~Plugin()
    {
#ifdef HAVE_MFX_PLUGIN
        if (isGood())
            MFXVideoUSER_UnLoad(session, &uid);
#endif
    }
    bool isGood() const { return res >= MFX_ERR_NONE; }
private:
    mfxStatus res;
private:
#ifdef HAVE_MFX_PLUGIN
    MFXVideoSession_WRAP &session;
    mfxPluginUID uid;
    Plugin(MFXVideoSession_WRAP &_session, mfxPluginUID _uid) : session(_session), uid(_uid)
    {
        res = MFXVideoUSER_Load(session, &uid, 1);
    }
#endif
    Plugin(const Plugin &);
    Plugin &operator=(const Plugin &);
};

//==================================================================================================

class ReadBitstream
{
public:
    ReadBitstream(const char * filename, size_t maxSize = 10 * 1024 * 1024);
    ~ReadBitstream();
    bool isOpened() const;
    bool isDone() const;
    bool read();
private:
    ReadBitstream(const ReadBitstream &);
    ReadBitstream &operator=(const ReadBitstream &);
public:
    std::fstream input;
    mfxBitstream stream;
    bool drain;
};

//==================================================================================================

class WriteBitstream
{
public:
    WriteBitstream(const char * filename, size_t maxSize);
    ~WriteBitstream();
    bool write();
    bool isOpened() const;
private:
    WriteBitstream(const WriteBitstream &);
    WriteBitstream &operator=(const WriteBitstream &);
public:
    std::fstream output;
    mfxBitstream stream;
};

//==================================================================================================

class SurfacePool
{
public:
    SurfacePool(ushort width_, ushort height_, ushort count, const mfxFrameInfo & frameInfo, uchar bpp = 12);
    ~SurfacePool();
    mfxFrameSurface1 *getFreeSurface();

    template <typename T>
    static SurfacePool * create(T * instance, mfxVideoParam &params)
    {
        CV_Assert(instance);
        mfxFrameAllocRequest request;
        memset(&request, 0, sizeof(request));
        mfxStatus res = instance->QueryIOSurf(&params, &request);
        DBG(std::cout << "MFX QueryIOSurf: " << res << std::endl);
        if (res < MFX_ERR_NONE)
            return 0;
        return _create(request, params);
    }
private:
    static SurfacePool* _create(const mfxFrameAllocRequest& request, const mfxVideoParam& params);
private:
    SurfacePool(const SurfacePool &);
    SurfacePool &operator=(const SurfacePool &);
public:
    size_t width, height;
    size_t oneSize;
    cv::AutoBuffer<uchar, 0> buffers;
    std::vector<mfxFrameSurface1> surfaces;
};

//==================================================================================================

class DeviceHandler {
public:
    virtual ~DeviceHandler() {}
    bool init(MFXVideoSession_WRAP &session);
protected:
    virtual bool initDeviceSession(MFXVideoSession_WRAP &session) = 0;
};


// TODO: move to core::util?
#include <thread>
static void sleep_ms(int64 ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


// Linux specific
#ifdef __linux__

#include <unistd.h>
#include <va/va_drm.h>

class VAHandle : public DeviceHandler {
public:
    VAHandle();
    ~VAHandle();
private:
    VAHandle(const VAHandle &);
    VAHandle &operator=(const VAHandle &);
    bool initDeviceSession(MFXVideoSession_WRAP &session) CV_OVERRIDE;
private:
    VADisplay display;
    int file;
};

#endif // __linux__

// Windows specific
#ifdef _WIN32

#include <Windows.h>

class DXHandle : public DeviceHandler {
public:
    DXHandle() {}
    ~DXHandle() {}
private:
    DXHandle(const DXHandle &);
    DXHandle &operator=(const DXHandle &);
    bool initDeviceSession(MFXVideoSession_WRAP &) CV_OVERRIDE { return true; }
};

#endif // _WIN32

DeviceHandler * createDeviceHandler();

#endif // MFXHELPER_H
