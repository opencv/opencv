// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#if IPP_VERSION_X100 >= 810 && !DISABLE_IPP_MEDIAN_BLUR

// Copied from core/private.hpp (gated by HAVE_IPP, which the plugin lacks).
#if IPP_VERSION_X100 >= 201700
#define CV_IPP_MALLOC(SIZE) ippMalloc_L(SIZE)
#else
#define CV_IPP_MALLOC(SIZE) ippMalloc((int)SIZE)
#endif

template<typename T>
class IppAutoBuffer
{
public:
    IppAutoBuffer() { m_size = 0; m_pBuffer = NULL; }
    explicit IppAutoBuffer(size_t size) { m_size = 0; m_pBuffer = NULL; allocate(size); }
    ~IppAutoBuffer() { deallocate(); }
    T* allocate(size_t size)   { if(m_size < size) { deallocate(); m_pBuffer = (T*)CV_IPP_MALLOC(size); m_size = size; } return m_pBuffer; }
    void deallocate() { if(m_pBuffer) { ippFree(m_pBuffer); m_pBuffer = NULL; } m_size = 0; }
    inline T* get() { return (T*)m_pBuffer; }
    inline operator T* () { return (T*)m_pBuffer; }
    inline operator const T* () const { return (const T*)m_pBuffer; }
private:
    // Disable copy operations
    IppAutoBuffer(IppAutoBuffer &) {}
    IppAutoBuffer& operator =(const IppAutoBuffer &) { return *this; }

    size_t m_size;
    T*     m_pBuffer;
};

int ipp_hal_medianBlur(const uchar* src_data, size_t src_step,
                       uchar* dst_data, size_t dst_step,
                       int width, int height, int depth, int cn, int ksize)
{
    CV_HAL_CHECK_USE_IPP();

#if IPP_VERSION_X100 < 201801
    // Degradations for big kernel
    if(ksize > 7)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    int         bufSize;
    IppiSize    dstRoiSize = ippiSize(width, height), maskSize = ippiSize(ksize, ksize);
    IppDataType ippType    = ippiGetDataType(depth);
    int         channels   = cn;
    IppAutoBuffer<Ipp8u> buffer;

    // Wrap the raw input in a Mat header; for the in-place case (src == dst) make a
    // private copy so IPP does not read from and write to the same buffer.
    cv::Mat src(height, width, CV_MAKETYPE(depth, cn), (void*)src_data, src_step);
    if(src_data == dst_data)
        src = src.clone();

    if(ippiFilterMedianBorderGetBufferSize(dstRoiSize, maskSize, ippType, channels, &bufSize) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    buffer.allocate(bufSize);

    bool ok = false;
    switch(ippType)
    {
    case ipp8u:
        if(channels == 1)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C1R, src.ptr<Ipp8u>(), (int)src.step, (Ipp8u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        else if(channels == 3)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C3R, src.ptr<Ipp8u>(), (int)src.step, (Ipp8u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        else if(channels == 4)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C4R, src.ptr<Ipp8u>(), (int)src.step, (Ipp8u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        break;
    case ipp16u:
        if(channels == 1)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C1R, src.ptr<Ipp16u>(), (int)src.step, (Ipp16u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        else if(channels == 3)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C3R, src.ptr<Ipp16u>(), (int)src.step, (Ipp16u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        else if(channels == 4)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C4R, src.ptr<Ipp16u>(), (int)src.step, (Ipp16u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        break;
    case ipp16s:
        if(channels == 1)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C1R, src.ptr<Ipp16s>(), (int)src.step, (Ipp16s*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        else if(channels == 3)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C3R, src.ptr<Ipp16s>(), (int)src.step, (Ipp16s*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        else if(channels == 4)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C4R, src.ptr<Ipp16s>(), (int)src.step, (Ipp16s*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        break;
    case ipp32f:
        if(channels == 1)
            ok = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_32f_C1R, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
        break;
    default:
        break;
    }

    return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 810 && !DISABLE_IPP_MEDIAN_BLUR
