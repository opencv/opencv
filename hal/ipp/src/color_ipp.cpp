// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_imgproc.hpp"
#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include <limits>

#if IPP_VERSION_X100 >= 700

#define MAX_IPP8u   255
#define MAX_IPP16u  65535
#define MAX_IPP32f  1.0

// moved from color_rgb.dispatch.cpp
#define IPP_DISABLE_CVTCOLOR_GRAY2BGR_8UC3 1

// replicated from core/private.hpp
#define IPP_DISABLE_RGB_HSV 1
#define IPP_DISABLE_RGB_YUV 1
#define IPP_DISABLE_YUV_RGB 1
#define IPP_DISABLE_RGB_XYZ 1
#define IPP_DISABLE_XYZ_RGB 1
#define IPP_DISABLE_RGB_LAB 1
#define IPP_DISABLE_LAB_RGB 1

namespace {

typedef IppStatus (CV_STDCALL* ippiGeneralFunc)(const void *, int, void *, int, IppiSize);
typedef IppStatus (CV_STDCALL* ippiColor2GrayFunc)(const void *, int, void *, int, IppiSize, const Ipp32f *);
typedef IppStatus (CV_STDCALL* ippiReorderFunc)(const void *, int, void *, int, IppiSize, const int *);

// BT.601 BGR->Y weights
static const float B2YF = 0.114f;
static const float G2YF = 0.587f;
static const float R2YF = 0.299f;

template<typename _Tp> struct ColorChannel
{
    static inline _Tp max() { return std::numeric_limits<_Tp>::max(); }
};
template<> struct ColorChannel<float>
{
    static inline float max() { return 1.f; }
};

template <typename Cvt>
class CvtColorIPPLoop_Invoker : public cv::ParallelLoopBody
{
public:
    CvtColorIPPLoop_Invoker(const uchar * src_data_, size_t src_step_, uchar * dst_data_, size_t dst_step_,
                            int width_, const Cvt& _cvt, bool *_ok) :
        ParallelLoopBody(), src_data(src_data_), src_step(src_step_), dst_data(dst_data_), dst_step(dst_step_),
        width(width_), cvt(_cvt), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        const void *yS = src_data + src_step * range.start;
        void *yD = dst_data + dst_step * range.start;
        if( !cvt(yS, static_cast<int>(src_step), yD, static_cast<int>(dst_step), width, range.end - range.start) )
            *ok = false;
        else
        {
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        }
    }

private:
    const uchar * src_data;
    const size_t src_step;
    uchar * dst_data;
    const size_t dst_step;
    const int width;
    const Cvt& cvt;
    bool *ok;

    const CvtColorIPPLoop_Invoker& operator= (const CvtColorIPPLoop_Invoker&);
};

template <typename Cvt>
bool CvtColorIPPLoop(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                     int width, int height, const Cvt& cvt)
{
    bool ok;
    cv::parallel_for_(cv::Range(0, height),
                      CvtColorIPPLoop_Invoker<Cvt>(src_data, src_step, dst_data, dst_step, width, cvt, &ok),
                      (width * height)/(double)(1<<16) );
    return ok;
}

template <typename Cvt>
bool CvtColorIPPLoopCopy(const uchar * src_data, size_t src_step, int src_type, uchar * dst_data, size_t dst_step,
                         int width, int height, const Cvt& cvt)
{
    cv::Mat temp;
    cv::Mat src(cv::Size(width, height), src_type, const_cast<uchar*>(src_data), src_step);
    cv::Mat source = src;
    if( src_data == dst_data )
    {
        src.copyTo(temp);
        source = temp;
    }
    bool ok;
    cv::parallel_for_(cv::Range(0, source.rows),
                      CvtColorIPPLoop_Invoker<Cvt>(source.data, source.step, dst_data, dst_step,
                                                   source.cols, cvt, &ok),
                      source.total()/(double)(1<<16) );
    return ok;
}

struct IPPGeneralFunctor
{
    IPPGeneralFunctor(ippiGeneralFunc _func) : ippiColorConvertGeneral(_func){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiColorConvertGeneral ? CV_INSTRUMENT_FUN_IPP(ippiColorConvertGeneral, src, srcStep, dst, dstStep, ippiSize(cols, rows)) >= 0 : false;
    }
private:
    ippiGeneralFunc ippiColorConvertGeneral;
};

struct IPPReorderFunctor
{
    IPPReorderFunctor(ippiReorderFunc _func, int _order0, int _order1, int _order2) : ippiColorConvertReorder(_func)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiColorConvertReorder ? CV_INSTRUMENT_FUN_IPP(ippiColorConvertReorder, src, srcStep, dst, dstStep, ippiSize(cols, rows), order) >= 0 : false;
    }
private:
    ippiReorderFunc ippiColorConvertReorder;
    int order[4];
};

struct IPPReorderGeneralFunctor
{
    IPPReorderGeneralFunctor(ippiReorderFunc _func1, ippiGeneralFunc _func2, int _order0, int _order1, int _order2, int _depth) :
        ippiColorConvertReorder(_func1), ippiColorConvertGeneral(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        if (ippiColorConvertReorder == 0 || ippiColorConvertGeneral == 0)
            return false;

        cv::Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(CV_INSTRUMENT_FUN_IPP(ippiColorConvertReorder, src, srcStep, temp.ptr(), (int)temp.step[0], ippiSize(cols, rows), order) < 0)
            return false;
        return CV_INSTRUMENT_FUN_IPP(ippiColorConvertGeneral, temp.ptr(), (int)temp.step[0], dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiReorderFunc ippiColorConvertReorder;
    ippiGeneralFunc ippiColorConvertGeneral;
    int order[4];
    int depth;
};

struct IPPGeneralReorderFunctor
{
    IPPGeneralReorderFunctor(ippiGeneralFunc _func1, ippiReorderFunc _func2, int _order0, int _order1, int _order2, int _depth) :
        ippiColorConvertGeneral(_func1), ippiColorConvertReorder(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        if (ippiColorConvertGeneral == 0 || ippiColorConvertReorder == 0)
            return false;

        cv::Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(CV_INSTRUMENT_FUN_IPP(ippiColorConvertGeneral, src, srcStep, temp.ptr(), (int)temp.step[0], ippiSize(cols, rows)) < 0)
            return false;
        return CV_INSTRUMENT_FUN_IPP(ippiColorConvertReorder, temp.ptr(), (int)temp.step[0], dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiGeneralFunc ippiColorConvertGeneral;
    ippiReorderFunc ippiColorConvertReorder;
    int order[4];
    int depth;
};

struct IPPColor2GrayFunctor
{
    IPPColor2GrayFunctor(ippiColor2GrayFunc _func) : ippiColorToGray(_func)
    {
        coeffs[0] = B2YF;
        coeffs[1] = G2YF;
        coeffs[2] = R2YF;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiColorToGray ? CV_INSTRUMENT_FUN_IPP(ippiColorToGray, src, srcStep, dst, dstStep, ippiSize(cols, rows), coeffs) >= 0 : false;
    }
private:
    ippiColor2GrayFunc ippiColorToGray;
    Ipp32f coeffs[3];
};

#if !IPP_DISABLE_CVTCOLOR_GRAY2BGR_8UC3
static IppStatus ippiGrayToRGB_C1C3R(const Ipp8u*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize roiSize)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_8u_C1C3R, pSrc, srcStep, pDst, dstStep, roiSize);
}
#endif
static IppStatus ippiGrayToRGB_C1C3R(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_16u_C1C3R, pSrc, srcStep, pDst, dstStep, roiSize);
}
static IppStatus ippiGrayToRGB_C1C3R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_32f_C1C3R, pSrc, srcStep, pDst, dstStep, roiSize);
}

static IppStatus ippiGrayToRGB_C1C4R(const Ipp8u*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize roiSize, Ipp8u  aval)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_8u_C1C4R, pSrc, srcStep, pDst, dstStep, roiSize, aval);
}
static IppStatus ippiGrayToRGB_C1C4R(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize, Ipp16u aval)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_16u_C1C4R, pSrc, srcStep, pDst, dstStep, roiSize, aval);
}
static IppStatus ippiGrayToRGB_C1C4R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f aval)
{
    return CV_INSTRUMENT_FUN_IPP(ippiGrayToRGB_32f_C1C4R, pSrc, srcStep, pDst, dstStep, roiSize, aval);
}

template <typename T>
struct IPPGray2BGRFunctor
{
    IPPGray2BGRFunctor(){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiGrayToRGB_C1C3R((T*)src, srcStep, (T*)dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
};

template <typename T>
struct IPPGray2BGRAFunctor
{
    IPPGray2BGRAFunctor()
    {
        alpha = ColorChannel<T>::max();
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiGrayToRGB_C1C4R((T*)src, srcStep, (T*)dst, dstStep, ippiSize(cols, rows), alpha) >= 0;
    }
    T alpha;
};

static IppStatus CV_STDCALL ippiSwapChannels_8u_C3C4Rf(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_8u_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP8u);
}
static IppStatus CV_STDCALL ippiSwapChannels_16u_C3C4Rf(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_16u_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP16u);
}
static IppStatus CV_STDCALL ippiSwapChannels_32f_C3C4Rf(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP32f);
}

static const ippiColor2GrayFunc ippiColor2GrayC3Tab[CV_DEPTH_MAX] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_C3C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_C3C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_C3C1R, 0, 0
};

static const ippiColor2GrayFunc ippiColor2GrayC4Tab[CV_DEPTH_MAX] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_AC4C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_AC4C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_AC4C1R, 0, 0
};

static const ippiGeneralFunc ippiRGB2GrayC3Tab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_C3C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_C3C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_C3C1R, 0, 0
};

static const ippiGeneralFunc ippiRGB2GrayC4Tab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_AC4C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_AC4C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_AC4C1R, 0, 0
};

static const ippiReorderFunc ippiSwapChannelsC3C4RTab[CV_DEPTH_MAX] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3C4Rf, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3C4Rf, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3C4Rf, 0, 0
};

static const ippiGeneralFunc ippiCopyAC4C3RTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiCopy_8u_AC4C3R, 0, (ippiGeneralFunc)ippiCopy_16u_AC4C3R, 0,
    0, (ippiGeneralFunc)ippiCopy_32f_AC4C3R, 0, 0
};

static const ippiReorderFunc ippiSwapChannelsC4C3RTab[CV_DEPTH_MAX] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4C3R, 0, 0
};

static const ippiReorderFunc ippiSwapChannelsC3RTab[CV_DEPTH_MAX] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3R, 0, 0
};

#if IPP_VERSION_X100 >= 810
static const ippiReorderFunc ippiSwapChannelsC4RTab[CV_DEPTH_MAX] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4R, 0, 0
};
#endif

#if !IPP_DISABLE_RGB_HSV
static const ippiGeneralFunc ippiRGB2HSVTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToHSV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHSV_16u_C3R, 0,
    0, 0, 0, 0
};
#endif

static const ippiGeneralFunc ippiHSV2RGBTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiHSVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHSVToRGB_16u_C3R, 0,
    0, 0, 0, 0
};

static const ippiGeneralFunc ippiRGB2HLSTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToHLS_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHLS_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToHLS_32f_C3R, 0, 0
};

static const ippiGeneralFunc ippiHLS2RGBTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiHLSToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHLSToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiHLSToRGB_32f_C3R, 0, 0
};

#if !IPP_DISABLE_RGB_XYZ
static const ippiGeneralFunc ippiRGB2XYZTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToXYZ_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToXYZ_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToXYZ_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_XYZ_RGB
static const ippiGeneralFunc ippiXYZ2RGBTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiXYZToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiXYZToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiXYZToRGB_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_RGB_LAB
static const ippiGeneralFunc ippiRGBToLUVTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToLUV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToLUV_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToLUV_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_LAB_RGB
static const ippiGeneralFunc ippiLUVToRGBTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiLUVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiLUVToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiLUVToRGB_32f_C3R, 0, 0
};
#endif

} // namespace

int ipp_hal_cvtBGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, int dcn, bool swapBlue)
{
    CV_HAL_CHECK_USE_IPP();

    if(scn == 3 && dcn == 4 && !swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 0, 1, 2)) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 4 && dcn == 3 && !swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralFunctor(ippiCopyAC4C3RTab[depth])) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 3 && dcn == 4 && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 2, 1, 0)) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 4 && dcn == 3 && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderFunctor(ippiSwapChannelsC4C3RTab[depth], 2, 1, 0)) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 3 && dcn == 3 && swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                IPPReorderFunctor(ippiSwapChannelsC3RTab[depth], 2, 1, 0)) )
            return CV_HAL_ERROR_OK;
    }
#if IPP_VERSION_X100 >= 810
    else if(scn == 4 && dcn == 4 && swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                IPPReorderFunctor(ippiSwapChannelsC4RTab[depth], 2, 1, 0)) )
            return CV_HAL_ERROR_OK;
    }
#endif

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtGraytoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                         int width, int height, int depth, int dcn)
{
    CV_HAL_CHECK_USE_IPP();

    bool ippres = false;
    if(dcn == 3)
    {
        if( depth == CV_8U )
        {
#if !IPP_DISABLE_CVTCOLOR_GRAY2BGR_8UC3
            ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRFunctor<Ipp8u>());
#endif
        }
        else if( depth == CV_16U )
            ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRFunctor<Ipp16u>());
        else
            ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRFunctor<Ipp32f>());
    }
    else if(dcn == 4)
    {
        if( depth == CV_8U )
            ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRAFunctor<Ipp8u>());
        else if( depth == CV_16U )
            ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRAFunctor<Ipp16u>());
        else
            ippres = CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height, IPPGray2BGRAFunctor<Ipp32f>());
    }

    return ippres ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtBGRtoGray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                         int width, int height, int depth, int scn, bool swapBlue)
{
    CV_HAL_CHECK_USE_IPP();

    // preserves original cvtBGRtoGray routing: only 32f was sent to IPP, other depths use the dispatch path
    if (depth != CV_32F)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(scn == 3 && !swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPColor2GrayFunctor(ippiColor2GrayC3Tab[depth])) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 3 && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralFunctor(ippiRGB2GrayC3Tab[depth])) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 4 && !swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPColor2GrayFunctor(ippiColor2GrayC4Tab[depth])) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 4 && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralFunctor(ippiRGB2GrayC4Tab[depth])) )
            return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtBGRtoHSV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_HAL_CHECK_USE_IPP();

    if(depth == CV_8U && isFullRange)
    {
        if (isHSV)
        {
#if !IPP_DISABLE_RGB_HSV
            if(scn == 3 && !swapBlue)
            {
                if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(scn == 4 && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(scn == 4 && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 0, 1, 2, depth)) )
                    return CV_HAL_ERROR_OK;
            }
#endif
        }
        else
        {
            if(scn == 3 && !swapBlue)
            {
                if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(scn == 4 && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(scn == 3 && swapBlue)
            {
                if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                        IPPGeneralFunctor(ippiRGB2HLSTab[depth])) )
                    return CV_HAL_ERROR_OK;
            }
            else if(scn == 4 && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 0, 1, 2, depth)) )
                    return CV_HAL_ERROR_OK;
            }
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtHSVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_HAL_CHECK_USE_IPP();

    if (depth == CV_8U && isFullRange)
    {
        if (isHSV)
        {
            if(dcn == 3 && !swapBlue)
            {
                if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(dcn == 4 && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(dcn == 3 && swapBlue)
            {
                if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                        IPPGeneralFunctor(ippiHSV2RGBTab[depth])) )
                    return CV_HAL_ERROR_OK;
            }
            else if(dcn == 4 && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                    return CV_HAL_ERROR_OK;
            }
        }
        else
        {
            if(dcn == 3 && !swapBlue)
            {
                if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(dcn == 4 && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if(dcn == 3 && swapBlue)
            {
                if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                        IPPGeneralFunctor(ippiHLS2RGBTab[depth])) )
                    return CV_HAL_ERROR_OK;
            }
            else if(dcn == 4 && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                    return CV_HAL_ERROR_OK;
            }
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                                    int width, int height)
{
    CV_HAL_CHECK_USE_IPP();

    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                        IPPGeneralFunctor((ippiGeneralFunc)ippiAlphaPremul_8u_AC4R)) )
        return CV_HAL_ERROR_OK;

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isCbCr)
{
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(depth); CV_UNUSED(scn); CV_UNUSED(swapBlue); CV_UNUSED(isCbCr);
    CV_HAL_CHECK_USE_IPP();

#if !IPP_DISABLE_RGB_YUV
    if (scn == 3 && depth == CV_8U && swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralFunctor((ippiGeneralFunc)ippiRGBToYUV_8u_C3R)))
            return CV_HAL_ERROR_OK;
    }
    else if (scn == 3 && depth == CV_8U && !swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                     (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 2, 1, 0, depth)))
            return CV_HAL_ERROR_OK;
    }
    else if (scn == 4 && depth == CV_8U && swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                     (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 0, 1, 2, depth)))
            return CV_HAL_ERROR_OK;
    }
    else if (scn == 4 && depth == CV_8U && !swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                     (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 2, 1, 0, depth)))
            return CV_HAL_ERROR_OK;
    }
#endif

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isCbCr)
{
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(depth); CV_UNUSED(dcn); CV_UNUSED(swapBlue); CV_UNUSED(isCbCr);
    CV_HAL_CHECK_USE_IPP();

#if !IPP_DISABLE_YUV_RGB
    if (dcn == 3 && depth == CV_8U && swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R)))
            return CV_HAL_ERROR_OK;
    }
    else if (dcn == 3 && depth == CV_8U && !swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                     ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)))
            return CV_HAL_ERROR_OK;
    }
    else if (dcn == 4 && depth == CV_8U && swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                     ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)))
            return CV_HAL_ERROR_OK;
    }
    else if (dcn == 4 && depth == CV_8U && !swapBlue && !isCbCr)
    {
        if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                     ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)))
            return CV_HAL_ERROR_OK;
    }
#endif

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtBGRtoXYZ(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue)
{
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(depth); CV_UNUSED(scn); CV_UNUSED(swapBlue);
    CV_HAL_CHECK_USE_IPP();

#if !IPP_DISABLE_RGB_XYZ
    if(scn == 3 && depth != CV_32F && !swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 4 && depth != CV_32F && !swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 3 && depth != CV_32F && swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                IPPGeneralFunctor(ippiRGB2XYZTab[depth])) )
            return CV_HAL_ERROR_OK;
    }
    else if(scn == 4 && depth != CV_32F && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 0, 1, 2, depth)) )
            return CV_HAL_ERROR_OK;
    }
#endif

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtXYZtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue)
{
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(depth); CV_UNUSED(dcn); CV_UNUSED(swapBlue);
    CV_HAL_CHECK_USE_IPP();

#if !IPP_DISABLE_XYZ_RGB
    if(dcn == 3 && depth != CV_32F && !swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
            return CV_HAL_ERROR_OK;
    }
    else if(dcn == 4 && depth != CV_32F && !swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
            return CV_HAL_ERROR_OK;
    }
    if(dcn == 3 && depth != CV_32F && swapBlue)
    {
        if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                IPPGeneralFunctor(ippiXYZ2RGBTab[depth])) )
            return CV_HAL_ERROR_OK;
    }
    else if(dcn == 4 && depth != CV_32F && swapBlue)
    {
        if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                            IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
            return CV_HAL_ERROR_OK;
    }
#endif

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtBGRtoLab(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(depth); CV_UNUSED(scn); CV_UNUSED(swapBlue); CV_UNUSED(isLab); CV_UNUSED(srgb);
    CV_HAL_CHECK_USE_IPP();

#if !IPP_DISABLE_RGB_LAB
    if (!srgb)
    {
        if (isLab)
        {
            if (scn == 3 && depth == CV_8U && !swapBlue)
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralFunctor((ippiGeneralFunc)ippiBGRToLab_8u_C3R)))
                    return CV_HAL_ERROR_OK;
            }
            else if (scn == 4 && depth == CV_8U && !swapBlue)
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                             (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 0, 1, 2, depth)))
                    return CV_HAL_ERROR_OK;
            }
            else if (scn == 3 && depth == CV_8U && swapBlue) // slower than OpenCV
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                             (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 2, 1, 0, depth)))
                    return CV_HAL_ERROR_OK;
            }
            else if (scn == 4 && depth == CV_8U && swapBlue) // slower than OpenCV
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                             (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 2, 1, 0, depth)))
                    return CV_HAL_ERROR_OK;
            }
        }
        else
        {
            if (scn == 3 && swapBlue)
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralFunctor(ippiRGBToLUVTab[depth])))
                    return CV_HAL_ERROR_OK;
            }
            else if (scn == 4 && swapBlue)
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                             ippiRGBToLUVTab[depth], 0, 1, 2, depth)))
                    return CV_HAL_ERROR_OK;
            }
            else if (scn == 3 && !swapBlue)
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                             ippiRGBToLUVTab[depth], 2, 1, 0, depth)))
                    return CV_HAL_ERROR_OK;
            }
            else if (scn == 4 && !swapBlue)
            {
                if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                             ippiRGBToLUVTab[depth], 2, 1, 0, depth)))
                    return CV_HAL_ERROR_OK;
            }
        }
    }
#endif

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_cvtLabtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(depth); CV_UNUSED(dcn); CV_UNUSED(swapBlue); CV_UNUSED(isLab); CV_UNUSED(srgb);
    CV_HAL_CHECK_USE_IPP();

#if !IPP_DISABLE_LAB_RGB
    if (!srgb)
    {
        if (isLab)
        {
            if( dcn == 3 && depth == CV_8U && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R)) )
                    return CV_HAL_ERROR_OK;
            }
            else if( dcn == 4 && depth == CV_8U && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                             ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            if( dcn == 3 && depth == CV_8U && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                             ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if( dcn == 4 && depth == CV_8U && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                             ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
        }
        else
        {
            if( dcn == 3 && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralFunctor(ippiLUVToRGBTab[depth])) )
                    return CV_HAL_ERROR_OK;
            }
            else if( dcn == 4 && swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                             ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            if( dcn == 3 && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                             ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
            else if( dcn == 4 && !swapBlue)
            {
                if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                             ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                    return CV_HAL_ERROR_OK;
            }
        }
    }
#endif

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 700
