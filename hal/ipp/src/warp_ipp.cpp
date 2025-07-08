// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include "precomp_ipp.hpp"

#ifdef HAVE_IPP_IW
#include "iw++/iw.hpp"
#endif

#define IPP_WARPAFFINE_PARALLEL 1
#define CV_TYPE(src_type) (src_type & (CV_DEPTH_MAX - 1))
#ifdef HAVE_IPP_IW

class ipp_warpAffineParallel: public cv::ParallelLoopBody
{
public:
    ipp_warpAffineParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, IppiInterpolationType _inter, double (&_coeffs)[2][3], ::ipp::IwiBorderType _borderType, IwTransDirection _iwTransDirection, bool *_ok):m_src(src), m_dst(dst)
    {
        pOk = _ok;

        inter          = _inter;
        borderType     = _borderType;
        iwTransDirection = _iwTransDirection;

        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = _coeffs[i][j];

        *pOk = true;
    }
    ~ipp_warpAffineParallel() {}

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        //CV_INSTRUMENT_REGION_IPP();

        if(*pOk == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, m_src, m_dst, coeffs, iwTransDirection, inter, ::ipp::IwiWarpAffineParams(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    IppiInterpolationType inter;
    double coeffs[2][3];
    ::ipp::IwiBorderType borderType;
    IwTransDirection iwTransDirection;

    bool  *pOk;
    const ipp_warpAffineParallel& operator= (const ipp_warpAffineParallel&);
};

#if (IPP_VERSION_X100 >= 700)
int ipp_hal_warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width,
                              int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4])
{
    //CV_INSTRUMENT_REGION_IPP();

    IppiInterpolationType ippInter    = ippiGetInterpolation(interpolation);
    if((int)ippInter < 0 || interpolation > 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

                                     /* C1         C2         C3         C4 */
    char impl[CV_DEPTH_MAX][4][3]={{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8U
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8S
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}},   //16U
                                   {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}},   //16S
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32S
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32F
                                   {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}}};  //64F

    if(impl[CV_TYPE(src_type)][CV_MAT_CN(src_type)-1][interpolation] == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Acquire data and begin processing
    try
    {
        ::ipp::IwiImage        iwSrc;
        iwSrc.Init({src_width, src_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), NULL, src_data, IwSize(src_step));
        ::ipp::IwiImage        iwDst({dst_width, dst_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), NULL, dst_data, dst_step);
        ::ipp::IwiBorderType   ippBorder(ippiGetBorderType(borderType), {borderValue[0], borderValue[1], borderValue[2], borderValue[3]});
        IwTransDirection       iwTransDirection = iwTransForward;

        if((int)ippBorder == -1)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        double coeffs[2][3];
        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = M[i*3 + j];

        const int threads = ippiSuggestThreadsNum(iwDst, 2);

        if(IPP_WARPAFFINE_PARALLEL && threads > 1)
        {
            bool  ok      = true;
            cv::Range range(0, (int)iwDst.m_size.height);
            ipp_warpAffineParallel invoker(iwSrc, iwDst, ippInter, coeffs, ippBorder, iwTransDirection, &ok);
            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;

            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, iwSrc, iwDst, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpAffineParams(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}
#endif
#endif

typedef IppStatus (CV_STDCALL* ippiSetFunc)(const void*, void *, int, IppiSize);

template <int channels, typename Type>
bool IPPSetSimple(const double value[4], void *dataPointer, int step, IppiSize &size, ippiSetFunc func)
{
    //CV_INSTRUMENT_REGION_IPP();

    Type values[channels];
    for( int i = 0; i < channels; i++ )
        values[i] = cv::saturate_cast<Type>(value[i]);
    return CV_INSTRUMENT_FUN_IPP(func, values, dataPointer, step, size) >= 0;
}

static bool IPPSet(const double value[4], void *dataPointer, int step, IppiSize &size, int channels, int depth)
{
    //CV_INSTRUMENT_REGION_IPP();

    if( channels == 1 )
    {
        switch( depth )
        {
        case CV_8U:
            return CV_INSTRUMENT_FUN_IPP(ippiSet_8u_C1R, cv::saturate_cast<Ipp8u>(value[0]), (Ipp8u *)dataPointer, step, size) >= 0;
        case CV_16U:
            return CV_INSTRUMENT_FUN_IPP(ippiSet_16u_C1R, cv::saturate_cast<Ipp16u>(value[0]), (Ipp16u *)dataPointer, step, size) >= 0;
        case CV_32F:
            return CV_INSTRUMENT_FUN_IPP(ippiSet_32f_C1R, cv::saturate_cast<Ipp32f>(value[0]), (Ipp32f *)dataPointer, step, size) >= 0;
        }
    }
    else
    {
        if( channels == 3 )
        {
            switch( depth )
            {
            case CV_8U:
                return IPPSetSimple<3, Ipp8u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_8u_C3R);
            case CV_16U:
                return IPPSetSimple<3, Ipp16u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_16u_C3R);
            case CV_32F:
                return IPPSetSimple<3, Ipp32f>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_32f_C3R);
            }
        }
        else if( channels == 4 )
        {
            switch( depth )
            {
            case CV_8U:
                return IPPSetSimple<4, Ipp8u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_8u_C4R);
            case CV_16U:
                return IPPSetSimple<4, Ipp16u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_16u_C4R);
            case CV_32F:
                return IPPSetSimple<4, Ipp32f>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_32f_C4R);
            }
        }
    }
    return false;
}

#if (IPP_VERSION_X100 >= 810)
typedef IppStatus (CV_STDCALL* ippiWarpPerspectiveFunc)(const Ipp8u*, int, Ipp8u*, int,IppiPoint, IppiSize, const IppiWarpSpec*,Ipp8u*);

class IPPWarpPerspectiveInvoker :
    public cv::ParallelLoopBody
{
public:
    IPPWarpPerspectiveInvoker(int _src_type, cv::Mat &_src, size_t _src_step, cv::Mat &_dst, size_t _dst_step, IppiInterpolationType _interpolation,
                              double (&_coeffs)[3][3], int &_borderType, const double _borderValue[4], ippiWarpPerspectiveFunc _func,
                              bool *_ok) :
        ParallelLoopBody(), src_type(_src_type), src(_src), src_step(_src_step), dst(_dst), dst_step(_dst_step), inter(_interpolation), coeffs(_coeffs),
        borderType(_borderType), func(_func), ok(_ok)
    {
        memcpy(this->borderValue, _borderValue, sizeof(this->borderValue));
        *ok = true;
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        //CV_INSTRUMENT_REGION_IPP();
        IppiWarpSpec* pSpec = 0;
        int specSize = 0, initSize = 0, bufSize = 0; Ipp8u* pBuffer  = 0;
        IppiPoint dstRoiOffset = {0, 0};
        IppiWarpDirection direction = ippWarpBackward;  //fixed for IPP
        const Ipp32u numChannels = CV_MAT_CN(src_type);

        IppiSize srcsize = {src.cols, src.rows};
        IppiSize dstsize = {dst.cols, dst.rows};
        IppiRect srcroi = {0, 0, src.cols, src.rows};

        /* Spec and init buffer sizes */
        IppStatus status = ippiWarpPerspectiveGetSize(srcsize, srcroi, dstsize, ippiGetDataType(src_type), coeffs, inter, ippWarpBackward, ippiGetBorderType(borderType), &specSize, &initSize);

        pSpec = (IppiWarpSpec*)ippMalloc_L(specSize);

        if (inter == ippLinear)
        {
            status = ippiWarpPerspectiveLinearInit(srcsize, srcroi, dstsize, ippiGetDataType(src_type), coeffs, direction, numChannels, ippiGetBorderType(borderType),
                                            borderValue, 0, pSpec);
        } else
        {
            status = ippiWarpPerspectiveNearestInit(srcsize, srcroi, dstsize, ippiGetDataType(src_type), coeffs, direction, numChannels, ippiGetBorderType(borderType),
                                            borderValue, 0, pSpec);
        }

        status = ippiWarpGetBufferSize(pSpec, dstsize, &bufSize);
        pBuffer = (Ipp8u*)ippMalloc_L(bufSize);
        IppiSize dstRoiSize = dstsize;

        int cnn = src.channels();

        if( borderType == cv::BorderTypes::BORDER_CONSTANT )
        {
            IppiSize setSize = {dst.cols, range.end - range.start};
            void *dataPointer = dst.ptr(range.start);
            if( !IPPSet( borderValue, dataPointer, (int)dst.step[0], setSize, cnn, src.depth() ) )
            {
                *ok = false;
                return;
            }
        }

        status = CV_INSTRUMENT_FUN_IPP(func, src.ptr(), (int)src_step, dst.ptr(), (int)dst_step, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
        if (status != ippStsNoErr)
            *ok = false;
        else
        {
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        }
    }
private:
    int src_type;
    cv::Mat &src;
    size_t src_step;
    cv::Mat &dst;
    size_t dst_step;
    IppiInterpolationType inter;
    double (&coeffs)[3][3];
    int borderType;
    double borderValue[4];
    ippiWarpPerspectiveFunc func;
    bool *ok;

    const IPPWarpPerspectiveInvoker& operator= (const IPPWarpPerspectiveInvoker&);
};

int ipp_hal_warpPerspective(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar * dst_data, size_t dst_step,
                            int dst_width, int dst_height, const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    //CV_INSTRUMENT_REGION_IPP();
    ippiWarpPerspectiveFunc ippFunc = 0;
    if (interpolation == cv::InterpolationFlags::INTER_NEAREST)
    {
        ippFunc = src_type == CV_8UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_8u_C1R :
        src_type == CV_8UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_8u_C3R :
        src_type == CV_8UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_8u_C4R :
        src_type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16u_C1R :
        src_type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16u_C3R :
        src_type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16u_C4R :
        src_type == CV_16SC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16s_C1R :
        src_type == CV_16SC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16s_C3R :
        src_type == CV_16SC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16s_C4R :
        src_type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_32f_C1R :
        src_type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_32f_C3R :
        src_type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_32f_C4R : 0;
    }
    else if (interpolation == cv::InterpolationFlags::INTER_LINEAR)
    {
        ippFunc = src_type == CV_8UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_8u_C1R :
        src_type == CV_8UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_8u_C3R :
        src_type == CV_8UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_8u_C4R :
        src_type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16u_C1R :
        src_type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16u_C3R :
        src_type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16u_C4R :
        src_type == CV_16SC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16s_C1R :
        src_type == CV_16SC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16s_C3R :
        src_type == CV_16SC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16s_C4R :
        src_type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_32f_C1R :
        src_type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_32f_C3R :
        src_type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_32f_C4R : 0;
    }
    else
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if(src_height == 1 || src_width == 1) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int mode =
    interpolation == cv::InterpolationFlags::INTER_NEAREST ? IPPI_INTER_NN :
    interpolation == cv::InterpolationFlags::INTER_LINEAR ? IPPI_INTER_LINEAR : 0;

    if (mode == 0 || ippFunc == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

                                    /* C1      C2      C3      C4 */
    char impl[CV_DEPTH_MAX][4][2]={{{0, 0}, {1, 1}, {0, 0}, {0, 0}},   //8U
                                   {{1, 1}, {1, 1}, {1, 1}, {1, 1}},   //8S
                                   {{0, 0}, {1, 1}, {0, 1}, {0, 1}},   //16U
                                   {{1, 1}, {1, 1}, {1, 1}, {1, 1}},   //16S
                                   {{1, 1}, {1, 1}, {1, 0}, {1, 1}},   //32S
                                   {{1, 0}, {1, 0}, {0, 0}, {1, 0}},   //32F
                                   {{1, 1}, {1, 1}, {1, 1}, {1, 1}}};  //64F

    if(impl[CV_TYPE(src_type)][CV_MAT_CN(src_type)-1][interpolation] == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    double coeffs[3][3];
    for( int i = 0; i < 3; i++ )
        for( int j = 0; j < 3; j++ )
            coeffs[i][j] = M[i*3 + j];

    bool ok;
    cv::Range range(0, dst_height);
    cv::Mat src(cv::Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    cv::Mat dst(cv::Size(dst_width, dst_height), src_type, dst_data, dst_step);
    IppiInterpolationType ippInter    = ippiGetInterpolation(interpolation);
    IPPWarpPerspectiveInvoker invoker(src_type, src, src_step, dst, dst_step, ippInter, coeffs, borderType, borderValue, ippFunc, &ok);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));

    if( ok )
    {
        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
    }
    return CV_HAL_ERROR_OK;
}
#endif

typedef IppStatus(CV_STDCALL *ippiRemap)(const void *pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi,
                                         const Ipp32f *pxMap, int xMapStep, const Ipp32f *pyMap, int yMapStep,
                                         void *pDst, int dstStep, IppiSize dstRoiSize, int interpolation);

class IPPRemapInvoker : public cv::ParallelLoopBody
{
public:
    IPPRemapInvoker(int _src_type, const uchar *_src_data, size_t _src_step, int _src_width, int _src_height,
                    uchar *_dst_data, size_t _dst_step, int _dst_width, float *_mapx, size_t _mapx_step, float *_mapy,
                    size_t _mapy_step, ippiRemap _ippFunc, int _ippInterpolation, int _borderType, const double _borderValue[4], bool *_ok) :
        ParallelLoopBody(),
        src_type(_src_type), src(_src_data), src_step(_src_step), src_width(_src_width), src_height(_src_height),
        dst(_dst_data), dst_step(_dst_step), dst_width(_dst_width), mapx(_mapx), mapx_step(_mapx_step), mapy(_mapy),
        mapy_step(_mapy_step), ippFunc(_ippFunc), ippInterpolation(_ippInterpolation), borderType(_borderType), ok(_ok)
    {
        memcpy(this->borderValue, _borderValue, sizeof(this->borderValue));
        *ok = true;
    }

    virtual void operator()(const cv::Range &range) const
    {
        IppiRect srcRoiRect = {0, 0, src_width, src_height};
        uchar *dst_roi_data = dst + range.start * dst_step;
        IppiSize dstRoiSize = ippiSize(dst_width, range.size());
        int depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);

        if (borderType == cv::BORDER_CONSTANT &&
            !IPPSet(borderValue, dst_roi_data, (int)dst_step, dstRoiSize, cn, depth))
        {
            *ok = false;
            return;
        }

        if (CV_INSTRUMENT_FUN_IPP(ippFunc, src, {src_width, src_height}, (int)src_step, srcRoiRect,
                                  mapx, (int)mapx_step, mapy, (int)mapy_step,
                                  dst_roi_data, (int)dst_step, dstRoiSize, ippInterpolation) < 0)
            *ok = false;
        else
        {
            CV_IMPL_ADD(CV_IMPL_IPP | CV_IMPL_MT);
        }
    }

private:
    int src_type;
    const uchar *src;
    size_t src_step;
    int src_width, src_height;
    uchar *dst;
    size_t dst_step;
    int dst_width;
    float *mapx;
    size_t mapx_step;
    float *mapy;
    size_t mapy_step;
    ippiRemap ippFunc;
    int ippInterpolation, borderType;
    double borderValue[4];
    bool *ok;
};

int ipp_hal_remap32f(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
                     uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
                     float *mapx, size_t mapx_step, float *mapy, size_t mapy_step,
                     int interpolation, int border_type, const double border_value[4])
{
    if ((interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC || interpolation == cv::INTER_NEAREST) &&
        (border_type == cv::BORDER_CONSTANT || border_type == cv::BORDER_TRANSPARENT))
    {
        int ippInterpolation =
            interpolation == cv::INTER_NEAREST ? IPPI_INTER_NN : interpolation == cv::INTER_LINEAR ? IPPI_INTER_LINEAR
                                                                                                   : IPPI_INTER_CUBIC;

                                         /* C1         C2         C3         C4 */
        char impl[CV_DEPTH_MAX][4][3]={{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8U
                                       {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8S
                                       {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //16U
                                       {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //16S
                                       {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32S
                                       {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32F
                                       {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};  //64F

        if (impl[CV_TYPE(src_type)][CV_MAT_CN(src_type) - 1][interpolation] == 0)
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        ippiRemap ippFunc =
            src_type == CV_8UC1 ? (ippiRemap)ippiRemap_8u_C1R : src_type == CV_8UC3 ? (ippiRemap)ippiRemap_8u_C3R
                                                            : src_type == CV_8UC4   ? (ippiRemap)ippiRemap_8u_C4R
                                                            : src_type == CV_16UC1  ? (ippiRemap)ippiRemap_16u_C1R
                                                            : src_type == CV_16UC3  ? (ippiRemap)ippiRemap_16u_C3R
                                                            : src_type == CV_16UC4  ? (ippiRemap)ippiRemap_16u_C4R
                                                            : src_type == CV_32FC1  ? (ippiRemap)ippiRemap_32f_C1R
                                                            : src_type == CV_32FC3  ? (ippiRemap)ippiRemap_32f_C3R
                                                            : src_type == CV_32FC4  ? (ippiRemap)ippiRemap_32f_C4R
                                                                                    : 0;

        if (ippFunc)
        {
            bool ok;

            IPPRemapInvoker invoker(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width,
                                    mapx, mapx_step, mapy, mapy_step, ippFunc, ippInterpolation, border_type, border_value, &ok);
            cv::Range range(0, dst_height);
            cv::parallel_for_(range, invoker, dst_width * dst_height / (double)(1 << 16));

            if (ok)
            {
                CV_IMPL_ADD(CV_IMPL_IPP | CV_IMPL_MT);
                return CV_HAL_ERROR_OK;
            }
        }
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
