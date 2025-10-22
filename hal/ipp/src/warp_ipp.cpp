// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_imgproc.hpp"

#if IPP_VERSION_X100 >= 810 // integrated IPP warping/remap ABI is available since IPP v8.1

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

// Uncomment to enforce IPP calls for all supported by IPP configurations
// #define IPP_CALLS_ENFORCED

#define CV_IPP_SAFE_CALL(pFunc, pFlag, ...) if (pFunc(__VA_ARGS__) != ippStsNoErr) {*pFlag = false; return;}
#define CV_TYPE(src_type) (src_type & (CV_DEPTH_MAX - 1))

#ifdef HAVE_IPP_IW
// Warp affine section

#include "iw++/iw.hpp"

class ipp_warpAffineParallel: public cv::ParallelLoopBody
{
public:
    ipp_warpAffineParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, IppiInterpolationType _inter, double (&_coeffs)[2][3], ::ipp::IwiBorderType _borderType, IwTransDirection _iwTransDirection, bool *_ok):m_src(src), m_dst(dst)
    {
        ok = _ok;

        inter          = _inter;
        borderType     = _borderType;
        iwTransDirection = _iwTransDirection;

        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = _coeffs[i][j];

        *ok = true;
    }
    ~ipp_warpAffineParallel() {}

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        //CV_INSTRUMENT_REGION_IPP();
        if(*ok == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, m_src, m_dst, coeffs, iwTransDirection, inter, ::ipp::IwiWarpAffineParams(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *ok = false;
            return;
        }
        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    IppiInterpolationType inter;
    double coeffs[2][3];
    ::ipp::IwiBorderType borderType;
    IwTransDirection iwTransDirection;

    bool  *ok;
    const ipp_warpAffineParallel& operator= (const ipp_warpAffineParallel&);
};

int ipp_hal_warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width,
                              int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4])
{
    //CV_INSTRUMENT_REGION_IPP();

    IppiInterpolationType ippInter  = ippiGetInterpolation(interpolation);
    if((int)ippInter < 0 || interpolation > 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if defined(IPP_CALLS_ENFORCED)
                                     /* C1         C2         C3         C4 */
    char impl[CV_DEPTH_MAX][4][3]={{{1, 1, 0}, {0, 0, 0}, {1, 1, 0}, {1, 1, 0}},   //8U
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8S
                                   {{1, 1, 0}, {0, 0, 0}, {1, 1, 0}, {1, 1, 0}},   //16U
                                   {{1, 1, 0}, {0, 0, 0}, {1, 1, 0}, {1, 1, 0}},   //16S
                                   {{1, 1, 0}, {0, 0, 0}, {1, 1, 0}, {1, 1, 0}},   //32S
                                   {{1, 1, 0}, {0, 0, 0}, {1, 1, 0}, {1, 1, 0}},   //32F
                                   {{1, 1, 0}, {0, 0, 0}, {1, 1, 0}, {1, 1, 0}}};  //64F
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation
                                     /* C1         C2         C3         C4 */
    char impl[CV_DEPTH_MAX][4][3]={{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8U
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8S
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}},   //16U
                                   {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}},   //16S
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32S
                                   {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32F
                                   {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}}};  //64F
#endif

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

        int min_payload = 1 << 16; // 64KB shall be minimal per thread to maximize scalability for warping functions
        const int threads = ippiSuggestRowThreadsNum(iwDst, min_payload);

        if (threads > 1)
        {
            bool ok = true;
            cv::Range range(0, (int)iwDst.m_size.height);
            ipp_warpAffineParallel invoker(iwSrc, iwDst, ippInter, coeffs, ippBorder, iwTransDirection, &ok);
            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;

            parallel_for_(range, invoker, threads);

            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        else
        {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, iwSrc, iwDst, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpAffineParams(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#endif // HAVE_IPP_IW

// End of Warp affine section

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

// Warp perspective section

typedef IppStatus (CV_STDCALL* ippiWarpPerspectiveFunc)(const Ipp8u*, int, Ipp8u*, int,IppiPoint, IppiSize, const IppiWarpSpec*,Ipp8u*);
typedef IppStatus (CV_STDCALL* ippiWarpPerspectiveInitFunc)(IppiSize, IppiRect, IppiSize, IppDataType,const double [3][3], IppiWarpDirection, int, IppiBorderType, const Ipp64f *, int, IppiWarpSpec*);

class IPPWarpPerspectiveInvoker :
    public cv::ParallelLoopBody
{
    // Mem object ot simplify IPP memory lifetime control
    struct IPPWarpPerspectiveMem
    {
        IppiWarpSpec* pSpec = nullptr;
        Ipp8u* pBuffer  = nullptr;

        IPPWarpPerspectiveMem() = default;
        IPPWarpPerspectiveMem (const IPPWarpPerspectiveMem&) = delete;
        IPPWarpPerspectiveMem& operator= (const IPPWarpPerspectiveMem&) = delete;

        void AllocateSpec(int size)
        {
            pSpec = (IppiWarpSpec*)ippMalloc_L(size);
        }

        void AllocateBuffer(int size)
        {
            pBuffer = (Ipp8u*)ippMalloc_L(size);
        }

        ~IPPWarpPerspectiveMem()
        {
            if (nullptr != pSpec) ippFree(pSpec);
            if (nullptr != pBuffer) ippFree(pBuffer);
        }
    };
public:
    IPPWarpPerspectiveInvoker(int _src_type, cv::Mat &_src, size_t _src_step, cv::Mat &_dst, size_t _dst_step, IppiInterpolationType _interpolation,
                              double (&_coeffs)[3][3], int &_borderType, const double _borderValue[4], ippiWarpPerspectiveFunc _func, ippiWarpPerspectiveInitFunc _initFunc,
                              bool *_ok) :
        ParallelLoopBody(), src_type(_src_type), src(_src), src_step(_src_step), dst(_dst), dst_step(_dst_step), inter(_interpolation), coeffs(_coeffs),
        borderType(_borderType), func(_func), initFunc(_initFunc), ok(_ok)
    {
        memcpy(this->borderValue, _borderValue, sizeof(this->borderValue));
        *ok = true;
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        //CV_INSTRUMENT_REGION_IPP();
        if (*ok == false)
            return;

        IPPWarpPerspectiveMem mem;

        int specSize = 0, initSize = 0, bufSize = 0;

        IppiWarpDirection direction = ippWarpBackward;  //fixed for IPP
        const Ipp32u numChannels = CV_MAT_CN(src_type);

        IppiSize srcsize = {src.cols, src.rows};
        IppiSize dstsize = {dst.cols, dst.rows};
        IppiRect srcroi = {0, 0, src.cols, src.rows};

        /* Spec and init buffer sizes */
        CV_IPP_SAFE_CALL(ippiWarpPerspectiveGetSize, ok, srcsize, srcroi, dstsize, ippiGetDataType(src_type), coeffs, inter, ippWarpBackward, ippiGetBorderType(borderType), &specSize, &initSize);

        mem.AllocateSpec(specSize);

        CV_IPP_SAFE_CALL(initFunc, ok, srcsize, srcroi, dstsize, ippiGetDataType(src_type), coeffs, direction, numChannels, ippiGetBorderType(borderType),
                                            borderValue, 0, mem.pSpec);

        CV_IPP_SAFE_CALL(ippiWarpGetBufferSize, ok, mem.pSpec, dstsize, &bufSize);

        mem.AllocateBuffer(bufSize);

        IppiPoint dstRoiOffset = {0, range.start};
        IppiSize dstRoiSize = {dst.cols, range.size()};
        auto* pDst = dst.ptr(range.start);
        if (borderType == cv::BorderTypes::BORDER_CONSTANT &&
            !IPPSet(borderValue, pDst, (int)dst_step, dstRoiSize, src.channels(), src.depth()))
        {
            *ok = false;
            return;
        }

        if (ippStsNoErr != CV_INSTRUMENT_FUN_IPP(func, src.ptr(), (int)src_step, pDst, (int)dst_step, dstRoiOffset, dstRoiSize, mem.pSpec, mem.pBuffer))
        {
            *ok = false;
            return;
        }

        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
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
    ippiWarpPerspectiveInitFunc initFunc;
    bool *ok;
    const IPPWarpPerspectiveInvoker& operator= (const IPPWarpPerspectiveInvoker&);
};

int ipp_hal_warpPerspective(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar * dst_data, size_t dst_step,
                            int dst_width, int dst_height, const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    //CV_INSTRUMENT_REGION_IPP();

    if (src_height <= 1 || src_width <= 1)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    ippiWarpPerspectiveFunc ippFunc = nullptr;
    ippiWarpPerspectiveInitFunc ippInitFunc = nullptr;

    if (interpolation == cv::InterpolationFlags::INTER_NEAREST)
    {
        ippInitFunc = ippiWarpPerspectiveNearestInit;
        ippFunc =
            src_type == CV_8UC1  ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_8u_C1R :
            src_type == CV_8UC3  ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_8u_C3R :
            src_type == CV_8UC4  ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_8u_C4R :
            src_type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16u_C1R :
            src_type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16u_C3R :
            src_type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16u_C4R :
            src_type == CV_16SC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16s_C1R :
            src_type == CV_16SC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16s_C3R :
            src_type == CV_16SC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_16s_C4R :
            src_type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_32f_C1R :
            src_type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_32f_C3R :
            src_type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveNearest_32f_C4R : nullptr;
    }
    else if (interpolation == cv::InterpolationFlags::INTER_LINEAR)
    {
        ippInitFunc = ippiWarpPerspectiveLinearInit;
        ippFunc =
            src_type == CV_8UC1  ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_8u_C1R :
            src_type == CV_8UC3  ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_8u_C3R :
            src_type == CV_8UC4  ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_8u_C4R :
            src_type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16u_C1R :
            src_type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16u_C3R :
            src_type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16u_C4R :
            src_type == CV_16SC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16s_C1R :
            src_type == CV_16SC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16s_C3R :
            src_type == CV_16SC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_16s_C4R :
            src_type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_32f_C1R :
            src_type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_32f_C3R :
            src_type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveLinear_32f_C4R : nullptr;
    }
    else
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if (ippFunc == nullptr)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

#if defined(IPP_CALLS_ENFORCED)
                                    /* C1      C2      C3      C4 */
    char impl[CV_DEPTH_MAX][4][2]={{{1, 1}, {0, 0}, {1, 1}, {1, 1}},   //8U
                                   {{0, 0}, {0, 0}, {0, 0}, {0, 0}},   //8S
                                   {{1, 1}, {0, 0}, {1, 1}, {1, 1}},   //16U
                                   {{1, 1}, {0, 0}, {1, 1}, {1, 1}},   //16S
                                   {{1, 1}, {0, 0}, {1, 1}, {1, 1}},   //32S
                                   {{1, 1}, {0, 0}, {1, 1}, {1, 1}},   //32F
                                   {{0, 0}, {0, 0}, {0, 0}, {0, 0}}};  //64F
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation
                                    /* C1      C2      C3      C4 */
    char impl[CV_DEPTH_MAX][4][2]={{{0, 0}, {0, 0}, {0, 0}, {0, 0}},   //8U
                                   {{0, 0}, {0, 0}, {0, 0}, {0, 0}},   //8S
                                   {{0, 0}, {0, 0}, {0, 1}, {0, 1}},   //16U
                                   {{1, 1}, {0, 0}, {1, 1}, {1, 1}},   //16S
                                   {{1, 1}, {0, 0}, {1, 0}, {1, 1}},   //32S
                                   {{1, 0}, {0, 0}, {0, 0}, {1, 0}},   //32F
                                   {{0, 0}, {0, 0}, {0, 0}, {0, 0}}};  //64F
#endif

    const char type_size[CV_DEPTH_MAX] = {1,1,2,2,4,4,8};
    if(impl[CV_TYPE(src_type)][CV_MAT_CN(src_type)-1][interpolation] == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    double coeffs[3][3];
    for( int i = 0; i < 3; i++ )
        for( int j = 0; j < 3; j++ )
            coeffs[i][j] = M[i*3 + j];

    bool ok = true;
    cv::Range range(0, dst_height);
    cv::Mat src(cv::Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    cv::Mat dst(cv::Size(dst_width, dst_height), src_type, dst_data, dst_step);
    IppiInterpolationType ippInter = ippiGetInterpolation(interpolation);

    int min_payload = 1 << 16; // 64KB shall be minimal per thread to maximize scalability for warping functions
    int num_threads = ippiSuggestRowThreadsNum(dst_width, dst_height, type_size[CV_TYPE(src_type)]*CV_MAT_CN(src_type), min_payload);

    IPPWarpPerspectiveInvoker invoker(src_type, src, src_step, dst, dst_step, ippInter, coeffs, borderType, borderValue, ippFunc, ippInitFunc, &ok);

    (num_threads > 1) ? parallel_for_(range, invoker, num_threads) : invoker(range);

    if (ok)
    {
        CV_IMPL_ADD(CV_IMPL_IPP | CV_IMPL_MT);
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// End of Warp perspective section

// Remap section

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
        //CV_INSTRUMENT_REGION_IPP();
        if (*ok == false)
            return;

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

        if (ippStsNoErr != CV_INSTRUMENT_FUN_IPP(ippFunc, src, {src_width, src_height}, (int)src_step, srcRoiRect,
                                  mapx, (int)mapx_step, mapy, (int)mapy_step,
                                  dst_roi_data, (int)dst_step, dstRoiSize, ippInterpolation))
        {
            *ok = false;
            return;
        }

        CV_IMPL_ADD(CV_IMPL_IPP | CV_IMPL_MT);
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
    if (!((interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC || interpolation == cv::INTER_NEAREST) &&
        (border_type == cv::BORDER_CONSTANT || border_type == cv::BORDER_TRANSPARENT)))
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    int ippInterpolation = ippiGetInterpolation(interpolation);

#if defined(IPP_CALLS_ENFORCED)
                                    /* C1         C2         C3         C4 */
    char impl[CV_DEPTH_MAX][4][3] = {{{1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},   //8U
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8S
                                     {{1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},   //16U
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //16S
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32S
                                     {{1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},   //32F
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};  //64F
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation
                                    /* C1         C2         C3         C4 */
    char impl[CV_DEPTH_MAX][4][3] = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8U
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8S
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //16U
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //16S
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32S
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32F
                                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};  //64F
#endif
    const char type_size[CV_DEPTH_MAX] = {1,1,2,2,4,4,8};

    if (impl[CV_TYPE(src_type)][CV_MAT_CN(src_type) - 1][interpolation] == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    ippiRemap ippFunc =
        src_type == CV_8UC1  ? (ippiRemap)ippiRemap_8u_C1R :
        src_type == CV_8UC3  ? (ippiRemap)ippiRemap_8u_C3R :
        src_type == CV_8UC4  ? (ippiRemap)ippiRemap_8u_C4R :
        src_type == CV_16UC1 ? (ippiRemap)ippiRemap_16u_C1R :
        src_type == CV_16UC3 ? (ippiRemap)ippiRemap_16u_C3R :
        src_type == CV_16UC4 ? (ippiRemap)ippiRemap_16u_C4R :
        src_type == CV_32FC1 ? (ippiRemap)ippiRemap_32f_C1R :
        src_type == CV_32FC3 ? (ippiRemap)ippiRemap_32f_C3R :
        src_type == CV_32FC4 ? (ippiRemap)ippiRemap_32f_C4R : 0;

    if (ippFunc)
    {
        bool ok = true;

        IPPRemapInvoker invoker(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width,
                                mapx, mapx_step, mapy, mapy_step, ippFunc, ippInterpolation, border_type, border_value, &ok);
        cv::Range range(0, dst_height);

        int min_payload = 1 << 16; // 64KB shall be minimal per thread to maximize scalability for warping functions
        int num_threads = ippiSuggestRowThreadsNum(dst_width, dst_height, type_size[CV_TYPE(src_type)]*CV_MAT_CN(src_type), min_payload);

        cv::parallel_for_(range, invoker, num_threads);

        if (ok)
        {
            CV_IMPL_ADD(CV_IMPL_IPP | CV_IMPL_MT);
            return CV_HAL_ERROR_OK;
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// End of Remap section

#endif // IPP_VERSION_X100 >= 810
