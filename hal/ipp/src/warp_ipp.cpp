// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_imgproc.hpp"

#if IPP_VERSION_X100 >= 810 // integrated IPP warping/remap ABI is available since IPP v8.1

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"
#include <atomic>

// Uncomment to enforce IPP calls for all supported by IPP configurations
// #define IPP_CALLS_ENFORCED

#define CV_TYPE(src_type) (src_type & (CV_DEPTH_MAX - 1))

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

#ifdef HAVE_IPP_IW

#include "iw++/iw.hpp"

int ipp_hal_warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step,
                       int dst_width, int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4])
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
    double coeffs[2][3];
    for( int i = 0; i < 2; i++ )
        for( int j = 0; j < 3; j++ )
            coeffs[i][j] = M[i*3 + j];

    try
    {
        std::atomic_bool  ok{true};
        cv::Range cv_range(0, dst_height);
        ::ipp::IwiImage        iwSrc;
        iwSrc.Init(IwiSize{src_width, src_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), IwiBorderSize(), src_data, IwSize(src_step));
        ::ipp::IwiImage        iwDst(IwiSize{dst_width, dst_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), IwiBorderSize(), dst_data, IwSize(dst_step));
        ::ipp::IwiBorderType   ippBorder(ippiGetBorderType(borderType), {borderValue, 4});
        // OpenCV inverts the affine matrix before calling the HAL (lines 2401-2411 of imgwarp.cpp), so the HAL receives the inverse transform.
        IwTransDirection       iwTransDirection = iwTransInverse;

        if ((int)ippBorder == -1)
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        // The lambda function is used to invoke IPP warping function in parallel for different image stripes.
        // The function is exception safe and sets the 'ok' flag to false if any exception occurs during processing.
        // The 'ok' flag is checked before and after parallel processing to determine if the operation was successful or
        // if it should fall back to a non-IPP implementation.
        auto IPPWarpAffineInvokerLambda = [&iwSrc, &iwDst, dst_width, ippInter, &coeffs, ippBorder, iwTransDirection, &ok](const cv::Range& range)
        {
            //CV_INSTRUMENT_REGION_IPP();
            if (!ok.load(std::memory_order_relaxed))
            {
                return;
            }

            try
            {
                ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, dst_width, range.end - range.start);
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, iwSrc, iwDst, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpAffineParams(), ippBorder, tile);
            }
            catch (const ::ipp::IwException &)
            {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        };

        int min_payload = 1 << 16; // 64KB shall be minimal per thread to maximize scalability for warping functions
        const int num_threads = ippiSuggestRowThreadsNum(iwDst, min_payload);

        if (num_threads > 1)
        {
            parallel_for_(cv_range, IPPWarpAffineInvokerLambda, num_threads);
        }
        else
        {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, iwSrc, iwDst, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpAffineParams(), ippBorder);
        }

        if (!ok)
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#if IPP_VERSION_X100 >= 202600

int ipp_hal_warpPerspective(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar * dst_data, size_t dst_step,
                            int dst_width, int dst_height, const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    //CV_INSTRUMENT_REGION_IPP();

    IppiInterpolationType ippInter = ippiGetInterpolation(interpolation);

    if (src_height <= 1 || src_width <= 1)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    int mode =
    interpolation == cv::InterpolationFlags::INTER_NEAREST ? IPPI_INTER_NN :
    interpolation == cv::InterpolationFlags::INTER_LINEAR ? IPPI_INTER_LINEAR : 0;

    if (mode == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Unsupported source type
    if (src_type !=  CV_8UC1 && src_type !=  CV_8UC3 && src_type !=  CV_8UC4 &&
        src_type != CV_16UC1 && src_type != CV_16UC3 && src_type != CV_16UC4 &&
        src_type != CV_16SC1 && src_type != CV_16SC3 && src_type != CV_16SC4 &&
        src_type != CV_32FC1 && src_type != CV_32FC3 && src_type != CV_32FC4)
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
                                   {{0, 0}, {0, 0}, {0, 0}, {0, 1}},   //16U
                                   {{1, 1}, {0, 0}, {1, 0}, {1, 1}},   //16S
                                   {{1, 1}, {0, 0}, {1, 0}, {1, 1}},   //32S
                                   {{0, 0}, {0, 0}, {0, 0}, {1, 0}},   //32F
                                   {{0, 0}, {0, 0}, {0, 0}, {0, 0}}};  //64F
#endif

    if (impl[CV_TYPE(src_type)][CV_MAT_CN(src_type)-1][interpolation] == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Acquire data and begin processing
    double coeffs[3][3];
    for( int i = 0; i < 3; i++ )
        for( int j = 0; j < 3; j++ )
            coeffs[i][j] = M[i*3 + j];

    try
    {
        std::atomic_bool ok{true};
        cv::Range cv_range(0, dst_height);
        ::ipp::IwiImage iwSrc; // src_data is const pointer. So, we need to call an init function
        iwSrc.Init(IwiSize{src_width, src_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), IwiBorderSize(), src_data, IwSize(src_step));
        ::ipp::IwiImage iwDst(IwiSize{dst_width, dst_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), IwiBorderSize(), dst_data, IwSize(dst_step));
        ::ipp::IwiBorderType   ippBorder(ippiGetBorderType(borderType), {borderValue, 4});
        IwTransDirection iwTransDirection = iwTransInverse;  //fixed for IPP
        if ((int)ippBorder == -1)
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        // The lambda function is used to invoke IPP warping function in parallel for different image stripes.
        // The function is exception safe and sets the 'ok' flag to false if any exception occurs during processing.
        // The 'ok' flag is checked before and after parallel processing to determine
        // if the operation was successful or if it should fall back to a non-IPP implementation.
        auto IPPWarpPerspectiveInvokerLambda = [&iwSrc, &iwDst, dst_width, ippInter, &coeffs, ippBorder, iwTransDirection, &ok](const cv::Range& range)
        {
            //CV_INSTRUMENT_REGION_IPP();
            if (!ok.load(std::memory_order_relaxed))
            {
                return;
            }

            try
            {
                ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, dst_width, range.end - range.start);
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpPerspective, iwSrc, iwDst, ippRectInfinite, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpPerspectiveParams(), ippBorder, tile);
            }
            catch (const ::ipp::IwException &)
            {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        };

        int min_payload = 1 << 16; // 64KB shall be minimal per thread to maximize scalability for warping functions
        const char type_size[CV_DEPTH_MAX] = {1,1,2,2,4,4,8};
        const int num_threads = ippiSuggestRowThreadsNum(dst_width, dst_height, type_size[CV_TYPE(src_type)]*CV_MAT_CN(src_type), min_payload);

        if (num_threads > 1)
        {
            parallel_for_(cv_range, IPPWarpPerspectiveInvokerLambda, num_threads);
        }
        else
        {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpPerspective, iwSrc, iwDst, ippRectInfinite, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpPerspectiveParams(), ippBorder);
        }

        if (!ok)
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}
#endif // IPP_VERSION_X100 >= 202600

#endif // HAVE_IPP_IW

// Remap section
typedef IppStatus(CV_STDCALL *ippiRemap)(const void *pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi,
                                         const Ipp32f *pxMap, int xMapStep, const Ipp32f *pyMap, int yMapStep,
                                         void *pDst, int dstStep, IppiSize dstRoiSize, int interpolation);

class IPPRemapInvoker : public cv::ParallelLoopBody
{
public:
    IPPRemapInvoker(int _src_type, const uchar *_src_data, size_t _src_step, int _src_width, int _src_height,
                    uchar *_dst_data, size_t _dst_step, int _dst_width, float *_mapx, size_t _mapx_step, float *_mapy,
                    size_t _mapy_step, ippiRemap _ippFunc, int _ippInterpolation, int _borderType, const double _borderValue[4], std::atomic_bool *_ok) :
        src_type(_src_type), src(_src_data), src_step(_src_step), src_width(_src_width), src_height(_src_height),
        dst(_dst_data), dst_step(_dst_step), dst_width(_dst_width), mapx(_mapx), mapx_step(_mapx_step), mapy(_mapy),
        mapy_step(_mapy_step), ippFunc(_ippFunc), ippInterpolation(_ippInterpolation), borderType(_borderType), ok(_ok)
    {
        memcpy(this->borderValue, _borderValue, sizeof(this->borderValue));
    }

    virtual void operator()(const cv::Range &range) const
    {
        //CV_INSTRUMENT_REGION_IPP();
        if(!ok->load(std::memory_order_relaxed))
            return;

        IppiRect srcRoiRect = {0, 0, src_width, src_height};
        uchar *dst_roi_data = dst + range.start * dst_step;
        IppiSize dstRoiSize = ippiSize(dst_width, range.size());
        int depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);

        if (borderType == cv::BORDER_CONSTANT &&
            !IPPSet(borderValue, dst_roi_data, (int)dst_step, dstRoiSize, cn, depth))
        {
            ok->store(false, std::memory_order_relaxed);
            return;
        }

        if (ippStsNoErr != CV_INSTRUMENT_FUN_IPP(ippFunc, src, {src_width, src_height}, (int)src_step, srcRoiRect,
                                  mapx, (int)mapx_step, mapy, (int)mapy_step,
                                  dst_roi_data, (int)dst_step, dstRoiSize, ippInterpolation))
        {
            ok->store(false, std::memory_order_relaxed);
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
    std::atomic_bool *ok;
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
        std::atomic_bool ok{true};

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
