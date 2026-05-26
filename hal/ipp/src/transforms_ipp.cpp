// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_core.hpp"
#include "precomp_ipp.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#ifdef HAVE_IPP_IW
#include "iw++/iw.hpp"
#endif

// HACK: Should be removed, when IPP management moved to HAL
namespace cv
{
    namespace ipp
    {
        unsigned long long getIppTopFeatures(); // Returns top major enabled IPP feature flag
    }
}

//bool ipp_transpose( Mat &src, Mat &dst )
int ipp_hal_transpose2d(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int src_width,
                        int src_height, int element_size)
{
    typedef IppStatus (CV_STDCALL * IppiTranspose)(const void * pSrc, int srcStep, void * pDst, int dstStep, IppiSize roiSize);
    typedef IppStatus (CV_STDCALL * IppiTransposeI)(const void * pSrcDst, int srcDstStep, IppiSize roiSize);
    IppiTranspose ippiTranspose = nullptr;
    IppiTransposeI ippiTranspose_I = nullptr;

    if (dst_data == src_data && src_width == src_height)
    {
        CV_SUPPRESS_DEPRECATED_START
        ippiTranspose_I =
            element_size == 1*sizeof(char) ? (IppiTransposeI)ippiTranspose_8u_C1IR :
            element_size == 3*sizeof(char) ? (IppiTransposeI)ippiTranspose_8u_C3IR :
            element_size == 1*sizeof(short) ? (IppiTransposeI)ippiTranspose_16u_C1IR :
            element_size == 4*sizeof(char) ? (IppiTransposeI)ippiTranspose_8u_C4IR :
            element_size == 3*sizeof(short) ? (IppiTransposeI)ippiTranspose_16u_C3IR :
            element_size == 4*sizeof(short) ? (IppiTransposeI)ippiTranspose_16u_C4IR :
            element_size == 3*sizeof(int) ? (IppiTransposeI)ippiTranspose_32s_C3IR :
            element_size == 4*sizeof(int) ? (IppiTransposeI)ippiTranspose_32s_C4IR : 0;
        CV_SUPPRESS_DEPRECATED_END
    }
    else
    {
        ippiTranspose =
            element_size == 1*sizeof(char) ? (IppiTranspose)ippiTranspose_8u_C1R :
            element_size == 3*sizeof(char) ? (IppiTranspose)ippiTranspose_8u_C3R :
            element_size == 4*sizeof(char) ? (IppiTranspose)ippiTranspose_8u_C4R :
            element_size == 1*sizeof(short) ? (IppiTranspose)ippiTranspose_16u_C1R :
            element_size == 3*sizeof(short) ? (IppiTranspose)ippiTranspose_16u_C3R :
            element_size == 4*sizeof(short) ? (IppiTranspose)ippiTranspose_16u_C4R :
            element_size == 1*sizeof(int) ? (IppiTranspose)ippiTranspose_32s_C1R :
            element_size == 3*sizeof(int) ? (IppiTranspose)ippiTranspose_32s_C3R :
            element_size == 4*sizeof(int) ? (IppiTranspose)ippiTranspose_32s_C4R : 0;
    }

    IppiSize roiSize = { src_width, src_height };
    if (ippiTranspose != 0)
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiTranspose, src_data, (int)src_step, dst_data, (int)dst_step, roiSize) >= 0)
            return CV_HAL_ERROR_OK;
    }
    else if (ippiTranspose_I != 0)
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiTranspose_I, dst_data, (int)dst_step, roiSize) >= 0)
            return CV_HAL_ERROR_OK;
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#ifdef HAVE_IPP_IW

static inline ::ipp::IwiImage ippiGetImage(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height)
{
    ::ipp::IwiImage dst;
    ::ipp::IwiBorderSize inMemBorder;
//     if(src.isSubmatrix()) // already have physical border
//     {
//         cv::Size  origSize;
//         cv::Point offset;
//         src.locateROI(origSize, offset);
//
//         inMemBorder.left   = (IwSize)offset.x;
//         inMemBorder.top    = (IwSize)offset.y;
//         inMemBorder.right  = (IwSize)(origSize.width - src.cols - offset.x);
//         inMemBorder.bottom = (IwSize)(origSize.height - src.rows - offset.y);
//     }

    dst.Init({src_width, src_height}, ippiGetDataType(CV_MAT_DEPTH(src_type)),
             CV_MAT_CN(src_type), inMemBorder, (void*)src_data, src_step);

    return dst;
}

int ipp_hal_flip(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
                 uchar* dst_data, size_t dst_step, int flip_mode)

{
    int64_t total = src_step*src_height*CV_ELEM_SIZE(src_type);
    // Details: https://github.com/opencv/opencv/issues/12943
    if (flip_mode <= 0 /* swap rows */
        && total > 0 && total >= CV_BIG_INT(0x80000000)/*2Gb*/
        && cv::ipp::getIppTopFeatures() != ippCPUID_SSE42)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    IppiAxis ippMode;
    if(flip_mode < 0)
        ippMode = ippAxsBoth;
    else if(flip_mode == 0)
        ippMode = ippAxsHorizontal;
    else
        ippMode = ippAxsVertical;

    try
    {
        ::ipp::IwiImage iwSrc = ippiGetImage(src_type, src_data, src_step, src_width, src_height);
        ::ipp::IwiImage iwDst = ippiGetImage(src_type, dst_data, dst_step, src_width, src_height);

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiMirror, iwSrc, iwDst, ippMode);
    }
    catch(const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#if IPP_VERSION_X100 >= 202600
#include <atomic>
#include "ipp_hal_imgproc.hpp"

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
    // Unsupported source type
    if (src_type !=  CV_8UC1 && src_type !=  CV_8UC3 && src_type !=  CV_8UC4 &&
        src_type != CV_16UC1 && src_type != CV_16UC3 && src_type != CV_16UC4 &&
        src_type != CV_32FC1 && src_type != CV_32FC3 && src_type != CV_32FC4)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

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

    if (impl[CV_TYPE(src_type)][CV_MAT_CN(src_type)-1][interpolation] == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    try
    {
        std::atomic_bool ok{true};
        cv::Range cv_range(0, dst_height);

        ::ipp::IwiImage iwSrc, iwMapx, iwMapy; // There are const pointers. So, we need to call an init function
        iwSrc.Init( IwiSize{src_width, src_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), IwiBorderSize(), src_data, IwSize(src_step));
        iwMapx.Init(IwiSize{dst_width, dst_height}, ippiGetDataType(CV_32FC1), 1, IwiBorderSize(), mapx, IwSize(mapx_step));
        iwMapy.Init(IwiSize{dst_width, dst_height}, ippiGetDataType(CV_32FC1), 1, IwiBorderSize(), mapy, IwSize(mapy_step));
        ::ipp::IwiImage iwDst(IwiSize{dst_width, dst_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), IwiBorderSize(), dst_data, IwSize(dst_step));

        auto IPPRemapInvokerLambda = [dst_data, border_type, border_value, src_type,
            &iwSrc, &iwDst, &iwMapx, &iwMapy, ippInterpolation, &ok](const cv::Range& range)
        {
            //CV_INSTRUMENT_REGION_IPP();
            if (!ok.load(std::memory_order_relaxed))
                return;

            try
            {
                IppiSize dstRoiSize = ippiSize(iwDst.m_size.width, range.size());
                IppiRect srcRoiRect = {0, 0, (int)iwSrc.m_size.width, (int)iwSrc.m_size.height};
                uchar *dst_roi_data = dst_data + range.start * iwDst.m_step;
                int depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);

                if (border_type == cv::BORDER_CONSTANT &&
                    !IPPSet(border_value, dst_roi_data, iwDst.m_step, dstRoiSize, cn, depth))
                {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, iwDst.m_size.width, range.end - range.start);
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiRemap, iwSrc, iwDst, srcRoiRect, iwMapx, iwMapy, dstRoiSize, ippInterpolation, tile);
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

        cv::parallel_for_(cv_range, IPPRemapInvokerLambda, num_threads);

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
