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

#endif
