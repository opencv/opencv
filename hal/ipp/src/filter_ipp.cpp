// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_imgproc.hpp"

#if IPP_VERSION_X100 >= 810

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>

#if defined(HAVE_IPP_IW)

#include "iw++/iw.hpp"

static inline IppiBorderType ippiFilterGetBorderType(int borderTypeNI)
{
    return borderTypeNI == cv::BORDER_CONSTANT    ? ippBorderConst  :
           borderTypeNI == cv::BORDER_TRANSPARENT ? ippBorderTransp :
           borderTypeNI == cv::BORDER_REPLICATE   ? ippBorderRepl   :
           borderTypeNI == cv::BORDER_REFLECT_101 ? ippBorderMirror :
           (IppiBorderType)-1;
}

static inline IppiBorderType ippiGetBorder(::ipp::IwiImage &image, int ocvBorderType, ::ipp::IwiBorderSize &borderSize)
{
    int            inMemFlags = 0;
    IppiBorderType border     = ippiFilterGetBorderType(ocvBorderType & ~cv::BORDER_ISOLATED);
    if((int)border == -1)
        return (IppiBorderType)0;

    if(!(ocvBorderType & cv::BORDER_ISOLATED))
    {
        if(image.m_inMemSize.left)
        {
            if(image.m_inMemSize.left >= borderSize.left)
                inMemFlags |= ippBorderInMemLeft;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.left = 0;
        if(image.m_inMemSize.top)
        {
            if(image.m_inMemSize.top >= borderSize.top)
                inMemFlags |= ippBorderInMemTop;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.top = 0;
        if(image.m_inMemSize.right)
        {
            if(image.m_inMemSize.right >= borderSize.right)
                inMemFlags |= ippBorderInMemRight;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.right = 0;
        if(image.m_inMemSize.bottom)
        {
            if(image.m_inMemSize.bottom >= borderSize.bottom)
                inMemFlags |= ippBorderInMemBottom;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.bottom = 0;
    }
    else
        borderSize.left = borderSize.right = borderSize.top = borderSize.bottom = 0;

    return (IppiBorderType)(border | inMemFlags);
}

int ipp_hal_filter2D(const uchar * src_data, size_t src_step, int src_type,
                     uchar * dst_data, size_t dst_step, int dst_type,
                     int width, int height, int full_width, int full_height,
                     int offset_x, int offset_y,
                     const uchar * kernel_data, size_t kernel_step, int kernel_type,
                     int kernel_width, int kernel_height,
                     int anchor_x, int anchor_y, double delta, int borderType,
                     bool isSubmatrix, bool allowInplace)
{
    CV_HAL_CHECK_USE_IPP();

    CV_UNUSED(isSubmatrix);
    CV_UNUSED(allowInplace); // in-place calls are handled by filtering from a copy of the source

    const int depth    = CV_MAT_DEPTH(src_type);
    const int channels = CV_MAT_CN(src_type);
    const int kdepth   = CV_MAT_DEPTH(kernel_type);

    // ippiFilterBorder has no type-converting variants
    if(src_type != dst_type)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if(depth >= CV_DEPTH_MAX || channels < 1 || channels > 4 || CV_MAT_CN(kernel_type) != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if(kernel_width <= 0 || kernel_height <= 0 ||
       anchor_x < 0 || anchor_x >= kernel_width || anchor_y < 0 || anchor_y >= kernel_height)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if(dst_step > (size_t)INT_MAX)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Supported (depth x channels x kernel type) combinations.
    // ippiFilterBorder provides C1/C3/C4 for 8u, 16u, 16s and 32f; a 16s kernel cannot be used with 32f data
    // (ippiFilterBorderInit_16s returns an error). Checked against ICV and full IPP 2026.1.0.
    // Full IPP also has ippiFilterBorder_64f_C1R (64f kernel only), but it is not in ICV and
    // iwiFilter does not dispatch it.
    // Column "16S": a 16S kernel is passed to IPP as is (integer arithmetic).
    // Column "32F": the kernel is passed as 32F. Kernels of any other depth (64F, 8U, 32S, ...) and 16S kernels
    // whose "16S" entry is 0 are converted to 32F, which is what OpenCV itself does (getLinearFilter).
#if defined(IPP_CALLS_ENFORCED)
                                                   /*  kernel: 16S 32F */
    static const char impl[CV_DEPTH_MAX][4][2] = { /* C1     C2     C3     C4   */
        /* 8U  */ {{1, 1}, {0, 0}, {1, 1}, {1, 1}},
        /* 8S  */ {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        /* 16U */ {{1, 1}, {0, 0}, {1, 1}, {1, 1}},
        /* 16S */ {{1, 1}, {0, 0}, {1, 1}, {1, 1}},
        /* 32S */ {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        /* 32F */ {{0, 1}, {0, 0}, {0, 1}, {0, 1}},
        /* 64F */ {{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation
                                                   /*  kernel: 16S 32F */
    static const char impl[CV_DEPTH_MAX][4][2] = { /* C1     C2     C3     C4   */
        /* 8U  */ {{1, 1}, {0, 0}, {1, 1}, {0, 1}},
        /* 8S  */ {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        /* 16U */ {{1, 1}, {0, 0}, {1, 1}, {1, 1}},
        /* 16S */ {{1, 1}, {0, 0}, {1, 1}, {1, 1}},
        /* 32S */ {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        /* 32F */ {{0, 1}, {0, 0}, {0, 0}, {0, 0}},   // 32FC3/C4: small gain, slower with zeros in the kernel
        /* 64F */ {{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
#endif // IPP_CALLS_ENFORCED

    bool useKernel16s;
    if(kdepth == CV_16S && impl[depth][channels - 1][0])
        useKernel16s = true;
    else if(impl[depth][channels - 1][1])
        useKernel16s = false;
    else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // IPP supports only centered anchors: an off-center anchor (or an even kernel size) is handled by
    // zero-padding the kernel so that the anchor becomes its center. The extra taps have zero weight.
    const int radius_x = std::max(anchor_x, kernel_width  - 1 - anchor_x);
    const int radius_y = std::max(anchor_y, kernel_height - 1 - anchor_y);
    const int ipp_kernel_width  = 2 * radius_x + 1;
    const int ipp_kernel_height = 2 * radius_y + 1;

    // delta: 8u/16u apply an integer offset before the final conversion (same saturation as OpenCV),
    // 32f adds it after filtering, 16s has no offset support
    int  ippOffset = 0;
    bool addDelta  = false;
    if(delta != 0)
    {
        if((depth == CV_8U || depth == CV_16U) && std::fabs(delta) <= (double)(1 << 30) && delta == (double)cvRound(delta))
            ippOffset = cvRound(delta);
        else if(depth == CV_32F)
            addDelta = true;
        else
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

#if !defined(IPP_CALLS_ENFORCED)
    // OpenCV switches to FFT-based dftFilter2D for big kernels on whole (non-ROI) images. That path is
    // multi-threaded and much faster there.
    // Results of IPP match it within the usual tolerance (checked with WITH_IPP_CALLS_ENFORCED).
    // Mirrors the threshold in dftFilter2D() (modules/imgproc/src/filter.dispatch.cpp, keep in sync), original kernel size.
    {
        const bool isROI = offset_x != 0 || offset_y != 0 || width != full_width || height != full_height;
        const int  dftFilterSize = cv::checkHardwareSupport(CV_CPU_SSE3) && (depth == CV_8U || depth == CV_32F) ? 130 : 50;
        if(!isROI && kernel_width * kernel_height >= dftFilterSize)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Performance: the cases below are slower with IPP than with OpenCV.
    {
        const int ippKernelArea = ipp_kernel_width * ipp_kernel_height;

        // 32F 3x3 with an extra full-image pass (copy of the source for in-place calls, adding delta)
        if(depth == CV_32F && ippKernelArea <= 9 && (src_data == dst_data || addDelta))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        // Sparse kernels (sharpen, Laplacian, cross shapes) and zero-padded kernels (off-center anchors, even
        // kernel sizes): IPP processes every coefficient of the padded kernel, OpenCV skips zero coefficients.
        // 8U/32F with a kernel passed as 32F and less than 60% non-zero coefficients in the padded kernel are
        // slower; 32F with a 3x3 kernel is slower with any zero coefficient. The native 16S path, 16U and 16S are
        // faster with any kernel.
        if(!useKernel16s && (depth == CV_8U || depth == CV_32F))
        {
            const int nonZero = cv::countNonZero(cv::Mat(kernel_height, kernel_width, kernel_type, (void*)kernel_data, kernel_step));
            if((depth == CV_32F && ippKernelArea <= 9) ? nonZero < ippKernelArea : nonZero < 0.6 * ippKernelArea)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }

    // ippBorderTransp does not reproduce OpenCV behavior
    if((borderType & ~cv::BORDER_ISOLATED) == cv::BORDER_TRANSPARENT)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif // IPP_CALLS_ENFORCED

    // OpenCV rounds with cvRound (round half to even), which is ippRndNear.
    // ippRndFinancial (round half away from zero) gives +-1 differences on exact .5 results.
    const IppRoundMode roundMode = ippRndNear;

    cv::Mat  srcCopy;      // in-place calls only: copy of the source ROI and the margins IPP reads
    cv::Rect srcCopyRoi;   // the source ROI inside srcCopy
    // Any failure after IPP may have written dst: for in-place calls dst (= src) is restored from the copy,
    // so OpenCV can always recompute the result from a valid source
    auto fallback = [&]() -> int
    {
        if(!srcCopy.empty())
            srcCopy(srcCopyRoi).copyTo(cv::Mat(height, width, src_type, dst_data, dst_step));
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    };

    try
    {
        // Kernel for IPP: always a continuous, centered copy (16S as is, anything else converted to 32F)
        cv::Mat ippKernel(ipp_kernel_height, ipp_kernel_width, useKernel16s ? CV_16SC1 : CV_32FC1, cv::Scalar::all(0));
        cv::Mat(kernel_height, kernel_width, kernel_type, (void*)kernel_data, kernel_step)
            .convertTo(ippKernel(cv::Rect(radius_x - anchor_x, radius_y - anchor_y, kernel_width, kernel_height)), ippKernel.type());

        // Pixels around the ROI that IPP may read from memory (non-isolated borders)
        const bool isolated = (borderType & cv::BORDER_ISOLATED) != 0;
        int margin_left   = isolated ? 0 : offset_x;
        int margin_top    = isolated ? 0 : offset_y;
        int margin_right  = isolated ? 0 : full_width  - offset_x - width;
        int margin_bottom = isolated ? 0 : full_height - offset_y - height;

        const uchar* srcPtr  = src_data;
        size_t       srcStep = src_step;
        if(src_data == dst_data)
        {
            // iwiFilter cannot run in place: filter from a copy of the source and the margins it needs
            margin_left   = std::min(margin_left,   radius_x);
            margin_right  = std::min(margin_right,  radius_x);
            margin_top    = std::min(margin_top,    radius_y);
            margin_bottom = std::min(margin_bottom, radius_y);
            const size_t esz = CV_ELEM_SIZE(src_type);
            cv::Mat(height + margin_top + margin_bottom, width + margin_left + margin_right, src_type,
                    (void*)(src_data - margin_top * src_step - margin_left * esz), src_step).copyTo(srcCopy);
            srcCopyRoi = cv::Rect(margin_left, margin_top, width, height);
            srcPtr  = srcCopy.ptr(margin_top) + margin_left * esz;
            srcStep = srcCopy.step;
        }

        const IppDataType       type = ippiGetDataType(depth);
        ::ipp::IwiSize          iwSize(width, height);
        ::ipp::IwiImage         iwKernel(::ipp::IwiSize(ipp_kernel_width, ipp_kernel_height), ippiGetDataType(ippKernel.depth()), 1, 0, ippKernel.ptr(), ippKernel.step);
        ::ipp::IwiImage         iwSrc(iwSize, type, channels, ::ipp::IwiBorderSize(margin_left, margin_top, margin_right, margin_bottom), (void*)srcPtr, srcStep);
        ::ipp::IwiImage         iwDst(iwSize, type, channels, ::ipp::IwiBorderSize(), (void*)dst_data, dst_step);
        ::ipp::IwiBorderSize    iwBorderSize = ::ipp::iwiSizeToBorderSize(::ipp::IwiSize(ipp_kernel_width, ipp_kernel_height));
        ::ipp::IwiBorderType    iwBorderType = ippiGetBorder(iwSrc, borderType, iwBorderSize);
        if(!iwBorderType)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const ::ipp::IwiFilterParams iwParams(1, ippOffset, ippAlgHintNone, roundMode);

        // Same condition as OpenCV's tiled parallel path in FilterEngine__apply() (filter.simd.hpp, keep in sync):
        // filter in parallel horizontal stripes exactly when OpenCV itself would run in parallel.
        const int nthreads = cv::getNumThreads();
        if(nthreads > 1 && (size_t)width * height >= std::max((size_t)1024 * 1024, (size_t)nthreads * 64 * 1024))
        {
            // A stripe can't be lower than the kernel radius: otherwise the next stripe has a mixed top border
            // (pixels from both memory and outside of the image), which IPP does not support.
            int numTiles = nthreads * 4;
            if(radius_y > 0)
                numTiles = std::max(1, std::min(numTiles, height / radius_y));

            // Exception safe: a failure in any stripe sets 'ok' to false, the other stripes are skipped then
            std::atomic_bool ok{true};
            const Ipp32f     deltaValue = (Ipp32f)delta;
            auto filterStripe = [&iwSrc, &iwDst, &iwKernel, &iwParams, &iwBorderType, addDelta, deltaValue, &ok](const cv::Range& range)
            {
                if(!ok.load(std::memory_order_relaxed))
                    return;
                try
                {
                    ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, iwDst.m_size.width, range.end - range.start);
                    CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilter, iwSrc, iwDst, iwKernel, iwParams, iwBorderType, tile);

                    if(addDelta)
                    {
                        // the same value for every channel: process C3/C4 as C1 (ICV has only ippiAddC_32f_C1IR)
                        const IppiSize roi = { (int)iwDst.m_size.width * iwDst.m_channels, range.end - range.start };
                        Ipp32f* rows = (Ipp32f*)((Ipp8u*)iwDst.m_ptr + (size_t)range.start * (size_t)iwDst.m_step);
                        if(CV_INSTRUMENT_FUN_IPP(ippiAddC_32f_C1IR, deltaValue, rows, (int)iwDst.m_step, roi) < 0)
                            ok.store(false, std::memory_order_relaxed);
                    }
                }
                catch(const ::ipp::IwException &)
                {
                    ok.store(false, std::memory_order_relaxed);
                }
            };
            cv::parallel_for_(cv::Range(0, height), filterStripe, numTiles);
            if(!ok)
                return fallback();
        }
        else
        {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilter, iwSrc, iwDst, iwKernel, iwParams, iwBorderType);

            if(addDelta)
            {
                // the same value for every channel: process C3/C4 as C1 (ICV has only ippiAddC_32f_C1IR)
                const IppiSize roi = { width * channels, height };
                IppStatus status = CV_INSTRUMENT_FUN_IPP(ippiAddC_32f_C1IR, (Ipp32f)delta, (Ipp32f*)dst_data, (int)dst_step, roi);
                if(status < 0)
                    return fallback();
            }
        }
    }
    catch(const ::ipp::IwException&)
    {
        return fallback();
    }

    CV_IMPL_ADD(CV_IMPL_IPP);
    return CV_HAL_ERROR_OK;
}

#endif // defined(HAVE_IPP_IW)

#endif // IPP_VERSION_X100 >= 810
