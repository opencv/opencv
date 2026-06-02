// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "ipp_hal_dnn.hpp"
#include "precomp_ipp.hpp"
#include <iostream>

#if IPP_VERSION_X100 >= 201700

// Requires `status` variable (IppStatus) to be declared in the calling scope.
#define CV_HAL_IPP_STATUS_CHECK(call) if ((status = (call)) < 0) return status

// set a constant value to the destination blob when paddingmode is DNN_PMODE_CONSTANT
static inline IppStatus ipp_set_constant(cv::Mat& dst, const double borderValue[4])
{
    IppiSize size = ippiSize(dst.cols, dst.rows);
    int depth = dst.depth();
    int channels = dst.channels();

    if (channels == 1)
    {
        if (depth == CV_8U)
            return CV_INSTRUMENT_FUN_IPP(ippiSet_8u_C1R, cv::saturate_cast<Ipp8u>(borderValue[0]), dst.ptr<Ipp8u>(), (int)dst.step, size);
        if (depth == CV_32F)
            return CV_INSTRUMENT_FUN_IPP(ippiSet_32f_C1R, cv::saturate_cast<Ipp32f>(borderValue[0]), dst.ptr<Ipp32f>(), (int)dst.step, size);
    }
    else if (channels == 3)
    {
        if (depth == CV_8U)
        {
            Ipp8u values[3] = { cv::saturate_cast<Ipp8u>(borderValue[0]), cv::saturate_cast<Ipp8u>(borderValue[1]), cv::saturate_cast<Ipp8u>(borderValue[2]) };
            return CV_INSTRUMENT_FUN_IPP(ippiSet_8u_C3R, values, dst.ptr<Ipp8u>(), (int)dst.step, size);
        }
        if (depth == CV_32F)
        {
            Ipp32f values[3] = { (Ipp32f)borderValue[0], (Ipp32f)borderValue[1], (Ipp32f)borderValue[2] };
            return CV_INSTRUMENT_FUN_IPP(ippiSet_32f_C3R, values, dst.ptr<Ipp32f>(), (int)dst.step, size);
        }
    }
    else if (channels == 4)
    {
        if (depth == CV_8U)
        {
            Ipp8u values[4] = { cv::saturate_cast<Ipp8u>(borderValue[0]), cv::saturate_cast<Ipp8u>(borderValue[1]),
                               cv::saturate_cast<Ipp8u>(borderValue[2]), cv::saturate_cast<Ipp8u>(borderValue[3]) };
            return CV_INSTRUMENT_FUN_IPP(ippiSet_8u_C4R, values, dst.ptr<Ipp8u>(), (int)dst.step, size);
        }
        if (depth == CV_32F)
        {
            Ipp32f values[4] = { (Ipp32f)borderValue[0], (Ipp32f)borderValue[1], (Ipp32f)borderValue[2], (Ipp32f)borderValue[3] };
            return CV_INSTRUMENT_FUN_IPP(ippiSet_32f_C4R, values, dst.ptr<Ipp32f>(), (int)dst.step, size);
        }
    }

    return ippStsAlgTypeErr;
}

// copy data from source image to destination blob in interleaved format (NHWC)
static inline IppStatus ipp_copy_interleaved(const cv::Mat& src, uchar* dst, size_t row_step, int channels, int depth, bool swapRB = false)
{
    IppiSize roi = ippiSize(src.cols, src.rows);
    IppiSizeL roiL = {(IppSizeL)src.cols, (IppSizeL)src.rows};
    const int order3[3] = {2, 1, 0};
    const int order4[4] = {2, 1, 0, 3};

    // Out-of-place channel swap directly into destination (avoids mutating src)
    if (swapRB && channels >= 3)
    {
        if (depth == CV_8U)
        {
            if (channels == 3)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_8u_C3R, src.ptr<Ipp8u>(), (int)src.step, dst, (int)row_step, roi, order3);
            if (channels == 4)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_8u_C4R, src.ptr<Ipp8u>(), (int)src.step, dst, (int)row_step, roi, order4);
        }
        else if (depth == CV_32F)
        {
            if (channels == 3)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C3R, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst, (int)row_step, roi, order3);
            if (channels == 4)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C4R, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst, (int)row_step, roi, order4);
        }
        return ippStsAlgTypeErr;
    }

    // Fast path: use memcpy if src is continuous and destination step matches
    size_t expected_step = src.cols * src.elemSize();
    if (src.isContinuous() && row_step == expected_step)
    {
        memcpy(dst, src.data, src.total() * src.elemSize());
        return ippStsNoErr;
    }

    // Use IPP copy functions
    if (depth == CV_8U)
    {
        if (channels == 1)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, src.ptr<Ipp8u>(), (int)src.step, dst, (int)row_step, roiL);
        if (channels == 3)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C3R_L, src.ptr<Ipp8u>(), (int)src.step, dst, (int)row_step, roiL);
        if (channels == 4)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C4R_L, src.ptr<Ipp8u>(), (int)src.step, dst, (int)row_step, roiL);
    }
    else if (depth == CV_32F)
    {
        if (channels == 1)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C1R_L, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst, (int)row_step, roiL);
        if (channels == 3)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C3R_L, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst, (int)row_step, roiL);
        if (channels == 4)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C4R_L, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst, (int)row_step, roiL);
    }

    return ippStsAlgTypeErr;
}

// deinterleave 32-bit float image from HWC to CHW format (used when converting to planar blob)
static inline IppStatus ipp_deinterleave_32f(const Ipp32f* src, int srcStep, Ipp32f* const planes[], int dstStep, IppiSizeL roiL, int channels)
{
    IppiSize roi = ippiSize((int)roiL.width, (int)roiL.height);
    if (channels == 1)
        return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C1R_L, src, srcStep, planes[0], dstStep, roiL);
    if (channels == 3)
        return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C3P3R, src, srcStep, planes, dstStep, roi);
    if (channels == 4)
        return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C4P4R, src, srcStep, planes, dstStep, roi);
    return ippStsNoErr;
}

// copy data from source image to destination blob in planar format (NCHW)
// swapRB: if true, swap R and B channels (like OpenCV's pointer swap trick)
static inline IppStatus ipp_copy_planar(const cv::Mat& src, uchar* dst_base, size_t plane_step, size_t row_step, int channels, int depth, bool swapRB)
{
    IppiSizeL roiL = {(IppSizeL)src.cols, (IppSizeL)src.rows};
    IppiSize roi = ippiSize(src.cols, src.rows);

    if (depth == CV_8U)
    {
        uchar* planes[4] = { dst_base, dst_base + plane_step, dst_base + 2 * plane_step, dst_base + 3 * plane_step };

        // Swap R and B plane pointers if needed (zero-cost)
        if (swapRB && channels >= 3)
            std::swap(planes[0], planes[2]);

        if (channels == 1)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, src.ptr<Ipp8u>(), (int)src.step, planes[0], (int)row_step, roiL);
        if (channels == 3)
        {
            Ipp8u* dstPlanes[3] = { planes[0], planes[1], planes[2] };
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C3P3R, src.ptr<Ipp8u>(), (int)src.step, dstPlanes, (int)row_step, roi);
        }
        if (channels == 4)
        {
            Ipp8u* dstPlanes[4] = { planes[0], planes[1], planes[2], planes[3] };
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C4P4R, src.ptr<Ipp8u>(), (int)src.step, dstPlanes, (int)row_step, roi);
        }
    }
    else if (depth == CV_32F)
    {
        float* planes[4] = {
            (float*)dst_base,
            (float*)(dst_base + plane_step),
            (float*)(dst_base + 2 * plane_step),
            (float*)(dst_base + 3 * plane_step)
        };

        // Swap R and B plane pointers if needed (zero-cost)
        if (swapRB && channels >= 3)
            std::swap(planes[0], planes[2]);

        return ipp_deinterleave_32f(src.ptr<Ipp32f>(), (int)src.step, planes, (int)row_step, roiL, channels);
    }

    return ippStsAlgTypeErr;
}

static IppStatus ipp_resize_hwc(cv::Mat& src, int dst_width, int dst_height, int interpolation,
                                int paddingmode, const double borderValue[4])
{
    IppStatus status;
    int width = src.cols;
    int height = src.rows;
    if (dst_width <= 0 || dst_height <= 0)
        return ippStsAlgTypeErr;

    if (paddingmode == /*DNN_PMODE_CROP_CENTER*/ 1)
    {
        float resizeFactor = std::max(dst_width / (float)width, dst_height / (float)height);
        // call cv::resize since it's more effective. It is parallelized and uses IPP underneath
        cv::resize(src, src, cv::Size(), resizeFactor, resizeFactor, interpolation);
        cv::Rect crop(cv::Point(0.5 * (src.cols - dst_width), 0.5 * (src.rows - dst_height)),
                      cv::Size(dst_width, dst_height));
        src = src(crop);
    }
    else if (paddingmode == /*DNN_PMODE_LETTERBOX*/ 2)
    {
        float resizeFactor = std::min(dst_width / (float)width, dst_height / (float)height);
        int rw = (int)(width * resizeFactor);
        int rh = (int)(height * resizeFactor);
        cv::resize(src, src, cv::Size(rw, rh), 0, 0, interpolation);

        int top = (dst_height - rh) / 2;
        int left = (dst_width - rw) / 2;

        cv::Mat bordered(dst_height, dst_width, src.type());
        CV_HAL_IPP_STATUS_CHECK(ipp_set_constant(bordered, borderValue));

        cv::Mat roi = bordered(cv::Rect(left, top, rw, rh));
        CV_HAL_IPP_STATUS_CHECK(ipp_copy_interleaved(src, roi.ptr(), roi.step, src.channels(), src.depth()));
        src = bordered;
    }
    else
    {
        cv::resize(src, src, cv::Size(dst_width, dst_height), 0, 0, interpolation);
    }

    return ippStsNoErr;
}

// Forward declaration
static IppStatus ipp_apply_mean_scale_32f(cv::Mat& img, const float mean[4], const float scalefactor[4], int channels);

// convert 8-bit unsigned image to 32-bit float image with scaling and mean subtraction
// This combines conversion, scaling, and mean subtraction to reduce passes
static inline IppStatus ipp_convert_8u_to_32f_scaled(const cv::Mat& src, cv::Mat& dst, int channels,
                                                       const float scalefactor[4], const float mean[4])
{
    IppiSize roi = ippiSize(src.cols, src.rows);
    dst.create(src.rows, src.cols, CV_MAKETYPE(CV_32F, channels));

    bool needScale = (scalefactor[0] != 1.0f || (channels > 1 && (scalefactor[1] != 1.0f || scalefactor[2] != 1.0f)));
    bool needMean = (mean[0] != 0.0f || (channels > 1 && (mean[1] != 0.0f || mean[2] != 0.0f)));

    // ippiScale_8u32f does: dst = (src - min) * (max / (max - min))
    // For our case: dst = src * scalefactor, so min=0, max=scalefactor
    // This combines convert + multiply in one pass
    if (needScale && !needMean && channels == 1)
    {
        // dst = (src - 0) * scalefactor
        return CV_INSTRUMENT_FUN_IPP(ippiScale_8u32f_C1R,
                                    src.ptr<Ipp8u>(), (int)src.step,
                                    dst.ptr<Ipp32f>(), (int)dst.step,
                                    roi, 0.0f, scalefactor[0]);
    }

    // Otherwise do convert, then apply mean/scale
    IppStatus status;
    if (channels == 1)
        status = CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C1R, src.ptr<Ipp8u>(), (int)src.step,
                                       dst.ptr<Ipp32f>(), (int)dst.step, roi);
    else if (channels == 3)
        status = CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C3R, src.ptr<Ipp8u>(), (int)src.step,
                                       dst.ptr<Ipp32f>(), (int)dst.step, roi);
    else if (channels == 4)
        status = CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C4R, src.ptr<Ipp8u>(), (int)src.step,
                                       dst.ptr<Ipp32f>(), (int)dst.step, roi);
    else
        return ippStsAlgTypeErr;

    CV_HAL_IPP_STATUS_CHECK(status);

    // Apply mean and/or scale if needed
    if (needMean || needScale)
    {
        return ipp_apply_mean_scale_32f(dst, mean, scalefactor, channels);
    }

    return ippStsNoErr;
}

// apply mean subtraction and scaling to 32-bit float image in-place (HWC layout)
static IppStatus ipp_apply_mean_scale_32f(cv::Mat& img, const float mean[4], const float scalefactor[4], int channels)
{
    IppStatus status;
    IppiSizeL roi = { (IppSizeL)img.cols, (IppSizeL)img.rows };
    if (channels == 1)
    {
        if (mean[0] != 0.0f)
            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiSubC_32f_C1IR_L, mean[0], img.ptr<Ipp32f>(), (int)img.step, roi));
        if (scalefactor[0] != 1.0f)
            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C1IR_L, scalefactor[0], img.ptr<Ipp32f>(), (int)img.step, roi));
        return ippStsNoErr;
    }
    if (channels == 3)
    {
        if (mean[0] != 0.0f || mean[1] != 0.0f || mean[2] != 0.0f)
            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiSubC_32f_C3IR_L, mean, img.ptr<Ipp32f>(), (int)img.step, roi));
        if (scalefactor[0] != 1.0f || scalefactor[1] != 1.0f || scalefactor[2] != 1.0f)
            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C3IR_L, scalefactor, img.ptr<Ipp32f>(), (int)img.step, roi));
        return ippStsNoErr;
    }
    if (channels == 4)
    {
        if (mean[0] != 0.0f || mean[1] != 0.0f || mean[2] != 0.0f || mean[3] != 0.0f)
            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiSubC_32f_C4IR_L, mean, img.ptr<Ipp32f>(), (int)img.step, roi));
        if (scalefactor[0] != 1.0f || scalefactor[1] != 1.0f || scalefactor[2] != 1.0f || scalefactor[3] != 1.0f)
            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C4IR_L, scalefactor, img.ptr<Ipp32f>(), (int)img.step, roi));
        return ippStsNoErr;
    }

    return ippStsAlgTypeErr;
}
#endif // IPP_VERSION_X100 >= 201700

// Main function to convert a batch of images to a blob suitable for DNN input
int ipp_hal_blobFromImages(const uchar* const* src_data, const size_t* src_step,
                               int nimages, int width, int height, int depth, int channels,
                               int ddepth, int swapRB,
                               int dst_width, int dst_height, int paddingmode,
                               const double borderValue[4],
                               const float mean[4], const float scalefactor[4],
                               uchar* dst_data, size_t* dst_step, int layout,
                               int interpolation)
{
#if IPP_VERSION_X100 >= 201700
    #ifndef HAVE_IPP_IW
    // IPP IW (Integrated Wrapper) is required for resize operations
    // Without it, IPP acceleration is not available
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
    #endif

    if (channels < 1 || channels > 4 || depth < 0 || depth >= CV_DEPTH_MAX)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if defined(IPP_CALLS_ENFORCED)
                  /* C1 C2 C3 C4 */
    char impl[2][4]={{1, 0, 1, 1},   //8U
                     {1, 0, 1, 1}};  //32F
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation

                  /* C1 C2 C3 C4 */
    char impl[2][4]={{1, 0, 1, 1},   //8U
                     {1, 0, 1, 1}};  //32F
#endif // IPP_CALLS_ENFORCED

    if (impl[(depth == CV_32F) ? 1 : 0][channels - 1] == 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (layout != /*DNN_LAYOUT_NCHW*/ 2 && layout != /*DNN_LAYOUT_NHWC*/ 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (!src_data || !src_step || !dst_data || nimages <= 0 || width <= 0 || height <= 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (ddepth != CV_8U && ddepth != CV_32F)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (ddepth == CV_8U)
    {
        if (depth != CV_8U)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        if (mean[0] != 0.0f || mean[1] != 0.0f || mean[2] != 0.0f || mean[3] != 0.0f)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        if (scalefactor[0] != 1.0f || scalefactor[1] != 1.0f || scalefactor[2] != 1.0f || scalefactor[3] != 1.0f)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if (dst_width <= 0 || dst_height <= 0)
    {
        dst_width = width;
        dst_height = height;
    }

    if ((size_t)dst_width * (size_t)dst_height > (size_t)INT_MAX)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Preallocate buffers to reuse across loop iterations
    cv::Mat floatImg;
    cv::Mat temp8uPlanes;
    IppStatus status = -1;

    // Precompute whether mean/scale operations are needed
    bool needMean = (mean[0] != 0.0f || mean[1] != 0.0f || mean[2] != 0.0f || mean[3] != 0.0f);
    bool needScale = (scalefactor[0] != 1.0f || scalefactor[1] != 1.0f || scalefactor[2] != 1.0f || scalefactor[3] != 1.0f);
    bool needResize = (dst_width != width || dst_height != height || paddingmode != 0);

    // NCHW 8u->32f with resize: multi-pass (deinterleave + convert + scale) loses to
    // OpenCV's fused single-pass loop that does deinterleave+convert+scale in one pass.
    if (layout == /*DNN_LAYOUT_NCHW*/ 2 && depth == CV_8U && ddepth == CV_32F
        && channels > 1 && needResize && !needMean && !needScale)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Preallocate temp 8u plane buffer for NCHW 8u -> 32f path (4x smaller than 32f HWC buffer)
    if (layout == /*DNN_LAYOUT_NCHW*/ 2 && depth == CV_8U && ddepth == CV_32F)
    {
        temp8uPlanes.create(dst_height * channels, dst_width, CV_8UC1);
    }

    for (int i = 0; i < nimages; ++i)
    {
        const uchar* src = src_data[i];
        size_t step = src_step[i];

        uchar* dst_img = dst_data + (size_t)i * dst_step[0];

        cv::Mat srcMat(height, width, CV_MAKETYPE(depth, channels), const_cast<uchar*>(src), step);
        cv::Mat& processedImg = srcMat;

        if (needResize)
        {
            CV_HAL_IPP_STATUS_CHECK(ipp_resize_hwc(srcMat, dst_width, dst_height, interpolation, paddingmode, borderValue));
        }

        if (layout == /*DNN_LAYOUT_NCHW*/ 2)
        {
            IppiSize planeRoi = ippiSize(processedImg.cols, processedImg.rows);
            IppiSizeL planeRoiL = {(IppSizeL)planeRoi.width, (IppSizeL)planeRoi.height};
            if (depth == CV_8U && ddepth == CV_32F)
            {
                // Optimized: deinterleave 8u -> convert per-plane to 32f -> mean/scale per-plane
                // Eliminates intermediate 32f HWC buffer
                int planeSize = processedImg.rows * processedImg.cols;
                int planeStep8u = processedImg.cols;

                // Deinterleave 8u HWC -> 8u temp planes (no swapRB here, handled later via dst pointer swap)
                CV_HAL_IPP_STATUS_CHECK(ipp_copy_planar(processedImg, temp8uPlanes.ptr(), planeSize, planeStep8u, channels, CV_8U, false));

                Ipp8u* tempPtrs[4];
                for (int c = 0; c < channels; c++)
                {
                    tempPtrs[c] = temp8uPlanes.ptr<Ipp8u>() + c * planeSize;
                }

                // Dst plane pointers with swapRB: source ch c -> dstPlanes[c]
                Ipp32f* dstPlanes[4];
                for (int c = 0; c < channels; c++)
                {
                    dstPlanes[c] = (Ipp32f*)(dst_img + c * dst_step[1]);
                }

                // Mean/scale are applied by destination plane index, so swap
                // indices 0 and 2 when swapRB to match the reference implementation.
                float localMean[4] = {mean[0], mean[1], mean[2], mean[3]};
                float localScale[4] = {scalefactor[0], scalefactor[1], scalefactor[2], scalefactor[3]};
                if (swapRB && channels >= 3)
                {
                    std::swap(dstPlanes[0], dstPlanes[2]);
                    std::swap(localMean[0], localMean[2]);
                    std::swap(localScale[0], localScale[2]);
                }

                // Convert each 8u temp plane -> 32f dst plane, then apply mean/scale
                for (int c = 0; c < channels; c++)
                {
                    if (localMean[c] == 0.0f)
                    {
                        if (localScale[c] != 1.0f)
                        {
                            const Ipp8u* srcPlane = tempPtrs[c];
                            Ipp32f* dstPlane = dstPlanes[c];
                            for (int j = 0; j < planeSize; j++)
                            {
                                dstPlane[j] = (float)srcPlane[j] * localScale[c];
                            }
                        }
                        else
                        {
                            // Fast path: just convert without mean/scale
                            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C1R, tempPtrs[c], planeStep8u, dstPlanes[c], (int)dst_step[2], planeRoi));
                        }
                    }
                    else
                    {
                        const float scale = localScale[c];
                        const float bias = -localMean[c] * scale;
                        const Ipp8u* srcPlane = tempPtrs[c];
                        Ipp32f* dstPlane = dstPlanes[c];
                        for (int j = 0; j < planeSize; j++)
                        {
                            dstPlane[j] = (float)srcPlane[j] * scale + bias;
                        }
                    }
                }
            }
            else if (depth == CV_32F && ddepth == CV_32F)
            {
                // Deinterleave 32f HWC -> 32f dst planes directly
                CV_HAL_IPP_STATUS_CHECK(ipp_copy_planar(processedImg, dst_img, dst_step[1], dst_step[2], channels, CV_32F, swapRB));

                // Apply mean/scale per-plane on dst (data is already in dst, no extra copy)
                if (needMean || needScale)
                {
                    Ipp32f* dstPlanes[4];
                    for (int c = 0; c < channels; c++)
                    {
                        dstPlanes[c] = (Ipp32f*)(dst_img + c * dst_step[1]);
                    }

                    // Mean/scale are applied by destination plane index, so swap
                    // indices 0 and 2 when swapRB to match the reference implementation.
                    float localMean[4] = {mean[0], mean[1], mean[2], mean[3]};
                    float localScale[4] = {scalefactor[0], scalefactor[1], scalefactor[2], scalefactor[3]};
                    if (swapRB && channels >= 3)
                    {
                        std::swap(dstPlanes[0], dstPlanes[2]);
                        std::swap(localMean[0], localMean[2]);
                        std::swap(localScale[0], localScale[2]);
                    }

                    for (int c = 0; c < channels; c++)
                    {
                        if (localMean[c] == 0.0f)
                        {
                            if (localScale[c] != 1.0f)
                            {
                                // Only scale needed: dst = dst * scale
                                CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C1IR_L, localScale[c], dstPlanes[c], (int)dst_step[2], planeRoiL));
                            }
                            // No mean/scale needed if mean == 0 and scale == 1, skip
                        }
                        else
                        {
                            const float scale = localScale[c];
                            const float bias = -localMean[c] * scale;
                            Ipp32f* dstPlane = dstPlanes[c];
                            const int planeSize = planeRoi.width * planeRoi.height;
                            for (int j = 0; j < planeSize; j++)
                            {
                                dstPlane[j] = dstPlane[j] * scale + bias;
                            }
                        }
                    }
                }
            }
            else
            {
                // 8u->8u: single deinterleave, no conversion needed
                CV_HAL_IPP_STATUS_CHECK(ipp_copy_planar(processedImg, dst_img, dst_step[1], dst_step[2], channels, ddepth, swapRB));
            }
        }
        else if (layout == /*DNN_LAYOUT_NHWC*/ 4)
        {
            if (ddepth == CV_32F && depth == CV_8U)
            {
                // Mean/scale are applied in destination channel order (after swapRB).
                // Swap indices 0 and 2 so that the correct mean/scale is applied
                // to each channel before the channel swap during copy.
                float localMean[4] = {mean[0], mean[1], mean[2], mean[3]};
                float localScale[4] = {scalefactor[0], scalefactor[1], scalefactor[2], scalefactor[3]};
                if (swapRB && channels >= 3)
                {
                    std::swap(localMean[0], localMean[2]);
                    std::swap(localScale[0], localScale[2]);
                }
                CV_HAL_IPP_STATUS_CHECK(ipp_convert_8u_to_32f_scaled(processedImg, floatImg, channels, localScale, localMean));
                CV_HAL_IPP_STATUS_CHECK(ipp_copy_interleaved(floatImg, dst_img, dst_step[1], channels, CV_32F, swapRB));
            }
            else if (ddepth == CV_32F && depth == CV_32F)
            {
                if (needMean || needScale)
                {
                    // Mean/scale are applied in destination channel order (after swapRB).
                    // Swap indices 0 and 2 so that the correct mean/scale is applied
                    // to each channel before the channel swap during copy.
                    float localMean[4] = {mean[0], mean[1], mean[2], mean[3]};
                    float localScale[4] = {scalefactor[0], scalefactor[1], scalefactor[2], scalefactor[3]};
                    if (swapRB && channels >= 3)
                    {
                        std::swap(localMean[0], localMean[2]);
                        std::swap(localScale[0], localScale[2]);
                    }
                    CV_HAL_IPP_STATUS_CHECK(ipp_copy_interleaved(processedImg, dst_img, dst_step[1], channels, CV_32F, false));
                    cv::Mat dstMat(processedImg.rows, processedImg.cols, CV_MAKETYPE(CV_32F, channels), dst_img, dst_step[1]);
                    CV_HAL_IPP_STATUS_CHECK(ipp_apply_mean_scale_32f(dstMat, localMean, localScale, channels));
                    if (swapRB && channels >= 3)
                    {
                        IppiSize roi = ippiSize(dstMat.cols, dstMat.rows);
                        const int order3[3] = {2, 1, 0};
                        const int order4[4] = {2, 1, 0, 3};
                        if (channels == 3){
                            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C3R, dstMat.ptr<Ipp32f>(), (int)dstMat.step, dstMat.ptr<Ipp32f>(), (int)dstMat.step, roi, order3));
                        }
                        else if (channels == 4){
                            CV_HAL_IPP_STATUS_CHECK(CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C4R, dstMat.ptr<Ipp32f>(), (int)dstMat.step, dstMat.ptr<Ipp32f>(), (int)dstMat.step, roi, order4));
                        }
                    }
                }
                else
                {
                    CV_HAL_IPP_STATUS_CHECK(ipp_copy_interleaved(processedImg, dst_img, dst_step[1], channels, CV_32F, swapRB));
                }
            }
            else
            {
                // 8u->8u: single copy with optional swapRB
                CV_HAL_IPP_STATUS_CHECK(ipp_copy_interleaved(processedImg, dst_img, dst_step[1], channels, ddepth, swapRB));
            }
        }
    }

    return CV_HAL_ERROR_OK;
#else
    CV_UNUSED(src_data);
    CV_UNUSED(src_step);
    CV_UNUSED(nimages);
    CV_UNUSED(width);
    CV_UNUSED(height);
    CV_UNUSED(depth);
    CV_UNUSED(channels);
    CV_UNUSED(ddepth);
    CV_UNUSED(swapRB);
    CV_UNUSED(dst_width);
    CV_UNUSED(dst_height);
    CV_UNUSED(paddingmode);
    CV_UNUSED(borderValue);
    CV_UNUSED(mean);
    CV_UNUSED(scalefactor);
    CV_UNUSED(dst_data);
    CV_UNUSED(dst_step);
    CV_UNUSED(layout);
    CV_UNUSED(interpolation);
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif // IPP_VERSION_X100 >= 201700
}

