// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#include "ipp_hal_dnn.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"
#include "resize_ipp.hpp"

#include <atomic>
#include <functional>

#if IPP_VERSION_X100 >= 202610

// The resize uses IPP Integration Wrappers (see resize_ipp.hpp)
#ifdef HAVE_IPP_IW

// Helpers propagate the IPP status, ipp_hal_blobFromImages() falls back to OpenCV on any IPP error
#define IPP_RETURN_ON_ERROR(call) do { IppStatus sts_ = (call); if (sts_ < 0) return sts_; } while (0)
#define HAL_RETURN_ON_ERROR(call) do { if ((call) < 0) return CV_HAL_ERROR_NOT_IMPLEMENTED; } while (0)

// fill the destination with borderValue (letterbox padding)
static IppStatus ipp_set_constant(cv::Mat& dst, const double borderValue[4])
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

    return ippStsDataTypeErr;
}

// copy an HWC image keeping it interleaved (NHWC), optionally swapping R and B
static IppStatus ipp_copy_interleaved(const cv::Mat& src, uchar* dst, size_t dst_step, bool swapRB)
{
    const int depth = src.depth(), channels = src.channels();
    IppiSize roi = ippiSize(src.cols, src.rows);
    IppiSizeL roiL = ippiSizeL(src.cols, src.rows);
    const int order3[3] = {2, 1, 0};
    const int order4[4] = {2, 1, 0, 3};

    if (swapRB)
    {
        if (depth == CV_8U)
        {
            if (channels == 3)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_8u_C3R, src.ptr<Ipp8u>(), (int)src.step, dst, (int)dst_step, roi, order3);
            if (channels == 4)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_8u_C4R, src.ptr<Ipp8u>(), (int)src.step, dst, (int)dst_step, roi, order4);
        }
        else if (depth == CV_32F)
        {
            if (channels == 3)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C3R, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst, (int)dst_step, roi, order3);
            if (channels == 4)
                return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C4R, src.ptr<Ipp32f>(), (int)src.step, (Ipp32f*)dst, (int)dst_step, roi, order4);
        }
        return ippStsDataTypeErr;
    }

    if (src.isContinuous() && dst_step == src.cols * src.elemSize())
    {
        memcpy(dst, src.data, src.total() * src.elemSize());
        return ippStsNoErr;
    }

    if (depth == CV_8U)
    {
        if (channels == 1)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, src.ptr<Ipp8u>(), (IppSizeL)src.step, dst, (IppSizeL)dst_step, roiL);
        if (channels == 3)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C3R_L, src.ptr<Ipp8u>(), (IppSizeL)src.step, dst, (IppSizeL)dst_step, roiL);
        if (channels == 4)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C4R_L, src.ptr<Ipp8u>(), (IppSizeL)src.step, dst, (IppSizeL)dst_step, roiL);
    }
    else if (depth == CV_32F)
    {
        if (channels == 1)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C1R_L, src.ptr<Ipp32f>(), (IppSizeL)src.step, (Ipp32f*)dst, (IppSizeL)dst_step, roiL);
        if (channels == 3)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C3R_L, src.ptr<Ipp32f>(), (IppSizeL)src.step, (Ipp32f*)dst, (IppSizeL)dst_step, roiL);
        if (channels == 4)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C4R_L, src.ptr<Ipp32f>(), (IppSizeL)src.step, (Ipp32f*)dst, (IppSizeL)dst_step, roiL);
    }

    return ippStsDataTypeErr;
}

// deinterleave an HWC image into channel planes (NCHW); swapRB swaps the R and B plane pointers
static IppStatus ipp_copy_planar(const cv::Mat& src, uchar* dst, size_t plane_step, size_t row_step, bool swapRB)
{
    const int depth = src.depth(), channels = src.channels();
    IppiSize roi = ippiSize(src.cols, src.rows);
    IppiSizeL roiL = ippiSizeL(src.cols, src.rows);
    uchar* planes[4] = { dst, dst + plane_step, dst + 2 * plane_step, dst + 3 * plane_step };
    if (swapRB)
        std::swap(planes[0], planes[2]);

    if (depth == CV_8U)
    {
        Ipp8u* const dstPlanes[4] = { planes[0], planes[1], planes[2], planes[3] };
        if (channels == 1)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, src.ptr<Ipp8u>(), (IppSizeL)src.step, dstPlanes[0], (IppSizeL)row_step, roiL);
        if (channels == 3)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C3P3R, src.ptr<Ipp8u>(), (int)src.step, dstPlanes, (int)row_step, roi);
        if (channels == 4)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C4P4R, src.ptr<Ipp8u>(), (int)src.step, dstPlanes, (int)row_step, roi);
    }
    else if (depth == CV_32F)
    {
        Ipp32f* const dstPlanes[4] = { (Ipp32f*)planes[0], (Ipp32f*)planes[1], (Ipp32f*)planes[2], (Ipp32f*)planes[3] };
        if (channels == 1)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C1R_L, src.ptr<Ipp32f>(), (IppSizeL)src.step, dstPlanes[0], (IppSizeL)row_step, roiL);
        if (channels == 3)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C3P3R, src.ptr<Ipp32f>(), (int)src.step, dstPlanes, (int)row_step, roi);
        if (channels == 4)
            return CV_INSTRUMENT_FUN_IPP(ippiCopy_32f_C4P4R, src.ptr<Ipp32f>(), (int)src.step, dstPlanes, (int)row_step, roi);
    }

    return ippStsDataTypeErr;
}

static IppStatus ipp_convert_8u32f(const cv::Mat& src, cv::Mat& dst)
{
    IppiSize roi = ippiSize(src.cols, src.rows);
    switch (src.channels())
    {
    case 1: return CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C1R, src.ptr<Ipp8u>(), (int)src.step, dst.ptr<Ipp32f>(), (int)dst.step, roi);
    case 3: return CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C3R, src.ptr<Ipp8u>(), (int)src.step, dst.ptr<Ipp32f>(), (int)dst.step, roi);
    case 4: return CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C4R, src.ptr<Ipp8u>(), (int)src.step, dst.ptr<Ipp32f>(), (int)dst.step, roi);
    default: return ippStsDataTypeErr;
    }
}

// in-place img = (img - mean) * scalefactor for an interleaved 32F image, mean/scalefactor are in image channel order
static IppStatus ipp_apply_mean_scale_32f(cv::Mat& img, const float mean[4], const float scalefactor[4])
{
    const int channels = img.channels();
    IppiSizeL roi = ippiSizeL(img.cols, img.rows);
    bool needMean = false, needScale = false;
    for (int c = 0; c < channels; c++)
    {
        needMean |= mean[c] != 0.0f;
        needScale |= scalefactor[c] != 1.0f;
    }

    if (channels == 1)
    {
        if (needMean)
            IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiSubC_32f_C1IR_L, mean[0], img.ptr<Ipp32f>(), (IppSizeL)img.step, roi));
        if (needScale)
            IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C1IR_L, scalefactor[0], img.ptr<Ipp32f>(), (IppSizeL)img.step, roi));
        return ippStsNoErr;
    }
    if (channels == 3)
    {
        if (needMean)
            IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiSubC_32f_C3IR_L, mean, img.ptr<Ipp32f>(), (IppSizeL)img.step, roi));
        if (needScale)
            IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C3IR_L, scalefactor, img.ptr<Ipp32f>(), (IppSizeL)img.step, roi));
        return ippStsNoErr;
    }
    if (channels == 4)
    {
        if (needMean)
            IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiSubC_32f_C4IR_L, mean, img.ptr<Ipp32f>(), (IppSizeL)img.step, roi));
        if (needScale)
            IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C4IR_L, scalefactor, img.ptr<Ipp32f>(), (IppSizeL)img.step, roi));
        return ippStsNoErr;
    }

    return ippStsDataTypeErr;
}

// in-place plane = (plane - mean) * scale with the same operations as the OpenCV implementation
static IppStatus ipp_mean_scale_plane_32f(Ipp32f* plane, size_t step, IppiSizeL roi, float mean, float scale)
{
    if (mean != 0.0f)
        IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiSubC_32f_C1IR_L, mean, plane, (IppSizeL)step, roi));
    if (scale != 1.0f)
        IPP_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C1IR_L, scale, plane, (IppSizeL)step, roi));
    return ippStsNoErr;
}

// How an image is mapped to the dsize blob image: the image is resized to `rsize`, then the blob takes the dsize window
// at `crop` of it, or puts it at (`left`, `top`) and pads it (`letterbox`)
struct ipp_blob_geometry
{
    cv::Size rsize;
    double inv_scale_x, inv_scale_y;
    cv::Point crop;
    bool letterbox; // the blob image has padding around the resized image
    int top, left;
};

// The resized image is placed at `offset` in the blob image: a negative offset is a crop window, a resized image that
// doesn't cover the blob image is padded. A mix of both (crop along one axis, padding along the other) isn't supported.
static bool ipp_blob_get_geometry(const cv::Size& rsize, double inv_scale_x, double inv_scale_y, const cv::Point& offset,
                                  const cv::Size& dsize, ipp_blob_geometry& g)
{
    if (rsize.width <= 0 || rsize.height <= 0 || !(inv_scale_x > 0) || !(inv_scale_y > 0))
        return false;
    g.rsize = rsize;
    g.inv_scale_x = inv_scale_x;
    g.inv_scale_y = inv_scale_y;
    g.letterbox = offset.x > 0 || offset.y > 0 ||
                  offset.x + rsize.width < dsize.width || offset.y + rsize.height < dsize.height;
    if (g.letterbox)
    {
        if (offset.x < 0 || offset.y < 0 || offset.x + rsize.width > dsize.width || offset.y + rsize.height > dsize.height)
            return false;
        g.crop = cv::Point();
        g.left = offset.x;
        g.top = offset.y;
    }
    else
    {
        g.crop = -offset;
        g.top = g.left = 0;
    }
    return true;
}

// Fill rows [r0, r1) of the letterbox padding around the resized image in the blob image `img`
static int ipp_blob_fill_padding(cv::Mat& img, const ipp_blob_geometry& g, int r0, int r1, const double borderValue[4])
{
    const int bottom = g.top + g.rsize.height, right = g.left + g.rsize.width;
    const int m0 = std::max(r0, g.top), m1 = std::min(r1, bottom); // rows with the resized image
    const cv::Rect strips[] = { cv::Rect(0, r0, img.cols, std::min(r1, g.top) - r0),
                                cv::Rect(0, std::max(r0, bottom), img.cols, r1 - std::max(r0, bottom)),
                                cv::Rect(0, m0, g.left, m1 - m0),
                                cv::Rect(right, m0, img.cols - right, m1 - m0) };
    for (const cv::Rect& strip : strips)
    {
        if (strip.width <= 0 || strip.height <= 0)
            continue;
        cv::Mat border = img(strip);
        HAL_RETURN_ON_ERROR(ipp_set_constant(border, borderValue));
    }
    return CV_HAL_ERROR_OK;
}

// The blob image of an image that keeps its size (resize factor 1): the crop window, or the padded image.
// cv::resize() copies an image of the same size without calling the HAL.
static int ipp_blob_no_resize(const cv::Mat& src, cv::Mat& dst, cv::Mat& bordered, const ipp_blob_geometry& g,
                              const cv::Size& dsize, const double borderValue[4])
{
    if (!g.letterbox)
    {
        dst = src(cv::Rect(g.crop, dsize));
        return CV_HAL_ERROR_OK;
    }
    bordered.create(dsize, src.type());
    HAL_RETURN_ON_ERROR(ipp_blob_fill_padding(bordered, g, 0, dsize.height, borderValue));
    cv::Mat roi = bordered(cv::Rect(cv::Point(g.left, g.top), g.rsize));
    HAL_RETURN_ON_ERROR(ipp_copy_interleaved(src, roi.ptr(), roi.step, false));
    dst = bordered;
    return CV_HAL_ERROR_OK;
}

struct ipp_blob_params
{
    int depth, ddepth, channels, layout;
    bool swap, needMeanScale;
    const float* mean;
    const float* scalefactor;
    const size_t* dst_step;
};

// Convert rows [r0, r1) of the HWC image `img`, which already has the blob width x height, to the blob of one image at `dst`.
// `buf` is a scratch buffer reused between calls.
static int ipp_blob_rows(const cv::Mat& img, uchar* dst, int r0, int r1, const ipp_blob_params& p, cv::AutoBuffer<uchar>& buf)
{
    const cv::Mat rows = img.rowRange(r0, r1);
    const int width = img.cols, nrows = r1 - r0;
    const size_t planeLen = (size_t)width * nrows;
    const IppiSizeL roiL = ippiSizeL(width, nrows);
    const size_t* dst_step = p.dst_step;

    if (p.layout == cv::DATA_LAYOUT_NCHW)
    {
        uchar* dst0 = dst + r0 * dst_step[2];
        if (p.depth == CV_8U && p.ddepth == CV_32F)
        {
            // Deinterleave to 8U planes, then convert each plane: no 32F HWC intermediate.
            // A single channel image is a plane already.
            const uchar* planes = rows.ptr();
            size_t planeStep = rows.step;
            if (p.channels > 1)
            {
                buf.allocate(planeLen * p.channels);
                HAL_RETURN_ON_ERROR(ipp_copy_planar(rows, buf.data(), planeLen, width, p.swap));
                planes = buf.data();
                planeStep = width;
            }
            for (int c = 0; c < p.channels; c++)
            {
                const Ipp8u* srcPlane = planes + c * planeLen;
                Ipp32f* dstPlane = (Ipp32f*)(dst0 + c * dst_step[1]);
                HAL_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C1R, srcPlane, (int)planeStep, dstPlane, (int)dst_step[2], ippiSize(width, nrows)));
                HAL_RETURN_ON_ERROR(ipp_mean_scale_plane_32f(dstPlane, dst_step[2], roiL, p.mean[c], p.scalefactor[c]));
            }
        }
        else
        {
            HAL_RETURN_ON_ERROR(ipp_copy_planar(rows, dst0, dst_step[1], dst_step[2], p.swap));
            for (int c = 0; c < p.channels && p.needMeanScale; c++)
                HAL_RETURN_ON_ERROR(ipp_mean_scale_plane_32f((Ipp32f*)(dst0 + c * dst_step[1]), dst_step[2], roiL,
                                                             p.mean[c], p.scalefactor[c]));
        }
    }
    else // DATA_LAYOUT_NHWC
    {
        uchar* dst0 = dst + r0 * dst_step[1];
        if (p.depth == p.ddepth && !p.needMeanScale)
        {
            HAL_RETURN_ON_ERROR(ipp_copy_interleaved(rows, dst0, dst_step[1], p.swap));
        }
        else
        {
            cv::Mat dstMat(nrows, width, CV_MAKETYPE(CV_32F, p.channels), dst0, dst_step[1]);
            if (p.depth == CV_8U)
            {
                // There is no in-place 32F channel swap, so swap the 8U source before the conversion
                cv::Mat src8u = rows;
                if (p.swap)
                {
                    buf.allocate(planeLen * p.channels);
                    src8u = cv::Mat(nrows, width, rows.type(), buf.data());
                    HAL_RETURN_ON_ERROR(ipp_copy_interleaved(rows, src8u.ptr(), src8u.step, true));
                }
                HAL_RETURN_ON_ERROR(ipp_convert_8u32f(src8u, dstMat));
            }
            else
            {
                HAL_RETURN_ON_ERROR(ipp_copy_interleaved(rows, dst0, dst_step[1], p.swap));
            }
            HAL_RETURN_ON_ERROR(ipp_apply_mean_scale_32f(dstMat, p.mean, p.scalefactor));
        }
    }
    return CV_HAL_ERROR_OK;
}

// Convert a batch of HWC images to an NCHW or NHWC blob.
// mean[c] and scalefactor[c] apply to channel c of the blob, i.e. after the swapRB channel swap.
int ipp_hal_blobFromImages(const uchar* const* src_data, const size_t* src_step, int nimages,
                           int width, int height, int src_type, int ddepth, int swapRB,
                           int dst_width, int dst_height, int resized_width, int resized_height,
                           double inv_scale_x, double inv_scale_y, int offset_x, int offset_y,
                           const double borderValue[4], const double mean_[4], const double scalefactor_[4],
                           uchar* dst_data, const size_t* dst_step, int layout, int interpolation)
{
    CV_HAL_CHECK_USE_IPP();

    const int depth = CV_MAT_DEPTH(src_type), channels = CV_MAT_CN(src_type);

    if (!src_data || !src_step || !dst_data || !dst_step || nimages <= 0 || width <= 0 || height <= 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((depth != CV_8U && depth != CV_32F) || (ddepth != CV_8U && ddepth != CV_32F) || channels < 1 || channels > 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (layout != cv::DATA_LAYOUT_NCHW && layout != cv::DATA_LAYOUT_NHWC)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if defined(IPP_CALLS_ENFORCED)
                       /* C1 C2 C3 C4 */
    const char impl[2][4] = {{1, 0, 1, 1},   // 8U
                             {1, 0, 1, 1}};  // 32F
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation
                       /* C1 C2 C3 C4 */
    const char impl[2][4] = {{1, 0, 1, 1},   // 8U
                             {1, 0, 1, 1}};  // 32F
#endif // IPP_CALLS_ENFORCED

    if (impl[depth == CV_32F ? 1 : 0][channels - 1] == 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    float mean[4], scalefactor[4];
    bool needMeanScale = false;
    for (int c = 0; c < 4; c++)
    {
        mean[c] = (float)mean_[c];
        scalefactor[c] = (float)scalefactor_[c];
        if (c < channels)
            needMeanScale |= mean[c] != 0.0f || scalefactor[c] != 1.0f;
    }

    // 8U blob is a plain copy: no conversion from 32F, no mean or scale
    if (ddepth == CV_8U && (depth != CV_8U || needMeanScale))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (dst_width <= 0 || dst_height <= 0 || (size_t)dst_width * (size_t)dst_height > (size_t)INT_MAX)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const cv::Size dsize(dst_width, dst_height);
    ipp_blob_geometry g;
    if (!ipp_blob_get_geometry(cv::Size(resized_width, resized_height), inv_scale_x, inv_scale_y,
                               cv::Point(offset_x, offset_y), dsize, g))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    const bool needResize = g.rsize != cv::Size(width, height);

#if !defined(IPP_CALLS_ENFORCED)
    // ipp_resize_prepare() declines 8U INTER_LINEAR in this mode: give up before any work is done
    if (needResize && depth == CV_8U && (interpolation & cv::INTER_MAX) == cv::INTER_LINEAR)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    const ipp_blob_params p = { depth, ddepth, channels, layout, swapRB && channels >= 3, needMeanScale,
                                mean, scalefactor, dst_step };

    // Large images are converted in bands of rows, so the intermediate data of a band stays in cache.
    // The bands of all images are split into one contiguous run per thread.
    const size_t srcPixelSize = (size_t)channels * CV_ELEM_SIZE1(depth);
    const size_t dstPixelSize = (size_t)channels * CV_ELEM_SIZE1(ddepth);
    const size_t imageBytes = dstPixelSize * dst_width * dst_height;
    int l2cache = 0;
    ippGetL2CacheSize(&l2cache);
    if (l2cache <= 0)
        l2cache = 1 << 18;

    // Convert `count` images of the blob size starting with blob image `first`. `prepareRows`, if set, makes rows
    // [r0, r1) of image i ready (resize, padding) right before their conversion, in the same thread, so that the
    // conversion reads them from the cache of that core. `srcBytes` is the data read per blob pixel.
    auto convertImages = [&](const cv::Mat* imgs, int first, int count, double srcBytes,
                             const std::function<bool(int, int, int)>& prepareRows) -> int
    {
        const int threads = ippiSuggestThreadsNum(dst_width, (size_t)count * dst_height, dstPixelSize,
                                                  1 + srcBytes / dstPixelSize);
        int bandRows = threads == 1 && imageBytes <= (size_t)l2cache ? dst_height
                     : std::max(1, std::min(dst_height, (int)((128 << 10) / (dstPixelSize * dst_width))));
        if (threads > 1) // at least one band per thread
            bandRows = std::max(1, std::min(bandRows, (int)(((int64)count * dst_height + threads - 1) / threads)));
        const int nbands = (dst_height + bandRows - 1) / bandRows;
        const int ntasks = count * nbands;

        std::atomic_bool ok{true};
        auto convertTasks = [&](int t0, int t1)
        {
            cv::AutoBuffer<uchar> buf;
            for (int t = t0; t < t1 && ok; t++)
            {
                const int i = t / nbands, r0 = (t % nbands) * bandRows, r1 = std::min(r0 + bandRows, dst_height);
                if ((prepareRows && !prepareRows(i, r0, r1)) ||
                    ipp_blob_rows(imgs[i], dst_data + (size_t)(first + i) * dst_step[0], r0, r1, p, buf) != CV_HAL_ERROR_OK)
                    ok = false;
            }
        };
        const int nchunks = std::min(threads, ntasks);
        if (nchunks > 1)
        {
            // One range element per thread: the thread pool wakes only the threads that get work
            cv::parallel_for_(cv::Range(0, nchunks), [&](const cv::Range& range)
            {
                for (int c = range.start; c < range.end; c++)
                    convertTasks((int)((int64)ntasks * c / nchunks), (int)((int64)ntasks * (c + 1) / nchunks));
            }, nchunks);
        }
        else
        {
            convertTasks(0, ntasks);
        }
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    };

    if (!needResize && g.rsize == dsize) // no crop or padding either
    {
        std::vector<cv::Mat> imgs(nimages);
        for (int i = 0; i < nimages; ++i)
            imgs[i] = cv::Mat(height, width, src_type, const_cast<uchar*>(src_data[i]), src_step[i]);
        return convertImages(imgs.data(), 0, nimages, (double)srcPixelSize, nullptr);
    }

    // Resize and convert one image: every band of blob rows is resized (only the part the blob takes from the
    // resized image) and converted by the same thread, so the resized data does not travel between cores
    cv::Mat resized, bordered;
    auto resizeAndConvert = [&](const cv::Mat& src, int i) -> int
    {
        IppiInterpolationType inter;
        bool affine;
        if (ipp_resize_prepare(src_type, src.cols, src.rows, g.rsize.width, g.rsize.height,
                               g.inv_scale_x, g.inv_scale_y, interpolation, inter, affine) != CV_HAL_ERROR_OK)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        // The resize writes the resized image, or the middle of the padded blob image
        cv::Mat out, img;
        if (g.letterbox)
        {
            bordered.create(dsize, src_type);
            out = bordered(cv::Rect(cv::Point(g.left, g.top), g.rsize));
            img = bordered;
        }
        else
        {
            resized.create(g.rsize, src_type);
            out = resized;
            img = resized(cv::Rect(g.crop, dsize));
        }

        try
        {
            const IppDataType ippType = ippiGetDataType(depth);
            ::ipp::IwiImage iwSrc(::ipp::IwiSize(src.cols, src.rows), ippType, channels, 0, (void*)src.ptr(), src.step);
            ::ipp::IwiImage iwDst(::ipp::IwiSize(out.cols, out.rows), ippType, channels, 0, (void*)out.ptr(), out.step);
            std::atomic_bool resizeOk{true};
            ipp_resizeParallel       resizeGeneral(iwSrc, iwDst, resizeOk);
            ipp_resizeAffineParallel resizeAffine(iwSrc, iwDst, resizeOk);
            if (affine)
                resizeAffine.Init(inter, g.inv_scale_x, g.inv_scale_y);
            else
                resizeGeneral.Init(inter);

            auto prepareRows = [&](int, int r0, int r1) -> bool
            {
                if (g.letterbox && ipp_blob_fill_padding(bordered, g, r0, r1, borderValue) != CV_HAL_ERROR_OK)
                    return false;
                // Blob rows [r0, r1) are these rows of the resized image
                const int y0 = std::max(r0 + g.crop.y - g.top, 0), y1 = std::min(r1 + g.crop.y - g.top, g.rsize.height);
                if (y0 >= y1)
                    return true;
                const ::ipp::IwiTile tile = ::ipp::IwiRoi(g.crop.x, y0, g.letterbox ? g.rsize.width : dsize.width, y1 - y0);
                return affine ? resizeAffine.runTile(tile) : resizeGeneral.runTile(tile);
            };
            return convertImages(&img, i, 1, srcPixelSize * ((double)src.total() / dsize.area() + 1), prepareRows);
        }
        catch(const ::ipp::IwException &)
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    };

    for (int i = 0; i < nimages; ++i)
    {
        const cv::Mat src(height, width, src_type, const_cast<uchar*>(src_data[i]), src_step[i]);
        int res;
        if (!needResize)
        {
            cv::Mat img;
            res = ipp_blob_no_resize(src, img, bordered, g, dsize, borderValue);
            if (res == CV_HAL_ERROR_OK)
                res = convertImages(&img, i, 1, (double)srcPixelSize, nullptr);
        }
        else
        {
            res = resizeAndConvert(src, i);
        }
        if (res != CV_HAL_ERROR_OK)
            return res;
    }

    return CV_HAL_ERROR_OK;
}

#else // HAVE_IPP_IW

int ipp_hal_blobFromImages(const uchar* const*, const size_t*, int, int, int, int, int, int, int, int, int, int,
                           double, double, int, int, const double[4], const double[4], const double[4],
                           uchar*, const size_t*, int, int)
{
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // HAVE_IPP_IW

#endif // IPP_VERSION_X100 >= 202610
