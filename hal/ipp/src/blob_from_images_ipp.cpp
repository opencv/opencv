// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_dnn.hpp"
#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#if IPP_VERSION_X100 >= 202610

// Resize goes through ipp_hal_resize(), which is available with IW only
#ifdef HAVE_IPP_IW

// cv::dnn::ImagePaddingMode values (the HAL does not depend on the dnn module)
enum { IPP_DNN_PMODE_NULL = 0, IPP_DNN_PMODE_CROP_CENTER = 1, IPP_DNN_PMODE_LETTERBOX = 2 };

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

// dst = (src - mean) * scale over a continuous plane, the same expression as the OpenCV implementation uses
template <typename T>
static void mean_scale_plane(const T* src, float* dst, size_t len, float mean, float scale)
{
    for (size_t i = 0; i < len; i++)
        dst[i] = ((float)src[i] - mean) * scale;
}

// Resize (+ crop or letterbox) one HWC image to dsize exactly like blobFromImagesWithParams() does,
// with cv::resize() replaced by the IPP HAL resize. `resized` and `bordered` are reusable buffers.
static int ipp_resize_hwc(const cv::Mat& src, cv::Mat& dst, cv::Mat& resized, cv::Mat& bordered,
                          const cv::Size& dsize, int interpolation, int paddingmode, const double borderValue[4])
{
    cv::Size rsize = dsize;
    double inv_scale_x = (double)dsize.width / src.cols;
    double inv_scale_y = (double)dsize.height / src.rows;
    if (paddingmode == IPP_DNN_PMODE_CROP_CENTER)
    {
        // cv::resize(src, dst, Size(), f, f): the size is derived from the factor
        float resizeFactor = std::max(dsize.width / (float)src.cols, dsize.height / (float)src.rows);
        inv_scale_x = inv_scale_y = resizeFactor;
        rsize = cv::Size(cv::saturate_cast<int>(src.cols * inv_scale_x), cv::saturate_cast<int>(src.rows * inv_scale_y));
    }
    else if (paddingmode == IPP_DNN_PMODE_LETTERBOX)
    {
        float resizeFactor = std::min(dsize.width / (float)src.cols, dsize.height / (float)src.rows);
        rsize = cv::Size((int)(src.cols * resizeFactor), (int)(src.rows * resizeFactor));
        inv_scale_x = (double)rsize.width / src.cols;
        inv_scale_y = (double)rsize.height / src.rows;
    }
    if (rsize.empty())
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // cv::resize() copies an image of the same size without calling the HAL
    cv::Mat img = src;
    if (rsize != src.size())
    {
        resized.create(rsize, src.type());
        if (ipp_hal_resize(src.type(), src.ptr(), src.step, src.cols, src.rows,
                           resized.ptr(), resized.step, resized.cols, resized.rows,
                           inv_scale_x, inv_scale_y, interpolation) != CV_HAL_ERROR_OK)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        img = resized;
    }

    if (paddingmode == IPP_DNN_PMODE_CROP_CENTER)
    {
        if (img.cols < dsize.width || img.rows < dsize.height)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        dst = img(cv::Rect(cv::Point((int)(0.5 * (img.cols - dsize.width)), (int)(0.5 * (img.rows - dsize.height))), dsize));
    }
    else if (paddingmode == IPP_DNN_PMODE_LETTERBOX)
    {
        int top = (dsize.height - rsize.height) / 2;
        int left = (dsize.width - rsize.width) / 2;
        bordered.create(dsize, src.type());
        HAL_RETURN_ON_ERROR(ipp_set_constant(bordered, borderValue));
        cv::Mat roi = bordered(cv::Rect(left, top, rsize.width, rsize.height));
        HAL_RETURN_ON_ERROR(ipp_copy_interleaved(img, roi.ptr(), roi.step, false));
        dst = bordered;
    }
    else
    {
        dst = img;
    }
    return CV_HAL_ERROR_OK;
}

// Convert a batch of HWC images to an NCHW or NHWC blob.
// mean[c] and scalefactor[c] apply to channel c of the blob, i.e. after the swapRB channel swap.
int ipp_hal_blobFromImages(const uchar* const* src_data, const size_t* src_step, int nimages,
                           int width, int height, int src_type, int ddepth, int swapRB,
                           int dst_width, int dst_height, int paddingmode, const double borderValue[4],
                           const double mean_[4], const double scalefactor_[4],
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

    if (dst_width <= 0 || dst_height <= 0)
    {
        dst_width = width;
        dst_height = height;
    }
    if ((size_t)dst_width * (size_t)dst_height > (size_t)INT_MAX)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    // NCHW channel planes are processed as continuous arrays
    if (layout == cv::DATA_LAYOUT_NCHW && dst_step[1] != dst_step[2] * dst_height)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const cv::Size dsize(dst_width, dst_height);
    const bool needResize = dst_width != width || dst_height != height;
    const bool swap = swapRB && channels >= 3;

    // NCHW 8U -> 32F with resize and without mean/scale: the multi-pass IPP path loses to
    // the OpenCV loop that does deinterleave + convert in a single pass
    if (layout == cv::DATA_LAYOUT_NCHW && depth == CV_8U && ddepth == CV_32F
        && channels > 1 && needResize && !needMeanScale)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    cv::Mat resized, bordered, tmp;
    for (int i = 0; i < nimages; ++i)
    {
        const cv::Mat src(height, width, CV_MAKETYPE(depth, channels), const_cast<uchar*>(src_data[i]), src_step[i]);
        cv::Mat img = src;
        if (needResize)
        {
            int res = ipp_resize_hwc(src, img, resized, bordered, dsize, interpolation, paddingmode, borderValue);
            if (res != CV_HAL_ERROR_OK)
                return res;
        }

        uchar* dst = dst_data + (size_t)i * dst_step[0];
        const size_t planeSize = (size_t)dst_width * dst_height;

        if (layout == cv::DATA_LAYOUT_NCHW)
        {
            if (depth == CV_8U && ddepth == CV_32F)
            {
                // Deinterleave to 8U planes, then convert each plane: no 32F HWC intermediate
                tmp.create(dst_height * channels, dst_width, CV_8UC1);
                HAL_RETURN_ON_ERROR(ipp_copy_planar(img, tmp.ptr(), planeSize, dst_width, swap));
                for (int c = 0; c < channels; c++)
                {
                    const Ipp8u* srcPlane = tmp.ptr<Ipp8u>() + c * planeSize;
                    Ipp32f* dstPlane = (Ipp32f*)(dst + c * dst_step[1]);
                    if (mean[c] == 0.0f && scalefactor[c] == 1.0f)
                        HAL_RETURN_ON_ERROR(CV_INSTRUMENT_FUN_IPP(ippiConvert_8u32f_C1R, srcPlane, dst_width, dstPlane, (int)dst_step[2], ippiSize(dsize)));
                    else
                        mean_scale_plane(srcPlane, dstPlane, planeSize, mean[c], scalefactor[c]);
                }
            }
            else
            {
                HAL_RETURN_ON_ERROR(ipp_copy_planar(img, dst, dst_step[1], dst_step[2], swap));
                for (int c = 0; c < channels && needMeanScale; c++)
                {
                    Ipp32f* dstPlane = (Ipp32f*)(dst + c * dst_step[1]);
                    if (mean[c] != 0.0f || scalefactor[c] != 1.0f)
                        mean_scale_plane(dstPlane, dstPlane, planeSize, mean[c], scalefactor[c]);
                }
            }
        }
        else // DATA_LAYOUT_NHWC
        {
            if (depth == ddepth && !needMeanScale)
            {
                HAL_RETURN_ON_ERROR(ipp_copy_interleaved(img, dst, dst_step[1], swap));
            }
            else
            {
                cv::Mat dstMat(dst_height, dst_width, CV_MAKETYPE(CV_32F, channels), dst, dst_step[1]);
                if (depth == CV_8U)
                {
                    // There is no in-place 32F channel swap, so swap the 8U source before the conversion
                    cv::Mat src8u = img;
                    if (swap)
                    {
                        tmp.create(img.size(), img.type());
                        HAL_RETURN_ON_ERROR(ipp_copy_interleaved(img, tmp.ptr(), tmp.step, true));
                        src8u = tmp;
                    }
                    HAL_RETURN_ON_ERROR(ipp_convert_8u32f(src8u, dstMat));
                }
                else
                {
                    HAL_RETURN_ON_ERROR(ipp_copy_interleaved(img, dst, dst_step[1], swap));
                }
                HAL_RETURN_ON_ERROR(ipp_apply_mean_scale_32f(dstMat, mean, scalefactor));
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

#else // HAVE_IPP_IW

int ipp_hal_blobFromImages(const uchar* const*, const size_t*, int, int, int, int, int, int, int, int, int,
                           const double[4], const double[4], const double[4], uchar*, const size_t*, int, int)
{
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // HAVE_IPP_IW

#endif // IPP_VERSION_X100 >= 202610
