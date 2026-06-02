// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include "dnn_ipp.hpp"
#include <opencv2/imgproc.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// UMat version - always returns false (IPP is CPU-only)
bool blobFromImages_HAL(const std::vector<UMat>& images, UMat& blob, const Image2BlobParams& param, const Size& targetSize)
{
    CV_UNUSED(images);
    CV_UNUSED(blob);
    CV_UNUSED(param);
    CV_UNUSED(targetSize);
    return false;
}

// Mat version - uses IPP HAL if available
bool blobFromImages_HAL(const std::vector<Mat>& images, Mat& blob, const Image2BlobParams& param, const Size& targetSize)
{
    if(param.datalayout != DNN_LAYOUT_NCHW && param.datalayout != DNN_LAYOUT_NHWC)
        return false;

#ifdef IPP_HAL_DNN
    if (images.empty())
        return false;
    if (param.ddepth != CV_8U && param.ddepth != CV_32F)
        return false;

    const Mat& image0 = images[0];
    if (image0.dims != 2)
        return false;

    const int nch = image0.channels();
    if (nch != 1 && nch != 3 && nch != 4)
        return false;

    // Validate all images against image0
    // HAL receives a single width/height/depth/channels for the whole batch,
    // so any mismatch leads to incorrect processing or out-of-bounds reads.
    // If resize is needed, input sizes don't need to match (resize normalizes them).
    bool needResize = (targetSize != image0.size() || param.paddingmode != DNN_PMODE_NULL);

    auto images_size = images.size();

    for (size_t i = 1; i < images_size; ++i)
    {
        if (images[i].dims != 2 ||
            images[i].depth() != image0.depth() ||
            images[i].channels() != image0.channels() ||
            (!needResize && images[i].size() != image0.size()))
        {
            return false;
        }
    }

    std::vector<const uchar*> src_data(images_size);
    std::vector<size_t> src_step(images_size);
    for (size_t i = 0; i < images_size; ++i)
    {
        src_data[i] = images[i].ptr();
        src_step[i] = images[i].step;
    }

    float mean[4] = { (float)param.mean[0], (float)param.mean[1], (float)param.mean[2], (float)param.mean[3] };
    float scalefactor[4] = { (float)param.scalefactor[0], (float)param.scalefactor[1], (float)param.scalefactor[2], (float)param.scalefactor[3] };

    int status = -1;
    int sz[4];
    sz[0] = (int)images_size;
    if(param.datalayout == DNN_LAYOUT_NCHW)
    {
        sz[1] = nch;
        sz[2] = targetSize.height;
        sz[3] = targetSize.width;
    }
    else if(param.datalayout == DNN_LAYOUT_NHWC)
    {
        sz[1] = targetSize.height;
        sz[2] = targetSize.width;
        sz[3] = nch;
    }

    blob.create(4, sz, param.ddepth);
    if (!blob.isContinuous())
        return false;

    status = ipp_hal_blobFromImages(
        src_data.data(), src_step.data(), (int)images_size, image0.cols, image0.rows,
        image0.depth(), nch, param.ddepth, param.swapRB ? 1 : 0,
        targetSize.width, targetSize.height, (int)param.paddingmode, param.borderValue.val,
        mean, scalefactor, blob.ptr(), blob.step.p, param.datalayout, INTER_LINEAR);
    return status == CV_HAL_ERROR_OK;
#else
    CV_UNUSED(images);
    CV_UNUSED(blob);
    CV_UNUSED(param);
    CV_UNUSED(targetSize);
    return false;
#endif
}

CV__DNN_INLINE_NS_END
} // namespace dnn
} // namespace cv
