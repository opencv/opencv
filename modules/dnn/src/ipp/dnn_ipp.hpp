// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_DNN_IPP_HPP
#define OPENCV_DNN_DNN_IPP_HPP

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "ipp_hal_dnn.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

bool blobFromImages_HAL(const std::vector<Mat>& images, Mat& blob, const Image2BlobParams& param, const Size& targetSize);
bool blobFromImages_HAL(const std::vector<UMat>& images, UMat& blob, const Image2BlobParams& param, const Size& targetSize);

CV__DNN_INLINE_NS_END
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_DNN_IPP_HPP
