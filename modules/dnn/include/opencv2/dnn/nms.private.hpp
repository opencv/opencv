// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_DNN_HPP
#define OPENCV_DNN_DNN_HPP

#include <vector>
#include <opencv2/core.hpp>
#include "../dnn/version.hpp"

namespace cv {
namespace dnn {

CV__DNN_INLINE_NS_BEGIN


CV_EXPORTS void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                            const float score_threshold, const float nms_threshold,
                            CV_OUT std::vector<int>& indices,
                            const float eta = 1.f, const int top_k = 0);

CV_EXPORTS_W void NMSBoxes(const std::vector<Rect2d>& bboxes, const std::vector<float>& scores,
                            const float score_threshold, const float nms_threshold,
                            CV_OUT std::vector<int>& indices,
                            const float eta = 1.f, const int top_k = 0);

CV_EXPORTS_AS(NMSBoxesRotated) void NMSBoxes(const std::vector<RotatedRect>& bboxes, const std::vector<float>& scores,
                            const float score_threshold, const float nms_threshold,
                            CV_OUT std::vector<int>& indices,
                            const float eta = 1.f, const int top_k = 0);

CV_EXPORTS void NMSBoxesBatched(const std::vector<Rect>& bboxes, const std::vector<float>& scores, const std::vector<int>& class_ids,
                                const float score_threshold, const float nms_threshold,
                                CV_OUT std::vector<int>& indices,
                                const float eta = 1.f, const int top_k = 0);

CV_EXPORTS_W void NMSBoxesBatched(const std::vector<Rect2d>& bboxes, const std::vector<float>& scores, const std::vector<int>& class_ids,
                                    const float score_threshold, const float nms_threshold,
                                    CV_OUT std::vector<int>& indices,
                                    const float eta = 1.f, const int top_k = 0);

CV_EXPORTS_W void softNMSBoxes(const std::vector<Rect>& bboxes,
                                const std::vector<float>& scores,
                                CV_OUT std::vector<float>& updated_scores,
                                const float score_threshold,
                                const float nms_threshold,
                                CV_OUT std::vector<int>& indices,
                                size_t top_k = 0,
                                const float sigma = 0.5,
                                SoftNMSMethod method = SoftNMSMethod::SOFTNMS_GAUSSIAN);

CV__DNN_INLINE_NS_END
}
}

#endif  /* OPENCV_DNN_DNN_HPP */
