// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/imgproc.hpp"
#include "opencv2/dnn/nms.private.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta, const int top_k)
{
    cv::NMSBoxes(bboxes, scores, score_threshold, nms_threshold, indices, eta, top_k);
}

void NMSBoxes(const std::vector<Rect2d>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta, const int top_k)
{
    cv::NMSBoxes(bboxes, scores, score_threshold, nms_threshold, indices, eta, top_k);
}

void NMSBoxes(const std::vector<RotatedRect>& bboxes, const std::vector<float>& scores,
              const float score_threshold, const float nms_threshold,
              std::vector<int>& indices, const float eta, const int top_k)
{
    cv::NMSBoxes(bboxes, scores, score_threshold, nms_threshold, indices, eta, top_k);
}

void NMSBoxesBatched(const std::vector<Rect>& bboxes,
                     const std::vector<float>& scores, const std::vector<int>& class_ids,
                     const float score_threshold, const float nms_threshold,
                     std::vector<int>& indices, const float eta, const int top_k)
{
    cv::NMSBoxesBatched(bboxes, scores,  class_ids, score_threshold, nms_threshold, indices, eta, top_k);
}

void NMSBoxesBatched(const std::vector<Rect2d>& bboxes,
                     const std::vector<float>& scores, const std::vector<int>& class_ids,
                     const float score_threshold, const float nms_threshold,
                     std::vector<int>& indices, const float eta, const int top_k)
{
    cv::NMSBoxesBatched(bboxes, scores,  class_ids, score_threshold, nms_threshold, indices, eta, top_k);
}

void softNMSBoxes(const std::vector<Rect>& bboxes,
                  const std::vector<float>& scores,
                  std::vector<float>& updated_scores,
                  const float score_threshold,
                  const float nms_threshold,
                  std::vector<int>& indices,
                  size_t top_k,
                  const float sigma,
                  SoftNMSMethod method)
{
    cv::softNMSBoxes(bboxes, scores, updated_scores, score_threshold, nms_threshold, indices, top_k, sigma, method);
}

CV__DNN_INLINE_NS_END
}// dnn
}// cv
