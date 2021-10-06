// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "nms.inl.hpp"

#include <opencv2/imgproc.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta, const int top_k)
{
    CV_Assert_N(bboxes.size() == scores.size(), score_threshold >= 0,
        nms_threshold >= 0, eta > 0);
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

void NMSBoxes(const std::vector<Rect2d>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta, const int top_k)
{
    CV_Assert_N(bboxes.size() == scores.size(), score_threshold >= 0,
        nms_threshold >= 0, eta > 0);
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

static inline float rotatedRectIOU(const RotatedRect& a, const RotatedRect& b)
{
    std::vector<Point2f> inter;
    int res = rotatedRectangleIntersection(a, b, inter);
    if (inter.empty() || res == INTERSECT_NONE)
        return 0.0f;
    if (res == INTERSECT_FULL)
        return 1.0f;
    float interArea = contourArea(inter);
    return interArea / (a.size.area() + b.size.area() - interArea);
}

void NMSBoxes(const std::vector<RotatedRect>& bboxes, const std::vector<float>& scores,
              const float score_threshold, const float nms_threshold,
              std::vector<int>& indices, const float eta, const int top_k)
{
    CV_Assert_N(bboxes.size() == scores.size(), score_threshold >= 0,
        nms_threshold >= 0, eta > 0);
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rotatedRectIOU);
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
    CV_Assert_N(bboxes.size() == scores.size(), score_threshold >= 0,
                nms_threshold >= 0, sigma >= 0);

    indices.clear();
    updated_scores.clear();

    std::vector<std::pair<float, size_t> > score_index_vec(scores.size());
    for (size_t i = 0; i < scores.size(); i++)
    {
        score_index_vec[i].first = scores[i];
        score_index_vec[i].second = i;
    }

    const auto score_cmp = [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b)
    {
        return a.first == b.first ? a.second > b.second : a.first < b.first;
    };

    top_k = top_k == 0 ? scores.size() : std::min(top_k, scores.size());
    ptrdiff_t start = 0;
    while (indices.size() < top_k)
    {
        auto it = std::max_element(score_index_vec.begin() + start, score_index_vec.end(), score_cmp);

        float bscore = it->first;
        size_t bidx = it->second;

        if (bscore < score_threshold)
        {
            break;
        }

        indices.push_back(static_cast<int>(bidx));
        updated_scores.push_back(bscore);
        std::swap(score_index_vec[start], *it); // first start elements are chosen

        for (size_t i = start + 1; i < scores.size(); ++i)
        {
            float& bscore_i = score_index_vec[i].first;
            const size_t bidx_i = score_index_vec[i].second;

            if (bscore_i < score_threshold)
            {
                continue;
            }

            float overlap = rectOverlap(bboxes[bidx], bboxes[bidx_i]);

            switch (method)
            {
                case SoftNMSMethod::SOFTNMS_LINEAR:
                    if (overlap > nms_threshold)
                    {
                        bscore_i *= 1.f - overlap;
                    }
                    break;
                case SoftNMSMethod::SOFTNMS_GAUSSIAN:
                    bscore_i *= exp(-(overlap * overlap) / sigma);
                    break;
                default:
                    CV_Error(Error::StsBadArg, "Not supported SoftNMS method.");
            }
        }
        ++start;
    }
}

CV__DNN_INLINE_NS_END
}// dnn
}// cv
