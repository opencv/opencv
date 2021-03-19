// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_NMS_INL_HPP
#define OPENCV_DNN_NMS_INL_HPP

#include <opencv2/dnn.hpp>

namespace cv {
namespace dnn {

namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    CV_DbgAssert(score_index_vec.empty());
    // Generate index score pairs.
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

// Do non maximum suppression given bboxes and scores.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    top_k: if not > 0, keep at most top_k picked indices.
//    limit: early terminate once the # of picked indices has reached it.
//    indices: the kept indices of bboxes after nms.
template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices,
      float (*computeOverlap)(const BoxType&, const BoxType&),
      int limit = std::numeric_limits<int>::max())
{
    CV_Assert(bboxes.size() == scores.size());

    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep) {
            indices.push_back(idx);
            if (indices.size() >= limit) {
                break;
            }
        }
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}

enum SoftNMSMethod{ SOFTNMS_LINEAR = 1, SOFTNMS_GAUSSIAN = 2 };

static inline void updateScore(float& score, float IOU, float sigma) {
    if (IOU <= 0.0)
        return;
    score = score * exp(-(IOU * IOU) / sigma);
}

static inline void updateScore(float& score, float IOU) {
    score = score * (1.0 - IOU);
}

// Implement soft NMS.
// https://arxiv.org/abs/1704.04503
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    updated_scores: the adjusted scores after soft NMS.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    top_k: if not > 0, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
//    sigma: parameter of Gaussian weighting.
//    soft_nms_threshold: threshold once below it box will be discarded.
//    method: Gaussian or linear.
template <typename BoxType>
void SoftNMS_(
      const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores,
      std::vector<float>& updated_scores,
      const float score_threshold,
      const float nms_threshold,
      const int top_k,
      std::vector<int>& indices,
      float (*computeOverlap)(const BoxType&, const BoxType&),
      const float sigma = 0.5,
      const float soft_nms_threshold = 0.001,
      SoftNMSMethod method = SOFTNMS_GAUSSIAN)
{
    CV_Assert(bboxes.size() == scores.size());

    std::vector<std::pair<float, int> > score_index_vec(scores.size());
    for (size_t i = 0; i < scores.size(); i++) {
        score_index_vec[i].first = scores[i];
        score_index_vec[i].second = i;
    }

    int top_k_ = top_k == 0 ? scores.size() : (top_k < scores.size() ? top_k : scores.size());
    float threshold = score_threshold > soft_nms_threshold ? score_threshold : soft_nms_threshold;
    int s = 0; // start index; elements before it has been ouput.
    while (indices.size() < top_k_) {
        std::vector<std::pair<float, int> >::iterator it = 
            std::max_element(std::begin(score_index_vec) + s, std::end(score_index_vec));
        if (it->first < threshold) {
            break;
        }
        int bidx = it->second; // box index
        indices.push_back(bidx);
        int idx = std::distance(std::begin(score_index_vec), it);
        std::swap(score_index_vec[s], score_index_vec[idx]);
        for (int i = s + 1; i < scores.size(); i++) {
            if (score_index_vec[i].first < threshold) {
                continue;
            }
            int bidx_i = score_index_vec[i].second;
            float overlap = computeOverlap(bboxes[bidx], bboxes[bidx_i]);
            if ((method == SOFTNMS_LINEAR) && (overlap > nms_threshold)) {
                updateScore(score_index_vec[bidx_i].first, overlap);
            } else {
                CV_Assert(method == SOFTNMS_GAUSSIAN);
                updateScore(score_index_vec[bidx_i].first, overlap, sigma);
            }
        }
        s++;
    }
    updated_scores.resize(scores.size());
    for (size_t i = 0; i < score_index_vec.size(); i++) {
        updated_scores[score_index_vec[i].second] = score_index_vec[i].first;
    }
}

}// dnn
}// cv

#endif
