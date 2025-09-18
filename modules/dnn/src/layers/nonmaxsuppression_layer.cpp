// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <limits>
#include <numeric>

namespace cv {
namespace dnn {

// ONNX NonMaxSuppression (opset >=10)
// Spec: https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html
// Inputs: boxes [B, N, 4], scores [B, C, N], (optional) max_output_boxes_per_class, iou_threshold, score_threshold
// Output: selected_indices [K, 3] (int64): [batch_idx, class_idx, box_idx]

template <typename T>
static inline T readScalarAs(const cv::Mat& m, T def)
{
    if (m.empty())
        return def;
    CV_Assert(m.channels() == 1 && m.total() == 1);

    switch (m.depth())
    {
    case CV_8U:  return static_cast<T>(m.at<uchar>(0));
    case CV_8S:  return static_cast<T>(m.at<schar>(0));
    case CV_16U: return static_cast<T>(m.at<unsigned short>(0));
    case CV_16S: return static_cast<T>(m.at<short>(0));
    case CV_32S: return static_cast<T>(m.at<int>(0));
    case CV_64S: return static_cast<T>(m.at<long long>(0));
    case CV_32F:
        if (std::is_integral<T>::value)
            return static_cast<T>(cvRound(m.at<float>(0)));
        else
            return static_cast<T>(m.at<float>(0));
    case CV_64F:
        if (std::is_integral<T>::value)
            return static_cast<T>(cvRound(m.at<double>(0)));
        else
            return static_cast<T>(m.at<double>(0));
    default:
        return def;
    }
}

class NonMaxSuppressionLayerImpl CV_FINAL : public NonMaxSuppressionLayer
{
public:
    NonMaxSuppressionLayerImpl(const LayerParams& p) {
        setParamsFrom(p);
        centerPointBox = p.get<int>("center_point_box", 0);
        defaultMaxOut = p.get<int>("default_max_output_boxes_per_class", 0);
        defaultIouThr = p.get<float>("default_iou_threshold", 0.f);
        defaultScoreThr = p.get<float>("default_score_threshold", 0.f);
    }

    bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE { return true; }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, const int, std::vector<MatShape>& outputs, std::vector<MatShape>&) const CV_OVERRIDE {
        CV_Assert(inputs.size() >= 2);
        const MatShape& boxes = inputs[0];
        const MatShape& scores = inputs[1];
        CV_Assert(boxes.size() == 3 && boxes[2] == 4);
        CV_Assert(scores.size() == 3 && boxes[0] == scores[0] && boxes[1] == scores[2]);

        outputs.assign(1, MatShape({1, 3}));
        return false;
    }

    void getTypes(const std::vector<MatType>&, const int requiredOutputs, const int requiredInternals,
                   std::vector<MatType>& outputs, std::vector<MatType>& internals) const CV_OVERRIDE {
        outputs.assign(requiredOutputs, MatType(CV_64S));
        internals.assign(requiredInternals, MatType(CV_32F));
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays) CV_OVERRIDE {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& boxesBlob  = inputs[0];
        const Mat& scoresBlob = inputs[1];

        const int B = boxesBlob.size[0];
        const int N = boxesBlob.size[1];
        const int C = scoresBlob.size[1];

        int maxOut = (int)readScalarAs<long long>((inputs.size() >= 3 ? inputs[2] : Mat()), (long long)defaultMaxOut);
        float iouThr = readScalarAs<float>((inputs.size() >= 4 ? inputs[3] : Mat()), defaultIouThr);
        float scoreThr = readScalarAs<float>((inputs.size() >= 5 ? inputs[4] : Mat()), defaultScoreThr);

        const int tasks = B * C;
        std::vector<std::vector<Vec<long long,3>>> tripletsPerBC(tasks);

        cv::parallel_for_(cv::Range(0, tasks), [&](const cv::Range& r){
            for (int taskIdx = r.start; taskIdx < r.end; ++taskIdx) {
                const int b = taskIdx / C;
                const int c = taskIdx % C;

                std::vector<Rect2f> rects;
                std::vector<float> scores;
                std::vector<int> globalIndices;
                rects.reserve(N); scores.reserve(N); globalIndices.reserve(N);

                const float* sPtr = scoresBlob.ptr<float>(b, c);
                for (int n = 0; n < N; ++n) {
                    float sc = sPtr[n];
                    const float* bx = boxesBlob.ptr<float>(b, n);
                    Rect2f r;
                    if (centerPointBox == 0) {
                        const float y1 = bx[0], x1 = bx[1], y2 = bx[2], x2 = bx[3];
                        const float x = std::min(x1, x2), y = std::min(y1, y2);
                        const float w = std::max(0.f, std::abs(x2 - x1));
                        const float h = std::max(0.f, std::abs(y2 - y1));
                        r = Rect2f(x, y, w, h);
                    } else {
                        const float yc = bx[0], xc = bx[1];
                        const float h = std::max(0.f, bx[2]), w = std::max(0.f, bx[3]);
                        r = Rect2f(xc - 0.5f*w, yc - 0.5f*h, w, h);
                    }
                    rects.push_back(r);
                    scores.push_back(sc);
                    globalIndices.push_back(n);
                }

                if (rects.empty())
                    continue;

                std::vector<int> order(rects.size());
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int i, int j){ return scores[i] > scores[j]; });

                std::vector<Rect2d> rects2d; rects2d.reserve(order.size());
                std::vector<float> sortedScores; sortedScores.reserve(order.size());
                for (int idx : order) {
                    const Rect2f& rf = rects[idx];
                    rects2d.emplace_back((double)rf.x, (double)rf.y, (double)rf.width, (double)rf.height);
                    sortedScores.push_back(scores[idx]);
                }
                std::vector<int> keep;
                cv::dnn::NMSBoxes(rects2d, sortedScores, /*score_threshold*/ scoreThr, /*nms_threshold*/ iouThr, keep, 1.f,
                                  0);
                if (maxOut > 0 && (int)keep.size() > maxOut)
                    keep.resize(maxOut);

                auto& local = tripletsPerBC[taskIdx];
                local.reserve(keep.size());
                for (int kept : keep) {
                    const int filteredIdx = order[kept];
                    const int globalIdx = globalIndices[filteredIdx];
                    local.push_back({(long long)b, (long long)c, (long long)globalIdx});
                }
            }
        });

        int K = 0;
        for (int t = 0; t < tasks; ++t) K += (int)tripletsPerBC[t].size();
        outputs_arr.getMatRef(0).create(2, std::array<int,2>{K, 3}.data(), CV_64S);
        auto* out = outputs_arr.getMatRef(0).ptr<long long>();
        std::vector<int> offsets(tasks + 1, 0);
        for (int t = 0; t < tasks; ++t)
            offsets[t + 1] = offsets[t] + (int)tripletsPerBC[t].size() * 3;
        cv::parallel_for_(cv::Range(0, tasks), [&](const cv::Range& r){
            for (int t = r.start; t < r.end; ++t) {
                long long* dst = out + offsets[t];
                const auto& v = tripletsPerBC[t];
                for (const auto& trip : v) {
                    *dst++ = trip[0]; *dst++ = trip[1]; *dst++ = trip[2];
                }
            }
        });
    }

private:
    int   centerPointBox = 0;
    int   defaultMaxOut = 0;
    float defaultIouThr = 0.f;
    float defaultScoreThr = 0.f;
};

Ptr<NonMaxSuppressionLayer> NonMaxSuppressionLayer::create(const LayerParams& params) {
    return Ptr<NonMaxSuppressionLayer>(new NonMaxSuppressionLayerImpl(params));
}

}} // namespace cv::dnn
