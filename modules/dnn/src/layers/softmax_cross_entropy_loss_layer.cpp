// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/softmax.hpp"

// ONNX operator: SoftmaxCrossEntropyLoss
// Spec: https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html
// Supported opsets: 12-23

namespace cv {
namespace dnn {

class SoftmaxCrossEntropyLossImpl CV_FINAL : public SoftmaxCrossEntropyLossLayer
{
public:
    SoftmaxCrossEntropyLossImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        String red = toLowerCase(params.get<String>("reduction", "mean"));
        if (red == "none") reduction = DNN_LOSS_REDUCTION_NONE;
        else if (red == "mean") reduction = DNN_LOSS_REDUCTION_MEAN;
        else if (red == "sum") reduction = DNN_LOSS_REDUCTION_SUM;
        else CV_Error(Error::StsBadArg, "Unsupported reduction: " + red);
        ignoreIndex = params.get<int>("ignore_index", -1);
        labelSmoothing = params.get<float>("label_smoothing", 0.f);
        softLabel = params.get<int>("soft_label", 0) != 0;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& in, const int requiredOutputs,
                     std::vector<MatShape>& out, std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(in.size() >= 2);
        const MatShape& x = in[0];
        CV_Assert(x.size() >= 2);

        if (reduction == DNN_LOSS_REDUCTION_NONE) {
            MatShape y; y.push_back(x[0]);
            for (size_t i = 2; i < x.size(); ++i) y.push_back(x[i]);
            out.push_back(y);
        } else {
            out.push_back(MatShape(1,1));
        }

        if (requiredOutputs >= 2)
            out.push_back(x);

        return false;
    }

    void getTypes(const std::vector<MatType>&, const int requiredOutputs, const int,
                std::vector<MatType>& out, std::vector<MatType>&) const CV_OVERRIDE
    {
        out.clear();
        out.push_back(CV_32F);
        if (requiredOutputs >= 2) out.push_back(CV_32F);
    }

    void forward(InputArrayOfArrays in_arr, OutputArrayOfArrays out_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inp;
        in_arr.getMatVector(inp);

        const Mat& logits = inp[0];
        const Mat& labels = inp[1];
        const bool hasWeight = inp.size() >= 3;
        const Mat& w = hasWeight ? inp[2] : Mat();

        CV_Assert(logits.dims >= 2);
        const int N = logits.size[0], C = logits.size[1];
        int S = 1; for (int i = 2; i < logits.dims; ++i) S *= logits.size[i];
        MatShape logitsShape = shape(logits);
        MatShape lossShape;
        bool isReduced = (reduction != DNN_LOSS_REDUCTION_NONE);
        if (!isReduced) {
            lossShape.push_back(N);
            for (int i = 2; i < logits.dims; ++i) lossShape.push_back(logits.size[i]);
        }

        auto kind = out_arr.kind();
        Mat out_loss, out_logprob;
        bool wantLogProb = false;
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = out_arr.getMatVecRef();
            CV_Assert(outs.size() == 1 || outs.size() == 2);
            wantLogProb = (outs.size() == 2);
            if (!isReduced) {
                outs[0].fit(lossShape, CV_32F);
            }
            out_loss = outs[0];
            if (wantLogProb) {
                outs[1].fit(logitsShape, CV_32F);
                out_logprob = outs[1];
            }
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& uouts = out_arr.getUMatVecRef();
            CV_Assert(uouts.size() == 1 || uouts.size() == 2);
            wantLogProb = (uouts.size() == 2);
            if (!isReduced) {
                uouts[0].fit(lossShape, CV_32F);
            }
            out_loss = uouts[0].getMat(ACCESS_WRITE);
            if (wantLogProb) {
                uouts[1].fit(logitsShape, CV_32F);
                out_logprob = uouts[1].getMat(ACCESS_WRITE);
            }
        } else {
            CV_Error(cv::Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }
        CV_Assert(out_loss.type() == CV_32F);
        if (isReduced) {
            CV_Assert(out_loss.total() == 1);
        }
        const int nstripes = 16;

        Mat logits2D; tensorNCX_to_NSxC(logits, logits2D, nstripes);
        CV_Assert(logits2D.isContinuous());
        Mat logp2D; logp2D.create(logits2D.size(), CV_32F);
        CV_Assert(logp2D.isContinuous());
        logSoftmax(logp2D, logits2D, 1);
        if (wantLogProb) {
            NSxC_to_tensorNCX(logp2D, logits.dims, logits.size.p, out_logprob, nstripes);
        }

        const bool expanded = (labels.dims == logits.dims && labels.size[1] == C) || softLabel;
        const float eps = std::max(0.f, std::min(1.f, labelSmoothing));

        Mat per(N*S, 1, CV_32F, Scalar(0));
        Mat validMask(N*S, 1, CV_8U, Scalar(1));
        Mat effW(N*S, 1, CV_32F, Scalar(1));

        if (!expanded)
        {
            CV_Assert(labels.depth() == CV_32S || labels.depth() == CV_64S);
            CV_Assert((int)labels.total() == N*S);
            Mat idx1D = labels.reshape(1, N*S);

            std::vector<float> meanLogRow;
            float* meanLogRowData = nullptr;
            if (eps > 0.f) {
                meanLogRow.resize(N*S);
                parallel_for_(Range(0, N*S), [&](const Range& rr){
                    for (int r = rr.start; r < rr.end; ++r) {
                        const float* p = logp2D.ptr<float>(r);
                        double s = 0.0;
                        for (int c = 0; c < C; ++c) {
                            s += p[c];
                        }
                        meanLogRow[r] = static_cast<float>(s / C);
                    }
                }, nstripes);
                meanLogRowData = meanLogRow.data();
            }

            if (labels.depth() == CV_32S)
            {
                reduceSCEPerSample<int32_t>(logp2D, idx1D, C, ignoreIndex, eps, meanLogRowData, w, hasWeight, per, effW, validMask, nstripes);
            }
            else
            {
                reduceSCEPerSample<int64_t>(logp2D, idx1D, C, ignoreIndex, eps, meanLogRowData, w, hasWeight, per, effW, validMask, nstripes);
            }
        }
        else
        {
            Mat lab2D; tensorNCX_to_NSxC(labels, lab2D, nstripes);

            if (!softLabel && eps > 0.f) {
                parallel_for_(Range(0, N*S), [&](const Range& rr){
                    for (int r = rr.start; r < rr.end; ++r) {
                        float* a = lab2D.ptr<float>(r);
                        for (int c = 0; c < C; ++c) {
                            a[c] = (1.f - eps) * a[c] + eps / C;
                        }
                    }
                }, nstripes);
            }

            if (hasWeight) {
                for (int c = 0; c < C; ++c) {
                    const float wc = w.at<float>(c);
                    if (wc == 1.f) continue;
                    parallel_for_(Range(0, N*S), [&](const Range& rr){
                        for (int r = rr.start; r < rr.end; ++r) lab2D.at<float>(r, c) *= wc;
                    }, nstripes);
                }
            }

            for (int r = 0; r < N*S; ++r)
            {
                const float* a = lab2D.ptr<float>(r);
                const float* b = logp2D.ptr<float>(r);

                double dot = 0.0, wsum = 0.0;
                for (int c = 0; c < C; ++c) {
                    dot += static_cast<double>(a[c]) * static_cast<double>(b[c]);
                    wsum += static_cast<double>(a[c]);
                }

                if (wsum == 0.0) {
                    validMask.at<uchar>(r) = 0;
                    effW.at<float>(r) = 0.f;
                    per.at<float>(r) = 0.f;
                } else {
                    per.at<float>(r) = static_cast<float>(-dot);
                    effW.at<float>(r) = static_cast<float>(wsum);
                }
            }
        }

        if (!isReduced)
        {
            per.reshape(1, out_loss.size).copyTo(out_loss);
        }
        else
        {
            double num = 0.0, den = 0.0;
            for (int r = 0; r < per.rows; ++r) {
                const float wEff = effW.at<float>(r);
                if (validMask.at<uchar>(r)) {
                    num += per.at<float>(r);
                    if (reduction == DNN_LOSS_REDUCTION_MEAN) den += std::max(1e-12f, wEff);
                }
            }
            const float out = (reduction == DNN_LOSS_REDUCTION_SUM) ? static_cast<float>(num)
                                                : static_cast<float>((den > 0.0) ? (num / den) : 0.0);
            out_loss.at<float>(0) = out;
        }
    }

private:
    template<typename IndexT>
    static inline void reduceSCEPerSample(const Mat& logp2D,
                                          const Mat& idx1D,
                                          const int C,
                                          const int ignoreIndex,
                                          const float eps,
                                          const float* meanLogRowData,
                                          const Mat& w,
                                          const bool hasWeight,
                                          Mat& per,
                                          Mat& effW,
                                          Mat& validMask,
                                          const int nstripes)
    {
        CV_Assert(idx1D.cols == 1);
        parallel_for_(Range(0, idx1D.rows), [&](const Range& rr){
            for (int r = rr.start; r < rr.end; ++r)
            {
                const IndexT y_raw = idx1D.at<IndexT>(r);
                if ((int64_t)y_raw == (int64_t)ignoreIndex)
                {
                    validMask.at<uchar>(r) = 0;
                    effW.at<float>(r) = 0.f;
                    continue;
                }

                const int64_t yi64 = (int64_t)y_raw;
                CV_Assert(yi64 >= 0 && yi64 < (int64_t)C);
                const int y = (int)yi64;

                const float lp = logp2D.at<float>(r, y);
                const float cw = hasWeight ? w.at<float>(y) : 1.f;

                float loss = -(1.f - eps) * lp;
                if (eps > 0.f) loss += -eps * meanLogRowData[r];

                per.at<float>(r)  = loss * cw;
                effW.at<float>(r) =  cw;
            }
        }, nstripes);
    }
};

Ptr<SoftmaxCrossEntropyLossLayer> SoftmaxCrossEntropyLossLayer::create(const LayerParams& params)
{
    return Ptr<SoftmaxCrossEntropyLossLayer>(new SoftmaxCrossEntropyLossImpl(params));
}
}}
