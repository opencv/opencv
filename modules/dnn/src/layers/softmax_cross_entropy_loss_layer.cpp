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
        Mat logits32; logits.convertTo(logits32, CV_32F);
        const size_t sN = logits32.step1(0);
        const size_t sC = logits32.step1(1);

        std::vector<float> rowLogSumExp(N*S);
        std::vector<float> meanLogRow(N*S);

        bool writeLogProb = (wantLogProb);
        size_t sN_out = 0, sC_out = 0;
        float* outlpBase = nullptr;
        if (writeLogProb) {
            CV_Assert(out_logprob.type() == CV_32F);
            sN_out = out_logprob.step1(0);
            sC_out = out_logprob.step1(1);
            outlpBase = out_logprob.ptr<float>();
        }

        parallel_for_(Range(0, N*S), [&](const Range& rr){
            const float* base = logits32.ptr<float>();
            for (int r = rr.start; r < rr.end; ++r) {
                const int n = r / S;
                const int s = r % S;
                const float* rowBase = base + n * sN + s;

                float maxv = rowBase[0];
                for (int c = 1; c < C; ++c) maxv = std::max(maxv, rowBase[c * sC]);

                double sumExp = 0.0;
                double sumLogits = 0.0;
                for (int c = 0; c < C; ++c) {
                    const float v = rowBase[c * sC];
                    sumExp += std::exp(static_cast<double>(v - maxv));
                    sumLogits += v;
                }
                const float lse = static_cast<float>(std::log(sumExp) + maxv);
                rowLogSumExp[r] = lse;
                meanLogRow[r] = static_cast<float>(sumLogits / C) - lse;

                if (writeLogProb) {
                    float* dst = outlpBase + n * sN_out + s;
                    for (int c = 0; c < C; ++c) dst[c * sC_out] = rowBase[c * sC] - lse;
                }
            }
        }, nstripes);

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

            const float* wptr = hasWeight ? w.ptr<float>() : nullptr;
            if (labels.depth() == CV_32S)
            {
                if (hasWeight)
                    reduceSCEPerSample<int32_t, true>(logits32, sN, sC, S, idx1D, C, ignoreIndex, eps, meanLogRow.data(), rowLogSumExp.data(), wptr, per, effW, validMask, nstripes);
                else
                    reduceSCEPerSample<int32_t, false>(logits32, sN, sC, S, idx1D, C, ignoreIndex, eps, meanLogRow.data(), rowLogSumExp.data(), wptr, per, effW, validMask, nstripes);
            }
            else
            {
                if (hasWeight)
                    reduceSCEPerSample<int64_t, true>(logits32, sN, sC, S, idx1D, C, ignoreIndex, eps, meanLogRow.data(), rowLogSumExp.data(), wptr, per, effW, validMask, nstripes);
                else
                    reduceSCEPerSample<int64_t, false>(logits32, sN, sC, S, idx1D, C, ignoreIndex, eps, meanLogRow.data(), rowLogSumExp.data(), wptr, per, effW, validMask, nstripes);
            }
        }
        else
        {
            Mat labels32; labels.convertTo(labels32, CV_32F);
            const size_t sN_lab = labels32.step1(0);
            const size_t sC_lab = labels32.step1(1);

            const float smoothMul = (!softLabel && eps > 0.f) ? (1.f - eps) : 1.f;
            const float smoothAdd = (!softLabel && eps > 0.f) ? (eps / C) : 0.f;
            std::vector<float> onesW;
            const float* wBase = nullptr;
            if (hasWeight){
                wBase = w.ptr<float>();
            } else {
                onesW.assign(C, 1.f);
                wBase = onesW.data();
            }

            parallel_for_(Range(0, N*S), [&](const Range& rr){
                const float* baseLogits = logits32.ptr<float>();
                const float* baseLabels = labels32.ptr<float>();
                const float* wlocal = wBase;
                for (int r = rr.start; r < rr.end; ++r) {
                    const int n = r / S;
                    const int s = r % S;
                    const float* arow = baseLabels + n * sN_lab + s;
                    const float* lrow = baseLogits + n * sN + s;

                    double dot = 0.0, wsum = 0.0;
                    for (int c = 0; c < C; ++c) {
                        float a = arow[c * sC_lab] * smoothMul + smoothAdd;
                        a *= wlocal[c];
                        wsum += static_cast<double>(a);
                        dot  += static_cast<double>(a) * static_cast<double>(lrow[c * sC]);
                    }

                    const float m = (wsum != 0.0) ? 1.f : 0.f;
                    validMask.at<uchar>(r) = static_cast<uchar>(m > 0.f);
                    effW.at<float>(r) = static_cast<float>(wsum * m);
                    per.at<float>(r) = static_cast<float>(-(dot - static_cast<double>(rowLogSumExp[r]) * wsum) * m);
                }
            }, nstripes);
        }

        if (!isReduced)
        {
            per.reshape(1, out_loss.size).copyTo(out_loss);
        }
        else
        {
            double num = 0.0, den = 0.0;
            const float* perData = per.ptr<float>();
            const float* effData = effW.ptr<float>();
            const uchar* validData = validMask.ptr<uchar>();
            const int R = per.rows;

            for (int r = 0; r < R; ++r) {
                const float m = validData[r] ? 1.f : 0.f;
                num += static_cast<double>(perData[r] * m);
                if (reduction == DNN_LOSS_REDUCTION_MEAN)
                {
                    den += static_cast<double>(std::max(1e-12f, effData[r]) * m);
                }
            }

            const float out = (reduction == DNN_LOSS_REDUCTION_SUM) ? static_cast<float>(num)
                                                 : static_cast<float>((den > 0.0) ? (num / den) : 0.0);
            out_loss.at<float>(0) = out;
        }
    }

private:
    template<typename T, bool UseWeight>
    static inline void reduceSCEPerSample(const Mat& logits32,
                                          const size_t sN,
                                          const size_t sC,
                                          const int S,
                                          const Mat& idx1D,
                                          const int C,
                                          const int ignoreIndex,
                                          const float eps,
                                          const float* meanLogRowData,
                                          const float* rowLogSumExp,
                                          const float* weightBase,
                                          Mat& per,
                                          Mat& effW,
                                          Mat& validMask,
                                          const int nstripes)
    {
        CV_Assert(idx1D.cols == 1);
        parallel_for_(Range(0, idx1D.rows), [&](const Range& rr){
            const float* base = logits32.ptr<float>();
            for (int r = rr.start; r < rr.end; ++r)
            {
                const T y_raw = idx1D.at<T>(r);
                if ((int64_t)y_raw == (int64_t)ignoreIndex)
                {
                    validMask.at<uchar>(r) = 0;
                    effW.at<float>(r) = 0.f;
                    continue;
                }

                const int64_t yi64 = (int64_t)y_raw;
                CV_Assert(yi64 >= 0 && yi64 < (int64_t)C);
                const int y = static_cast<int>(yi64);

                const int n = r / S;
                const int s = r % S;
                const float logits_y = base[n * sN + y * sC + s];
                const float lp_y = logits_y - rowLogSumExp[r];
                const float cw = UseWeight ? weightBase[y] : 1.f;

                float loss = -(1.f - eps) * lp_y;
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
