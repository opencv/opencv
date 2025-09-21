// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

// Opset's 12 to 23 are covered.

namespace cv {
namespace dnn {

class SoftmaxCrossEntropyLossImpl CV_FINAL : public SoftmaxCrossEntropyLossLayer
{
public:
    SoftmaxCrossEntropyLossImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        reduction = params.get<String>("reduction", "mean");
        ignore_index = params.get<int>("ignore_index", -1);
        label_smoothing = params.get<float>("label_smoothing", 0.f);
        soft_label = params.get<int>("soft_label", 0) != 0;
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

        if (reduction == "none") {
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
        std::vector<Mat> inp, outv;
        in_arr.getMatVector(inp); out_arr.getMatVector(outv);

        const Mat& logits = inp[0];
        const Mat& labels = inp[1];
        const bool hasWeight = inp.size() >= 3;
        const Mat& w = hasWeight ? inp[2] : Mat();

        CV_Assert(logits.dims >= 2);
        const int N = logits.size[0], C = logits.size[1];
        int S = 1; for (int i = 2; i < logits.dims; ++i) S *= logits.size[i];
        const bool wantLogProb = (outv.size() >= 2);

        Mat logits2D; tensorNCX_to_NSxC(logits, logits2D);
        Mat logp2D;   computeLogSoftmax(logits2D, logp2D);
        if (wantLogProb)
            NSxC_to_tensorNCX(logp2D, logits.dims, logits.size.p, outv[1]);

        const bool expanded = (labels.dims == logits.dims && labels.size[1] == C) || soft_label;
        const float eps = std::max(0.f, std::min(1.f, label_smoothing));

        Mat per(N*S, 1, CV_32F, Scalar(0));
        Mat validMask(N*S, 1, CV_8U, Scalar(1));
        Mat effW(N*S, 1, CV_32F, Scalar(1));

        if (!expanded)
        {
            Mat idx; labels.convertTo(idx, CV_32S);
            Mat idx1D = idx.reshape(1, N*S).clone();

            std::vector<float> meanLogRow;
                        if (eps > 0.f) {
            meanLogRow.resize(N*S);
            parallel_for_(Range(0, N*S), [&](const Range& rr){
                for (int r = rr.start; r < rr.end; ++r) {
                    const float* p = logp2D.ptr<float>(r);
                    double s = 0.0; for (int c = 0; c < C; ++c) s += p[c];
                    meanLogRow[r] = (float)(s / C);
                }
            }, 16);
        }

        parallel_for_(Range(0, N*S), [&](const Range& rr){
            for (int r = rr.start; r < rr.end; ++r)
            {
                const int y = idx1D.at<int>(r);
                if (y == ignore_index) { validMask.at<uchar>(r) = 0; effW.at<float>(r) = 0.f; continue; }
                CV_Assert(0 <= y && y < C);

                const float lp = logp2D.at<float>(r, y);
                const float cw = hasWeight ? w.at<float>(y) : 1.f;

                float loss = -(1.f - eps) * lp;
                if (eps > 0.f) loss += -eps * meanLogRow[r];

                per.at<float>(r) = loss * cw;
                effW.at<float>(r) = cw;
            }
        }, 16);
        }
        else
        {
            Mat lab2D; tensorNCX_to_NSxC(labels, lab2D);

            if (!soft_label && eps > 0.f) {
                            parallel_for_(Range(0, N*S), [&](const Range& rr){
                for (int r = rr.start; r < rr.end; ++r) {
                    float* a = lab2D.ptr<float>(r);
                    for (int c = 0; c < C; ++c)
                        a[c] = (1.f - eps) * a[c] + eps / C;
                }
            }, 16);
            }

            if (hasWeight) {
                            for (int c = 0; c < C; ++c) {
                const float wc = w.at<float>(c);
                if (wc == 1.f) continue;
                parallel_for_(Range(0, N*S), [&](const Range& rr){
                    for (int r = rr.start; r < rr.end; ++r) lab2D.at<float>(r, c) *= wc;
                }, 16);
            }
            }

            for (int r = 0; r < N*S; ++r)
            {
                const float* a = lab2D.ptr<float>(r);
                const float* b = logp2D.ptr<float>(r);

                double dot = 0.0, wsum = 0.0;
                for (int c = 0; c < C; ++c) { dot += (double)a[c] * (double)b[c]; wsum += (double)a[c]; }

                if (wsum == 0.0) {
                    validMask.at<uchar>(r) = 0;
                    effW.at<float>(r) = 0.f;
                    per.at<float>(r) = 0.f;
                } else {
                    per.at<float>(r) = (float)(-dot);
                    effW.at<float>(r) = (float)wsum;
                }
            }
        }

        if (reduction == "none")
        {
            std::vector<int> outShape; outShape.push_back(N);
            for (int i = 2; i < logits.dims; ++i) outShape.push_back(logits.size[i]);
            outv[0].create((int)outShape.size(), outShape.data(), CV_32F);
            per.reshape(1, outv[0].size).copyTo(outv[0]);
        }
        else
        {
            double num = 0.0, den = 0.0;
            for (int r = 0; r < per.rows; ++r) {
                const float wEff = effW.at<float>(r);
                if (validMask.at<uchar>(r)) {
                    num += per.at<float>(r);
                    if (reduction == "mean") den += std::max(1e-12f, wEff);
                }
            }
            const float out = (reduction == "sum") ? (float)num
                                                : (float)((den > 0.0) ? (num / den) : 0.0);
            outv[0].create(1, 1, CV_32F);
            outv[0].at<float>(0) = out;
        }
    }

    static inline void tensorNCX_to_NSxC(const Mat& src, Mat& dst, int nstripes = 16)
    {
        CV_Assert(src.dims >= 2);
        const int N = src.size[0], C = src.size[1];
        int S = 1; for (int i = 2; i < src.dims; ++i) S *= src.size[i];

        Mat src32; src.convertTo(src32, CV_32F);
        int sz3[3] = { N, C, S };
        Mat src3 = src32.reshape(1, 3, sz3);

        dst.create(N * S, C, CV_32F);

        const size_t sN = src3.step1(0); // stride when n++
        const size_t sC = src3.step1(1); // stride when c++

        const float* srcData = src3.ptr<float>();
        parallel_for_(Range(0, N * S), [&](const Range& r){
            for (int row = r.start; row < r.end; ++row)
            {
                const int n = row / S;
                const int s = row % S;

                float* dstRow = dst.ptr<float>(row);
                const float* nBase = srcData + n * sN;
                for (int c = 0; c < C; ++c)
                    dstRow[c] = nBase[c * sC + s];
            }
        }, nstripes);
    }

    static inline void NSxC_to_tensorNCX(const Mat& src2D, int ndims, const int* sizes, Mat& dst, int nstripes = 16)
    {
        CV_Assert(ndims >= 2);
        const int N = sizes[0], C = sizes[1];
        int S = 1; for (int i = 2; i < ndims; ++i) S *= sizes[i];

        dst.create(ndims, sizes, CV_32F);

        int sz3[3] = { N, C, S };
        Mat dst3 = dst.reshape(1, 3, sz3);

        const size_t sN = dst3.step1(0);
        const size_t sC = dst3.step1(1);
        float* dstData = dst3.ptr<float>();

        parallel_for_(Range(0, N * S), [&](const Range& r){
            for (int row = r.start; row < r.end; ++row)
            {
                const int n = row / S;
                const int s = row % S;

                const float* srcRow = src2D.ptr<float>(row);
                float* nBase = dstData + n * sN;
                for (int c = 0; c < C; ++c)
                    nBase[c * sC + s] = srcRow[c];
            }
        }, nstripes);
    }


private:
    String reduction;
    int ignore_index;
    float label_smoothing;
    bool soft_label;

    static void computeLogSoftmax(const Mat& logits2D, Mat& logp2D)
    {
        logp2D.create(logits2D.size(), CV_32F);
        parallel_for_(Range(0, logits2D.rows), [&](const Range& rr){
            for (int r = rr.start; r < rr.end; ++r)
            {
                const float* x = logits2D.ptr<float>(r);
                float m = x[0]; for (int c=1;c<logits2D.cols;++c) m = std::max(m, x[c]);
                double sum = 0.0;
                for (int c=0;c<logits2D.cols;++c) sum += std::exp((double)x[c] - m);
                float logZ = (float)(std::log(sum) + m);
                float* y = logp2D.ptr<float>(r);
                for (int c=0;c<logits2D.cols;++c) y[c] = x[c] - logZ;
            }
        }, 16);
    }
};

Ptr<SoftmaxCrossEntropyLossLayer> SoftmaxCrossEntropyLossLayer::create(const LayerParams& params)
{
    return Ptr<SoftmaxCrossEntropyLossLayer>(new SoftmaxCrossEntropyLossImpl(params));
}

}} // namespace
