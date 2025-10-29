// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

// ONNX operator: NegativeLogLikelihoodLoss
// Spec: https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html
// Supported opsets: 12-23

namespace cv {
namespace dnn {

class NegativeLogLikelihoodLossImpl CV_FINAL : public NegativeLogLikelihoodLossLayer
{
public:
    NegativeLogLikelihoodLossImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        String red = toLowerCase(params.get<String>("reduction", "mean"));
        if (red == "none") reduction = LOSS_REDUCTION_NONE;
        else if (red == "mean") reduction = LOSS_REDUCTION_MEAN;
        else if (red == "sum") reduction = LOSS_REDUCTION_SUM;
        else CV_Error(Error::StsBadArg, "Unsupported reduction: " + red);
        ignoreIndex = params.get<int>("ignore_index", -1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& in, const int reqOut,
                         std::vector<MatShape>& out, std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(in.size() >= 2);
        const MatShape& x = in[0];
        CV_Assert(x.size() >= 2);
        if (reduction == LOSS_REDUCTION_NONE)
        {
            MatShape shp = x;
            MatShape y; y.reserve(shp.size()-1);
            y.push_back(shp[0]);
            for (size_t i = 2; i < shp.size(); ++i) {
                y.push_back(shp[i]);
            }
            out.assign(1, y);
        }
        else
        {
            out.assign(1, MatShape(1, 1));
        }
        return false;
    }

    void getTypes(const std::vector<MatType>&, const int reqOut, const int reqInt,
                  std::vector<MatType>& out, std::vector<MatType>& internals) const CV_OVERRIDE
    {
        out.assign(1, MatType(CV_32F));
        internals.assign(reqInt, MatType(CV_32F));
    }

    void forward(InputArrayOfArrays in_arr, OutputArrayOfArrays out_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inp;
        in_arr.getMatVector(inp);

        const Mat& logp  = inp[0];
        const Mat& label = inp[1];
        const bool hasWeight = inp.size() >= 3;

        CV_Assert(logp.dims >= 2);
        const int N = logp.size[0], C = logp.size[1];
        int S = 1; for (int i = 2; i < logp.dims; ++i) S *= logp.size[i];

        MatShape lossShape;
        bool isReduced = (reduction != LOSS_REDUCTION_NONE);
        if (!isReduced) {
            lossShape.push_back(N);
            for (int i = 2; i < logp.dims; ++i) {
                lossShape.push_back(logp.size[i]);
            }
        }

        auto kind = out_arr.kind();
        Mat out_loss;
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = out_arr.getMatVecRef();
            CV_Assert(outs.size() == 1);
            if (!isReduced) {
                outs[0].fit(lossShape, CV_32F);
            }
            out_loss = outs[0];
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& uouts = out_arr.getUMatVecRef();
            CV_Assert(uouts.size() == 1);
            if (!isReduced) {
                uouts[0].fit(lossShape, CV_32F);
            }
            out_loss = uouts[0].getMat(ACCESS_WRITE);
        } else {
            CV_Error(cv::Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }
        CV_Assert(out_loss.type() == CV_32F);
        if (isReduced) {
            CV_Assert(out_loss.total() == 1);
        }

        Mat w_f;
        if (hasWeight) {
            Mat wsrc = inp[2];
            CV_Assert(wsrc.total() == (size_t)C);
            Mat wflat = wsrc.reshape(1, (int)wsrc.total()).clone();
            wflat.convertTo(w_f, CV_32F);
        }

        const float* wfDataPtr = hasWeight ? w_f.ptr<float>() : nullptr;

        const int nstripes = 16;
        Mat logp32; logp.convertTo(logp32, CV_32F);
        const size_t sN = logp32.step1(0);
        const size_t sC = logp32.step1(1);

        CV_Assert(label.depth() == CV_32S || label.depth() == CV_64S);
        CV_Assert((int)label.total() == N*S);
        Mat idx1D = label.reshape(1, N*S);

        Mat per(N*S, 1, CV_32F, Scalar(0));
        Mat validMask(N*S, 1, CV_8U, Scalar(1));
        Mat effW(N*S, 1, CV_32F, Scalar(1));

        if (label.depth() == CV_32S)
        {
            reduceNLLPerSample<int32_t>(logp32, sN, sC, S, idx1D, C, ignoreIndex, hasWeight, wfDataPtr, per, effW, validMask, nstripes);
        }
        else
        {
            reduceNLLPerSample<int64_t>(logp32, sN, sC, S, idx1D, C, ignoreIndex, hasWeight, wfDataPtr, per, effW, validMask, nstripes);
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

                if (reduction == LOSS_REDUCTION_MEAN)
                {
                    den += static_cast<double>(std::max(1e-12f, effData[r]) * m);
                }
            }
            const float out = (reduction == LOSS_REDUCTION_SUM) ? static_cast<float>(num) : static_cast<float>((den > 0.0) ? (num / den) : 0.0);
            out_loss.at<float>(0) = out;
        }
    }

private:
    template<typename T>
    static inline void reduceNLLPerSample(const Mat& logp32,
                                          const size_t sN_,
                                          const size_t sC_,
                                          const int S_,
                                          const Mat& idx1D,
                                          const int C_,
                                          const int ignoreIndex_,
                                          const bool hasWeight_,
                                          const float* wfData_,
                                          Mat& per,
                                          Mat& effW,
                                          Mat& validMask,
                                          const int nstripes)
    {
        CV_Assert(idx1D.cols == 1);
        parallel_for_(Range(0, idx1D.rows), [&](const Range& rr){
            const float* base = logp32.ptr<float>();
            const T* idx1DData = idx1D.ptr<T>();
            uchar* validMaskData = validMask.ptr<uchar>();
            float* effWData = effW.ptr<float>();
            float* perData = per.ptr<float>();

            const size_t sIdx = idx1D.step1(0);
            const size_t sVM = validMask.step1(0);
            const size_t sEff = effW.step1(0);
            const size_t sPer = per.step1(0);

            const size_t sN = sN_;
            const size_t sC = sC_;
            const int S = S_;
            const int C = C_;
            const int ignoreIndex = ignoreIndex_;
            const bool hasWeight = hasWeight_;
            const float* wfData = wfData_;

            for (int r = rr.start; r < rr.end; ++r)
            {
                const T y_raw = idx1DData[r * sIdx];
                if ((int64_t)y_raw == (int64_t)ignoreIndex)
                {
                    validMaskData[r * sVM] = 0;
                    effWData[r * sEff] = 0.f;
                    continue;
                }

                const int64_t yi64 = (int64_t)y_raw;
                CV_Assert(yi64 >= 0 && yi64 < (int64_t)C);
                const int y = static_cast<int>(yi64);

                const int n = r / S;
                const int s = r % S;
                const float lp = base[n * sN + y * sC + s];
                const float cw = hasWeight ? wfData[y] : 1.f;

                perData[r * sPer]  = -lp * cw;
                effWData[r * sEff] =  cw;
            }
        }, nstripes);
    }
};

Ptr<NegativeLogLikelihoodLossLayer> NegativeLogLikelihoodLossLayer::create(const LayerParams& params)
{
    return Ptr<NegativeLogLikelihoodLossLayer>(new NegativeLogLikelihoodLossImpl(params));
}
}}
