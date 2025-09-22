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
        if (red == "none") reduction = DNN_LOSS_REDUCTION_NONE;
        else if (red == "mean") reduction = DNN_LOSS_REDUCTION_MEAN;
        else if (red == "sum") reduction = DNN_LOSS_REDUCTION_SUM;
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
        if (reduction == DNN_LOSS_REDUCTION_NONE)
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
        bool isReduced = (reduction != DNN_LOSS_REDUCTION_NONE);
        if (!isReduced) {
            lossShape.push_back(N);
            for (int i = 2; i < logp.dims; ++i) lossShape.push_back(logp.size[i]);
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

        auto weightAt = [&](int c) -> float {
            if (!hasWeight) return 1.f;
            CV_DbgAssert(0 <= c && c < C);
            return w_f.at<float>(c, 0);
        };

        const int nstripes = 16;
        Mat logp2D; tensorNCX_to_NSxC(logp, logp2D, nstripes);

        CV_Assert(label.depth() == CV_32S || label.depth() == CV_64S);
        CV_Assert((int)label.total() == N*S);
        Mat idx1D = label.reshape(1, N*S);

        Mat per(N*S, 1, CV_32F, Scalar(0));
        Mat validMask(N*S, 1, CV_8U, Scalar(1));
        Mat effW(N*S, 1, CV_32F, Scalar(1));

        if (label.depth() == CV_32S)
        {
            reduceNLLPerSample<int32_t>(logp2D, idx1D, C, ignoreIndex, weightAt, per, effW, validMask, nstripes);
        }
        else
        {
            reduceNLLPerSample<int64_t>(logp2D, idx1D, C, ignoreIndex, weightAt, per, effW, validMask, nstripes);
        }

        if (!isReduced)
        {
            per.reshape(1, out_loss.size).copyTo(out_loss);
        }
        else
        {
            double num = 0.0, den = 0.0;
            for (int r = 0; r < per.rows; ++r) {
                if (!validMask.at<uchar>(r)) continue;
                num += per.at<float>(r);
                if (reduction == DNN_LOSS_REDUCTION_MEAN) den += std::max(1e-12f, effW.at<float>(r));
            }
            const float out = (reduction == DNN_LOSS_REDUCTION_SUM) ? static_cast<float>(num) : static_cast<float>((den > 0.0) ? (num / den) : 0.0);
            out_loss.at<float>(0) = out;
        }
    }

private:
    template<typename IndexT, typename WeightGetter>
    static inline void reduceNLLPerSample(const Mat& logp2D,
                                          const Mat& idx1D,
                                          const int C,
                                          const int ignoreIndex,
                                          WeightGetter weightAt,
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
                const float cw = weightAt(y);

                per.at<float>(r)  = -lp * cw;
                effW.at<float>(r) =  cw;
            }
        }, nstripes);
    }
};

Ptr<NegativeLogLikelihoodLossLayer> NegativeLogLikelihoodLossLayer::create(const LayerParams& params)
{
    return Ptr<NegativeLogLikelihoodLossLayer>(new NegativeLogLikelihoodLossImpl(params));
}
}}
