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

class NegativeLogLikelihoodLossImpl CV_FINAL : public NegativeLogLikelihoodLossLayer
{
public:
    NegativeLogLikelihoodLossImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        reduction = params.get<String>("reduction", "mean");
        ignore_index = params.get<int>("ignore_index", -1);
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
        if (reduction == "none")
        {
            MatShape shp = x;
            MatShape y; y.reserve(shp.size()-1);
            y.push_back(shp[0]);
            for (size_t i = 2; i < shp.size(); ++i) y.push_back(shp[i]);
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
        std::vector<Mat> inp, outv;
        in_arr.getMatVector(inp); out_arr.getMatVector(outv);

        const Mat& logp  = inp[0];
        const Mat& label = inp[1];
        const bool hasWeight = inp.size() >= 3;

        CV_Assert(logp.dims >= 2);
        const int N = logp.size[0], C = logp.size[1];
        int S = 1; for (int i = 2; i < logp.dims; ++i) S *= logp.size[i];

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

        Mat logp2D; tensorNCX_to_NSxC(logp, logp2D);
        Mat idx32;
        if (label.type() == CV_32S) idx32 = label;
        else                        label.convertTo(idx32, CV_32S);
        CV_Assert((int)idx32.total() == N*S);
        Mat idx1D = idx32.reshape(1, N*S).clone();

        Mat per(N*S, 1, CV_32F, Scalar(0));
        Mat validMask(N*S, 1, CV_8U, Scalar(1));
        Mat effW(N*S, 1, CV_32F, Scalar(1));

        parallel_for_(Range(0, N*S), [&](const Range& rr){
            for (int r = rr.start; r < rr.end; ++r)
            {
                const int y = idx1D.at<int>(r);
                if (y == ignore_index) { validMask.at<uchar>(r) = 0; effW.at<float>(r) = 0.f; continue; }
                CV_Assert(0 <= y && y < C);

                const float lp = logp2D.at<float>(r, y);
                const float cw = weightAt(y);

                per.at<float>(r)  = -lp * cw;
                effW.at<float>(r) =  cw;
            }
        }, 16);

        if (reduction == "none")
        {
            std::vector<int> outShape; outShape.push_back(N);
            for (int i = 2; i < logp.dims; ++i) outShape.push_back(logp.size[i]);
            outv[0].create((int)outShape.size(), outShape.data(), CV_32F);
            per.reshape(1, outv[0].size).copyTo(outv[0]);
        }
        else
        {
            double num = 0.0, den = 0.0;
            for (int r = 0; r < per.rows; ++r) {
                if (!validMask.at<uchar>(r)) continue;
                num += per.at<float>(r);
                if (reduction == "mean") den += std::max(1e-12f, effW.at<float>(r));
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

private:
    String reduction;
    int ignore_index;
};

Ptr<NegativeLogLikelihoodLossLayer> NegativeLogLikelihoodLossLayer::create(const LayerParams& params)
{
    return Ptr<NegativeLogLikelihoodLossLayer>(new NegativeLogLikelihoodLossImpl(params));
}

}}
