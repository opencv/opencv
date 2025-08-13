// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "opencv2/core/fast_math.hpp"  // for cvIsInf

namespace cv {
namespace dnn {

template <typename T>
static inline void computeIsInfMask(const T* src, uchar* dst, const size_t count, const bool detectPositive, const bool detectNegative)
{
    for (size_t i = 0; i < count; ++i)
    {
        const T v = src[i];
        const bool pos = cvIsInf(v) && (v > 0) && detectPositive;
        const bool neg = cvIsInf(v) && (v < 0) && detectNegative;
        dst[i] = static_cast<uchar>(pos || neg);
    }
}

class IsInfLayerImpl CV_FINAL : public IsInfLayer
{
    bool detect_pos = true, detect_neg = true;

public:
    IsInfLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        detect_pos = params.get<bool>("detect_positive", true);
        detect_neg = params.get<bool>("detect_negative", true);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, int,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>&) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>&, const int requiredOutputs,
                  const int requiredInternals, std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        outputs.assign(requiredOutputs, CV_Bool);
        internals.assign(requiredInternals, MatType(-1));
    }

    void forward(InputArrayOfArrays in, OutputArrayOfArrays out, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs; in.getMatVector(inputs); out.getMatVector(outputs);
        CV_Assert(inputs.size() == 1 && outputs.size() == 1);

        const Mat& X = inputs[0];
        Mat& Y = outputs[0];

        const int defaultOutType = CV_BoolC1;
        const int outType = Y.empty() ? defaultOutType : Y.type();
        Y.create(X.dims, X.size.p, outType);

        const int depth = CV_MAT_DEPTH(X.type());
        const size_t total = X.total();
        uchar* dst = Y.ptr<uchar>();

        switch (depth) {
            case CV_32F: computeIsInfMask<float>(X.ptr<float>(), dst, total, detect_pos, detect_neg);    break;
            case CV_64F: computeIsInfMask<double>(X.ptr<double>(), dst, total, detect_pos, detect_neg);   break;
            default: CV_Error_(Error::StsError, ("IsInf: Unsupported type depth=%d", depth));
        }
    }
};

Ptr<IsInfLayer> IsInfLayer::create(const LayerParams& p) { return makePtr<IsInfLayerImpl>(p); }

}} // namespace cv::dnn
