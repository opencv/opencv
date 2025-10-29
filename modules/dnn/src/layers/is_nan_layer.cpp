// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "opencv2/core/fast_math.hpp"  // for cvIsNaN

namespace cv {
namespace dnn {

/*
    IsNaN layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__IsNaN.html

    Opset's 9 to 20 are covered.
*/

template <typename T, typename WT = T>
static inline void computeIsNaNMask(const T* src, uchar* dst, const size_t count)
{
    parallel_for_(Range(0, (int)count), [&](const Range& r){
        for (int i = r.start; i < r.end; ++i)
        {
            WT v = (WT)src[i];
            dst[i] = static_cast<uchar>(cvIsNaN(v));
        }
    });
}

class IsNaNLayerImpl CV_FINAL : public IsNaNLayer
{
public:
    IsNaNLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
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
        const int outType = (Y.empty() || Y.type() < 0) ? defaultOutType : Y.type();
        Y.create(X.size, outType);

        const int depth = CV_MAT_DEPTH(X.type());
        const size_t total = X.total();
        uchar* dst = Y.ptr<uchar>();

        switch (depth) {
            case CV_32F: computeIsNaNMask<float>(X.ptr<float>(), dst, total);    break;
            case CV_64F: computeIsNaNMask<double>(X.ptr<double>(), dst, total);   break;
            case CV_16F: computeIsNaNMask<hfloat, float>(X.ptr<hfloat>(), dst, total); break;
            case CV_16BF: computeIsNaNMask<bfloat, float>(X.ptr<bfloat>(), dst, total); break;
            default: CV_Error_(Error::StsError, ("IsNaN: Unsupported type depth=%d", depth));
        }
    }
};

Ptr<IsNaNLayer> IsNaNLayer::create(const LayerParams& p) { return makePtr<IsNaNLayerImpl>(p); }

}} // namespace cv::dnn
