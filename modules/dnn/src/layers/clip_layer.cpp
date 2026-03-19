// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/hal/interface.h>
#include <limits>
#include <cfloat>
#include <algorithm>

namespace cv {
namespace dnn {

static double typeMin(int depth)
{
    switch (depth)
    {
        case CV_8U:  return std::numeric_limits<uchar>::lowest();
        case CV_8S:  return std::numeric_limits<schar>::lowest();
        case CV_16U: return std::numeric_limits<ushort>::lowest();
        case CV_16S: return std::numeric_limits<short>::lowest();
        case CV_32S: return std::numeric_limits<int>::lowest();
        case CV_32F: return -FLT_MAX;
        case CV_64F: return -DBL_MAX;
        default:     CV_Error(Error::StsUnsupportedFormat, "Clip: unsupported depth");
    }
}

static double typeMax(int depth)
{
    switch (depth)
    {
        case CV_8U:  return std::numeric_limits<uchar>::max();
        case CV_8S:  return std::numeric_limits<schar>::max();
        case CV_16U: return std::numeric_limits<ushort>::max();
        case CV_16S: return std::numeric_limits<short>::max();
        case CV_32S: return std::numeric_limits<int>::max();
        case CV_32F: return  FLT_MAX;
        case CV_64F: return  DBL_MAX;
        default:     CV_Error(Error::StsUnsupportedFormat, "Clip: unsupported depth");
    }
}

class ClipLayerImpl CV_FINAL : public ClipLayer
{
public:
    float minValue, maxValue;
    bool  hasMin,   hasMax;

    ClipLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasMin = params.has("min");
        hasMax = params.has("max");
        if (hasMin) minValue = params.get<float>("min");
        if (hasMax) maxValue = params.get<float>("max");
        if (hasMin && hasMax)
            CV_Assert(minValue <= maxValue);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(requiredOutputs, inputs[0]);
        internals.assign(requiredInternals, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_Assert(!inputs.empty());
        const Mat& data = inputs[0];
        Mat& dst = outputs[0];

        bool dynMin = inputs.size() >= 2 && !inputs[1].empty();
        bool dynMax = inputs.size() >= 3 && !inputs[2].empty();

        auto getScalar = [](const Mat& m)->double {
            CV_Assert(m.total()==1);
            Mat tmp;
            m.convertTo(tmp, CV_64F);
            return tmp.at<double>(0);
        };

        double actualMin = dynMin ? getScalar(inputs[1]) : (hasMin ? minValue : typeMin(data.depth()));
        double actualMax = dynMax ? getScalar(inputs[2]) : (hasMax ? maxValue : typeMax(data.depth()));
        CV_Assert(actualMin <= actualMax);

        Scalar lowS  = Scalar::all(actualMin);
        Scalar highS = Scalar::all(actualMax);
        cv::max(data, lowS, dst);
        cv::min(dst, highS, dst);
    }
};

Ptr<ClipLayer> ClipLayer::create(const LayerParams& params)
{
    return Ptr<ClipLayer>(new ClipLayerImpl(params));
}
}}
