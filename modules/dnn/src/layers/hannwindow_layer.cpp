// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

/*
    HannWindow layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__HannWindow.html

    Supported opsets: 17
*/


template<typename T>
static void hannWindowFill(Mat& out, int N, double pi, double N1)
{
    CV_Assert(out.dims == 1);
    CV_Assert((int)out.total() == N);

    T* dst = out.ptr<T>();
    for (int n = 0; n < N; ++n)
    {
        double arg = (double)n * pi / N1;
        double v = std::sin(arg);
        v = v * v;
        dst[n] = saturate_cast<T>(v);
    }
}

class HannWindowLayerImpl CV_FINAL : public HannWindowLayer
{
public:
    HannWindowLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        outputType = params.get<int>("output_type", CV_32F);
        periodic = params.get<int>("periodic", 1);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }


    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return true;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);

        outputs.assign(1, MatShape());
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& /*inputs*/,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        int t = outputType;
        if (t < 0)
            t = CV_32F;

        outputs.assign(requiredOutputs, MatType(t));
        internals.assign(requiredInternals, MatType(-1));
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size sz = inputs_arr.size();
        CV_Assert(sz.area() == 1);

        Mat sizeTensor = inputs_arr.getMat(0);

        CV_Assert(!sizeTensor.empty());
        CV_Assert(sizeTensor.total() == (size_t)1);

        int64_t size64 = 0;
        tensorToScalar(sizeTensor, CV_64S, &size64);
        CV_CheckGT(size64, 0, "HannWindow: size must be > 0");
        int N = saturate_cast<int>(size64);

        int outType = outputType >= 0 ? outputType : CV_32F;
        int depth = CV_MAT_DEPTH(outType);

        MatShape outShape(1);
        outShape[0] = N;

        int outKind = outputs_arr.kind();
        Mat Y;
        std::vector<Mat>* out_mats = nullptr;
        std::vector<UMat>* out_umats = nullptr;

        if (outKind == _InputArray::STD_VECTOR_MAT)
        {
            out_mats = &outputs_arr.getMatVecRef();
            out_mats->resize(1);
            out_mats->at(0).fit(outShape, outType);
            Y = out_mats->at(0);
        }
        else if (outKind == _InputArray::STD_VECTOR_UMAT)
        {
            out_umats = &outputs_arr.getUMatVecRef();
            out_umats->resize(1);
            out_umats->at(0).fit(outShape, outType);
            Y = Mat(outShape, outType);
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "HannWindow: unsupported output kind");
        }

        const double pi = CV_PI;
        const double N1 = (periodic == 0) ? std::max(N - 1, 1) : std::max(N, 1);

        switch (depth)
        {
            case CV_32F: hannWindowFill<float>(Y, N, pi, N1); break;
            case CV_64F: hannWindowFill<double>(Y, N, pi, N1); break;
            case CV_16F: hannWindowFill<hfloat>(Y, N, pi, N1); break;
            case CV_16BF: hannWindowFill<bfloat>(Y, N, pi, N1); break;
            default: CV_Error(Error::BadDepth, "HannWindow: unsupported output depth");
        }

        if (outKind == _InputArray::STD_VECTOR_UMAT)
        {
            Y.copyTo(out_umats->at(0));
        }
    }

private:
    int outputType;
    int periodic;
};

Ptr<HannWindowLayer> HannWindowLayer::create(const LayerParams& params)
{
    return makePtr<HannWindowLayerImpl>(params);
}

}} // namespace cv::dnn
