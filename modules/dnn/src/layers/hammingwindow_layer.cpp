// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {

/*
    HammingWindow layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__HammingWindow.html

    Supported opsets: 17
*/

template<typename T>
static void HammingWindowFill(Mat& out, int N, double pi, double N1)
{
    CV_Assert(out.dims == 1);
    CV_Assert((int)out.total() == N);

    const double alpha = 25.0 / 46.0;
    const double beta = 1.0 - alpha;
    const double coeff = (2.0 * pi) / N1;

    cv::AutoBuffer<double> _w(N);
    double* w = _w.data();

    int i = 0;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    const int nlanes64 = VTraits<v_float64>::vlanes();
    const int max_nlanes = VTraits<v_float64>::max_nlanes;
    std::array<double, max_nlanes> index;
    std::iota(index.data(), index.data() + max_nlanes, 0.0);
    v_float64 vindex = vx_load(index.data());
    v_float64 delta = vx_setall_f64(nlanes64);
    v_float64 vcoeff = vx_setall_f64(coeff);
    v_float64 valpha = vx_setall_f64(alpha);
    v_float64 vbeta = vx_setall_f64(beta);

    for (; i <= N - nlanes64; i += nlanes64)
    {
        v_float64 varg = v_mul(vcoeff, vindex);
        v_float64 vc = v_cos(varg);
        v_float64 v = v_sub(valpha, v_mul(vc, vbeta));
        vx_store(w + i, v);
        vindex = v_add(vindex, delta);
    }
#endif

    for (; i < N; ++i)
    {
        double arg = coeff * i;
        w[i] = alpha - std::cos(arg) * beta;
    }

    T* dst = out.ptr<T>();
    for (int n = 0; n < N; ++n)
        dst[n] = saturate_cast<T>(w[n]);
}

class HammingWindowLayerImpl CV_FINAL : public HammingWindowLayer
{
public:
    HammingWindowLayerImpl(const LayerParams& params)
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
        // Shape is fully dynamic here; it will be resolved in forward().
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
        CV_CheckGT(size64, 0, "HammingWindow: size must be > 0");
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
            CV_Error(Error::StsNotImplemented, "HammingWindow: unsupported output kind");
        }

        const double pi = CV_PI;
        const double N1 = (periodic == 0) ? std::max(N - 1, 1) : std::max(N, 1);

        switch (depth)
        {
            case CV_32F: HammingWindowFill<float>(Y, N, pi, N1); break;
            case CV_64F: HammingWindowFill<double>(Y, N, pi, N1); break;
            case CV_16F: HammingWindowFill<hfloat>(Y, N, pi, N1); break;
            case CV_16BF: HammingWindowFill<bfloat>(Y, N, pi, N1); break;
            default:
                CV_Error(Error::BadDepth, "HammingWindow: unsupported output depth");
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

Ptr<HammingWindowLayer> HammingWindowLayer::create(const LayerParams& params)
{
    return makePtr<HammingWindowLayerImpl>(params);
}

}} // namespace cv::dnn
