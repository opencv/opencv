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
    BlackmanWindow layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__BlackmanWindow.html

    Supported opsets: 17
*/

template<typename T>
static void blackmanWindowFill(Mat& out, int N, double alpha, double beta, double pi, double N1)
{
    CV_Assert(out.dims == 1);
    CV_Assert((int)out.total() == N);

    const double coeff1 = (2.0 * pi) / N1;
    const double coeff2 = (4.0 * pi) / N1;

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
    v_float64 vcoeff1 = vx_setall_f64(coeff1);
    v_float64 vcoeff2 = vx_setall_f64(coeff2);
    v_float64 valpha = vx_setall_f64(alpha);
    v_float64 vbeta = vx_setall_f64(beta);
    v_float64 vnegHalf = vx_setall_f64(-0.5);

    for (; i <= N - nlanes64; i += nlanes64)
    {
        v_float64 varg1 = v_mul(vcoeff1, vindex);
        v_float64 varg2 = v_mul(vcoeff2, vindex);
        v_float64 vc1 = v_cos(varg1);
        v_float64 vc2 = v_cos(varg2);
        v_float64 v = v_add(valpha, v_add(v_mul(vc1, vnegHalf), v_mul(vc2, vbeta)));
        vx_store(w + i, v);
        vindex = v_add(vindex, delta);
    }
#endif

    for (; i < N; ++i)
    {
        double arg1 = coeff1 * i;
        double arg2 = coeff2 * i;
        double v = std::cos(arg1) * (-0.5);
        v += std::cos(arg2) * beta;
        v += alpha;
        w[i] = v;
    }

    T* dst = out.ptr<T>();
    for (int n = 0; n < N; ++n)
        dst[n] = saturate_cast<T>(w[n]);
}

class BlackmanWindowLayerImpl CV_FINAL : public BlackmanWindowLayer
{
public:
    BlackmanWindowLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        outputType = params.get<int>("output_type", CV_32F);
        periodic = params.get<int>("periodic", 1);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    // Output shape depends on runtime scalar "size" input.
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
        CV_CheckGT(size64, 0, "BlackmanWindow: size must be > 0");
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
            CV_Error(Error::StsNotImplemented, "BlackmanWindow: unsupported output kind");
        }

        const double alpha = 0.42;
        const double beta = 0.08;
        const double pi = CV_PI;
        const double N1 = (periodic == 0) ? std::max(N - 1, 1) : std::max(N, 1);

        switch (depth)
        {
            case CV_32F: blackmanWindowFill<float>(Y, N, alpha, beta, pi, N1); break;
            case CV_64F: blackmanWindowFill<double>(Y, N, alpha, beta, pi, N1); break;
            case CV_16F: blackmanWindowFill<hfloat>(Y, N, alpha, beta, pi, N1); break;
            case CV_16BF: blackmanWindowFill<bfloat>(Y, N, alpha, beta, pi, N1); break;
            default:
                CV_Error(Error::BadDepth, "BlackmanWindow: unsupported output depth");
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

Ptr<BlackmanWindowLayer> BlackmanWindowLayer::create(const LayerParams& params)
{
    return makePtr<BlackmanWindowLayerImpl>(params);
}

}} // namespace cv::dnn
