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
    Window layers, as defined in ONNX specification:
      - BlackmanWindow: https://onnx.ai/onnx/operators/onnx__BlackmanWindow.html
      - HannWindow:     https://onnx.ai/onnx/operators/onnx__HannWindow.html
      - HammingWindow:  https://onnx.ai/onnx/operators/onnx__HammingWindow.html

    Supported opsets: 17
*/

enum class WindowKind
{
    Blackman,
    Hann,
    Hamming
};

template<typename T>
static void BlackmanWindowFill(Mat& out, int N, double N1, bool useDouble)
{
    CV_Assert(out.dims == 1);
    CV_Assert((int)out.total() == N);

    if (useDouble)
    {
        const double alpha = 0.42;
        const double beta = 0.08;
        const double pi = CV_PI;
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
        v_float64 valpha  = vx_setall_f64(alpha);
        v_float64 vbeta   = vx_setall_f64(beta);
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
    else
    {
        const float alpha = 0.42f;
        const float beta = 0.08f;
        const float pi = (float)CV_PI;
        const float coeff1 = (2.0f * pi) / (float)N1;
        const float coeff2 = (4.0f * pi) / (float)N1;

        cv::AutoBuffer<float> _w(N);
        float* w = _w.data();

        int i = 0;
#if (defined(CV_SIMD_32F) && CV_SIMD_32F) || (defined(CV_SIMD_SCALABLE_32F) && CV_SIMD_SCALABLE_32F)
        const int nlanes32 = VTraits<v_float32>::vlanes();
        const int max_nlanes32 = VTraits<v_float32>::max_nlanes;
        std::array<float, max_nlanes32> index;
        std::iota(index.data(), index.data() + max_nlanes32, 0.0f);
        v_float32 vindex = vx_load(index.data());
        v_float32 delta = vx_setall_f32(nlanes32);
        v_float32 vcoeff1 = vx_setall_f32(coeff1);
        v_float32 vcoeff2 = vx_setall_f32(coeff2);
        v_float32 valpha  = vx_setall_f32(alpha);
        v_float32 vbeta   = vx_setall_f32(beta);
        v_float32 vnegHalf = vx_setall_f32(-0.5f);

        for (; i <= N - nlanes32; i += nlanes32)
        {
            v_float32 varg1 = v_mul(vcoeff1, vindex);
            v_float32 varg2 = v_mul(vcoeff2, vindex);
            v_float32 vc1 = v_cos(varg1);
            v_float32 vc2 = v_cos(varg2);
            v_float32 v = v_add(valpha, v_add(v_mul(vc1, vnegHalf), v_mul(vc2, vbeta)));
            vx_store(w + i, v);
            vindex = v_add(vindex, delta);
        }
#endif

        for (; i < N; ++i)
        {
            float arg1 = coeff1 * i;
            float arg2 = coeff2 * i;
            float v = std::cos(arg1) * (-0.5f);
            v += std::cos(arg2) * beta;
            v += alpha;
            w[i] = v;
        }

        T* dst = out.ptr<T>();
        for (int n = 0; n < N; ++n)
            dst[n] = saturate_cast<T>(w[n]);
    }
}

template<typename T>
static void HannWindowFill(Mat& out, int N, double N1, bool useDouble)
{
    CV_Assert(out.dims == 1);
    CV_Assert((int)out.total() == N);

    if (useDouble)
    {
        cv::AutoBuffer<double> _w(N);
        double* w = _w.data();

        const double pi = CV_PI;
        const double coeff = (2.0 * pi) / N1;

        int i = 0;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        const int nlanes64 = VTraits<v_float64>::vlanes();
        const int max_nlanes = VTraits<v_float64>::max_nlanes;
        std::array<double, max_nlanes> index;
        std::iota(index.data(), index.data() + max_nlanes, 0.0);
        v_float64 vindex = vx_load(index.data());
        v_float64 delta = vx_setall_f64(nlanes64);
        v_float64 vcoeff = vx_setall_f64(coeff);
        v_float64 one = vx_setall_f64(1.0);
        v_float64 half = vx_setall_f64(0.5);

        for (; i <= N - nlanes64; i += nlanes64)
        {
            v_float64 v = v_mul(half, v_sub(one, v_cos(v_mul(vcoeff, vindex))));
            vx_store(w + i, v);
            vindex = v_add(vindex, delta);
        }
#endif

        for (; i < N; ++i)
            w[i] = 0.5 * (1.0 - std::cos(coeff * i));

        T* dst = out.ptr<T>();
        for (int n = 0; n < N; ++n)
            dst[n] = saturate_cast<T>(w[n]);
    }
    else
    {
        const float pi = (float)CV_PI;
        const float coeff = (2.0f * pi) / (float)N1;

        cv::AutoBuffer<float> _w(N);
        float* w = _w.data();

        int i = 0;
#if (defined(CV_SIMD_32F) && CV_SIMD_32F) || (defined(CV_SIMD_SCALABLE_32F) && CV_SIMD_SCALABLE_32F)
        const int nlanes32 = VTraits<v_float32>::vlanes();
        const int max_nlanes32 = VTraits<v_float32>::max_nlanes;
        std::array<float, max_nlanes32> index;
        std::iota(index.data(), index.data() + max_nlanes32, 0.0f);
        v_float32 vindex = vx_load(index.data());
        v_float32 delta = vx_setall_f32(nlanes32);
        v_float32 vcoeff = vx_setall_f32(coeff);
        v_float32 one = vx_setall_f32(1.0f);
        v_float32 half = vx_setall_f32(0.5f);

        for (; i <= N - nlanes32; i += nlanes32)
        {
            v_float32 v = v_mul(half, v_sub(one, v_cos(v_mul(vcoeff, vindex))));
            vx_store(w + i, v);
            vindex = v_add(vindex, delta);
        }
#endif

        for (; i < N; ++i)
            w[i] = 0.5f * (1.0f - std::cos(coeff * i));

        T* dst = out.ptr<T>();
        for (int n = 0; n < N; ++n)
            dst[n] = saturate_cast<T>(w[n]);
    }
}

template<typename T>
static void HammingWindowFill(Mat& out, int N, double N1, bool useDouble)
{
    CV_Assert(out.dims == 1);
    CV_Assert((int)out.total() == N);

    if (useDouble)
    {
        const double alpha = 25.0 / 46.0;
        const double beta = 1.0 - alpha;
        const double pi = CV_PI;
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
    else
    {
        const float alpha = (float)(25.0 / 46.0);
        const float beta = 1.0f - alpha;
        const float pi = (float)CV_PI;
        const float coeff = (2.0f * pi) / (float)N1;

        cv::AutoBuffer<float> _w(N);
        float* w = _w.data();

        int i = 0;
#if (defined(CV_SIMD_32F) && CV_SIMD_32F) || (defined(CV_SIMD_SCALABLE_32F) && CV_SIMD_SCALABLE_32F)
        const int nlanes32 = VTraits<v_float32>::vlanes();
        const int max_nlanes32 = VTraits<v_float32>::max_nlanes;
        std::array<float, max_nlanes32> index;
        std::iota(index.data(), index.data() + max_nlanes32, 0.0f);
        v_float32 vindex = vx_load(index.data());
        v_float32 delta = vx_setall_f32(nlanes32);
        v_float32 vcoeff = vx_setall_f32(coeff);
        v_float32 valpha = vx_setall_f32(alpha);
        v_float32 vbeta = vx_setall_f32(beta);

        for (; i <= N - nlanes32; i += nlanes32)
        {
            v_float32 varg = v_mul(vcoeff, vindex);
            v_float32 vc = v_cos(varg);
            v_float32 v = v_sub(valpha, v_mul(vc, vbeta));
            vx_store(w + i, v);
            vindex = v_add(vindex, delta);
        }
#endif

        for (; i < N; ++i)
        {
            float arg = coeff * i;
            w[i] = alpha - std::cos(arg) * beta;
        }

        T* dst = out.ptr<T>();
        for (int n = 0; n < N; ++n)
            dst[n] = saturate_cast<T>(w[n]);
    }
}

template<typename BaseLayer>
class WindowLayerImpl CV_FINAL : public BaseLayer
{
public:
    WindowLayerImpl(const LayerParams& params, WindowKind kind_)
        : outputType(onnxDataTypeToCV(static_cast<OnnxDataType>(
              params.get<int>("output_type", static_cast<int>(ONNX_FLOAT))))),
          periodic(params.get<int>("periodic", 1)),
          kind(kind_)
    {
        this->setParamsFrom(params);
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
        forwardImpl(this->name, outputType, inputs_arr, outputs_arr);
    }

private:
    void forwardImpl(const String& name,
                     int outputType_,
                     InputArrayOfArrays inputs_arr,
                     OutputArrayOfArrays outputs_arr) const
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(layer_name, "name", name.c_str());

        Size sz = inputs_arr.size();
        CV_Assert(sz.area() == 1);

        Mat sizeTensor = inputs_arr.getMat(0);

        CV_Assert(!sizeTensor.empty());
        CV_Assert(sizeTensor.total() == (size_t)1);

        int64_t size64 = 0;
        tensorToScalar(sizeTensor, CV_64S, &size64);

        const char* layerPrefix = nullptr;
        switch (kind)
        {
        case WindowKind::Blackman: layerPrefix = "BlackmanWindow"; break;
        case WindowKind::Hann:     layerPrefix = "HannWindow";     break;
        case WindowKind::Hamming:  layerPrefix = "HammingWindow";  break;
        default:                   layerPrefix = "Window";         break;
        }

        if (size64 <= 0)
            CV_Error(Error::StsOutOfRange, String(layerPrefix) + ": size must be > 0");
        int N = saturate_cast<int>(size64);

        int outType = outputType_ >= 0 ? outputType_ : CV_32F;
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
            CV_Error(Error::StsNotImplemented, String(layerPrefix) + ": unsupported output kind");
        }

        const double N1 = (periodic == 0) ? std::max(N - 1, 1) : std::max(N, 1);

        switch (kind)
        {
        case WindowKind::Blackman:
            switch (depth)
            {
            case CV_64F: BlackmanWindowFill<double>(Y, N, N1, true); break;
            case CV_32F: BlackmanWindowFill<float>(Y, N, N1, false); break;
            case CV_16F: BlackmanWindowFill<hfloat>(Y, N, N1, false); break;
            case CV_16BF: BlackmanWindowFill<bfloat>(Y, N, N1, false); break;
            default:
                CV_Error(Error::BadDepth, "BlackmanWindow: unsupported output depth");
            }
            break;
        case WindowKind::Hann:
            switch (depth)
            {
            case CV_64F: HannWindowFill<double>(Y, N, N1, true); break;
            case CV_32F: HannWindowFill<float>(Y, N, N1, false); break;
            case CV_16F: HannWindowFill<hfloat>(Y, N, N1, false); break;
            case CV_16BF: HannWindowFill<bfloat>(Y, N, N1, false); break;
            default:
                CV_Error(Error::BadDepth, "HannWindow: unsupported output depth");
            }
            break;
        case WindowKind::Hamming:
            switch (depth)
            {
            case CV_64F: HammingWindowFill<double>(Y, N, N1, true); break;
            case CV_32F: HammingWindowFill<float>(Y, N, N1, false); break;
            case CV_16F: HammingWindowFill<hfloat>(Y, N, N1, false); break;
            case CV_16BF: HammingWindowFill<bfloat>(Y, N, N1, false); break;
            default:
                CV_Error(Error::BadDepth, "HammingWindow: unsupported output depth");
            }
            break;
        default:
            CV_Error(Error::StsInternal, "Unknown WindowKind");
        }

        if (outKind == _InputArray::STD_VECTOR_UMAT)
        {
            Y.copyTo(out_umats->at(0));
        }
    }

    int outputType;
    int periodic;
    WindowKind kind;
};

Ptr<BlackmanWindowLayer> BlackmanWindowLayer::create(const LayerParams& params)
{
    return makePtr<WindowLayerImpl<BlackmanWindowLayer>>(params, WindowKind::Blackman);
}

Ptr<HannWindowLayer> HannWindowLayer::create(const LayerParams& params)
{
    return makePtr<WindowLayerImpl<HannWindowLayer>>(params, WindowKind::Hann);
}

Ptr<HammingWindowLayer> HammingWindowLayer::create(const LayerParams& params)
{
    return makePtr<WindowLayerImpl<HammingWindowLayer>>(params, WindowKind::Hamming);
}

}} // namespace cv::dnn
