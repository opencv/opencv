// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

namespace cv { namespace dnn {

// ONNX Pow operator spec: https://onnx.ai/onnx/operators/onnx__Pow.html
// Supported opsets: 1, 7, 12, 13, 15

static void fillBroadcastSteps(const Mat& m, const std::vector<int>& outputShape, std::vector<size_t>& strides)
{
    std::vector<int> ms(m.dims);
    for (int i = 0; i < m.dims; ++i) ms[i] = m.size[i];
    const size_t dim = std::max(outputShape.size(), ms.size());
    std::vector<int> msx = ms; msx.insert(msx.begin(), dim - msx.size(), 1);
    std::vector<size_t> sx(dim, 0);
    std::vector<size_t> orig(m.step.p, m.step.p + m.dims);
    std::vector<size_t> origx = orig; origx.insert(origx.begin(), dim - origx.size(), 0);
    for (size_t i = 0; i < dim; ++i)
    {
        if (msx[i] == 1) sx[i] = 0;
        else sx[i] = origx[i];
    }
    strides = sx;
}

template<typename T>
static void applyPowSaturate(const double* a, const double* b, Mat& out)
{
    T* o = (T*)out.data;
    const size_t total = out.total();
    for (size_t i = 0; i < total; i++) o[i] = saturate_cast<T>(std::pow(a[i], b[i]));
}

static inline void powPlaneFloat(const float* a, const float* b, float* o, int plane_size, size_t dpA, size_t dpB, size_t dpO)
{
    if (dpA == 1 && dpB == 1 && dpO == 1) {
        for (int i = 0; i < plane_size; i++) o[i] = std::pow(a[i], b[i]);
    } else {
        for (int i = 0; i < plane_size; i++, a += dpA, b += dpB, o += dpO) *o = std::pow(*a, *b);
    }
}

class PowLayerImpl CV_FINAL : public PowLayer {
public:
    PowLayerImpl(const LayerParams& params) { setParamsFrom(params); }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        MatShape a = inputs[0];
        MatShape b = inputs[1];
        const size_t dim = std::max(a.size(), b.size());
        a.insert(a.begin(), dim - a.size(), 1);
        b.insert(b.begin(), dim - b.size(), 1);
        MatShape out(dim, 1);
        for (size_t i = 0; i < dim; ++i)
        {
            CV_Assert(a[i] == b[i] || a[i] == 1 || b[i] == 1);
            out[i] = std::max(a[i], b[i]);
        }
        outputs.assign(1, out);
        internals.clear();
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        auto isIntegerType = [](int t) {
            return t == CV_8S || t == CV_8U || t == CV_16S || t == CV_16U || t == CV_32S || t == CV_32U || t == CV_64S || t == CV_64U;
        };
        auto isFloatType = [](int t) {
            return t == CV_32F || t == CV_64F || t == CV_16F || t == CV_16BF;
        };

        int out_type;
        const bool baseIsInt   = isIntegerType(inputs[0]);
        const bool expIsInt    = isIntegerType(inputs[1]);
        const bool baseIsFloat = isFloatType(inputs[0]);
        const bool expIsFloat  = isFloatType(inputs[1]);

        if ((baseIsInt && expIsInt) || (baseIsFloat && expIsFloat))
        {
            out_type = (inputs[0] == inputs[1]) ? inputs[0] : CV_32F;
        }
        else if (baseIsFloat != expIsFloat)
        {
            out_type = inputs[0];
        }
        else
        {
            out_type = CV_32F;
        }
        outputs.assign(1, out_type);
        internals.clear();
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_Assert(inputs.size() == 2);

        Mat A = inputs[0];
        Mat B = inputs[1];

        if (A.type() != outputs[0].type()) A.convertTo(A, outputs[0].type());
        if (B.type() != outputs[0].type()) B.convertTo(B, outputs[0].type());

        CV_Assert(outputs[0].isContinuous());
        const int nd = outputs[0].dims;
        std::vector<int> shape(outputs[0].size.p, outputs[0].size.p + nd);
        std::vector<size_t> step_out(outputs[0].step.p, outputs[0].step.p + nd);
        std::vector<size_t> step_a(nd, 0), step_b(nd, 0);

        fillBroadcastSteps(A, shape, step_a);
        fillBroadcastSteps(B, shape, step_b);

        const size_t total = outputs[0].total();
        size_t dpA = nd ? step_a.back() / outputs[0].elemSize() : 0;
        size_t dpB = nd ? step_b.back() / outputs[0].elemSize() : 0;
        size_t dpO = nd ? step_out.back() / outputs[0].elemSize() : 0;

        if (outputs[0].type() == CV_32F) {
            const float* a = (const float*)A.data;
            const float* b = (const float*)B.data;
            float* o = (float*)outputs[0].data;
            const size_t a_total = A.total();
            const size_t b_total = B.total();
            if (a_total == total && b_total == total && dpA == 1 && dpB == 1 && dpO == 1) {
                powPlaneFloat(a, b, o, (int)total, 1, 1, 1);
            } else {
                const char* pa0 = (const char*)A.data;
                const char* pb0 = (const char*)B.data;
                char* po0 = (char*)outputs[0].data;
                int plane_size = nd ? shape.back() : 1;
                size_t nplanes = 1;
                if (nd >= 2) nplanes = std::accumulate(shape.begin(), shape.end()-1, 1, std::multiplies<size_t>());
                parallel_for_(Range(0, (int)nplanes), [&](const Range& r){
                    for (int p = r.start; p < r.end; ++p) {
                        const char* pa = pa0;
                        const char* pb = pb0;
                        char* po = po0;
                        size_t idx = (size_t)p;
                        for (int k = nd - 2; k >= 0; k--) {
                            size_t next_idx = idx / shape[k];
                            size_t i_k = (int)(idx - next_idx * shape[k]);
                            pa += i_k * step_a[k];
                            pb += i_k * step_b[k];
                            po += i_k * step_out[k];
                            idx = next_idx;
                        }
                        const float* a_ = (const float*)pa;
                        const float* b_ = (const float*)pb;
                        float* o_ = (float*)po;
                        powPlaneFloat(a_, b_, o_, plane_size, dpA, dpB, dpO);
                    }
                });
            }
        } else {
            Mat Ad, Bd; A.convertTo(Ad, CV_64F); B.convertTo(Bd, CV_64F);
            const double* a = (const double*)Ad.data;
            const double* b = (const double*)Bd.data;
            switch (outputs[0].type()) {
            case CV_32S: applyPowSaturate<int>(a, b, outputs[0]); break;
            case CV_64S: applyPowSaturate<int64>(a, b, outputs[0]); break;
            case CV_8U:  applyPowSaturate<uchar>(a, b, outputs[0]); break;
            case CV_8S:  applyPowSaturate<schar>(a, b, outputs[0]); break;
            case CV_16U: applyPowSaturate<ushort>(a, b, outputs[0]); break;
            case CV_16S: applyPowSaturate<short>(a, b, outputs[0]); break;
            case CV_32U: applyPowSaturate<uint32_t>(a, b, outputs[0]); break;
            case CV_64U: applyPowSaturate<uint64_t>(a, b, outputs[0]); break;
            case CV_64F: applyPowSaturate<double>(a, b, outputs[0]); break;
            default:
                CV_Error(Error::StsNotImplemented, "Unsupported integer type for PowLayer");
            }
        }
    }
};

Ptr<PowLayer> PowLayer::create(const LayerParams& params)
{
    return Ptr<PowLayer>(new PowLayerImpl(params));
}

}} // namespace cv::dnn
