// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv {
namespace dnn {

class BatchNorm2LayerImpl CV_FINAL : public BatchNorm2Layer {
public:
    BatchNorm2LayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        epsilon = params.get<float>("epsilon", params.get<float>("eps", 1e-5f));
        useGlobalStats = params.get<bool>("use_global_stats", true);
        hasWeights = params.get<bool>("has_weight", false);
        hasBias = params.get<bool>("has_bias", false);

        if (blobs.size() >= 4) {
            dynamicInputs = false;

            const Mat& mean  = blobs[0];
            const Mat& var   = blobs[1];
            const Mat& scale = blobs[2];
            const Mat& beta  = blobs[3];

            weights_.create(scale.size(), CV_32F);
            bias_.create(scale.size(), CV_32F);

            cv::sqrt(var + epsilon, bias_);
            cv::divide(scale, bias_, weights_);
            bias_ = beta - mean.mul(weights_);
        } else {
            dynamicInputs = true;
        }
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return dynamicInputs;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(requiredOutputs, inputs[0]);
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
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                    std::vector<DataLayout>& desiredInputs,
                    const int requiredOutputs,
                    std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        size_t ninputs = actualInputs.size();
        CV_Assert(ninputs >= 1u);
        desiredInputs.assign(ninputs, DATA_LAYOUT_UNKNOWN);
        desiredInputs[0] = actualInputs[0];
        outputs.assign(requiredOutputs, actualInputs[0]);
        return 0;
    }

    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        const Mat &X = inputs[0];
        Mat Y;
        Mat w, b;

        if (dynamicInputs) {
            CV_Assert(inputs.size() == 5);

            const Mat& scale   = inputs[1];
            const Mat& beta    = inputs[2];
            const Mat& mean    = inputs[3];
            const Mat& var     = inputs[4];

            w.create(scale.size(), CV_32F);
            b.create(scale.size(), CV_32F);

            cv::sqrt(var + epsilon, b);
            cv::divide(scale, b, w);
            b = beta - mean.mul(w);
        } else {
            w = weights_;
            b = bias_;
        }

        if (w.empty() || b.empty())
             CV_Error(Error::StsBadArg, "BatchNorm2Layer: Weights not initialized");

        MatShape outShape = shape(X);
        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            CV_Assert(outs.size() >= 1);
            outs[0].fit(outShape, X.type());
            Y = outs[0];
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            CV_Assert(uouts.size() >= 1);
            uouts[0].fit(outShape, X.type());
            Y = uouts[0].getMat(ACCESS_WRITE);
        } else {
            CV_Error(Error::StsBadArg, "Unsupported output array kind");
        }

        const int C = (X.dims >= 2) ? X.size[1] : 1;
        const int N = X.size[0];
        const size_t planeSize = X.total() / (N * C);

        CV_Assert(w.total() == C);

        parallel_for_(Range(0, N * C), [&](const Range& r) {
            for (int i = r.start; i < r.end; ++i) {
                int c = i % C;

                float scale_val = w.ptr<float>()[c];
                float shift_val = b.ptr<float>()[c];

                const float* srcPtr = X.ptr<float>() + i * planeSize;
                float* dstPtr = Y.ptr<float>() + i * planeSize;

                int j = 0;
#if CV_SIMD128
                v_float32x4 v_scale = v_setall_f32(scale_val);
                v_float32x4 v_shift = v_setall_f32(shift_val);
                for (; j <= (int)planeSize - 4; j += 4) {
                    v_float32x4 v_src = v_load(srcPtr + j);
                    v_float32x4 v_dst = v_muladd(v_src, v_scale, v_shift);
                    v_store(dstPtr + j, v_dst);
                }
#endif
                for (; j < (int)planeSize; ++j) {
                    dstPtr[j] = srcPtr[j] * scale_val + shift_val;
                }
            }
        });
    }
private:
    bool dynamicInputs;
    Mat weights_, bias_;
};

Ptr<BatchNorm2Layer> BatchNorm2Layer::create(const LayerParams& params)
{
    return makePtr<BatchNorm2LayerImpl>(params);
}
}} // namespace cv::dnn
