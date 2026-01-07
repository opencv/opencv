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

    virtual void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        size_t ninputs = inputs_arr.total(-1);
        CV_Assert(ninputs > 0);

        if (ninputs > 1) {
            CV_Assert(ninputs == 5);
            Mat scale_ = inputs_arr.getMat(1);
            Mat bias_ = inputs_arr.getMat(2);
            Mat mean_ = inputs_arr.getMat(3);
            Mat var_ = inputs_arr.getMat(4);
            BatchNorm2Layer::getScaleBias(scale_, bias_, mean_, var_, epsilon, scale, bias);
        }

        MatShape inpShape = inputs_arr.shape(0);
        int inpType = inputs_arr.type(0);

        MatShape outShape = getOutShape(inpShape);
        int outKind = outputs_arr.kind();

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runOp(inp, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            Mat temp(outShape, inpType);
            runOp(inp, temp);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& inp, Mat& out)
    {
        auto netimpl_ = getNetImpl(this);
        batchnorm(inp, out, scale, bias, netimpl_->originalLayout);
    }

    virtual bool freezeScaleBias() CV_OVERRIDE
    {
        auto netimpl_ = getNetImpl(this);
        size_t ninputs = inputs.size();
        if (ninputs != 5)
            return false;
        if (netimpl_->isConstArg(inputs[1]) ||
            netimpl_->isConstArg(inputs[2]) ||
            netimpl_->isConstArg(inputs[3]) ||
            netimpl_->isConstArg(inputs[4]))
            return false;
        Mat scale_ = netimpl_->argTensor(inputs[1]);
        Mat bias_ = netimpl_->argTensor(inputs[2]);
        Mat mean_ = netimpl_->argTensor(inputs[3]);
        Mat var_ = netimpl_->argTensor(inputs[4]);
        BatchNorm2Layer::getScaleBias(scale_, bias_, mean_, var_, epsilon, scale, bias);
        inputs.resize(1);
        return true;
    }

    virtual void getScaleBias(OutputArray scale_, OutputArray bias_) const CV_OVERRIDE
    {
        scale.copyTo(scale_);
        bias.copyTo(bias_);
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
