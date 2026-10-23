// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
//#include "../op_cuda.hpp"
//#include "../op_inf_engine.hpp"
//#include "../ie_ngraph.hpp"
//#include "../op_webnn.hpp"
//#include "../op_timvx.hpp"
//#include "../op_cann.hpp"

//#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

/*
    Split2 layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Split2.html

    Opset's 1 to 13 are covered.
*/

class Split2LayerImpl CV_FINAL : public Split2Layer
{
public:
    Split2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 1);
        split = params.getVector<int>("split");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    // In block layout, split sizes along channels count real channels, not blocks of 8.
    static bool isBlockChannelSplit(const MatShape& inpShape, int axis_)
    {
        return inpShape.layout == DATA_LAYOUT_BLOCK && axis_ == 1;
    }

    static int axisSize(const MatShape& inpShape, int axis_)
    {
        return isBlockChannelSplit(inpShape, axis_) ? inpShape.C : inpShape[axis_];
    }

    void getOutShapes(const MatShape& inpShape, int axis_,
                      const std::vector<int>& split,
                      std::vector<MatShape>& outShapes) const
    {
        size_t noutputs = split.size();
        CV_Assert(noutputs == outputs.size());

        int inpDims = inpShape.dims;
        CV_Assert(0 <= axis_ && axis_ < inpDims);
        const bool blockC = isBlockChannelSplit(inpShape, axis_);
        const int C0 = blockC ? inpShape.back() : 1;
        int totalSize_a = 0;

        outShapes.resize(noutputs);
        for (size_t i = 0; i < noutputs; i++) {
            MatShape outShape = inpShape;
            int s = split[i];
            CV_Assert(s >= 0);
            CV_Assert(s <= axisSize(inpShape, axis_) - totalSize_a);
            if (blockC) {
                outShape[axis_] = (s + C0 - 1) / C0;
                outShape.C = s;
            } else {
                outShape[axis_] = s;
            }
            outShapes[i] = outShape;
            totalSize_a += s;
        }
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        desiredInputs.assign(actualInputs.size(), DATA_LAYOUT_UNKNOWN);
        // Stay in block layout for a non-negative axis (negative axes count from the end, which
        // differs between layouts). Splits not on multiples of 8 channels are repacked in forward().
        if (!actualInputs.empty() && actualInputs[0] == DATA_LAYOUT_BLOCK && axis >= 0) {
            desiredInputs[0] = DATA_LAYOUT_BLOCK;
            outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
            return netimpl_->defaultC0;
        }
        outputs.assign(requiredOutputs, DATA_LAYOUT_UNKNOWN);
        return 0;
    }

    void makeDefaultSplit(int totalSize, size_t noutputs, std::vector<int>& split_) const
    {
        split_.resize(noutputs);
        int chunkSize = (int)((totalSize + noutputs - 1) / noutputs);
        for (size_t i = 0; i < noutputs; i++) {
            int sz_i = std::min(totalSize, chunkSize);
            split_[i] = sz_i;
            totalSize -= sz_i;
        }
    }

    bool isDataShuffling() const CV_OVERRIDE { return true; }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int noutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(noutputs == (int)this->outputs.size());

        size_t ninputs = inputs.size();
        CV_Assert(ninputs == 1 || ninputs == 2);

        MatShape inpShape = inputs[0];
        std::vector<int> tempSplit;
        const std::vector<int>* split_ = &split;
        int axis_ = normalize_axis(axis, inpShape.dims);

        if (ninputs == 2) {
            Net::Impl* netimpl_ = getNetImpl(this);
            Mat splitTensor = netimpl_->argTensor(this->inputs[1]);
            tensorToIntVec(splitTensor, tempSplit);
            split_ = &tempSplit;
        }
        else if (split.empty()) {
            makeDefaultSplit(axisSize(inpShape, axis_), noutputs, tempSplit);
            split_ = &tempSplit;
        }

        getOutShapes(inputs[0], axis_, *split_, outputs);
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(ninputs == 1 || ninputs == 2);
        outputs.assign(requiredOutputs, inputs[0]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        int noutputs = (int)outputs.size();

        CV_Assert(ninputs == 1 || ninputs == 2);

        int inpType = inputs_arr.type(0);
        MatShape inpShape = inputs_arr.shape(0);
        std::vector<int> tempSplit;
        const std::vector<int>* split_ = &split;
        std::vector<MatShape> outShapes;

        int axis_ = normalize_axis(axis, inpShape.dims);

        if (ninputs == 2) {
            Mat splitTensor = inputs_arr.getMat(1);
            tensorToIntVec(splitTensor, tempSplit);
            split_ = &tempSplit;
        }
        else if (split.empty()) {
            makeDefaultSplit(axisSize(inpShape, axis_), noutputs, tempSplit);
            split_ = &tempSplit;
        }
        getOutShapes(inpShape, axis_, *split_, outShapes);
        CV_Assert(outShapes.size() == (size_t)noutputs);

        int outKind = outputs_arr.kind();

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(noutputs);
            for (int i = 0; i < noutputs; i++) {
                MatShape outShape = outShapes[i];
                outs[i].fit(outShape, inpType);
            }
            runOp(inp, outs, axis_);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(noutputs);

            std::vector<Mat> temps(noutputs);
            for (int i = 0; i < noutputs; i++) {
                MatShape outShape = outShapes[i];
                temps[i].fit(outShape, inpType);
            }
            runOp(inp, temps, axis_);
            for (int i = 0; i < noutputs; i++) {
                MatShape outShape = outShapes[i];
                outs[i].fit(outShape, inpType);
                temps[i].copyTo(outs[i]);
                temps[i].release();
            }
        }
    }

    void runOp(const Mat& inp, std::vector<Mat>& outs, int axis_)
    {
        // An empty input (e.g. a zero batch) has nothing to copy; the outputs already have their shapes.
        if (inp.total() == 0)
            return;
        if (isBlockChannelSplit(inp.size, axis_)) {
            runOpBlockAxis1(inp, outs);
            return;
        }
        std::vector<int> sizes(outs.size());
        for (size_t i = 0; i < outs.size(); i++)
            sizes[i] = outs[i].size[axis_];
        cv::split(inp, axis_, sizes, outs);
    }

    // Channel split in block layout, like Concat's: if every part starts on a multiple of 8
    // channels, copy the channel blocks directly; otherwise go through NCHW.
    void runOpBlockAxis1(const Mat& inp, std::vector<Mat>& outs)
    {
        const int C0 = inp.size[inp.dims - 1];
        const size_t nout = outs.size();
        bool aligned = true;
        for (size_t k = 0; k + 1 < nout; k++)
            aligned = aligned && (outs[k].size.C % C0) == 0;
        if (aligned) {
            copyChannelBlocks(inp, outs, true);
            return;
        }

        DataLayout origLayout = getNetImpl(this)->originalLayout;
        Mat plainInp;
        transformLayout(inp, plainInp, origLayout, origLayout, C0);
        std::vector<int> sizes(nout);
        for (size_t k = 0; k < nout; k++)
            sizes[k] = outs[k].size.C;
        std::vector<Mat> plainOuts;
        cv::split(plainInp, 1, sizes, plainOuts);
        for (size_t k = 0; k < nout; k++)
            transformLayout(plainOuts[k], outs[k], DATA_LAYOUT_BLOCK, origLayout, C0);
    }
};

Ptr<Split2Layer> Split2Layer::create(const LayerParams& params)
{
    return Ptr<Split2Layer>(new Split2LayerImpl(params));
}

}
}
