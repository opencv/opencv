// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/slice.hpp"
#endif
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
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA) {
            Net::Impl* netimpl_ = getNetImpl(this);
            if (!netimpl_ || inputs.empty() || outputs.empty())
                return false;
            // offsets are fixed in initCUDA(), so the split sizes must be constant
            if (inputs.size() == 2 && !netimpl_->isConstArg(inputs[1]))
                return false;
            // forwardOpCUDA() skips the whole op when any output is empty
            for (const Arg& out : outputs) {
                const MatShape& s = netimpl_->argData(out).shape;
                for (int j = 0; j < s.dims; j++)
                    if (s[j] == 0)
                        return false;
            }
            const ArgData& adata = netimpl_->argData(inputs[0]);
            if (adata.shape.dims > CSL_MAX_TENSOR_RANK)
                return false;
            // an intermediate arg has no type until the first forward; refuse only a known unsupported one
            const int t = adata.type;
            return t < 0 || isCudaNodeType(t);
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void* context_,
                              InputArrayOfArrays inputs_arr,
                              InputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        auto context = reinterpret_cast<cuda4dnn::csl::CSLContext*>(context_);
        const int ndims = inputs_arr.shape(0).dims;
        const int axis_ = normalize_axis(axis, ndims);

        // output i starts where the previous outputs end along the split axis
        const size_t noutputs = outputs_arr.total();
        std::vector<std::vector<std::size_t>> offsets(noutputs, std::vector<std::size_t>(ndims, 0));
        std::size_t start = 0;
        for (size_t i = 0; i < noutputs; i++) {
            offsets[i][axis_] = start;
            start += (std::size_t)outputs_arr.shape((int)i)[axis_];
        }

        if (inputs_arr.depth(0) == CV_Bool)
            return make_cuda_node_bool<cuda4dnn::SliceOp>(std::move(context->stream), std::move(offsets));
        return make_cuda_node_with_type<cuda4dnn::SliceOp>(preferableTarget, inputs_arr.type(0),
                                                           std::move(context->stream), std::move(offsets));
    }
#endif

    void getOutShapes(const MatShape& inpShape, int axis_,
                      const std::vector<int>& split,
                      std::vector<MatShape>& outShapes) const
    {
        size_t noutputs = split.size();
        CV_Assert(noutputs == outputs.size());

        int inpDims = inpShape.dims;
        CV_Assert(0 <= axis_ && axis_ < inpDims);
        int totalSize_a = 0;

        outShapes.resize(noutputs);
        for (size_t i = 0; i < noutputs; i++) {
            MatShape outShape = inpShape;
            int s = split[i];
            CV_Assert(s >= 0);
            CV_Assert(s <= inpShape[axis_] - totalSize_a);
            outShape[axis_] = s;
            outShapes[i] = outShape;
            totalSize_a += s;
        }
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
            makeDefaultSplit(inpShape[axis_], noutputs, tempSplit);
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
            makeDefaultSplit(inpShape[axis_], noutputs, tempSplit);
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
        std::vector<int> sizes(outs.size());
        for (size_t i = 0; i < outs.size(); i++)
            sizes[i] = outs[i].size[axis_];
        cv::split(inp, axis_, sizes, outs);
    }
};

Ptr<Split2Layer> Split2Layer::create(const LayerParams& params)
{
    return Ptr<Split2Layer>(new Split2LayerImpl(params));
}

}
}
