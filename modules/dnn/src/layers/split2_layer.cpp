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

// all outputs must be pre-allocated.
// axis must be normalized
static void split(const Mat& inp, std::vector<Mat>& outs, int axis)
{
    CV_Assert(inp.isContinuous());

    MatShape inpShape = inp.shape();
    int ndims = inpShape.dims;

    CV_Assert_N(0 <= axis, axis <= inp.dims);

    int nslices = 1;
    int inpType = inp.type();
    size_t esz = inp.elemSize();
    size_t sliceSize = esz;
    size_t inpStep = 0;
    size_t totalSize = inp.total()*esz;
    int outSize_a = 0;
    for (int i = ndims-1; i > axis; i--)
        sliceSize *= inpShape[i];
    inpStep = sliceSize*inpShape[axis];
    for (int i = 0; i < axis; i++)
        nslices *= inpShape[i];

    size_t noutputs = outs.size();
    for (size_t k = 0; k < noutputs; k++) {
        Mat& out = outs[k];
        MatShape outShape = out.shape();
        CV_Assert(out.isContinuous());
        CV_Assert(out.type() == inpType);
        CV_Assert(out.dims == ndims);
        for (int i = 0; i < ndims; i++) {
            if (i == axis)
                outSize_a += outShape[i];
            else {
                CV_Assert(inpShape[i] == outShape[i]);
            }
        }
    }

    CV_Assert(outSize_a == inpShape[axis]);

    parallel_for_(Range(0, (int)noutputs), [&](const Range& r) {
        for (int k = r.start; k < r.end; k++) {
            const uchar* inptr = inp.data;
            Mat& out_k = outs[k];
            uchar* outptr_k = out_k.data;
            int sz_a;
            for (int i = 0; i < k; i++) {
                sz_a = outs[i].size[axis];
                inptr += sliceSize*sz_a;
            }
            sz_a = out_k.size[axis];
            size_t sliceSize_k = sliceSize*sz_a;
            for (int i = 0; i < nslices; i++)
                memcpy(outptr_k + i*sliceSize_k, inptr + i*inpStep, sliceSize_k);
        }
    }, (totalSize > 1000000 ? noutputs : 1));
}

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
        cv::dnn::split(inp, outs, axis_);
    }
};

Ptr<Split2Layer> Split2Layer::create(const LayerParams& params)
{
    return Ptr<Split2Layer>(new Split2LayerImpl(params));
}

}
}
