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
    Slice2 layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Slice2.html

    Opset's 1 to 13 are covered.
*/

// Block tensor [N, C1, S..., C0] cut on the channel axis with step 1: output channel c is input
// channel starts[1] + c, so when starts[1] is not a multiple of C0 a block spans two input blocks.
template <typename T>
static void sliceChannelsBlock_(const Mat& inp, Mat& out, const int* starts, const int* steps)
{
    CV_Assert(inp.isContinuous() && out.isContinuous());
    CV_Assert(steps[1] == 1);
    const int ndims = out.dims, wAxis = ndims - 2;
    const int C0 = out.size[ndims - 1], C1out = out.size[1], Cout = out.size.C;
    const int Wout = out.size[wAxis];
    size_t istep[MatShape::MAX_DIMS];
    for (int d = 0; d < ndims; d++)
        istep[d] = inp.step.p[d] / sizeof(T);
    // for k >= len0 below, src[nextBlock + k] is lane k - len0 of the next input block
    const size_t nextBlock = istep[1] - C0;

    int nrows = out.size[0] * C1out;
    for (int d = 2; d < wAxis; d++)
        nrows *= out.size[d];

    parallel_for_(Range(0, nrows), [&](const Range& r) {
        const T* inpdata = inp.ptr<T>();
        T* outdata = out.ptr<T>();
        for (int row = r.start; row < r.end; row++) {
            int t = row;
            size_t ofs = 0;
            for (int d = wAxis - 1; d >= 2; d--) {
                int i = t % out.size[d];
                t /= out.size[d];
                ofs += (size_t)(starts[d] + i*steps[d])*istep[d];
            }
            int oc1 = t % C1out, n = t / C1out;
            int c = starts[1] + oc1*C0;
            int nvalid = std::min(C0, Cout - oc1*C0);
            int len0 = std::min(nvalid, C0 - c % C0);
            ofs += (size_t)(starts[0] + n*steps[0])*istep[0] + (size_t)(c / C0)*istep[1] + c % C0;
            const T* inprow = inpdata + ofs + (size_t)starts[wAxis]*istep[wAxis];
            T* outrow = outdata + (size_t)row*Wout*C0;
            if (len0 == C0 && steps[wAxis] == 1) {
                memcpy(outrow, inprow, (size_t)Wout*C0*sizeof(T));
                continue;
            }
            for (int w = 0; w < Wout; w++) {
                const T* src = inprow + (ptrdiff_t)w*steps[wAxis]*(ptrdiff_t)istep[wAxis];
                T* dst = outrow + (size_t)w*C0;
                int k = 0;
                for (; k < len0; k++)
                    dst[k] = src[k];
                for (; k < nvalid; k++)
                    dst[k] = src[nextBlock + k];
                for (; k < C0; k++)
                    dst[k] = T(0);
            }
        }
    });
}

static void sliceChannelsBlock(const Mat& inp, Mat& out, const int* starts, const int* steps)
{
    switch (inp.elemSize()) {
    case 1: sliceChannelsBlock_<uint8_t>(inp, out, starts, steps); break;
    case 2: sliceChannelsBlock_<uint16_t>(inp, out, starts, steps); break;
    case 4: sliceChannelsBlock_<uint32_t>(inp, out, starts, steps); break;
    case 8: sliceChannelsBlock_<uint64_t>(inp, out, starts, steps); break;
    default: CV_Error(Error::StsNotImplemented, "unsupported element size");
    }
}

class Slice2LayerImpl CV_FINAL : public Slice2Layer
{
public:
    Slice2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axes = params.getVector<int>("axes");
        starts = params.getVector<int>("starts");
        ends = params.getVector<int>("ends");
    }

    void checkNumInputs(size_t ninputs) const
    {
        CV_Assert(ninputs == 1 || (3 <= ninputs && ninputs <= 5));
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        size_t ninputs = inputs.size();

        for (size_t i = 1; i < ninputs; i++) {
            if (!netimpl_->isConstArg(inputs[i]))
                return true;
        }
        return false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const MatShape& inpShape,
                         const std::vector<int>& starts_,
                         const std::vector<int>& ends_,
                         const std::vector<int>& axes_,
                         const std::vector<int>& steps_,
                         int* allStarts = nullptr,
                         int* allEnds = nullptr,
                         int* allSteps = nullptr) const
    {
        bool sliceMask[MatShape::MAX_DIMS];

        int ndims = inpShape.dims;
        int nstarts = (int)starts_.size(), nends = (int)ends_.size();
        int naxes = (int)axes_.size(), nsteps = (int)steps_.size();

        CV_Assert_N(nstarts > 0, nstarts <= ndims, nstarts == nends);
        CV_Assert(naxes == 0 || naxes == nstarts);
        CV_Assert(nsteps == 0 || nsteps == nstarts);

        MatShape outShape = inpShape;

        for (int i = 0; i < ndims; i++) {
            sliceMask[i] = false;
            if (allStarts)
                allStarts[i] = 0;
            if (allEnds)
                allEnds[i] = inpShape[i];
            if (allSteps)
                allSteps[i] = 1;
        }

        for (int i = 0; i < nstarts; i++) {
            int axis = i;
            if (!axes_.empty()) {
                axis = axes_[i];
                axis = normalize_axis(axis, ndims);
                if (sliceMask[axis]) {
                    CV_Error(Error::StsBadArg, "duplicate axis occurs in Slice");
                }
            }
            sliceMask[axis] = true;
            int inpsz = inpShape[axis];
            int start = starts_[i];
            int end = ends_[i];
            int step = 1;
            if (!steps_.empty())
                step = steps_[i];
            CV_Assert(step != 0);
            start = start < 0 ? std::max(start + inpsz, 0) :
                                std::min(start, inpsz - (step < 0));
            end = end < 0 ? std::max(end + inpsz, -(step < 0)) :
                            std::min(end, inpsz);
            if (allStarts)
                allStarts[axis] = start;
            if (allSteps)
                allSteps[axis] = step;
            int outsz = step > 0 ? (end - start + step-1)/step :
                                   (start - end - step-1)/(-step);
            if (outsz < 0) {
                outsz = 0;
                end = start;
            }
            if (allEnds)
                allEnds[axis] = end;
            outShape[axis] = outsz;
        }

        return outShape;
    }

    // A block input cut on the channel axis is sliced in its NCHW form, where the bounds count
    // channels rather than blocks, and blocked again; any other input keeps its own shape.
    MatShape getSliceShape(const MatShape& inpShape,
                           const std::vector<int>& starts_,
                           const std::vector<int>& ends_,
                           const std::vector<int>& axes_,
                           const std::vector<int>& steps_,
                           int* allStarts, int* allEnds, int* allSteps,
                           bool* channelsCut = nullptr) const
    {
        if (channelsCut)
            *channelsCut = false;
        if (inpShape.layout == DATA_LAYOUT_BLOCK &&
            getNetImpl(this)->originalLayout == DATA_LAYOUT_NCHW) {
            MatShape nchwShape = inpShape.toLayout(DATA_LAYOUT_NCHW);
            MatShape outShape = getOutShape(nchwShape, starts_, ends_, axes_, steps_,
                                            allStarts, allEnds, allSteps);
            if (allStarts[1] != 0 || allEnds[1] != nchwShape[1] || allSteps[1] != 1) {
                if (channelsCut)
                    *channelsCut = true;
                return outShape.toLayout(DATA_LAYOUT_BLOCK, inpShape.back());
            }
        }
        return getOutShape(inpShape, starts_, ends_, axes_, steps_, allStarts, allEnds, allSteps);
    }

    // Constant channel bounds keep block layout only on whole blocks, i.e. multiples of C0 or the
    // "to the end" value; runtime bounds may fall anywhere, sliceChannelsBlock_() takes any start.
    bool channelCutKeepsBlock(const std::vector<int>& axes_, int C0) const
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        const size_t ninputs = this->inputs.size();
        const size_t k = std::find(axes_.begin(), axes_.end(), 1) - axes_.begin();
        if (ninputs > 4 && !this->inputs[4].empty()) {
            std::vector<int> steps_;
            if (!netimpl_->isConstArg(this->inputs[4]))
                return false;
            tensorToIntVec(netimpl_->argTensor(this->inputs[4]), steps_);
            if (!steps_.empty() && (k >= steps_.size() || steps_[k] != 1))
                return false;
        }
        for (size_t i = 0; i < 2; i++) {
            std::vector<int> bounds = i == 0 ? starts : ends;
            if (ninputs > 1) {
                if (!netimpl_->isConstArg(this->inputs[1 + i]))
                    continue;
                tensorToIntVec(netimpl_->argTensor(this->inputs[1 + i]), bounds);
            }
            if (k >= bounds.size() || bounds[k] < 0 ||
                (bounds[k] % C0 != 0 && bounds[k] != INT_MAX))
                return false;
        }
        return true;
    }

    bool isDataShuffling() const CV_OVERRIDE { return true; }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        DataLayout defaultLayout = netimpl_->originalLayout;
        const size_t ninputs = actualInputs.size();
        desiredInputs = actualInputs;
        outputs.assign(requiredOutputs, DATA_LAYOUT_UNKNOWN);

        const bool inputIsBlock = ninputs >= 1 && actualInputs[0] == DATA_LAYOUT_BLOCK;

        std::vector<int> resolvedAxes = axes;
        if (resolvedAxes.empty() && this->inputs.size() > 3 &&
            netimpl_->isConstArg(this->inputs[3])) {
            Mat axesT = netimpl_->argTensor(this->inputs[3]);
            tensorToIntVec(axesT, resolvedAxes);
        }
        bool axesOK = !resolvedAxes.empty();
        bool cutsChannels = false;
        if (axesOK) {
            const int channelAxis = (defaultLayout == DATA_LAYOUT_NCHW) ? 1 :
                                    (defaultLayout == DATA_LAYOUT_NHWC) ? 3 : -1;
            for (int a : resolvedAxes) {
                if (a < 0) { axesOK = false; break; }
                cutsChannels = cutsChannels || a == channelAxis;
            }
        }
        if (inputIsBlock && axesOK && cutsChannels)
            axesOK = defaultLayout == DATA_LAYOUT_NCHW &&
                     channelCutKeepsBlock(resolvedAxes, netimpl_->defaultC0);

        if (inputIsBlock && axesOK) {
            outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        } else if (inputIsBlock) {
            desiredInputs[0] = defaultLayout;
        }
        return outputs[0] == DATA_LAYOUT_BLOCK ? netimpl_->defaultC0 : 0;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        checkNumInputs(ninputs);
        std::vector<int> tempStarts, tempEnds, tempAxes, steps;
        const std::vector<int> *starts_ = &starts, *ends_ = &ends, *axes_ = &axes;

        if (ninputs > 1) {
            Net::Impl* netimpl_ = getNetImpl(this);
            Mat startsTensor = netimpl_->argTensor(this->inputs[1]);
            tensorToIntVec(startsTensor, tempStarts);
            starts_ = &tempStarts;
            Mat endsTensor = netimpl_->argTensor(this->inputs[2]);
            tensorToIntVec(endsTensor, tempEnds);
            ends_ = &tempEnds;
            if (ninputs > 3) {
                Mat axesTensor = netimpl_->argTensor(this->inputs[3]);
                tensorToIntVec(axesTensor, tempAxes);
                axes_ = &tempAxes;
            }
            if (ninputs > 4) {
                Mat stepsTensor = netimpl_->argTensor(this->inputs[4]);
                tensorToIntVec(stepsTensor, steps);
            }
        }
        int allStarts[MatShape::MAX_DIMS];
        int allEnds[MatShape::MAX_DIMS];
        int allSteps[MatShape::MAX_DIMS];
        MatShape outShape = getSliceShape(inputs[0], *starts_, *ends_, *axes_, steps,
                                          allStarts, allEnds, allSteps);
        outputs.assign(1, outShape);
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
        checkNumInputs(ninputs);
        outputs.assign(requiredOutputs, inputs[0]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }


private:
    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        checkNumInputs(ninputs);

        int inpType = inputs_arr.type(0);
        MatShape inpShape = inputs_arr.shape(0);
        std::vector<int> tempStarts, tempEnds, tempAxes, steps;
        const std::vector<int> *starts_ = &starts, *ends_ = &ends, *axes_ = &axes;

        if (ninputs > 1) {
            Mat startsTensor = inputs_arr.getMat(1);
            tensorToIntVec(startsTensor, tempStarts);
            starts_ = &tempStarts;
            Mat endsTensor = inputs_arr.getMat(2);
            tensorToIntVec(endsTensor, tempEnds);
            ends_ = &tempEnds;
            if (ninputs > 3) {
                Mat axesTensor = inputs_arr.getMat(3);
                tensorToIntVec(axesTensor, tempAxes);
                axes_ = &tempAxes;
            }
            if (ninputs > 4) {
                Mat stepsTensor = inputs_arr.getMat(4);
                tensorToIntVec(stepsTensor, steps);
            }
        }
        int allStarts[MatShape::MAX_DIMS];
        int allEnds[MatShape::MAX_DIMS];
        int allSteps[MatShape::MAX_DIMS];
        bool channelsCut = false;
        MatShape outShape = getSliceShape(inpShape, *starts_, *ends_, *axes_, steps,
                                          allStarts, allEnds, allSteps, &channelsCut);
        // an empty output may keep an out-of-range end, which cv::slice would reject
        const bool emptyOut = outShape.empty();
        auto runSlice = [&](const Mat& inp, Mat& out) {
            if (emptyOut)
                return;
            if (channelsCut) {
                sliceChannelsBlock(inp, out, allStarts, allSteps);
                return;
            }
            const int ndims = inpShape.dims;
            cv::slice(inp, std::vector<int>(allStarts, allStarts + ndims),
                      std::vector<int>(allEnds, allEnds + ndims),
                      std::vector<int>(allSteps, allSteps + ndims), out);
        };

        int outKind = outputs_arr.kind();
        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runSlice(inp, outs[0]);
        } else {
             Mat inp = inputs_arr.getMat(0);
             std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
             outs.resize(1);
             outs[0].fit(outShape, inpType);
             Mat temp(outShape, inpType);
             runSlice(inp, temp);
             temp.copyTo(outs[0]);
        }
    }
};

Ptr<Slice2Layer> Slice2Layer::create(const LayerParams& params)
{
    return Ptr<Slice2Layer>(new Slice2LayerImpl(params));
}

}
}
