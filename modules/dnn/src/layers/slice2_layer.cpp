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

/* Slice op for CPU.
   starts_, ends_ and steps_ must contain as many elements as
   the dimensionality in inp and out.
*/
static void slice(const Mat& inp, const int* starts_,
                  const int*, const int* steps_,
                  Mat& out)
{
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// in this function steps can be negative, so
    /// please don't replace int64_t's with size_t's
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    enum {SLICE_MAX_DIMS=7};

    CV_Assert_N(inp.isContinuous(), out.isContinuous());
    CV_Assert(inp.type() == out.type());
    CV_Assert_N(inp.dims <= SLICE_MAX_DIMS, inp.dims == out.dims);

    MatShape inpShape = inp.shape();
    MatShape outShape = out.shape();
    int64_t esz = (int64_t)inp.elemSize();

    int ndims = inpShape.dims;
    int starts[SLICE_MAX_DIMS], steps[SLICE_MAX_DIMS];
    int inpsz[SLICE_MAX_DIMS], outsz[SLICE_MAX_DIMS];
    int64_t inpstep[SLICE_MAX_DIMS];

    int delta = SLICE_MAX_DIMS - ndims;
    bool emptyOut = false;

    for (int i = 0; i < SLICE_MAX_DIMS; i++) {
        inpsz[i] = outsz[i] = steps[i] = 1;
        starts[i] = 0;
    }

    for (int i = 0; i < ndims; i++) {
        inpsz[delta + i] = inpShape[i];
        outsz[delta + i] = outShape[i];
        starts[delta + i] = starts_[i];
        steps[delta + i] = steps_[i];
        if (outShape[i] == 0)
            emptyOut = true;
    }

    for (int i = SLICE_MAX_DIMS-1; i >= 0; i--)
        inpstep[i] = i == SLICE_MAX_DIMS-1 ? 1 : inpstep[i+1]*inpsz[i+1];

    const uchar* inptr0 = inp.data;

    for (int i = 0; i < SLICE_MAX_DIMS; i++) {
        inptr0 += starts[i]*inpstep[i]*esz;
        inpstep[i] *= steps[i];
    }

    int sz0 = outsz[6], sz1 = outsz[5];
    int sz2 = outsz[4], sz3 = outsz[3];
    int sz4 = outsz[2], sz5 = outsz[1], sz6 = outsz[0];
    int64_t p0 = inpstep[6], p1 = inpstep[5];
    int64_t p2 = inpstep[4], p3 = inpstep[3];
    int64_t p4 = inpstep[2], p5 = inpstep[1], p6 = inpstep[0];

    #undef CV_IMPLEMENT_SLICE
    #define CV_IMPLEMENT_SLICE(typ) \
        typ* outptr = (typ*)(out.data); \
        for(int i6 = 0; i6 < sz6; i6++) { \
        for(int i5 = 0; i5 < sz5; i5++) { \
        for(int i4 = 0; i4 < sz4; i4++) { \
        for(int i3 = 0; i3 < sz3; i3++) { \
        for(int i2 = 0; i2 < sz2; i2++) { \
        for(int i1 = 0; i1 < sz1; i1++, outptr += sz0) { \
            const typ* inptr = (const typ*)inptr0 + i6*p6 + \
                    i5*p5 + i4*p4 + i3*p3 + i2*p2 + i1*p1; \
            int i0 = 0; \
            if (p0 == 1) { \
                for (; i0 < sz0; i0++) \
                    outptr[i0] = inptr[i0]; \
            } \
            else { \
                for (; i0 <= sz0 - 4; i0 += 4) { \
                    int64_t ip0 = i0*p0; \
                    typ t0 = inptr[ip0], t1 = inptr[ip0 + p0]; \
                    typ t2 = inptr[ip0 + p0*2], t3 = inptr[ip0 + p0*3]; \
                    outptr[i0] = t0; outptr[i0+1] = t1; \
                    outptr[i0+2] = t2; outptr[i0+3] = t3; \
                } \
                for (; i0 < sz0; i0++) \
                    outptr[i0] = inptr[i0*p0]; \
            } \
        }}}}}}

    if (emptyOut) return;
    if (esz == 4) {
        CV_IMPLEMENT_SLICE(int)
    } else if (esz == 2) {
        CV_IMPLEMENT_SLICE(int16_t)
    } else if (esz == 1) {
        CV_IMPLEMENT_SLICE(int8_t)
    } else if (esz == 8) {
        CV_IMPLEMENT_SLICE(int64_t)
    } else {
        CV_Error(Error::StsNotImplemented, "");
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
            if (allEnds)
                allEnds[axis] = end;
            if (allSteps)
                allSteps[axis] = step;
            int outsz = step > 0 ? (end - start + step-1)/step :
                                   (start - end - step-1)/(-step);
            CV_Assert(outsz >= 0);
            outShape[axis] = outsz;
        }

        return outShape;
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
        MatShape outShape = getOutShape(inputs[0], *starts_, *ends_, *axes_, steps);
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
        MatShape outShape = getOutShape(inpShape, *starts_, *ends_, *axes_, steps,
                                        allStarts, allEnds, allSteps);

        int outKind = outputs_arr.kind();

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runOp(inp, allStarts, allEnds, allSteps, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            Mat temp(outShape, inpType);
            runOp(inp, allStarts, allEnds, allSteps, temp);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& inp, const int* starts_,
               const int* ends_, const int* steps_, Mat& out)
    {
        slice(inp, starts_, ends_, steps_, out);
    }
};

Ptr<Slice2Layer> Slice2Layer::create(const LayerParams& params)
{
    return Ptr<Slice2Layer>(new Slice2LayerImpl(params));
}

}
}
