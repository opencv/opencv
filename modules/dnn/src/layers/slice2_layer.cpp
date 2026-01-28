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


private:
    template <typename T>
    class ParallelSlice : public cv::ParallelLoopBody
    {
    public:
        ParallelSlice(const Mat& inp, Mat& out,
                      const std::vector<Range>& ranges,
                      const std::vector<int>& steps)
            : inp_(inp), out_(out), ranges_(ranges), steps_(steps)
        {
            dims_ = inp.dims;
            es_ = inp.elemSize();

            inp_strides_.resize(dims_);
            out_strides_.resize(dims_);
            for(int i=0; i<dims_; ++i) {
                inp_strides_[i] = inp.step.p[i];
                out_strides_[i] = out.step.p[i];
            }
        }

        void operator()(const Range& range) const CV_OVERRIDE
        {
            int b = ranges_[0].start;
            int s = steps_[0];

            const uchar* src_base = inp_.ptr();
            uchar* dst_base = out_.ptr();

            for (int i = range.start; i < range.end; ++i)
            {
                int k = b + i * s;
                size_t src_offset = k * inp_strides_[0];
                size_t dst_offset = i * out_strides_[0];
                if (dims_ == 1)
                    std::memcpy(dst_base + dst_offset, src_base + src_offset, es_);
                else
                    recursive_copy(1, src_base + src_offset, dst_base + dst_offset);
            }
        }

        void recursive_copy(int dim, const uchar* src_ptr, uchar* dst_ptr) const
        {
            if (dim >= dims_) return;

            int begin = ranges_[dim].start;
            int end = ranges_[dim].end;
            int step = steps_[dim];

            if (dim == dims_ - 1)
            {
                if (step == 1)
                {
                    size_t count = end - begin;
                    std::memcpy(dst_ptr, src_ptr + begin * inp_strides_[dim], count * es_);
                }
                else
                {
                    const uchar* s_ptr = src_ptr + begin * inp_strides_[dim];
                    uchar* d_ptr = dst_ptr;
                    size_t s_stride = step * inp_strides_[dim];
                    size_t d_stride = out_strides_[dim];

                    if (step > 0)
                    {
                        for (int k = begin; k < end; k += step)
                        {
                            *(T*)d_ptr = *(const T*)s_ptr;
                            s_ptr += s_stride;
                            d_ptr += d_stride;
                        }
                    }
                    else
                    {
                        for (int k = begin; k > end; k += step)
                        {
                            *(T*)d_ptr = *(const T*)s_ptr;
                            s_ptr += s_stride;
                            d_ptr += d_stride;
                        }
                    }
                }
                return;
            }

            if (step == 1 && is_fully_contiguous(dim))
            {
               size_t count = end - begin;
               size_t bytes = count * inp_strides_[dim];
               std::memcpy(dst_ptr, src_ptr + begin * inp_strides_[dim], bytes);
               return;
            }

            size_t src_stride = step * inp_strides_[dim];
            size_t dst_stride = out_strides_[dim];

            const uchar* s_ptr = src_ptr + begin * inp_strides_[dim];
            uchar* d_ptr = dst_ptr;

            if (step > 0)
            {
                for (int k = begin; k < end; k += step)
                {
                    recursive_copy(dim + 1, s_ptr, d_ptr);
                    s_ptr += src_stride;
                    d_ptr += dst_stride;
                }
            }
            else
            {
                for (int k = begin; k > end; k += step)
                {
                    recursive_copy(dim + 1, s_ptr, d_ptr);
                    s_ptr += src_stride;
                    d_ptr += dst_stride;
                }
            }
        }

        bool is_fully_contiguous(int dim) const
        {
            size_t expected_step = es_;
            for (int d = dims_ - 1; d >= dim; --d)
            {
                 if (inp_.step[d] != expected_step) return false;
                 if (d > dim) {
                     if (steps_[d] != 1) return false;
                     if (ranges_[d].start != 0 || ranges_[d].end != inp_.size[d]) return false;
                     expected_step *= inp_.size[d];
                 }
            }
            return true;
        }

    private:
        const Mat& inp_;
        Mat& out_;
        const std::vector<Range>& ranges_;
        const std::vector<int>& steps_;
        int dims_;
        size_t es_;
        std::vector<size_t> inp_strides_;
        std::vector<size_t> out_strides_;
    };

    template <typename T>
    void run_parallel(const Mat& inp, Mat& out, const std::vector<Range>& ranges, const std::vector<int>& steps)
    {
        ParallelSlice<T> body(inp, out, ranges, steps);
        int dim0_size = out.size[0];
        parallel_for_(Range(0, dim0_size), body);
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

        std::vector<Range> ranges;
        std::vector<int> steps_vec;
        for (int i = 0; i < inpShape.dims; ++i) {
            ranges.push_back(Range(allStarts[i], allEnds[i]));
            steps_vec.push_back(allSteps[i]);
        }

        int outKind = outputs_arr.kind();
        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);

            if (inp.depth() == CV_32S) run_parallel<int32_t>(inp, outs[0], ranges, steps_vec);
            else if (inp.depth() == CV_64S) run_parallel<int64_t>(inp, outs[0], ranges, steps_vec);
            else if (inp.depth() == CV_16F) run_parallel<int16_t>(inp, outs[0], ranges, steps_vec);
            else if (inp.depth() == CV_8S) run_parallel<int8_t>(inp, outs[0], ranges, steps_vec);
            else if (inp.depth() == CV_8U) run_parallel<uint8_t>(inp, outs[0], ranges, steps_vec);
            else if (inp.depth() == CV_Bool) run_parallel<uint8_t>(inp, outs[0], ranges, steps_vec);
            else run_parallel<float>(inp, outs[0], ranges, steps_vec);
        } else {
             Mat inp = inputs_arr.getMat(0);
             std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
             outs.resize(1);
             outs[0].fit(outShape, inpType);
             Mat temp(outShape, inpType);

             if (inp.depth() == CV_32S) run_parallel<int32_t>(inp, temp, ranges, steps_vec);
             else if (inp.depth() == CV_64S) run_parallel<int64_t>(inp, temp, ranges, steps_vec);
             else if (inp.depth() == CV_16F) run_parallel<int16_t>(inp, temp, ranges, steps_vec);
             else if (inp.depth() == CV_8S) run_parallel<int8_t>(inp, temp, ranges, steps_vec);
             else if (inp.depth() == CV_8U) run_parallel<uint8_t>(inp, temp, ranges, steps_vec);
             else if (inp.depth() == CV_Bool) run_parallel<uint8_t>(inp, temp, ranges, steps_vec);
             else run_parallel<float>(inp, temp, ranges, steps_vec);

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
