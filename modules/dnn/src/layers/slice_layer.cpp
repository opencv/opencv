/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/slice.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

void sliceRangesFromShape(const MatShape& inpShape, int& axis, std::vector<std::vector<cv::Range> >& sliceRanges)
{
    CV_Assert(inpShape.size() > 0);
    bool axisNeg = (axis < 0);
    axis = (axis + static_cast<int>(inpShape.size())) % inpShape.size();
    int n = inpShape[axis];

    for (size_t i = 0; i < sliceRanges.size(); ++i){
        std::vector<Range>& ranges = sliceRanges[i];
        if (axisNeg)
        {
            ranges.insert(ranges.begin(), axis, Range::all());
        }
        Range& range = ranges.back();

        if (range.start >= 0)
        {
            continue;
        }

        CV_Assert(n != 0);
        range.start = (n + range.start) % n;
    }
}

class SliceLayerImpl : public SliceLayer
{
public:
    SliceLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasSteps = false;
        axis = params.get<int>("axis", 1);
        num_split = params.get<int>("num_split", 0);
        hasDynamicShapes = params.get<bool>("has_dynamic_shapes", false);
        shapesInitialized = !hasDynamicShapes;

        if (params.has("slice_point"))
        {
            CV_Assert(!params.has("begin") && !params.has("size") && !params.has("end"));
            const DictValue &indicesValue = params.get("slice_point");
            int size = axis > 0 ? axis + 1 : 1;
            sliceRanges.resize(indicesValue.size() + 1,
                               std::vector<Range>(size, Range::all()));
            int prevSlice = 0;
            for (int i = 0; i < indicesValue.size(); ++i)
            {
                sliceRanges[i][size - 1].start = prevSlice;
                sliceRanges[i][size - 1].end = indicesValue.get<int>(i);
                prevSlice = sliceRanges[i][size - 1].end;
            }
            sliceRanges.back()[size - 1].start = prevSlice;
        }
        else if (params.has("begin"))
        {
            CV_Assert(params.has("size") ^ params.has("end"));
            const DictValue &begins = params.get("begin");
            const DictValue &sizesOrEnds = params.has("size") ? params.get("size") : params.get("end");
            CV_Assert(begins.size() == sizesOrEnds.size());

            sliceRanges.resize(1);
            sliceRanges[0].resize(begins.size(), Range::all());
            for (int i = 0; i < begins.size(); ++i)
            {
                int start = begins.get<int>(i);
                int sizeOrEnd = sizesOrEnds.get<int>(i);  // It may be negative to reverse indexation.

                sliceRanges[0][i].start = start;
                if (params.has("size"))
                {
                    int size = sizeOrEnd;
                    CV_Assert(size == -1 || size > 0);  // -1 value means range [start, axis_size).
                    sliceRanges[0][i].end = size > 0 ? (start + size) : -1;  // We'll finalize a negative value later.
                }
                else
                {
                    int end = sizeOrEnd;
                    CV_Assert(end < 0 || end > start);  // End index is excluded.
                    sliceRanges[0][i].end = end;  // We'll finalize a negative value later.
                }
            }

            if (params.has("steps"))
            {
                const DictValue &steps = params.get("steps");
                sliceSteps.resize(1);
                sliceSteps[0].resize(steps.size());

                for (int i = 0; i < steps.size(); ++i)
                {
                    int step = steps.get<int>(i);
                    CV_Assert(step >= 1);
                    if (step > 1)
                        hasSteps = true;
                    sliceSteps[0][i] = step;
                }
            }
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return sliceRanges.size() == 1 && !hasSteps;
#endif
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA)
            return !hasSteps;
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                            const int requiredOutputs,
                            std::vector<MatShape> &outputs,
                            std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        MatShape inpShape = inputs[0];

        int axis_rw = axis;
        std::vector<std::vector<cv::Range> > sliceRanges_rw = sliceRanges;
        sliceRangesFromShape(inpShape, axis_rw, sliceRanges_rw);

        if (!sliceRanges_rw.empty())
        {
            outputs.resize(sliceRanges_rw.size(), inpShape);
            for (int i = 0; i < outputs.size(); ++i)
            {
                CV_Assert(sliceRanges_rw[i].size() <= inpShape.size());
                for (int j = 0; j < sliceRanges_rw[i].size(); ++j)
                {
                    if (shapesInitialized || inpShape[j] > 0)
                        outputs[i][j] = normalize_axis_range(sliceRanges_rw[i][j], inpShape[j]).size();

                    if (!sliceSteps.empty() && (i < sliceSteps.size()) && (j < sliceSteps[i].size()) && (sliceSteps[i][j] > 1))
                        outputs[i][j] = (outputs[i][j] + sliceSteps[i][j] - 1) / sliceSteps[i][j];
                }
            }
        }
        else  // Divide input blob on equal parts by axis.
        {
            CV_Assert(0 <= axis_rw && axis_rw < inpShape.size());
            int splits = num_split ? num_split : requiredOutputs;
            CV_Assert(splits > 0 && inpShape[axis_rw] % splits == 0);
            inpShape[axis_rw] /= splits;
            outputs.resize(splits, inpShape);
        }
        return false;
    }

    bool updateMemoryShapes(const std::vector<MatShape> &inputs) CV_OVERRIDE
    {
        shapesInitialized = true;
        return true;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
#ifdef HAVE_OPENCL
        ocl_exec_cache.clear();
#endif

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 1);
        const MatSize& inpShape = inputs[0].size;

        sliceRangesFromShape(shape(inputs[0]), axis, sliceRanges);
        finalSliceRanges = sliceRanges;

        if (sliceRanges.empty())
        {
            // Divide input blob on equal parts by axis.
            int outAxisSize = inpShape[axis] / outputs.size();
            finalSliceRanges.resize(outputs.size(),
                                    std::vector<Range>(axis + 1, Range::all()));
            int prevSlice = 0;
            for (int i = 0; i < outputs.size(); ++i)
            {
                finalSliceRanges[i][axis].start = prevSlice;
                finalSliceRanges[i][axis].end = finalSliceRanges[i][axis].start + outAxisSize;
                prevSlice = finalSliceRanges[i][axis].end;
            }
        }
        else
            CV_Assert(outputs.size() == sliceRanges.size());

        for (int i = 0; i < outputs.size(); ++i)
        {
            CV_Assert(finalSliceRanges[i].size() <= inpShape.dims());
            // Fill the rest of ranges.
            for (int j = finalSliceRanges[i].size(); j < inpShape.dims(); ++j)
            {
                finalSliceRanges[i].push_back(Range::all());
            }
            // Clamp.
            for (int j = 0; j < finalSliceRanges[i].size(); ++j)
            {
                finalSliceRanges[i][j] = normalize_axis_range(finalSliceRanges[i][j], inpShape[j]);
            }
        }

        if (!sliceSteps.empty() && sliceSteps[0].size() != inputs[0].dims)
            sliceSteps[0].resize(inputs[0].dims, 1);

#if 0
        std::cout << "DEBUG: DNN/Slice: " << outputs.size() << " inpShape=" << inpShape << std::endl;
        for (int i = 0; i < outputs.size(); ++i)
        {
            for (int j = 0; j < finalSliceRanges[i].size(); ++j)
            {
                std::cout << finalSliceRanges[i][j];
            }
            std::cout << std::endl;
        }
#endif
    }

#ifdef HAVE_OPENCL
    struct OpenCLExecInfo
    {
        std::string kernel_name;
        std::string build_opts;
        size_t local_size[2];
        size_t global_size[2];

        OpenCLExecInfo()
        {
            local_size[0] = local_size[1] = 0;
            global_size[0] = global_size[1] = 0;
        }
    };
    std::vector<OpenCLExecInfo> ocl_exec_cache;

    void ocl_prepare(const std::vector<UMat>& inputs, const std::vector<UMat>& outputs)
    {
        CV_TRACE_FUNCTION();

        CV_Assert(outputs.size() == finalSliceRanges.size());
        ocl_exec_cache.resize(outputs.size());

        const UMat& input = inputs[0];
        const int dims = input.dims;

        size_t WSZ = 128;

        const int elemSize = (int)input.elemSize();
        String opts0 = cv::format(
                "-DDIMS=%d -DELEMSIZE=%d",
                dims, elemSize
            );
        for (int d = 0; d < dims; d++)
        {
            opts0 += cv::format(" -DSRC_STEP_%d=%d", d, (int)input.step[dims - 1 - d]);
        }
        for (size_t i = 0; i < outputs.size(); i++)
        {
            OpenCLExecInfo& ocl = ocl_exec_cache[i];

            const UMat& output = outputs[i];
            const std::vector<Range>& range = finalSliceRanges[i];

            String opts = opts0;

            CV_CheckEQ(output.dims, dims, "");
            for (int d = 0; d < dims; d++)
            {
                opts += cv::format(" -DDST_STEP_%d=%d -DDST_SZ_%d=%d -DSRC_START_%d=%d",
                        d, (int)output.step[dims - 1 - d],
                        d, (int)output.size[dims - 1 - d],
                        d, (int)range[dims - 1 - d].start
                    );
                CV_CheckEQ(range[d].size(), (int)output.size[d], "");
            }

            const size_t param_LIMIT_BLOCK_SIZE_PER_WG = WSZ * 64;

            int block_dims = 0;
            size_t block_size = elemSize;
            for (int i = dims - 1; i >= 0; --i)
            {
                if (input.step[i] != output.step[i])
                    break;
                block_size *= output.size[i];
                block_dims++;
                if (block_size >= param_LIMIT_BLOCK_SIZE_PER_WG)
                    break;
            }

            const size_t total = output.total() * elemSize;
            size_t num_blocks = total / block_size;

            if ((num_blocks <= 8 && block_size >= WSZ * 4) || (block_size >= param_LIMIT_BLOCK_SIZE_PER_WG))
            {
                // use 1D copy mode
                opts += cv::format(" -DUSE_COPY_1D=1");

                opts += cv::format(" -DBLOCK_DIMS=%d", block_dims);
                opts += cv::format(" -DBLOCK_DIMS_CONTIGUOUS=%d", block_dims);
                opts += cv::format(" -DBLOCK_SIZE=%d", (int)block_size);

                opts += cv::format(" -DBLOCK_COLS=%d", (int)block_size);
            }
            else
            {
                // use 2D copy mode
                int block_cols = block_size;
                int block_dims_contiguous = block_dims;
                size_t input_base_step = input.step[dims - 1 - block_dims_contiguous];
                size_t output_base_step = output.step[dims - 1 - block_dims_contiguous];

                size_t block_rows = 1;
                for (int i = dims - 1 - block_dims_contiguous; i >= 0; --i)
                {
                    if (input.step[i] * output_base_step != output.step[i] * input_base_step)
                        break;
                    block_rows *= output.size[i];
                    block_dims++;
                }

                block_size *= block_rows;

                num_blocks = total / block_size;

                if (block_rows > 1)
                {
                    opts += cv::format(" -DBLOCK_DIMS=%d", block_dims);
                    opts += cv::format(" -DBLOCK_DIMS_CONTIGUOUS=%d", block_dims_contiguous);
                    opts += cv::format(" -DBLOCK_SIZE=%d", (int)block_size);

                    opts += cv::format(" -DBLOCK_COLS=%d", (int)block_cols);

                    opts += cv::format(" -DBLOCK_ROWS=%d", (int)block_rows);
                    opts += cv::format(" -DBLOCK_SRC_STRIDE=%d", (int)input_base_step);
                }
                else
                {
                    // use 1D copy mode
                    opts += cv::format(" -DUSE_COPY_1D=1");

                    opts += cv::format(" -DBLOCK_DIMS=%d", block_dims_contiguous);
                    opts += cv::format(" -DBLOCK_DIMS_CONTIGUOUS=%d", block_dims_contiguous);
                    opts += cv::format(" -DBLOCK_SIZE=%d", (int)block_size);

                    opts += cv::format(" -DBLOCK_COLS=%d", (int)block_size);
                }
            }

            const size_t MIN_WORK_ITEMS = 16;
            if (block_size <= 4 * MIN_WORK_ITEMS)
                WSZ = 4;
            else if (block_size <= 8 * MIN_WORK_ITEMS)
                WSZ = 8;
            else if (block_size <= 16 * MIN_WORK_ITEMS)
                WSZ = 16;
            else if (block_size <= 32 * MIN_WORK_ITEMS)
                WSZ = 32;
            else if (block_size <= 64 * MIN_WORK_ITEMS)
                WSZ = 64;

            opts += cv::format(" -DWSZ=%d", (int)WSZ);

            std::ostringstream kernel_suffix;
            kernel_suffix << dims << 'x' << elemSize << "_bsz" << block_size;
            kernel_suffix << "__src_";
            for (int d = 0; d < dims; d++)
            {
                kernel_suffix << input.size[dims - 1 - d] << '_';
            }
            kernel_suffix << '_';
            /*for (int d = 0; d < dims; d++)
            {
                kernel_suffix << input.step[dims - 1 - d] << '_';
            }
            kernel_suffix << '_';*/

            kernel_suffix << "dst_";
            for (int d = 0; d < dims; d++)
            {
                kernel_suffix << output.size[dims - 1 - d] << '_';
            }
            /*kernel_suffix << '_';
            for (int d = 0; d < dims; d++)
            {
                kernel_suffix << output.step[dims - 1 - d] << '_';
            }*/
            kernel_suffix << "_slice_";
            for (int d = 0; d < dims; d++)
            {
                kernel_suffix << range[dims - 1 - d].start << '_';
            }
            for (int d = 0; d < dims; d++)
            {
                kernel_suffix << '_' << range[dims - 1 - d].end;
            }

            std::string kernel_suffix_str = kernel_suffix.str();
            opts += cv::format(" -DSLICE_KERNEL_SUFFIX=%s", kernel_suffix_str.c_str());

            ocl.kernel_name = cv::format("slice_%s", kernel_suffix_str.c_str());
            ocl.build_opts = opts;
            ocl.local_size[0] = WSZ;
            ocl.local_size[1] = 1;
            ocl.global_size[0] = WSZ;
            ocl.global_size[1] = num_blocks;
        }  // for outputs.size()
    }  // ocl_prepare

    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        CV_TRACE_FUNCTION();

        if (hasSteps)
            return false;  // TODO not implemented yet: https://github.com/opencv/opencv/pull/19546

        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        CV_Assert(outputs.size() == finalSliceRanges.size());

        const UMat& input = inputs[0];
        const int dims = input.dims;
        if (dims > 5)
        {
            CV_LOG_INFO(NULL, "DNN/OpenCL/Slice: implementation doesn't support dims=" << dims << ". Fallback to CPU");
            return false;
        }

        if (ocl_exec_cache.empty())
        {
            ocl_prepare(inputs, outputs);
        }
        CV_CheckEQ(ocl_exec_cache.size(), outputs.size(), "");

        for (size_t i = 0; i < outputs.size(); i++)
        {
            const OpenCLExecInfo& ocl = ocl_exec_cache[i];

            UMat& output = outputs[i];

            ocl::Kernel kernel(ocl.kernel_name.c_str(), ocl::dnn::slice_oclsrc, ocl.build_opts);
            if (kernel.empty())
                return false;
            bool ret = kernel.args(
                    ocl::KernelArg::PtrReadOnly(input),
                    ocl::KernelArg::PtrWriteOnly(output)
                )
                .run_(2, (size_t*)ocl.global_size, (size_t*)ocl.local_size, false);
            if (!ret)
                return false;
        }  // for outputs.size()

        return true;
    }  // forward_ocl
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& inpMat = inputs[0];
        CV_Assert(outputs.size() == finalSliceRanges.size());

        if (!hasSteps)
        {
            for (size_t i = 0; i < outputs.size(); i++)
            {
                inpMat(finalSliceRanges[i]).copyTo(outputs[i]);
            }
        }
        else
        {
            int dimsNum = inpMat.dims;

            for (size_t i = 0; i < outputs.size(); i++)
            {
                std::vector<int> inpIdx(dimsNum, 0);
                std::vector<int> outIdx(dimsNum, 0);
                if (inpMat.type() == CV_16S)
                    getSliceRecursive<int16_t>(inpMat, inpIdx, finalSliceRanges[i], sliceSteps[i], 0, dimsNum, outputs[i], outIdx);
                else if (inpMat.type() == CV_8S)
                    getSliceRecursive<int8_t>(inpMat, inpIdx, finalSliceRanges[i], sliceSteps[i], 0, dimsNum, outputs[i], outIdx);
                else
                    getSliceRecursive<float>(inpMat, inpIdx, finalSliceRanges[i], sliceSteps[i], 0, dimsNum, outputs[i], outIdx);
            }
        }
    }


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert_N(nodes.size() <= 2);
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        CV_Assert(finalSliceRanges[0].size() == ieInpNode->get_shape().size());

        std::vector<int64_t> offsets, dims;
        for (int i = 0; i < finalSliceRanges[0].size(); ++i)
        {
            offsets.push_back(finalSliceRanges[0][i].start);
            dims.push_back(finalSliceRanges[0][i].end);
        }

        auto lower_bounds = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                             ngraph::Shape{offsets.size()}, offsets.data());
        auto upper_bounds = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                             ngraph::Shape{dims.size()}, dims.data());
        auto strides = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                        ngraph::Shape{dims.size()}, std::vector<int64_t>((int64_t)dims.size(), 1));

        auto slice = std::make_shared<ngraph::op::v1::StridedSlice>(ieInpNode,
                                      lower_bounds, upper_bounds, strides, std::vector<int64_t>{}, std::vector<int64_t>{});

        return Ptr<BackendNode>(new InfEngineNgraphNode(slice));
    }
#endif  // HAVE_DNN_NGRAPH


#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        std::vector<std::vector<std::size_t>> offsets;
        for (const auto& ranges : finalSliceRanges)
        {
            std::vector<std::size_t> offsets_i;
            for (const auto& range : ranges)
                offsets_i.push_back(range.start);
            offsets.push_back(std::move(offsets_i));
        }

        return make_cuda_node<cuda4dnn::SliceOp>(preferableTarget, std::move(context->stream), std::move(offsets));
    }
#endif

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        const int numOutputs = scales[1].size();
        for (int i = 0; i < numOutputs; i++)
        {
            if (scales[1][i] != scales[0][0])
             return false;
        }
        return true;
    }

private:
    template <typename T>
    void getSliceRecursive(const Mat &inpMat, std::vector<int> &inpIdx,
                           const std::vector<Range> &sliceRanges,
                           const std::vector<int> &sliceSteps, int dim, int dimsNum,
                           Mat &outputs, std::vector<int> &outIdx)
    {
        int begin = sliceRanges[dim].start;
        int end = sliceRanges[dim].end;
        int step = !sliceSteps.empty() ? sliceSteps[dim] : 1;

        // TODO optimization is required (for 2D tail case at least)
        for (int k = begin, j = 0; k < end; k += step, j++)
        {
            inpIdx[dim] = k;
            outIdx[dim] = j;

            if (dim + 1 < dimsNum)
                getSliceRecursive<T>(inpMat, inpIdx, sliceRanges, sliceSteps, dim + 1, dimsNum, outputs, outIdx);
            else
                outputs.at<T>(outIdx.data()) = inpMat.at<T>(inpIdx.data());
        }
    }

protected:
    // The actual non-negative values determined from @p sliceRanges depends on input size.
    std::vector<std::vector<Range> > finalSliceRanges;
    bool hasDynamicShapes;
    bool shapesInitialized;
    bool hasSteps;
};

class CropLayerImpl CV_FINAL : public SliceLayerImpl
{
public:
    CropLayerImpl(const LayerParams& params) : SliceLayerImpl(LayerParams())
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 2);
        const DictValue *paramOffset = params.ptr("offset");

        if (paramOffset)
        {
            for (int i = 0; i < paramOffset->size(); i++)
                offset.push_back(paramOffset->get<int>(i));
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);

        MatShape dstShape = inputs[0];
        int start = normalize_axis(axis, dstShape);
        for (int i = start; i < dstShape.size(); i++)
        {
            dstShape[i] = inputs[1][i];
        }
        outputs.resize(1, dstShape);
        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(2 == inputs.size());

        const Mat &inpBlob = inputs[0];
        const Mat &inpSzBlob = inputs[1];

        int dims = inpBlob.dims;
        int start_axis = normalize_axis(axis, dims);

        std::vector<int> offset_final(dims, 0);
        if (offset.size() == 1)
        {
            for (int i = start_axis; i < dims; i++)
                offset_final[i] = offset[0];
        }
        else if (offset.size() > 1)
        {
            if ((int)offset.size() != dims - start_axis)
                CV_Error(Error::StsBadArg, "number of offset values specified must be "
                                           "equal to the number of dimensions following axis.");

            for (int i = start_axis; i < dims; i++)
                offset_final[i] = offset[i - start_axis];
        }

        finalSliceRanges.resize(1);
        finalSliceRanges[0].resize(dims);
        for (int i = 0; i < start_axis; i++)
        {
            finalSliceRanges[0][i] = Range(0, inpBlob.size[i]);
        }
        for (int i = start_axis; i < dims; i++)
        {
            if (offset_final[i] < 0 || offset_final[i] + inpSzBlob.size[i] > inpBlob.size[i])
                CV_Error(Error::StsBadArg, "invalid crop parameters or blob sizes");

            finalSliceRanges[0][i] = Range(offset_final[i], offset_final[i] + inpSzBlob.size[i]);
        }
    }

private:
    std::vector<int> offset;
};

Ptr<SliceLayer> SliceLayer::create(const LayerParams& params)
{
    return Ptr<SliceLayer>(new SliceLayerImpl(params));
}

Ptr<Layer> CropLayer::create(const LayerParams& params)
{
    return Ptr<Layer>(new CropLayerImpl(params));
}

}
}
