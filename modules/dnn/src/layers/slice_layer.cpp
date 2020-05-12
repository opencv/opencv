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

class SliceLayerImpl : public SliceLayer
{
public:
    SliceLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 1);
        num_split = params.get<int>("num_split", 0);
        if (params.has("slice_point"))
        {
            CV_Assert(!params.has("begin") && !params.has("size") && !params.has("end"));
            const DictValue &indicesValue = params.get("slice_point");
            sliceRanges.resize(indicesValue.size() + 1,
                               std::vector<Range>(axis + 1, Range::all()));
            int prevSlice = 0;
            for (int i = 0; i < indicesValue.size(); ++i)
            {
                sliceRanges[i][axis].start = prevSlice;
                sliceRanges[i][axis].end = indicesValue.get<int>(i);
                prevSlice = sliceRanges[i][axis].end;
            }
            sliceRanges.back()[axis].start = prevSlice;
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
                CV_Assert(start >= 0);

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
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_DNN_IE_NN_BUILDER_2019
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
            return INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1) &&
                sliceRanges.size() == 1 && sliceRanges[0].size() == 4;
#endif
#ifdef HAVE_DNN_NGRAPH
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return sliceRanges.size() == 1;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                            const int requiredOutputs,
                            std::vector<MatShape> &outputs,
                            std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        MatShape inpShape = inputs[0];

        if (!sliceRanges.empty())
        {
            outputs.resize(sliceRanges.size(), inpShape);
            for (int i = 0; i < outputs.size(); ++i)
            {
                CV_Assert(sliceRanges[i].size() <= inpShape.size());
                for (int j = 0; j < sliceRanges[i].size(); ++j)
                {
                    outputs[i][j] = clamp(sliceRanges[i][j], inpShape[j]).size();
                }
            }
        }
        else  // Divide input blob on equal parts by axis.
        {
            CV_Assert(0 <= axis && axis < inpShape.size());
            int splits = num_split ? num_split : requiredOutputs;
            CV_Assert(splits > 0 && inpShape[axis] % splits == 0);
            inpShape[axis] /= splits;
            outputs.resize(splits, inpShape);
        }
        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 1);
        const MatSize& inpShape = inputs[0].size;

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
                finalSliceRanges[i][j] = clamp(finalSliceRanges[i][j], inpShape[j]);
            }
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inputs_.depth() == CV_16S);
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        if (inputs[0].dims < 4 || (total(shape(outputs[0]), 0, 2) % 4 != 0) ||
            (total(shape(outputs[0]), 2) % 4 != 0))
            return false;

        String opts;
        if (use_half)
            opts = "-DDtype=half -DDtype4=half4 -DDtype8=half8";
        else
            opts = "-DDtype=float -DDtype4=float4 -DDtype8=float8";
        const UMat& inpMat = inputs[0];
        for (size_t i = 0; i < outputs.size(); i++)
        {
            int groups = outputs[i].size[0];
            int channels = outputs[i].size[1];
            int rows = outputs[i].size[2];
            int cols = outputs[i].size[3];

            ocl::Kernel kernel("slice", ocl::dnn::slice_oclsrc, opts);
            size_t local[] = { 128 };
            size_t global[] = { (size_t)groups * channels / 4 * local[0] };
            int idx = 0;
            kernel.set(idx++, ocl::KernelArg::PtrReadOnly(inpMat));
            kernel.set(idx++, (int)(inpMat.size[2] * inpMat.size[3]));
            kernel.set(idx++, (int)(rows * cols));
            kernel.set(idx++, (int)inpMat.size[3]);
            kernel.set(idx++, (int)cols);
            kernel.set(idx++, (int)finalSliceRanges[i][2].start);
            kernel.set(idx++, (int)finalSliceRanges[i][3].start);
            kernel.set(idx++, ocl::KernelArg::PtrWriteOnly(outputs[i]));
            bool ret = kernel.run(1, global, local, false);
            if (!ret)
                return false;
        }

        return true;
    }
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
        for (size_t i = 0; i < outputs.size(); i++)
        {
            inpMat(finalSliceRanges[i]).copyTo(outputs[i]);
        }
    }


#ifdef HAVE_DNN_IE_NN_BUILDER_2019
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1)
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
        CV_Assert_N(finalSliceRanges.size() == 1, inputs.size() <= 2);

        std::vector<size_t> axes, offsets, dims;
        int from, to, step;
        int numDims = finalSliceRanges[0].size();
        if (preferableTarget == DNN_TARGET_MYRIAD)
        {
            from = axis;
            to = numDims;
            step = 1;
        }
        else
        {
            from = numDims - 1;
            to = axis - 1;
            step = -1;
        }
        for (int i = from; i != to; i += step)
        {
            axes.push_back(i);
            offsets.push_back(finalSliceRanges[0][i].start);
            dims.push_back(finalSliceRanges[0][i].size());
        }

        InferenceEngine::Builder::Layer ieLayer(name);
        ieLayer.setName(name);
        ieLayer.setType("Crop");
        ieLayer.getParameters()["axis"] = axes;
        ieLayer.getParameters()["dim"] = dims;
        ieLayer.getParameters()["offset"] = offsets;
        ieLayer.setInputPorts(std::vector<InferenceEngine::Port>(2));
        ieLayer.setOutputPorts(std::vector<InferenceEngine::Port>(1));

        if (inputs.size() != 2)
        {
            std::vector<size_t> outShape(numDims);
            for (int i = 0; i < numDims; ++i)
                outShape[i] = finalSliceRanges[0][i].size();

            ieLayer.getInputPorts()[1].setParameter("type", "weights");

            auto shapeSource = InferenceEngine::make_shared_blob<float>({
                                   InferenceEngine::Precision::FP32, outShape,
                                   InferenceEngine::Layout::ANY
                               });
            shapeSource->allocate();
            addConstantData("weights", shapeSource, ieLayer);
        }
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif
#endif


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


protected:
    // The actual non-negative values determined from @p sliceRanges depends on input size.
    std::vector<std::vector<Range> > finalSliceRanges;
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
        int start = clamp(axis, dstShape);
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
        int start_axis = clamp(axis, dims);

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
