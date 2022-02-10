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
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "../op_vkcom.hpp"
#include "../op_webnn.hpp"

#include <float.h>
#include <algorithm>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/permute.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{
class PermuteLayerImpl CV_FINAL : public PermuteLayer
{
public:
    void checkNeedForPermutation()
    {
        _needsPermute = false;
        for (size_t i = 0; i < _numAxes; ++i)
        {
            if (_order[i] != i)
            {
                _needsPermute = true;
                break;
            }
        }
    }

    PermuteLayerImpl(const LayerParams &params)
        : _count(0), _needsPermute(false), _numAxes(0)
    {
        if (!params.has("order"))
        {
            return;
        }

        DictValue paramOrder = params.get("order");
        _numAxes = paramOrder.size();

        for (size_t i = 0; i < _numAxes; i++)
        {
            int currentOrder = paramOrder.get<int>(i);
            if (currentOrder < 0 || currentOrder > _numAxes)
            {
                CV_Error(Error::StsBadArg,
                         format("Orders of dimensions in Permute layer parameter"
                                "must be in [0...%zu]", _numAxes - 1));
            }
            if (std::find(_order.begin(), _order.end(), currentOrder) != _order.end())
            {
                CV_Error(Error::StsBadArg,
                         "Permute layer parameter contains duplicated orders.");
            }
            _order.push_back(currentOrder);
        }

        setParamsFrom(params);
        checkNeedForPermutation();
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && preferableTarget == DNN_TARGET_CPU)
            return _order.size() <= 4 || !isArmComputePlugin();
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_WEBNN ||
               ((backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && haveInfEngine()) ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan());
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        if(!_needsPermute)
        {
            Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
            return true;
        }

        CV_Assert(inputs.size() > 0);
        CV_Assert((int)_numAxes == inputs[0].size());

        MatShape shapeBefore = inputs[0], shapeAfter;
        for (size_t i = 0; i < _numAxes; i++)
        {
            shapeAfter.push_back(shapeBefore[_order[i]]);
        }

        outputs.clear();

        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(total(inputs[i]) == total(shapeAfter));
            outputs.push_back(shapeAfter);
        }

        return false;
    }

    void computeStrides(const MatShape &shapeBefore, const MatShape &shapeAfter)
    {
        _oldStride.resize(_numAxes);
        _newStride.resize(_numAxes);

        _oldStride[_numAxes - 1] = 1;
        _newStride[_numAxes - 1] = 1;

        for(int i = _numAxes - 2; i >= 0; i--)
        {
            _oldStride[i] = _oldStride[i + 1] * shapeBefore[i + 1];
            _newStride[i] = _newStride[i + 1] * shapeAfter[i + 1];
        }

        _count = _oldStride[0] * shapeBefore[0];
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        if(!_needsPermute)
        {
            return;
        }
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() > 0);
        const Mat& inp0 = inputs[0];
        CV_Assert((int)_numAxes == inp0.dims);

        computeStrides(shape(inputs[0]), shape(outputs[0]));

#ifdef HAVE_OPENCL
        uorder.release();
        uold_stride.release();
        unew_stride.release();
#endif
    }

    template <class T>
    class PermuteInvoker : public ParallelLoopBody
    {
    public:
        const Mat* inp;
        Mat* out;
        const std::vector<size_t>* order;
        int nstripes;

        static void run(const Mat& inp, Mat& out, const std::vector<size_t>& order, int nstripes)
        {
            PermuteInvoker p;
            p.inp = &inp;
            p.out = &out;
            p.order = &order;
            p.nstripes = nstripes;

            CV_Assert( out.size[0] == inp.size[order[0]] &&
                      out.size[1] == inp.size[order[1]] &&
                      out.size[2] == inp.size[order[2]] &&
                      out.size[3] == inp.size[order[3]]);

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        PermuteInvoker() : inp(0), out(0), order(0), nstripes(0) {}

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int n0 = out->size[0], n1 = out->size[1], n2 = out->size[2], n3 = out->size[3];

            size_t orows = (size_t)n0*n1*n2;
            size_t stripeSize = (orows + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, orows);

            const size_t esz = sizeof(T);
            size_t ostep0 = out->step[0]/esz, ostep1 = out->step[1]/esz, ostep2 = out->step[2]/esz;
            const size_t* ord = &order->at(0);
            size_t istep0 = inp->step[ord[0]]/esz, istep1 = inp->step[ord[1]]/esz,
            istep2 = inp->step[ord[2]]/esz, istep3 = inp->step[ord[3]]/esz;

            size_t val = stripeStart;
            int i2 = (int)(val % n2);
            val /= n2;
            int i1 = (int)(val % n1);
            int i0 = (int)(val / n1);

            const T* inptr_orig = inp->ptr<T>();
            T* outptr_orig = out->ptr<T>();

            for( size_t ofs = stripeStart; ofs < stripeEnd; ofs++ )
            {
                const T* inptr = inptr_orig + i0*istep0 + i1*istep1 + i2*istep2;
                T* outptr = outptr_orig + i0*ostep0 + i1*ostep1 + i2*ostep2;

                for( int i3 = 0; i3 < n3; i3++ )
                    outptr[i3] = inptr[i3*istep3];

                if( ++i2 >= n2 )
                {
                    i2 = 0;
                    if( ++i1 >= n1 )
                    {
                        i1 = 0;
                        if( ++i0 >= n0 )
                            break;
                    }
                }
            }
        }
    };

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        if (!_needsPermute)
            return false;

        if (uorder.empty())
        {
            std::vector<int> orderVec(_order.begin(), _order.end());;
            Mat morder(1, orderVec.size(), CV_32SC1, &orderVec[0]);

            std::vector<int> oldStrideVec(_oldStride.begin(), _oldStride.end());
            Mat mold_stride(1, _oldStride.size(), CV_32SC1, &oldStrideVec[0]);

            std::vector<int> newStrideVec(_newStride.begin(), _newStride.end());
            Mat mnew_stride(1, newStrideVec.size(), CV_32SC1, &newStrideVec[0]);

            morder.copyTo(uorder);
            mold_stride.copyTo(uold_stride);
            mnew_stride.copyTo(unew_stride);
        }

        bool use_half = (inps.depth() == CV_16S);
        String opts = format("-DDtype=%s", use_half ? "half" : "float");
        for (size_t i = 0; i < inputs.size(); i++)
        {
            ocl::Kernel kernel("permute", ocl::dnn::permute_oclsrc, opts);

            kernel.set(0, (int)_count);
            kernel.set(1, ocl::KernelArg::PtrReadOnly(inputs[i]));
            kernel.set(2, ocl::KernelArg::PtrReadOnly(uorder));
            kernel.set(3, ocl::KernelArg::PtrReadOnly(uold_stride));
            kernel.set(4, ocl::KernelArg::PtrReadOnly(unew_stride));
            kernel.set(5, (int)_numAxes);
            kernel.set(6, ocl::KernelArg::PtrWriteOnly(outputs[i]));

            if (!kernel.run(1, &_count, NULL, false))
                return false;
        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) &&
                   inputs_arr.depth() != CV_8S,
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        size_t k, ninputs = inputs.size();
        if(!_needsPermute)
        {
            for (k = 0; k < ninputs; k++)
            {
                CV_Assert(outputs[k].total() == inputs[k].total());
                if (outputs[k].data != inputs[k].data)
                    inputs[k].copyTo(outputs[k]);
            }
        }
        else
        {
            size_t i, j, count = _count, numAxes = _numAxes;
            const size_t* newStride = &_newStride[0];
            const size_t* oldStride = &_oldStride[0];
            const size_t* order = &_order[0];

            for (k = 0; k < ninputs; k++)
            {
                const Mat& inp = inputs[k];
                Mat& out = outputs[k];

                CV_Assert(inp.dims == numAxes && inp.size == inputs[0].size);
                CV_Assert(out.dims == numAxes && out.size == outputs[0].size);

                CV_Assert(inp.isContinuous() && out.isContinuous());
                // CV_Assert(inp.type() == CV_32F && out.type() == CV_32F);

                if( numAxes == 4 )
                {
                    int nstripes = getNumThreads();
                    if (inp.type() == CV_8S)
                        PermuteInvoker<int8_t>::run(inp, out, _order, nstripes);
                    else
                        PermuteInvoker<float>::run(inp, out, _order, nstripes);
                }
                else
                {
                    if (inp.type() == CV_8S)
                    {
                        const int8_t *srcData = inp.ptr<int8_t>();
                        int8_t *dstData = out.ptr<int8_t>();

                        for (i = 0; i < count; ++i)
                        {
                            size_t oldPosition = 0;
                            size_t newPosition = i;

                            for (j = 0; j < numAxes; ++j)
                            {
                                oldPosition += (newPosition / newStride[j]) * oldStride[order[j]];
                                newPosition %= newStride[j];
                            }
                            dstData[i] = srcData[oldPosition];
                        }
                    }
                    else
                    {
                        const float *srcData = inp.ptr<float>();
                        float *dstData = out.ptr<float>();

                        for (i = 0; i < count; ++i)
                        {
                            size_t oldPosition = 0;
                            size_t newPosition = i;

                            for (j = 0; j < numAxes; ++j)
                            {
                                oldPosition += (newPosition / newStride[j]) * oldStride[order[j]];
                                newPosition %= newStride[j];
                            }
                            dstData[i] = srcData[oldPosition];
                        }
                    }
                }
            }
        }
    }


#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
        InferenceEngine::Builder::PermuteLayer ieLayer(name);
        ieLayer.setOrder(_order);
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<int64_t> order(_order.begin(), _order.end());
        auto tr_axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                       ngraph::Shape({order.size()}), order.data());
        auto transpose = std::make_shared<ngraph::op::Transpose>(ieInpNode, tr_axes);
        return Ptr<BackendNode>(new InfEngineNgraphNode(transpose));
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        std::vector<int32_t> permutation(_order.begin(), _order.end());
        ml::TransposeOptions options;
        options.permutation = permutation.data();
        options.permutationCount = permutation.size();
        auto operand = webnnGraphBuilder.Transpose(webnnInpOperand, &options);
        return Ptr<BackendNode>(new WebnnBackendNode(operand));
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        return make_cuda_node<cuda4dnn::PermuteOp>(preferableTarget, std::move(context->stream), _order);
    }
#endif


#ifdef HAVE_VULKAN
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &input) CV_OVERRIDE
    {
        CV_Assert(!_order.empty());
        std::shared_ptr<vkcom::OpBase> op(new vkcom::OpPermute(_order));
        return Ptr<BackendNode>(new VkComBackendNode(input, op));
    }
#endif // HAVE_VULKAN

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        return true;
    }

    size_t _count;
    std::vector<size_t> _order;

    std::vector<int> _oldDimensionSize;
    std::vector<int> _newDimensionSize;

    std::vector<size_t> _oldStride;
    std::vector<size_t> _newStride;
    bool _needsPermute;

#ifdef HAVE_OPENCL
    UMat uorder, uold_stride, unew_stride;
#endif

    size_t _numAxes;
};

Ptr<PermuteLayer> PermuteLayer::create(const LayerParams &params)
{
    return Ptr<PermuteLayer>(new PermuteLayerImpl(params));
}

}
}
