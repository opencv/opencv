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
#include "../op_webnn.hpp"
#include "../op_cann.hpp"

#include <algorithm>
#include <stdlib.h>
#include <opencv2/core/utils/logger.hpp>
#include "cpu_kernels/softmax.hpp"
using std::max;

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/softmax.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class SoftMaxLayerImpl CV_FINAL : public SoftmaxLayer
{
public:

    SoftMaxLayerImpl(const LayerParams& params)
    {
        axisRaw = params.get<int>("axis", -1);
        logSoftMax = params.get<bool>("log_softmax", false);
        setParamsFrom(params);
    }

#ifdef HAVE_OPENCL
    Ptr<OCL4DNNSoftmax<float> > softmaxOp;
#endif

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        bool inplace = Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        MatShape shape = inputs[0];
        if (shape.dims > 0) {
            int cAxis = normalize_axis(axisRaw, shape.dims);
            shape[cAxis] = 1;
        }
        internals.assign(1, shape);
        return inplace;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
#ifdef HAVE_WEBNN
        if (backendId == DNN_BACKEND_WEBNN) {
            // TODO: support logSoftMax
            if (logSoftMax)
            {
                CV_LOG_WARNING(NULL, "logSoftMax is not supported by WebNN backend.")
            }
            return !logSoftMax;
        }
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

#ifdef HAVE_OPENCL
    virtual void finalize(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs) CV_OVERRIDE
    {
        softmaxOp.release();
    }

    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        bool use_half = (inputs_.depth() == CV_16F);
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        internals_.getUMatVector(internals);

        UMat& src = inputs[0];
        UMat& dstMat = outputs[0];
        int axis = normalize_axis(axisRaw, src.dims);

        if (softmaxOp.empty())
        {
            OCL4DNNSoftmaxConfig config;
            config.in_shape = shape(inputs[0]);
            config.axis = axis;
            config.channels = inputs[0].size[axis];
            config.logsoftmax = logSoftMax;
            config.use_half = use_half;

            softmaxOp = Ptr<OCL4DNNSoftmax<float> >(new OCL4DNNSoftmax<float>(config));
        }

        if (softmaxOp->Forward(src, dstMat))
            return true;

        UMat& bufMat = internals[0];
        MatShape s = shape(src);
        size_t outerSize = total(s, 0, axis);
        size_t channels = src.size[axis];
        size_t innerSize = total(s, axis + 1);

        String buildOpts = format("-DT=%s", use_half ? "half" : "float");
        ocl::Kernel kmax, ksub, ksum, kdiv;

        if (!kmax.create("kernel_channel_max", ocl::dnn::softmax_oclsrc, buildOpts))
            return false;

        if (!ksub.create("kernel_channel_subtract", ocl::dnn::softmax_oclsrc, buildOpts))
            return false;

        if (!ksum.create("kernel_channel_sum", ocl::dnn::softmax_oclsrc, buildOpts))
            return false;

        if (logSoftMax) buildOpts += " -DLOG_SOFTMAX ";
        if (!kdiv.create("kernel_channel_div", ocl::dnn::softmax_oclsrc, buildOpts))
            return false;

        size_t bufSize = internals[0].total();
        size_t totalSize = src.total();

        size_t internal_globalSize[1] = { bufSize };
        size_t total_globalSize[1] = { totalSize };

        kmax.args((int)outerSize, (int)channels, (int)innerSize,
                  ocl::KernelArg::PtrReadOnly(src), ocl::KernelArg::PtrReadWrite(bufMat));
        if (!kmax.run(1, internal_globalSize, NULL, false))
            return false;

        ksub.args((int)totalSize, (int)outerSize, (int)channels, (int)innerSize,
                  ocl::KernelArg::PtrReadOnly(bufMat),
                  ocl::KernelArg::PtrReadOnly(src), ocl::KernelArg::PtrWriteOnly(dstMat));
        if (!ksub.run(1, total_globalSize, NULL, false))
            return false;

        ksum.args((int)outerSize, (int)channels, (int)innerSize,
                  ocl::KernelArg::PtrReadOnly(dstMat), ocl::KernelArg::PtrReadWrite(bufMat));
        if (!ksum.run(1, internal_globalSize, NULL, false))
            return false;

        kdiv.args((int)totalSize, (int)outerSize, (int)channels, (int)innerSize,
                  ocl::KernelArg::PtrReadOnly(bufMat), ocl::KernelArg::PtrReadWrite(dstMat));
        if (!kdiv.run(1, total_globalSize, NULL, false))
            return false;

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat &src = inputs[0];
        Mat &dst = outputs[0];
        int axis = normalize_axis(axisRaw, src.dims);

        if(logSoftMax)
            logSoftmax(dst, src, axis);
        else
            softmax(dst, src, axis);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto channel_axis = normalize_axis(axisRaw, input_wrapper->getRank());
        return make_cuda_node<cuda4dnn::SoftmaxOp>(preferableTarget, std::move(context->cudnn_handle), channel_axis, logSoftMax);
    }
#endif

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        // create operator
        auto op = std::make_shared<ge::op::SoftmaxV2>(name);

        // set attributes
        op->set_attr_axes(ge::Operator::OpListInt(
            {(int64_t)axisRaw}
        ));

        // set inputs
        // set inputs : x
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        // set outputs
        auto output_y_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_y_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        int axis = normalize_axis(axisRaw, ieInpNode.get_shape().size());
        if (logSoftMax) {
            return new InfEngineNgraphNode(std::make_shared<ov::op::v5::LogSoftmax>(ieInpNode, axis));
        } else {
            return new InfEngineNgraphNode(std::make_shared<ov::op::v1::Softmax>(ieInpNode, axis));
        }
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        auto operand = webnnGraphBuilder.Softmax(webnnInpOperand);
        return Ptr<BackendNode>(new WebnnBackendNode(operand));
    }

#endif

    int64 getFLOPS(const std::vector<MatShape> &inputs,
                  const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        int64 flops = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += 4*total(inputs[i]);
        }

        return flops;
    }

    int axisRaw;
};

Ptr<SoftmaxLayer> SoftmaxLayer::create(const LayerParams& params)
{
    return Ptr<SoftmaxLayer>(new SoftMaxLayerImpl(params));
}

}
}
