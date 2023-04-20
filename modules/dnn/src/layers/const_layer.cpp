// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../op_cuda.hpp"
#include "layers_common.hpp"
#include "../ie_ngraph.hpp"
#include "../op_webnn.hpp"
#include "../op_cann.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/const.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv { namespace dnn {

class ConstLayerImpl CV_FINAL : public ConstLayer
{
public:
    ConstLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_WEBNN ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.empty());
        outputs.assign(1, shape(blobs[0]));
        return false;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> outputs;
        outs.getUMatVector(outputs);
        if (outs.depth() == CV_16S)
            convertFp16(blobs[0], outputs[0]);
        else
            blobs[0].copyTo(outputs[0]);
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);
        blobs[0].copyTo(outputs[0]);
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto mat_shape = shape(blobs[0]);
        std::vector<int64_t> mat_shape_{mat_shape.begin(), mat_shape.end()};

        auto ge_shape = ge::Shape(mat_shape_);
        auto ge_dtype = ge::DT_FLOAT;
        switch (blobs[0].type())
        {
            case CV_32F: break;
            case CV_32S: ge_dtype = ge::DT_INT32; break;
            default: CV_Error(Error::StsNotImplemented, "Unsuppported data type");
        }
        auto size_of_type = sizeof(float);
        switch (blobs[0].type())
        {
            case CV_32F: break;
            case CV_32S: size_of_type = sizeof(int); break;
            default: CV_Error(Error::StsNotImplemented, "Unsuppported data type");
        }

        auto desc = std::make_shared<ge::TensorDesc>(ge_shape, ge::FORMAT_NCHW, ge_dtype);
        auto ge_tensor = std::make_shared<ge::Tensor>();
        ge_tensor->SetTensorDesc(*desc);
        ge_tensor->SetData(blobs[0].data, ge_shape.GetShapeSize() * size_of_type);

        auto op = std::make_shared<ge::op::Const>(name);
        op->set_attr_value(*ge_tensor);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto node = std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                                           getShape<size_t>(blobs[0]),
                                                           blobs[0].data);
        return Ptr<BackendNode>(new InfEngineNgraphNode(node));
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        ml::Operand operand = nullptr;
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnGraphBuilder = node->net->builder;
        operand = webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs[0]), blobs[0].data, blobs[0].total()*blobs[0].elemSize(), ml::OperandType::Float32);
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

        CV_Assert(blobs.size() == 1);
        return make_cuda_node<cuda4dnn::ConstOp>(preferableTarget, std::move(context->stream), blobs[0]);
    }
#endif

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        Mat quantizedBlob;
        blobs[0].convertTo(quantizedBlob, CV_8S, 1.f/scales[1][0], zeropoints[1][0]);
        params.blobs.clear();
        params.blobs.push_back(quantizedBlob);
        return true;
    }
};

Ptr<Layer> ConstLayer::create(const LayerParams& params)
{
    return Ptr<Layer>(new ConstLayerImpl(params));
}

}}  // namespace cv::dnn
