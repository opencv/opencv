// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include <opencv2/imgproc.hpp>

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#include <ngraph/op/experimental/layers/interpolate.hpp>
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/resize.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv { namespace dnn {

class ResizeLayerImpl : public ResizeLayer
{
public:
    ResizeLayerImpl(const LayerParams& params) : zoomFactorWidth(params.get<int>("zoom_factor_x", params.get<int>("zoom_factor", 0))),
                                                 zoomFactorHeight(params.get<int>("zoom_factor_y", params.get<int>("zoom_factor", 0))),
                                                 scaleWidth(0), scaleHeight(0)
    {
        setParamsFrom(params);
        outWidth = params.get<float>("width", 0);
        outHeight = params.get<float>("height", 0);
        if (params.has("zoom_factor"))
        {
            CV_Assert(!params.has("zoom_factor_x") && !params.has("zoom_factor_y"));
        }
        else if (params.has("zoom_factor_x") || params.has("zoom_factor_y"))
        {
            CV_Assert(params.has("zoom_factor_x") && params.has("zoom_factor_y"));
        }
        interpolation = params.get<String>("interpolation");
        CV_Assert(interpolation == "nearest" || interpolation == "opencv_linear" || interpolation == "bilinear");

        alignCorners = params.get<bool>("align_corners", false);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert_N(inputs.size() == 1, inputs[0].size() == 4);
        outputs.resize(1, inputs[0]);
        outputs[0][2] = zoomFactorHeight > 0 ? (outputs[0][2] * zoomFactorHeight) : outHeight;
        outputs[0][3] = zoomFactorWidth > 0 ? (outputs[0][3] * zoomFactorWidth) : outWidth;
        // We can work in-place (do nothing) if input shape == output shape.
        return (outputs[0][2] == inputs[0][2]) && (outputs[0][3] == inputs[0][3]);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_CUDA)
            return interpolation == "nearest" || interpolation == "bilinear";

#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
            backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            return (interpolation == "nearest" && scaleWidth == scaleHeight) ||
                   (interpolation == "bilinear");
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        outHeight = outputs[0].size[2];
        outWidth = outputs[0].size[3];
        if (alignCorners && outHeight > 1)
            scaleHeight = static_cast<float>(inputs[0].size[2] - 1) / (outHeight - 1);
        else
            scaleHeight = static_cast<float>(inputs[0].size[2]) / outHeight;

        if (alignCorners && outWidth > 1)
            scaleWidth = static_cast<float>(inputs[0].size[3] - 1) / (outWidth - 1);
        else
            scaleWidth = static_cast<float>(inputs[0].size[3]) / outWidth;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        if (outHeight == inputs[0].size[2] && outWidth == inputs[0].size[3])
            return;

        Mat& inp = inputs[0];
        Mat& out = outputs[0];
        if (interpolation == "nearest" || interpolation == "opencv_linear")
        {
            InterpolationFlags mode = interpolation == "nearest" ? INTER_NEAREST : INTER_LINEAR;
            for (size_t n = 0; n < inputs[0].size[0]; ++n)
            {
                for (size_t ch = 0; ch < inputs[0].size[1]; ++ch)
                {
                    resize(getPlane(inp, n, ch), getPlane(out, n, ch),
                           Size(outWidth, outHeight), 0, 0, mode);
                }
            }
        }
        else if (interpolation == "bilinear")
        {
            const int inpHeight = inp.size[2];
            const int inpWidth = inp.size[3];
            const int inpSpatialSize = inpHeight * inpWidth;
            const int outSpatialSize = outHeight * outWidth;
            const int numPlanes = inp.size[0] * inp.size[1];
            CV_Assert_N(inp.isContinuous(), out.isContinuous());

            Mat inpPlanes = inp.reshape(1, numPlanes * inpHeight);
            Mat outPlanes = out.reshape(1, numPlanes * outHeight);
            for (int y = 0; y < outHeight; ++y)
            {
                float input_y = y * scaleHeight;
                int y0 = static_cast<int>(input_y);
                const float* inpData_row0 = inpPlanes.ptr<float>(y0);
                const float* inpData_row1 = inpPlanes.ptr<float>(std::min(y0 + 1, inpHeight - 1));
                for (int x = 0; x < outWidth; ++x)
                {
                    float input_x = x * scaleWidth;
                    int x0 = static_cast<int>(input_x);
                    int x1 = std::min(x0 + 1, inpWidth - 1);

                    float* outData = outPlanes.ptr<float>(y, x);
                    const float* inpData_row0_c = inpData_row0;
                    const float* inpData_row1_c = inpData_row1;
                    for (int c = 0; c < numPlanes; ++c)
                    {
                        *outData = inpData_row0_c[x0] +
                            (input_y - y0) * (inpData_row1_c[x0] - inpData_row0_c[x0]) +
                            (input_x - x0) * (inpData_row0_c[x1] - inpData_row0_c[x0] +
                            (input_y - y0) * (inpData_row1_c[x1] - inpData_row0_c[x1] - inpData_row1_c[x0] + inpData_row0_c[x0]));

                        inpData_row0_c += inpSpatialSize;
                        inpData_row1_c += inpSpatialSize;
                        outData += outSpatialSize;
                    }
                }
            }
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown interpolation: " + interpolation);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        cuda4dnn::InterpolationType itype;
        if (interpolation == "nearest")
            itype = InterpolationType::NEAREST_NEIGHBOUR;
        else if (interpolation == "bilinear")
            itype = InterpolationType::BILINEAR;
        else
            CV_Error(Error::StsNotImplemented, "Requested interpolation mode is not available in resize layer.");

        return make_cuda_node<cuda4dnn::ResizeOp>(preferableTarget, std::move(context->stream), itype, scaleHeight, scaleWidth);
    }
#endif

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::Builder::Layer ieLayer(name);
        ieLayer.setName(name);
        if (interpolation == "nearest")
        {
            ieLayer.setType("Resample");
            ieLayer.getParameters()["type"] = std::string("caffe.ResampleParameter.NEAREST");
            ieLayer.getParameters()["antialias"] = false;
            if (scaleWidth != scaleHeight)
                CV_Error(Error::StsNotImplemented, "resample with sw != sh");
            ieLayer.getParameters()["factor"] = 1.0f / scaleWidth;
        }
        else if (interpolation == "bilinear")
        {
            ieLayer.setType("Interp");
            ieLayer.getParameters()["pad_beg"] = 0;
            ieLayer.getParameters()["pad_end"] = 0;
            ieLayer.getParameters()["align_corners"] = alignCorners;
        }
        else
            CV_Error(Error::StsNotImplemented, "Unsupported interpolation: " + interpolation);
        ieLayer.getParameters()["width"] = outWidth;
        ieLayer.getParameters()["height"] = outHeight;
        ieLayer.setInputPorts(std::vector<InferenceEngine::Port>(1));
        ieLayer.setOutputPorts(std::vector<InferenceEngine::Port>(1));
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;

        ngraph::op::InterpolateAttrs attrs;
        attrs.pads_begin.push_back(0);
        attrs.pads_end.push_back(0);
        attrs.axes = ngraph::AxisSet{2, 3};
        attrs.align_corners = alignCorners;

        if (interpolation == "nearest") {
            attrs.mode = "nearest";
            attrs.antialias = false;
        } else if (interpolation == "bilinear") {
            attrs.mode = "linear";
        } else {
            CV_Error(Error::StsNotImplemented, "Unsupported interpolation: " + interpolation);
        }

        std::vector<int64_t> shape = {outHeight, outWidth};
        auto out_shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, shape.data());
        auto interp = std::make_shared<ngraph::op::Interpolate>(ieInpNode, out_shape, attrs);
        return Ptr<BackendNode>(new InfEngineNgraphNode(interp));
    }
#endif  // HAVE_DNN_NGRAPH

protected:
    int outWidth, outHeight;
    const int zoomFactorWidth, zoomFactorHeight;
    String interpolation;
    float scaleWidth, scaleHeight;
    bool alignCorners;
};


Ptr<ResizeLayer> ResizeLayer::create(const LayerParams& params)
{
    return Ptr<ResizeLayer>(new ResizeLayerImpl(params));
}

class InterpLayerImpl CV_FINAL : public ResizeLayerImpl
{
public:
    InterpLayerImpl(const LayerParams& params) : ResizeLayerImpl(params) {}

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert_N(inputs.size() == 1, inputs[0].size() == 4);
        outputs.resize(1, inputs[0]);
        outputs[0][2] = zoomFactorHeight > 0 ? (1 + zoomFactorHeight * (outputs[0][2] - 1)) : outHeight;
        outputs[0][3] = zoomFactorWidth > 0 ? (1 + zoomFactorWidth * (outputs[0][3] - 1)) : outWidth;
        // We can work in-place (do nothing) if input shape == output shape.
        return (outputs[0][2] == inputs[0][2]) && (outputs[0][3] == inputs[0][3]);
    }
};

Ptr<Layer> InterpLayer::create(const LayerParams& params)
{
    LayerParams lp(params);
    lp.set("interpolation", "bilinear");
    lp.set("align_corners", true);
    return Ptr<Layer>(new InterpLayerImpl(lp));
}

}  // namespace dnn
}  // namespace cv
