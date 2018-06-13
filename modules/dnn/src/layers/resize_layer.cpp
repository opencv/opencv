// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_inf_engine.hpp"
#include <opencv2/imgproc.hpp>

namespace cv { namespace dnn {

class ResizeLayerImpl CV_FINAL : public ResizeLayer
{
public:
    ResizeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        outWidth = params.get<float>("width", 0);
        outHeight = params.get<float>("height", 0);
        if (params.has("zoom_factor"))
        {
            CV_Assert(!params.has("zoom_factor_x") && !params.has("zoom_factor_y"));
            zoomFactorWidth = zoomFactorHeight = params.get<int>("zoom_factor");
        }
        else if (params.has("zoom_factor_x") || params.has("zoom_factor_y"))
        {
            CV_Assert(params.has("zoom_factor_x") && params.has("zoom_factor_y"));
            zoomFactorWidth = params.get<int>("zoom_factor_x");
            zoomFactorHeight = params.get<int>("zoom_factor_y");
        }
        interpolation = params.get<String>("interpolation");
        CV_Assert(interpolation == "nearest" || interpolation == "bilinear");

        alignCorners = params.get<bool>("align_corners", false);
        if (alignCorners)
            CV_Error(Error::StsNotImplemented, "Resize with align_corners=true is not implemented");
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1, inputs[0].size() == 4);
        outputs.resize(1, inputs[0]);
        outputs[0][2] = outHeight > 0 ? outHeight : (outputs[0][2] * zoomFactorHeight);
        outputs[0][3] = outWidth > 0 ? outWidth : (outputs[0][3] * zoomFactorWidth);
        // We can work in-place (do nothing) if input shape == output shape.
        return (outputs[0][2] == inputs[0][2]) && (outputs[0][3] == inputs[0][3]);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE)
            return interpolation == "nearest" && preferableTarget != DNN_TARGET_MYRIAD;
        else
            return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void finalize(const std::vector<Mat*>& inputs, std::vector<Mat> &outputs) CV_OVERRIDE
    {
        if (!outWidth && !outHeight)
        {
            outHeight = outputs[0].size[2];
            outWidth = outputs[0].size[3];
        }
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (outHeight == inputs[0]->size[2] && outWidth == inputs[0]->size[3])
            return;

        Mat& inp = *inputs[0];
        Mat& out = outputs[0];
        if (interpolation == "nearest")
        {
            for (size_t n = 0; n < inputs[0]->size[0]; ++n)
            {
                for (size_t ch = 0; ch < inputs[0]->size[1]; ++ch)
                {
                    resize(getPlane(inp, n, ch), getPlane(out, n, ch),
                           Size(outWidth, outHeight), 0, 0, INTER_NEAREST);
                }
            }
        }
        else if (interpolation == "bilinear")
        {
            const int inpHeight = inp.size[2];
            const int inpWidth = inp.size[3];
            const int inpSpatialSize = inpHeight * inpWidth;
            const int outSpatialSize = outHeight * outWidth;
            const float heightScale = static_cast<float>(inpHeight) / (outHeight);
            const float widthScale = static_cast<float>(inpWidth) / (outWidth);
            const int numPlanes = inp.size[0] * inp.size[1];
            CV_Assert(inp.isContinuous(), out.isContinuous());

            Mat inpPlanes = inp.reshape(1, numPlanes * inpHeight);
            Mat outPlanes = out.reshape(1, numPlanes * outHeight);
            for (int y = 0; y < outHeight; ++y)
            {
                float input_y = y * heightScale;
                int y0 = static_cast<int>(input_y);
                const float* inpData_row0 = inpPlanes.ptr<float>(y0);
                const float* inpData_row1 = inpPlanes.ptr<float>(std::min(y0 + 1, inpHeight - 1));
                for (int x = 0; x < outWidth; ++x)
                {
                    float input_x = x * widthScale;
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

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "Resample";
        lp.precision = InferenceEngine::Precision::FP32;

        std::shared_ptr<InferenceEngine::CNNLayer> ieLayer(new InferenceEngine::CNNLayer(lp));
        ieLayer->params["type"] = "caffe.ResampleParameter.NEAREST";
        ieLayer->params["antialias"] = "0";
        ieLayer->params["width"] = cv::format("%d", outWidth);
        ieLayer->params["height"] = cv::format("%d", outHeight);

        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

private:
    int outWidth, outHeight, zoomFactorWidth, zoomFactorHeight;
    String interpolation;
    bool alignCorners;
};


Ptr<ResizeLayer> ResizeLayer::create(const LayerParams& params)
{
    return Ptr<ResizeLayer>(new ResizeLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
