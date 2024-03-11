// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class CastLayerImpl CV_FINAL : public CastLayer
{
public:
    CastLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        outputType = params.get<int>("outputType");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        outputs.assign(1, inputs[0]);
        return false;
    }

    virtual  void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        if (preferableTarget == DNN_TARGET_OPENCL_FP16 && outputType == CV_32F)
            outputs.assign(1, CV_16F);
        else
            outputs.assign(1, outputType);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs, outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(outputs.size(), (size_t)1, "");

        if ((outputType == CV_16F && inputs[0].depth() != CV_32F) ||
            (inputs[0].depth() == CV_16F && outputType != CV_32F))
        {
            UMat input_32f;
            inputs[0].convertTo(input_32f, CV_32F);
            input_32f.convertTo(outputs[0], outputType);
        }
        else
        {
            inputs[0].convertTo(outputs[0], outputType);
        }
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
            forward_ocl(inputs_arr, outputs_arr, internals_arr));

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(outputs.size(), (size_t)1, "");

        inputs[0].convertTo(outputs[0], outputType);
    }

private:
    int outputType;
};

Ptr<CastLayer> CastLayer::create(const LayerParams& params)
{
    return makePtr<CastLayerImpl>(params);
}

}}  // namespace cv::dnn
