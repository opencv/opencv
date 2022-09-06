// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


Layer::Layer() { preferableTarget = DNN_TARGET_CPU; }

Layer::Layer(const LayerParams& params)
    : blobs(params.blobs)
    , name(params.name)
    , type(params.type)
{
    preferableTarget = DNN_TARGET_CPU;
}

void Layer::setParamsFrom(const LayerParams& params)
{
    blobs = params.blobs;
    name = params.name;
    type = params.type;
}

int Layer::inputNameToIndex(String)
{
    return -1;
}

int Layer::outputNameToIndex(const String&)
{
    return 0;
}

bool Layer::supportBackend(int backendId)
{
    return backendId == DNN_BACKEND_OPENCV;
}

Ptr<BackendNode> Layer::initCUDA(
        void*,
        const std::vector<Ptr<BackendWrapper>>&,
        const std::vector<Ptr<BackendWrapper>>&)
{
    CV_Error(Error::StsNotImplemented, "CUDA pipeline of " + type + " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initVkCom(const std::vector<Ptr<BackendWrapper>>&)
{
    CV_Error(Error::StsNotImplemented, "VkCom pipeline of " + type + " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initHalide(const std::vector<Ptr<BackendWrapper>>&)
{
    CV_Error(Error::StsNotImplemented, "Halide pipeline of " + type + " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initNgraph(const std::vector<Ptr<BackendWrapper>>& inputs, const std::vector<Ptr<BackendNode>>& nodes)
{
    CV_Error(Error::StsNotImplemented, "Inference Engine pipeline of " + type + " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initWebnn(const std::vector<Ptr<BackendWrapper>>& inputs, const std::vector<Ptr<BackendNode>>& nodes)
{
    CV_Error(Error::StsNotImplemented, "WebNN pipeline of " + type + " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initTimVX(void* timVxInfo,
                                  const std::vector<Ptr<BackendWrapper> > & inputsWrapper,
                                  const std::vector<Ptr<BackendWrapper> > & outputsWrapper,
                                  bool isLast)
{
    CV_Error(Error::StsNotImplemented, "TimVX pipeline of " + type +
                                       " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::initCann(const std::vector<Ptr<BackendWrapper> > &inputsWrapper, const int index, const std::vector<Ptr<BackendNode> >& nodes)
{
    CV_Error(Error::StsNotImplemented, "CANN pipeline of " + type + " layers is not defined.");
    return Ptr<BackendNode>();
}

Ptr<BackendNode> Layer::tryAttach(const Ptr<BackendNode>& node)
{
    return Ptr<BackendNode>();
}

bool Layer::setActivation(const Ptr<ActivationLayer>&) { return false; }
bool Layer::tryFuse(Ptr<Layer>&) { return false; }
void Layer::getScaleShift(Mat& scale, Mat& shift) const
{
    scale = Mat();
    shift = Mat();
}

void Layer::getScaleZeropoint(float& scale, int& zeropoint) const
{
    scale = 1.f;
    zeropoint = 0;
}

void Layer::unsetAttached()
{
    setActivation(Ptr<ActivationLayer>());
}

template <typename T>
static void vecToPVec(const std::vector<T>& v, std::vector<T*>& pv)
{
    pv.resize(v.size());
    for (size_t i = 0; i < v.size(); i++)
        pv[i] = const_cast<T*>(&v[i]);
}

void Layer::finalize(const std::vector<Mat>& inputs, std::vector<Mat>& outputs)
{
    CV_TRACE_FUNCTION();
    this->finalize((InputArrayOfArrays)inputs, (OutputArrayOfArrays)outputs);
}

void Layer::finalize(const std::vector<Mat*>& input, std::vector<Mat>& output)
{
    CV_UNUSED(input);
    CV_UNUSED(output);
}

void Layer::finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> inputs, outputs;
    inputs_arr.getMatVector(inputs);
    outputs_arr.getMatVector(outputs);

    std::vector<Mat*> inputsp;
    vecToPVec(inputs, inputsp);
    this->finalize(inputsp, outputs);
}

std::vector<Mat> Layer::finalize(const std::vector<Mat>& inputs)
{
    CV_TRACE_FUNCTION();

    std::vector<Mat> outputs;
    this->finalize(inputs, outputs);
    return outputs;
}

void Layer::forward(std::vector<Mat*>& input, std::vector<Mat>& output, std::vector<Mat>& internals)
{
    // We kept this method for compatibility. DNN calls it now only to support users' implementations.
}

void Layer::forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());

    Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
}

void Layer::forward_fallback(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());

    if (preferableTarget == DNN_TARGET_OPENCL_FP16 && inputs_arr.depth() == CV_16S)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        std::vector<UMat> orig_inputs;
        std::vector<UMat> orig_outputs;
        std::vector<UMat> orig_internals;

        inputs_arr.getUMatVector(orig_inputs);
        outputs_arr.getUMatVector(orig_outputs);
        internals_arr.getUMatVector(orig_internals);

        inputs.resize(orig_inputs.size());
        for (size_t i = 0; i < orig_inputs.size(); i++)
            convertFp16(orig_inputs[i], inputs[i]);

        outputs.resize(orig_outputs.size());
        for (size_t i = 0; i < orig_outputs.size(); i++)
            outputs[i].create(shape(orig_outputs[i]), CV_32F);

        internals.resize(orig_internals.size());
        for (size_t i = 0; i < orig_internals.size(); i++)
            internals[i].create(shape(orig_internals[i]), CV_32F);

        forward(inputs, outputs, internals);

        for (size_t i = 0; i < outputs.size(); i++)
            convertFp16(outputs[i], orig_outputs[i]);

        // sync results back
        outputs_arr.assign(orig_outputs);
        internals_arr.assign(orig_internals);
        return;
    }
    std::vector<Mat> inpvec;
    std::vector<Mat> outputs;
    std::vector<Mat> internals;

    inputs_arr.getMatVector(inpvec);
    outputs_arr.getMatVector(outputs);
    internals_arr.getMatVector(internals);

    std::vector<Mat*> inputs(inpvec.size());
    for (int i = 0; i < inpvec.size(); i++)
        inputs[i] = &inpvec[i];

    this->forward(inputs, outputs, internals);

    // sync results back
    outputs_arr.assign(outputs);
    internals_arr.assign(internals);
}

void Layer::run(const std::vector<Mat>& inputs, std::vector<Mat>& outputs, std::vector<Mat>& internals)
{
    CV_TRACE_FUNCTION();

    this->finalize(inputs, outputs);
    this->forward(inputs, outputs, internals);
}

bool Layer::tryQuantize(const std::vector<std::vector<float>>& scales,
        const std::vector<std::vector<int>>& zeropoints, LayerParams& params)
{
    return false;
}

Layer::~Layer() {}

bool Layer::getMemoryShapes(const std::vector<MatShape>& inputs,
        const int requiredOutputs,
        std::vector<MatShape>& outputs,
        std::vector<MatShape>& internals) const
{
    CV_Assert(inputs.size());
    outputs.assign(std::max(requiredOutputs, (int)inputs.size()), inputs[0]);
    return false;
}

bool Layer::updateMemoryShapes(const std::vector<MatShape>& inputs)
{
    return true;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
