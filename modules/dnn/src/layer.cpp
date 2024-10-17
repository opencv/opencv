// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


Layer::Layer() {
    netimpl = nullptr;
    preferableTarget = DNN_TARGET_CPU;
}

Layer::Layer(const LayerParams& params)
    : blobs(params.blobs)
    , name(params.name)
    , type(params.type)
{
    netimpl = nullptr;
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

Ptr<BackendNode> Layer::initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs,
                                  std::vector<Ptr<BackendWrapper> > &outputs)
{
    CV_Error(Error::StsNotImplemented, "VkCom pipeline of " + type + " layers is not defined.");
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

Ptr<BackendNode> Layer::initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                 const std::vector<Ptr<BackendWrapper> > &outputs,
                                 const std::vector<Ptr<BackendNode> >& nodes)
{
    CV_Error(Error::StsNotImplemented, "CANN pipeline of " + type + " layers is not defined.");
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

    if (preferableTarget == DNN_TARGET_OPENCL_FP16 && inputs_arr.depth() == CV_16F)
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
            if (orig_inputs[i].depth() == CV_16F)
                orig_inputs[i].convertTo(inputs[i], CV_32F);
            else
                inputs[i] = orig_inputs[i];

        outputs.resize(orig_outputs.size());
        for (size_t i = 0; i < orig_outputs.size(); i++)
            if (orig_outputs[i].depth() == CV_16F)
                outputs[i].create(shape(orig_outputs[i]), CV_32F);
            else
                outputs[i] = orig_outputs[i];

        internals.resize(orig_internals.size());
        for (size_t i = 0; i < orig_internals.size(); i++)
            if (orig_internals[i].depth() == CV_16F)
                internals[i].create(shape(orig_internals[i]), CV_32F);
            else
                internals[i] = orig_internals[i];

        forward(inputs, outputs, internals);

        for (size_t i = 0; i < outputs.size(); i++)
            if (orig_outputs[i].depth() == CV_16F)
                outputs[i].convertTo(orig_outputs[i], CV_16F);
            else
                outputs[i] = orig_outputs[i];

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

void Layer::getTypes(const std::vector<MatType>&inputs,
                     const int requiredOutputs,
                     const int requiredInternals,
                     std::vector<MatType>&outputs,
                     std::vector<MatType>&internals) const
{
    CV_Assert(inputs.size());
    for (auto input : inputs)
    {
        if (preferableTarget == DNN_TARGET_CUDA_FP16 || preferableTarget == DNN_TARGET_CUDA)
            CV_CheckTypeEQ(input, CV_32F, "");
        else if (preferableTarget == DNN_TARGET_OPENCL_FP16)
            CV_CheckType(input, input == CV_16F || input == CV_8S, "");
        else
            CV_CheckType(input, input == CV_32F || input == CV_8S, "");
    }

    outputs.assign(requiredOutputs, inputs[0]);
    internals.assign(requiredInternals, inputs[0]);
}

int64 Layer::getFLOPS(const std::vector<MatShape>&,
                      const std::vector<MatShape>&) const
{
    return 0;
}

bool Layer::updateMemoryShapes(const std::vector<MatShape>& inputs)
{
    return true;
}

std::vector<Ptr<Graph> >* Layer::subgraphs() const
{
    return nullptr;
}

bool Layer::alwaysSupportInplace() const
{
    return false;
}

bool Layer::dynamicOutputShapes() const
{
    return false;
}

std::ostream& Layer::dumpAttrs(std::ostream& strm, int) const
{
    return strm;
}

std::ostream& Layer::dump(std::ostream& strm, int indent, bool comma) const
{
    CV_Assert(netimpl);
    size_t ninputs = inputs.size();
    size_t noutputs = outputs.size();
    size_t nblobs = blobs.size();
    const std::vector<Ptr<Graph> >* subgraphs_ = subgraphs();
    size_t nsubgraphs = subgraphs_ ? subgraphs_->size() : 0;
    Net::Impl* netimpl = getNetImpl(this);
    int delta_indent = netimpl->dump_indent;
    int subindent = indent + delta_indent;
    int argindent = subindent + delta_indent;
    prindent(strm, indent);
    std::string opname = type;
    strm << opname << " {\n";
    prindent(strm, subindent);
    strm << "name: \"" << name << "\",\n";

    if (!blobs.empty()) {
        prindent(strm, subindent);
        strm << "blobs: [\n";
        for (size_t i = 0; i < nblobs; i++) {
            if (i > 0)
                strm << ",\n";
            const Mat& blob = blobs[i];
            prindent(strm, argindent);
            netimpl->dumpTypeShape(strm, blob.type(), blob.shape());
        }
        strm << "\n";
        prindent(strm, subindent);
        strm << "],\n";
    }
    dumpAttrs(strm, subindent);
    prindent(strm, subindent);
    strm << "inputs: [\n";
    for (size_t i = 0; i < ninputs; i++) {
        netimpl->dumpArg(strm, inputs[i], argindent, i+1 < ninputs, true);
    }
    prindent(strm, subindent);
    strm << "],\n";
    prindent(strm, subindent);
    strm << "outputs: [\n";
    for (size_t i = 0; i < noutputs; i++) {
        netimpl->dumpArg(strm, outputs[i], argindent, i+1 < noutputs, true);
    }
    prindent(strm, subindent);
    strm << "],\n";

    if (nsubgraphs > 0) {
        std::vector<std::string> names;
        if (opname == "If")
            names = {"then", "else"};
        else if (opname == "Loop")
            names = {"body"};
        else {
            CV_Error(Error::StsError,
                     format("unsupported operation '%s' with subgraphs",
                            std::string(opname).c_str()));
        }
        CV_Assert(names.size() == nsubgraphs);
        for (size_t i = 0; i < nsubgraphs; i++) {
            prindent(strm, subindent);
            strm << names[i] << ": ";
            subgraphs_->at(i)->dump(strm, argindent, i+1 < nsubgraphs);
        }
    }
    prindent(strm, indent);
    strm << '}';
    if (comma)
        strm << ',';
    strm << '\n';
    return strm;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
