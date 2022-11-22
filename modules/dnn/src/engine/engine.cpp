// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

Net2::Net2()
{
    impl_ = makePtr<Impl>();
    impl_->net = this;
}

Net2::~Net2() {}

Ptr<Net2::Impl> Net2::impl() const { return impl_; }

bool Net2::empty() const { return impl_->empty(); }

void Net2::forward(InputArrayOfArrays inputBlobs,
                   OutputArrayOfArrays outputBlobs)
{
    impl_->forward(inputBlobs, outputBlobs);
}

void Net2::getInputNames(std::vector<String>& inputs) const
{
    size_t i, ninputs = impl_->graph.inputs.size();
    inputs.resize(ninputs);
    for (i = 0; i < ninputs; i++) {
        int argidx = impl_->graph.inputs[i];
        inputs[i] = impl_->args.at(argidx).name;
    }
}

void Net2::getOutputNames(std::vector<String>& outputs) const
{
    size_t i, noutputs = impl_->graph.outputs.size();
    outputs.resize(noutputs);
    for (i = 0; i < noutputs; i++) {
        int argidx = impl_->graph.outputs[i];
        outputs[i] = impl_->args.at(argidx).name;
    }
}

void Net2::set(int propId, double value)
{
    impl_->set(propId, value);
}

double Net2::get(int propId) const
{
    return impl_->get(propId);
}

Net2::Impl::Impl()
{
    net = 0;
    modelFormat = DNN_MODEL_GENERIC;
    defaultLayout = DNN_LAYOUT_NCHW;
    enableFP16 = false;
    haveFP16 = false; // [TODO] update based on the architecture
    trace = false;
    profile = true;
    traceProfile = false;

    defaultDevice = 0;
    defaultMemoryManager = 0;
}

void Net2::Impl::clear()
{
    argnames.clear();
    dimnames.clear();
    dimnames_.clear();
    args.clear();
    tensors.clear();
    bufidxs.clear();
    buffers.clear();
    graph.clear();
    addConstTensor("", Tensor());
    findDim("?");
}

Net2::Impl::~Impl() {}

LayerArg::LayerArg()
{
    kind = DNN_ARG_TEMP;
    typ = -1;
}

int Net2::Impl::addConstTensor(const std::string& name, const Tensor& t, int idx)
{
    if (idx < 0) {
        ArgInfo arginfo;
        arginfo.name = name;
        arginfo.typ = t.typ;
        int i, ndims = t.shape.ndims;
        for (i = 0; i < ndims; i++)
            arginfo.shape.push_back(TensorDim((int64_t)t.shape.shape[i]));
        idx = addArg(DNN_ARG_CONST, arginfo);
    }
    tensors[idx] = t;
    return idx;
}

int64_t Net2::Impl::findDim(const std::string& dimname)
{
    auto it = dimnames.find(dimname);
    if (it != dimnames.end())
        return it->second;
    dimnames_.push_back(dimname);
    int dimidx = -(int)dimnames_.size();
    dimnames[dimname] = dimidx;
    return dimidx;
}

int Net2::Impl::addArg(int argkind, const ArgInfo& arginfo)
{
    int argidx = (int)args.size();
    LayerArg arg;
    arg.name = arginfo.name;
    arg.kind = argkind;
    arg.typ = arginfo.typ;
    size_t i, ndims = arginfo.shape.size();
    arg.shape.ndims = (int)ndims;
    for (i = 0; i < ndims; i++) {
        TensorDim dim = arginfo.shape[i];
        int64_t value = dim.value;
        if (value < 0)
            value = findDim(dim.param);
        arg.shape.shape[i] = value;
    }
    args.push_back(arg);
    tensors.push_back(Tensor());
    CV_Assert(argnames.find(arg.name) == argnames.end());
    argnames[arg.name] = argidx;
    return argidx;
}

int Net2::Impl::findArg(const std::string& argname)
{
    auto it = argnames.find(argname);
    return it == argnames.end() ? -1 : it->second;
}

int Net2::Impl::findOutputArg(const std::string& argname)
{
    auto it = argnames.find(argname);
    if (it != argnames.end())
        return it->second;
    {
    ArgInfo arginfo;
    arginfo.name = argname;
    arginfo.typ = -1;
    return addArg(DNN_ARG_TEMP, arginfo);
    }
}

bool Net2::Impl::isConst(int argidx) const
{
    return args.at(argidx).kind == DNN_ARG_CONST;
}

int Net2::Impl::kind(int argidx) const
{
    return args.at(argidx).kind;
}

bool Net2::Impl::empty() const { return graph.empty(); }

void Net2::Impl::updateUseCounts(std::vector<int>& usecounts, const Graph& graph) const
{
    /* 1. when we have the main graph, we gonna use its outputs in one way or another,
          so we increment the use count.
       2. when we have a subgraph, we need to copy (which could possibly done in-place, i.e.
          without actual copying) its formal outputs to the actual outputs,
          specified in If, Loop, Scan etc. To reflect it, we increment the use counts as well.
       So, whether it's the main graph or a subgraph, we increment 'usage counter' of each
       its output
    */
    for (auto output: graph.outputs)
        usecounts.at(output)++;
    for (const auto& op: graph.prog) {
        for (auto op_input: op.inputs)
            usecounts.at(op_input)++;
        for (const auto& pgr: op.subgraphs)
            if (!pgr.empty()) {
                const Graph& subgraph = *pgr.get();
                updateUseCounts(usecounts, subgraph);
            }
    }
}

void Net2::Impl::useCounts(vector<int>& usecounts) const
{
    usecounts.assign(args.size(), 0);
    updateUseCounts(usecounts, graph);
}

void Net2::Impl::set(int propId, double value)
{
    if (propId == Net2::PROP_TRACE)
        trace = value != 0;
    else if (propId == Net2::PROP_PROFILE)
        profile = value != 0;
    else
        CV_Error(Error::StsBadArg, "unknown property");
}

double Net2::Impl::get(int propId) const
{
    if (propId == Net2::PROP_TRACE)
        return (double)trace;
    else if (propId == Net2::PROP_PROFILE)
        return (double)profile;
    else
        CV_Error(Error::StsBadArg, "unknown property");
    return 0.;
}

bool Net2::Impl::useFP16() const { return enableFP16 && haveFP16; }

void Net2::Impl::forward(InputArrayOfArrays inputBlobs,
                         OutputArrayOfArrays outputBlobs)
{
    size_t i, ninputs = inputBlobs.total();
    CV_Assert(ninputs == graph.inputs.size());

    // set inputs
    for (i = 0; i < ninputs; i++) {
        int argidx = graph.inputs[i];
        LayerArg& arg = args.at(argidx);
        Mat m = inputBlobs.getMat((int)i);
        DataLayout layout = arg.shape.layout;
        if (layout == DNN_LAYOUT_UNKNOWN)
            layout = m.dims < 3 ? DNN_LAYOUT_ND : defaultLayout;
        CV_Assert(arg.kind == DNN_ARG_INPUT);
        TensorShape mshape = TensorShape::fromArray(m, arg.shape.ndims, layout);
        int mtype = m.type();
        int argtype = arg.typ != CV_16F && arg.typ != CV_32F ? arg.typ :
                    useFP16() ? CV_16F : CV_32F;

        CV_Assert(mtype == argtype ||
                  (mtype == CV_32F && argtype == CV_16F) ||
                  (mtype == CV_16F && argtype == CV_32F));
        // [TODO] check for the proper shape
        tensors[argidx].fit(mshape, argtype);
        Mat dst(m.dims, m.size, argtype, tensors[argidx].data());
        m.convertTo(dst, argtype);
    }

    forwardGraph(graph);

    // copy the results to outputs
    size_t noutputs = graph.outputs.size();
    vector<Mat> blobs(noutputs);

    for (i = 0; i < noutputs; i++) {
        int argidx = graph.outputs[i];
        blobs[i] = tensors.at(argidx).getMat();
    }
    outputBlobs.create((int)noutputs, 1, CV_32F/*FIXIT*/, -1);  // allocate vector
    outputBlobs.assign(blobs);
}

void Net2::Impl::forwardGraph(const Graph& graph)
{
    /*
       1. check that all the inputs are initialized
       2. for each operation:
          * check that all the inputs are initialized
          * run shape inference
          * initialize output tensors
          * convert inputs & outputs to mat's
          * call the operation
          * convert outputs back to tensors
    */
    std::vector<TensorShape> inpshapes, outshapes;
    std::vector<int> inptypes, outtypes;
    std::vector<Mat> inptensors, outtensors;
    std::vector<Mat> scratchbufs;
    size_t opidx, nops = graph.prog.size();

    for (opidx = 0; opidx < nops; opidx++) {
        const Node& node = graph.prog[opidx];
        const PLayer& layer = node.op;
        size_t j, ninputs = node.inputs.size(), noutputs = node.outputs.size();
        if (layer.empty())
            continue;
        if (trace) {
            printf("-----------\n");
            printf("'%s' [%d/%d]. %s node: %s\n", graph.name.c_str(),
                   (int)opidx, (int)nops, layer->type.c_str(), layer->name.c_str());
            for (j = 0; j < ninputs; j++) {
                int argidx = node.inputs[j];
                dumpArg("Input", (int)j, argidx, false);
            }
        }
        if (layer->type == "Reshape")
            putchar('.');

        inptensors.clear();
        inpshapes.clear();
        inptypes.clear();
        outtensors.clear();
        outtypes.clear();
        outshapes.clear();

        // initialize inputs
        for (j = 0; j < ninputs; j++) {
            int argidx = node.inputs[j];
            // [TODO] check that argidx is computed
            Tensor& t = tensors.at(argidx);
            int bufidx = bufidxs.at(argidx);
            if (bufidx >= 0)
                t.buf = buffers.at(bufidx);
            // [TODO] map the inputs using MemoryManager API
            inptensors.push_back(t.getMat());
            inpshapes.push_back(t.shape);
            inptypes.push_back(t.typ);
        }
        layer->inferOutputShapes(*net, node.inputs, inptypes, inpshapes,
                                  node.outputs, outtypes, outshapes);
        CV_Assert(outshapes.size() == noutputs);
        CV_Assert(outtypes.size() == noutputs);

        // allocate output tensors
        for (j = 0; j < noutputs; j++) {
            int argidx = node.outputs[j];
            Tensor& t = tensors.at(argidx);
            int argkind = kind(argidx);
            if (argkind == DNN_ARG_TEMP) {
                int bufidx = bufidxs.at(argidx);
                CV_Assert(bufidx >= 0);
                t.buf = buffers.at(bufidx);
            } else {
                CV_Assert(argkind == DNN_ARG_OUTPUT);
            }
            t.fit(outshapes[j], outtypes[j]);
            outtensors.push_back(t.getMat());
        }

        // [TODO] handle If, Loop ...
        layer->forward(inptensors, outtensors, scratchbufs);
        CV_Assert(outtensors.size() == noutputs);
        for (j = 0; j < noutputs; j++) {
            int argidx = node.outputs[j];
            LayerArg& arg = args.at(argidx);
            const Mat& m = outtensors[j];
            Tensor& t = tensors.at(argidx);
            arg.typ = t.typ;
            /*
                the j-th output has been reallocated during execution;
                it's possible with "truly dynamic" operations, such as
                If, Loop, NonMaxSuppression etc.
                In this case we need to copy it back to the buffer
            */
            if (t.data() != (void*)m.data) {
                t.set(TensorShape::fromArray(m, outshapes[j].ndims), m.type(), m.data, true);
            }
            int bufidx = bufidxs.at(argidx);
            // Even if the operation does not reallocates certain output tensor,
            // it can be extended according to the shape inference results.
            // So we need to copy it back. If the buffer was not reallocated,
            // the assignment below will not do any copying.
            if (bufidx >= 0)
                buffers.at(bufidx) = t.buf;
        }
        if (trace) {
            printf("-----------\n");
            for (j = 0; j < noutputs; j++) {
                int argidx = node.outputs[j];
                dumpArg("Output", (int)j, argidx, true);
            }
            printf("\n");
        }
    }
}

bool Graph::empty() const { return prog.empty(); }
void Graph::clear()
{
    inputs.clear();
    outputs.clear();
    prog.clear();
}

CV__DNN_INLINE_NS_END
}}
