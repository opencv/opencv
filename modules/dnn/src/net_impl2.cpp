// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

std::string modelFormatToString(ModelFormat modelFormat)
{
    return
        modelFormat == DNN_MODEL_ONNX ? "ONNX" :
        modelFormat == DNN_MODEL_TF ? "TF" :
        modelFormat == DNN_MODEL_TFLITE ? "TFLite" :
        modelFormat == DNN_MODEL_CAFFE ? "Caffe" : "Unknown/Generic";
}

std::string argKindToString(ArgKind kind)
{
    return
        kind == DNN_ARG_CONST ? "Const" :
        kind == DNN_ARG_INPUT ? "Input" :
        kind == DNN_ARG_OUTPUT ? "Output" :
        kind == DNN_ARG_TEMP ? "Temp" :
        kind == DNN_ARG_PATTERN ? "Pattern" : "???";
}

ArgData::ArgData()
{
    kind = DNN_ARG_EMPTY;
    type = -1;
}

class GraphImpl : public Graph
{
public:
    GraphImpl(const Net& net, const std::string& name,
              const std::vector<Arg>& inputs)
    {
        netimpl_ = net.getImpl();
        name_ = name;
        inputs_ = inputs;
    }

    virtual ~GraphImpl()
    {
    }

    virtual std::string name() const override { return name_; }
    virtual bool empty() const override { return prog_.empty(); }
    virtual void clear() override
    {
        prog_.clear();
    }

    /*Ptr<Graph> clone(Net* newnet) const
    {
        Graph g = std::make_shared<GraphData>((newnet ? *newnet : *net_), name_, inputs_, ispattern_);
        g->outputs_ = outputs_;
        g->backend_ = backend_;
        // don't copy optigraph_. It has to be re-created
        for (auto n : prog_) {
            g->prog_.push_back(n->clone(g->net_));
        }
        return g;
    }*/

    virtual const std::vector<Arg>& append(Ptr<Layer>& layer,
                const std::vector<std::string>& outnames) override
    {
        CV_Assert(layer);
        int i, noutputs = (int)outnames.size();
        //CV_Assert(layer->minNumOutputs() <= noutputs && noutputs <= layer->maxNumOutputs());

        layer->outputs.resize(noutputs);
        for (i = 0; i < noutputs; i++) {
            Arg outarg = netimpl_->getArg(outnames[i]);
            ArgKind kind = netimpl_->argKind(outarg);
            CV_Assert(kind == DNN_ARG_TEMP || kind == DNN_ARG_OUTPUT);
            layer->outputs[i] = outarg;
        }

        prog_.push_back(layer);
        return layer->outputs;
    }

    virtual Arg append(Ptr<Layer>& layer,
               const std::string& outname) override
    {
        std::vector<std::string> outnames = {outname};
        const std::vector<Arg>& outputs = append(layer, outnames);
        CV_Assert(outputs.size() == 1);
        return outputs[0];
    }

    virtual std::ostream& dump(std::ostream& strm, int indent, bool comma) override
    {
        CV_Assert(netimpl_);
        size_t ninputs = inputs_.size(), noutputs = outputs_.size();
        int delta_indent = netimpl_->dump_indent;
        int subindent = indent + delta_indent;
        int argindent = subindent + delta_indent;
        strm << "{\n";
        prindent(strm, subindent);
        strm << "name: ";
        if (name_.empty())
            strm << "<noname>\n";
        else
            strm << '\"' << name_ << "\"\n";
        prindent(strm, subindent);
        strm << "inputs: [\n";
        for (size_t i = 0; i < ninputs; i++) {
            netimpl_->dumpArg(strm, inputs_[i], argindent, i+1 < ninputs, true);
        }
        prindent(strm, subindent);
        strm << "],\n";
        prindent(strm, subindent);
        strm << "outputs: [\n";
        for (size_t i = 0; i < noutputs; i++) {
            netimpl_->dumpArg(strm, outputs_[i], argindent, i+1 < noutputs, true);
        }
        prindent(strm, subindent);
        strm << "],\n";
        prindent(strm, subindent);
        strm << "nodes: [\n";
        size_t nlayers = prog_.size();
        for (size_t i = 0; i < nlayers; i++) {
            prindent(strm, argindent);
            strm << "// op #" << i << "\n";
            const Ptr<Layer>& layer = prog_[i];
            layer->dump(strm, argindent, i+1 < nlayers);
        }
        prindent(strm, subindent);
        strm << "]\n";
        prindent(strm, indent);
        strm << '}';
        if (comma)
            strm << ',';
        strm << '\n';
        return strm;
    }

    //virtual Net* net() const override { return net_; }
    virtual const std::vector<Arg>& inputs() const override { return inputs_; }
    virtual const std::vector<Arg>& outputs() const override { return outputs_; }

    virtual void setOutputs(const std::vector<Arg>& outputs) override {
        netimpl_->checkArgs(outputs);
        outputs_ = outputs;
    }
    virtual const std::vector<Ptr<Layer> >& prog() const override { return prog_; }
    virtual void setProg(const std::vector<Ptr<Layer> >& newprog) override { prog_ = newprog; }

protected:
    Net::Impl* netimpl_;
    std::string name_;
    std::vector<Arg> inputs_;
    std::vector<Arg> outputs_;
    std::vector<Ptr<Layer> > prog_;
};

Ptr<Graph> Graph::create(Net& net, const std::string& name,
                         const std::vector<Arg>& inputs)
{
    return Ptr<Graph>(new GraphImpl(net, name, inputs));
}

Graph::~Graph() {}

bool Net::Impl::isConstArg(Arg arg) const
{
    return argKind(arg) == DNN_ARG_CONST;
}

const ArgData& Net::Impl::argData(Arg arg) const
{
    CV_Assert((size_t)arg.idx < args.size());
    return args[arg.idx];
}

const std::string& Net::Impl::argName(Arg arg) const
{
    return argData(arg).name;
}

ArgKind Net::Impl::argKind(Arg arg) const
{
    return argData(arg).kind;
}

Mat& Net::Impl::argTensor(Arg arg) const
{
    CV_Assert((size_t)arg.idx < tensors.size());
    return const_cast<Mat&>(tensors.at(arg.idx));
}

Arg Net::Impl::getArg(const std::string& name)
{
    if (!name.empty()) {
        auto it = argnames.find(name);
        if (it != argnames.end()) {
            return Arg((int)it->second);
        }
    }
    return newArg(name, DNN_ARG_TEMP);
}

bool Net::Impl::haveArg(const std::string& name) const
{
    return argnames.find(name) != argnames.end();
}

Arg Net::Impl::newConstArg(const std::string& name, const Mat& m)
{
    Arg arg = newArg(name, DNN_ARG_CONST);
    tensors[arg.idx] = m;
    ArgData& adata = args[arg.idx];
    adata.type = m.type();
    adata.shape = m.shape();
    return arg;
}

Arg Net::Impl::newArg(const std::string& name, ArgKind kind)
{
    int idx = (int)args.size();

    if (!name.empty()) {
        CV_Assert(argnames.find(name) == argnames.end());
        argnames.insert(std::make_pair(name, idx));
    }

    ArgData adata;
    adata.name = name;
    adata.kind = kind;
    args.push_back(adata);
    tensors.push_back(Mat());
    bufidxs.push_back(-1);

    return Arg(idx);
}


int Net::Impl::findDim(const std::string& dimname, bool insert)
{
    if (!dimname.empty()) {
        auto it = dimnames.find(dimname);
        if (it != dimnames.end()) {
            return it->second;
        }
    }
    if (!insert) {
        CV_Error_(Error::StsObjectNotFound, ("symbolic dimension '%s' is not found",
                                             dimname.empty() ? "<some unique name>" : dimname.c_str()));
    }
    int value = -(int)dimnames_vec.size() - 1;
    std::string inserted_dimname = dimname.empty() ? format("N!%d", -value) : dimname;
    dimnames.insert(std::make_pair(inserted_dimname, value));
    dimnames_vec.push_back(inserted_dimname);
    return value;
}

void Net::Impl::prepareForInference()
{
    if (!prepared) {
        constFold();
        //inferTypes();
        //constArgs();
        //inferShapes(true);
        //fuse();
        //useBlockLayout();
        //inferShapes(true);
        assignBuffers();
        prepared = true;
        finalizeLayers = true;
    }
}

void Net::Impl::allocateLayerOutputs(
                          const Ptr<Layer>& layer,
                          const std::vector<int>& inpTypes,
                          const std::vector<MatShape>& inpShapes,
                          std::vector<int>& outTypes,
                          std::vector<MatShape>& outShapes,
                          std::vector<Mat>& outputs,
                          std::vector<int>& tempTypes,
                          std::vector<MatShape>& tempShapes,
                          std::vector<Mat>& temps,
                          std::vector<Mat>& globalTemps,
                          bool useBufferPool)
{
    // In theory, when
    // 1) useBufferPool==true,
    // 2) the buffers in the pool are already big enough (e.g. when we already run inference a few times to let them grow)
    // 3) getMemoryShapes() and getTypes() are implemented efficiently without any memory allocations
    // the method allocateLayerOutputs() should not allocate any memory either.
    //
    // Well, currently it still may do so, because Mat::fit() may create e.g. 4D tensor on top of 1D buffer and then
    // MatSize and MatStep will require dynamic memory allocation (those are very small buffers though).
    // But we plan to make MatSize and MatStep lighter so that they don't use dynamic memory.
    size_t noutputs = layer->outputs.size();
    outShapes.clear();
    outTypes.clear();
    layer->getMemoryShapes(inpShapes, (int)noutputs, outShapes, tempShapes);
    layer->getTypes(inpTypes, (int)noutputs, (int)tempShapes.size(), outTypes, tempTypes);
    CV_Assert(tempShapes.size() == tempTypes.size());
    CV_Assert(outShapes.size() == outTypes.size());
    CV_Assert(outShapes.size() == noutputs);
    outputs.resize(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
        Arg out = layer->outputs[i];
        const ArgData& adata = args.at(out.idx);
        if (useBufferPool && adata.kind == DNN_ARG_TEMP) {
            int bufidx = bufidxs.at(out.idx);
            Mat& buf = buffers.at(bufidx);
            buf.fit(outShapes[i], outTypes[i]);
            outputs[i] = buf;
        } else {
            outputs[i].fit(outShapes[i], outTypes[i]);
        }
    }
    // [TODO] probably there should be a smarter algorithm that e.g. sorts
    // temp buffers by size in decreasing order and assigns global temps accordingly
    // in order to minimize the total size of temp buffers
    size_t ntemps = tempShapes.size();
    temps.resize(ntemps);
    globalTemps.resize(std::max(ntemps, globalTemps.size()));
    for (size_t i = 0; i < ntemps; i++) {
        globalTemps[i].fit(tempShapes[i], tempTypes[i]);
        temps[i] = globalTemps[i];
    }
}

void Net::Impl::forwardMainGraph(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
{
    if (!mainGraph) {
        CV_Error(Error::StsNullPtr, "the model was not loaded");
    }
    // [for debugging] tracingMode = DNN_TRACE_OP;
    // [TODO] initialize profile, tracer, symbolic shapes etc.
    size_t nsymdims = dimnames_vec.size();
    dimvalues.assign(nsymdims, -1);
    forwardGraph(mainGraph, inputs, outputs, true);
}

Mat Net::Impl::forwardWithSingleOutput(const std::string& outname)
{
    if (!mainGraph) {
        CV_Error(Error::StsNullPtr, "the model was not loaded");
    }
    const std::vector<Arg>& outargs = mainGraph->outputs();
    CV_Assert(outargs.size() == 1);
    if (!outname.empty()) {
        const ArgData& outdata = args.at(outargs[0].idx);
        CV_Assert(outdata.name == outname);
    }
    std::vector<Mat> inps={}, outs;
    forwardMainGraph(inps, outs);
    return outs[0];
}

/*void Net::Impl::checkAndUpdateDim(const Ptr<Graph>& g, const Ptr<Layer>& layer, Arg inp, int j, int value)
{
    const ArgData& adata = args[inp.idx];
    int64_t value0 = adata.size.size[j];
    if (value0 >= 0) {
        if (value0 != value) {
            CV_Error_(Error::StsBadArg, ("graph '%s': node '%s': %d-th dimension of argument '%s' is wrong: %lld given, %lld expected",
                                        g->name().data(), node ? node->name().data() : "none (graph input)", j, adata.name.c_str(), value, value0));
        }
    } else {
        int64_t idx = -value0-1;
        CV_Assert(0 <= idx && idx < (int64_t)dimvalues.size());
        value0 = dimvalues[idx];
        if (value0 < 0) {
            dimvalues[idx] = value;
        } else if (value0 != value) {
            CV_Error_(Error::StsBadArg,
            ("graph '%s': node '%s': %d-th dimension '%s' of argument '%s' is wrong: %lld given, but '%s' is already set to %lld",
                    g->name().data(), node ? node->name().data() : "none (graph input)",
                    j, dimnames_[idx].c_str(), adata.name.c_str(),
                    value, dimnames_[idx].c_str(), value0));
        }
    }
}*/

void Net::Impl::traceArg(std::ostream& strm_, const char* prefix, size_t i, Arg arg, bool dumpdata)
{
    const Mat& m = tensors.at(arg.idx);
    const ArgData& adata = args.at(arg.idx);
    CV_Assert(m.type() == adata.type);
    strm_ << prefix << " " << i << ". Name: " << adata.name << "\n";
    strm_ << "  Buf: " << bufidxs.at(arg.idx) << "\n";
    strm_ << "  Type: " << typeToString(adata.type) << " \n";
    MatShape shape = m.shape();
    strm_ << "  Shape: " << shape << "\n  Layout: " << layoutToString(shape.layout) << "\n";
    if (dumpdata) {
        Mat temp;
        /*if (size.layout == DNN_LAYOUT_BLOCK) {
            fromBlock->forward(*net, mainGraph, {m}, {fromBlockResult}, scratch_bufs);
            temp = fromBlockResult;
        } else*/ {
            temp = m;
        }
        // [TODO] dump(strm_, m, 0);
        strm_ << "\n";
    }
}

void Net::Impl::setMainGraphInput(InputArray m, const std::string& inpname)
{
    CV_Assert(mainGraph);
    const std::vector<Arg>& gr_inputs = mainGraph->inputs();
    size_t i, ninputs = gr_inputs.size();
    if (inpname.empty()) {
        CV_Assert(ninputs == 1 && "empty name can only be used to set input if there is just one input");
        i = 0;
    } else {
        for (i = 0; i < ninputs; i++) {
            const ArgData& adata = args.at(gr_inputs[i].idx);
            CV_Assert(adata.kind == DNN_ARG_INPUT);
            if (adata.name == inpname)
                break;
        }
        if ((i == ninputs) && (!isdigit(inpname[0]) || !sscanf(inpname.c_str(), "%zu", &i))) {
            CV_Error_(Error::StsObjectNotFound, ("input '%s' is not found", inpname.c_str()));
        }
    }
    setGraphInput(mainGraph, i, m.getMat());
}

void Net::Impl::setGraphInput(Ptr<Graph>& graph, size_t idx, const Mat& m)
{
    int mtype = m.type();
    MatShape mshape = m.shape();
    const std::vector<Arg>& gr_inputs = graph->inputs();
    CV_Assert(idx < gr_inputs.size());
    Arg inp = gr_inputs[idx];
    const ArgData& adata = args.at(inp.idx);
    /*
     [TODO] add more detailed shape check
     if (adata.shape.dims != mshape.dims) {
     CV_Error_(Error::StsBadArg, ("wrong dimensionality of argument '%s': %d given, %d expected",
     adata.name.c_str(), tsize.ndims, adata.size.ndims));
     }

     for (int k = 0; k < mshape.dims; k++) {
        checkAndUpdateDim(graph, Node(), inp, k, tsize.size[k]);
     }
    */

    if (adata.kind == DNN_ARG_INPUT) {
        if (adata.type != mtype) {
            // [TODO] add conversion BF16/FP16/FP32 <=> BF16/FP16/FP32 when the type is 'slightly' mismatched
            CV_Error_(Error::StsBadArg, ("wrong type of argument '%s': %s given, %s expected",
                                         adata.name.c_str(), typeToString(mtype).c_str(),
                                         typeToString(adata.type).c_str()));
        }
        Mat& inp_t = tensors.at(inp.idx);
        inp_t.fit(mshape, mtype);
        m.copyTo(inp_t);
    } else if (adata.kind == DNN_ARG_TEMP) {
        int bufidx = bufidxs.at(inp.idx);
        Mat& buf = buffers.at(bufidx);
        buf.fit(mshape, mtype); // minimize reallocations
        m.copyTo(buf);
        tensors[inp.idx] = buf;
    } else {
        CV_Error_(Error::StsBadArg, ("graph %s: argument '%s' must be 'INPUT' or 'TEMP', not '%s'",
                                     graph->name().data(), adata.name.c_str(), argKindToString(adata.kind).c_str()));
    }
}

void Net::Impl::forwardGraph(Ptr<Graph>& graph, InputArrayOfArrays inputs_,
                             OutputArrayOfArrays outputs_, bool isMainGraph)
{
    std::ostream& strm_ = dump_strm ? *dump_strm : std::cout;
    const std::vector<Ptr<Layer> >& prog = graph->prog();
    size_t i, nops = prog.size();
    const std::vector<Arg>& gr_inputs = graph->inputs();
    const std::vector<Arg>& gr_outputs = graph->outputs();
    size_t n_gr_inputs = gr_inputs.size(), n_gr_outputs = gr_outputs.size();
    std::vector<Mat> inpMats, outMats, tempMats;
    std::vector<int> inpTypes, outTypes, tempTypes;
    std::vector<MatShape> inpShapes, outShapes, tempShapes;
    double timestamp = 0;
    bool already_set_input = false;

    if (inputs_.empty()) {
        // inputs are already set; it's only possible to do with the main graph
        CV_Assert(isMainGraph);
    }
    else {
        if (inputs_.total() != n_gr_inputs) {
            CV_Error_(Error::StsBadArg, ("wrong number of inputs in graph '%s': %zu given, %zu expected",
                                         graph->name().data(), inputs_.total(), n_gr_inputs));
        }
        for (i = 0; i < n_gr_inputs; i++) {
            Mat m = inputs_.getMat((int)i);
            setGraphInput(graph, i, m);
        }
    }

    for (size_t opidx = 0; opidx < nops; opidx++) {
        const Ptr<Layer>& layer = prog.at(opidx);
        if (!layer) // in theory we shouldn't have any 'nops' at this stage, but just in case we skip them.
            continue;
        const std::vector<Arg>& inputs = layer->inputs;
        const std::vector<Arg>& outputs = layer->outputs;
        size_t ninputs = inputs.size(), noutputs = outputs.size();

        inpMats.resize(ninputs);
        inpTypes.resize(ninputs);
        inpShapes.resize(ninputs);
        for (i = 0; i < ninputs; i++) {
            Arg inp = inputs[i];
            //const ArgData& adata = args[inp.idx];
            const Mat& m = tensors[inp.idx];
            inpMats[i] = m;
            inpTypes[i] = m.type();
            inpShapes[i] = m.shape();
        }

        bool dynamicOutShapes = layer->dynamicOutputShapes();
        if (!dynamicOutShapes) {
            allocateLayerOutputs(layer, inpTypes, inpShapes, outTypes, outShapes, outMats,
                                 tempTypes, tempShapes, tempMats, scratchBufs, true);
        } else {
            outMats.resize(noutputs);
            for (i = 0; i < noutputs; i++) {
                Arg out = outputs[i];
                const ArgData& adata = args.at(out.idx);
                if (adata.kind == DNN_ARG_TEMP) {
                    int bufidx = bufidxs.at(out.idx);
                    outMats[i] = buffers.at(bufidx);
                } else {
                    outMats[i] = tensors.at(out.idx);
                }
            }
            tempMats = scratchBufs;
        }

        if (tracingMode != DNN_TRACE_NONE) {
            strm_ << "-----------\n";
            strm_ << "'" << graph->name() << "' [" << opidx << "/" << nops << "]. " << layer->type << " node: " << layer->name << "\n";
            for (i = 0; i < ninputs; i++) {
                Arg inp = inputs[i];
                traceArg(strm_, "Input", i, inp, false);
            }
            timestamp = (double)getTickCount();
        }

        // [TODO] handle If/Loop/...
        CV_Assert(!layer->subgraphs());
        if (finalizeLayers)
            layer->finalize(inpMats, outMats);
        layer->forward(inpMats, outMats, tempMats);
        CV_Assert(outMats.size() == noutputs);

        for (i = 0; i < noutputs; i++) {
            Arg out = outputs[i];
            ArgData& adata = args[out.idx];
            const Mat& m = outMats[i];
            adata.type = m.type();
            adata.shape = m.shape();
            tensors.at(out.idx) = m;
            if (adata.kind == DNN_ARG_TEMP) {
                int bufidx = bufidxs.at(out.idx);
                Mat& buf = buffers.at(bufidx);

                if (!dynamicOutShapes) {
                    // a sanity check: make sure that the data was not reallocated during Layer::forward()
                    // if the layer claims it does not produce dynamic-shape outputs.
                    CV_Assert(buf.data == m.data);
                } else if (m.u->size > buf.u->size) {
                    buf = m;
                } else {
                    // this branch means that the layer still calls create rather than fit and needs to be fixed.
                    // [TODO] should probably log it.
                    buf.fit(m.shape(), m.type());
                    m.copyTo(buf);
                }
            }
        }

        if (tracingMode != DNN_TRACE_NONE) {
            timestamp = (double)getTickCount() - timestamp;
            strm_ << "TIME (\"" << layer->name << "\", \"" << layer->type << "\"): " <<
                format("%.2fms", timestamp*1000/getTickFrequency()) << "\n";
            for (i = 0; i < noutputs; i++) {
                Arg out = outputs[i];
                traceArg(strm_, "Output", i, out, tracingMode == DNN_TRACE_ALL);
            }
        }
    }

    std::vector<Mat>& outputsVec = outputs_.getMatVecRef();
    outputsVec.resize(n_gr_outputs);
    for (i = 0; i < n_gr_outputs; i++) {
        Arg out = gr_outputs[i];
        const Mat& outm = tensors.at(out.idx);
        if (isMainGraph) {
            outputsVec[i].fit(outm.shape(), outm.type());
            outm.copyTo(outputsVec[i]);
        } else {
            outputsVec[i] = outm;
        }
    }
}


void Net::Impl::updateUseCounts(const Ptr<Graph>& graph, std::vector<int>& usecounts) const
{
    if (!graph)
        return;
    const std::vector<Ptr<Layer> >& prog = graph->prog();
    for (const Ptr<Layer>& layer: prog) {
        const std::vector<Arg>& inputs = layer->inputs;
        for (const Arg& input: inputs) {
            CV_Assert(input.idx < (int)usecounts.size());
            usecounts[input.idx]++;
        }
        const std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
        if (subgraphs) {
            for (const Ptr<Graph>& subgraph: *subgraphs) {
                updateUseCounts(subgraph, usecounts);
            }
        }
    }
}

void Net::Impl::useCounts(std::vector<int>& usecounts) const
{
    size_t nargs = args.size();
    usecounts.assign(nargs, 0);
    usecounts[0] = 1; // empty Arg() is always useful
    updateUseCounts(mainGraph, usecounts);
}

void Net::Impl::checkArgs(const std::vector<Arg>& args_) const
{
    for (const Arg& a: args_) {
        checkArg(a);
    }
}

void Net::Impl::checkArg(Arg a) const
{
    CV_Assert(a.idx >= 0);
    CV_Assert(a.idx < (int)args.size());
}

std::ostream& Net::Impl::dumpDim(std::ostream& strm, int value) const
{
    if (value >= 0) {
        strm << value;
    } else {
        size_t idx = -value;
        if (idx < dimnames_vec.size())
            strm << dimnames_vec[idx];
        else
            strm << "sym(" << idx << ")";
    }
    return strm;
}

std::ostream& Net::Impl::dumpTypeShape(std::ostream& strm, int type, const MatShape& shape) const
{
    if (shape.empty()) {
        strm << "<empty>";
    } else {
        strm << typeToString(type);
        if (shape.dims > 0 && shape.layout != DATA_LAYOUT_UNKNOWN) {
            strm << " " << layoutToString(shape.layout);
        }
        strm << " [";
        for (int i = 0; i < shape.dims; i++) {
            strm << (i > 0 ? " x " : "");
            dumpDim(strm, shape[i]);
        }
        strm << "]";
    }
    return strm;
}

std::ostream& Net::Impl::dumpArg(std::ostream& strm, Arg arg, int indent, bool comma, bool dump_details) const
{
    checkArg(arg);
    const ArgData& adata = args.at(arg.idx);
    prindent(strm, indent);
    if (arg.empty()) {
        strm << "<empty>" << (comma ? "," : "");
    } else {
        strm << '\"' << adata.name << (comma ? "\"," : "\"");
        if (dump_details && arg.idx > 0) {
            strm << " // ";
            strm << (adata.kind == DNN_ARG_INPUT ? "<Input>" :
                     adata.kind == DNN_ARG_OUTPUT ? "<Output>" :
                     adata.kind == DNN_ARG_CONST ? "<Const>" :
                     adata.kind == DNN_ARG_TEMP ? "<Temp>" :
                     "<Uknown kind>");
            if (adata.type >= 0) {
                strm << " ";
                dumpTypeShape(strm, adata.type, adata.shape);
            }
            if (adata.kind == DNN_ARG_TEMP && ((size_t)arg.idx < bufidxs.size()))
                strm << " (buf #" << bufidxs[arg.idx] << ")";
        }
    }
    strm << "\n";
    return strm;
}

std::ostream& Net::Impl::dump(std::ostream& strm)
{
    int indent = dump_indent;
    strm << "{\n";
    prindent(strm, indent);
    strm << "model_format: \"" << modelFormatToString(modelFormat) << "\",\n";
    if (modelFormat == DNN_MODEL_ONNX) {
        prindent(strm, indent);
        strm << "onnx_opset: " << onnx_opset << ",\n";
    }
    prindent(strm, indent);
    strm << "layout: \"" << layoutToString(originalLayout) << "\",\n";
    if (mainGraph) {
        prindent(strm, indent);
        strm << "main_graph: ";
        mainGraph->dump(strm, indent, false);
    }
    strm << "}\n";
    return strm;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
