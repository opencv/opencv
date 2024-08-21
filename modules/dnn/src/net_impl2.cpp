// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

std::string argKindToString(ArgKind kind)
{
    return
        kind == DNN_ARG_CONST ? "Const" :
        kind == DNN_ARG_INPUT ? "Input" :
        kind == DNN_ARG_OUTPUT ? "Output" :
        kind == DNN_ARG_TEMP ? "Temp" :
        kind == DNN_ARG_PATTERN ? "Pattern" : "???";
}

/*
[TODO] move code from these methods into net_impl.cpp.
Net::Impl::Impl(Net2* net_)
{
    CV_Assert(net_ != nullptr);
    net = net_;
    modelFormat = DNN_MODEL_GENERIC;
    defaultLayout = LAYOUT_NCHW;
    onnx_opset = 0;

    defaultDevice = Device::CPU();
    defaultMemoryManager = MemoryManager::forCPU();

    accuracy = CV_32F;
    enableFP16 = haveFP16 = false;
    if (checkHardwareSupport(CV_CPU_FP16)) {
        enableFP16 = haveFP16 = true;
    }

    tracingMode = DNN_TRACE_NONE;
    profilingMode = DNN_PROFILE_NONE;
    prepared = false;

    strm = &std::cout;
    dump_indent = 3;

    clear();
}

Net::Impl::~Impl() { clear(); }

void Net::Impl::prepareForInference()
{
    if (!prepared) {
        constFold();
        inferTypes();
        constArgs();
        inferShapes(true);
        fuse();
        useBlockLayout();
        inferShapes(true);
        assignBuffers();
        prepared = true;
    }
}

void Net::Impl::clear()
{
    modelFormat = DNN_MODEL_GENERIC;

    dimnames = NamesHash();
    dimnames_ = std::vector<std::string>();

    args = std::vector<ArgInfo>();
    argnames = NamesHash();

    tensors = std::vector<Tensor>();
    bufidxs = std::vector<int>();
    buffers = std::vector<Buffer>();

    mainGraph = Graph();

    pattern_args = std::vector<ArgInfo>();
    pattern_tensors = std::vector<Tensor>();

    ArgInfo info;
    args.push_back(info);
    pattern_args.push_back(info);
    tensors.push_back(Tensor());
    bufidxs.push_back(-1);

    fromBlock = TransformLayoutOp::create(LAYOUT_NCHW);
}*/

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
    layer->getMemoryShapes(inpShapes, (int)noutputs, outShapes, tempShapes);
    layer->getTypes(inpTypes, (int)noutputs, (int)tempShapes.size(), outTypes, tempTypes);
    CV_Assert(tempShapes.size() == tempTypes.size());
    CV_Assert(outShapes.size() == outTypes.size());
    CV_Assert(outShapes.size() == noutputs);
    outputs.resize(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
        Arg out = layer->outputs[i];
        const ArgInfo& info = args.at(out.idx);
        if (useBufferPool && info.kind == DNN_ARG_TEMP) {
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
    // [TODO] initialize profile, tracer, symbolic shapes etc.
    size_t nsymdims = dimnames_.size();
    dimvalues.assign(nsymdims, -1);
    forwardGraph(mainGraph, inputs, outputs, true);
}

/*void Net::Impl::checkAndUpdateDim(const Ptr<Graph>& g, const Ptr<Layer>& layer, Arg inp, int j, int value)
{
    const ArgInfo& info = args[inp.idx];
    int64_t value0 = info.size.size[j];
    if (value0 >= 0) {
        if (value0 != value) {
            CV_Error_(Error::StsBadArg, ("graph '%s': node '%s': %d-th dimension of argument '%s' is wrong: %lld given, %lld expected",
                                        g->name().data(), node ? node->name().data() : "none (graph input)", j, info.name.c_str(), value, value0));
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
                    j, dimnames_[idx].c_str(), info.name.c_str(),
                    value, dimnames_[idx].c_str(), value0));
        }
    }
}*/

void Net::Impl::traceArg(std::ostream& strm_, const char* prefix, size_t i, Arg arg, bool dumpdata)
{
    const Mat& m = tensors.at(arg.idx);
    const ArgInfo& info = args.at(arg.idx);
    CV_Assert(m.type() == info.type);
    strm_ << prefix << " " << i << ". Name: " << info.name << "\n";
    strm_ << "  Buf: " << bufidxs.at(arg.idx) << "\n";
    strm_ << "  Type: " << typeToString(info.type) << " \n";
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

void Net::Impl::forwardGraph(Ptr<Graph>& graph, InputArrayOfArrays inputs_,
                             OutputArrayOfArrays outputs_, bool isMainGraph)
{
    std::ostream& strm_ = traceStream ? *traceStream : std::cout;
    const std::vector<Ptr<Layer> >& prog = graph->prog();
    size_t i, nops = prog.size();
    const std::vector<Arg>& gr_inputs = graph->inputs();
    const std::vector<Arg>& gr_outputs = graph->outputs();
    size_t n_gr_inputs = gr_inputs.size(), n_gr_outputs = gr_outputs.size();
    std::vector<Mat> inpMats, outMats, tempMats;
    std::vector<int> inpTypes, outTypes, tempTypes;
    std::vector<MatShape> inpShapes, outShapes, tempShapes;
    double timestamp = 0;

    if (inputs_.total() != n_gr_inputs) {
        CV_Error_(Error::StsBadArg, ("wrong number of inputs in graph '%s': %zu given, %zu expected",
                                     graph->name().data(), inputs_.total(), n_gr_inputs));
    }

    for (i = 0; i < n_gr_inputs; i++) {
        // [TODO] add conversion if needed
        Mat m = inputs_.getMat((int)i);
        int mtype = m.type();
        MatShape mshape = m.shape();
        Arg inp = gr_inputs[i];
        const ArgInfo& info = args.at(inp.idx);
        if (info.type != mtype) {
            CV_Error_(Error::StsBadArg, ("wrong type of argument '%s': %s given, %s expected",
                                         info.name.c_str(), typeToString(mtype).c_str(),
                                         typeToString(info.type).c_str()));
        }

        /*
        [TODO] ignore extensive shape check for now
        if (info.shape.dims != mshape.dims) {
            CV_Error_(Error::StsBadArg, ("wrong dimensionality of argument '%s': %d given, %d expected",
                                         info.name.c_str(), tsize.ndims, info.size.ndims));
        }

        for (int k = 0; k < mshape.dims; k++) {
            checkAndUpdateDim(graph, Node(), inp, k, tsize.size[k]);
        }*/

        if (info.kind == DNN_ARG_INPUT) {
            tensors.at(inp.idx) = m;
        } else if (info.kind == DNN_ARG_TEMP) {
            int bufidx = bufidxs.at(inp.idx);
            Mat& buf = buffers.at(bufidx);
            buf.fit(mshape, mtype); // minimize reallocations
            m.copyTo(buf);
            tensors[inp.idx] = buf;
        } else {
            CV_Error_(Error::StsBadArg, ("graph %s: argument '%s' must be 'INPUT' or 'TEMP', not '%s'",
                                         graph->name().data(), info.name.c_str(), argKindToString(info.kind).c_str()));
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
            //const ArgInfo& info = args[inp.idx];
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
                const ArgInfo& info = args.at(out.idx);
                if (info.kind == DNN_ARG_TEMP) {
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
        layer->forward(inpMats, outMats, tempMats);
        CV_Assert(outMats.size() == noutputs);

        for (i = 0; i < noutputs; i++) {
            Arg out = outputs[i];
            ArgInfo& info = args[out.idx];
            const Mat& m = outMats[i];
            info.type = m.type();
            info.shape = m.shape();
            tensors.at(out.idx) = m;
            if (info.kind == DNN_ARG_TEMP) {
                int bufidx = bufidxs.at(out.idx);
                Mat& buf = buffers.at(bufidx);
                
                if (!dynamicOutShapes) {
                    // a sanity check: make sure that the data was not reallocated during Layer::forward()
                    // if the layer claims it does not produce dynamic-shape outputs.
                    CV_Assert(buf.data == m.data);
                } else if (m.u->size > buf.u->size) {
                    buf = m;
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

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
