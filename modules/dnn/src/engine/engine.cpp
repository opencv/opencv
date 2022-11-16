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
}

Net2::~Net2() {}

Ptr<Net2::Impl> Net2::impl() const { return impl_; }

bool Net2::empty() const { return impl_->empty(); }

Net2::Impl::Impl()
{
    modelFormat = DNN_MODEL_GENERIC;
    defaultLayout = DNN_LAYOUT_NCHW;
    enableFP16 = true;
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

void Net2::Impl::updateUseCounts(std::vector<int>& usecounts, const Graph& graph)
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

void Net2::Impl::useCounts(vector<int>& usecounts)
{
    int i, nargs = (int)args.size();
    usecounts.assign(nargs, 0);
    updateUseCounts(usecounts, graph);
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
