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

Ptr<Net2::Impl> Net2::impl() { return impl_; }

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

void Net2::Impl::useCounts(std::vector<int>& usecounts)
{
    size_t nargs = args.size();
    usecounts.assign(nargs, 0);
}

void Net2::Impl::assignBuffers()
{

}

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

bool Graph::empty() const { return prog.empty(); }
void Graph::clear()
{
    inputs.clear();
    outputs.clear();
    prog.clear();
}

CV__DNN_INLINE_NS_END
}}
