// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;
typedef std::pair<int, Arg> int_arg_pair;

struct ConstFolding
{
    Net::Impl* netimpl;
    std::vector<int> usecounts;

    ConstFolding(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void process()
    {
        size_t nargs = netimpl->args.size();
        netimpl->tensors.resize(nargs);
        netimpl->useCounts(usecounts);
        netimpl->scratchBufs.clear();
        processGraph(netimpl->mainGraph);
        netimpl->scratchBufs.clear();
    }

    Layer* getLayer(std::vector<Ptr<Layer> >& newprog, int op_idx) const
    {
        return op_idx >= 0 ? newprog.at(op_idx).get() : 0;
    }

    void unuse(Arg inp)
    {
        CV_Assert(usecounts[inp.idx] > 0);
        if (--usecounts[inp.idx] == 0 && netimpl->isConstArg(inp)) {
            netimpl->tensors[inp.idx] = Mat(); // deallocate unused tensor
        }
    }

    bool processGraph(Ptr<Graph>& graph)
    {
        bool modified = false;
        const std::vector<Ptr<Layer> >& prog = graph->prog();
        size_t i, nops = prog.size();
        std::vector<Ptr<Layer> > newprog;
        std::vector<Arg> removed_args;
        std::vector<Mat> inpMats, tempMats;
        std::vector<int> inpTypes, outTypes, tempTypes;
        std::vector<MatShape> inpShapes, outShapes, tempShapes;

        for (i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = prog[i];
            std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g: *subgraphs) {
                    if (processGraph(g))
                        modified = true;
                }
            }
            const std::vector<Arg>& inputs = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;
            size_t j, ninputs = inputs.size(), noutputs = outputs.size();
            bool all_const = true;
            inpMats.assign(ninputs, Mat());
            inpTypes.resize(ninputs);
            inpShapes.resize(ninputs);
            for (j = 0; j < ninputs; j++) {
                Arg inp = inputs[j];
                bool const_arg = netimpl->isConstArg(inp);
                if (!const_arg)
                    all_const = false;
                if (all_const) {
                    const Mat& m = netimpl->tensors.at(inp.idx);
                    inpMats[j] = m;
                    inpTypes[j] = m.type();
                    inpShapes[j] = m.shape();
                }
            }

            if (all_const /*&&
                op->supportBlockLayout(0, (int)ninputs) <= 0 // we don't currently support constant folding
                                               // for block-layout operations (Convolution, MaxPool, AveragePool)
                */) {
                // Use a fresh vector of Mat's for outputs since we want to make these outputs the new constant tensors.
                // So, they must be unique and don't interfere with other tensors.
                std::vector<Mat> outMats(noutputs);
                if (!layer->dynamicOutputShapes())
                    netimpl->allocateLayerOutputs(layer, inpTypes, inpShapes, outTypes,
                                                  outShapes, outMats, tempTypes, tempShapes, tempMats,
                                                  netimpl->scratchBufs, false);
                layer->finalize(inpMats, outMats);
                layer->forward(inpMats, outMats, tempMats);
                CV_Assert(outMats.size() == noutputs);
                for (j = 0; j < noutputs; j++) {
                    Arg out = outputs[j];
                    ArgData& out_data = netimpl->args.at(out.idx);
                    const Mat& m = outMats[j];
                    out_data.type = m.type();
                    out_data.shape = m.shape();
                    out_data.kind = DNN_ARG_CONST; // re-classify each output as constant
                    netimpl->tensors.at(out.idx) = m;
                }

                modified = true;
                for (size_t i = 0; i < ninputs; i++)
                    unuse(inputs[i]);
                //printf("folded %s: %s\n", op->name().data(), node->name().data());
                // we don't add operation into the new program,
                // because the output of the all-const inputs operation is now a constant,
                // stored in a separate tensor
            } else {
                newprog.push_back(layer);
            }
        }

        if (modified) {
            graph->setProg(newprog);
        }

        return modified;
    }
};

void Net::Impl::constFold()
{
    ConstFolding constfolder(this);
    constfolder.process();
}

CV__DNN_INLINE_NS_END
}}
