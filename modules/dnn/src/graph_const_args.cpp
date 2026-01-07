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

struct ConstArgs
{
    ConstArgs(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void process()
    {
        size_t nargs = netimpl->args.size();
        netimpl->__tensors__.resize(nargs);
        netimpl->useCounts(usecounts);
        processGraph(netimpl->mainGraph);
    }

    void unuse(Arg inp)
    {
        CV_Assert(usecounts[inp.idx] > 0);
        if (--usecounts[inp.idx] == 0 && netimpl->isConstArg(inp)) {
            netimpl->__tensors__[inp.idx] = Mat(); // deallocate unused tensor
        }
    }

    void processGraph(Ptr<Graph>& graph)
    {
        const std::vector<Ptr<Layer> >& prog = graph->prog();
        size_t i, nops = prog.size();
        std::vector<Arg> removed_args;
        std::vector<Arg> saved_tail_inputs;

        for (i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = prog[i];
            Layer* layer_ptr = const_cast<Layer*>(layer.get());
            std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g: *subgraphs) {
                    processGraph(g);
                }
            }
            const std::vector<Arg>& inputs = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;
            size_t j, ninputs = inputs.size();
            if (ninputs == 1) {
                continue;
            }
            bool tail_const = true, unuse_tail = false;
            saved_tail_inputs.clear();
            for (j = 1; j < ninputs; j++) {
                Arg inp = inputs[j];
                bool const_arg = netimpl->isConstArg(inp);
                if (!const_arg)
                    tail_const = false;
                saved_tail_inputs.push_back(inp);
            }

            Conv2Layer* conv = dynamic_cast<Conv2Layer*>(layer_ptr);
            BatchNorm2Layer* bn = dynamic_cast<BatchNorm2Layer*>(layer_ptr);
            //ActivationLayer* activ = dynamic_cast<ActivationLayer*>(layer_ptr);

            if (tail_const) {
                if (conv) {
                    // convolution with constant weights and bias
                    conv->setWeights(netimpl->__tensors__[inputs[1]],
                                     ninputs > 2 ? netimpl->__tensors__[inputs[2]] : Mat(),
                                     netimpl->defaultC0, netimpl->accuracy);
                    conv->inputs.resize(1);
                    unuse_tail = true;
                } else if (bn && bn->freezeScaleBias()) {
                    // batch norm with constant parameters
                    unuse_tail = true;
                }/* else if (activ && dynamic_cast<ReLU6Layer>(activ)) {
                    // [TODO] ...
                    unuse_tail = true;
                }*/
            }

            if (unuse_tail) {
                for (Arg inp: saved_tail_inputs)
                    unuse(inp);
            }
        }
    }

    Net::Impl* netimpl;
    std::vector<int> usecounts;
};

void Net::Impl::constArgs()
{
    ConstArgs constargs(this);
    constargs.process();
}

CV__DNN_INLINE_NS_END
}}

