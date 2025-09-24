// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#if 0
using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;
typedef std::pair<int, Arg> int_arg_pair;

struct ConstArgs
{
    ConstArgs(Net* net_) : net(net_), netimpl(net_->getImpl()) {}

    void process()
    {
        size_t nargs = netimpl->args.size();
        netimpl->tensors.resize(nargs);
        netimpl->useCounts(usecounts);
        C0 = 8; // [TODO] netimpl->backends(0)
        processGraph(netimpl->mainGraph);
    }

    void unuse(Arg inp)
    {
        CV_Assert(usecounts[inp.idx] > 0);
        if (--usecounts[inp.idx] == 0 && net->isConstArg(inp)) {
            netimpl->tensors[inp.idx] = Mat(); // deallocate unused tensor
        }
    }

    bool processGraph(Ptr<Graph>& graph)
    {
        bool modified = false;
        const std::vector<Ptr<Layer> >& prog = graph->prog();
        size_t i, nops = prog.size();
        std::vector<Node> newprog;
        std::vector<Buffer> temp;
        std::vector<Arg> removed_args;
        std::vector<Tensor> t_inputs;

        for (i = 0; i < nops; i++) {
            const Node& node = prog[i];
            std::vector<Graph>& subgraphs = const_cast<std::vector<Graph>&>(node->subgraphs());
            for (Graph& g: subgraphs) {
                if (processGraph(g))
                    modified = true;
            }
            const std::vector<Arg>& inputs = node->inputs();
            const std::vector<Arg>& outputs = node->outputs();
            const Op& op = node->op();
            size_t j, ninputs = inputs.size(), noutputs = outputs.size();
            bool tail_const = true, unuse_tail = false;
            t_inputs.assign(ninputs, Tensor());
            for (j = 1; j < ninputs; j++) {
                Arg inp = inputs[j];
                bool const_arg = net->isConstArg(inp);
                if (!const_arg)
                    tail_const = false;
                if (tail_const)
                    t_inputs[j] = netimpl->tensors.at(inp.idx);
            }

            ConvOp* conv_op = getOp<ConvOp>(&node);
            BatchNormOp* bn_op = getOp<BatchNormOp>(&node);
            ElemwiseOp* elemwise_op = getOp<ElemwiseOp>(&node);

            if (tail_const) {
                if (conv_op) {
                    // convolution with constant weights and bias
                    conv_op->setWeights(t_inputs[1], ninputs > 2 ? t_inputs[2] : Tensor(), C0);
                    modified = unuse_tail = true;
                } else if (bn_op) {
                    // batch norm with constant parameters
                    bn_op->computeScaleBias(t_inputs[1], t_inputs[2], t_inputs[3], t_inputs[4]);
                    modified = unuse_tail = true;
                } else if (elemwise_op && elemwise_op->opcode == ELWISE_CLIP && ninputs == 3) {
                    elemwise_op->setParams({t_inputs[1], t_inputs[2]});
                    modified = unuse_tail = true;
                }
            }

            if (unuse_tail) {
                //printf("simplified %s: %s\n", op->name().data(), node->name().data());
                std::vector<Arg> newinputs = {inputs[0]};
                Node newnode = NodeData::create(node->name(), op, newinputs, outputs);
                newprog.push_back(newnode);
            } else {
                newprog.push_back(node);
            }

            if (unuse_tail) {
                for (size_t i = 1; i < ninputs; i++)
                    unuse(inputs[i]);
            }
        }

        if (modified) {
            graph->setProg(newprog);
        }

        return modified;
    }

    Net* net;
    Net::Impl* netimpl;
    std::vector<int> usecounts;
    int64_t C0;
};

void Net::Impl::constArgs()
{
    ConstArgs constargs(net);
    constargs.process();
}
#endif

CV__DNN_INLINE_NS_END
}}

