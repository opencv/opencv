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

struct ModelFusionBasic
{
    ModelFusionBasic(Net* net_) : net(net_), netimpl(net_->getImpl()) {}

    void fuse()
    {
        int i, niter = 10;
        netimpl->useCounts(usecounts);
        for (i = 0; i < niter; i++) {
            bool fused_any = fuseGraph(netimpl->mainGraph);
            if (!fused_any)
                break;
        }
    }

    bool isConstScalarTensor(Arg arg, float* compare_with) const
    {
        const Mat* m;
        void* data;
        if (!net->isConstArg(arg))
            return false;
        m = &netimpl->tensors.at(arg.idx);
        data = m->data;
        if (m->total() != 1 || m->type() != CV_32F)
            return false;
        if (compare_with && *(float*)data != *compare_with)
            return false;
        return true;
    }

    Layer* getLayer(std::vector<Ptr<Layer> >& newprog, int op_idx) const
    {
        return op_idx >= 0 ? newprog.at(op_idx).get() : 0;
    }

    /*template<typename _OpType>
    int_arg_pair isUnary(std::vector<Node>& newprog, const std::vector<int>& producer_of,
                         int t_inp, bool check_used_once) const
    {
        int op_idx;
        const Node* node;
        if (t_inp < 0) return std::make_pair(-1, Arg());
        op_idx = producer_of.at(t_inp);
        node = getNode(newprog, op_idx);
        if (!node || (*node)->ninputs() != 1 || !getOp<_OpType>(node) ||
            (check_used_once && usecounts.at((*node)->inputs(0).idx) != 1))
            return std::make_pair(-1, Arg());
        return std::make_pair(op_idx, (*node)->inputs(0));
    }*/

    bool fuseGraph(Ptr<Graph>& graph)
    {
        vector<Arg> removed_args;
        bool modified = false;
        const std::vector<Ptr<Layer> >& prog = graph->prog();
        size_t i, nargs = netimpl->args.size(), nops = prog.size();
        std::vector<int> producer_of(nargs, -1);
        std::vector<Ptr<Layer> > newprog;
        std::vector<Arg> fused_inputs;

        for (i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = prog[i];
            Layer* layer_ptr = (Layer*)layer.get();
            std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g: *subgraphs) {
                    if (fuseGraph(g))
                        modified = true;
                }
            }
            const std::vector<Arg>& inputs = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;
            size_t ninputs = inputs.size();
            Ptr<Layer> fused_layer;
            int fused_node_idx = -1;
            removed_args.clear();
            fused_inputs.clear(); // leave it empty in the merge patterns below to re-use original fused node inputs as-is.

            for(;;) {
                BatchNormLayer* bn = dynamic_cast<BatchNormLayer*>(layer_ptr);
                EltW  emwiseOp* elemwise = getOp<ElemwiseOp>(&node);

                // merge convolution and batch norm
                if (bn && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg bn_inp = inputs[0];
                    int conv_node_idx = producer_of.at(bn_inp.idx);
                    Node* conv_node = getNode(newprog, conv_node_idx);
                    ConvOp* conv_op = getOp<ConvOp>(conv_node);
                    if (conv_op && (*conv_node)->ninputs() == 1) {
                        bool ok = conv_op->fuseBatchNorm(node->op());
                        if (ok) {
                            fused_node_idx = conv_node_idx;
                            fused_op = (*conv_node)->op();
                            removed_args.push_back(bn_inp);
                            break;
                        }
                    }
                }

                // merge residual 'add' into 'conv' node
                if (elemwise && elemwise->opcode == ELWISE_ADD && ninputs == 2) {
                    ArgData& adata0 = netimpl->args[inputs[0].idx];
                    ArgData& adata1 = netimpl->args[inputs[1].idx];

                    if (adata0.type == adata1.type && adata0.shape == adata1.shape) {
                        int op0 = producer_of.at(inputs[0].idx);
                        int op1 = producer_of.at(inputs[1].idx);
                        int conv_node_idx;
                        Arg residual, conv_out;

                        if (op0 > op1) { // choose the latter op to ensure that the other component is already computed
                            conv_node_idx = op0;
                            conv_out = inputs[0];
                            residual = inputs[1];
                        } else {
                            conv_node_idx = op1;
                            conv_out = inputs[1];
                            residual = inputs[0];
                        }

                        Node* conv_node = getNode(newprog, conv_node_idx);
                        ConvOp* conv_op = getOp<ConvOp>(conv_node);
                        if (conv_op && !conv_op->activ && !conv_op->add_residual && usecounts[conv_out.idx] == 1) {
                            conv_op->add_residual = true;
                            fused_node_idx = conv_node_idx;
                            fused_op = (*conv_node)->op();
                            const std::vector<Arg>& conv_inputs = (*conv_node)->inputs();
                            fused_inputs.assign(conv_inputs.begin(), conv_inputs.end());
                            fused_inputs.push_back(residual);
                            removed_args.push_back(conv_out);
                            break;
                        }
                    }
                }

                // merge convolution and activation
                if (elemwise && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg activ_inp = inputs[0];
                    int conv_node_idx = producer_of.at(activ_inp.idx);
                    Node* conv_node = getNode(newprog, conv_node_idx);
                    ConvOp* conv_op = getOp<ConvOp>(conv_node);
                    if (conv_op) {
                        bool ok = conv_op->fuseActivation(node->op());
                        if (ok) {
                            fused_node_idx = conv_node_idx;
                            fused_op = (*conv_node)->op();
                            removed_args.push_back(activ_inp);
                            break;
                        }
                    }
                }
                break;
            }

            if (fused_node_idx >= 0) {
                modified = true;
                const Node& orig_node = newprog.at(fused_node_idx);
                if (fused_inputs.empty()) {
                    const std::vector<Arg>& orig_inputs = orig_node->inputs();
                    fused_inputs.assign(orig_inputs.begin(), orig_inputs.end());
                }
                Node fused_node = NodeData::create(orig_node->name(), fused_op,
                                                   fused_inputs, outputs,
                                                   orig_node->subgraphs());
                newprog.at(fused_node_idx) = fused_node;
                for (Arg new_out: outputs)
                    producer_of[new_out.idx] = fused_node_idx;
                for (Arg old_out: removed_args) {
                    usecounts.at(old_out.idx) = 0;
                    producer_of.at(old_out.idx) = -1;
                }
            } else {
                for (auto out: outputs)
                    producer_of[out.idx] = (int)newprog.size();
                newprog.push_back(node);
            }
        }

        if (modified) {
            size_t i, j = 0, nops = newprog.size();
            for (i = 0; i < nops; i++) {
                if (newprog[i]->op()) {
                    if (j < i)
                        newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            printf("fused some ops in graph %s. size before: %zu ops, size after: %zu ops\n",
                   graph->name().data(), nops, j);
            graph->setProg(newprog);
        }

        return modified;
    }

    Net* net;
    Net::Impl* netimpl;
    vector<int> usecounts;
};

void Net::Impl::fuse()
{
    ModelFusionBasic basicFusion(net);
    basicFusion.fuse();
}
#endif

CV__DNN_INLINE_NS_END
}}
