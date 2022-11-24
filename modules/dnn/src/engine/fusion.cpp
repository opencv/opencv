// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;

struct ModelFusionBasic
{
    ModelFusionBasic(Net2::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse()
    {
        int i, niter = 10;
        netimpl->useCounts(usecounts);
        for (i = 0; i < niter; i++) {
            bool fused_any = fuseGraph(netimpl->graph);
            if (!fused_any)
                break;
        }
    }

    bool isConstScalarTensor(int argidx, float* compare_with) const
    {
        const Tensor* t;
        void* data;
        if (!netimpl->isConst(argidx))
            return false;
        t = &netimpl->tensors.at(argidx);
        data = t->data();
        if (t->shape.total() != 1 || t->typ != CV_32F)
            return false;
        if (compare_with && *(float*)data != *compare_with)
            return false;
        return true;
    }

    Node* getOp(int op_idx) const
    {
        return op_idx >= 0 ? (Node*)&newprog.at(op_idx) : 0;
    }

    template<typename _LayerType> bool isOp(const Node* n) const
    { return n && dynamic_cast<_LayerType*>(n->op.get()) != 0; }

    template<typename _LayerType>
    int_pair isUnary(int t_inp, bool check_used_once) const
    {
        int op_idx;
        const Node* node;
        if (t_inp < 0) return std::make_pair(-1, -1);
        op_idx = producer_of.at(t_inp);
        node = getOp(op_idx);
        if (!node || node->inputs.size() != 1 ||
            !isOp<_LayerType>(node) ||
            (check_used_once && usecounts.at(node->inputs.at(0)) != 1))
            return std::make_pair(-1, -1);
        return std::make_pair(op_idx, node->inputs.at(0));
    }

    bool fuseGraph(Graph& graph)
    {
        vector<int> t_out_removed;
        bool modified = false;
        size_t i, nargs = netimpl->args.size(), nops = graph.prog.size();
        producer_of.assign(nargs, -1);
        newprog.clear();

        for (i = 0; i < nops; i++) {
            Node& node = graph.prog[i];
            int ninputs = (int)node.inputs.size();
            PLayer fused_op;
            int fused_op_idx = -1, t_out_new = -1;
            t_out_removed.clear();

            for(;;){
                // merge convolution batch norm
                if (isOp<BatchNormLayer>(&node) && ninputs == 1 &&
                    usecounts.at(node.inputs[0]) == 1) {
                    int t_bn_inp = node.inputs[0];
                    int conv_op_idx = producer_of.at(t_bn_inp);
                    Node* conv_op = getOp(conv_op_idx);
                    if (isOp<ConvolutionLayer>(conv_op) && conv_op->inputs.size() == 1) {
                        bool ok = conv_op->op->tryFuse(node.op);
                        if (ok) {
                            fused_op_idx = conv_op_idx;
                            fused_op = conv_op->op;
                            t_out_new = node.outputs[0];
                            t_out_removed.push_back(t_bn_inp);
                            break;
                        }
                    }
                }

                // merge convolution and activation
                if (isOp<ActivationLayer>(&node) && ninputs == 1 &&
                    usecounts.at(node.inputs[0]) == 1) {
                    int t_activ_inp = node.inputs[0];
                    int conv_op_idx = producer_of.at(t_activ_inp);
                    Node* conv_op = getOp(conv_op_idx);
                    if (isOp<ConvolutionLayer>(conv_op)) {
                        auto activ = node.op.dynamicCast<ActivationLayer>();
                        if (!activ.empty()) {
                            conv_op->op->setActivation(activ);
                            fused_op_idx = conv_op_idx;
                            fused_op = conv_op->op;
                            t_out_new = node.outputs[0];
                            t_out_removed.push_back(t_activ_inp);
                            break;
                        }
                    }
                }
                break;
            }
            if (fused_op_idx >= 0) {
                modified = true;
                Node& fused_node = newprog.at(fused_op_idx);
                fused_node.op = fused_op;
                fused_node.outputs[0] = t_out_new;
                producer_of[t_out_new] = fused_op_idx;
                for (auto t_out_old: t_out_removed) {
                    usecounts.at(t_out_old) = 0;
                    producer_of.at(t_out_old) = -1;
                }
            } else {
                for (auto outidx: node.outputs)
                    producer_of[outidx] = (int)newprog.size();
                newprog.push_back(node);
            }
        }

        if (modified) {
            size_t i, j = 0, nops = newprog.size();
            for (i = 0; i < nops; i++) {
                if (!newprog[i].op.empty()) {
                    if (j < i)
                        newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            printf("fused some ops in graph %s. size before: %d ops, size after: %d ops",
                   graph.name.c_str(), (int)nops, (int)j);
            graph.prog = newprog;
        }
    }

    Net2::Impl* netimpl;
    vector<int> usecounts;
    vector<int> producer_of;
    vector<Node> newprog;
};

void Net2::Impl::fuse()
{
    ModelFusionBasic basicFusion(this);
    basicFusion.fuse();
}

CV__DNN_INLINE_NS_END
}}
