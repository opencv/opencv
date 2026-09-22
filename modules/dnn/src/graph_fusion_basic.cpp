// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "adjacency_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;
typedef std::pair<int, Arg> int_arg_pair;

//! Reshape+InstanceNorm(1,0)+Reshape+Mul+Add collapsed into one layer. It matches
//! backward from the Add and can swap the survivor's type, so it is not a chain.
struct InstanceNormAffineFusion
{
    InstanceNormAffineFusion(Net::Impl* netimpl_) : netimpl(netimpl_) {}

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

    template<typename _LayerType> _LayerType*
    getLayer(std::vector<Ptr<LayerInfo> >& newprog, int op_idx) const
    {
        return op_idx >= 0 ? dynamic_cast<_LayerType*>(newprog.at(op_idx).get()) : 0;
    }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        vector<Arg> removed_args;
        bool modified = false;
        const std::vector<Ptr<LayerInfo> >& prog = graph->prog();
        size_t i, nargs = netimpl->args.size(), nops = prog.size();
        std::vector<int> producer_of(nargs, -1);
        std::vector<Ptr<LayerInfo> > newprog;

        for (i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            Layer* layer_ptr = (Layer*)layer.get();
            int fused_layer_idx = -1;
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
            removed_args.clear();

            NaryEltwiseLayer* elemwise = dynamic_cast<NaryEltwiseLayer*>(layer_ptr);

            // fuse Reshape + InstanceNorm(scale=ones,bias=zeros) + Reshape + Mul + Add
            if (elemwise && elemwise->op == NaryEltwiseLayer::OPERATION::ADD &&
                ninputs == 2) {
                int mul_input_idx = -1;
                Arg add_bias_arg;
                for (int k = 0; k < 2; k++) {
                    int pidx = producer_of.at(inputs[k].idx);
                    NaryEltwiseLayer* mul = getLayer<NaryEltwiseLayer>(newprog, pidx);
                    if (mul && mul->op == NaryEltwiseLayer::OPERATION::PROD) {
                        mul_input_idx = k;
                        add_bias_arg = inputs[1 - k];
                        break;
                    }
                }
                if (mul_input_idx >= 0 && netimpl->isConstArg(add_bias_arg)) {
                    Arg mul_out = inputs[mul_input_idx];
                    int mul_idx = producer_of.at(mul_out.idx);
                    NaryEltwiseLayer* mul = getLayer<NaryEltwiseLayer>(newprog, mul_idx);
                    if (mul && mul->inputs.size() == 2 &&
                        usecounts.at(mul_out.idx) == 1) {
                        int reshape2_input_idx = -1;
                        Arg mul_scale_arg;
                        for (int k = 0; k < 2; k++) {
                            int pidx = producer_of.at(mul->inputs[k].idx);
                            if (getLayer<Reshape2Layer>(newprog, pidx)) {
                                reshape2_input_idx = k;
                                mul_scale_arg = mul->inputs[1 - k];
                                break;
                            }
                        }
                        if (reshape2_input_idx >= 0 && netimpl->isConstArg(mul_scale_arg)) {
                            Arg reshape2_out = mul->inputs[reshape2_input_idx];
                            int reshape2_idx = producer_of.at(reshape2_out.idx);
                            Reshape2Layer* reshape2_lyr = getLayer<Reshape2Layer>(newprog, reshape2_idx);
                            if (reshape2_lyr && reshape2_lyr->inputs.size() >= 1 &&
                                usecounts.at(reshape2_out.idx) == 1) {
                                Arg reshape2_inp = reshape2_lyr->inputs[0];
                                int instnorm_idx = producer_of.at(reshape2_inp.idx);
                                InstanceNormLayer* instnorm = getLayer<InstanceNormLayer>(newprog, instnorm_idx);
                                if (instnorm && instnorm->inputs.size() == 3 &&
                                    usecounts.at(reshape2_inp.idx) == 1) {
                                    Arg instnorm_inp = instnorm->inputs[0];
                                    int reshape1_idx = producer_of.at(instnorm_inp.idx);
                                    Reshape2Layer* reshape1_lyr = getLayer<Reshape2Layer>(newprog, reshape1_idx);
                                    if (reshape1_lyr && reshape1_lyr->outputs.size() == 1 &&
                                        usecounts.at(instnorm_inp.idx) == 1) {
                                        Mat in_scale = netimpl->isConstArg(instnorm->inputs[1]) ?
                                                       netimpl->argTensor(instnorm->inputs[1]) : Mat();
                                        Mat in_bias = netimpl->isConstArg(instnorm->inputs[2]) ?
                                                      netimpl->argTensor(instnorm->inputs[2]) : Mat();
                                        bool valid = !in_scale.empty() && !in_bias.empty() &&
                                                     in_scale.type() == CV_32F && in_bias.type() == CV_32F;
                                        if (valid) {
                                            const float* sp = in_scale.ptr<float>();
                                            const float* bp = in_bias.ptr<float>();
                                            bool all_ones = true, all_zeros = true;
                                            for (size_t k = 0; k < in_scale.total() && all_ones; k++)
                                                all_ones = (std::abs(sp[k] - 1.f) < 1e-6f);
                                            for (size_t k = 0; k < in_bias.total() && all_zeros; k++)
                                                all_zeros = (std::abs(bp[k]) < 1e-6f);
                                            if (all_ones && all_zeros) {
                                                Arg orig_inp = reshape1_lyr->inputs[0];
                                                Mat mul_scale_mat = netimpl->argTensor(mul_scale_arg);
                                                if (in_scale.total() == mul_scale_mat.total()) {
                                                    // Channel dim preserved — fuse into InstanceNorm
                                                    instnorm->inputs[0] = orig_inp;
                                                    instnorm->inputs[1] = mul_scale_arg;
                                                    instnorm->inputs[2] = add_bias_arg;
                                                } else {
                                                    // Channel dim changed (e.g. [1,C,H,W]->[1,1,C*H*W]):
                                                    // original is global norm + per-channel affine.
                                                    // Replace with GroupNorm(num_groups=reshaped_C).
                                                    int num_groups = (int)in_scale.total();
                                                    LayerParams gnparams;
                                                    gnparams.name = instnorm->name;
                                                    gnparams.type = "GroupNormalization";
                                                    gnparams.set("epsilon", instnorm->epsilon);
                                                    gnparams.set("num_groups", num_groups);
                                                    Ptr<LayerInfo> gnlayer = GroupNormLayer::create(gnparams);
                                                    gnlayer->netimpl = netimpl;
                                                    gnlayer->inputs = {orig_inp, mul_scale_arg, add_bias_arg};
                                                    newprog[instnorm_idx] = gnlayer;
                                                }
                                                fused_layer_idx = instnorm_idx;
                                                removed_args.push_back(instnorm_inp);
                                                removed_args.push_back(reshape2_inp);
                                                removed_args.push_back(reshape2_out);
                                                removed_args.push_back(mul_out);
                                                newprog[reshape1_idx] = Ptr<LayerInfo>();
                                                newprog[reshape2_idx] = Ptr<LayerInfo>();
                                                newprog[mul_idx] = Ptr<LayerInfo>();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (fused_layer_idx >= 0) {
                modified = true;
                Layer* fused_layer = (Layer*)newprog[fused_layer_idx].get();
                fused_layer->outputs = outputs;
                for (Arg new_out: outputs)
                    producer_of[new_out.idx] = fused_layer_idx;
                for (Arg old_out: removed_args) {
                    usecounts.at(old_out.idx) = 0;
                    producer_of.at(old_out.idx) = -1;
                }
            } else {
                for (auto out: outputs)
                    producer_of[out.idx] = (int)newprog.size();
                newprog.push_back(layer);
            }
        }

        if (modified) {
            size_t i, j = 0, newops = newprog.size();
            for (i = 0; i < newops; i++) {
                if (!newprog[i].empty()) {
                    if (j < i)
                        newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            //printf("fused some ops in graph %s. size before: %zu ops, size after: %zu ops\n",
            //       graph->name().data(), nops, j);
            graph->setProg(newprog);
        }

        return modified;
    }

    Net::Impl* netimpl;
    vector<int> usecounts;
};

void Net::Impl::fuseInstanceNormAffine()
{
    InstanceNormAffineFusion fusion(this);
    fusion.fuse();
}

// fold BN scale/bias into the weights of the immediately following Conv2 (pre-constArgs only)
struct FuseBNPass
{
    FuseBNPass(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void run()
    {
        netimpl->useCounts(usecounts);
        fuseGraph(netimpl->mainGraph);
    }

    void fuseGraph(Ptr<Graph>& graph)
    {
        const std::vector<Ptr<LayerInfo> >& prog = graph->prog();
        size_t nops = prog.size(), nargs = netimpl->args.size();
        std::vector<Ptr<LayerInfo> > newprog;
        newprog.reserve(nops);
        std::vector<int> producer_of((int)nargs, -1);
        bool modified = false;

        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            Layer* layer_ptr = (Layer*)layer.get();

            std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs)
                for (Ptr<Graph>& g : *subgraphs) fuseGraph(g);

            const std::vector<Arg>& inputs  = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;

            Conv2Layer* conv = dynamic_cast<Conv2Layer*>(layer_ptr);
            if (conv && !inputs.empty()) {
                Arg conv_inp0 = inputs[0];
                int bn_idx = conv_inp0.idx >= 0 && conv_inp0.idx < (int)producer_of.size()
                             ? producer_of[conv_inp0.idx] : -1;
                if (bn_idx >= 0 && usecounts[conv_inp0.idx] == 1) {
                    BatchNorm2Layer* bn = dynamic_cast<BatchNorm2Layer*>(newprog[bn_idx].get());
                    if (bn && fuseForward(conv, bn)) {
                        Arg bn_inp0 = bn->inputs[0];
                        layer_ptr->inputs[0] = bn_inp0;
                        usecounts[conv_inp0.idx] = 0;
                        if (bn_inp0.idx >= 0)
                            usecounts[bn_inp0.idx]++;
                        newprog[bn_idx] = Ptr<LayerInfo>();
                        modified = true;
                    }
                }
            }

            for (Arg out : outputs)
                if (out.idx >= 0 && out.idx < (int)producer_of.size())
                    producer_of[out.idx] = (int)newprog.size();
            newprog.push_back(layer);
        }

        if (modified) {
            size_t j = 0;
            for (size_t i = 0; i < newprog.size(); i++) {
                if (!newprog[i].empty()) {
                    if (j < i) newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            graph->setProg(newprog);
        }
    }

    // Ask the conv to take the BatchNorm's scale and shift into its own weights. It owns
    // the layout, so this works after constArgs() has packed them.
    bool fuseForward(Conv2Layer* conv, BatchNorm2Layer* bn)
    {
        if (bn->inputs.size() != 1)
            return false;   // constArgs() freezes a const-parameter BN down to one input
        Mat scale, shift;
        bn->getScaleBias(scale, shift);
        if (scale.empty() || shift.empty())
            return false;
        const FusionOps* ops = fusionOpsFor(conv);
        return ops && ops->foldInputScale && ops->foldInputScale(conv, scale, shift);
    }

    Net::Impl* netimpl;
    std::vector<int> usecounts;
};

void Net::Impl::fuseBN()
{
    FuseBNPass pass(this);
    pass.run();
}

CV__DNN_INLINE_NS_END
}}
