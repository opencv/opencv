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

struct ModelFusionBasic
{
    ModelFusionBasic(Net::Impl* netimpl_) : netimpl(netimpl_) {}

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
    getLayer(std::vector<Ptr<Layer> >& newprog, int op_idx) const
    {
        return op_idx >= 0 ? dynamic_cast<_LayerType*>(newprog.at(op_idx).get()) : 0;
    }

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
            fused_inputs.clear(); // leave it empty in the merge patterns below to re-use original fused node inputs as-is.

            for(;;) {
                BatchNorm2Layer* bn = dynamic_cast<BatchNorm2Layer*>(layer_ptr);
                ActivationLayer* activ = dynamic_cast<ActivationLayer*>(layer_ptr);
                NaryEltwiseLayer* elemwise = dynamic_cast<NaryEltwiseLayer*>(layer_ptr);

                // merge convolution and batch norm
                if (bn && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg bn_inp = inputs[0];
                    int conv_layer_idx = producer_of.at(bn_inp.idx);
                    Conv2Layer* conv = getLayer<Conv2Layer>(newprog, conv_layer_idx);
                    if (conv) {
                        bool ok = conv->fuseBatchNorm(layer);
                        if (ok) {
                            fused_layer_idx = conv_layer_idx;
                            removed_args.push_back(bn_inp);
                            break;
                        }
                    }
                }

                // merge residual 'add' into 'conv' node
                if (elemwise && (elemwise->op == NaryEltwiseLayer::OPERATION::ADD ||
                    elemwise->op == NaryEltwiseLayer::OPERATION::SUM) &&
                    ninputs == 2) {

                    int op0 = producer_of.at(inputs[0].idx);
                    int op1 = producer_of.at(inputs[1].idx);

                    if (op0 >= 0 && op1 >= 0) {
                        int conv_layer_idx;
                        Arg residual, conv_out;

                        if (op0 > op1) { // choose the latter op to ensure that the other component is already computed
                            conv_layer_idx = op0;
                            conv_out = inputs[0];
                            residual = inputs[1];
                        } else {
                            conv_layer_idx = op1;
                            conv_out = inputs[1];
                            residual = inputs[0];
                        }

                        Conv2Layer* conv = getLayer<Conv2Layer>(newprog, conv_layer_idx);
                        if (conv && usecounts[conv_out.idx] == 1 &&
                            conv->fuseAddResidual(residual)) {
                            fused_layer_idx = conv_layer_idx;
                            removed_args.push_back(conv_out);
                            break;
                        }
                    }
                }

                // merge convolution and activation
                if (activ && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg activ_inp = inputs[0];
                    int conv_layer_idx = producer_of.at(activ_inp.idx);
                    Conv2Layer* conv = getLayer<Conv2Layer>(newprog, conv_layer_idx);
                    if (conv) {
                        bool ok = conv->fuseActivation(layer);
                        if (ok) {
                            fused_layer_idx = conv_layer_idx;
                            removed_args.push_back(activ_inp);
                            break;
                        }
                    }
                }

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
                                                        Ptr<Layer> gnlayer = GroupNormLayer::create(gnparams);
                                                        gnlayer->netimpl = netimpl;
                                                        gnlayer->inputs = {orig_inp, mul_scale_arg, add_bias_arg};
                                                        newprog[instnorm_idx] = gnlayer;
                                                    }
                                                    fused_layer_idx = instnorm_idx;
                                                    removed_args.push_back(instnorm_inp);
                                                    removed_args.push_back(reshape2_inp);
                                                    removed_args.push_back(reshape2_out);
                                                    removed_args.push_back(mul_out);
                                                    newprog[reshape1_idx] = Ptr<Layer>();
                                                    newprog[reshape2_idx] = Ptr<Layer>();
                                                    newprog[mul_idx] = Ptr<Layer>();
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }

            if (fused_layer_idx >= 0) {
                modified = true;
                Layer* fused_layer = newprog[fused_layer_idx];
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

void Net::Impl::fuseBasic()
{
    ModelFusionBasic basicFusion(this);
    basicFusion.fuse();
}

CV__DNN_INLINE_NS_END
}}
