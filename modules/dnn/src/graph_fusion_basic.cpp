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

    // fold BN scale/bias into Conv2 weight and bias tensors (raw NCHW fp32 only)
    bool fuseForward(Conv2Layer* conv, BatchNorm2Layer* bn)
    {
        // skip padded convolutions: border taps get incorrect bias from folded BN
        for (int p : conv->pads)
            if (p > 0) return false;

        if (conv->inputs.size() < 2 || !netimpl->isConstArg(conv->inputs[1]))
            return false;

        Mat& Wref = netimpl->argTensor(conv->inputs[1]);
        if (Wref.empty() || Wref.type() != CV_32F || Wref.dims != 4)
            return false;

        int OC    = Wref.size[0];
        int IC_pg = Wref.size[1];   // input channels per group
        int KH    = Wref.size[2];
        int KW    = Wref.size[3];
        int ngroups = conv->ngroups;
        if (ngroups <= 0 || OC % ngroups != 0)
            return false;
        int OC_pg = OC / ngroups;

        // resolve BN effective scale and bias; validate all params are const
        Mat bn_scale, bn_bias_vec;
        size_t bn_nin = bn->inputs.size();
        if (bn_nin == 5) {
            for (int k = 1; k <= 4; k++)
                if (!netimpl->isConstArg(bn->inputs[k])) return false;
            BatchNorm2Layer::getScaleBias(
                netimpl->argTensor(bn->inputs[1]),
                netimpl->argTensor(bn->inputs[2]),
                netimpl->argTensor(bn->inputs[3]),
                netimpl->argTensor(bn->inputs[4]),
                bn->epsilon, bn_scale, bn_bias_vec);
        } else if (bn_nin == 1) {
            bn->getScaleBias(bn_scale, bn_bias_vec);
        } else {
            return false;
        }

        bn_scale.convertTo(bn_scale, CV_32F);
        bn_bias_vec.convertTo(bn_bias_vec, CV_32F);

        if ((int)bn_scale.total() != IC_pg * ngroups)
            return false;

        const float* bn_s = bn_scale.ptr<float>();
        const float* bn_b = bn_bias_vec.ptr<float>();

        // reshape to 2D [OC, IC_pg*KH*KW]; shares buffer with Wref
        Mat W2d = Wref.reshape(1, OC);

        // scale weights in-place; accumulate per-output-channel bias correction
        bias_adj.assign(OC, 0.f);
        for (int g = 0; g < ngroups; g++) {
            int ic_off = g * IC_pg;
            for (int oc = g * OC_pg, oc_end = oc + OC_pg; oc < oc_end; oc++) {
                float* wrow = W2d.ptr<float>(oc);
                for (int ic = 0; ic < IC_pg; ic++) {
                    float  s   = bn_s[ic_off + ic];
                    float  b   = bn_b[ic_off + ic];
                    float* wic = wrow + ic * KH * KW;
                    float  sw  = 0.f;
                    for (int k = 0; k < KH * KW; k++) {
                        sw += wic[k];
                        wic[k] *= s;
                    }
                    bias_adj[oc] += b * sw;
                }
            }
        }

        // apply bias correction into existing bias tensor or create a new one
        size_t conv_nin = conv->inputs.size();
        if (conv_nin >= 3 && netimpl->isConstArg(conv->inputs[2])) {
            Mat& Bref = netimpl->argTensor(conv->inputs[2]);
            if (Bref.type() != CV_32F || (int)Bref.total() != OC)
                return false;
            float* bp = Bref.ptr<float>();
            for (int oc = 0; oc < OC; oc++)
                bp[oc] += bias_adj[oc];
        } else {
            Mat newB(1, &OC, CV_32F);
            std::memcpy(newB.ptr<float>(), bias_adj.data(), OC * sizeof(float));
            Arg ba = netimpl->newConstArg(
                "__fused_bn_bias_w" + std::to_string(conv->inputs[1].idx), newB);
            if (conv_nin == 2) conv->inputs.push_back(ba);
            else               conv->inputs[2] = ba;
        }

        return true;
    }

    Net::Impl* netimpl;
    std::vector<int> usecounts;
    std::vector<float> bias_adj;
};

void Net::Impl::fuseBN()
{
    FuseBNPass pass(this);
    pass.run();
}

CV__DNN_INLINE_NS_END
}}
