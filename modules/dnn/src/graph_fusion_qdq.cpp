// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;
typedef std::pair<int, Arg> int_arg_pair;

struct ModelFusionQDQ
{
    ModelFusionQDQ(Net::Impl* netimpl_) : netimpl(netimpl_) {}

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

    LayerParams makeLayerParamsFromOriginal(const Layer* layer, const String& newType) const
    {
        LayerParams params;
        int lid = netimpl->getLayerId(layer->name);
        if (lid >= 0)
            params = netimpl->getLayerData(lid).params;
        params.name = layer->name;
        params.type = newType;
        return params;
    }

    Ptr<Layer> createFusedLayer(const LayerParams& src) const
    {
        LayerParams params = src;
        Ptr<Layer> layer = LayerFactory::createLayerInstance(params.type, params);
        if (!layer.empty())
            layer->netimpl = netimpl;
        return layer;
    }

    static bool getInt8OutputParams(Layer* layer, float& sc, int& zp)
    {
        if (auto* p = dynamic_cast<Conv2Int8Layer*>(layer))
            { sc = p->output_sc; zp = p->output_zp; return true; }
        if (auto* p = dynamic_cast<Eltwise2Int8Layer*>(layer))
            { sc = p->output_sc; zp = p->output_zp; return true; }
        if (auto* p = dynamic_cast<InnerProductLayerInt8*>(layer))
            { sc = p->output_sc; zp = p->output_zp; return true; }
        return false;
    }

    static void setInt8OutputParams(Layer* layer, float sc, int zp)
    {
        if (auto* p = dynamic_cast<Conv2Int8Layer*>(layer))
            { p->output_sc = sc; p->output_zp = zp; }
        else if (auto* p = dynamic_cast<Eltwise2Int8Layer*>(layer))
            { p->output_sc = sc; p->output_zp = zp; }
        else if (auto* p = dynamic_cast<InnerProductLayerInt8*>(layer))
            { p->output_sc = sc; p->output_zp = zp; }
    }

    template<typename LayerT>
    bool getQdqPatternContext(Layer* layer_ptr,
                              size_t ninputs,
                              const std::vector<Arg>& inputs,
                              const std::vector<int>& producer_of,
                              std::vector<Ptr<Layer> >& newprog,
                              Arg& q_data_in,
                              Arg& out_scale,
                              Arg& out_zp,
                              int& mid_layer_idx,
                              LayerT*& mid_layer) const
    {
        QuantizeLinearLayer* ql = dynamic_cast<QuantizeLinearLayer*>(layer_ptr);
        if (!(ql && ninputs == 3 && usecounts.at(inputs[0].idx) == 1))
            return false;
        q_data_in = inputs[0];
        out_scale = inputs[1];
        out_zp = inputs[2];
        mid_layer_idx = producer_of.at(q_data_in.idx);
        mid_layer = getLayer<LayerT>(newprog, mid_layer_idx);
        return mid_layer != 0;
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
        std::set<int> skip_indices;
        std::vector<Arg> override_outputs;
        std::map<int, std::pair<Arg, Eltwise2Int8Layer*>> relu_to_eltwise;

        for (i = 0; i < nops; i++) {
            if (skip_indices.count((int)i)) continue;
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
            fused_inputs.clear();
            override_outputs.clear();

            for(;;) {
                Arg q_data_in, out_scale, out_zp;
                int add_layer_idx = -1;
                NaryEltwiseLayer* add = 0;
                if (getQdqPatternContext<NaryEltwiseLayer>(layer_ptr, ninputs, inputs, producer_of,
                                                           newprog, q_data_in, out_scale, out_zp,
                                                           add_layer_idx, add) &&
                    add && add->inputs.size() >= 2) {
                        vector<DequantizeLinearLayer*> dq_ptrs;
                        vector<int> dq_prog_indices;
                        vector<Arg> int8_inputs;

                        for (size_t k = 0; k < add->inputs.size(); k++) {
                            const Arg& add_inp = add->inputs[k];
                            int dq_idx = producer_of.at(add_inp.idx);
                            DequantizeLinearLayer* dq =
                                getLayer<DequantizeLinearLayer>(newprog, dq_idx);

                            if (!dq || dq->inputs.size() < 3 ||
                                usecounts.at(add_inp.idx) != 1) {
                                break;
                            }
                            dq_ptrs.push_back(dq);
                            dq_prog_indices.push_back(dq_idx);
                            int8_inputs.push_back(dq->inputs[0]);
                        }

                        const int eltwise_out_type = !outputs.empty() ? netimpl->argData(outputs[0]).type : -1;
                        const bool eltwise_out_int8 = (eltwise_out_type == CV_8S || eltwise_out_type == CV_8U);
                        if (dq_ptrs.size() == add->inputs.size() && eltwise_out_int8) {
                            vector<float> in_scales(2);
                            vector<int>   in_zps(2);
                            bool eltwise_in_int8 = true;
                            for (int k = 0; k < 2; k++) {
                                int inp_type = netimpl->argData(dq_ptrs[k]->inputs[0]).type;
                                eltwise_in_int8 = eltwise_in_int8 && (inp_type == CV_8S || inp_type == CV_8U);
                                in_scales[k] = netimpl->argTensor(dq_ptrs[k]->inputs[1]).at<float>(0);
                                const Mat& zp_m = netimpl->argTensor(dq_ptrs[k]->inputs[2]);
                                in_zps[k] = zp_m.depth() == CV_8U
                                    ? (int)zp_m.at<uint8_t>(0)
                                    : (int)zp_m.at<int8_t>(0);
                            }
                            if (!eltwise_in_int8)
                                break;

                            float out_scale_val = netimpl->argTensor(out_scale).at<float>(0);
                            const Mat& elt_out_zp_m = netimpl->argTensor(out_zp);
                            int out_zp_val = elt_out_zp_m.depth() == CV_8U
                                ? (int)elt_out_zp_m.at<uint8_t>(0)
                                : (int)elt_out_zp_m.at<int8_t>(0);

                            LayerParams eltwiseParams = makeLayerParamsFromOriginal(add, "Eltwise2Int8");
                            eltwiseParams.blobs.clear();
                            eltwiseParams.set("input_scales", DictValue::arrayReal(in_scales.data(), (int)in_scales.size()));
                            eltwiseParams.set("input_zeropoints", DictValue::arrayInt(in_zps.data(), (int)in_zps.size()));
                            eltwiseParams.set("scales", out_scale_val);
                            eltwiseParams.set("zeropoints", out_zp_val);
                            Ptr<Layer> eltwiseInt8 = createFusedLayer(eltwiseParams);
                            if (!eltwiseInt8.empty()) {
                                CV_Assert(dynamic_cast<Eltwise2Int8Layer*>(eltwiseInt8.get()));
                                fused_layer_idx = add_layer_idx;
                                newprog[add_layer_idx] = eltwiseInt8;
                                fused_inputs.swap(int8_inputs);
                                removed_args.push_back(q_data_in);     // float add_out
                                for (const Arg& add_inp : add->inputs)
                                    removed_args.push_back(add_inp);

                                for (int dq_prog_idx : dq_prog_indices)
                                    newprog[dq_prog_idx] = Ptr<Layer>();

                                break;
                            }
                        }
                    }

                int relu_layer_idx = -1;
                ReLULayer* relu = 0;
                if (getQdqPatternContext<ReLULayer>(layer_ptr, ninputs, inputs, producer_of,
                                                    newprog, q_data_in, out_scale, out_zp,
                                                    relu_layer_idx, relu) && relu->inputs.size() == 1) {
                    Arg relu_in = relu->inputs[0];
                    int dq_idx = producer_of.at(relu_in.idx);
                    DequantizeLinearLayer* dq = getLayer<DequantizeLinearLayer>(newprog, dq_idx);

                    const int relu_out_type = !outputs.empty() ? netimpl->argData(outputs[0]).type : -1;
                    const bool relu_out_int8 = (relu_out_type == CV_8S || relu_out_type == CV_8U);
                    const int relu_in_type = (dq && !dq->inputs.empty()) ? netimpl->argData(dq->inputs[0]).type : -1;
                    const bool relu_in_int8 = (relu_in_type == CV_8S || relu_in_type == CV_8U);
                    if (dq && dq->inputs.size() >= 3 &&
                        relu_in_int8 && relu_out_int8 &&
                        usecounts.at(relu_in.idx) == 1) {
                        const float inp_sc = netimpl->argTensor(dq->inputs[1]).at<float>(0);
                        const Mat& relu_zp_m = netimpl->argTensor(dq->inputs[2]);
                        const int inp_zp = relu_zp_m.depth() == CV_8U
                            ? (int)relu_zp_m.at<uint8_t>(0)
                            : (int)relu_zp_m.at<int8_t>(0);
                        const float out_sc = netimpl->argTensor(out_scale).at<float>(0);
                        const Mat& out_zp_relu_m = netimpl->argTensor(out_zp);
                        const int out_zp_i = out_zp_relu_m.depth() == CV_8U
                            ? (int)out_zp_relu_m.at<uint8_t>(0)
                            : (int)out_zp_relu_m.at<int8_t>(0);

                        if (inp_sc > 0.f && out_sc > 0.f) {
                            const bool isU8 = (relu_in_type == CV_8U);
                            Mat lookUpTable(1, 256, isU8 ? CV_8U : CV_8S);
                            if (isU8) {
                                uint8_t* table = lookUpTable.ptr<uint8_t>();
                                for (int t = 0; t < 256; t++) {
                                    float x = inp_sc * (t - inp_zp);
                                    float y = std::max(0.0f, x);
                                    int quantized = out_zp_i + cvRound(y / out_sc);
                                    table[t] = saturate_cast<uint8_t>(quantized);
                                }
                            } else {
                                int8_t* table = lookUpTable.ptr<int8_t>();
                                for (int t = -128; t < 128; t++) {
                                    float x = inp_sc * (t - inp_zp);
                                    float y = std::max(0.0f, x);
                                    int quantized = out_zp_i + cvRound(y / out_sc);
                                    table[t + 128] = saturate_cast<int8_t>(quantized);
                                }
                            }

                            LayerParams reluInt8Params = makeLayerParamsFromOriginal(relu, "ReLUInt8");
                            reluInt8Params.blobs.clear();
                            Ptr<Layer> reluInt8 = createFusedLayer(reluInt8Params);
                            if (!reluInt8.empty()) {
                                auto* reluInt8Layer = dynamic_cast<ActivationLayerInt8*>(reluInt8.get());
                                CV_Assert(reluInt8Layer);
                                reluInt8Layer->input_sc = inp_sc;
                                reluInt8Layer->input_zp = inp_zp;
                                reluInt8Layer->output_sc = out_sc;
                                reluInt8Layer->output_zp = out_zp_i;
                                reluInt8Layer->activationLUT = lookUpTable;
                                fused_layer_idx = relu_layer_idx;
                                newprog[relu_layer_idx] = reluInt8;
                                fused_inputs.assign(1, dq->inputs[0]);
                                removed_args.push_back(q_data_in);
                                removed_args.push_back(relu_in);
                                newprog[dq_idx] = Ptr<Layer>();
                                break;
                            }
                        }
                    }
                }

                QuantizeLinearLayer* ql_compound = dynamic_cast<QuantizeLinearLayer*>(layer_ptr);
                if (ql_compound && ninputs == 3) {
                    Arg q_inp = inputs[0];
                    int relu_layer_idx2 = (q_inp.idx < (int)producer_of.size()) ? producer_of.at(q_inp.idx) : -1;
                    ReLULayer* relu2 = getLayer<ReLULayer>(newprog, relu_layer_idx2);
                    (void)relu_layer_idx2;
                    if (relu2 && relu2->inputs.size() == 1) {
                        Arg relu_in2 = relu2->inputs[0];
                        int add_idx2 = producer_of.at(relu_in2.idx);
                        NaryEltwiseLayer* add2 = getLayer<NaryEltwiseLayer>(newprog, add_idx2);
                        int relu_in_uc = usecounts.at(relu_in2.idx);
                        if (add2 && add2->inputs.size() >= 2 &&
                            relu_in_uc == 1) {
                            vector<DequantizeLinearLayer*> dq_ptrs2;
                            vector<int> dq_prog_indices2;
                            vector<Arg> int8_inputs2;
                            for (size_t k = 0; k < add2->inputs.size(); k++) {
                                const Arg& add_inp = add2->inputs[k];
                                int dq_idx2 = producer_of.at(add_inp.idx);
                                DequantizeLinearLayer* dq2 = getLayer<DequantizeLinearLayer>(newprog, dq_idx2);

                                if (dq2 && dq2->inputs.size() >= 3) {
                                    dq_ptrs2.push_back(dq2);
                                    dq_prog_indices2.push_back(dq_idx2);
                                    int8_inputs2.push_back(dq2->inputs[0]);
                                } else {
                                    int arg_type = netimpl->argData(add_inp).type;
                                    bool found_int8 = (arg_type == CV_8S || arg_type == CV_8U);
                                    Arg int8_arg = add_inp;
                                    if (!found_int8) {
                                        auto it = relu_to_eltwise.find(add_inp.idx);
                                        if (it != relu_to_eltwise.end()) {
                                            int8_arg = it->second.first;
                                            found_int8 = true;
                                        }
                                    }
                                    if (found_int8) {
                                        dq_ptrs2.push_back(nullptr);
                                        dq_prog_indices2.push_back(-1);
                                        int8_inputs2.push_back(int8_arg);
                                    } else {
                                        break;
                                    }
                                }
                            }

                            int elt_out_type2 = !outputs.empty() ? netimpl->argData(outputs[0]).type : -1;
                            if (elt_out_type2 < 0) {
                                const Mat& zp_tensor = netimpl->argTensor(inputs[2]);
                                elt_out_type2 = !zp_tensor.empty() ? zp_tensor.type() : CV_8S;
                            }
                            const bool elt_out_int82 = (elt_out_type2 == CV_8S || elt_out_type2 == CV_8U);
                            if (int8_inputs2.size() == add2->inputs.size() && elt_out_int82) {
                                vector<float> in_scales2(2);
                                vector<int>   in_zps2(2);
                                bool elt_in_int82 = true;
                                for (int k = 0; k < 2; k++) {
                                    if (dq_ptrs2[k]) {
                                        int it = netimpl->argData(dq_ptrs2[k]->inputs[0]).type;
                                        if (it < 0) {
                                            const Mat& zp_infer = netimpl->argTensor(dq_ptrs2[k]->inputs[2]);
                                            it = !zp_infer.empty() ? zp_infer.type() : CV_8S;
                                        }
                                        elt_in_int82 = elt_in_int82 && (it == CV_8S || it == CV_8U);
                                        in_scales2[k] = netimpl->argTensor(dq_ptrs2[k]->inputs[1]).at<float>(0);
                                        const Mat& zp_m2 = netimpl->argTensor(dq_ptrs2[k]->inputs[2]);
                                        in_zps2[k] = zp_m2.depth() == CV_8U
                                            ? (int)zp_m2.at<uint8_t>(0)
                                            : (int)zp_m2.at<int8_t>(0);
                                    } else {
                                        auto relu_it = relu_to_eltwise.find(add2->inputs[k].idx);
                                        if (relu_it != relu_to_eltwise.end()) {
                                            Eltwise2Int8Layer* elt2 = relu_it->second.second;
                                            if (elt2) {
                                                in_scales2[k] = elt2->output_sc;
                                                in_zps2[k] = elt2->output_zp;
                                                continue;
                                            }
                                        }
                                        int prod_idx = producer_of.at(add2->inputs[k].idx);
                                        Layer* prod = prod_idx >= 0 && !newprog[prod_idx].empty()
                                            ? newprog[prod_idx].get() : nullptr;
                                        if (!prod || !getInt8OutputParams(prod, in_scales2[k], in_zps2[k]))
                                            elt_in_int82 = false;
                                    }
                                }
                                if (elt_in_int82) {
                                    float out_sc2 = netimpl->argTensor(inputs[1]).at<float>(0);
                                    const Mat& out_zp_m2 = netimpl->argTensor(inputs[2]);
                                    int out_zp_val2 = out_zp_m2.depth() == CV_8U
                                        ? (int)out_zp_m2.at<uint8_t>(0)
                                        : (int)out_zp_m2.at<int8_t>(0);
                                    if (out_sc2 > 0.f) {
                                        int relu_out_uc = usecounts.at(q_inp.idx);
                                        LayerParams eltParams = makeLayerParamsFromOriginal(add2, "Eltwise2Int8");
                                        eltParams.blobs.clear();
                                        eltParams.set("input_scales", DictValue::arrayReal(in_scales2.data(), (int)in_scales2.size()));
                                        eltParams.set("input_zeropoints", DictValue::arrayInt(in_zps2.data(), (int)in_zps2.size()));
                                        eltParams.set("scales", out_sc2);
                                        eltParams.set("zeropoints", out_zp_val2);
                                        eltParams.set("with_relu", true);
                                        Ptr<Layer> eltInt8 = createFusedLayer(eltParams);
                                        if (!eltInt8.empty()) {
                                            CV_Assert(dynamic_cast<Eltwise2Int8Layer*>(eltInt8.get()));

                                            fused_inputs = int8_inputs2;
                                            if (relu_out_uc <= 1) {
                                                fused_layer_idx = add_idx2;
                                                newprog[add_idx2] = eltInt8;
                                                newprog[relu_layer_idx2] = Ptr<Layer>();
                                                removed_args.push_back(q_inp);   // relu_out
                                                removed_args.push_back(relu_in2); // add_out
                                                for (size_t dk = 0; dk < add2->inputs.size(); dk++) {
                                                    removed_args.push_back(add2->inputs[dk]);
                                                }
                                                for (int dq_prog_idx : dq_prog_indices2) {
                                                    if (dq_prog_idx >= 0)
                                                        newprog[dq_prog_idx] = Ptr<Layer>();
                                                }
                                            } else {
                                                int new_idx = (int)newprog.size();
                                                newprog.push_back(eltInt8);
                                                fused_layer_idx = new_idx;
                                                usecounts.at(q_inp.idx) -= 1;
                                                Eltwise2Int8Layer* elt2ptr = dynamic_cast<Eltwise2Int8Layer*>(eltInt8.get());
                                                relu_to_eltwise[q_inp.idx] = {outputs[0], elt2ptr};
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Arg out_scale_arg, out_zp_arg;
                int conv_layer_idx = -1;
                Conv2Layer* conv = 0;
                if (getQdqPatternContext<Conv2Layer>(layer_ptr, ninputs, inputs, producer_of,
                                                           newprog, q_data_in, out_scale_arg, out_zp_arg,
                                                           conv_layer_idx, conv) &&
                    (conv->inputs.size() == 2 || conv->inputs.size() == 3)) {
                        const Arg conv_x = conv->inputs[0];
                        const Arg conv_w = conv->inputs[1];
                        const int dq_x_idx = producer_of.at(conv_x.idx);
                        const int dq_w_idx = producer_of.at(conv_w.idx);
                        DequantizeLinearLayer* dq_x = getLayer<DequantizeLinearLayer>(newprog, dq_x_idx);
                        DequantizeLinearLayer* dq_w = getLayer<DequantizeLinearLayer>(newprog, dq_w_idx);

                        if (dq_x && dq_w &&
                            dq_x->inputs.size() >= 3 && dq_w->inputs.size() >= 3 &&
                            usecounts.at(conv_w.idx) == 1) {
                            float inp_sc = netimpl->argTensor(dq_x->inputs[1]).at<float>(0);
                            float out_sc = netimpl->argTensor(out_scale_arg).at<float>(0);
                            int inp_zp = 0, out_zp = 0;
                            const Mat& x_zp_m_read = netimpl->argTensor(dq_x->inputs[2]);
                            if (x_zp_m_read.depth() == CV_8S)
                                inp_zp = (int)x_zp_m_read.at<int8_t>(0);
                            else
                                inp_zp = (int)x_zp_m_read.at<uint8_t>(0);
                            const Mat& out_zp_m = netimpl->argTensor(out_zp_arg);
                            if (out_zp_m.depth() == CV_8S)
                                out_zp = (int)out_zp_m.at<int8_t>(0);
                            else if (out_zp_m.depth() == CV_8U)
                                out_zp = (int)out_zp_m.at<uint8_t>(0);
                            else
                                out_zp = out_zp_m.at<int>(0);
                            if (!(inp_sc > 0.f && out_sc > 0.f))
                                break;
                            Mat w_q = netimpl->argTensor(dq_w->inputs[0]);
                            Mat w_sc_m = netimpl->argTensor(dq_w->inputs[1]);
                            Mat w_zp_m = netimpl->argTensor(dq_w->inputs[2]);
                            const Mat& x_zp_m = netimpl->argTensor(dq_x->inputs[2]);
                            if (!w_q.empty() && w_q.depth() == CV_8S && w_q.dims >= 3) {
                                const int outCn = w_q.size[0];
                                Mat wt_sc = (w_sc_m.total() == (size_t)outCn)
                                            ? w_sc_m.reshape(1, 1)
                                            : Mat(1, outCn, CV_32F, Scalar(w_sc_m.at<float>(0))).clone();
                                bool per_channel = w_sc_m.total() == (size_t)outCn;

                                bool all_wzp_zero = true;
                                if (w_zp_m.total() > 1 && w_zp_m.total() != (size_t)outCn)
                                    all_wzp_zero = false;
                                for (size_t t = 0; all_wzp_zero && t < w_zp_m.total(); t++) {
                                    int wz = w_zp_m.depth() == CV_8S ? (int)w_zp_m.at<int8_t>((int)t)
                                                                        : (int)w_zp_m.at<uint8_t>((int)t);
                                    if (wz != 0) all_wzp_zero = false;
                                }

                                bool symmetric_pads = true;
                                size_t npads = conv->pads.size();
                                size_t ndims_pad = npads / 2;
                                for (size_t d = 0; d < ndims_pad && symmetric_pads; d++) {
                                    if (conv->pads[d] != conv->pads[d + ndims_pad])
                                        symmetric_pads = false;
                                }

                                if (all_wzp_zero && symmetric_pads) {
                                    Mat bias = Mat::zeros(1, outCn, CV_32S);
                                    bool biasOk = true;
                                    int dq_bias_idx = -1;
                                    if (conv->inputs.size() == 3) {
                                        if (netimpl->isConstArg(conv->inputs[2])) {
                                            Mat b = netimpl->argTensor(conv->inputs[2]);
                                            if (b.empty() || b.total() != (size_t)outCn) {
                                                biasOk = false;
                                            } else if (b.depth() == CV_32S) {
                                                bias = b.reshape(1, 1);
                                            } else if (b.depth() == CV_32F || b.depth() == CV_64F) {
                                                Mat b1 = b.reshape(1, 1);
                                                for (int oc = 0; oc < outCn; oc++) {
                                                    const float b_real = b1.depth() == CV_32F
                                                        ? b1.at<float>(oc)
                                                        : (float)b1.at<double>(oc);
                                                    const float denom = inp_sc * wt_sc.at<float>(oc);
                                                    if (std::abs(denom) < 1e-12f) { biasOk = false; break; }
                                                    bias.at<int>(oc) = cvRound(b_real / denom);
                                                }
                                            } else {
                                                biasOk = false;
                                            }
                                        } else {
                                            dq_bias_idx = producer_of.at(conv->inputs[2].idx);
                                            DequantizeLinearLayer* dq_b =
                                                getLayer<DequantizeLinearLayer>(newprog, dq_bias_idx);
                                            if (!dq_b || dq_b->inputs.size() < 2 ||
                                                usecounts.at(conv->inputs[2].idx) != 1 ||
                                                !netimpl->isConstArg(dq_b->inputs[0])) {
                                                biasOk = false;
                                            } else {
                                                Mat bq = netimpl->argTensor(dq_b->inputs[0]);
                                                if (bq.empty() || bq.total() != (size_t)outCn || bq.depth() != CV_32S)
                                                    biasOk = false;
                                                else
                                                    bias = bq.reshape(1, 1);
                                            }
                                        }
                                    }
                                    if (!biasOk)
                                        break;

                                    const bool inputIsU8 =
                                        (netimpl->argData(dq_x->inputs[0]).type == CV_8U) ||
                                        (x_zp_m.depth() == CV_8U);
                                    const int inp_zp_kernel = inputIsU8 ? (inp_zp - 128) : inp_zp;
                                    Mat weights_2d = w_q.reshape(1, outCn);
                                    Mat biasFused(1, outCn, CV_32S);
                                    Mat outputMultiplier(1, outCn, CV_32F);
                                    for (int oc = 0; oc < outCn; oc++) {
                                        biasFused.at<int>(oc) = bias.at<int>(oc) - inp_zp_kernel * (int)cv::sum(weights_2d.row(oc))[0];
                                        outputMultiplier.at<float>(oc) = (inp_sc * wt_sc.at<float>(oc)) / out_sc;
                                    }

                                    LayerParams convInt8Params = makeLayerParamsFromOriginal(conv, "Conv2Int8");
                                    {
                                        if (!conv->strides.empty())
                                            convInt8Params.set("stride", DictValue::arrayInt(conv->strides.data(), (int)conv->strides.size()));
                                        if (!conv->dilations.empty())
                                            convInt8Params.set("dilation", DictValue::arrayInt(conv->dilations.data(), (int)conv->dilations.size()));
                                        if (!conv->pads.empty())
                                            convInt8Params.set("pad", DictValue::arrayInt(conv->pads.data(), (int)conv->pads.size()));
                                    }
                                    convInt8Params.set("num_output", outCn);
                                    convInt8Params.set("group", conv->ngroups);
                                    convInt8Params.set("input_scale", inp_sc);
                                    convInt8Params.set("input_zeropoint", inp_zp);
                                    convInt8Params.set("scales", out_sc);
                                    convInt8Params.set("zeropoints", out_zp);
                                    convInt8Params.set("per_channel", per_channel);
                                    convInt8Params.blobs.resize(3);
                                    convInt8Params.blobs[0] = w_q;
                                    convInt8Params.blobs[1] = biasFused;
                                    convInt8Params.blobs[2] = outputMultiplier;
                                    Ptr<Layer> convInt8 = createFusedLayer(convInt8Params);
                                    if (!convInt8.empty()) {
                                        auto* convInt8Layer = dynamic_cast<Conv2Int8Layer*>(convInt8.get());
                                        CV_Assert(convInt8Layer);
                                        fused_layer_idx = conv_layer_idx;
                                        newprog[conv_layer_idx] = convInt8;
                                        fused_inputs.assign(1, dq_x->inputs[0]);
                                        removed_args.push_back(q_data_in);
                                        removed_args.push_back(conv_w);
                                        if (conv->inputs.size() == 3) {
                                            removed_args.push_back(conv->inputs[2]);
                                            if (dq_bias_idx >= 0)
                                                newprog[dq_bias_idx] = Ptr<Layer>();
                                        }
                                        if (usecounts.at(conv_x.idx) == 1) {
                                            removed_args.push_back(conv_x);
                                            newprog[dq_x_idx] = Ptr<Layer>();
                                        }
                                        newprog[dq_w_idx] = Ptr<Layer>();
                                        break;
                                    }
                                }
                            }
                        }
                    }

                int mm_layer_idx = -1;
                MatMulLayer* mm = 0;
                if (getQdqPatternContext<MatMulLayer>(layer_ptr, ninputs, inputs, producer_of,
                                                      newprog, q_data_in, out_scale, out_zp,
                                                      mm_layer_idx, mm) &&
                    mm->inputs.size() == 2) {
                        const Arg mm_x = mm->inputs[0];
                        const Arg mm_w = mm->inputs[1];
                        int dq_x_idx = producer_of.at(mm_x.idx);
                        int dq_w_idx = producer_of.at(mm_w.idx);
                        DequantizeLinearLayer* dq_x = getLayer<DequantizeLinearLayer>(newprog, dq_x_idx);
                        DequantizeLinearLayer* dq_w = getLayer<DequantizeLinearLayer>(newprog, dq_w_idx);
                        float inp_sc = 0.f, out_sc = 0.f;
                        int inp_zp = 0, out_zp_i = 0;
                        int fc_out_type = !outputs.empty() ? netimpl->argData(outputs[0]).type : -1;
                        if (fc_out_type < 0) {
                            const Mat& zp_t = netimpl->argTensor(inputs[2]);
                            fc_out_type = !zp_t.empty() ? zp_t.type() : CV_8S;
                        }
                        const bool fc_out_int8 = (fc_out_type == CV_8S || fc_out_type == CV_8U);
                        int fc_in_type = (dq_x && !dq_x->inputs.empty()) ? netimpl->argData(dq_x->inputs[0]).type : -1;
                        if (fc_in_type < 0 && dq_x && dq_x->inputs.size() >= 3) {
                            const Mat& zp_t = netimpl->argTensor(dq_x->inputs[2]);
                            fc_in_type = !zp_t.empty() ? zp_t.type() : CV_8S;
                        }
                        const bool fc_in_int8 = (fc_in_type == CV_8S || fc_in_type == CV_8U);
                        if (dq_x && dq_w &&
                            dq_x->inputs.size() >= 3 && dq_w->inputs.size() >= 3 &&
                            fc_in_int8 && fc_out_int8 &&
                            usecounts.at(mm_x.idx) == 1 && usecounts.at(mm_w.idx) == 1) {
                            inp_sc = netimpl->argTensor(dq_x->inputs[1]).at<float>(0);
                            const Mat& fc_zp_m = netimpl->argTensor(dq_x->inputs[2]);

                            inp_zp = fc_zp_m.depth() == CV_8U
                                ? (int)fc_zp_m.at<uint8_t>(0) - 128
                                : (int)fc_zp_m.at<int8_t>(0);
                            out_sc = netimpl->argTensor(out_scale).at<float>(0);
                            const Mat& fc_out_zp_m = netimpl->argTensor(out_zp);
                            out_zp_i = fc_out_zp_m.depth() == CV_8U
                                ? (int)fc_out_zp_m.at<uint8_t>(0)
                                : (int)fc_out_zp_m.at<int8_t>(0);
                            if (!(inp_sc > 0.f && out_sc > 0.f))
                                break;
                            Mat w_q = netimpl->argTensor(dq_w->inputs[0]);
                            Mat w_sc_m = netimpl->argTensor(dq_w->inputs[1]);
                            Mat w_zp_m = netimpl->argTensor(dq_w->inputs[2]);
                            if (!w_q.empty() && w_q.depth() == CV_8S && w_q.dims == 2) {
                                bool all_wzp_zero = true;
                                for (size_t t = 0; all_wzp_zero && t < w_zp_m.total(); t++) {
                                    int wz = w_zp_m.depth() == CV_8S ? (int)w_zp_m.at<int8_t>((int)t)
                                                                     : (int)w_zp_m.at<uint8_t>((int)t);
                                    if (wz != 0) all_wzp_zero = false;
                                }
                                if (all_wzp_zero) {
                                    Mat weights = w_q.t();
                                    int outCn = weights.size[0];
                                    Mat wt_sc = (w_sc_m.total() == (size_t)outCn)
                                              ? w_sc_m.reshape(1, 1)
                                              : Mat(1, outCn, CV_32F, Scalar(w_sc_m.at<float>(0))).clone();
                                    bool per_channel = w_sc_m.total() == (size_t)outCn;
                                    Mat bias(1, outCn, CV_32S);
                                    Mat outputMultiplier(1, outCn, CV_32F);
                                    for (int ioc = 0; ioc < outCn; ioc++) {
                                        bias.at<int>(ioc) = -inp_zp * (int)cv::sum(weights.row(ioc))[0];
                                        outputMultiplier.at<float>(ioc) = (inp_sc * wt_sc.at<float>(ioc)) / out_sc;
                                    }
                                    int firstInpDims = (int)netimpl->argData(mm_x).shape.size();
                                    int axis = std::max(1, firstInpDims - w_q.dims + 1);

                                    LayerParams fcInt8Params = makeLayerParamsFromOriginal(mm, "InnerProductInt8");
                                    fcInt8Params.set("num_output", outCn);
                                    fcInt8Params.set("axis", axis);
                                    fcInt8Params.blobs.resize(3);
                                    fcInt8Params.blobs[0] = weights;
                                    fcInt8Params.blobs[1] = bias;
                                    fcInt8Params.blobs[2] = outputMultiplier;
                                    Ptr<Layer> fcInt8 = createFusedLayer(fcInt8Params);
                                    if (!fcInt8.empty()) {
                                        auto* fcInt8Layer = dynamic_cast<InnerProductLayerInt8*>(fcInt8.get());
                                        CV_Assert(fcInt8Layer);
                                        fcInt8Layer->input_zp = inp_zp;
                                        fcInt8Layer->input_sc = inp_sc;
                                        fcInt8Layer->output_zp = out_zp_i;
                                        fcInt8Layer->output_sc = out_sc;
                                        fcInt8Layer->output_type = fc_out_type;
                                        fcInt8Layer->per_channel = per_channel;
                                        fused_layer_idx = mm_layer_idx;
                                        newprog[mm_layer_idx] = fcInt8;
                                        fused_inputs.assign(1, dq_x->inputs[0]);
                                        removed_args.push_back(q_data_in);
                                        removed_args.push_back(mm_x);
                                        removed_args.push_back(mm_w);
                                        newprog[dq_x_idx] = Ptr<Layer>();
                                        newprog[dq_w_idx] = Ptr<Layer>();
                                        break;
                                    }
                                }
                            }
                        }
                    }

                int add_bias_idx = -1;
                NaryEltwiseLayer* add_bias = 0;
                if (getQdqPatternContext<NaryEltwiseLayer>(layer_ptr, ninputs, inputs, producer_of,
                                                            newprog, q_data_in, out_scale, out_zp,
                                                            add_bias_idx, add_bias) &&
                    add_bias && add_bias->inputs.size() == 2) {
                    int mm_inp_k = -1;
                    int bias_inp_k = -1;
                    MatMulLayer* mm2 = nullptr;
                    int mm2_idx = -1;
                    for (int k = 0; k < 2; k++) {
                        int prod_idx = producer_of.at(add_bias->inputs[k].idx);
                        MatMulLayer* candidate = getLayer<MatMulLayer>(newprog, prod_idx);
                        if (candidate && candidate->inputs.size() == 2) {
                            mm_inp_k = k;
                            mm2 = candidate;
                            mm2_idx = prod_idx;
                        } else if (netimpl->isConstArg(add_bias->inputs[k])) {
                            bias_inp_k = k;
                        }
                    }
                    if (mm2 && bias_inp_k >= 0 && mm_inp_k >= 0 &&
                        usecounts.at(add_bias->inputs[mm_inp_k].idx) == 1) {
                        const Arg mm_x = mm2->inputs[0];
                        const Arg mm_w = mm2->inputs[1];
                        int dq_x_idx = producer_of.at(mm_x.idx);
                        int dq_w_idx = producer_of.at(mm_w.idx);
                        DequantizeLinearLayer* dq_x = getLayer<DequantizeLinearLayer>(newprog, dq_x_idx);
                        DequantizeLinearLayer* dq_w = getLayer<DequantizeLinearLayer>(newprog, dq_w_idx);
                        int fc_out_type = !outputs.empty() ? netimpl->argData(outputs[0]).type : -1;
                        if (fc_out_type < 0) {
                            const Mat& zp_t = netimpl->argTensor(inputs[2]);
                            fc_out_type = !zp_t.empty() ? zp_t.type() : CV_8S;
                        }
                        const bool fc_out_int8 = (fc_out_type == CV_8S || fc_out_type == CV_8U);
                        int fc_in_type = (dq_x && !dq_x->inputs.empty()) ? netimpl->argData(dq_x->inputs[0]).type : -1;
                        if (fc_in_type < 0 && dq_x && dq_x->inputs.size() >= 3) {
                            const Mat& zp_t = netimpl->argTensor(dq_x->inputs[2]);
                            fc_in_type = !zp_t.empty() ? zp_t.type() : CV_8S;
                        }
                        const bool fc_in_int8 = (fc_in_type == CV_8S || fc_in_type == CV_8U);
                        if (dq_x && dq_w &&
                            dq_x->inputs.size() >= 3 && dq_w->inputs.size() >= 3 &&
                            fc_in_int8 && fc_out_int8 &&
                            usecounts.at(mm_x.idx) == 1 && usecounts.at(mm_w.idx) == 1) {
                            float inp_sc = netimpl->argTensor(dq_x->inputs[1]).at<float>(0);
                            const Mat& fc_zp_m = netimpl->argTensor(dq_x->inputs[2]);
                            int inp_zp = fc_zp_m.depth() == CV_8U
                                ? (int)fc_zp_m.at<uint8_t>(0) - 128
                                : (int)fc_zp_m.at<int8_t>(0);
                            float out_sc_val = netimpl->argTensor(out_scale).at<float>(0);
                            const Mat& fc_out_zp_m = netimpl->argTensor(out_zp);
                            int out_zp_i = fc_out_zp_m.depth() == CV_8U
                                ? (int)fc_out_zp_m.at<uint8_t>(0)
                                : (int)fc_out_zp_m.at<int8_t>(0);
                            if (inp_sc > 0.f && out_sc_val > 0.f) {
                                Mat w_q = netimpl->argTensor(dq_w->inputs[0]);
                                Mat w_sc_m = netimpl->argTensor(dq_w->inputs[1]);
                                Mat w_zp_m = netimpl->argTensor(dq_w->inputs[2]);
                                if (!w_q.empty() && w_q.depth() == CV_8S && w_q.dims == 2) {
                                    bool all_wzp_zero = true;
                                    for (size_t t = 0; all_wzp_zero && t < w_zp_m.total(); t++) {
                                        int wz = w_zp_m.depth() == CV_8S ? (int)w_zp_m.at<int8_t>((int)t)
                                                                            : (int)w_zp_m.at<uint8_t>((int)t);
                                        if (wz != 0) all_wzp_zero = false;
                                    }
                                    if (all_wzp_zero) {
                                        Mat weights = w_q.t();
                                        int outCn = weights.size[0];
                                        Mat wt_sc = (w_sc_m.total() == (size_t)outCn)
                                                    ? w_sc_m.reshape(1, 1)
                                                    : Mat(1, outCn, CV_32F, Scalar(w_sc_m.at<float>(0))).clone();
                                        bool per_channel = w_sc_m.total() == (size_t)outCn;

                                        // Fuse the float bias into int32 bias
                                        Mat float_bias = netimpl->argTensor(add_bias->inputs[bias_inp_k]);
                                        Mat bias(1, outCn, CV_32S);
                                        Mat outputMultiplier(1, outCn, CV_32F);
                                        bool biasOk = (float_bias.total() == (size_t)outCn);
                                        for (int ioc = 0; ioc < outCn && biasOk; ioc++) {
                                            float denom = inp_sc * wt_sc.at<float>(ioc);
                                            if (std::abs(denom) < 1e-12f) { biasOk = false; break; }
                                            float b_val = 0.f;
                                            if (float_bias.depth() == CV_32F)
                                                b_val = float_bias.at<float>(ioc);
                                            else if (float_bias.depth() == CV_64F)
                                                b_val = (float)float_bias.at<double>(ioc);
                                            bias.at<int>(ioc) = cvRound(b_val / denom) - inp_zp * (int)cv::sum(weights.row(ioc))[0];
                                            outputMultiplier.at<float>(ioc) = denom / out_sc_val;
                                        }
                                        if (biasOk) {
                                            int firstInpDims = (int)netimpl->argData(mm_x).shape.size();
                                            int fc_axis = std::max(1, firstInpDims - w_q.dims + 1);

                                            LayerParams fcInt8Params = makeLayerParamsFromOriginal(mm2, "InnerProductInt8");
                                            fcInt8Params.set("num_output", outCn);
                                            fcInt8Params.set("axis", fc_axis);
                                            fcInt8Params.blobs.resize(3);
                                            fcInt8Params.blobs[0] = weights;
                                            fcInt8Params.blobs[1] = bias;
                                            fcInt8Params.blobs[2] = outputMultiplier;
                                            Ptr<Layer> fcInt8 = createFusedLayer(fcInt8Params);
                                            if (!fcInt8.empty()) {
                                                auto* fcInt8Layer = dynamic_cast<InnerProductLayerInt8*>(fcInt8.get());
                                                CV_Assert(fcInt8Layer);
                                                fcInt8Layer->input_zp = inp_zp;
                                                fcInt8Layer->input_sc = inp_sc;
                                                fcInt8Layer->output_zp = out_zp_i;
                                                fcInt8Layer->output_sc = out_sc_val;
                                                fcInt8Layer->output_type = fc_out_type;
                                                fcInt8Layer->per_channel = per_channel;
                                                fused_layer_idx = add_bias_idx;
                                                newprog[add_bias_idx] = fcInt8;
                                                fused_inputs.assign(1, dq_x->inputs[0]);
                                                removed_args.push_back(q_data_in);
                                                removed_args.push_back(add_bias->inputs[mm_inp_k]); // matmul out
                                                removed_args.push_back(mm_x);
                                                removed_args.push_back(mm_w);
                                                newprog[mm2_idx] = Ptr<Layer>();
                                                newprog[dq_x_idx] = Ptr<Layer>();
                                                newprog[dq_w_idx] = Ptr<Layer>();
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                int pool_layer_idx = -1;
                PoolingLayer* pool = 0;
                if (getQdqPatternContext<PoolingLayer>(layer_ptr, ninputs, inputs, producer_of,
                                                       newprog, q_data_in, out_scale, out_zp,
                                                       pool_layer_idx, pool) && pool->inputs.size() == 1) {
                    Arg pool_in = pool->inputs[0];
                    int dq_idx = producer_of.at(pool_in.idx);
                    DequantizeLinearLayer* dq = getLayer<DequantizeLinearLayer>(newprog, dq_idx);
                    float inp_sc = 0.f, out_sc = 0.f;
                    int inp_zp = 0, out_zp_i = 0;
                    const int pool_out_type = !outputs.empty() ? netimpl->argData(outputs[0]).type : -1;
                    const bool pool_out_int8 = (pool_out_type == CV_8S || pool_out_type == CV_8U);
                    const int pool_in_type = (dq && !dq->inputs.empty()) ? netimpl->argData(dq->inputs[0]).type : -1;
                    const bool pool_in_int8 = (pool_in_type == CV_8S || pool_in_type == CV_8U);

                    if (dq && dq->inputs.size() >= 3 && pool_in_int8 && pool_out_int8 && usecounts.at(pool_in.idx) == 1) {
                        inp_sc = netimpl->argTensor(dq->inputs[1]).at<float>(0);
                        const Mat& pool_zp_m = netimpl->argTensor(dq->inputs[2]);
                        inp_zp = pool_zp_m.depth() == CV_8U
                            ? (int)pool_zp_m.at<uint8_t>(0)
                            : (int)pool_zp_m.at<int8_t>(0);
                        out_sc = netimpl->argTensor(out_scale).at<float>(0);
                        const Mat& pool_out_zp_m = netimpl->argTensor(out_zp);
                        out_zp_i = pool_out_zp_m.depth() == CV_8U
                            ? (int)pool_out_zp_m.at<uint8_t>(0)
                            : (int)pool_out_zp_m.at<int8_t>(0);
                        bool isGlobalAve = pool->globalPooling;
                        bool isMax = !isGlobalAve;
                        if ((isGlobalAve && inp_sc > 0.f && out_sc > 0.f) ||
                            (isMax && std::abs(inp_sc - out_sc) < 1e-6f && inp_zp == out_zp_i)) {
                            LayerParams poolInt8Params = makeLayerParamsFromOriginal(pool, "Pool2Int8");
                            poolInt8Params.blobs.clear();
                            poolInt8Params.set("input_scale", inp_sc);
                            poolInt8Params.set("input_zeropoint", inp_zp);
                            poolInt8Params.set("scales", out_sc);
                            poolInt8Params.set("zeropoints", out_zp_i);
                            poolInt8Params.set("is_max_pool", isMax);
                            poolInt8Params.set("global_pooling", isGlobalAve);
                            Ptr<Layer> poolInt8 = createFusedLayer(poolInt8Params);
                            if (!poolInt8.empty()) {
                                auto* poolInt8Layer = dynamic_cast<Pool2Int8Layer*>(poolInt8.get());
                                CV_Assert(poolInt8Layer);
                                fused_layer_idx = pool_layer_idx;
                                newprog[pool_layer_idx] = poolInt8;
                                fused_inputs.assign(1, dq->inputs[0]);
                                removed_args.push_back(q_data_in);
                                removed_args.push_back(pool_in);
                                newprog[dq_idx] = Ptr<Layer>();
                                break;
                            }
                        }
                    }
                }

                ActivationLayerInt8* activ_int8 = dynamic_cast<ActivationLayerInt8*>(layer_ptr);
                if (activ_int8 && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg activ_inp = inputs[0];
                    int producer_idx = producer_of.at(activ_inp.idx);
                    if (producer_idx >= 0 && !newprog[producer_idx].empty()) {
                        Layer* producer_layer = newprog[producer_idx].get();
                        float prod_sc = 0.f;
                        int prod_zp = 0;
                        if (getInt8OutputParams(producer_layer, prod_sc, prod_zp) &&
                            prod_sc == activ_int8->input_sc &&
                            prod_zp == activ_int8->input_zp) {
                            Ptr<ActivationLayer> activ_layer = layer.dynamicCast<ActivationLayer>();
                            if (newprog[producer_idx]->setActivation(activ_layer)) {
                                setInt8OutputParams(producer_layer,
                                                    activ_int8->output_sc,
                                                    activ_int8->output_zp);
                                fused_layer_idx = producer_idx;
                                removed_args.push_back(activ_inp);
                                break;
                            }
                        }
                    }
                }
                break;
            }

            if (fused_layer_idx >= 0) {
                modified = true;
                Layer* fused_layer = newprog[fused_layer_idx];
                const std::vector<Arg>& final_outputs = override_outputs.empty() ? outputs : override_outputs;
                fused_layer->outputs = final_outputs;
                if (!fused_inputs.empty())
                    fused_layer->inputs = fused_inputs;
                for (const Arg& new_out: final_outputs)
                    producer_of[new_out.idx] = fused_layer_idx;
                for (const Arg& old_out: removed_args) {
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
            std::vector<int> uc(nargs, 0);
            for (const auto& layer : newprog) {
                if (layer.empty()) continue;
                for (const Arg& inp : layer->inputs) {
                    if (inp.idx >= 0 && inp.idx < (int)nargs)
                        uc[inp.idx]++;
                }
            }
            const auto& graph_outputs = graph->outputs();
            for (const Arg& out : graph_outputs) {
                if (out.idx >= 0 && out.idx < (int)nargs)
                    uc[out.idx]++;
            }
            bool changed = true;
            while (changed) {
                changed = false;
                for (auto& layer : newprog) {
                    if (layer.empty()) continue;
                    bool all_dead = true;
                    for (const Arg& out : layer->outputs) {
                        if (out.idx >= 0 && out.idx < (int)nargs && uc[out.idx] > 0) {
                            all_dead = false;
                            break;
                        }
                    }
                    if (all_dead) {
                        for (const Arg& inp : layer->inputs) {
                            if (inp.idx >= 0 && inp.idx < (int)nargs)
                                uc[inp.idx]--;
                        }
                        layer = Ptr<Layer>();
                        changed = true;
                    }
                }
            }
            size_t i, j = 0, newops = newprog.size();
            for (i = 0; i < newops; i++) {
                if (!newprog[i].empty()) {
                    if (j < i)
                        newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            graph->setProg(newprog);
        }

        return modified;
    }

    Net::Impl* netimpl;
    vector<int> usecounts;
};

void Net::Impl::fuseQDQ()
{
    ModelFusionQDQ qdqFusion(this);
    qdqFusion.fuse();
}

CV__DNN_INLINE_NS_END
}}  // cv::dnn
