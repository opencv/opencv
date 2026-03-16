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
        return LayerFactory::createLayerInstance(params.type, params);
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
            fused_inputs.clear(); // leave it empty to re-use original fused node inputs as-is.

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
                            int8_inputs.push_back(dq->inputs[0]); // int8 quantized tensor
                        }

                        const bool eltwise_out_s8 = !outputs.empty() && netimpl->argData(outputs[0]).type == CV_8S;
                        if (dq_ptrs.size() == add->inputs.size() && eltwise_out_s8) {
                            // Read per-input scale and zero-point from the DQ const args.
                            vector<float> in_scales(2);
                            vector<int>   in_zps(2);
                            bool eltwise_in_s8 = true;
                            for (int k = 0; k < 2; k++) {
                                eltwise_in_s8 = eltwise_in_s8 && (netimpl->argData(dq_ptrs[k]->inputs[0]).type == CV_8S);
                                in_scales[k] = netimpl->argTensor(dq_ptrs[k]->inputs[1]).at<float>(0);
                                in_zps[k] = (int)netimpl->argTensor(dq_ptrs[k]->inputs[2]).at<int8_t>(0);
                            }
                            if (!eltwise_in_s8)
                                break;

                            float out_scale_val = netimpl->argTensor(out_scale).at<float>(0);
                            int   out_zp_val    = (int)netimpl->argTensor(out_zp).at<int8_t>(0);

                            LayerParams eltwiseParams = makeLayerParamsFromOriginal(add, "EltwiseInt8");
                            eltwiseParams.blobs.clear();
                            Ptr<Layer> eltwiseInt8 = createFusedLayer(eltwiseParams);
                            if (!eltwiseInt8.empty()) {
                                auto* elt = dynamic_cast<EltwiseLayerInt8*>(eltwiseInt8.get());
                                CV_Assert(elt);
                                elt->scales = in_scales;
                                elt->zeropoints = in_zps;
                                elt->output_sc = out_scale_val;
                                elt->output_zp = out_zp_val;
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
                                                    relu_layer_idx, relu) &&
                    relu->inputs.size() == 1) {
                        Arg relu_in = relu->inputs[0];
                        int dq_idx = producer_of.at(relu_in.idx);
                        DequantizeLinearLayer* dq = getLayer<DequantizeLinearLayer>(newprog, dq_idx);

                        const bool relu_out_s8 = !outputs.empty() && netimpl->argData(outputs[0]).type == CV_8S;
                        const bool relu_in_s8 = dq && !dq->inputs.empty() && netimpl->argData(dq->inputs[0]).type == CV_8S;
                        if (dq && dq->inputs.size() >= 3 &&
                            relu_in_s8 && relu_out_s8 &&
                            usecounts.at(relu_in.idx) == 1) {
                            const float inp_sc = netimpl->argTensor(dq->inputs[1]).at<float>(0);
                            const int inp_zp = (int)netimpl->argTensor(dq->inputs[2]).at<int8_t>(0);
                            const float out_sc = netimpl->argTensor(out_scale).at<float>(0);
                            const int out_zp_i = (int)netimpl->argTensor(out_zp).at<int8_t>(0);

                            if (inp_sc > 0.f && out_sc > 0.f) {
                                Mat lookUpTable(1, 256, CV_8S);
                                int8_t* table = lookUpTable.ptr<int8_t>();
                                for (int t = -128; t < 128; t++) {
                                    float x = inp_sc * (t - inp_zp);
                                    float y = std::max(0.0f, x);
                                    int quantized = out_zp_i + cvRound(y / out_sc);
                                    table[t + 128] = saturate_cast<int8_t>(quantized);
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

                Arg out_scale_arg, out_zp_arg;
                int conv_layer_idx = -1;
                ConvolutionLayer* conv = 0;
                if (getQdqPatternContext<ConvolutionLayer>(layer_ptr, ninputs, inputs, producer_of,
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
                            usecounts.at(conv_x.idx) == 1 && usecounts.at(conv_w.idx) == 1) {
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

                                    if (all_wzp_zero && conv->pads_begin == conv->pads_end) {
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

                                        LayerParams convInt8Params = makeLayerParamsFromOriginal(conv, "ConvolutionInt8");
                                        convInt8Params.blobs.resize(3);
                                        convInt8Params.blobs[0] = w_q;
                                        convInt8Params.blobs[1] = biasFused;
                                        convInt8Params.blobs[2] = outputMultiplier;
                                        Ptr<Layer> convInt8 = createFusedLayer(convInt8Params);
                                        if (!convInt8.empty()) {
                                            auto* convInt8Layer = dynamic_cast<ConvolutionLayerInt8*>(convInt8.get());
                                            CV_Assert(convInt8Layer);
                                            const String convInt8Type = static_cast<Layer&>(*convInt8Layer).type;
                                            std::vector<Mat> convInt8Blobs = convInt8Layer->blobs;
                                            static_cast<BaseConvolutionLayer&>(*convInt8Layer) = *conv;
                                            static_cast<Layer&>(*convInt8Layer).type = convInt8Type;
                                            convInt8Layer->blobs = convInt8Blobs;
                                            convInt8Layer->useWinograd = conv->useWinograd;
                                            convInt8Layer->input_zp = inp_zp;
                                            convInt8Layer->input_sc = inp_sc;
                                            convInt8Layer->output_zp = out_zp;
                                            convInt8Layer->output_sc = out_sc;
                                            convInt8Layer->per_channel = per_channel;
                                            fused_layer_idx = conv_layer_idx;
                                            newprog[conv_layer_idx] = convInt8;
                                            fused_inputs.assign(1, dq_x->inputs[0]);
                                            removed_args.push_back(q_data_in);
                                            removed_args.push_back(conv_x);
                                            removed_args.push_back(conv_w);
                                            if (conv->inputs.size() == 3) {
                                                removed_args.push_back(conv->inputs[2]);
                                                if (dq_bias_idx >= 0)
                                                    newprog[dq_bias_idx] = Ptr<Layer>();
                                            }
                                            newprog[dq_x_idx] = Ptr<Layer>();
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
                        const bool fc_out_s8 = !outputs.empty() && netimpl->argData(outputs[0]).type == CV_8S;
                        const bool fc_in_s8 = dq_x && !dq_x->inputs.empty() && netimpl->argData(dq_x->inputs[0]).type == CV_8S;
                        if (dq_x && dq_w &&
                            dq_x->inputs.size() >= 3 && dq_w->inputs.size() >= 3 &&
                            fc_in_s8 && fc_out_s8 &&
                            usecounts.at(mm_x.idx) == 1 && usecounts.at(mm_w.idx) == 1) {
                            inp_sc = netimpl->argTensor(dq_x->inputs[1]).at<float>(0);
                            inp_zp = (int)netimpl->argTensor(dq_x->inputs[2]).at<int8_t>(0);
                            out_sc = netimpl->argTensor(out_scale).at<float>(0);
                            out_zp_i = (int)netimpl->argTensor(out_zp).at<int8_t>(0);
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
                // Fuse QDQ Pooling pattern:
                //   DQ(x_int8) -> Pooling -> QuantizeLinear
                int pool_layer_idx = -1;
                PoolingLayer* pool = 0;
                if (getQdqPatternContext<PoolingLayer>(layer_ptr, ninputs, inputs, producer_of,
                                                       newprog, q_data_in, out_scale, out_zp,
                                                       pool_layer_idx, pool) &&
                    pool->inputs.size() == 1) {
                        Arg pool_in = pool->inputs[0];
                        int dq_idx = producer_of.at(pool_in.idx);
                        DequantizeLinearLayer* dq = getLayer<DequantizeLinearLayer>(newprog, dq_idx);
                        float inp_sc = 0.f, out_sc = 0.f;
                        int inp_zp = 0, out_zp_i = 0;
                        const bool pool_out_s8 = !outputs.empty() && netimpl->argData(outputs[0]).type == CV_8S;
                        const bool pool_in_s8 = dq && !dq->inputs.empty() && netimpl->argData(dq->inputs[0]).type == CV_8S;
                        if (dq && dq->inputs.size() >= 3 &&
                            pool_in_s8 && pool_out_s8 &&
                            usecounts.at(pool_in.idx) == 1) {
                            inp_sc = netimpl->argTensor(dq->inputs[1]).at<float>(0);
                            inp_zp = (int)netimpl->argTensor(dq->inputs[2]).at<int8_t>(0);
                            out_sc = netimpl->argTensor(out_scale).at<float>(0);
                            out_zp_i = (int)netimpl->argTensor(out_zp).at<int8_t>(0);
                            bool isGlobalAve = pool->globalPooling;
                            bool isMax = !isGlobalAve;
                            if ((isGlobalAve && inp_sc > 0.f && out_sc > 0.f) ||
                                (isMax && std::abs(inp_sc - out_sc) < 1e-6f && inp_zp == out_zp_i)) {
                                LayerParams poolInt8Params = makeLayerParamsFromOriginal(pool, "PoolingInt8");
                                poolInt8Params.blobs.clear();
                                Ptr<Layer> poolInt8 = createFusedLayer(poolInt8Params);
                                if (!poolInt8.empty()) {
                                    auto* poolInt8Layer = dynamic_cast<PoolingLayerInt8*>(poolInt8.get());
                                    CV_Assert(poolInt8Layer);
                                    const String poolInt8Type = static_cast<Layer&>(*poolInt8Layer).type;
                                    std::vector<Mat> poolInt8Blobs = poolInt8Layer->blobs;
                                    static_cast<PoolingLayer&>(*poolInt8Layer) = *pool;
                                    static_cast<Layer&>(*poolInt8Layer).type = poolInt8Type;
                                    poolInt8Layer->blobs = poolInt8Blobs;
                                    poolInt8Layer->input_sc = inp_sc;
                                    poolInt8Layer->input_zp = inp_zp;
                                    poolInt8Layer->output_sc = out_sc;
                                    poolInt8Layer->output_zp = out_zp_i;
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
                break;
            }

            if (fused_layer_idx >= 0) {
                modified = true;
                Layer* fused_layer = newprog[fused_layer_idx];
                fused_layer->outputs = outputs;
                if (!fused_inputs.empty())
                    fused_layer->inputs = fused_inputs;
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
