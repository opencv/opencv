// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

namespace cv {
namespace dnn {

/*
    Implementation of BatchNormalization, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__BatchNormalization.html

    Opset's 1 to 15 are covered.
*/

#undef CV_SIMD_ONLY
#if CV_SIMD || CV_SIMD_SCALABLE
#define CV_SIMD_ONLY(expr) expr
#else
#define CV_SIMD_ONLY(expr)
#endif

// out must be pre-allocated
static void batchnorm(const Mat& inp, Mat& out, const Mat& scale,
                      const Mat& bias, DataLayout defaultLayout)
{
    CV_Assert_N(inp.isContinuous(), out.isContinuous());
    CV_Assert_N(scale.dims == 1, bias.dims == 1);
    CV_Assert_N(scale.type() == CV_32F, bias.type() == CV_32F);

    MatShape shape = inp.shape();
    DataLayout layout = shape.layout;
    if (layout == DATA_LAYOUT_UNKNOWN)
        layout = defaultLayout;
    CV_Assert(layout == DATA_LAYOUT_BLOCK ||
              layout == DATA_LAYOUT_NCHW ||
              layout == DATA_LAYOUT_NHWC);
    int N = shape[0];
    int C_ = layout == DATA_LAYOUT_BLOCK ? shape.C :
             layout == DATA_LAYOUT_NCHW ? shape[1] : shape[shape.dims-1];
    CV_Assert_N(scale.cols == C_, bias.cols == C_);
    int C1_ = layout != DATA_LAYOUT_NHWC ? shape[1] : 1;
    int C0_ = layout != DATA_LAYOUT_NCHW ? shape[shape.dims-1] : 1;
    int type_ = inp.type();
    CV_SIMD_ONLY(int vlanes_ = VTraits<v_float32>::vlanes());

    size_t esz = inp.elemSize();

    CV_Assert(type_ == CV_32F || type_ == CV_16F || type_ == CV_16BF);

    CV_Assert(inp.shape() == out.shape());
    CV_Assert(inp.isContinuous());
    CV_Assert(out.isContinuous());

    int planesize_ = 1;
    for (int i = 1; i < shape.dims; i++) {
        planesize_ *= shape[i];
    }
    planesize_ /= (C0_*C1_);

    parallel_for_(Range(0, N*C1_), [&](const Range& r) {
        int C0 = C0_, C1 = C1_, C = C_;
        int planesize_C0 = planesize_*C0;
        int type = type_;
        CV_SIMD_ONLY(int vlanes = vlanes_);
        constexpr int max_lanes = VTraits<v_float32>::max_nlanes;
        constexpr int MAX_UNROLL = 4;
        float scalebuf[max_lanes*MAX_UNROLL], biasbuf[max_lanes*MAX_UNROLL];

        for (int k = r.start; k < r.end; k++) {
            int i = 0, n = k/C1, c1 = k % C1;
            int c_start = c1*C0, c_end = std::min(C, c_start + C0), c_delta = c_end - c_start;
            size_t ofs_k = (n*C1 + c1)*planesize_C0*esz;
            const uchar* inptr_ = inp.data + ofs_k;
            uchar* outptr_ = out.data + ofs_k;
            const float* scaleptr = scale.ptr<float>() + c_start;
            const float* biasptr = bias.ptr<float>() + c_start;

            if (C0 == 1) {
                // NCHW case
                float scale_c = 0.f, bias_c = 0.f;
                if (c_start < c_end) {
                    scale_c = scaleptr[0];
                    bias_c = biasptr[0];
                }
                CV_SIMD_ONLY(v_float32 vscale_c = vx_setall_f32(scale_c);
                             v_float32 vbias_c = vx_setall_f32(bias_c));
                if (type == CV_32F) {
                    const float* inptr = (const float*)inptr_;
                    float* outptr = (float*)outptr_;
                    CV_SIMD_ONLY(for (; i <= planesize_C0 - vlanes*2; i += vlanes*2) {
                        v_float32 x0 = vx_load(inptr + i);
                        v_float32 x1 = vx_load(inptr + i + vlanes);
                        x0 = v_fma(x0, vscale_c, vbias_c);
                        x1 = v_fma(x1, vscale_c, vbias_c);
                        v_store(outptr + i, x0);
                        v_store(outptr + i + vlanes, x1);
                    });
                    for (; i < planesize_C0; i++) {
                        outptr[i] = inptr[i]*scale_c + bias_c;
                    }
                } else if (type == CV_16F) {
                    const hfloat* inptr = (const hfloat*)inptr_;
                    hfloat* outptr = (hfloat*)outptr_;
                    CV_SIMD_ONLY(for (; i <= planesize_C0 - vlanes*2; i += vlanes*2) {
                        v_float32 x0 = vx_load_expand(inptr + i);
                        v_float32 x1 = vx_load_expand(inptr + i + vlanes);
                        x0 = v_fma(x0, vscale_c, vbias_c);
                        x1 = v_fma(x1, vscale_c, vbias_c);
                        v_pack_store(outptr + i, x0);
                        v_pack_store(outptr + i + vlanes, x1);
                    });
                    for (; i < planesize_C0; i++) {
                        outptr[i] = hfloat(float(inptr[i])*scale_c + bias_c);
                    }
                } else if (type == CV_16BF) {
                    const bfloat* inptr = (const bfloat*)inptr_;
                    bfloat* outptr = (bfloat*)outptr_;
                    CV_SIMD_ONLY(for (; i <= planesize_C0 - vlanes*2; i += vlanes*2) {
                        v_float32 x0 = vx_load_expand(inptr + i);
                        v_float32 x1 = vx_load_expand(inptr + i + vlanes);
                        x0 = v_fma(x0, vscale_c, vbias_c);
                        x1 = v_fma(x1, vscale_c, vbias_c);
                        v_pack_store(outptr + i, x0);
                        v_pack_store(outptr + i + vlanes, x1);
                    });
                    for (; i < planesize_C0; i++) {
                        outptr[i] = bfloat(float(inptr[i])*scale_c + bias_c);
                    }
                }
            }
        #if CV_SIMD || CV_SIMD_SCALABLE
            /*
                [TODO] support C0 == vlanes/2, maybe C0 == vlanes/4.
                in this case, load everything into vsc0 and vb0, process
                most part of the plane using vector code as if C0 == vlanes and
                then process the tail using scalar code
            */
            else if (C0 == vlanes*4 || C0 == vlanes*2 || C0 == vlanes) {
                // accelerated block layout case
                int c = 0;
                for (; c < c_delta; c++) {
                    scalebuf[c] = scaleptr[c];
                    biasbuf[c] = biasptr[c];
                }
                for (; c < MAX_UNROLL*vlanes; c++) {
                    scalebuf[c] = biasbuf[c] = 0.f;
                }
                v_float32 vsc0, vsc1, vsc2, vsc3;
                v_float32 vb0, vb1, vb2, vb3, vb4;
                vsc0 = vx_load(scalebuf);
                vsc1 = vx_load(scalebuf + vlanes);
                vsc2 = vx_load(scalebuf + vlanes*2);
                vsc3 = vx_load(scalebuf + vlanes*3);
                vb0 = vx_load(biasbuf);
                vb1 = vx_load(biasbuf + vlanes);
                vb2 = vx_load(biasbuf + vlanes*2);
                vb3 = vx_load(biasbuf + vlanes*3);
                if (type == CV_32F) {
                    const float* inptr = (const float*)inptr_;
                    float* outptr = (float*)outptr_;
                    if (C0 == vlanes*4) {
                        for (; i < planesize_C0; i += C0) {
                            v_float32 x0 = vx_load(inptr + i);
                            v_float32 x1 = vx_load(inptr + i + vlanes);
                            v_float32 x2 = vx_load(inptr + i + vlanes*2);
                            v_float32 x3 = vx_load(inptr + i + vlanes*3);
                            x0 = v_fma(x0, vsc0, vb0);
                            x1 = v_fma(x1, vsc1, vb1);
                            x2 = v_fma(x2, vsc2, vb2);
                            x3 = v_fma(x3, vsc3, vb3);
                            v_store(outptr + i, x0);
                            v_store(outptr + i + vlanes, x1);
                            v_store(outptr + i + vlanes*2, x2);
                            v_store(outptr + i + vlanes*3, x3);
                        }
                    } else if (C0 == vlanes*2) {
                        for (; i < planesize_C0; i += C0) {
                            v_float32 x0 = vx_load(inptr + i);
                            v_float32 x1 = vx_load(inptr + i + vlanes);
                            x0 = v_fma(x0, vsc0, vb0);
                            x1 = v_fma(x1, vsc1, vb1);
                            v_store(outptr + i, x0);
                            v_store(outptr + i + vlanes, x1);
                        }
                    } else {
                        for (; i < planesize_C0; i += C0) {
                            v_float32 x0 = vx_load(inptr + i);
                            x0 = v_fma(x0, vsc0, vb0);
                            v_store(outptr + i, x0);
                        }
                    }
                } else if (type == CV_16F) {
                    const hfloat* inptr = (const hfloat*)inptr_;
                    hfloat* outptr = (hfloat*)outptr_;
                    if (type == CV_32F) {
                        if (C0 == vlanes*4) {
                            for (; i < planesize_C0; i += C0) {
                                v_float32 x0 = vx_load_expand(inptr + i);
                                v_float32 x1 = vx_load_expand(inptr + i + vlanes);
                                v_float32 x2 = vx_load_expand(inptr + i + vlanes*2);
                                v_float32 x3 = vx_load_expand(inptr + i + vlanes*3);
                                x0 = v_fma(x0, vsc0, vb0);
                                x1 = v_fma(x1, vsc1, vb1);
                                x2 = v_fma(x2, vsc2, vb2);
                                x3 = v_fma(x3, vsc3, vb3);
                                v_pack_store(outptr + i, x0);
                                v_pack_store(outptr + i + vlanes, x1);
                                v_pack_store(outptr + i + vlanes*2, x2);
                                v_pack_store(outptr + i + vlanes*3, x3);
                            }
                        } else if (C0 == vlanes*2) {
                            for (; i < planesize_C0; i += C0) {
                                v_float32 x0 = vx_load_expand(inptr + i);
                                v_float32 x1 = vx_load_expand(inptr + i + vlanes);
                                x0 = v_fma(x0, vsc0, vb0);
                                x1 = v_fma(x1, vsc1, vb1);
                                v_pack_store(outptr + i, x0);
                                v_pack_store(outptr + i + vlanes, x1);
                            }
                        } else {
                            for (; i < planesize_C0; i += C0) {
                                v_float32 x0 = vx_load_expand(inptr + i);
                                x0 = v_fma(x0, vsc0, vb0);
                                v_pack_store(outptr + i, x0);
                            }
                        }
                    }
                } else if (type == CV_16BF) {
                    const bfloat* inptr = (const bfloat*)inptr_;
                    bfloat* outptr = (bfloat*)outptr_;
                    if (C0 == vlanes*4) {
                        for (; i < planesize_C0; i += C0) {
                            v_float32 x0 = vx_load_expand(inptr + i);
                            v_float32 x1 = vx_load_expand(inptr + i + vlanes);
                            v_float32 x2 = vx_load_expand(inptr + i + vlanes*2);
                            v_float32 x3 = vx_load_expand(inptr + i + vlanes*3);
                            x0 = v_fma(x0, vsc0, vb0);
                            x1 = v_fma(x1, vsc1, vb1);
                            x2 = v_fma(x2, vsc2, vb2);
                            x3 = v_fma(x3, vsc3, vb3);
                            v_pack_store(outptr + i, x0);
                            v_pack_store(outptr + i + vlanes, x1);
                            v_pack_store(outptr + i + vlanes*2, x2);
                            v_pack_store(outptr + i + vlanes*3, x3);
                        }
                    } else if (C0 == vlanes*2) {
                        for (; i < planesize_C0; i += C0) {
                            v_float32 x0 = vx_load_expand(inptr + i);
                            v_float32 x1 = vx_load_expand(inptr + i + vlanes);
                            x0 = v_fma(x0, vsc0, vb0);
                            x1 = v_fma(x1, vsc1, vb1);
                            v_pack_store(outptr + i, x0);
                            v_pack_store(outptr + i + vlanes, x1);
                        }
                    } else {
                        for (; i < planesize_C0; i += C0) {
                            v_float32 x0 = vx_load_expand(inptr + i);
                            x0 = v_fma(x0, vsc0, vb0);
                            v_pack_store(outptr + i, x0);
                        }
                    }
                }
            }
        #endif
            else {
                // general block layout or NHWC case
                if (type == CV_32F) {
                    const float* inptr = (const float*)inptr_;
                    float* outptr = (float*)outptr_;
                    for (; i < planesize_C0; i += C0) {
                        int c = 0;
                        CV_SIMD_ONLY(for (; c <= c_delta - vlanes*2; c += vlanes*2) {
                            v_float32 x0 = vx_load(inptr + i + c);
                            v_float32 x1 = vx_load(inptr + i + c + vlanes);
                            v_float32 sc0 = vx_load(scaleptr + c);
                            v_float32 sc1 = vx_load(scaleptr + c + vlanes);
                            v_float32 b0 = vx_load(biasptr + c);
                            v_float32 b1 = vx_load(biasptr + c + vlanes);
                            x0 = v_fma(x0, sc0, b0);
                            x1 = v_fma(x1, sc1, b1);
                            v_store(outptr + i + c, x0);
                            v_store(outptr + i + c + vlanes, x1);
                        });
                        for (; c < c_delta; c++)
                            outptr[i + c] = inptr[i + c]*scaleptr[c] + biasptr[c];
                        for (; c < C0; c++)
                            outptr[i + c] = 0.f;
                    }
                } else if (type == CV_16F) {
                    const hfloat* inptr = (const hfloat*)inptr_;
                    hfloat* outptr = (hfloat*)outptr_;
                    hfloat z = hfloat(0.f);
                    for (; i < planesize_C0; i += C0) {
                        int c = 0;
                        CV_SIMD_ONLY(for (; c <= c_delta - vlanes*2; c += vlanes*2) {
                            v_float32 x0 = vx_load_expand(inptr + i + c);
                            v_float32 x1 = vx_load_expand(inptr + i + c + vlanes);
                            v_float32 sc0 = vx_load(scaleptr + c);
                            v_float32 sc1 = vx_load(scaleptr + c + vlanes);
                            v_float32 b0 = vx_load(biasptr + c);
                            v_float32 b1 = vx_load(biasptr + c + vlanes);
                            x0 = v_fma(x0, sc0, b0);
                            x1 = v_fma(x1, sc1, b1);
                            v_pack_store(outptr + i + c, x0);
                            v_pack_store(outptr + i + c + vlanes, x1);
                        });
                        for (; c < c_delta; c++)
                            outptr[i + c] = hfloat(float(inptr[i + c])*scaleptr[c] + biasptr[c]);
                        for (; c < C0; c++)
                            outptr[i + c] = z;
                    }
                } else if (type == CV_16BF) {
                    const bfloat* inptr = (const bfloat*)inptr_;
                    bfloat* outptr = (bfloat*)outptr_;
                    bfloat z = bfloat(0.f);
                    for (; i < planesize_C0; i += C0) {
                        int c = 0;
                        CV_SIMD_ONLY(for (; c <= c_delta - vlanes*2; c += vlanes*2) {
                            v_float32 x0 = vx_load_expand(inptr + i + c);
                            v_float32 x1 = vx_load_expand(inptr + i + c + vlanes);
                            v_float32 sc0 = vx_load(scaleptr + c);
                            v_float32 sc1 = vx_load(scaleptr + c + vlanes);
                            v_float32 b0 = vx_load(biasptr + c);
                            v_float32 b1 = vx_load(biasptr + c + vlanes);
                            x0 = v_fma(x0, sc0, b0);
                            x1 = v_fma(x1, sc1, b1);
                            v_pack_store(outptr + i + c, x0);
                            v_pack_store(outptr + i + c + vlanes, x1);
                        });
                        for (; c < c_delta; c++)
                            outptr[i + c] = bfloat(float(inptr[i + c])*scaleptr[c] + biasptr[c]);
                        for (; c < C0; c++)
                            outptr[i + c] = z;
                    }
                }
            }
        }
    }, (planesize_*C0_ > 1000000 ? N*C1_ : 1));
}

class BatchNorm2LayerImpl CV_FINAL : public BatchNorm2Layer
{
public:
    BatchNorm2LayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        epsilon = params.get<float>("epsilon", 1e-5);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const MatShape& inpShape) const
    {
        return inpShape;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(1, getOutShape(inputs[0]));
        internals.clear();
        return true;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(requiredOutputs, inputs[0]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                    std::vector<DataLayout>& desiredInputs,
                    const int requiredOutputs,
                    std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        size_t ninputs = actualInputs.size();
        CV_Assert(ninputs >= 1u);
        desiredInputs.assign(ninputs, DATA_LAYOUT_UNKNOWN);
        desiredInputs[0] = actualInputs[0];
        outputs.assign(requiredOutputs, actualInputs[0]);
        return 0;
    }

    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    virtual void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        size_t ninputs = inputs_arr.total(-1);
        CV_Assert(ninputs > 0);

        if (ninputs > 1) {
            CV_Assert(ninputs == 5);
            Mat scale_ = inputs_arr.getMat(1);
            Mat bias_ = inputs_arr.getMat(2);
            Mat mean_ = inputs_arr.getMat(3);
            Mat var_ = inputs_arr.getMat(4);
            BatchNorm2Layer::getScaleBias(scale_, bias_, mean_, var_, epsilon, scale, bias);
        }

        MatShape inpShape = inputs_arr.shape(0);
        int inpType = inputs_arr.type(0);

        MatShape outShape = getOutShape(inpShape);
        int outKind = outputs_arr.kind();

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runOp(inp, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            Mat temp(outShape, inpType);
            runOp(inp, temp);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& inp, Mat& out)
    {
        auto netimpl_ = getNetImpl(this);
        batchnorm(inp, out, scale, bias, netimpl_->originalLayout);
    }

    virtual bool freezeScaleBias() CV_OVERRIDE
    {
        auto netimpl_ = getNetImpl(this);
        size_t ninputs = inputs.size();
        if (ninputs != 5)
            return false;
        if (!netimpl_->isConstArg(inputs[1]) ||
            !netimpl_->isConstArg(inputs[2]) ||
            !netimpl_->isConstArg(inputs[3]) ||
            !netimpl_->isConstArg(inputs[4]))
            return false;
        Mat scale_ = netimpl_->argTensor(inputs[1]);
        Mat bias_ = netimpl_->argTensor(inputs[2]);
        Mat mean_ = netimpl_->argTensor(inputs[3]);
        Mat var_ = netimpl_->argTensor(inputs[4]);
        BatchNorm2Layer::getScaleBias(scale_, bias_, mean_, var_, epsilon, scale, bias);
        inputs.resize(1);
        return true;
    }

    virtual void getScaleBias(OutputArray scale_, OutputArray bias_) const CV_OVERRIDE
    {
        scale.copyTo(scale_);
        bias.copyTo(bias_);
    }

    Mat scale, bias;
};

void BatchNorm2Layer::getScaleBias(InputArray scale_, InputArray bias_,
                                   InputArray mean_, InputArray variance_, float eps,
                                   OutputArray outscale_, OutputArray outbias_)
{
    Mat scale = scale_.getMat(), bias = bias_.getMat();
    Mat mean = mean_.getMat(), var = variance_.getMat();

    int sctype = scale.type(), btype = bias.type();
    int mtype = mean.type(), vtype = var.type();

    CV_Assert(sctype == CV_32F || sctype == CV_16F || sctype == CV_16BF);
    CV_Assert(btype == CV_32F || btype == CV_16F || btype == CV_16BF);
    CV_Assert(mtype == CV_32F || mtype == CV_16F || mtype == CV_16BF);
    CV_Assert(vtype == CV_32F || vtype == CV_16F || vtype == CV_16BF);

    CV_Assert_N(scale.dims == 1, bias.dims == 1, mean.dims == 1, var.dims == 1);
    int C = scale.cols;
    CV_Assert_N(bias.cols == C, mean.cols == C, var.cols == C);

    Mat outscale(1, &C, CV_32F);
    Mat outbias(1, &C, CV_32F);

    const uchar* scdata = scale.data;
    const uchar* bdata = bias.data;
    const uchar* mdata = mean.data;
    const uchar* vdata = var.data;
    float* outsc = outscale.ptr<float>();
    float* outb = outbias.ptr<float>();

    #undef LOAD_AS_FLOAT
    #define LOAD_AS_FLOAT(typ, ptr, i) \
        (typ == CV_32F ? ((const float*)ptr)[i] : \
         typ == CV_16F ? float(((const hfloat*)ptr)[i]) : \
         float(((const bfloat*)ptr)[i]))

    // ONNX documentation: Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
    for (int i = 0; i < C; i++) {
        float sc = LOAD_AS_FLOAT(sctype, scdata, i);
        float b = LOAD_AS_FLOAT(btype, bdata, i);
        float m = LOAD_AS_FLOAT(mtype, mdata, i);
        float v = LOAD_AS_FLOAT(vtype, vdata, i);

        float outscval = sc/sqrtf(fabsf(v) + eps);
        float outbval = b - m*outscval;
        outsc[i] = outscval;
        outb[i] = outbval;
    }
    outscale.copyTo(outscale_);
    outbias.copyTo(outbias_);
}

Ptr<BatchNorm2Layer> BatchNorm2Layer::create(const LayerParams& params)
{
    return makePtr<BatchNorm2LayerImpl>(params);
}
}} // namespace cv::dnn
