// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "../layers/conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {

template <typename T>
static void maxPoolImpl(const void* inp_, void* out_, const ConvState& cs)
{
    constexpr int MAX_POOL_DIMS = ConvState::MAX_CONV_DIMS;
    int NC1 = cs.inpshape[0] * cs.inpshape[1];

    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.dims == cs.outshape.dims);

    parallel_for_(Range(0, NC1), [&](const Range& r) {
        CV_Assert(cs.nspatialdims <= MAX_POOL_DIMS && MAX_POOL_DIMS == 3);

        int sdims = cs.nspatialdims;
        int nc0 = r.start, nc1 = r.end;
        int C0 = cs.inpshape.back();
        int Di = sdims > 2 ? cs.inpshape[sdims - 1] : 1;
        int Hi = sdims > 1 ? cs.inpshape[sdims] : 1;
        int Wi = cs.inpshape[sdims + 1];
        int D = sdims > 2 ? cs.outshape[sdims - 1] : 1;
        int H = sdims > 1 ? cs.outshape[sdims] : 1;
        int W = cs.outshape[sdims + 1];
        int iplanesize = Di * Hi * Wi * C0;
        int planesize = D * H * W * C0;
        int SZ = cs.strides[0], SY = cs.strides[1], SX = cs.strides[2];
        int padZ0 = cs.pads[0], padY0 = cs.pads[1], padX0 = cs.pads[2];
        int inner_z0 = cs.inner[0], inner_z1 = cs.inner[MAX_POOL_DIMS];
        int inner_y0 = cs.inner[1], inner_y1 = cs.inner[MAX_POOL_DIMS + 1];
        int inner_x0 = cs.inner[2], inner_x1 = cs.inner[MAX_POOL_DIMS + 2];
        int ksize = (int)cs.ofstab.size();
        const int* zyxtab = cs.coordtab.data();
        const int* ofstab = cs.ofstab.data();

        const T* inp = (const T*)inp_ + nc0 * iplanesize;
        T* out = (T*)out_ + nc0 * planesize;
        const T INITVAL = std::numeric_limits<T>::min();

        for (int nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            for (int z0 = 0; z0 < D; z0++) {
                int zi_ = z0 * SZ - padZ0;
                for (int y0 = 0; y0 < H; y0++, out += W * C0) {
                    int x0 = 0;
                    int x1 = z0 >= inner_z0 && z0 < inner_z1 &&
                             y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                    int yi_ = y0 * SY - padY0;

                    for (;;) {
                        for (; x0 < x1; x0++) {
                            int xi_ = x0 * SX - padX0;
                            for (int c = 0; c < C0; c++)
                                out[x0 * C0 + c] = INITVAL;

                            for (int k = 0; k < ksize; k++) {
                                int zi = zi_ + zyxtab[k * MAX_POOL_DIMS];
                                int yi = yi_ + zyxtab[k * MAX_POOL_DIMS + 1];
                                int xi = xi_ + zyxtab[k * MAX_POOL_DIMS + 2];
                                if ((unsigned)zi >= (unsigned)Di ||
                                    (unsigned)yi >= (unsigned)Hi ||
                                    (unsigned)xi >= (unsigned)Wi)
                                    continue;
                                const T* inptr = inp + ((zi * Hi + yi) * Wi + xi) * C0;
                                for (int c = 0; c < C0; c++)
                                    out[x0 * C0 + c] = std::max(out[x0 * C0 + c], inptr[c]);
                            }
                        }

                        if (x0 == W)
                            break;
                        x1 = inner_x1;

                        for (; x0 < x1; x0++) {
                            int xi_ = x0 * SX - padX0;
                            const T* inp_xi = inp + ((Hi * zi_ + yi_) * Wi + xi_) * C0;

                            for (int c = 0; c < C0; c++) {
                                T s = inp_xi[ofstab[0] + c];
                                for (int k = 1; k < ksize; k++)
                                    s = std::max(s, inp_xi[ofstab[k] + c]);
                                out[x0 * C0 + c] = s;
                            }
                        }

                        x1 = W;
                    }
                }
            }
        }
    });
}

static void maxPoolInt8(const void* inp_, void* out_, const ConvState& cs, bool isU8)
{
    if (isU8)
        maxPoolImpl<uint8_t>(inp_, out_, cs);
    else
        maxPoolImpl<int8_t>(inp_, out_, cs);
}

static void avgPoolInt8(const void* inp_, void* out_, const ConvState& cs,
                        float inp_sc, int inp_zp, float out_sc, int out_zp,
                        bool count_include_pad_, bool isU8)
{
    constexpr int MAX_POOL_DIMS = ConvState::MAX_CONV_DIMS;
    int NC1 = cs.inpshape[0] * cs.inpshape[1];

    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.dims == cs.outshape.dims);

    parallel_for_(Range(0, NC1), [&](const Range& r) {
        CV_Assert(cs.nspatialdims <= MAX_POOL_DIMS && MAX_POOL_DIMS == 3);

        bool count_include_pad = count_include_pad_;
        int sdims = cs.nspatialdims;
        int nc0 = r.start, nc1 = r.end;
        int C0 = cs.inpshape.back();
        int Di = sdims > 2 ? cs.inpshape[sdims - 1] : 1;
        int Hi = sdims > 1 ? cs.inpshape[sdims] : 1;
        int Wi = cs.inpshape[sdims + 1];
        int D = sdims > 2 ? cs.outshape[sdims - 1] : 1;
        int H = sdims > 1 ? cs.outshape[sdims] : 1;
        int W = cs.outshape[sdims + 1];
        int iplanesize = Di * Hi * Wi * C0;
        int planesize = D * H * W * C0;
        int SZ = cs.strides[0], SY = cs.strides[1], SX = cs.strides[2];
        int padZ0 = cs.pads[0], padY0 = cs.pads[1], padX0 = cs.pads[2];
        int ksize_total = (int)cs.ofstab.size();
        const int* zyxtab = cs.coordtab.data();

        const uint8_t* inp = (const uint8_t*)inp_ + nc0 * iplanesize;
        uint8_t* out = (uint8_t*)out_ + nc0 * planesize;

        float scale_ratio = inp_sc / out_sc;

        std::vector<int> accum(C0);

        for (int nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            for (int z0 = 0; z0 < D; z0++) {
                int zi_ = z0 * SZ - padZ0;
                for (int y0 = 0; y0 < H; y0++, out += W * C0) {
                    int yi_ = y0 * SY - padY0;

                    for (int x0 = 0; x0 < W; x0++) {
                        int xi_ = x0 * SX - padX0;

                        for (int c = 0; c < C0; c++)
                            accum[c] = 0;

                        int count = 0;
                        for (int k = 0; k < ksize_total; k++) {
                            int zi = zi_ + zyxtab[k * MAX_POOL_DIMS];
                            int yi = yi_ + zyxtab[k * MAX_POOL_DIMS + 1];
                            int xi = xi_ + zyxtab[k * MAX_POOL_DIMS + 2];

                            if ((unsigned)zi >= (unsigned)Di ||
                                (unsigned)yi >= (unsigned)Hi ||
                                (unsigned)xi >= (unsigned)Wi) {
                                if (count_include_pad)
                                    count++;
                                continue;
                            }

                            count++;
                            const uint8_t* inptr = inp + ((zi * Hi + yi) * Wi + xi) * C0;
                            for (int c = 0; c < C0; c++) {
                                int v = isU8 ? (int)inptr[c]
                                             : (int)(int8_t)inptr[c];
                                accum[c] += v - inp_zp;
                            }
                        }

                        if (count == 0) count = 1;
                        float inv_count = 1.f / count;

                        for (int c = 0; c < C0; c++) {
                            float val = (float)accum[c] * inv_count * scale_ratio + (float)out_zp;
                            int ival = cvRound(val);
                            if (isU8)
                                out[x0 * C0 + c] = (uint8_t)std::max(0, std::min(255, ival));
                            else
                                out[x0 * C0 + c] = (uint8_t)(int8_t)std::max(-128, std::min(127, ival));
                        }
                    }
                }
            }
        }
    });
}

static void globalAvgPoolInt8(const void* inp_, void* out_,
                              int N, int C1, int spatialSize, int C0,
                              float inp_sc, int inp_zp, float out_sc, int out_zp,
                              bool isU8)
{
    int NC1 = N * C1;
    float scale_ratio = inp_sc / out_sc;

    parallel_for_(Range(0, NC1), [&](const Range& r) {
        for (int nc = r.start; nc < r.end; nc++) {
            const uint8_t* inptr = (const uint8_t*)inp_ + (size_t)nc * spatialSize * C0;
            uint8_t* outptr = (uint8_t*)out_ + (size_t)nc * C0;

            for (int c = 0; c < C0; c++) {
                int sum = 0;
                for (int s = 0; s < spatialSize; s++) {
                    int v = isU8 ? (int)inptr[s * C0 + c]
                                 : (int)(int8_t)inptr[s * C0 + c];
                    sum += v - inp_zp;
                }

                float val = (float)sum / spatialSize * scale_ratio + (float)out_zp;
                int ival = cvRound(val);
                if (isU8)
                    outptr[c] = (uint8_t)std::max(0, std::min(255, ival));
                else
                    outptr[c] = (uint8_t)(int8_t)std::max(-128, std::min(127, ival));
            }
        }
    });
}

class Pool2Int8LayerImpl CV_FINAL : public Pool2Int8Layer
{
public:
    bool count_include_pad;
    ConvState cs;
    MatShape prevInpshape;

    Pool2Int8LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        kernel_shape = params.getVector<int>("kernel_size");
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        ceil_mode = params.get<bool>("ceil_mode", false);
        is_global_pooling = params.get<bool>("global_pooling", false);
        is_max_pool = params.get<bool>("is_max_pool", true);
        count_include_pad = params.get<bool>("count_include_pad", false);

        input_sc = params.get<float>("input_scale", 1.f);
        input_zp = params.get<int>("input_zeropoint", 0);
        output_sc = params.get<float>("scales", 1.f);
        output_zp = params.get<int>("zeropoints", 0);
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs, const int,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                         const int,
                         std::vector<MatShape>& outshapes,
                         std::vector<MatShape>& tempshapes) const CV_OVERRIDE
    {
        CV_Assert(inpshapes.size() == 1);

        if (is_global_pooling) {
            MatShape outshape = inpshapes[0];
            for (int d = 2; d < outshape.dims - 1; d++)
                outshape[d] = 1;
            outshapes.assign(1, outshape);
        } else {
            outshapes.assign(1, convInferShape(inpshapes[0], MatShape(),
                                               kernel_shape, 0, strides, dilations,
                                               pads, auto_pad, ceil_mode));
        }
        tempshapes.clear();
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        CV_Assert(actualInputs.size() == 1u);
        desiredInputs.assign(1, DATA_LAYOUT_BLOCK);
        outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        return getNetImpl(this)->defaultC0;
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        size_t ninputs = inputs_arr.total();
        CV_Assert(ninputs == 1);

        const Mat& inp = inputs_arr.getMat(0);
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inptype == CV_8SC1 || inptype == CV_8UC1);

        MatShape outshape;
        if (is_global_pooling) {
            outshape = inpshape;
            for (int d = 2; d < outshape.dims - 1; d++)
                outshape[d] = 1;
        } else {
            outshape = convInferShape(inpshape, MatShape(),
                                      kernel_shape, 0, strides, dilations,
                                      pads, auto_pad, ceil_mode);
        }

        int outkind = outputs_arr.kind();
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            out = outs[0];
        } else {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            out.fit(outshape, inptype);
        }

        if (is_global_pooling) {
            int N = inpshape[0];
            int C1 = inpshape[1];
            int C0 = inpshape.back();
            int spatialSize = 1;
            for (int d = 2; d < inpshape.dims - 1; d++)
                spatialSize *= inpshape[d];

            globalAvgPoolInt8(inp.data, out.data, N, C1, spatialSize, C0,
                              input_sc, input_zp, output_sc, output_zp,
                              inptype == CV_8UC1);
        } else {
            ConvState cs_local;
            cs_local.initPooling(inpshape, outshape, kernel_shape, strides,
                                 dilations, pads, auto_pad, ceil_mode);

            if (is_max_pool) {
                maxPoolInt8(inp.data, out.data, cs_local, inptype == CV_8UC1);
            } else {
                avgPoolInt8(inp.data, out.data, cs_local,
                            input_sc, input_zp, output_sc, output_zp,
                            count_include_pad, inptype == CV_8UC1);
            }
        }

        if (outkind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            out.copyTo(outs[0]);
        }
    }
};

Ptr<Pool2Int8Layer> Pool2Int8Layer::create(const LayerParams& params)
{
    return Ptr<Pool2Int8Layer>(new Pool2Int8LayerImpl(params));
}

} // namespace dnn
} // namespace cv
