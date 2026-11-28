// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include <opencv2/dnn/shape_utils.hpp>

// ONNX operator: DeformConv
// Spec: https://onnx.ai/onnx/operators/onnx__DeformConv.html
// Supported opsets: 19-22 (2D input only)

namespace cv {
namespace dnn {

// Bilinear sample of one HxW plane at (y, x), as GridSample with align_corners=1 and
// padding_mode=zeros does it: every corner outside the plane counts as 0.
static inline float deformConvSample(const float* plane, int H, int W, float y, float x)
{
    if (y <= -1.f || y >= (float)H || x <= -1.f || x >= (float)W)
        return 0.f;

    int y0 = (int)std::floor(y), x0 = (int)std::floor(x);
    int y1 = y0 + 1, x1 = x0 + 1;
    float ly = y - (float)y0, lx = x - (float)x0;
    float hy = 1.f - ly, hx = 1.f - lx;

    bool y0in = y0 >= 0, y1in = y1 < H, x0in = x0 >= 0, x1in = x1 < W;
    float v00 = (y0in && x0in) ? plane[y0 * W + x0] : 0.f;
    float v01 = (y0in && x1in) ? plane[y0 * W + x1] : 0.f;
    float v10 = (y1in && x0in) ? plane[y1 * W + x0] : 0.f;
    float v11 = (y1in && x1in) ? plane[y1 * W + x1] : 0.f;
    return hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
}

class DeformConvLayerImpl CV_FINAL : public DeformConvLayer
{
public:
    int group = 1;
    int offset_group = 1;
    std::vector<int> kernel_shape;  // empty: take it from W
    std::vector<int> strides, dilations, pads;
    FastGemmOpt gemmOpt;

    // The ONNX importer renames the conv attributes (kernel_shape -> kernel_size, strides -> stride,
    // pads -> pad, dilations -> dilation); the ONNX names are accepted too.
    static std::vector<int> getIntVec(const LayerParams& params, const String& name, const String& onnxName)
    {
        std::vector<int> v;
        const String& key = params.has(name) ? name : onnxName;
        if (params.has(key))
        {
            const DictValue& dv = params.get(key);
            for (int i = 0; i < (int)dv.size(); i++)
                v.push_back(dv.get<int>(i));
        }
        return v;
    }

    DeformConvLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        gemmOpt.init();
        group = params.get<int>("group", 1);
        offset_group = params.get<int>("offset_group", 1);
        kernel_shape = getIntVec(params, "kernel_size", "kernel_shape");
        strides = getIntVec(params, "stride", "strides");
        dilations = getIntVec(params, "dilation", "dilations");
        pads = getIntVec(params, "pad", "pads");

        CV_CheckGT(group, 0, "DeformConv: group must be positive");
        CV_CheckGT(offset_group, 0, "DeformConv: offset_group must be positive");
        if (strides.empty()) strides.assign(2, 1);
        if (dilations.empty()) dilations.assign(2, 1);
        if (pads.empty()) pads.assign(4, 0);
        CV_CheckEQ((int)strides.size(), 2, "DeformConv: only 2D input is supported");
        CV_CheckEQ((int)dilations.size(), 2, "DeformConv: only 2D input is supported");
        CV_CheckEQ((int)pads.size(), 4, "DeformConv: only 2D input is supported");
        if (!kernel_shape.empty())
            CV_CheckEQ((int)kernel_shape.size(), 2, "DeformConv: only 2D input is supported");
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    static bool hasInput(const std::vector<MatShape>& inputs, size_t i)
    {
        return inputs.size() > i && !inputs[i].empty();
    }

    void checkShapes(const MatShape& x, const MatShape& w, const MatShape& off,
                     const MatShape* b, const MatShape* mask, int& kh, int& kw) const
    {
        CV_CheckEQ(x.dims, 4, "DeformConv: only 2D input (N x C x H x W) is supported");
        CV_CheckEQ(w.dims, 4, "DeformConv: W must be oC x C/group x kH x kW");
        CV_CheckEQ(off.dims, 4, "DeformConv: offset must be N x (offset_group*kH*kW*2) x oH x oW");
        kh = kernel_shape.empty() ? w[2] : kernel_shape[0];
        kw = kernel_shape.empty() ? w[3] : kernel_shape[1];
        CV_CheckEQ(kh, w[2], "DeformConv: kernel_shape does not match W");
        CV_CheckEQ(kw, w[3], "DeformConv: kernel_shape does not match W");

        int C = x[1], oC = w[0];
        CV_CheckEQ(C, w[1] * group, "DeformConv: input channels must be W.shape[1] * group");
        CV_CheckEQ(oC % group, 0, "DeformConv: output channels must be divisible by group");
        CV_CheckEQ(C % offset_group, 0, "DeformConv: input channels must be divisible by offset_group");
        CV_CheckEQ(off[0], x[0], "DeformConv: offset batch size does not match X");
        CV_CheckEQ(off[1], offset_group * kh * kw * 2, "DeformConv: offset channels must be offset_group*kH*kW*2");

        int oh = (x[2] + pads[0] + pads[2] - ((kh - 1) * dilations[0] + 1)) / strides[0] + 1;
        int ow = (x[3] + pads[1] + pads[3] - ((kw - 1) * dilations[1] + 1)) / strides[1] + 1;
        CV_CheckEQ(off[2], oh, "DeformConv: offset height does not match the output height");
        CV_CheckEQ(off[3], ow, "DeformConv: offset width does not match the output width");

        if (b)
            CV_CheckEQ((int)b->total(), oC, "DeformConv: B must have oC elements");
        if (mask)
        {
            CV_CheckEQ(mask->dims, 4, "DeformConv: mask must be N x (offset_group*kH*kW) x oH x oW");
            CV_Assert((*mask)[0] == x[0] && (*mask)[1] == offset_group * kh * kw &&
                      (*mask)[2] == oh && (*mask)[3] == ow &&
                      "DeformConv: mask must be N x (offset_group*kH*kW) x oH x oW");
        }
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_CheckGE((int)inputs.size(), 3, "DeformConv: X, W and offset are required");
        int kh, kw;
        MatShape b, mask;
        if (hasInput(inputs, 3)) b = inputs[3];
        if (hasInput(inputs, 4)) mask = inputs[4];
        checkShapes(inputs[0], inputs[1], inputs[2],
                    b.empty() ? nullptr : &b, mask.empty() ? nullptr : &mask, kh, kw);

        const MatShape& x = inputs[0];
        const MatShape& off = inputs[2];
        outputs.assign(requiredOutputs, MatShape({x[0], inputs[1][0], off[2], off[3]}));
        internals.clear();
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckGE((int)inputs.size(), 3, "DeformConv: X, W and offset are required");
        CV_CheckType(inputs[0], inputs[0] == CV_32F, "DeformConv: only float32 is supported");
        CV_CheckTypeEQ(inputs[1], inputs[0], "DeformConv: W must have the same type as X");
        CV_CheckTypeEQ(inputs[2], inputs[0], "DeformConv: offset must have the same type as X");
        outputs.assign(requiredOutputs, inputs[0]);
        internals.assign(requiredInternals, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_CheckGE((int)inputs.size(), 3, "DeformConv: X, W and offset are required");

        Mat X = inputs[0], Wt = inputs[1], Off = inputs[2];
        Mat B = inputs.size() > 3 ? inputs[3] : Mat();
        Mat Mask = inputs.size() > 4 ? inputs[4] : Mat();
        if (!B.empty()) CV_CheckTypeEQ(B.type(), X.type(), "DeformConv: B must have the same type as X");
        if (!Mask.empty()) CV_CheckTypeEQ(Mask.type(), X.type(), "DeformConv: mask must have the same type as X");

        int kh, kw;
        MatShape bshape = B.shape(), mshape = Mask.shape();
        checkShapes(X.shape(), Wt.shape(), Off.shape(),
                    B.empty() ? nullptr : &bshape, Mask.empty() ? nullptr : &mshape, kh, kw);

        X = X.isContinuous() ? X : X.clone();
        Wt = Wt.isContinuous() ? Wt : Wt.clone();
        Off = Off.isContinuous() ? Off : Off.clone();
        if (!B.empty() && !B.isContinuous()) B = B.clone();
        if (!Mask.empty() && !Mask.isContinuous()) Mask = Mask.clone();

        Mat& Y = outputs[0];
        CV_Assert(Y.isContinuous() && Y.type() == CV_32F);

        const int N = X.size[0], C = X.size[1], H = X.size[2], W = X.size[3];
        const int oC = Wt.size[0], oH = Off.size[2], oW = Off.size[3];
        const int K = kh * kw, L = oH * oW;
        const int Cg = C / group, oCg = oC / group, Cog = C / offset_group;
        const int sh = strides[0], sw = strides[1], dh = dilations[0], dw = dilations[1];
        const int padT = pads[0], padL = pads[1];

        const float* xdata = X.ptr<float>();
        const float* wdata = Wt.ptr<float>();
        const float* offdata = Off.ptr<float>();
        const float* bdata = B.empty() ? nullptr : B.ptr<float>();
        const float* mdata = Mask.empty() ? nullptr : Mask.ptr<float>();
        float* ydata = Y.ptr<float>();

        if (Y.total() == 0)
            return;

        // im2col with deformed sampling positions: row c*K + k holds input channel c sampled at
        // kernel tap k for every output position; then each group is one GEMM.
        Mat cols(C * K, L, CV_32F);
        float* cdata = cols.ptr<float>();

        for (int n = 0; n < N; n++)
        {
            const float* xn = xdata + (size_t)n * C * H * W;
            const float* offn = offdata + (size_t)n * offset_group * K * 2 * L;
            const float* mn = mdata ? mdata + (size_t)n * offset_group * K * L : nullptr;

            parallel_for_(Range(0, C * K * oH), [&](const Range& r) {
                for (int idx = r.start; idx < r.end; idx++)
                {
                    int row = idx / oH, i = idx % oH;
                    int c = row / K, k = row % K;
                    int ki = k / kw, kj = k % kw;
                    int og = c / Cog;

                    const float* plane = xn + (size_t)c * H * W;
                    const float* offY = offn + ((size_t)og * K * 2 + (size_t)k * 2) * L + (size_t)i * oW;
                    const float* offX = offY + L;
                    const float* mrow = mn ? mn + ((size_t)og * K + k) * L + (size_t)i * oW : nullptr;
                    float* crow = cdata + (size_t)row * L + (size_t)i * oW;

                    float ybase = (float)(i * sh - padT + ki * dh);
                    for (int j = 0; j < oW; j++)
                    {
                        float xbase = (float)(j * sw - padL + kj * dw);
                        float v = deformConvSample(plane, H, W, ybase + offY[j], xbase + offX[j]);
                        crow[j] = mrow ? v * mrow[j] : v;
                    }
                }
            });

            // Y_g [oCg x L] = W_g [oCg x Cg*K] * cols_g [Cg*K x L]
            const int Kg = Cg * K;
            for (int g = 0; g < group; g++)
            {
                const float* wg = wdata + (size_t)g * oCg * Kg;
                const float* cg = cdata + (size_t)g * Kg * L;
                float* yg = ydata + ((size_t)n * oC + (size_t)g * oCg) * L;
                fastGemm(false, false, oCg, Kg, Kg, L, 1.f, wg, Kg, 1, cg, L, 1, 0.f, yg, L, gemmOpt);
            }

            if (bdata)
            {
                for (int oc = 0; oc < oC; oc++)
                {
                    float* yrow = ydata + ((size_t)n * oC + oc) * L;
                    float bv = bdata[oc];
                    for (int l = 0; l < L; l++)
                        yrow[l] += bv;
                }
            }
        }
    }
};

Ptr<DeformConvLayer> DeformConvLayer::create(const LayerParams& params)
{
    return Ptr<DeformConvLayer>(new DeformConvLayerImpl(params));
}

}}  // namespace cv::dnn
