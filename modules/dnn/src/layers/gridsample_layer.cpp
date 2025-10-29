// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/utility.hpp>

// ONNX operator: GridSample
// Spec: https://onnx.ai/onnx/operators/onnx__GridSample.html
// Supported opsets: 16-22

namespace cv {
namespace dnn {

static inline float reflectCoord(float x, int limit, bool align_corners) {
    if (limit <= 1) return 0.f;

    const float minv = align_corners ? 0.f : -0.5f;
    const float maxv = align_corners ? float(limit - 1) : float(limit) - 0.5f;
    const float m = maxv - minv;
    const float two_m = 2.f * m;

    float t = fmodf(x - minv, two_m);
    if (t < 0) t += two_m;
    if (t > m) t = two_m - t;
    return t + minv;
}

static inline void getCubicCoeffs(float x, float A, float* coeffs )
{
    coeffs[0] = ((A*(x + 1.0f) - 5.0f*A)*(x + 1.0f) + 8.0f*A)*(x + 1.0f) - 4.0f*A;
    coeffs[1] = ((A + 2.0f)*x - (A + 3.0f))*x*x + 1.0f;
    coeffs[2] = ((A + 2.0f)*(1.0f - x) - (A + 3.0f))*(1.0f - x)*(1.0f - x) + 1.0f;
    coeffs[3] = 1.0f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void computeNormToPixParams(int W, int H, bool align_corners,
                                           float& xscale, float& yscale,
                                           float& xdelta, float& ydelta)
{
    const float delta = align_corners ? 1.f : 0.f;
    xscale = 0.5f * (W - delta);
    yscale = 0.5f * (H - delta);
    xdelta = 0.5f * (W - delta) + 0.5f * (delta - 1.f);
    ydelta = 0.5f * (H - delta) + 0.5f * (delta - 1.f);
}

static inline void normToPix(float nx, float ny,
                             float xscale, float yscale,
                             float xdelta, float ydelta,
                             float& x, float& y)
{
    x = nx * xscale + xdelta;
    y = ny * yscale + ydelta;
}

enum Mode { M_NEAREST=0, M_BILINEAR=1, M_BICUBIC=2 };
enum Pad  { P_ZEROS=0,  P_BORDER=1,  P_REFLECTION=2 };

template<typename T, int MODE, int PAD>
static inline void gridSampleComputeRows(
        const T* Xptr,
        const float* Gptr,
        T* Yptr,
        int N, int C, int H, int W,
        int Hout, int Wout,
        bool align_corners)
{
    size_t xNStride = (size_t)C * H * W;
    size_t xCStride = (size_t)H * W;
    size_t xHStride = (size_t)W;

    size_t gNStride = (size_t)Hout * Wout * 2;
    size_t gHStride = (size_t)Wout * 2;
    size_t gWStride = 2;

    size_t yNStride = (size_t)C * Hout * Wout;
    size_t yCStride = (size_t)Hout * Wout;
    size_t yHStride = (size_t)Wout;

    auto fetch = [&](const T* baseNC, int yy, int xx)->float {
        int px = xx, py = yy;
        if (PAD == P_BORDER) {
            px = saturate_cast<int>(std::min(float(W - 1), std::max(0.f, float(px))));
            py = saturate_cast<int>(std::min(float(H - 1), std::max(0.f, float(py))));
        } else if (PAD == P_REFLECTION) {
            px = saturate_cast<int>(std::floor(reflectCoord((float)px, W, align_corners) + 0.5f));
            py = saturate_cast<int>(std::floor(reflectCoord((float)py, H, align_corners) + 0.5f));
        }
        if (px < 0 || py < 0 || px >= W || py >= H) return 0.f;
        const T* p = baseNC + (size_t)py * xHStride + (size_t)px;
        return (float)(*p);
    };

    int nstripes = Hout * C * N;
    parallel_for_(Range(0, nstripes), [&](const Range& range) {
        for (int row = range.start; row < range.end; row++) {
            int n = row / (Hout * C);
            int c = (row - n * Hout * C) / Hout;
            int h = row % Hout;

            const T* baseNC = Xptr + n * xNStride + c * xCStride;
            size_t yRowBase = n * yNStride + c * yCStride + h * yHStride;
            size_t gRowBase = n * gNStride + h * gHStride;

            float xscale, yscale, xdelta, ydelta;
            computeNormToPixParams(W, H, align_corners, xscale, yscale, xdelta, ydelta);
            for (int w = 0; w < Wout; w++) {
                float nx = Gptr[gRowBase + w * gWStride + 0];
                float ny = Gptr[gRowBase + w * gWStride + 1];
                float xf, yf; normToPix(nx, ny, xscale, yscale, xdelta, ydelta, xf, yf);

                float outv = 0.f;
                if (MODE == M_NEAREST) {
                    int px = cvRound(xf);
                    int py = cvRound(yf);
                    outv = fetch(baseNC, py, px);
                } else if (MODE == M_BILINEAR) {
                    int x0 = saturate_cast<int>(floorf(xf));
                    int y0 = saturate_cast<int>(floorf(yf));
                    float dx = xf - x0, dy = yf - y0;

                    if (x0 >= 0 && y0 >= 0 && x0 + 1 < W && y0 + 1 < H)
                    {
                        const T* p_y0 = baseNC + (size_t)y0 * xHStride;
                        const T* p_y1 = baseNC + (size_t)(y0 + 1) * xHStride;
                        float v00 = (float)p_y0[x0];
                        float v01 = (float)p_y0[x0 + 1];
                        float v10 = (float)p_y1[x0];
                        float v11 = (float)p_y1[x0 + 1];
                        float vx0 = v00 * (1.f - dx) + v01 * dx;
                        float vx1 = v10 * (1.f - dx) + v11 * dx;
                        outv = vx0 * (1.f - dy) + vx1 * dy;
                    }
                    else
                    {
                        float v00 = fetch(baseNC, y0,     x0);
                        float v01 = fetch(baseNC, y0,     x0 + 1);
                        float v10 = fetch(baseNC, y0 + 1, x0);
                        float v11 = fetch(baseNC, y0 + 1, x0 + 1);
                        float vx0 = v00 * (1.f - dx) + v01 * dx;
                        float vx1 = v10 * (1.f - dx) + v11 * dx;
                        outv = vx0 * (1.f - dy) + vx1 * dy;
                    }
                } else {
                    const float alpha = -0.75f;
                    int x1 = saturate_cast<int>(floorf(xf));
                    int y1 = saturate_cast<int>(floorf(yf));
                    float tx = xf - x1, ty = yf - y1;

                    float wx[4], wy[4];
                    getCubicCoeffs(tx, alpha, wx);
                    getCubicCoeffs(ty, alpha, wy);

                    if (x1 >= 1 && y1 >= 1 && x1 + 2 < W && y1 + 2 < H) {
                        const T* p = baseNC + (size_t)(y1 - 1) * xHStride + (x1 - 1);
                        float rowv0 = (float)p[0] * wx[0] + (float)p[1] * wx[1] + (float)p[2] * wx[2] + (float)p[3] * wx[3];
                        p += xHStride;
                        float rowv1 = (float)p[0] * wx[0] + (float)p[1] * wx[1] + (float)p[2] * wx[2] + (float)p[3] * wx[3];
                        p += xHStride;
                        float rowv2 = (float)p[0] * wx[0] + (float)p[1] * wx[1] + (float)p[2] * wx[2] + (float)p[3] * wx[3];
                        p += xHStride;
                        float rowv3 = (float)p[0] * wx[0] + (float)p[1] * wx[1] + (float)p[2] * wx[2] + (float)p[3] * wx[3];
                        outv = rowv0 * wy[0] + rowv1 * wy[1] + rowv2 * wy[2] + rowv3 * wy[3];
                    } else {
                        float rowv0 = fetch(baseNC, y1 - 1, x1 - 1) * wx[0] + fetch(baseNC, y1 - 1, x1    ) * wx[1] +
                                      fetch(baseNC, y1 - 1, x1 + 1) * wx[2] + fetch(baseNC, y1 - 1, x1 + 2) * wx[3];
                        float rowv1 = fetch(baseNC, y1,     x1 - 1) * wx[0] + fetch(baseNC, y1,     x1    ) * wx[1] +
                                      fetch(baseNC, y1,     x1 + 1) * wx[2] + fetch(baseNC, y1,     x1 + 2) * wx[3];
                        float rowv2 = fetch(baseNC, y1 + 1, x1 - 1) * wx[0] + fetch(baseNC, y1 + 1, x1    ) * wx[1] +
                                      fetch(baseNC, y1 + 1, x1 + 1) * wx[2] + fetch(baseNC, y1 + 1, x1 + 2) * wx[3];
                        float rowv3 = fetch(baseNC, y1 + 2, x1 - 1) * wx[0] + fetch(baseNC, y1 + 2, x1    ) * wx[1] +
                                      fetch(baseNC, y1 + 2, x1 + 1) * wx[2] + fetch(baseNC, y1 + 2, x1 + 2) * wx[3];
                        outv = rowv0 * wy[0] + rowv1 * wy[1] + rowv2 * wy[2] + rowv3 * wy[3];
                    }
                }
                Yptr[yRowBase + w] = saturate_cast<T>(outv);
            }
        }
    });
}

template<typename T>
static inline void gridSampleDispatch(
        const T* Xptr,
        const float* Gptr,
        T* Yptr,
        int N, int C, int H, int W,
        int Hout, int Wout,
        bool align_corners,
        int mode, int padding)
{
    if (mode == M_NEAREST) {
        if (padding == P_ZEROS) gridSampleComputeRows<T, M_NEAREST, P_ZEROS>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
        else if (padding == P_BORDER) gridSampleComputeRows<T, M_NEAREST, P_BORDER>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
        else gridSampleComputeRows<T, M_NEAREST, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
    } else if (mode == M_BILINEAR) {
        if (padding == P_ZEROS) gridSampleComputeRows<T, M_BILINEAR, P_ZEROS>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
        else if (padding == P_BORDER) gridSampleComputeRows<T, M_BILINEAR, P_BORDER>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
        else gridSampleComputeRows<T, M_BILINEAR, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
    } else {
        if (padding == P_ZEROS) gridSampleComputeRows<T, M_BICUBIC, P_ZEROS>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
        else if (padding == P_BORDER) gridSampleComputeRows<T, M_BICUBIC, P_BORDER>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
        else gridSampleComputeRows<T, M_BICUBIC, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners);
    }
}

class GridSampleLayerImpl CV_FINAL : public GridSampleLayer
{
public:
    int mode = M_BILINEAR;
    int padding = P_ZEROS;
    bool align_corners = false;

    GridSampleLayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        String m = params.get<String>("mode", "bilinear");
        String p = params.get<String>("padding_mode", "zeros");
        align_corners = params.get<bool>("align_corners", false);

        if (m == "nearest") mode = M_NEAREST;
        else if (m == "bicubic") mode = M_BICUBIC;
        else mode = M_BILINEAR;

        if (p == "border") padding = P_BORDER;
        else if (p == "reflection") padding = P_REFLECTION;
        else padding = P_ZEROS;
    }

    bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        const MatShape& x = inputs[0];
        const MatShape& g = inputs[1];
        CV_Assert(x.size() == 4);
        CV_Assert(g.size() == 4 && g[3] == 2);
        outputs.assign(1, MatShape({x[0], x[1], g[1], g[2]}));
        return false;
    }

    void getTypes(const std::vector<MatType>& inTypes,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outTypes,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inTypes.empty());
        outTypes.assign(requiredOutputs, inTypes[0]);
        internals.assign(requiredInternals, inTypes[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& X = inputs[0];
        const Mat& G = inputs[1];

        CV_Assert(X.dims == 4 && G.dims == 4 && G.size[3] == 2);
        const int N = X.size[0], C = X.size[1], H = X.size[2], W = X.size[3];
        const int Hout = G.size[1], Wout = G.size[2];

        Mat Gf = G;
        if (G.depth() != CV_32F) G.convertTo(Gf, CV_32F);

        const float* Gptr = (const float*)Gf.data;

        CV_Assert(!outputs.empty());
        int outType  = outputs[0].type();

        Mat Ytmp({N, C, Hout, Wout}, outType);
        Ytmp.setTo(Scalar(0));

        switch (X.depth()) {
        case CV_8U: {
            const uchar* Xptr = X.ptr<uchar>();
            uchar* Yptr = Ytmp.ptr<uchar>();
            gridSampleDispatch<uchar>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding);
            break;
        }
        case CV_16F: {
            const hfloat* Xptr = X.ptr<hfloat>();
            hfloat* Yptr = Ytmp.ptr<hfloat>();
            gridSampleDispatch<hfloat>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding);
            break;
        }
#ifdef CV_16BF
        case CV_16BF: {
            const bfloat* Xptr = X.ptr<bfloat>();
            bfloat* Yptr = Ytmp.ptr<bfloat>();
            gridSampleDispatch<bfloat>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding);
            break;
        }
#endif
        case CV_32F: {
            const float* Xptr = X.ptr<float>();
            float* Yptr = Ytmp.ptr<float>();
            gridSampleDispatch<float>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding);
            break;
        }
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported input depth for GridSample");
        }

        Ytmp.copyTo(outputs[0]);
    }
};

Ptr<GridSampleLayer> GridSampleLayer::create(const LayerParams& params) {
    return Ptr<GridSampleLayer>(new GridSampleLayerImpl(params));
}

}}
