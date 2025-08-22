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
// Supported opsets: 16, 20, 22

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

static inline float cubicWeight(float t, float alpha) {
    t = std::fabs(t);
    const float t2 = t * t;
    const float t3 = t2 * t;
    if (t <= 1.f) {
        return (alpha + 2.f) * t3 - (alpha + 3.f) * t2 + 1.f;
    } else if (t < 2.f) {
        return alpha * t3 - 5.f * alpha * t2 + 8.f * alpha * t - 4.f * alpha;
    } else {
        return 0.f;
    }
}

static inline void normToPix(float nx, float ny, int W, int H, bool align_corners, float& x, float& y) {
    if (align_corners) {
        x = (nx + 1.f) * 0.5f * (W - 1);
        y = (ny + 1.f) * 0.5f * (H - 1);
    } else {
        x = ((nx + 1.f) * W - 1.f) * 0.5f;
        y = ((ny + 1.f) * H - 1.f) * 0.5f;
    }
}

enum Mode { M_NEAREST=0, M_BILINEAR=1, M_BICUBIC=2 };
enum Pad  { P_ZEROS=0,  P_BORDER=1,  P_REFLECTION=2 };

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

        Mat Xf = X;
        if (X.depth() != CV_32F) X.convertTo(Xf, CV_32F);
        Mat Gf = G;
        if (G.depth() != CV_32F) G.convertTo(Gf, CV_32F);

        Mat Ytmp({N, C, Hout, Wout}, CV_32F);
        Ytmp.setTo(Scalar(0));

        auto Xptr = (const float*)Xf.data;
        auto Gptr = (const float*)Gf.data;
        auto Yptr = (float*)Ytmp.data;

        size_t xNStride = (size_t)C*H*W;
        size_t xCStride = (size_t)H*W;
        size_t xHStride = (size_t)W;

        size_t gNStride = (size_t)Hout*Wout*2;
        size_t gHStride = (size_t)Wout*2;
        size_t gWStride = 2;

        size_t yNStride = (size_t)C*Hout*Wout;
        size_t yCStride = (size_t)Hout*Wout;
        size_t yHStride = (size_t)Wout;

        auto sampleAt = [&](int n, int c, float x, float y)->float {
            auto fetch = [&](int yy, int xx)->float {
                int px = xx, py = yy;
                if (padding == P_BORDER) {
                    px = int(std::min(float(W - 1), std::max(0.f, float(px))));
                    py = int(std::min(float(H - 1), std::max(0.f, float(py))));
                } else if (padding == P_REFLECTION) {
                    px = int(std::floor(reflectCoord((float)px, W, align_corners) + 0.5f));
                    py = int(std::floor(reflectCoord((float)py, H, align_corners) + 0.5f));
                }
                if (px < 0 || py < 0 || px >= W || py >= H) return 0.f;
                const float* base = Xptr + n*xNStride + c*xCStride + (size_t)py*xHStride + (size_t)px;
                return *base;
            };

            if (mode == M_NEAREST) {
                int px = int(std::floor(x + 0.5f));
                int py = int(std::floor(y + 0.5f));
                return fetch(py, px);
            } else if (mode == M_BILINEAR) {
                int x0 = (int)floorf(x), y0 = (int)floorf(y);
                float dx = x - x0, dy = y - y0;
                float v00 = fetch(y0,   x0);
                float v01 = fetch(y0,   x0+1);
                float v10 = fetch(y0+1, x0);
                float v11 = fetch(y0+1, x0+1);
                float vx0 = v00*(1.f-dx) + v01*dx;
                float vx1 = v10*(1.f-dx) + v11*dx;
                return vx0*(1.f-dy) + vx1*dy;
            } else {
                const float alpha = -0.75f;
                int x1 = (int)floorf(x);
                int y1 = (int)floorf(y);
                float tx = x - x1, ty = y - y1;

                float wx[4], wy[4];
                for (int i = 0; i < 4; ++i) {
                    wx[i] = cubicWeight((i - 1) - tx, alpha);
                    wy[i] = cubicWeight((i - 1) - ty, alpha);
                }
                float sumwx = wx[0]+wx[1]+wx[2]+wx[3];
                float sumwy = wy[0]+wy[1]+wy[2]+wy[3];
                if (sumwx != 0.f) { for (int i = 0; i < 4; ++i) wx[i] /= sumwx; }
                if (sumwy != 0.f) { for (int i = 0; i < 4; ++i) wy[i] /= sumwy; }

                float acc = 0.f;
                for (int j = 0; j < 4; ++j) {
                    float row = 0.f;
                    for (int i = 0; i < 4; ++i) {
                        row += fetch(y1 + (j - 1), x1 + (i - 1)) * wx[i];
                    }
                    acc += row * wy[j];
                }
                return acc;
            }
        };

        cv::parallel_for_(cv::Range(0, N * C * Hout * Wout), [&](const cv::Range& r) {
            for (int idx = r.start; idx < r.end; ++idx) {
                int t = idx;
                int w = t % Wout; t /= Wout;
                int h = t % Hout; t /= Hout;
                int c = t % C;    t /= C;
                int n = t;

                float nx = Gptr[n*gNStride + h*gHStride + w*gWStride + 0];
                float ny = Gptr[n*gNStride + h*gHStride + w*gWStride + 1];
                float x, y; normToPix(nx, ny, W, H, align_corners, x, y);

                Yptr[n*yNStride + c*yCStride + h*yHStride + w] = sampleAt(n, c, x, y);
            }
        });

        CV_Assert(!outputs.empty());
        int outType  = outputs[0].type();
        int outDepth = CV_MAT_DEPTH(outType);
        if (outDepth == CV_32F) {
            Ytmp.copyTo(outputs[0]);
        } else {
            Ytmp.convertTo(outputs[0], outType);
        }
    }
};

Ptr<GridSampleLayer> GridSampleLayer::create(const LayerParams& params) {
    return Ptr<GridSampleLayer>(new GridSampleLayerImpl(params));
}

}}
