// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../op_cann.hpp"
#include "../net_impl.hpp"
#include <opencv2/imgproc.hpp>

// Implements ONNX Resize operator semantics (ai.onnx) as per onnx.ai documentation.
// See: https://onnx.ai/onnx/operators/onnx__Resize.html (opsets 10, 11, 13, 18 supported)

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#include <openvino/op/interpolate.hpp>
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/resize.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv { namespace dnn {

namespace {

enum class CoordTransMode {
    HALF_PIXEL,
    PYTORCH_HALF_PIXEL,
    TF_HALF_PIXEL_FOR_NN,
    TF_CROP_AND_RESIZE,
    ASYMMETRIC
};

static inline CoordTransMode parseCoordTransMode(const String& s)
{
    if (s == "half_pixel") return CoordTransMode::HALF_PIXEL;
    if (s == "pytorch_half_pixel") return CoordTransMode::PYTORCH_HALF_PIXEL;
    if (s == "tf_half_pixel_for_nn") return CoordTransMode::TF_HALF_PIXEL_FOR_NN;
    if (s == "tf_crop_and_resize") return CoordTransMode::TF_CROP_AND_RESIZE;
    return CoordTransMode::ASYMMETRIC;
}

enum class NearestMode {
    FLOOR,
    CEIL,
    ROUND_PREFER_CEIL,
    ROUND_PREFER_FLOOR
};

#if __cplusplus < 201703L
template<typename T>
static T clamp(T d, T min, T max)
{
    return std::min(std::max(d, min), max);
}
#else
#define clamp std::clamp
#endif

static inline NearestMode parseNearestMode(const String& s)
{
    if (s == "floor") return NearestMode::FLOOR;
    if (s == "ceil") return NearestMode::CEIL;
    if (s == "round_prefer_ceil") return NearestMode::ROUND_PREFER_CEIL;
    return NearestMode::ROUND_PREFER_FLOOR;
}

static constexpr int kResizeNumStripes = 16;

inline float computeSrcGeneric(int dst, float scale, int limit, int len,
                               CoordTransMode coordTransMode, bool /*halfPixelCenters*/,
                               float start_coord = 0.0f, float end_coord = 1.0f)
{
    if (coordTransMode == CoordTransMode::TF_CROP_AND_RESIZE)
    {
        if (len > 1)
            return start_coord * (limit - 1) + dst * (end_coord - start_coord) * (limit - 1) / float(len - 1);
        else
            return 0.5f * (start_coord + end_coord) * (limit - 1);
    }
    if (coordTransMode == CoordTransMode::PYTORCH_HALF_PIXEL)
        return (len > 1) ? (dst + 0.5f)*scale - 0.5f : 0.f;
    if (coordTransMode == CoordTransMode::HALF_PIXEL)
        return (dst + 0.5f)*scale - 0.5f;
    if (coordTransMode == CoordTransMode::TF_HALF_PIXEL_FOR_NN)
        return (dst + 0.5f)*scale;
    return dst*scale;
}

static inline void buildNearestIndexMap(std::vector<int>& map,
                                        int outLen,
                                        int inLen,
                                        float scale,
                                        int len,
                                        float start_coord,
                                        float end_coord,
                                        CoordTransMode coordTransMode,
                                        NearestMode nearestMode,
                                        bool halfPixelCenters)
{
    auto nearestIndex = [&](float src) {
        const int f = cvFloor(src);
        const float frac = src - f;
        const float eps = 1e-6f;
        int idx;
        if (nearestMode == NearestMode::FLOOR) idx = cvFloor(src);
        else if (nearestMode == NearestMode::CEIL)  idx = cvCeil(src);
        else if (nearestMode == NearestMode::ROUND_PREFER_CEIL) {
            idx = (abs(frac - 0.5f) <= eps) ? (f + 1) : cvRound(src);
        } else {
            idx = (abs(frac - 0.5f) <= eps) ? f : cvRound(src);
        }
        return clamp(idx, 0, inLen - 1);
    };

    map.resize(outLen);
    for (int i = 0; i < outLen; ++i)
    {
        float src = computeSrcGeneric(i, scale, inLen - 1, len,
                                      coordTransMode, halfPixelCenters, start_coord, end_coord);
        if (coordTransMode == CoordTransMode::TF_CROP_AND_RESIZE) {
            if (src < 0.f || src >= float(inLen)) {
                map[i] = -1; // out of bounds
                continue;
            }
        } else {
            src = std::min(std::max(src, 0.f), float(inLen - 1));
        }
        map[i] = nearestIndex(src);
    }
}

static inline void buildBilinearIndexAndLerp(std::vector<int>& i0,
                                             std::vector<int>& i1,
                                             std::vector<float>& frac,
                                             std::vector<uint8_t>& outOfBounds,
                                             int outLen,
                                             int inLen,
                                             float scale,
                                             int len,
                                             float start_coord,
                                             float end_coord,
                                             CoordTransMode coordTransMode,
                                             bool halfPixelCenters,
                                             bool tf_crop_and_resize_mode)
{
    i0.resize(outLen);
    i1.resize(outLen);
    frac.resize(outLen);
    outOfBounds.assign(outLen, 0);

    for (int o = 0; o < outLen; ++o)
    {
        float src = computeSrcGeneric(o, scale, inLen, len,
                                      coordTransMode, halfPixelCenters, start_coord, end_coord);
        if (tf_crop_and_resize_mode)
        {
            int base = int(std::floor(src));
            if (base < 0 || base >= inLen - 1) {
                outOfBounds[o] = 1;
                i0[o] = 0;
                i1[o] = 0;
                frac[o] = 0.0f;
            } else {
                i0[o] = base;
                i1[o] = base + 1;
                frac[o] = src - float(base);
            }
        }
        else
        {
            src = std::min(std::max(src, 0.f), float(inLen - 1) - 1e-6f);
            int base = int(std::floor(src));
            i0[o] = base;
            i1[o] = clamp(base + 1, 0, inLen - 1);
            frac[o] = src - float(base);
        }
    }
}

static inline void interpolateCubicResize(float x, float A, float* coeffs )
{
    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void buildCubicIndexAndWeights(std::vector<std::array<int,4>>& ids,
                                             std::vector<std::array<float,4>>& weights,
                                             std::vector<uint8_t>& outOfBounds,
                                             int outLen,
                                             int inLen,
                                             float scale,
                                             int len,
                                             float start_coord,
                                             float end_coord,
                                             CoordTransMode coordTransMode,
                                             bool halfPixelCenters,
                                             bool tf_crop_and_resize_mode,
                                             bool excludeOutside,
                                             float cubicA)
{
    ids.resize(outLen);
    weights.resize(outLen);
    outOfBounds.assign(outLen, 0);

    for (int o = 0; o < outLen; ++o)
    {
        float src = computeSrcGeneric(o, scale, inLen, len,
                                      coordTransMode, halfPixelCenters, start_coord, end_coord);
        int i = int(std::floor(src));
        float d = src - i;
        float sw = 0.f;
        bool hasOutOfBounds = false;
        interpolateCubicResize(d, cubicA, weights[o].data());

        if (!tf_crop_and_resize_mode && !excludeOutside)
        {
            for (int k = -1; k <= 2; ++k)
            {
                int idx = clamp(i + k, 0, inLen - 1);
                ids[o][k+1] = idx;
                sw += weights[o][k+1];
            }
        }
        else if (tf_crop_and_resize_mode)
        {
            for (int k = -1; k <= 2; ++k)
            {
                int idx = i + k;
                unsigned valid = (unsigned)idx < (unsigned)inLen;
                ids[o][k+1] = valid ? idx : -1;
                float w = weights[o][k+1];
                float wv = valid ? w : 0.f;
                weights[o][k+1] = wv;
                sw += wv;
                hasOutOfBounds |= !valid;
            }
        }
        else
        {
            for (int k = -1; k <= 2; ++k)
            {
                int idx = i + k;
                unsigned valid = (unsigned)idx < (unsigned)inLen;
                ids[o][k+1] = valid ? idx : -1;
                float w = weights[o][k+1];
                float wv = valid ? w : 0.f;
                weights[o][k+1] = wv;
                sw += wv;
            }
        }

        if (sw != 0.f)
            for (int k = 0; k < 4; ++k)
                weights[o][k] /= sw;

        if (tf_crop_and_resize_mode && hasOutOfBounds)
            outOfBounds[o] = 1;
    }
}

template<typename T>
void resizeNearest(const Mat &inp, Mat &out,
                   float scaleH, float scaleW,
                   int lenY, int lenX,
                   NearestMode nearestMode,
                   const String &coordTransMode,
                   bool halfPixelCenters,
                   float start_y = 0.0f, float end_y = 1.0f,
                   float start_x = 0.0f, float end_x = 1.0f,
                   float extrapolation_value = 0.0f)
{
    int numPlanes = inp.size[0] * inp.size[1];
    int inH       = inp.size[2], inW       = inp.size[3];
    int outH      = out.size[2], outW      = out.size[3];
    CV_Assert(inp.isContinuous() && out.isContinuous());

    Mat inpP = inp.reshape(1, numPlanes * inH);
    Mat outP = out.reshape(1, numPlanes * outH);

    CoordTransMode coordMode = parseCoordTransMode(coordTransMode);

    std::vector<int> mapY(outH);
    buildNearestIndexMap(mapY, outH, inH, scaleH, lenY, start_y, end_y,
                         coordMode, nearestMode, halfPixelCenters);

    std::vector<int> mapX(outW);
    buildNearestIndexMap(mapX, outW, inW, scaleW, lenX, start_x, end_x,
                         coordMode, nearestMode, halfPixelCenters);

    const int nstripes = kResizeNumStripes;
    parallel_for_(Range(0, nstripes), [&](const Range& range) {
        const bool tf_crop_and_resize_mode = (coordMode == CoordTransMode::TF_CROP_AND_RESIZE);
        int row0 = range.start * (outH * numPlanes) / nstripes;
        float extrapolation_value_ = extrapolation_value;
        int row1 = range.end   * (outH * numPlanes) / nstripes - 1;
        int plane0 = row0 / outH, plane1 = row1 / outH;
        row0 %= outH;
        row1 %= outH;

        const int* mapYptr = mapY.data();
        const int* mapXptr = mapX.data();

        for (int p = plane0; p <= plane1; p++) {
            int y0 = p == plane0 ? row0 : 0;
            int y1 = p == plane1 ? row1 : outH - 1;
            for (int y = y0; y <= y1; y++) {
                int my = mapYptr[y];
                if (tf_crop_and_resize_mode && my == -1) {
                    T* outRowFill = outP.ptr<T>(p * outH + y);
                    for (int x = 0; x < outW; ++x)
                        outRowFill[x] = T(extrapolation_value_);
                    continue;
                }
                const T* inpRow = inpP.ptr<T>(p * inH + my);
                T*       outRow = outP.ptr<T>(p * outH + y);
                for (int x = 0; x < outW; ++x)
                {
                    int mx = mapXptr[x];
                    if (tf_crop_and_resize_mode && mx == -1) {
                        outRow[x] = T(extrapolation_value_);
                    } else {
                        outRow[x] = inpRow[mx];
                    }
                }
            }
        }
    }, nstripes);
}

template<typename T>
void resizeBilinear(const Mat &inp, Mat &out,
                    float scaleH, float scaleW,
                    int lenY, int lenX,
                    const String &coordTransMode,
                    bool halfPixelCenters,
                    float start_y = 0.0f, float end_y = 1.0f,
                    float start_x = 0.0f, float end_x = 1.0f,
                    float extrapolation_value = 0.0f)
{
    int numPlanes = inp.size[0]*inp.size[1];
    int inH       = inp.size[2], inW = inp.size[3];
    int outH      = out.size[2], outW = out.size[3];
    CV_Assert(inp.isContinuous() && out.isContinuous());

    Mat inpP = inp.reshape(1, numPlanes*inH);
    Mat outP = out.reshape(1, numPlanes*outH);

    CoordTransMode coordMode = parseCoordTransMode(coordTransMode);
    const bool tf_crop_and_resize_mode = (coordMode == CoordTransMode::TF_CROP_AND_RESIZE);

    std::vector<int>    x0(outW), x1(outW);
    std::vector<float> lx(outW);
    std::vector<uint8_t>  outOfBoundsX(outW);
    buildBilinearIndexAndLerp(x0, x1, lx, outOfBoundsX,
                              outW, inW, scaleW, lenX, start_x, end_x,
                              coordMode, halfPixelCenters, tf_crop_and_resize_mode);

    std::vector<int>    y0(outH), y1(outH);
    std::vector<float> ly(outH);
    std::vector<uint8_t>  outOfBoundsY(outH);
    buildBilinearIndexAndLerp(y0, y1, ly, outOfBoundsY,
                              outH, inH, scaleH, lenY, start_y, end_y,
                              coordMode, halfPixelCenters, tf_crop_and_resize_mode);

    const int nstripes = kResizeNumStripes;
    parallel_for_(Range(0, nstripes), [&](const Range& range) {
        int row0 = range.start * (outH * numPlanes) / nstripes;
        int row1 = range.end   * (outH * numPlanes) / nstripes - 1;
        int plane0 = row0 / outH, plane1 = row1 / outH;
        row0 %= outH;
        row1 %= outH;

        const int* y0ptr = y0.data();
        const int* y1ptr = y1.data();
        const float* lyptr = ly.data();
        const int* x0ptr = x0.data();
        const float* lxptr = lx.data();
        const uint8_t* outOfBoundsYptr = outOfBoundsY.data();
        const uint8_t* outOfBoundsXptr = outOfBoundsX.data();
        float extrapolation_value_ = extrapolation_value;
        const bool tf_crop_and_resize_mode_ = tf_crop_and_resize_mode;
        std::vector<float> hbufbuf(inW + 3);
        float* hbuf = hbufbuf.data() + 1;

        for (int p = plane0; p <= plane1; ++p)
        {
            int oy0 = (p == plane0) ? row0 : 0;
            int oy1 = (p == plane1) ? row1 : outH - 1;
            for (int oy = oy0; oy <= oy1; ++oy)
            {
                if (tf_crop_and_resize_mode_ && outOfBoundsYptr[oy]) {
                    T* outRowFill = outP.ptr<T>(p * outH + oy);
                    for (int ox = 0; ox < outW; ++ox)
                        outRowFill[ox] = T(extrapolation_value_);
                    continue;
                }

                const T* row0ptr = inpP.ptr<T>( p * inH + y0ptr[oy] );
                const T* row1ptr = inpP.ptr<T>( p * inH + y1ptr[oy] );
                float    fy      = lyptr[oy];

                T* outRowBase = outP.ptr<T>( p * outH + oy );

                for (int ix = 0; ix < inW; ++ix)
                {
                    float v0 = static_cast<float>(row0ptr[ix]);
                    float v1 = static_cast<float>(row1ptr[ix]);
                    hbuf[ix] = v0 + fy * (v1 - v0);
                }
                hbuf[-1] = hbuf[0];
                hbuf[inW] = hbuf[inW - 1];
                hbuf[inW + 1] = hbuf[inW - 1];

                for (int ox = 0; ox < outW; ++ox)
                {
                    if (tf_crop_and_resize_mode_ && outOfBoundsXptr[ox]) {
                        outRowBase[ox] = T(extrapolation_value_);
                    } else {
                        int   xi = x0ptr[ox];
                        float fx = lxptr[ox];

                        float left = hbuf[xi];
                        float res  = left + fx * (hbuf[xi + 1] - left);
                        outRowBase[ox] = T(res);
                    }
                }
            }
        }
    }, nstripes);
}

template<typename T>
void resizeCubic(const Mat &inp, Mat &out,
                 float scaleH, float scaleW,
                 int lenY, int lenX,
                 float cubicA, bool excludeOutside,
                 const String &coordTransMode, bool halfPixelCenters,
                 float start_y = 0.0f, float end_y = 1.0f,
                 float start_x = 0.0f, float end_x = 1.0f,
                 float extrapolation_value = 0.0f)
{
    int numPlanes = inp.size[0] * inp.size[1];
    int inH = inp.size[2], inW = inp.size[3];
    int outH = out.size[2], outW = out.size[3];

    Mat inpPlanes = inp.reshape(1, numPlanes * inH);
    Mat outPlanes = out.reshape(1, numPlanes * outH);

    CoordTransMode coordMode = parseCoordTransMode(coordTransMode);
    const bool tf_crop_and_resize_mode = (coordMode == CoordTransMode::TF_CROP_AND_RESIZE);

    std::vector<std::array<int,4>> x_id(outW);
    std::vector<std::array<float,4>> x_w (outW);
    std::vector<uint8_t> outOfBoundsX(outW);
    buildCubicIndexAndWeights(x_id, x_w, outOfBoundsX,
                              outW, inW, scaleW, lenX, start_x, end_x,
                              coordMode, halfPixelCenters, tf_crop_and_resize_mode,
                              excludeOutside, cubicA);

    std::vector<std::array<int,4>> y_id(outH);
    std::vector<std::array<float,4>> y_w (outH);
    std::vector<uint8_t> outOfBoundsY(outH);
    buildCubicIndexAndWeights(y_id, y_w, outOfBoundsY,
                              outH, inH, scaleH, lenY, start_y, end_y,
                              coordMode, halfPixelCenters, tf_crop_and_resize_mode,
                              excludeOutside, cubicA);

    const int nstripes = kResizeNumStripes;
    parallel_for_(Range(0, nstripes), [&](const Range& range) {
        int row0 = range.start * (outH * numPlanes) / nstripes;
        int row1 = range.end   * (outH * numPlanes) / nstripes - 1;
        int plane0 = row0 / outH, plane1 = row1 / outH;
        row0 %= outH;
        row1 %= outH;

        const bool tf_crop_and_resize_mode_ = tf_crop_and_resize_mode;
        const uint8_t* outOfBoundsYptr = outOfBoundsY.data();
        const uint8_t* outOfBoundsXptr = outOfBoundsX.data();
        float extrapolation_value_ = extrapolation_value;
        std::vector<float> hbuf(inW, 0.f);

        for (int p = plane0; p <= plane1; ++p)
        {
            int oy0 = (p == plane0) ? row0 : 0;
            int oy1 = (p == plane1) ? row1 : outH - 1;
            const T* inpBase = inpPlanes.ptr<T>(p * inH);
            for (int oy = oy0; oy <= oy1; ++oy)
            {
                T* outRow = outPlanes.ptr<T>(p * outH) + oy * outW;

                if (tf_crop_and_resize_mode_ && outOfBoundsYptr[oy]) {
                    for (int ox = 0; ox < outW; ++ox)
                        outRow[ox] = cv::saturate_cast<T>(extrapolation_value_);
                    continue;
                }

                const float w0y = y_w[oy][0];
                const float w1y = y_w[oy][1];
                const float w2y = y_w[oy][2];
                const float w3y = y_w[oy][3];

                int yy0, yy1, yy2, yy3;
                yy0 = y_id[oy][0];
                yy1 = y_id[oy][1];
                yy2 = y_id[oy][2];
                yy3 = y_id[oy][3];

                const T* ptr0 = (yy0 >= 0) ? (inpBase + (size_t)yy0 * inW) : nullptr;
                const T* ptr1 = (yy1 >= 0) ? (inpBase + (size_t)yy1 * inW) : nullptr;
                const T* ptr2 = (yy2 >= 0) ? (inpBase + (size_t)yy2 * inW) : nullptr;
                const T* ptr3 = (yy3 >= 0) ? (inpBase + (size_t)yy3 * inW) : nullptr;

                if (!ptr0 && !ptr1 && !ptr2 && !ptr3) {
                    for (int ix = 0; ix < inW; ++ix)
                    {
                        hbuf[ix] = 0.f;
                    }
                } else {
                    const T* ptrNZ = ptr0 ? ptr0 : (ptr1 ? ptr1 : (ptr2 ? ptr2 : ptr3));
                    float w0 = ptr0 ? w0y : 0.f;
                    float w1 = ptr1 ? w1y : 0.f;
                    float w2 = ptr2 ? w2y : 0.f;
                    float w3 = ptr3 ? w3y : 0.f;
                    if (!ptr0) ptr0 = ptrNZ;
                    if (!ptr1) ptr1 = ptrNZ;
                    if (!ptr2) ptr2 = ptrNZ;
                    if (!ptr3) ptr3 = ptrNZ;

                    for (int ix = 0; ix < inW; ++ix)
                    {
                        hbuf[ix] = static_cast<float>(ptr0[ix]) * w0 +
                                   static_cast<float>(ptr1[ix]) * w1 +
                                   static_cast<float>(ptr2[ix]) * w2 +
                                   static_cast<float>(ptr3[ix]) * w3;
                    }
                }

                for (int ox = 0; ox < outW; ++ox)
                {
                    if (tf_crop_and_resize_mode_ && outOfBoundsXptr[ox]) {
                        outRow[ox] = cv::saturate_cast<T>(extrapolation_value_);
                        continue;
                    }
                    const int xx = x_id[ox][1];
                    const float w0x = x_w[ox][0];
                    const float w1x = x_w[ox][1];
                    const float w2x = x_w[ox][2];
                    const float w3x = x_w[ox][3];
                    float val;
                    if (1 <= xx && xx + 3 < inW) {
                        val = hbuf[xx - 1] * w0x + hbuf[xx] * w1x + hbuf[xx + 1] * w2x + hbuf[xx + 2] * w3x;
                    } else {
                        const int xx0 = x_id[ox][0];
                        const int xx1 = x_id[ox][1];
                        const int xx2 = x_id[ox][2];
                        const int xx3 = x_id[ox][3];
                        val = 0.f;
                        if (xx0 >= 0) val += hbuf[xx0] * w0x;
                        if (xx1 >= 0) val += hbuf[xx1] * w1x;
                        if (xx2 >= 0) val += hbuf[xx2] * w2x;
                        if (xx3 >= 0) val += hbuf[xx3] * w3x;
                    }
                    outRow[ox] = cv::saturate_cast<T>(val);
                }
            }
        }
    }, nstripes);
}
}

class Resize2LayerImpl : public Resize2Layer
{
public:
    int outWidth0, outHeight0;
    Resize2LayerImpl(const LayerParams& params) : zoomFactorWidth(params.get<float>("zoom_factor_x", params.get<float>("zoom_factor", 0))),
                                                 zoomFactorHeight(params.get<float>("zoom_factor_y", params.get<float>("zoom_factor", 0))),
                                                 scaleWidth(0), scaleHeight(0), cubicCoeffA(params.get<float>("cubic_coeff_a", -0.75f)),
                                                 roi_start_y(0.0f), roi_end_y(1.0f), roi_start_x(0.0f), roi_end_x(1.0f),
                                                 extrapolation_value(params.get<float>("extrapolation_value", 0.0f))
    {
        setParamsFrom(params);
        outWidth = outWidth0 = params.get<float>("width", 0);
        outHeight = outHeight0 = params.get<float>("height", 0);
        if (params.has("zoom_factor"))
        {
            CV_Assert(!params.has("zoom_factor_x") && !params.has("zoom_factor_y"));
        }
        else if (params.has("zoom_factor_x") || params.has("zoom_factor_y"))
        {
            CV_Assert(params.has("zoom_factor_x") && params.has("zoom_factor_y"));
        }
        interpolation = params.get<String>("interpolation");
        // Keep nearest_mode if provided (ONNX attribute). Default is "round_prefer_floor" as per ONNX spec.
        nearestModeE = parseNearestMode(params.get<String>("nearest_mode", "round_prefer_floor"));
        CV_Check(interpolation, interpolation == "nearest" || interpolation == "opencv_linear" || interpolation == "bilinear" || interpolation == "cubic", "");

        excludeOutside = params.get<bool>("exclude_outside", false);
        dynamicROI = params.get<bool>("dynamic_roi", false);

        alignCorners = params.get<bool>("align_corners", false);
        halfPixelCenters = params.get<bool>("half_pixel_centers", false);
        coordTransMode = params.get<String>("coordinate_transformation_mode", "half_pixel");
        coordTransModeE = parseCoordTransMode(coordTransMode);

        if (interpolation == "opencv_linear")
            halfPixelCenters = true;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        if (dynamicROI) return true;
        size_t ninputs = inputs.size();
        if (ninputs <= 1 &&
            ((outWidth0 > 0 && outHeight0 > 0) ||
            (zoomFactorWidth > 0 && zoomFactorHeight > 0)))
            return false;
        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_)
            return true;
        for (size_t i = 1; i < ninputs; i++) {
            if (!netimpl_->isConstArg(inputs[i]))
                return true;
        }
        return false;
    }

    MatShape getOutShape(const MatShape& inpShape, const std::vector<int>& sizes,
                         const std::vector<float>& scales) const
    {
        // ONNX Resize allows either "sizes" or "scales" input. These tensors may
        // describe all 4 dims (N,C,H,W) or only spatial dims (H,W) when accompanied
        // by an "axes" input equal to {2,3}. To stay backwards-compatible, we keep
        // the legacy 4-element handling but also accept 2-element vectors.

        CV_Assert((sizes.empty() ^ scales.empty()) &&
                  (sizes.empty() ? (scales.size() == 4 || scales.size() == 2)
                                 : (sizes.size() == 4 || sizes.size() == 2)));

        MatShape outShape = inpShape;
        if (!sizes.empty()) {
            if (sizes.size() == 4) {
                outShape[2] = sizes[2];
                outShape[3] = sizes[3];
            } else /* sizes.size() == 2 */ {
                outShape[2] = sizes[0];
                outShape[3] = sizes[1];
            }
        } else {
            if (scales.size() == 4) {
                outShape[2] = cvFloor(inpShape[2] * scales[2]);
                outShape[3] = cvFloor(inpShape[3] * scales[3]);
            } else /* scales.size() == 2 */ {
                outShape[2] = cvFloor(inpShape[2] * scales[0]);
                outShape[3] = cvFloor(inpShape[3] * scales[1]);
            }
        }
        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(ninputs == 1 || ninputs == 2 || ninputs >= 4);
        outputs.resize(1, inputs[0]);
        // New ONNX importer may provide "sizes" or "scales" via constant blobs
        // (blobs[0] = roi, blobs[1] = scales, blobs[2] = sizes, blobs[3] = axes).
        if (ninputs == 1 && !this->blobs.empty()) {
            std::vector<int> sizes;
            std::vector<float> scales;
            if (this->blobs.size() >= 3 && this->blobs[2].total() > 0)
                tensorToIntVec(this->blobs[2], sizes);
            if (this->blobs.size() >= 2 && this->blobs[1].total() > 0)
                tensorToFloatVec(this->blobs[1], scales);

            if (!sizes.empty() || !scales.empty()) {
                outputs[0] = getOutShape(inputs[0], sizes, scales);
                // in-place if spatial dims unchanged
                return (outputs[0][2] == inputs[0][2]) && (outputs[0][3] == inputs[0][3]);
            }
        }

        if (ninputs == 1) {
            outputs[0][2] = zoomFactorHeight > 0 ? (int)(inputs[0][2] * zoomFactorHeight) : outHeight0;
            outputs[0][3] = zoomFactorWidth > 0 ? (int)(inputs[0][3] * zoomFactorWidth) : outWidth0;
        } else if (ninputs == 2 && inputs[1].dims == 4) {
            outputs[0][2] = inputs[1][2];
            outputs[0][3] = inputs[1][3];
        } else {
            Net::Impl* netimpl_ = getNetImpl(this);
            std::vector<int> sizes;
            std::vector<float> scales;
            if (ninputs >= 4) {
                Mat sizesTensor = netimpl_->argTensor(this->inputs[3]);
                tensorToIntVec(sizesTensor, sizes);
            }

            Mat scalesTensor = netimpl_->argTensor(this->inputs[(ninputs == 2) ? 1 : 2]);
            tensorToFloatVec(scalesTensor, scales);
            outputs[0] = getOutShape(inputs[0], sizes, scales);
        }
        // We can work in-place (do nothing) if input shape == output shape.
        return (outputs[0][2] == inputs[0][2]) && (outputs[0][3] == inputs[0][3]);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_CUDA)
        {
            EngineType engine_forced = getForcedDnnEngine();
            if (engine_forced != ENGINE_CLASSIC)
                return false;
            return interpolation == "nearest" || interpolation == "bilinear" || interpolation == "opencv_linear";
        }

        if (backendId == DNN_BACKEND_CANN)
            return interpolation == "nearest" || interpolation == "bilinear" || interpolation == "opencv_linear";

#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            return (interpolation == "nearest" && scaleWidth == scaleHeight) ||
                   (interpolation == "bilinear");
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    void updateOutSizeAndScale(const MatShape& inpShape, const MatShape& outShape)
    {
        CV_Assert(outShape.dims == 4);
        outHeight = outShape[2];
        outWidth = outShape[3];
        if (alignCorners && outHeight > 1)
            scaleHeight = float(inpShape[2] - 1) / (outHeight - 1);
        else
            scaleHeight = float(inpShape[2]) / outHeight;

        if (alignCorners && outWidth > 1)
            scaleWidth = float(inpShape[3] - 1) / (outWidth - 1);
        else
            scaleWidth = float(inpShape[3]) / outWidth;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<int> sizes;
        std::vector<float> scales;
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        size_t ninputs = inputs.size();
        CV_Assert(ninputs > 0);

        Mat& inp_ = inputs[0];

        MatShape inpShape = inp_.shape();
        MatShape outShape;

        if (ninputs == 1) {
            outShape = inpShape;
            outShape[2] = zoomFactorHeight > 0 ? (int)(inpShape[2] * zoomFactorHeight) : outHeight0;
            outShape[3] = zoomFactorWidth > 0 ? (int)(inpShape[3] * zoomFactorWidth) : outWidth0;
        } else if (ninputs == 2 && inputs[0].dims == 4 && inputs[1].dims == 4) {
            outShape = inpShape;
            outShape[2] = inputs[1].size[2];
            outShape[3] = inputs[1].size[3];
        } else {
            if (ninputs >= 4) {
                Mat sizesTensor = inputs[3];
                tensorToIntVec(sizesTensor, sizes);
            }
            Mat scalesTensor = inputs[(ninputs == 2) ? 1 : 2];
            tensorToFloatVec(scalesTensor, scales);
            outShape = getOutShape(inpShape, sizes, scales);
        }

        int length_resized_y = outShape[2];
        int length_resized_x = outShape[3];
        updateOutSizeAndScale(inpShape, outShape);

        // Read ROI if dynamicROI is enabled
        if (dynamicROI && coordTransModeE == CoordTransMode::TF_CROP_AND_RESIZE && ninputs >= 2)
        {
            Mat roiTensor = inputs[1];
            std::vector<float> roi;
            tensorToFloatVec(roiTensor, roi);
            if (roi.size() >= 4)
            {
                if (roi.size() == 4) {
                    roi_start_y = roi[0];
                    roi_start_x = roi[1];
                    roi_end_y = roi[2];
                    roi_end_x = roi[3];
                } else if (roi.size() == 6) {
                    roi_start_y = roi[1];
                    roi_start_x = roi[2];
                    roi_end_y = roi[4];
                    roi_end_x = roi[5];
                } else if (roi.size() == 8) {
                    roi_start_y = roi[2];
                    roi_start_x = roi[3];
                    roi_end_y = roi[6];
                    roi_end_x = roi[7];
                }
            }
        }

        if (sizes.empty() && !scales.empty() && halfPixelCenters)
        {
            float sH = (scales.size() == 4) ? scales[2] : scales[0];
            float sW = (scales.size() == 4) ? scales[3] : scales[1];
            scaleHeight = 1.f / sH;
            scaleWidth  = 1.f / sW;
        }

        auto kind = outputs_arr.kind();
        Mat out_;
        UMat uout_;
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outputs = outputs_arr.getMatVecRef();
            outputs[0].fit(outShape, inp_.type());
            out_ = outputs[0];

            if (outShape == inpShape)
            {
                inp_.copyTo(out_);
                return;
            }
        }
        else {
            CV_Assert(kind == _InputArray::STD_VECTOR_UMAT);
            std::vector<UMat>& u_outputs = outputs_arr.getUMatVecRef();
            u_outputs[0].fit(outShape, inp_.type());
            uout_ = u_outputs[0];
            if (outShape == inpShape)
            {
                inp_.copyTo(uout_);
                return;
            }
            out_.create(outShape, inp_.type());
        }

        int depth = inp_.type(), orig_depth = depth;

        Mat inp, out;
        if (depth != CV_32F && depth != CV_8S && depth != CV_8U && depth != CV_16F && depth != CV_16BF) {
            inp_.convertTo(inp, CV_32F);
            out.fit(outShape, CV_32F);
            depth = CV_32F;
        } else {
            inp = inp_;
            out = out_;
        }

        if(interpolation=="nearest"){
            switch(depth){
            case CV_8S:
            case CV_8U:
                resizeNearest<int8_t>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,nearestModeE,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_16F:
                resizeNearest<hfloat>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,nearestModeE,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_16BF:
                resizeNearest<bfloat>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,nearestModeE,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_32F:
                resizeNearest<float>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,nearestModeE,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            default: CV_Error(Error::StsUnsupportedFormat,"Unsupported depth");
            }
        }
        else if(interpolation=="bilinear"||interpolation=="opencv_linear"){
            switch(depth){
            case CV_8S:
                resizeBilinear<int8_t>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_8U:
                resizeBilinear<uint8_t>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_16F:
                resizeBilinear<hfloat>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_16BF:
                resizeBilinear<bfloat>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_32F:
                resizeBilinear<float>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            default: CV_Error(Error::StsUnsupportedFormat,"Unsupported depth");
            }
        }
        else if(interpolation=="cubic"){
            switch (depth) {
            case CV_8S:
                resizeCubic<int8_t>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,cubicCoeffA,excludeOutside,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_8U:
                resizeCubic<uint8_t>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,cubicCoeffA,excludeOutside,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_16F:
                resizeCubic<hfloat>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,cubicCoeffA,excludeOutside,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_16BF:
                resizeCubic<bfloat>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,cubicCoeffA,excludeOutside,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            case CV_32F:
                resizeCubic<float>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,cubicCoeffA,excludeOutside,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
                break;
            default:
                CV_Error(Error::StsUnsupportedFormat, "Unsupported depth");
            }
        }
        else
            CV_Error(Error::StsNotImplemented,"Unknown interpolation: "+interpolation);

        if (orig_depth != depth) {
            if (!uout_.empty())
                out.convertTo(uout_, orig_depth);
            else
                out.convertTo(out_, orig_depth);
        }
        else if (!uout_.empty()) {
            out.copyTo(uout_);
        }
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto x_desc = x->getTensorDesc();
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto output_y_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        // create operator
        if (interpolation == "nearest")
        {
            auto op = std::make_shared<ge::op::ResizeNearestNeighborV2>(name);

            // set attributes
            op->set_attr_align_corners(alignCorners);
            op->set_attr_half_pixel_centers(halfPixelCenters);

            // set inputs : x
            op->set_input_x_by_name(*op_x, x->name.c_str());
            op->update_input_desc_x(*x_desc);
            // set inputs : size
            std::vector<int> shape_of_size_mat{2};
            std::vector<int> size_vec{outHeight, outWidth};
            Mat size_mat(shape_of_size_mat, CV_32S, size_vec.data());
            auto op_const_size = std::make_shared<CannConstOp>(size_mat.data, size_mat.type(), shape_of_size_mat, cv::format("%s_size", name.c_str()));
            op->set_input_size(*(op_const_size->getOp()));
            op->update_input_desc_size(*(op_const_size->getTensorDesc()));

            // set outputs
            op->update_output_desc_y(*output_y_desc);

            return Ptr<BackendNode>(new CannBackendNode(op));
        }
        else if (interpolation == "opencv_linear" || interpolation == "bilinear")
        {
            auto op = std::make_shared<ge::op::ResizeBilinearV2D>(name);

            // set attributes
            op->set_attr_align_corners(alignCorners);
            op->set_attr_half_pixel_centers(halfPixelCenters);
            std::vector<int64_t> taget_size{(int64_t)outHeight, (int64_t)outWidth};
            op->set_attr_size(taget_size);

            // set inputs : x
            op->set_input_x_by_name(*op_x, x->name.c_str());
            op->update_input_desc_x(*x_desc);

            // set outputs
            op->update_output_desc_y(*output_y_desc);

            return Ptr<BackendNode>(new CannBackendNode(op));
        }
        else
            CV_Error(Error::StsNotImplemented, "Unsupported interpolation by CANN backend: " + interpolation);
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;

        ov::op::v4::Interpolate::InterpolateAttrs attrs;

        if (interpolation == "nearest") {
            attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        } else if (interpolation == "bilinear") {
            attrs.mode = ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX;
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        } else {
            CV_Error(Error::StsNotImplemented, format("Unsupported interpolation: %s", interpolation.c_str()));
        }
        attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SIZES;

        CV_Assert(!halfPixelCenters || !alignCorners);
        if (halfPixelCenters) {
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        } else if (alignCorners) {
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        }

        attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;


        std::vector<int64_t> shape = {outHeight, outWidth};
        auto out_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, shape.data());

        auto& input_shape = ieInpNode.get_shape();
        CV_Assert_N(input_shape[2] != 0, input_shape[3] != 0);
        std::vector<float> scales = {static_cast<float>(outHeight) / input_shape[2], static_cast<float>(outWidth) / input_shape[3]};
        auto scales_shape = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, scales.data());

        auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
        auto interp = std::make_shared<ov::op::v4::Interpolate>(ieInpNode, out_shape, scales_shape, axes, attrs);
        return Ptr<BackendNode>(new InfEngineNgraphNode(interp));
    }
#endif  // HAVE_DNN_NGRAPH


#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        cuda4dnn::ResizeConfiguration config;
        if (interpolation == "nearest")
        {
            config.type = InterpolationType::NEAREST_NEIGHBOUR;
            config.align_corners = alignCorners;
            config.half_pixel_centers = halfPixelCenters;
        }
        else if (interpolation == "bilinear")
        {
            config.type = InterpolationType::BILINEAR;
            config.align_corners = alignCorners;
            config.half_pixel_centers = halfPixelCenters;
        }
        else if (interpolation == "opencv_linear")
        {
            config.type = InterpolationType::BILINEAR;
            config.align_corners = false;
            config.half_pixel_centers = true;
        }
        else
            CV_Error(Error::StsNotImplemented, "Requested interpolation mode is not available in resize layer.");
        return make_cuda_node<cuda4dnn::ResizeOp>(preferableTarget, std::move(context->stream), config);
    }
#endif

protected:
    int outWidth, outHeight;
    const float zoomFactorWidth, zoomFactorHeight;
    String interpolation;
    float scaleWidth, scaleHeight;
    bool alignCorners;
    bool dynamicROI;
    bool halfPixelCenters;
    String coordTransMode;
    CoordTransMode coordTransModeE;
    NearestMode nearestModeE;  // ONNX "nearest_mode" attribute
    bool excludeOutside; // ONNX attribute for cubic
    float cubicCoeffA;
    float roi_start_y, roi_end_y, roi_start_x, roi_end_x;
    float extrapolation_value; // Extrapolation value for tf_crop_and_resize mode
};

Ptr<Resize2Layer> Resize2Layer::create(const LayerParams& params)
{
    return Ptr<Resize2Layer>(new Resize2LayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
