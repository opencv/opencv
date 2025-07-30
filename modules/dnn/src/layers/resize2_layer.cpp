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
#include <algorithm>

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

inline float computeSrcGeneric(int dst, float scale, int limit, int len,
                               const String &coordTransMode, bool halfPixelCenters,
                               float start_coord = 0.0f, float end_coord = 1.0f)
{
    if (coordTransMode == "tf_crop_and_resize")
    {
        if (len > 1)
            return start_coord * (limit - 1) + dst * (end_coord - start_coord) * (limit - 1) / float(len - 1);
        else
            return 0.5f * (start_coord + end_coord) * (limit - 1);
    }
    if (coordTransMode == "pytorch_half_pixel")
        return (len > 1) ? (dst + 0.5f)*scale - 0.5f : 0.f;
    if (coordTransMode == "half_pixel")
        return (dst + 0.5f)*scale - 0.5f;
    if (coordTransMode == "tf_half_pixel_for_nn")
        return (dst + 0.5f)*scale;
    return dst*scale;
}

template<typename T>
void resizeNearest(const Mat &inp, Mat &out,
                   float scaleH, float scaleW,
                   int lenY, int lenX,
                   const String &nearestMode,
                   const String &coordTransMode,
                   bool halfPixelCenters,
                   float start_y = 0.0f, float end_y = 1.0f,
                   float start_x = 0.0f, float end_x = 1.0f,
                   float extrapolation_value = 0.0f)
{
    int numPlanes = inp.size[0] * inp.size[1];
    int inH       = inp.size[2], inW       = inp.size[3];
    int outH      = out.size[2], outW      = out.size[3];
    int inPlane   = inH * inW,    outPlane = outH * outW;
    CV_Assert(inp.isContinuous() && out.isContinuous());

    Mat inpP = inp.reshape(1, numPlanes * inH);
    Mat outP = out.reshape(1, numPlanes * outH);

    auto comp = [&](int dst, float scale, int limit, int len, float start_coord, float end_coord) {
        float src = computeSrcGeneric(dst, scale, limit, len,
                                      coordTransMode, halfPixelCenters, start_coord, end_coord);
        if (coordTransMode == "tf_crop_and_resize") {
            return src; // Don't clamp for tf_crop_and_resize
        }
        return std::min(std::max(src, 0.f), float(limit));
    };
    auto nidx = [&](float src, int lim) {
        float fv = std::floor(src), frac = src - fv;
        int   idx;
        if      (nearestMode == "floor")            idx = int(fv);
        else if (nearestMode == "ceil")             idx = int(std::ceil(src));
        else if (nearestMode == "round_prefer_ceil")
            idx = (frac >= 0.5f ? int(fv+1) : int(fv));
        else /* round_prefer_floor */               idx = (frac >  0.5f ? int(fv+1) : int(fv));
        return std::clamp(idx, 0, lim);
    };

    std::vector<int> mapY(outH);
    for (int y = 0; y < outH; ++y)
    {
        float yf = comp(y, scaleH, inH - 1, lenY, start_y, end_y);
        if (coordTransMode == "tf_crop_and_resize" && (yf < 0.f || yf >= float(inH))) {
            mapY[y] = -1; // Mark as out of bounds
        } else {
            mapY[y] = nidx(yf, inH - 1);
        }
    }

    std::vector<int> mapX(outW);
    for (int x = 0; x < outW; ++x)
    {
        float xf = comp(x, scaleW, inW - 1, lenX, start_x, end_x);
        if (coordTransMode == "tf_crop_and_resize" && (xf < 0.f || xf >= float(inW))) {
            mapX[x] = -1; // Mark as out of bounds
        } else {
            mapX[x] = nidx(xf, inW - 1);
        }
    }

    for (int y = 0; y < outH; ++y)
    {
        const T* inpRow = inpP.ptr<T>( mapY[y] );
        T*       outRow = outP.ptr<T>( y );
        for (int x = 0; x < outW; ++x)
        {
            const T* srcPtr = inpRow + mapX[x];
            T*       dst    = outRow + x;
            for (int p = 0; p < numPlanes; ++p)
            {
                if (coordTransMode == "tf_crop_and_resize" && (mapY[y] == -1 || mapX[x] == -1)) {
                    *dst = T(extrapolation_value);
                } else {
                    *dst = *srcPtr;
                }
                srcPtr += inPlane;
                dst    += outPlane;
            }
        }
    }
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
    int inPlane   = inH * inW, outPlane = outH * outW;
    CV_Assert(inp.isContinuous() && out.isContinuous());

    Mat inpP = inp.reshape(1, numPlanes*inH);
    Mat outP = out.reshape(1, numPlanes*outH);

    auto clampC = [&](int dst, float scale, int lim, int len, float start_coord, float end_coord) {
        float src = computeSrcGeneric(dst, scale, lim, len,
                                      coordTransMode, halfPixelCenters, start_coord, end_coord);
        if (coordTransMode == "tf_crop_and_resize") {
            int i0 = int(std::floor(src));
            return std::make_pair(i0, src - float(i0));
        } else {
            src = std::min(std::max(src, 0.f), float(lim-1) - 1e-6f);
            int i0 = int(std::floor(src));
            return std::make_pair(i0, src - float(i0));
        }
    };

    std::vector<int>    x0(outW), x1(outW);
    std::vector<float> lx(outW);
    std::vector<bool>  outOfBoundsX(outW, false);
    for (int x = 0; x < outW; ++x)
    {
        auto [xi, xfrac] = clampC(x, scaleW, inW, lenX, start_x, end_x);
        if (coordTransMode == "tf_crop_and_resize" && (xi < 0 || xi >= inW - 1)) {
            outOfBoundsX[x] = true;
            x0[x] = 0;
            x1[x] = 0;
            lx[x] = 0.0f;
        } else {
            x0[x] = xi;
            x1[x] = std::clamp(xi + 1, 0, inW - 1);
            lx[x] = xfrac;
        }
    }

    std::vector<int>    y0(outH), y1(outH);
    std::vector<float> ly(outH);
    std::vector<bool>  outOfBoundsY(outH, false);
    for (int y = 0; y < outH; ++y)
    {
        auto [yi, yfrac] = clampC(y, scaleH, inH, lenY, start_y, end_y);
        if (coordTransMode == "tf_crop_and_resize" && (yi < 0 || yi >= inH - 1)) {
            outOfBoundsY[y] = true;
            y0[y] = 0;
            y1[y] = 0;
            ly[y] = 0.0f;
        } else {
            y0[y] = yi;
            y1[y] = std::clamp(yi + 1, 0, inH - 1);
            ly[y] = yfrac;
        }
    }

    for (int y = 0; y < outH; ++y)
    {
        const T* row0 = inpP.ptr<T>( y0[y] );
        const T* row1 = inpP.ptr<T>( y1[y] );
        float    fy   = ly[y];

        T* outRowBase = outP.ptr<T>( y );

        for (int x = 0; x < outW; ++x)
        {
            int   xi = x0[x];
            float fx = lx[x];

            T* dst = outRowBase + x;
            for (int p = 0; p < numPlanes; ++p)
            {
                if (coordTransMode == "tf_crop_and_resize" && (outOfBoundsY[y] || outOfBoundsX[x])) {
                    *dst = T(extrapolation_value);
                } else {
                    const T* c0 = row0 + p * inPlane + xi;
                    const T* c1 = row1 + p * inPlane + xi;
                    float top    = c0[0] + fx * (c0[1] - c0[0]);
                    float bottom = c1[0] + fx * (c1[1] - c1[0]);
                    *dst = T(top + fy * (bottom - top));
                }
                dst += outPlane;
            }
        }
    }
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
    CV_Assert(inp.depth() == CV_32F);
    int numPlanes = inp.size[0] * inp.size[1];
    int inH = inp.size[2], inW = inp.size[3];
    int outH = out.size[2], outW = out.size[3];

    Mat inpPlanes = inp.reshape(1, numPlanes * inH);
    Mat outPlanes = out.reshape(1, numPlanes * outH);

    auto clampInt = [](int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };
    auto cubicWeight = [&](float x) {
        float a = cubicA;
        x = std::abs(x);
        if (x < 1.f)
            return (a + 2.f) * x*x*x - (a + 3.f) * x*x + 1.f;
        else if (x < 2.f)
            return a * x*x*x - 5.f*a * x*x + 8.f*a * x - 4.f*a;
        return 0.f;
    };

    std::vector<std::array<int,4>> x_id(outW);
    std::vector<std::array<float,4>> x_w (outW);
    std::vector<bool> outOfBoundsX(outW, false);
    for (int ox = 0; ox < outW; ++ox)
    {
        float src_x = computeSrcGeneric(ox, scaleW, inW, lenX, coordTransMode, halfPixelCenters, start_x, end_x);
        int ix = int(std::floor(src_x));
        float dx = src_x - ix;
        float sw = 0.f;
        bool hasOutOfBounds = false;
        for (int k = -1; k <= 2; ++k)
        {
            float w = cubicWeight(k - dx);
            int idx = ix + k;
            if (coordTransMode == "tf_crop_and_resize") {
                if (idx < 0 || idx >= inW) {
                    hasOutOfBounds = true;
                    x_id[ox][k+1] = -1;
                    x_w [ox][k+1] = 0.f;
                } else {
                    x_id[ox][k+1] = idx;
                    x_w [ox][k+1] = w;
                    sw += w;
                }
            } else if (idx < 0 || idx >= inW) {
                if (excludeOutside) {
                    x_id[ox][k+1] = -1;
                    x_w [ox][k+1] = 0.f;
                } else {
                    idx = clampInt(idx, 0, inW - 1);
                    x_id[ox][k+1] = idx;
                    x_w [ox][k+1] = w;
                    sw += w;
                }
            } else {
                x_id[ox][k+1] = idx;
                x_w [ox][k+1] = w;
                sw += w;
            }
        }
        if (sw != 0.f)
            for (int k = 0; k < 4; ++k)
                x_w[ox][k] /= sw;
        if (coordTransMode == "tf_crop_and_resize" && hasOutOfBounds) {
            outOfBoundsX[ox] = true;
        }
    }

    std::vector<std::array<int,4>> y_id(outH);
    std::vector<std::array<float,4>> y_w (outH);
    std::vector<bool> outOfBoundsY(outH, false);
    for (int oy = 0; oy < outH; ++oy)
    {
        float src_y = computeSrcGeneric(oy, scaleH, inH, lenY, coordTransMode, halfPixelCenters, start_y, end_y);
        int iy = int(std::floor(src_y));
        float dy = src_y - iy;
        float swy = 0.f;
        bool hasOutOfBounds = false;
        for (int k = -1; k <= 2; ++k)
        {
            float w = cubicWeight(k - dy);
            int idy = iy + k;
            if (coordTransMode == "tf_crop_and_resize") {
                if (idy < 0 || idy >= inH) {
                    hasOutOfBounds = true;
                    y_id[oy][k+1] = -1;
                    y_w [oy][k+1] = 0.f;
                } else {
                    y_id[oy][k+1] = idy;
                    y_w [oy][k+1] = w;
                    swy += w;
                }
            } else if (idy < 0 || idy >= inH) {
                if (excludeOutside) {
                    y_id[oy][k+1] = -1;
                    y_w [oy][k+1] = 0.f;
                } else {
                    idy = clampInt(idy, 0, inH - 1);
                    y_id[oy][k+1] = idy;
                    y_w [oy][k+1] = w;
                    swy += w;
                }
            } else {
                y_id[oy][k+1] = idy;
                y_w [oy][k+1] = w;
                swy += w;
            }
        }
        if (swy != 0.f)
            for (int k = 0; k < 4; ++k)
                y_w[oy][k] /= swy;
        if (coordTransMode == "tf_crop_and_resize" && hasOutOfBounds) {
            outOfBoundsY[oy] = true;
        }
    }

    const int R = numPlanes * outH;
    parallel_for_(Range(0, R), [&](const Range& range) {
        for (int i = range.start; i < range.end; ++i)
        {
            int p  = i / outH;    // plane
            int oy = i % outH;    // output row
            const float* inpBase = inpPlanes.ptr<float>(p * inH);
            float*       outRow  = outPlanes.ptr<float>(p * outH) + oy * outW;

            for (int ox = 0; ox < outW; ++ox)
            {
                if (coordTransMode == "tf_crop_and_resize" && (outOfBoundsY[oy] || outOfBoundsX[ox])) {
                    outRow[ox] = extrapolation_value;
                } else {
                    float val = 0.f;
                    bool  hasValid = false;
                    for (int ky = 0; ky < 4; ++ky)
                    {
                        int yy = y_id[oy][ky];
                        if (yy < 0) continue;
                        const float* row = inpBase + yy * inW;
                        for (int kx = 0; kx < 4; ++kx)
                        {
                            int xx = x_id[ox][kx];
                            if (xx < 0) continue;
                            val      += y_w[oy][ky] * x_w[ox][kx] * row[xx];
                            hasValid  = true;
                        }
                    }
                    outRow[ox] = hasValid ? val : 0.f;
                }
            }
        }
    });
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
        nearestMode = params.get<String>("nearest_mode", "round_prefer_floor");
        CV_Check(interpolation, interpolation == "nearest" || interpolation == "opencv_linear" || interpolation == "bilinear" || interpolation == "cubic", "");

        excludeOutside = params.get<bool>("exclude_outside", false);
        dynamicROI = params.get<bool>("dynamic_roi", false);

        alignCorners = params.get<bool>("align_corners", false);
        halfPixelCenters = params.get<bool>("half_pixel_centers", false);
        coordTransMode = params.get<String>("coordinate_transformation_mode", "half_pixel");

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
            // [TODO] this workaround needs to be removed
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
            return interpolation == "nearest" || interpolation == "bilinear" || interpolation == "opencv_linear";

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
        if (dynamicROI && coordTransMode == "tf_crop_and_resize" && ninputs >= 2)
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
        if (depth != CV_32F && depth != CV_8S) {
            inp_.convertTo(inp, CV_32F);
            out.fit(outShape, CV_32F);
            depth = CV_32F;
        } else {
            inp = inp_;
            out = out_;
        }

        if(interpolation=="nearest"){
            switch(depth){
            case CV_8S:  resizeNearest<int8_t>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,nearestMode,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);break;
            case CV_32F: resizeNearest<float>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,nearestMode,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);break;
            default: CV_Error(Error::StsUnsupportedFormat,"Nearest supports only CV_8S & CV_32F");
            }
        }
        else if(interpolation=="bilinear"||interpolation=="opencv_linear"){
            switch(depth){
            case CV_8S:  resizeBilinear<int8_t>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);break;
            case CV_32F: resizeBilinear<float>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);break;
            default: CV_Error(Error::StsUnsupportedFormat,"Bilinear supports only CV_8S & CV_32F");
            }
        }
        else if(interpolation=="cubic"){
            if(depth!=CV_32F){inp.convertTo(inp,CV_32F);out.convertTo(out,CV_32F);}
            resizeCubic<float>(inp,out,scaleHeight,scaleWidth,length_resized_y,length_resized_x,cubicCoeffA,excludeOutside,coordTransMode,halfPixelCenters,roi_start_y,roi_end_y,roi_start_x,roi_end_x,extrapolation_value);
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
    String nearestMode;  // ONNX "nearest_mode" attribute
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
