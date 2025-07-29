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

class Resize2LayerImpl : public Resize2Layer
{
public:
    int outWidth0, outHeight0;
    Resize2LayerImpl(const LayerParams& params) : zoomFactorWidth(params.get<float>("zoom_factor_x", params.get<float>("zoom_factor", 0))),
                                                 zoomFactorHeight(params.get<float>("zoom_factor_y", params.get<float>("zoom_factor", 0))),
                                                 scaleWidth(0), scaleHeight(0), cubicCoeffA(params.get<float>("cubic_coeff_a", -0.75f))
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

        alignCorners = params.get<bool>("align_corners", false);
        halfPixelCenters = params.get<bool>("half_pixel_centers", false);
        if (interpolation == "opencv_linear")
            halfPixelCenters = true;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
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
                outShape[2] = (float)(inpShape[2] * scales[2]);
                outShape[3] = (float)(inpShape[3] * scales[3]);
            } else /* scales.size() == 2 */ {
                outShape[2] = (float)(inpShape[2] * scales[0]);
                outShape[3] = (float)(inpShape[3] * scales[1]);
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
            std::vector<int> sizes;
            std::vector<float> scales;
            if (ninputs >= 4) {
                Mat sizesTensor = inputs[3];
                tensorToIntVec(sizesTensor, sizes);
            }
            Mat scalesTensor = inputs[(ninputs == 2) ? 1 : 2];
            tensorToFloatVec(scalesTensor, scales);
            outShape = getOutShape(inpShape, sizes, scales);
        }

        //printf("name: %s, outShape: %d x %d x %d x %d\n", name.c_str(), outShape[0], outShape[1], outShape[2], outShape[3]);

        updateOutSizeAndScale(inpShape, outShape);

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

        if (interpolation == "nearest")
        {
            const int inpHeight = inp.size[2];
            const int inpWidth = inp.size[3];
            const int inpSpatialSize = inpHeight * inpWidth;
            const int outSpatialSize = outHeight * outWidth;
            const int numPlanes = inp.size[0] * inp.size[1];
            CV_Assert_N(inp.isContinuous(), out.isContinuous());

            Mat inpPlanes = inp.reshape(1, numPlanes * inpHeight);
            Mat outPlanes = out.reshape(1, numPlanes * outHeight);

            auto compute_input_coord = [&](int dst, float scale, int limit) -> float {
                if (halfPixelCenters)
                    return (dst + 0.5f) * scale - 0.5f;
                return dst * scale;
            };

            auto nearest_index = [&](float src, int limit) -> int {
                int idx = 0;
                if (nearestMode == "floor")
                {
                    idx = static_cast<int>(std::floor(src));
                }
                else if (nearestMode == "ceil")
                {
                    idx = static_cast<int>(std::ceil(src));
                }
                else if (nearestMode == "round_prefer_ceil")
                {
                    float floor_v = std::floor(src);
                    float frac = src - floor_v;
                    if (frac > 0.5f)
                        idx = floor_v + 1;
                    else if (frac < 0.5f)
                        idx = static_cast<int>(floor_v);
                    else  // exactly 0.5
                        idx = static_cast<int>(floor_v + 1);
                }
                else /* round_prefer_floor (default) */
                {
                    float floor_v = std::floor(src);
                    float frac = src - floor_v;
                    if (frac > 0.5f)
                        idx = floor_v + 1;
                    else if (frac < 0.5f)
                        idx = static_cast<int>(floor_v);
                    else  // exactly 0.5, prefer floor
                        idx = static_cast<int>(floor_v);
                }
                if (idx < 0) idx = 0;
                if (idx > limit) idx = limit;
                return idx;
            };

            if (depth == CV_8S)
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    float input_y_f = compute_input_coord(y, scaleHeight, inpHeight - 1);
                    int y0 = nearest_index(input_y_f, inpHeight - 1);

                    const int8_t* inpData_row = inpPlanes.ptr<int8_t>(y0);

                    for (int x = 0; x < outWidth; ++x)
                    {
                        float input_x_f = compute_input_coord(x, scaleWidth, inpWidth - 1);
                        int x0 = nearest_index(input_x_f, inpWidth - 1);

                        int8_t* outData = outPlanes.ptr<int8_t>(y, x);
                        const int8_t* inpData_row_c = inpData_row;

                        for (int c = 0; c < numPlanes; ++c)
                        {
                            *outData = inpData_row_c[x0];

                            inpData_row_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    float input_y_f = compute_input_coord(y, scaleHeight, inpHeight - 1);
                    int y0 = nearest_index(input_y_f, inpHeight - 1);

                    const float* inpData_row = inpPlanes.ptr<float>(y0);

                    for (int x = 0; x < outWidth; ++x)
                    {
                        float input_x_f = compute_input_coord(x, scaleWidth, inpWidth - 1);
                        int x0 = nearest_index(input_x_f, inpWidth - 1);

                        float* outData = outPlanes.ptr<float>(y, x);
                        const float* inpData_row_c = inpData_row;

                        for (int c = 0; c < numPlanes; ++c)
                        {
                            *outData = inpData_row_c[x0];

                            inpData_row_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
        }
        else if (interpolation == "bilinear" || interpolation == "opencv_linear")
        {
            const int inpHeight = inp.size[2];
            const int inpWidth = inp.size[3];
            const int inpSpatialSize = inpHeight * inpWidth;
            const int outSpatialSize = outHeight * outWidth;
            const int numPlanes = inp.size[0] * inp.size[1];
            CV_Assert_N(inp.isContinuous(), out.isContinuous());

            Mat inpPlanes = inp.reshape(1, numPlanes * inpHeight);
            Mat outPlanes = out.reshape(1, numPlanes * outHeight);
            auto clampInt = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
            auto compute_input_coord_clamped = [&](int dst, float scale, int limit) -> std::pair<int,float> {
                float src = halfPixelCenters ? (dst + 0.5f) * scale - 0.5f : dst * scale;
                if (src < 0.f) src = 0.f;
                if (src > limit - 1.f) src = limit - 1.f - 1e-6f; // keep within range so that x1/y1 exists
                int i0 = static_cast<int>(std::floor(src));
                float l = src - i0;
                return {i0, l};
            };

            if (depth == CV_8S)
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    auto cy = compute_input_coord_clamped(y, scaleHeight, inpHeight);
                    int y0 = cy.first;
                    int y1 = clampInt(y0 + 1, 0, inpHeight - 1);
                    float ly = cy.second;

                    const int8_t* inpData_row0 = inpPlanes.ptr<int8_t>(y0);
                    const int8_t* inpData_row1 = inpPlanes.ptr<int8_t>(y1);

                    for (int x = 0; x < outWidth; ++x)
                    {
                        auto cx = compute_input_coord_clamped(x, scaleWidth, inpWidth);
                        int x0 = cx.first;
                        int x1 = clampInt(x0 + 1, 0, inpWidth - 1);
                        float lx = cx.second;

                        int8_t* outData = outPlanes.ptr<int8_t>(y, x);
                        const int8_t* inpData_row0_c = inpData_row0;
                        const int8_t* inpData_row1_c = inpData_row1;
                        for (int c = 0; c < numPlanes; ++c)
                        {
                            float top = inpData_row0_c[x0] + lx * (inpData_row0_c[x1] - inpData_row0_c[x0]);
                            float bottom = inpData_row1_c[x0] + lx * (inpData_row1_c[x1] - inpData_row1_c[x0]);
                            *outData = static_cast<int8_t>(top + ly * (bottom - top));

                            inpData_row0_c += inpSpatialSize;
                            inpData_row1_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    auto cy = compute_input_coord_clamped(y, scaleHeight, inpHeight);
                    int y0 = cy.first;
                    int y1 = clampInt(y0 + 1, 0, inpHeight - 1);
                    float ly = cy.second;

                    const float* inpData_row0 = inpPlanes.ptr<float>(y0);
                    const float* inpData_row1 = inpPlanes.ptr<float>(y1);

                    for (int x = 0; x < outWidth; ++x)
                    {
                        auto cx = compute_input_coord_clamped(x, scaleWidth, inpWidth);
                        int x0 = cx.first;
                        int x1 = clampInt(x0 + 1, 0, inpWidth - 1);
                        float lx = cx.second;

                        float* outData = outPlanes.ptr<float>(y, x);
                        const float* inpData_row0_c = inpData_row0;
                        const float* inpData_row1_c = inpData_row1;
                        for (int c = 0; c < numPlanes; ++c)
                        {
                            float top = inpData_row0_c[x0] + lx * (inpData_row0_c[x1] - inpData_row0_c[x0]);
                            float bottom = inpData_row1_c[x0] + lx * (inpData_row1_c[x1] - inpData_row1_c[x0]);
                            *outData = top + ly * (bottom - top);

                            inpData_row0_c += inpSpatialSize;
                            inpData_row1_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
        }
        else if (interpolation == "cubic")
        {
            if (depth == CV_8S)
            {
                inp.convertTo(inp, CV_32F);
                out.convertTo(out, CV_32F);
                depth = CV_32F;
            }

            CV_Assert(depth == CV_32F);

            const int inpHeight = inp.size[2];
            const int inpWidth = inp.size[3];
            const int numPlanes = inp.size[0] * inp.size[1];

            auto clampInt = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };

            auto cubic = [this](float x) {
                float a = cubicCoeffA;
                x = std::abs(x);
                if (x < 1.f)
                    return (a + 2.f) * x * x * x - (a + 3.f) * x * x + 1.f;
                else if (x < 2.f)
                    return a * x * x * x - 5.f * a * x * x + 8.f * a * x - 4.f * a;
                return 0.f;
            };

            std::vector<std::array<int, 4>> x_id(outWidth);
            std::vector<std::array<float, 4>> x_w(outWidth);
            for (int ox = 0; ox < outWidth; ++ox)
            {
                float src_x = alignCorners ? ox * scaleWidth : (halfPixelCenters ? (ox + 0.5f) * scaleWidth - 0.5f : ox * scaleWidth);
                int ix = static_cast<int>(std::floor(src_x));
                float dx = src_x - ix;
                float sw = 0.f;
                for (int k = -1; k <= 2; ++k)
                {
                    int src_idx = ix + k;
                    float w = cubic(k - dx);
                    if (src_idx < 0 || src_idx >= inpWidth)
                    {
                        if (excludeOutside)
                        {
                            x_id[ox][k + 1] = -1;
                            x_w[ox][k + 1] = 0.f;
                        }
                        else
                        {
                            src_idx = clampInt(src_idx, 0, inpWidth - 1);
                            x_id[ox][k + 1] = src_idx;
                            x_w[ox][k + 1] = w;
                            sw += w;
                        }
                    }
                    else
                    {
                        x_id[ox][k + 1] = src_idx;
                        x_w[ox][k + 1] = w;
                        sw += w;
                    }
                }
                if (sw != 0.f)
                    for (int k = 0; k < 4; ++k)
                        x_w[ox][k] /= sw;
            }

            std::vector<std::array<int, 4>> y_id(outHeight);
            std::vector<std::array<float, 4>> y_w(outHeight);
            for (int oy = 0; oy < outHeight; ++oy)
            {
                float src_y = alignCorners ? oy * scaleHeight : (halfPixelCenters ? (oy + 0.5f) * scaleHeight - 0.5f : oy * scaleHeight);
                int iy = static_cast<int>(std::floor(src_y));
                float dy = src_y - iy;
                float swy = 0.f;
                for (int k = -1; k <= 2; ++k)
                {
                    int src_idy = iy + k;
                    float w = cubic(k - dy);
                    if (src_idy < 0 || src_idy >= inpHeight)
                    {
                        if (excludeOutside)
                        {
                            y_id[oy][k + 1] = -1;
                            y_w[oy][k + 1] = 0.f;
                        }
                        else
                        {
                            src_idy = clampInt(src_idy, 0, inpHeight - 1);
                            y_id[oy][k + 1] = src_idy;
                            y_w[oy][k + 1] = w;
                            swy += w;
                        }
                    }
                    else
                    {
                        y_id[oy][k + 1] = src_idy;
                        y_w[oy][k + 1] = w;
                        swy += w;
                    }
                }
                if (swy != 0.f)
                    for (int k = 0; k < 4; ++k)
                        y_w[oy][k] /= swy;
            }

            Mat inpPlanes = inp.reshape(1, numPlanes * inpHeight);
            Mat outPlanes = out.reshape(1, numPlanes * outHeight);

            for (int p = 0; p < numPlanes; ++p)
            {
                const float* inpBase = inpPlanes.ptr<float>(p * inpHeight);
                float* outBase = outPlanes.ptr<float>(p * outHeight);
                for (int oy = 0; oy < outHeight; ++oy)
                {
                    float* outRow = outBase + oy * outWidth;
                    for (int ox = 0; ox < outWidth; ++ox)
                    {
                        float val = 0.f;
                        bool hasValidContribution = false;
                        for (int ky = 0; ky < 4; ++ky)
                        {
                            if (y_id[oy][ky] == -1) continue;
                            const float* inRow = inpBase + y_id[oy][ky] * inpWidth;
                            float sumx = 0.f;
                            for (int kx = 0; kx < 4; ++kx)
                            {
                                if (x_id[ox][kx] == -1) continue;
                                sumx += x_w[ox][kx] * inRow[x_id[ox][kx]];
                                hasValidContribution = true;
                            }
                            val += y_w[oy][ky] * sumx;
                        }
                        outRow[ox] = hasValidContribution ? val : 0.f;
                    }
                }
            }
        }

        else
            CV_Error(Error::StsNotImplemented, "Unknown interpolation: " + interpolation);

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
    bool halfPixelCenters;
    String nearestMode;  // ONNX "nearest_mode" attribute
    bool excludeOutside; // ONNX attribute for cubic
    float cubicCoeffA;
};

Ptr<Resize2Layer> Resize2Layer::create(const LayerParams& params)
{
    return Ptr<Resize2Layer>(new Resize2LayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
