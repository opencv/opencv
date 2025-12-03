// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

// ONNX RoiAlign operator
// Spec: https://onnx.ai/onnx/operators/onnx__RoiAlign.html
// Supported opsets: 10-22

namespace {

enum class RoiAlignMode
{
    AVG = 0,
    MAX = 1
};

enum class CoordTransformMode
{
    HALF_PIXEL = 0,
    OUTPUT_HALF_PIXEL = 1
};

template<typename T, bool MaxMode>
class RoiAlignForwardInvoker CV_FINAL : public ParallelLoopBody
{
public:
    RoiAlignForwardInvoker(const Mat& X_, Mat& Y_,
                           const Mat& rois_, const Mat& batch_indices_,
                           const int* Xsize,
                           int pooled_h_, int pooled_w_, int sampling_ratio_,
                           float spatial_scale_, float offset_,
                           bool clampMalformedRoi_)
            : X(X_), Y(Y_), rois(rois_), batch_indices(batch_indices_),
              output_height(pooled_h_), output_width(pooled_w_), sampling_ratio(sampling_ratio_),
              spatial_scale(spatial_scale_), offset(offset_),
              clampMalformedRoi(clampMalformedRoi_)
    {
        CV_Assert(X.isContinuous() && Y.isContinuous());
        CV_Assert(rois.isContinuous() && batch_indices.isContinuous());
        CV_Assert(Xsize);
        N = Xsize[0]; C = Xsize[1]; H = Xsize[2]; W = Xsize[3];

        Xdata = X.ptr<T>();
        Ydata = Y.ptr<T>();

        spatialSize = static_cast<size_t>(H) * W;
        channelStride = spatialSize;
        batchStride = static_cast<size_t>(C) * spatialSize;
        outSpatial = output_height * output_width;

        Rdepth = rois.depth();

        const int bdepth = batch_indices.depth();
        if (bdepth == CV_32S)
        {
            batch32 = batch_indices.ptr<int>();
        }
        else if (bdepth == CV_64S)
        {
            batch64 = batch_indices.ptr<int64>();
        }
        else
        {
            CV_Error(Error::StsUnsupportedFormat, "Unsupported batch_indices depth (expected int32 or int64)");
        }
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        for (int r = range.start; r < range.end; ++r)
        {
            const int b = batch64 ? static_cast<int>(batch64[r]) : batch32[r];
            CV_Assert(0 <= b && b < N);

            float x1, y1, x2, y2;
            if (Rdepth == CV_32F)
                readScaledRoiCoords<float>(rois, r, spatial_scale, offset, x1, y1, x2, y2);
            else if (Rdepth == CV_64F)
                readScaledRoiCoords<double>(rois, r, spatial_scale, offset, x1, y1, x2, y2);
            else if (Rdepth == CV_16F)
                readScaledRoiCoords<hfloat>(rois, r, spatial_scale, offset, x1, y1, x2, y2);
            else if (Rdepth == CV_16BF)
                readScaledRoiCoords<bfloat>(rois, r, spatial_scale, offset, x1, y1, x2, y2);
            else
                CV_Error(Error::StsUnsupportedFormat, "Unsupported rois depth");

            float roi_width = x2 - x1;
            float roi_height = y2 - y1;

            if (clampMalformedRoi)
            {
                roi_width = std::max(roi_width, 1.f);
                roi_height = std::max(roi_height, 1.f);
            }

            const float bin_size_w = roi_width / output_width;
            const float bin_size_h = roi_height / output_height;

            const int gw = std::max((sampling_ratio > 0) ? sampling_ratio
                                 : cvCeil(roi_width / output_width), 1);
            const int gh = std::max((sampling_ratio > 0) ? sampling_ratio
                                 : cvCeil(roi_height / output_height), 1);
            const int sample_count_i = gw * gh;
            const float inv_sample_count = 1.f / static_cast<float>(sample_count_i);
            const float step_y = bin_size_h / gh;
            const float step_x = bin_size_w / gw;

            AutoBuffer<BilinearSample> samples(static_cast<size_t>(outSpatial) * sample_count_i);
            for (int outIdx = 0; outIdx < outSpatial; ++outIdx)
            {
                const int ph = outIdx / output_width;
                const int pw = outIdx - ph * output_width;
                const float ph_off = ph * bin_size_h;
                const float pw_off = pw * bin_size_w;

                BilinearSample* dst = samples.data() + static_cast<size_t>(outIdx) * sample_count_i;
                for (int s = 0; s < sample_count_i; ++s)
                {
                    const int iy = s / gw;
                    const int ix = s - iy * gw;
                    const float yy = y1 + ph_off + (iy + 0.5f) * step_y;
                    const float xx = x1 + pw_off + (ix + 0.5f) * step_x;
                    dst[s] = precomputeBilinearSample(H, W, yy, xx);
                }
            }

            const size_t y_roi_base = static_cast<size_t>(r) * static_cast<size_t>(C) * outSpatial;
            const size_t x_batch_base = static_cast<size_t>(b) * batchStride;

            for (int outPlane = 0; outPlane < C * outSpatial; ++outPlane)
            {
                const int c = outPlane / outSpatial;
                const int outIdx = outPlane - c * outSpatial;

                const T* img = Xdata + x_batch_base + static_cast<size_t>(c) * channelStride;
                const BilinearSample* src = samples.data() + static_cast<size_t>(outIdx) * sample_count_i;

                float outv = MaxMode ? -FLT_MAX : 0.f;
                for (int s = 0; s < sample_count_i; ++s)
                {
                    const BilinearSample& bs = src[s];
                    const float v00 = static_cast<float>(img[bs.idx00]);
                    const float v01 = static_cast<float>(img[bs.idx01]);
                    const float v10 = static_cast<float>(img[bs.idx10]);
                    const float v11 = static_cast<float>(img[bs.idx11]);

                    const float v = MaxMode
                            ? std::max(std::max(bs.w1 * v00, bs.w2 * v01), std::max(bs.w3 * v10, bs.w4 * v11))
                            : (bs.w1 * v00 + bs.w2 * v01 + bs.w3 * v10 + bs.w4 * v11);

                    outv = MaxMode ? std::max(outv, v) : (outv + v);
                }

                outv = MaxMode ? ((outv == -FLT_MAX) ? 0.f : outv) : (outv * inv_sample_count);

                Ydata[y_roi_base + static_cast<size_t>(outPlane)] = saturate_cast<T>(outv);
            }
        }
    }

private:
    struct BilinearSample
    {
        size_t idx00 = 0, idx01 = 0, idx10 = 0, idx11 = 0;
        float w1 = 0.f, w2 = 0.f, w3 = 0.f, w4 = 0.f;
    };

    template<typename TRoi>
    static inline void readScaledRoiCoords(const Mat& rois, int r, float spatial_scale, float offset,
                                          float& x1, float& y1, float& x2, float& y2)
    {
        const TRoi* p = rois.ptr<TRoi>(r);
        x1 = static_cast<float>(p[0]) * spatial_scale - offset;
        y1 = static_cast<float>(p[1]) * spatial_scale - offset;
        x2 = static_cast<float>(p[2]) * spatial_scale - offset;
        y2 = static_cast<float>(p[3]) * spatial_scale - offset;
    }

    static inline BilinearSample precomputeBilinearSample(int height, int width, float y, float x)
    {
        BilinearSample s;
        if (y < -1.f || y > height || x < -1.f || x > width)
            return s;

        y = std::min(std::max(y, 0.f), static_cast<float>(height - 1));
        x = std::min(std::max(x, 0.f), static_cast<float>(width - 1));

        const int y_low = static_cast<int>(y);
        const int x_low = static_cast<int>(x);
        const int y_high = std::min(y_low + 1, height - 1);
        const int x_high = std::min(x_low + 1, width - 1);

        const float ly = y - y_low;
        const float lx = x - x_low;
        const float hy = 1.f - ly;
        const float hx = 1.f - lx;

        s.w1 = hy * hx;
        s.w2 = hy * lx;
        s.w3 = ly * hx;
        s.w4 = ly * lx;

        s.idx00 = static_cast<size_t>(y_low) * width + x_low;
        s.idx01 = static_cast<size_t>(y_low) * width + x_high;
        s.idx10 = static_cast<size_t>(y_high) * width + x_low;
        s.idx11 = static_cast<size_t>(y_high) * width + x_high;

        return s;
    }

    const Mat& X;
    Mat& Y;
    const Mat& rois;
    const Mat& batch_indices;

    int N = 0, C = 0, H = 0, W = 0;
    int output_height, output_width, sampling_ratio;
    float spatial_scale, offset;
    bool clampMalformedRoi;

    const T* Xdata = nullptr;
    T* Ydata = nullptr;

    int Rdepth = -1;
    int outSpatial = 0;

    size_t spatialSize = 0;
    size_t channelStride = 0;
    size_t batchStride = 0;

    const int* batch32 = nullptr;
    const int64* batch64 = nullptr;
};

}

class RoiAlignLayerImpl CV_FINAL : public RoiAlignLayer
{
public:
    RoiAlignLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        output_height = params.get<int>("output_height", 1);
        output_width = params.get<int>("output_width", 1);
        sampling_ratio = params.get<int>("sampling_ratio", 0);
        spatial_scale = params.get<float>("spatial_scale", 1.f);
        const String modeStr = params.get<String>("mode", "avg");
        CV_Assert(modeStr == "avg" || modeStr == "max");
        mode_ = (modeStr == "max") ? RoiAlignMode::MAX : RoiAlignMode::AVG;

        const String coordStr = params.get<String>("coordinate_transformation_mode", "half_pixel");
        CV_Assert(coordStr == "half_pixel" || coordStr == "output_half_pixel");
        coord_mode_ = (coordStr == "half_pixel") ? CoordTransformMode::HALF_PIXEL
                                                    : CoordTransformMode::OUTPUT_HALF_PIXEL;

        CV_Assert(output_height > 0 && output_width > 0);
        CV_Assert(sampling_ratio >= 0);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 3);

        const MatShape& X = inputs[0];
        const MatShape& rois = inputs[1];
        const MatShape& batch_indices = inputs[2];

        CV_Assert(X.size() == 4);
        CV_Assert(rois.size() == 2);
        CV_Assert(rois[1] == 4);
        CV_Assert(batch_indices.size() == 1 || (batch_indices.size() == 2 && batch_indices[1] == 1));

        MatShape out(4);
        out[0] = rois[0];
        out[1] = X[1];
        out[2] = output_height;
        out[3] = output_width;

        outputs.assign(1, out);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 3);
        CV_Assert(CV_MAT_DEPTH(inputs[0]) == CV_32F || CV_MAT_DEPTH(inputs[0]) == CV_64F ||
                  CV_MAT_DEPTH(inputs[0]) == CV_16F || CV_MAT_DEPTH(inputs[0]) == CV_16BF);
        CV_Assert(CV_MAT_DEPTH(inputs[1]) == CV_32F || CV_MAT_DEPTH(inputs[1]) == CV_64F ||
                  CV_MAT_DEPTH(inputs[1]) == CV_16F || CV_MAT_DEPTH(inputs[1]) == CV_16BF);
        CV_Assert(CV_MAT_DEPTH(inputs[2]) == CV_32S || CV_MAT_DEPTH(inputs[2]) == CV_64S);

        outputs.assign(requiredOutputs, MatType(inputs[0]));
        internals.assign(requiredInternals, MatType(inputs[0]));
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 3);
        CV_Assert(outputs.size() == 1);

        const Mat& X = inputs[0];
        Mat rois = inputs[1];
        Mat batch_indices = inputs[2];
        Mat& Y = outputs[0];

        CV_Assert(X.dims == 4);
        const int C = X.size[1];

        CV_Assert(X.depth() == CV_32F || X.depth() == CV_64F || X.depth() == CV_16F || X.depth() == CV_16BF);
        CV_Assert(rois.depth() == CV_32F || rois.depth() == CV_64F || rois.depth() == CV_16F || rois.depth() == CV_16BF);
        CV_Assert(batch_indices.depth() == CV_32S || batch_indices.depth() == CV_64S);

        CV_Assert(rois.total() % 4 == 0);
        const int num_rois = static_cast<int>(rois.total() / 4);
        rois = rois.reshape(1, num_rois);
        batch_indices = batch_indices.reshape(1, static_cast<int>(batch_indices.total()));

        CV_Assert(rois.rows == num_rois && rois.cols == 4);
        CV_Assert(static_cast<int>(batch_indices.total()) == num_rois);

        CV_Assert(Y.dims == 4);
        CV_Assert(Y.size[0] == num_rois && Y.size[1] == C && Y.size[2] == output_height && Y.size[3] == output_width);

        const bool isMaxMode = (mode_ == RoiAlignMode::MAX);
        const float offset = (coord_mode_ == CoordTransformMode::HALF_PIXEL) ? 0.5f : 0.0f;
        const bool clampMalformedRoi = (coord_mode_ != CoordTransformMode::HALF_PIXEL);
        const int* Xsize = X.size.p;
        const int Xdepth = X.depth();
        if (Xdepth == CV_32F)
        {
            if (isMaxMode)
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<float, true>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
            else
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<float, false>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
        }
        else if (Xdepth == CV_64F)
        {
            if (isMaxMode)
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<double, true>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
            else
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<double, false>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
        }
        else if (Xdepth == CV_16F)
        {
            if (isMaxMode)
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<hfloat, true>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
            else
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<hfloat, false>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
        }
        else if (Xdepth == CV_16BF)
        {
            if (isMaxMode)
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<bfloat, true>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
            else
                parallel_for_(Range(0, num_rois), RoiAlignForwardInvoker<bfloat, false>(X, Y, rois, batch_indices, Xsize, output_height, output_width, sampling_ratio, spatial_scale, offset, clampMalformedRoi));
        }
        else
            CV_Error(Error::StsUnsupportedFormat, "Unsupported X depth");
    }

private:
    int output_height = 1;
    int output_width = 1;
    int sampling_ratio = 0;
    float spatial_scale = 1.f;
    RoiAlignMode mode_ = RoiAlignMode::AVG;
    CoordTransformMode coord_mode_ = CoordTransformMode::HALF_PIXEL;
};

Ptr<RoiAlignLayer> RoiAlignLayer::create(const LayerParams& params)
{
    return Ptr<RoiAlignLayer>(new RoiAlignLayerImpl(params));
}

}} // namespace cv::dnn
