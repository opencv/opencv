// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

// ONNX operator: AffineGrid
// Spec: https://onnx.ai/onnx/operators/onnx__AffineGrid.html
// Supported opsets: 16-22

namespace cv {
namespace dnn {

class AffineGridLayerImpl CV_FINAL : public AffineGridLayer
{
public:
    bool align_corners;

    AffineGridLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        align_corners = params.get<bool>("align_corners", false);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckGE((int)inputs.size(), 1, "AffineGrid requires at least transform input");

        const MatShape thetaShape = inputs[0];
        CV_CheckGE((int)thetaShape.size(), 3, "theta must be at least 3D [N, r, c]");
        int N = thetaShape[0];
        int r = thetaShape[1];
        int c = thetaShape[2];
        CV_CheckTrue((r == 2 && c == 3) || (r == 3 && c == 4), "theta must be [N,2,3] or [N,3,4]");
        const bool is3d = (r == 3);

        MatShape outShape;
        if ((int)inputs.size() >= 2) {
            const MatShape sizeShape = inputs[1];
            CV_CheckTrue(sizeShape.size() == 1, "size input must be 1D tensor");
            const int dims = (int)sizeShape[0];
            CV_CheckTrue(dims == (is3d ? 5 : 4), "size input must have 4 (2D) or 5 (3D) elements");
            MatShape resolved = is3d ? MatShape{N, -1, -1, -1, 3} : MatShape{N, -1, -1, 2};
            Net::Impl* netimpl_ = getNetImpl(this);
            if (netimpl_) {
                try {
                    Mat sz = netimpl_->argTensor(this->inputs[1]);
                    if (!sz.empty()) {
                        if (!is3d) {
                            int H = -1, W = -1;
                            if (sz.depth() == CV_64S) { H = (int)sz.at<int64_t>(2); W = (int)sz.at<int64_t>(3); }
                            else if (sz.depth() == CV_32S) { H = sz.at<int32_t>(2); W = sz.at<int32_t>(3); }
                            if (H > 0 && W > 0) resolved = MatShape{N, H, W, 2};
                        } else {
                            int D = -1, H = -1, W = -1;
                            if (sz.depth() == CV_64S) { D = (int)sz.at<int64_t>(2); H = (int)sz.at<int64_t>(3); W = (int)sz.at<int64_t>(4); }
                            else if (sz.depth() == CV_32S) { D = sz.at<int32_t>(2); H = sz.at<int32_t>(3); W = sz.at<int32_t>(4); }
                            if (D > 0 && H > 0 && W > 0) resolved = MatShape{N, D, H, W, 3};
                        }
                    }
                } catch (const cv::Exception& e) {
                    CV_Error(cv::Error::StsError, cv::format("DNN/AffineGrid: failed to resolve output shape from 'size' tensor: %s", e.what()));
                }
            }
            outShape = resolved;
        } else {
            if (is3d)
                outShape = MatShape{N, -1, -1, -1, 3};
            else
                outShape = MatShape{N, -1, -1, 2};
        }
        outputs.assign(1, outShape);
        return false;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_) return true;
        if (this->inputs.size() < 2) return true;

        if (netimpl_->isConstArg(this->inputs[1])) return false;
        try {
            const Mat& sz = netimpl_->argTensor(this->inputs[1]);
            return sz.empty();
        } catch (const cv::Exception& e) {
            CV_Error(cv::Error::StsError, cv::format("DNN/AffineGrid: cannot query 'size' tensor at runtime: %s", e.what()));
        }
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckGE((int)inputs.size(), 1, "");
        const bool allow_fp16 = (preferableTarget == DNN_TARGET_OPENCL_FP16 || preferableTarget == DNN_TARGET_CPU_FP16);
        const int thetaType = inputs[0];
        const bool is_float_ok = thetaType == CV_32F || thetaType == CV_64F || thetaType == CV_16BF || (allow_fp16 && thetaType == CV_16F) || thetaType == CV_16F;
        CV_CheckType(thetaType, is_float_ok, "AffineGrid: theta must be a floating tensor");

        if (inputs.size() >= 2)
            CV_CheckType(inputs[1], inputs[1] == CV_64S || inputs[1] == CV_32S, "");

        outputs.assign(std::max(1, requiredOutputs), CV_32F);
        internals.assign(requiredInternals, CV_32F);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& theta = inputs[0];
        CV_CheckTrue(theta.dims == 3, "theta must be 3D [N, r, c]");
        const int N = theta.size[0];
        const int r = theta.size[1];
        const bool is3d = (r == 3);

        int D = 1, H = 0, W = 0;
        if ((int)inputs.size() >= 2) {
            Mat sz = inputs[1];
            CV_CheckTrue(sz.total() == (size_t)(is3d ? 5 : 4), "size input must have 4 (2D) or 5 (3D) elements");
            if (is3d) {
                if (sz.depth() == CV_64S) {
                    D = (int)sz.at<int64_t>(2);
                    H = (int)sz.at<int64_t>(3);
                    W = (int)sz.at<int64_t>(4);
                } else if (sz.depth() == CV_32S) {
                    D = (int)sz.at<int32_t>(2);
                    H = (int)sz.at<int32_t>(3);
                    W = (int)sz.at<int32_t>(4);
                } else {
                    CV_CheckType(sz.depth(), sz.depth() == CV_64S || sz.depth() == CV_32S, "");
                }
            } else {
                if (sz.depth() == CV_64S) {
                    H = (int)sz.at<int64_t>(2);
                    W = (int)sz.at<int64_t>(3);
                } else if (sz.depth() == CV_32S) {
                    H = (int)sz.at<int32_t>(2);
                    W = (int)sz.at<int32_t>(3);
                } else {
                    CV_CheckType(sz.depth(), sz.depth() == CV_64S || sz.depth() == CV_32S, "");
                }
            }
        } else {
            const Mat& out = outputs[0];
            CV_CheckTrue(out.dims >= 3, "output must be preallocated with proper shape when size input is absent");
            if (is3d) {
                CV_CheckTrue(out.dims == 5, "3D grid must be [N, D, H, W, 3]");
                D = out.size[1]; H = out.size[2]; W = out.size[3];
            } else {
                CV_CheckTrue(out.dims == 4, "2D grid must be [N, H, W, 2]");
                H = out.size[1]; W = out.size[2];
            }
        }
        CV_CheckGT(H, 0, "H must be > 0");
        CV_CheckGT(W, 0, "W must be > 0");
        if (is3d) CV_CheckGT(D, 0, "D must be > 0");

        std::vector<int> outShape = is3d ? std::vector<int>{N, D, H, W, 3}
                                         : std::vector<int>{N, H, W, 2};
        Mat& grid = outputs[0];
        grid.create((int)outShape.size(), outShape.data(), CV_32F);

        auto computeLinspace = [&](int len, float& scale, float& delta){
            if (align_corners) {
                scale = len > 1 ? 2.f / float(len - 1) : 0.f;
                delta = -1.f;
            } else {
                scale = 2.f / float(len);
                delta = scale * 0.5f - 1.f;
            }
        };

        float xs, xd, ys, yd, zs=0.f, zd=0.f;
        computeLinspace(W, xs, xd);
        computeLinspace(H, ys, yd);
        if (is3d) computeLinspace(D, zs, zd);

        Mat thetaF;
        if (theta.depth() == CV_32F) thetaF = theta; else theta.convertTo(thetaF, CV_32F);

        parallel_for_(Range(0, N), [&](const Range& range){
            for (int n = range.start; n < range.end; n++) {
                int i0[3] = {n, 0, 0};
                int i1[3] = {n, 1, 0};
                int i2[3] = {n, 2, 0};
                const float* T0 = thetaF.ptr<float>(i0);
                const float* T1 = thetaF.ptr<float>(i1);
                const float* T2 = is3d ? thetaF.ptr<float>(i2) : nullptr;
                if (!is3d) {
                    for (int y = 0; y < H; y++) {
                        float ny = yd + ys * y;
                        float base0 = T0[1]*ny + T0[2];
                        float base1 = T1[1]*ny + T1[2];
                        float* out = grid.ptr<float>(n, y, 0);
                        for (int x = 0; x < W; x++) {
                            float nx = xd + xs * x;
                            out[2*x + 0] = T0[0]*nx + base0;
                            out[2*x + 1] = T1[0]*nx + base1;
                        }
                    }
                } else {
                    for (int zh = 0; zh < D * H; zh++) {
                        int z = zh / H;
                        int y = zh - z * H;
                        float ny = yd + ys * y;
                        float nz = zd + zs * z;
                        float base0 = T0[1]*ny + T0[2]*nz + T0[3];
                        float base1 = T1[1]*ny + T1[2]*nz + T1[3];
                        float base2 = T2[1]*ny + T2[2]*nz + T2[3];
                        int idx[5] = {n, z, y, 0, 0};
                        float* out = grid.ptr<float>(idx);
                        for (int x = 0; x < W; x++) {
                            float nx = xd + xs * x;
                            int o = 3*x;
                            out[o + 0] = T0[0]*nx + base0;
                            out[o + 1] = T1[0]*nx + base1;
                            out[o + 2] = T2[0]*nx + base2;
                        }
                    }
                }
            }
        });
    }
};

Ptr<AffineGridLayer> AffineGridLayer::create(const LayerParams& params)
{
    return Ptr<AffineGridLayer>(new AffineGridLayerImpl(params));
}

}} // namespace cv::dnn
