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
// Supported opsets: 20

namespace cv {
namespace dnn {

namespace {

template<typename T>
void computeGrid2D(const Mat& theta, Mat& grid, int N, int H, int W,
                   float xs, float xd, float ys, float yd)
{
    const int total = N * H;
    parallel_for_(Range(0, total), [&](const Range& range){
        const int H_ = H;
        const int W_ = W;
        const float xs_ = xs, xd_ = xd, ys_ = ys, yd_ = yd;
        const Mat theta_ = theta;
        Mat grid_ = grid;
        int lastN = -1;
        const T* T0 = nullptr;
        const T* T1 = nullptr;
        float T0_0 = 0.f, T1_0 = 0.f;
        for (int nyi = range.start; nyi < range.end; nyi++)
        {
            int n = nyi / H_;
            int y = nyi - n * H_;
            if (n != lastN)
            {
                int i0[3] = {n, 0, 0};
                int i1[3] = {n, 1, 0};
                T0 = theta_.ptr<T>(i0);
                T1 = theta_.ptr<T>(i1);
                T0_0 = (float)T0[0];
                T1_0 = (float)T1[0];
                lastN = n;
            }
            float ny = yd_ + ys_ * y;
            float base0 = (float)T0[1]*ny + (float)T0[2];
            float base1 = (float)T1[1]*ny + (float)T1[2];
            float* out = grid_.ptr<float>(n, y, 0);
            for (int x = 0; x < W_; x++)
            {
                float nx = xd_ + xs_ * x;
                out[2*x + 0] = T0_0*nx + base0;
                out[2*x + 1] = T1_0*nx + base1;
            }
        }
    });
}

template<typename T>
void computeGrid3D(const Mat& theta, Mat& grid, int N, int D, int H, int W,
                   float xs, float xd, float ys, float yd, float zs, float zd)
{
    const int stride = D * H;
    const int total = N * stride;
    parallel_for_(Range(0, total), [&](const Range& range){
        const int H_ = H;
        const int W_ = W;
        const int stride_ = stride;
        const float xs_ = xs, xd_ = xd, ys_ = ys, yd_ = yd, zs_ = zs, zd_ = zd;
        const Mat theta_ = theta;
        Mat grid_ = grid;
        int lastN = -1;
        const T* T0 = nullptr;
        const T* T1 = nullptr;
        const T* T2 = nullptr;
        float T0_0 = 0.f, T1_0 = 0.f, T2_0 = 0.f;
        for (int ndz = range.start; ndz < range.end; ndz++)
        {
            int n = ndz / stride_;
            int rem = ndz - n * stride_;
            int z = rem / H_;
            int y = rem - z * H_;
            if (n != lastN)
            {
                int i0[3] = {n, 0, 0};
                int i1[3] = {n, 1, 0};
                int i2[3] = {n, 2, 0};
                T0 = theta_.ptr<T>(i0);
                T1 = theta_.ptr<T>(i1);
                T2 = theta_.ptr<T>(i2);
                T0_0 = (float)T0[0];
                T1_0 = (float)T1[0];
                T2_0 = (float)T2[0];
                lastN = n;
            }
            float ny = yd_ + ys_ * y;
            float nz = zd_ + zs_ * z;
            float base0 = (float)T0[1]*ny + (float)T0[2]*nz + (float)T0[3];
            float base1 = (float)T1[1]*ny + (float)T1[2]*nz + (float)T1[3];
            float base2 = (float)T2[1]*ny + (float)T2[2]*nz + (float)T2[3];
            int idx[5] = {n, z, y, 0, 0};
            float* out = grid_.ptr<float>(idx);
            for (int x = 0; x < W_; x++)
            {
                float nx = xd_ + xs_ * x;
                int o = 3*x;
                out[o + 0] = T0_0*nx + base0;
                out[o + 1] = T1_0*nx + base1;
                out[o + 2] = T2_0*nx + base2;
            }
        }
    });
}
}

class AffineGridLayerImpl CV_FINAL : public AffineGridLayer
{
public:
    AffineGridLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        align_corners = params.get<bool>("align_corners", false);
    }

private:
    void resolveSizeAndShape(int N, bool is3d, const Mat* explicitSz,
                         MatShape& outShape, int& D, int& H, int& W) const
    {
        outShape = is3d ? MatShape{N, -1, -1, -1, 3} : MatShape{N, -1, -1, 2};
        D = 1; H = -1; W = -1;

        Mat sz;
        if (explicitSz)
        {
            sz = *explicitSz;
        }
        else
        {
            Net::Impl* netimpl_ = getNetImpl(this);
            if (!netimpl_)
                return;
            try
            {
                sz = netimpl_->argTensor(this->inputs[1]);
            }
            catch (const cv::Exception& e)
            {
                CV_Error(cv::Error::StsError,
                        cv::format("DNN/AffineGrid: failed to resolve output shape from 'size' tensor: %s", e.what()));
            }
        }

        if (sz.empty())
            return;

        std::vector<int> sizeVec;
        tensorToIntVec(sz, sizeVec);
        CV_CheckTrue((int)sizeVec.size() == (is3d ? 5 : 4), "size input must have 4 (2D) or 5 (3D) elements");

        if (is3d) {
            D = sizeVec[2];
            H = sizeVec[3];
            W = sizeVec[4];
            if (D > 0 && H > 0 && W > 0)
                outShape = MatShape{N, D, H, W, 3};
        } else {
            H = sizeVec[2];
            W = sizeVec[3];
            if (H > 0 && W > 0)
                outShape = MatShape{N, H, W, 2};
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckGE((int)inputs.size(), 2, "AffineGrid requires two inputs: theta and size");

        const MatShape thetaShape = inputs[0];
        CV_CheckGE((int)thetaShape.size(), 3, "theta must be at least 3D [N, r, c]");
        int N = thetaShape[0];
        int r = thetaShape[1];
        int c = thetaShape[2];
        CV_CheckTrue((r == 2 && c == 3) || (r == 3 && c == 4), "theta must be [N,2,3] or [N,3,4]");
        const bool is3d = (r == 3);

        MatShape outShape;
        const MatShape sizeShape = inputs[1];
        CV_CheckTrue(sizeShape.size() == 1, "size input must be 1D tensor");
        const int dims = (int)sizeShape[0];
        CV_CheckTrue(dims == (is3d ? 5 : 4), "size input must have 4 (2D) or 5 (3D) elements");
        int D, H, W;
        resolveSizeAndShape(N, is3d, /*explicitSz*/nullptr, outShape, D, H, W);

        outputs.assign(1, outShape);
        return false;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_) return true;
        CV_CheckGE((int)this->inputs.size(), 2, "AffineGrid requires two inputs: theta and size");

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
        CV_CheckGE((int)inputs.size(), 2, "AffineGrid requires two inputs: theta and size");
        const bool allow_fp16 = (preferableTarget == DNN_TARGET_OPENCL_FP16 || preferableTarget == DNN_TARGET_CPU_FP16);
        const int thetaType = inputs[0];
        const bool is_float_ok = thetaType == CV_32F || thetaType == CV_64F || thetaType == CV_16BF || (allow_fp16 && thetaType == CV_16F) || thetaType == CV_16F;
        CV_CheckType(thetaType, is_float_ok, "AffineGrid: theta must be a floating tensor");
        CV_CheckType(inputs[1], inputs[1] == CV_64S || inputs[1] == CV_32S, "");

        outputs.assign(std::max(1, requiredOutputs), CV_32F);
        internals.assign(requiredInternals, CV_32F);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        auto outKind = outputs_arr.kind();
        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT || outKind == _InputArray::STD_VECTOR_UMAT);

        std::vector<Mat> inMats;
        inputs_arr.getMatVector(inMats);
        CV_CheckGE((int)inMats.size(), 2, "AffineGrid requires two inputs: theta and size");
        Mat theta = inMats[0];
        Mat sz = inMats[1];

        CV_CheckTrue(theta.dims == 3, "theta must be 3D [N, r, c]");
        const int N = theta.size[0];
        const int r = theta.size[1];
        const bool is3d = (r == 3);

        int D = 1, H = 0, W = 0;
        MatShape dummy;
        resolveSizeAndShape(N, is3d, &sz, dummy, D, H, W);

        CV_CheckGT(H, 0, "H must be > 0");
        CV_CheckGT(W, 0, "W must be > 0");
        if (is3d) CV_CheckGT(D, 0, "D must be > 0");

        std::vector<int> outShape = is3d ? std::vector<int>{N, D, H, W, 3}
                                         : std::vector<int>{N, H, W, 2};
        MatShape fitShape(outShape.begin(), outShape.end());
        Mat grid;
        if (outKind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(fitShape, CV_32F);
            grid = outs[0];
        } else {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            uouts.resize(1);
            uouts[0].fit(fitShape, CV_32F);
            grid = uouts[0].getMat(ACCESS_WRITE);
        }

        float xs, xd, ys, yd, zs=0.f, zd=0.f;
        if (align_corners) {
            xs = W > 1 ? 2.f / float(W - 1) : 0.f;
            xd = -1.f;
            ys = H > 1 ? 2.f / float(H - 1) : 0.f;
            yd = -1.f;
            if (is3d) {
                zs = D > 1 ? 2.f / float(D - 1) : 0.f;
                zd = -1.f;
            }
        } else {
            xs = 2.f / float(W);
            xd = xs * 0.5f - 1.f;
            ys = 2.f / float(H);
            yd = ys * 0.5f - 1.f;
            if (is3d) {
                zs = 2.f / float(D);
                zd = zs * 0.5f - 1.f;
            }
        }

        int depth = theta.depth();
        if (!is3d){
            switch (depth) {
                case CV_32F: computeGrid2D<float>(theta, grid, N, H, W, xs, xd, ys, yd); break;
                case CV_64F: computeGrid2D<double>(theta, grid, N, H, W, xs, xd, ys, yd); break;
                case CV_16F: computeGrid2D<hfloat>(theta, grid, N, H, W, xs, xd, ys, yd); break;
                case CV_16BF: computeGrid2D<bfloat>(theta, grid, N, H, W, xs, xd, ys, yd); break;
                default: CV_Error(cv::Error::BadDepth, "AffineGrid: unsupported theta depth");
            }
        }
        else{
            switch (depth) {
                case CV_32F: computeGrid3D<float>(theta, grid, N, D, H, W, xs, xd, ys, yd, zs, zd); break;
                case CV_64F: computeGrid3D<double>(theta, grid, N, D, H, W, xs, xd, ys, yd, zs, zd); break;
                case CV_16F: computeGrid3D<hfloat>(theta, grid, N, D, H, W, xs, xd, ys, yd, zs, zd); break;
                case CV_16BF: computeGrid3D<bfloat>(theta, grid, N, D, H, W, xs, xd, ys, yd, zs, zd); break;
                default: CV_Error(cv::Error::BadDepth, "AffineGrid: unsupported theta depth");
            }
        }
    }
};

Ptr<AffineGridLayer> AffineGridLayer::create(const LayerParams& params)
{
    return Ptr<AffineGridLayer>(new AffineGridLayerImpl(params));
}

}} // namespace cv::dnn
