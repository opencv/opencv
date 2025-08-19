// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

// ONNX operator: Unique
// Spec: https://onnx.ai/onnx/operators/onnx__Unique.html
// Supported opsets: 11 (axis attribute introduced in opset 11)

namespace cv {
namespace dnn {

class UniqueLayerImpl CV_FINAL : public UniqueLayer
{
public:
    UniqueLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        sorted_ = params.get<bool>("sorted", true);
        hasAxis_ = params.has("axis");
        axis_ = hasAxis_ ? params.get<int>("axis") : 0;
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return true;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.resize(requiredOutputs);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.resize(requiredOutputs);
        outputs[0] = inputs[0];
        for (int i = 1; i < requiredOutputs; ++i)
            outputs[i] = MatType(CV_64S);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 1);

        const Mat& Xin = inputs[0];
        CV_Assert(Xin.dims >= 1);

        int ax = hasAxis_ ? normalize_axis(axis_, Xin.dims) : -1;

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            CV_Assert(!outs.empty());
            dispatchByDepth(Xin, outs, ax, sorted_);
        }
        else if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            CV_Assert(!uouts.empty());
            std::vector<Mat> tmp(uouts.size());
            dispatchByDepth(Xin, tmp, ax, sorted_);
            for (size_t i = 0; i < uouts.size(); ++i)
            {
                if (tmp[i].empty())
                    continue;
                MatShape sh = shape(tmp[i]);
                uouts[i].fit(sh, tmp[i].type());
                tmp[i].copyTo(uouts[i]);
            }
        }
        else
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }
    }

private:
    template<typename T, typename WT = T>
    void uniqueAxisImpl(const Mat& X, std::vector<Mat>& outs, int ax, bool sorted) const
    {
        const int r = X.dims;

        MatShape inShape = shape(X);
        int dimAxis;
        size_t outer = 1, inner = 1;
        if (ax < 0) {
            dimAxis = (int)X.total();
        } else {
            dimAxis = inShape[ax];
            outer = std::accumulate(inShape.begin(), inShape.begin() + ax, (size_t)1, std::multiplies<int>());
            inner = std::accumulate(inShape.begin() + ax + 1, inShape.end(), (size_t)1, std::multiplies<int>());
        }

        const int A = dimAxis;
        const int64 sliceElems = (ax < 0) ? 1 : (int64)outer * (int64)inner;

        const T* inPtr = X.ptr<const T>();

        auto getVal = [&](int u, size_t ob, size_t ij)->WT {
            if (ax < 0)
                return static_cast<WT>(inPtr[(size_t)u]);
            size_t off = ob * (size_t)dimAxis * inner + (size_t)u * inner + ij;
            return static_cast<WT>(inPtr[off]);
        };

        std::vector<int> perm(A);
        std::iota(perm.begin(), perm.end(), 0);
        auto lessIdx = [&](int ua, int ub){
            if (ax < 0) {
                WT va = getVal(ua, 0, 0);
                WT vb = getVal(ub, 0, 0);
                if (va < vb) return true;
                if (va > vb) return false;
                return ua < ub;
            }
            for (size_t ob = 0; ob < outer; ++ob) {
                for (size_t ij = 0; ij < inner; ++ij) {
                    WT va = getVal(ua, ob, ij);
                    WT vb = getVal(ub, ob, ij);
                    if (va < vb) return true;
                    if (va > vb) return false;
                }
            }
            return false;
        };
        std::sort(perm.begin(), perm.end(), lessIdx);

        auto eqIdx = [&](int ua, int ub){
            if (ax < 0)
                return getVal(ua, 0, 0) == getVal(ub, 0, 0);
            for (size_t ob = 0; ob < outer; ++ob) {
                for (size_t ij = 0; ij < inner; ++ij) {
                    if (getVal(ua, ob, ij) != getVal(ub, ob, ij))
                        return false;
                }
            }
            return true;
        };

        std::vector<int> groupRepIndex; groupRepIndex.reserve(A);
        std::vector<int> firstJ; firstJ.reserve(A);
        std::vector<int64> counts; counts.reserve(A);
        std::vector<int> inv(A, -1);

        int g = 0;
        for (int pos = 0; pos < A; )
        {
            int start = pos;
            int minJ = perm[pos];
            int64 cnt = 0;
            while (pos < A && eqIdx(perm[start], perm[pos])) {
                inv[ perm[pos] ] = g;
                minJ = std::min(minJ, perm[pos]);
                ++cnt;
                ++pos;
            }
            groupRepIndex.push_back(perm[start]);
            firstJ.push_back(minJ);
            counts.push_back(cnt);
            ++g;
        }

        std::vector<int> order((int)groupRepIndex.size());
        std::iota(order.begin(), order.end(), 0);
        if (sorted) {
            auto lessRep = [&](int ga, int gb){
                return lessIdx(groupRepIndex[ga], groupRepIndex[gb]);
            };
            std::sort(order.begin(), order.end(), lessRep);
        } else {
            auto firstOccurLess = [&](int ga, int gb){ return firstJ[ga] < firstJ[gb]; };
            std::sort(order.begin(), order.end(), firstOccurLess);
        }

        std::vector<int> remap(order.size());
        for (int newi = 0; newi < (int)order.size(); ++newi)
            remap[ order[newi] ] = newi;

        if (ax < 0)
        {
            MatShape yshape(1); yshape[0] = (int)order.size();
            outs[0].fit(yshape, X.type());
            T* yp = outs[0].ptr<T>();
            for (int yi = 0; yi < (int)order.size(); ++yi)
            {
                int u = groupRepIndex[ order[yi] ];
                yp[yi] = inPtr[(size_t)u];
            }
        }
        else
        {
            std::vector<int> ysz(r);
            for (int k = 0; k < r; ++k) ysz[k] = X.size[k];
            ysz[ax] = (int)order.size();
            MatShape yshape(ysz);
            outs[0].fit(yshape, X.type());

            parallel_for_(Range(0, (int)order.size()), [&](const Range& rq){
                std::vector<int> yidx(r, 0);
                std::vector<size_t> ystepB(r);
                for (int k = 0; k < r; ++k) ystepB[k] = outs[0].step[k];

                for (int q = rq.start; q < rq.end; ++q) {
                    int oldq = order[q];
                    int oldu = groupRepIndex[oldq];

                    std::fill(yidx.begin(), yidx.end(), 0);
                    int64 wrote = 0;
                    for (int64 t = 0; t < sliceElems; ++t)
                    {
                        yidx[ax] = q;
                        size_t yoff = 0;
                        for (int k = 0; k < r; ++k) yoff += (size_t)yidx[k] * ystepB[k];
                        T* yptrT = reinterpret_cast<T*>(outs[0].ptr() + yoff);

                        size_t ob = (size_t)(wrote / (int64)inner);
                        size_t ij = (size_t)(wrote % (int64)inner);
                        size_t srcOff = ob * (size_t)dimAxis * inner + (size_t)oldu * inner + ij;
                        const T vs = inPtr[srcOff];
                        *yptrT = vs;
                        wrote++;

                        for (int k = r - 1; k >= 0; --k) {
                            if (k == ax) continue;
                            if (++yidx[k] < X.size[k]) break;
                            yidx[k] = 0;
                        }
                    }
                }
            });
        }

        if (outs.size() > 1) {
            MatShape ishape(1); ishape[0] = (int)order.size();
            outs[1].fit(ishape, CV_64S);
            auto ip = outs[1].ptr<int64_t>();
            for (int yi = 0; yi < (int)order.size(); ++yi)
                ip[yi] = firstJ[ order[yi] ];
        }

        if (outs.size() > 2) {
            MatShape invshape(1); invshape[0] = A;
            outs[2].fit(invshape, CV_64S);
            auto invp = outs[2].ptr<int64_t>();
            for (int j = 0; j < A; ++j)
                invp[j] = remap[ inv[j] ];
        }

        if (outs.size() > 3) {
            MatShape cshape(1); cshape[0] = (int)order.size();
            outs[3].fit(cshape, CV_64S);
            auto cp = outs[3].ptr<int64_t>();
            for (int yi = 0; yi < (int)order.size(); ++yi)
                cp[yi] = counts[ order[yi] ];
        }
    }

    void dispatchByDepth(const Mat& X, std::vector<Mat>& outs, int ax, bool sorted) const
    {
        switch (X.depth())
        {
        case CV_8U:   uniqueAxisImpl<uint8_t          >(X, outs, ax, sorted); break;
        case CV_8S:   uniqueAxisImpl<int8_t           >(X, outs, ax, sorted); break;
        case CV_16U:  uniqueAxisImpl<uint16_t         >(X, outs, ax, sorted); break;
        case CV_16S:  uniqueAxisImpl<int16_t          >(X, outs, ax, sorted); break;
        case CV_16F:  uniqueAxisImpl<hfloat, float    >(X, outs, ax, sorted); break;
        case CV_16BF: uniqueAxisImpl<bfloat, float    >(X, outs, ax, sorted); break;
        case CV_32U:  uniqueAxisImpl<uint32_t         >(X, outs, ax, sorted); break;
        case CV_32S:  uniqueAxisImpl<int32_t          >(X, outs, ax, sorted); break;
        case CV_32F:  uniqueAxisImpl<float            >(X, outs, ax, sorted); break;
        case CV_64U:  uniqueAxisImpl<uint64_t         >(X, outs, ax, sorted); break;
        case CV_64S:  uniqueAxisImpl<int64_t          >(X, outs, ax, sorted); break;
        case CV_64F:  uniqueAxisImpl<double           >(X, outs, ax, sorted); break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "Unsupported data type for Unique");
        }
    }

    bool sorted_;
    bool hasAxis_;
    int axis_;
};

Ptr<UniqueLayer> UniqueLayer::create(const LayerParams& params)
{
    return Ptr<UniqueLayer>(new UniqueLayerImpl(params));
}

}} // namespace cv::dnn
