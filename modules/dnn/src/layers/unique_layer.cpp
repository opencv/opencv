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
// Supported opsets: 11+ (axis attribute introduced in opset 11)

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
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.resize(requiredOutputs);
        outputs[0] = MatType(CV_32F);
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

        int ax = axis_;
        if (hasAxis_) {
            if (ax < 0) ax += Xin.dims;
            CV_Assert(0 <= ax && ax < Xin.dims);
        } else {
            ax = -1;
        }

        Mat X;
        if (Xin.depth() == CV_32F) X = Xin;
        else Xin.convertTo(X, CV_32F);

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            CV_Assert(!outs.empty());
            uniqueAxisBytes(X, outs, ax, sorted_);
        }
        else if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            CV_Assert(!uouts.empty());
            std::vector<Mat> tmp(uouts.size());
            uniqueAxisBytes(X, tmp, ax, sorted_);
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
    void uniqueAxisBytes(const Mat& X, std::vector<Mat>& outs, int ax, bool sorted) const
    {
        const int r = X.dims;

        int A = 0;
        int64 sliceElems = 0;
        if (ax < 0)
        {
            A = (int)X.total();
            sliceElems = 1;
        }
        else
        {
            A = X.size[ax];
            sliceElems = 1;
            for (int k = 0; k < r; ++k) if (k != ax) sliceElems *= X.size[k];
        }

        const size_t elemSz = X.elemSize();

        std::vector<std::vector<uchar>> keys(A);
        keys.assign(A, std::vector<uchar>( (size_t)sliceElems * elemSz ));

        if (ax < 0)
        {
            const uchar* base = X.ptr();
            for (int j = 0; j < A; ++j)
            {
                std::memcpy(keys[j].data(), base + (size_t)j * elemSz, elemSz);
            }
        }
        else
        {
            std::vector<int> idx(r, 0);
            std::vector<size_t> stepB(r);
            for (int k = 0; k < r; ++k) stepB[k] = X.step[k];

            std::vector<int64> mult(r, 0);
            {
                int64 m = 1;
                for (int k = r - 1; k >= 0; --k) {
                    if (k == ax) continue;
                    mult[k] = m;
                    m *= X.size[k];
                }
            }

            const uchar* base = X.ptr();
            int64 total = X.total();
            for (int64 t = 0; t < total; ++t)
            {
                int j = idx[ax];
                int64 lin = 0;
                for (int k = 0; k < r; ++k) if (k != ax) lin += (int64)idx[k] * mult[k];
                uchar* dst = keys[j].data() + (size_t)lin * elemSz;
                size_t off = 0;
                for (int k = 0; k < r; ++k) off += (size_t)idx[k] * stepB[k];
                const uchar* src = base + off;
                std::memcpy(dst, src, elemSz);
                for (int k = r - 1; k >= 0; --k) {
                    if (++idx[k] < X.size[k]) break;
                    idx[k] = 0;
                }
            }
        }

        std::unordered_map<std::string, int> where; where.reserve(A*2+1);
        std::vector<std::string> uniqKeyStr; uniqKeyStr.reserve(A);
        std::vector<int> firstJ; firstJ.reserve(A);
        std::vector<int64> counts; counts.reserve(A);
        std::vector<int> inv(A, -1);

        auto keyToStr = [](const std::vector<uchar>& v)->std::string {
            return std::string(reinterpret_cast<const char*>(v.data()), v.size());
        };

        for (int j = 0; j < A; ++j) {
            std::string s = keyToStr(keys[j]);
            auto it = where.find(s);
            if (it == where.end()) {
                int yi = (int)uniqKeyStr.size();
                where.emplace(s, yi);
                uniqKeyStr.push_back(std::move(s));
                firstJ.push_back(j);
                counts.push_back(1);
                inv[j] = yi;
            } else {
                counts[it->second] += 1;
                inv[j] = it->second;
            }
        }

        std::vector<int> order(uniqKeyStr.size());
        std::iota(order.begin(), order.end(), 0);
        if (sorted) {
            const size_t elemSzBytes = elemSz;
            const size_t elemsPerSlice = (size_t)sliceElems;
            auto numericLess = [&](int a, int b){
                const std::string& sa = uniqKeyStr[a];
                const std::string& sb = uniqKeyStr[b];
                for (size_t i = 0; i < elemsPerSlice; ++i) {
                    float va, vb;
                    std::memcpy(&va, sa.data() + i * elemSzBytes, sizeof(float));
                    std::memcpy(&vb, sb.data() + i * elemSzBytes, sizeof(float));
                    if (va < vb) return true;
                    if (va > vb) return false;
                }
                return false;
            };
            std::sort(order.begin(), order.end(), numericLess);
        }

        std::vector<int> remap(order.size());
        for (int newi = 0; newi < (int)order.size(); ++newi)
            remap[ order[newi] ] = newi;

        if (ax < 0)
        {
            MatShape yshape(1); yshape[0] = (int)order.size();
            outs[0].fit(yshape, CV_32F);
            float* yp = outs[0].ptr<float>();
            for (int yi = 0; yi < (int)order.size(); ++yi)
            {
                float v;
                std::memcpy(&v, uniqKeyStr[ order[yi] ].data(), sizeof(float));
                yp[yi] = v;
            }
        }
        else
        {
            std::vector<int> ysz(r);
            for (int k = 0; k < r; ++k) ysz[k] = X.size[k];
            ysz[ax] = (int)order.size();
            MatShape yshape(ysz);
            outs[0].fit(yshape, CV_32F);

            parallel_for_(Range(0, (int)order.size()), [&](const Range& rq){
                std::vector<int> yidx(r, 0);
                const size_t outElemSz = outs[0].elemSize();
                std::vector<size_t> ystepB(r);
                for (int k = 0; k < r; ++k) ystepB[k] = outs[0].step[k];

                for (int q = rq.start; q < rq.end; ++q) {
                    int oldq = order[q];
                    const std::string& blob = uniqKeyStr[oldq];

                    std::fill(yidx.begin(), yidx.end(), 0);
                    int64 wrote = 0;
                    for (int64 t = 0; t < sliceElems; ++t)
                    {
                        yidx[ax] = q;
                        size_t yoff = 0;
                        for (int k = 0; k < r; ++k) yoff += (size_t)yidx[k] * ystepB[k];
                        uchar* yptr = outs[0].ptr() + yoff;
                        const uchar* sp = reinterpret_cast<const uchar*>(blob.data()) + (size_t)wrote * outElemSz;
                        std::memcpy(yptr, sp, outElemSz);
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
            MatShape invshape(1); invshape[0] = (ax < 0 ? A : A);
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

    bool sorted_;
    bool hasAxis_;
    int axis_;
};

Ptr<UniqueLayer> UniqueLayer::create(const LayerParams& params)
{
    return Ptr<UniqueLayer>(new UniqueLayerImpl(params));
}

}} // namespace cv::dnn
