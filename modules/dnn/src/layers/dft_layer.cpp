// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core.hpp>
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <numeric>

// ONNX DFT operator
// Spec: https://onnx.ai/onnx/operators/onnx__DFT.html
// Supported opset: 17-22

namespace cv {
namespace dnn {

template<typename T, typename ComplexVec, typename FillFn>
static void dftAlongAxisWorker(const Mat& src,
                               Mat& dst,
                               const std::vector<int>& dimSizesSrc,
                               const std::vector<size_t>& stridesSrc,
                               const std::vector<size_t>& stridesDst,
                               const std::vector<int>& iterDims,
                               const std::vector<int>& outerSizes,
                               const std::vector<size_t>& outerStep,
                               const size_t totalOuter,
                               const int axis,
                               const int N,
                               const int outN,
                               const size_t strideAxisSrc,
                               const size_t strideAxisDst,
                               const int inMatType,
                               const int flagsBase,
                               const bool inverse,
                               FillFn&& fill)
{
    const T* sp = src.ptr<T>();
    T* dp = dst.ptr<T>();
    cv::parallel_for_(Range(0, (int)totalOuter), [&](const Range& r){
        Mat inRow(1, N, inMatType);
        Mat outRow;
        for (int pos = r.start; pos < r.end; ++pos)
        {
            size_t baseSrc = 0;
            size_t baseDst = 0;
            for (size_t t = 0; t < iterDims.size(); ++t)
            {
                int idxVal = outerStep.empty() ? 0 : (int)((pos / outerStep[t]) % (size_t)outerSizes[t]);
                int d = iterDims[t];
                baseSrc += (size_t)idxVal * stridesSrc[d];
                baseDst += (size_t)idxVal * stridesDst[d];
            }
            const T* in = sp + baseSrc;
            T* out = dp + baseDst;
            fill(inRow, in, dimSizesSrc[axis], N, strideAxisSrc);
            int flags = flagsBase | (inverse ? (DFT_INVERSE | DFT_SCALE) : 0);
            cv::dft(inRow, outRow, flags);
            const ComplexVec* p = outRow.ptr<ComplexVec>(0);
            for (int k = 0; k < outN; ++k)
            {
                size_t ok = (size_t)k * strideAxisDst;
                out[ok + 0] = p[k][0];
                out[ok + 1] = p[k][1];
            }
        }
    });
}

template<typename T, typename ComplexVec>
static void runTypedDFT(const Mat& src,
                        Mat& dst,
                        const std::vector<int>& dimSizesSrc,
                        const std::vector<size_t>& stridesSrc,
                        const std::vector<size_t>& stridesDst,
                        const std::vector<int>& iterDims,
                        const std::vector<int>& outerSizes,
                        const std::vector<size_t>& outerStep,
                        const size_t totalOuter,
                        const int axis,
                        const int N,
                        const int outN,
                        const size_t strideAxisSrc,
                        const size_t strideAxisDst,
                        const bool srcHasComplex,
                        const bool inverse)
{
    const int matTypeReal = std::is_same<T, float>::value ? CV_32F : CV_64F;
    const int matTypeComplex = std::is_same<T, float>::value ? CV_32FC2 : CV_64FC2;

    if (srcHasComplex)
    {
        dftAlongAxisWorker<T, ComplexVec>(
            src, dst, dimSizesSrc, stridesSrc, stridesDst,
            iterDims, outerSizes, outerStep, totalOuter,
            axis, N, outN, strideAxisSrc, strideAxisDst,
            matTypeComplex, 0, inverse,
            [&](Mat& inRow, const T* in, int origLen, int len, size_t stride){
                ComplexVec* ptr = inRow.ptr<ComplexVec>(0);
                for (int n = 0; n < origLen; ++n)
                {
                    size_t offSrc = (size_t)n * stride;
                    ptr[n][0] = in[offSrc + 0];
                    ptr[n][1] = in[offSrc + 1];
                }
                const ComplexVec zeroVal(T(0), T(0));
                for (int n = origLen; n < len; ++n) ptr[n] = zeroVal;
            }
        );
    }
    else
    {
        dftAlongAxisWorker<T, ComplexVec>(
            src, dst, dimSizesSrc, stridesSrc, stridesDst,
            iterDims, outerSizes, outerStep, totalOuter,
            axis, N, outN, strideAxisSrc, strideAxisDst,
            matTypeReal, DFT_COMPLEX_OUTPUT, inverse,
            [&](Mat& inRow, const T* in, int origLen, int len, size_t stride){
                T* ptr = inRow.ptr<T>(0);
                for (int n = 0; n < origLen; ++n)
                {
                    size_t offSrc = (size_t)n * stride;
                    ptr[n] = in[offSrc];
                }
                for (int n = origLen; n < len; ++n) ptr[n] = T(0);
            }
        );
    }
}

// Offset of the pos-th signal, walking the dims that are not transformed.
static inline size_t dftSignalOffset(int pos,
                                     const std::vector<int>& iterDims,
                                     const std::vector<int>& outerSizes,
                                     const std::vector<size_t>& outerStep,
                                     const std::vector<size_t>& strides)
{
    size_t base = 0;
    for (size_t t = 0; t < iterDims.size(); ++t)
    {
        int idxVal = outerStep.empty() ? 0 : (int)((pos / outerStep[t]) % (size_t)outerSizes[t]);
        base += (size_t)idxVal * strides[iterDims[t]];
    }
    return base;
}

// Inverse real DFT: cv::dft() takes the one-sided spectrum in CCS layout directly.
template<typename T>
static void runIRFFT(const Mat& src, Mat& dst,
                     const std::vector<size_t>& stridesSrc,
                     const std::vector<size_t>& stridesDst,
                     const std::vector<int>& iterDims,
                     const std::vector<int>& outerSizes,
                     const std::vector<size_t>& outerStep,
                     const size_t totalOuter,
                     const int N, const int L,
                     const size_t strideAxisSrc, const size_t strideAxisDst)
{
    const int matTypeReal = std::is_same<T, float>::value ? CV_32F : CV_64F;
    const T* sp = src.ptr<T>();
    T* dp = dst.ptr<T>();

    // Innermost axis: dst rows are already the N-element rows cv::dft() wants.
    const bool inPlaceOnDst = (strideAxisDst == 1);
    Mat rows = inPlaceOnDst ? Mat((int)totalOuter, N, matTypeReal, dp)
                            : Mat((int)totalOuter, N, matTypeReal);

    const int nbins = std::min(L, N / 2 + 1);   // bins a CCS row can hold
    const bool zeroPad = nbins < N / 2 + 1;

    // One DFT_ROWS call per stripe: cv::dft() rebuilds its tables on every call.
    const double nstripes = std::min<double>((double)totalOuter, std::max(getNumThreads(), 1));
    parallel_for_(Range(0, (int)totalOuter), [&](const Range& r)
    {
        for (int pos = r.start; pos < r.end; ++pos)
        {
            const T* in = sp + dftSignalOffset(pos, iterDims, outerSizes, outerStep, stridesSrc);
            T* ccs = rows.ptr<T>(pos);
            if (zeroPad)
                std::fill(ccs, ccs + N, T(0));
            // CCS: [Re0, Re1, Im1, ...] plus a trailing Re(N/2) for even N.
            ccs[0] = in[0];
            for (int k = 1; k < nbins; ++k)
            {
                const size_t o = (size_t)k * strideAxisSrc;
                ccs[2 * k - 1] = in[o];
                if (2 * k < N)
                    ccs[2 * k] = in[o + 1];
            }
        }

        Mat stripe = rows.rowRange(r.start, r.end);
        cv::dft(stripe, stripe, DFT_INVERSE | DFT_SCALE | DFT_ROWS);

        if (inPlaceOnDst)
            return;   // dst already holds the result
        for (int pos = r.start; pos < r.end; ++pos)
        {
            const T* res = rows.ptr<T>(pos);
            T* out = dp + dftSignalOffset(pos, iterDims, outerSizes, outerStep, stridesDst);
            for (int n = 0; n < N; ++n)
                out[(size_t)n * strideAxisDst] = res[n];
        }
    }, nstripes);
}

class DFTLayerImpl CV_FINAL : public DFTLayer {
public:
    DFTLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        inverse = params.get<int>("inverse", 0) != 0;
        onesided = params.get<int>("onesided", 0) != 0;
        axis_attr = params.get<int>("axis", 1);
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        // Non-const dft_length/axis inputs make the output shape known only at forward time.
        Net::Impl* netimpl_ = getNetImpl(const_cast<DFTLayerImpl*>(this));
        for (size_t i = 1; i < this->inputs.size(); ++i)
        {
            if (this->inputs[i].idx <= 0)
                continue;   // empty optional input
            if (!netimpl_ || !netimpl_->isConstArg(this->inputs[i]))
                return true;
        }
        return false;
    }

private:
    int getDftLengthFromConstant() const
    {
        if (this->inputs.size() < 2)
        {
            return -1;
        }

        Net::Impl* netimpl_ = getNetImpl(const_cast<DFTLayerImpl*>(this));
        if (!netimpl_)
        {
            return -1;
        }

        Mat dft_length_tensor = netimpl_->argTensor(this->inputs[1]).getMat(ACCESS_READ);
        if (dft_length_tensor.empty() || dft_length_tensor.total() != 1)
        {
            return -1;
        }

        int64_t dft_length64 = 0;
        tensorToScalar(dft_length_tensor, CV_64S, &dft_length64);
        if (dft_length64 > 0)
        {
            return static_cast<int>(dft_length64);
        }
        return -1;
    }

    // Read the opset-20 axis input when it is a constant.
    int getAxisFromConstant() const
    {
        if (this->inputs.size() < 3 || this->inputs[2].idx <= 0)
        {
            return INT_MIN;   // no axis input, or empty optional input
        }

        Net::Impl* netimpl_ = getNetImpl(const_cast<DFTLayerImpl*>(this));
        if (!netimpl_ || !netimpl_->isConstArg(this->inputs[2]))
        {
            return INT_MIN;
        }

        Mat axis_tensor = netimpl_->argTensor(this->inputs[2]).getMat(ACCESS_READ);
        if (axis_tensor.empty() || axis_tensor.total() != 1)
        {
            return INT_MIN;
        }

        int64_t axis64 = 0;
        tensorToScalar(axis_tensor, CV_64S, &axis64);
        return static_cast<int>(axis64);
    }

public:
    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int /*requiredOutputs*/,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &/*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        const MatShape &inshape = inputs[0];
        CV_Assert(!inshape.empty());
        MatShape out = inshape;

        int ndims_in = (int)inshape.size();
        int ax = axis_attr;
        if (ax == INT_MIN)
        {
            ax = (inshape.back() == 2 || inshape.back() == 1) ? ndims_in - 2 : ndims_in - 1;
        }
        if (ax < 0) ax += ndims_in;

        // A constant opset-20 axis input overrides the attribute (matches forward()).
        int ax_const = getAxisFromConstant();
        if (ax_const != INT_MIN)
        {
            ax = ax_const < 0 ? ax_const + ndims_in : ax_const;
        }

        // Inverse real DFT (irfft): real output, axis restored to full length N.
        if (inverse && onesided)
        {
            out.back() = 1;
            if (ax >= 0 && ax < (int)out.size() - 1)
            {
                int dft_length = getDftLengthFromConstant();
                out[ax] = dft_length > 0 ? dft_length : 2 * (inshape[ax] - 1);
            }
            outputs.assign(1, out);
            return false;
        }

        int last = out.back();
        if (last == 1)
            out.back() = 2;
        else if (last != 2)
            out.push_back(2);

        if (ax >= 0 && ax < (int)out.size() - 1)
        {
            int dft_length = getDftLengthFromConstant();
            int signalLen = dft_length > 0 ? dft_length : (ax < (int)inshape.size() ? inshape[ax] : out[ax]);
            out[ax] = onesided ? (signalLen / 2 + 1) : signalLen;
        }
        outputs.assign(1, out);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);

        CV_Assert(!inputs.empty());
        CV_Assert(inputs[0].dims >= 1);

        const Mat &src = inputs[0];

        const int ndims = src.dims;
        CV_Assert(ndims >= 1);
        const bool srcHasComplex = (src.size[ndims - 1] == 2);
        const bool srcLastIsOne = (src.size[ndims - 1] == 1);
        int axis = (srcHasComplex ? ndims - 2 : ndims - 1);
        if (axis_attr != INT_MIN)
        {
            axis = axis_attr;
            if (axis < 0) axis += ndims;
        }
        else if (!axes.empty())
        {
            CV_Assert(axes.size() == 1);
            axis = axes[0];
            if (axis < 0) axis += ndims;
        }
        // opset-20 provides the axis as the 3rd input tensor.
        if (inputs.size() >= 3 && !inputs[2].empty())
        {
            CV_Assert(inputs[2].total() == 1);
            int64_t ax64 = 0;
            tensorToScalar(inputs[2], CV_64S, &ax64);
            axis = static_cast<int>(ax64);
            if (axis < 0) axis += ndims;
        }
        CV_Assert(axis >= 0 && axis < (srcHasComplex ? ndims - 1 : ndims));

        const bool irfft = inverse && onesided;
        if (onesided && !irfft)
        {
            CV_Assert(!srcHasComplex);
            CV_Assert(!inverse);
        }
        if (irfft)
            CV_Assert(srcHasComplex);

        int dft_length = -1;
        if (inputs.size() >= 2 && !inputs[1].empty())
        {
            CV_Assert(inputs[1].total() == 1);
            int64_t dft_length64 = -1;
            tensorToScalar(inputs[1], CV_64S, &dft_length64);
            dft_length = static_cast<int>(dft_length64);
        }

        if (irfft)
        {
            const int L = src.size[axis];
            const int N = dft_length > 0 ? dft_length : 2 * (L - 1);

            std::vector<int> outSizesVec(ndims);
            for (int i = 0; i < ndims; ++i) outSizesVec[i] = src.size[i];
            outSizesVec[ndims - 1] = 1;   // real output
            outSizesVec[axis] = N;

            MatShape outShape(outSizesVec.begin(), outSizesVec.end());
            if (outputs_arr.kind() == _InputArray::STD_VECTOR_MAT)
                outputs_arr.getMatVecRef()[0].fit(outShape, src.type());
            else
                outputs_arr.getUMatVecRef()[0].fit(outShape, src.type());
            outputs_arr.getMatVector(outputs);
            Mat &dst = outputs[0];

            std::vector<size_t> stridesSrc(ndims, 1), stridesDst(ndims, 1);
            for (int i = ndims - 2; i >= 0; --i) stridesSrc[i] = stridesSrc[i + 1] * (size_t)src.size[i + 1];
            for (int i = ndims - 2; i >= 0; --i) stridesDst[i] = stridesDst[i + 1] * (size_t)outSizesVec[i + 1];

            std::vector<int> iterDims;
            for (int i = 0; i < ndims - 1; ++i) if (i != axis) iterDims.push_back(i);
            std::vector<int> outerSizes(iterDims.size());
            for (size_t j = 0; j < iterDims.size(); ++j) outerSizes[j] = src.size[iterDims[j]];
            std::vector<size_t> outerStep(iterDims.size(), 1);
            for (int j = (int)iterDims.size() - 2; j >= 0; --j) outerStep[j] = outerStep[j + 1] * (size_t)outerSizes[j + 1];
            size_t totalOuter = 1;
            for (int s : outerSizes) totalOuter *= (size_t)s;

            const size_t strideAxisSrc = stridesSrc[axis];
            const size_t strideAxisDst = stridesDst[axis];
            if (src.depth() == CV_32F)
                runIRFFT<float>(src, dst, stridesSrc, stridesDst, iterDims, outerSizes, outerStep, totalOuter, N, L, strideAxisSrc, strideAxisDst);
            else if (src.depth() == CV_64F)
                runIRFFT<double>(src, dst, stridesSrc, stridesDst, iterDims, outerSizes, outerStep, totalOuter, N, L, strideAxisSrc, strideAxisDst);
            else
                CV_Error(Error::StsNotImplemented, "DFT supports float32/float64 only");
            return;
        }

        std::vector<int> outSizesVec;
        outSizesVec.resize(srcHasComplex ? ndims : ndims + (srcLastIsOne ? 0 : 1));
        for (int i = 0; i < ndims; ++i) outSizesVec[i] = src.size[i];
        int complexDim = (int)outSizesVec.size() - 1;
        outSizesVec[complexDim] = 2;
        {
            int dstDimsNoComplex = (int)outSizesVec.size() - 1;
            if (axis >= 0 && axis < dstDimsNoComplex)
            {
                int signalLen = dft_length > 0 ? dft_length : outSizesVec[axis];
                outSizesVec[axis] = onesided ? (signalLen / 2 + 1) : signalLen;
            }
        }
        MatShape outShape;
        outShape.assign(outSizesVec.begin(), outSizesVec.end());
        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            outputs_arr.getMatVecRef()[0].fit(outShape, src.type());
        } else {
            CV_Assert(kind == _InputArray::STD_VECTOR_UMAT);
            outputs_arr.getUMatVecRef()[0].fit(outShape, src.type());
        }
        outputs_arr.getMatVector(outputs);
        CV_Assert(outputs.size() == 1);
        Mat &dst = outputs[0];

        std::vector<int> dimSizesSrc(ndims);
        for (int i = 0; i < ndims; ++i) {
            dimSizesSrc[i] = src.size[i];
        }
        std::vector<size_t> stridesSrc(ndims, 1);
        for (int i = ndims - 2; i >= 0; --i) {
            stridesSrc[i] = stridesSrc[i + 1] * (size_t)dimSizesSrc[i + 1];
        }
        const int ndimsDst = (int)outSizesVec.size();
        std::vector<size_t> stridesDst(ndimsDst, 1);
        for (int i = ndimsDst - 2; i >= 0; --i) {
            stridesDst[i] = stridesDst[i + 1] * (size_t)outSizesVec[i + 1];
        }

        int N = dimSizesSrc[axis];
        if (dft_length > 0) N = dft_length;
        int outN = onesided ? (N / 2 + 1) : N;
        const size_t strideAxisSrc = stridesSrc[axis];
        const size_t strideAxisDst = stridesDst[axis];
        std::vector<int> iterDims;
        for (int i = 0; i < (srcHasComplex ? ndims - 1 : ndims); ++i){
            if (i != axis){
                iterDims.push_back(i);
            }
        }
        std::vector<int> outerSizes(iterDims.size(), 0);
        for (size_t j = 0; j < iterDims.size(); ++j){
            outerSizes[j] = dimSizesSrc[iterDims[j]];
        }
        std::vector<size_t> outerStep(iterDims.size(), 1);
        for (int j = (int)iterDims.size() - 2; j >= 0; --j){
            outerStep[j] = outerStep[j + 1] * (size_t)outerSizes[j + 1];
        }
        size_t totalOuter = 1;
        for (int s : outerSizes){
            totalOuter *= (size_t)s;
        }

        const int depth = src.depth();
        if (depth == CV_32F)
        {
            runTypedDFT<float, Vec2f>(src, dst, dimSizesSrc, stridesSrc, stridesDst,
                                      iterDims, outerSizes, outerStep, totalOuter,
                                      axis, N, outN, strideAxisSrc, strideAxisDst,
                                      srcHasComplex, inverse);
        }
        else if (depth == CV_64F)
        {
            runTypedDFT<double, Vec2d>(src, dst, dimSizesSrc, stridesSrc, stridesDst,
                                       iterDims, outerSizes, outerStep, totalOuter,
                                       axis, N, outN, strideAxisSrc, strideAxisDst,
                                       srcHasComplex, inverse);
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "DFT supports float32/float64 only");
        }
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int /*requiredOutputs*/,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        outputs.assign(1, inputs[0]);
    }
};

Ptr<DFTLayer> DFTLayer::create(const LayerParams& params)
{
    return makePtr<DFTLayerImpl>(params);
}

}} // namespace
