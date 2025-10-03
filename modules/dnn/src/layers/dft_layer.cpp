// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core.hpp>
#include <numeric>

// ONNX DFT operator
// Spec: https://onnx.ai/onnx/operators/onnx__DFT.html
// Supported opset: 17-22

namespace cv {
namespace dnn {

template<typename Scalar, typename ComplexVec, typename FillFn>
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
                               const size_t strideAxisSrc,
                               const size_t strideAxisDst,
                               const int inMatType,
                               const int flagsBase,
                               const bool inverse,
                               FillFn&& fill)
{
    const Scalar* sp = src.ptr<Scalar>();
    Scalar* dp = dst.ptr<Scalar>();
    cv::parallel_for_(Range(0, (int)totalOuter), [&](const Range& r){
        Mat inRow(1, N, inMatType);
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
            const Scalar* in = sp + baseSrc;
            Scalar* out = dp + baseDst;
            fill(inRow, in, dimSizesSrc[axis], N, strideAxisSrc);
            Mat outRow;
            int flags = flagsBase | (inverse ? (DFT_INVERSE | DFT_SCALE) : 0);
            cv::dft(inRow, outRow, flags);
            const ComplexVec* p = outRow.ptr<ComplexVec>(0);
            for (int k = 0; k < N; ++k)
            {
                size_t ok = (size_t)k * strideAxisDst;
                out[ok + 0] = p[k][0];
                out[ok + 1] = p[k][1];
            }
        }
    });
}

template<typename Scalar, typename ComplexVec>
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
                        const size_t strideAxisSrc,
                        const size_t strideAxisDst,
                        const bool srcHasComplex,
                        const bool inverse)
{
    const int matTypeReal = std::is_same<Scalar, float>::value ? CV_32F : CV_64F;
    const int matTypeComplex = std::is_same<Scalar, float>::value ? CV_32FC2 : CV_64FC2;

    if (srcHasComplex)
    {
        dftAlongAxisWorker<Scalar, ComplexVec>(
            src, dst, dimSizesSrc, stridesSrc, stridesDst,
            iterDims, outerSizes, outerStep, totalOuter,
            axis, N, strideAxisSrc, strideAxisDst,
            matTypeComplex, 0, inverse,
            [&](Mat& inRow, const Scalar* in, int origLen, int len, size_t stride){
                ComplexVec* ptr = inRow.ptr<ComplexVec>(0);
                for (int n = 0; n < origLen; ++n)
                {
                    size_t offSrc = (size_t)n * stride;
                    ptr[n][0] = in[offSrc + 0];
                    ptr[n][1] = in[offSrc + 1];
                }
                const ComplexVec zeroVal(Scalar(0), Scalar(0));
                for (int n = origLen; n < len; ++n) ptr[n] = zeroVal;
            }
        );
    }
    else
    {
        dftAlongAxisWorker<Scalar, ComplexVec>(
            src, dst, dimSizesSrc, stridesSrc, stridesDst,
            iterDims, outerSizes, outerStep, totalOuter,
            axis, N, strideAxisSrc, strideAxisDst,
            matTypeReal, DFT_COMPLEX_OUTPUT, inverse,
            [&](Mat& inRow, const Scalar* in, int origLen, int len, size_t stride){
                Scalar* ptr = inRow.ptr<Scalar>(0);
                for (int n = 0; n < origLen; ++n)
                {
                    size_t offSrc = (size_t)n * stride;
                    ptr[n] = in[offSrc];
                }
                for (int n = origLen; n < len; ++n) ptr[n] = Scalar(0);
            }
        );
    }
}

class DFTLayerImpl CV_FINAL : public DFTLayer {
public:
    DFTLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        inverse = params.get<int>("inverse", 0) != 0;
        onesided = params.get<int>("onesided", 0) != 0;
        axis_attr = params.get<int>("axis", 1);
        dft_length = params.get<int>("dft_length", -1);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int /*requiredOutputs*/,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &/*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        const MatShape &inshape = inputs[0];
        MatShape out = inshape;

        if (!out.empty())
        {
            int last = out.back();
            if (last == 1)
                out.back() = 2;
            else if (last != 2)
                out.push_back(2);
        }
        if (dft_length > 0 && !out.empty())
        {
            int ndims_in = (int)inshape.size();
            int ax = axis_attr;
            if (ax == INT_MIN)
            {
                ax = (inshape.back() == 2 || inshape.back() == 1) ? ndims_in - 2 : ndims_in - 1;
            }
            if (ax < 0) ax += ndims_in;
            if (ax >= 0 && ax < (int)out.size() - 1)
                out[ax] = dft_length;
        }
        outputs.assign(1, out);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty());
        CV_Assert(outputs.size() == 1);
        CV_Assert(inputs[0].dims >= 1);

        const Mat &src = inputs[0];
        Mat &dst = outputs[0];

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
        CV_Assert(axis >= 0 && axis < (srcHasComplex ? ndims - 1 : ndims));
        CV_Assert(!onesided);

        std::vector<int> outSizesVec;
        outSizesVec.resize(srcHasComplex ? ndims : ndims + (srcLastIsOne ? 0 : 1));
        for (int i = 0; i < ndims; ++i) outSizesVec[i] = src.size[i];
        int complexDim = (int)outSizesVec.size() - 1;
        outSizesVec[complexDim] = 2;
        if (dft_length > 0)
        {
            int dstDimsNoComplex = (int)outSizesVec.size() - 1;
            if (axis >= 0 && axis < dstDimsNoComplex)
                outSizesVec[axis] = dft_length;
        }
        dst.create((int)outSizesVec.size(), &outSizesVec[0], src.type());

        std::vector<int> dimSizesSrc(ndims);
        for (int i = 0; i < ndims; ++i) {
            dimSizesSrc[i] = src.size[i];
        }
        std::vector<size_t> stridesSrc(ndims, 1);
        for (int i = ndims - 2; i >= 0; --i) {
            stridesSrc[i] = stridesSrc[i + 1] * (size_t)dimSizesSrc[i + 1];
        }
        const int ndimsDst = dst.dims;
        std::vector<int> dimSizesDst(ndimsDst);
        for (int i = 0; i < ndimsDst; ++i) {
            dimSizesDst[i] = dst.size[i];
        }
        std::vector<size_t> stridesDst(ndimsDst, 1);
        for (int i = ndimsDst - 2; i >= 0; --i) {
            stridesDst[i] = stridesDst[i + 1] * (size_t)dimSizesDst[i + 1];
        }

        int N = dimSizesSrc[axis];
        if (dft_length > 0) N = dft_length;
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
                                      axis, N, strideAxisSrc, strideAxisDst,
                                      srcHasComplex, inverse);
        }
        else if (depth == CV_64F)
        {
            runTypedDFT<double, Vec2d>(src, dst, dimSizesSrc, stridesSrc, stridesDst,
                                       iterDims, outerSizes, outerStep, totalOuter,
                                       axis, N, strideAxisSrc, strideAxisDst,
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
