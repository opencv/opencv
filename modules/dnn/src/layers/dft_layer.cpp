// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <numeric>

using namespace std;

namespace cv { namespace dnn {

class DFTLayerImpl CV_FINAL : public DFTLayer {
public:
    DFTLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        inverse = params.get<int>("inverse", 0) != 0;
        onesided = params.get<int>("onesided", 0) != 0;
        if (params.has("axes"))
        {
            DictValue dv = params.get("axes");
            axes.resize(dv.size());
            for (int i = 0; i < dv.size(); ++i)
                axes[i] = dv.get<int>(i);
        }
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs);
        CV_UNUSED(internals);
        CV_Assert(inputs.size() >= 1);
        const MatShape &inshape = inputs[0];
        MatShape out = inshape;
        // If input ends with 2 (complex), keep.
        // If input ends with 1, replace it with 2.
        // Otherwise, append a trailing 2.
        if (!out.empty())
        {
            if (out.back() == 2)
            {
                // already complex
            }
            else if (out.back() == 1)
            {
                out.back() = 2;
            }
            else
            {
                out.push_back(2);
            }
        }
        outputs.assign(1, out);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_UNUSED(internals_arr);
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty());
        CV_Assert(outputs.size() == 1);
        CV_Assert(inputs[0].dims >= 1);

        const Mat &src = inputs[0];
        Mat &dst = outputs[0];

        // Dimensions
        const int ndims = src.dims;
        CV_Assert(ndims >= 1);
        const bool srcHasComplex = (src.size[ndims - 1] == 2);
        const bool srcLastIsOne = (src.size[ndims - 1] == 1);

        // Create output with proper shape following getMemoryShapes logic
        if (srcHasComplex)
        {
            dst.create(src.dims, src.size.p, src.type());
        }
        else if (srcLastIsOne)
        {
            std::vector<int> outSizes(ndims);
            for (int i = 0; i < ndims; ++i) outSizes[i] = src.size[i];
            outSizes[ndims - 1] = 2;
            dst.create((int)outSizes.size(), &outSizes[0], src.type());
        }
        else
        {
            std::vector<int> outSizes(ndims + 1);
            for (int i = 0; i < ndims; ++i) outSizes[i] = src.size[i];
            outSizes[ndims] = 2;
            dst.create((int)outSizes.size(), &outSizes[0], src.type());
        }

        // Determine single transform axis (only 1D supported for now)
        int axis = (srcHasComplex ? ndims - 2 : ndims - 1); // default axis excludes complex dim when present
        if (!axes.empty())
        {
            CV_Assert(axes.size() == 1);
            axis = axes[0];
            if (axis < 0) axis += ndims; // axis is defined on input dims
        }
        CV_Assert(axis >= 0 && axis < (srcHasComplex ? ndims - 1 : ndims));
        CV_Assert(!onesided); // onesided not supported yet

        // Prepare strides (in element counts)
        // Src sizes/strides
        std::vector<int> dimSizesSrc(ndims);
        for (int i = 0; i < ndims; ++i) dimSizesSrc[i] = src.size[i];
        std::vector<size_t> stridesSrc(ndims, 1);
        // if src has complex, last dim is complex with stride 1; if not, last stride is for last real dim
        int startStrideFrom = srcHasComplex ? ndims - 2 : ndims - 1;
        for (int i = startStrideFrom; i >= 0; --i)
            stridesSrc[i] = stridesSrc[i + 1] * (size_t)dimSizesSrc[i + 1];

        // Dst sizes/strides
        const int ndimsDst = dst.dims;
        std::vector<int> dimSizesDst(ndimsDst);
        for (int i = 0; i < ndimsDst; ++i) dimSizesDst[i] = dst.size[i];
        std::vector<size_t> stridesDst(ndimsDst, 1);
        for (int i = ndimsDst - 2; i >= 0; --i)
            stridesDst[i] = stridesDst[i + 1] * (size_t)dimSizesDst[i + 1];

        const int N = dimSizesSrc[axis];
        const size_t strideAxisSrc = stridesSrc[axis];
        const size_t strideAxisDst = stridesDst[axis];

        // Build list of dims to iterate (exclude axis and exclude complex dim in src if present)
        std::vector<int> iterDims;
        for (int i = 0; i < (srcHasComplex ? ndims - 1 : ndims); ++i) if (i != axis) iterDims.push_back(i);

        // Total sequences = product of iterDims sizes
        size_t totalSeq = 1;
        for (int d : iterDims) totalSeq *= (size_t)dimSizesSrc[d];

        // Prepare index counters
        std::vector<int> idx(iterDims.size(), 0);
        auto nextIndex = [&](bool &done){
            for (int p = (int)idx.size() - 1; p >= 0; --p)
            {
                idx[p]++;
                if (idx[p] < dimSizesSrc[iterDims[p]]) return;
                idx[p] = 0;
            }
            done = true;
        };

        const int depth = src.depth();
        if (depth == CV_32F)
        {
            const float* sp = src.ptr<float>();
            float* dp = dst.ptr<float>();
            bool done = false;
            while (!done)
            {
                // Compute base offset in elements for current sequence (axis index = 0, last complex index = 0)
                size_t baseSrc = 0;
                for (size_t t = 0; t < idx.size(); ++t)
                {
                    int d = iterDims[t];
                    baseSrc += (size_t)idx[t] * stridesSrc[d];
                }

                // For dst base, same indices map to same dims (complex dim is appended)
                size_t baseDst = 0;
                for (size_t t = 0; t < idx.size(); ++t)
                {
                    int d = iterDims[t];
                    baseDst += (size_t)idx[t] * stridesDst[d];
                }

                const float* in = sp + baseSrc;
                float* out = dp + baseDst;
                const float sign = inverse ? 1.f : -1.f;

                for (int k = 0; k < N; ++k)
                {
                    double accRe = 0.0, accIm = 0.0;
                    for (int n = 0; n < N; ++n)
                    {
                        double angle = 2.0 * CV_PI * k * n / N;
                        double c = cos(angle);
                        double s = sin(angle) * sign;
                        size_t offSrc = (size_t)n * strideAxisSrc;
                        double re, im;
                        if (srcHasComplex)
                        {
                            re = in[offSrc + 0];
                            im = in[offSrc + 1];
                        }
                        else
                        {
                            re = in[offSrc];
                            im = 0.0;
                        }
                        accRe += re * c - im * s;
                        accIm += re * s + im * c;
                    }
                    if (inverse)
                    {
                        accRe /= N; accIm /= N;
                    }
                    size_t ok = (size_t)k * strideAxisDst;
                    out[ok + 0] = (float)accRe;
                    out[ok + 1] = (float)accIm;
                }

                nextIndex(done);
            }
        }
        else if (depth == CV_64F)
        {
            const double* sp = src.ptr<double>();
            double* dp = dst.ptr<double>();
            bool done = false;
            while (!done)
            {
                size_t baseSrc = 0;
                for (size_t t = 0; t < idx.size(); ++t)
                {
                    int d = iterDims[t];
                    baseSrc += (size_t)idx[t] * stridesSrc[d];
                }

                size_t baseDst = 0;
                for (size_t t = 0; t < idx.size(); ++t)
                {
                    int d = iterDims[t];
                    baseDst += (size_t)idx[t] * stridesDst[d];
                }

                const double* in = sp + baseSrc;
                double* out = dp + baseDst;
                const double sign = inverse ? 1.0 : -1.0;

                for (int k = 0; k < N; ++k)
                {
                    double accRe = 0.0, accIm = 0.0;
                    for (int n = 0; n < N; ++n)
                    {
                        double angle = 2.0 * CV_PI * k * n / N;
                        double c = cos(angle);
                        double s = sin(angle) * sign;
                        size_t offSrc = (size_t)n * strideAxisSrc;
                        double re, im;
                        if (srcHasComplex)
                        {
                            re = in[offSrc + 0];
                            im = in[offSrc + 1];
                        }
                        else
                        {
                            re = in[offSrc];
                            im = 0.0;
                        }
                        accRe += re * c - im * s;
                        accIm += re * s + im * c;
                    }
                    if (inverse)
                    {
                        accRe /= N; accIm /= N;
                    }
                    size_t ok = (size_t)k * strideAxisDst;
                    out[ok + 0] = accRe;
                    out[ok + 1] = accIm;
                }

                nextIndex(done);
            }
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "DFT supports float32/float64 only");
        }
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs);
        CV_UNUSED(requiredInternals);
        CV_UNUSED(internals);
        outputs.assign(1, inputs[0]);
    }

private:
    bool inverse;
    bool onesided;
    std::vector<int> axes;
};

Ptr<DFTLayer> DFTLayer::create(const LayerParams& params)
{
    return makePtr<DFTLayerImpl>(params);
}

}} // namespace
