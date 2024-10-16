// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

namespace cv
{
namespace dnn
{

/*
    DequantizeLinear layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

    Opset's 10 to 23 are covered.
*/

template <typename _InpTp, typename _ScaleTp, typename _OutTp>
static void dequantizeLinear(const _InpTp* inp_, const _ScaleTp* scale_,
                             const _InpTp* zp_, _OutTp* out_,
                             int64_t nslices, int sz_a_,
                             int64_t slice_size_, int block_size_)
{
    int bsz_ = std::max(block_size_, 1);
    int nblocks_per_axis = (sz_a_ + bsz_ - 1) / bsz_;
    int64_t nmacro_blocks = nslices * nblocks_per_axis;
    CV_Assert(nmacro_blocks <= (int64_t)INT_MAX);

    parallel_for_(Range(0, (int)nmacro_blocks), [&](const Range& r) {
        int sz_a = sz_a_;
        int64_t slice_size = slice_size_;
        int block_size = block_size_;
        int delta = 0;
        int64_t scale_step = block_size > 0 ? slice_size : 1;
        int64_t zp_step = zp_ ? scale_step : 0;

        for (int i = r.start; i < r.end; i += delta) {
            int slice_idx = i / nblocks_per_axis;
            int block_idx = i - slice_idx * nblocks_per_axis;
            int64_t block_ofs, scale_ofs;
            if (block_size > 0) {
                delta = std::min(nblocks_per_axis - block_idx, r.end - i);
                block_ofs = (slice_idx*sz_a + block_idx*block_size)*slice_size;
                scale_ofs = (slice_idx*nblocks_per_axis + block_idx)*slice_size;
            } else {
                delta = std::min(sz_a - block_idx, r.end - i);
                block_ofs = (slice_idx*sz_a + block_idx)*slice_size;
                scale_ofs = block_idx;
            }
            const _InpTp* inp = inp_ + block_ofs;
            const _InpTp* zp = zp_ ? zp_ + scale_ofs : nullptr;
            const _ScaleTp* sc = scale_ + scale_ofs;
            _OutTp* out = out_ + block_ofs;

            // [TODO] vectorize using intrinsics
            if (slice_size > 1) {
                for (int k = 0; k < delta; k++, inp += slice_size, out += slice_size,
                                                sc += scale_step, zp += zp_step) {
                    float scval = (float)*sc;
                    _InpTp zpval = zp ? *zp : (_InpTp)0;

                    for (int64_t j = 0; j < slice_size; j++)
                        out[j] = _OutTp((inp[j] - zpval)*scval);
                }
            } else if (block_size > 0 ) {
                int bsz = block_size;
                for (int k = 0; k < delta; k++, inp += bsz, out += bsz) {
                    bsz = std::min(bsz, sz_a - (block_idx + k)*block_size);
                    float scval = (float)sc[k];
                    _InpTp zpval = zp ? zp[k] : (_InpTp)0;

                    for (int j = 0; j < bsz; j++)
                        out[j] = _OutTp((inp[j] - zpval)*scval);
                }
                sc += delta;
                zp += zp ? delta : 0;
            } else {
                if (zp) {
                    for (int j = 0; j < delta; j++) {
                        float scval = (float)sc[j];
                        _InpTp zpval = zp[j];
                        out[j] = _OutTp((inp[j] - zpval)*scval);
                    }
                } else {
                    for (int j = 0; j < delta; j++) {
                        float scval = (float)sc[j];
                        out[j] = _OutTp(inp[j]*scval);
                    }
                }
                inp += delta;
                out += delta;
            }
        }
    });
}

// Dequantize INT8/UINT8 to FP32/FP16; out must be preallocated
static void dequantizeLinear(const Mat& inp, const Mat& scale_, const Mat& zp,
                             int axis, int block_size, Mat& out)
{
    Mat scale = scale_;
    CV_Assert(inp.isContinuous());
    CV_Assert(scale.isContinuous());
    CV_Assert(out.isContinuous());

    int inptype = inp.type();
    int outtype = out.type();
    int sctype = scale.type();
    int zptype = zp.type();
    MatShape inpshape = inp.shape();
    MatShape scshape = scale.shape();
    MatShape zpshape = zp.shape();
    int i, ndims = inpshape.dims;
    int64_t nslices = 1, slice_size = 1;

    CV_Assert(inptype == CV_8U || inptype == CV_8S);
    CV_Assert(sctype == CV_32F || sctype == CV_16F);
    CV_Assert(outtype == CV_32F || outtype == CV_16F);

    if (!zp.empty()) {
        CV_Assert(zp.isContinuous());
        CV_Assert(zptype == inptype);
        CV_Assert(zpshape == scshape);
    }

    axis = normalize_axis(axis, ndims);
    for (i = 0; i < axis; i++)
        nslices *= inpshape[i];
    for (i = axis+1; i < ndims; i++)
        slice_size *= inpshape[i];
    int sz_a = inpshape[axis];

    if (block_size == 0) {
        size_t sc_total = scshape.total();
        CV_Assert(scale.dims <= 1);
        CV_Assert(sc_total == 1 || sc_total == (size_t)sz_a);

        // unroll the innermost loop if the scale's/zp's are the same
        if (sc_total == 1) {
            slice_size *= sz_a;
            sz_a = 1;
        }

        // avoid FP16 => FP32 conversion for scale inside the innermost loop
        if (sctype == CV_16F && slice_size == 1 && nslices > 1) {
            Mat temp;
            scale_.convertTo(temp, CV_32F);
            scale = temp;
            sctype = CV_32F;
        }
    } else {
        CV_Assert(block_size > 0);
        CV_Assert(scale.dims == ndims);
        for (int i = 0; i < ndims; i++) {
            int inp_i = inpshape[i];
            int sc_i = scshape[i];
            if (i == axis) {
                CV_Assert((inp_i + block_size - 1)/block_size == sc_i);
            } else {
                CV_Assert(sc_i == inp_i);
            }
        }
    }

    if (inptype == CV_8U && sctype == CV_32F && outtype == CV_32F)
        dequantizeLinear(reinterpret_cast<const uint8_t*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<float*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (inptype == CV_8U && sctype == CV_16F && outtype == CV_32F)
        dequantizeLinear(reinterpret_cast<const uint8_t*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<float*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (inptype == CV_8U && sctype == CV_32F && outtype == CV_16F)
        dequantizeLinear(reinterpret_cast<const uint8_t*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<hfloat*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (inptype == CV_8U && sctype == CV_16F && outtype == CV_16F)
        dequantizeLinear(reinterpret_cast<const uint8_t*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<hfloat*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (inptype == CV_8S && sctype == CV_32F && outtype == CV_32F)
        dequantizeLinear(reinterpret_cast<const int8_t*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<float*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (inptype == CV_8S && sctype == CV_16F && outtype == CV_32F)
        dequantizeLinear(reinterpret_cast<const int8_t*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<float*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (inptype == CV_8S && sctype == CV_32F && outtype == CV_16F)
        dequantizeLinear(reinterpret_cast<const int8_t*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<hfloat*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (inptype == CV_8S && sctype == CV_16F && outtype == CV_16F)
        dequantizeLinear(reinterpret_cast<const int8_t*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<hfloat*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else {
        CV_Error_(Error::StsNotImplemented,
                  ("the following combination of types is not supported in "
                   "DequantizeLinear: inp=%s, scale=%s, out=%s",
                   typeToString(inptype).c_str(),
                   typeToString(sctype).c_str(),
                   typeToString(outtype).c_str()));
    }
}

class DequantizeLinearLayerImpl CV_FINAL : public DequantizeLinearLayer
{
public:
    DequantizeLinearLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", 1);
        block_size = params.get<int>("block_size", 0);
        CV_Assert(block_size >= 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(2 <= ninputs && ninputs <= 3);
        CV_Assert(requiredOutputs == 1);
        outputs.assign(1, inputs[0]);
        return true;
    }

    int getOutType() const
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        return netimpl_->enableFP16 ? CV_16F : CV_32F;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(2 <= ninputs && ninputs <= 3);
        if (ninputs == 3) {
            CV_Assert(inputs[0] == inputs[2]);
        }
        outputs.assign(1, getOutType());
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        int ninputs = inputs_arr.size(-1).area();
        CV_Assert(2 <= ninputs && ninputs <= 3);

        Mat inp = inputs_arr.getMat(0);
        Mat scale = inputs_arr.getMat(1);
        Mat zeropoint;
        int outtype = getOutType();
        MatShape inpshape = inp.shape();

        if (ninputs >= 3) {
            zeropoint = inputs_arr.getMat(2);
        }

        auto kind = outputs_arr.kind();

        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(inpshape, outtype);
            dequantizeLinear(inp, scale, zeropoint, axis, block_size, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(inpshape, outtype);
            Mat temp(inpshape, outtype);
            dequantizeLinear(inp, scale, zeropoint, axis, block_size, temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
};

Ptr<DequantizeLinearLayer> DequantizeLinearLayer::create(const LayerParams& params)
{
    return Ptr<DequantizeLinearLayer>(new DequantizeLinearLayerImpl(params));
}

}}
