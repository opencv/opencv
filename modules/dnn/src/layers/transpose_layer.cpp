// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "cpu_kernels/transpose_kernels.simd.hpp"
#include "layers/cpu_kernels/transpose_kernels.simd_declarations.hpp"
#define CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN namespace cpu_baseline {
#define CV_CPU_OPTIMIZATION_NAMESPACE_END }
#undef CV_CPU_DISPATCH_MODES_ALL

#include "layers_common.hpp"
#include "../net_impl.hpp"
//#include "../op_cuda.hpp"
//#include "../op_inf_engine.hpp"
//#include "../ie_ngraph.hpp"
//#include "../op_webnn.hpp"
//#include "../op_timvx.hpp"
//#include "../op_cann.hpp"

//#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

/*
    Transpose layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Transpose.html

    Opset's 1 to 23 are covered.
*/

// Detect a "swap innermost two dims" permutation, collapsing leading dims.
// Returns true and fills outer/rows/cols when applicable.
static bool is_swap_last_two_f32(const MatShape& inpShape,
                                 const std::vector<int>& perm,
                                 int esz,
                                 int64_t& outer, int64_t& rows, int64_t& cols)
{
    if (esz != 4) return false;
    int nd = inpShape.dims;
    if (nd < 2 || (int)perm.size() != nd) return false;
    for (int i = 0; i < nd - 2; i++)
        if (perm[i] != i) return false;
    if (perm[nd - 2] != nd - 1 || perm[nd - 1] != nd - 2) return false;
    outer = 1;
    for (int i = 0; i < nd - 2; i++) outer *= inpShape[i];
    rows = inpShape[nd - 1]; // output rows = input inner dim
    cols = inpShape[nd - 2]; // output cols = input outer-of-pair dim
    return true;
}

static void transpose(const Mat& inp, const std::vector<int>& perm, Mat& out)
{
    enum {TRANSPOSE_MAX_DIMS=7};
    MatShape inpShape = inp.shape();
    MatShape outShape = out.shape();
    int ndims = inpShape.dims;
    size_t esz = inp.elemSize();
    CV_Assert(esz == 1 || esz == 2 || esz == 4 || esz == 8);

    int64_t outer = 1, rows = 0, cols = 0;
    if (is_swap_last_two_f32(inpShape, perm, (int)esz, outer, rows, cols)
        && inp.isContinuous() && out.isContinuous()) {
        CV_CPU_DISPATCH(transpose2D_f32_,
                        (inp.ptr<float>(), out.ptr<float>(), outer, rows, cols),
                        NEON, AVX2, AVX, BASELINE);
        return;
    }

    if (inp.isContinuous() && out.isContinuous() &&
        ndims >= 2 && (int)perm.size() == ndims &&
        perm[ndims - 1] == ndims - 1)
    {
        std::vector<int64_t> inStride(ndims, 0);
        inStride[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            inStride[i] = inStride[i + 1] * (int64_t)inpShape[i + 1];
        const int64_t inner = inpShape[ndims - 1];

        std::vector<int> outOuterShape(ndims - 1);
        for (int i = 0; i < ndims - 1; i++) outOuterShape[i] = outShape[i];
        int64_t outerTotal = 1;
        for (int i = 0; i < ndims - 1; i++) outerTotal *= outOuterShape[i];

        if (outerTotal > 0 && inner > 0) {
            const size_t innerBytes = (size_t)inner * esz;
            const char* in_base = (const char*)inp.data;
            char* out_base = (char*)out.data;

            parallel_for_(Range(0, (int)outerTotal), [&](const Range& r) {
                std::vector<int> outIdx(ndims - 1, 0);
                // initialize outIdx to r.start in row-major
                int64_t rem = r.start;
                for (int k = ndims - 2; k >= 0; k--) {
                    outIdx[k] = (int)(rem % outOuterShape[k]);
                    rem /= outOuterShape[k];
                }
                for (int64_t idx = r.start; idx < r.end; idx++) {
                    // Compute input outer offset using perm.
                    int64_t inOff = 0;
                    for (int i = 0; i < ndims - 1; i++) {
                        // perm[i] is the input axis providing this output axis.
                        inOff += (int64_t)outIdx[i] * inStride[perm[i]];
                    }
                    // Output position: linear idx * inner.
                    std::memcpy(out_base + (size_t)idx * innerBytes,
                                in_base + (size_t)inOff * esz,
                                innerBytes);
                    // Advance outIdx (row-major over outOuterShape).
                    for (int k = ndims - 2; k >= 0; k--) {
                        if (++outIdx[k] < outOuterShape[k]) break;
                        outIdx[k] = 0;
                    }
                }
            }, std::max(1.0, (double)outerTotal * (double)innerBytes / 16384.0));
            return;
        }
    }

    int perm_[TRANSPOSE_MAX_DIMS];
    int inpShape_[TRANSPOSE_MAX_DIMS];
    int outShape_[TRANSPOSE_MAX_DIMS];
    size_t inpStep_[TRANSPOSE_MAX_DIMS];
    int delta = TRANSPOSE_MAX_DIMS - ndims;

    CV_Assert(ndims <= TRANSPOSE_MAX_DIMS);
    CV_Assert(inp.isContinuous());
    CV_Assert(out.isContinuous());

    for (int i = 0; i < TRANSPOSE_MAX_DIMS; i++) {
        perm_[i] = i;
        inpShape_[i] = outShape_[i] = 1;
        inpStep_[i] = 0;
    }
    inpStep_[TRANSPOSE_MAX_DIMS-1] = 1; // step's are measured in elements, not bytes

    for(int i = 0; i < ndims; i++) {
        int j = perm.empty() ? ndims - i - 1 : perm[i];
        if (j < 0)
            j += ndims;
        CV_Assert(0 <= j && j < ndims);
        perm_[i + delta] = j + delta;
        int inpsz = inpShape[j];
        int outsz = outShape[i];
        CV_Assert(inpsz == outsz);
        inpShape_[i + delta] = inpShape[i];
        outShape_[i + delta] = outShape[i];
    }

    for (int i = TRANSPOSE_MAX_DIMS-2; i >= 0; i--)
        inpStep_[i] = inpStep_[i+1]*inpShape_[i+1];

    int sz6 = outShape_[0], sz5 = outShape_[1];
    int sz4 = outShape_[2], sz3 = outShape_[3];
    int sz2 = outShape_[4], sz1 = outShape_[5], sz0 = outShape_[6];
    size_t p6 = inpStep_[perm_[0]], p5 = inpStep_[perm_[1]];
    size_t p4 = inpStep_[perm_[2]], p3 = inpStep_[perm_[3]];
    size_t p2 = inpStep_[perm_[4]], p1 = inpStep_[perm_[5]], p0 = inpStep_[perm_[6]];

    int64_t outerTotal = (int64_t)sz6 * sz5 * sz4 * sz3 * sz2;

#undef CV_IMPLEMENT_TRANSPOSE
#define CV_IMPLEMENT_TRANSPOSE(typ) \
    parallel_for_(Range(0, (int)outerTotal), [&](const Range& r) { \
        const typ* inptr0 = (const typ*)inp.data; \
        typ* outptr0 = (typ*)out.data; \
        for (int64_t idx = r.start; idx < r.end; idx++) { \
            int64_t q = idx; \
            int i2 = (int)(q % sz2); q /= sz2; \
            int i3 = (int)(q % sz3); q /= sz3; \
            int i4 = (int)(q % sz4); q /= sz4; \
            int i5 = (int)(q % sz5); q /= sz5; \
            int i6 = (int)q; \
            const typ* inptrBase = inptr0 + i6*p6 + i5*p5 + i4*p4 + i3*p3 + i2*p2; \
            typ* outptr = outptr0 + idx * ((int64_t)sz1 * sz0); \
            for (int i1 = 0; i1 < sz1; i1++, outptr += sz0) { \
                const typ* inptr = inptrBase + i1*p1; \
                int i0 = 0; \
                for (; i0 <= sz0 - 3; i0 += 3) { \
                    size_t ip0 = i0*p0; \
                    typ t0 = inptr[ip0]; \
                    typ t1 = inptr[ip0+p0]; \
                    typ t2 = inptr[ip0+p0*2]; \
                    outptr[i0] = t0; \
                    outptr[i0+1] = t1; \
                    outptr[i0+2] = t2; \
                } \
                for (; i0 < sz0; i0++) \
                    outptr[i0] = inptr[i0*p0]; \
            } \
        } \
    });

    if (esz == 4) {
        CV_IMPLEMENT_TRANSPOSE(int)
    } else if (esz == 2) {
        CV_IMPLEMENT_TRANSPOSE(short)
    } else if (esz == 1) {
        CV_IMPLEMENT_TRANSPOSE(char)
    } else if (esz == 8) {
        CV_IMPLEMENT_TRANSPOSE(int64_t)
    }
}

class TransposeLayerImpl CV_FINAL : public TransposeLayer
{
public:
    TransposeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        perm = params.getVector<int>("perm");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const MatShape& inpShape) const
    {
        MatShape outShape(inpShape.dims);
        CV_Assert(perm.empty() || perm.size() == (size_t)inpShape.dims);

        for (int i = 0; i < inpShape.dims; i++) {
            int j = perm.empty() ? inpShape.dims - i - 1 : perm[i];
            CV_Assert(0 <= j && j < inpShape.dims);
            outShape[i] = inpShape[j];
        }

        return outShape;
    }

    bool isDataShuffling() const CV_OVERRIDE { return true; }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.assign(1, getOutShape(inputs[0]));
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.assign(requiredOutputs, inputs[0]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        CV_Assert(ninputs == 1);

        MatShape inpShape = inputs_arr.shape(0);
        MatShape outShape = getOutShape(inpShape);
        int inpType = inputs_arr.type(0);
        int outKind = outputs_arr.kind();

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runOp(inp, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            Mat temp(outShape, inpType);
            runOp(inp, temp);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& inp, Mat& out)
    {
        transpose(inp, perm, out);
    }
};

Ptr<TransposeLayer> TransposeLayer::create(const LayerParams& params)
{
    return Ptr<TransposeLayer>(new TransposeLayerImpl(params));
}

}
}
