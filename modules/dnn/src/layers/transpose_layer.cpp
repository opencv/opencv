// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
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

static void transpose(const Mat& inp, const std::vector<int>& perm, Mat& out)
{
    enum {TRANSPOSE_MAX_DIMS=7};
    MatShape inpShape = inp.shape();
    MatShape outShape = out.shape();
    int ndims = inpShape.dims;
    size_t esz = inp.elemSize();
    CV_Assert(esz == 1 || esz == 2 || esz == 4 || esz == 8);

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

#undef CV_IMPLEMENT_TRANSPOSE
#define CV_IMPLEMENT_TRANSPOSE(typ) \
    const typ* inptr0 = (const typ*)inp.data; \
    typ* outptr = (typ*)out.data; \
    for (int i6 = 0; i6 < sz6; i6++) { \
    for (int i5 = 0; i5 < sz5; i5++) { \
    for (int i4 = 0; i4 < sz4; i4++) { \
    for (int i3 = 0; i3 < sz3; i3++) { \
    for (int i2 = 0; i2 < sz2; i2++) { \
    for (int i1 = 0; i1 < sz1; i1++, outptr += sz0) { \
        int i0 = 0; \
        const typ* inptr = inptr0 + i6*p6 + i5*p5 + i4*p4 + i3*p3 + i2*p2 + i1*p1; \
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
    }}}}}}

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
        std::cout << "\n==>TransposeLayerImpl::forward" << std::endl;
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        CV_Assert(ninputs == 1);

        MatShape inpShape = inputs_arr.shape(0);
        std::cout << "input shape: " << inpShape << std::endl;
        auto *inp_ptr = inputs_arr.getMat(0).ptr<float>();
        for (int i = 0; i < inputs_arr.getMat(0).total(); i++){
            std::cout << inp_ptr[i] << " ";
        }
        MatShape outShape = getOutShape(inpShape);
        std::cout << "output shape" << outShape << std::endl;
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
        auto *out_ptr = outputs_arr.getMat(0).ptr<float>();
        for (int i = 0; i < outputs_arr.getMat(0).total(); i++){
            std::cout << out_ptr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "==>TransposeLayerImpl::forward::done\n" << std::endl;
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
