// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/fast_gemm.hpp"

namespace cv { namespace dnn {

class MatMulLayerImpl CV_FINAL : public MatMulLayer {
 public:
    MatMulLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.f);
        beta = params.get<float>("beta", 1.f);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(2), "DNN/MatMul: two inputs required");

        const auto shape_A = inputs[0], shape_B = inputs[1];
        CV_CheckGE(shape_A.size(), static_cast<size_t>(2), "DNN/MatMul: invalid shape of input A");
        CV_CheckGE(shape_B.size(), static_cast<size_t>(2), "DNN/MatMul: invalid shape of input B");

        // Check legal matrix multiplication
        int mA = shape_A[shape_A.size() - 2], nA = shape_A.back();
        int mB = shape_B[shape_B.size() - 2], nB = shape_B.back();
        int M = trans_a ? nA : mA;
        int N = trans_b ? mB : nB;
        int K_A = trans_a ? mA : nA;
        int K_B = trans_b ? nB : mB;
        CV_CheckEQ(K_A, K_B, "DNN/MatMul: invalid dimension K");

        // Check legal broadcast. It is legal for sure if A and B are 2d, or one of them is 2d.
        MatShape common_shape;
        if (shape_A.size() != 2 || shape_B.size() != 2) {
            const auto &shape_more_dims = shape_A.size() > shape_B.size() ? shape_A : shape_B;
            const auto &shape_less_dims = shape_A.size() > shape_B.size() ? shape_B : shape_A;
            size_t diff_dims = shape_more_dims.size() - shape_less_dims.size();
            common_shape = shape_more_dims;
            for (size_t i = 0; i < shape_less_dims.size() - 2; i++) {
                const auto dl = shape_less_dims[i], dm = shape_more_dims[i + diff_dims];
                if (dl != 1 && dm != 1 && dl != dm) {
                    CV_Error(Error::StsBadSize, format("DNN/MatMul: invalid shape for broadcasting, shape_A[%zu]=%d, shape_B[%zu]=%d\n", i, shape_less_dims[i], i, shape_more_dims[i + diff_dims]));
                }

                if (dm == 1) {
                    common_shape[i + diff_dims] = dl;
                }
            }
            common_shape[common_shape.size() - 2] = M;
            common_shape[common_shape.size() - 1] = N;
        } else {
            common_shape.resize(2);
            common_shape[0] = M;
            common_shape[1] = N;
        }

        outputs.assign(1, common_shape);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        const auto A_shape = shape(inputs[0]), B_shape = shape(inputs[1]), C_shape = shape(outputs[0]);
        helper.compute(trans_a, trans_b, A_shape, B_shape, C_shape);
    }

    // works like Y = numpy.matmul(A, B)
    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto &A = inputs[0], &B = inputs[1];
        auto &Y = outputs[0];

        const auto *a = A.ptr<const float>(), *b = B.ptr<const float>();
        auto *y = Y.ptr<float>();
        std::memset(y, 0, Y.total() * sizeof(float));
        fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                      helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1, b, helper.ldb0,
                      helper.ldb1, beta, y, helper.ldc, opt);
    }

 private:
    bool trans_a;
    bool trans_b;
    float alpha;
    float beta;

    FastGemmOpt opt;
    MatMulHelper helper;
};

Ptr<MatMulLayer> MatMulLayer::create(const LayerParams& params)
{
    return makePtr<MatMulLayerImpl>(params);
}

}} // cv::dnn
