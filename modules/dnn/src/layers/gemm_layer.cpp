// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/gemm.impl.hpp"

namespace cv { namespace dnn {

class GemmLayerImpl CV_FINAL : public GemmLayer {
public:
    GemmLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.0f);
        beta = params.get<float>("beta", 1.0f);

        // C is initialized and broadcast in finalize()
        real_ndims_C = params.get<int>("real_ndims_C", -1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        size_t num_inputs = inputs.size();
        CV_CheckGE(num_inputs, static_cast<size_t>(2), "DNN/Gemm: Gemm takes at least two inputs");
        CV_CheckLE(num_inputs, static_cast<size_t>(3), "DNN/Gemm: Gemm takes at most three inputs");

        // Check whether A and B are two dimensional
        const auto shape_A = inputs[0], shape_B = inputs[1];
        CV_CheckEQ(shape_A.size(), static_cast<size_t>(2), "DNN/Gemm: Input A should be two dimensional");
        CV_CheckEQ(shape_B.size(), static_cast<size_t>(2), "DNN/Gemm: Input B should be two dimensional");

        // Check legal matrix multiplication
        int ma = shape_A[0], na = shape_A[1];
        int mb = shape_B[0], nb = shape_B[1];
        int M = trans_a ? na : ma;
        int N = trans_b ? mb : nb;
        int K_a = trans_a ? ma : na;
        int K_b = trans_b ? nb : mb;
        CV_CheckEQ(K_a, K_b, "DNN/Gemm: Invalid dimension of dim K");

        // Check whether C can be unidirectional broadcast to (M, N). Handle carefully with 1D Mat.
        if (inputs.size() == static_cast<size_t>(3)) {
            const auto shape_C = inputs[2];

            auto ndims_C = shape_C.size();
            CV_CheckLE(ndims_C, static_cast<size_t>(2), "DNN/Gemm: C can only be 0d (scalar) / 1d / 2d tensor");

            if (ndims_C == 1) { // scalar
                CV_Check(shape_C[0], shape_C[0] == 1, "DNN/Gemm: scalar C cannot be broadcast");
            } else if (ndims_C == 2 && real_ndims_C == 1) { // 1d tensor
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == N, "DNN/Gemm: 1d C cannot be broadcast");
            } else if (ndims_C == 2 && real_ndims_C == 2) {
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == M, "DNN/Gemm: 2d C cannot be broadcast");
                CV_Check(shape_C[1], shape_C[1] == 1 || shape_C[1] == N, "DNN/Gemm: 2d C cannot be broadcast");
            } else {
                CV_Error(Error::StsBadArg, "DNN/Gemm: Input C can not be unidirectional broadcastable to (M, N)");
            }
        }

        MatShape shape_y{M, N};
        outputs.assign(1, shape_y);
        return false;
    }

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

        // broadcast C and copy C to output
        if (inputs.size() == 3) {
            auto C = inputs[2].clone();

            const auto shape_A = shape(A),
                       shape_B = shape(B),
                       shape_C = shape(C);

            int ma = shape_A[0], na = shape_A[1];
            int mb = shape_B[0], nb = shape_B[1];
            int M = trans_a ? na : ma;
            int N = trans_b ? mb : nb;

            // broadcast
            float *ptr_y = Y.ptr<float>();
            const float *ptr_c = C.ptr<const float>();
            if (real_ndims_C == 0 || (real_ndims_C == 1 && shape_C[0] == 1) ||
                (real_ndims_C == 2 && shape_C[0] == 1 && shape_C[1] == 1)) {
                // (), (1,), (1, 1)
                float c = C.at<float>(0);
                int total = M * N;
                for (int i = 0; i < total; ++i) {
                    ptr_y[i] = c;
                }
            } else if ((real_ndims_C == 1 && shape_C[0] != 1) ||
                       (real_ndims_C == 2 && shape_C[0] == 1)) {
                // (N,), (1, N)
                for (int i = 0; i < M; ++i) {
                    std::memcpy(ptr_y + i * N, ptr_c, N * sizeof(float));
                }
            } else if (real_ndims_C == 2 && shape_C[1] == 1) {
                // (M, 1)
                float *ptr_c = C.ptr<float>();
                for (int i = 0; i < M; ++i) {
                    int step = i * M;
                    for (int j = 0; j < N; ++j) {
                        ptr_y[step + j] = ptr_c[i];
                    }
                }
            } else {
                // (M, N)
                std::memcpy(ptr_y, ptr_c, M * N * sizeof(float));
            }
        }

        ocv_gemm(trans_a, trans_b, alpha, A, B, beta, Y);
    }

private:
    int real_ndims_C;
};

Ptr<GemmLayer> GemmLayer::create(const LayerParams& params) {
    return makePtr<GemmLayerImpl>(params);
}

}} // namespace cv::dnn
