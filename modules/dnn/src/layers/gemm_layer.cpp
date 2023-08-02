// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/fast_gemm.hpp"

namespace cv { namespace dnn {

class GemmLayerImpl CV_FINAL : public GemmLayer {
public:
    GemmLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.0f);
        beta = params.get<float>("beta", 1.0f);

        const_A = params.get<bool>("constA", false);
        const_B = params.get<bool>("constB", false);
        const_C = params.get<bool>("constC", false);
        have_bias = params.get<bool>("have_bias", false);

        real_ndims_C = params.get<int>("real_ndims_C", -1);

        // printf("const_A=%d, const_B=%d, const_C=%d\n", const_A, const_B, const_C);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    void prepare_broadcast_C(int M, int N, const Mat& C) {
        // printf("In broadcast, M=%d, N=%d\n", M, N);
        broadcast_C.clear();
        broadcast_C.resize(M * N);
        const auto shape_C = shape(C);
        float *ptr_bc = broadcast_C.data();
        const float *ptr_c = C.ptr<const float>();
        if (real_ndims_C == 0 || (real_ndims_C == 1 && shape_C[0] == 1) ||
            (real_ndims_C == 2 && shape_C[0] == 1 && shape_C[1] == 1)) {
            // (), (1,), (1, 1)
            float c = *ptr_c;
            int total = M * N;
            for (int i = 0; i < total; ++i) {
                ptr_bc[i] = c;
            }
        } else if ((real_ndims_C == 1 && shape_C[0] != 1) ||
                    (real_ndims_C == 2 && shape_C[0] == 1)) {
            // (N,), (1, N)
            for (int i = 0; i < M; ++i) {
                std::memcpy(ptr_bc + i * N, ptr_c, N * sizeof(float));
            }
        } else if (real_ndims_C == 2 && shape_C[1] == 1) {
            // (M, 1)
            for (int i = 0; i < M; ++i) {
                int step = i * M;
                for (int j = 0; j < N; ++j) {
                    ptr_bc[step + j] = ptr_c[i];
                }
            }
        } else {
            // (M, N)
            std::memcpy(ptr_bc, ptr_c, M * N * sizeof(float));
        }

        // std::cout << "broadcast_C=";
        // for (int i = 0; i < broadcast_C.size(); i++) {
        //     std::cout << broadcast_C[i] << " ";
        // }
        // std::cout << std::endl;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        // pack A or B if one of them is const
        if (const_A || const_B) {
            /*
            blobs:
                none,        invalid
                bias,        invalid
                a,
                b,
                a, bias
                b, bias
                a, b
                a, b, bias
            */
            if (blobs.size() == 1 || (blobs.size() == 2 && const_C)) {
                auto packer = const_A ? fast_gemm_packA : fast_gemm_packB;
                auto trans = const_A ? trans_a : trans_b;
                auto &packed = const_A ? packed_A : packed_B;
                packer(blobs[0], packed, trans);
            } else if (blobs.size() >= 2 && const_A && const_B) {
                fast_gemm_packA(blobs[0], packed_A);
                fast_gemm_packB(blobs[1], packed_B);
            }
        }

        // also pre-broadcast bias
        if (const_C) {
            const auto &C = blobs.back();

            std::vector<Mat> outputs;
            outputs_arr.getMatVector(outputs);
            const auto &Y = outputs[0];
            const auto shape_Y = shape(Y);
            int M = shape_Y[0], N = shape_Y[1];

            // broadcast
            prepare_broadcast_C(M, N, C);
        }
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        int num_inputs = static_cast<int>(inputs.size() + blobs.size());
        CV_CheckGE(num_inputs, 2, "DNN/Gemm: Gemm takes at least two inputs");
        CV_CheckLE(num_inputs, 3, "DNN/Gemm: Gemm takes at most three inputs");

        // Check whether A and B are two dimensional
        MatShape shape_A = const_A ? shape(blobs[0]) : inputs[0];
        MatShape shape_B;
        if (const_B) {
            shape_B = const_A ? shape(blobs[1]) : shape(blobs[0]);
        } else {
            shape_B = const_A ? inputs[0] : inputs[1];
        }

        // Check legal matrix multiplication
        int ma = shape_A[0], na = shape_A[1];
        int mb = shape_B[0], nb = shape_B[1];
        int M = trans_a ? na : ma;
        int N = trans_b ? mb : nb;
        int K_a = trans_a ? ma : na;
        int K_b = trans_b ? nb : mb;
        // printf("transA=%d, transB=%d\n", trans_a, trans_b);
        // printf("ma=%d, na=%d, mb=%d, nb=%d\n", ma, na, mb, nb);
        // printf("M=%d, N=%d, K_a=%d, K_b=%d\n", M, N, K_a, K_b);
        // printf("have_bias=%d\n", have_bias);
        CV_CheckEQ(K_a, K_b, "DNN/Gemm: Invalid dimension of dim K");

        // Check whether C can be unidirectional broadcast to (M, N). Handle carefully with 1D Mat.
        if (have_bias) {
            const auto shape_C = const_C ? shape(blobs.back()) : inputs.back();

            auto ndims_C = shape_C.size();
            CV_CheckLE(ndims_C, static_cast<size_t>(2), "DNN/Gemm: C can only be 0d (scalar) / 1d / 2d tensor");

            if (real_ndims_C == 1) { // (1,) or (N,)
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == N, "DNN/Gemm: C must be either of shape (1,) or (N,)");
            } else if (real_ndims_C == 2) { // (1, 1) or (1, N) or (M, 1) or (M, N)
                // printf("shape_C=[%d, %d]\n", shape_C[0], shape_C[1]);
                CV_Check(shape_C[0], (shape_C[0] == 1 && shape_C[1] == 1) ||
                                        (shape_C[0] == 1 && shape_C[1] == N) ||
                                        (shape_C[0] == M && shape_C[1] == 1) ||
                                        (shape_C[0] == M && shape_C[1] == N),
                                        "DNN/Gemm: C must be of shape (1, 1) or (1, N) or (M, 1) or (M, N)");
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

        const auto &A = const_A ? blobs[0] : inputs[0],
                   &B = const_B ? (const_A ? blobs[1] : blobs[0]) : (const_A ? inputs[0] : inputs[1]);
        auto &Y = outputs[0];

        const auto shape_A = shape(A),
                   shape_B = shape(B);

        int ma = shape_A[0], na = shape_A[1];
        int mb = shape_B[0], nb = shape_B[1];
        int M = trans_a ? na : ma;
        int N = trans_b ? mb : nb;
        int K = trans_a ? ma : na;

        // broadcast C and copy C to output
        if (have_bias) {
            if (!const_C) {
                prepare_broadcast_C(M, N, inputs.back());
            }
            CV_CheckEQ(broadcast_C.size(), static_cast<size_t>(M * N), "DNN/Gemm: broadcast_C is not prepared");
            float *ptr_y = Y.ptr<float>();
            std::memcpy(ptr_y, broadcast_C.data(), M * N * sizeof(float));
        } else { // initialization
            float *ptr_y = Y.ptr<float>();
            std::memset(ptr_y, 0, M * N * sizeof(float));
        }

        if (const_A && !const_B) {
            CV_CheckGT(packed_A.size(), static_cast<size_t>(0), "DNN/Gemm: constant A is not pre-packed");
            fast_gemm(trans_b, M, N, K, alpha, packed_A.data(), B.ptr<const float>(), nb, beta, Y.ptr<float>(), N);
        } else if (const_B && !const_A) {
            CV_CheckGT(packed_B.size(), static_cast<size_t>(0), "DNN/Gemm: constant B is not pre-packed");
            fast_gemm(trans_a, M, N, K, alpha, A.ptr<const float>(), na, packed_B.data(), beta, Y.ptr<float>(), N);
        } else if (const_A && const_B) {
            CV_CheckGT(packed_A.size(), static_cast<size_t>(0), "DNN/Gemm: constant A is not pre-packed");
            CV_CheckGT(packed_B.size(), static_cast<size_t>(0), "DNN/Gemm: constant B is not pre-packed");
            fast_gemm(M, N, K, alpha, packed_A.data(), packed_B.data(), beta, Y.ptr<float>(), N);
        } else {
            fast_gemm(trans_a, trans_b, alpha, A, B, beta, Y);
        }

    }

private:
    bool const_A;
    bool const_B;
    bool const_C;
    bool have_bias;
    std::vector<float> packed_A;
    std::vector<float> packed_B;
    std::vector<float> broadcast_C;
    int real_ndims_C;
};

Ptr<GemmLayer> GemmLayer::create(const LayerParams& params) {
    return makePtr<GemmLayerImpl>(params);
}

}} // namespace cv::dnn
