// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_EINSUM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_EINSUM_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cublas.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/permute.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct EinsumMatMulPlan {
        std::vector<std::size_t> permA, permB, permOut;
        int numBatch = 0, numM = 0, numK = 0, numN = 0;
    };

    template <class T>
    class EinsumMatMulOp final : public CUDABackendNode {
    public:
        EinsumMatMulOp(csl::Stream stream_, csl::cublas::Handle handle, EinsumMatMulPlan plan_)
            : stream(std::move(stream_)), cublasHandle(std::move(handle)), plan(std::move(plan_))
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto a = csl::viewOf<T>(inputs[0]);
            auto b = csl::viewOf<T>(inputs[1]);
            auto output = csl::spanOf<T>(outputs[0]);

            const MatShape shapeA = cv::dnn::shape(inputs[0]);
            const MatShape shapeB = cv::dnn::shape(inputs[1]);
            CV_Assert(shapeA.dims == (int)plan.permA.size() && shapeB.dims == (int)plan.permB.size());

            std::vector<std::size_t> dimsA(plan.permA.size()), dimsB(plan.permB.size());
            for (std::size_t i = 0; i < plan.permA.size(); i++)
                dimsA[i] = static_cast<std::size_t>(shapeA[(int)plan.permA[i]]);
            for (std::size_t i = 0; i < plan.permB.size(); i++)
                dimsB[i] = static_cast<std::size_t>(shapeB[(int)plan.permB[i]]);

            const int nb = plan.numBatch, nm = plan.numM, nk = plan.numK, nn = plan.numN;
            std::size_t batch = 1, M = 1, K = 1, N = 1;
            for (int i = 0; i < nb; i++) {
                CV_Assert(dimsA[i] == dimsB[i]);
                batch *= dimsA[i];
            }
            for (int i = 0; i < nm; i++)
                M *= dimsA[nb + i];
            for (int i = 0; i < nk; i++) {
                CV_Assert(dimsA[nb + nm + i] == dimsB[nb + i]);
                K *= dimsA[nb + nm + i];
            }
            for (int i = 0; i < nn; i++)
                N *= dimsB[nb + nk + i];
            CV_Assert(output.size() == batch * M * N);

            csl::TensorView<T> pa = a, pb = b;
            if (!isIdentity(plan.permA)) {
                grow(bufA, a.size());
                csl::TensorSpan<T> span(bufA.get(), dimsA.begin(), dimsA.end());
                kernels::permute<T>(stream, span, a, plan.permA);
                pa = span;
            }
            if (!isIdentity(plan.permB)) {
                grow(bufB, b.size());
                csl::TensorSpan<T> span(bufB.get(), dimsB.begin(), dimsB.end());
                kernels::permute<T>(stream, span, b, plan.permB);
                pb = span;
            }

            const std::size_t shapeA3[] = { batch, M, K }, shapeB3[] = { batch, K, N }, shapeC3[] = { batch, M, N };
            csl::TensorView<T> a3(pa.get(), std::begin(shapeA3), std::end(shapeA3));
            csl::TensorView<T> b3(pb.get(), std::begin(shapeB3), std::end(shapeB3));

            if (isIdentity(plan.permOut)) {
                csl::TensorSpan<T> c3(output.get(), std::begin(shapeC3), std::end(shapeC3));
                csl::tensor_ops::gemmStridedBatched<T>(cublasHandle, static_cast<T>(0.f), c3, static_cast<T>(1.f),
                                                       false, a3, false, b3);
                return;
            }

            grow(bufC, batch * M * N);
            csl::TensorSpan<T> c3(bufC.get(), std::begin(shapeC3), std::end(shapeC3));
            csl::tensor_ops::gemmStridedBatched<T>(cublasHandle, static_cast<T>(0.f), c3, static_cast<T>(1.f),
                                                   false, a3, false, b3);

            std::vector<std::size_t> dimsC;
            dimsC.insert(dimsC.end(), dimsA.begin(), dimsA.begin() + nb + nm);
            dimsC.insert(dimsC.end(), dimsB.begin() + nb + nk, dimsB.end());
            csl::TensorView<T> cfull(bufC.get(), dimsC.begin(), dimsC.end());
            kernels::permute<T>(stream, output, cfull, plan.permOut);
        }

    private:
        static bool isIdentity(const std::vector<std::size_t>& perm)
        {
            for (std::size_t i = 0; i < perm.size(); i++)
                if (perm[i] != i)
                    return false;
            return true;
        }

        static void grow(csl::Tensor<T>& buf, std::size_t n)
        {
            if (buf.empty() || buf.size() < n)
                buf.resize(n);
        }

        csl::Stream stream;
        csl::cublas::Handle cublasHandle;
        EinsumMatMulPlan plan;
        csl::Tensor<T> bufA, bufB, bufC;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_EINSUM_HPP */
