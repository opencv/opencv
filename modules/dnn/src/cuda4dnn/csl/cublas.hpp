// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_CUBLAS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_CUBLAS_HPP

#include "error.hpp"
#include "stream.hpp"
#include "pointer.hpp"
#include "memory.hpp"

#include <opencv2/core.hpp>

#include <cublas_v2.h>

#include <cstddef>
#include <memory>
#include <utility>

#define CUDA4DNN_CHECK_CUBLAS(call) \
    ::cv::dnn::cuda4dnn::csl::cublas::detail::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cublas {

    /** @brief exception class for errors thrown by the cuBLAS API */
    class cuBLASException : public CUDAException {
    public:
        using CUDAException::CUDAException;
    };

    namespace detail {
        static void check(cublasStatus_t status, const char* func, const char* file, int line) {
            auto cublasGetErrorString = [](cublasStatus_t err) {
                switch (err) {
                case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
                case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
                case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
                case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
                case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
                case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
                case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
                case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
                case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
                case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
                }
                return "UNKNOWN_CUBLAS_ERROR";
            };

            if (status != CUBLAS_STATUS_SUCCESS)
                throw cuBLASException(Error::GpuApiCallError, cublasGetErrorString(status), func, file, line);
        }
    }

    /** non-copyable cuBLAS smart handle
     *
     * UniqueHandle is a smart non-sharable wrapper for cuBLAS handle which ensures that the handle
     * is destroyed after use. The handle must always be associated with a non-default stream. The stream
     * must be specified during construction.
     *
     * Refer to stream API for more information for the choice of forcing non-default streams.
     */
    class UniqueHandle {
    public:
        UniqueHandle() noexcept : handle{ nullptr } { }
        UniqueHandle(UniqueHandle&) = delete;
        UniqueHandle(UniqueHandle&& other) noexcept {
            stream = std::move(other.stream);
            handle = other.handle;
            other.handle = nullptr;
        }

        /** creates a cuBLAS handle and associates it with the stream specified
         *
         * Exception Guarantee: Basic
         */
        UniqueHandle(Stream strm) : stream(std::move(strm)) {
            CV_Assert(stream);
            CUDA4DNN_CHECK_CUBLAS(cublasCreate(&handle));
            try {
                CUDA4DNN_CHECK_CUBLAS(cublasSetStream(handle, stream.get()));
            } catch (...) {
                /* cublasDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUBLAS(cublasDestroy(handle));
                throw;
            }
        }

        ~UniqueHandle() noexcept {
            if (handle) {
                /* cublasDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUBLAS(cublasDestroy(handle));
            }
        }

        UniqueHandle& operator=(const UniqueHandle&) = delete;
        UniqueHandle& operator=(UniqueHandle&& other) noexcept {
            CV_Assert(other);
            if (&other != this) {
                UniqueHandle(std::move(*this)); /* destroy current handle */
                stream = std::move(other.stream);
                handle = other.handle;
                other.handle = nullptr;
            }
            return *this;
        }

        /** returns the raw cuBLAS handle */
        cublasHandle_t get() const noexcept {
            CV_Assert(handle);
            return handle;
        }

        /** returns true if the handle is valid */
        explicit operator bool() const noexcept { return static_cast<bool>(handle); }

    private:
        Stream stream;
        cublasHandle_t handle;
    };

    /** @brief sharable cuBLAS smart handle
     *
     * Handle is a smart sharable wrapper for cuBLAS handle which ensures that the handle
     * is destroyed after all references to the handle are destroyed. The handle must always
     * be associated with a non-default stream. The stream must be specified during construction.
     *
     * @note Moving a Handle object to another invalidates the former
     */
    class Handle {
    public:
        Handle() = default;
        Handle(const Handle&) = default;
        Handle(Handle&&) = default;

        /** creates a cuBLAS handle and associates it with the stream specified
         *
         * Exception Guarantee: Basic
         */
        Handle(Stream strm) : handle(std::make_shared<UniqueHandle>(std::move(strm))) { }

        Handle& operator=(const Handle&) = default;
        Handle& operator=(Handle&&) = default;

        /** returns true if the handle is valid */
        explicit operator bool() const noexcept { return static_cast<bool>(handle); }

        /** returns the raw cuBLAS handle */
        cublasHandle_t get() const noexcept {
            CV_Assert(handle);
            return handle->get();
        }

    private:
        std::shared_ptr<UniqueHandle> handle;
    };

    /** @brief GEMM for colummn-major matrices
     *
     * \f$ C = \alpha AB + \beta C \f$
     *
     * @tparam          T           matrix element type (must be `half` or `float`)
     *
     * @param           handle      valid cuBLAS Handle
     * @param           transa      use transposed matrix of A for computation
     * @param           transb      use transposed matrix of B for computation
     * @param           rows_c      number of rows in C
     * @param           cols_c      number of columns in C
     * @param           common_dim  common dimension of A (or trans A) and B (or trans B)
     * @param           alpha       scale factor for AB
     * @param[in]       A           pointer to column-major matrix A in device memory
     * @param           lda         leading dimension of matrix A
     * @param[in]       B           pointer to column-major matrix B in device memory
     * @param           ldb         leading dimension of matrix B
     * @param           beta        scale factor for C
     * @param[in,out]   C           pointer to column-major matrix C in device memory
     * @param           ldc         leading dimension of matrix C
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void gemm(const Handle& handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        T alpha, const DevicePtr<const T> A, std::size_t lda,
        const DevicePtr<const T> B, std::size_t ldb,
        T beta, const DevicePtr<T> C, std::size_t ldc);

    template <> inline
    void gemm<half>(const Handle& handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        half alpha, const DevicePtr<const half> A, std::size_t lda,
        const DevicePtr<const half> B, std::size_t ldb,
        half beta, const DevicePtr<half> C, std::size_t ldc)
    {
        CV_Assert(handle);

        auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
            opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        int irows_c = static_cast<int>(rows_c),
            icols_c = static_cast<int>(cols_c),
            icommon_dim = static_cast<int>(common_dim),
            ilda = static_cast<int>(lda),
            ildb = static_cast<int>(ldb),
            ildc = static_cast<int>(ldc);

        CUDA4DNN_CHECK_CUBLAS(
            cublasHgemm(
                handle.get(),
                opa, opb,
                irows_c, icols_c, icommon_dim,
                &alpha, A.get(), ilda,
                B.get(), ildb,
                &beta, C.get(), ildc
            )
        );
    }

    template <> inline
    void gemm<float>(const Handle& handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        float alpha, const DevicePtr<const float> A, std::size_t lda,
        const DevicePtr<const float> B, std::size_t ldb,
        float beta, const DevicePtr<float> C, std::size_t ldc)
    {
        CV_Assert(handle);

        auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
            opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        int irows_c = static_cast<int>(rows_c),
            icols_c = static_cast<int>(cols_c),
            icommon_dim = static_cast<int>(common_dim),
            ilda = static_cast<int>(lda),
            ildb = static_cast<int>(ldb),
            ildc = static_cast<int>(ldc);

        CUDA4DNN_CHECK_CUBLAS(
            cublasSgemm(
                handle.get(),
                opa, opb,
                irows_c, icols_c, icommon_dim,
                &alpha, A.get(), ilda,
                B.get(), ildb,
                &beta, C.get(), ildc
            )
        );
    }

    /** @brief Strided batched GEMM for colummn-major matrices
     *
     * \f$ C_i = \alpha A_i B_i + \beta C_i \f$ for a stack of matrices A, B and C indexed by i
     *
     * @tparam          T           matrix element type (must be `half` or `float`)
     *
     * @param           handle      valid cuBLAS Handle
     * @param           transa      use transposed matrix of A_i for computation
     * @param           transb      use transposed matrix of B_i for computation
     * @param           rows_c      number of rows in C_i
     * @param           cols_c      number of columns in C_i
     * @param           common_dim  common dimension of A_i (or trans A_i) and B_i (or trans B_i)
     * @param           alpha       scale factor for A_i B_i
     * @param[in]       A           pointer to stack of column-major matrices A in device memory
     * @param           lda         leading dimension of matrix A_i
     * @param           strideA     stride between matrices in A
     * @param[in]       B           pointer to stack of column-major matrices B in device memory
     * @param           ldb         leading dimension of matrix B_i
     * @param           strideB     stride between matrices in B
     * @param           beta        scale factor for C_i
     * @param[in,out]   C           pointer to stack of column-major matrices C in device memory
     * @param           ldc         leading dimension of matrix C_i
     * @param           strideC     stride between matrices in C
     * @param           batchCount  number of matrices in the batch
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void gemmStridedBatched(const Handle& handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        T alpha, const DevicePtr<const T> A, std::size_t lda, std::size_t strideA,
        const DevicePtr<const T> B, std::size_t ldb, std::size_t strideB,
        T beta, const DevicePtr<T> C, std::size_t ldc, std::size_t strideC,
        std::size_t batchCount);

    template <> inline
    void gemmStridedBatched<half>(const Handle& handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        half alpha, const DevicePtr<const half> A, std::size_t lda, std::size_t strideA,
        const DevicePtr<const half> B, std::size_t ldb, std::size_t strideB,
        half beta, const DevicePtr<half> C, std::size_t ldc, std::size_t strideC,
        std::size_t batchCount)
    {
        CV_Assert(handle);

        const auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                   opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto irows_c = static_cast<int>(rows_c),
                   icols_c = static_cast<int>(cols_c),
                   icommon_dim = static_cast<int>(common_dim),
                   ilda = static_cast<int>(lda),
                   ildb = static_cast<int>(ldb),
                   ildc = static_cast<int>(ldc);

        const auto batch_count = static_cast<int>(batchCount);
        const auto stride_a = static_cast<long long int>(strideA),
                   stride_b = static_cast<long long int>(strideB),
                   stride_c = static_cast<long long int>(strideC);

        CV_Assert(stride_c >= irows_c * icols_c); // output matrices must not overlap

        CUDA4DNN_CHECK_CUBLAS(
            cublasHgemmStridedBatched(
                handle.get(),
                opa, opb,
                irows_c, icols_c, icommon_dim,
                &alpha, A.get(), ilda, stride_a,
                B.get(), ildb, stride_b,
                &beta, C.get(), ildc, stride_c,
                batch_count
            )
        );
    }

    template <> inline
    void gemmStridedBatched<float>(const Handle& handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        float alpha, const DevicePtr<const float> A, std::size_t lda, std::size_t strideA,
        const DevicePtr<const float> B, std::size_t ldb, std::size_t strideB,
        float beta, const DevicePtr<float> C, std::size_t ldc, std::size_t strideC,
        std::size_t batchCount)
    {
        CV_Assert(handle);

        const auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                   opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto irows_c = static_cast<int>(rows_c),
                   icols_c = static_cast<int>(cols_c),
                   icommon_dim = static_cast<int>(common_dim),
                   ilda = static_cast<int>(lda),
                   ildb = static_cast<int>(ldb),
                   ildc = static_cast<int>(ldc);

        const auto batch_count = static_cast<int>(batchCount);
        const auto stride_a = static_cast<long long int>(strideA),
                   stride_b = static_cast<long long int>(strideB),
                   stride_c = static_cast<long long int>(strideC);

        CV_Assert(stride_c >= irows_c * icols_c); // output matrices must not overlap

        CUDA4DNN_CHECK_CUBLAS(
            cublasSgemmStridedBatched(
                handle.get(),
                opa, opb,
                irows_c, icols_c, icommon_dim,
                &alpha, A.get(), ilda, stride_a,
                B.get(), ildb, stride_b,
                &beta, C.get(), ildc, stride_c,
                batch_count
            )
        );
    }

    /** @brief Strided batched GEMM for colummn-major matrices
     *
     * \f$ C_i = \alpha A_i B_i + \beta C_i \f$ for a stack of matrices A, B and C indexed by i
     *
     * @tparam          T           matrix element type (must be `half` or `float`)
     *
     * @param           handle      valid cuBLAS Handle
     * @param           trans_a     use transposed matrix of A_i for computation
     * @param           trans_b     use transposed matrix of B_i for computation
     * @param           M           number of rows in C
     * @param           N           number of columns in C
     * @param           K           common dimension of A (or trans A) and B (or trans B)
     * @param           alpha       scale factor for A B
     * @param[in]       A           pointer to stack of column-major matrices A in device memory
     * @param           lda         leading dimension of matrix A
     * @param           A_offsets   offsets to get A slices
     * @param[in]       B           pointer to stack of column-major matrices B in device memory
     * @param           ldb         leading dimension of matrix B
     * @param           B_offsets   offsets to get B slices
     * @param           beta        scale factor for C
     * @param[in,out]   C           pointer to stack of column-major matrices C in device memory
     * @param           ldc         leading dimension of matrix C
     * @param           C_offsets   offsets to get C slices
     * @param           batchCount  number of matrices in the batch
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void gemmBatched(const Handle &handle,
                     bool trans_a, bool trans_b,
                     std::size_t M, std::size_t N, std::size_t K,
                     T alpha,
                     const DevicePtr<const T> A, std::size_t lda, std::vector<std::size_t> A_offsets,
                     const DevicePtr<const T> B, std::size_t ldb, std::vector<std::size_t> B_offsets,
                     T beta,
                     const DevicePtr<T> C, std::size_t ldc, std::vector<std::size_t> C_offsets,
                     std::size_t batchCount);

    template <> inline
    void gemmBatched<half>(const Handle &handle,
                           bool trans_a, bool trans_b,
                           std::size_t M, std::size_t N, std::size_t K,
                           half alpha,
                           const DevicePtr<const half> A, std::size_t lda, std::vector<std::size_t> A_offsets,
                           const DevicePtr<const half> B, std::size_t ldb, std::vector<std::size_t> B_offsets,
                           half beta,
                           const DevicePtr<half> C, std::size_t ldc, std::vector<std::size_t> C_offsets,
                           std::size_t batchCount) {
        CV_Assert(handle);

        const auto opa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                   opb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto iM = static_cast<int>(M),
                   iN = static_cast<int>(N),
                   iK = static_cast<int>(K),
                   ilda = static_cast<int>(lda),
                   ildb = static_cast<int>(ldb),
                   ildc = static_cast<int>(ldc);

        const auto batch_count = static_cast<int>(batchCount);

        AutoBuffer<half*> buffer(3 * batch_count);
        auto A_slices = buffer.data();
        auto B_slices = A_slices + batch_count;
        auto C_slices = B_slices + batch_count;
        // collect A, B and C slices
        for (int i = 0; i < batch_count; i++) {
            A_slices[i] = (half*)(A.get()) + A_offsets[i];
            B_slices[i] = (half*)(B.get()) + B_offsets[i];
            C_slices[i] = (half*)(C.get()) + C_offsets[i];
        }

        const half **dev_A_slices = 0, **dev_B_slices = 0;
        half **dev_C_slices = 0;
        CUDA4DNN_CHECK_CUDA(cudaMalloc((void**)&dev_A_slices, batch_count * sizeof(half*)));
        CUDA4DNN_CHECK_CUDA(cudaMalloc((void**)&dev_B_slices, batch_count * sizeof(half*)));
        CUDA4DNN_CHECK_CUDA(cudaMalloc((void**)&dev_C_slices, batch_count * sizeof(half*)));
        CUDA4DNN_CHECK_CUDA(cudaMemcpy(dev_A_slices, A_slices, batch_count * sizeof(half*), cudaMemcpyHostToDevice));
        CUDA4DNN_CHECK_CUDA(cudaMemcpy(dev_B_slices, B_slices, batch_count * sizeof(half*), cudaMemcpyHostToDevice));
        CUDA4DNN_CHECK_CUDA(cudaMemcpy(dev_C_slices, C_slices, batch_count * sizeof(half*), cudaMemcpyHostToDevice));

        CUDA4DNN_CHECK_CUBLAS(cublasHgemmBatched(handle.get(), opa, opb, iM, iN, iK, &alpha, dev_A_slices, ilda, dev_B_slices, ildb, &beta, dev_C_slices, ildc, batch_count));

        CUDA4DNN_CHECK_CUDA(cudaFree(dev_A_slices));
        CUDA4DNN_CHECK_CUDA(cudaFree(dev_B_slices));
        CUDA4DNN_CHECK_CUDA(cudaFree(dev_C_slices));
    }

    template <> inline
    void gemmBatched<float>(const Handle &handle,
                           bool trans_a, bool trans_b,
                           std::size_t M, std::size_t N, std::size_t K,
                           float alpha,
                           const DevicePtr<const float> A, std::size_t lda, std::vector<std::size_t> A_offsets,
                           const DevicePtr<const float> B, std::size_t ldb, std::vector<std::size_t> B_offsets,
                           float beta,
                           const DevicePtr<float> C, std::size_t ldc, std::vector<std::size_t> C_offsets,
                           std::size_t batchCount) {
        CV_Assert(handle);

        const auto opa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                   opb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto iM = static_cast<int>(M),
                   iN = static_cast<int>(N),
                   iK = static_cast<int>(K),
                   ilda = static_cast<int>(lda),
                   ildb = static_cast<int>(ldb),
                   ildc = static_cast<int>(ldc);

        const auto batch_count = static_cast<int>(batchCount);

        AutoBuffer<float*> buffer(3 * batch_count);
        auto A_slices = buffer.data();
        auto B_slices = A_slices + batch_count;
        auto C_slices = B_slices + batch_count;
        // collect A, B and C slices
        for (int i = 0; i < batch_count; i++) {
            A_slices[i] = (float*)(A.get()) + A_offsets[i];
            B_slices[i] = (float*)(B.get()) + B_offsets[i];
            C_slices[i] = (float*)(C.get()) + C_offsets[i];
        }

        const float **dev_A_slices = 0, **dev_B_slices = 0;
        float **dev_C_slices = 0;
        CUDA4DNN_CHECK_CUDA(cudaMalloc((void**)&dev_A_slices, batch_count * sizeof(float*)));
        CUDA4DNN_CHECK_CUDA(cudaMalloc((void**)&dev_B_slices, batch_count * sizeof(float*)));
        CUDA4DNN_CHECK_CUDA(cudaMalloc((void**)&dev_C_slices, batch_count * sizeof(float*)));
        CUDA4DNN_CHECK_CUDA(cudaMemcpy(dev_A_slices, A_slices, batch_count * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA4DNN_CHECK_CUDA(cudaMemcpy(dev_B_slices, B_slices, batch_count * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA4DNN_CHECK_CUDA(cudaMemcpy(dev_C_slices, C_slices, batch_count * sizeof(float*), cudaMemcpyHostToDevice));

        // cuBLAS is column-major
        CUDA4DNN_CHECK_CUBLAS(cublasSgemmBatched(handle.get(), opa, opb, iM, iN, iK, &alpha, dev_A_slices, ilda, dev_B_slices, ildb, &beta, dev_C_slices, ildc, batch_count));

        CUDA4DNN_CHECK_CUDA(cudaFree(dev_A_slices));
        CUDA4DNN_CHECK_CUDA(cudaFree(dev_B_slices));
        CUDA4DNN_CHECK_CUDA(cudaFree(dev_C_slices));
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cublas */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_CUBLAS_HPP */
