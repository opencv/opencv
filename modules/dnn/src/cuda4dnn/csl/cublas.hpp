// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_CUBLAS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_CUBLAS_HPP

#include "error.hpp"
#include "stream.hpp"
#include "pointer.hpp"
#include "fp16.hpp"

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

    /** noncopyable cuBLAS smart handle
     *
     * UniqueHandle is a smart non-sharable wrapper for cuBLAS handle which ensures that the handle
     * is destroyed after use. The handle can be associated with a CUDA stream by specifying the
     * stream during construction. By default, the handle is associated with the default stream.
     */
    class UniqueHandle {
    public:
        UniqueHandle() { CUDA4DNN_CHECK_CUBLAS(cublasCreate(&handle)); }
        UniqueHandle(UniqueHandle&) = delete;
        UniqueHandle(UniqueHandle&& other) noexcept
            : stream(std::move(other.stream)), handle{ other.handle } {
            other.handle = nullptr;
        }

        UniqueHandle(Stream strm) : stream(std::move(strm)) {
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
            if (handle != nullptr) {
                /* cublasDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUBLAS(cublasDestroy(handle));
            }
        }

        UniqueHandle& operator=(const UniqueHandle&) = delete;
        UniqueHandle& operator=(UniqueHandle&& other) noexcept {
            stream = std::move(other.stream);
            handle = other.handle;
            other.handle = nullptr;
            return *this;
        }

        /** @brief returns the raw cuBLAS handle */
        cublasHandle_t get() const noexcept { return handle; }

    private:
        Stream stream;
        cublasHandle_t handle;
    };

    /** @brief sharable cuBLAS smart handle
     *
     * Handle is a smart sharable wrapper for cuBLAS handle which ensures that the handle
     * is destroyed after all references to the handle are destroyed. The handle can be
     * associated with a CUDA stream by specifying the stream during construction. By default,
     * the handle is associated with the default stream.
     *
     * @note Moving a Handle object to another invalidates the former
     */
    class Handle {
    public:
        Handle() : handle(std::make_shared<UniqueHandle>()) { }
        Handle(const Handle&) = default;
        Handle(Handle&&) = default;
        Handle(Stream strm) : handle(std::make_shared<UniqueHandle>(std::move(strm))) { }

        Handle& operator=(const Handle&) = default;
        Handle& operator=(Handle&&) = default;

        /** returns true if the handle is valid */
        explicit operator bool() const noexcept { return static_cast<bool>(handle); }

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

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cublas */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_CUBLAS_HPP */
