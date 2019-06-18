// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "cublas.hpp"
#include "stream.hpp"

#include <opencv2/core.hpp>

#include <cublas_v2.h>

#include <cstddef>
#include <memory>
#include <utility>

#define CUDA4DNN_CHECK_CUBLAS(call) \
    ::cv::dnn::cuda4dnn::csl::cublas::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cublas {

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

    /** noncopyable cuBLAS smart handle
     *
     * UniqueHandle is a smart non-sharable wrapper for cuBLAS handle which ensures that the handle
     * is destroyed after use. The handle can be associated with a CUDA stream by specifying the
     * stream during construction. By default, the handle is associated with the default stream.
     */
    class Handle::UniqueHandle {
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
                CUDA4DNN_CHECK_CUBLAS(cublasSetStream(handle, StreamAccessor::get(stream)));
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

        //!< returns the raw cuBLAS handle
        cublasHandle_t get() const noexcept { return handle; }

    private:
        Stream stream;
        cublasHandle_t handle;
    };

    /* used to access the raw cuBLAS handle held by Handle */
    class HandleAccessor {
    public:
        static cublasHandle_t get(Handle& handle) {
            CV_Assert(handle);
            return handle.handle->get();
        }
    };

    Handle::Handle() : handle(std::make_shared<Handle::UniqueHandle>()) { }
    Handle::Handle(const Handle&) noexcept = default;
    Handle::Handle(Handle&&) noexcept = default;
    Handle::Handle(Stream strm) : handle(std::make_shared<Handle::UniqueHandle>(std::move(strm))) { }

    Handle& Handle::operator=(const Handle&) noexcept = default;
    Handle& Handle::operator=(Handle&&) noexcept = default;

    Handle::operator bool() const noexcept { return static_cast<bool>(handle); }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cublas */
