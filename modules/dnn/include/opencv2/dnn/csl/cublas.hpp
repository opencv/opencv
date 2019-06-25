// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CSL_CUBLAS_HPP
#define OPENCV_DNN_CSL_CUBLAS_HPP

#include "error.hpp"
#include "stream.hpp"

#include <memory>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cublas {

    /** @brief exception class for errors thrown by the cuBLAS API */
    class cuBLASException : public CUDAException {
    public:
        using CUDAException::CUDAException;
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
        Handle();
        Handle(const Handle&) noexcept;
        Handle(Handle&&) noexcept;
        Handle(Stream strm);

        Handle& operator=(const Handle&) noexcept;
        Handle& operator=(Handle&&) noexcept;

        /** returns true if the handle is valid */
        explicit operator bool() const noexcept;

    private:
        /*! \cond PRIVATE */
        friend class HandleAccessor;
        /*! \endcond */

        class UniqueHandle;
        std::shared_ptr<UniqueHandle> handle;
    };

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cublas */

#endif /* OPENCV_DNN_CSL_CUBLAS_HPP */
