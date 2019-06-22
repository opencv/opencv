// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUBLAS_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUBLAS_HPP

#include "pointer.hpp"

#include <opencv2/dnn/csl/cublas.hpp>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cublas {

    /** @brief GEMM for colummn-major matrices
     *
     * \f$ C = \alpha AB + \beta C \f$
     *
     * @tparam          T           matrix element type (must be `float` or `double`)
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
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type gemm(const Handle& handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        T alpha, const DevicePtr<const T> A, std::size_t lda,
        const DevicePtr<const T> B, std::size_t ldb,
        T beta, const DevicePtr<T> C, std::size_t ldc);

}}}}} /* cv::dnn::cuda4dnn::csl::cublas */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUBLAS_HPP */
