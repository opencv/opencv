/*
 * greentea_math_functions.hpp
 *
 *  Created on: Apr 6, 2015
 *      Author: fabian
 */
#ifndef _OPENCV_GREENTEA_MATH_FUNCTIONS_HPP_
#define _OPENCV_GREENTEA_MATH_FUNCTIONS_HPP_
#include "../../precomp.hpp"
#include "common.hpp"


namespace greentea {

#ifdef HAVE_OPENCL
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

template<typename Dtype>
void greentea_gpu_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC , const bool is_image_a = false,
                       const bool is_image_b = false);

void greentea_gpu_gemm_copy_buffer_to_image(int_tp ctx_id,
                                            cl_mem *image, cl_mem buffer, int offset,
                                            bool is_matrix_a, bool transpose,
                                            bool padding, int padded_height,
                                            int padded_width, int height,
                                            int width, int wait_list_size,
                                            cl_event *wait_list, cl_event *event);

template<typename Dtype>
void greentea_gpu_gemv(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int_tp M, const int_tp N, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem x,
                       const int_tp offx, const Dtype beta, cl_mem y,
                       const int_tp offy);

template<typename Dtype>
void greentea_gpu_axpy(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       const cl_mem x, const int_tp offx, cl_mem y,
                       const int_tp offy);

template<typename Dtype>
void greentea_gpu_set(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                      cl_mem Y, const int_tp offY);
#endif  // HAVE_OPENCL

}  // namespace greentea

#endif  /* _OPENCV_GREENTEA_MATH_FUNCTIONS_HPP_ */
