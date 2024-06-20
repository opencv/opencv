// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

// You can find here the fast convolution routines for symmetrical kernels
// (gaussianblur)

#include "precomp.hpp" // defined here.

// If we have AVX intrinsics, and we can use OPENMP, then build vk_gaussian.hpp
//#if defined HAVE_OPENMP && defined CV_AVX
#define HAVE_VK_SMOOTH
//#endif


#include <immintrin.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h> /* for uint64 definition */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace cv {
    
/**
 * DEPENDENT ROUTINES
 * for sub-functions called by the optimised routines.
 */

__m256i division(const unsigned int division_case, __m256i m2, const __m256i f);
unsigned int prepare_for_division(const unsigned short int divisor);
__m256i fill_zeros(__m256i r0, const __m256i mask_prelude); // make one routine
__m256i fill_3zeros(__m256i r0, const __m256i mask_prelude);
__m256i fill_2zeros(__m256i r0, const __m256i mask_prelude);
__m256i fill_1zeros(__m256i r0, const __m256i mask_prelude);

__m256i insert_one_zeros_front(__m256i r0,__m256i mask_prelude);
__m256i insert_two_zeros_front(__m256i r0,__m256i mask_prelude);

void prelude_9x9_16_Ymask_3(const int row, const unsigned int M,
                            unsigned char **frame1, unsigned char *temp,
                            const signed char mask_vector_y[][32],
                            const unsigned int division_case, const __m256i f,
                            const __m256i mask_prelude);
void prelude_9x9_16_Ymask_2(const int row, const unsigned int M,
                            unsigned char **frame1, unsigned char *temp,
                            const signed char mask_vector_y[][32],
                            const unsigned int division_case, const __m256i f,
                            const __m256i mask_prelude);
void prelude_9x9_16_Ymask_1(const int row, const unsigned int M,
                            unsigned char **frame1, unsigned char *temp,
                            const signed char mask_vector_y[][32],
                            const unsigned int division_case, const __m256i f,
                            const __m256i mask_prelude);
void prelude_9x9_16_Ymask_0(const int row, const unsigned int M,
                            unsigned char **frame1, unsigned char *temp,
                            const signed char mask_vector_y[][32],
                            const unsigned int division_case, const __m256i f,
                            const __m256i mask_prelude);
void prelude_9x9_16_Xmask(unsigned char **frame1, unsigned char **filt,
                          unsigned char *temp, const unsigned int N,
                          const unsigned int M, const int row,
                          const signed char mask_vector_x[][32],
                          const unsigned int division_case, const __m256i f,
                          const unsigned int REMINDER_ITERATIONS_X,
                          const unsigned int REMINDER_ITERATIONS_Y,
                          const signed char mask_vector_y[][32],
                          const unsigned short int divisor_xy,
                          signed char *kernel_x);
void loop_reminder_9x9_16_blur_Y(unsigned char **frame1, unsigned char **filt,
                                 unsigned char *temp, const unsigned int N,
                                 const unsigned int M, const unsigned int row,
                                 const unsigned int REMINDER_ITERATIONS_Y,
                                 const unsigned int division_case,
                                 const signed char mask_vector_y[][32],
                                 const __m256i f,
                                 const unsigned short int divisor_xy);
void loop_reminder_9x9_16_blur_X(
    unsigned char **frame1, unsigned char **filt, unsigned char *temp,
    const unsigned int N, const unsigned int M, const unsigned int row,
    const unsigned int col, const unsigned int REMINDER_ITERATIONS_X,
    const unsigned int division_case, const signed char mask_vector_x[][32],
    const __m256i f, const unsigned short int divisor_xy,
    signed char *kernel_x);

void prelude_7x7_16_Ymask_2_new(const int row, const unsigned int M,
                                unsigned char **frame1, unsigned char *temp,
                                const signed char mask_vector_y[][32],
                                const unsigned int division_case,
                                const __m256i f, const __m256i mask_prelude);
void prelude_7x7_16_Ymask_1_new(const int row, const unsigned int M,
                                unsigned char **frame1, unsigned char *temp,
                                const signed char mask_vector_y[][32],
                                const unsigned int division_case,
                                const __m256i f, const __m256i mask_prelude);
void prelude_7x7_16_Ymask_0_new(const int row, const unsigned int M,
                                unsigned char **frame1, unsigned char *temp,
                                const signed char mask_vector_y[][32],
                                const unsigned int division_case,
                                const __m256i f, const __m256i mask_prelude);
void prelude_7x7_16_Xmask_new(unsigned char **frame1, unsigned char **filt,
                              unsigned char *temp, const unsigned int N,
                              const unsigned int M, const int row,
                              const signed char mask_vector_x[][32],
                              const unsigned int division_case, const __m256i f,
                              const unsigned int REMINDER_ITERATIONS_XY,
                              const signed char mask_vector_y[][32],
                              const unsigned short int divisor_xy,
                              signed char *kernel_x);
void loop_reminder_7x7_16_blur_Y(unsigned char **frame1, unsigned char **filt,
                                 unsigned char *temp, const unsigned int N,
                                 const unsigned int M, const unsigned int row,
                                 const unsigned int REMINDER_ITERATIONS_Y,
                                 const unsigned int division_case,
                                 const signed char mask_vector_y[][32],
                                 const __m256i f,
                                 const unsigned short int divisor_xy);
int loop_reminder_7x7_16_blur_X(
    unsigned char **frame1, unsigned char **filt, unsigned char *temp,
    const unsigned int N, const unsigned int M, const unsigned int row,
    const unsigned int col, const unsigned int REMINDER_ITERATIONS_X,
    const unsigned int division_case, const signed char mask_vector_x[][32],
    const __m256i f, const unsigned short int divisor_xy,
    signed char *kernel_x);


int loop_reminder_high_reminder_values_less_div(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int row, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const __m256i c0, const __m256i c1, const __m256i c2, const __m256i c0_sh1,
    const __m256i c1_sh1, const __m256i c2_sh1, const __m256i c0_sh2,
    const __m256i c1_sh2, const __m256i c2_sh2, const __m256i f,
    const unsigned short int divisor, signed char **filter5x5);
int loop_reminder_low_reminder_values_less_div(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int row, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const __m256i c0, const __m256i c1, const __m256i c2, const __m256i c0_sh1,
    const __m256i c1_sh1, const __m256i c2_sh1, const __m256i c0_sh2,
    const __m256i c1_sh2, const __m256i c2_sh2, const __m256i c0_sh3,
    const __m256i c1_sh3, const __m256i c2_sh3, const __m256i c0_sh4,
    const __m256i c1_sh4, const __m256i c2_sh4, const __m256i c0_sh5,
    const __m256i c1_sh5, const __m256i c2_sh5, const __m256i f);
int loop_reminder_first_less_div(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const __m256i c0, const __m256i c1, const __m256i c2, const __m256i c0_sh1,
    const __m256i c1_sh1, const __m256i c2_sh1, const __m256i c0_sh2,
    const __m256i c1_sh2, const __m256i c2_sh2, const __m256i f,
    const unsigned short int divisor, signed char **filter5x5);
int loop_reminder_last_less_div(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const __m256i c0, const __m256i c1, const __m256i c2, const __m256i c0_sh1,
    const __m256i c1_sh1, const __m256i c2_sh1, const __m256i c0_sh2,
    const __m256i c1_sh2, const __m256i c2_sh2, const __m256i f,
    const unsigned short int divisor, signed char **filter5x5);
int loop_reminder_last_less_div_special_case(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const __m256i c0, const __m256i c1, const __m256i c2, const __m256i c0_sh1,
    const __m256i c1_sh1, const __m256i c2_sh1, const __m256i c0_sh2,
    const __m256i c1_sh2, const __m256i c2_sh2, const __m256i f,
    const unsigned short int divisor, signed char **filter5x5);
    
void prelude_5x5_16_Ymask_1_new(const int row, const unsigned int M,
                                unsigned char **frame1, unsigned char *temp,
                                const signed char mask_vector_y[][32],
                                const unsigned int division_case,
                                const __m256i f);
void prelude_5x5_16_Ymask_0_new(const int row, const unsigned int M,
                                unsigned char **frame1, unsigned char *temp,
                                const signed char mask_vector_y[][32],
                                const unsigned int division_case,
                                const __m256i f);
void prelude_5x5_16_Xmask_new(unsigned char **frame1, unsigned char **filt,
                              unsigned char *temp, const unsigned int N,
                              const unsigned int M, const int row,
                              const signed char mask_vector_x[][32],
                              const unsigned int division_case, const __m256i f,
                              const unsigned int REMINDER_ITERATIONS_XY,
                              const signed char mask_vector_y[][32],
                              const unsigned short int divisor_xy,
                              signed char *kernel_x);
void loop_reminder_5x5_16_blur_Y(unsigned char **frame1, unsigned char **filt,
                                 unsigned char *temp, const unsigned int N,
                                 const unsigned int M, const unsigned int row,
                                 const unsigned int REMINDER_ITERATIONS_Y,
                                 const unsigned int division_case,
                                 const signed char mask_vector_y[][32],
                                 const __m256i f,
                                 const unsigned short int divisor_xy);
void loop_reminder_5x5_16_blur_X(
    unsigned char **frame1, unsigned char **filt, unsigned char *temp,
    const unsigned int N, const unsigned int M, const unsigned int row,
    const unsigned int col, const unsigned int REMINDER_ITERATIONS_X,
    const unsigned int division_case, const signed char mask_vector_x[][32],
    const __m256i f, const unsigned short int divisor_xy,
    signed char *kernel_x);
    
int loop_reminder_3x3(unsigned char **frame1, unsigned char **filt,
                      const unsigned int M, const unsigned int N,
                      const unsigned int row, const unsigned int col,
                      const unsigned int REMINDER_ITERATIONS,
                      const unsigned int division_case,
                      const unsigned short int divisor, signed char **filter,
                      const __m256i c0, const __m256i c1, const __m256i c0_sh1,
                      const __m256i c1_sh1, const __m256i c0_sh2,
                      const __m256i c1_sh2, const __m256i c0_sh3,
                      const __m256i c1_sh3, const __m256i f);
int loop_reminder_3x3_first_values(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const unsigned short int divisor, signed char **filter, const __m256i c0,
    const __m256i c1, const __m256i c0_sh1, const __m256i c1_sh1,
    const __m256i c0_sh2, const __m256i c1_sh2, const __m256i c0_sh3,
    const __m256i c1_sh3, const __m256i f);
int loop_reminder_3x3_last_values(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const unsigned short int divisor, signed char **filter, const __m256i c0,
    const __m256i c1, const __m256i c0_sh1, const __m256i c1_sh1,
    const __m256i c0_sh2, const __m256i c1_sh2, const __m256i c0_sh3,
    const __m256i c1_sh3, const __m256i f);
int loop_reminder_3x3_last_row_only(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const unsigned short int divisor, signed char **filter, const __m256i c0,
    const __m256i c1, const __m256i c0_sh1, const __m256i c1_sh1,
    const __m256i c0_sh2, const __m256i c1_sh2, const __m256i c0_sh3,
    const __m256i c1_sh3, const __m256i f);
    

int loop_reminder_3x3_new(unsigned char **frame1, unsigned char **filt,
                          const unsigned int M, const unsigned int N,
                          const unsigned int row, const unsigned int col,
                          const unsigned int REMINDER_ITERATIONS,
                          const unsigned int division_case,
                          const unsigned short int divisor,
                          signed char **filter, const __m256i c0,
                          const __m256i c1, const __m256i f);
int loop_reminder_3x3_new_first_last_rows(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned int row, const unsigned int col,
    const unsigned int REMINDER_ITERATIONS, const unsigned int division_case,
    const unsigned short int divisor, signed char **filter, const __m256i c0,
    const __m256i c1, const __m256i f);

    
/**
 * OPTIMISED ROUTINES
 * the optimised routines themselves.
 */

void Gaussian_Blur_3x3_16_more_load(unsigned char **frame1,
                                    unsigned char **filt, const unsigned int M,
                                    const unsigned int N,
                                    const unsigned short int divisor,
                                    signed char **filter);


void Gaussian_Blur_optimized_3x3_16_reg_blocking(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned short int divisor,
    signed char **filter);

void Gaussian_Blur_optimized_5x5_16_reg_blocking(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, const unsigned short int divisor,
    signed char **filter);

void Gaussian_Blur_optimized_5x5_16_seperable(
    unsigned char **frame1, unsigned char **filt, const unsigned int M,
    const unsigned int N, signed char *kernel_y, signed char *kernel_x,
    const unsigned short int divisor_xy);

void Gaussian_Blur_7x7_16_separable(unsigned char **frame1,
                                    unsigned char **filt, const unsigned int M,
                                    const unsigned int N, signed char *kernel_y,
                                    signed char *kernel_x,
                                    const unsigned short int divisor_xy);

void Gaussian_Blur_9x9_16_separable(unsigned char **frame1,
                                    unsigned char **filt, const unsigned int M,
                                    const unsigned int N, signed char *kernel_y,
                                    signed char *kernel_x,
                                    const unsigned short int divisor_xy);

}

