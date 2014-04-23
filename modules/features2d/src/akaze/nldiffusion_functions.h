/**
 * @file nldiffusion_functions.h
 * @brief Functions for nonlinear diffusion filtering applications
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
// Includes
#include "precomp.hpp"

/* ************************************************************************* */
// Declaration of functions
void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst, const size_t& ksize_x,
                             const size_t& ksize_y, const float& sigma);
void image_derivatives_scharr(const cv::Mat& src, cv::Mat& dst,
                              const size_t& xorder, const size_t& yorder);
void pm_g1(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void pm_g2(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void weickert_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void charbonnier_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
float compute_k_percentile(const cv::Mat& img, float perc, float gscale,
                           size_t nbins, size_t ksize_x, size_t ksize_y);
void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, const size_t& xorder,
                                const size_t& yorder, const size_t& scale);
void nld_step_scalar(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, const float& stepsize);
void downsample_image(const cv::Mat& src, cv::Mat& dst);
void halfsample_image(const cv::Mat& src, cv::Mat& dst);
void compute_derivative_kernels(cv::OutputArray kx_, cv::OutputArray ky_,
                                const size_t& dx, const size_t& dy, const size_t& scale);
bool check_maximum_neighbourhood(const cv::Mat& img, int dsize, float value,
                                 int row, int col, bool same_img);
