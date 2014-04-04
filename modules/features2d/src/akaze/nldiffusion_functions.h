#ifndef _NLDIFFUSION_FUNCTIONS_H_
#define _NLDIFFUSION_FUNCTIONS_H_

//******************************************************************************
//******************************************************************************

// Includes
#include "precomp.hpp"

// OpenMP Includes
#ifdef _OPENMP
# include <omp.h>
#endif

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst, const size_t& ksize_x,
                             const size_t& ksize_y, const float& sigma);
void image_derivatives_scharr(const cv::Mat& src, cv::Mat& dst,
                              const size_t& xorder, const size_t& yorder);
void pm_g1(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void pm_g2(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void weickert_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
void charbonnier_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
float compute_k_percentile(const cv::Mat& img, const float& perc, const float& gscale,
                           const size_t& nbins, const size_t& ksize_x, const size_t& ksize_y);
void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, const size_t& xorder,
                                const size_t& yorder, const size_t& scale);
void nld_step_scalar(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, const float& stepsize);
void downsample_image(const cv::Mat& src, cv::Mat& dst);
void halfsample_image(const cv::Mat& src, cv::Mat& dst);
void compute_derivative_kernels(cv::OutputArray kx_, cv::OutputArray ky_,
                                const size_t& dx, const size_t& dy, const size_t& scale);

//*************************************************************************************
//*************************************************************************************


#endif
