/**
 * @file nldiffusion_functions.h
 * @brief Functions for non-linear diffusion applications:
 * 2D Gaussian Derivatives
 * Perona and Malik conductivity equations
 * Perona and Malik evolution
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#ifndef __OPENCV_FEATURES_2D_NLDIFFUSION_FUNCTIONS_H__
#define __OPENCV_FEATURES_2D_NLDIFFUSION_FUNCTIONS_H__

/* ************************************************************************* */
// Declaration of functions

namespace cv
{

// Gaussian 2D convolution
void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst, int ksize_x, int ksize_y, float sigma);

// Diffusivity functions
void pm_g1(InputArray Lx, InputArray Ly, OutputArray dst, float k);
void pm_g2(InputArray Lx, InputArray Ly, OutputArray dst, float k);
void weickert_diffusivity(InputArray Lx, InputArray Ly, OutputArray dst, float k);
void charbonnier_diffusivity(InputArray Lx, InputArray Ly, OutputArray dst, float k);

float compute_k_percentile(const cv::Mat& img, float perc, float gscale, int nbins, int ksize_x, int ksize_y);

// Image derivatives
void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, int xorder, int yorder, int scale);
void compute_derivative_kernels(cv::OutputArray _kx, cv::OutputArray _ky, int dx, int dy, int scale);
void image_derivatives_scharr(const cv::Mat& src, cv::Mat& dst, int xorder, int yorder);

// Nonlinear diffusion filtering scalar step
void nld_step_scalar(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, float stepsize);

// For non-maxima suppresion
bool check_maximum_neighbourhood(const cv::Mat& img, int dsize, float value, int row, int col, bool same_img);

// Image downsampling
void halfsample_image(const cv::Mat& src, cv::Mat& dst);

}

#endif
