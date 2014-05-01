/**
 * @file nldiffusion_functions.h
 * @brief Functions for nonlinear diffusion filtering applications
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#ifndef AKAZE_NLDIFFUSION_FUNCTIONS_H
#define AKAZE_NLDIFFUSION_FUNCTIONS_H

/* ************************************************************************* */
// Includes
#include "precomp.hpp"

/* ************************************************************************* */
// Declaration of functions

namespace cv {
    namespace details {
        namespace akaze {

            void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst, int ksize_x, int ksize_y, float sigma);
            void image_derivatives_scharr(const cv::Mat& src, cv::Mat& dst, int xorder, int yorder);
            void pm_g1(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
            void pm_g2(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
            void weickert_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
            void charbonnier_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, const float& k);
            float compute_k_percentile(const cv::Mat& img, float perc, float gscale, int nbins, int ksize_x, int ksize_y);
            void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, int xorder, int, int scale);
            void nld_step_scalar(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, const float& stepsize);
            void halfsample_image(const cv::Mat& src, cv::Mat& dst);
            void compute_derivative_kernels(cv::OutputArray kx_, cv::OutputArray ky_, int dx, int dy, int scale);
            bool check_maximum_neighbourhood(const cv::Mat& img, int dsize, float value, int row, int col, bool same_img);

        }
    }
}

#endif
