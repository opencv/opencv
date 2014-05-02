//=============================================================================
//
// nldiffusion_functions.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 27/12/2011
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file nldiffusion_functions.cpp
 * @brief Functions for non-linear diffusion applications:
 * 2D Gaussian Derivatives
 * Perona and Malik conductivity equations
 * Perona and Malik evolution
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#include "nldiffusion_functions.h"

// Namespaces
using namespace std;
using namespace cv;

/* ************************************************************************* */

namespace cv {
    namespace details {
        namespace kaze {

            /* ************************************************************************* */
            /**
             * @brief This function smoothes an image with a Gaussian kernel
             * @param src Input image
             * @param dst Output image
             * @param ksize_x Kernel size in X-direction (horizontal)
             * @param ksize_y Kernel size in Y-direction (vertical)
             * @param sigma Kernel standard deviation
             */
            void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst, int ksize_x, int ksize_y, float sigma) {

                int ksize_x_ = 0, ksize_y_ = 0;

                // Compute an appropriate kernel size according to the specified sigma
                if (sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0) {
                    ksize_x_ = (int)ceil(2.0f*(1.0f + (sigma - 0.8f) / (0.3f)));
                    ksize_y_ = ksize_x_;
                }

                // The kernel size must be and odd number
                if ((ksize_x_ % 2) == 0) {
                    ksize_x_ += 1;
                }

                if ((ksize_y_ % 2) == 0) {
                    ksize_y_ += 1;
                }

                // Perform the Gaussian Smoothing with border replication
                GaussianBlur(src, dst, Size(ksize_x_, ksize_y_), sigma, sigma, BORDER_REPLICATE);
            }

            /* ************************************************************************* */
            /**
             * @brief This function computes image derivatives with Scharr kernel
             * @param src Input image
             * @param dst Output image
             * @param xorder Derivative order in X-direction (horizontal)
             * @param yorder Derivative order in Y-direction (vertical)
             * @note Scharr operator approximates better rotation invariance than
             * other stencils such as Sobel. See Weickert and Scharr,
             * A Scheme for Coherence-Enhancing Diffusion Filtering with Optimized Rotation Invariance,
             * Journal of Visual Communication and Image Representation 2002
             */
            void image_derivatives_scharr(const cv::Mat& src, cv::Mat& dst, int xorder, int yorder) {
                Scharr(src, dst, CV_32F, xorder, yorder, 1.0, 0, BORDER_DEFAULT);
            }

            /* ************************************************************************* */
            /**
             * @brief This function computes the Perona and Malik conductivity coefficient g1
             * g1 = exp(-|dL|^2/k^2)
             * @param Lx First order image derivative in X-direction (horizontal)
             * @param Ly First order image derivative in Y-direction (vertical)
             * @param dst Output image
             * @param k Contrast factor parameter
             */
            void pm_g1(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {
                cv::exp(-(Lx.mul(Lx) + Ly.mul(Ly)) / (k*k), dst);
            }

            /* ************************************************************************* */
            /**
             * @brief This function computes the Perona and Malik conductivity coefficient g2
             * g2 = 1 / (1 + dL^2 / k^2)
             * @param Lx First order image derivative in X-direction (horizontal)
             * @param Ly First order image derivative in Y-direction (vertical)
             * @param dst Output image
             * @param k Contrast factor parameter
             */
            void pm_g2(const cv::Mat &Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {
                dst = 1.0f / (1.0f + (Lx.mul(Lx) + Ly.mul(Ly)) / (k*k));
            }

            /* ************************************************************************* */
            /**
             * @brief This function computes Weickert conductivity coefficient gw
             * @param Lx First order image derivative in X-direction (horizontal)
             * @param Ly First order image derivative in Y-direction (vertical)
             * @param dst Output image
             * @param k Contrast factor parameter
             * @note For more information check the following paper: J. Weickert
             * Applications of nonlinear diffusion in image processing and computer vision,
             * Proceedings of Algorithmy 2000
             */
            void weickert_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {
                Mat modg;
                cv::pow((Lx.mul(Lx) + Ly.mul(Ly)) / (k*k), 4, modg);
                cv::exp(-3.315f / modg, dst);
                dst = 1.0f - dst;
            }

            /* ************************************************************************* */
            /**
            * @brief This function computes Charbonnier conductivity coefficient gc
            * gc = 1 / sqrt(1 + dL^2 / k^2)
            * @param Lx First order image derivative in X-direction (horizontal)
            * @param Ly First order image derivative in Y-direction (vertical)
            * @param dst Output image
            * @param k Contrast factor parameter
            * @note For more information check the following paper: J. Weickert
            * Applications of nonlinear diffusion in image processing and computer vision,
            * Proceedings of Algorithmy 2000
            */
            void charbonnier_diffusivity(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {
                Mat den;
                cv::sqrt(1.0f + (Lx.mul(Lx) + Ly.mul(Ly)) / (k*k), den);
                dst = 1.0f / den;
            }

            /* ************************************************************************* */
            /**
             * @brief This function computes a good empirical value for the k contrast factor
             * given an input image, the percentile (0-1), the gradient scale and the number of
             * bins in the histogram
             * @param img Input image
             * @param perc Percentile of the image gradient histogram (0-1)
             * @param gscale Scale for computing the image gradient histogram
             * @param nbins Number of histogram bins
             * @param ksize_x Kernel size in X-direction (horizontal) for the Gaussian smoothing kernel
             * @param ksize_y Kernel size in Y-direction (vertical) for the Gaussian smoothing kernel
             * @return k contrast factor
             */
            float compute_k_percentile(const cv::Mat& img, float perc, float gscale, int nbins, int ksize_x, int ksize_y) {

                int nbin = 0, nelements = 0, nthreshold = 0, k = 0;
                float kperc = 0.0, modg = 0.0, lx = 0.0, ly = 0.0;
                float npoints = 0.0;
                float hmax = 0.0;

                // Create the array for the histogram
                std::vector<int> hist(nbins, 0);

                // Create the matrices
                Mat gaussian = Mat::zeros(img.rows, img.cols, CV_32F);
                Mat Lx = Mat::zeros(img.rows, img.cols, CV_32F);
                Mat Ly = Mat::zeros(img.rows, img.cols, CV_32F);

                // Perform the Gaussian convolution
                gaussian_2D_convolution(img, gaussian, ksize_x, ksize_y, gscale);

                // Compute the Gaussian derivatives Lx and Ly
                Scharr(gaussian, Lx, CV_32F, 1, 0, 1, 0, cv::BORDER_DEFAULT);
                Scharr(gaussian, Ly, CV_32F, 0, 1, 1, 0, cv::BORDER_DEFAULT);

                // Skip the borders for computing the histogram
                for (int i = 1; i < gaussian.rows - 1; i++) {
                    for (int j = 1; j < gaussian.cols - 1; j++) {
                        lx = *(Lx.ptr<float>(i)+j);
                        ly = *(Ly.ptr<float>(i)+j);
                        modg = sqrt(lx*lx + ly*ly);

                        // Get the maximum
                        if (modg > hmax) {
                            hmax = modg;
                        }
                    }
                }

                // Skip the borders for computing the histogram
                for (int i = 1; i < gaussian.rows - 1; i++) {
                    for (int j = 1; j < gaussian.cols - 1; j++) {
                        lx = *(Lx.ptr<float>(i)+j);
                        ly = *(Ly.ptr<float>(i)+j);
                        modg = sqrt(lx*lx + ly*ly);

                        // Find the correspondent bin
                        if (modg != 0.0) {
                            nbin = (int)floor(nbins*(modg / hmax));

                            if (nbin == nbins) {
                                nbin--;
                            }

                            hist[nbin]++;
                            npoints++;
                        }
                    }
                }

                // Now find the perc of the histogram percentile
                nthreshold = (int)(npoints*perc);

                for (k = 0; nelements < nthreshold && k < nbins; k++) {
                    nelements = nelements + hist[k];
                }

                if (nelements < nthreshold)  {
                    kperc = 0.03f;
                }
                else {
                    kperc = hmax*((float)(k) / (float)nbins);
                }

                return kperc;
            }

            /* ************************************************************************* */
            /**
             * @brief This function computes Scharr image derivatives
             * @param src Input image
             * @param dst Output image
             * @param xorder Derivative order in X-direction (horizontal)
             * @param yorder Derivative order in Y-direction (vertical)
             * @param scale Scale factor for the derivative size
             */
            void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, int xorder, int yorder, int scale) {
                Mat kx, ky;
                compute_derivative_kernels(kx, ky, xorder, yorder, scale);
                sepFilter2D(src, dst, CV_32F, kx, ky);
            }

            /* ************************************************************************* */
            /**
             * @brief Compute derivative kernels for sizes different than 3
             * @param _kx Horizontal kernel values
             * @param _ky Vertical kernel values
             * @param dx Derivative order in X-direction (horizontal)
             * @param dy Derivative order in Y-direction (vertical)
             * @param scale_ Scale factor or derivative size
             */
            void compute_derivative_kernels(cv::OutputArray _kx, cv::OutputArray _ky, int dx, int dy, int scale) {

                int ksize = 3 + 2 * (scale - 1);

                // The standard Scharr kernel
                if (scale == 1) {
                    getDerivKernels(_kx, _ky, dx, dy, 0, true, CV_32F);
                    return;
                }

                _kx.create(ksize, 1, CV_32F, -1, true);
                _ky.create(ksize, 1, CV_32F, -1, true);
                Mat kx = _kx.getMat();
                Mat ky = _ky.getMat();

                float w = 10.0f / 3.0f;
                float norm = 1.0f / (2.0f*scale*(w + 2.0f));

                for (int k = 0; k < 2; k++) {
                    Mat* kernel = k == 0 ? &kx : &ky;
                    int order = k == 0 ? dx : dy;
                    std::vector<float> kerI(ksize, 0.0f);

                    if (order == 0) {
                        kerI[0] = norm, kerI[ksize / 2] = w*norm, kerI[ksize - 1] = norm;
                    }
                    else if (order == 1) {
                        kerI[0] = -1, kerI[ksize / 2] = 0, kerI[ksize - 1] = 1;
                    }

                    Mat temp(kernel->rows, kernel->cols, CV_32F, &kerI[0]);
                    temp.copyTo(*kernel);
                }
            }

            class Nld_Step_Scalar_Invoker : public cv::ParallelLoopBody
            {
            public:
                Nld_Step_Scalar_Invoker(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, float _stepsize)
                    : _Ld(&Ld)
                    , _c(&c)
                    , _Lstep(&Lstep)
                    , stepsize(_stepsize)
                {
                }

                virtual ~Nld_Step_Scalar_Invoker()
                {

                }

                void operator()(const cv::Range& range) const
                {
                    cv::Mat& Ld = *_Ld;
                    const cv::Mat& c = *_c;
                    cv::Mat& Lstep = *_Lstep;

                    for (int i = range.start; i < range.end; i++)
                    {
                        for (int j = 1; j < Lstep.cols - 1; j++)
                        {
                            float xpos = ((*(c.ptr<float>(i)+j)) + (*(c.ptr<float>(i)+j + 1)))*((*(Ld.ptr<float>(i)+j + 1)) - (*(Ld.ptr<float>(i)+j)));
                            float xneg = ((*(c.ptr<float>(i)+j - 1)) + (*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j)) - (*(Ld.ptr<float>(i)+j - 1)));
                            float ypos = ((*(c.ptr<float>(i)+j)) + (*(c.ptr<float>(i + 1) + j)))*((*(Ld.ptr<float>(i + 1) + j)) - (*(Ld.ptr<float>(i)+j)));
                            float yneg = ((*(c.ptr<float>(i - 1) + j)) + (*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j)) - (*(Ld.ptr<float>(i - 1) + j)));
                            *(Lstep.ptr<float>(i)+j) = 0.5f*stepsize*(xpos - xneg + ypos - yneg);
                        }
                    }
                }

            private:
                cv::Mat * _Ld;
                const cv::Mat * _c;
                cv::Mat * _Lstep;
                float stepsize;
            };

            /* ************************************************************************* */
            /**
            * @brief This function performs a scalar non-linear diffusion step
            * @param Ld2 Output image in the evolution
            * @param c Conductivity image
            * @param Lstep Previous image in the evolution
            * @param stepsize The step size in time units
            * @note Forward Euler Scheme 3x3 stencil
            * The function c is a scalar value that depends on the gradient norm
            * dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
            */
            void nld_step_scalar(cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep, float stepsize) {

                cv::parallel_for_(cv::Range(1, Lstep.rows - 1), Nld_Step_Scalar_Invoker(Ld, c, Lstep, stepsize));

                for (int j = 1; j < Lstep.cols - 1; j++) {
                    float xpos = ((*(c.ptr<float>(0) + j)) + (*(c.ptr<float>(0) + j + 1)))*((*(Ld.ptr<float>(0) + j + 1)) - (*(Ld.ptr<float>(0) + j)));
                    float xneg = ((*(c.ptr<float>(0) + j - 1)) + (*(c.ptr<float>(0) + j)))*((*(Ld.ptr<float>(0) + j)) - (*(Ld.ptr<float>(0) + j - 1)));
                    float ypos = ((*(c.ptr<float>(0) + j)) + (*(c.ptr<float>(1) + j)))*((*(Ld.ptr<float>(1) + j)) - (*(Ld.ptr<float>(0) + j)));
                    *(Lstep.ptr<float>(0) + j) = 0.5f*stepsize*(xpos - xneg + ypos);
                }

                for (int j = 1; j < Lstep.cols - 1; j++) {
                    float xpos = ((*(c.ptr<float>(Lstep.rows - 1) + j)) + (*(c.ptr<float>(Lstep.rows - 1) + j + 1)))*((*(Ld.ptr<float>(Lstep.rows - 1) + j + 1)) - (*(Ld.ptr<float>(Lstep.rows - 1) + j)));
                    float xneg = ((*(c.ptr<float>(Lstep.rows - 1) + j - 1)) + (*(c.ptr<float>(Lstep.rows - 1) + j)))*((*(Ld.ptr<float>(Lstep.rows - 1) + j)) - (*(Ld.ptr<float>(Lstep.rows - 1) + j - 1)));
                    float ypos = ((*(c.ptr<float>(Lstep.rows - 1) + j)) + (*(c.ptr<float>(Lstep.rows - 1) + j)))*((*(Ld.ptr<float>(Lstep.rows - 1) + j)) - (*(Ld.ptr<float>(Lstep.rows - 1) + j)));
                    float yneg = ((*(c.ptr<float>(Lstep.rows - 2) + j)) + (*(c.ptr<float>(Lstep.rows - 1) + j)))*((*(Ld.ptr<float>(Lstep.rows - 1) + j)) - (*(Ld.ptr<float>(Lstep.rows - 2) + j)));
                    *(Lstep.ptr<float>(Lstep.rows - 1) + j) = 0.5f*stepsize*(xpos - xneg + ypos - yneg);
                }

                for (int i = 1; i < Lstep.rows - 1; i++) {
                    float xpos = ((*(c.ptr<float>(i))) + (*(c.ptr<float>(i)+1)))*((*(Ld.ptr<float>(i)+1)) - (*(Ld.ptr<float>(i))));
                    float xneg = ((*(c.ptr<float>(i))) + (*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i))) - (*(Ld.ptr<float>(i))));
                    float ypos = ((*(c.ptr<float>(i))) + (*(c.ptr<float>(i + 1))))*((*(Ld.ptr<float>(i + 1))) - (*(Ld.ptr<float>(i))));
                    float yneg = ((*(c.ptr<float>(i - 1))) + (*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i))) - (*(Ld.ptr<float>(i - 1))));
                    *(Lstep.ptr<float>(i)) = 0.5f*stepsize*(xpos - xneg + ypos - yneg);
                }

                for (int i = 1; i < Lstep.rows - 1; i++) {
                    float xneg = ((*(c.ptr<float>(i)+Lstep.cols - 2)) + (*(c.ptr<float>(i)+Lstep.cols - 1)))*((*(Ld.ptr<float>(i)+Lstep.cols - 1)) - (*(Ld.ptr<float>(i)+Lstep.cols - 2)));
                    float ypos = ((*(c.ptr<float>(i)+Lstep.cols - 1)) + (*(c.ptr<float>(i + 1) + Lstep.cols - 1)))*((*(Ld.ptr<float>(i + 1) + Lstep.cols - 1)) - (*(Ld.ptr<float>(i)+Lstep.cols - 1)));
                    float yneg = ((*(c.ptr<float>(i - 1) + Lstep.cols - 1)) + (*(c.ptr<float>(i)+Lstep.cols - 1)))*((*(Ld.ptr<float>(i)+Lstep.cols - 1)) - (*(Ld.ptr<float>(i - 1) + Lstep.cols - 1)));
                    *(Lstep.ptr<float>(i)+Lstep.cols - 1) = 0.5f*stepsize*(-xneg + ypos - yneg);
                }

                Ld = Ld + Lstep;
            }

            /* ************************************************************************* */
            /**
            * @brief This function downsamples the input image using OpenCV resize
            * @param img Input image to be downsampled
            * @param dst Output image with half of the resolution of the input image
            */
            void halfsample_image(const cv::Mat& src, cv::Mat& dst) {

                // Make sure the destination image is of the right size
                CV_Assert(src.cols / 2 == dst.cols);
                CV_Assert(src.rows / 2 == dst.rows);
                resize(src, dst, dst.size(), 0, 0, cv::INTER_AREA);
            }

            /* ************************************************************************* */
            /**
             * @brief This function checks if a given pixel is a maximum in a local neighbourhood
             * @param img Input image where we will perform the maximum search
             * @param dsize Half size of the neighbourhood
             * @param value Response value at (x,y) position
             * @param row Image row coordinate
             * @param col Image column coordinate
             * @param same_img Flag to indicate if the image value at (x,y) is in the input image
             * @return 1->is maximum, 0->otherwise
             */
            bool check_maximum_neighbourhood(const cv::Mat& img, int dsize, float value, int row, int col, bool same_img) {

                bool response = true;

                for (int i = row - dsize; i <= row + dsize; i++) {
                    for (int j = col - dsize; j <= col + dsize; j++) {
                        if (i >= 0 && i < img.rows && j >= 0 && j < img.cols) {
                            if (same_img == true) {
                                if (i != row || j != col) {
                                    if ((*(img.ptr<float>(i)+j)) > value) {
                                        response = false;
                                        return response;
                                    }
                                }
                            }
                            else {
                                if ((*(img.ptr<float>(i)+j)) > value) {
                                    response = false;
                                    return response;
                                }
                            }
                        }
                    }
                }

                return response;
            }
        }
    }
}