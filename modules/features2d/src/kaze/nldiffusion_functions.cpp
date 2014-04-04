
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

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function smoothes an image with a Gaussian kernel
 * @param src Input image
 * @param dst Output image
 * @param ksize_x Kernel size in X-direction (horizontal)
 * @param ksize_y Kernel size in Y-direction (vertical)
 * @param sigma Kernel standard deviation
 */
void gaussian_2D_convolution(const cv::Mat& src, cv::Mat& dst,
                             int ksize_x, int ksize_y, float sigma) {

  size_t ksize_x_ = 0, ksize_y_ = 0;

  // Compute an appropriate kernel size according to the specified sigma
  if (sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0) {
    ksize_x_ = ceil(2.0*(1.0 + (sigma-0.8)/(0.3)));
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
  GaussianBlur(src,dst,Size(ksize_x_,ksize_y_),sigma,sigma,cv::BORDER_REPLICATE);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the Perona and Malik conductivity coefficient g1
 * g1 = exp(-|dL|^2/k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void pm_g1(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {
  cv::exp(-(Lx.mul(Lx) + Ly.mul(Ly))/(k*k),dst);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the Perona and Malik conductivity coefficient g2
 * g2 = 1 / (1 + dL^2 / k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void pm_g2(const cv::Mat &Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {
  dst = 1./(1. + (Lx.mul(Lx) + Ly.mul(Ly))/(k*k));
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes Weickert conductivity coefficient g3
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
  cv::pow((Lx.mul(Lx) + Ly.mul(Ly))/(k*k),4,modg);
  cv::exp(-3.315/modg, dst);
  dst = 1.0 - dst;
}

//*************************************************************************************
//*************************************************************************************

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
float compute_k_percentile(const cv::Mat& img, float perc, float gscale,
                           int nbins, int ksize_x, int ksize_y) {

  int nbin = 0, nelements = 0, nthreshold = 0, k = 0;
  float kperc = 0.0, modg = 0.0, lx = 0.0, ly = 0.0;
  float npoints = 0.0;
  float hmax = 0.0;

  // Create the array for the histogram
  float *hist = new float[nbins];

  // Create the matrices
  Mat gaussian = Mat::zeros(img.rows,img.cols,CV_32F);
  Mat Lx = Mat::zeros(img.rows,img.cols,CV_32F);
  Mat Ly = Mat::zeros(img.rows,img.cols,CV_32F);

  // Set the histogram to zero, just in case
  for (int i = 0; i < nbins; i++) {
    hist[i] = 0.0;
  }

  // Perform the Gaussian convolution
  gaussian_2D_convolution(img,gaussian,ksize_x,ksize_y,gscale);

  // Compute the Gaussian derivatives Lx and Ly
  Scharr(gaussian,Lx,CV_32F,1,0,1,0,cv::BORDER_DEFAULT);
  Scharr(gaussian,Ly,CV_32F,0,1,1,0,cv::BORDER_DEFAULT);

  // Skip the borders for computing the histogram
  for (int i = 1; i < gaussian.rows-1; i++) {
    for (int j = 1; j < gaussian.cols-1; j++) {
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
  for (int i = 1; i < gaussian.rows-1; i++) {
    for (int j = 1; j < gaussian.cols-1; j++) {
      lx = *(Lx.ptr<float>(i)+j);
      ly = *(Ly.ptr<float>(i)+j);
      modg = sqrt(lx*lx + ly*ly);

      // Find the correspondent bin
      if (modg != 0.0) {
        nbin = floor(nbins*(modg/hmax));

        if (nbin == nbins) {
          nbin--;
        }

        hist[nbin]++;
        npoints++;
      }
    }
  }

  // Now find the perc of the histogram percentile
  nthreshold = (size_t)(npoints*perc);


  for (k = 0; nelements < nthreshold && k < nbins; k++) {
    nelements = nelements + hist[k];
  }

  if (nelements < nthreshold)  {
    kperc = 0.03;
  }
  else {
    kperc = hmax*((float)(k)/(float)nbins);
  }

  delete hist;
  return kperc;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes Scharr image derivatives
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 * @param scale Scale factor or derivative size
 */
void compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst,
                                int xorder, int yorder, int scale) {
  Mat kx, ky;
  compute_derivative_kernels(kx,ky,xorder,yorder,scale);
  sepFilter2D(src,dst,CV_32F,kx,ky);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief Compute derivative kernels for sizes different than 3
 * @param _kx Horizontal kernel values
 * @param _ky Vertical kernel values
 * @param dx Derivative order in X-direction (horizontal)
 * @param dy Derivative order in Y-direction (vertical)
 * @param scale_ Scale factor or derivative size
 */
void compute_derivative_kernels(cv::OutputArray _kx, cv::OutputArray _ky,
                                int dx, int dy, int scale) {

  int ksize = 3 + 2*(scale-1);

  // The standard Scharr kernel
  if (scale == 1) {
    getDerivKernels(_kx,_ky,dx,dy,0,true,CV_32F);
    return;
  }

  _kx.create(ksize,1,CV_32F,-1,true);
  _ky.create(ksize,1,CV_32F,-1,true);
  Mat kx = _kx.getMat();
  Mat ky = _ky.getMat();

  float w = 10.0/3.0;
  float norm = 1.0/(2.0*scale*(w+2.0));

  for (int k = 0; k < 2; k++) {
    Mat* kernel = k == 0 ? &kx : &ky;
    int order = k == 0 ? dx : dy;
	std::vector<float> kerI(ksize);

    for (int t=0; t<ksize; t++) {
      kerI[t] = 0;
    }

    if (order == 0) {
      kerI[0] = norm, kerI[ksize/2] = w*norm, kerI[ksize-1] = norm;
    }
    else if (order == 1) {
      kerI[0] = -1, kerI[ksize/2] = 0, kerI[ksize-1] = 1;
    }

    Mat temp(kernel->rows,kernel->cols,CV_32F,&kerI[0]);
    temp.copyTo(*kernel);
  }
}

//*************************************************************************************
//*************************************************************************************

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

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 1; i < Lstep.rows-1; i++) {
    for (int j = 1; j < Lstep.cols-1; j++) {
      float xpos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(i)+j+1)))*((*(Ld.ptr<float>(i)+j+1))-(*(Ld.ptr<float>(i)+j)));
      float xneg = ((*(c.ptr<float>(i)+j-1))+(*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j))-(*(Ld.ptr<float>(i)+j-1)));
      float ypos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(i+1)+j)))*((*(Ld.ptr<float>(i+1)+j))-(*(Ld.ptr<float>(i)+j)));
      float yneg = ((*(c.ptr<float>(i-1)+j))+(*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j))-(*(Ld.ptr<float>(i-1)+j)));
      *(Lstep.ptr<float>(i)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
    }
  }

  for (int j = 1; j < Lstep.cols-1; j++) {
    float xpos = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(0)+j+1)))*((*(Ld.ptr<float>(0)+j+1))-(*(Ld.ptr<float>(0)+j)));
    float xneg = ((*(c.ptr<float>(0)+j-1))+(*(c.ptr<float>(0)+j)))*((*(Ld.ptr<float>(0)+j))-(*(Ld.ptr<float>(0)+j-1)));
    float ypos = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(1)+j)))*((*(Ld.ptr<float>(1)+j))-(*(Ld.ptr<float>(0)+j)));
    float yneg = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(0)+j)))*((*(Ld.ptr<float>(0)+j))-(*(Ld.ptr<float>(0)+j)));
    *(Lstep.ptr<float>(0)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  for (int j = 1; j < Lstep.cols-1; j++) {
    float xpos = ((*(c.ptr<float>(Lstep.rows-1)+j))+(*(c.ptr<float>(Lstep.rows-1)+j+1)))*((*(Ld.ptr<float>(Lstep.rows-1)+j+1))-(*(Ld.ptr<float>(Lstep.rows-1)+j)));
    float xneg = ((*(c.ptr<float>(Lstep.rows-1)+j-1))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-1)+j-1)));
    float ypos = ((*(c.ptr<float>(Lstep.rows-1)+j))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-1)+j)));
    float yneg = ((*(c.ptr<float>(Lstep.rows-2)+j))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-2)+j)));
    *(Lstep.ptr<float>(Lstep.rows-1)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  for (int i = 1; i < Lstep.rows-1; i++) {
    float xpos = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i)+1)))*((*(Ld.ptr<float>(i)+1))-(*(Ld.ptr<float>(i))));
    float xneg = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i)))-(*(Ld.ptr<float>(i))));
    float ypos = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i+1))))*((*(Ld.ptr<float>(i+1)))-(*(Ld.ptr<float>(i))));
    float yneg = ((*(c.ptr<float>(i-1)))+(*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i)))-(*(Ld.ptr<float>(i-1))));
    *(Lstep.ptr<float>(i)) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  for (int i = 1; i < Lstep.rows-1; i++) {
    float xpos = ((*(c.ptr<float>(i)+Lstep.cols-1))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-1)));
    float xneg = ((*(c.ptr<float>(i)+Lstep.cols-2))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-2)));
    float ypos = ((*(c.ptr<float>(i)+Lstep.cols-1))+(*(c.ptr<float>(i+1)+Lstep.cols-1)))*((*(Ld.ptr<float>(i+1)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-1)));
    float yneg = ((*(c.ptr<float>(i-1)+Lstep.cols-1))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i-1)+Lstep.cols-1)));
    *(Lstep.ptr<float>(i)+Lstep.cols-1) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  Ld = Ld + Lstep;
}

//*************************************************************************************
//*************************************************************************************

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
bool check_maximum_neighbourhood(const cv::Mat& img, int dsize, float value,
                                 int row, int col, bool same_img) {

  bool response = true;

  for (int i = row-dsize; i <= row+dsize; i++) {
    for (int j = col-dsize; j <= col+dsize; j++) {
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
