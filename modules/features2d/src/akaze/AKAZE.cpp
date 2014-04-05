//=============================================================================
//
// AKAZE.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Georgia Institute of Technology (1)
//               TrueVision Solutions (2)
// Date: 15/09/2013
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2013, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file AKAZE.cpp
 * @brief Main class for detecting and describing binary features in an
 * accelerated nonlinear scale space
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "AKAZE.h"

using namespace std;
using namespace cv;

//*******************************************************************************
//*******************************************************************************

/**
 * @brief AKAZE constructor with input options
 * @param options AKAZE configuration options
 * @note This constructor allocates memory for the nonlinear scale space
*/
AKAZEFeatures::AKAZEFeatures(const AKAZEOptions& options) {

  soffset_ = options.soffset;
  factor_size_ = DEFAULT_FACTOR_SIZE;
  sderivatives_ = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
  omax_ = options.omax;
  nsublevels_ = options.nsublevels;
  dthreshold_ = options.dthreshold;
  descriptor_ = options.descriptor;
  diffusivity_ = options.diffusivity;
  save_scale_space_ = options.save_scale_space;
  verbosity_ = options.verbosity;
  img_width_ = options.img_width;
  img_height_ = options.img_height;
  noctaves_ = omax_;
  ncycles_ = 0;
  reordering_ = true;
  descriptor_size_ = options.descriptor_size;
  descriptor_channels_ = options.descriptor_channels;
  descriptor_pattern_size_ = options.descriptor_pattern_size;
  tkcontrast_ = 0.0;
  tscale_ = 0.0;
  tderivatives_ = 0.0;
  tdetector_ = 0.0;
  textrema_ = 0.0;
  tsubpixel_ = 0.0;
  tdescriptor_ = 0.0;

  if (descriptor_size_ > 0 && descriptor_ >= MLDB_UPRIGHT) {
    generateDescriptorSubsample(descriptorSamples_,descriptorBits_,descriptor_size_,
                                descriptor_pattern_size_,descriptor_channels_);
  }

  Allocate_Memory_Evolution();
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief AKAZE destructor
*/
AKAZEFeatures::~AKAZEFeatures(void) {

  evolution_.clear();
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief This method allocates the memory for the nonlinear diffusion evolution
*/
void AKAZEFeatures::Allocate_Memory_Evolution(void) {

  float rfactor = 0.0;
  int level_height = 0, level_width = 0;

  // Allocate the dimension of the matrices for the evolution
  for (int i = 0; i <= omax_-1 && i <= DEFAULT_OCTAVE_MAX; i++) {
    rfactor = 1.0/pow(2.f,i);
    level_height = (int)(img_height_*rfactor);
    level_width = (int)(img_width_*rfactor);

    // Smallest possible octave
    if (level_width < 80 || level_height < 40) {
      noctaves_ = i;
      i = omax_;
      break;
    }

    for (int j = 0; j < nsublevels_; j++) {
      tevolution aux;
      aux.Lx  = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Ly  = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Lxx = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Lxy = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Lyy = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Lt  = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Ldet = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Lflow  = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.Lstep  = cv::Mat::zeros(level_height,level_width,CV_32F);
      aux.esigma = soffset_*pow(2.f,(float)(j)/(float)(nsublevels_) + i);
      aux.sigma_size = fRound(aux.esigma);
      aux.etime = 0.5*(aux.esigma*aux.esigma);
      aux.octave = i;
      aux.sublevel = j;
      evolution_.push_back(aux);
    }
  }

  // Allocate memory for the number of cycles and time steps
  for (size_t i = 1; i < evolution_.size(); i++) {
    int naux = 0;
    std::vector<float> tau;
    float ttime = 0.0;
    ttime = evolution_[i].etime-evolution_[i-1].etime;
    naux = fed_tau_by_process_time(ttime,1,0.25,reordering_,tau);
    nsteps_.push_back(naux);
    tsteps_.push_back(tau);
    ncycles_++;
  }
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param img Input image for which the nonlinear scale space needs to be created
 * @return 0 if the nonlinear scale space was created successfully, -1 otherwise
*/
int AKAZEFeatures::Create_Nonlinear_Scale_Space(const cv::Mat &img) {

  double t1 = 0.0, t2 = 0.0;

  if (evolution_.size() == 0) {
    cout << "Error generating the nonlinear scale space!!" << endl;
    cout << "Firstly you need to call AKAZE::Allocate_Memory_Evolution()" << endl;
    return -1;
  }

  t1 = getTickCount();

  // Copy the original image to the first level of the evolution
  img.copyTo(evolution_[0].Lt);
  gaussian_2D_convolution(evolution_[0].Lt,evolution_[0].Lt,0,0,soffset_);
  evolution_[0].Lt.copyTo(evolution_[0].Lsmooth);

  // Firstly compute the kcontrast factor
  kcontrast_ = compute_k_percentile(img,KCONTRAST_PERCENTILE,1.0,KCONTRAST_NBINS,0,0);

  t2 = getTickCount();
  tkcontrast_ = 1000.0*(t2-t1) / getTickFrequency();

  // Now generate the rest of evolution levels
  for (size_t i = 1; i < evolution_.size(); i++) {

    if (evolution_[i].octave > evolution_[i-1].octave) {
      halfsample_image(evolution_[i-1].Lt,evolution_[i].Lt);
      kcontrast_ = kcontrast_*0.75;
    }
    else {
      evolution_[i-1].Lt.copyTo(evolution_[i].Lt);
    }

    gaussian_2D_convolution(evolution_[i].Lt,evolution_[i].Lsmooth,0,0,1.0);

    // Compute the Gaussian derivatives Lx and Ly
    image_derivatives_scharr(evolution_[i].Lsmooth,evolution_[i].Lx,1,0);
    image_derivatives_scharr(evolution_[i].Lsmooth,evolution_[i].Ly,0,1);

    // Compute the conductivity equation
    switch (diffusivity_) {
      case 0:
        pm_g1(evolution_[i].Lx,evolution_[i].Ly,evolution_[i].Lflow,kcontrast_);
      break;
      case 1:
        pm_g2(evolution_[i].Lx,evolution_[i].Ly,evolution_[i].Lflow,kcontrast_);
      break;
      case 2:
        weickert_diffusivity(evolution_[i].Lx,evolution_[i].Ly,evolution_[i].Lflow,kcontrast_);
      break;
      case 3:
        charbonnier_diffusivity(evolution_[i].Lx,evolution_[i].Ly,evolution_[i].Lflow,kcontrast_);
      break;
      default:
        std::cerr << "Diffusivity: " << diffusivity_ << " is not supported" << std::endl;
    }

    // Perform FED n inner steps
    for (int j = 0; j < nsteps_[i-1]; j++) {
      nld_step_scalar(evolution_[i].Lt,evolution_[i].Lflow,evolution_[i].Lstep,tsteps_[i-1][j]);
    }
  }

  t2 = getTickCount();
  tscale_ = 1000.0*(t2-t1) / getTickFrequency();

  return 0;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method selects interesting keypoints through the nonlinear scale space
 * @param kpts Vector of detected keypoints
*/
void AKAZEFeatures::Feature_Detection(std::vector<cv::KeyPoint>& kpts) {

  double t1 = 0.0, t2 = 0.0;

  t1 = getTickCount();

  Compute_Determinant_Hessian_Response();
  Find_Scale_Space_Extrema(kpts);
  Do_Subpixel_Refinement(kpts);

  t2 = getTickCount();
  tdetector_ = 1000.0*(t2-t1) / getTickFrequency();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the multiscale derivatives for the nonlinear scale space
*/
void AKAZEFeatures::Compute_Multiscale_Derivatives(void) {

  double t1 = 0.0, t2 = 0.0;

  t1 = getTickCount();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < evolution_.size(); i++) {
    float ratio = pow(2.f,evolution_[i].octave);
    int sigma_size_ = fRound(evolution_[i].esigma*factor_size_/ratio);

    compute_scharr_derivatives(evolution_[i].Lsmooth,evolution_[i].Lx,1,0,sigma_size_);
    compute_scharr_derivatives(evolution_[i].Lsmooth,evolution_[i].Ly,0,1,sigma_size_);
    compute_scharr_derivatives(evolution_[i].Lx,evolution_[i].Lxx,1,0,sigma_size_);
    compute_scharr_derivatives(evolution_[i].Ly,evolution_[i].Lyy,0,1,sigma_size_);
    compute_scharr_derivatives(evolution_[i].Lx,evolution_[i].Lxy,0,1,sigma_size_);

    evolution_[i].Lx = evolution_[i].Lx*((sigma_size_));
    evolution_[i].Ly = evolution_[i].Ly*((sigma_size_));
    evolution_[i].Lxx = evolution_[i].Lxx*((sigma_size_)*(sigma_size_));
    evolution_[i].Lxy = evolution_[i].Lxy*((sigma_size_)*(sigma_size_));
    evolution_[i].Lyy = evolution_[i].Lyy*((sigma_size_)*(sigma_size_));
  }

  t2 = getTickCount();
  tderivatives_ = 1000.0*(t2-t1) / getTickFrequency();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the feature detector response for the nonlinear scale space
 * @note We use the Hessian determinant as the feature detector response
*/
void AKAZEFeatures::Compute_Determinant_Hessian_Response(void) {

  // Firstly compute the multiscale derivatives
  Compute_Multiscale_Derivatives();

  for (size_t i = 0; i < evolution_.size(); i++) {
    if (verbosity_ == true) {
      cout << "Computing detector response. Determinant of Hessian. Evolution time: " << evolution_[i].etime << endl;
    }

    for (int ix = 0; ix < evolution_[i].Ldet.rows; ix++) {
      for (int jx = 0; jx < evolution_[i].Ldet.cols; jx++) {
        float lxx = *(evolution_[i].Lxx.ptr<float>(ix)+jx);
        float lxy = *(evolution_[i].Lxy.ptr<float>(ix)+jx);
        float lyy = *(evolution_[i].Lyy.ptr<float>(ix)+jx);
        *(evolution_[i].Ldet.ptr<float>(ix)+jx) = (lxx*lyy-lxy*lxy);
      }
    }
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method finds extrema in the nonlinear scale space
 * @param kpts Vector of detected keypoints
*/
void AKAZEFeatures::Find_Scale_Space_Extrema(std::vector<cv::KeyPoint>& kpts) {

  double t1 = 0.0, t2 = 0.0;
  float value = 0.0;
  float dist = 0.0, ratio = 0.0, smax = 0.0;
  int npoints = 0, id_repeated = 0;
  int sigma_size_ = 0, left_x = 0, right_x = 0, up_y = 0, down_y = 0;
  bool is_extremum = false, is_repeated = false, is_out = false;
  cv::KeyPoint point;

  // Set maximum size
  if (descriptor_ == SURF_UPRIGHT || descriptor_ == SURF ||
      descriptor_ == MLDB_UPRIGHT || descriptor_ == MLDB) {
    smax = 10.0*sqrtf(2.0);
  }
  else if (descriptor_ == MSURF_UPRIGHT || descriptor_ == MSURF) {
    smax = 12.0*sqrtf(2.0);
  }

  t1 = getTickCount();

  for (size_t i = 0; i < evolution_.size(); i++) {
    for (int ix = 1; ix < evolution_[i].Ldet.rows-1; ix++) {
      for (int jx = 1; jx < evolution_[i].Ldet.cols-1; jx++) {
        is_extremum = false;
        is_repeated = false;
        is_out = false;
        value = *(evolution_[i].Ldet.ptr<float>(ix)+jx);

        // Filter the points with the detector threshold
        if (value > dthreshold_ && value >= DEFAULT_MIN_DETECTOR_THRESHOLD &&
            value > *(evolution_[i].Ldet.ptr<float>(ix)+jx-1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix)+jx+1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix-1)+jx-1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix-1)+jx) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix-1)+jx+1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix+1)+jx-1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix+1)+jx) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix+1)+jx+1)) {
          is_extremum = true;

          point.response = fabs(value);
          point.size = evolution_[i].esigma*factor_size_;
          point.octave = evolution_[i].octave;
          point.class_id = i;
          ratio = pow(2.f,point.octave);
          sigma_size_ = fRound(point.size/ratio);
          point.pt.x = jx;
          point.pt.y = ix;

          for (size_t ik = 0; ik < kpts.size(); ik++) {
            if (point.class_id == kpts[ik].class_id-1 ||
                point.class_id == kpts[ik].class_id   ||
                point.class_id == kpts[ik].class_id+1) {
              dist = sqrt(pow(point.pt.x*ratio-kpts[ik].pt.x,2)+pow(point.pt.y*ratio-kpts[ik].pt.y,2));
              if (dist <= point.size) {
                if (point.response > kpts[ik].response) {
                  id_repeated = ik;
                  is_repeated = true;
                }
                else {
                  is_extremum = false;
                }
                break;
              }
            }
          }

          // Check out of bounds
          if (is_extremum == true) {
            // Check that the point is under the image limits for the descriptor computation
            left_x = fRound(point.pt.x-smax*sigma_size_)-1;
            right_x = fRound(point.pt.x+smax*sigma_size_) +1;
            up_y = fRound(point.pt.y-smax*sigma_size_)-1;
            down_y = fRound(point.pt.y+smax*sigma_size_)+1;

            if (left_x < 0 || right_x >= evolution_[i].Ldet.cols ||
                up_y < 0 || down_y >= evolution_[i].Ldet.rows) {
              is_out = true;
            }

            if (is_out == false) {
              if (is_repeated == false) {
                point.pt.x *= ratio;
                point.pt.y *= ratio;
                kpts.push_back(point);
                npoints++;
              }
              else {
                point.pt.x *= ratio;
                point.pt.y *= ratio;
                kpts[id_repeated] = point;
              }
            } // if is_out
          } //if is_extremum
        }
      } // for jx
    } // for ix
  } // for i

  t2 = getTickCount();
  textrema_ = 1000.0*(t2-t1) / getTickFrequency();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs subpixel refinement of the detected keypoints
 * @param kpts Vector of detected keypoints
*/
void AKAZEFeatures::Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts) {

  double t1 = 0.0, t2 = 0.0;
  float Dx = 0.0, Dy = 0.0, ratio = 0.0;
  float Dxx = 0.0, Dyy = 0.0, Dxy = 0.0;
  int x = 0, y = 0;
  Mat A = Mat::zeros(2,2,CV_32F);
  Mat b = Mat::zeros(2,1,CV_32F);
  Mat dst = Mat::zeros(2,1,CV_32F);

  t1 = getTickCount();

  for (size_t i = 0; i < kpts.size(); i++) {
    ratio = pow(2.f,kpts[i].octave);
    x = fRound(kpts[i].pt.x/ratio);
    y = fRound(kpts[i].pt.y/ratio);

    // Compute the gradient
    Dx = (0.5)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x+1)
                -*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x-1));
    Dy = (0.5)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x)
                -*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x));

    // Compute the Hessian
    Dxx = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x+1)
           + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x-1)
           -2.0*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

    Dyy = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x)
           + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x)
           -2.0*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

    Dxy = (0.25)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x+1)
                  +(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x-1)))
        -(0.25)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y-1)+x+1)
                 +(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y+1)+x-1)));

    // Solve the linear system
    *(A.ptr<float>(0)) = Dxx;
    *(A.ptr<float>(1)+1) = Dyy;
    *(A.ptr<float>(0)+1) = *(A.ptr<float>(1)) = Dxy;
    *(b.ptr<float>(0)) = -Dx;
    *(b.ptr<float>(1)) = -Dy;

    solve(A,b,dst,DECOMP_LU);

    if (fabs(*(dst.ptr<float>(0))) <= 1.0 && fabs(*(dst.ptr<float>(1))) <= 1.0) {
      kpts[i].pt.x = x + (*(dst.ptr<float>(0)));
      kpts[i].pt.y = y + (*(dst.ptr<float>(1)));
      kpts[i].pt.x *= powf(2.f,evolution_[kpts[i].class_id].octave);
      kpts[i].pt.y *= powf(2.f,evolution_[kpts[i].class_id].octave);
      kpts[i].angle = 0.0;

      // In OpenCV the size of a keypoint its the diameter
      kpts[i].size *= 2.0;
    }
    // Delete the point since its not stable
    else {
      kpts.erase(kpts.begin()+i);
      i--;
    }
  }

  t2 = getTickCount();
  tsubpixel_ = 1000.0*(t2-t1) / getTickFrequency();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs feature suppression based on 2D distance
 * @param kpts Vector of keypoints
 * @param mdist Maximum distance in pixels
*/
void AKAZEFeatures::Feature_Suppression_Distance(std::vector<cv::KeyPoint>& kpts, float mdist) {

  vector<KeyPoint> aux;
  vector<int> to_delete;
  float dist = 0.0, x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
  bool found = false;

  for (size_t i = 0; i < kpts.size(); i++) {
    x1 = kpts[i].pt.x;
    y1 = kpts[i].pt.y;
    for (size_t j = i+1; j < kpts.size(); j++) {
      x2 = kpts[j].pt.x;
      y2 = kpts[j].pt.y;
      dist = sqrt(pow(x1-x2,2)+pow(y1-y2,2));
      if (dist < mdist) {
        if (fabs(kpts[i].response) >= fabs(kpts[j].response)) {
          to_delete.push_back(j);
        }
        else {
          to_delete.push_back(i);
          break;
        }
      }
    }
  }

  for (size_t i = 0; i < kpts.size(); i++) {
    found = false;
    for (size_t j = 0; j < to_delete.size(); j++) {
      if (i == (size_t)(to_delete[j])) {
        found = true;
        break;
      }
    }
    if (found == false) {
      aux.push_back(kpts[i]);
    }
  }

  kpts.clear();
  kpts = aux;
  aux.clear();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method  computes the set of descriptors through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 * @param desc Matrix to store the descriptors
*/
void AKAZEFeatures::Compute_Descriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {

  double t1 = 0.0, t2 = 0.0;

  t1 = getTickCount();

  // Allocate memory for the matrix with the descriptors
  if (descriptor_ < MLDB_UPRIGHT) {
    desc = cv::Mat::zeros(kpts.size(),64,CV_32FC1);
  }
  else {
    // We use the full length binary descriptor -> 486 bits
    if (descriptor_size_ == 0) {
      int t = (6+36+120)*descriptor_channels_;
      desc = cv::Mat::zeros(kpts.size(),ceil(t/8.),CV_8UC1);
    }
    else {
      // We use the random bit selection length binary descriptor
      desc = cv::Mat::zeros(kpts.size(),ceil(descriptor_size_/8.),CV_8UC1);
    }
  }

  switch (descriptor_)
  {
    case SURF_UPRIGHT : // Upright descriptors, not invariant to rotation
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < kpts.size(); i++) {
        Get_SURF_Descriptor_Upright_64(kpts[i],desc.ptr<float>(i));
      }
    }
    break;
    case SURF :
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < kpts.size(); i++) {
        Compute_Main_Orientation_SURF(kpts[i]);
        Get_SURF_Descriptor_64(kpts[i],desc.ptr<float>(i));
      }
    }
    break;
    case MSURF_UPRIGHT : // Upright descriptors, not invariant to rotation
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < kpts.size(); i++) {
        Get_MSURF_Upright_Descriptor_64(kpts[i],desc.ptr<float>(i));
      }
    }
    break;
    case MSURF :
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < kpts.size(); i++) {
        Compute_Main_Orientation_SURF(kpts[i]);
        Get_MSURF_Descriptor_64(kpts[i],desc.ptr<float>(i));
      }
    }
    break;
    case MLDB_UPRIGHT : // Upright descriptors, not invariant to rotation
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < kpts.size(); i++) {
        if (descriptor_size_ == 0)
          Get_Upright_MLDB_Full_Descriptor(kpts[i],desc.ptr<unsigned char>(i));
        else
          Get_Upright_MLDB_Descriptor_Subset(kpts[i],desc.ptr<unsigned char>(i));
      }
    }
    break;
    case MLDB :
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < kpts.size(); i++) {
        Compute_Main_Orientation_SURF(kpts[i]);
        if (descriptor_size_ == 0)
          Get_MLDB_Full_Descriptor(kpts[i],desc.ptr<unsigned char>(i));
        else
          Get_MLDB_Descriptor_Subset(kpts[i],desc.ptr<unsigned char>(i));
      }
    }
    break;
  }

  t2 = getTickCount();
  tdescriptor_ = 1000.0*(t2-t1) / getTickFrequency();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the main orientation for a given keypoint
 * @param kpt Input keypoint
 * @note The orientation is computed using a similar approach as described in the
 * original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
*/
void AKAZEFeatures::Compute_Main_Orientation_SURF(cv::KeyPoint& kpt) {

  int ix = 0, iy = 0, idx = 0, s = 0, level = 0;
  float xf = 0.0, yf = 0.0, gweight = 0.0, ratio = 0.0;
  std::vector<float> resX(109), resY(109), Ang(109);
  const int id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};

  // Variables for computing the dominant direction
  float sumX = 0.0, sumY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;

  // Get the information from the keypoint
  level = kpt.class_id;
  ratio = (float)(1<<evolution_[level].octave);
  s = fRound(0.5*kpt.size/ratio);
  xf = kpt.pt.x/ratio;
  yf = kpt.pt.y/ratio;

  // Calculate derivatives responses for points within radius of 6*scale
  for (int i = -6; i <= 6; ++i) {
    for (int j = -6; j <= 6; ++j) {
      if (i*i + j*j < 36) {
        iy = fRound(yf + j*s);
        ix = fRound(xf + i*s);

        gweight = gauss25[id[i+6]][id[j+6]];
        resX[idx] = gweight*(*(evolution_[level].Lx.ptr<float>(iy)+ix));
        resY[idx] = gweight*(*(evolution_[level].Ly.ptr<float>(iy)+ix));

        Ang[idx] = get_angle(resX[idx],resY[idx]);
        ++idx;
      }
    }
  }

  // Loop slides pi/3 window around feature point
  for (ang1 = 0; ang1 < 2.0*CV_PI;  ang1+=0.15f) {
    ang2 =(ang1+CV_PI/3.0f > 2.0*CV_PI ? ang1-5.0f*CV_PI/3.0f : ang1+CV_PI/3.0f);
    sumX = sumY = 0.f;

    for (size_t k = 0; k < Ang.size(); ++k) {
      // Get angle from the x-axis of the sample point
      const float & ang = Ang[k];

      // Determine whether the point is within the window
      if (ang1 < ang2 && ang1 < ang && ang < ang2) {
        sumX+=resX[k];
        sumY+=resY[k];
      }
      else if (ang2 < ang1 &&
               ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2.0*CV_PI) )) {
        sumX+=resX[k];
        sumY+=resY[k];
      }
    }

    // if the vector produced from this window is longer than all
    // previous vectors then this forms the new dominant direction
    if (sumX*sumX + sumY*sumY > max) {
      // store largest orientation
      max = sumX*sumX + sumY*sumY;
      kpt.angle = get_angle(sumX, sumY);
    }
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the upright descriptor of the provided keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 64. No additional
 * Gaussian weighting is performed. The descriptor is inspired from Bay et al.,
 * Speeded Up Robust Features, ECCV, 2006
*/
void AKAZEFeatures::Get_SURF_Descriptor_Upright_64(const cv::KeyPoint& kpt, float *desc) {

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int scale = 0, dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 64;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  ratio = (float)(1<<kpt.octave);
  scale = fRound(0.5*kpt.size/ratio);
  level = kpt.class_id;
  yf = kpt.pt.y/ratio;
  xf = kpt.pt.x/ratio;

  // Calculate descriptor for this interest point
  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {
      dx=dy=mdx=mdy=0.0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {
          // Get the coordinates of the sample point on the rotated axis
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = (int)(sample_y-.5);
          x1 = (int)(sample_x-.5);

          y2 = (int)(sample_y+.5);
          x2 = (int)(sample_x+.5);

          fx = sample_x-x1;
          fy = sample_y-y1;

          res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          // Sum the derivatives to the cumulative descriptor
          dx += rx;
          dy += ry;
          mdx += fabs(rx);
          mdy += fabs(ry);
        }
      }

      // Add the values to the descriptor vector
      desc[dcount++] = dx;
      desc[dcount++] = dy;
      desc[dcount++] = mdx;
      desc[dcount++] = mdy;

      // Store the current length^2 of the vector
      len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for (int i = 0; i < dsize; i++) {
    desc[i] /= len;
  }
}

//*************************************************************************************
//*************************************************************************************
/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 64. No additional
 * Gaussian weighting is performed. The descriptor is inspired from Bay et al.,
 * Speeded Up Robust Features, ECCV, 2006
*/
void AKAZEFeatures::Get_SURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc) {

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int scale = 0, dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 64;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  ratio = (float)(1<<kpt.octave);
  scale = fRound(0.5*kpt.size/ratio);
  angle = kpt.angle;
  level = kpt.class_id;
  yf = kpt.pt.y/ratio;
  xf = kpt.pt.x/ratio;
  co = cos(angle);
  si = sin(angle);

  // Calculate descriptor for this interest point
  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {
      dx=dy=mdx=mdy=0.0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {
          // Get the coordinates of the sample point on the rotated axis
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          y1 = (int)(sample_y-.5);
          x1 = (int)(sample_x-.5);

          y2 = (int)(sample_y+.5);
          x2 = (int)(sample_x+.5);

          fx = sample_x-x1;
          fy = sample_y-y1;

          res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          // Get the x and y derivatives on the rotated axis
          rry = rx*co + ry*si;
          rrx = -rx*si + ry*co;

          // Sum the derivatives to the cumulative descriptor
          dx += rrx;
          dy += rry;
          mdx += fabs(rrx);
          mdy += fabs(rry);
        }
      }

      // Add the values to the descriptor vector
      desc[dcount++] = dx;
      desc[dcount++] = dy;
      desc[dcount++] = mdx;
      desc[dcount++] = mdy;

      // Store the current length^2 of the vector
      len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for (int i = 0; i < dsize; i++) {
    desc[i] /= len;
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
*/
void AKAZEFeatures::Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float *desc) {

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int scale = 0, dsize = 0, level = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5, cy = 0.5;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 64;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1<<kpt.octave);
  scale = fRound(0.5*kpt.size/ratio);
  level = kpt.class_id;
  yf = kpt.pt.y/ratio;
  xf = kpt.pt.x/ratio;

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while (i < pattern_size) {
    j = -8;
    i = i-4;

    cx += 1.0;
    cy = -0.5;

    while (j < pattern_size) {
      dx=dy=mdx=mdy=0.0;
      cy += 1.0;
      j = j-4;

      ky = i + sample_step;
      kx = j + sample_step;

      ys = yf + (ky*scale);
      xs = xf + (kx*scale);

      for (int k = i; k < i+9; k++) {
        for (int l = j; l < j+9; l++) {
          sample_y = k*scale + yf;
          sample_x = l*scale + xf;

          //Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.50*scale);

          y1 = (int)(sample_y-.5);
          x1 = (int)(sample_x-.5);

          y2 = (int)(sample_y+.5);
          x2 = (int)(sample_x+.5);

          fx = sample_x-x1;
          fy = sample_y-y1;

          res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          rx = gauss_s1*rx;
          ry = gauss_s1*ry;

          // Sum the derivatives to the cumulative descriptor
          dx += rx;
          dy += ry;
          mdx += fabs(rx);
          mdy += fabs(ry);
        }
      }

      // Add the values to the descriptor vector
      gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);

      desc[dcount++] = dx*gauss_s2;
      desc[dcount++] = dy*gauss_s2;
      desc[dcount++] = mdx*gauss_s2;
      desc[dcount++] = mdy*gauss_s2;

      len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

      j += 9;
    }

    i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

  for (int i = 0; i < dsize; i++) {
    desc[i] /= len;
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
*/
void AKAZEFeatures::Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc) {

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
  int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  int scale = 0, dsize = 0, level = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5, cy = 0.5;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 64;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1<<kpt.octave);
  scale = fRound(0.5*kpt.size/ratio);
  angle = kpt.angle;
  level = kpt.class_id;
  yf = kpt.pt.y/ratio;
  xf = kpt.pt.x/ratio;
  co = cos(angle);
  si = sin(angle);

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while (i < pattern_size) {
    j = -8;
    i = i-4;

    cx += 1.0;
    cy = -0.5;

    while (j < pattern_size) {
      dx=dy=mdx=mdy=0.0;
      cy += 1.0;
      j = j - 4;

      ky = i + sample_step;
      kx = j + sample_step;

      xs = xf + (-kx*scale*si + ky*scale*co);
      ys = yf + (kx*scale*co + ky*scale*si);

      for (int k = i; k < i + 9; ++k) {
        for (int l = j; l < j + 9; ++l) {
          // Get coords of sample point on the rotated axis
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          // Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5*scale);

          y1 = fRound(sample_y-.5);
          x1 = fRound(sample_x-.5);

          y2 = fRound(sample_y+.5);
          x2 = fRound(sample_x+.5);

          fx = sample_x-x1;
          fy = sample_y-y1;

          res1 = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution_[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution_[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution_[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution_[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

          // Get the x and y derivatives on the rotated axis
          rry = gauss_s1*(rx*co + ry*si);
          rrx = gauss_s1*(-rx*si + ry*co);

          // Sum the derivatives to the cumulative descriptor
          dx += rrx;
          dy += rry;
          mdx += fabs(rrx);
          mdy += fabs(rry);
        }
      }

      // Add the values to the descriptor vector
      gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);
      desc[dcount++] = dx*gauss_s2;
      desc[dcount++] = dy*gauss_s2;
      desc[dcount++] = mdx*gauss_s2;
      desc[dcount++] = mdy*gauss_s2;

      len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

      j += 9;
    }

    i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

  for (int i = 0; i < dsize; i++) {
    desc[i] /= len;
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the rupright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
*/
void AKAZEFeatures::Get_Upright_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char *desc) {

  float di = 0.0, dx = 0.0, dy = 0.0;
  float ri = 0.0, rx = 0.0, ry = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, ratio = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int level = 0, nsamples = 0, scale = 0;
  int dcount1 = 0, dcount2 = 0;

  // Matrices for the M-LDB descriptor
  Mat values_1 = Mat::zeros(4,descriptor_channels_,CV_32FC1);
  Mat values_2 = Mat::zeros(9,descriptor_channels_,CV_32FC1);
  Mat values_3 = Mat::zeros(16,descriptor_channels_,CV_32FC1);

  // Get the information from the keypoint
  ratio = (float)(1<<kpt.octave);
  scale = fRound(0.5*kpt.size/ratio);
  level = kpt.class_id;
  yf = kpt.pt.y/ratio;
  xf = kpt.pt.x/ratio;

  // First 2x2 grid
  pattern_size = descriptor_pattern_size_;
  sample_step = pattern_size;

  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {
      di=dx=dy=0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {
          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

          di += ri;
          dx += rx;
          dy += ry;
          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_1.ptr<float>(dcount2)) = di;
      *(values_1.ptr<float>(dcount2)+1) = dx;
      *(values_1.ptr<float>(dcount2)+2) = dy;
      dcount2++;
    }
  }

  // Do binary comparison first level
  for(int i = 0; i < 4; i++) {
    for (int j = i+1; j < 4; j++) {
      if (*(values_1.ptr<float>(i)) > *(values_1.ptr<float>(j))) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;

      if (*(values_1.ptr<float>(i)+1) > *(values_1.ptr<float>(j)+1)) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;

      if (*(values_1.ptr<float>(i)+2) > *(values_1.ptr<float>(j)+2)) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;
    }
  }

  // Second 3x3 grid
  sample_step = ceil(pattern_size*2./3.);
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {
      di=dx=dy=0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {
          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

          di += ri;
          dx += rx;
          dy += ry;
          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_2.ptr<float>(dcount2)) = di;
      *(values_2.ptr<float>(dcount2)+1) = dx;
      *(values_2.ptr<float>(dcount2)+2) = dy;
      dcount2++;
    }
  }

  dcount2 = 0;
  //Do binary comparison second level
  for (int i = 0; i < 9; i++) {
    for (int j = i+1; j < 9; j++) {
      if (*(values_2.ptr<float>(i)) > *(values_2.ptr<float>(j))) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;

      if (*(values_2.ptr<float>(i)+1) > *(values_2.ptr<float>(j)+1)) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;

      if(*(values_2.ptr<float>(i)+2) > *(values_2.ptr<float>(j)+2)) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;
    }
  }

  // Third 4x4 grid
  sample_step = pattern_size/2;
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {
      di=dx=dy=0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {
          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

          di += ri;
          dx += rx;
          dy += ry;
          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_3.ptr<float>(dcount2)) = di;
      *(values_3.ptr<float>(dcount2)+1) = dx;
      *(values_3.ptr<float>(dcount2)+2) = dy;
      dcount2++;
    }
  }

  dcount2 = 0;
  //Do binary comparison third level
  for (int i = 0; i < 16; i++) {
    for (int j = i+1; j < 16; j++) {
      if (*(values_3.ptr<float>(i)) > *(values_3.ptr<float>(j))) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;

      if (*(values_3.ptr<float>(i)+1) > *(values_3.ptr<float>(j)+1)) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;

      if (*(values_3.ptr<float>(i)+2) > *(values_3.ptr<float>(j)+2)) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;
    }
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
*/
void AKAZEFeatures::Get_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char *desc) {

  float di = 0.0, dx = 0.0, dy = 0.0, ratio = 0.0;
  float ri = 0.0, rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int level = 0, nsamples = 0, scale = 0;
  int dcount1 = 0, dcount2 = 0;

  // Matrices for the M-LDB descriptor
  Mat values_1 = Mat::zeros(4,descriptor_channels_,CV_32FC1);
  Mat values_2 = Mat::zeros(9,descriptor_channels_,CV_32FC1);
  Mat values_3 = Mat::zeros(16,descriptor_channels_,CV_32FC1);

  // Get the information from the keypoint
  ratio = (float)(1<<kpt.octave);
  scale = fRound(0.5*kpt.size/ratio);
  angle = kpt.angle;
  level = kpt.class_id;
  yf = kpt.pt.y/ratio;
  xf = kpt.pt.x/ratio;
  co = cos(angle);
  si = sin(angle);

  // First 2x2 grid
  pattern_size = descriptor_pattern_size_;
  sample_step = pattern_size;

  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {

      di=dx=dy=0.0;
      nsamples = 0;

      for (float k = i; k < i + sample_step; k++) {
        for (float l = j; l < j + sample_step; l++) {
          // Get the coordinates of the sample point
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

          di += ri;

          if (descriptor_channels_ == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (descriptor_channels_ == 3) {
            // Get the x and y derivatives on the rotated axis
            rry = rx*co + ry*si;
            rrx = -rx*si + ry*co;
            dx += rrx;
            dy += rry;
          }

          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_1.ptr<float>(dcount2)) = di;
      if ( descriptor_channels_ > 1 ) {
        *(values_1.ptr<float>(dcount2)+1) = dx;
      }

      if ( descriptor_channels_ > 2 ) {
        *(values_1.ptr<float>(dcount2)+2) = dy;
      }

      dcount2++;
    }
  }

  // Do binary comparison first level
  for (int i = 0; i < 4; i++) {
    for (int j = i+1; j < 4; j++) {
      if (*(values_1.ptr<float>(i)) > *(values_1.ptr<float>(j))) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;
    }
  }

  if (descriptor_channels_ > 1) {
    for (int i = 0; i < 4; i++) {
      for (int j = i+1; j < 4; j++) {
        if (*(values_1.ptr<float>(i)+1) > *(values_1.ptr<float>(j)+1)) {
          desc[dcount1/8] |= (1<<(dcount1%8));
        }

        dcount1++;
      }
    }
  }

  if (descriptor_channels_ > 2) {
    for (int i = 0; i < 4; i++) {
      for ( int j = i+1; j < 4; j++) {
        if (*(values_1.ptr<float>(i)+2) > *(values_1.ptr<float>(j)+2)) {
          desc[dcount1/8] |= (1<<(dcount1%8));
        }
        dcount1++;
      }
    }
  }

  // Second 3x3 grid
  sample_step = ceil(pattern_size*2./3.);
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {

      di=dx=dy=0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);
          di += ri;

          if (descriptor_channels_ == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (descriptor_channels_ == 3) {
            // Get the x and y derivatives on the rotated axis
            rry = rx*co + ry*si;
            rrx = -rx*si + ry*co;
            dx += rrx;
            dy += rry;
          }

          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_2.ptr<float>(dcount2)) = di;
      if (descriptor_channels_ > 1) {
        *(values_2.ptr<float>(dcount2)+1) = dx;
      }

      if (descriptor_channels_ > 2) {
        *(values_2.ptr<float>(dcount2)+2) = dy;
      }

      dcount2++;
    }
  }

  // Do binary comparison second level
  for (int i = 0; i < 9; i++) {
    for (int j = i+1; j < 9; j++) {
      if (*(values_2.ptr<float>(i)) > *(values_2.ptr<float>(j))) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;
    }
  }

  if (descriptor_channels_ > 1) {
    for (int i = 0; i < 9; i++) {
      for (int j = i+1; j < 9; j++) {
        if (*(values_2.ptr<float>(i)+1) > *(values_2.ptr<float>(j)+1)) {
          desc[dcount1/8] |= (1<<(dcount1%8));
        }
        dcount1++;
      }
    }
  }

  if (descriptor_channels_ > 2) {
    for (int i = 0; i < 9; i++) {
      for (int j = i+1; j < 9; j++) {
        if (*(values_2.ptr<float>(i)+2) > *(values_2.ptr<float>(j)+2)) {
          desc[dcount1/8] |= (1<<(dcount1%8));
        }
        dcount1++;
      }
    }
  }

  // Third 4x4 grid
  sample_step = pattern_size/2;
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i+=sample_step) {
    for (int j = -pattern_size; j < pattern_size; j+=sample_step) {
      di=dx=dy=0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {
          // Get the coordinates of the sample point
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution_[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);
          di += ri;

          if (descriptor_channels_ == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (descriptor_channels_ == 3) {
            // Get the x and y derivatives on the rotated axis
            rry = rx*co + ry*si;
            rrx = -rx*si + ry*co;
            dx += rrx;
            dy += rry;
          }

          nsamples++;
        }
      }

      di /= nsamples;
      dx /= nsamples;
      dy /= nsamples;

      *(values_3.ptr<float>(dcount2)) = di;
      if (descriptor_channels_ > 1) {
        *(values_3.ptr<float>(dcount2)+1) = dx;
      }

      if (descriptor_channels_ > 2) {
        *(values_3.ptr<float>(dcount2)+2) = dy;
      }

      dcount2++;
    }
  }

  // Do binary comparison third level
  for(int i = 0; i < 16; i++) {
    for(int j = i+1; j < 16; j++) {
      if (*(values_3.ptr<float>(i)) > *(values_3.ptr<float>(j))) {
        desc[dcount1/8] |= (1<<(dcount1%8));
      }
      dcount1++;
    }
  }

  if (descriptor_channels_ > 1) {
    for (int i = 0; i < 16; i++) {
      for (int j = i+1; j < 16; j++) {
        if (*(values_3.ptr<float>(i)+1) > *(values_3.ptr<float>(j)+1)) {
          desc[dcount1/8] |= (1<<(dcount1%8));
        }
        dcount1++;
      }
    }
  }

  if (descriptor_channels_ > 2)
  {
    for (int i = 0; i < 16; i++) {
      for (int j = i+1; j < 16; j++) {
        if (*(values_3.ptr<float>(i)+2) > *(values_3.ptr<float>(j)+2)) {
          desc[dcount1/8] |= (1<<(dcount1%8));
        }
        dcount1++;
      }
    }
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the M-LDB descriptor of the provided keypoint given the
 * main orientation of the keypoint. The descriptor is computed based on a subset of
 * the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
*/
void AKAZEFeatures::Get_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char *desc) {

  float di, dx, dy;
  float  rx, ry;
  float sample_x = 0.f, sample_y = 0.f;
  int x1 = 0, y1 = 0;

  // Get the information from the keypoint
  float ratio = (float)(1<<kpt.octave);
  int scale = fRound(0.5*kpt.size/ratio);
  float angle = kpt.angle;
  float level = kpt.class_id;
  float yf = kpt.pt.y/ratio;
  float xf = kpt.pt.x/ratio;
  float co = cos(angle);
  float si = sin(angle);

  // Allocate memory for the matrix of values
  Mat values = cv::Mat_<float>::zeros((4+9+16)*descriptor_channels_,1);

  // Sample everything, but only do the comparisons
  vector<int> steps(3);
  steps.at(0) = descriptor_pattern_size_;
  steps.at(1) = ceil(2.f*descriptor_pattern_size_/3.f);
  steps.at(2) = descriptor_pattern_size_/2;

  for (int i=0; i<descriptorSamples_.rows; i++) {
    int *coords = descriptorSamples_.ptr<int>(i);
    int sample_step = steps.at(coords[0]);
    di=0.0f;
    dx=0.0f;
    dy=0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {
        // Get the coordinates of the sample point
        sample_y = yf + (l*scale*co + k*scale*si);
        sample_x = xf + (-l*scale*si + k*scale*co);

        y1 = fRound(sample_y);
        x1 = fRound(sample_x);

        di += *(evolution_[level].Lt.ptr<float>(y1)+x1);

        if (descriptor_channels_ > 1) {
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

          if (descriptor_channels_ == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (descriptor_channels_ == 3) {
            // Get the x and y derivatives on the rotated axis
            dx += rx*co + ry*si;
            dy += -rx*si + ry*co;
          }
        }
      }
    }

    *(values.ptr<float>(descriptor_channels_*i)) = di;

    if (descriptor_channels_ == 2) {
      *(values.ptr<float>(descriptor_channels_*i+1)) = dx;
    }
    else if (descriptor_channels_ == 3) {
      *(values.ptr<float>(descriptor_channels_*i+1)) = dx;
      *(values.ptr<float>(descriptor_channels_*i+2)) = dy;
    }
  }

  // Do the comparisons
  const float *vals = values.ptr<float>(0);
  const int *comps = descriptorBits_.ptr<int>(0);

  for (int i=0; i<descriptorBits_.rows; i++) {
    if (vals[comps[2*i]] > vals[comps[2*i +1]]) {
      desc[i/8] |= (1<<(i%8));
    }
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the upright (not rotation invariant) M-LDB descriptor
 * of the provided keypoint given the main orientation of the keypoint.
 * The descriptor is computed based on a subset of the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
*/
void AKAZEFeatures::Get_Upright_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char *desc) {

  float di = 0.0f, dx = 0.0f, dy = 0.0f;
  float rx = 0.0f, ry = 0.0f;
  float sample_x = 0.0f, sample_y = 0.0f;
  int x1 = 0, y1 = 0;

  // Get the information from the keypoint
  float ratio = (float)(1<<kpt.octave);
  int scale = fRound(0.5*kpt.size/ratio);
  float level = kpt.class_id;
  float yf = kpt.pt.y/ratio;
  float xf = kpt.pt.x/ratio;

  // Allocate memory for the matrix of values
  Mat values = cv::Mat_<float>::zeros((4+9+16)*descriptor_channels_,1);

  vector<int> steps(3);
  steps.at(0) = descriptor_pattern_size_;
  steps.at(1) = ceil(2.f*descriptor_pattern_size_/3.f);
  steps.at(2) = descriptor_pattern_size_/2;

  for (int i=0; i < descriptorSamples_.rows; i++) {
    int *coords = descriptorSamples_.ptr<int>(i);
    int sample_step = steps.at(coords[0]);
    di=0.0f;
    dx=0.0f;
    dy=0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {
        // Get the coordinates of the sample point
        sample_y = yf + l*scale;
        sample_x = xf + k*scale;

        y1 = fRound(sample_y);
        x1 = fRound(sample_x);
        di += *(evolution_[level].Lt.ptr<float>(y1)+x1);

        if (descriptor_channels_ > 1) {
          rx = *(evolution_[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution_[level].Ly.ptr<float>(y1)+x1);

          if (descriptor_channels_ == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (descriptor_channels_ == 3) {
            dx += rx;
            dy += ry;
          }
        }
      }
    }

    *(values.ptr<float>(descriptor_channels_*i)) = di;

    if (descriptor_channels_ == 2) {
      *(values.ptr<float>(descriptor_channels_*i+1)) = dx;
    }
    else if (descriptor_channels_ == 3) {
      *(values.ptr<float>(descriptor_channels_*i+1)) = dx;
      *(values.ptr<float>(descriptor_channels_*i+2)) = dy;
    }
  }

  // Do the comparisons
  const float *vals = values.ptr<float>(0);
  const int *comps = descriptorBits_.ptr<int>(0);

  for (int i=0; i<descriptorBits_.rows; i++) {
    if (vals[comps[2*i]] > vals[comps[2*i +1]]) {
      desc[i/8] |= (1<<(i%8));
    }
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes a (quasi-random) list of bits to be taken
 * from the full descriptor. To speed the extraction, the function creates
 * a list of the samples that are involved in generating at least a bit (sampleList)
 * and a list of the comparisons between those samples (comparisons)
 * @param sampleList
 * @param comparisons The matrix with the binary comparisons
 * @param nbits The number of bits of the descriptor
 * @param pattern_size The pattern size for the binary descriptor
 * @param nchannels Number of channels to consider in the descriptor (1-3)
 * @note The function keeps the 18 bits (3-channels by 6 comparisons) of the
 * coarser grid, since it provides the most robust estimations
 */
void generateDescriptorSubsample(cv::Mat& sampleList, cv::Mat& comparisons, int nbits,
                                 int pattern_size, int nchannels) {

  int ssz = 0;
  for (int i=0; i<3; i++) {
    int gz = (i+2)*(i+2);
    ssz += gz*(gz-1)/2;
  }
  ssz *= nchannels;

  CV_Assert(nbits<=ssz && "descriptor size can't be bigger than full descriptor");

  // Since the full descriptor is usually under 10k elements, we pick
  // the selection from the full matrix.  We take as many samples per
  // pick as the number of channels. For every pick, we
  // take the two samples involved and put them in the sampling list

  Mat_<int> fullM(ssz/nchannels,5);
  for (size_t i=0, c=0; i<3; i++) {
    int gdiv = i+2; //grid divisions, per row
    int gsz = gdiv*gdiv;
    int psz = ceil(2.*pattern_size/(float)gdiv);

    for (int j=0; j<gsz; j++) {
      for (int k=j+1; k<gsz; k++,c++) {
        fullM(c,0) = i;
        fullM(c,1) = psz*(j % gdiv) - pattern_size;
        fullM(c,2) = psz*(j / gdiv) - pattern_size;
        fullM(c,3) = psz*(k % gdiv) - pattern_size;
        fullM(c,4) = psz*(k / gdiv) - pattern_size;
      }
    }
  }

  srand(1024);
  Mat_<int> comps = Mat_<int>(nchannels*ceil(nbits/(float)nchannels),2);
  comps = 1000;

  // Select some samples. A sample includes all channels
  int count =0;
  size_t npicks = ceil(nbits/(float)nchannels);
  Mat_<int> samples(29,3);
  Mat_<int> fullcopy = fullM.clone();
  samples = -1;

  for (size_t i=0; i<npicks; i++) {
    size_t k = rand() % (fullM.rows-i);
    if (i < 6) {
      // Force use of the coarser grid values and comparisons
      k = i;
    }

    bool n = true;

    for (int j=0; j<count; j++) {
      if (samples(j,0) == fullcopy(k,0) && samples(j,1) == fullcopy(k,1) && samples(j,2) == fullcopy(k,2)) {
        n = false;
        comps(i*nchannels,0) = nchannels*j;
        comps(i*nchannels+1,0) = nchannels*j+1;
        comps(i*nchannels+2,0) = nchannels*j+2;
        break;
      }
    }

    if (n) {
      samples(count,0) = fullcopy(k,0);
      samples(count,1) = fullcopy(k,1);
      samples(count,2) = fullcopy(k,2);
      comps(i*nchannels,0) = nchannels*count;
      comps(i*nchannels+1,0) = nchannels*count+1;
      comps(i*nchannels+2,0) = nchannels*count+2;
      count++;
    }

    n = true;
    for (int j=0; j<count; j++) {
      if (samples(j,0) == fullcopy(k,0) && samples(j,1) == fullcopy(k,3) && samples(j,2) == fullcopy(k,4)) {
        n = false;
        comps(i*nchannels,1) = nchannels*j;
        comps(i*nchannels+1,1) = nchannels*j+1;
        comps(i*nchannels+2,1) = nchannels*j+2;
        break;
      }
    }

    if (n) {
      samples(count,0) = fullcopy(k,0);
      samples(count,1) = fullcopy(k,3);
      samples(count,2) = fullcopy(k,4);
      comps(i*nchannels,1) = nchannels*count;
      comps(i*nchannels+1,1) = nchannels*count+1;
      comps(i*nchannels+2,1) = nchannels*count+2;
      count++;
    }

    Mat tmp = fullcopy.row(k);
    fullcopy.row(fullcopy.rows-i-1).copyTo(tmp);
  }

  sampleList = samples.rowRange(0,count).clone();
  comparisons = comps.rowRange(0,nbits).clone();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the angle from the vector given by (X Y). From 0 to 2*Pi
*/
inline float get_angle(float x, float y) {

  if (x >= 0 && y >= 0) {
    return atanf(y/x);
  }

  if (x < 0 && y >= 0) {
    return CV_PI - atanf(-y/x);
  }

  if (x < 0 && y < 0) {
    return CV_PI + atanf(y/x);
  }

  if(x >= 0 && y < 0) {
    return 2.0*CV_PI - atanf(-y/x);
  }

  return 0;
}

//**************************************************************************************
//**************************************************************************************

/**
 * @brief This function computes the value of a 2D Gaussian function
 * @param x X Position
 * @param y Y Position
 * @param sig Standard Deviation
*/
inline float gaussian(float x, float y, float sigma) {

  return expf(-(x*x+y*y)/(2.0f*sigma*sigma));
}

//**************************************************************************************
//**************************************************************************************

/**
 * @brief This function checks descriptor limits
 * @param x X Position
 * @param y Y Position
 * @param width Image width
 * @param height Image height
*/
inline void check_descriptor_limits(int &x, int &y, const int width, const int height) {

  if (x < 0) {
    x = 0;
  }

  if (y < 0) {
    y = 0;
  }

  if (x > width-1) {
    x = width-1;
  }

  if (y > height-1) {
    y = height-1;
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This funtion rounds float to nearest integer
 * @param flt Input float
 * @return dst Nearest integer
 */
inline int fRound(float flt)
{
  return (int)(flt+0.5f);
}

