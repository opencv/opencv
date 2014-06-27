/**
 * @file AKAZEFeatures.cpp
 * @brief Main class for detecting and describing binary features in an
 * accelerated nonlinear scale space
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "AKAZEFeatures.h"
#include "fed.h"
#include "nldiffusion_functions.h"
#include "utils.h"

#include <iostream>

// Namespaces
using namespace std;
using namespace cv;
using namespace cv::details::kaze;

/* ************************************************************************* */
/**
 * @brief AKAZEFeatures constructor with input options
 * @param options AKAZEFeatures configuration options
 * @note This constructor allocates memory for the nonlinear scale space
 */
AKAZEFeatures::AKAZEFeatures(const AKAZEOptions& options) : options_(options) {

  ncycles_ = 0;
  reordering_ = true;

  if (options_.descriptor_size > 0 && options_.descriptor >= cv::DESCRIPTOR_MLDB_UPRIGHT) {
    generateDescriptorSubsample(descriptorSamples_, descriptorBits_, options_.descriptor_size,
                                options_.descriptor_pattern_size, options_.descriptor_channels);
  }

  Allocate_Memory_Evolution();
}

/* ************************************************************************* */
/**
 * @brief This method allocates the memory for the nonlinear diffusion evolution
 */
void AKAZEFeatures::Allocate_Memory_Evolution(void) {

  float rfactor = 0.0f;
  int level_height = 0, level_width = 0;

  // Allocate the dimension of the matrices for the evolution
  for (int i = 0; i <= options_.omax - 1; i++) {
    rfactor = 1.0f / pow(2.f, i);
    level_height = (int)(options_.img_height*rfactor);
    level_width = (int)(options_.img_width*rfactor);

    // Smallest possible octave and allow one scale if the image is small
    if ((level_width < 80 || level_height < 40) && i != 0) {
      options_.omax = i;
      break;
    }

    for (int j = 0; j < options_.nsublevels; j++) {
      TEvolution step;
      step.Lx = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.Ly = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.Lxx = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.Lxy = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.Lyy = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.Lt = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.Ldet = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.Lsmooth = cv::Mat::zeros(level_height, level_width, CV_32F);
      step.esigma = options_.soffset*pow(2.f, (float)(j) / (float)(options_.nsublevels) + i);
      step.sigma_size = fRound(step.esigma);
      step.etime = 0.5f*(step.esigma*step.esigma);
      step.octave = i;
      step.sublevel = j;
      evolution_.push_back(step);
    }
  }

  // Allocate memory for the number of cycles and time steps
  for (size_t i = 1; i < evolution_.size(); i++) {
    int naux = 0;
    vector<float> tau;
    float ttime = 0.0f;
    ttime = evolution_[i].etime - evolution_[i - 1].etime;
    naux = fed_tau_by_process_time(ttime, 1, 0.25f, reordering_, tau);
    nsteps_.push_back(naux);
    tsteps_.push_back(tau);
    ncycles_++;
  }
}

/* ************************************************************************* */
/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param img Input image for which the nonlinear scale space needs to be created
 * @return 0 if the nonlinear scale space was created successfully, -1 otherwise
 */
int AKAZEFeatures::Create_Nonlinear_Scale_Space(const cv::Mat& img)
{
  CV_Assert(evolution_.size() > 0);

  // Copy the original image to the first level of the evolution
  img.copyTo(evolution_[0].Lt);
  gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);
  evolution_[0].Lt.copyTo(evolution_[0].Lsmooth);

  // Allocate memory for the flow and step images
  cv::Mat Lflow = cv::Mat::zeros(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);
  cv::Mat Lstep = cv::Mat::zeros(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);

  // First compute the kcontrast factor
  options_.kcontrast = compute_k_percentile(img, options_.kcontrast_percentile, 1.0f, options_.kcontrast_nbins, 0, 0);

  // Now generate the rest of evolution levels
  for (size_t i = 1; i < evolution_.size(); i++) {

    if (evolution_[i].octave > evolution_[i - 1].octave) {
      halfsample_image(evolution_[i - 1].Lt, evolution_[i].Lt);
      options_.kcontrast = options_.kcontrast*0.75f;

      // Allocate memory for the resized flow and step images
      Lflow = cv::Mat::zeros(evolution_[i].Lt.rows, evolution_[i].Lt.cols, CV_32F);
      Lstep = cv::Mat::zeros(evolution_[i].Lt.rows, evolution_[i].Lt.cols, CV_32F);
    }
    else {
      evolution_[i - 1].Lt.copyTo(evolution_[i].Lt);
    }

    gaussian_2D_convolution(evolution_[i].Lt, evolution_[i].Lsmooth, 0, 0, 1.0f);

    // Compute the Gaussian derivatives Lx and Ly
    image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Lx, 1, 0);
    image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Ly, 0, 1);

    // Compute the conductivity equation
    switch (options_.diffusivity) {
      case cv::DIFF_PM_G1:
        pm_g1(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
      break;
      case cv::DIFF_PM_G2:
        pm_g2(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
      break;
      case cv::DIFF_WEICKERT:
        weickert_diffusivity(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
      break;
      case cv::DIFF_CHARBONNIER:
        charbonnier_diffusivity(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
      break;
      default:
        CV_Error(options_.diffusivity, "Diffusivity is not supported");
      break;
    }

    // Perform FED n inner steps
    for (int j = 0; j < nsteps_[i - 1]; j++) {
      cv::details::kaze::nld_step_scalar(evolution_[i].Lt, Lflow, Lstep, tsteps_[i - 1][j]);
    }
  }

  return 0;
}

/* ************************************************************************* */
/**
 * @brief This method selects interesting keypoints through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 */
void AKAZEFeatures::Feature_Detection(std::vector<cv::KeyPoint>& kpts)
{
  kpts.clear();
  Compute_Determinant_Hessian_Response();
  Find_Scale_Space_Extrema(kpts);
  Do_Subpixel_Refinement(kpts);
}

/* ************************************************************************* */
class MultiscaleDerivativesAKAZEInvoker : public cv::ParallelLoopBody
{
public:
    explicit MultiscaleDerivativesAKAZEInvoker(std::vector<TEvolution>& ev, const AKAZEOptions& opt)
    : evolution_(&ev)
    , options_(opt)
  {
  }

  void operator()(const cv::Range& range) const
  {
    std::vector<TEvolution>& evolution = *evolution_;

    for (int i = range.start; i < range.end; i++)
    {
      float ratio = pow(2.f, (float)evolution[i].octave);
      int sigma_size_ = fRound(evolution[i].esigma * options_.derivative_factor / ratio);

      compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Lx, 1, 0, sigma_size_);
      compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Ly, 0, 1, sigma_size_);
      compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxx, 1, 0, sigma_size_);
      compute_scharr_derivatives(evolution[i].Ly, evolution[i].Lyy, 0, 1, sigma_size_);
      compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxy, 0, 1, sigma_size_);

      evolution[i].Lx = evolution[i].Lx*((sigma_size_));
      evolution[i].Ly = evolution[i].Ly*((sigma_size_));
      evolution[i].Lxx = evolution[i].Lxx*((sigma_size_)*(sigma_size_));
      evolution[i].Lxy = evolution[i].Lxy*((sigma_size_)*(sigma_size_));
      evolution[i].Lyy = evolution[i].Lyy*((sigma_size_)*(sigma_size_));
    }
  }

private:
  std::vector<TEvolution>*  evolution_;
  AKAZEOptions              options_;
};

/* ************************************************************************* */
/**
 * @brief This method computes the multiscale derivatives for the nonlinear scale space
 */
void AKAZEFeatures::Compute_Multiscale_Derivatives(void)
{
  cv::parallel_for_(cv::Range(0, (int)evolution_.size()),
                                        MultiscaleDerivativesAKAZEInvoker(evolution_, options_));
}

/* ************************************************************************* */
/**
 * @brief This method computes the feature detector response for the nonlinear scale space
 * @note We use the Hessian determinant as the feature detector response
 */
void AKAZEFeatures::Compute_Determinant_Hessian_Response(void) {

  // Firstly compute the multiscale derivatives
  Compute_Multiscale_Derivatives();

  for (size_t i = 0; i < evolution_.size(); i++)
  {
    for (int ix = 0; ix < evolution_[i].Ldet.rows; ix++)
    {
      for (int jx = 0; jx < evolution_[i].Ldet.cols; jx++)
      {
        float lxx = *(evolution_[i].Lxx.ptr<float>(ix)+jx);
        float lxy = *(evolution_[i].Lxy.ptr<float>(ix)+jx);
        float lyy = *(evolution_[i].Lyy.ptr<float>(ix)+jx);
        *(evolution_[i].Ldet.ptr<float>(ix)+jx) = (lxx*lyy - lxy*lxy);
      }
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This method finds extrema in the nonlinear scale space
 * @param kpts Vector of detected keypoints
 */
void AKAZEFeatures::Find_Scale_Space_Extrema(std::vector<cv::KeyPoint>& kpts)
{

  float value = 0.0;
  float dist = 0.0, ratio = 0.0, smax = 0.0;
  int npoints = 0, id_repeated = 0;
  int sigma_size_ = 0, left_x = 0, right_x = 0, up_y = 0, down_y = 0;
  bool is_extremum = false, is_repeated = false, is_out = false;
  cv::KeyPoint point;
  vector<cv::KeyPoint> kpts_aux;

  // Set maximum size
  if (options_.descriptor == cv::DESCRIPTOR_MLDB_UPRIGHT || options_.descriptor == cv::DESCRIPTOR_MLDB) {
    smax = 10.0f*sqrtf(2.0f);
  }
  else if (options_.descriptor == cv::DESCRIPTOR_KAZE_UPRIGHT || options_.descriptor == cv::DESCRIPTOR_KAZE) {
    smax = 12.0f*sqrtf(2.0f);
  }

  for (size_t i = 0; i < evolution_.size(); i++) {
    for (int ix = 1; ix < evolution_[i].Ldet.rows - 1; ix++) {
      for (int jx = 1; jx < evolution_[i].Ldet.cols - 1; jx++) {
        is_extremum = false;
        is_repeated = false;
        is_out = false;
        value = *(evolution_[i].Ldet.ptr<float>(ix)+jx);

        // Filter the points with the detector threshold
        if (value > options_.dthreshold && value >= options_.min_dthreshold &&
            value > *(evolution_[i].Ldet.ptr<float>(ix)+jx - 1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix)+jx + 1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix - 1) + jx - 1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix - 1) + jx) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix - 1) + jx + 1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix + 1) + jx - 1) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix + 1) + jx) &&
            value > *(evolution_[i].Ldet.ptr<float>(ix + 1) + jx + 1)) {

          is_extremum = true;
          point.response = fabs(value);
          point.size = evolution_[i].esigma*options_.derivative_factor;
          point.octave = (int)evolution_[i].octave;
          point.class_id = (int)i;
          ratio = pow(2.f, point.octave);
          sigma_size_ = fRound(point.size / ratio);
          point.pt.x = static_cast<float>(jx);
          point.pt.y = static_cast<float>(ix);

          // Compare response with the same and lower scale
          for (size_t ik = 0; ik < kpts_aux.size(); ik++) {

            if ((point.class_id - 1) == kpts_aux[ik].class_id ||
                point.class_id == kpts_aux[ik].class_id) {
              dist = sqrt(pow(point.pt.x*ratio - kpts_aux[ik].pt.x, 2) + pow(point.pt.y*ratio - kpts_aux[ik].pt.y, 2));
              if (dist <= point.size) {
                if (point.response > kpts_aux[ik].response) {
                  id_repeated = (int)ik;
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
            left_x = fRound(point.pt.x - smax*sigma_size_) - 1;
            right_x = fRound(point.pt.x + smax*sigma_size_) + 1;
            up_y = fRound(point.pt.y - smax*sigma_size_) - 1;
            down_y = fRound(point.pt.y + smax*sigma_size_) + 1;

            if (left_x < 0 || right_x >= evolution_[i].Ldet.cols ||
                up_y < 0 || down_y >= evolution_[i].Ldet.rows) {
              is_out = true;
            }

            if (is_out == false) {
              if (is_repeated == false) {
                point.pt.x *= ratio;
                point.pt.y *= ratio;
                kpts_aux.push_back(point);
                npoints++;
              }
              else {
                point.pt.x *= ratio;
                point.pt.y *= ratio;
                kpts_aux[id_repeated] = point;
              }
            } // if is_out
          } //if is_extremum
        }
      } // for jx
    } // for ix
  } // for i

  // Now filter points with the upper scale level
  for (size_t i = 0; i < kpts_aux.size(); i++) {

    is_repeated = false;
    const cv::KeyPoint& pt = kpts_aux[i];
    for (size_t j = i + 1; j < kpts_aux.size(); j++) {

      // Compare response with the upper scale
      if ((pt.class_id + 1) == kpts_aux[j].class_id) {
        dist = sqrt(pow(pt.pt.x - kpts_aux[j].pt.x, 2) + pow(pt.pt.y - kpts_aux[j].pt.y, 2));
        if (dist <= pt.size) {
          if (pt.response < kpts_aux[j].response) {
            is_repeated = true;
            break;
          }
        }
      }
    }

    if (is_repeated == false)
      kpts.push_back(pt);
  }
}

/* ************************************************************************* */
/**
 * @brief This method performs subpixel refinement of the detected keypoints
 * @param kpts Vector of detected keypoints
 */
void AKAZEFeatures::Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts)
{
  float Dx = 0.0, Dy = 0.0, ratio = 0.0;
  float Dxx = 0.0, Dyy = 0.0, Dxy = 0.0;
  int x = 0, y = 0;
  cv::Mat A = cv::Mat::zeros(2, 2, CV_32F);
  cv::Mat b = cv::Mat::zeros(2, 1, CV_32F);
  cv::Mat dst = cv::Mat::zeros(2, 1, CV_32F);

  for (size_t i = 0; i < kpts.size(); i++) {
    ratio = pow(2.f, kpts[i].octave);
    x = fRound(kpts[i].pt.x / ratio);
    y = fRound(kpts[i].pt.y / ratio);

    // Compute the gradient
    Dx = (0.5f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x + 1)
        - *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x - 1));
    Dy = (0.5f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x)
        - *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x));

    // Compute the Hessian
    Dxx = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x + 1)
        + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x - 1)
        - 2.0f*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

    Dyy = (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x)
        + *(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x)
        - 2.0f*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y)+x)));

    Dxy = (0.25f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x + 1)
        + (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x - 1)))
        - (0.25f)*(*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y - 1) + x + 1)
        + (*(evolution_[kpts[i].class_id].Ldet.ptr<float>(y + 1) + x - 1)));

    // Solve the linear system
    *(A.ptr<float>(0)) = Dxx;
    *(A.ptr<float>(1) + 1) = Dyy;
    *(A.ptr<float>(0) + 1) = *(A.ptr<float>(1)) = Dxy;
    *(b.ptr<float>(0)) = -Dx;
    *(b.ptr<float>(1)) = -Dy;

    cv::solve(A, b, dst, DECOMP_LU);

    if (fabs(*(dst.ptr<float>(0))) <= 1.0f && fabs(*(dst.ptr<float>(1))) <= 1.0f) {
      kpts[i].pt.x = x + (*(dst.ptr<float>(0)));
      kpts[i].pt.y = y + (*(dst.ptr<float>(1)));
      kpts[i].pt.x *= powf(2.f, (float)evolution_[kpts[i].class_id].octave);
      kpts[i].pt.y *= powf(2.f, (float)evolution_[kpts[i].class_id].octave);
      kpts[i].angle = 0.0;

      // In OpenCV the size of a keypoint its the diameter
      kpts[i].size *= 2.0f;
    }
    // Delete the point since its not stable
    else {
      kpts.erase(kpts.begin() + i);
      i--;
    }
  }
}

/* ************************************************************************* */

class SURF_Descriptor_Upright_64_Invoker : public cv::ParallelLoopBody
{
public:
  SURF_Descriptor_Upright_64_Invoker(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_SURF_Descriptor_Upright_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_SURF_Descriptor_Upright_64(const cv::KeyPoint& kpt, float* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class SURF_Descriptor_64_Invoker : public cv::ParallelLoopBody
{
public:
  SURF_Descriptor_64_Invoker(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator()(const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_SURF_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_SURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class MSURF_Upright_Descriptor_64_Invoker : public cv::ParallelLoopBody
{
public:
  MSURF_Upright_Descriptor_64_Invoker(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator()(const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_MSURF_Upright_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class MSURF_Descriptor_64_Invoker : public cv::ParallelLoopBody
{
public:
  MSURF_Descriptor_64_Invoker(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, std::vector<TEvolution>& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_MSURF_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i));
    }
  }

  void Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
};

class Upright_MLDB_Full_Descriptor_Invoker : public cv::ParallelLoopBody
{
public:
  Upright_MLDB_Full_Descriptor_Invoker(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, std::vector<TEvolution>& evolution, AKAZEOptions& options)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_Upright_MLDB_Full_Descriptor((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_Upright_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;
};

class Upright_MLDB_Descriptor_Subset_Invoker : public cv::ParallelLoopBody
{
public:
  Upright_MLDB_Descriptor_Subset_Invoker(std::vector<cv::KeyPoint>& kpts,
                                         cv::Mat& desc,
                                         std::vector<TEvolution>& evolution,
                                         AKAZEOptions& options,
                                         cv::Mat descriptorSamples,
                                         cv::Mat descriptorBits)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
    , descriptorSamples_(descriptorSamples)
    , descriptorBits_(descriptorBits)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_Upright_MLDB_Descriptor_Subset((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_Upright_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;

  cv::Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  cv::Mat descriptorBits_;
};

class MLDB_Full_Descriptor_Invoker : public cv::ParallelLoopBody
{
public:
  MLDB_Full_Descriptor_Invoker(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, std::vector<TEvolution>& evolution, AKAZEOptions& options)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_MLDB_Full_Descriptor((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;
};

class MLDB_Descriptor_Subset_Invoker : public cv::ParallelLoopBody
{
public:
  MLDB_Descriptor_Subset_Invoker(std::vector<cv::KeyPoint>& kpts,
                                 cv::Mat& desc,
                                 std::vector<TEvolution>& evolution,
                                 AKAZEOptions& options,
                                 cv::Mat descriptorSamples,
                                 cv::Mat descriptorBits)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
    , descriptorSamples_(descriptorSamples)
    , descriptorBits_(descriptorBits)
  {
  }

  void operator() (const Range& range) const
  {
    for (int i = range.start; i < range.end; i++)
    {
      AKAZEFeatures::Compute_Main_Orientation((*keypoints_)[i], *evolution_);
      Get_MLDB_Descriptor_Subset((*keypoints_)[i], descriptors_->ptr<unsigned char>(i));
    }
  }

  void Get_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char* desc) const;

private:
  std::vector<cv::KeyPoint>* keypoints_;
  cv::Mat*                   descriptors_;
  std::vector<TEvolution>*   evolution_;
  AKAZEOptions*              options_;

  cv::Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  cv::Mat descriptorBits_;
};

/**
 * @brief This method  computes the set of descriptors through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 * @param desc Matrix to store the descriptors
 */
void AKAZEFeatures::Compute_Descriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc)
{
  for(size_t i = 0; i < kpts.size(); i++)
  {
      CV_Assert(0 <= kpts[i].class_id && kpts[i].class_id < static_cast<int>(evolution_.size()));
  }

  // Allocate memory for the matrix with the descriptors
  if (options_.descriptor < cv::DESCRIPTOR_MLDB_UPRIGHT) {
    desc = cv::Mat::zeros((int)kpts.size(), 64, CV_32FC1);
  }
  else {
    // We use the full length binary descriptor -> 486 bits
    if (options_.descriptor_size == 0) {
      int t = (6 + 36 + 120)*options_.descriptor_channels;
      desc = cv::Mat::zeros((int)kpts.size(), (int)ceil(t / 8.), CV_8UC1);
    }
    else {
      // We use the random bit selection length binary descriptor
      desc = cv::Mat::zeros((int)kpts.size(), (int)ceil(options_.descriptor_size / 8.), CV_8UC1);
    }
  }

  switch (options_.descriptor)
  {
    case cv::DESCRIPTOR_KAZE_UPRIGHT: // Upright descriptors, not invariant to rotation
    {
      cv::parallel_for_(cv::Range(0, (int)kpts.size()), MSURF_Upright_Descriptor_64_Invoker(kpts, desc, evolution_));
    }
    break;
    case cv::DESCRIPTOR_KAZE:
    {
      cv::parallel_for_(cv::Range(0, (int)kpts.size()), MSURF_Descriptor_64_Invoker(kpts, desc, evolution_));
    }
    break;
    case cv::DESCRIPTOR_MLDB_UPRIGHT: // Upright descriptors, not invariant to rotation
    {
      if (options_.descriptor_size == 0)
        cv::parallel_for_(cv::Range(0, (int)kpts.size()), Upright_MLDB_Full_Descriptor_Invoker(kpts, desc, evolution_, options_));
      else
        cv::parallel_for_(cv::Range(0, (int)kpts.size()), Upright_MLDB_Descriptor_Subset_Invoker(kpts, desc, evolution_, options_, descriptorSamples_, descriptorBits_));
    }
    break;
    case cv::DESCRIPTOR_MLDB:
    {
      if (options_.descriptor_size == 0)
        cv::parallel_for_(cv::Range(0, (int)kpts.size()), MLDB_Full_Descriptor_Invoker(kpts, desc, evolution_, options_));
      else
        cv::parallel_for_(cv::Range(0, (int)kpts.size()), MLDB_Descriptor_Subset_Invoker(kpts, desc, evolution_, options_, descriptorSamples_, descriptorBits_));
    }
    break;
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the main orientation for a given keypoint
 * @param kpt Input keypoint
 * @note The orientation is computed using a similar approach as described in the
 * original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
 */
void AKAZEFeatures::Compute_Main_Orientation(cv::KeyPoint& kpt, const std::vector<TEvolution>& evolution_) {

  int ix = 0, iy = 0, idx = 0, s = 0, level = 0;
  float xf = 0.0, yf = 0.0, gweight = 0.0, ratio = 0.0;
  std::vector<float> resX(109), resY(109), Ang(109);
  const int id[] = { 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6 };

  // Variables for computing the dominant direction
  float sumX = 0.0, sumY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;

  // Get the information from the keypoint
  level = kpt.class_id;
  ratio = (float)(1 << evolution_[level].octave);
  s = fRound(0.5f*kpt.size / ratio);
  xf = kpt.pt.x / ratio;
  yf = kpt.pt.y / ratio;

  // Calculate derivatives responses for points within radius of 6*scale
  for (int i = -6; i <= 6; ++i) {
    for (int j = -6; j <= 6; ++j) {
      if (i*i + j*j < 36) {
        iy = fRound(yf + j*s);
        ix = fRound(xf + i*s);

        gweight = gauss25[id[i + 6]][id[j + 6]];
        resX[idx] = gweight*(*(evolution_[level].Lx.ptr<float>(iy)+ix));
        resY[idx] = gweight*(*(evolution_[level].Ly.ptr<float>(iy)+ix));

        Ang[idx] = getAngle(resX[idx], resY[idx]);
        ++idx;
      }
    }
  }
  // Loop slides pi/3 window around feature point
  for (ang1 = 0; ang1 < (float)(2.0 * CV_PI); ang1 += 0.15f) {
    ang2 = (ang1 + (float)(CV_PI / 3.0) >(float)(2.0*CV_PI) ? ang1 - (float)(5.0*CV_PI / 3.0) : ang1 + (float)(CV_PI / 3.0));
    sumX = sumY = 0.f;

    for (size_t k = 0; k < Ang.size(); ++k) {
      // Get angle from the x-axis of the sample point
      const float & ang = Ang[k];

      // Determine whether the point is within the window
      if (ang1 < ang2 && ang1 < ang && ang < ang2) {
        sumX += resX[k];
        sumY += resY[k];
      }
      else if (ang2 < ang1 &&
               ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2.0f*CV_PI))) {
        sumX += resX[k];
        sumY += resY[k];
      }
    }

    // if the vector produced from this window is longer than all
    // previous vectors then this forms the new dominant direction
    if (sumX*sumX + sumY*sumY > max) {
      // store largest orientation
      max = sumX*sumX + sumY*sumY;
      kpt.angle = getAngle(sumX, sumY);
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void MSURF_Upright_Descriptor_64_Invoker::Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float *desc) const {

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int scale = 0, dsize = 0, level = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5f, cy = 0.5f;

  const std::vector<TEvolution>& evolution = *evolution_;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 64;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
  scale = fRound(0.5f*kpt.size / ratio);
  level = kpt.class_id;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while (i < pattern_size) {
    j = -8;
    i = i - 4;

    cx += 1.0f;
    cy = -0.5f;

    while (j < pattern_size) {
      dx = dy = mdx = mdy = 0.0;
      cy += 1.0f;
      j = j - 4;

      ky = i + sample_step;
      kx = j + sample_step;

      ys = yf + (ky*scale);
      xs = xf + (kx*scale);

      for (int k = i; k < i + 9; k++) {
        for (int l = j; l < j + 9; l++) {
          sample_y = k*scale + yf;
          sample_x = l*scale + xf;

          //Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.50f*scale);

          y1 = (int)(sample_y - .5);
          x1 = (int)(sample_x - .5);

          y2 = (int)(sample_y + .5);
          x2 = (int)(sample_x + .5);

          fx = sample_x - x1;
          fy = sample_y - y1;

          res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

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
      gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

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

  for (i = 0; i < dsize; i++) {
    desc[i] /= len;
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void MSURF_Descriptor_64_Invoker::Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc) const {

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
  int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  int scale = 0, dsize = 0, level = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5f, cy = 0.5f;

  const std::vector<TEvolution>& evolution = *evolution_;

  // Set the descriptor size and the sample and pattern sizes
  dsize = 64;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
  scale = fRound(0.5f*kpt.size / ratio);
  angle = kpt.angle;
  level = kpt.class_id;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;
  co = cos(angle);
  si = sin(angle);

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while (i < pattern_size) {
    j = -8;
    i = i - 4;

    cx += 1.0f;
    cy = -0.5f;

    while (j < pattern_size) {
      dx = dy = mdx = mdy = 0.0;
      cy += 1.0f;
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
          gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);

          y1 = fRound(sample_y - 0.5f);
          x1 = fRound(sample_x - 0.5f);

          y2 = fRound(sample_y + 0.5f);
          x2 = fRound(sample_x + 0.5f);

          fx = sample_x - x1;
          fy = sample_y - y1;

          res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
          rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
          res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
          res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
          res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
          ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

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
      gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);
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

  for (i = 0; i < dsize; i++) {
    desc[i] /= len;
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the rupright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void Upright_MLDB_Full_Descriptor_Invoker::Get_Upright_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char *desc) const {

  float di = 0.0, dx = 0.0, dy = 0.0;
  float ri = 0.0, rx = 0.0, ry = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, ratio = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int level = 0, nsamples = 0, scale = 0;
  int dcount1 = 0, dcount2 = 0;

  const AKAZEOptions & options = *options_;
  const std::vector<TEvolution>& evolution = *evolution_;

  // Matrices for the M-LDB descriptor
  cv::Mat values_1 = cv::Mat::zeros(4, options.descriptor_channels, CV_32FC1);
  cv::Mat values_2 = cv::Mat::zeros(9, options.descriptor_channels, CV_32FC1);
  cv::Mat values_3 = cv::Mat::zeros(16, options.descriptor_channels, CV_32FC1);

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
  scale = fRound(0.5f*kpt.size / ratio);
  level = kpt.class_id;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;

  // First 2x2 grid
  pattern_size = options_->descriptor_pattern_size;
  sample_step = pattern_size;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {
      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

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
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      if (*(values_1.ptr<float>(i)) > *(values_1.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_1.ptr<float>(i)+1) > *(values_1.ptr<float>(j)+1)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_1.ptr<float>(i)+2) > *(values_1.ptr<float>(j)+2)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }

  // Second 3x3 grid
  sample_step = static_cast<int>(ceil(pattern_size*2. / 3.));
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {
      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

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

  //Do binary comparison second level
  dcount2 = 0;
  for (int i = 0; i < 9; i++) {
    for (int j = i + 1; j < 9; j++) {
      if (*(values_2.ptr<float>(i)) > *(values_2.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_2.ptr<float>(i)+1) > *(values_2.ptr<float>(j)+1)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_2.ptr<float>(i)+2) > *(values_2.ptr<float>(j)+2)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }

  // Third 4x4 grid
  sample_step = pattern_size / 2;
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {
      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + l*scale;
          sample_x = xf + k*scale;

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

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

  //Do binary comparison third level
  dcount2 = 0;
  for (int i = 0; i < 16; i++) {
    for (int j = i + 1; j < 16; j++) {
      if (*(values_3.ptr<float>(i)) > *(values_3.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_3.ptr<float>(i)+1) > *(values_3.ptr<float>(j)+1)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;

      if (*(values_3.ptr<float>(i)+2) > *(values_3.ptr<float>(j)+2)) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void MLDB_Full_Descriptor_Invoker::Get_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char *desc) const {

  float di = 0.0, dx = 0.0, dy = 0.0, ratio = 0.0;
  float ri = 0.0, rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int level = 0, nsamples = 0, scale = 0;
  int dcount1 = 0, dcount2 = 0;

  const AKAZEOptions & options = *options_;
  const std::vector<TEvolution>& evolution = *evolution_;

  // Matrices for the M-LDB descriptor
  cv::Mat values_1 = cv::Mat::zeros(4, options.descriptor_channels, CV_32FC1);
  cv::Mat values_2 = cv::Mat::zeros(9, options.descriptor_channels, CV_32FC1);
  cv::Mat values_3 = cv::Mat::zeros(16, options.descriptor_channels, CV_32FC1);

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
  scale = fRound(0.5f*kpt.size / ratio);
  angle = kpt.angle;
  level = kpt.class_id;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;
  co = cos(angle);
  si = sin(angle);

  // First 2x2 grid
  pattern_size = options.descriptor_pattern_size;
  sample_step = pattern_size;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {

      di = dx = dy = 0.0;
      nsamples = 0;

      for (float k = (float)i; k < i + sample_step; k++) {
        for (float l = (float)j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          di += ri;

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
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
      if (options.descriptor_channels > 1) {
        *(values_1.ptr<float>(dcount2)+1) = dx;
      }

      if (options.descriptor_channels > 2) {
        *(values_1.ptr<float>(dcount2)+2) = dy;
      }

      dcount2++;
    }
  }

  // Do binary comparison first level
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      if (*(values_1.ptr<float>(i)) > *(values_1.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }

  if (options.descriptor_channels > 1) {
    for (int i = 0; i < 4; i++) {
      for (int j = i + 1; j < 4; j++) {
        if (*(values_1.ptr<float>(i)+1) > *(values_1.ptr<float>(j)+1)) {
          desc[dcount1 / 8] |= (1 << (dcount1 % 8));
        }

        dcount1++;
      }
    }
  }

  if (options.descriptor_channels > 2) {
    for (int i = 0; i < 4; i++) {
      for (int j = i + 1; j < 4; j++) {
        if (*(values_1.ptr<float>(i)+2) > *(values_1.ptr<float>(j)+2)) {
          desc[dcount1 / 8] |= (1 << (dcount1 % 8));
        }
        dcount1++;
      }
    }
  }

  // Second 3x3 grid
  sample_step = static_cast<int>(ceil(pattern_size*2. / 3.));
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {

      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);
          di += ri;

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
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
      if (options.descriptor_channels > 1) {
        *(values_2.ptr<float>(dcount2)+1) = dx;
      }

      if (options.descriptor_channels > 2) {
        *(values_2.ptr<float>(dcount2)+2) = dy;
      }

      dcount2++;
    }
  }

  // Do binary comparison second level
  for (int i = 0; i < 9; i++) {
    for (int j = i + 1; j < 9; j++) {
      if (*(values_2.ptr<float>(i)) > *(values_2.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }

  if (options.descriptor_channels > 1) {
    for (int i = 0; i < 9; i++) {
      for (int j = i + 1; j < 9; j++) {
        if (*(values_2.ptr<float>(i)+1) > *(values_2.ptr<float>(j)+1)) {
          desc[dcount1 / 8] |= (1 << (dcount1 % 8));
        }
        dcount1++;
      }
    }
  }

  if (options.descriptor_channels > 2) {
    for (int i = 0; i < 9; i++) {
      for (int j = i + 1; j < 9; j++) {
        if (*(values_2.ptr<float>(i)+2) > *(values_2.ptr<float>(j)+2)) {
          desc[dcount1 / 8] |= (1 << (dcount1 % 8));
        }
        dcount1++;
      }
    }
  }

  // Third 4x4 grid
  sample_step = pattern_size / 2;
  dcount2 = 0;

  for (int i = -pattern_size; i < pattern_size; i += sample_step) {
    for (int j = -pattern_size; j < pattern_size; j += sample_step) {
      di = dx = dy = 0.0;
      nsamples = 0;

      for (int k = i; k < i + sample_step; k++) {
        for (int l = j; l < j + sample_step; l++) {

          // Get the coordinates of the sample point
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          y1 = fRound(sample_y);
          x1 = fRound(sample_x);

          ri = *(evolution[level].Lt.ptr<float>(y1)+x1);
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);
          di += ri;

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
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
      if (options.descriptor_channels > 1)
        *(values_3.ptr<float>(dcount2)+1) = dx;

      if (options.descriptor_channels > 2)
        *(values_3.ptr<float>(dcount2)+2) = dy;

      dcount2++;
    }
  }

  // Do binary comparison third level
  for (int i = 0; i < 16; i++) {
    for (int j = i + 1; j < 16; j++) {
      if (*(values_3.ptr<float>(i)) > *(values_3.ptr<float>(j))) {
        desc[dcount1 / 8] |= (1 << (dcount1 % 8));
      }
      dcount1++;
    }
  }

  if (options.descriptor_channels > 1) {
    for (int i = 0; i < 16; i++) {
      for (int j = i + 1; j < 16; j++) {
        if (*(values_3.ptr<float>(i)+1) > *(values_3.ptr<float>(j)+1)) {
          desc[dcount1 / 8] |= (1 << (dcount1 % 8));
        }
        dcount1++;
      }
    }
  }

  if (options.descriptor_channels > 2) {
    for (int i = 0; i < 16; i++) {
      for (int j = i + 1; j < 16; j++) {
        if (*(values_3.ptr<float>(i)+2) > *(values_3.ptr<float>(j)+2)) {
          desc[dcount1 / 8] |= (1 << (dcount1 % 8));
        }
        dcount1++;
      }
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the M-LDB descriptor of the provided keypoint given the
 * main orientation of the keypoint. The descriptor is computed based on a subset of
 * the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void MLDB_Descriptor_Subset_Invoker::Get_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char *desc) const {

  float di = 0.f, dx = 0.f, dy = 0.f;
  float rx = 0.f, ry = 0.f;
  float sample_x = 0.f, sample_y = 0.f;
  int x1 = 0, y1 = 0;

  const AKAZEOptions & options = *options_;
  const std::vector<TEvolution>& evolution = *evolution_;

  // Get the information from the keypoint
  float ratio = (float)(1 << kpt.octave);
  int scale = fRound(0.5f*kpt.size / ratio);
  float angle = kpt.angle;
  int level = kpt.class_id;
  float yf = kpt.pt.y / ratio;
  float xf = kpt.pt.x / ratio;
  float co = cos(angle);
  float si = sin(angle);

  // Allocate memory for the matrix of values
  cv::Mat values = cv::Mat_<float>::zeros((4 + 9 + 16)*options.descriptor_channels, 1);

  // Sample everything, but only do the comparisons
  vector<int> steps(3);
  steps.at(0) = options.descriptor_pattern_size;
  steps.at(1) = (int)ceil(2.f*options.descriptor_pattern_size / 3.f);
  steps.at(2) = options.descriptor_pattern_size / 2;

  for (int i = 0; i < descriptorSamples_.rows; i++) {
    const int *coords = descriptorSamples_.ptr<int>(i);
    int sample_step = steps.at(coords[0]);
    di = 0.0f;
    dx = 0.0f;
    dy = 0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {

        // Get the coordinates of the sample point
        sample_y = yf + (l*scale*co + k*scale*si);
        sample_x = xf + (-l*scale*si + k*scale*co);

        y1 = fRound(sample_y);
        x1 = fRound(sample_x);

        di += *(evolution[level].Lt.ptr<float>(y1)+x1);

        if (options.descriptor_channels > 1) {
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
            // Get the x and y derivatives on the rotated axis
            dx += rx*co + ry*si;
            dy += -rx*si + ry*co;
          }
        }
      }
    }

    *(values.ptr<float>(options.descriptor_channels*i)) = di;

    if (options.descriptor_channels == 2) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
    }
    else if (options.descriptor_channels == 3) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
      *(values.ptr<float>(options.descriptor_channels*i + 2)) = dy;
    }
  }

  // Do the comparisons
  const float *vals = values.ptr<float>(0);
  const int *comps = descriptorBits_.ptr<int>(0);

  for (int i = 0; i<descriptorBits_.rows; i++) {
    if (vals[comps[2 * i]] > vals[comps[2 * i + 1]]) {
      desc[i / 8] |= (1 << (i % 8));
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the upright (not rotation invariant) M-LDB descriptor
 * of the provided keypoint given the main orientation of the keypoint.
 * The descriptor is computed based on a subset of the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void Upright_MLDB_Descriptor_Subset_Invoker::Get_Upright_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char *desc) const {

  float di = 0.0f, dx = 0.0f, dy = 0.0f;
  float rx = 0.0f, ry = 0.0f;
  float sample_x = 0.0f, sample_y = 0.0f;
  int x1 = 0, y1 = 0;

  const AKAZEOptions & options = *options_;
  const std::vector<TEvolution>& evolution = *evolution_;

  // Get the information from the keypoint
  float ratio = (float)(1 << kpt.octave);
  int scale = fRound(0.5f*kpt.size / ratio);
  int level = kpt.class_id;
  float yf = kpt.pt.y / ratio;
  float xf = kpt.pt.x / ratio;

  // Allocate memory for the matrix of values
  Mat values = cv::Mat_<float>::zeros((4 + 9 + 16)*options.descriptor_channels, 1);

  vector<int> steps(3);
  steps.at(0) = options.descriptor_pattern_size;
  steps.at(1) = static_cast<int>(ceil(2.f*options.descriptor_pattern_size / 3.f));
  steps.at(2) = options.descriptor_pattern_size / 2;

  for (int i = 0; i < descriptorSamples_.rows; i++) {
    const int *coords = descriptorSamples_.ptr<int>(i);
    int sample_step = steps.at(coords[0]);
    di = 0.0f, dx = 0.0f, dy = 0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {

        // Get the coordinates of the sample point
        sample_y = yf + l*scale;
        sample_x = xf + k*scale;

        y1 = fRound(sample_y);
        x1 = fRound(sample_x);
        di += *(evolution[level].Lt.ptr<float>(y1)+x1);

        if (options.descriptor_channels > 1) {
          rx = *(evolution[level].Lx.ptr<float>(y1)+x1);
          ry = *(evolution[level].Ly.ptr<float>(y1)+x1);

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
            dx += rx;
            dy += ry;
          }
        }
      }
    }

    *(values.ptr<float>(options.descriptor_channels*i)) = di;

    if (options.descriptor_channels == 2) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
    }
    else if (options.descriptor_channels == 3) {
      *(values.ptr<float>(options.descriptor_channels*i + 1)) = dx;
      *(values.ptr<float>(options.descriptor_channels*i + 2)) = dy;
    }
  }

  // Do the comparisons
  const float *vals = values.ptr<float>(0);
  const int *comps = descriptorBits_.ptr<int>(0);

  for (int i = 0; i<descriptorBits_.rows; i++) {
    if (vals[comps[2 * i]] > vals[comps[2 * i + 1]]) {
      desc[i / 8] |= (1 << (i % 8));
    }
  }
}

/* ************************************************************************* */
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
  for (int i = 0; i < 3; i++) {
    int gz = (i + 2)*(i + 2);
    ssz += gz*(gz - 1) / 2;
  }
  ssz *= nchannels;

  CV_Assert(nbits <= ssz); // Descriptor size can't be bigger than full descriptor

  // Since the full descriptor is usually under 10k elements, we pick
  // the selection from the full matrix.  We take as many samples per
  // pick as the number of channels. For every pick, we
  // take the two samples involved and put them in the sampling list

  Mat_<int> fullM(ssz / nchannels, 5);
  for (int i = 0, c = 0; i < 3; i++) {
    int gdiv = i + 2; //grid divisions, per row
    int gsz = gdiv*gdiv;
    int psz = (int)ceil(2.f*pattern_size / (float)gdiv);

    for (int j = 0; j < gsz; j++) {
      for (int k = j + 1; k < gsz; k++, c++) {
        fullM(c, 0) = i;
        fullM(c, 1) = psz*(j % gdiv) - pattern_size;
        fullM(c, 2) = psz*(j / gdiv) - pattern_size;
        fullM(c, 3) = psz*(k % gdiv) - pattern_size;
        fullM(c, 4) = psz*(k / gdiv) - pattern_size;
      }
    }
  }

  srand(1024);
  Mat_<int> comps = Mat_<int>(nchannels * (int)ceil(nbits / (float)nchannels), 2);
  comps = 1000;

  // Select some samples. A sample includes all channels
  int count = 0;
  int npicks = (int)ceil(nbits / (float)nchannels);
  Mat_<int> samples(29, 3);
  Mat_<int> fullcopy = fullM.clone();
  samples = -1;

  for (int i = 0; i < npicks; i++) {
    int k = rand() % (fullM.rows - i);
    if (i < 6) {
      // Force use of the coarser grid values and comparisons
      k = i;
    }

    bool n = true;

    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 1) && samples(j, 2) == fullcopy(k, 2)) {
        n = false;
        comps(i*nchannels, 0) = nchannels*j;
        comps(i*nchannels + 1, 0) = nchannels*j + 1;
        comps(i*nchannels + 2, 0) = nchannels*j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 1);
      samples(count, 2) = fullcopy(k, 2);
      comps(i*nchannels, 0) = nchannels*count;
      comps(i*nchannels + 1, 0) = nchannels*count + 1;
      comps(i*nchannels + 2, 0) = nchannels*count + 2;
      count++;
    }

    n = true;
    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 3) && samples(j, 2) == fullcopy(k, 4)) {
        n = false;
        comps(i*nchannels, 1) = nchannels*j;
        comps(i*nchannels + 1, 1) = nchannels*j + 1;
        comps(i*nchannels + 2, 1) = nchannels*j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 3);
      samples(count, 2) = fullcopy(k, 4);
      comps(i*nchannels, 1) = nchannels*count;
      comps(i*nchannels + 1, 1) = nchannels*count + 1;
      comps(i*nchannels + 2, 1) = nchannels*count + 2;
      count++;
    }

    Mat tmp = fullcopy.row(k);
    fullcopy.row(fullcopy.rows - i - 1).copyTo(tmp);
  }

  sampleList = samples.rowRange(0, count).clone();
  comparisons = comps.rowRange(0, nbits).clone();
}
