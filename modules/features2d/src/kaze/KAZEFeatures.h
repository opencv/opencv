
/**
 * @file KAZE.h
 * @brief Main program for detecting and computing descriptors in a nonlinear
 * scale space
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 */

#ifndef KAZE_H_
#define KAZE_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include "KAZEConfig.h"
#include "nldiffusion_functions.h"
#include "fed.h"

//*************************************************************************************
//*************************************************************************************

// KAZE Class Declaration
class KAZEFeatures {

private:

  // Parameters of the Nonlinear diffusion class
  float soffset_;	// Base scale offset
  float sderivatives_; // Standard deviation of the Gaussian for the nonlinear diff. derivatives
  int omax_;		// Maximum octave level
  int nsublevels_;	// Number of sublevels per octave level
  int img_width_;	// Width of the original image
  int img_height_; // Height of the original image
  std::vector<TEvolution> evolution_;	// Vector of nonlinear diffusion evolution
  float kcontrast_; // The contrast parameter for the scalar nonlinear diffusion
  float dthreshold_;	// Feature detector threshold response
  int diffusivity_; 	// Diffusivity type, 0->PM G1, 1->PM G2, 2-> Weickert
  int descriptor_mode_; // Descriptor mode
  bool use_fed_;        // Set to true in case we want to use FED for the nonlinear diffusion filtering. Set false for using AOS
  bool use_upright_;	// Set to true in case we want to use the upright version of the descriptors
  bool use_extended_;	// Set to true in case we want to use the extended version of the descriptors
  bool use_normalization;

  // Vector of keypoint vectors for finding extrema in multiple threads
  std::vector<std::vector<cv::KeyPoint> > kpts_par_;

  // FED parameters
  int ncycles_;                  // Number of cycles
  bool reordering_;              // Flag for reordering time steps
  std::vector<std::vector<float > > tsteps_;  // Vector of FED dynamic time steps
  std::vector<int> nsteps_;      // Vector of number of steps per cycle

  // Computation times variables in ms
  //double tkcontrast_;       // Kcontrast factor computation
  //double tnlscale_;         // Nonlinear Scale space generation
  //double tdetector_;        // Feature detector
  //double tmderivatives_;    // Multiscale derivatives computation
  //double tdresponse_;       // Detector response computation
  //double tdescriptor_;      // Feature descriptor
  //double tsubpixel_;        // Subpixel refinement

  // Some auxiliary variables used in the AOS step
  cv::Mat Ltx_, Lty_, px_, py_, ax_, ay_, bx_, by_, qr_, qc_;

public:

  // Constructor
  KAZEFeatures(KAZEOptions& options);

  // Public methods for KAZE interface
  void Allocate_Memory_Evolution(void);
  int Create_Nonlinear_Scale_Space(const cv::Mat& img);
  void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
  void Feature_Description(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

private:

  // Feature Detection Methods
  void Compute_KContrast(const cv::Mat& img, const float& kper);
  void Compute_Multiscale_Derivatives(void);
  void Compute_Detector_Response(void);
  void Determinant_Hessian_Parallel(std::vector<cv::KeyPoint>& kpts);
  void Find_Extremum_Threading(const int& level);
  void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);
  void Feature_Suppression_Distance(std::vector<cv::KeyPoint>& kpts, const float& mdist);

  // AOS Methods
  void AOS_Step_Scalar(cv::Mat &Ld, const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
  void AOS_Rows(const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
  void AOS_Columns(const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
  void Thomas(const cv::Mat &a, const cv::Mat &b, const cv::Mat &Ld, cv::Mat &x);

  // Feature Description methods
  void Compute_Main_Orientation_SURF(cv::KeyPoint& kpt);

  // Descriptor Mode -> 0 SURF 64
  void Get_SURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc);
  void Get_SURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc);

  // Descriptor Mode -> 0 SURF 128
  void Get_SURF_Upright_Descriptor_128(const cv::KeyPoint& kpt, float* desc);
  void Get_SURF_Descriptor_128(const cv::KeyPoint& kpt, float* desc);

  // Descriptor Mode -> 1 M-SURF 64
  void Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc);
  void Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc);

  // Descriptor Mode -> 1 M-SURF 128
  void Get_MSURF_Upright_Descriptor_128(const cv::KeyPoint& kpt, float* desc);
  void Get_MSURF_Descriptor_128(const cv::KeyPoint& kpt, float *desc);

  // Descriptor Mode -> 2 G-SURF 64
  void Get_GSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc);
  void Get_GSURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc);

  // Descriptor Mode -> 2 G-SURF 128
  void Get_GSURF_Upright_Descriptor_128(const cv::KeyPoint& kpt, float* desc);
  void Get_GSURF_Descriptor_128(const cv::KeyPoint& kpt, float* desc);
};

//*************************************************************************************
//*************************************************************************************

// Inline functions
float getAngle(const float& x, const float& y);
float gaussian(const float& x, const float& y, const float& sig);
void checkDescriptorLimits(int &x, int &y, const int& width, const int& height);
void clippingDescriptor(float *desc, const int& dsize, const int& niter, const float& ratio);
int fRound(const float& flt);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_H_
