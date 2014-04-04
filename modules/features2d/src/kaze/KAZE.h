
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
#include "config.h"
#include "nldiffusion_functions.h"
#include "fed.h"
#include "utils.h"

//*************************************************************************************
//*************************************************************************************

// KAZE Class Declaration
class KAZE {

private:

  // Parameters of the Nonlinear diffusion class
  float soffset_;	// Base scale offset
  float sderivatives_; // Standard deviation of the Gaussian for the nonlinear diff. derivatives
  int omax_;		// Maximum octave level
  int nsublevels_;	// Number of sublevels per octave level
  int img_width_;	// Width of the original image
  int img_height_; // Height of the original image
  bool save_scale_space_; // For saving scale space images
  bool verbosity_;	// Verbosity level
  std::vector<TEvolution> evolution_;	// Vector of nonlinear diffusion evolution
  float kcontrast_; // The contrast parameter for the scalar nonlinear diffusion
  float dthreshold_;	// Feature detector threshold response
  int diffusivity_; 	// Diffusivity type, 0->PM G1, 1->PM G2, 2-> Weickert
  int descriptor_mode_; // Descriptor mode
  bool use_fed_;        // Set to true in case we want to use FED for the nonlinear diffusion filtering. Set false for using AOS
  bool use_upright_;	// Set to true in case we want to use the upright version of the descriptors
  bool use_extended_;	// Set to true in case we want to use the extended version of the descriptors

  // Vector of keypoint vectors for finding extrema in multiple threads
  std::vector<std::vector<cv::KeyPoint> > kpts_par_;

  // FED parameters
  int ncycles_;                  // Number of cycles
  bool reordering_;              // Flag for reordering time steps
  std::vector<std::vector<float > > tsteps_;  // Vector of FED dynamic time steps
  std::vector<int> nsteps_;      // Vector of number of steps per cycle

  // Computation times variables in ms
  double tkcontrast_;       // Kcontrast factor computation
  double tnlscale_;         // Nonlinear Scale space generation
  double tdetector_;        // Feature detector
  double tmderivatives_;    // Multiscale derivatives computation
  double tdresponse_;       // Detector response computation
  double tdescriptor_;      // Feature descriptor
  double tsubpixel_;        // Subpixel refinement

  // Some auxiliary variables used in the AOS step
  cv::Mat Ltx_, Lty_, px_, py_, ax_, ay_, bx_, by_, qr_, qc_;

public:

  // Constructor
  KAZE(KAZEOptions& options);

  // Destructor
  ~KAZE(void);

  // Public methods for KAZE interface
  void Allocate_Memory_Evolution(void);
  int Create_Nonlinear_Scale_Space(const cv::Mat& img);
  void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
  void Feature_Description(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

  // Methods for saving the scale space set of images and detector responses
  void Save_Nonlinear_Scale_Space(void);
  void Save_Detector_Responses(void);
  void Save_Flow_Responses(void);

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

public:

  // Setters
  void Set_Scale_Offset(float soffset) {
    soffset_ = soffset;
  }

  void Set_SDerivatives(float sderivatives) {
    sderivatives_ = sderivatives;
  }

  void Set_Octave_Max(int omax) {
    omax_ = omax;
  }

  void Set_NSublevels(int nsublevels) {
    nsublevels_ = nsublevels;
  }

  void Set_Save_Scale_Space_Flag(bool save_scale_space) {
    save_scale_space_ = save_scale_space;
  }

  void Set_Image_Width(int img_width) {
    img_width_ = img_width;
  }

  void Set_Image_Height(int img_height) {
    img_height_ = img_height;
  }

  void Set_Verbosity_Level(bool verbosity) {
    verbosity_ = verbosity;
  }

  void Set_KContrast(float kcontrast) {
    kcontrast_ = kcontrast;
  }

  void Set_Detector_Threshold(float dthreshold) {
    dthreshold_ = dthreshold;
  }

  void Set_Diffusivity_Type(int diffusivity) {
    diffusivity_ = diffusivity;
  }

  void Set_Descriptor_Mode(int descriptor_mode) {
    descriptor_mode_ = descriptor_mode;
  }

  void Set_Use_FED(bool use_fed) {
    use_fed_ = use_fed;
  }

  void Set_Upright(bool use_upright) {
    use_upright_ = use_upright;
  }

  void Set_Extended(bool use_extended) {
    use_extended_ = use_extended;
  }

  // Getters
  float Get_Scale_Offset(void) {
    return soffset_;
  }

  float Get_SDerivatives(void) {
    return sderivatives_;
  }

  int Get_Octave_Max(void) {
    return omax_;
  }

  int Get_NSublevels(void) {
    return nsublevels_;
  }

  bool Get_Save_Scale_Space_Flag(void) {
    return save_scale_space_;
  }

  int Get_Image_Width(void) {
    return img_width_;
  }

  int Get_Image_Height(void) {
    return img_height_;
  }

  bool Get_Verbosity_Level(void) {
    return verbosity_;
  }

  float Get_KContrast(void) {
    return kcontrast_;
  }

  float Get_Detector_Threshold(void) {
    return dthreshold_;
  }

  int Get_Diffusivity_Type(void) {
    return diffusivity_;
  }

  int Get_Descriptor_Mode(void) {
    return descriptor_mode_;
  }

  bool Get_Upright(void) {
    return use_upright_;
  }

  bool Get_Extended(void) {
    return use_extended_;
  }

  float Get_Time_KContrast(void) {
    return tkcontrast_;
  }

  float Get_Time_NLScale(void) {
    return tnlscale_;
  }

  float Get_Time_Detector(void) {
    return tdetector_;
  }

  float Get_Time_Multiscale_Derivatives(void) {
    return tmderivatives_;
  }

  float Get_Time_Detector_Response(void) {
    return tdresponse_;
  }

  float Get_Time_Descriptor(void) {
    return tdescriptor_;
  }

  float Get_Time_Subpixel(void) {
    return tsubpixel_;
  }
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
