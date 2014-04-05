/**
 * @file AKAZE.h
 * @brief Main class for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Mar 27, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#ifndef _AKAZE_H_
#define _AKAZE_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include "config.h"
#include "fed.h"
#include "utils.h"
#include "nldiffusion_functions.h"

//*************************************************************************************
//*************************************************************************************

// AKAZE Class Declaration
class AKAZEFeatures {

private:

  // Parameters of the AKAZE class
  int omax_;               // Maximum octave level
  int noctaves_;           // Number of octaves
  int nsublevels_;         // Number of sublevels per octave level
  int img_width_;          // Width of the original image
  int img_height_;         // Height of the original image
  float soffset_;          // Base scale offset
  float factor_size_;      // Factor for the multiscale derivatives
  float sderivatives_;     // Standard deviation of the Gaussian for the nonlinear diff. derivatives
  float kcontrast_;        // The contrast parameter for the scalar nonlinear diffusion
  float dthreshold_;       // Feature detector threshold response
  int diffusivity_;        // Diffusivity type, 0->PM G1, 1->PM G2, 2-> Weickert, 3->Charbonnier
  int descriptor_;         // Descriptor mode:
                           // 0-> SURF_UPRIGHT, 1->SURF
                           // 2-> M-SURF_UPRIGHT, 3->M-SURF
                           // 4-> M-LDB_UPRIGHT, 5->M-LDB
  int descriptor_size_;    // Size of the descriptor in bits. Use 0 for the full descriptor
  int descriptor_pattern_size_;    // Size of the pattern. Actual size sampled is 2*pattern_size
  int descriptor_channels_;        // Number of channels to consider in the M-LDB descriptor
  bool save_scale_space_; // For saving scale space images
  bool verbosity_;	// Verbosity level
  std::vector<tevolution> evolution_;	// Vector of nonlinear diffusion evolution

  // FED parameters
  int ncycles_;                  // Number of cycles
  bool reordering_;              // Flag for reordering time steps
  std::vector<std::vector<float > > tsteps_;  // Vector of FED dynamic time steps
  std::vector<int> nsteps_;      // Vector of number of steps per cycle

  // Some matrices for the M-LDB descriptor computation
  cv::Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  cv::Mat descriptorBits_;
  cv::Mat bitMask_;

  // Computation times variables in ms
  double tkcontrast_;      // Kcontrast factor computation
  double tscale_;          // Nonlinear Scale space generation
  double tderivatives_;    // Multiscale derivatives
  double tdetector_;       // Feature detector
  double textrema_;        // Scale Space extrema
  double tsubpixel_;       // Subpixel refinement
  double tdescriptor_;     // Feature descriptors

public:

  // Constructor
  AKAZEFeatures(const AKAZEOptions &options);

  // Destructor
  ~AKAZEFeatures(void);

  // Setters
  void Set_Octave_Max(const int& omax) {
    omax_ = omax;
  }
  void Set_NSublevels(const int& nsublevels) {
    nsublevels_ = nsublevels;
  }
  void Set_Save_Scale_Space_Flag(const bool& save_scale_space) {
    save_scale_space_ = save_scale_space;
  }
  void Set_Image_Width(const int& img_width) {
    img_width_ = img_width;
  }
  void Set_Image_Height(const int& img_height) {
    img_height_ = img_height;
  }

  // Getters
  int Get_Image_Width(void) {
    return img_width_;
  }
  int Get_Image_Height(void) {
    return img_height_;
  }
  double Get_Time_KContrast(void) {
    return tkcontrast_;
  }
  double Get_Time_Scale_Space(void) {
    return tscale_;
  }
  double Get_Time_Derivatives(void) {
    return tderivatives_;
  }
  double Get_Time_Detector(void) {
    return tdetector_;
  }
  double Get_Time_Descriptor(void) {
    return tdescriptor_;
  }

  // Scale Space methods
  void Allocate_Memory_Evolution(void);
  int Create_Nonlinear_Scale_Space(const cv::Mat& img);
  void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
  void Compute_Determinant_Hessian_Response(void);
  void Compute_Multiscale_Derivatives(void);
  void Find_Scale_Space_Extrema(std::vector<cv::KeyPoint>& kpts);
  void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);
  void Feature_Suppression_Distance(std::vector<cv::KeyPoint>& kpts, float mdist);

  // Feature description methods
  void Compute_Descriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);
  void Compute_Main_Orientation_SURF(cv::KeyPoint& kpt);

  // SURF Pattern Descriptor
  void Get_SURF_Descriptor_Upright_64(const cv::KeyPoint& kpt, float *desc);
  void Get_SURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc);

  // M-SURF Pattern Descriptor
  void Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float *desc);
  void Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float *desc);

  // M-LDB Pattern Descriptor
  void Get_Upright_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char *desc);
  void Get_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char *desc);
  void Get_Upright_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char *desc);
  void Get_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char *desc);
};

//*************************************************************************************
//*************************************************************************************

// Inline functions
/**
 * @brief This function sets default parameters for the A-KAZE detector.
 * @param options AKAZE options
 */
void setDefaultAKAZEOptions(AKAZEOptions& options);

// Inline functions
void generateDescriptorSubsample(cv::Mat& sampleList, cv::Mat& comparisons,
                                 int nbits, int pattern_size, int nchannels);
float get_angle(float x, float y);
float gaussian(float x, float y, float sigma);
void check_descriptor_limits(int& x, int& y, const int width, const int height);
int fRound(float flt);

//*************************************************************************************
//*************************************************************************************

#endif
