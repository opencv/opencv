/**
 * @file AKAZE.h
 * @brief Main class for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Mar 27, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#ifndef __OPENCV_FEATURES_2D_AKAZE_FEATURES_H__
#define __OPENCV_FEATURES_2D_AKAZE_FEATURES_H__

/* ************************************************************************* */
// Includes
#include "AKAZEConfig.h"

namespace cv
{

/// A-KAZE nonlinear diffusion filtering evolution
struct Evolution
{
  Evolution() {
    etime = 0.0f;
    esigma = 0.0f;
    octave = 0;
    sublevel = 0;
    sigma_size = 0;
    octave_ratio = 0.0f;
    border = 0;
  }

  Mat Lx, Ly;           ///< First order spatial derivatives
  Mat Lt;               ///< Evolution image
  Mat Lsmooth;          ///< Smoothed image, used only for computing determinant, released afterwards
  Mat Ldet;             ///< Detector response

  Size size;                ///< Size of the layer
  float etime;              ///< Evolution time
  float esigma;             ///< Evolution sigma. For linear diffusion t = sigma^2 / 2
  int octave;               ///< Image octave
  int sublevel;             ///< Image sublevel in each octave
  int sigma_size;           ///< Integer esigma. For computing the feature detector responses
  float octave_ratio;       ///< Scaling ratio of this octave. ratio = 2^octave
  int border;               ///< Width of border where descriptors cannot be computed
};

/* ************************************************************************* */
// AKAZE Class Declaration
class AKAZEFeatures {

private:

  AKAZEOptions options_;                ///< Configuration options for AKAZE
  std::vector<Evolution> evolution_;        ///< Vector of nonlinear diffusion evolution

  /// FED parameters
  int ncycles_;                  ///< Number of cycles
  bool reordering_;              ///< Flag for reordering time steps
  std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
  std::vector<int> nsteps_;      ///< Vector of number of steps per cycle

  /// Matrices for the M-LDB descriptor computation
  cv::Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  cv::Mat descriptorBits_;
  cv::Mat bitMask_;

public:

  /// Constructor with input arguments
  AKAZEFeatures(const AKAZEOptions& options);

  /// Scale Space methods
  void Allocate_Memory_Evolution();
  void Create_Nonlinear_Scale_Space(InputArray img);
  void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
  void Compute_Determinant_Hessian_Response(void);
  void Find_Scale_Space_Extrema(std::vector<Mat>& keypoints_by_layers);
  void Do_Subpixel_Refinement(std::vector<Mat>& keypoints_by_layers,
    std::vector<KeyPoint>& kpts);

  /// Feature description methods
  void Compute_Descriptors(std::vector<cv::KeyPoint>& kpts, OutputArray desc);
  void Compute_Keypoints_Orientation(std::vector<cv::KeyPoint>& kpts) const;
};

/* ************************************************************************* */
/// Inline functions
void generateDescriptorSubsample(cv::Mat& sampleList, cv::Mat& comparisons,
                                 int nbits, int pattern_size, int nchannels);

}

#endif
