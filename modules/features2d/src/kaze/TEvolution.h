/**
 * @file TEvolution.h
 * @brief Header file with the declaration of the TEvolution struct
 * @date Jun 02, 2014
 * @author Pablo F. Alcantarilla
 */

#ifndef __OPENCV_FEATURES_2D_TEVOLUTION_H__
#define __OPENCV_FEATURES_2D_TEVOLUTION_H__

namespace cv
{

/* ************************************************************************* */
/// KAZE/A-KAZE nonlinear diffusion filtering evolution
template <typename MatType>
struct TEvolutionT
{
  TEvolutionT() {
    etime = 0.0f;
    esigma = 0.0f;
    octave = 0;
    sublevel = 0;
    sigma_size = 0;
  }

  MatType Lx, Ly;           ///< First order spatial derivatives
  MatType Lxx, Lxy, Lyy;    ///< Second order spatial derivatives
  MatType Lt;               ///< Evolution image
  MatType Lsmooth;          ///< Smoothed image
  MatType Ldet;             ///< Detector response

  float etime;              ///< Evolution time
  float esigma;             ///< Evolution sigma. For linear diffusion t = sigma^2 / 2
  int octave;               ///< Image octave
  int sublevel;             ///< Image sublevel in each octave
  int sigma_size;           ///< Integer esigma. For computing the feature detector responses
};

typedef TEvolutionT<Mat> TEvolution;
typedef TEvolutionT<UMat> UTEvolution;
typedef std::vector<TEvolution> TPyramid;
typedef std::vector<UTEvolution> UTPyramid;

}

#endif
