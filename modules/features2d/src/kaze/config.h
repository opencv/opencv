
/**
 * @file config.h
 * @brief Configuration file
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#ifndef _CONFIG_H_
#define _CONFIG_H_

//******************************************************************************
//******************************************************************************

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>

// OpenCV Includes
#include "precomp.hpp"

// OpenMP Includes
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

//*************************************************************************************
//*************************************************************************************

// Some defines
#define NMAX_CHAR 400

// Some default options
const float DEFAULT_SCALE_OFFSET = 1.60; // Base scale offset (sigma units)
const float DEFAULT_OCTAVE_MAX = 4.0; // Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
const int DEFAULT_NSUBLEVELS = 4; // Default number of sublevels per scale level
const float DEFAULT_DETECTOR_THRESHOLD = 0.001; // Detector response threshold to accept point
const float DEFAULT_MIN_DETECTOR_THRESHOLD = 0.00001;     // Minimum Detector response threshold to accept point
const int DEFAULT_DESCRIPTOR_MODE = 1; // Descriptor Mode 0->SURF, 1->M-SURF
const bool DEFAULT_USE_FED = true;  // 0->AOS, 1->FED
const bool DEFAULT_UPRIGHT = false;  // Upright descriptors, not invariant to rotation
const bool DEFAULT_EXTENDED = false; // Extended descriptor, dimension 128
const bool DEFAULT_SAVE_SCALE_SPACE = false; // For saving the scale space images
const bool DEFAULT_VERBOSITY = false; // Verbosity level (0->no verbosity)
const bool DEFAULT_SHOW_RESULTS = true; // For showing the output image with the detected features plus some ratios
const bool DEFAULT_SAVE_KEYPOINTS = false; // For saving the list of keypoints

// Some important configuration variables
const float DEFAULT_SIGMA_SMOOTHING_DERIVATIVES = 1.0;
const float DEFAULT_KCONTRAST = .01;
const float KCONTRAST_PERCENTILE = 0.7;
const int KCONTRAST_NBINS = 300;
const bool COMPUTE_KCONTRAST = true;
const int DEFAULT_DIFFUSIVITY_TYPE = 1;  // 0 -> PM G1, 1 -> PM G2, 2 -> Weickert
const bool USE_CLIPPING_NORMALIZATION = false;
const float CLIPPING_NORMALIZATION_RATIO = 1.6;
const int CLIPPING_NORMALIZATION_NITER = 5;

//*************************************************************************************
//*************************************************************************************

struct KAZEOptions {

  KAZEOptions() {
    // Load the default options
    soffset = DEFAULT_SCALE_OFFSET;
    omax = DEFAULT_OCTAVE_MAX;
    nsublevels = DEFAULT_NSUBLEVELS;
    dthreshold = DEFAULT_DETECTOR_THRESHOLD;
    use_fed = DEFAULT_USE_FED;
    upright = DEFAULT_UPRIGHT;
    extended = DEFAULT_EXTENDED;
    descriptor = DEFAULT_DESCRIPTOR_MODE;
    diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
    sderivatives = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
    save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
    save_keypoints = DEFAULT_SAVE_KEYPOINTS;
    verbosity = DEFAULT_VERBOSITY;
    show_results = DEFAULT_SHOW_RESULTS;
  }

  float soffset;
  int omax;
  int nsublevels;
  int img_width;
  int img_height;
  int diffusivity;
  float sderivatives;
  float dthreshold;
  bool use_fed;
  bool upright;
  bool extended;
  int descriptor;
  bool save_scale_space;
  bool save_keypoints;
  bool verbosity;
  bool show_results;
};

struct TEvolution {
  cv::Mat Lx, Ly;	// First order spatial derivatives
  cv::Mat Lxx, Lxy, Lyy;	// Second order spatial derivatives
  cv::Mat Lflow;	// Diffusivity image
  cv::Mat Lt;	// Evolution image
  cv::Mat Lsmooth; // Smoothed image
  cv::Mat Lstep; // Evolution step update
  cv::Mat Ldet; // Detector response
  float etime;	// Evolution time
  float esigma;	// Evolution sigma. For linear diffusion t = sigma^2 / 2
  float octave;	// Image octave
  float sublevel;	// Image sublevel in each octave
  int sigma_size;	// Integer esigma. For computing the feature detector responses
};

//*************************************************************************************
//*************************************************************************************

#endif




