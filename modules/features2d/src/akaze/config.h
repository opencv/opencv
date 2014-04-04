#ifndef _CONFIG_H_
#define _CONFIG_H_

// STL
#include <string>
#include <vector>
#include <cmath>
#include <bitset>
#include <iomanip>

// OpenCV
#include "precomp.hpp"

// OpenMP
#ifdef _OPENMP
# include <omp.h>
#endif

// Lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
const float gauss25[7][7] = {
  {0.02546481f,	0.02350698f,	0.01849125f,	0.01239505f,	0.00708017f,	0.00344629f,	0.00142946f},
  {0.02350698f,	0.02169968f,	0.01706957f,	0.01144208f,	0.00653582f,	0.00318132f,	0.00131956f},
  {0.01849125f,	0.01706957f,	0.01342740f,	0.00900066f,	0.00514126f,	0.00250252f,	0.00103800f},
  {0.01239505f,	0.01144208f,	0.00900066f,	0.00603332f,	0.00344629f,	0.00167749f,	0.00069579f},
  {0.00708017f,	0.00653582f,	0.00514126f,	0.00344629f,	0.00196855f,	0.00095820f,	0.00039744f},
  {0.00344629f,	0.00318132f,	0.00250252f,	0.00167749f,	0.00095820f,	0.00046640f,	0.00019346f},
  {0.00142946f,	0.00131956f,	0.00103800f,	0.00069579f,	0.00039744f,	0.00019346f,	0.00008024f}
};


// Scale Space parameters
const float DEFAULT_SCALE_OFFSET = 1.60f;    // Base scale offset (sigma units)
const float DEFAULT_FACTOR_SIZE = 1.5f;      // Factor for the multiscale derivatives
const int DEFAULT_OCTAVE_MIN = 0;            // Initial octave level (-1 means that the size of the input image is duplicated)
const int DEFAULT_OCTAVE_MAX = 4;            // Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
const int DEFAULT_NSUBLEVELS = 4;            // Default number of sublevels per scale level
const int DEFAULT_DIFFUSIVITY_TYPE = 1;
const float KCONTRAST_PERCENTILE = 0.7f;
const int KCONTRAST_NBINS = 300;
const float DEFAULT_SIGMA_SMOOTHING_DERIVATIVES = 1.0f;
const float DEFAULT_KCONTRAST = .01f;


// Detector Parameters
const float DEFAULT_DETECTOR_THRESHOLD = 0.001f;           // Detector response threshold to accept point
const float DEFAULT_MIN_DETECTOR_THRESHOLD = 0.00001f;     // Minimum Detector response threshold to accept point
const int DEFAULT_LDB_DESCRIPTOR_SIZE = 0;  // Use 0 for the full descriptor, or the number of bits
const int DEFAULT_LDB_PATTERN_SIZE = 10;    // Actual patch size is 2*pattern_size*point.scale;
const int DEFAULT_LDB_CHANNELS = 3;

// Descriptor Parameters
enum DESCRIPTOR_TYPE
{
  SURF_UPRIGHT = 0, // Upright descriptors, not invariant to rotation
  SURF = 1,
  MSURF_UPRIGHT = 2, // Upright descriptors, not invariant to rotation
  MSURF = 3,
  MLDB_UPRIGHT = 4, // Upright descriptors, not invariant to rotation
  MLDB = 5
};

const int DEFAULT_DESCRIPTOR = MLDB;

// Some debugging options
const bool DEFAULT_SAVE_SCALE_SPACE = false; // For saving the scale space images
const bool DEFAULT_VERBOSITY = false; // Verbosity level (0->no verbosity)
const bool DEFAULT_SHOW_RESULTS = true; // For showing the output image with the detected features plus some ratios
const bool DEFAULT_SAVE_KEYPOINTS = false; // For saving the list of keypoints

// Options structure
struct AKAZEOptions
{
  int omin;
  int omax;
  int nsublevels;
  int img_width;
  int img_height;
  int diffusivity;
  float soffset;
  float sderivatives;
  float dthreshold;
  float dthreshold2;
  int descriptor;
  int descriptor_size;
  int descriptor_channels;
  int descriptor_pattern_size;
  bool save_scale_space;
  bool save_keypoints;
  bool verbosity;

  AKAZEOptions()
  {
    // Load the default options
    soffset = DEFAULT_SCALE_OFFSET;
    omax = DEFAULT_OCTAVE_MAX;
    nsublevels = DEFAULT_NSUBLEVELS;
    dthreshold = DEFAULT_DETECTOR_THRESHOLD;
    diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
    descriptor = DEFAULT_DESCRIPTOR;
    descriptor_size = DEFAULT_LDB_DESCRIPTOR_SIZE;
    descriptor_channels = DEFAULT_LDB_CHANNELS;
    descriptor_pattern_size = DEFAULT_LDB_PATTERN_SIZE;
    sderivatives = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
    save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
    save_keypoints = DEFAULT_SAVE_KEYPOINTS;
    verbosity = DEFAULT_VERBOSITY;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const AKAZEOptions& akaze_options)
  {
    os << std::left;
#define CHECK_AKAZE_OPTION(option) \
    os << std::setw(33) << #option << " =  " << option << std::endl

    // Scale-space parameters.
    CHECK_AKAZE_OPTION(akaze_options.omax);
    CHECK_AKAZE_OPTION(akaze_options.nsublevels);
    CHECK_AKAZE_OPTION(akaze_options.soffset);
    CHECK_AKAZE_OPTION(akaze_options.sderivatives);
    CHECK_AKAZE_OPTION(akaze_options.diffusivity);
    // Detection parameters.
    CHECK_AKAZE_OPTION(akaze_options.dthreshold);
    // Descriptor parameters.
    CHECK_AKAZE_OPTION(akaze_options.descriptor);
    CHECK_AKAZE_OPTION(akaze_options.descriptor_channels);
    CHECK_AKAZE_OPTION(akaze_options.descriptor_size);
    // Save scale-space
    CHECK_AKAZE_OPTION(akaze_options.save_scale_space);
    // Verbose option for debug.
    CHECK_AKAZE_OPTION(akaze_options.verbosity);
#undef CHECK_AKAZE_OPTIONS

    return os;
  }
};

struct tevolution
{
	cv::Mat Lx, Ly;	// First order spatial derivatives
	cv::Mat Lxx, Lxy, Lyy;	// Second order spatial derivatives
	cv::Mat Lflow;	// Diffusivity image
	cv::Mat Lt;	// Evolution image
	cv::Mat Lsmooth; // Smoothed image
	cv::Mat Lstep; // Evolution step update
	cv::Mat Ldet; // Detector response
	float etime;	// Evolution time
	float esigma;	// Evolution sigma. For linear diffusion t = sigma^2 / 2
  int octave;	// Image octave
  int sublevel;	// Image sublevel in each octave
	int sigma_size;	// Integer sigma. For computing the feature detector responses
};


#endif