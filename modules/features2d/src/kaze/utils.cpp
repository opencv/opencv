
//=============================================================================
//
// utils.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 29/12/2011
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file utils.cpp
 * @brief Some useful functions
 * @date Dec 29, 2011
 * @author Pablo F. Alcantarilla
 */

#include "utils.h"

using namespace std;
using namespace cv;

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function copies the input image and converts the scale of the copied
 * image prior visualization
 * @param src Input image
 * @param dst Output image
 */
void copy_and_convert_scale(const cv::Mat& src, cv::Mat& dst) {

  float min_val = 0, max_val = 0;

  src.copyTo(dst);
  compute_min_32F(dst,min_val);

  dst = dst - min_val;

  compute_max_32F(dst,max_val);
  dst = dst / max_val;
}

//*************************************************************************************
//*************************************************************************************

/*
void show_input_options_help(int example) {

  fflush(stdout);

  cout << endl;
  cout << endl;
  cout << "KAZE Features" << endl;
  cout << "***********************************************************" << endl;
  cout << "For running the program you need to type in the command line the following arguments: " << endl;

  if (example == 0) {
    cout << "./kaze_features img.jpg [options]" << endl;
  }
  else if (example == 1) {
    cout << "./kaze_match img1.jpg img2.pgm homography.txt [options]" << endl;
  }
  else if (example == 2) {
    cout << "./kaze_compare img1.jpg img2.pgm homography.txt [options]" << endl;
  }

  cout << endl;
  cout << "The options are not mandatory. In case you do not specify additional options, default arguments will be used" << endl << endl;
  cout << "Here is a description of the additional options: " << endl;
  cout << "--verbose " << "\t\t if verbosity is required" << endl;
  cout << "--help" << "\t\t for showing the command line options" << endl;
  cout << "--soffset" << "\t\t the base scale offset (sigma units)" << endl;
  cout << "--omax" << "\t\t maximum octave evolution of the image 2^sigma (coarsest scale)" << endl;
  cout << "--nsublevels" << "\t\t number of sublevels per octave" << endl;
  cout << "--dthreshold" << "\t\t Feature detector threshold response for accepting points (0.001 can be a good value)" << endl;
  cout << "--descriptor" << "\t\t Descriptor Type 0 -> SURF, 1 -> M-SURF, 2 -> G-SURF" << endl;
  cout << "--use_fed" "\t\t 1 -> Use FED, 0 -> Use AOS for the nonlinear diffusion filtering" << endl;
  cout << "--upright" << "\t\t 0 -> Rotation Invariant, 1 -> No Rotation Invariant" << endl;
  cout << "--extended" << "\t\t 0 -> Normal Descriptor (64), 1 -> Extended Descriptor (128)" << endl;
  cout << "--output keypoints.txt" << "\t\t For saving the detected keypoints into a .txt file" << endl;
  cout << "--save_scale_space" << "\t\t 1 in case we want to save the nonlinear scale space images. 0 otherwise" << endl;
  cout << "--show_results" << "\t\t 1 in case we want to show detection results. 0 otherwise" << endl;
  cout << endl;
}
*/