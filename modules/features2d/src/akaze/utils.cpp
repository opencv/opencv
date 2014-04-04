//=============================================================================
//
// utils.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Georgia Institute of Technology (1)
//               TrueVision Solutions (2)
//
// Date: 15/09/2013
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2013, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file utils.cpp
 * @brief Some utilities functions
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "precomp.hpp"
#include "utils.h"

// Namespaces
using namespace std;
using namespace cv;

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the minimum value of a float image
 * @param src Input image
 * @param value Minimum value
 */
void compute_min_32F(const cv::Mat &src, float &value) {

  float aux = 1000.0;

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<float>(i,j) < aux) {
        aux = src.at<float>(i,j);
      }
    }
  }

  value = aux;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the maximum value of a float image
 * @param src Input image
 * @param value Maximum value
 */
void compute_max_32F(const cv::Mat &src, float &value) {

  float aux = 0.0;

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<float>(i,j) > aux) {
        aux = src.at<float>(i,j);
      }
    }
  }

  value = aux;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function converts the scale of the input image prior to visualization
 * @param src Input/Output image
 * @param value Maximum value
 */
void convert_scale(cv::Mat &src) {

  float min_val = 0, max_val = 0;

  compute_min_32F(src,min_val);

  src = src - min_val;

  compute_max_32F(src,max_val);
  src = src / max_val;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function copies the input image and converts the scale of the copied
 * image prior visualization
 * @param src Input image
 * @param dst Output image
 */
void copy_and_convert_scale(const cv::Mat &src, cv::Mat dst) {

  float min_val = 0, max_val = 0;

  src.copyTo(dst);
  compute_min_32F(dst,min_val);

  dst = dst - min_val;

  compute_max_32F(dst,max_val);
  dst = dst / max_val;
}

//*************************************************************************************
//*************************************************************************************

const size_t length = string("--descriptor_channels").size() + 2;
static inline std::ostream& cout_help()
{ cout << setw(length); return cout; }

static inline std::string toUpper(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function shows the possible command line configuration options
 */
void show_input_options_help(int example) {

  fflush(stdout);
  cout << "A-KAZE Features" << endl;
  cout << "Usage: ";
  if (example == 0) {
    cout << "./akaze_features -i img.jpg [options]" << endl;
  }
  else if (example == 1) {
    cout << "./akaze_match img1.jpg img2.pgm homography.txt [options]" << endl;
  }
  else if (example == 2) {
    cout << "./akaze_compare img1.jpg img2.pgm homography.txt [options]" << endl;
  }
  
  cout << endl;
  cout_help() << "Options below are not mandatory. Unless specified, default arguments are used." << endl << endl;  
  // Justify on the left
  cout << left;
  // Generalities
  cout_help() << "--help" << "Show the command line options" << endl;
  cout_help() << "--verbose " << "Verbosity is required" << endl;
  cout_help() << endl;
  // Scale-space parameters
  cout_help() << "--soffset" << "Base scale offset (sigma units)" << endl;
  cout_help() << "--omax" << "Maximum octave of image evolution" << endl;
  cout_help() << "--nsublevels" << "Number of sublevels per octave" << endl;
  cout_help() << "--diffusivity" << "Diffusivity function. Possible values:" << endl;
  cout_help() << " " << "0 -> Perona-Malik, g1 = exp(-|dL|^2/k^2)" << endl;
  cout_help() << " " << "1 -> Perona-Malik, g2 = 1 / (1 + dL^2 / k^2)" << endl;
  cout_help() << " " << "2 -> Weickert diffusivity" << endl;
  cout_help() << " " << "3 -> Charbonnier diffusivity" << endl;
  cout_help() << endl;
  // Feature detection parameters.
  cout_help() << "--dthreshold" << "Feature detector threshold response for keypoints" << endl;
  cout_help() << " " << "(0.001 can be a good value)" << endl;
  cout_help() << endl;
  // Descriptor parameters.
  cout_help() << "--descriptor" << "Descriptor Type. Possible values:" << endl;
  cout_help() << " " << "0 -> SURF_UPRIGHT" << endl;
  cout_help() << " " << "1 -> SURF" << endl;
  cout_help() << " " << "2 -> M-SURF_UPRIGHT," << endl;
  cout_help() << " " << "3 -> M-SURF" << endl;
  cout_help() << " " << "4 -> M-LDB_UPRIGHT" << endl;
  cout_help() << " " << "5 -> M-LDB" << endl;
  
  cout_help() << "--descriptor_channels " << "Descriptor Channels for M-LDB. Valid values: " << endl;
  cout_help() << " " << "1 -> intensity" << endl;
  cout_help() << " " << "2 -> intensity + gradient magnitude" << endl;
  cout_help() << " " << "3 -> intensity + X and Y gradients" <<endl;
  
  cout_help() << "--descriptor_size" << "Descriptor size for M-LDB in bits." << endl;
  cout_help() << " " << "0: means the full length descriptor (486)!!" << endl;
  cout_help() << endl;
  // Save results?
  cout_help() << "--show_results" << "Possible values below:" << endl;
  cout_help() << " " << "1 -> show detection results." << endl;
  cout_help() << " " << "0 -> don't show detection results" << endl;
  cout_help() << endl;
}
