
#ifndef _UTILS_H_
#define _UTILS_H_

//******************************************************************************
//******************************************************************************

// OpenCV Includes
#include "precomp.hpp"

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

//******************************************************************************
//******************************************************************************

// Stringify common types such as int, double and others.
template <typename T>
inline std::string to_string(const T& x) {
  std::stringstream oss;
  oss << x;
  return oss.str();
}

//******************************************************************************
//******************************************************************************

// Stringify and format integral types as follows:
// to_formatted_string(  1, 2) produces string:  '01'
// to_formatted_string(  5, 2) produces string:  '05'
// to_formatted_string( 19, 2) produces string:  '19'
// to_formatted_string( 19, 3) produces string: '019'
template <typename Integer>
inline std::string to_formatted_string(Integer x, int num_digits) {
  std::stringstream oss;
  oss << std::setfill('0') << std::setw(num_digits) << x;
  return oss.str();
}

//******************************************************************************
//******************************************************************************

void compute_min_32F(const cv::Mat& src, float& value);
void compute_max_32F(const cv::Mat& src, float& value);
void convert_scale(cv::Mat& src);
void copy_and_convert_scale(const cv::Mat& src, cv::Mat& dst);

#endif
