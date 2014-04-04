
/**
 * @file utils.h
 * @brief Some useful functions
 * @date Dec 29, 2011
 * @author Pablo F. Alcantarilla
 */

#ifndef UTILS_H_
#define UTILS_H_

//******************************************************************************
//******************************************************************************

// OPENCV Includes
#include "precomp.hpp"

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <math.h>

//*************************************************************************************
//*************************************************************************************

// Declaration of Functions
void compute_min_32F(const cv::Mat& src, float& value);
void compute_max_32F(const cv::Mat& src, float& value);
void convert_scale(cv::Mat& src);
void copy_and_convert_scale(const cv::Mat &src, cv::Mat& dst);

//*************************************************************************************
//*************************************************************************************

#endif // UTILS_H_
