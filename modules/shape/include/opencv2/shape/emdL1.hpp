/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2012, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_EMD_L1_HPP
#define OPENCV_EMD_L1_HPP

#include "opencv2/core.hpp"

namespace cv
{
/****************************************************************************************\
*                                   EMDL1 Function                                      *
\****************************************************************************************/

//! @addtogroup shape
//! @{

/** @brief Computes the "minimal work" distance between two weighted point configurations base on the papers
"EMD-L1: An efficient and Robust Algorithm for comparing histogram-based descriptors", by Haibin
Ling and Kazunori Okuda; and "The Earth Mover's Distance is the Mallows Distance: Some Insights from
Statistics", by Elizaveta Levina and Peter Bickel.

@param signature1 First signature, a single column floating-point matrix. Each row is the value of
the histogram in each bin.
@param signature2 Second signature of the same format and size as signature1.
 */
CV_EXPORTS float EMDL1(InputArray signature1, InputArray signature2);

//! @}

}//namespace cv

#endif
