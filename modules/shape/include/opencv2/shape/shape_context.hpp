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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_SHAPE_SHAPE_CONTEXT_HPP__
#define __OPENCV_SHAPE_SHAPE_CONTEXT_HPP__

namespace cv
{

/*!
 SCD implementation.
 The class implements SCD algorithm by Belongie et al.
 */
class CV_EXPORTS_W SCD 
{
public:
    //! the default constructor
    CV_WRAP SCD();
    //! the full constructor taking all the necessary parameters
    explicit CV_WRAP SCD(int nAngularBins=5,
                  int nRadialBins = 4, bool logscale = true);

    //! returns the descriptor size in float's 
    CV_WRAP int descriptorSize() const;

    //! Compute keypoints descriptors. 
    void operator()(InputArray img, CV_OUT std::vector<KeyPoint>& keypoints,
                    OutputArray descriptors) const;

    CV_PROP_RW int nAngularBins;
    CV_PROP_RW int nRadialBins;
    CV_PROP_RW bool logscale;

protected:

};

typedef SCD ShapeContextDescriptorExtractor;

} /* namespace cv */

#endif
