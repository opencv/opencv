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

////////////////////////////////////////////////////////////////////////////////
//
// NVIDIA CUDA implementation of Brox et al Optical Flow algorithm
//
// Algorithm is explained in the original paper:
//      T. Brox, A. Bruhn, N. Papenberg, J. Weickert:
//      High accuracy optical flow estimation based on a theory for warping.
//      ECCV 2004.
//
// Implementation by Mikhail Smirnov
// email: msmirnov@nvidia.com, devsupport@nvidia.com
//
// Credits for help with the code to:
// Alexey Mendelenko, Anton Obukhov, and Alexander Kharlamov.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _ncv_optical_flow_h_
#define _ncv_optical_flow_h_

#include "NCV.hpp"

/// \brief Model and solver parameters
struct NCVBroxOpticalFlowDescriptor
{
    /// flow smoothness
    Ncv32f alpha;
    /// gradient constancy importance
    Ncv32f gamma;
    /// pyramid scale factor
    Ncv32f scale_factor;
    /// number of lagged non-linearity iterations (inner loop)
    Ncv32u number_of_inner_iterations;
    /// number of warping iterations (number of pyramid levels)
    Ncv32u number_of_outer_iterations;
    /// number of linear system solver iterations
    Ncv32u number_of_solver_iterations;
};

/////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute optical flow
///
/// Based on method by Brox et al [2004]
/// \param [in]  desc              model and solver parameters
/// \param [in]  gpu_mem_allocator GPU memory allocator
/// \param [in]  frame0            source frame
/// \param [in]  frame1            frame to track
/// \param [out] u                 flow horizontal component (along \b x axis)
/// \param [out] v                 flow vertical component (along \b y axis)
/// \return                        computation status
/////////////////////////////////////////////////////////////////////////////////////////

NCV_EXPORTS
NCVStatus NCVBroxOpticalFlow(const NCVBroxOpticalFlowDescriptor desc,
                             INCVMemAllocator &gpu_mem_allocator,
                             const NCVMatrix<Ncv32f> &frame0,
                             const NCVMatrix<Ncv32f> &frame1,
                             NCVMatrix<Ncv32f> &u,
                             NCVMatrix<Ncv32f> &v,
                             cudaStream_t stream);

#endif
