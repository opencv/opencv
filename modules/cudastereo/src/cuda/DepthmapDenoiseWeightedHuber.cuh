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

//! OpenDTAM Variant of Chambolle & Pock denoising
//!
//! The complicated half of the DTAM algorithm's mapping core,
//! but can be used independently to refine depthmaps.
//!
//! Written by Paul Foster for GSoC 2014 OpenDTAM project.
//! High level algorithm described by Richard Newcombe, Steven J. Lovegrove, and Andrew J. Davison. 
//! "DTAM: Dense tracking and mapping in real-time."
//! Which was in turn based on Chambolle & Pock's
//! "A first-order primal-dual algorithm for convex problems with applications to imaging."

#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH
#include <opencv2/core/cuda/common.hpp>//for cudaStream_t
namespace cv { namespace cuda { namespace device { namespace dtam_denoise{
    struct m33{
            float data[9];
        };
        struct m34{
            float data[12];
        };
        void loadConstants(uint h_rows, uint h_cols, uint h_layers, uint h_layerStep,
                float* h_a, float* h_d, float* h_cdata, float* h_lo, float* h_hi,
                float* h_loInd);

    void computeGCaller  (float* pp, float* g1p, float* gxp, float* gyp, int cols);
    void updateQDCaller  (float* gqxpt, float* gqypt, float *dpt, float * apt,
                    float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                    float theta);
    extern cudaStream_t localStream;
}}}}
#endif
