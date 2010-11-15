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

#include "precomp.hpp"


using namespace cv;
using namespace cv::gpu;


#if !defined (HAVE_CUDA)

#else /* !defined (HAVE_CUDA) */


namespace 
{
    struct NppError
    {
        int error;
        string str;
    } 
    npp_errors [] = 
    {
        { NPP_NOT_SUPPORTED_MODE_ERROR, "NPP_NOT_SUPPORTED_MODE_ERROR" },
        { NPP_ROUND_MODE_NOT_SUPPORTED_ERROR, "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR" },
        { NPP_RESIZE_NO_OPERATION_ERROR, "NPP_RESIZE_NO_OPERATION_ERROR" },
        { NPP_BAD_ARG_ERROR, "NPP_BAD_ARG_ERROR" },
        { NPP_LUT_NUMBER_OF_LEVELS_ERROR, "NPP_LUT_NUMBER_OF_LEVELS_ERROR" },
        { NPP_TEXTURE_BIND_ERROR, "NPP_TEXTURE_BIND_ERROR" },
        { NPP_COEFF_ERROR, "NPP_COEFF_ERROR" },
        { NPP_RECT_ERROR, "NPP_RECT_ERROR" },
        { NPP_QUAD_ERROR, "NPP_QUAD_ERROR" },
        { NPP_WRONG_INTERSECTION_ROI_ERROR, "NPP_WRONG_INTERSECTION_ROI_ERROR" },
        { NPP_NOT_EVEN_STEP_ERROR, "NPP_NOT_EVEN_STEP_ERROR" },
        { NPP_INTERPOLATION_ERROR, "NPP_INTERPOLATION_ERROR" },
        { NPP_RESIZE_FACTOR_ERROR, "NPP_RESIZE_FACTOR_ERROR" },
        { NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR, "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR" },
        { NPP_MEMFREE_ERR, "NPP_MEMFREE_ERR" },
        { NPP_MEMSET_ERR, "NPP_MEMSET_ERR" },
        { NPP_MEMCPY_ERROR, "NPP_MEMCPY_ERROR" },
        { NPP_MEM_ALLOC_ERR, "NPP_MEM_ALLOC_ERR" },
        { NPP_HISTO_NUMBER_OF_LEVELS_ERROR, "NPP_HISTO_NUMBER_OF_LEVELS_ERROR" },
        { NPP_MIRROR_FLIP_ERR, "NPP_MIRROR_FLIP_ERR" },
        { NPP_INVALID_INPUT, "NPP_INVALID_INPUT" },
        { NPP_ALIGNMENT_ERROR, "NPP_ALIGNMENT_ERROR" },
        { NPP_STEP_ERROR, "NPP_STEP_ERROR" },
        { NPP_SIZE_ERROR, "NPP_SIZE_ERROR" },
        { NPP_POINTER_ERROR, "NPP_POINTER_ERROR" },
        { NPP_NULL_POINTER_ERROR, "NPP_NULL_POINTER_ERROR" },
        { NPP_CUDA_KERNEL_EXECUTION_ERROR, "NPP_CUDA_KERNEL_EXECUTION_ERROR" },
        { NPP_NOT_IMPLEMENTED_ERROR, "NPP_NOT_IMPLEMENTED_ERROR" },
        { NPP_ERROR, "NPP_ERROR" }, 
        { NPP_NO_ERROR, "NPP_NO_ERROR" },
        { NPP_SUCCESS, "NPP_SUCCESS" },
        { NPP_WARNING, "NPP_WARNING" },
        { NPP_WRONG_INTERSECTION_QUAD_WARNING, "NPP_WRONG_INTERSECTION_QUAD_WARNING" },
        { NPP_MISALIGNED_DST_ROI_WARNING, "NPP_MISALIGNED_DST_ROI_WARNING" },
        { NPP_AFFINE_QUAD_INCORRECT_WARNING, "NPP_AFFINE_QUAD_INCORRECT_WARNING" },
        //disabled in NPP for cuda 3.2-rc
        //{ NPP_AFFINE_QUAD_CHANGED_WARNING, "NPP_AFFINE_QUAD_CHANGED_WARNING" },
        //{ NPP_ADJUSTED_ROI_SIZE_WARNING, "NPP_ADJUSTED_ROI_SIZE_WARNING" },
        { NPP_DOUBLE_SIZE_WARNING, "NPP_DOUBLE_SIZE_WARNING" },
        { NPP_ODD_ROI_WARNING, "NPP_ODD_ROI_WARNING" }
    };

    int error_num = sizeof(npp_errors)/sizeof(npp_errors[0]);

    struct Searcher
    {
        int err;
        Searcher(int err_) : err(err_) {};
        bool operator()(const NppError& e) const { return e.error == err; }
    };

}

namespace cv
{
    namespace gpu
    {
        const string getNppErrorString( int err )
        {
            int idx = std::find_if(npp_errors, npp_errors + error_num, Searcher(err)) - npp_errors;
            const string& msg = (idx != error_num) ? npp_errors[idx].str : string("Unknown error code");

            std::stringstream interpreter;
            interpreter << msg <<" [Code = " << err << "]";

            return interpreter.str();
        }

        void nppError( int err, const char *file, const int line, const char *func)
        {                    
            cv::error( cv::Exception(CV_GpuNppCallError, getNppErrorString(err), func, file, line) );                
        }

        void error(const char *error_string, const char *file, const int line, const char *func)
        {          
            //if (uncaught_exception())
            cv::error( cv::Exception(CV_GpuApiCallError, error_string, func, file, line) );
        }
    }
}

#endif