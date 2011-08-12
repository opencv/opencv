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
    #define error_entry(entry)  { entry, #entry }

    //////////////////////////////////////////////////////////////////////////
    // NPP errors

    struct NppError
    {
        int error;
        string str;
    } 
    
    npp_errors [] = 
    {
        error_entry( NPP_NOT_SUPPORTED_MODE_ERROR ),
        error_entry( NPP_ROUND_MODE_NOT_SUPPORTED_ERROR ),
        error_entry( NPP_RESIZE_NO_OPERATION_ERROR ),

#if defined (_MSC_VER)
        error_entry( NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY ),
#endif
        error_entry( NPP_BAD_ARG_ERROR ),
        error_entry( NPP_LUT_NUMBER_OF_LEVELS_ERROR ),
        error_entry( NPP_TEXTURE_BIND_ERROR ),
        error_entry( NPP_COEFF_ERROR ),
        error_entry( NPP_RECT_ERROR ),
        error_entry( NPP_QUAD_ERROR ),
        error_entry( NPP_WRONG_INTERSECTION_ROI_ERROR ),
        error_entry( NPP_NOT_EVEN_STEP_ERROR ),
        error_entry( NPP_INTERPOLATION_ERROR ),
        error_entry( NPP_RESIZE_FACTOR_ERROR ),
        error_entry( NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR ),
        error_entry( NPP_MEMFREE_ERR ),
        error_entry( NPP_MEMSET_ERR ),
        error_entry( NPP_MEMCPY_ERROR ),
        error_entry( NPP_MEM_ALLOC_ERR ),
        error_entry( NPP_HISTO_NUMBER_OF_LEVELS_ERROR ),
        error_entry( NPP_MIRROR_FLIP_ERR ),
        error_entry( NPP_INVALID_INPUT ),
        error_entry( NPP_ALIGNMENT_ERROR ),
        error_entry( NPP_STEP_ERROR ),
        error_entry( NPP_SIZE_ERROR ),
        error_entry( NPP_POINTER_ERROR ),
        error_entry( NPP_NULL_POINTER_ERROR ),
        error_entry( NPP_CUDA_KERNEL_EXECUTION_ERROR ),
        error_entry( NPP_NOT_IMPLEMENTED_ERROR ),
        error_entry( NPP_ERROR ),
        error_entry( NPP_NO_ERROR ),
        error_entry( NPP_SUCCESS ),
        error_entry( NPP_WARNING ),
        error_entry( NPP_WRONG_INTERSECTION_QUAD_WARNING ),
        error_entry( NPP_MISALIGNED_DST_ROI_WARNING ),
        error_entry( NPP_AFFINE_QUAD_INCORRECT_WARNING ),
        error_entry( NPP_DOUBLE_SIZE_WARNING ),
        error_entry( NPP_ODD_ROI_WARNING )
    };

    const size_t error_num = sizeof(npp_errors) / sizeof(npp_errors[0]);

    struct Searcher
    {
        int err;
        Searcher(int err_) : err(err_) {};
        bool operator()(const NppError& e) const { return e.error == err; }
    };

    //////////////////////////////////////////////////////////////////////////
    // CUFFT errors

    struct CufftError
    {
        int code;
        string message;
    };

    const CufftError cufft_errors[] = 
    {
        error_entry(CUFFT_INVALID_PLAN),
        error_entry(CUFFT_ALLOC_FAILED),
        error_entry(CUFFT_INVALID_TYPE),
        error_entry(CUFFT_INVALID_VALUE),
        error_entry(CUFFT_INTERNAL_ERROR),
        error_entry(CUFFT_EXEC_FAILED),
        error_entry(CUFFT_SETUP_FAILED),
        error_entry(CUFFT_INVALID_SIZE),
        error_entry(CUFFT_UNALIGNED_DATA)
    };

    struct CufftErrorComparer
    {
        CufftErrorComparer(int code_): code(code_) {}
        bool operator()(const CufftError& other) const 
        { 
            return other.code == code; 
        }
        int code;
    };

    const int cufft_error_num = sizeof(cufft_errors) / sizeof(cufft_errors[0]);

}

namespace cv
{
    namespace gpu
    {
        const string getNppErrorString( int err )
        {
            size_t idx = std::find_if(npp_errors, npp_errors + error_num, Searcher(err)) - npp_errors;
            const string& msg = (idx != error_num) ? npp_errors[idx].str : string("Unknown error code");

            std::stringstream interpreter;
            interpreter << msg <<" [Code = " << err << "]";

            return interpreter.str();
        }

        void nppError( int err, const char *file, const int line, const char *func)
        {                    
            cv::error( cv::Exception(CV_GpuNppCallError, getNppErrorString(err), func, file, line) );                
        }

        const string getCufftErrorString(int err_code)
        {
            const CufftError* cufft_error = std::find_if(
                    cufft_errors, cufft_errors + cufft_error_num, 
                    CufftErrorComparer(err_code));

            bool found = cufft_error != cufft_errors + cufft_error_num;

            std::stringstream ss;
            ss << (found ? cufft_error->message : "Unknown error code");
            ss << " [Code = " << err_code << "]";

            return ss.str();
        }

        void cufftError(int err, const char *file, const int line, const char *func)
        {
            cv::error(cv::Exception(CV_GpuCufftCallError, getCufftErrorString(err), func, file, line));
        }

        void error(const char *error_string, const char *file, const int line, const char *func)
        {          
            int code = CV_GpuApiCallError;

            if (std::uncaught_exception())
            {
                const char* errorStr = cvErrorStr(code);            
                const char* function = func ? func : "unknown function";    

                std::cerr << "OpenCV Error: " << errorStr << "(" << error_string << ") in " << function << ", file " << file << ", line " << line;
                std::cerr.flush();            
            }
            else    
                cv::error( cv::Exception(code, error_string, func, file, line) );
        }
    }
}

#endif
