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
using namespace std;

#ifdef HAVE_CUDA

namespace 
{
    #define error_entry(entry)  { entry, #entry }

    struct ErrorEntry
    {
        int code;
        string str;
    }; 

    struct ErrorEntryComparer
    {
        int code;
        ErrorEntryComparer(int code_) : code(code_) {};
        bool operator()(const ErrorEntry& e) const { return e.code == code; }
    };

    string getErrorString(int code, const ErrorEntry* errors, size_t n)
    {
        size_t idx = find_if(errors, errors + n, ErrorEntryComparer(code)) - errors;

        const string& msg = (idx != n) ? errors[idx].str : string("Unknown error code");

        ostringstream ostr;
        ostr << msg << " [Code = " << code << "]";

        return ostr.str();
    }

    //////////////////////////////////////////////////////////////////////////
    // NPP errors
    
    const ErrorEntry npp_errors [] = 
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

    const size_t npp_error_num = sizeof(npp_errors) / sizeof(npp_errors[0]);

    //////////////////////////////////////////////////////////////////////////
    // NCV errors
    
    const ErrorEntry ncv_errors [] = 
    {
        error_entry( NCV_SUCCESS ),
        error_entry( NCV_UNKNOWN_ERROR ),
        error_entry( NCV_CUDA_ERROR ),
        error_entry( NCV_NPP_ERROR ),
        error_entry( NCV_FILE_ERROR ),
        error_entry( NCV_NULL_PTR ),
        error_entry( NCV_INCONSISTENT_INPUT ),
        error_entry( NCV_TEXTURE_BIND_ERROR ),
        error_entry( NCV_DIMENSIONS_INVALID ),
        error_entry( NCV_INVALID_ROI ),
        error_entry( NCV_INVALID_STEP ),
        error_entry( NCV_INVALID_SCALE ),
        error_entry( NCV_INVALID_SCALE ),
        error_entry( NCV_ALLOCATOR_NOT_INITIALIZED ),
        error_entry( NCV_ALLOCATOR_BAD_ALLOC ),
        error_entry( NCV_ALLOCATOR_BAD_DEALLOC ),
        error_entry( NCV_ALLOCATOR_INSUFFICIENT_CAPACITY ),
        error_entry( NCV_ALLOCATOR_DEALLOC_ORDER ),
        error_entry( NCV_ALLOCATOR_BAD_REUSE ),
        error_entry( NCV_MEM_COPY_ERROR ),
        error_entry( NCV_MEM_RESIDENCE_ERROR ),
        error_entry( NCV_MEM_INSUFFICIENT_CAPACITY ),
        error_entry( NCV_HAAR_INVALID_PIXEL_STEP ),
        error_entry( NCV_HAAR_TOO_MANY_FEATURES_IN_CLASSIFIER ),
        error_entry( NCV_HAAR_TOO_MANY_FEATURES_IN_CASCADE ),
        error_entry( NCV_HAAR_TOO_LARGE_FEATURES ),
        error_entry( NCV_HAAR_XML_LOADING_EXCEPTION ),
        error_entry( NCV_NOIMPL_HAAR_TILTED_FEATURES ),
        error_entry( NCV_WARNING_HAAR_DETECTIONS_VECTOR_OVERFLOW ),
        error_entry( NPPST_SUCCESS ),
        error_entry( NPPST_ERROR ),
        error_entry( NPPST_CUDA_KERNEL_EXECUTION_ERROR ),
        error_entry( NPPST_NULL_POINTER_ERROR ),
        error_entry( NPPST_TEXTURE_BIND_ERROR ),
        error_entry( NPPST_MEMCPY_ERROR ),
        error_entry( NPPST_MEM_ALLOC_ERR ),
        error_entry( NPPST_MEMFREE_ERR ),
        error_entry( NPPST_INVALID_ROI ),
        error_entry( NPPST_INVALID_STEP ),
        error_entry( NPPST_INVALID_SCALE ),
        error_entry( NPPST_MEM_INSUFFICIENT_BUFFER ),
        error_entry( NPPST_MEM_RESIDENCE_ERROR ),
        error_entry( NPPST_MEM_INTERNAL_ERROR )
    };

    const size_t ncv_error_num = sizeof(ncv_errors) / sizeof(ncv_errors[0]);

    //////////////////////////////////////////////////////////////////////////
    // CUFFT errors

    const ErrorEntry cufft_errors[] = 
    {
        error_entry( CUFFT_INVALID_PLAN ),
        error_entry( CUFFT_ALLOC_FAILED ),
        error_entry( CUFFT_INVALID_TYPE ),
        error_entry( CUFFT_INVALID_VALUE ),
        error_entry( CUFFT_INTERNAL_ERROR ),
        error_entry( CUFFT_EXEC_FAILED ),
        error_entry( CUFFT_SETUP_FAILED ),
        error_entry( CUFFT_INVALID_SIZE ),
        error_entry( CUFFT_UNALIGNED_DATA )
    };

    const int cufft_error_num = sizeof(cufft_errors) / sizeof(cufft_errors[0]);

    //////////////////////////////////////////////////////////////////////////
    // CUBLAS errors

    const ErrorEntry cublas_errors[] = 
    {
        error_entry( CUBLAS_STATUS_SUCCESS ),
        error_entry( CUBLAS_STATUS_NOT_INITIALIZED ),
        error_entry( CUBLAS_STATUS_ALLOC_FAILED ),
        error_entry( CUBLAS_STATUS_INVALID_VALUE ),
        error_entry( CUBLAS_STATUS_ARCH_MISMATCH ),
        error_entry( CUBLAS_STATUS_MAPPING_ERROR ),
        error_entry( CUBLAS_STATUS_EXECUTION_FAILED ),
        error_entry( CUBLAS_STATUS_INTERNAL_ERROR )
    };

    const int cublas_error_num = sizeof(cublas_errors) / sizeof(cublas_errors[0]);
}

namespace cv
{
    namespace gpu
    {
        void error(const char *error_string, const char *file, const int line, const char *func)
        {          
            int code = CV_GpuApiCallError;

            if (uncaught_exception())
            {
                const char* errorStr = cvErrorStr(code);            
                const char* function = func ? func : "unknown function";    

                cerr << "OpenCV Error: " << errorStr << "(" << error_string << ") in " << function << ", file " << file << ", line " << line;
                cerr.flush();            
            }
            else    
                cv::error( cv::Exception(code, error_string, func, file, line) );
        }

        void nppError(int code, const char *file, const int line, const char *func)
        {
            string msg = getErrorString(code, npp_errors, npp_error_num);
            cv::gpu::error(msg.c_str(), file, line, func);
        }

        void ncvError(int code, const char *file, const int line, const char *func)
        {
            string msg = getErrorString(code, ncv_errors, ncv_error_num);
            cv::gpu::error(msg.c_str(), file, line, func);
        }

        void cufftError(int code, const char *file, const int line, const char *func)
        {
            string msg = getErrorString(code, cufft_errors, cufft_error_num);
            cv::gpu::error(msg.c_str(), file, line, func);
        }

        void cublasError(int code, const char *file, const int line, const char *func)
        {
            string msg = getErrorString(code, cublas_errors, cublas_error_num);
            cv::gpu::error(msg.c_str(), file, line, func);
        }
    }
}

#endif
