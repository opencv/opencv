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
// any express or bpied warranties, including, but not limited to, the bpied
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

#ifndef __OPENCV_GPU_DATAMOV_UTILS_HPP__
#define __OPENCV_GPU_DATAMOV_UTILS_HPP__

#include "common.hpp"

namespace cv { namespace gpu { namespace device 
{
    #if __CUDA_ARCH__ >= 200

        // for Fermi memory space is detected automatically
        template <typename T> struct ForceGlob
        {
            __device__ __forceinline__ static void Load(const T* ptr, int offset, T& val)  { val = ptr[offset];  }
        };
            
    #else // __CUDA_ARCH__ >= 200        

        #if defined(_WIN64) || defined(__LP64__)		
            // 64-bit register modifier for inlined asm
            #define OPENCV_GPU_ASM_PTR "l"
        #else	
            // 32-bit register modifier for inlined asm
            #define OPENCV_GPU_ASM_PTR "r"
        #endif

        template<class T> struct ForceGlob;

        #define OPENCV_GPU_DEFINE_FORCE_GLOB(base_type, ptx_type, reg_mod) \
            template <> struct ForceGlob<base_type> \
            { \
                __device__ __forceinline__ static void Load(const base_type* ptr, int offset, base_type& val) \
                { \
                    asm("ld.global."#ptx_type" %0, [%1];" : "="#reg_mod(val) : OPENCV_GPU_ASM_PTR(ptr + offset)); \
                } \
            };

        #define OPENCV_GPU_DEFINE_FORCE_GLOB_B(base_type, ptx_type) \
            template <> struct ForceGlob<base_type> \
            { \
                __device__ __forceinline__ static void Load(const base_type* ptr, int offset, base_type& val) \
                { \
                    asm("ld.global."#ptx_type" %0, [%1];" : "=r"(*reinterpret_cast<uint*>(&val)) : OPENCV_GPU_ASM_PTR(ptr + offset)); \
                } \
            };
        
            OPENCV_GPU_DEFINE_FORCE_GLOB_B(uchar,  u8)
            OPENCV_GPU_DEFINE_FORCE_GLOB_B(schar,  s8)
            OPENCV_GPU_DEFINE_FORCE_GLOB_B(char,   b8)
            OPENCV_GPU_DEFINE_FORCE_GLOB  (ushort, u16, h)
            OPENCV_GPU_DEFINE_FORCE_GLOB  (short,  s16, h)
            OPENCV_GPU_DEFINE_FORCE_GLOB  (uint,   u32, r)
            OPENCV_GPU_DEFINE_FORCE_GLOB  (int,    s32, r)	
            OPENCV_GPU_DEFINE_FORCE_GLOB  (float,  f32, f)	
            OPENCV_GPU_DEFINE_FORCE_GLOB  (double, f64, d)	            

        #undef OPENCV_GPU_DEFINE_FORCE_GLOB
        #undef OPENCV_GPU_DEFINE_FORCE_GLOB_B
        #undef OPENCV_GPU_ASM_PTR
        
    #endif // __CUDA_ARCH__ >= 200
}}} // namespace cv { namespace gpu { namespace device

#endif // __OPENCV_GPU_DATAMOV_UTILS_HPP__
