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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/transform.hpp"

namespace cv { namespace gpu
{
    namespace transform_points
    {
        __constant__ float3 crot0;
        __constant__ float3 crot1;
        __constant__ float3 crot2;
        __constant__ float3 ctransl;

        struct TransformOp
        {
            __device__ float3 operator()(float3 p) const
            {
                return make_float3(
                        crot0.x * p.x + crot0.y * p.y + crot0.z * p.z + ctransl.x,
                        crot1.x * p.x + crot1.y * p.y + crot1.z * p.z + ctransl.y,
                        crot2.x * p.x + crot2.y * p.y + crot2.z * p.z + ctransl.z);
            }
        };

        void call(const DevMem2D_<float> src, const float* rot,
                  const float* transl, DevMem2D_<float> dst)
        {
            cudaSafeCall(cudaMemcpyToSymbol(crot0, rot, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot1, rot + 3, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot2, rot + 6, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(ctransl, transl, sizeof(float) * 3));
            transform((const DevMem2D_<float3>)src, (DevMem2D_<float3>)dst, TransformOp());
        }
    } // namespace transform_points

    namespace project_points
    {
        __constant__ float3 crot0;
        __constant__ float3 crot1;
        __constant__ float3 crot2;
        __constant__ float3 ctransl;
        __constant__ float3 cproj0;
        __constant__ float3 cproj1;

        struct ProjectOp
        {
            __device__ float2 operator()(float3 p) const
            {
                // Rotate and translate in 3D
                float3 t = make_float3(
                        crot0.x * p.x + crot0.y * p.y + crot0.z * p.z + ctransl.x,
                        crot1.x * p.x + crot1.y * p.y + crot1.z * p.z + ctransl.y,
                        crot2.x * p.x + crot2.y * p.y + crot2.z * p.z + ctransl.z);
                // Project on 2D plane
                return make_float2(
                        (cproj0.x * t.x + cproj0.y * t.y) / t.z + cproj0.z,
                        (cproj1.x * t.x + cproj1.y * t.y) / t.z + cproj1.z);
            }
        };

        void call(const DevMem2D_<float> src, const float* rot,
                  const float* transl, const float* proj, DevMem2D_<float> dst)
        {
            cudaSafeCall(cudaMemcpyToSymbol(crot0, rot, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot1, rot + 3, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot2, rot + 6, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(ctransl, transl, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(cproj0, proj, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(cproj1, proj + 3, sizeof(float) * 3));
            transform((const DevMem2D_<float3>)src, (DevMem2D_<float2>)dst, ProjectOp());
        }
    } // namespace project_points

}} // namespace cv { namespace gpu
