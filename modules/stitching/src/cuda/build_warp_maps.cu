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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        // TODO use intrinsics like __sinf and so on

        namespace build_warp_maps
        {

            __constant__ float ck_rinv[9];
            __constant__ float cr_kinv[9];
            __constant__ float ct[3];
            __constant__ float cscale;
        }


        class PlaneMapper
        {
        public:
            static __device__ __forceinline__ void mapBackward(float u, float v, float &x, float &y)
            {
                using namespace build_warp_maps;

                float x_ = u / cscale - ct[0];
                float y_ = v / cscale - ct[1];

                float z;
                x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * (1 - ct[2]);
                y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * (1 - ct[2]);
                z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * (1 - ct[2]);

                x /= z;
                y /= z;
            }
        };


        class CylindricalMapper
        {
        public:
            static __device__ __forceinline__ void mapBackward(float u, float v, float &x, float &y)
            {
                using namespace build_warp_maps;

                u /= cscale;
                float x_ = ::sinf(u);
                float y_ = v / cscale;
                float z_ = ::cosf(u);

                float z;
                x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * z_;
                y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * z_;
                z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * z_;

                if (z > 0) { x /= z; y /= z; }
                else x = y = -1;
            }
        };


        class SphericalMapper
        {
        public:
            static __device__ __forceinline__ void mapBackward(float u, float v, float &x, float &y)
            {
                using namespace build_warp_maps;

                v /= cscale;
                u /= cscale;

                float sinv = ::sinf(v);
                float x_ = sinv * ::sinf(u);
                float y_ = -::cosf(v);
                float z_ = sinv * ::cosf(u);

                float z;
                x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * z_;
                y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * z_;
                z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * z_;

                if (z > 0) { x /= z; y /= z; }
                else x = y = -1;
            }
        };


        template <typename Mapper>
        __global__ void buildWarpMapsKernel(int tl_u, int tl_v, int cols, int rows,
                                            PtrStepf map_x, PtrStepf map_y)
        {
            int du = blockIdx.x * blockDim.x + threadIdx.x;
            int dv = blockIdx.y * blockDim.y + threadIdx.y;
            if (du < cols && dv < rows)
            {
                float u = tl_u + du;
                float v = tl_v + dv;
                float x, y;
                Mapper::mapBackward(u, v, x, y);
                map_x.ptr(dv)[du] = x;
                map_y.ptr(dv)[du] = y;
            }
        }


        void buildWarpPlaneMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
                                const float k_rinv[9], const float r_kinv[9], const float t[3],
                                float scale, cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ck_rinv, k_rinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cr_kinv, r_kinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ct, t, 3*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cscale, &scale, sizeof(float)));

            int cols = map_x.cols;
            int rows = map_x.rows;

            dim3 threads(32, 8);
            dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

            buildWarpMapsKernel<PlaneMapper><<<grid,threads>>>(tl_u, tl_v, cols, rows, map_x, map_y);
            cudaSafeCall(cudaGetLastError());
            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        void buildWarpCylindricalMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
                                      const float k_rinv[9], const float r_kinv[9], float scale,
                                      cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ck_rinv, k_rinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cr_kinv, r_kinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cscale, &scale, sizeof(float)));

            int cols = map_x.cols;
            int rows = map_x.rows;

            dim3 threads(32, 8);
            dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

            buildWarpMapsKernel<CylindricalMapper><<<grid,threads>>>(tl_u, tl_v, cols, rows, map_x, map_y);
            cudaSafeCall(cudaGetLastError());
            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        void buildWarpSphericalMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
                                    const float k_rinv[9], const float r_kinv[9], float scale,
                                    cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ck_rinv, k_rinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cr_kinv, r_kinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cscale, &scale, sizeof(float)));

            int cols = map_x.cols;
            int rows = map_x.rows;

            dim3 threads(32, 8);
            dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

            buildWarpMapsKernel<SphericalMapper><<<grid,threads>>>(tl_u, tl_v, cols, rows, map_x, map_y);
            cudaSafeCall(cudaGetLastError());
            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    } // namespace imgproc
}}} // namespace cv { namespace cuda { namespace cudev {


#endif /* CUDA_DISABLER */
