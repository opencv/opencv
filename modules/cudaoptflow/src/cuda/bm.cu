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
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/reduce.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace optflowbm
{
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_prev(false, cudaFilterModePoint, cudaAddressModeClamp);
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_curr(false, cudaFilterModePoint, cudaAddressModeClamp);

    __device__ int cmpBlocks(int X1, int Y1, int X2, int Y2, int2 blockSize)
    {
        int s = 0;

        for (int y = 0; y < blockSize.y; ++y)
        {
            for (int x = 0; x < blockSize.x; ++x)
                s += ::abs(tex2D(tex_prev, X1 + x, Y1 + y) - tex2D(tex_curr, X2 + x, Y2 + y));
        }

        return s;
    }

    __global__ void calcOptFlowBM(PtrStepSzf velx, PtrStepf vely, const int2 blockSize, const int2 shiftSize, const bool usePrevious,
                                  const int maxX, const int maxY, const int acceptLevel, const int escapeLevel,
                                  const short2* ss, const int ssCount)
    {
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= velx.rows || j >= velx.cols)
            return;

        const int X1 = j * shiftSize.x;
        const int Y1 = i * shiftSize.y;

        const int offX = usePrevious ? __float2int_rn(velx(i, j)) : 0;
        const int offY = usePrevious ? __float2int_rn(vely(i, j)) : 0;

        int X2 = X1 + offX;
        int Y2 = Y1 + offY;

        int dist = numeric_limits<int>::max();

        if (0 <= X2 && X2 <= maxX && 0 <= Y2 && Y2 <= maxY)
            dist = cmpBlocks(X1, Y1, X2, Y2, blockSize);

        int countMin = 1;
        int sumx = offX;
        int sumy = offY;

        if (dist > acceptLevel)
        {
            // do brute-force search
            for (int k = 0; k < ssCount; ++k)
            {
                const short2 ssVal = ss[k];

                const int dx = offX + ssVal.x;
                const int dy = offY + ssVal.y;

                X2 = X1 + dx;
                Y2 = Y1 + dy;

                if (0 <= X2 && X2 <= maxX && 0 <= Y2 && Y2 <= maxY)
                {
                    const int tmpDist = cmpBlocks(X1, Y1, X2, Y2, blockSize);
                    if (tmpDist < acceptLevel)
                    {
                        sumx = dx;
                        sumy = dy;
                        countMin = 1;
                        break;
                    }

                    if (tmpDist < dist)
                    {
                        dist = tmpDist;
                        sumx = dx;
                        sumy = dy;
                        countMin = 1;
                    }
                    else if (tmpDist == dist)
                    {
                        sumx += dx;
                        sumy += dy;
                        countMin++;
                    }
                }
            }

            if (dist > escapeLevel)
            {
                sumx = offX;
                sumy = offY;
                countMin = 1;
            }
        }

        velx(i, j) = static_cast<float>(sumx) / countMin;
        vely(i, j) = static_cast<float>(sumy) / countMin;
    }

    void calc(PtrStepSzb prev, PtrStepSzb curr, PtrStepSzf velx, PtrStepSzf vely, int2 blockSize, int2 shiftSize, bool usePrevious,
              int maxX, int maxY, int acceptLevel, int escapeLevel, const short2* ss, int ssCount, cudaStream_t stream)
    {
        bindTexture(&tex_prev, prev);
        bindTexture(&tex_curr, curr);

        const dim3 block(32, 8);
        const dim3 grid(divUp(velx.cols, block.x), divUp(vely.rows, block.y));

        calcOptFlowBM<<<grid, block, 0, stream>>>(velx, vely, blockSize, shiftSize, usePrevious,
                                                  maxX, maxY, acceptLevel,  escapeLevel, ss, ssCount);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

#endif // !defined CUDA_DISABLER
