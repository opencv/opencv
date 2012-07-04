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

#include <opencv2/gpu/device/lbp.hpp>

namespace cv { namespace gpu { namespace device
{
    namespace lbp
    {
        __global__ void lbp_classify_stump(Stage* stages, int nstages, ClNode* nodes, const float* leaves, const int* subsets, const uchar4* features,
            const DevMem2Di integral, int workWidth, int workHeight, int clWidth, int clHeight, float scale, int step, int subsetSize, DevMem2D_<int4> objects, unsigned int* n)
        {
            int y = threadIdx.x * scale;
            int x = blockIdx.x * scale;

            int current_node = 0;
            int current_leave = 0;

            LBP evaluator;
            for (int s = 0; s < nstages; s++ )
            {
                float sum = 0;
                Stage stage = stages[s];

                for (int t = 0; t < stage.ntrees; t++)
                {
                    ClNode node = nodes[current_node];

                    uchar4 feature = features[node.featureIdx];
                    int c = evaluator(y, x, feature, integral);
                    const int* subsetIdx = subsets + (current_node * subsetSize);

                    int idx =  (subsetIdx[c >> 5] & ( 1 << (c & 31))) ? current_leave : current_leave + 1;
                    sum += leaves[idx];
                    current_node += 1;
                    current_leave += 2;
                }

                if (sum < stage.threshold)
                    return;
            }

            int4 rect;
            rect.x = roundf(x * scale);
            rect.y = roundf(y * scale);
            rect.z = roundf(clWidth);
            rect.w = roundf(clHeight);

            int res = atomicInc(n, 100);
            objects(0, res) = rect;
        }

        classifyStump(const DevMem2Db mstages, const int nstages, const DevMem2Di mnodes, const DevMem2Df mleaves, const DevMem2Di msubsets, const DevMem2Db mfeatures,
                           const DevMem2Di integral, const int workWidth, const int workHeight, const int clWidth, const int clHeight, float scale, int step, int subsetSize,
                           DevMem2D_<int4> objects, unsigned int* classified)
        {
            int blocks  = ceilf(workHeight / (float)step);
            int threads = ceilf(workWidth / (float)step);
            // printf("blocks %d, threads %d\n", blocks, threads);

            Stage* stages = (Stage*)(mstages.ptr());
            ClNode* nodes = (ClNode*)(mnodes.ptr());
            const float* leaves = mleaves.ptr();
            const int* subsets = msubsets.ptr();
            const uchar4* features = (uchar4*)(mfeatures.ptr());

            lbp_classify_stump<<<blocks, threads>>>(stages, nstages, nodes, leaves, subsets, features, integral,
                workWidth, workHeight, clWidth, clHeight, scale, step, subsetSize, objects, classified);
        }
    }
}}}