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
        __global__ void lbp_classify(const DevMem2D_< ::cv::gpu::device::Stage> stages, const DevMem2Di trees, const DevMem2Db nodes, const DevMem2Df leaves, const DevMem2Di subsets,
            const DevMem2Db integral, float step, int subsetSize, DevMem2D_<int4> objects)
        {
            unsigned int x = threadIdx.x;
            unsigned int y = blockIdx.x;
            int nodeOfs = 0, leafOfs = 0;
            ::cv::gpu::device::Feature feature;

            for (int s = 0; s < stages.cols; s++ )
            {
                ::cv::gpu::device::Stage stage = stages(0, s);
                int sum = 0;
                for (int w = 0; w < stage.ntrees; w++)
                {
                    ::cv::gpu::device::ClNode node = nodes(0, nodeOfs);
                    char c = feature();// TODO: inmplement it
                    const int subsetIdx = (nodeOfs * subsetSize);
                    int idx = subsetIdx + ((c >> 5) & ( 1 << (c & 31)) ? leafOfs : leafOfs + 1);
                    sum += leaves(0, subsets(0, idx) );
                    nodeOfs++;
                    leafOfs += 2;
                }

                if (sum < stage.threshold)
                    return; // nothing matched
                return;//mathed
            }

        }

        void cascadeClassify(const DevMem2Db bstages, const DevMem2Di trees, const DevMem2Db nodes, const DevMem2Df leaves, const DevMem2Di subsets,
            const DevMem2Db integral, int workWidth, int workHeight, int step, int subsetSize, DevMem2D_<int4> objects, int minNeighbors, cudaStream_t stream)
        {
            printf("CascadeClassify");
            int blocks = ceilf(workHeight / (float)step);
            int threads = ceilf(workWidth / (float)step);
            DevMem2D_< ::cv::gpu::device::Stage> stages = DevMem2D_< ::cv::gpu::device::Stage>(bstages);

            lbp_classify<<<blocks, threads>>>(stages, trees, nodes, leaves, subsets, integral, step, subsetSize, objects);
        }
    }
}}}