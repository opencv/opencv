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

////////////////////////////////////////////////////////////////////////////////
//
// NVIDIA CUDA implementation of Viola-Jones Object Detection Framework
//
// The algorithm and code are explained in the upcoming GPU Computing Gems
// chapter in detail:
//
//   Anton Obukhov, "Haar Classifiers for Object Detection with CUDA"
//   PDF URL placeholder
//   email: aobukhov@nvidia.com, devsupport@nvidia.com
//
// Credits for help with the code to:
// Alexey Mendelenko, Cyril Crassin, and Mikhail Smirnov.
//
////////////////////////////////////////////////////////////////////////////////

#if !defined CUDA_DISABLER

#include <algorithm>
#include <cstdio>

#include "NCV.hpp"
#include "NCVAlg.hpp"
#include "NPP_staging/NPP_staging.hpp"
#include "NCVRuntimeTemplates.hpp"
#include "NCVHaarObjectDetection.hpp"
#include "opencv2/gpu/device/warp.hpp"
#include "opencv2/gpu/device/warp_shuffle.hpp"


//==============================================================================
//
// BlockScan file
//
//==============================================================================


NCV_CT_ASSERT(K_WARP_SIZE == 32); //this is required for the manual unroll of the loop in warpScanInclusive


//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE and size is power of 2
__device__ Ncv32u warpScanInclusive(Ncv32u idata, volatile Ncv32u *s_Data)
{
#if __CUDA_ARCH__ >= 300
    const unsigned int laneId = cv::gpu::device::Warp::laneId();

    // scan on shuffl functions
    #pragma unroll
    for (int i = 1; i <= (K_WARP_SIZE / 2); i *= 2)
    {
        const Ncv32u n = cv::gpu::device::shfl_up(idata, i);
        if (laneId >= i)
              idata += n;
    }

    return idata;
#else
    Ncv32u pos = 2 * threadIdx.x - (threadIdx.x & (K_WARP_SIZE - 1));
    s_Data[pos] = 0;
    pos += K_WARP_SIZE;
    s_Data[pos] = idata;

    s_Data[pos] += s_Data[pos - 1];
    s_Data[pos] += s_Data[pos - 2];
    s_Data[pos] += s_Data[pos - 4];
    s_Data[pos] += s_Data[pos - 8];
    s_Data[pos] += s_Data[pos - 16];

    return s_Data[pos];
#endif
}

__device__ __forceinline__ Ncv32u warpScanExclusive(Ncv32u idata, volatile Ncv32u *s_Data)
{
    return warpScanInclusive(idata, s_Data) - idata;
}

template <Ncv32u tiNumScanThreads>
__device__ Ncv32u scan1Inclusive(Ncv32u idata, volatile Ncv32u *s_Data)
{
    if (tiNumScanThreads > K_WARP_SIZE)
    {
        //Bottom-level inclusive warp scan
        Ncv32u warpResult = warpScanInclusive(idata, s_Data);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if( (threadIdx.x & (K_WARP_SIZE - 1)) == (K_WARP_SIZE - 1) )
        {
            s_Data[threadIdx.x >> K_LOG2_WARP_SIZE] = warpResult;
        }

        //wait for warp scans to complete
        __syncthreads();

        if( threadIdx.x < (tiNumScanThreads / K_WARP_SIZE) )
        {
            //grab top warp elements
            Ncv32u val = s_Data[threadIdx.x];
            //calculate exclusive scan and write back to shared memory
            s_Data[threadIdx.x] = warpScanExclusive(val, s_Data);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        return warpResult + s_Data[threadIdx.x >> K_LOG2_WARP_SIZE];
    }
    else
    {
        return warpScanInclusive(idata, s_Data);
    }
}


//==============================================================================
//
// HaarClassifierCascade file
//
//==============================================================================


const Ncv32u MAX_GRID_DIM = 65535;


const Ncv32u NUM_THREADS_ANCHORSPARALLEL = 64;


#define NUM_THREADS_CLASSIFIERPARALLEL_LOG2     6
#define NUM_THREADS_CLASSIFIERPARALLEL          (1 << NUM_THREADS_CLASSIFIERPARALLEL_LOG2)


/** \internal
* Haar features solid array.
*/
texture<uint2, 1, cudaReadModeElementType> texHaarFeatures;


/** \internal
* Haar classifiers flattened trees container.
* Two parts: first contains root nodes, second - nodes that are referred by root nodes.
* Drawback: breaks tree locality (might cause more cache misses
* Advantage: No need to introduce additional 32-bit field to index root nodes offsets
*/
texture<uint4, 1, cudaReadModeElementType> texHaarClassifierNodes;


texture<Ncv32u, 1, cudaReadModeElementType> texIImage;


__device__ HaarStage64 getStage(Ncv32u iStage, HaarStage64 *d_Stages)
{
    return d_Stages[iStage];
}


template <NcvBool tbCacheTextureCascade>
__device__ HaarClassifierNode128 getClassifierNode(Ncv32u iNode, HaarClassifierNode128 *d_ClassifierNodes)
{
    HaarClassifierNode128 tmpNode;
    if (tbCacheTextureCascade)
    {
        tmpNode._ui4 = tex1Dfetch(texHaarClassifierNodes, iNode);
    }
    else
    {
        tmpNode = d_ClassifierNodes[iNode];
    }
    return tmpNode;
}


template <NcvBool tbCacheTextureCascade>
__device__ void getFeature(Ncv32u iFeature, HaarFeature64 *d_Features,
                           Ncv32f *weight,
                           Ncv32u *rectX, Ncv32u *rectY, Ncv32u *rectWidth, Ncv32u *rectHeight)
{
    HaarFeature64 feature;
    if (tbCacheTextureCascade)
    {
        feature._ui2 = tex1Dfetch(texHaarFeatures, iFeature);
    }
    else
    {
        feature = d_Features[iFeature];
    }
    feature.getRect(rectX, rectY, rectWidth, rectHeight);
    *weight = feature.getWeight();
}


template <NcvBool tbCacheTextureIImg>
__device__ Ncv32u getElemIImg(Ncv32u x, Ncv32u *d_IImg)
{
    if (tbCacheTextureIImg)
    {
        return tex1Dfetch(texIImage, x);
    }
    else
    {
        return d_IImg[x];
    }
}


__device__ Ncv32u d_outMaskPosition;


__device__ void compactBlockWriteOutAnchorParallel(Ncv32u threadPassFlag, Ncv32u threadElem, Ncv32u *vectorOut)
{
#if __CUDA_ARCH__ && __CUDA_ARCH__ >= 110

    __shared__ Ncv32u shmem[NUM_THREADS_ANCHORSPARALLEL * 2];
    __shared__ Ncv32u numPassed;
    __shared__ Ncv32u outMaskOffset;

    Ncv32u incScan = scan1Inclusive<NUM_THREADS_ANCHORSPARALLEL>(threadPassFlag, shmem);
    __syncthreads();

    if (threadIdx.x == NUM_THREADS_ANCHORSPARALLEL-1)
    {
        numPassed = incScan;
        outMaskOffset = atomicAdd(&d_outMaskPosition, incScan);
    }

    if (threadPassFlag)
    {
        Ncv32u excScan = incScan - threadPassFlag;
        shmem[excScan] = threadElem;
    }

    __syncthreads();

    if (threadIdx.x < numPassed)
    {
        vectorOut[outMaskOffset + threadIdx.x] = shmem[threadIdx.x];
    }
#endif
}


template <NcvBool tbInitMaskPositively,
          NcvBool tbCacheTextureIImg,
          NcvBool tbCacheTextureCascade,
          NcvBool tbReadPixelIndexFromVector,
          NcvBool tbDoAtomicCompaction>
__global__ void applyHaarClassifierAnchorParallel(Ncv32u *d_IImg, Ncv32u IImgStride,
                                                  Ncv32f *d_weights, Ncv32u weightsStride,
                                                  HaarFeature64 *d_Features, HaarClassifierNode128 *d_ClassifierNodes, HaarStage64 *d_Stages,
                                                  Ncv32u *d_inMask, Ncv32u *d_outMask,
                                                  Ncv32u mask1Dlen, Ncv32u mask2Dstride,
                                                  NcvSize32u anchorsRoi, Ncv32u startStageInc, Ncv32u endStageExc, Ncv32f scaleArea)
{
    Ncv32u y_offs;
    Ncv32u x_offs;
    Ncv32u maskOffset;
    Ncv32u outMaskVal;

    NcvBool bInactiveThread = false;

    if (tbReadPixelIndexFromVector)
    {
        maskOffset = (MAX_GRID_DIM * blockIdx.y + blockIdx.x) * NUM_THREADS_ANCHORSPARALLEL + threadIdx.x;

        if (maskOffset >= mask1Dlen)
        {
            if (tbDoAtomicCompaction) bInactiveThread = true; else return;
        }

        if (!tbDoAtomicCompaction || tbDoAtomicCompaction && !bInactiveThread)
        {
            outMaskVal = d_inMask[maskOffset];
            y_offs = outMaskVal >> 16;
            x_offs = outMaskVal & 0xFFFF;
        }
    }
    else
    {
        y_offs = blockIdx.y;
        x_offs = blockIdx.x * NUM_THREADS_ANCHORSPARALLEL + threadIdx.x;

        if (x_offs >= mask2Dstride)
        {
            if (tbDoAtomicCompaction) bInactiveThread = true; else return;
        }

        if (!tbDoAtomicCompaction || tbDoAtomicCompaction && !bInactiveThread)
        {
            maskOffset = y_offs * mask2Dstride + x_offs;

            if ((x_offs >= anchorsRoi.width) ||
                (!tbInitMaskPositively &&
                 d_inMask != d_outMask &&
                 d_inMask[maskOffset] == OBJDET_MASK_ELEMENT_INVALID_32U))
            {
                if (tbDoAtomicCompaction)
                {
                    bInactiveThread = true;
                }
                else
                {
                    d_outMask[maskOffset] = OBJDET_MASK_ELEMENT_INVALID_32U;
                    return;
                }
            }

            outMaskVal = (y_offs << 16) | x_offs;
        }
    }

    NcvBool bPass = true;

    if (!tbDoAtomicCompaction || tbDoAtomicCompaction)
    {
        Ncv32f pixelStdDev = 0.0f;

        if (!bInactiveThread)
            pixelStdDev = d_weights[y_offs * weightsStride + x_offs];

        for (Ncv32u iStage = startStageInc; iStage < endStageExc; iStage++)
        {
            Ncv32f curStageSum = 0.0f;

            HaarStage64 curStage = getStage(iStage, d_Stages);
            Ncv32u numRootNodesInStage = curStage.getNumClassifierRootNodes();
            Ncv32u curRootNodeOffset = curStage.getStartClassifierRootNodeOffset();
            Ncv32f stageThreshold = curStage.getStageThreshold();

            while (numRootNodesInStage--)
            {
                NcvBool bMoreNodesToTraverse = true;
                Ncv32u iNode = curRootNodeOffset;

                if (bPass && !bInactiveThread)
                {
                    while (bMoreNodesToTraverse)
                    {
                        HaarClassifierNode128 curNode = getClassifierNode<tbCacheTextureCascade>(iNode, d_ClassifierNodes);
                        HaarFeatureDescriptor32 featuresDesc = curNode.getFeatureDesc();
                        Ncv32u curNodeFeaturesNum = featuresDesc.getNumFeatures();
                        Ncv32u iFeature = featuresDesc.getFeaturesOffset();

                        Ncv32f curNodeVal = 0.0f;

                        for (Ncv32u iRect=0; iRect<curNodeFeaturesNum; iRect++)
                        {
                            Ncv32f rectWeight;
                            Ncv32u rectX, rectY, rectWidth, rectHeight;
                            getFeature<tbCacheTextureCascade>
                                (iFeature + iRect, d_Features,
                                &rectWeight, &rectX, &rectY, &rectWidth, &rectHeight);

                            Ncv32u iioffsTL = (y_offs + rectY) * IImgStride + (x_offs + rectX);
                            Ncv32u iioffsTR = iioffsTL + rectWidth;
                            Ncv32u iioffsBL = iioffsTL + rectHeight * IImgStride;
                            Ncv32u iioffsBR = iioffsBL + rectWidth;

                            Ncv32u rectSum = getElemIImg<tbCacheTextureIImg>(iioffsBR, d_IImg) -
                                             getElemIImg<tbCacheTextureIImg>(iioffsBL, d_IImg) +
                                             getElemIImg<tbCacheTextureIImg>(iioffsTL, d_IImg) -
                                             getElemIImg<tbCacheTextureIImg>(iioffsTR, d_IImg);

    #if defined CPU_FP_COMPLIANCE || defined DISABLE_MAD_SELECTIVELY
                        curNodeVal += __fmul_rn((Ncv32f)rectSum, rectWeight);
    #else
                        curNodeVal += (Ncv32f)rectSum * rectWeight;
    #endif
                        }

                        HaarClassifierNodeDescriptor32 nodeLeft = curNode.getLeftNodeDesc();
                        HaarClassifierNodeDescriptor32 nodeRight = curNode.getRightNodeDesc();
                        Ncv32f nodeThreshold = curNode.getThreshold();

                        HaarClassifierNodeDescriptor32 nextNodeDescriptor;
                        NcvBool nextNodeIsLeaf;

                        if (curNodeVal < scaleArea * pixelStdDev * nodeThreshold)
                        {
                            nextNodeDescriptor = nodeLeft;
                            nextNodeIsLeaf = featuresDesc.isLeftNodeLeaf();
                        }
                        else
                        {
                            nextNodeDescriptor = nodeRight;
                            nextNodeIsLeaf = featuresDesc.isRightNodeLeaf();
                        }

                        if (nextNodeIsLeaf)
                        {
                            Ncv32f tmpLeafValue = nextNodeDescriptor.getLeafValue();
                            curStageSum += tmpLeafValue;
                            bMoreNodesToTraverse = false;
                        }
                        else
                        {
                            iNode = nextNodeDescriptor.getNextNodeOffset();
                        }
                    }
                }

                __syncthreads();
                curRootNodeOffset++;
            }

            if (curStageSum < stageThreshold)
            {
                bPass = false;
                outMaskVal = OBJDET_MASK_ELEMENT_INVALID_32U;
            }
        }
    }

    __syncthreads();

    if (!tbDoAtomicCompaction)
    {
        if (!tbReadPixelIndexFromVector ||
            (tbReadPixelIndexFromVector && (!bPass || d_inMask != d_outMask)))
        {
            d_outMask[maskOffset] = outMaskVal;
        }
    }
    else
    {
        compactBlockWriteOutAnchorParallel(bPass && !bInactiveThread,
                                           outMaskVal,
                                           d_outMask);
    }
}


template <NcvBool tbCacheTextureIImg,
          NcvBool tbCacheTextureCascade,
          NcvBool tbDoAtomicCompaction>
__global__ void applyHaarClassifierClassifierParallel(Ncv32u *d_IImg, Ncv32u IImgStride,
                                                      Ncv32f *d_weights, Ncv32u weightsStride,
                                                      HaarFeature64 *d_Features, HaarClassifierNode128 *d_ClassifierNodes, HaarStage64 *d_Stages,
                                                      Ncv32u *d_inMask, Ncv32u *d_outMask,
                                                      Ncv32u mask1Dlen, Ncv32u mask2Dstride,
                                                      NcvSize32u anchorsRoi, Ncv32u startStageInc, Ncv32u endStageExc, Ncv32f scaleArea)
{
    Ncv32u maskOffset = MAX_GRID_DIM * blockIdx.y + blockIdx.x;

    if (maskOffset >= mask1Dlen)
    {
        return;
    }

    Ncv32u outMaskVal = d_inMask[maskOffset];
    Ncv32u y_offs = outMaskVal >> 16;
    Ncv32u x_offs = outMaskVal & 0xFFFF;

    Ncv32f pixelStdDev = d_weights[y_offs * weightsStride + x_offs];
    NcvBool bPass = true;

    for (Ncv32u iStage = startStageInc; iStage<endStageExc; iStage++)
    {
        //this variable is subject to reduction
        Ncv32f curStageSum = 0.0f;

        HaarStage64 curStage = getStage(iStage, d_Stages);
        Ncv32s numRootNodesInStage = curStage.getNumClassifierRootNodes();
        Ncv32u curRootNodeOffset = curStage.getStartClassifierRootNodeOffset() + threadIdx.x;
        Ncv32f stageThreshold = curStage.getStageThreshold();

        Ncv32u numRootChunks = (numRootNodesInStage + NUM_THREADS_CLASSIFIERPARALLEL - 1) >> NUM_THREADS_CLASSIFIERPARALLEL_LOG2;

        for (Ncv32u chunkId=0; chunkId<numRootChunks; chunkId++)
        {
            NcvBool bMoreNodesToTraverse = true;

            if (chunkId * NUM_THREADS_CLASSIFIERPARALLEL + threadIdx.x < numRootNodesInStage)
            {
                Ncv32u iNode = curRootNodeOffset;

                while (bMoreNodesToTraverse)
                {
                    HaarClassifierNode128 curNode = getClassifierNode<tbCacheTextureCascade>(iNode, d_ClassifierNodes);
                    HaarFeatureDescriptor32 featuresDesc = curNode.getFeatureDesc();
                    Ncv32u curNodeFeaturesNum = featuresDesc.getNumFeatures();
                    Ncv32u iFeature = featuresDesc.getFeaturesOffset();

                    Ncv32f curNodeVal = 0.0f;
                    //TODO: fetch into shmem if size suffices. Shmem can be shared with reduce
                    for (Ncv32u iRect=0; iRect<curNodeFeaturesNum; iRect++)
                    {
                        Ncv32f rectWeight;
                        Ncv32u rectX, rectY, rectWidth, rectHeight;
                        getFeature<tbCacheTextureCascade>
                            (iFeature + iRect, d_Features,
                            &rectWeight, &rectX, &rectY, &rectWidth, &rectHeight);

                        Ncv32u iioffsTL = (y_offs + rectY) * IImgStride + (x_offs + rectX);
                        Ncv32u iioffsTR = iioffsTL + rectWidth;
                        Ncv32u iioffsBL = iioffsTL + rectHeight * IImgStride;
                        Ncv32u iioffsBR = iioffsBL + rectWidth;

                        Ncv32u rectSum = getElemIImg<tbCacheTextureIImg>(iioffsBR, d_IImg) -
                                         getElemIImg<tbCacheTextureIImg>(iioffsBL, d_IImg) +
                                         getElemIImg<tbCacheTextureIImg>(iioffsTL, d_IImg) -
                                         getElemIImg<tbCacheTextureIImg>(iioffsTR, d_IImg);

#if defined CPU_FP_COMPLIANCE || defined DISABLE_MAD_SELECTIVELY
                        curNodeVal += __fmul_rn((Ncv32f)rectSum, rectWeight);
#else
                        curNodeVal += (Ncv32f)rectSum * rectWeight;
#endif
                    }

                    HaarClassifierNodeDescriptor32 nodeLeft = curNode.getLeftNodeDesc();
                    HaarClassifierNodeDescriptor32 nodeRight = curNode.getRightNodeDesc();
                    Ncv32f nodeThreshold = curNode.getThreshold();

                    HaarClassifierNodeDescriptor32 nextNodeDescriptor;
                    NcvBool nextNodeIsLeaf;

                    if (curNodeVal < scaleArea * pixelStdDev * nodeThreshold)
                    {
                        nextNodeDescriptor = nodeLeft;
                        nextNodeIsLeaf = featuresDesc.isLeftNodeLeaf();
                    }
                    else
                    {
                        nextNodeDescriptor = nodeRight;
                        nextNodeIsLeaf = featuresDesc.isRightNodeLeaf();
                    }

                    if (nextNodeIsLeaf)
                    {
                        Ncv32f tmpLeafValue = nextNodeDescriptor.getLeafValue();
                        curStageSum += tmpLeafValue;
                        bMoreNodesToTraverse = false;
                    }
                    else
                    {
                        iNode = nextNodeDescriptor.getNextNodeOffset();
                    }
                }
            }
            __syncthreads();

            curRootNodeOffset += NUM_THREADS_CLASSIFIERPARALLEL;
        }

        Ncv32f finalStageSum = subReduce<Ncv32f, functorAddValues<Ncv32f>, NUM_THREADS_CLASSIFIERPARALLEL>(curStageSum);

        if (finalStageSum < stageThreshold)
        {
            bPass = false;
            outMaskVal = OBJDET_MASK_ELEMENT_INVALID_32U;
            break;
        }
    }

    if (!tbDoAtomicCompaction)
    {
        if (!bPass || d_inMask != d_outMask)
        {
            if (!threadIdx.x)
            {
                d_outMask[maskOffset] = outMaskVal;
            }
        }
    }
    else
    {
#if __CUDA_ARCH__ && __CUDA_ARCH__ >= 110
        if (bPass && !threadIdx.x)
        {
            Ncv32u outMaskOffset = atomicAdd(&d_outMaskPosition, 1);
            d_outMask[outMaskOffset] = outMaskVal;
        }
#endif
    }
}


template <NcvBool tbMaskByInmask,
          NcvBool tbDoAtomicCompaction>
__global__ void initializeMaskVector(Ncv32u *d_inMask, Ncv32u *d_outMask,
                                     Ncv32u mask1Dlen, Ncv32u mask2Dstride,
                                     NcvSize32u anchorsRoi, Ncv32u step)
{
    Ncv32u y_offs = blockIdx.y;
    Ncv32u x_offs = blockIdx.x * NUM_THREADS_ANCHORSPARALLEL + threadIdx.x;
    Ncv32u outMaskOffset = y_offs * gridDim.x * blockDim.x + x_offs;

    Ncv32u y_offs_upsc = step * y_offs;
    Ncv32u x_offs_upsc = step * x_offs;
    Ncv32u inMaskOffset = y_offs_upsc * mask2Dstride + x_offs_upsc;

    Ncv32u outElem = OBJDET_MASK_ELEMENT_INVALID_32U;

    if (x_offs_upsc < anchorsRoi.width &&
        (!tbMaskByInmask || d_inMask[inMaskOffset] != OBJDET_MASK_ELEMENT_INVALID_32U))
    {
        outElem = (y_offs_upsc << 16) | x_offs_upsc;
    }

    if (!tbDoAtomicCompaction)
    {
        d_outMask[outMaskOffset] = outElem;
    }
    else
    {
        compactBlockWriteOutAnchorParallel(outElem != OBJDET_MASK_ELEMENT_INVALID_32U,
                                           outElem,
                                           d_outMask);
    }
}


struct applyHaarClassifierAnchorParallelFunctor
{
    dim3 gridConf, blockConf;
    cudaStream_t cuStream;

    //Kernel arguments are stored as members;
    Ncv32u *d_IImg;
    Ncv32u IImgStride;
    Ncv32f *d_weights;
    Ncv32u weightsStride;
    HaarFeature64 *d_Features;
    HaarClassifierNode128 *d_ClassifierNodes;
    HaarStage64 *d_Stages;
    Ncv32u *d_inMask;
    Ncv32u *d_outMask;
    Ncv32u mask1Dlen;
    Ncv32u mask2Dstride;
    NcvSize32u anchorsRoi;
    Ncv32u startStageInc;
    Ncv32u endStageExc;
    Ncv32f scaleArea;

    //Arguments are passed through the constructor
    applyHaarClassifierAnchorParallelFunctor(dim3 _gridConf, dim3 _blockConf, cudaStream_t _cuStream,
                                             Ncv32u *_d_IImg, Ncv32u _IImgStride,
                                             Ncv32f *_d_weights, Ncv32u _weightsStride,
                                             HaarFeature64 *_d_Features, HaarClassifierNode128 *_d_ClassifierNodes, HaarStage64 *_d_Stages,
                                             Ncv32u *_d_inMask, Ncv32u *_d_outMask,
                                             Ncv32u _mask1Dlen, Ncv32u _mask2Dstride,
                                             NcvSize32u _anchorsRoi, Ncv32u _startStageInc,
                                             Ncv32u _endStageExc, Ncv32f _scaleArea) :
    gridConf(_gridConf),
    blockConf(_blockConf),
    cuStream(_cuStream),
    d_IImg(_d_IImg),
    IImgStride(_IImgStride),
    d_weights(_d_weights),
    weightsStride(_weightsStride),
    d_Features(_d_Features),
    d_ClassifierNodes(_d_ClassifierNodes),
    d_Stages(_d_Stages),
    d_inMask(_d_inMask),
    d_outMask(_d_outMask),
    mask1Dlen(_mask1Dlen),
    mask2Dstride(_mask2Dstride),
    anchorsRoi(_anchorsRoi),
    startStageInc(_startStageInc),
    endStageExc(_endStageExc),
    scaleArea(_scaleArea)
    {}

    template<class TList>
    void call(TList tl)
    {
        (void)tl;
        applyHaarClassifierAnchorParallel <
            Loki::TL::TypeAt<TList, 0>::Result::value,
            Loki::TL::TypeAt<TList, 1>::Result::value,
            Loki::TL::TypeAt<TList, 2>::Result::value,
            Loki::TL::TypeAt<TList, 3>::Result::value,
            Loki::TL::TypeAt<TList, 4>::Result::value >
            <<<gridConf, blockConf, 0, cuStream>>>
            (d_IImg, IImgStride,
            d_weights, weightsStride,
            d_Features, d_ClassifierNodes, d_Stages,
            d_inMask, d_outMask,
            mask1Dlen, mask2Dstride,
            anchorsRoi, startStageInc,
            endStageExc, scaleArea);
    }
};


void applyHaarClassifierAnchorParallelDynTemplate(NcvBool tbInitMaskPositively,
                                                  NcvBool tbCacheTextureIImg,
                                                  NcvBool tbCacheTextureCascade,
                                                  NcvBool tbReadPixelIndexFromVector,
                                                  NcvBool tbDoAtomicCompaction,

                                                  dim3 gridConf, dim3 blockConf, cudaStream_t cuStream,

                                                  Ncv32u *d_IImg, Ncv32u IImgStride,
                                                  Ncv32f *d_weights, Ncv32u weightsStride,
                                                  HaarFeature64 *d_Features, HaarClassifierNode128 *d_ClassifierNodes, HaarStage64 *d_Stages,
                                                  Ncv32u *d_inMask, Ncv32u *d_outMask,
                                                  Ncv32u mask1Dlen, Ncv32u mask2Dstride,
                                                  NcvSize32u anchorsRoi, Ncv32u startStageInc,
                                                  Ncv32u endStageExc, Ncv32f scaleArea)
{

    applyHaarClassifierAnchorParallelFunctor functor(gridConf, blockConf, cuStream,
                                                     d_IImg, IImgStride,
                                                     d_weights, weightsStride,
                                                     d_Features, d_ClassifierNodes, d_Stages,
                                                     d_inMask, d_outMask,
                                                     mask1Dlen, mask2Dstride,
                                                     anchorsRoi, startStageInc,
                                                     endStageExc, scaleArea);

    //Second parameter is the number of "dynamic" template parameters
    NCVRuntimeTemplateBool::KernelCaller<Loki::NullType, 5, applyHaarClassifierAnchorParallelFunctor>
        ::call( &functor,
                tbInitMaskPositively,
                tbCacheTextureIImg,
                tbCacheTextureCascade,
                tbReadPixelIndexFromVector,
                tbDoAtomicCompaction);
}


struct applyHaarClassifierClassifierParallelFunctor
{
    dim3 gridConf, blockConf;
    cudaStream_t cuStream;

    //Kernel arguments are stored as members;
    Ncv32u *d_IImg;
    Ncv32u IImgStride;
    Ncv32f *d_weights;
    Ncv32u weightsStride;
    HaarFeature64 *d_Features;
    HaarClassifierNode128 *d_ClassifierNodes;
    HaarStage64 *d_Stages;
    Ncv32u *d_inMask;
    Ncv32u *d_outMask;
    Ncv32u mask1Dlen;
    Ncv32u mask2Dstride;
    NcvSize32u anchorsRoi;
    Ncv32u startStageInc;
    Ncv32u endStageExc;
    Ncv32f scaleArea;

    //Arguments are passed through the constructor
    applyHaarClassifierClassifierParallelFunctor(dim3 _gridConf, dim3 _blockConf, cudaStream_t _cuStream,
                                                 Ncv32u *_d_IImg, Ncv32u _IImgStride,
                                                 Ncv32f *_d_weights, Ncv32u _weightsStride,
                                                 HaarFeature64 *_d_Features, HaarClassifierNode128 *_d_ClassifierNodes, HaarStage64 *_d_Stages,
                                                 Ncv32u *_d_inMask, Ncv32u *_d_outMask,
                                                 Ncv32u _mask1Dlen, Ncv32u _mask2Dstride,
                                                 NcvSize32u _anchorsRoi, Ncv32u _startStageInc,
                                                 Ncv32u _endStageExc, Ncv32f _scaleArea) :
    gridConf(_gridConf),
    blockConf(_blockConf),
    cuStream(_cuStream),
    d_IImg(_d_IImg),
    IImgStride(_IImgStride),
    d_weights(_d_weights),
    weightsStride(_weightsStride),
    d_Features(_d_Features),
    d_ClassifierNodes(_d_ClassifierNodes),
    d_Stages(_d_Stages),
    d_inMask(_d_inMask),
    d_outMask(_d_outMask),
    mask1Dlen(_mask1Dlen),
    mask2Dstride(_mask2Dstride),
    anchorsRoi(_anchorsRoi),
    startStageInc(_startStageInc),
    endStageExc(_endStageExc),
    scaleArea(_scaleArea)
    {}

    template<class TList>
    void call(TList tl)
    {
        (void)tl;
        applyHaarClassifierClassifierParallel <
            Loki::TL::TypeAt<TList, 0>::Result::value,
            Loki::TL::TypeAt<TList, 1>::Result::value,
            Loki::TL::TypeAt<TList, 2>::Result::value >
            <<<gridConf, blockConf, 0, cuStream>>>
            (d_IImg, IImgStride,
            d_weights, weightsStride,
            d_Features, d_ClassifierNodes, d_Stages,
            d_inMask, d_outMask,
            mask1Dlen, mask2Dstride,
            anchorsRoi, startStageInc,
            endStageExc, scaleArea);
    }
};


void applyHaarClassifierClassifierParallelDynTemplate(NcvBool tbCacheTextureIImg,
                                                      NcvBool tbCacheTextureCascade,
                                                      NcvBool tbDoAtomicCompaction,

                                                      dim3 gridConf, dim3 blockConf, cudaStream_t cuStream,

                                                      Ncv32u *d_IImg, Ncv32u IImgStride,
                                                      Ncv32f *d_weights, Ncv32u weightsStride,
                                                      HaarFeature64 *d_Features, HaarClassifierNode128 *d_ClassifierNodes, HaarStage64 *d_Stages,
                                                      Ncv32u *d_inMask, Ncv32u *d_outMask,
                                                      Ncv32u mask1Dlen, Ncv32u mask2Dstride,
                                                      NcvSize32u anchorsRoi, Ncv32u startStageInc,
                                                      Ncv32u endStageExc, Ncv32f scaleArea)
{
    applyHaarClassifierClassifierParallelFunctor functor(gridConf, blockConf, cuStream,
                                                         d_IImg, IImgStride,
                                                         d_weights, weightsStride,
                                                         d_Features, d_ClassifierNodes, d_Stages,
                                                         d_inMask, d_outMask,
                                                         mask1Dlen, mask2Dstride,
                                                         anchorsRoi, startStageInc,
                                                         endStageExc, scaleArea);

    //Second parameter is the number of "dynamic" template parameters
    NCVRuntimeTemplateBool::KernelCaller<Loki::NullType, 3, applyHaarClassifierClassifierParallelFunctor>
        ::call( &functor,
                tbCacheTextureIImg,
                tbCacheTextureCascade,
                tbDoAtomicCompaction);
}


struct initializeMaskVectorFunctor
{
    dim3 gridConf, blockConf;
    cudaStream_t cuStream;

    //Kernel arguments are stored as members;
    Ncv32u *d_inMask;
    Ncv32u *d_outMask;
    Ncv32u mask1Dlen;
    Ncv32u mask2Dstride;
    NcvSize32u anchorsRoi;
    Ncv32u step;

    //Arguments are passed through the constructor
    initializeMaskVectorFunctor(dim3 _gridConf, dim3 _blockConf, cudaStream_t _cuStream,
                                Ncv32u *_d_inMask, Ncv32u *_d_outMask,
                                Ncv32u _mask1Dlen, Ncv32u _mask2Dstride,
                                NcvSize32u _anchorsRoi, Ncv32u _step) :
    gridConf(_gridConf),
    blockConf(_blockConf),
    cuStream(_cuStream),
    d_inMask(_d_inMask),
    d_outMask(_d_outMask),
    mask1Dlen(_mask1Dlen),
    mask2Dstride(_mask2Dstride),
    anchorsRoi(_anchorsRoi),
    step(_step)
    {}

    template<class TList>
    void call(TList tl)
    {
        (void)tl;
        initializeMaskVector <
            Loki::TL::TypeAt<TList, 0>::Result::value,
            Loki::TL::TypeAt<TList, 1>::Result::value >
            <<<gridConf, blockConf, 0, cuStream>>>
            (d_inMask, d_outMask,
             mask1Dlen, mask2Dstride,
             anchorsRoi, step);
    }
};


void initializeMaskVectorDynTemplate(NcvBool tbMaskByInmask,
                                     NcvBool tbDoAtomicCompaction,

                                     dim3 gridConf, dim3 blockConf, cudaStream_t cuStream,

                                     Ncv32u *d_inMask, Ncv32u *d_outMask,
                                     Ncv32u mask1Dlen, Ncv32u mask2Dstride,
                                     NcvSize32u anchorsRoi, Ncv32u step)
{
    initializeMaskVectorFunctor functor(gridConf, blockConf, cuStream,
                                        d_inMask, d_outMask,
                                        mask1Dlen, mask2Dstride,
                                        anchorsRoi, step);

    //Second parameter is the number of "dynamic" template parameters
    NCVRuntimeTemplateBool::KernelCaller<Loki::NullType, 2, initializeMaskVectorFunctor>
        ::call( &functor,
                tbMaskByInmask,
                tbDoAtomicCompaction);
}


Ncv32u getStageNumWithNotLessThanNclassifiers(Ncv32u N, HaarClassifierCascadeDescriptor &haar,
                                              NCVVector<HaarStage64> &h_HaarStages)
{
    Ncv32u i = 0;
    for (; i<haar.NumStages; i++)
    {
        if (h_HaarStages.ptr()[i].getNumClassifierRootNodes() >= N)
        {
            break;
        }
    }
    return i;
}


NCVStatus ncvApplyHaarClassifierCascade_device(NCVMatrix<Ncv32u> &integral,
                                               NCVMatrix<Ncv32f> &d_weights,
                                               NCVMatrixAlloc<Ncv32u> &d_pixelMask,
                                               Ncv32u &numDetections,
                                               HaarClassifierCascadeDescriptor &haar,
                                               NCVVector<HaarStage64> &h_HaarStages,
                                               NCVVector<HaarStage64> &d_HaarStages,
                                               NCVVector<HaarClassifierNode128> &d_HaarNodes,
                                               NCVVector<HaarFeature64> &d_HaarFeatures,
                                               NcvBool bMaskElements,
                                               NcvSize32u anchorsRoi,
                                               Ncv32u pixelStep,
                                               Ncv32f scaleArea,
                                               INCVMemAllocator &gpuAllocator,
                                               INCVMemAllocator &cpuAllocator,
                                               cudaDeviceProp &devProp,
                                               cudaStream_t cuStream)
{
    ncvAssertReturn(integral.memType() == d_weights.memType()&&
                    integral.memType() == d_pixelMask.memType() &&
                    integral.memType() == gpuAllocator.memType() &&
                   (integral.memType() == NCVMemoryTypeDevice ||
                    integral.memType() == NCVMemoryTypeNone), NCV_MEM_RESIDENCE_ERROR);

    ncvAssertReturn(d_HaarStages.memType() == d_HaarNodes.memType() &&
                    d_HaarStages.memType() == d_HaarFeatures.memType() &&
                     (d_HaarStages.memType() == NCVMemoryTypeDevice ||
                      d_HaarStages.memType() == NCVMemoryTypeNone), NCV_MEM_RESIDENCE_ERROR);

    ncvAssertReturn(h_HaarStages.memType() != NCVMemoryTypeDevice, NCV_MEM_RESIDENCE_ERROR);

    ncvAssertReturn(gpuAllocator.isInitialized() && cpuAllocator.isInitialized(), NCV_ALLOCATOR_NOT_INITIALIZED);

    ncvAssertReturn((integral.ptr() != NULL && d_weights.ptr() != NULL && d_pixelMask.ptr() != NULL &&
                     h_HaarStages.ptr() != NULL && d_HaarStages.ptr() != NULL && d_HaarNodes.ptr() != NULL &&
                     d_HaarFeatures.ptr() != NULL) || gpuAllocator.isCounting(), NCV_NULL_PTR);

    ncvAssertReturn(anchorsRoi.width > 0 && anchorsRoi.height > 0 &&
                    d_pixelMask.width() >= anchorsRoi.width && d_pixelMask.height() >= anchorsRoi.height &&
                    d_weights.width() >= anchorsRoi.width && d_weights.height() >= anchorsRoi.height &&
                    integral.width() >= anchorsRoi.width + haar.ClassifierSize.width &&
                    integral.height() >= anchorsRoi.height + haar.ClassifierSize.height, NCV_DIMENSIONS_INVALID);

    ncvAssertReturn(scaleArea > 0, NCV_INVALID_SCALE);

    ncvAssertReturn(d_HaarStages.length() >= haar.NumStages &&
                    d_HaarNodes.length() >= haar.NumClassifierTotalNodes &&
                    d_HaarFeatures.length() >= haar.NumFeatures &&
                    d_HaarStages.length() == h_HaarStages.length() &&
                    haar.NumClassifierRootNodes <= haar.NumClassifierTotalNodes, NCV_DIMENSIONS_INVALID);

    ncvAssertReturn(haar.bNeedsTiltedII == false || gpuAllocator.isCounting(), NCV_NOIMPL_HAAR_TILTED_FEATURES);

    ncvAssertReturn(pixelStep == 1 || pixelStep == 2, NCV_HAAR_INVALID_PIXEL_STEP);

    NCV_SET_SKIP_COND(gpuAllocator.isCounting());

#if defined _SELF_TEST_

    NCVStatus ncvStat;

    NCVMatrixAlloc<Ncv32u> h_integralImage(cpuAllocator, integral.width, integral.height, integral.pitch);
    ncvAssertReturn(h_integralImage.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVMatrixAlloc<Ncv32f> h_weights(cpuAllocator, d_weights.width, d_weights.height, d_weights.pitch);
    ncvAssertReturn(h_weights.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVMatrixAlloc<Ncv32u> h_pixelMask(cpuAllocator, d_pixelMask.width, d_pixelMask.height, d_pixelMask.pitch);
    ncvAssertReturn(h_pixelMask.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVVectorAlloc<HaarClassifierNode128> h_HaarNodes(cpuAllocator, d_HaarNodes.length);
    ncvAssertReturn(h_HaarNodes.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVVectorAlloc<HaarFeature64> h_HaarFeatures(cpuAllocator, d_HaarFeatures.length);
    ncvAssertReturn(h_HaarFeatures.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCVMatrixAlloc<Ncv32u> h_pixelMask_d(cpuAllocator, d_pixelMask.width, d_pixelMask.height, d_pixelMask.pitch);
    ncvAssertReturn(h_pixelMask_d.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCV_SKIP_COND_BEGIN

    ncvStat = d_pixelMask.copySolid(h_pixelMask, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvStat = integral.copySolid(h_integralImage, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvStat = d_weights.copySolid(h_weights, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvStat = d_HaarNodes.copySolid(h_HaarNodes, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvStat = d_HaarFeatures.copySolid(h_HaarFeatures, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);

    for (Ncv32u i=0; i<(Ncv32u)anchorsRoi.height; i++)
    {
        for (Ncv32u j=0; j<d_pixelMask.stride(); j++)
        {
            if ((i%pixelStep==0) && (j%pixelStep==0) && (j<(Ncv32u)anchorsRoi.width))
            {
                if (!bMaskElements || h_pixelMask.ptr[i*d_pixelMask.stride()+j] != OBJDET_MASK_ELEMENT_INVALID_32U)
                {
                    h_pixelMask.ptr[i*d_pixelMask.stride()+j] = (i << 16) | j;
                }
            }
            else
            {
                h_pixelMask.ptr[i*d_pixelMask.stride()+j] = OBJDET_MASK_ELEMENT_INVALID_32U;
            }
        }
    }

    NCV_SKIP_COND_END

#endif

    NCVVectorReuse<Ncv32u> d_vecPixelMask(d_pixelMask.getSegment(), anchorsRoi.height * d_pixelMask.stride());
    ncvAssertReturn(d_vecPixelMask.isMemReused(), NCV_ALLOCATOR_BAD_REUSE);

    NCVVectorAlloc<Ncv32u> d_vecPixelMaskTmp(gpuAllocator, static_cast<Ncv32u>(d_vecPixelMask.length()));
    ncvAssertReturn(d_vecPixelMaskTmp.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCVVectorAlloc<Ncv32u> hp_pool32u(cpuAllocator, 2);
    ncvAssertReturn(hp_pool32u.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    Ncv32u *hp_zero = &hp_pool32u.ptr()[0];
    Ncv32u *hp_numDet = &hp_pool32u.ptr()[1];

    NCV_SKIP_COND_BEGIN
    *hp_zero = 0;
    *hp_numDet = 0;
    NCV_SKIP_COND_END

    Ncv32f scaleAreaPixels = scaleArea * ((haar.ClassifierSize.width - 2*HAAR_STDDEV_BORDER) *
                                          (haar.ClassifierSize.height - 2*HAAR_STDDEV_BORDER));

    NcvBool bTexCacheCascade = devProp.major < 2;
    NcvBool bTexCacheIImg = true; //this works better even on Fermi so far
    NcvBool bDoAtomicCompaction = devProp.major >= 2 || (devProp.major == 1 && devProp.minor >= 3);

    NCVVector<Ncv32u> *d_ptrNowData = &d_vecPixelMask;
    NCVVector<Ncv32u> *d_ptrNowTmp = &d_vecPixelMaskTmp;

    Ncv32u szNppCompactTmpBuf;
    nppsStCompactGetSize_32u(static_cast<Ncv32u>(d_vecPixelMask.length()), &szNppCompactTmpBuf, devProp);
    if (bDoAtomicCompaction)
    {
        szNppCompactTmpBuf = 0;
    }
    NCVVectorAlloc<Ncv8u> d_tmpBufCompact(gpuAllocator, szNppCompactTmpBuf);

    NCV_SKIP_COND_BEGIN

    if (bTexCacheIImg)
    {
        cudaChannelFormatDesc cfdTexIImage;
        cfdTexIImage = cudaCreateChannelDesc<Ncv32u>();

        size_t alignmentOffset;
        ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, texIImage, integral.ptr(), cfdTexIImage,
            (anchorsRoi.height + haar.ClassifierSize.height) * integral.pitch()), NCV_CUDA_ERROR);
        ncvAssertReturn(alignmentOffset==0, NCV_TEXTURE_BIND_ERROR);
    }

    if (bTexCacheCascade)
    {
        cudaChannelFormatDesc cfdTexHaarFeatures;
        cudaChannelFormatDesc cfdTexHaarClassifierNodes;
        cfdTexHaarFeatures = cudaCreateChannelDesc<uint2>();
        cfdTexHaarClassifierNodes = cudaCreateChannelDesc<uint4>();

        size_t alignmentOffset;
        ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, texHaarFeatures,
            d_HaarFeatures.ptr(), cfdTexHaarFeatures,sizeof(HaarFeature64) * haar.NumFeatures), NCV_CUDA_ERROR);
        ncvAssertReturn(alignmentOffset==0, NCV_TEXTURE_BIND_ERROR);
        ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, texHaarClassifierNodes,
            d_HaarNodes.ptr(), cfdTexHaarClassifierNodes, sizeof(HaarClassifierNode128) * haar.NumClassifierTotalNodes), NCV_CUDA_ERROR);
        ncvAssertReturn(alignmentOffset==0, NCV_TEXTURE_BIND_ERROR);
    }

    Ncv32u stageStartAnchorParallel = 0;
    Ncv32u stageMiddleSwitch = getStageNumWithNotLessThanNclassifiers(NUM_THREADS_CLASSIFIERPARALLEL,
        haar, h_HaarStages);
    Ncv32u stageEndClassifierParallel = haar.NumStages;
    if (stageMiddleSwitch == 0)
    {
        stageMiddleSwitch = 1;
    }

    //create stages subdivision for pixel-parallel processing
    const Ncv32u compactEveryNstage = bDoAtomicCompaction ? 7 : 1;
    Ncv32u curStop = stageStartAnchorParallel;
    std::vector<Ncv32u> pixParallelStageStops;
    while (curStop < stageMiddleSwitch)
    {
        pixParallelStageStops.push_back(curStop);
        curStop += compactEveryNstage;
    }
    if (curStop > compactEveryNstage && curStop - stageMiddleSwitch > compactEveryNstage / 2)
    {
        pixParallelStageStops[pixParallelStageStops.size()-1] =
            (stageMiddleSwitch - (curStop - 2 * compactEveryNstage)) / 2;
    }
    pixParallelStageStops.push_back(stageMiddleSwitch);
    Ncv32u pixParallelStageStopsIndex = 0;

    if (pixelStep != 1 || bMaskElements)
    {
        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaMemcpyToSymbolAsync(d_outMaskPosition, hp_zero, sizeof(Ncv32u),
                                                        0, cudaMemcpyHostToDevice, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }

        dim3 gridInit((((anchorsRoi.width + pixelStep - 1) / pixelStep + NUM_THREADS_ANCHORSPARALLEL - 1) / NUM_THREADS_ANCHORSPARALLEL),
                        (anchorsRoi.height + pixelStep - 1) / pixelStep);
        dim3 blockInit(NUM_THREADS_ANCHORSPARALLEL);

        if (gridInit.x == 0 || gridInit.y == 0)
        {
            numDetections = 0;
            return NCV_SUCCESS;
        }

        initializeMaskVectorDynTemplate(bMaskElements,
                                        bDoAtomicCompaction,
                                        gridInit, blockInit, cuStream,
                                        d_ptrNowData->ptr(),
                                        d_ptrNowTmp->ptr(),
                                        static_cast<Ncv32u>(d_vecPixelMask.length()), d_pixelMask.stride(),
                                        anchorsRoi, pixelStep);
        ncvAssertCUDAReturn(cudaGetLastError(), NCV_CUDA_ERROR);

        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaMemcpyFromSymbolAsync(hp_numDet, d_outMaskPosition, sizeof(Ncv32u),
                                                          0, cudaMemcpyDeviceToHost, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
            swap(d_ptrNowData, d_ptrNowTmp);
        }
        else
        {
            NCVStatus nppSt;
            nppSt = nppsStCompact_32u(d_ptrNowTmp->ptr(), static_cast<Ncv32u>(d_vecPixelMask.length()),
                                      d_ptrNowData->ptr(), hp_numDet, OBJDET_MASK_ELEMENT_INVALID_32U,
                                      d_tmpBufCompact.ptr(), szNppCompactTmpBuf, devProp);
            ncvAssertReturn(nppSt == NPPST_SUCCESS, NCV_NPP_ERROR);
        }
        numDetections = *hp_numDet;
    }
    else
    {
        //
        // 1. Run the first pixel-input pixel-parallel classifier for few stages
        //

        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaMemcpyToSymbolAsync(d_outMaskPosition, hp_zero, sizeof(Ncv32u),
                                                        0, cudaMemcpyHostToDevice, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }

        dim3 grid1(((d_pixelMask.stride() + NUM_THREADS_ANCHORSPARALLEL - 1) / NUM_THREADS_ANCHORSPARALLEL),
                   anchorsRoi.height);
        dim3 block1(NUM_THREADS_ANCHORSPARALLEL);
        applyHaarClassifierAnchorParallelDynTemplate(
            true,                         //tbInitMaskPositively
            bTexCacheIImg,                //tbCacheTextureIImg
            bTexCacheCascade,             //tbCacheTextureCascade
            pixParallelStageStops[pixParallelStageStopsIndex] != 0,//tbReadPixelIndexFromVector
            bDoAtomicCompaction,          //tbDoAtomicCompaction
            grid1,
            block1,
            cuStream,
            integral.ptr(), integral.stride(),
            d_weights.ptr(), d_weights.stride(),
            d_HaarFeatures.ptr(), d_HaarNodes.ptr(), d_HaarStages.ptr(),
            d_ptrNowData->ptr(),
            bDoAtomicCompaction ? d_ptrNowTmp->ptr() : d_ptrNowData->ptr(),
            0,
            d_pixelMask.stride(),
            anchorsRoi,
            pixParallelStageStops[pixParallelStageStopsIndex],
            pixParallelStageStops[pixParallelStageStopsIndex+1],
            scaleAreaPixels);
        ncvAssertCUDAReturn(cudaGetLastError(), NCV_CUDA_ERROR);

        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaMemcpyFromSymbolAsync(hp_numDet, d_outMaskPosition, sizeof(Ncv32u),
                                                          0, cudaMemcpyDeviceToHost, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }
        else
        {
            NCVStatus nppSt;
            nppSt = nppsStCompact_32u(d_ptrNowData->ptr(), static_cast<Ncv32u>(d_vecPixelMask.length()),
                                      d_ptrNowTmp->ptr(), hp_numDet, OBJDET_MASK_ELEMENT_INVALID_32U,
                                      d_tmpBufCompact.ptr(), szNppCompactTmpBuf, devProp);
            ncvAssertReturnNcvStat(nppSt);
        }

        swap(d_ptrNowData, d_ptrNowTmp);
        numDetections = *hp_numDet;

        pixParallelStageStopsIndex++;
    }

    //
    // 2. Run pixel-parallel stages
    //

    for (; pixParallelStageStopsIndex < pixParallelStageStops.size()-1; pixParallelStageStopsIndex++)
    {
        if (numDetections == 0)
        {
            break;
        }

        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaMemcpyToSymbolAsync(d_outMaskPosition, hp_zero, sizeof(Ncv32u),
                                                        0, cudaMemcpyHostToDevice, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }

        dim3 grid2((numDetections + NUM_THREADS_ANCHORSPARALLEL - 1) / NUM_THREADS_ANCHORSPARALLEL);
        if (numDetections > MAX_GRID_DIM)
        {
            grid2.x = MAX_GRID_DIM;
            grid2.y = (numDetections + MAX_GRID_DIM - 1) / MAX_GRID_DIM;
        }
        dim3 block2(NUM_THREADS_ANCHORSPARALLEL);

        applyHaarClassifierAnchorParallelDynTemplate(
            false,                        //tbInitMaskPositively
            bTexCacheIImg,                //tbCacheTextureIImg
            bTexCacheCascade,             //tbCacheTextureCascade
            pixParallelStageStops[pixParallelStageStopsIndex] != 0 || pixelStep != 1 || bMaskElements,//tbReadPixelIndexFromVector
            bDoAtomicCompaction,          //tbDoAtomicCompaction
            grid2,
            block2,
            cuStream,
            integral.ptr(), integral.stride(),
            d_weights.ptr(), d_weights.stride(),
            d_HaarFeatures.ptr(), d_HaarNodes.ptr(), d_HaarStages.ptr(),
            d_ptrNowData->ptr(),
            bDoAtomicCompaction ? d_ptrNowTmp->ptr() : d_ptrNowData->ptr(),
            numDetections,
            d_pixelMask.stride(),
            anchorsRoi,
            pixParallelStageStops[pixParallelStageStopsIndex],
            pixParallelStageStops[pixParallelStageStopsIndex+1],
            scaleAreaPixels);
        ncvAssertCUDAReturn(cudaGetLastError(), NCV_CUDA_ERROR);

        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaMemcpyFromSymbolAsync(hp_numDet, d_outMaskPosition, sizeof(Ncv32u),
                                                          0, cudaMemcpyDeviceToHost, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }
        else
        {
            NCVStatus nppSt;
            nppSt = nppsStCompact_32u(d_ptrNowData->ptr(), numDetections,
                                      d_ptrNowTmp->ptr(), hp_numDet, OBJDET_MASK_ELEMENT_INVALID_32U,
                                      d_tmpBufCompact.ptr(), szNppCompactTmpBuf, devProp);
            ncvAssertReturnNcvStat(nppSt);
        }

        swap(d_ptrNowData, d_ptrNowTmp);
        numDetections = *hp_numDet;
    }

    //
    // 3. Run all left stages in one stage-parallel kernel
    //

    if (numDetections > 0 && stageMiddleSwitch < stageEndClassifierParallel)
    {
        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaMemcpyToSymbolAsync(d_outMaskPosition, hp_zero, sizeof(Ncv32u),
                                                        0, cudaMemcpyHostToDevice, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }

        dim3 grid3(numDetections);
        if (numDetections > MAX_GRID_DIM)
        {
            grid3.x = MAX_GRID_DIM;
            grid3.y = (numDetections + MAX_GRID_DIM - 1) / MAX_GRID_DIM;
        }
        dim3 block3(NUM_THREADS_CLASSIFIERPARALLEL);

        applyHaarClassifierClassifierParallelDynTemplate(
            bTexCacheIImg,                //tbCacheTextureIImg
            bTexCacheCascade,             //tbCacheTextureCascade
            bDoAtomicCompaction,          //tbDoAtomicCompaction
            grid3,
            block3,
            cuStream,
            integral.ptr(), integral.stride(),
            d_weights.ptr(), d_weights.stride(),
            d_HaarFeatures.ptr(), d_HaarNodes.ptr(), d_HaarStages.ptr(),
            d_ptrNowData->ptr(),
            bDoAtomicCompaction ? d_ptrNowTmp->ptr() : d_ptrNowData->ptr(),
            numDetections,
            d_pixelMask.stride(),
            anchorsRoi,
            stageMiddleSwitch,
            stageEndClassifierParallel,
            scaleAreaPixels);
        ncvAssertCUDAReturn(cudaGetLastError(), NCV_CUDA_ERROR);

        if (bDoAtomicCompaction)
        {
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaMemcpyFromSymbolAsync(hp_numDet, d_outMaskPosition, sizeof(Ncv32u),
                                                          0, cudaMemcpyDeviceToHost, cuStream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }
        else
        {
            NCVStatus nppSt;
            nppSt = nppsStCompact_32u(d_ptrNowData->ptr(), numDetections,
                                      d_ptrNowTmp->ptr(), hp_numDet, OBJDET_MASK_ELEMENT_INVALID_32U,
                                      d_tmpBufCompact.ptr(), szNppCompactTmpBuf, devProp);
            ncvAssertReturnNcvStat(nppSt);
        }

        swap(d_ptrNowData, d_ptrNowTmp);
        numDetections = *hp_numDet;
    }

    if (d_ptrNowData != &d_vecPixelMask)
    {
        d_vecPixelMaskTmp.copySolid(d_vecPixelMask, cuStream);
        ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
    }

#if defined _SELF_TEST_

    ncvStat = d_pixelMask.copySolid(h_pixelMask_d, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);

    if (bDoAtomicCompaction)
    {
        std::sort(h_pixelMask_d.ptr, h_pixelMask_d.ptr + numDetections);
    }

    Ncv32u fpu_oldcw, fpu_cw;
    _controlfp_s(&fpu_cw, 0, 0);
    fpu_oldcw = fpu_cw;
    _controlfp_s(&fpu_cw, _PC_24, _MCW_PC);
    Ncv32u numDetGold;
    ncvStat = ncvApplyHaarClassifierCascade_host(h_integralImage, h_weights, h_pixelMask, numDetGold, haar,
                                                 h_HaarStages, h_HaarNodes, h_HaarFeatures,
                                                 bMaskElements, anchorsRoi, pixelStep, scaleArea);
    ncvAssertReturnNcvStat(ncvStat);
    _controlfp_s(&fpu_cw, fpu_oldcw, _MCW_PC);

    bool bPass = true;

    if (numDetGold != numDetections)
    {
        printf("NCVHaarClassifierCascade::applyHaarClassifierCascade numdetections don't match: cpu=%d, gpu=%d\n", numDetGold, numDetections);
        bPass = false;
    }
    else
    {
        for (Ncv32u i=0; i<std::max(numDetGold, numDetections) && bPass; i++)
        {
            if (h_pixelMask.ptr[i] != h_pixelMask_d.ptr[i])
            {
                printf("NCVHaarClassifierCascade::applyHaarClassifierCascade self test failed: i=%d, cpu=%d, gpu=%d\n", i, h_pixelMask.ptr[i], h_pixelMask_d.ptr[i]);
                bPass = false;
            }
        }
    }

    printf("NCVHaarClassifierCascade::applyHaarClassifierCascade %s\n", bPass?"PASSED":"FAILED");
#endif

    NCV_SKIP_COND_END

    return NCV_SUCCESS;
}


//==============================================================================
//
// HypothesesOperations file
//
//==============================================================================


const Ncv32u NUM_GROW_THREADS = 128;


__device__ __host__ NcvRect32u pixelToRect(Ncv32u pixel, Ncv32u width, Ncv32u height, Ncv32f scale)
{
    NcvRect32u res;
    res.x = (Ncv32u)(scale * (pixel & 0xFFFF));
    res.y = (Ncv32u)(scale * (pixel >> 16));
    res.width = (Ncv32u)(scale * width);
    res.height = (Ncv32u)(scale * height);
    return res;
}


__global__ void growDetectionsKernel(Ncv32u *pixelMask, Ncv32u numElements,
                                     NcvRect32u *hypotheses,
                                     Ncv32u rectWidth, Ncv32u rectHeight, Ncv32f curScale)
{
    Ncv32u blockId = blockIdx.y * 65535 + blockIdx.x;
    Ncv32u elemAddr = blockId * NUM_GROW_THREADS + threadIdx.x;
    if (elemAddr >= numElements)
    {
        return;
    }
    hypotheses[elemAddr] = pixelToRect(pixelMask[elemAddr], rectWidth, rectHeight, curScale);
}


NCVStatus ncvGrowDetectionsVector_device(NCVVector<Ncv32u> &pixelMask,
                                         Ncv32u numPixelMaskDetections,
                                         NCVVector<NcvRect32u> &hypotheses,
                                         Ncv32u &totalDetections,
                                         Ncv32u totalMaxDetections,
                                         Ncv32u rectWidth,
                                         Ncv32u rectHeight,
                                         Ncv32f curScale,
                                         cudaStream_t cuStream)
{
    ncvAssertReturn(pixelMask.ptr() != NULL && hypotheses.ptr() != NULL, NCV_NULL_PTR);

    ncvAssertReturn(pixelMask.memType() == hypotheses.memType() &&
                    pixelMask.memType() == NCVMemoryTypeDevice, NCV_MEM_RESIDENCE_ERROR);

    ncvAssertReturn(rectWidth > 0 && rectHeight > 0 && curScale > 0, NCV_INVALID_ROI);

    ncvAssertReturn(curScale > 0, NCV_INVALID_SCALE);

    ncvAssertReturn(totalMaxDetections <= hypotheses.length() &&
                    numPixelMaskDetections <= pixelMask.length() &&
                    totalMaxDetections <= totalMaxDetections, NCV_INCONSISTENT_INPUT);

    NCVStatus ncvStat = NCV_SUCCESS;
    Ncv32u numDetsToCopy = numPixelMaskDetections;

    if (numDetsToCopy == 0)
    {
        return ncvStat;
    }

    if (totalDetections + numPixelMaskDetections > totalMaxDetections)
    {
        ncvStat = NCV_WARNING_HAAR_DETECTIONS_VECTOR_OVERFLOW;
        numDetsToCopy = totalMaxDetections - totalDetections;
    }

    dim3 block(NUM_GROW_THREADS);
    dim3 grid((numDetsToCopy + NUM_GROW_THREADS - 1) / NUM_GROW_THREADS);
    if (grid.x > 65535)
    {
        grid.y = (grid.x + 65534) / 65535;
        grid.x = 65535;
    }
    growDetectionsKernel<<<grid, block, 0, cuStream>>>(pixelMask.ptr(), numDetsToCopy,
                                                       hypotheses.ptr() + totalDetections,
                                                       rectWidth, rectHeight, curScale);
    ncvAssertCUDAReturn(cudaGetLastError(), NCV_CUDA_ERROR);

    totalDetections += numDetsToCopy;
    return ncvStat;
}


//==============================================================================
//
// Pipeline file
//
//==============================================================================


NCVStatus ncvDetectObjectsMultiScale_device(NCVMatrix<Ncv8u> &d_srcImg,
                                            NcvSize32u srcRoi,
                                            NCVVector<NcvRect32u> &d_dstRects,
                                            Ncv32u &dstNumRects,

                                            HaarClassifierCascadeDescriptor &haar,
                                            NCVVector<HaarStage64> &h_HaarStages,
                                            NCVVector<HaarStage64> &d_HaarStages,
                                            NCVVector<HaarClassifierNode128> &d_HaarNodes,
                                            NCVVector<HaarFeature64> &d_HaarFeatures,

                                            NcvSize32u minObjSize,
                                            Ncv32u minNeighbors,      //default 4
                                            Ncv32f scaleStep,         //default 1.2f
                                            Ncv32u pixelStep,         //default 1
                                            Ncv32u flags,             //default NCVPipeObjDet_Default

                                            INCVMemAllocator &gpuAllocator,
                                            INCVMemAllocator &cpuAllocator,
                                            cudaDeviceProp &devProp,
                                            cudaStream_t cuStream)
{
    ncvAssertReturn(d_srcImg.memType() == d_dstRects.memType() &&
                    d_srcImg.memType() == gpuAllocator.memType() &&
                     (d_srcImg.memType() == NCVMemoryTypeDevice ||
                      d_srcImg.memType() == NCVMemoryTypeNone), NCV_MEM_RESIDENCE_ERROR);

    ncvAssertReturn(d_HaarStages.memType() == d_HaarNodes.memType() &&
                    d_HaarStages.memType() == d_HaarFeatures.memType() &&
                     (d_HaarStages.memType() == NCVMemoryTypeDevice ||
                      d_HaarStages.memType() == NCVMemoryTypeNone), NCV_MEM_RESIDENCE_ERROR);

    ncvAssertReturn(h_HaarStages.memType() != NCVMemoryTypeDevice, NCV_MEM_RESIDENCE_ERROR);

    ncvAssertReturn(gpuAllocator.isInitialized() && cpuAllocator.isInitialized(), NCV_ALLOCATOR_NOT_INITIALIZED);

    ncvAssertReturn((d_srcImg.ptr() != NULL && d_dstRects.ptr() != NULL &&
                     h_HaarStages.ptr() != NULL && d_HaarStages.ptr() != NULL && d_HaarNodes.ptr() != NULL &&
                     d_HaarFeatures.ptr() != NULL) || gpuAllocator.isCounting(), NCV_NULL_PTR);
    ncvAssertReturn(srcRoi.width > 0 && srcRoi.height > 0 &&
                    d_srcImg.width() >= srcRoi.width && d_srcImg.height() >= srcRoi.height &&
                    srcRoi.width >= minObjSize.width && srcRoi.height >= minObjSize.height &&
                    d_dstRects.length() >= 1, NCV_DIMENSIONS_INVALID);

    ncvAssertReturn(scaleStep > 1.0f, NCV_INVALID_SCALE);

    ncvAssertReturn(d_HaarStages.length() >= haar.NumStages &&
                    d_HaarNodes.length() >= haar.NumClassifierTotalNodes &&
                    d_HaarFeatures.length() >= haar.NumFeatures &&
                    d_HaarStages.length() == h_HaarStages.length() &&
                    haar.NumClassifierRootNodes <= haar.NumClassifierTotalNodes, NCV_DIMENSIONS_INVALID);

    ncvAssertReturn(haar.bNeedsTiltedII == false, NCV_NOIMPL_HAAR_TILTED_FEATURES);

    ncvAssertReturn(pixelStep == 1 || pixelStep == 2, NCV_HAAR_INVALID_PIXEL_STEP);

    //TODO: set NPP active stream to cuStream

    NCVStatus ncvStat;
    NCV_SET_SKIP_COND(gpuAllocator.isCounting());

    Ncv32u integralWidth = d_srcImg.width() + 1;
    Ncv32u integralHeight = d_srcImg.height() + 1;

    NCVMatrixAlloc<Ncv32u> integral(gpuAllocator, integralWidth, integralHeight);
    ncvAssertReturn(integral.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVMatrixAlloc<Ncv64u> d_sqIntegralImage(gpuAllocator, integralWidth, integralHeight);
    ncvAssertReturn(d_sqIntegralImage.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCVMatrixAlloc<Ncv32f> d_rectStdDev(gpuAllocator, d_srcImg.width(), d_srcImg.height());
    ncvAssertReturn(d_rectStdDev.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVMatrixAlloc<Ncv32u> d_pixelMask(gpuAllocator, d_srcImg.width(), d_srcImg.height());
    ncvAssertReturn(d_pixelMask.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCVMatrixAlloc<Ncv32u> d_scaledIntegralImage(gpuAllocator, integralWidth, integralHeight);
    ncvAssertReturn(d_scaledIntegralImage.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVMatrixAlloc<Ncv64u> d_scaledSqIntegralImage(gpuAllocator, integralWidth, integralHeight);
    ncvAssertReturn(d_scaledSqIntegralImage.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCVVectorAlloc<NcvRect32u> d_hypothesesIntermediate(gpuAllocator, d_srcImg.width() * d_srcImg.height());
    ncvAssertReturn(d_hypothesesIntermediate.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVVectorAlloc<NcvRect32u> h_hypothesesIntermediate(cpuAllocator, d_srcImg.width() * d_srcImg.height());
    ncvAssertReturn(h_hypothesesIntermediate.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCVStatus nppStat;
    Ncv32u szTmpBufIntegral, szTmpBufSqIntegral;
    nppStat = nppiStIntegralGetSize_8u32u(NcvSize32u(d_srcImg.width(), d_srcImg.height()), &szTmpBufIntegral, devProp);
    ncvAssertReturnNcvStat(nppStat);
    nppStat = nppiStSqrIntegralGetSize_8u64u(NcvSize32u(d_srcImg.width(), d_srcImg.height()), &szTmpBufSqIntegral, devProp);
    ncvAssertReturnNcvStat(nppStat);
    NCVVectorAlloc<Ncv8u> d_tmpIIbuf(gpuAllocator, std::max(szTmpBufIntegral, szTmpBufSqIntegral));
    ncvAssertReturn(d_tmpIIbuf.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCV_SKIP_COND_BEGIN

    nppStat = nppiStIntegral_8u32u_C1R(d_srcImg.ptr(), d_srcImg.pitch(),
                                       integral.ptr(), integral.pitch(),
                                       NcvSize32u(d_srcImg.width(), d_srcImg.height()),
                                       d_tmpIIbuf.ptr(), szTmpBufIntegral, devProp);
    ncvAssertReturnNcvStat(nppStat);

    nppStat = nppiStSqrIntegral_8u64u_C1R(d_srcImg.ptr(), d_srcImg.pitch(),
                                          d_sqIntegralImage.ptr(), d_sqIntegralImage.pitch(),
                                          NcvSize32u(d_srcImg.width(), d_srcImg.height()),
                                          d_tmpIIbuf.ptr(), szTmpBufSqIntegral, devProp);
    ncvAssertReturnNcvStat(nppStat);

    NCV_SKIP_COND_END

    dstNumRects = 0;

    Ncv32u lastCheckedScale = 0;
    NcvBool bReverseTraverseScale = ((flags & NCVPipeObjDet_FindLargestObject) != 0);
    std::vector<Ncv32u> scalesVector;

    NcvBool bFoundLargestFace = false;

    for (Ncv32f scaleIter = 1.0f; ; scaleIter *= scaleStep)
    {
        Ncv32u scale = (Ncv32u)scaleIter;
        if (lastCheckedScale == scale)
        {
            continue;
        }
        lastCheckedScale = scale;

        if (haar.ClassifierSize.width * (Ncv32s)scale < minObjSize.width ||
            haar.ClassifierSize.height * (Ncv32s)scale < minObjSize.height)
        {
            continue;
        }

        NcvSize32s srcRoi_, srcIIRo_i, scaledIIRoi, searchRoi;

        srcRoi_.width = d_srcImg.width();
        srcRoi_.height = d_srcImg.height();

        srcIIRo_i.width = srcRoi_.width + 1;
        srcIIRo_i.height = srcRoi_.height + 1;

        scaledIIRoi.width = srcIIRo_i.width / scale;
        scaledIIRoi.height = srcIIRo_i.height / scale;

        searchRoi.width = scaledIIRoi.width - haar.ClassifierSize.width;
        searchRoi.height = scaledIIRoi.height - haar.ClassifierSize.height;

        if (searchRoi.width <= 0 || searchRoi.height <= 0)
        {
            break;
        }

        scalesVector.push_back(scale);

        if (gpuAllocator.isCounting())
        {
            break;
        }
    }

    if (bReverseTraverseScale)
    {
        std::reverse(scalesVector.begin(), scalesVector.end());
    }

    //TODO: handle _fair_scale_ flag
    for (Ncv32u i=0; i<scalesVector.size(); i++)
    {
        Ncv32u scale = scalesVector[i];

        NcvSize32u srcRoi_, scaledIIRoi, searchRoi;
        NcvSize32u srcIIRoi;
        srcRoi_.width = d_srcImg.width();
        srcRoi_.height = d_srcImg.height();
        srcIIRoi.width = srcRoi_.width + 1;
        srcIIRoi.height = srcRoi_.height + 1;
        scaledIIRoi.width = srcIIRoi.width / scale;
        scaledIIRoi.height = srcIIRoi.height / scale;
        searchRoi.width = scaledIIRoi.width - haar.ClassifierSize.width;
        searchRoi.height = scaledIIRoi.height - haar.ClassifierSize.height;

        NCV_SKIP_COND_BEGIN

        nppStat = nppiStDecimate_32u_C1R(
            integral.ptr(), integral.pitch(),
            d_scaledIntegralImage.ptr(), d_scaledIntegralImage.pitch(),
            srcIIRoi, scale, true);
        ncvAssertReturnNcvStat(nppStat);

        nppStat = nppiStDecimate_64u_C1R(
            d_sqIntegralImage.ptr(), d_sqIntegralImage.pitch(),
            d_scaledSqIntegralImage.ptr(), d_scaledSqIntegralImage.pitch(),
            srcIIRoi, scale, true);
        ncvAssertReturnNcvStat(nppStat);

        const NcvRect32u rect(
            HAAR_STDDEV_BORDER,
            HAAR_STDDEV_BORDER,
            haar.ClassifierSize.width - 2*HAAR_STDDEV_BORDER,
            haar.ClassifierSize.height - 2*HAAR_STDDEV_BORDER);
        nppStat = nppiStRectStdDev_32f_C1R(
            d_scaledIntegralImage.ptr(), d_scaledIntegralImage.pitch(),
            d_scaledSqIntegralImage.ptr(), d_scaledSqIntegralImage.pitch(),
            d_rectStdDev.ptr(), d_rectStdDev.pitch(),
            NcvSize32u(searchRoi.width, searchRoi.height), rect,
            (Ncv32f)scale*scale, true);
        ncvAssertReturnNcvStat(nppStat);

        NCV_SKIP_COND_END

        Ncv32u detectionsOnThisScale;
        ncvStat = ncvApplyHaarClassifierCascade_device(
            d_scaledIntegralImage, d_rectStdDev, d_pixelMask,
            detectionsOnThisScale,
            haar, h_HaarStages, d_HaarStages, d_HaarNodes, d_HaarFeatures, false,
            searchRoi, pixelStep, (Ncv32f)scale*scale,
            gpuAllocator, cpuAllocator, devProp, cuStream);
        ncvAssertReturnNcvStat(nppStat);

        NCV_SKIP_COND_BEGIN

        NCVVectorReuse<Ncv32u> d_vecPixelMask(d_pixelMask.getSegment());
        ncvStat = ncvGrowDetectionsVector_device(
            d_vecPixelMask,
            detectionsOnThisScale,
            d_hypothesesIntermediate,
            dstNumRects,
            static_cast<Ncv32u>(d_hypothesesIntermediate.length()),
            haar.ClassifierSize.width,
            haar.ClassifierSize.height,
            (Ncv32f)scale,
            cuStream);
        ncvAssertReturn(ncvStat == NCV_SUCCESS, ncvStat);

        if (flags & NCVPipeObjDet_FindLargestObject)
        {
            if (dstNumRects == 0)
            {
                continue;
            }

            if (dstNumRects != 0)
            {
                ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
                ncvStat = d_hypothesesIntermediate.copySolid(h_hypothesesIntermediate, cuStream,
                                                             dstNumRects * sizeof(NcvRect32u));
                ncvAssertReturnNcvStat(ncvStat);
                ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
            }

            Ncv32u numStrongHypothesesNow = dstNumRects;
            ncvStat = ncvGroupRectangles_host(
                h_hypothesesIntermediate,
                numStrongHypothesesNow,
                minNeighbors,
                RECT_SIMILARITY_PROPORTION,
                NULL);
            ncvAssertReturnNcvStat(ncvStat);

            if (numStrongHypothesesNow > 0)
            {
                NcvRect32u maxRect = h_hypothesesIntermediate.ptr()[0];
                for (Ncv32u j=1; j<numStrongHypothesesNow; j++)
                {
                    if (maxRect.width < h_hypothesesIntermediate.ptr()[j].width)
                    {
                        maxRect = h_hypothesesIntermediate.ptr()[j];
                    }
                }

                h_hypothesesIntermediate.ptr()[0] = maxRect;
                dstNumRects = 1;

                ncvStat = h_hypothesesIntermediate.copySolid(d_dstRects, cuStream, sizeof(NcvRect32u));
                ncvAssertReturnNcvStat(ncvStat);

                bFoundLargestFace = true;

                break;
            }
        }

        NCV_SKIP_COND_END

        if (gpuAllocator.isCounting())
        {
            break;
        }
    }

    NCVStatus ncvRetCode = NCV_SUCCESS;

    NCV_SKIP_COND_BEGIN

    if (flags & NCVPipeObjDet_FindLargestObject)
    {
        if (!bFoundLargestFace)
        {
            dstNumRects = 0;
        }
    }
    else
    {
        //TODO: move hypotheses filtration to GPU pipeline (the only CPU-resident element of the pipeline left)
        if (dstNumRects != 0)
        {
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
            ncvStat = d_hypothesesIntermediate.copySolid(h_hypothesesIntermediate, cuStream,
                                                         dstNumRects * sizeof(NcvRect32u));
            ncvAssertReturnNcvStat(ncvStat);
            ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        }

        ncvStat = ncvGroupRectangles_host(
            h_hypothesesIntermediate,
            dstNumRects,
            minNeighbors,
            RECT_SIMILARITY_PROPORTION,
            NULL);
        ncvAssertReturnNcvStat(ncvStat);

        if (dstNumRects > d_dstRects.length())
        {
            ncvRetCode = NCV_WARNING_HAAR_DETECTIONS_VECTOR_OVERFLOW;
            dstNumRects = static_cast<Ncv32u>(d_dstRects.length());
        }

        if (dstNumRects != 0)
        {
            ncvStat = h_hypothesesIntermediate.copySolid(d_dstRects, cuStream,
                                                         dstNumRects * sizeof(NcvRect32u));
            ncvAssertReturnNcvStat(ncvStat);
        }
    }

    if (flags & NCVPipeObjDet_VisualizeInPlace)
    {
        ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);
        ncvDrawRects_8u_device(d_srcImg.ptr(), d_srcImg.stride(),
                               d_srcImg.width(), d_srcImg.height(),
                               d_dstRects.ptr(), dstNumRects, 255, cuStream);
    }

    NCV_SKIP_COND_END

    return ncvRetCode;
}


//==============================================================================
//
// Purely Host code: classifier IO, mock-ups
//
//==============================================================================


#ifdef _SELF_TEST_
#include <float.h>
#endif


NCVStatus ncvApplyHaarClassifierCascade_host(NCVMatrix<Ncv32u> &h_integralImage,
                                             NCVMatrix<Ncv32f> &h_weights,
                                             NCVMatrixAlloc<Ncv32u> &h_pixelMask,
                                             Ncv32u &numDetections,
                                             HaarClassifierCascadeDescriptor &haar,
                                             NCVVector<HaarStage64> &h_HaarStages,
                                             NCVVector<HaarClassifierNode128> &h_HaarNodes,
                                             NCVVector<HaarFeature64> &h_HaarFeatures,
                                             NcvBool bMaskElements,
                                             NcvSize32u anchorsRoi,
                                             Ncv32u pixelStep,
                                             Ncv32f scaleArea)
{
    ncvAssertReturn(h_integralImage.memType() == h_weights.memType() &&
                    h_integralImage.memType() == h_pixelMask.memType() &&
                     (h_integralImage.memType() == NCVMemoryTypeHostPageable ||
                      h_integralImage.memType() == NCVMemoryTypeHostPinned), NCV_MEM_RESIDENCE_ERROR);
    ncvAssertReturn(h_HaarStages.memType() == h_HaarNodes.memType() &&
                    h_HaarStages.memType() == h_HaarFeatures.memType() &&
                     (h_HaarStages.memType() == NCVMemoryTypeHostPageable ||
                      h_HaarStages.memType() == NCVMemoryTypeHostPinned), NCV_MEM_RESIDENCE_ERROR);
    ncvAssertReturn(h_integralImage.ptr() != NULL && h_weights.ptr() != NULL && h_pixelMask.ptr() != NULL &&
                    h_HaarStages.ptr() != NULL && h_HaarNodes.ptr() != NULL && h_HaarFeatures.ptr() != NULL, NCV_NULL_PTR);
    ncvAssertReturn(anchorsRoi.width > 0 && anchorsRoi.height > 0 &&
                    h_pixelMask.width() >= anchorsRoi.width && h_pixelMask.height() >= anchorsRoi.height &&
                    h_weights.width() >= anchorsRoi.width && h_weights.height() >= anchorsRoi.height &&
                    h_integralImage.width() >= anchorsRoi.width + haar.ClassifierSize.width &&
                    h_integralImage.height() >= anchorsRoi.height + haar.ClassifierSize.height, NCV_DIMENSIONS_INVALID);
    ncvAssertReturn(scaleArea > 0, NCV_INVALID_SCALE);
    ncvAssertReturn(h_HaarStages.length() >= haar.NumStages &&
                    h_HaarNodes.length() >= haar.NumClassifierTotalNodes &&
                    h_HaarFeatures.length() >= haar.NumFeatures &&
                    h_HaarStages.length() == h_HaarStages.length() &&
                    haar.NumClassifierRootNodes <= haar.NumClassifierTotalNodes, NCV_DIMENSIONS_INVALID);
    ncvAssertReturn(haar.bNeedsTiltedII == false, NCV_NOIMPL_HAAR_TILTED_FEATURES);
    ncvAssertReturn(pixelStep == 1 || pixelStep == 2, NCV_HAAR_INVALID_PIXEL_STEP);

    Ncv32f scaleAreaPixels = scaleArea * ((haar.ClassifierSize.width - 2*HAAR_STDDEV_BORDER) *
                                          (haar.ClassifierSize.height - 2*HAAR_STDDEV_BORDER));

    for (Ncv32u i=0; i<anchorsRoi.height; i++)
    {
        for (Ncv32u j=0; j<h_pixelMask.stride(); j++)
        {
            if (i % pixelStep != 0 || j % pixelStep != 0 || j >= anchorsRoi.width)
            {
                h_pixelMask.ptr()[i * h_pixelMask.stride() + j] = OBJDET_MASK_ELEMENT_INVALID_32U;
            }
            else
            {
                for (Ncv32u iStage = 0; iStage < haar.NumStages; iStage++)
                {
                    Ncv32f curStageSum = 0.0f;
                    Ncv32u numRootNodesInStage = h_HaarStages.ptr()[iStage].getNumClassifierRootNodes();
                    Ncv32u curRootNodeOffset = h_HaarStages.ptr()[iStage].getStartClassifierRootNodeOffset();

                    if (iStage == 0)
                    {
                        if (bMaskElements && h_pixelMask.ptr()[i * h_pixelMask.stride() + j] == OBJDET_MASK_ELEMENT_INVALID_32U)
                        {
                            break;
                        }
                        else
                        {
                            h_pixelMask.ptr()[i * h_pixelMask.stride() + j] = ((i << 16) | j);
                        }
                    }
                    else if (h_pixelMask.ptr()[i * h_pixelMask.stride() + j] == OBJDET_MASK_ELEMENT_INVALID_32U)
                    {
                        break;
                    }

                    while (numRootNodesInStage--)
                    {
                        NcvBool bMoreNodesToTraverse = true;
                        Ncv32u curNodeOffset = curRootNodeOffset;

                        while (bMoreNodesToTraverse)
                        {
                            HaarClassifierNode128 curNode = h_HaarNodes.ptr()[curNodeOffset];
                            HaarFeatureDescriptor32 curFeatDesc = curNode.getFeatureDesc();
                            Ncv32u curNodeFeaturesNum = curFeatDesc.getNumFeatures();
                            Ncv32u curNodeFeaturesOffs = curFeatDesc.getFeaturesOffset();

                            Ncv32f curNodeVal = 0.f;
                            for (Ncv32u iRect=0; iRect<curNodeFeaturesNum; iRect++)
                            {
                                HaarFeature64 feature = h_HaarFeatures.ptr()[curNodeFeaturesOffs + iRect];
                                Ncv32u rectX, rectY, rectWidth, rectHeight;
                                feature.getRect(&rectX, &rectY, &rectWidth, &rectHeight);
                                Ncv32f rectWeight = feature.getWeight();
                                Ncv32u iioffsTL = (i + rectY) * h_integralImage.stride() + (j + rectX);
                                Ncv32u iioffsTR = iioffsTL + rectWidth;
                                Ncv32u iioffsBL = iioffsTL + rectHeight * h_integralImage.stride();
                                Ncv32u iioffsBR = iioffsBL + rectWidth;

                                Ncv32u iivalTL = h_integralImage.ptr()[iioffsTL];
                                Ncv32u iivalTR = h_integralImage.ptr()[iioffsTR];
                                Ncv32u iivalBL = h_integralImage.ptr()[iioffsBL];
                                Ncv32u iivalBR = h_integralImage.ptr()[iioffsBR];
                                Ncv32u rectSum = iivalBR - iivalBL + iivalTL - iivalTR;
                                curNodeVal += (Ncv32f)rectSum * rectWeight;
                            }

                            HaarClassifierNodeDescriptor32 nodeLeft = curNode.getLeftNodeDesc();
                            HaarClassifierNodeDescriptor32 nodeRight = curNode.getRightNodeDesc();
                            Ncv32f nodeThreshold = curNode.getThreshold();

                            HaarClassifierNodeDescriptor32 nextNodeDescriptor;
                            NcvBool nextNodeIsLeaf;

                            if (curNodeVal < scaleAreaPixels * h_weights.ptr()[i * h_weights.stride() + j] * nodeThreshold)
                            {
                                nextNodeDescriptor = nodeLeft;
                                nextNodeIsLeaf = curFeatDesc.isLeftNodeLeaf();
                            }
                            else
                            {
                                nextNodeDescriptor = nodeRight;
                                nextNodeIsLeaf = curFeatDesc.isRightNodeLeaf();
                            }

                            if (nextNodeIsLeaf)
                            {
                                Ncv32f tmpLeafValue = nextNodeDescriptor.getLeafValueHost();
                                curStageSum += tmpLeafValue;
                                bMoreNodesToTraverse = false;
                            }
                            else
                            {
                                curNodeOffset = nextNodeDescriptor.getNextNodeOffset();
                            }
                        }

                        curRootNodeOffset++;
                    }

                    Ncv32f tmpStageThreshold = h_HaarStages.ptr()[iStage].getStageThreshold();
                    if (curStageSum < tmpStageThreshold)
                    {
                        //drop
                        h_pixelMask.ptr()[i * h_pixelMask.stride() + j] = OBJDET_MASK_ELEMENT_INVALID_32U;
                        break;
                    }
                }
            }
        }
    }

    std::sort(h_pixelMask.ptr(), h_pixelMask.ptr() + anchorsRoi.height * h_pixelMask.stride());
    Ncv32u i = 0;
    for (; i<anchorsRoi.height * h_pixelMask.stride(); i++)
    {
        if (h_pixelMask.ptr()[i] == OBJDET_MASK_ELEMENT_INVALID_32U)
        {
            break;
        }
    }
    numDetections = i;

    return NCV_SUCCESS;
}


NCVStatus ncvGrowDetectionsVector_host(NCVVector<Ncv32u> &pixelMask,
                                       Ncv32u numPixelMaskDetections,
                                       NCVVector<NcvRect32u> &hypotheses,
                                       Ncv32u &totalDetections,
                                       Ncv32u totalMaxDetections,
                                       Ncv32u rectWidth,
                                       Ncv32u rectHeight,
                                       Ncv32f curScale)
{
    ncvAssertReturn(pixelMask.ptr() != NULL && hypotheses.ptr() != NULL, NCV_NULL_PTR);
    ncvAssertReturn(pixelMask.memType() == hypotheses.memType() &&
                    pixelMask.memType() != NCVMemoryTypeDevice, NCV_MEM_RESIDENCE_ERROR);
    ncvAssertReturn(rectWidth > 0 && rectHeight > 0 && curScale > 0, NCV_INVALID_ROI);
    ncvAssertReturn(curScale > 0, NCV_INVALID_SCALE);
    ncvAssertReturn(totalMaxDetections <= hypotheses.length() &&
                    numPixelMaskDetections <= pixelMask.length() &&
                    totalMaxDetections <= totalMaxDetections, NCV_INCONSISTENT_INPUT);

    NCVStatus ncvStat = NCV_SUCCESS;
    Ncv32u numDetsToCopy = numPixelMaskDetections;

    if (numDetsToCopy == 0)
    {
        return ncvStat;
    }

    if (totalDetections + numPixelMaskDetections > totalMaxDetections)
    {
        ncvStat = NCV_WARNING_HAAR_DETECTIONS_VECTOR_OVERFLOW;
        numDetsToCopy = totalMaxDetections - totalDetections;
    }

    for (Ncv32u i=0; i<numDetsToCopy; i++)
    {
        hypotheses.ptr()[totalDetections + i] = pixelToRect(pixelMask.ptr()[i], rectWidth, rectHeight, curScale);
    }

    totalDetections += numDetsToCopy;
    return ncvStat;
}


NCVStatus loadFromXML(const std::string &filename,
                      HaarClassifierCascadeDescriptor &haar,
                      std::vector<HaarStage64> &haarStages,
                      std::vector<HaarClassifierNode128> &haarClassifierNodes,
                      std::vector<HaarFeature64> &haarFeatures);


#define NVBIN_HAAR_SIZERESERVED     16
#define NVBIN_HAAR_VERSION          0x1


static NCVStatus loadFromNVBIN(const std::string &filename,
                               HaarClassifierCascadeDescriptor &haar,
                               std::vector<HaarStage64> &haarStages,
                               std::vector<HaarClassifierNode128> &haarClassifierNodes,
                               std::vector<HaarFeature64> &haarFeatures)
{
    size_t readCount;
    FILE *fp = fopen(filename.c_str(), "rb");
    ncvAssertReturn(fp != NULL, NCV_FILE_ERROR);
    Ncv32u fileVersion;
    readCount = fread(&fileVersion, sizeof(Ncv32u), 1, fp);
    ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
    ncvAssertReturn(fileVersion == NVBIN_HAAR_VERSION, NCV_FILE_ERROR);
    Ncv32u fsize;
    readCount = fread(&fsize, sizeof(Ncv32u), 1, fp);
    ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
    fseek(fp, 0, SEEK_END);
    Ncv32u fsizeActual = ftell(fp);
    ncvAssertReturn(fsize == fsizeActual, NCV_FILE_ERROR);

    std::vector<unsigned char> fdata;
    fdata.resize(fsize);
    Ncv32u dataOffset = 0;
    fseek(fp, 0, SEEK_SET);
    readCount = fread(&fdata[0], fsize, 1, fp);
    ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
    fclose(fp);

    //data
    dataOffset = NVBIN_HAAR_SIZERESERVED;
    haar.NumStages = *(Ncv32u *)(&fdata[0]+dataOffset);
    dataOffset += sizeof(Ncv32u);
    haar.NumClassifierRootNodes = *(Ncv32u *)(&fdata[0]+dataOffset);
    dataOffset += sizeof(Ncv32u);
    haar.NumClassifierTotalNodes = *(Ncv32u *)(&fdata[0]+dataOffset);
    dataOffset += sizeof(Ncv32u);
    haar.NumFeatures = *(Ncv32u *)(&fdata[0]+dataOffset);
    dataOffset += sizeof(Ncv32u);
    haar.ClassifierSize = *(NcvSize32u *)(&fdata[0]+dataOffset);
    dataOffset += sizeof(NcvSize32u);
    haar.bNeedsTiltedII = *(NcvBool *)(&fdata[0]+dataOffset);
    dataOffset += sizeof(NcvBool);
    haar.bHasStumpsOnly = *(NcvBool *)(&fdata[0]+dataOffset);
    dataOffset += sizeof(NcvBool);

    haarStages.resize(haar.NumStages);
    haarClassifierNodes.resize(haar.NumClassifierTotalNodes);
    haarFeatures.resize(haar.NumFeatures);

    Ncv32u szStages = haar.NumStages * sizeof(HaarStage64);
    Ncv32u szClassifiers = haar.NumClassifierTotalNodes * sizeof(HaarClassifierNode128);
    Ncv32u szFeatures = haar.NumFeatures * sizeof(HaarFeature64);

    memcpy(&haarStages[0], &fdata[0]+dataOffset, szStages);
    dataOffset += szStages;
    memcpy(&haarClassifierNodes[0], &fdata[0]+dataOffset, szClassifiers);
    dataOffset += szClassifiers;
    memcpy(&haarFeatures[0], &fdata[0]+dataOffset, szFeatures);
    dataOffset += szFeatures;

    return NCV_SUCCESS;
}


NCVStatus ncvHaarGetClassifierSize(const std::string &filename, Ncv32u &numStages,
                                   Ncv32u &numNodes, Ncv32u &numFeatures)
{
    size_t readCount;
    NCVStatus ncvStat;

    std::string fext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);

    if (fext == "nvbin")
    {
        FILE *fp = fopen(filename.c_str(), "rb");
        ncvAssertReturn(fp != NULL, NCV_FILE_ERROR);
        Ncv32u fileVersion;
        readCount = fread(&fileVersion, sizeof(Ncv32u), 1, fp);
        ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
        ncvAssertReturn(fileVersion == NVBIN_HAAR_VERSION, NCV_FILE_ERROR);
        fseek(fp, NVBIN_HAAR_SIZERESERVED, SEEK_SET);
        Ncv32u tmp;
        readCount = fread(&numStages,   sizeof(Ncv32u), 1, fp);
        ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
        readCount = fread(&tmp,         sizeof(Ncv32u), 1, fp);
        ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
        readCount = fread(&numNodes,    sizeof(Ncv32u), 1, fp);
        ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
        readCount = fread(&numFeatures, sizeof(Ncv32u), 1, fp);
        ncvAssertReturn(1 == readCount, NCV_FILE_ERROR);
        fclose(fp);
    }
    else if (fext == "xml")
    {
        HaarClassifierCascadeDescriptor haar;
        std::vector<HaarStage64> haarStages;
        std::vector<HaarClassifierNode128> haarNodes;
        std::vector<HaarFeature64> haarFeatures;

        ncvStat = loadFromXML(filename, haar, haarStages, haarNodes, haarFeatures);
        ncvAssertReturnNcvStat(ncvStat);

        numStages = haar.NumStages;
        numNodes = haar.NumClassifierTotalNodes;
        numFeatures = haar.NumFeatures;
    }
    else
    {
        return NCV_HAAR_XML_LOADING_EXCEPTION;
    }

    return NCV_SUCCESS;
}


NCVStatus ncvHaarLoadFromFile_host(const std::string &filename,
                                   HaarClassifierCascadeDescriptor &haar,
                                   NCVVector<HaarStage64> &h_HaarStages,
                                   NCVVector<HaarClassifierNode128> &h_HaarNodes,
                                   NCVVector<HaarFeature64> &h_HaarFeatures)
{
    ncvAssertReturn(h_HaarStages.memType() == NCVMemoryTypeHostPinned &&
                    h_HaarNodes.memType() == NCVMemoryTypeHostPinned &&
                    h_HaarFeatures.memType() == NCVMemoryTypeHostPinned, NCV_MEM_RESIDENCE_ERROR);

    NCVStatus ncvStat;

    std::string fext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);

    std::vector<HaarStage64> haarStages;
    std::vector<HaarClassifierNode128> haarNodes;
    std::vector<HaarFeature64> haarFeatures;

    if (fext == "nvbin")
    {
        ncvStat = loadFromNVBIN(filename, haar, haarStages, haarNodes, haarFeatures);
        ncvAssertReturnNcvStat(ncvStat);
    }
    else if (fext == "xml")
    {
        ncvStat = loadFromXML(filename, haar, haarStages, haarNodes, haarFeatures);
        ncvAssertReturnNcvStat(ncvStat);
    }
    else
    {
        return NCV_HAAR_XML_LOADING_EXCEPTION;
    }

    ncvAssertReturn(h_HaarStages.length() >= haarStages.size(), NCV_MEM_INSUFFICIENT_CAPACITY);
    ncvAssertReturn(h_HaarNodes.length() >= haarNodes.size(), NCV_MEM_INSUFFICIENT_CAPACITY);
    ncvAssertReturn(h_HaarFeatures.length() >= haarFeatures.size(), NCV_MEM_INSUFFICIENT_CAPACITY);

    memcpy(h_HaarStages.ptr(), &haarStages[0], haarStages.size()*sizeof(HaarStage64));
    memcpy(h_HaarNodes.ptr(), &haarNodes[0], haarNodes.size()*sizeof(HaarClassifierNode128));
    memcpy(h_HaarFeatures.ptr(), &haarFeatures[0], haarFeatures.size()*sizeof(HaarFeature64));

    return NCV_SUCCESS;
}


NCVStatus ncvHaarStoreNVBIN_host(const std::string &filename,
                                 HaarClassifierCascadeDescriptor haar,
                                 NCVVector<HaarStage64> &h_HaarStages,
                                 NCVVector<HaarClassifierNode128> &h_HaarNodes,
                                 NCVVector<HaarFeature64> &h_HaarFeatures)
{
    ncvAssertReturn(h_HaarStages.length() >= haar.NumStages, NCV_INCONSISTENT_INPUT);
    ncvAssertReturn(h_HaarNodes.length() >= haar.NumClassifierTotalNodes, NCV_INCONSISTENT_INPUT);
    ncvAssertReturn(h_HaarFeatures.length() >= haar.NumFeatures, NCV_INCONSISTENT_INPUT);
    ncvAssertReturn(h_HaarStages.memType() == NCVMemoryTypeHostPinned &&
                    h_HaarNodes.memType() == NCVMemoryTypeHostPinned &&
                    h_HaarFeatures.memType() == NCVMemoryTypeHostPinned, NCV_MEM_RESIDENCE_ERROR);

    Ncv32u szStages = haar.NumStages * sizeof(HaarStage64);
    Ncv32u szClassifiers = haar.NumClassifierTotalNodes * sizeof(HaarClassifierNode128);
    Ncv32u szFeatures = haar.NumFeatures * sizeof(HaarFeature64);

    Ncv32u dataOffset = 0;
    std::vector<unsigned char> fdata;
    fdata.resize(szStages+szClassifiers+szFeatures+1024, 0);

    //header
    *(Ncv32u *)(&fdata[0]+dataOffset) = NVBIN_HAAR_VERSION;

    //data
    dataOffset = NVBIN_HAAR_SIZERESERVED;
    *(Ncv32u *)(&fdata[0]+dataOffset) = haar.NumStages;
    dataOffset += sizeof(Ncv32u);
    *(Ncv32u *)(&fdata[0]+dataOffset) = haar.NumClassifierRootNodes;
    dataOffset += sizeof(Ncv32u);
    *(Ncv32u *)(&fdata[0]+dataOffset) = haar.NumClassifierTotalNodes;
    dataOffset += sizeof(Ncv32u);
    *(Ncv32u *)(&fdata[0]+dataOffset) = haar.NumFeatures;
    dataOffset += sizeof(Ncv32u);
    *(NcvSize32u *)(&fdata[0]+dataOffset) = haar.ClassifierSize;
    dataOffset += sizeof(NcvSize32u);
    *(NcvBool *)(&fdata[0]+dataOffset) = haar.bNeedsTiltedII;
    dataOffset += sizeof(NcvBool);
    *(NcvBool *)(&fdata[0]+dataOffset) = haar.bHasStumpsOnly;
    dataOffset += sizeof(NcvBool);

    memcpy(&fdata[0]+dataOffset, h_HaarStages.ptr(), szStages);
    dataOffset += szStages;
    memcpy(&fdata[0]+dataOffset, h_HaarNodes.ptr(), szClassifiers);
    dataOffset += szClassifiers;
    memcpy(&fdata[0]+dataOffset, h_HaarFeatures.ptr(), szFeatures);
    dataOffset += szFeatures;
    Ncv32u fsize = dataOffset;

    //TODO: CRC32 here

    //update header
    dataOffset = sizeof(Ncv32u);
    *(Ncv32u *)(&fdata[0]+dataOffset) = fsize;

    FILE *fp = fopen(filename.c_str(), "wb");
    ncvAssertReturn(fp != NULL, NCV_FILE_ERROR);
    fwrite(&fdata[0], fsize, 1, fp);
    fclose(fp);
    return NCV_SUCCESS;
}

#endif /* CUDA_DISABLER */
