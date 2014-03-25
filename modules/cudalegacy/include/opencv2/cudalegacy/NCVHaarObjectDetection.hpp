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

#ifndef _ncvhaarobjectdetection_hpp_
#define _ncvhaarobjectdetection_hpp_

#include "opencv2/cudalegacy/NCV.hpp"


//==============================================================================
//
// Guaranteed size cross-platform classifier structures
//
//==============================================================================
#if defined __GNUC__ && __GNUC__ > 2 && __GNUC_MINOR__  > 4
typedef Ncv32f __attribute__((__may_alias__)) Ncv32f_a;
#else
typedef Ncv32f Ncv32f_a;
#endif

struct HaarFeature64
{
    uint2 _ui2;

#define HaarFeature64_CreateCheck_MaxRectField                  0xFF

    __host__ NCVStatus setRect(Ncv32u rectX, Ncv32u rectY, Ncv32u rectWidth, Ncv32u rectHeight, Ncv32u /*clsWidth*/, Ncv32u /*clsHeight*/)
    {
        ncvAssertReturn(rectWidth <= HaarFeature64_CreateCheck_MaxRectField && rectHeight <= HaarFeature64_CreateCheck_MaxRectField, NCV_HAAR_TOO_LARGE_FEATURES);
        ((NcvRect8u*)&(this->_ui2.x))->x = (Ncv8u)rectX;
        ((NcvRect8u*)&(this->_ui2.x))->y = (Ncv8u)rectY;
        ((NcvRect8u*)&(this->_ui2.x))->width = (Ncv8u)rectWidth;
        ((NcvRect8u*)&(this->_ui2.x))->height = (Ncv8u)rectHeight;
        return NCV_SUCCESS;
    }

    __host__ NCVStatus setWeight(Ncv32f weight)
    {
        ((Ncv32f_a*)&(this->_ui2.y))[0] = weight;
        return NCV_SUCCESS;
    }

    __device__ __host__ void getRect(Ncv32u *rectX, Ncv32u *rectY, Ncv32u *rectWidth, Ncv32u *rectHeight)
    {
        NcvRect8u tmpRect = *(NcvRect8u*)(&this->_ui2.x);
        *rectX = tmpRect.x;
        *rectY = tmpRect.y;
        *rectWidth = tmpRect.width;
        *rectHeight = tmpRect.height;
    }

    __device__ __host__ Ncv32f getWeight(void)
    {
        return *(Ncv32f_a*)(&this->_ui2.y);
    }
};


struct HaarFeatureDescriptor32
{
private:

#define HaarFeatureDescriptor32_Interpret_MaskFlagTilted        0x80000000
#define HaarFeatureDescriptor32_Interpret_MaskFlagLeftNodeLeaf  0x40000000
#define HaarFeatureDescriptor32_Interpret_MaskFlagRightNodeLeaf 0x20000000
#define HaarFeatureDescriptor32_CreateCheck_MaxNumFeatures      0x1F
#define HaarFeatureDescriptor32_NumFeatures_Shift               24
#define HaarFeatureDescriptor32_CreateCheck_MaxFeatureOffset    0x00FFFFFF

    Ncv32u desc;

public:

    __host__ NCVStatus create(NcvBool bTilted, NcvBool bLeftLeaf, NcvBool bRightLeaf,
                              Ncv32u numFeatures, Ncv32u offsetFeatures)
    {
        if (numFeatures > HaarFeatureDescriptor32_CreateCheck_MaxNumFeatures)
        {
            return NCV_HAAR_TOO_MANY_FEATURES_IN_CLASSIFIER;
        }
        if (offsetFeatures > HaarFeatureDescriptor32_CreateCheck_MaxFeatureOffset)
        {
            return NCV_HAAR_TOO_MANY_FEATURES_IN_CASCADE;
        }
        this->desc = 0;
        this->desc |= (bTilted ? HaarFeatureDescriptor32_Interpret_MaskFlagTilted : 0);
        this->desc |= (bLeftLeaf ? HaarFeatureDescriptor32_Interpret_MaskFlagLeftNodeLeaf : 0);
        this->desc |= (bRightLeaf ? HaarFeatureDescriptor32_Interpret_MaskFlagRightNodeLeaf : 0);
        this->desc |= (numFeatures << HaarFeatureDescriptor32_NumFeatures_Shift);
        this->desc |= offsetFeatures;
        return NCV_SUCCESS;
    }

    __device__ __host__ NcvBool isTilted(void)
    {
        return (this->desc & HaarFeatureDescriptor32_Interpret_MaskFlagTilted) != 0;
    }

    __device__ __host__ NcvBool isLeftNodeLeaf(void)
    {
        return (this->desc & HaarFeatureDescriptor32_Interpret_MaskFlagLeftNodeLeaf) != 0;
    }

    __device__ __host__ NcvBool isRightNodeLeaf(void)
    {
        return (this->desc & HaarFeatureDescriptor32_Interpret_MaskFlagRightNodeLeaf) != 0;
    }

    __device__ __host__ Ncv32u getNumFeatures(void)
    {
        return (this->desc >> HaarFeatureDescriptor32_NumFeatures_Shift) & HaarFeatureDescriptor32_CreateCheck_MaxNumFeatures;
    }

    __device__ __host__ Ncv32u getFeaturesOffset(void)
    {
        return this->desc & HaarFeatureDescriptor32_CreateCheck_MaxFeatureOffset;
    }
};

struct HaarClassifierNodeDescriptor32
{
    uint1 _ui1;

    __host__ NCVStatus create(Ncv32f leafValue)
    {
        *(Ncv32f_a *)&this->_ui1 = leafValue;
        return NCV_SUCCESS;
    }

    __host__ NCVStatus create(Ncv32u offsetHaarClassifierNode)
    {
        this->_ui1.x = offsetHaarClassifierNode;
        return NCV_SUCCESS;
    }

    __host__ Ncv32f getLeafValueHost(void)
    {
        return *(Ncv32f_a *)&this->_ui1.x;
    }

#ifdef __CUDACC__
    __device__ Ncv32f getLeafValue(void)
    {
        return __int_as_float(this->_ui1.x);
    }
#endif

    __device__ __host__ Ncv32u getNextNodeOffset(void)
    {
        return this->_ui1.x;
    }
};

#if defined __GNUC__ && __GNUC__ > 2 && __GNUC_MINOR__  > 4
typedef Ncv32u __attribute__((__may_alias__)) Ncv32u_a;
#else
typedef Ncv32u Ncv32u_a;
#endif

struct HaarClassifierNode128
{
    uint4 _ui4;

    __host__ NCVStatus setFeatureDesc(HaarFeatureDescriptor32 f)
    {
        this->_ui4.x = *(Ncv32u *)&f;
        return NCV_SUCCESS;
    }

    __host__ NCVStatus setThreshold(Ncv32f t)
    {
        this->_ui4.y = *(Ncv32u_a *)&t;
        return NCV_SUCCESS;
    }

    __host__ NCVStatus setLeftNodeDesc(HaarClassifierNodeDescriptor32 nl)
    {
        this->_ui4.z = *(Ncv32u_a *)&nl;
        return NCV_SUCCESS;
    }

    __host__ NCVStatus setRightNodeDesc(HaarClassifierNodeDescriptor32 nr)
    {
        this->_ui4.w = *(Ncv32u_a *)&nr;
        return NCV_SUCCESS;
    }

    __host__ __device__ HaarFeatureDescriptor32 getFeatureDesc(void)
    {
        return *(HaarFeatureDescriptor32 *)&this->_ui4.x;
    }

    __host__ __device__ Ncv32f getThreshold(void)
    {
        return *(Ncv32f_a*)&this->_ui4.y;
    }

    __host__ __device__ HaarClassifierNodeDescriptor32 getLeftNodeDesc(void)
    {
        return *(HaarClassifierNodeDescriptor32 *)&this->_ui4.z;
    }

    __host__ __device__ HaarClassifierNodeDescriptor32 getRightNodeDesc(void)
    {
        return *(HaarClassifierNodeDescriptor32 *)&this->_ui4.w;
    }
};


struct HaarStage64
{
#define HaarStage64_Interpret_MaskRootNodes         0x0000FFFF
#define HaarStage64_Interpret_MaskRootNodeOffset    0xFFFF0000
#define HaarStage64_Interpret_ShiftRootNodeOffset   16

    uint2 _ui2;

    __host__ NCVStatus setStageThreshold(Ncv32f t)
    {
        this->_ui2.x = *(Ncv32u_a *)&t;
        return NCV_SUCCESS;
    }

    __host__ NCVStatus setStartClassifierRootNodeOffset(Ncv32u val)
    {
        if (val > (HaarStage64_Interpret_MaskRootNodeOffset >> HaarStage64_Interpret_ShiftRootNodeOffset))
        {
            return NCV_HAAR_XML_LOADING_EXCEPTION;
        }
        this->_ui2.y = (val << HaarStage64_Interpret_ShiftRootNodeOffset) | (this->_ui2.y & HaarStage64_Interpret_MaskRootNodes);
        return NCV_SUCCESS;
    }

    __host__ NCVStatus setNumClassifierRootNodes(Ncv32u val)
    {
        if (val > HaarStage64_Interpret_MaskRootNodes)
        {
            return NCV_HAAR_XML_LOADING_EXCEPTION;
        }
        this->_ui2.y = val | (this->_ui2.y & HaarStage64_Interpret_MaskRootNodeOffset);
        return NCV_SUCCESS;
    }

    __host__ __device__ Ncv32f getStageThreshold(void)
    {
        return *(Ncv32f_a*)&this->_ui2.x;
    }

    __host__ __device__ Ncv32u getStartClassifierRootNodeOffset(void)
    {
        return (this->_ui2.y >> HaarStage64_Interpret_ShiftRootNodeOffset);
    }

    __host__ __device__ Ncv32u getNumClassifierRootNodes(void)
    {
        return (this->_ui2.y & HaarStage64_Interpret_MaskRootNodes);
    }
};


NCV_CT_ASSERT(sizeof(HaarFeature64) == 8);
NCV_CT_ASSERT(sizeof(HaarFeatureDescriptor32) == 4);
NCV_CT_ASSERT(sizeof(HaarClassifierNodeDescriptor32) == 4);
NCV_CT_ASSERT(sizeof(HaarClassifierNode128) == 16);
NCV_CT_ASSERT(sizeof(HaarStage64) == 8);


//==============================================================================
//
// Classifier cascade descriptor
//
//==============================================================================


struct HaarClassifierCascadeDescriptor
{
    Ncv32u NumStages;
    Ncv32u NumClassifierRootNodes;
    Ncv32u NumClassifierTotalNodes;
    Ncv32u NumFeatures;
    NcvSize32u ClassifierSize;
    NcvBool bNeedsTiltedII;
    NcvBool bHasStumpsOnly;
};


//==============================================================================
//
// Functional interface
//
//==============================================================================


enum
{
    NCVPipeObjDet_Default               = 0x000,
    NCVPipeObjDet_UseFairImageScaling   = 0x001,
    NCVPipeObjDet_FindLargestObject     = 0x002,
    NCVPipeObjDet_VisualizeInPlace      = 0x004,
};


CV_EXPORTS NCVStatus ncvDetectObjectsMultiScale_device(NCVMatrix<Ncv8u> &d_srcImg,
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
                                                        cudaStream_t cuStream);


#define OBJDET_MASK_ELEMENT_INVALID_32U     0xFFFFFFFF
#define HAAR_STDDEV_BORDER                  1


CV_EXPORTS NCVStatus ncvApplyHaarClassifierCascade_device(NCVMatrix<Ncv32u> &d_integralImage,
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
                                                           cudaStream_t cuStream);


CV_EXPORTS NCVStatus ncvApplyHaarClassifierCascade_host(NCVMatrix<Ncv32u> &h_integralImage,
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
                                                         Ncv32f scaleArea);


#define RECT_SIMILARITY_PROPORTION      0.2f


CV_EXPORTS NCVStatus ncvGrowDetectionsVector_device(NCVVector<Ncv32u> &pixelMask,
                                                     Ncv32u numPixelMaskDetections,
                                                     NCVVector<NcvRect32u> &hypotheses,
                                                     Ncv32u &totalDetections,
                                                     Ncv32u totalMaxDetections,
                                                     Ncv32u rectWidth,
                                                     Ncv32u rectHeight,
                                                     Ncv32f curScale,
                                                     cudaStream_t cuStream);


CV_EXPORTS NCVStatus ncvGrowDetectionsVector_host(NCVVector<Ncv32u> &pixelMask,
                                                   Ncv32u numPixelMaskDetections,
                                                   NCVVector<NcvRect32u> &hypotheses,
                                                   Ncv32u &totalDetections,
                                                   Ncv32u totalMaxDetections,
                                                   Ncv32u rectWidth,
                                                   Ncv32u rectHeight,
                                                   Ncv32f curScale);


CV_EXPORTS NCVStatus ncvHaarGetClassifierSize(const cv::String &filename, Ncv32u &numStages,
                                               Ncv32u &numNodes, Ncv32u &numFeatures);


CV_EXPORTS NCVStatus ncvHaarLoadFromFile_host(const cv::String &filename,
                                               HaarClassifierCascadeDescriptor &haar,
                                               NCVVector<HaarStage64> &h_HaarStages,
                                               NCVVector<HaarClassifierNode128> &h_HaarNodes,
                                               NCVVector<HaarFeature64> &h_HaarFeatures);


CV_EXPORTS NCVStatus ncvHaarStoreNVBIN_host(const cv::String &filename,
                                             HaarClassifierCascadeDescriptor haar,
                                             NCVVector<HaarStage64> &h_HaarStages,
                                             NCVVector<HaarClassifierNode128> &h_HaarNodes,
                                             NCVVector<HaarFeature64> &h_HaarFeatures);



#endif // _ncvhaarobjectdetection_hpp_
