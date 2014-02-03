/*M/////////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
#include "surf.hpp"

#include <cstdio>
#include <sstream>
#include "opencl_kernels.hpp"

namespace cv
{

enum { ORI_SEARCH_INC=5, ORI_LOCAL_SIZE=(360 / ORI_SEARCH_INC) };

/*static void openCLExecuteKernelSURF(Context2 *clCxt, const ProgramEntry* source, String kernelName, size_t globalThreads[3],
    size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels, int depth)
{
    std::stringstream optsStr;
    optsStr << "-D ORI_LOCAL_SIZE=" << ORI_LOCAL_SIZE << " ";
    optsStr << "-D ORI_SEARCH_INC=" << ORI_SEARCH_INC << " ";
    cl_kernel kernel;
    kernel = openCLGetKernelFromSource(clCxt, source, kernelName, optsStr.str().c_str());
    size_t wave_size = queryWaveFrontSize(kernel);
    CV_Assert(clReleaseKernel(kernel) == CL_SUCCESS);
    optsStr << "-D WAVE_SIZE=" << wave_size;
    openCLExecuteKernel(clCxt, source, kernelName, globalThreads, localThreads, args, channels, depth, optsStr.str().c_str());
}*/

static inline int calcSize(int octave, int layer)
{
    /* Wavelet size at first layer of first octave. */
    const int HAAR_SIZE0 = 9;

    /* Wavelet size increment between layers. This should be an even number,
    such that the wavelet sizes in an octave are either all even or all odd.
    This ensures that when looking for the neighbors of a sample, the layers

    above and below are aligned correctly. */
    const int HAAR_SIZE_INC = 6;

    return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave;
}


SURF_OCL::SURF_OCL()
{
    img_cols = img_rows = maxCandidates = maxFeatures = 0;
    haveImageSupport = false;
    status = -1;
}

bool SURF_OCL::init(const SURF* p)
{
    params = p;
    if(status < 0)
    {
        status = 0;
        if(ocl::haveOpenCL())
        {
            const ocl::Device& dev = ocl::Device::getDefault();
            if( dev.type() == ocl::Device::TYPE_CPU )
                return false;
            haveImageSupport = dev.imageSupport();
            String opts = haveImageSupport ? "-D DISABLE_IMAGE2D" : "";

            if( kerCalcDetTrace.create("SURF_calcLayerDetAndTrace", ocl::nonfree::surf_oclsrc, opts) &&
                kerFindMaxima.create("SURF_findMaximaInLayer", ocl::nonfree::surf_oclsrc, opts) &&
                kerFindMaximaMask.create("SURF_findMaximaInLayerWithMask", ocl::nonfree::surf_oclsrc, opts) &&
                kerInterp.create("SURF_interpolateKeypoint", ocl::nonfree::surf_oclsrc, opts) &&
                kerUpRight.create("SURF_setUpRight", ocl::nonfree::surf_oclsrc, opts) &&
                kerOri.create("SURF_calcOrientation", ocl::nonfree::surf_oclsrc, opts) &&
                kerCalcDesc64.create("SURF_computeDescriptors64", ocl::nonfree::surf_oclsrc, opts) &&
                kerCalcDesc128.create("SURF_computeDescriptors128", ocl::nonfree::surf_oclsrc, opts) &&
                kerNormDesc64.create("SURF_normalizeDescriptors64", ocl::nonfree::surf_oclsrc, opts) &&
                kerNormDesc128.create("SURF_normalizeDescriptors128", ocl::nonfree::surf_oclsrc, opts))
                status = 1;
        }
    }
    return status > 0;
}


bool SURF_OCL::setImage(InputArray _img, InputArray _mask)
{
    if( status <= 0 )
        return false;
    CV_Assert(!_img.empty() && _img.type() == CV_8UC1);
    CV_Assert(_mask.empty() || (_mask.size() == _img.size() && _mask.type() == CV_8UC1));
    CV_Assert(params && params->nOctaves > 0 && params->nOctaveLayers > 0);

    int min_size = calcSize(params->nOctaves - 1, 0);
    Size sz = _img.size();
    img_cols = sz.width;
    img_rows = sz.height;
    CV_Assert(img_rows >= min_size && img_cols >= min_size);

    const int layer_rows = img_rows >> (params->nOctaves - 1);
    const int layer_cols = img_cols >> (params->nOctaves - 1);
    const int min_margin = ((calcSize((params->nOctaves - 1), 2) >> 1) >> (params->nOctaves - 1)) + 1;
    CV_Assert(layer_rows - 2 * min_margin > 0);
    CV_Assert(layer_cols - 2 * min_margin > 0);

    maxFeatures   = std::min(static_cast<int>(img_cols*img_rows * 0.01f), 65535);
    maxCandidates = std::min(static_cast<int>(1.5 * maxFeatures), 65535);

    CV_Assert(maxFeatures > 0);

    counters.create(1, params->nOctaves + 1, CV_32SC1);
    counters.setTo(Scalar::all(0));

    img.release();
    if(_img.isUMat())
        img = _img.getUMat();
    else
        _img.copyTo(img);

    integral(img, sum);

    if(haveImageSupport)
    {
        imgTex = ocl::Image2D(img);
        sumTex = ocl::Image2D(sum);
    }

    maskSumTex = ocl::Image2D();

    if(!_mask.empty())
    {
        CV_Error(Error::StsBadFunc, "Masked SURF detector is not implemented yet");
    }
    return true;
}


bool SURF_OCL::detectKeypoints(UMat &keypoints)
{
    // create image pyramid buffers
    // different layers have same sized buffers, but they are sampled from Gaussian kernel.
    det.create(img_rows * (params->nOctaveLayers + 2), img_cols, CV_32F);
    trace.create(img_rows * (params->nOctaveLayers + 2), img_cols, CV_32FC1);

    maxPosBuffer.create(1, maxCandidates, CV_32SC4);
    keypoints.create(SURF_OCL::ROWS_COUNT, maxFeatures, CV_32F);
    keypoints.setTo(Scalar::all(0));
    Mat cpuCounters;

    for (int octave = 0; octave < params->nOctaves; ++octave)
    {
        const int layer_rows = img_rows >> octave;
        const int layer_cols = img_cols >> octave;

        if(!calcLayerDetAndTrace(det, trace, octave, layer_rows))
            return false;

        if(!findMaximaInLayer(det, trace, maxPosBuffer, counters, 1 + octave, octave,
                              layer_rows, layer_cols))
            return false;

        cpuCounters = counters.getMat(ACCESS_READ);
        int maxCounter = cpuCounters.at<int>(1 + octave);
        maxCounter = std::min(maxCounter, maxCandidates);
        cpuCounters.release();

        if (maxCounter > 0)
        {
            if(!interpolateKeypoint(det, maxPosBuffer, maxCounter, keypoints,
                                    counters, octave, layer_rows, maxFeatures))
                return false;
        }
    }

    cpuCounters = counters.getMat(ACCESS_READ);
    int featureCounter = cpuCounters.at<int>(0);
    featureCounter = std::min(featureCounter, maxFeatures);
    cpuCounters.release();

    keypoints = UMat(keypoints, Rect(0, 0, featureCounter, 1));

    if (params->upright)
        return setUpRight(keypoints);
    else
        return calcOrientation(keypoints);
}


bool SURF_OCL::setUpRight(UMat &keypoints)
{
    int nFeatures = keypoints.cols;
    if( nFeatures == 0 )
        return true;

    size_t globalThreads[3] = {nFeatures, 1};
    return kerUpRight.args(ocl::KernelArg::ReadWrite(keypoints)).run(2, globalThreads, 0, false);
}

bool SURF_OCL::computeDescriptors(const UMat &keypoints, OutputArray _descriptors)
{
    int descriptorSize = params->descriptorSize();
    int nFeatures = keypoints.cols;
    if (nFeatures == 0)
    {
        _descriptors.release();
        return true;
    }
    _descriptors.create(nFeatures, descriptorSize, CV_32F);
    UMat descriptors;
    if( _descriptors.isUMat() )
        descriptors = _descriptors.getUMat();
    else
        descriptors.create(nFeatures, descriptorSize, CV_32F);

    ocl::Kernel kerCalcDesc, kerNormDesc;

    if( descriptorSize == 64 )
    {
        kerCalcDesc = kerCalcDesc64;
        kerNormDesc = kerNormDesc64;
    }
    else
    {
        CV_Assert(descriptorSize == 128);
        kerCalcDesc = kerCalcDesc128;
        kerNormDesc = kerNormDesc128;
    }

    size_t localThreads[] = {6, 6};
    size_t globalThreads[] = {nFeatures*localThreads[0], localThreads[1]};

    if(haveImageSupport)
    {
        kerCalcDesc.args(imgTex,
                         ocl::KernelArg::ReadOnlyNoSize(keypoints),
                         ocl::KernelArg::WriteOnlyNoSize(descriptors));
    }
    else
    {
        kerCalcDesc.args(ocl::KernelArg::ReadOnly(img),
                         ocl::KernelArg::ReadOnlyNoSize(keypoints),
                         ocl::KernelArg::WriteOnlyNoSize(descriptors));
    }

    if(!kerCalcDesc.run(2, globalThreads, localThreads, false))
        return false;

    size_t localThreads_n[] = {descriptorSize, 1};
    size_t globalThreads_n[] = {nFeatures*localThreads_n[0], localThreads_n[1]};

    globalThreads[0] = nFeatures * localThreads[0];
    globalThreads[1] = localThreads[1];
    bool ok = kerNormDesc.args(ocl::KernelArg::ReadWriteNoSize(descriptors)).
                        run(2, globalThreads_n, localThreads_n, false);
    if(ok && !_descriptors.isUMat())
        descriptors.copyTo(_descriptors);
    return ok;
}


void SURF_OCL::uploadKeypoints(const std::vector<KeyPoint> &keypoints, UMat &keypointsGPU)
{
    if (keypoints.empty())
        keypointsGPU.release();
    else
    {
        Mat keypointsCPU(SURF_OCL::ROWS_COUNT, static_cast<int>(keypoints.size()), CV_32FC1);

        float *kp_x = keypointsCPU.ptr<float>(SURF_OCL::X_ROW);
        float *kp_y = keypointsCPU.ptr<float>(SURF_OCL::Y_ROW);
        int *kp_laplacian = keypointsCPU.ptr<int>(SURF_OCL::LAPLACIAN_ROW);
        int *kp_octave = keypointsCPU.ptr<int>(SURF_OCL::OCTAVE_ROW);
        float *kp_size = keypointsCPU.ptr<float>(SURF_OCL::SIZE_ROW);
        float *kp_dir = keypointsCPU.ptr<float>(SURF_OCL::ANGLE_ROW);
        float *kp_hessian = keypointsCPU.ptr<float>(SURF_OCL::HESSIAN_ROW);

        for (size_t i = 0, size = keypoints.size(); i < size; ++i)
        {
            const KeyPoint &kp = keypoints[i];
            kp_x[i] = kp.pt.x;
            kp_y[i] = kp.pt.y;
            kp_octave[i] = kp.octave;
            kp_size[i] = kp.size;
            kp_dir[i] = kp.angle;
            kp_hessian[i] = kp.response;
            kp_laplacian[i] = 1;
        }

        keypointsCPU.copyTo(keypointsGPU);
    }
}

void SURF_OCL::downloadKeypoints(const UMat &keypointsGPU, std::vector<KeyPoint> &keypoints)
{
    const int nFeatures = keypointsGPU.cols;

    if (nFeatures == 0)
        keypoints.clear();
    else
    {
        CV_Assert(keypointsGPU.type() == CV_32FC1 && keypointsGPU.rows == ROWS_COUNT);

        Mat keypointsCPU = keypointsGPU.getMat(ACCESS_READ);
        keypoints.resize(nFeatures);

        float *kp_x = keypointsCPU.ptr<float>(SURF_OCL::X_ROW);
        float *kp_y = keypointsCPU.ptr<float>(SURF_OCL::Y_ROW);
        int *kp_laplacian = keypointsCPU.ptr<int>(SURF_OCL::LAPLACIAN_ROW);
        int *kp_octave = keypointsCPU.ptr<int>(SURF_OCL::OCTAVE_ROW);
        float *kp_size = keypointsCPU.ptr<float>(SURF_OCL::SIZE_ROW);
        float *kp_dir = keypointsCPU.ptr<float>(SURF_OCL::ANGLE_ROW);
        float *kp_hessian = keypointsCPU.ptr<float>(SURF_OCL::HESSIAN_ROW);

        for (int i = 0; i < nFeatures; ++i)
        {
            KeyPoint &kp = keypoints[i];
            kp.pt.x = kp_x[i];
            kp.pt.y = kp_y[i];
            kp.class_id = kp_laplacian[i];
            kp.octave = kp_octave[i];
            kp.size = kp_size[i];
            kp.angle = kp_dir[i];
            kp.response = kp_hessian[i];
        }
    }
}

bool SURF_OCL::detect(InputArray img, InputArray mask, UMat& keypoints)
{
    if( !setImage(img, mask) )
        return false;

    return detectKeypoints(keypoints);
}


bool SURF_OCL::detectAndCompute(InputArray img, InputArray mask, UMat& keypoints,
                                OutputArray _descriptors, bool useProvidedKeypoints )
{
    if( !setImage(img, mask) )
        return false;

    if( !useProvidedKeypoints && !detectKeypoints(keypoints) )
        return false;

    return computeDescriptors(keypoints, _descriptors);
}

inline int divUp(int a, int b) { return (a + b-1)/b; }

////////////////////////////
// kernel caller definitions
bool SURF_OCL::calcLayerDetAndTrace(UMat &det, UMat &trace, int octave, int c_layer_rows)
{
    int nOctaveLayers = params->nOctaveLayers;
    const int min_size = calcSize(octave, 0);
    const int max_samples_i = 1 + ((img_rows - min_size) >> octave);
    const int max_samples_j = 1 + ((img_cols - min_size) >> octave);

    String kernelName = "SURF_calcLayerDetAndTrace";
    std::vector< std::pair<size_t, const void *> > args;

    size_t localThreads[3]  = {16, 16};
    size_t globalThreads[3] =
    {
        divUp(max_samples_j, localThreads[0]) *localThreads[0],
        divUp(max_samples_i, localThreads[1]) *localThreads[1] *(nOctaveLayers + 2)
    };
    if(haveImageSupport)
    {
        kerCalcDetTrace.args(sumTex,
                             img_rows, img_cols, nOctaveLayers,
                             octave, c_layer_rows,
                             ocl::KernelArg::WriteOnlyNoSize(det),
                             ocl::KernelArg::WriteOnlyNoSize(trace));
    }
    else
    {
        kerCalcDetTrace.args(ocl::KernelArg::ReadOnlyNoSize(sum),
                             img_rows, img_cols, nOctaveLayers,
                             octave, c_layer_rows,
                             ocl::KernelArg::WriteOnlyNoSize(det),
                             ocl::KernelArg::WriteOnlyNoSize(trace));
    }
    return kerCalcDetTrace.run(2, globalThreads, localThreads, false);
}

bool SURF_OCL::findMaximaInLayer(const UMat &det, const UMat &trace,
                                 UMat &maxPosBuffer, UMat &maxCounter,
                                 int counterOffset, int octave,
                                 int layer_rows, int layer_cols)
{
    const int min_margin = ((calcSize(octave, 2) >> 1) >> octave) + 1;
    bool haveMask = !maskSum.empty() || (maskSumTex.ptr() != 0);
    int nOctaveLayers = params->nOctaveLayers;

    ocl::Kernel ker;
    if( haveMask )
    {
        if( haveImageSupport )
            ker = kerFindMaximaMask.args(maskSumTex,
                           ocl::KernelArg::ReadOnlyNoSize(det),
                           ocl::KernelArg::ReadOnlyNoSize(trace),
                           ocl::KernelArg::PtrReadWrite(maxPosBuffer),
                           ocl::KernelArg::PtrReadWrite(maxCounter),
                           counterOffset, img_rows, img_cols,
                           octave, nOctaveLayers,
                           layer_rows, layer_cols,
                           maxCandidates,
                           (float)params->hessianThreshold);
        else
            ker = kerFindMaximaMask.args(ocl::KernelArg::ReadOnlyNoSize(maskSum),
                           ocl::KernelArg::ReadOnlyNoSize(det),
                           ocl::KernelArg::ReadOnlyNoSize(trace),
                           ocl::KernelArg::PtrReadWrite(maxPosBuffer),
                           ocl::KernelArg::PtrReadWrite(maxCounter),
                           counterOffset, img_rows, img_cols,
                           octave, nOctaveLayers,
                           layer_rows, layer_cols,
                           maxCandidates,
                           (float)params->hessianThreshold);
    }
    else
    {
        ker = kerFindMaxima.args(ocl::KernelArg::ReadOnlyNoSize(det),
                                 ocl::KernelArg::ReadOnlyNoSize(trace),
                                 ocl::KernelArg::PtrReadWrite(maxPosBuffer),
                                 ocl::KernelArg::PtrReadWrite(maxCounter),
                                 counterOffset, img_rows, img_cols,
                                 octave, nOctaveLayers,
                                 layer_rows, layer_cols,
                                 maxCandidates,
                                 (float)params->hessianThreshold);
    }
    size_t localThreads[3]  = {16, 16};
    size_t globalThreads[3] =
    {
        divUp(layer_cols - 2 * min_margin, localThreads[0] - 2) *localThreads[0],
        divUp(layer_rows - 2 * min_margin, localThreads[1] - 2) *nOctaveLayers *localThreads[1]
    };

    return ker.run(2, globalThreads, localThreads, false);
}

bool SURF_OCL::interpolateKeypoint(const UMat &det, const UMat &maxPosBuffer, int maxCounter,
        UMat &keypoints, UMat &counters_, int octave, int layer_rows, int max_features)
{
    size_t localThreads[3]  = {3, 3, 3};
    size_t globalThreads[3] = {maxCounter*localThreads[0], localThreads[1], 3};

    return kerInterp.args(ocl::KernelArg::ReadOnlyNoSize(det),
                   ocl::KernelArg::PtrReadOnly(maxPosBuffer),
                   ocl::KernelArg::ReadWriteNoSize(keypoints),
                   ocl::KernelArg::PtrReadWrite(counters_),
                   img_rows, img_cols, octave, layer_rows, max_features).
        run(3, globalThreads, localThreads, false);
}

bool SURF_OCL::calcOrientation(UMat &keypoints)
{
    int nFeatures = keypoints.cols;
    if( nFeatures == 0 )
        return true;
    if( haveImageSupport )
        kerOri.args(sumTex,
                    ocl::KernelArg::ReadWriteNoSize(keypoints),
                    img_rows, img_cols);
    else
        kerOri.args(ocl::KernelArg::ReadOnlyNoSize(sum),
                    ocl::KernelArg::ReadWriteNoSize(keypoints),
                    img_rows, img_cols);

    size_t localThreads[3]  = {ORI_LOCAL_SIZE, 1};
    size_t globalThreads[3] = {nFeatures * localThreads[0], 1};
    return kerOri.run(2, globalThreads, localThreads, false);
}

}
