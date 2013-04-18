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

#ifndef _npp_staging_hpp_
#define _npp_staging_hpp_

#include "NCV.hpp"


/**
* \file NPP_staging.hpp
* NPP Staging Library
*/


/** \defgroup core_npp NPPST Core
 * Basic functions for CUDA streams management.
 * @{
 */


/**
 * Gets an active CUDA stream used by NPPST
 * NOT THREAD SAFE
 * \return Current CUDA stream
 */
NCV_EXPORTS
cudaStream_t nppStGetActiveCUDAstream();


/**
 * Sets an active CUDA stream used by NPPST
 * NOT THREAD SAFE
 * \param cudaStream        [IN] cudaStream CUDA stream to become current
 * \return CUDA stream used before
 */
NCV_EXPORTS
cudaStream_t nppStSetActiveCUDAstream(cudaStream_t cudaStream);


/*@}*/


/** \defgroup nppi NPPST Image Processing
* @{
*/


/** Border type
 *
 * Filtering operations assume that each pixel has a neighborhood of pixels.
 * The following structure describes possible ways to define non-existent pixels.
 */
enum NppStBorderType
{
    nppStBorderNone   = 0, ///< There is no need to define additional pixels, image is extended already
    nppStBorderClamp  = 1, ///< Clamp out of range position to borders
    nppStBorderWrap   = 2, ///< Wrap out of range position. Image becomes periodic.
    nppStBorderMirror = 3  ///< reflect out of range position across borders
};


/**
 * Filter types for image resizing
 */
enum NppStInterpMode
{
    nppStSupersample, ///< Supersampling. For downscaling only
    nppStBicubic      ///< Bicubic convolution filter, a = -0.5 (cubic Hermite spline)
};


/** Frame interpolation state
 *
 * This structure holds parameters required for frame interpolation.
 * Forward displacement field is a per-pixel mapping from frame 0 to frame 1.
 * Backward displacement field is a per-pixel mapping from frame 1 to frame 0.
 */

 struct NppStInterpolationState
{
    NcvSize32u size;      ///< frame size
    Ncv32u nStep;         ///< pitch
    Ncv32f pos;           ///< new frame position
    Ncv32f *pSrcFrame0;   ///< frame 0
    Ncv32f *pSrcFrame1;   ///< frame 1
    Ncv32f *pFU;          ///< forward horizontal displacement
    Ncv32f *pFV;          ///< forward vertical displacement
    Ncv32f *pBU;          ///< backward horizontal displacement
    Ncv32f *pBV;          ///< backward vertical displacement
    Ncv32f *pNewFrame;    ///< new frame
    Ncv32f *ppBuffers[6]; ///< temporary buffers
};


/** Size of a buffer required for interpolation.
 *
 * Requires several such buffers. See \see NppStInterpolationState.
 *
 * \param srcSize           [IN]  Frame size (both frames must be of the same size)
 * \param nStep             [IN]  Frame line step
 * \param hpSize            [OUT] Where to store computed size (host memory)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStGetInterpolationBufferSize(NcvSize32u srcSize,
                                           Ncv32u nStep,
                                           Ncv32u *hpSize);


/** Interpolate frames (images) using provided optical flow (displacement field).
 * 32-bit floating point images, single channel
 *
 * \param pState            [IN] structure containing all required parameters (host memory)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStInterpolateFrames(const NppStInterpolationState *pState);


/** Row linear filter. 32-bit floating point image, single channel
 *
 * Apply horizontal linear filter
 *
 * \param pSrc              [IN]  Source image pointer (CUDA device memory)
 * \param srcSize           [IN]  Source image size
 * \param nSrcStep          [IN]  Source image line step
 * \param pDst              [OUT] Destination image pointer (CUDA device memory)
 * \param dstSize           [OUT] Destination image size
 * \param oROI              [IN]  Region of interest in the source image
 * \param borderType        [IN]  Type of border
 * \param pKernel           [IN]  Pointer to row kernel values (CUDA device memory)
 * \param nKernelSize       [IN]  Size of the kernel in pixels
 * \param nAnchor           [IN]  The kernel row alignment with respect to the position of the input pixel
 * \param multiplier        [IN]  Value by which the computed result is multiplied
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStFilterRowBorder_32f_C1R(const Ncv32f *pSrc,
                                        NcvSize32u srcSize,
                                        Ncv32u nSrcStep,
                                        Ncv32f *pDst,
                                        NcvSize32u dstSize,
                                        Ncv32u nDstStep,
                                        NcvRect32u oROI,
                                        NppStBorderType borderType,
                                        const Ncv32f *pKernel,
                                        Ncv32s nKernelSize,
                                        Ncv32s nAnchor,
                                        Ncv32f multiplier);


/** Column linear filter. 32-bit floating point image, single channel
 *
 * Apply vertical linear filter
 *
 * \param pSrc              [IN]  Source image pointer (CUDA device memory)
 * \param srcSize           [IN]  Source image size
 * \param nSrcStep          [IN]  Source image line step
 * \param pDst              [OUT] Destination image pointer (CUDA device memory)
 * \param dstSize           [OUT] Destination image size
 * \param oROI              [IN]  Region of interest in the source image
 * \param borderType        [IN]  Type of border
 * \param pKernel           [IN]  Pointer to column kernel values (CUDA device memory)
 * \param nKernelSize       [IN]  Size of the kernel in pixels
 * \param nAnchor           [IN]  The kernel column alignment with respect to the position of the input pixel
 * \param multiplier        [IN]  Value by which the computed result is multiplied
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStFilterColumnBorder_32f_C1R(const Ncv32f *pSrc,
                                           NcvSize32u srcSize,
                                           Ncv32u nSrcStep,
                                           Ncv32f *pDst,
                                           NcvSize32u dstSize,
                                           Ncv32u nDstStep,
                                           NcvRect32u oROI,
                                           NppStBorderType borderType,
                                           const Ncv32f *pKernel,
                                           Ncv32s nKernelSize,
                                           Ncv32s nAnchor,
                                           Ncv32f multiplier);


/** Size of buffer required for vector image warping.
 *
 * \param srcSize           [IN]  Source image size
 * \param nStep             [IN]  Source image line step
 * \param hpSize            [OUT] Where to store computed size (host memory)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStVectorWarpGetBufferSize(NcvSize32u srcSize,
                                        Ncv32u nSrcStep,
                                        Ncv32u *hpSize);


/** Warp image using provided 2D vector field and 1x1 point spread function.
 * 32-bit floating point image, single channel
 *
 * During warping pixels from the source image may fall between pixels of the destination image.
 * PSF (point spread function) describes how the source image pixel affects pixels of the destination.
 * For 1x1 PSF only single pixel with the largest intersection is affected (similar to nearest interpolation).
 *
 * Destination image size and line step must be the same as the source image size and line step
 *
 * \param pSrc              [IN]  Source image pointer (CUDA device memory)
 * \param srcSize           [IN]  Source image size
 * \param nSrcStep          [IN]  Source image line step
 * \param pU                [IN]  Pointer to horizontal displacement field (CUDA device memory)
 * \param pV                [IN]  Pointer to vertical displacement field (CUDA device memory)
 * \param nVFStep           [IN]  Displacement field line step
 * \param timeScale         [IN]  Value by which displacement field will be scaled for warping
 * \param pDst              [OUT] Destination image pointer (CUDA device memory)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStVectorWarp_PSF1x1_32f_C1(const Ncv32f *pSrc,
                                         NcvSize32u srcSize,
                                         Ncv32u nSrcStep,
                                         const Ncv32f *pU,
                                         const Ncv32f *pV,
                                         Ncv32u nVFStep,
                                         Ncv32f timeScale,
                                         Ncv32f *pDst);


/** Warp image using provided 2D vector field and 2x2 point spread function.
 * 32-bit floating point image, single channel
 *
 * During warping pixels from the source image may fall between pixels of the destination image.
 * PSF (point spread function) describes how the source image pixel affects pixels of the destination.
 * For 2x2 PSF all four intersected pixels will be affected.
 *
 * Destination image size and line step must be the same as the source image size and line step
 *
 * \param pSrc              [IN]  Source image pointer (CUDA device memory)
 * \param srcSize           [IN]  Source image size
 * \param nSrcStep          [IN]  Source image line step
 * \param pU                [IN]  Pointer to horizontal displacement field (CUDA device memory)
 * \param pV                [IN]  Pointer to vertical displacement field (CUDA device memory)
 * \param nVFStep           [IN]  Displacement field line step
 * \param timeScale         [IN]  Value by which displacement field will be scaled for warping
 * \param pDst              [OUT] Destination image pointer (CUDA device memory)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStVectorWarp_PSF2x2_32f_C1(const Ncv32f *pSrc,
                                         NcvSize32u srcSize,
                                         Ncv32u nSrcStep,
                                         const Ncv32f *pU,
                                         const Ncv32f *pV,
                                         Ncv32u nVFStep,
                                         Ncv32f *pBuffer,
                                         Ncv32f timeScale,
                                         Ncv32f *pDst);


/** Resize. 32-bit floating point image, single channel
 *
 * Resizes image using specified filter (interpolation type)
 *
 * \param pSrc              [IN]  Source image pointer (CUDA device memory)
 * \param srcSize           [IN]  Source image size
 * \param nSrcStep          [IN]  Source image line step
 * \param srcROI            [IN]  Source image region of interest
 * \param pDst              [OUT] Destination image pointer (CUDA device memory)
 * \param dstSize           [IN]  Destination image size
 * \param nDstStep          [IN]  Destination image line step
 * \param dstROI            [IN]  Destination image region of interest
 * \param xFactor           [IN]  Row scale factor
 * \param yFactor           [IN]  Column scale factor
 * \param interpolation     [IN]  Interpolation type
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStResize_32f_C1R(const Ncv32f *pSrc,
                               NcvSize32u srcSize,
                               Ncv32u nSrcStep,
                               NcvRect32u srcROI,
                               Ncv32f *pDst,
                               NcvSize32u dstSize,
                               Ncv32u nDstStep,
                               NcvRect32u dstROI,
                               Ncv32f xFactor,
                               Ncv32f yFactor,
                               NppStInterpMode interpolation);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit unsigned pixels, single channel.
 *
 * \param d_src             [IN] Source image pointer (CUDA device memory)
 * \param srcStep           [IN] Source image line step
 * \param d_dst             [OUT] Destination image pointer (CUDA device memory)
 * \param dstStep           [IN] Destination image line step
 * \param srcRoi            [IN] Region of interest in the source image
 * \param scale             [IN] Downsampling scale factor (positive integer)
 * \param readThruTexture   [IN] Performance hint to cache source in texture (true) or read directly (false)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_32u_C1R(Ncv32u *d_src, Ncv32u srcStep,
                                 Ncv32u *d_dst, Ncv32u dstStep,
                                 NcvSize32u srcRoi, Ncv32u scale,
                                 NcvBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit signed pixels, single channel.
 * \see nppiStDecimate_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_32s_C1R(Ncv32s *d_src, Ncv32u srcStep,
                                 Ncv32s *d_dst, Ncv32u dstStep,
                                 NcvSize32u srcRoi, Ncv32u scale,
                                 NcvBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit float pixels, single channel.
 * \see nppiStDecimate_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_32f_C1R(Ncv32f *d_src, Ncv32u srcStep,
                                 Ncv32f *d_dst, Ncv32u dstStep,
                                 NcvSize32u srcRoi, Ncv32u scale,
                                 NcvBool readThruTexture);


/**
* Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit unsigned pixels, single channel.
* \see nppiStDecimate_32u_C1R
*/
NCV_EXPORTS
NCVStatus nppiStDecimate_64u_C1R(Ncv64u *d_src, Ncv32u srcStep,
                                 Ncv64u *d_dst, Ncv32u dstStep,
                                 NcvSize32u srcRoi, Ncv32u scale,
                                 NcvBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit signed pixels, single channel.
 * \see nppiStDecimate_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_64s_C1R(Ncv64s *d_src, Ncv32u srcStep,
                                 Ncv64s *d_dst, Ncv32u dstStep,
                                 NcvSize32u srcRoi, Ncv32u scale,
                                 NcvBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit float pixels, single channel.
 * \see nppiStDecimate_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_64f_C1R(Ncv64f *d_src, Ncv32u srcStep,
                                 Ncv64f *d_dst, Ncv32u dstStep,
                                 NcvSize32u srcRoi, Ncv32u scale,
                                 NcvBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit unsigned pixels, single channel. Host implementation.
 *
 * \param h_src             [IN] Source image pointer (Host or pinned memory)
 * \param srcStep           [IN] Source image line step
 * \param h_dst             [OUT] Destination image pointer (Host or pinned memory)
 * \param dstStep           [IN] Destination image line step
 * \param srcRoi            [IN] Region of interest in the source image
 * \param scale             [IN] Downsampling scale factor (positive integer)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_32u_C1R_host(Ncv32u *h_src, Ncv32u srcStep,
                                      Ncv32u *h_dst, Ncv32u dstStep,
                                      NcvSize32u srcRoi, Ncv32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit signed pixels, single channel. Host implementation.
 * \see nppiStDecimate_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_32s_C1R_host(Ncv32s *h_src, Ncv32u srcStep,
                                      Ncv32s *h_dst, Ncv32u dstStep,
                                      NcvSize32u srcRoi, Ncv32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit float pixels, single channel. Host implementation.
 * \see nppiStDecimate_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_32f_C1R_host(Ncv32f *h_src, Ncv32u srcStep,
                                      Ncv32f *h_dst, Ncv32u dstStep,
                                      NcvSize32u srcRoi, Ncv32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit unsigned pixels, single channel. Host implementation.
 * \see nppiStDecimate_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_64u_C1R_host(Ncv64u *h_src, Ncv32u srcStep,
                                      Ncv64u *h_dst, Ncv32u dstStep,
                                      NcvSize32u srcRoi, Ncv32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit signed pixels, single channel. Host implementation.
 * \see nppiStDecimate_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_64s_C1R_host(Ncv64s *h_src, Ncv32u srcStep,
                                      Ncv64s *h_dst, Ncv32u dstStep,
                                      NcvSize32u srcRoi, Ncv32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit float pixels, single channel. Host implementation.
 * \see nppiStDecimate_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStDecimate_64f_C1R_host(Ncv64f *h_src, Ncv32u srcStep,
                                      Ncv64f *h_dst, Ncv32u dstStep,
                                      NcvSize32u srcRoi, Ncv32u scale);


/**
 * Computes standard deviation for each rectangular region of the input image using integral images.
 *
 * \param d_sum             [IN] Integral image pointer (CUDA device memory)
 * \param sumStep           [IN] Integral image line step
 * \param d_sqsum           [IN] Squared integral image pointer (CUDA device memory)
 * \param sqsumStep         [IN] Squared integral image line step
 * \param d_norm            [OUT] Stddev image pointer (CUDA device memory). Each pixel contains stddev of a rect with top-left corner at the original location in the image
 * \param normStep          [IN] Stddev image line step
 * \param roi               [IN] Region of interest in the source image
 * \param rect              [IN] Rectangular region to calculate stddev over
 * \param scaleArea         [IN] Multiplication factor to account decimated scale
 * \param readThruTexture   [IN] Performance hint to cache source in texture (true) or read directly (false)
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStRectStdDev_32f_C1R(Ncv32u *d_sum, Ncv32u sumStep,
                                   Ncv64u *d_sqsum, Ncv32u sqsumStep,
                                   Ncv32f *d_norm, Ncv32u normStep,
                                   NcvSize32u roi, NcvRect32u rect,
                                   Ncv32f scaleArea, NcvBool readThruTexture);


/**
 * Computes standard deviation for each rectangular region of the input image using integral images. Host implementation
 *
 * \param h_sum             [IN] Integral image pointer (Host or pinned memory)
 * \param sumStep           [IN] Integral image line step
 * \param h_sqsum           [IN] Squared integral image pointer (Host or pinned memory)
 * \param sqsumStep         [IN] Squared integral image line step
 * \param h_norm            [OUT] Stddev image pointer (Host or pinned memory). Each pixel contains stddev of a rect with top-left corner at the original location in the image
 * \param normStep          [IN] Stddev image line step
 * \param roi               [IN] Region of interest in the source image
 * \param rect              [IN] Rectangular region to calculate stddev over
 * \param scaleArea         [IN] Multiplication factor to account decimated scale
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStRectStdDev_32f_C1R_host(Ncv32u *h_sum, Ncv32u sumStep,
                                        Ncv64u *h_sqsum, Ncv32u sqsumStep,
                                        Ncv32f *h_norm, Ncv32u normStep,
                                        NcvSize32u roi, NcvRect32u rect,
                                        Ncv32f scaleArea);


/**
 * Transposes an image. 32-bit unsigned pixels, single channel
 *
 * \param d_src             [IN] Source image pointer (CUDA device memory)
 * \param srcStride         [IN] Source image line step
 * \param d_dst             [OUT] Destination image pointer (CUDA device memory)
 * \param dstStride         [IN] Destination image line step
 * \param srcRoi            [IN] Region of interest of the source image
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_32u_C1R(Ncv32u *d_src, Ncv32u srcStride,
                                  Ncv32u *d_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 32-bit signed pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_32s_C1R(Ncv32s *d_src, Ncv32u srcStride,
                                  Ncv32s *d_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 32-bit float pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_32f_C1R(Ncv32f *d_src, Ncv32u srcStride,
                                  Ncv32f *d_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 64-bit unsigned pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_64u_C1R(Ncv64u *d_src, Ncv32u srcStride,
                                  Ncv64u *d_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 64-bit signed pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_64s_C1R(Ncv64s *d_src, Ncv32u srcStride,
                                  Ncv64s *d_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 64-bit float pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_64f_C1R(Ncv64f *d_src, Ncv32u srcStride,
                                  Ncv64f *d_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 128-bit pixels of any type, single channel
 * \see nppiStTranspose_32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_128_C1R(void *d_src, Ncv32u srcStep,
                                  void *d_dst, Ncv32u dstStep, NcvSize32u srcRoi);


/**
 * Transposes an image. 32-bit unsigned pixels, single channel. Host implementation
 *
 * \param h_src             [IN] Source image pointer (Host or pinned memory)
 * \param srcStride         [IN] Source image line step
 * \param h_dst             [OUT] Destination image pointer (Host or pinned memory)
 * \param dstStride         [IN] Destination image line step
 * \param srcRoi            [IN] Region of interest of the source image
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_32u_C1R_host(Ncv32u *h_src, Ncv32u srcStride,
                                       Ncv32u *h_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 32-bit signed pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_32s_C1R_host(Ncv32s *h_src, Ncv32u srcStride,
                                       Ncv32s *h_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 32-bit float pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_32f_C1R_host(Ncv32f *h_src, Ncv32u srcStride,
                                       Ncv32f *h_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 64-bit unsigned pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_64u_C1R_host(Ncv64u *h_src, Ncv32u srcStride,
                                       Ncv64u *h_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 64-bit signed pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_64s_C1R_host(Ncv64s *h_src, Ncv32u srcStride,
                                       Ncv64s *h_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 64-bit float pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_64f_C1R_host(Ncv64f *h_src, Ncv32u srcStride,
                                       Ncv64f *h_dst, Ncv32u dstStride, NcvSize32u srcRoi);


/**
 * Transposes an image. 128-bit pixels of any type, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStTranspose_128_C1R_host(void *d_src, Ncv32u srcStep,
                                       void *d_dst, Ncv32u dstStep, NcvSize32u srcRoi);


/**
 * Calculates the size of the temporary buffer for integral image creation
 *
 * \param roiSize           [IN] Size of the input image
 * \param pBufsize          [OUT] Pointer to host variable that returns the size of the temporary buffer (in bytes)
 * \param devProp           [IN] CUDA device properties structure, containing texture alignment information
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStIntegralGetSize_8u32u(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp);


/**
 * Calculates the size of the temporary buffer for integral image creation
 * \see nppiStIntegralGetSize_8u32u
 */
NCV_EXPORTS
NCVStatus nppiStIntegralGetSize_32f32f(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp);


/**
 * Creates an integral image representation for the input image
 *
 * \param d_src             [IN] Source image pointer (CUDA device memory)
 * \param srcStep           [IN] Source image line step
 * \param d_dst             [OUT] Destination integral image pointer (CUDA device memory)
 * \param dstStep           [IN] Destination image line step
 * \param roiSize           [IN] Region of interest of the source image
 * \param pBuffer           [IN] Pointer to the pre-allocated temporary buffer (CUDA device memory)
 * \param bufSize           [IN] Size of the pBuffer in bytes
 * \param devProp           [IN] CUDA device properties structure, containing texture alignment information
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStIntegral_8u32u_C1R(Ncv8u *d_src, Ncv32u srcStep,
                                   Ncv32u *d_dst, Ncv32u dstStep, NcvSize32u roiSize,
                                   Ncv8u *pBuffer, Ncv32u bufSize, cudaDeviceProp &devProp);


/**
 * Creates an integral image representation for the input image
 * \see nppiStIntegral_8u32u_C1R
 */
NCV_EXPORTS
NCVStatus nppiStIntegral_32f32f_C1R(Ncv32f *d_src, Ncv32u srcStep,
                                    Ncv32f *d_dst, Ncv32u dstStep, NcvSize32u roiSize,
                                    Ncv8u *pBuffer, Ncv32u bufSize, cudaDeviceProp &devProp);


/**
 * Creates an integral image representation for the input image. Host implementation
 *
 * \param h_src             [IN] Source image pointer (Host or pinned memory)
 * \param srcStep           [IN] Source image line step
 * \param h_dst             [OUT] Destination integral image pointer (Host or pinned memory)
 * \param dstStep           [IN] Destination image line step
 * \param roiSize           [IN] Region of interest of the source image
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStIntegral_8u32u_C1R_host(Ncv8u *h_src, Ncv32u srcStep,
                                        Ncv32u *h_dst, Ncv32u dstStep, NcvSize32u roiSize);


/**
 * Creates an integral image representation for the input image. Host implementation
 * \see nppiStIntegral_8u32u_C1R_host
 */
NCV_EXPORTS
NCVStatus nppiStIntegral_32f32f_C1R_host(Ncv32f *h_src, Ncv32u srcStep,
                                         Ncv32f *h_dst, Ncv32u dstStep, NcvSize32u roiSize);


/**
 * Calculates the size of the temporary buffer for squared integral image creation
 *
 * \param roiSize           [IN] Size of the input image
 * \param pBufsize          [OUT] Pointer to host variable that returns the size of the temporary buffer (in bytes)
 * \param devProp           [IN] CUDA device properties structure, containing texture alignment information
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStSqrIntegralGetSize_8u64u(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp);


/**
 * Creates a squared integral image representation for the input image
 *
 * \param d_src             [IN] Source image pointer (CUDA device memory)
 * \param srcStep           [IN] Source image line step
 * \param d_dst             [OUT] Destination squared integral image pointer (CUDA device memory)
 * \param dstStep           [IN] Destination image line step
 * \param roiSize           [IN] Region of interest of the source image
 * \param pBuffer           [IN] Pointer to the pre-allocated temporary buffer (CUDA device memory)
 * \param bufSize           [IN] Size of the pBuffer in bytes
 * \param devProp           [IN] CUDA device properties structure, containing texture alignment information
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStSqrIntegral_8u64u_C1R(Ncv8u *d_src, Ncv32u srcStep,
                                      Ncv64u *d_dst, Ncv32u dstStep, NcvSize32u roiSize,
                                      Ncv8u *pBuffer, Ncv32u bufSize, cudaDeviceProp &devProp);


/**
 * Creates a squared integral image representation for the input image. Host implementation
 *
 * \param h_src             [IN] Source image pointer (Host or pinned memory)
 * \param srcStep           [IN] Source image line step
 * \param h_dst             [OUT] Destination squared integral image pointer (Host or pinned memory)
 * \param dstStep           [IN] Destination image line step
 * \param roiSize           [IN] Region of interest of the source image
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppiStSqrIntegral_8u64u_C1R_host(Ncv8u *h_src, Ncv32u srcStep,
                                           Ncv64u *h_dst, Ncv32u dstStep, NcvSize32u roiSize);


/*@}*/


/** \defgroup npps NPPST Signal Processing
* @{
*/


/**
 * Calculates the size of the temporary buffer for vector compaction. 32-bit unsigned values
 *
 * \param srcLen            [IN] Length of the input vector in elements
 * \param pBufsize          [OUT] Pointer to host variable that returns the size of the temporary buffer (in bytes)
 * \param devProp           [IN] CUDA device properties structure, containing texture alignment information
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppsStCompactGetSize_32u(Ncv32u srcLen, Ncv32u *pBufsize, cudaDeviceProp &devProp);


/**
 * Calculates the size of the temporary buffer for vector compaction. 32-bit signed values
 * \see nppsStCompactGetSize_32u
 */
NCVStatus nppsStCompactGetSize_32s(Ncv32u srcLen, Ncv32u *pBufsize, cudaDeviceProp &devProp);


/**
 * Calculates the size of the temporary buffer for vector compaction. 32-bit float values
 * \see nppsStCompactGetSize_32u
 */
NCVStatus nppsStCompactGetSize_32f(Ncv32u srcLen, Ncv32u *pBufsize, cudaDeviceProp &devProp);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit unsigned values
 *
 * \param d_src             [IN] Source vector pointer (CUDA device memory)
 * \param srcLen            [IN] Source vector length
 * \param d_dst             [OUT] Destination vector pointer (CUDA device memory)
 * \param p_dstLen          [OUT] Pointer to the destination vector length (Pinned memory or NULL)
 * \param elemRemove        [IN] The value to be removed
 * \param pBuffer           [IN] Pointer to the pre-allocated temporary buffer (CUDA device memory)
 * \param bufSize           [IN] Size of the pBuffer in bytes
 * \param devProp           [IN] CUDA device properties structure, containing texture alignment information
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppsStCompact_32u(Ncv32u *d_src, Ncv32u srcLen,
                            Ncv32u *d_dst, Ncv32u *p_dstLen,
                            Ncv32u elemRemove, Ncv8u *pBuffer,
                            Ncv32u bufSize, cudaDeviceProp &devProp);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit signed values
 * \see nppsStCompact_32u
 */
NCV_EXPORTS
NCVStatus nppsStCompact_32s(Ncv32s *d_src, Ncv32u srcLen,
                            Ncv32s *d_dst, Ncv32u *p_dstLen,
                            Ncv32s elemRemove, Ncv8u *pBuffer,
                            Ncv32u bufSize, cudaDeviceProp &devProp);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit float values
 * \see nppsStCompact_32u
 */
NCV_EXPORTS
NCVStatus nppsStCompact_32f(Ncv32f *d_src, Ncv32u srcLen,
                            Ncv32f *d_dst, Ncv32u *p_dstLen,
                            Ncv32f elemRemove, Ncv8u *pBuffer,
                            Ncv32u bufSize, cudaDeviceProp &devProp);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit unsigned values. Host implementation
 *
 * \param h_src             [IN] Source vector pointer (CUDA device memory)
 * \param srcLen            [IN] Source vector length
 * \param h_dst             [OUT] Destination vector pointer (CUDA device memory)
 * \param dstLen            [OUT] Pointer to the destination vector length (can be NULL)
 * \param elemRemove        [IN] The value to be removed
 *
 * \return NCV status code
 */
NCV_EXPORTS
NCVStatus nppsStCompact_32u_host(Ncv32u *h_src, Ncv32u srcLen,
                                 Ncv32u *h_dst, Ncv32u *dstLen, Ncv32u elemRemove);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit signed values. Host implementation
 * \see nppsStCompact_32u_host
 */
NCV_EXPORTS
NCVStatus nppsStCompact_32s_host(Ncv32s *h_src, Ncv32u srcLen,
                                 Ncv32s *h_dst, Ncv32u *dstLen, Ncv32s elemRemove);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit float values. Host implementation
 * \see nppsStCompact_32u_host
 */
NCV_EXPORTS
NCVStatus nppsStCompact_32f_host(Ncv32f *h_src, Ncv32u srcLen,
                                 Ncv32f *h_dst, Ncv32u *dstLen, Ncv32f elemRemove);


/*@}*/


#endif // _npp_staging_hpp_
