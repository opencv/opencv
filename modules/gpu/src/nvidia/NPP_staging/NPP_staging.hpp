/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. 
// 
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2009-2010, NVIDIA Corporation, all rights reserved.
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
