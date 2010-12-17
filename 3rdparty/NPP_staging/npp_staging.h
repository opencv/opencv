/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/
#ifndef _npp_staging_h_
#define _npp_staging_h_


/**
* \file npp_staging.h
* NPP Staging Library (will become part of NPP next release)
*/


#ifdef __cplusplus


/** \defgroup ctassert Compile-time assert functionality
* @{
*/


    /**
     * Compile-time assert namespace
     */
    namespace NppStCTprep
    {
        template <bool x>
        struct CT_ASSERT_FAILURE;

        template <>
        struct CT_ASSERT_FAILURE<true> {};

        template <int x>
        struct assertTest{};
    }


    #define NPPST_CT_PREP_PASTE_AUX(a,b)      a##b                           ///< Concatenation indirection macro
    #define NPPST_CT_PREP_PASTE(a,b)          NPPST_CT_PREP_PASTE_AUX(a, b)  ///< Concatenation macro


    /**
     * Performs compile-time assertion of a condition on the file scope
     */
    #define NPPST_CT_ASSERT(X) \
        typedef NppStCTprep::assertTest<sizeof(NppStCTprep::CT_ASSERT_FAILURE< (bool)(X) >)> \
        NPPST_CT_PREP_PASTE(__ct_assert_typedef_, __LINE__)


/*@}*/


#endif


/** \defgroup typedefs NPP Integral and compound types of guaranteed size
 * @{
 */


typedef               bool NppStBool; ///< Bool of size less than integer
typedef          long long NppSt64s;  ///< 64-bit signed integer
typedef unsigned long long NppSt64u;  ///< 64-bit unsigned integer
typedef                int NppSt32s;  ///< 32-bit signed integer
typedef       unsigned int NppSt32u;  ///< 32-bit unsigned integer
typedef              short NppSt16s;  ///< 16-bit signed short
typedef     unsigned short NppSt16u;  ///< 16-bit unsigned short
typedef               char NppSt8s;   ///< 8-bit signed char
typedef      unsigned char NppSt8u;   ///< 8-bit unsigned char
typedef              float NppSt32f;  ///< 32-bit IEEE-754 (single precision) float
typedef             double NppSt64f;  ///< 64-bit IEEE-754 (double precision) float


/**
 * 2D Rectangle, 8-bit unsigned fields
 * This struct contains position and size information of a rectangle in two space
 */
struct NppStRect8u
{
    NppSt8u x;          ///< x-coordinate of upper left corner
    NppSt8u y;          ///< y-coordinate of upper left corner
    NppSt8u width;      ///< Rectangle width
    NppSt8u height;     ///< Rectangle height
#ifdef __cplusplus
    NppStRect8u() : x(0), y(0), width(0), height(0) {};
    NppStRect8u(NppSt8u x, NppSt8u y, NppSt8u width, NppSt8u height) : x(x), y(y), width(width), height(height) {}
#endif
};


/**
 * 2D Rectangle, 32-bit signed fields
 * This struct contains position and size information of a rectangle in two space
 */
struct NppStRect32s
{
    NppSt32s x;          ///< x-coordinate of upper left corner
    NppSt32s y;          ///< y-coordinate of upper left corner
    NppSt32s width;      ///< Rectangle width
    NppSt32s height;     ///< Rectangle height
#ifdef __cplusplus
    NppStRect32s() : x(0), y(0), width(0), height(0) {};
    NppStRect32s(NppSt32s x, NppSt32s y, NppSt32s width, NppSt32s height) : x(x), y(y), width(width), height(height) {}
#endif
};


/**
 * 2D Rectangle, 32-bit unsigned fields
 * This struct contains position and size information of a rectangle in two space
 */
struct NppStRect32u
{
    NppSt32u x;          ///< x-coordinate of upper left corner
    NppSt32u y;          ///< y-coordinate of upper left corner
    NppSt32u width;      ///< Rectangle width
    NppSt32u height;     ///< Rectangle height
#ifdef __cplusplus
    NppStRect32u() : x(0), y(0), width(0), height(0) {};
    NppStRect32u(NppSt32u x, NppSt32u y, NppSt32u width, NppSt32u height) : x(x), y(y), width(width), height(height) {}
#endif
};


/**
 * 2D Size, 32-bit signed fields
 * This struct typically represents the size of a a rectangular region in two space
 */
struct NppStSize32s
{
    NppSt32s width;  ///< Rectangle width
    NppSt32s height; ///< Rectangle height
#ifdef __cplusplus
    NppStSize32s() : width(0), height(0) {};
    NppStSize32s(NppSt32s width, NppSt32s height) : width(width), height(height) {}
#endif
};


/**
 * 2D Size, 32-bit unsigned fields
 * This struct typically represents the size of a a rectangular region in two space
 */
struct NppStSize32u
{
    NppSt32u width;  ///< Rectangle width
    NppSt32u height; ///< Rectangle height
#ifdef __cplusplus
    NppStSize32u() : width(0), height(0) {};
    NppStSize32u(NppSt32u width, NppSt32u height) : width(width), height(height) {}
#endif
};


/**
 * Error Status Codes
 *
 * Almost all NPP function return error-status information using
 * these return codes.
 * Negative return codes indicate errors, positive return codes indicate
 * warnings, a return code of 0 indicates success.
 */
enum NppStStatus
{
    //already present in NPP
 /*   NPP_SUCCESS                      = 0,   ///< Successful operation (same as NPP_NO_ERROR)
    NPP_ERROR                        = -1,  ///< Unknown error
    NPP_CUDA_KERNEL_EXECUTION_ERROR  = -3,  ///< CUDA kernel execution error
    NPP_NULL_POINTER_ERROR           = -4,  ///< NULL pointer argument error
    NPP_TEXTURE_BIND_ERROR           = -24, ///< CUDA texture binding error or non-zero offset returned
    NPP_MEMCPY_ERROR                 = -13, ///< CUDA memory copy error
    NPP_MEM_ALLOC_ERR                = -12, ///< CUDA memory allocation error
    NPP_MEMFREE_ERR                  = -15, ///< CUDA memory deallocation error*/

    //to be added
    NPP_INVALID_ROI,                        ///< Invalid region of interest argument
    NPP_INVALID_STEP,                       ///< Invalid image lines step argument (check sign, alignment, relation to image width)
    NPP_INVALID_SCALE,                      ///< Invalid scale parameter passed
    NPP_MEM_INSUFFICIENT_BUFFER,            ///< Insufficient user-allocated buffer
    NPP_MEM_RESIDENCE_ERROR,                ///< Memory residence error detected (check if pointers should be device or pinned)
    NPP_MEM_INTERNAL_ERROR,                 ///< Internal memory management error
};


/*@}*/


#ifdef __cplusplus


/** \defgroup ct_typesize_checks Client-side sizeof types compile-time check
* @{
*/
    NPPST_CT_ASSERT(sizeof(NppStBool) <= 4);
    NPPST_CT_ASSERT(sizeof(NppSt64s) == 8);
    NPPST_CT_ASSERT(sizeof(NppSt64u) == 8);
    NPPST_CT_ASSERT(sizeof(NppSt32s) == 4);
    NPPST_CT_ASSERT(sizeof(NppSt32u) == 4);
    NPPST_CT_ASSERT(sizeof(NppSt16s) == 2);
    NPPST_CT_ASSERT(sizeof(NppSt16u) == 2);
    NPPST_CT_ASSERT(sizeof(NppSt8s) == 1);
    NPPST_CT_ASSERT(sizeof(NppSt8u) == 1);
    NPPST_CT_ASSERT(sizeof(NppSt32f) == 4);
    NPPST_CT_ASSERT(sizeof(NppSt64f) == 8);
    NPPST_CT_ASSERT(sizeof(NppStRect8u) == sizeof(NppSt32u));
    NPPST_CT_ASSERT(sizeof(NppStRect32s) == 4 * sizeof(NppSt32s));
    NPPST_CT_ASSERT(sizeof(NppStRect32u) == 4 * sizeof(NppSt32u));
    NPPST_CT_ASSERT(sizeof(NppStSize32u) == 2 * sizeof(NppSt32u));
/*@}*/


#endif


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup core_npp NPP Core
 * Basic functions for CUDA streams management.
 * WARNING: These functions couldn't be exported from NPP_staging library, so they can't be used
 * @{
 */


/**
 * Gets an active CUDA stream used by NPP (Not an API yet!)
 * \return Current CUDA stream
 */
cudaStream_t nppStGetActiveCUDAstream();


/**
 * Sets an active CUDA stream used by NPP (Not an API yet!)
 * \param cudaStream        [IN] cudaStream CUDA stream to become current
 * \return CUDA stream used before
 */
cudaStream_t nppStSetActiveCUDAstream(cudaStream_t cudaStream);


/*@}*/


/** \defgroup nppi NPP Image Processing
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
 * \return NPP status code
 */
NppStStatus nppiStDownsampleNearest_32u_C1R(NppSt32u *d_src, NppSt32u srcStep,
                                            NppSt32u *d_dst, NppSt32u dstStep,
                                            NppStSize32u srcRoi, NppSt32u scale,
                                            NppStBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit signed pixels, single channel.
 * \see nppiStDownsampleNearest_32u_C1R
 */
NppStStatus nppiStDownsampleNearest_32s_C1R(NppSt32s *d_src, NppSt32u srcStep,
                                            NppSt32s *d_dst, NppSt32u dstStep,
                                            NppStSize32u srcRoi, NppSt32u scale,
                                            NppStBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit float pixels, single channel.
 * \see nppiStDownsampleNearest_32u_C1R
 */
NppStStatus nppiStDownsampleNearest_32f_C1R(NppSt32f *d_src, NppSt32u srcStep,
                                            NppSt32f *d_dst, NppSt32u dstStep,
                                            NppStSize32u srcRoi, NppSt32u scale,
                                            NppStBool readThruTexture);


/**
* Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit unsigned pixels, single channel.
* \see nppiStDownsampleNearest_32u_C1R
*/
NppStStatus nppiStDownsampleNearest_64u_C1R(NppSt64u *d_src, NppSt32u srcStep,
                                            NppSt64u *d_dst, NppSt32u dstStep,
                                            NppStSize32u srcRoi, NppSt32u scale,
                                            NppStBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit signed pixels, single channel.
 * \see nppiStDownsampleNearest_32u_C1R
 */
NppStStatus nppiStDownsampleNearest_64s_C1R(NppSt64s *d_src, NppSt32u srcStep,
                                            NppSt64s *d_dst, NppSt32u dstStep,
                                            NppStSize32u srcRoi, NppSt32u scale,
                                            NppStBool readThruTexture);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit float pixels, single channel.
 * \see nppiStDownsampleNearest_32u_C1R
 */
NppStStatus nppiStDownsampleNearest_64f_C1R(NppSt64f *d_src, NppSt32u srcStep,
                                            NppSt64f *d_dst, NppSt32u dstStep,
                                            NppStSize32u srcRoi, NppSt32u scale,
                                            NppStBool readThruTexture);


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
 * \return NPP status code
 */
NppStStatus nppiStDownsampleNearest_32u_C1R_host(NppSt32u *h_src, NppSt32u srcStep,
                                                 NppSt32u *h_dst, NppSt32u dstStep,
                                                 NppStSize32u srcRoi, NppSt32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit signed pixels, single channel. Host implementation.
 * \see nppiStDownsampleNearest_32u_C1R_host
 */
NppStStatus nppiStDownsampleNearest_32s_C1R_host(NppSt32s *h_src, NppSt32u srcStep,
                                                 NppSt32s *h_dst, NppSt32u dstStep,
                                                 NppStSize32u srcRoi, NppSt32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 32-bit float pixels, single channel. Host implementation.
 * \see nppiStDownsampleNearest_32u_C1R_host
 */
NppStStatus nppiStDownsampleNearest_32f_C1R_host(NppSt32f *h_src, NppSt32u srcStep,
                                                 NppSt32f *h_dst, NppSt32u dstStep,
                                                 NppStSize32u srcRoi, NppSt32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit unsigned pixels, single channel. Host implementation.
 * \see nppiStDownsampleNearest_32u_C1R_host
 */
NppStStatus nppiStDownsampleNearest_64u_C1R_host(NppSt64u *h_src, NppSt32u srcStep,
                                                 NppSt64u *h_dst, NppSt32u dstStep,
                                                 NppStSize32u srcRoi, NppSt32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit signed pixels, single channel. Host implementation.
 * \see nppiStDownsampleNearest_32u_C1R_host
 */
NppStStatus nppiStDownsampleNearest_64s_C1R_host(NppSt64s *h_src, NppSt32u srcStep,
                                                 NppSt64s *h_dst, NppSt32u dstStep,
                                                 NppStSize32u srcRoi, NppSt32u scale);


/**
 * Downsamples (decimates) an image using the nearest neighbor algorithm. 64-bit float pixels, single channel. Host implementation.
 * \see nppiStDownsampleNearest_32u_C1R_host
 */
NppStStatus nppiStDownsampleNearest_64f_C1R_host(NppSt64f *h_src, NppSt32u srcStep,
                                                 NppSt64f *h_dst, NppSt32u dstStep,
                                                 NppStSize32u srcRoi, NppSt32u scale);


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
 * \return NPP status code
 */
NppStStatus nppiStRectStdDev_32f_C1R(NppSt32u *d_sum, NppSt32u sumStep,
                                     NppSt64u *d_sqsum, NppSt32u sqsumStep,
                                     NppSt32f *d_norm, NppSt32u normStep,
                                     NppStSize32u roi, NppStRect32u rect,
                                     NppSt32f scaleArea, NppStBool readThruTexture);


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
 * \return NPP status code
 */
NppStStatus nppiStRectStdDev_32f_C1R_host(NppSt32u *h_sum, NppSt32u sumStep,
                                          NppSt64u *h_sqsum, NppSt32u sqsumStep,
                                          NppSt32f *h_norm, NppSt32u normStep,
                                          NppStSize32u roi, NppStRect32u rect,
                                          NppSt32f scaleArea);


/**
 * Transposes an image. 32-bit unsigned pixels, single channel
 *
 * \param d_src             [IN] Source image pointer (CUDA device memory)
 * \param srcStride         [IN] Source image line step
 * \param d_dst             [OUT] Destination image pointer (CUDA device memory)
 * \param dstStride         [IN] Destination image line step
 * \param srcRoi            [IN] Region of interest of the source image
 *
 * \return NPP status code
 */
NppStStatus nppiStTranspose_32u_C1R(NppSt32u *d_src, NppSt32u srcStride,
                                    NppSt32u *d_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 32-bit signed pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NppStStatus nppiStTranspose_32s_C1R(NppSt32s *d_src, NppSt32u srcStride,
                                    NppSt32s *d_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 32-bit float pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NppStStatus nppiStTranspose_32f_C1R(NppSt32f *d_src, NppSt32u srcStride,
                                    NppSt32f *d_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 64-bit unsigned pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NppStStatus nppiStTranspose_64u_C1R(NppSt64u *d_src, NppSt32u srcStride,
                                    NppSt64u *d_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 64-bit signed pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NppStStatus nppiStTranspose_64s_C1R(NppSt64s *d_src, NppSt32u srcStride,
                                    NppSt64s *d_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 64-bit float pixels, single channel
 * \see nppiStTranspose_32u_C1R
 */
NppStStatus nppiStTranspose_64f_C1R(NppSt64f *d_src, NppSt32u srcStride,
                                    NppSt64f *d_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 32-bit unsigned pixels, single channel. Host implementation
 *
 * \param h_src             [IN] Source image pointer (Host or pinned memory)
 * \param srcStride         [IN] Source image line step
 * \param h_dst             [OUT] Destination image pointer (Host or pinned memory)
 * \param dstStride         [IN] Destination image line step
 * \param srcRoi            [IN] Region of interest of the source image
 *
 * \return NPP status code
 */
NppStStatus nppiStTranspose_32u_C1R_host(NppSt32u *h_src, NppSt32u srcStride,
                                         NppSt32u *h_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 32-bit signed pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NppStStatus nppiStTranspose_32s_C1R_host(NppSt32s *h_src, NppSt32u srcStride,
                                         NppSt32s *h_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 32-bit float pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NppStStatus nppiStTranspose_32f_C1R_host(NppSt32f *h_src, NppSt32u srcStride,
                                         NppSt32f *h_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 64-bit unsigned pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NppStStatus nppiStTranspose_64u_C1R_host(NppSt64u *h_src, NppSt32u srcStride,
                                         NppSt64u *h_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 64-bit signed pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NppStStatus nppiStTranspose_64s_C1R_host(NppSt64s *h_src, NppSt32u srcStride,
                                         NppSt64s *h_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Transposes an image. 64-bit float pixels, single channel. Host implementation
 * \see nppiStTranspose_32u_C1R_host
 */
NppStStatus nppiStTranspose_64f_C1R_host(NppSt64f *h_src, NppSt32u srcStride,
                                         NppSt64f *h_dst, NppSt32u dstStride, NppStSize32u srcRoi);


/**
 * Calculates the size of the temporary buffer for integral image creation
 *
 * \param roiSize           [IN] Size of the input image
 * \param pBufsize          [OUT] Pointer to host variable that returns the size of the temporary buffer (in bytes)
 *
 * \return NPP status code
 */
NppStStatus nppiStIntegralGetSize_8u32u(NppStSize32u roiSize, NppSt32u *pBufsize);


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
 *
 * \return NPP status code
 */
NppStStatus nppiStIntegral_8u32u_C1R(NppSt8u *d_src, NppSt32u srcStep,
                                     NppSt32u *d_dst, NppSt32u dstStep, NppStSize32u roiSize,
                                     NppSt8u *pBuffer, NppSt32u bufSize);


/**
 * Creates an integral image representation for the input image. Host implementation
 *
 * \param h_src             [IN] Source image pointer (Host or pinned memory)
 * \param srcStep           [IN] Source image line step
 * \param h_dst             [OUT] Destination integral image pointer (Host or pinned memory)
 * \param dstStep           [IN] Destination image line step
 * \param roiSize           [IN] Region of interest of the source image
 *
 * \return NPP status code
 */
NppStStatus nppiStIntegral_8u32u_C1R_host(NppSt8u *h_src, NppSt32u srcStep,
                                          NppSt32u *h_dst, NppSt32u dstStep, NppStSize32u roiSize);


/**
 * Calculates the size of the temporary buffer for squared integral image creation
 *
 * \param roiSize           [IN] Size of the input image
 * \param pBufsize          [OUT] Pointer to host variable that returns the size of the temporary buffer (in bytes)
 *
 * \return NPP status code
 */
NppStStatus nppiStSqrIntegralGetSize_8u64u(NppStSize32u roiSize, NppSt32u *pBufsize);


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
 *
 * \return NPP status code
 */
NppStStatus nppiStSqrIntegral_8u64u_C1R(NppSt8u *d_src, NppSt32u srcStep,
                                        NppSt64u *d_dst, NppSt32u dstStep, NppStSize32u roiSize,
                                        NppSt8u *pBuffer, NppSt32u bufSize);


/**
 * Creates a squared integral image representation for the input image. Host implementation
 *
 * \param h_src             [IN] Source image pointer (Host or pinned memory)
 * \param srcStep           [IN] Source image line step
 * \param h_dst             [OUT] Destination squared integral image pointer (Host or pinned memory)
 * \param dstStep           [IN] Destination image line step
 * \param roiSize           [IN] Region of interest of the source image
 *
 * \return NPP status code
 */
NppStStatus nppiStSqrIntegral_8u64u_C1R_host(NppSt8u *h_src, NppSt32u srcStep,
                                             NppSt64u *h_dst, NppSt32u dstStep, NppStSize32u roiSize);


/*@}*/


/** \defgroup npps NPP Signal Processing
* @{
*/


/**
 * Calculates the size of the temporary buffer for vector compaction. 32-bit unsigned values
 *
 * \param srcLen            [IN] Length of the input vector in elements
 * \param pBufsize          [OUT] Pointer to host variable that returns the size of the temporary buffer (in bytes)
 *
 * \return NPP status code
 */
NppStStatus nppsStCompactGetSize_32u(NppSt32u srcLen, NppSt32u *pBufsize);


/**
 * Calculates the size of the temporary buffer for vector compaction. 32-bit signed values
 * \see nppsStCompactGetSize_32u
 */
NppStStatus nppsStCompactGetSize_32s(NppSt32u srcLen, NppSt32u *pBufsize);


/**
 * Calculates the size of the temporary buffer for vector compaction. 32-bit float values
 * \see nppsStCompactGetSize_32u
 */
NppStStatus nppsStCompactGetSize_32f(NppSt32u srcLen, NppSt32u *pBufsize);


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
 *
 * \return NPP status code
 */
NppStStatus nppsStCompact_32u(NppSt32u *d_src, NppSt32u srcLen,
                              NppSt32u *d_dst, NppSt32u *p_dstLen,
                              NppSt32u elemRemove,
                              NppSt8u *pBuffer, NppSt32u bufSize);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit signed values
 * \see nppsStCompact_32u
 */
NppStStatus nppsStCompact_32s(NppSt32s *d_src, NppSt32u srcLen,
                              NppSt32s *d_dst, NppSt32u *p_dstLen,
                              NppSt32s elemRemove,
                              NppSt8u *pBuffer, NppSt32u bufSize);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit float values
 * \see nppsStCompact_32u
 */
NppStStatus nppsStCompact_32f(NppSt32f *d_src, NppSt32u srcLen,
                              NppSt32f *d_dst, NppSt32u *p_dstLen,
                              NppSt32f elemRemove,
                              NppSt8u *pBuffer, NppSt32u bufSize);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit unsigned values. Host implementation
 *
 * \param h_src             [IN] Source vector pointer (CUDA device memory)
 * \param srcLen            [IN] Source vector length
 * \param h_dst             [OUT] Destination vector pointer (CUDA device memory)
 * \param dstLen            [OUT] Pointer to the destination vector length (can be NULL)
 * \param elemRemove        [IN] The value to be removed
 *
 * \return NPP status code
 */
NppStStatus nppsStCompact_32u_host(NppSt32u *h_src, NppSt32u srcLen,
                                   NppSt32u *h_dst, NppSt32u *dstLen, NppSt32u elemRemove);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit signed values. Host implementation
 * \see nppsStCompact_32u_host
 */
NppStStatus nppsStCompact_32s_host(NppSt32s *h_src, NppSt32u srcLen,
                                   NppSt32s *h_dst, NppSt32u *dstLen, NppSt32s elemRemove);


/**
 * Compacts the input vector by removing elements of specified value. 32-bit float values. Host implementation
 * \see nppsStCompact_32u_host
 */
NppStStatus nppsStCompact_32f_host(NppSt32f *h_src, NppSt32u srcLen,
                                   NppSt32f *h_dst, NppSt32u *dstLen, NppSt32f elemRemove);


/*@}*/


#ifdef __cplusplus
}
#endif


#endif // _npp_staging_h_
