/*
// Copyright 2014 2015 Intel Corporation All Rights Reserved.
//
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title
// to such Material remains with Intel Corporation or its suppliers or
// licensors. The Material contains proprietary information of Intel
// or its suppliers and licensors. The Material is protected by worldwide
// copyright laws and treaty provisions. No part of the Material may be used,
// copied, reproduced, modified, published, uploaded, posted, transmitted,
// distributed or disclosed in any way without Intel's prior express written
// permission. No license under any patent, copyright or other intellectual
// property rights in the Material is granted to or conferred upon you,
// either expressly, by implication, inducement, estoppel or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.
//
// Unless otherwise agreed by Intel in writing,
// you may not remove or alter this notice or any other notice embedded in
// Materials by Intel or Intel's suppliers or licensors in any way.
//
*/

#if !defined( __IPPICV_H__ )
#define __IPPICV_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ippicv_defs.h"
#include "ippicv_types.h"
#include "ippicv_redefs.h"
#include "ippversion.h"


/* =============================================================================
							ippCore
============================================================================= */


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippGetLibVersion
//  Purpose:    getting of the library version
//  Returns:    the structure of information about version
//              of ippcore library
//  Parameters:
//
//  Notes:      not necessary to release the returned structure
*/
IPPAPI( const IppLibraryVersion*, ippGetLibVersion, (void) )

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippInit
//  Purpose:    Automatic switching to best for current cpu library code using.
//  Returns:
//   ippStsNoErr
//
//  Parameter:  nothing
//
//  Notes:      At the moment of this function execution no any other IPP function
//              has to be working
*/
IPPAPI( IppStatus, ippInit, ( void ))


/* /////////////////////////////////////////////////////////////////////////////
//                   Functions to allocate and free memory
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippMalloc
//  Purpose:    64-byte aligned memory allocation
//  Parameter:
//    len       number of bytes
//  Returns:    pointer to allocated memory
//
//  Notes:      the memory allocated by ippMalloc has to be free by ippFree
//              function only.
*/

IPPAPI( void*, ippMalloc,  (int length) )


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippFree
//  Purpose:    free memory allocated by the ippMalloc function
//  Parameter:
//    ptr       pointer to the memory allocated by the ippMalloc function
//
//  Notes:      use the function to free memory allocated by ippMalloc
*/
IPPAPI( void, ippFree, (void* ptr) )


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippSetCpuFeatures
//
//  Purpose: Changes the set of enabled/disabled CPU features.
//           This function sets the processor-specific code of the Intel IPP
//           library according to the processor features specified in cpuFeatures.
//
//  Return:
//    ippStsNoErr                 No errors.
//    Warnings:
//    ippStsFeatureNotSupported   Current CPU doesn't support at least 1 of the
//                                desired features;
//    ippStsUnknownFeature        At least one of the desired features is unknown;
//    ippStsFeaturesCombination   Wrong combination of features;
//    ippStsCpuMismatch           Indicates that the specified processor features
//                                are not valid. Previously set code is used.
//
//  Arguments:
//    cpuFeatures                 Desired features to support by the library
//                                (see ippdefs.h for ippCPUID_XX definition)
//
//  NOTE:       this function can re-initializes dispatcher and after the
//              call another library (letter) may work
//  CAUTION:    At the moment of this function excecution no any other IPP
//              function has to be working
//
//  The next pre-defined sets of features can be used:
//  32-bit code:
// #define PX_FM ( ippCPUID_MMX | ippCPUID_SSE )
// #define W7_FM ( PX_FM | ippCPUID_SSE2 )
// #define V8_FM ( W7_FM | ippCPUID_SSE3 | ippCPUID_SSSE3 )
// #define S8_FM ( V8_FM | ippCPUID_MOVBE )
// #define P8_FM ( V8_FM | ippCPUID_SSE41 | ippCPUID_SSE42 | ippCPUID_AES | ippCPUID_CLMUL | ippCPUID_SHA )
// #define G9_FM ( P8_FM | ippCPUID_AVX | ippAVX_ENABLEDBYOS | ippCPUID_RDRAND | ippCPUID_F16C )
// #define H9_FM ( G9_FM | ippCPUID_AVX2 | ippCPUID_MOVBE | ippCPUID_ADCOX | ippCPUID_RDSEED | ippCPUID_PREFETCHW )
//
// 64-bit code:
// #define PX_FM ( ippCPUID_MMX | ippCPUID_SSE | ippCPUID_SSE2 )
// #define M7_FM ( PX_FM | ippCPUID_SSE3 )
// #define N8_FM ( S8_FM )
// #define U8_FM ( V8_FM )
// #define Y8_FM ( P8_FM )
// #define E9_FM ( G9_FM )
// #define L9_FM ( H9_FM )
//
*/

IPPAPI( IppStatus, ippSetCpuFeatures,( Ipp64u cpuFeatures ))


/* =============================================================================
							ippVM
============================================================================= */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippvmGetLibVersion
//  Purpose:    getting of the library version
//  Returns:    the structure of information about version
//              of ippVM library
//  Parameters:
//
//  Notes:      not necessary to release the returned structure
*/

IPPAPI( const IppLibraryVersion*, ippvmGetLibVersion, (void) )


IPPAPI( IppStatus, ippsExp_32f_A21, (const Ipp32f a[],Ipp32f r[],Ipp32s n))
IPPAPI( IppStatus, ippsExp_64f_A50, (const Ipp64f a[],Ipp64f r[],Ipp32s n))
IPPAPI( IppStatus, ippsLn_32f_A21, (const Ipp32f a[],Ipp32f r[],Ipp32s n))
IPPAPI( IppStatus, ippsLn_64f_A50, (const Ipp64f a[],Ipp64f r[],Ipp32s n))
IPPAPI( IppStatus, ippsInvSqrt_32f_A21, (const Ipp32f a[],Ipp32f r[],Ipp32s n))
IPPAPI( IppStatus, ippsInvSqrt_64f_A50, (const Ipp64f a[],Ipp64f r[],Ipp32s n))
IPPAPI( IppStatus, ippsSqrt_32f_A21, (const Ipp32f a[],Ipp32f r[],Ipp32s n))
IPPAPI( IppStatus, ippsSqrt_64f_A50, (const Ipp64f a[],Ipp64f r[],Ipp32s n))
IPPAPI( IppStatus, ippsPowx_32f_A21, (const Ipp32f a[],const Ipp32f b,Ipp32f r[],Ipp32s n))
IPPAPI( IppStatus, ippsPowx_64f_A50, (const Ipp64f a[],const Ipp64f b,Ipp64f r[],Ipp32s n))



/* =============================================================================
							ippSP
============================================================================= */


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsGetLibVersion
//  Purpose:    get the library version
//  Parameters:
//  Returns:    pointer to structure describing version of the ipps library
//
//  Notes:      don't free the pointer
*/
IPPAPI( const IppLibraryVersion*, ippsGetLibVersion, (void) )


/* /////////////////////////////////////////////////////////////////////////////
//                   Functions to allocate and free memory
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsMalloc
//  Purpose:    32-byte aligned memory allocation
//  Parameter:
//    len       number of elements (according to their type)
//  Returns:    pointer to allocated memory
//
//  Notes:      the memory allocated by ippsMalloc has to be free by ippsFree
//              function only.
*/

IPPAPI( Ipp8u*,   ippsMalloc_8u,  (int len) )


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsFree
//  Purpose:    free memory allocated by the ippsMalloc functions
//  Parameter:
//    ptr       pointer to the memory allocated by the ippsMalloc functions
//
//  Notes:      use the function to free memory allocated by ippsMalloc_*
*/
IPPAPI( void, ippsFree, (void* ptr) )


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsZero
//  Purpose:    set elements of the vector to zero of corresponding type
//  Parameters:
//    pDst       pointer to the destination vector
//    len        length of the vectors
//  Return:
//    ippStsNullPtrErr        pointer to the vector is NULL
//    ippStsSizeErr           length of the vectors is less or equal zero
//    ippStsNoErr             otherwise
*/
IPPAPI(IppStatus, ippsZero_8u,  ( Ipp8u*   pDst, int len ))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsSet
//  Purpose:    set elements of the destination vector to the value
//  Parameters:
//    val        value to set the elements of the vector
//    pDst       pointer to the destination vector
//    len        length of the vectors
//  Return:
//    ippStsNullPtrErr        pointer to the vector is NULL
//    ippStsSizeErr           length of the vector is less or equal zero
//    ippStsNoErr             otherwise
*/
IPPAPI(IppStatus, ippsSet_8u,  ( Ipp8u   val, Ipp8u*   pDst, int len ))
IPPAPI(IppStatus, ippsSet_16s, ( Ipp16s  val, Ipp16s*  pDst, int len ))
IPPAPI(IppStatus, ippsSet_32s, ( Ipp32s  val, Ipp32s*  pDst, int len ))
IPPAPI(IppStatus, ippsSet_32f, ( Ipp32f  val, Ipp32f*  pDst, int len ))
IPPAPI(IppStatus, ippsSet_64f, ( Ipp64f  val, Ipp64f*  pDst, int len ))


/* /////////////////////////////////////////////////////////////////////////////
//  Names:      ippsMagnitude
//  Purpose:    compute magnitude of every complex element of the source
//  Parameters:
//   pSrcDst            pointer to the source/destination vector
//   pSrc               pointer to the source vector
//   pDst               pointer to the destination vector
//   len                length of the vector(s), number of items
//   scaleFactor        scale factor value
//  Return:
//   ippStsNullPtrErr      pointer(s) to data vector is NULL
//   ippStsSizeErr         length of a vector is less or equal 0
//   ippStsNoErr           otherwise
//  Notes:
//         dst = sqrt( src.re^2 + src.im^2 )
*/

IPPAPI(IppStatus,ippsMagnitude_32f,(const Ipp32f* pSrcRe,const Ipp32f* pSrcIm,
                               Ipp32f* pDst,int len))

IPPAPI(IppStatus,ippsMagnitude_64f,(const Ipp64f* pSrcRe,const Ipp64f* pSrcIm,
                               Ipp64f* pDst,int len))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsFlip
//  Purpose:    dst[i] = src[len-i-1], i=0..len-1
//  Parameters:
//    pSrc      pointer to the input vector
//    pDst      pointer to the output vector
//    len       length of the vectors, number of items
//  Return:
//    ippStsNullPtrErr        pointer(s) to the data is NULL
//    ippStsSizeErr           length of the vectors is less or equal zero
//    ippStsNoErr             otherwise
*/

IPPAPI(IppStatus, ippsFlip_16u_I,( Ipp16u* pSrcDst, int len ))

IPPAPI(IppStatus, ippsFlip_32f_I,( Ipp32f* pSrcDst, int len ))

IPPAPI(IppStatus, ippsFlip_64f_I,( Ipp64f* pSrcDst, int len ))

IPPAPI(IppStatus, ippsFlip_8u_I,( Ipp8u* pSrcDst, int len ))


/* /////////////////////////////////////////////////////////////////////////////
//  Names:      ippsPolarToCart
//
//  Purpose:    Convert polar coordinate to cartesian. Output data are formed as
//              two real vectors.
//
//  Parameters:
//   pDstMagn     an input vector containing the magnitude components
//   pDstPhase    an input vector containing the phase components(in radians)
//   pSrcRe       an output complex vector to store the coordinates X
//   pSrcIm       an output complex vector to store the coordinates Y
//   len          a length of the arrays
//  Return:
//   ippStsNoErr           Ok
//   ippStsNullPtrErr      Some of pointers to input or output data are NULL
//   ippStsSizeErr         The length of the arrays is less or equal zero
//
*/

IPPAPI(IppStatus, ippsPolarToCart_32f,(const Ipp32f* pSrcMagn,
                  const Ipp32f* pSrcPhase, Ipp32f* pDstRe, Ipp32f* pDstIm, int len))
IPPAPI(IppStatus, ippsPolarToCart_64f,(const Ipp64f* pSrcMagn,
                  const Ipp64f* pSrcPhase, Ipp64f* pDstRe, Ipp64f* pDstIm, int len))


/* /////////////////////////////////////////////////////////////////////////////
//                  DFT Init Functions
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsDFTInit_C, ippsDFTInit_R
//  Purpose:    initialize of DFT context
//  Arguments:
//     length     Length of the DFT transform
//     flag       Flag to choose the results normalization factors
//     hint       Option to select the algorithmic implementation of the transform
//                function
//     pDFTSpec   Double pointer to the DFT context structure
//     pMemInit   Pointer to initialization buffer
//  Return:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers is NULL
//     ippStsOrderErr         Invalid length value
//     ippStsFFTFlagErr       Incorrect normalization flag value
//     ippStsSizeErr          Indicates an error when length is less than or equal to 0
*/

IPPAPI (IppStatus, ippsDFTInit_C_32fc,
                   ( int length, int flag, IppHintAlgorithm hint,
                   IppsDFTSpec_C_32fc* pDFTSpec, Ipp8u* pMemInit ))

IPPAPI (IppStatus, ippsDFTInit_C_64fc,
                   ( int length, int flag, IppHintAlgorithm hint,
                   IppsDFTSpec_C_64fc* pDFTSpec, Ipp8u* pMemInit ))

IPPAPI (IppStatus, ippsDFTInit_R_32f,
                   ( int length, int flag, IppHintAlgorithm hint,
                   IppsDFTSpec_R_32f* pDFTSpec, Ipp8u* pMemInit ))

IPPAPI (IppStatus, ippsDFTInit_R_64f,
                   ( int length, int flag, IppHintAlgorithm hint,
                   IppsDFTSpec_R_64f* pDFTSpec, Ipp8u* pMemInit ))


/* /////////////////////////////////////////////////////////////////////////////
//                  DFT Context Functions
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsDFTGetSize_C, ippsDFTGetSize_R
//  Purpose:    Computes the size of the DFT context structure and the size
                of the required work buffer (in bytes)
//  Arguments:
//     length     Length of the DFT transform
//     flag       Flag to choose the results normalization factors
//     hint       Option to select the algorithmic implementation of the transform
//                function
//     pSizeSpec  Pointer to the size value of DFT specification structure
//     pSizeInit  Pointer to the size value of the buffer for DFT initialization function
//     pSizeBuf   Pointer to the size value of the DFT external work buffer
//  Return:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers is NULL
//     ippStsOrderErr         Invalid length value
//     ippStsFFTFlagErr       Incorrect normalization flag value
//     ippStsSizeErr          Indicates an error when length is less than or equal to 0
*/

IPPAPI (IppStatus, ippsDFTGetSize_R_32f,
                   ( int length, int flag, IppHintAlgorithm hint,
                     int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))

IPPAPI (IppStatus, ippsDFTGetSize_R_64f,
                   ( int length, int flag, IppHintAlgorithm hint,
                     int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))

IPPAPI (IppStatus, ippsDFTGetSize_C_32fc,
                   ( int length, int flag, IppHintAlgorithm hint,
                     int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))

IPPAPI (IppStatus, ippsDFTGetSize_C_64fc,
                   ( int length, int flag, IppHintAlgorithm hint,
                     int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))


/* /////////////////////////////////////////////////////////////////////////////
//                  DFT Complex Transforms
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsDFTFwd_CToC, ippsDFTInv_CToC
//  Purpose:    Computes forward and inverse DFT of a complex signal
//  Arguments:
//     pDFTSpec     Pointer to the DFT context
//     pSrc         Pointer to the source complex signal
//     pDst         Pointer to the destination complex signal
//     pSrcRe       Pointer to the real      part of source signal
//     pSrcIm       Pointer to the imaginary part of source signal
//     pDstRe       Pointer to the real      part of destination signal
//     pDstIm       Pointer to the imaginary part of destination signal
//     pBuffer      Pointer to the work buffer
//     scaleFactor  Scale factor for output result
//  Return:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers with the exception of
//                            pBuffer is NULL
//     ippStsContextMatchErr  Invalid context structure
//     ippStsMemAllocErr      Memory allocation fails
*/

IPPAPI (IppStatus, ippsDFTInv_CToC_32fc,
                   ( const Ipp32fc* pSrc, Ipp32fc* pDst,
                     const IppsDFTSpec_C_32fc* pDFTSpec, Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippsDFTInv_CToC_64fc,
                   ( const Ipp64fc* pSrc, Ipp64fc* pDst,
                     const IppsDFTSpec_C_64fc* pDFTSpec, Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippsDFTFwd_CToC_32fc,
                   ( const Ipp32fc* pSrc, Ipp32fc* pDst,
                     const IppsDFTSpec_C_32fc* pDFTSpec, Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippsDFTFwd_CToC_64fc,
                   ( const Ipp64fc* pSrc, Ipp64fc* pDst,
                     const IppsDFTSpec_C_64fc* pDFTSpec, Ipp8u* pBuffer ))


/* /////////////////////////////////////////////////////////////////////////////
//                  DFT Real Packed Transforms
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsDFTFwd_RToPerm, ippsDFTFwd_RToPack, ippsDFTFwd_RToCCS
//              ippsDFTInv_PermToR, ippsDFTInv_PackToR, ippsDFTInv_CCSToR
//  Purpose:    Compute forward and inverse DFT of a real signal
//              using Perm, Pack or Ccs packed format
//  Arguments:
//     pFFTSpec       Pointer to the DFT context
//     pSrc           Pointer to the source signal
//     pDst           Pointer to the destination signal
//     pSrcDst        Pointer to the source/destination signal (in-place)
//     pBuffer        Pointer to the work buffer
//     scaleFactor    Scale factor for output result
//  Return:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers with the exception of
//                            pBuffer is NULL
//     ippStsContextMatchErr  Invalid context structure
//     ippStsMemAllocErr      Memory allocation fails
*/

IPPAPI (IppStatus, ippsDFTInv_PackToR_32f,
                   ( const Ipp32f* pSrc, Ipp32f* pDst,
                     const IppsDFTSpec_R_32f* pDFTSpec, Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippsDFTInv_PackToR_64f,
                   ( const Ipp64f* pSrc, Ipp64f* pDst,
                     const IppsDFTSpec_R_64f* pDFTSpec, Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippsDFTFwd_RToPack_32f,
                   ( const Ipp32f* pSrc, Ipp32f* pDst,
                     const IppsDFTSpec_R_32f* pDFTSpec, Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippsDFTFwd_RToPack_64f,
                   ( const Ipp64f* pSrc, Ipp64f* pDst,
                     const IppsDFTSpec_R_64f* pDFTSpec, Ipp8u* pBuffer ))


/* /////////////////////////////////////////////////////////////////////////////
//                  Dot Product Functions
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsDotProd
//  Purpose:    compute Dot Product value
//  Arguments:
//     pSrc1               pointer to the source vector
//     pSrc2               pointer to the another source vector
//     len                 vector's length, number of items
//     pDp                 pointer to the result
//     scaleFactor         scale factor value
//  Return:
//     ippStsNullPtrErr       pointer(s) pSrc pDst is NULL
//     ippStsSizeErr          length of the vectors is less or equal 0
//     ippStsNoErr            otherwise
//  Notes:
//     the functions don't conjugate one of the source vectors
*/

IPPAPI ( IppStatus, ippsDotProd_32f64f,
        ( const Ipp32f* pSrc1, const Ipp32f* pSrc2, int len, Ipp64f* pDp ))

IPPAPI(IppStatus, ippsDotProd_64f, (const Ipp64f* pSrc1,
       const Ipp64f* pSrc2, int len, Ipp64f* pDp))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:              ippsMinEvery, ippsMaxEvery
//  Purpose:            calculation min/max value for every element of two vectors
//  Parameters:
//   pSrc               pointer to input vector
//   pSrcDst            pointer to input/output vector
//   len                vector's length
//  Return:
//   ippStsNullPtrErr      pointer(s) to the data is NULL
//   ippStsSizeErr         vector`s length is less or equal zero
//   ippStsNoErr           otherwise
*/

IPPAPI(IppStatus, ippsMaxEvery_32f, ( const Ipp32f* pSrc1, const Ipp32f* pSrc2, Ipp32f* pDst, Ipp32u len ))
IPPAPI(IppStatus, ippsMinEvery_32f, ( const Ipp32f* pSrc1, const Ipp32f* pSrc2, Ipp32f* pDst, Ipp32u len ))

IPPAPI(IppStatus, ippsMinEvery_16u, ( const Ipp16u* pSrc1, const Ipp16u* pSrc2, Ipp16u* pDst, Ipp32u len ))
IPPAPI(IppStatus, ippsMaxEvery_16u, ( const Ipp16u* pSrc1, const Ipp16u* pSrc2, Ipp16u* pDst, Ipp32u len ))

IPPAPI(IppStatus, ippsMinEvery_8u,  ( const Ipp8u* pSrc1, const Ipp8u* pSrc2, Ipp8u* pDst, Ipp32u len ))
IPPAPI(IppStatus, ippsMaxEvery_8u,  ( const Ipp8u* pSrc1, const Ipp8u* pSrc2, Ipp8u* pDst, Ipp32u len ))

IPPAPI(IppStatus, ippsMinEvery_64f, ( const Ipp64f* pSrc1, const Ipp64f* pSrc2, Ipp64f* pDst, Ipp32u len ))
IPPAPI(IppStatus, ippsMaxEvery_64f, ( const Ipp64f* pSrc1, const Ipp64f* pSrc2, Ipp64f* pDst, Ipp32u len ))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:      ippsSortAscend, ippsSortDescend
//
//  Purpose:    Execute sorting of all elemens of the vector.
//              ippsSortAscend  is sorted in increasing order.
//              ippsSortDescend is sorted in decreasing order.
//  Arguments:
//    pSrcDst              pointer to the source/destination vector
//    len                  length of the vector
//  Return:
//    ippStsNullPtrErr     pointer to the data is NULL
//    ippStsSizeErr        length of the vector is less or equal zero
//    ippStsNoErr          otherwise
*/

IPPAPI(IppStatus, ippsSortAscend_8u_I,  (Ipp8u*  pSrcDst, int len))
IPPAPI(IppStatus, ippsSortAscend_16s_I, (Ipp16s* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortAscend_16u_I, (Ipp16u* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortAscend_32s_I, (Ipp32s* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortAscend_32f_I, (Ipp32f* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortAscend_64f_I, (Ipp64f* pSrcDst, int len))

IPPAPI(IppStatus, ippsSortDescend_8u_I,  (Ipp8u*  pSrcDst, int len))
IPPAPI(IppStatus, ippsSortDescend_16s_I, (Ipp16s* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortDescend_16u_I, (Ipp16u* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortDescend_32s_I, (Ipp32s* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortDescend_32f_I, (Ipp32f* pSrcDst, int len))
IPPAPI(IppStatus, ippsSortDescend_64f_I, (Ipp64f* pSrcDst, int len))

IPPAPI(IppStatus,ippsSortIndexAscend_8u_I, (Ipp8u*  pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexAscend_16s_I,(Ipp16s* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexAscend_16u_I,(Ipp16u* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexAscend_32s_I,(Ipp32s* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexAscend_32f_I,(Ipp32f* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexAscend_64f_I,(Ipp64f* pSrcDst, int* pDstIdx, int len ))

IPPAPI(IppStatus,ippsSortIndexDescend_8u_I, (Ipp8u*  pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexDescend_16s_I,(Ipp16s* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexDescend_16u_I,(Ipp16u* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexDescend_32s_I,(Ipp32s* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexDescend_32f_I,(Ipp32f* pSrcDst, int* pDstIdx, int len ))
IPPAPI(IppStatus,ippsSortIndexDescend_64f_I,(Ipp64f* pSrcDst, int* pDstIdx, int len ))



/* =============================================================================
							ippIP
============================================================================= */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiGetLibVersion
//  Purpose:    gets the version of the library
//  Returns:    structure containing information about the current version of
//  the Intel IPP library for image processing
//  Parameters:
//
//  Notes:      there is no need to release the returned structure
*/
IPPAPI( const IppLibraryVersion*, ippiGetLibVersion, (void) )


/*****************************************************************************************************
//  Name:       ippiFilterBorderGetSize, ippiFilterBorderInit, ippiFilterBorder
//  Purpose:    Filters an image using a general integer rectangular kernel
//  Returns:
//   ippStsNoErr       OK
//   ippStsNullPtrErr  One of the pointers is NULL
//   ippStsSizeErr     dstRoiSize or kernelSize has a field with zero or negative value
//   ippStsDivisorErr  Divisor value is zero, function execution is interrupted
//
//  Parameters:
//      pSrc        Distance, in bytes, between the starting points of consecutive lines in the source image
//      srcStep     Step in bytes through the source image buffer
//      pDst        Pointer to the destination buffer
//      dstStep     Distance, in bytes, between the starting points of consecutive lines in the destination image
//      dstRoiSize  Size of the source and destination ROI in pixels
//      pKernel     Pointer to the kernel values
//      kernelSize  Size of the rectangular kernel in pixels.
//      divisor     The integer value by which the computed result is divided.
//      kernelType  Kernel type {ipp16s|ipp32f}
//      dataType    Data type {ipp8u|ipp16u|ipp32f}
//      numChannels Number of channels, possible values are 1, 3 or 4
//      roundMode   Rounding mode (ippRndZero, ippRndNear or ippRndFinancial)
//      pSpecSize   Pointer to the size (in bytes) of the spec structure
//      pBufSize    Pointer to the size (in bytes) of the external buffer
//      pSpec       Pointer to pointer to the allocated and initialized context structure
//      borderType  Type of the border
//      borderValue Pointer to the constant value(s) if border type equals ippBorderConstant
//      pBuffer     Pointer to the work buffer. It can be equal to NULL if optimization algorithm doesn't demand a work buffer
*/

IPPAPI( IppStatus, ippiFilterBorderGetSize,  (IppiSize  kernelSize, IppiSize dstRoiSize, IppDataType dataType, IppDataType kernelType, int numChannels,int* pSpecSize, int* pBufferSize ))
IPPAPI( IppStatus, ippiFilterBorderInit_16s, ( const Ipp16s* pKernel, IppiSize  kernelSize, int divisor, IppDataType dataType, int numChannels , IppRoundMode roundMode, IppiFilterBorderSpec* pSpec ))
IPPAPI( IppStatus, ippiFilterBorderInit_32f, ( const Ipp32f* pKernel, IppiSize  kernelSize,              IppDataType dataType, int numChannels , IppRoundMode roundMode, IppiFilterBorderSpec* pSpec ))
IPPAPI( IppStatus, ippiFilterBorder_8u_C1R,  ( const Ipp8u*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp8u borderValue[1], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_8u_C3R,  ( const Ipp8u*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp8u borderValue[3], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_8u_C4R,  ( const Ipp8u*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp8u borderValue[4], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_16u_C1R, ( const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp16u borderValue[1], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_16u_C3R, ( const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp16u borderValue[3], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_16u_C4R, ( const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp16u borderValue[4], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_16s_C1R, ( const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp16s borderValue[1], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_16s_C3R, ( const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp16s borderValue[3], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_16s_C4R, ( const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp16s borderValue[4], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_32f_C1R, ( const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp32f borderValue[1], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_32f_C3R, ( const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp32f borderValue[3], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiFilterBorder_32f_C4R, ( const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType border, const Ipp32f borderValue[4], const IppiFilterBorderSpec* pSpec, Ipp8u* pBuffer ))


/* /////////////////////////////////////////////////////////////////////////////
//  Names:     ippiHistogramGetBufferSize
//  Purpose:   Get the sizes (in bytes) of the spec and the buffer for ippiHistogram_.
//  Parameters:
//    dataType    - Data type for source image. Possible values are ipp8u, ipp16u, ipp16s or ipp32f.
//    roiSize     - Size, in pixels, of the source image.
//    nLevels     - Number of levels values, separate for each channel.
//    numChannels - Number of image channels. Possible values are 1, 3, or 4.
//    uniform     - Type of levels distribution: 0 - with random step, 1 - with uniform step.
//    pSpecSize   - Pointer to the calculated spec size (in bytes).
//    pBufferSize - Pointer to the calculated buffer size (in bytes).
//  Return:
//    ippStsNoErr             - OK.
//    ippStsNullPtrErr        - Error when any of the specified pointers is NULL.
//    ippStsSizeErr           - Error when the roiSize has a zero or negative value.
//    ippStsHistoNofLevelsErr - Error when the number of levels is less than 2.
//    ippStsNumChannelsErr    - Error when the numChannels value differs from 1, 3, or 4.
//    ippStsDataTypeErr       - Error when the dataType value differs from the ipp8u, ipp16u, ipp16s or ipp32f.
*/
IPPAPI(IppStatus, ippiHistogramGetBufferSize, (IppDataType dataType, IppiSize roiSize, const int nLevels[], int numChannels, int uniform, int* pSpecSize, int* pBufferSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Names:     ippiHistogramInit, ippiHistogramUniformInit
//  Purpose:   Initializes the Spec for ippiHistogram.
//  Parameters:
//    dataType    - Data type for source image. Possible values are ipp8u, ipp16u, ipp16s or ipp32f.
//    pLevels     - Pointer to the array of level values. In case of multi-channel data, pLevels is an array of pointers to the level values array for each channel.
//    lowerLevel  - The lower levels for uniform histogram, separate for each channel.
//    upperLevel  - The upper levels for uniform histogram, separate for each channel.
//    nLevels     - Number of levels values, separate for each channel.
//    numChannels - Number of image channels. Possible values are 1, 3, or 4.
//    pSpec       - Pointer to the spec object.
//  Return:
//    ippStsNoErr             - OK.
//    ippStsNullPtrErr        - Error when any of the specified pointers is NULL.
//    ippStsNumChannelsErr    - Error when the numChannels value differs from 1, 3, or 4.
//    ippStsHistoNofLevelsErr - Error when the number of levels is less than 2.
//    ippStsRangeErr          - Error when consecutive pLevels values don't satisfy the condition: pLevel[i] < pLevel[i+1].
//    ippStsDataTypeErr       - Error when the dataType value differs from the ipp8u, ipp16u, ipp16s or ipp32f.
//    ippStsSizeWrn           - Warning ( in case of uniform histogram of integer data type) when rated level step is less than 1.
*/
IPPAPI(IppStatus, ippiHistogramInit, (IppDataType dataType, const Ipp32f *pLevels[], int nLevels[], int numChannels, IppiHistogramSpec* pSpec))
IPPAPI(IppStatus, ippiHistogramUniformInit, (IppDataType dataType, Ipp32f lowerLevel[], Ipp32f upperLevel[], int nLevels[], int numChannels, IppiHistogramSpec* pSpec))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:      ippiHistogramGetLevels
//  Purpose:   Returns levels arrays stored in the pSpec object.
//  Parameters:
//    pSpec       - Pointer to the spec object.
//    pLevels     - Pointer to the array of level values. In case of multi-channel data, pLevels is an array of pointers to the level values array for each channel.
//  Return:
//    ippStsNoErr             - OK.
//    ippStsNullPtrErr        - Error when any of the specified pointers is NULL.
//    ippStsBadArgErr         - Error when pSpec object doesn`t initialized.
*/
IPPAPI(IppStatus, ippiHistogramGetLevels, (const IppiHistogramSpec* pSpec, Ipp32f *pLevels[]))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:         ippiHistogram
//  Purpose:      Computes the intensity histogram of an image.
//  Parameters:
//    pSrc        - Pointer to the source image ROI.
//    srcStep     - Distance, in bytes, between the starting points of consecutive lines in the source image.
//    roiSize     - Size, in pixels, of the source image.
//    pHist       - Pointer to the computed histogram. In case of multi-channel data, pHist is an array of pointers to the histogram for each channel.
//    pSpec       - Pointer to the spec.
//    pBuffer     - Pointer to the buffer for internal calculations.
//  Returns:
//    ippStsNoErr             - OK.
//    ippStsNullPtrErr        - Error when any of the specified pointers is NULL.
//    ippStsSizeErr           - Error when the roiSize has a zero or negative value.
//    ippStsStepErr           - Error when the srcStep is less than roiSize.width*sizeof(*pSrc)*nChannels.
//    ippStsBadArgErr         - Error when pSpec object doesn`t initialized.
*/
IPPAPI(IppStatus, ippiHistogram_8u_C1R , (const Ipp8u*  pSrc, int srcStep, IppiSize roiSize, Ipp32u* pHist   , const IppiHistogramSpec* pSpec, Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiRemap
//  Purpose:            Transforms the source image by remapping its pixels
//                          dst[i,j] = src[xMap[i,j], yMap[i,j]]
//  Parameters:
//    pSrc              Pointer to the source image (point to pixel (0,0)). An array
//                      of pointers to each plane of the source image for planar data
//    srcSize           Size of the source image
//    srcStep           Step through the source image
//    srcROI            Region if interest in the source image
//    pxMap             Pointer to image with x coordinates of map
//    xMapStep          The step in xMap image
//    pyMap             The pointer to image with y coordinates of map
//    yMapStep          The step in yMap image
//    pDst              Pointer to the destination image. An array of pointers
//                      to each plane of the destination image for planar data
//    dstStep           Step through the destination image
//    dstRoiSize        Size of the destination ROI
//    interpolation     The type of interpolation to perform for image resampling
//                      The following types are currently supported:
//                        IPPI_INTER_NN       Nearest neighbor interpolation
//                        IPPI_INTER_LINEAR   Linear interpolation
//                        IPPI_INTER_CUBIC    Cubic interpolation
//                        IPPI_INTER_CUBIC2P_CATMULLROM  Catmull-Rom cubic filter
//                        IPPI_INTER_LANCZOS  Interpolation by Lanczos3-windowed sinc function
//                      The special feature in addition to one of general methods:
//                        IPPI_SMOOTH_EDGE    Edges smoothing
//  Returns:
//    ippStsNoErr       OK
//    ippStsNullPtrErr  One of the pointers is NULL
//    ippStsSizeErr     srcROI or dstRoiSize has a field with zero or negative value
//    ippStsStepErr     One of the step values is zero or negative
//    ippStsInterpolateErr  interpolation has an illegal value
*/

IPPAPI(IppStatus, ippiRemap_8u_C1R, (const Ipp8u* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp8u* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_8u_C3R, (const Ipp8u* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp8u* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_8u_C4R, (const Ipp8u* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp8u* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_16u_C1R, (const Ipp16u* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp16u* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_16u_C3R, (const Ipp16u* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp16u* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_16u_C4R, (const Ipp16u* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp16u* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_16s_C1R, (const Ipp16s* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp16s* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_16s_C3R, (const Ipp16s* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp16s* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_16s_C4R, (const Ipp16s* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp16s* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_32f_C1R, (const Ipp32f* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp32f* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_32f_C3R, (const Ipp32f* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp32f* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))
IPPAPI(IppStatus, ippiRemap_32f_C4R, (const Ipp32f* pSrc, IppiSize srcSize,
    int srcStep, IppiRect srcROI, const Ipp32f* pxMap, int xMapStep,
    const Ipp32f* pyMap, int yMapStep, Ipp32f* pDst, int dstStep,
    IppiSize dstRoiSize, int interpolation))


/* /////////////////////////////////////////////////////////////////////////////
//                     Warp Transform functions
// ////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiWarpGetBufferSize
//  Purpose:            Computes the size of external buffer for Warp transform
//
//  Parameters:
//    pSpec             Pointer to the Spec structure for warp transform
//    dstRoiSize        Size of the output image (in pixels)
//    numChannels       Number of channels, possible values are 1 or 3 or 4
//    pBufSize          Pointer to the size (in bytes) of the external buffer
//
//  Return Values:
//    ippStsNoErr           Indicates no error
//    ippStsNullPtrErr      Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation     Indicates a warning if width or height of output image is zero
//    ippStsContextMatchErr Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNumChannelsErr  Indicates an error if numChannels has illegal value
//    ippStsSizeErr         Indicates an error condition in the following cases:
//                          - if width or height of the source image is negative,
//                          - if the calculated buffer size exceeds maximum 32 bit signed integer
//                            positive value (the processed image ROIs are too large ).
//    ippStsSizeWrn         Indicates a warning if the destination image size is more than
//                          the destination image origin size
*/
IPPAPI (IppStatus, ippiWarpGetBufferSize, (const IppiWarpSpec* pSpec, IppiSize dstRoiSize, int* pBufSize))

/* /////////////////////////////////////////////////////////////////////////////
//                     Warp Affine Transform functions
// ////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiWarpAffineGetSize
//  Purpose:            Computes the size of Spec structure and temporal buffer for Affine transform
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    dstSize           Size of the output image (in pixels)
//    dataType          Data type of the source and destination images. Possible values
//                      are ipp8u, ipp16u, ipp16s, ipp32f and ipp64f.
//    coeffs            The affine transform coefficients
//    interpolation     Interpolation method. Supported values: ippNearest, ippLinear and ippCubic.
//    direction         Transformation direction. Possible values are:
//                          ippWarpForward  - Forward transformation.
//                          ippWarpBackward - Backward transformation.
//    border            Type of the border
//    pSpecSize         Pointer to the size (in bytes) of the Spec structure
//    pInitBufSize      Pointer to the size (in bytes) of the temporal buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error in the following cases:
//                              -  if width or height of the source or destination image is negative,
//                              -  if one of the calculated sizes exceeds maximum 32 bit signed integer
//                                 positive value (the size of the one of the processed images is too large).
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsWarpDirectionErr    Indicates an error when the direction value is illegal.
//    ippStsInterpolationErr    Indicates an error if interpolation has an illegal value
//    ippStsNotSupportedModeErr Indicates an error if the requested mode is not supported.
//    ippStsCoeffErr            Indicates an error condition, if affine transformation is singular.
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//
*/
IPPAPI (IppStatus, ippiWarpAffineGetSize, (IppiSize srcSize, IppiSize dstSize, IppDataType dataType, const double coeffs[2][3],
                                           IppiInterpolationType interpolation, IppiWarpDirection direction,
                                           IppiBorderType borderType, int* pSpecSize, int* pInitBufSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiWarpAffineNearestInit
//                      ippiWarpAffineLinearInit
//                      ippiWarpAffineCubicInit
//
//  Purpose:            Initializes the Spec structure for the Warp affine transform
//                      by different interpolation methods
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    dstSize           Size of the output image (in pixels)
//    dataType          Data type of the source and destination images. Possible values are:
//                      ipp8u, ipp16u, ipp16s, ipp32f, ipp64f.
//    coeffs            The affine transform coefficients
//    direction         Transformation direction. Possible values are:
//                          ippWarpForward  - Forward transformation.
//                          ippWarpBackward - Backward transformation.
//    numChannels       Number of channels, possible values are 1 or 3 or 4
//    valueB            The first parameter (B) for specifying Cubic filters
//    valueC            The second parameter (C) for specifying Cubic filters
//    border            Type of the border
//    borderValue       Pointer to the constant value(s) if border type equals ippBorderConstant
//    smoothEdge        The smooth edge flag. Supported values:
//                          0 - transform without edge smoothing
//                          1 - transform with edge smoothing
//    pSpec             Pointer to the Spec structure for resize filter
//    pInitBuf          Pointer to the temporal buffer for several initialization cases
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error if width or height of the source or destination
//                              image is negative
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsWarpDirectionErr    Indicates an error when the direction value is illegal.
//    ippStsCoeffErr            Indicates an error condition, if the affine transformation is singular.
//    ippStsNumChannelsErr      Indicates an error if numChannels has illegal value
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//    ippStsWrongIntersectQuad  Indicates a warning that no operation is performed, if the transformed
//                              source image has no intersection with the destination image.
//    ippStsNotSupportedModeErr Indicates an error if the requested mode is not supported.
//
//  Notes/References:
//    1. The equation shows the family of cubic filters:
//           ((12-9B-6C)*|x|^3 + (-18+12B+6C)*|x|^2                  + (6-2B)  ) / 6   for |x| < 1
//    K(x) = ((   -B-6C)*|x|^3 + (    6B+30C)*|x|^2 + (-12B-48C)*|x| + (8B+24C); / 6   for 1 <= |x| < 2
//           0   elsewhere
//    Some values of (B,C) correspond to known cubic splines: Catmull-Rom (B=0,C=0.5), B-Spline (B=1,C=0) and other.
//      Mitchell, Don P.; Netravali, Arun N. (Aug. 1988). "Reconstruction filters in computer graphics"
//      http://www.mentallandscape.com/Papers_siggraph88.pdf
//
//    2. Supported border types are ippBorderRepl, ippBorderConst, ippBorderTransp and ippBorderInMem
*/
IPPAPI (IppStatus, ippiWarpAffineNearestInit, (IppiSize srcSize, IppiSize dstSize, IppDataType dataType, const double coeffs[2][3],
                                               IppiWarpDirection direction, int numChannels, IppiBorderType borderType,
                                               const Ipp64f *pBorderValue, int smoothEdge, IppiWarpSpec* pSpec))
IPPAPI (IppStatus, ippiWarpAffineLinearInit, (IppiSize srcSize, IppiSize dstSize, IppDataType dataType, const double coeffs[2][3],
                                               IppiWarpDirection direction, int numChannels, IppiBorderType borderType,
                                               const Ipp64f *pBorderValue, int smoothEdge, IppiWarpSpec* pSpec))
IPPAPI (IppStatus, ippiWarpAffineCubicInit, (IppiSize srcSize, IppiSize dstSize, IppDataType dataType, const double coeffs[2][3],
                                             IppiWarpDirection direction, int numChannels, Ipp64f valueB, Ipp64f valueC,
                                             IppiBorderType borderType, const Ipp64f *pBorderValue, int smoothEdge, IppiWarpSpec* pSpec,
                                             Ipp8u* pInitBuf))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiWarpAffineNearest
//                      ippiWarpAffineLinear
//                      ippiWarpAffineCubic
//
//  Purpose:            Performs affine transform of an image with using different interpolation methods
//
//  Parameters:
//    pSrc              Pointer to the source image
//    srcStep           Distance (in bytes) between of consecutive lines in the source image
//    pDst              Pointer to the destination image
//    dstStep           Distance (in bytes) between of consecutive lines in the destination image
//    dstRoiOffset      Offset of tiled image respectively destination image origin
//    dstRoiSize        Size of the destination image (in pixels)
//    border            Type of the border
//    borderValue       Pointer to the constant value(s) if border type equals ippBorderConstant
//    pSpec             Pointer to the Spec structure for resize filter
//    pBuffer           Pointer to the work buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of output image is zero
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//    ippStsContextMatchErr     Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNotSupportedModeErr Indicates an error if requested mode is currently not supported
//    ippStsSizeErr             Indicates an error if width or height of the destination image
//                              is negative
//    ippStsStepErr             Indicates an error if the step value is not data type multiple
//    ippStsOutOfRangeErr       Indicates an error if the destination image offset point is outside the
//                              destination image origin
//    ippStsSizeWrn             Indicates a warning if the destination image size is more than
//                              the destination image origin size
//    ippStsWrongIntersectQuad  Indicates a warning that no operation is performed if the destination
//                              ROI has no intersection with the transformed source image origin.
//
//  Notes:
//    1. Supported border types are ippBorderRepl, ippBorderConst, ippBorderTransp and ippBorderRepl
*/
IPPAPI (IppStatus, ippiWarpAffineNearest_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_16u_C1R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_16u_C3R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_16u_C4R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_16s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_16s_C3R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_16s_C4R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_32f_C4R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_64f_C1R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_64f_C3R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineNearest_64f_C4R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiWarpAffineLinear_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_16u_C1R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_16u_C3R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_16u_C4R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_16s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_16s_C3R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_16s_C4R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_32f_C4R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_64f_C1R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_64f_C3R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineLinear_64f_C4R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiWarpAffineCubic_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_16u_C1R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_16u_C3R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_16u_C4R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_16s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_16s_C3R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_16s_C4R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_32f_C4R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_64f_C1R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_64f_C3R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpAffineCubic_64f_C4R, (const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))

/* /////////////////////////////////////////////////////////////////////////////
//                     Warp Perspective Transform functions
// ////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiWarpPerspectiveGetSize
//  Purpose:            Computes the size of Spec structure and temporal buffer
//                      for Perspective transform
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    srcRoi            Region of interest in the source image
//    dstSize           Size of the output image (in pixels)
//    dataType          Data type of the source and destination images. Possible values
//                      are ipp8u, ipp16u, ipp16s, ipp32f and ipp64f.
//    coeffs            The perspective transform coefficients
//    interpolation     Interpolation method. Supported values: ippNearest, ippLinear and ippCubic.
//    direction         Transformation direction. Possible values are:
//                          ippWarpForward  - Forward transformation.
//                          ippWarpBackward - Backward transformation.
//    border            Type of the border
//    pSpecSize         Pointer to the size (in bytes) of the Spec structure
//    pInitBufSize      Pointer to the size (in bytes) of the temporal buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error in the following cases:
//                              -  if width or height of the source or destination image is negative,
//                              -  if one of the calculated sizes exceeds maximum 32 bit signed integer
//                                 positive value (the size of the one of the processed images is too large).
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsWarpDirectionErr    Indicates an error when the direction value is illegal.
//    ippStsInterpolationErr    Indicates an error if interpolation has an illegal value
//    ippStsNotSupportedModeErr Indicates an error if the requested mode is not supported.
//    ippStsCoeffErr            Indicates an error condition, if perspective transformation is singular.
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//
*/
IPPAPI (IppStatus, ippiWarpPerspectiveGetSize, (IppiSize srcSize, IppiRect srcRoi, IppiSize dstSize, IppDataType dataType,
                                                const double coeffs[3][3], IppiInterpolationType interpolation,
                                                IppiWarpDirection direction, IppiBorderType borderType, int* pSpecSize,
                                                int* pInitBufSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiWarpPerspectiveNearestInit
//                      ippiWarpPerspectiveLinearInit
//                      ippiWarpPerspectiveCubicInit
//
//  Purpose:            Initializes the Spec structure for the Warp perspective transform
//                      by different interpolation methods
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    srcRoi            Region of interest in the source image
//    dstSize           Size of the output image (in pixels)
//    dataType          Data type of the source and destination images. Possible values are:
//                      ipp8u, ipp16u, ipp16s, ipp32f, ipp64f.
//    coeffs            The perspective transform coefficients
//    direction         Transformation direction. Possible values are:
//                          ippWarpForward  - Forward transformation.
//                          ippWarpBackward - Backward transformation.
//    numChannels       Number of channels, possible values are 1 or 3 or 4
//    valueB            The first parameter (B) for specifying Cubic filters
//    valueC            The second parameter (C) for specifying Cubic filters
//    border            Type of the border
//    borderValue       Pointer to the constant value(s) if border type equals ippBorderConstant
//    smoothEdge        The smooth edge flag. Supported values:
//                          0 - transform without edge smoothing
//                          1 - transform with edge smoothing
//    pSpec             Pointer to the Spec structure for resize filter
//    pInitBuf          Pointer to the temporal buffer for several initialization cases
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error if width or height of the source or destination
//                              image is negative
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsWarpDirectionErr    Indicates an error when the direction value is illegal.
//    ippStsCoeffErr            Indicates an error condition, if the perspective transformation is singular.
//    ippStsNumChannelsErr      Indicates an error if numChannels has illegal value
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//    ippStsWrongIntersectQuad  Indicates a warning that no operation is performed, if the transformed
//                              source image has no intersection with the destination image.
//    ippStsNotSupportedModeErr Indicates an error if the requested mode is not supported.
//
//  Notes/References:
//    1. The equation shows the family of cubic filters:
//           ((12-9B-6C)*|x|^3 + (-18+12B+6C)*|x|^2                  + (6-2B)  ) / 6   for |x| < 1
//    K(x) = ((   -B-6C)*|x|^3 + (    6B+30C)*|x|^2 + (-12B-48C)*|x| + (8B+24C); / 6   for 1 <= |x| < 2
//           0   elsewhere
//    Some values of (B,C) correspond to known cubic splines: Catmull-Rom (B=0,C=0.5), B-Spline (B=1,C=0) and other.
//      Mitchell, Don P.; Netravali, Arun N. (Aug. 1988). "Reconstruction filters in computer graphics"
//      http://www.mentallandscape.com/Papers_siggraph88.pdf
//
//    2. Supported border types are ippBorderRepl, ippBorderConst, ippBorderTransp and ippBorderRepl
*/
IPPAPI (IppStatus, ippiWarpPerspectiveNearestInit, (IppiSize srcSize, IppiRect srcRoi, IppiSize dstSize, IppDataType dataType,
                                                    const double coeffs[3][3], IppiWarpDirection direction, int numChannels,
                                                    IppiBorderType borderType, const Ipp64f *pBorderValue, int smoothEdge, IppiWarpSpec* pSpec))
IPPAPI (IppStatus, ippiWarpPerspectiveLinearInit, (IppiSize srcSize, IppiRect srcRoi, IppiSize dstSize, IppDataType dataType,
                                                   const double coeffs[3][3], IppiWarpDirection direction, int numChannels,
                                                   IppiBorderType borderType, const Ipp64f *pBorderValue, int smoothEdge, IppiWarpSpec* pSpec))
IPPAPI (IppStatus, ippiWarpPerspectiveCubicInit, (IppiSize srcSize, IppiRect srcRoi, IppiSize dstSize, IppDataType dataType, const double coeffs[3][3],
                                                  IppiWarpDirection direction, int numChannels, Ipp64f valueB, Ipp64f valueC,
                                                  IppiBorderType borderType, const Ipp64f *pBorderValue, int smoothEdge, IppiWarpSpec* pSpec,
                                                  Ipp8u* pInitBuf))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiWarpPerspectiveNearest
//                      ippiWarpPerspectiveLinear
//                      ippiWarpPerspectiveCubic
//
//  Purpose:            Performs perspective transform of an image with using
//                      different interpolation methods
//
//  Parameters:
//    pSrc              Pointer to the source image
//    srcStep           Distance (in bytes) between of consecutive lines in the source image
//    pDst              Pointer to the destination image
//    dstStep           Distance (in bytes) between of consecutive lines in the destination image
//    dstRoiOffset      Offset of tiled image respectively destination image origin
//    dstRoiSize        Size of the destination image (in pixels)
//    border            Type of the border
//    borderValue       Pointer to the constant value(s) if border type equals ippBorderConstant
//    pSpec             Pointer to the Spec structure for resize filter
//    pBuffer           Pointer to the work buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of output image is zero
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//    ippStsContextMatchErr     Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNotSupportedModeErr Indicates an error if requested mode is currently not supported
//    ippStsSizeErr             Indicates an error if width or height of the destination image
//                              is negative
//    ippStsStepErr             Indicates an error if the step value is not data type multiple
//    ippStsOutOfRangeErr       Indicates an error if the destination image offset point is outside the
//                              destination image origin
//    ippStsSizeWrn             Indicates a warning if the destination image size is more than
//                              the destination image origin size
//    ippStsWrongIntersectQuad  Indicates a warning that no operation is performed if the destination
//                              ROI has no intersection with the transformed source image origin.
//
//  Notes:
//    1. Supported border types are ippBorderRepl, ippBorderConst, ippBorderTransp and ippBorderRepl
*/
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_16u_C1R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_16u_C3R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_16u_C4R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_16s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_16s_C3R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_16s_C4R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveNearest_32f_C4R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                   IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                   Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiWarpPerspectiveLinear_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_16u_C1R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_16u_C3R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_16u_C4R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_16s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_16s_C3R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_16s_C4R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveLinear_32f_C4R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                  IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                  Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiWarpPerspectiveCubic_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_16u_C1R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_16u_C3R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_16u_C4R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_16s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_16s_C3R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_16s_C4R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiWarpPerspectiveCubic_32f_C4R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                 IppiPoint dstRoiOffset, IppiSize dstRoiSize, const IppiWarpSpec* pSpec,
                                                 Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//                  Alpha Compositing Operations
///////////////////////////////////////////////////////////////////////////// */
/*
//  Contents:
//      ippiAlphaPremul_8u_AC4R,  ippiAlphaPremul_16u_AC4R
//      ippiAlphaPremul_8u_AC4IR, ippiAlphaPremul_16u_AC4IR
//      ippiAlphaPremul_8u_AP4R,  ippiAlphaPremul_16u_AP4R
//      ippiAlphaPremul_8u_AP4IR, ippiAlphaPremul_16u_AP4IR
//   Pre-multiplies pixel values of an image by its alpha values.

//      ippiAlphaPremulC_8u_AC4R,  ippiAlphaPremulC_16u_AC4R
//      ippiAlphaPremulC_8u_AC4IR, ippiAlphaPremulC_16u_AC4IR
//      ippiAlphaPremulC_8u_AP4R,  ippiAlphaPremulC_16u_AP4R
//      ippiAlphaPremulC_8u_AP4IR, ippiAlphaPremulC_16u_AP4IR
//      ippiAlphaPremulC_8u_C4R,   ippiAlphaPremulC_16u_C4R
//      ippiAlphaPremulC_8u_C4IR,  ippiAlphaPremulC_16u_C4IR
//      ippiAlphaPremulC_8u_C3R,   ippiAlphaPremulC_16u_C3R
//      ippiAlphaPremulC_8u_C3IR,  ippiAlphaPremulC_16u_C3IR
//      ippiAlphaPremulC_8u_C1R,   ippiAlphaPremulC_16u_C1R
//      ippiAlphaPremulC_8u_C1IR,  ippiAlphaPremulC_16u_C1IR
//   Pre-multiplies pixel values of an image by constant alpha values.
//
//      ippiAlphaComp_8u_AC4R, ippiAlphaComp_16u_AC4R
//      ippiAlphaComp_8u_AC1R, ippiAlphaComp_16u_AC1R
//   Combines two images using alpha values of both images
//
//      ippiAlphaCompC_8u_AC4R, ippiAlphaCompC_16u_AC4R
//      ippiAlphaCompC_8u_AP4R, ippiAlphaCompC_16u_AP4R
//      ippiAlphaCompC_8u_C4R,  ippiAlphaCompC_16u_C4R
//      ippiAlphaCompC_8u_C3R,  ippiAlphaCompC_16u_C3R
//      ippiAlphaCompC_8u_C1R,  ippiAlphaCompC_16u_C1R
//   Combines two images using constant alpha values
//
//  Types of compositing operation (alphaType)
//      OVER   ippAlphaOver   ippAlphaOverPremul
//      IN     ippAlphaIn     ippAlphaInPremul
//      OUT    ippAlphaOut    ippAlphaOutPremul
//      ATOP   ippAlphaATop   ippAlphaATopPremul
//      XOR    ippAlphaXor    ippAlphaXorPremul
//      PLUS   ippAlphaPlus   ippAlphaPlusPremul
//
//  Type  result pixel           result pixel (Premul)    result alpha
//  OVER  aA*A+(1-aA)*aB*B         A+(1-aA)*B             aA+(1-aA)*aB
//  IN    aA*A*aB                  A*aB                   aA*aB
//  OUT   aA*A*(1-aB)              A*(1-aB)               aA*(1-aB)
//  ATOP  aA*A*aB+(1-aA)*aB*B      A*aB+(1-aA)*B          aA*aB+(1-aA)*aB
//  XOR   aA*A*(1-aB)+(1-aA)*aB*B  A*(1-aB)+(1-aA)*B      aA*(1-aB)+(1-aA)*aB
//  PLUS  aA*A+aB*B                A+B                    aA+aB
//      Here 1 corresponds significance VAL_MAX, multiplication is performed
//      with scaling
//          X * Y => (X * Y) / VAL_MAX
//      and VAL_MAX is the maximum presentable pixel value:
//          VAL_MAX == IPP_MAX_8U  for 8u
//          VAL_MAX == IPP_MAX_16U for 16u
*/

/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiAlphaPremul_8u_AC4R,  ippiAlphaPremul_16u_AC4R
//                  ippiAlphaPremul_8u_AC4IR, ippiAlphaPremul_16u_AC4IR
//                  ippiAlphaPremul_8u_AP4R,  ippiAlphaPremul_16u_AP4R
//                  ippiAlphaPremul_8u_AP4IR, ippiAlphaPremul_16u_AP4IR
//
//  Purpose:        Pre-multiplies pixel values of an image by its alpha values
//                  for 4-channel images
//               For channels 1-3
//                      dst_pixel = (src_pixel * src_alpha) / VAL_MAX
//               For alpha-channel (channel 4)
//                      dst_alpha = src_alpha
//  Parameters:
//     pSrc         Pointer to the source image for pixel-order data,
//                  array of pointers to separate source color planes for planar data
//     srcStep      Step through the source image
//     pDst         Pointer to the destination image for pixel-order data,
//                  array of pointers to separate destination color planes for planar data
//     dstStep      Step through the destination image
//     pSrcDst      Pointer to the source/destination image, or array of pointers
//                  to separate source/destination color planes for in-place functions
//     srcDstStep   Step through the source/destination image for in-place functions
//     roiSize      Size of the source and destination ROI
//  Returns:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       pSrc == NULL, or pDst == NULL, or pSrcDst == NULL
//     ippStsSizeErr          The roiSize has a field with negative or zero value
*/

IPPAPI (IppStatus, ippiAlphaPremul_8u_AC4R,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep,
                     IppiSize roiSize ))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:       ippiLUTPalette
//  Purpose:     intensity transformation of image using the palette lookup table pTable
//  Parameters:
//    pSrc       pointer to the source image
//    srcStep    line offset in input data in bytes
//    alphaValue constant alpha channel
//    pDst       pointer to the destination image
//    dstStep    line offset in output data in bytes
//    roiSize    size of source ROI in pixels
//    pTable     pointer to palette table of size 2^nBitSize or
//               array of pointers to each channel
//    nBitSize   number of valid bits in the source image
//               (range [1,8] for 8u source images and range [1,16] for 16u source images)
//  Returns:
//    ippStsNoErr         no errors
//    ippStsNullPtrErr    pSrc == NULL or pDst == NULL or pTable == NULL
//    ippStsSizeErr       width or height of ROI is less or equal zero
//    ippStsOutOfRangeErr nBitSize is out of range
//  Notes:
*/
IPPAPI(IppStatus, ippiLUTPalette_8u32u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp32u* pDst, int dstStep,
       IppiSize roiSize, const Ipp32u* pTable, int nBitSize))
IPPAPI(IppStatus, ippiLUTPalette_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
       IppiSize roiSize, const Ipp8u* pTable, int nBitSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:                ippiCountInRange
//
//  Purpose:  Computes the number of pixels with intensity values within the given range
//
//  Returns:             IppStatus
//      ippStsNoErr       No errors
//      ippStsNullPtrErr  pSrc == NULL
//      ippStsStepErr     srcStep is less than or equal to zero
//      ippStsSizeErr     roiSize has a field with zero or negative value
//      ippStsRangeErr    lowerBound is greater than upperBound
//
//  Parameters:
//      pSrc             Pointer to the source buffer
//      roiSize          Size of the source ROI
//      srcStep          Step through the source image buffer
//      counts           Number of pixels within the given intensity range
//      lowerBound       Lower limit of the range
//      upperBound       Upper limit of the range
*/

IPPAPI(IppStatus, ippiCountInRange_8u_C1R, (const Ipp8u* pSrc, int srcStep, IppiSize roiSize,
                                            int* counts, Ipp8u lowerBound, Ipp8u upperBound))
IPPAPI(IppStatus, ippiCountInRange_32f_C1R, (const Ipp32f* pSrc, int srcStep, IppiSize roiSize,
                                            int* counts, Ipp32f lowerBound, Ipp32f upperBound))


/* ////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiAdd_32f_C1IR, ippiAdd_32f_C3IR, ippiAdd_32f_C4IR, ippiAdd_32f_AC4IR,
//        ippiSub_32f_C1IR, ippiSub_32f_C3IR, ippiSub_32f_C4IR, ippiSub_32f_AC4IR,
//        ippiMul_32f_C1IR, ippiMul_32f_C3IR, ippiMul_32f_C4IR, ippiMul_32f_AC4IR
//
//  Purpose:    Adds, subtracts, or multiplies pixel values of two source images
//              and places the results in the first image.
//
//  Returns:
//    ippStsNoErr              OK
//    ippStsNullPtrErr         One of the pointers is NULL
//    ippStsSizeErr            Width or height of images is less than or equal to zero
//
//  Parameters:
//    pSrc                     Pointer to the second source image
//    srcStep                  Step through the second source image
//    pSrcDst                  Pointer to the  first source/destination image
//    srcDstStep               Step through the first source/destination image
//    roiSize                  Size of the ROI
*/

IPPAPI(IppStatus, ippiAdd_32f_C1IR,  (const Ipp32f* pSrc, int srcStep, Ipp32f* pSrcDst,
                                      int srcDstStep, IppiSize roiSize))



/* //////////////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiAddC_8u_C1IRSfs,  ippiAddC_8u_C3IRSfs,  ippiAddC_8u_C4IRSfs,   ippiAddC_8u_AC4IRSfs,
//        ippiAddC_16s_C1IRSfs, ippiAddC_16s_C3IRSfs, ippiAddC_16s_C4IRSfs,  ippiAddC_16s_AC4IRSfs,
//        ippiAddC_16u_C1IRSfs, ippiAddC_16u_C3IRSfs, ippiAddC_16u_C4IRSfs,  ippiAddC_16u_AC4IRSfs,
//        ippiSubC_8u_C1IRSfs,  ippiSubC_8u_C3IRSfs,  ippiSubC_8u_C4IRSfs,   ippiSubC_8u_AC4IRSfs,
//        ippiSubC_16s_C1IRSfs, ippiSubC_16s_C3IRSfs, ippiSubC_16s_C4IRSfs,  ippiSubC_16s_AC4IRSfs,
//        ippiSubC_16u_C1IRSfs, ippiSubC_16u_C3IRSfs, ippiSubC_16u_C4IRSfs,  ippiSubC_16u_AC4IRSfs,
//        ippiMulC_8u_C1IRSfs,  ippiMulC_8u_C3IRSfs,  ippiMulC_8u_C4IRSfs,   ippiMulC_8u_AC4IRSfs,
//        ippiMulC_16s_C1IRSfs, ippiMulC_16s_C3IRSfs, ippiMulC_16s_C4IRSfs,  ippiMulC_16s_AC4IRSfs
//        ippiMulC_16u_C1IRSfs, ippiMulC_16u_C3IRSfs, ippiMulC_16u_C4IRSfs,  ippiMulC_16u_AC4IRSfs
//
//  Purpose:    Adds, subtracts, or multiplies pixel values of an image and a constant
//              and places the scaled results in the same image.
//
//  Returns:
//    ippStsNoErr              OK
//    ippStsNullPtrErr         Pointer is NULL
//    ippStsSizeErr            Width or height of an image is less than or equal to zero
//
//  Parameters:
//    value                    Constant value (constant vector for multi-channel images)
//    pSrcDst                  Pointer to the image
//    srcDstStep               Step through the image
//    roiSize                  Size of the ROI
//    scaleFactor              Scale factor
*/

IPPAPI(IppStatus, ippiAddC_16s_C1IRSfs,  (Ipp16s value, Ipp16s* pSrcDst, int srcDstStep,
                                          IppiSize roiSize, int scaleFactor))

IPPAPI(IppStatus, ippiAddC_32f_C1IR,  (Ipp32f value, Ipp32f* pSrcDst, int srcDstStep,
                                       IppiSize roiSize))

IPPAPI(IppStatus, ippiAddC_32f_C1R,  (const Ipp32f* pSrc, int srcStep, Ipp32f value, Ipp32f* pDst,
                                      int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiMulC_32f_C1R,  (const Ipp32f* pSrc, int srcStep, Ipp32f value, Ipp32f* pDst,
                                      int dstStep, IppiSize roiSize))


/* ///////////////////////////////////////////////////////////////////////////
//             Filters with Fixed Kernel
/////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////////
//              Kernels:
//
//                                1  1  1
//              PrewittHoriz      0  0  0
//                               -1 -1 -1
//
//
//                               -1  0  1
//              PrewittVert      -1  0  1
//                               -1  0  1
//
//
//                                1  2  1
//              SobelHoriz        0  0  0
//                               -1 -2 -1
//
//
//                               -1  0  1
//              SobelVert        -2  0  2
//                               -1  0  1
//
//
//                                0  0  0
//              RobetsDown        0  1  0
//                                0  0 -1
//
//
//                                0  0  0
//              RobertsUp         0  1  0
//                               -1  0  0
//
//
//                               -1 -1  1
//              Sharpen          -1 16  1  X  1/8
//                               -1 -1  1
//
//
//                                3  0  -3
//              ScharrVert       10  0 -10
//                                3  0  -3
//
//
//                                3  10  3
//              ScharrHoriz       0   0  0
//                               -3 -10 -3
//
//
//                               -1 -1  1
//              Laplace (3x3)    -1  8  1
//                               -1 -1  1
//
//
//                                1  2  1
//              Gauss (3x3)       2  4  2  X  1/16
//                                1  2  1
//
//
//                                1  1  1
//              Lowpass (3x3)     1  1  1  X  1/9
//                                1  1  1
//
//
//                               -1 -1 -1
//              Hipass (3x3 )    -1  8 -1
//                               -1 -1 -1
//
//
//                               -1  0  1
//              SobelVert (3x3)  -2  0  2
//                               -1  0  1
//
//
//                                1  2  1
//              SobelHoriz (3x3)  0  0  0
//                               -1 -2 -1
//
//
//                                       1 -2  1
//              SobelVertSecond (3x3)    2 -4  2
//                                       1 -2  1
//
//
//                                       1  2  1
//              SobelHorizSecond (3x3)  -2 -4 -2
//                                       1  2  1
//
//
//                               -1  0  1
//              SobelCross (3x3)  0  0  0
//                                1  0 -1
//
//
//                               -1 -3 -4 -3 -1
//                               -3  0  6  0 -3
//              Laplace (5x5)    -4  6 20  6 -4
//                               -3  0  6  0 -3
//                               -1 -3 -4 -3 -1
//
//                                2   7  12   7   2
//                                7  31  52  31   7
//              Gauss (5x5)      12  52 127  52  12  X  1/571
//                                7  31  52  31   7
//                                2   7  12   7   2
//
//                                1 1 1 1 1
//                                1 1 1 1 1
//              Lowpass (5x5)     1 1 1 1 1  X  1/25
//                                1 1 1 1 1
//                                1 1 1 1 1
//
//
//                               -1 -1 -1 -1 -1
//                               -1 -1 -1 -1 -1
//              Hipass (5x5)     -1 -1 24 -1 -1
//                               -1 -1 -1 -1 -1
//                               -1 -1 -1 -1 -1
//
//                               -1  -2   0   2   1
//                               -4  -8   0   8   4
//              SobelVert (5x5)  -6 -12   0  12   6
//                               -4  -8   0   8   4
//                               -1  -2   0   2   1
//
//                                1   4   6   4   1
//                                2   8  12   8   2
//              SobelHoriz (5x5)  0   0   0   0   0
//                               -2  -8 -12  -8  -4
//                               -1  -4  -6  -4  -1
//
//                                       1   0  -2   0   1
//                                       4   0  -8   0   4
//              SobelVertSecond (5x5)    6   0 -12   0   6
//                                       4   0  -8   0   4
//                                       1   0  -2   0   1
//
//                                       1   4   6   4   1
//                                       0   0   0   0   0
//              SobelHorizSecond (5x5)  -2  -8 -12  -8  -2
//                                       0   0   0   0   0
//                                       1   4   6   4   1
//
//                               -1  -2   0   2   1
//                               -2  -4   0   4   2
//              SobelCross (5x5)  0   0   0   0   0
//                                2   4   0  -4  -2
//                                1   2   0  -2  -1
//
*/

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiFilterSobelHorizBorderGetBufferSize
//                      ippiFilterSobelVertBorderGetBufferSize
//                      ippiFilterScharrHorizMaskBorderGetBufferSize
//                      ippiFilterScharrVertMaskBorderGetBufferSize
//                      ippiFilterPrewittHorizBorderGetBufferSize
//                      ippiFilterPrewittVertBorderGetBufferSize
//                      ippiFilterRobertsDownBorderGetBufferSize
//                      ippiFilterRobertsUpBorderGetBufferSize
//                      ippiFilterSobelHorizSecondBorderGetBufferSize
//                      ippiFilterSobelVertSecondBorderGetBufferSize
//                      ippiFilterSobelNegVertBorderGetBufferSize
//
//  Purpose:            Computes the size of the external buffer for fixed filter with border
//
//  Parameters:
//   roiSize            Size of destination ROI in pixels.
//   mask               Predefined mask of IppiMaskSize type.
//   srcDataType        Data type of the source image.
//   dstDataType        Data type of the destination image.
//   numChannels        Number of channels in the images.
//   pBufferSize        Pointer to the size (in bytes) of the external work buffer.
//
//  Return Values:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when pBufferSize is NULL.
//   ippStsSizeErr      Indicates an error when roiSize is negative, or equal to zero.
//   ippStsMaskSizeErr  Indicates an error condition if mask has a wrong value.
//   ippStsDataTypeErr  Indicates an error when srcDataType or dstDataType has an illegal value.
//   ippStsNumChannelsErr Indicates an error when numChannels has an illegal value.
*/
IPPAPI(IppStatus, ippiFilterSobelHorizBorderGetBufferSize,(IppiSize dstRoiSize, IppiMaskSize mask,
    IppDataType srcDataType, IppDataType dstDataType, int numChannels, int* pBufferSize))
IPPAPI(IppStatus, ippiFilterSobelVertBorderGetBufferSize,(IppiSize dstRoiSize, IppiMaskSize mask,
    IppDataType srcDataType, IppDataType dstDataType, int numChannels, int* pBufferSize))
IPPAPI(IppStatus,  ippiFilterScharrHorizMaskBorderGetBufferSize,(IppiSize dstRoiSize, IppiMaskSize mask,
    IppDataType srcDataType, IppDataType dstDataType, int numChannels, int* pBufferSize))
IPPAPI(IppStatus,  ippiFilterScharrVertMaskBorderGetBufferSize,(IppiSize dstRoiSize, IppiMaskSize mask,
    IppDataType srcDataType, IppDataType dstDataType, int numChannels, int* pBufferSize))
IPPAPI(IppStatus, ippiFilterSobelHorizSecondBorderGetBufferSize,(IppiSize dstRoiSize, IppiMaskSize mask,
    IppDataType srcDataType, IppDataType dstDataType, int numChannels, int* pBufferSize))
IPPAPI(IppStatus, ippiFilterSobelVertSecondBorderGetBufferSize,(IppiSize dstRoiSize, IppiMaskSize mask,
    IppDataType srcDataType, IppDataType dstDataType, int numChannels, int* pBufferSize))
IPPAPI(IppStatus, ippiFilterSobelNegVertBorderGetBufferSize,(IppiSize dstRoiSize, IppiMaskSize mask,
    IppDataType srcDataType, IppDataType dstDataType, int numChannels, int* pBufferSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiFilterSobelVertBorder_8u16s_C1R
//                      ippiFilterSobelHorizBorder_8u16s_C1R
//                      ippiFilterScharrVertMaskBorder_8u16s_C1R
//                      ippiFilterScharrHorizMaskBorder_8u16s_C1R
//                      ippiFilterPrewittVertBorder_8u16s_C1R
//                      ippiFilterPrewittHorizBorder_8u16s_C1R
//                      ippiFilterRobertsDownBorder_8u16s_C1R
//                      ippiFilterRobertsUpBorder_8u16s_C1R
//                      ippiFilterSobelVertSecondBorder_8u16s_C1R
//                      ippiFilterSobelHorizSecondBorder_8u16s_C1R
//                      ippiFilterSobelNegVertBorder_8u16s_C1R
//                      ippiFilterSobelVertBorder_16s_C1R
//                      ippiFilterSobelHorizBorder_16s_C1R
//                      ippiFilterScharrVertMaskBorder_16s_C1R
//                      ippiFilterScharrHorizMaskBorder_16s_C1R
//                      ippiFilterPrewittVertBorder_16s_C1R
//                      ippiFilterPrewittHorizBorder_16s_C1R
//                      ippiFilterRobertsDownBorder_16s_C1R
//                      ippiFilterRobertsUpBorder_16s_C1R
//                      ippiFilterSobelVertBorder_32f_C1R
//                      ippiFilterSobelHorizBorder_32f_C1R
//                      ippiFilterScharrVertMaskBorder_32f_C1R
//                      ippiFilterScharrHorizMaskBorder_32f_C1R
//                      ippiFilterPrewittVertBorder_32f_C1R
//                      ippiFilterPrewittHorizBorder_32f_C1R
//                      ippiFilterRobertsDownBorder_32f_C1R
//                      ippiFilterRobertsUpBorder_32f_C1R
//                      ippiFilterSobelVertSecondBorder_32f_C1R
//                      ippiFilterSobelHorizSecondBorder_32f_C1R
//                      ippiFilterSobelNegVertBorder_32f_C1R
//                      ippiFilterLaplaceBorder_8u_C1R
//                      ippiFilterLaplaceBorder_8u_C3R
//                      ippiFilterLaplaceBorder_8u_C4R
//                      ippiFilterLaplaceBorder_8u_AC4R
//                      ippiFilterLaplaceBorder_16s_C1R
//                      ippiFilterLaplaceBorder_16s_C3R
//                      ippiFilterLaplaceBorder_16s_C4R
//                      ippiFilterLaplaceBorder_16s_AC4R
//                      ippiFilterLaplaceBorder_32f_C1R
//                      ippiFilterLaplaceBorder_32f_C3R
//                      ippiFilterLaplaceBorder_32f_C4R
//                      ippiFilterLaplaceBorder_32f_AC4R
//                      ippiFilterHipassBorder_8u_C1R
//                      ippiFilterHipassBorder_8u_C3R
//                      ippiFilterHipassBorder_8u_C4R
//                      ippiFilterHipassBorder_8u_AC4R
//                      ippiFilterHipassBorder_16s_C1R
//                      ippiFilterHipassBorder_16s_C3R
//                      ippiFilterHipassBorder_16s_C4R
//                      ippiFilterHipassBorder_16s_AC4R
//                      ippiFilterHipassBorder_32f_C1R
//                      ippiFilterHipassBorder_32f_C3R
//                      ippiFilterHipassBorder_32f_C4R
//                      ippiFilterHipassBorder_32f_AC4R
//                      ippiFilterSharpenBorder_8u_C1R
//                      ippiFilterSharpenBorder_8u_C3R
//                      ippiFilterSharpenBorder_8u_C4R
//                      ippiFilterSharpenBorder_8u_AC4R
//                      ippiFilterSharpenBorder_16s_C1R
//                      ippiFilterSharpenBorder_16s_C3R
//                      ippiFilterSharpenBorder_16s_C4R
//                      ippiFilterSharpenBorder_16s_AC4R
//                      ippiFilterSharpenBorder_32f_C1R
//                      ippiFilterSharpenBorder_32f_C3R
//                      ippiFilterSharpenBorder_32f_C4R
//                      ippiFilterSharpenBorder_32f_AC4R

//
//  Purpose:            Perform linear filtering of an image using one of
//                      predefined convolution kernels.
//
//  Parameters:
//   pSrc               Pointer to the source image ROI.
//   srcStep            Distance in bytes between starting points of consecutive lines in the sorce image.
//   pDst               Pointer to the destination image ROI.
//   dstStep            Distance in bytes between starting points of consecutive lines in the destination image.
//   dstRoiSize         Size of destination ROI in pixels.
//   mask               Predefined mask of IppiMaskSize type.
//   borderType         Type of border.
//   borderValue        Constant value to assign to pixels of the constant border. This parameter is applicable
//                      only to the ippBorderConst border type.
//   pBorderValue       The pointer to constant values to assign to pixels of the constant border. This parameter is applicable
//                      only to the ippBorderConst border type.
//   pBuffer            Pointer to the work buffer.
//
//  Return Values:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when pBufferSize is NULL.
//   ippStsSizeErr      Indicates an error when roiSize is negative, or equal to zero.
//   ippStsNotEvenStepErr Indicated an error when one of the step values is not divisible by 4
//                      for floating-point images, or by 2 for short-integer images.
//   ippStsBorderErr    Indicates an error when borderType has illegal value.
*/
IPPAPI(IppStatus, ippiFilterSobelVertBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelHorizBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterScharrVertMaskBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterScharrHorizMaskBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelVertSecondBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelHorizSecondBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelNegVertBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterSobelVertBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelHorizBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterScharrVertMaskBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterScharrHorizMaskBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelVertSecondBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelHorizSecondBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelNegVertBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterSobelVertBorder_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp16s borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterSobelHorizBorder_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp16s borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterScharrVertMaskBorder_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp16s borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterScharrHorizMaskBorder_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp16s borderValue, Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//                   Statistic functions
///////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:      ippiMomentGetStateSize_64s
//             ippiMomentGetStateSize_64f
//
//  Purpose:   Computes the size of the external buffer for the state
//             structure ippiMomentsState_64s in bytes
//
//  Returns:
//    ippStsNoErr         OK
//    ippStsNullPtrErr    pSize==NULL
//  Parameters:
//    hint                Option to specify the computation algorithm
//    pSize               Pointer to the value of the buffer size
//                        of the structure ippiMomentState_64s.
*/
IPPAPI(IppStatus,  ippiMomentGetStateSize_64f, (IppHintAlgorithm hint, int * pSize))


/* ////////////////////////////////////////////////////////////////////////////////////
//  Name:           ippiMomentInit64s
//                  ippiMomentInit64f
//
//  Purpose:        Initializes ippiMomentState_64s structure (without memory allocation)
//
//  Returns:
//    ippStsNoErr   No errors
//
//  Parameters:
//    pState        Pointer to the MomentState structure
//    hint          Option to specify the computation algorithm
*/
IPPAPI (IppStatus, ippiMomentInit_64f, (IppiMomentState_64f* pState, IppHintAlgorithm hint))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:      ippiMoments
//
//  Purpose:   Computes statistical moments of an image
//
//  Returns:
//    ippStsContextMatchErr   pState->idCtx != idCtxMoment
//    ippStsNullPtrErr        (pSrc == NULL) or (pState == NULL)
//    ippStsStepErr           pSrcStep <0
//    ippStsSizeErr           (roiSize.width  <1) or (roiSize.height <1)
//    ippStsNoErr             No errors
//
//  Parameters:
//    pSrc     Pointer to the source image
//    srcStep  Step in bytes through the source image
//    roiSize  Size of the source ROI
//    pState   Pointer to the MomentState structure
//
//  Notes:
//    These functions compute moments of order 0 to 3 only
//
*/
IPPAPI(IppStatus,ippiMoments64f_8u_C1R, (const Ipp8u* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))
IPPAPI(IppStatus,ippiMoments64f_8u_C3R, (const Ipp8u* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))
IPPAPI(IppStatus,ippiMoments64f_8u_AC4R,(const Ipp8u* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))

IPPAPI(IppStatus,ippiMoments64f_32f_C1R, (const Ipp32f* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))
IPPAPI(IppStatus,ippiMoments64f_32f_C3R, (const Ipp32f* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))
IPPAPI(IppStatus,ippiMoments64f_32f_AC4R,(const Ipp32f* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))

IPPAPI(IppStatus,ippiMoments64f_16u_C1R, (const Ipp16u* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))
IPPAPI(IppStatus,ippiMoments64f_16u_C3R, (const Ipp16u* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))
IPPAPI(IppStatus,ippiMoments64f_16u_AC4R,(const Ipp16u* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:      ippiGetSpatialMoment()
//             ippiGetCentralMoment()
//
//  Purpose:   Retrieves the value of the image spatial/central moment.
//
//  Returns:
//    ippStsNullPtrErr      (pState == NULL) or (pValue == NULL)
//    ippStsContextMatchErr pState->idCtx != idCtxMoment
//    ippStsSizeErr         (mOrd+nOrd) >3 or
//                          (nChannel<0) or (nChannel>=pState->nChannelInUse)
//    ippStsNoErr           No errors
//
//  Parameters:
//    pState      Pointer to the MomentState structure
//    mOrd        m- Order (X direction)
//    nOrd        n- Order (Y direction)
//    nChannel    Channel number
//    roiOffset   Offset of the ROI origin (ippiGetSpatialMoment ONLY!)
//    pValue      Pointer to the retrieved moment value
//    scaleFactor Factor to scale the moment value (for integer data)
//
//  NOTE:
//    ippiGetSpatialMoment uses Absolute Coordinates (left-top image has 0,0).
//
*/
IPPAPI(IppStatus,ippiGetSpatialMoment_64f,(const IppiMomentState_64f* pState,
                                       int mOrd, int nOrd, int nChannel,
                                       IppiPoint roiOffset, Ipp64f* pValue))
IPPAPI(IppStatus,ippiGetCentralMoment_64f,(const IppiMomentState_64f* pState,
                                       int mOrd, int nOrd, int nChannel,
                                       Ipp64f* pValue))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:      ippiGetNormalizedSpatialMoment()
//             ippiGetNormalizedCentralMoment()
//
//  Purpose:   Retrieves the normalized value of the image spatial/central moment.
//
//  Returns:
//    ippStsNullPtrErr      (pState == NULL) or (pValue == NULL)
//    ippStsContextMatchErr pState->idCtx != idCtxMoment
//    ippStsSizeErr         (mOrd+nOrd) >3 or
//                          (nChannel<0) or (nChannel>=pState->nChannelInUse)
//    ippStsMoment00ZeroErr mm[0][0] < IPP_EPS52
//    ippStsNoErr           No errors
//
//  Parameters:
//    pState      Pointer to the MomentState structure
//    mOrd        m- Order (X direction)
//    nOrd        n- Order (Y direction)
//    nChannel    Channel number
//    roiOffset   Offset of the ROI origin (ippiGetSpatialMoment ONLY!)
//    pValue      Pointer to the normalized moment value
//    scaleFactor Factor to scale the moment value (for integer data)
//
*/
IPPAPI(IppStatus,ippiGetNormalizedSpatialMoment_64f,(const IppiMomentState_64f* pState,
                                   int mOrd, int nOrd, int nChannel,
                                   IppiPoint roiOffset, Ipp64f* pValue))
IPPAPI(IppStatus,ippiGetNormalizedCentralMoment_64f,(const IppiMomentState_64f* pState,
                                   int mOrd, int nOrd, int nChannel,
                                   Ipp64f* pValue))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:      ippiGetHuMoments()
//
//  Purpose:   Retrieves image Hu moment invariants.
//
//  Returns:
//    ippStsNullPtrErr      (pState == NULL) or (pHu == NULL)
//    ippStsContextMatchErr pState->idCtx != idCtxMoment
//    ippStsSizeErr         (nChannel<0) or (nChannel>=pState->nChannelInUse)
//    ippStsMoment00ZeroErr mm[0][0] < IPP_EPS52
//    ippStsNoErr           No errors
//
//  Parameters:
//    pState      Pointer to the MomentState structure
//    nChannel    Channel number
//    pHm         Pointer to the array of the Hu moment invariants
//    scaleFactor Factor to scale the moment value (for integer data)
//
//  Notes:
//    We consider Hu moments up to the 7-th order only
*/
IPPAPI(IppStatus,ippiGetHuMoments_64f,(const IppiMomentState_64f* pState,
                                   int nChannel, IppiHuMoment_64f pHm))


/* /////////////////////////////////////////////////////////////////////////////
//
//  Name:        ippiTranspose
//
//  Purpose:     Transposing an image
//
//  Returns:     IppStatus
//    ippStsNoErr         No errors
//    ippStsNullPtrErr    pSrc == NULL, or pDst == NULL
//    ippStsSizeErr       roiSize has a field with zero or negative value,
//                        and roiSize.width != roiSize.height (in-place flavors)
//
//  Parameters:
//    pSrc       Pointer to the source image
//    srcStep    Step through the source image
//    pDst       Pointer to the destination image
//    dstStep    Step through the destination image
//    pSrcDst    Pointer to the source/destination image (in-place flavors)
//    srcDstStep Step through the source/destination image (in-place flavors)
//    roiSize    Size of the ROI
//
//  Notes: Parameters roiSize.width and roiSize.height are defined for the source image.
//
*/

IPPAPI ( IppStatus, ippiTranspose_8u_C1R,
                    ( const Ipp8u* pSrc, int srcStep,
                            Ipp8u* pDst, int dstStep,
                                    IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_8u_C3R,
                    ( const Ipp8u* pSrc, int srcStep,
                            Ipp8u* pDst, int dstStep,
                                    IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_8u_C4R,
                    ( const Ipp8u* pSrc, int srcStep,
                            Ipp8u* pDst, int dstStep,
                                    IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_8u_C1IR,
                    ( Ipp8u* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_8u_C3IR,
                    ( Ipp8u* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_8u_C4IR,
                    ( Ipp8u* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16u_C1R,
                    ( const Ipp16u* pSrc, int srcStep,
                            Ipp16u* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16u_C3R,
                    ( const Ipp16u* pSrc, int srcStep,
                            Ipp16u* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16u_C4R,
                    ( const Ipp16u* pSrc, int srcStep,
                            Ipp16u* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16u_C1IR,
                    ( Ipp16u* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16u_C3IR,
                    ( Ipp16u* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16u_C4IR,
                    ( Ipp16u* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32s_C1R,
                    ( const Ipp32s* pSrc, int srcStep,
                            Ipp32s* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32s_C3R,
                    ( const Ipp32s* pSrc, int srcStep,
                            Ipp32s* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32s_C4R,
                    ( const Ipp32s* pSrc, int srcStep,
                            Ipp32s* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32s_C1IR,
                    ( Ipp32s* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32s_C3IR,
                    ( Ipp32s* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32s_C4IR,
                    ( Ipp32s* pSrcDst, int srcDstStep, IppiSize roiSize ))

IPPAPI ( IppStatus, ippiTranspose_16s_C1R,
                    ( const Ipp16s* pSrc, int srcStep,
                            Ipp16s* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16s_C3R,
                    ( const Ipp16s* pSrc, int srcStep,
                            Ipp16s* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16s_C4R,
                    ( const Ipp16s* pSrc, int srcStep,
                            Ipp16s* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16s_C1IR,
                    ( Ipp16s* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16s_C3IR,
                    ( Ipp16s* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_16s_C4IR,
                    ( Ipp16s* pSrcDst, int srcDstStep, IppiSize roiSize ))


IPPAPI ( IppStatus, ippiTranspose_32f_C1R,
                    ( const Ipp32f* pSrc, int srcStep,
                            Ipp32f* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32f_C3R,
                    ( const Ipp32f* pSrc, int srcStep,
                            Ipp32f* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32f_C4R,
                    ( const Ipp32f* pSrc, int srcStep,
                            Ipp32f* pDst, int dstStep,
                                     IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32f_C1IR,
                    ( Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32f_C3IR,
                    ( Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiTranspose_32f_C4IR,
                    ( Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize ))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiConvert
//
//  Purpose:    Converts pixel values of an image from one bit depth to another
//
//  Returns:
//    ippStsNullPtrErr      One of the pointers is NULL
//    ippStsSizeErr         roiSize has a field with zero or negative value
//    ippStsStepErr         srcStep or dstStep has zero or negative value
//    ippStsNoErr           OK
//
//  Parameters:
//    pSrc                  Pointer  to the source image
//    srcStep               Step through the source image
//    pDst                  Pointer to the  destination image
//    dstStep               Step in bytes through the destination image
//    roiSize               Size of the ROI
//    roundMode             Rounding mode, ippRndZero or ippRndNear
*/
IPPAPI ( IppStatus, ippiConvert_8u8s_C1RSfs, (const Ipp8u*  pSrc, int srcStep, Ipp8s*  pDst, int dstStep, IppiSize roi,IppRoundMode rndMode, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_8u16u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_8u16s_C1R, (const Ipp8u* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_8u32s_C1R, (const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_8u32f_C1R, (const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize ))

IPPAPI ( IppStatus, ippiConvert_8s8u_C1Rs, (const Ipp8s*  pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize roi))
IPPAPI ( IppStatus, ippiConvert_8s16s_C1R, (const Ipp8s*  pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize roi))
IPPAPI ( IppStatus, ippiConvert_8s16u_C1Rs, (const Ipp8s*  pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roi))
IPPAPI ( IppStatus, ippiConvert_8s32s_C1R, (const Ipp8s* pSrc, int srcStep, Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_8s32f_C1R, (const Ipp8s* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize ))

IPPAPI ( IppStatus, ippiConvert_16u8u_C1R, ( const Ipp16u* pSrc, int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_16u8s_C1RSfs, (const Ipp16u* pSrc, int srcStep, Ipp8s*  pDst, int dstStep, IppiSize roi,IppRoundMode rndMode, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_16u16s_C1RSfs, (const Ipp16u* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize roi,IppRoundMode rndMode, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_16u32s_C1R, (const Ipp16u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_16u32f_C1R, (const Ipp16u* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize ))

IPPAPI ( IppStatus, ippiConvert_16s8s_C1RSfs, (const Ipp16s* pSrc, int srcStep, Ipp8s*  pDst, int dstStep, IppiSize roi,IppRoundMode rndMode, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_16s8u_C1R, ( const Ipp16s* pSrc, int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_16s16u_C1Rs, (const Ipp16s* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roi))
IPPAPI ( IppStatus, ippiConvert_16s32s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_16s32f_C1R, (const Ipp16s* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize ))

IPPAPI ( IppStatus, ippiConvert_32s8u_C1R, ( const Ipp32s* pSrc, int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_32s8s_C1R, ( const Ipp32s* pSrc, int srcStep, Ipp8s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiConvert_32s16u_C1RSfs, (const Ipp32s* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roi,IppRoundMode rndMode, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_32s16s_C1RSfs, (const Ipp32s* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize roi,IppRoundMode rndMode, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_32s32f_C1R, (const Ipp32s* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roi))

IPPAPI ( IppStatus, ippiConvert_32f8u_C1R, ( const Ipp32f* pSrc, int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize, IppRoundMode roundMode ))
IPPAPI ( IppStatus, ippiConvert_32f8u_C1RSfs, (const Ipp32f* pSrc, int srcStep, Ipp8u*  pDst, int dstStep, IppiSize roi, IppRoundMode round, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_32f8s_C1R, ( const Ipp32f* pSrc, int srcStep, Ipp8s* pDst, int dstStep, IppiSize roiSize, IppRoundMode roundMode ))
IPPAPI ( IppStatus, ippiConvert_32f8s_C1RSfs, (const Ipp32f* pSrc, int srcStep, Ipp8s*  pDst, int dstStep, IppiSize roi, IppRoundMode round, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_32f16u_C1R, ( const Ipp32f* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize, IppRoundMode roundMode ))
IPPAPI ( IppStatus, ippiConvert_32f16u_C1RSfs, (const Ipp32f* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roi, IppRoundMode round, int scaleFactor))
IPPAPI ( IppStatus, ippiConvert_32f16s_C1R, ( const Ipp32f* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize roiSize, IppRoundMode roundMode ))
IPPAPI ( IppStatus, ippiConvert_32f32s_C1RSfs, (const Ipp32f* pSrc, int srcStep, Ipp32s* pDst, int dstStep, IppiSize roi, IppRoundMode round, int scaleFactor))


/* /////////////////////////////////////////////////////////////////////////////
//
//  Name:        ippiMirror
//
//  Purpose:     Mirrors an image about a horizontal
//               or vertical axis, or both
//
//  Context:
//
//  Returns:     IppStatus
//    ippStsNoErr         No errors
//    ippStsNullPtrErr    pSrc == NULL, or pDst == NULL
//    ippStsSizeErr,      roiSize has a field with zero or negative value
//    ippStsMirrorFlipErr (flip != ippAxsHorizontal) &&
//                        (flip != ippAxsVertical) &&
//                        (flip != ippAxsBoth)
//
//  Parameters:
//    pSrc       Pointer to the source image
//    srcStep    Step through the source image
//    pDst       Pointer to the destination image
//    dstStep    Step through the destination image
//    pSrcDst    Pointer to the source/destination image (in-place flavors)
//    srcDstStep Step through the source/destination image (in-place flavors)
//    roiSize    Size of the ROI
//    flip       Specifies the axis to mirror the image about:
//                 ippAxsHorizontal     horizontal axis,
//                 ippAxsVertical       vertical axis,
//                 ippAxsBoth           both horizontal and vertical axes
//
//  Notes:
//
*/

IPPAPI(IppStatus, ippiMirror_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                      IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                      IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
                                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_8u_C1IR, (Ipp8u* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_8u_C3IR, (Ipp8u* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_8u_C4IR, (Ipp8u* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))

IPPAPI(IppStatus, ippiMirror_16u_C1R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16u_C3R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16u_C4R, (const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
                                                        IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16u_C1IR, (Ipp16u* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16u_C3IR, (Ipp16u* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI (IppStatus, ippiMirror_16u_C4IR, (Ipp16u* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))

IPPAPI(IppStatus, ippiMirror_32s_C1R, (const Ipp32s* pSrc, int srcStep, Ipp32s* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32s_C3R, (const Ipp32s* pSrc, int srcStep, Ipp32s* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32s_C4R, (const Ipp32s* pSrc, int srcStep, Ipp32s* pDst, int dstStep,
                                                        IppiSize roiSize, IppiAxis flip ) )

IPPAPI(IppStatus, ippiMirror_32s_C1IR, (Ipp32s* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32s_C3IR, (Ipp32s* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32s_C4IR, (Ipp32s* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))

IPPAPI(IppStatus, ippiMirror_16s_C1R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16s_C3R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16s_C4R, (const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,
                                                        IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16s_C1IR, (Ipp16s* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_16s_C3IR, (Ipp16s* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI (IppStatus, ippiMirror_16s_C4IR, (Ipp16s* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))

IPPAPI(IppStatus, ippiMirror_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                       IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32f_C4R, (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
                                                        IppiSize roiSize, IppiAxis flip ) )
IPPAPI(IppStatus, ippiMirror_32f_C1IR, (Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32f_C3IR, (Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))
IPPAPI(IppStatus, ippiMirror_32f_C4IR, (Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip))



/* ///////////////////////////////////////////////////////////////////////////////////////
//                  Arithmetic Functions
///////////////////////////////////////////////////////////////////////////// */
/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:  ippiAdd_8u_C1RSfs,  ippiAdd_8u_C3RSfs,  ippiAdd_8u_C4RSfs,  ippiAdd_8u_AC4RSfs,
//         ippiAdd_16s_C1RSfs, ippiAdd_16s_C3RSfs, ippiAdd_16s_C4RSfs, ippiAdd_16s_AC4RSfs,
//         ippiAdd_16u_C1RSfs, ippiAdd_16u_C3RSfs, ippiAdd_16u_C4RSfs, ippiAdd_16u_AC4RSfs,
//         ippiSub_8u_C1RSfs,  ippiSub_8u_C3RSfs,  ippiSub_8u_C4RSfs,  ippiSub_8u_AC4RSfs,
//         ippiSub_16s_C1RSfs, ippiSub_16s_C3RSfs, ippiSub_16s_C4RSfs, ippiSub_16s_AC4RSfs,
//         ippiSub_16u_C1RSfs, ippiSub_16u_C3RSfs, ippiSub_16u_C4RSfs, ippiSub_16u_AC4RSfs,
//         ippiMul_8u_C1RSfs,  ippiMul_8u_C3RSfs,  ippiMul_8u_C4RSfs,  ippiMul_8u_AC4RSfs,
//         ippiMul_16s_C1RSfs, ippiMul_16s_C3RSfs, ippiMul_16s_C4RSfs, ippiMul_16s_AC4RSfs
//         ippiMul_16u_C1RSfs, ippiMul_16u_C3RSfs, ippiMul_16u_C4RSfs, ippiMul_16u_AC4RSfs
//
//  Purpose:    Adds, subtracts, or multiplies pixel values of two
//              source images and places the scaled result in the destination image.
//
//  Returns:
//    ippStsNoErr              OK
//    ippStsNullPtrErr         One of the pointers is NULL
//    ippStsSizeErr            Width or height of images is less than or equal to zero
//
//  Parameters:
//    pSrc1, pSrc2             Pointers to the source images
//    src1Step, src2Step       Steps through the source images
//    pDst                     Pointer to the destination image
//    dstStep                  Step through the destination image
//    roiSize                  Size of the ROI
//    scaleFactor              Scale factor
*/

IPPAPI(IppStatus, ippiAdd_8u_C1RSfs,   (const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2,
                                        int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiAdd_16u_C1RSfs,  (const Ipp16u* pSrc1, int src1Step, const Ipp16u* pSrc2,
                                        int src2Step, Ipp16u* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiAdd_16s_C1RSfs,  (const Ipp16s* pSrc1, int src1Step, const Ipp16s* pSrc2,
                                        int src2Step, Ipp16s* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiSub_8u_C1RSfs,   (const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2,
                                        int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiSub_16u_C1RSfs,  (const Ipp16u* pSrc1, int src1Step, const Ipp16u* pSrc2,
                                        int src2Step, Ipp16u* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiSub_16s_C1RSfs,  (const Ipp16s* pSrc1, int src1Step, const Ipp16s* pSrc2,
                                        int src2Step, Ipp16s* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiMul_16s_C1RSfs,  (const Ipp16s* pSrc1, int src1Step, const Ipp16s* pSrc2,
                                        int src2Step, Ipp16s* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiMul_16u_C1RSfs,  (const Ipp16u* pSrc1, int src1Step, const Ipp16u* pSrc2,
                                        int src2Step, Ipp16u* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))

IPPAPI(IppStatus, ippiMul_8u_C1RSfs,   (const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2,
                                        int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                        int scaleFactor))


/* ////////////////////////////////////////////////////////////////////////////
//  Name: ippiAdd_32f_C1R, ippiAdd_32f_C3R, ippiAdd_32f_C4R, ippiAdd_32f_AC4R,
//        ippiSub_32f_C1R, ippiSub_32f_C3R, ippiSub_32f_C4R, ippiSub_32f_AC4R,
//        ippiMul_32f_C1R, ippiMul_32f_C3R, ippiMul_32f_C4R, ippiMul_32f_AC4R
//
//  Purpose:    Adds, subtracts, or multiplies pixel values of two
//              source images and places the results in a destination image.
//
//  Returns:
//    ippStsNoErr            OK
//    ippStsNullPtrErr       One of the pointers is NULL
//    ippStsSizeErr          Width or height of images is less than or equal to zero
//
//  Parameters:
//    pSrc1, pSrc2           Pointers to the source images
//    src1Step, src2Step     Steps through the source images
//    pDst                   Pointer to the destination image
//    dstStep                Step through the destination image
//    roiSize                Size of the ROI
*/

IPPAPI(IppStatus, ippiAdd_32f_C1R,  (const Ipp32f* pSrc1, int src1Step, const Ipp32f* pSrc2,
                                     int src2Step, Ipp32f* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiSub_32f_C1R,  (const Ipp32f* pSrc1, int src1Step, const Ipp32f* pSrc2,
                                     int src2Step, Ipp32f* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiMul_32f_C1R,  (const Ipp32f* pSrc1, int src1Step, const Ipp32f* pSrc2,
                                     int src2Step, Ipp32f* pDst, int dstStep, IppiSize roiSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiMax
//  Purpose:        computes the maximum of image pixel values
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      roiSize has a field with zero or negative value
//  Parameters:
//    pSrc        Pointer to the source image.
//    srcStep     Step in bytes through the source image
//    roiSize     Size of the source image ROI.
//    pMax        Pointer to the result (C1)
//    max         Array containing the results (C3, AC4, C4)
*/

IPPAPI(IppStatus, ippiMax_16s_C1R, (const Ipp16s* pSrc, int srcStep, IppiSize roiSize, Ipp16s* pMax))

IPPAPI(IppStatus, ippiMax_16u_C1R, (const Ipp16u* pSrc, int srcStep, IppiSize roiSize, Ipp16u* pMax))

IPPAPI(IppStatus, ippiMax_32f_C1R, (const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMax))

IPPAPI(IppStatus, ippiMax_8u_C1R, (const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMax))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiMin
//  Purpose:        computes the minimum of image pixel values
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      roiSize has a field with zero or negative value
//  Parameters:
//    pSrc        Pointer to the source image.
//    srcStep     Step through the source image
//    roiSize     Size of the source image ROI.
//    pMin        Pointer to the result (C1)
//    min         Array containing results (C3, AC4, C4)
*/

IPPAPI(IppStatus, ippiMin_16s_C1R, (const Ipp16s* pSrc, int srcStep, IppiSize roiSize, Ipp16s* pMin))

IPPAPI(IppStatus, ippiMin_16u_C1R, (const Ipp16u* pSrc, int srcStep, IppiSize roiSize, Ipp16u* pMin))

IPPAPI(IppStatus, ippiMin_32f_C1R, (const Ipp32f* pSrc, int srcStep, IppiSize roiSize, Ipp32f* pMin))

IPPAPI(IppStatus, ippiMin_8u_C1R, (const Ipp8u* pSrc, int srcStep, IppiSize roiSize, Ipp8u* pMin))


/* /////////////////////////////////////////////////////////////////////////////////////////////////
//                      Logical Operations and Shift Functions
///////////////////////////////////////////////////////////////////////////////////////////////// */
/*
//  Names:          ippiAnd, ippiAndC, ippiOr, ippiOrC, ippiXor, ippiXorC, ippiNot,
//  Purpose:        Performs corresponding bitwise logical operation between pixels of two image
//                  (AndC/OrC/XorC  - between pixel of the source image and a constant)
//
//  Names:          ippiLShiftC, ippiRShiftC
//  Purpose:        Shifts bits in each pixel value to the left and right
//  Parameters:
//   value         1) The constant value to be ANDed/ORed/XORed with each pixel of the source,
//                     constant vector for multi-channel images;
//                 2) The number of bits to shift, constant vector for multi-channel images.
//   pSrc          Pointer to the source image
//   srcStep       Step through the source image
//   pSrcDst       Pointer to the source/destination image (in-place flavors)
//   srcDstStep    Step through the source/destination image (in-place flavors)
//   pSrc1         Pointer to first source image
//   src1Step      Step through first source image
//   pSrc2         Pointer to second source image
//   src2Step      Step through second source image
//   pDst          Pointer to the destination image
//   dstStep       Step in destination image
//   roiSize       Size of the ROI
//
//  Returns:
//   ippStsNullPtrErr   One of the pointers is NULL
//   ippStsStepErr      One of the step values is less than or equal to zero
//   ippStsSizeErr      roiSize has a field with zero or negative value
//   ippStsShiftErr     Shift's value is less than zero
//   ippStsNoErr        No errors
*/

IPPAPI(IppStatus, ippiAnd_8u_C1R, (const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2, int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiOr_8u_C1R, (const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2, int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiXor_8u_C1R, (const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2, int src2Step, Ipp8u* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiNot_8u_C1R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize))


/* /////////////////////////////////////////////////////////////////////////////////////////////////
//                              Compare Operations
///////////////////////////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiCompare
//                  ippiCompareC
//  Purpose:  Compares pixel values of two images, or pixel values of an image to a constant
//            value using the following compare conditions: <, <=, ==, >, >= ;
//  Names:          ippiCompareEqualEps
//                  ippiCompareEqualEpsC
//  Purpose:  Compares 32f images for being equal, or equal to a given value within given tolerance
//  Context:
//
//  Returns:        IppStatus
//    ippStsNoErr        No errors
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsStepErr      One of the step values is less than or equal to zero
//    ippStsSizeErr      roiSize has a field with zero or negative value
//    ippStsEpsValErr    eps is negative
//
//  Parameters:
//    pSrc1         Pointer to the first source image;
//    src1Step      Step through the first source image;
//    pSrc2         Pointer to the second source image data;
//    src2Step      Step through the second source image;
//    pDst          Pointer to destination image data;
//    dstStep       Step in destination image;
//    roiSize       Size of the ROI;
//    ippCmpOp      Compare operation to be used
//    value         Value (array of values for multi-channel image) to compare
//                  each pixel to
//    eps           The tolerance value
//
//  Notes:
*/

IPPAPI (IppStatus, ippiCompare_8u_C1R, ( const Ipp8u* pSrc1, int src1Step,
                                         const Ipp8u* pSrc2, int src2Step,
                                               Ipp8u* pDst,  int dstStep,
                                         IppiSize roiSize,   IppCmpOp ippCmpOp))

IPPAPI (IppStatus, ippiCompare_16u_C1R, ( const Ipp16u* pSrc1, int src1Step,
                                          const Ipp16u* pSrc2, int src2Step,
                                                Ipp8u*  pDst,  int dstStep,
                                          IppiSize roiSize,    IppCmpOp ippCmpOp))

IPPAPI (IppStatus, ippiCompare_16s_C1R, ( const Ipp16s* pSrc1, int src1Step,
                                          const Ipp16s* pSrc2, int src2Step,
                                                Ipp8u*  pDst,  int dstStep,
                                          IppiSize roiSize,    IppCmpOp ippCmpOp))

IPPAPI (IppStatus, ippiCompare_32f_C1R, ( const Ipp32f* pSrc1, int src1Step,
                                          const Ipp32f* pSrc2, int src2Step,
                                                 Ipp8u*  pDst, int dstStep,
                                          IppiSize roiSize,    IppCmpOp ippCmpOp))


/* /////////////////////////////////////////////////////////////////////////////
//              Dot product of two images
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiDotProd
//  Purpose:        Computes the dot product of two images
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsStepErr      One of the step values is equal to zero
//  Parameters:
//    pSrc1       Pointer to the first source image.
//    src1Step    Step in bytes through the first source image
//    pSrc2       Pointer to the second source image.
//    src2Step    Step in bytes through the  source image
//    roiSize     Size of the source image ROI.
//    pDp         Pointer to the result (one-channel data) or array (multi-channel data) containing computed dot products of channel values of pixels in the source images.
//    hint        Option to select the algorithmic implementation of the function
//  Notes:
*/

IPPAPI(IppStatus, ippiDotProd_8u64f_C1R,(const Ipp8u* pSrc1, int src1Step, const Ipp8u* pSrc2, int src2Step, IppiSize roiSize, Ipp64f *pDp))

IPPAPI(IppStatus, ippiDotProd_16u64f_C1R,(const Ipp16u* pSrc1, int src1Step, const Ipp16u* pSrc2, int src2Step, IppiSize roiSize, Ipp64f *pDp))

IPPAPI(IppStatus, ippiDotProd_16s64f_C1R,(const Ipp16s* pSrc1, int src1Step, const Ipp16s* pSrc2, int src2Step, IppiSize roiSize, Ipp64f *pDp))

IPPAPI(IppStatus, ippiDotProd_32s64f_C1R,(const Ipp32s* pSrc1, int src1Step, const Ipp32s* pSrc2, int src2Step, IppiSize roiSize, Ipp64f *pDp))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiSum
//  Purpose:        computes the sum of image pixel values
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      roiSize has a field with zero or negative value
//  Parameters:
//    pSrc        Pointer to the source image.
//    srcStep     Step in bytes through the source image
//    roiSize     Size of the source image ROI.
//    pSum        Pointer to the result (one-channel data)
//    sum         Array containing the results (multi-channel data)
//    hint        Option to select the algorithmic implementation of the function
//  Notes:
*/

IPPAPI(IppStatus, ippiSum_8u_C1R, (const Ipp8u* pSrc, int srcStep,
                                   IppiSize roiSize, Ipp64f* pSum))

IPPAPI(IppStatus, ippiSum_8u_C3R, (const Ipp8u* pSrc, int srcStep,
                                   IppiSize roiSize, Ipp64f sum[3]))

IPPAPI(IppStatus, ippiSum_8u_C4R, (const Ipp8u* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f sum[4]))


IPPAPI(IppStatus, ippiSum_16u_C1R, (const Ipp16u* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f* pSum))

IPPAPI(IppStatus, ippiSum_16u_C3R, (const Ipp16u* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f sum[3]))

IPPAPI(IppStatus, ippiSum_16u_C4R, (const Ipp16u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f sum[4]))


IPPAPI(IppStatus, ippiSum_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f* pSum))

IPPAPI(IppStatus, ippiSum_16s_C3R, (const Ipp16s* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f sum[3]))

IPPAPI(IppStatus, ippiSum_16s_C4R, (const Ipp16s* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f sum[4]))


IPPAPI(IppStatus, ippiSum_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f* pSum, IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiSum_32f_C3R, (const Ipp32f* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f sum[3], IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiSum_32f_C4R, (const Ipp32f* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f sum[4], IppHintAlgorithm hint))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:      ippiSqr_8u_C1RSfs
//              ippiSqr_8u_C3RSfs
//              ippiSqr_8u_AC4RSfs
//              ippiSqr_8u_C4RSfs
//              ippiSqr_16u_C1RSfs
//              ippiSqr_16u_C3RSfs
//              ippiSqr_16u_AC4RSfs
//              ippiSqr_16u_C4RSfs
//              ippiSqr_16s_C1RSfs
//              ippiSqr_16s_C3RSfs
//              ippiSqr_16s_AC4RSfs
//              ippiSqr_16s_C4RSfs
//              ippiSqr_32f_C1R
//              ippiSqr_32f_C3R
//              ippiSqr_32f_AC4R
//              ippiSqr_32f_C4R
//
//              ippiSqr_8u_C1IRSfs
//              ippiSqr_8u_C3IRSfs
//              ippiSqr_8u_AC4IRSfs
//              ippiSqr_8u_C4IRSfs
//              ippiSqr_16u_C1IRSfs
//              ippiSqr_16u_C3IRSfs
//              ippiSqr_16u_AC4IRSfs
//              ippiSqr_16u_C4IRSfs
//              ippiSqr_16s_C1IRSfs
//              ippiSqr_16s_C3IRSfs
//              ippiSqr_16s_AC4IRSfs
//              ippiSqr_16s_C4IRSfs
//              ippiSqr_32f_C1IR
//              ippiSqr_32f_C3IR
//              ippiSqr_32f_AC4IR
//              ippiSqr_32f_C4IR
//
//  Purpose:    squares pixel values of an image and
//              places results in the destination image;
//              for in-place flavors - in  the same image
//  Returns:
//   ippStsNoErr       OK
//   ippStsNullPtrErr  One of the pointers is NULL
//   ippStsSizeErr     The roiSize has a field with negative or zero value
//
//  Parameters:
//   pSrc       pointer to the source image
//   srcStep    step through the source image
//   pDst       pointer to the destination image
//   dstStep    step through the destination image
//   pSrcDst    pointer to the source/destination image (for in-place function)
//   srcDstStep step through the source/destination image (for in-place function)
//   roiSize    size of the ROI
//   scaleFactor scale factor
*/

IPPAPI(IppStatus,ippiSqr_32f_C1R, (const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiMean
//  Purpose:        computes the mean of image pixel values
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      roiSize has a field with zero or negative value.
//  Parameters:
//    pSrc        Pointer to the source image.
//    srcStep     Step in bytes through the source image
//    roiSize     Size of the source ROI.
//    pMean       Pointer to the result (one-channel data)
//    mean        Array containing the results (multi-channel data)
//    hint        Option to select the algorithmic implementation of the function
//  Notes:
*/

IPPAPI(IppStatus, ippiMean_8u_C1R, (const Ipp8u* pSrc, int srcStep,
                                   IppiSize roiSize, Ipp64f* pMean))

IPPAPI(IppStatus, ippiMean_8u_C3R, (const Ipp8u* pSrc, int srcStep,
                                   IppiSize roiSize, Ipp64f mean[3]))

IPPAPI(IppStatus, ippiMean_8u_C4R, (const Ipp8u* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f mean[4]))

IPPAPI(IppStatus, ippiMean_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f* pMean))

IPPAPI(IppStatus, ippiMean_16s_C3R, (const Ipp16s* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f mean[3]))

IPPAPI(IppStatus, ippiMean_16s_C4R, (const Ipp16s* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f mean[4]))

IPPAPI(IppStatus, ippiMean_16u_C1R, (const Ipp16u* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f* pMean))

IPPAPI(IppStatus, ippiMean_16u_C3R, (const Ipp16u* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f mean[3]))

IPPAPI(IppStatus, ippiMean_16u_C4R, (const Ipp16u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f mean[4]))

IPPAPI(IppStatus, ippiMean_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f* pMean, IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiMean_32f_C3R, (const Ipp32f* pSrc, int srcStep,
                                    IppiSize roiSize, Ipp64f mean[3], IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiMean_32f_C4R, (const Ipp32f* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f mean[4], IppHintAlgorithm hint))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNorm_Inf
//  Purpose:        computes the C-norm of pixel values of the image: n = MAX |src1|
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      roiSize has a field with zero or negative value
//  Parameters:
//    pSrc        Pointer to the source image.
//    srcStep     Step through the source image
//    roiSize     Size of the source ROI.
//    pValue      Pointer to the computed norm (one-channel data)
//    value       Array of the computed norms for each channel (multi-channel data)
//  Notes:
*/

IPPAPI(IppStatus, ippiNorm_Inf_8u_C1R, (const Ipp8u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_Inf_8u_C3R, (const Ipp8u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_Inf_8u_C4R, (const Ipp8u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_Inf_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_Inf_16s_C3R, (const Ipp16s* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_Inf_16s_C4R, (const Ipp16s* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_Inf_16u_C1R, (const Ipp16u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_Inf_16u_C3R, (const Ipp16u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_Inf_16u_C4R, (const Ipp16u* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_Inf_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_Inf_32f_C3R, (const Ipp32f* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_Inf_32f_C4R, (const Ipp32f* pSrc, int srcStep,
                                     IppiSize roiSize, Ipp64f value[4]))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNorm_L1
//  Purpose:        computes the L1-norm of pixel values of the image: n = SUM |src1|
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      roiSize has a field with zero or negative value
//  Parameters:
//    pSrc        Pointer to the source image.
//    srcStep     Step through the source image
//    roiSize     Size of the source ROI.
//    pValue      Pointer to the computed norm (one-channel data)
//    value       Array of the computed norms for each channel (multi-channel data)
//    hint        Option to specify the computation algorithm
//  Notes:
*/

IPPAPI(IppStatus, ippiNorm_L1_8u_C1R, (const Ipp8u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_L1_8u_C3R, (const Ipp8u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_L1_8u_C4R, (const Ipp8u* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_L1_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_L1_16s_C3R, (const Ipp16s* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_L1_16s_C4R, (const Ipp16s* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_L1_16u_C1R, (const Ipp16u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_L1_16u_C3R, (const Ipp16u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_L1_16u_C4R, (const Ipp16u* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_L1_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue, IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNorm_L1_32f_C3R, (const Ipp32f* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3], IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNorm_L1_32f_C4R, (const Ipp32f* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4], IppHintAlgorithm hint))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNorm_L2
//  Purpose:        computes the L2-norm of pixel values of the image: n = SQRT(SUM |src1|^2)
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr        OK
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      roiSize has a field with zero or negative value
//  Parameters:
//    pSrc        Pointer to the source image.
//    srcStep     Step through the source image
//    roiSize     Size of the source ROI.
//    pValue      Pointer to the computed norm (one-channel data)
//    value       Array of the computed norms for each channel (multi-channel data)
//    hint        Option to specify the computation algorithm
//  Notes:
//    simple mul is better than table for P6 family
*/

IPPAPI(IppStatus, ippiNorm_L2_8u_C1R, (const Ipp8u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_L2_8u_C3R, (const Ipp8u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_L2_8u_C4R, (const Ipp8u* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_L2_16s_C1R, (const Ipp16s* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_L2_16s_C3R, (const Ipp16s* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_L2_16s_C4R, (const Ipp16s* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_L2_16u_C1R, (const Ipp16u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNorm_L2_16u_C3R, (const Ipp16u* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNorm_L2_16u_C4R, (const Ipp16u* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNorm_L2_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f* pValue, IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNorm_L2_32f_C3R, (const Ipp32f* pSrc, int srcStep,
                                      IppiSize roiSize, Ipp64f value[3], IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNorm_L2_32f_C4R, (const Ipp32f* pSrc, int srcStep,
                                       IppiSize roiSize, Ipp64f value[4], IppHintAlgorithm hint))



/* //////////////////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNormRel_Inf
//  Purpose:        computes the relative error for the C-norm of pixel values of two images:
//                      n = MAX |src1 - src2| / MAX |src2|
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr         OK
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       roiSize has a field with zero or negative value
//    ippStsDivByZero     MAX |src2| == 0
//  Parameters:
//    pSrc1, pSrc2        Pointers to the source images.
//    src1Step, src2Step  Steps in bytes through the source images
//    roiSize             Size of the source ROI.
//    pValue              Pointer to the computed norm (one-channel data)
//    value               Array of the computed norms for each channel (multi-channel data)
//  Notes:
*/

IPPAPI(IppStatus, ippiNormRel_Inf_8u_C1R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_Inf_16u_C1R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_Inf_16s_C1R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_Inf_32f_C1R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))


/* /////////////////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNormRel_L1
//  Purpose:        computes the relative error for the 1-norm of pixel values of two images:
//                      n = SUM |src1 - src2| / SUM |src2|
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr         OK
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       roiSize has a field with zero or negative value
//    ippStsDivByZero     SUM |src2| == 0
//  Parameters:
//    pSrc1, pSrc2        Pointers to the source images.
//    src1Step, src2Step  Steps in bytes through the source images
//    roiSize             Size of the source ROI.
//    pValue              Pointer to the computed norm (one-channel data)
//    value               Array of the computed norms for each channel (multi-channel data)
//    hint                Option to specify the computation algorithm
//  Notes:
*/

IPPAPI(IppStatus, ippiNormRel_L1_8u_C1R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_L1_16s_C1R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_L1_16u_C1R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_L1_32f_C1R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue, IppHintAlgorithm hint))



/* //////////////////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNormRel_L2
//  Purpose:        computes the relative error for the L2-norm of pixel values of two images:
//                      n = SQRT(SUM |src1 - src2|^2 / SUM |src2|^2)
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr         OK
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       roiSize has a field with zero or negative value
//    ippStsDivByZero     SUM |src2|^2 == 0
//  Parameters:
//    pSrc1, pSrc2        Pointers to the source images.
//    src1Step, src2Step  Steps in bytes through the source images
//    roiSize             Size of the source ROI.
//    pValue              Pointer to the computed norm (one-channel data)
//    value               Array of the computed norms for each channel (multi-channel data)
//    hint                Option to specify the computation algorithm
//  Notes:
*/

IPPAPI(IppStatus, ippiNormRel_L2_8u_C1R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_L2_16s_C1R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_L2_16u_C1R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormRel_L2_32f_C1R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue, IppHintAlgorithm hint))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNormDiff_Inf
//  Purpose:        computes the C-norm of pixel values of two images: n = MAX |src1 - src2|
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr         OK
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       roiSize has a field with zero or negative value
//  Parameters:
//    pSrc1, pSrc2        Pointers to the source images.
//    src1Step, src2Step  Steps in bytes through the source images
//    roiSize             Size of the source ROI.
//    pValue              Pointer to the computed norm (one-channel data)
//    value               Array of the computed norms for each channel (multi-channel data)
//  Notes:
*/

IPPAPI(IppStatus, ippiNormDiff_Inf_8u_C1R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_Inf_8u_C3R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_Inf_8u_C4R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_Inf_16s_C1R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_Inf_16s_C3R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_Inf_16s_C4R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_Inf_16u_C1R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_Inf_16u_C3R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_Inf_16u_C4R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_Inf_32f_C1R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_Inf_32f_C3R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_Inf_32f_C4R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNormDiff_L1
//  Purpose:        computes the L1-norm of pixel values of two images: n = SUM |src1 - src2|
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr         OK
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       roiSize has a field with zero or negative value
//  Parameters:
//    pSrc1, pSrc2        Pointers to the source images.
//    src1Step, src2Step  Steps in bytes through the source images
//    roiSize             Size of the source ROI.
//    pValue              Pointer to the computed norm (one-channel data)
//    value               Array of the computed norms for each channel (multi-channel data)
//    hint                Option to specify the computation algorithm
//  Notes:
*/

IPPAPI(IppStatus, ippiNormDiff_L1_8u_C1R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_L1_8u_C3R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_L1_8u_C4R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_L1_16s_C1R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_L1_16s_C3R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_L1_16s_C4R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_L1_16u_C1R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_L1_16u_C3R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_L1_16u_C4R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_L1_32f_C1R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue, IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNormDiff_L1_32f_C3R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3], IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNormDiff_L1_32f_C4R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4], IppHintAlgorithm hint))

/* /////////////////////////////////////////////////////////////////////////////////
//  Name:           ippiNormDiff_L2
//  Purpose:        computes the L2-norm of pixel values of two images:
//                    n = SQRT(SUM |src1 - src2|^2)
//  Context:
//  Returns:        IppStatus
//    ippStsNoErr         OK
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       roiSize has a field with zero or negative value
//  Parameters:
//    pSrc1, pSrc2        Pointers to the source images.
//    src1Step, src2Step  Steps in bytes through the source images
//    roiSize             Size of the source ROI.
//    pValue              Pointer to the computed norm (one-channel data)
//    value               Array of the computed norms for each channel (multi-channel data)
//    hint                Option to specify the computation algorithm
//  Notes:
*/

IPPAPI(IppStatus, ippiNormDiff_L2_8u_C1R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_L2_8u_C3R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_L2_8u_C4R, (const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_L2_16s_C1R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_L2_16s_C3R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_L2_16s_C4R, (const Ipp16s* pSrc1, int src1Step,
                                        const Ipp16s* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_L2_16u_C1R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue))

IPPAPI(IppStatus, ippiNormDiff_L2_16u_C3R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3]))

IPPAPI(IppStatus, ippiNormDiff_L2_16u_C4R, (const Ipp16u* pSrc1, int src1Step,
                                        const Ipp16u* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4]))

IPPAPI(IppStatus, ippiNormDiff_L2_32f_C1R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f* pValue, IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNormDiff_L2_32f_C3R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[3], IppHintAlgorithm hint))

IPPAPI(IppStatus, ippiNormDiff_L2_32f_C4R, (const Ipp32f* pSrc1, int src1Step,
                                        const Ipp32f* pSrc2, int src2Step,
                                        IppiSize roiSize, Ipp64f value[4], IppHintAlgorithm hint))



/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiSwapChannels C3C4R,C4C3R
//
//  Purpose:    Changes the order of channels of the image
//              The function performs operation for each pixel:
//  a) C3C4R.
//    if(dstOrder[i] < 3) dst[i] = src[dstOrder[i]];
//    if(dstOrder[i] == 3) dst[i] = val;
//    if(dstOrder[i] > 3) dst[i] does not change;
//    i = 0,1,2,3
//  b) C4C3R.
//    dst[0] = src [dstOrder[0]];
//    dst[1] = src [dstOrder[1]];
//    dst[2] = src [dstOrder[2]];
//
//  Returns:
//    ippStsNullPtrErr      One of the pointers is NULL
//    ippStsSizeErr         roiSize has a field with zero or negative value
//    ippStsChannelOrderErr dstOrder is out of the range, it should be:
//                            a) C3C4R.
//                              dstOrder[i] => 0, i = 0,1,2,3.
//                            b) C4C3R.
//                              0 <= dstOrder[i] <= 3, i = 0,1,2.
//    ippStsNoErr     OK
//
//  Parameters:
//    pSrc           Pointer  to the source image
//    srcStep        Step in bytes through the source image
//    pDst           Pointer to the  destination image
//    dstStep        Step in bytes through the destination image
//    roiSize        Size of the ROI
//    dstOrder       The order of channels in the destination image
//    val            Constant value for C3C4R
*/
IPPAPI ( IppStatus, ippiSwapChannels_8u_C3C4R,
                    ( const Ipp8u* pSrc, int srcStep,
                            Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   const int dstOrder[4], Ipp8u val ))

IPPAPI ( IppStatus, ippiSwapChannels_16u_C3C4R,
                    ( const Ipp16u* pSrc, int srcStep,
                            Ipp16u* pDst, int dstStep, IppiSize roiSize,
                                    const int dstOrder[4], Ipp16u val ))

IPPAPI ( IppStatus, ippiSwapChannels_32f_C3C4R,
                    ( const Ipp32f* pSrc, int srcStep,
                            Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                    const int dstOrder[4], Ipp32f val ))


IPPAPI ( IppStatus, ippiSwapChannels_8u_C4C3R,
                    ( const Ipp8u* pSrc, int srcStep,
                            Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   const int dstOrder[3] ))

IPPAPI ( IppStatus, ippiSwapChannels_16u_C4C3R,
                    ( const Ipp16u* pSrc, int srcStep,
                            Ipp16u* pDst, int dstStep, IppiSize roiSize,
                                    const int dstOrder[3] ))

IPPAPI ( IppStatus, ippiSwapChannels_32f_C4C3R,
                    ( const Ipp32f* pSrc, int srcStep,
                            Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                    const int dstOrder[3] ))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiSwapChannels
//
//  Purpose:    Changes the order of channels of the image
//              The function performs operation for each pixel:
//                  pDst[0] = pSrc[ dstOrder[0] ]
//                  pDst[1] = pSrc[ dstOrder[1] ]
//                  pDst[2] = pSrc[ dstOrder[2] ]
//                  pDst[3] = pSrc[ dstOrder[3] ]
//
//  Returns:
//    ippStsNullPtrErr      One of the pointers is NULL
//    ippStsSizeErr         roiSize has a field with zero or negative value
//    ippStsStepErr         One of the step values is less than or equal to zero
//    ippStsChannelOrderErr dstOrder is out of the range,
//                           it should be: dstOrder[3] = { 0..2, 0..2, 0..2 } for C3R, AC4R image
//                           and dstOrder[4] = { 0..3, 0..3, 0..3 } for C4R image
//    ippStsNoErr           OK
//
//  Parameters:
//    pSrc           Pointer  to the source image
//    srcStep        Step in bytes through the source image
//    pDst           Pointer to the  destination image
//    dstStep        Step in bytes through the destination image
//    pSrcDst        Pointer to the source/destination image (in-place flavors)
//    srcDstStep     Step through the source/destination image (in-place flavors)
//    roiSize        Size of the ROI
//    dstOrder       The order of channels in the destination image
*/

IPPAPI ( IppStatus, ippiSwapChannels_8u_C4R,
       ( const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
         IppiSize roiSize, const int dstOrder[4] ))

IPPAPI ( IppStatus, ippiSwapChannels_16u_C4R,
       ( const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
         IppiSize roiSize, const int dstOrder[4] ))

IPPAPI ( IppStatus, ippiSwapChannels_32f_C4R,
       ( const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
         IppiSize roiSize, const int dstOrder[4] ))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiSwapChannels
//
//  Purpose:    Changes the order of channels of the image
//              The function performs operation for each pixel:
//                  pDst[0] = pSrc[ dstOrder[0] ]
//                  pDst[1] = pSrc[ dstOrder[1] ]
//                  pDst[2] = pSrc[ dstOrder[2] ]
//                  pDst[3] = pSrc[ dstOrder[3] ]
//
//  Returns:
//    ippStsNullPtrErr      One of the pointers is NULL
//    ippStsSizeErr         roiSize has a field with zero or negative value
//    ippStsStepErr         One of the step values is less than or equal to zero
//    ippStsChannelOrderErr dstOrder is out of the range,
//                           it should be: dstOrder[3] = { 0..2, 0..2, 0..2 } for C3R, AC4R image
//                           and dstOrder[4] = { 0..3, 0..3, 0..3 } for C4R image
//    ippStsNoErr           OK
//
//  Parameters:
//    pSrc           Pointer  to the source image
//    srcStep        Step in bytes through the source image
//    pDst           Pointer to the  destination image
//    dstStep        Step in bytes through the destination image
//    pSrcDst        Pointer to the source/destination image (in-place flavors)
//    srcDstStep     Step through the source/destination image (in-place flavors)
//    roiSize        Size of the ROI
//    dstOrder       The order of channels in the destination image
*/

IPPAPI ( IppStatus, ippiSwapChannels_8u_C3R,
       ( const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
         IppiSize roiSize, const int dstOrder[3] ))

IPPAPI ( IppStatus, ippiSwapChannels_16u_C3R,
       ( const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
         IppiSize roiSize, const int dstOrder[3] ))

IPPAPI ( IppStatus, ippiSwapChannels_32f_C3R,
       ( const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
         IppiSize roiSize, const int dstOrder[3] ))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopy
//
//  Purpose:  copy pixel values from the source image to the destination  image
//
//
//  Returns:
//    ippStsNullPtrErr  One of the pointers is NULL
//    ippStsSizeErr     roiSize has a field with zero or negative value
//    ippStsNoErr       OK
//
//  Parameters:
//    pSrc              Pointer  to the source image buffer
//    srcStep           Step in bytes through the source image buffer
//    pDst              Pointer to the  destination image buffer
//    dstStep           Step in bytes through the destination image buffer
//    roiSize           Size of the ROI
//    pMask             Pointer to the mask image buffer
//    maskStep          Step in bytes through the mask image buffer
*/

IPPAPI( IppStatus, ippiCopy_8u_C3C1R,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_8u_C1C3R,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_8u_C4C1R,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_8u_C1C4R,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_8u_C3CR,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_8u_C4CR,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_8u_AC4C3R,
                   ( const Ipp8u* pSrc, int srcStep,
                     Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_8u_C1R,
                    ( const Ipp8u* pSrc, int srcStep,
                      Ipp8u* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_8u_C3R,
                    ( const Ipp8u* pSrc, int srcStep,
                      Ipp8u* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_8u_C4R,
                    ( const Ipp8u* pSrc, int srcStep,
                      Ipp8u* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_8u_C1MR,
                    ( const Ipp8u* pSrc, int srcStep,
                      Ipp8u* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_8u_C3MR,
                    ( const Ipp8u* pSrc, int srcStep,
                      Ipp8u* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_8u_C4MR,
                    ( const Ipp8u* pSrc, int srcStep,
                      Ipp8u* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI( IppStatus, ippiCopy_16s_C3C1R,
                   ( const Ipp16s* pSrc, int srcStep,
                     Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16s_C1C3R,
                   ( const Ipp16s* pSrc, int srcStep,
                     Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16s_C4C1R,
                   ( const Ipp16s* pSrc, int srcStep,
                     Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16s_C1C4R,
                   ( const Ipp16s* pSrc, int srcStep,
                     Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16s_C3CR,
                   ( const Ipp16s* pSrc, int srcStep,
                     Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16s_C4CR,
                   ( const Ipp16s* pSrc, int srcStep,
                     Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_C1R,
                    ( const Ipp16s* pSrc, int srcStep,
                      Ipp16s* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_C3R,
                    ( const Ipp16s* pSrc, int srcStep,
                      Ipp16s* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_C4R,
                    ( const Ipp16s* pSrc, int srcStep,
                      Ipp16s* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_C1MR,
                    ( const Ipp16s* pSrc, int srcStep,
                      Ipp16s* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_16s_C3MR,
                    ( const Ipp16s* pSrc, int srcStep,
                      Ipp16s* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_16s_C4MR,
                    ( const Ipp16s* pSrc, int srcStep,
                      Ipp16s* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI( IppStatus, ippiCopy_32f_C3C1R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32f_C1C3R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32f_C4C1R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32f_C1C4R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32f_AC4C3R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32f_C3CR,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32f_C4CR,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_C1R,
                    ( const Ipp32f* pSrc, int srcStep,
                      Ipp32f* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_C3R,
                    ( const Ipp32f* pSrc, int srcStep,
                      Ipp32f* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_C4R,
                    ( const Ipp32f* pSrc, int srcStep,
                      Ipp32f* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_C1MR,
                    ( const Ipp32f* pSrc, int srcStep,
                      Ipp32f* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_32f_C3MR,
                    ( const Ipp32f* pSrc, int srcStep,
                      Ipp32f* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_32f_C4MR,
                    ( const Ipp32f* pSrc, int srcStep,
                      Ipp32f* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_8u_C3P3R, ( const Ipp8u* pSrc, int srcStep,
                    Ipp8u* const pDst[3], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_8u_P3C3R, (const  Ipp8u* const pSrc[3],
                            int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_8u_C4P4R, ( const Ipp8u* pSrc, int srcStep,
                    Ipp8u* const pDst[4], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_8u_P4C4R, (const  Ipp8u* const pSrc[4],
                            int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_C3P3R, ( const Ipp16s* pSrc, int srcStep,
                    Ipp16s* const pDst[3], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_P3C3R, (const  Ipp16s* const pSrc[3],
                            int srcStep, Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_C4P4R, ( const Ipp16s* pSrc, int srcStep,
                    Ipp16s* const pDst[4], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16s_P4C4R, (const  Ipp16s* const pSrc[4],
                            int srcStep, Ipp16s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_C3P3R, ( const Ipp32f* pSrc, int srcStep,
                    Ipp32f* const pDst[3], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_P3C3R, (const  Ipp32f* const pSrc[3],
                            int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_C4P4R, ( const Ipp32f* pSrc, int srcStep,
                    Ipp32f* const pDst[4], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32f_P4C4R, (const  Ipp32f* const pSrc[4],
                            int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32s_C3C1R,
                   ( const Ipp32s* pSrc, int srcStep,
                     Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32s_C1C3R,
                   ( const Ipp32s* pSrc, int srcStep,
                     Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32s_C4C1R,
                   ( const Ipp32s* pSrc, int srcStep,
                     Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32s_C1C4R,
                   ( const Ipp32s* pSrc, int srcStep,
                     Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32s_C3CR,
                   ( const Ipp32s* pSrc, int srcStep,
                     Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_32s_C4CR,
                   ( const Ipp32s* pSrc, int srcStep,
                     Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32s_C1R,
                    ( const Ipp32s* pSrc, int srcStep,
                      Ipp32s* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32s_C3R,
                    ( const Ipp32s* pSrc, int srcStep,
                      Ipp32s* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32s_C4R,
                    ( const Ipp32s* pSrc, int srcStep,
                      Ipp32s* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32s_C1MR,
                    ( const Ipp32s* pSrc, int srcStep,
                      Ipp32s* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_32s_C3MR,
                    ( const Ipp32s* pSrc, int srcStep,
                      Ipp32s* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_32s_C4MR,
                    ( const Ipp32s* pSrc, int srcStep,
                      Ipp32s* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_32s_C3P3R, ( const Ipp32s* pSrc, int srcStep,
                    Ipp32s* const pDst[3], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32s_P3C3R, (const  Ipp32s* const pSrc[3],
                            int srcStep, Ipp32s* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32s_C4P4R, ( const Ipp32s* pSrc, int srcStep,
                    Ipp32s* const pDst[4], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_32s_P4C4R, (const  Ipp32s* const pSrc[4],
                            int srcStep, Ipp32s* pDst, int dstStep, IppiSize roiSize ))

IPPAPI( IppStatus, ippiCopy_16u_C3C1R,
                   ( const Ipp16u* pSrc, int srcStep,
                     Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16u_C1C3R,
                   ( const Ipp16u* pSrc, int srcStep,
                     Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16u_C4C1R,
                   ( const Ipp16u* pSrc, int srcStep,
                     Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16u_C1C4R,
                   ( const Ipp16u* pSrc, int srcStep,
                     Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16u_C3CR,
                   ( const Ipp16u* pSrc, int srcStep,
                     Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16u_C4CR,
                   ( const Ipp16u* pSrc, int srcStep,
                     Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI( IppStatus, ippiCopy_16u_AC4C3R,
                   ( const Ipp16u* pSrc, int srcStep,
                     Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16u_C1R,
                    ( const Ipp16u* pSrc, int srcStep,
                      Ipp16u* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16u_C3R,
                    ( const Ipp16u* pSrc, int srcStep,
                      Ipp16u* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16u_C4R,
                    ( const Ipp16u* pSrc, int srcStep,
                      Ipp16u* pDst, int dstStep,IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16u_C1MR,
                    ( const Ipp16u* pSrc, int srcStep,
                      Ipp16u* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_16u_C3MR,
                    ( const Ipp16u* pSrc, int srcStep,
                      Ipp16u* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiCopy_16u_C4MR,
                    ( const Ipp16u* pSrc, int srcStep,
                      Ipp16u* pDst, int dstStep,IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))

IPPAPI ( IppStatus, ippiCopy_16u_C3P3R, ( const Ipp16u* pSrc, int srcStep,
                    Ipp16u* const pDst[3], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16u_P3C3R, (const  Ipp16u* const pSrc[3],
                            int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16u_C4P4R, ( const Ipp16u* pSrc, int srcStep,
                    Ipp16u* const pDst[4], int dstStep, IppiSize roiSize ))
IPPAPI ( IppStatus, ippiCopy_16u_P4C4R, (const  Ipp16u* const pSrc[4],
                            int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize ))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopyReplicateBorder
//
//  Purpose:   Copies pixel values between two buffers and adds
//             the replicated border pixels.
//
//  Returns:
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       1). srcRoiSize or dstRoiSize has a field with negative or zero value
//                        2). topBorderHeight or leftBorderWidth is less than zero
//                        3). dstRoiSize.width < srcRoiSize.width + leftBorderWidth
//                        4). dstRoiSize.height < srcRoiSize.height + topBorderHeight
//    ippStsStepErr       srcStep or dstStep is less than or equal to zero
//    ippStsNoErr         OK
//
//  Parameters:
//    pSrc                Pointer  to the source image buffer
//    srcStep             Step in bytes through the source image
//    pDst                Pointer to the  destination image buffer
//    dstStep             Step in bytes through the destination image
//    scrRoiSize          Size of the source ROI in pixels
//    dstRoiSize          Size of the destination ROI in pixels
//    topBorderHeight     Height of the top border in pixels
//    leftBorderWidth     Width of the left border in pixels
*/

IPPAPI (IppStatus, ippiCopyReplicateBorder_8u_C1R,
            ( const Ipp8u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp8u* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_8u_C3R,
            ( const Ipp8u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp8u* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_8u_C4R,
            ( const Ipp8u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp8u* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_16s_C1R,
            ( const Ipp16s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16s* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_16s_C3R,
            ( const Ipp16s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16s* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_16s_C4R,
            ( const Ipp16s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16s* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_32s_C1R,
            ( const Ipp32s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32s* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_32s_C3R,
            ( const Ipp32s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32s* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_32s_C4R,
            ( const Ipp32s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32s* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )

IPPAPI ( IppStatus, ippiCopyReplicateBorder_8u_C1IR,
                    ( const Ipp8u* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_8u_C3IR,
                    ( const Ipp8u* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_8u_C4IR,
                    ( const Ipp8u* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )

IPPAPI ( IppStatus, ippiCopyReplicateBorder_16s_C1IR,
                    ( const Ipp16s* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_16s_C3IR,
                    ( const Ipp16s* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_16s_C4IR,
                    ( const Ipp16s* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )

IPPAPI ( IppStatus, ippiCopyReplicateBorder_32s_C1IR,
                    ( const Ipp32s* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_32s_C3IR,
                    ( const Ipp32s* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_32s_C4IR,
                    ( const Ipp32s* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )

IPPAPI ( IppStatus, ippiCopyReplicateBorder_16u_C1IR,
                    ( const Ipp16u* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_16u_C3IR,
                    ( const Ipp16u* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_16u_C4IR,
                    ( const Ipp16u* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )

IPPAPI (IppStatus, ippiCopyReplicateBorder_16u_C1R,
            ( const Ipp16u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16u* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_16u_C3R,
            ( const Ipp16u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16u* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_16u_C4R,
            ( const Ipp16u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16u* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )

IPPAPI (IppStatus, ippiCopyReplicateBorder_32f_C1R,
            ( const Ipp32f* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32f* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_32f_C3R,
            ( const Ipp32f* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32f* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI (IppStatus, ippiCopyReplicateBorder_32f_C4R,
            ( const Ipp32f* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32f* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )

IPPAPI ( IppStatus, ippiCopyReplicateBorder_32f_C1IR,
                    ( const Ipp32f* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_32f_C3IR,
                    ( const Ipp32f* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )
IPPAPI ( IppStatus, ippiCopyReplicateBorder_32f_C4IR,
                    ( const Ipp32f* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopyConstBorder
//
//  Purpose:    Copies pixel values between two buffers and adds
//              the border pixels with constant value.
//
//  Returns:
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      1). srcRoiSize or dstRoiSize has a field with negative or zero value
//                       2). topBorderHeight or leftBorderWidth is less than zero
//                       3). dstRoiSize.width < srcRoiSize.width + leftBorderWidth
//                       4). dstRoiSize.height < srcRoiSize.height + topBorderHeight
//    ippStsStepErr      srcStep or dstStep is less than or equal to zero
//    ippStsNoErr        OK
//
//  Parameters:
//    pSrc               Pointer  to the source image buffer
//    srcStep            Step in bytes through the source image
//    pDst               Pointer to the  destination image buffer
//    dstStep            Step in bytes through the destination image
//    srcRoiSize         Size of the source ROI in pixels
//    dstRoiSize         Size of the destination ROI in pixels
//    topBorderHeight    Height of the top border in pixels
//    leftBorderWidth    Width of the left border in pixels
//    value              Constant value to assign to the border pixels
*/

IPPAPI (IppStatus, ippiCopyConstBorder_8u_C1R,
            ( const Ipp8u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp8u* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            Ipp8u value ) )
IPPAPI (IppStatus, ippiCopyConstBorder_8u_C3R,
            ( const Ipp8u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp8u* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp8u value[3] ) )
IPPAPI (IppStatus, ippiCopyConstBorder_8u_C4R,
            ( const Ipp8u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp8u* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp8u value[4] ) )
IPPAPI (IppStatus, ippiCopyConstBorder_16s_C1R,
            ( const Ipp16s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16s* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            Ipp16s value ) )
IPPAPI (IppStatus, ippiCopyConstBorder_16s_C3R,
            ( const Ipp16s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16s* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp16s value[3] ) )
IPPAPI (IppStatus, ippiCopyConstBorder_16s_C4R,
            ( const Ipp16s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16s* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp16s value[4] ) )
IPPAPI (IppStatus, ippiCopyConstBorder_32s_C1R,
            ( const Ipp32s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32s* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            Ipp32s value ) )
IPPAPI (IppStatus, ippiCopyConstBorder_32s_C3R,
            ( const Ipp32s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32s* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp32s value[3] ) )
IPPAPI (IppStatus, ippiCopyConstBorder_32s_C4R,
            ( const Ipp32s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32s* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp32s value[4] ) )

IPPAPI (IppStatus, ippiCopyConstBorder_16u_C1R,
            ( const Ipp16u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16u* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            Ipp16u value ) )
IPPAPI (IppStatus, ippiCopyConstBorder_16u_C3R,
            ( const Ipp16u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16u* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp16u value[3] ) )
IPPAPI (IppStatus, ippiCopyConstBorder_16u_C4R,
            ( const Ipp16u* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp16u* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp16u value[4] ) )

IPPAPI (IppStatus, ippiCopyConstBorder_32f_C1R,
            ( const Ipp32f* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32f* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            Ipp32f value ) )
IPPAPI (IppStatus, ippiCopyConstBorder_32f_C3R,
            ( const Ipp32f* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32f* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp32f value[3] ) )
IPPAPI (IppStatus, ippiCopyConstBorder_32f_C4R,
            ( const Ipp32f* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32f* pDst,  int dstStep, IppiSize dstRoiSize,
                            int topBorderHeight, int leftBorderWidth,
                            const Ipp32f value[4] ) )


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopyWrapBorder
//
//  Purpose:    Copies pixel values between two buffers and adds the border pixels.
//
//  Returns:
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       1). srcRoiSize or dstRoiSize has a field with negative or zero value
//                        2). topBorderHeight or leftBorderWidth is less than zero
//                        3). dstRoiSize.width < srcRoiSize.width + leftBorderWidth
//                        4). dstRoiSize.height < srcRoiSize.height + topBorderHeight
//    ippStsStepErr       srcStep or dstStep is less than or equal to zero
//    ippStsNoErr         OK
//
//  Parameters:
//    pSrc                Pointer  to the source image buffer
//    srcStep             Step in bytes through the source image
//    pDst                Pointer to the  destination image buffer
//    dstStep             Step in bytes through the destination image
//    scrRoiSize          Size of the source ROI in pixels
//    dstRoiSize          Size of the destination ROI in pixels
//    topBorderHeight     Height of the top border in pixels
//    leftBorderWidth     Width of the left border in pixels
*/

IPPAPI (IppStatus, ippiCopyWrapBorder_32s_C1R,
            ( const Ipp32s* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32s* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI ( IppStatus, ippiCopyWrapBorder_32s_C1IR,
                    ( const Ipp32s* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )

IPPAPI (IppStatus, ippiCopyWrapBorder_32f_C1R,
            ( const Ipp32f* pSrc,  int srcStep, IppiSize srcRoiSize,
                    Ipp32f* pDst,  int dstStep, IppiSize dstRoiSize,
                    int topBorderHeight, int leftBorderWidth ) )
IPPAPI ( IppStatus, ippiCopyWrapBorder_32f_C1IR,
                    ( const Ipp32f* pSrc,  int srcDstStep,
                            IppiSize srcRoiSize, IppiSize dstRoiSize,
                            int topBorderHeight, int leftborderwidth ) )


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiSet
//
//  Purpose:    Sets pixels in the image buffer to a constant value
//
//  Returns:
//    ippStsNullPtrErr  One of pointers is NULL
//    ippStsSizeErr     roiSize has a field with negative or zero value
//    ippStsNoErr       OK
//
//  Parameters:
//    value      Constant value assigned to each pixel in the image buffer
//    pDst       Pointer to the destination image buffer
//    dstStep    Step in bytes through the destination image buffer
//    roiSize    Size of the ROI
//    pMask      Pointer to the mask image buffer
//    maskStep   Step in bytes through the mask image buffer
*/

IPPAPI ( IppStatus, ippiSet_8u_C1R,
                    ( Ipp8u value, Ipp8u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_8u_C3CR,
                    ( Ipp8u value, Ipp8u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_8u_C4CR,
                    ( Ipp8u value, Ipp8u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_8u_C3R,
                    ( const Ipp8u value[3], Ipp8u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_8u_C4R,
                    ( const Ipp8u value[4], Ipp8u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_8u_C1MR,
                    ( Ipp8u value, Ipp8u* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_8u_C3MR,
                    ( const Ipp8u value[3], Ipp8u* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))

IPPAPI ( IppStatus, ippiSet_8u_C4MR,
                    ( const Ipp8u value[4], Ipp8u* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_16s_C1R,
                    ( Ipp16s value, Ipp16s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16s_C3CR,
                    ( Ipp16s value, Ipp16s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16s_C4CR,
                    ( Ipp16s value, Ipp16s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16s_C3R,
                    ( const Ipp16s value[3], Ipp16s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16s_C4R,
                    ( const Ipp16s value[4], Ipp16s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16s_C1MR,
                    ( Ipp16s value, Ipp16s* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_16s_C3MR,
                    ( const Ipp16s value[3], Ipp16s* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_16s_C4MR,
                    ( const Ipp16s value[4], Ipp16s* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_32f_C1R,
                    ( Ipp32f value, Ipp32f* pDst, int dstStep,
                      IppiSize roiSize ))

IPPAPI ( IppStatus, ippiSet_32f_C3CR,
                    ( Ipp32f value, Ipp32f* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32f_C4CR,
                    ( Ipp32f value, Ipp32f* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32f_C3R,
                    ( const Ipp32f value[3], Ipp32f* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32f_C4R,
                    ( const Ipp32f value[4], Ipp32f* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32f_C1MR,
                    ( Ipp32f value, Ipp32f* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_32f_C3MR,
                    ( const Ipp32f value[3], Ipp32f* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_32f_C4MR,
                    ( const Ipp32f value[4], Ipp32f* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_32s_C1R,
                    ( Ipp32s value, Ipp32s* pDst, int dstStep,
                      IppiSize roiSize ))

IPPAPI ( IppStatus, ippiSet_32s_C3CR,
                    ( Ipp32s value, Ipp32s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32s_C4CR,
                    ( Ipp32s value, Ipp32s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32s_C3R,
                    ( const Ipp32s value[3], Ipp32s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32s_C4R,
                    ( const Ipp32s value[4], Ipp32s* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_32s_C1MR,
                    ( Ipp32s value, Ipp32s* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_32s_C3MR,
                    ( const Ipp32s value[3], Ipp32s* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_32s_C4MR,
                    ( const Ipp32s value[4], Ipp32s* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))

IPPAPI ( IppStatus, ippiSet_16u_C1R,
                    ( Ipp16u value, Ipp16u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16u_C3CR,
                    ( Ipp16u value, Ipp16u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16u_C4CR,
                    ( Ipp16u value, Ipp16u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16u_C3R,
                    ( const Ipp16u value[3], Ipp16u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16u_C4R,
                    ( const Ipp16u value[4], Ipp16u* pDst, int dstStep,
                      IppiSize roiSize ))
IPPAPI ( IppStatus, ippiSet_16u_C1MR,
                    ( Ipp16u value, Ipp16u* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_16u_C3MR,
                    ( const Ipp16u value[3], Ipp16u* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))
IPPAPI ( IppStatus, ippiSet_16u_C4MR,
                    ( const Ipp16u value[4], Ipp16u* pDst, int dstStep,
                      IppiSize roiSize,
                      const Ipp8u* pMask, int maskStep ))


/* //////////////////////////////////////////////////////////////////////////////////////
//                   Image Proximity Measures
////////////////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//  Names:  ippiCrossCorrNormGetBufferSize
//
//  Purpose:     Computes the size (in bytes) of the work buffer for the ippiCrossCorrNorm functions.
//
//  Parameters:
//    srcRoiSize  - Size of the source ROI in pixels.
//    tplRoiSize  - Size of the template ROI in pixels.
//    algType     - Bit-field mask for the algorithm type definition. Possible values are the results of composition of the IppAlgType, IppiROIShape, and IppiNormOp values.
//                  Usage example: algType=(ippiROIFull|ippAlgFFT|ippiNormalized); - full-shaped cross-correlation will be calculated
//                      using 2D FFT and normalization applied to result image.
//    pBufferSize - Pointer to the size of the work buffer (in bytes).
//  Return:
//    ippStsNoErr       - OK.
//    ippStsSizeErr     - Error when:
//                            srcRoiSize or tplRoiSize is negative, or equal to zero.
//                            The value of srcRoiSize is less than the corresponding value of the tplRoiSize.
//    ippStsAlgTypeErr  - Error when :
//                            The result of the bitwise AND operation between the algType and ippAlgMask differs from the ippAlgAuto, ippAlgDirect, or ippAlgFFT values.
//                            The result of the bitwise AND operation between the algType and ippiROIMask differs from the ippiROIFull, ippiROISame, or ippiROIValid values.
//                            The result of the bitwise AND operation between the algType and ippiNormMask differs from the ippiNormNone, ippiNormalized, or ippiNormCoefficients values.
//    ippStsNullPtrErr  - Error when the pBufferSize is NULL.
*/
IPPAPI( IppStatus, ippiCrossCorrNormGetBufferSize, ( IppiSize srcRoiSize, IppiSize tplRoiSize, IppEnum algType, int* pBufferSize ))


/* ////////////////////////////////////////////////////////////////////////////
//  Names: ippiCrossCorrNorm_32f_C1R
//         ippiCrossCorrNorm_16u32f_C1R
//         ippiCrossCorrNorm_8u32f_C1R
//         ippiCrossCorrNorm_8u_C1RSfs
//  Purpose: Computes normalized cross-correlation between an image and a template.
//           The result image size depends on operation shape selected in algType mask as follows :
//             (Wa+Wb-1)*(Ha+Hb-1) for ippiROIFull mask,
//             (Wa)*(Ha)           for ippiROISame mask,
//             (Wa-Wb+1)*(Ha-Hb+1) for ippiROIValid mask,
//           where Wa*Ha and Wb*Hb are the sizes of the image and template correspondingly.
//           Support of normalization operations (set in the algType mask) is set by selecting the following masks:
//             ippiNormNone   - the cross-correlation without normalization.
//             ippiNormalized - the normalized cross-correlation.
//             ippiNormLevel  - the normalized correlation coefficients.
//           If the IppAlgMask value in algType is equal to ippAlgAuto, the optimal algorithm is selected automatically.
//           For big data size, the function uses 2D FFT algorithm.
//  Parameters:
//    pSrc        - Pointer to the source image ROI.
//    srcStep     - Distance, in bytes, between the starting points of consecutive lines in the source image.
//    srcRoiSize  - Size of the source ROI in pixels.
//    pTpl        - Pointer to the template image.
//    tplStep     - Distance, in bytes, between the starting points of consecutive lines in the template image.
//    tplRoiSize  - Size of the template ROI in pixels.
//    pDst        - Pointer to the destination image ROI.
//    dstStep     - Distance, in bytes, between the starting points of consecutive lines in the destination image.
//    scaleFactor - Scale factor.
//    algType     - Bit-field mask for the algorithm type definition. Possible values are the results of composition of the IppAlgType, IppiROIShape, and IppiNormOp values.
//                  Usage example: algType=(ippiROIFull|ippAlgFFT|ippiNormNone); - full-shaped cross-correlation will be calculated using 2D FFT without result normalization.
//    pBuffer     - Pointer to the work buffer.
//  Returns:
//    ippStsNoErr      OK.
//    ippStsNullPtrErr Error when any of the specified pointers is NULL.
//    ippStsStepErr    Error when the value of srcStep, tplStep, or dstStep is negative, or equal to zero.
//    ippStsSizeErr    Error when :
//                         srcRoiSize or tplRoiSize is negative, or equal to zero.
//                         The value of srcRoiSize is less than the corresponding value of tplRoiSize.
//    ippStsAlgTypeErr Error when :
//                         The result of the bitwise AND operation between the algType and ippAlgMask differs from the ippAlgAuto, ippAlgDirect, or ippAlgFFT values.
//                         The result of the bitwise AND operation between the algType and ippiROIMask differs from the ippiROIFull, ippiROISame, or ippiROIValid values.
//                         The result of the bitwise AND operation between the algType and ippiNormMask differs from the ippiNormNone, ippiNormalized, or ippiNormCoefficients values.
*/

IPPAPI( IppStatus, ippiCrossCorrNorm_32f_C1R, ( const Ipp32f* pSrc, int srcStep, IppiSize srcRoiSize, const Ipp32f* pTpl, int tplStep, IppiSize tplRoiSize,
      Ipp32f* pDst, int dstStep, IppEnum algType, Ipp8u* pBuffer ))

IPPAPI( IppStatus, ippiCrossCorrNorm_8u32f_C1R, ( const Ipp8u* pSrc, int srcStep, IppiSize srcRoiSize, const Ipp8u* pTpl, int tplStep, IppiSize tplRoiSize,
      Ipp32f* pDst, int dstStep, IppEnum algType, Ipp8u* pBuffer ))



/* /////////////////////////////////////////////////////////////////////////////
//                  Definitions for DCT Functions
///////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//                  DCT Context Functions
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiDCTFwdInit, ippiDCTInvInit
//  Purpose:    Initializes the forward/inverse DCT context structure
//  Parameters:
//     pDCTSpec   Pointer to the DCT context structure
//     roiSize    Size of the ROI
//     pMemInit   Pointer to the temporary work buffer
//  Returns:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       pDCTSpec == NULL
//     ippStsSizeErr          roiSize has a field with zero or negative value
*/

IPPAPI (IppStatus, ippiDCTFwdInit_32f,
                   ( IppiDCTFwdSpec_32f* pDCTSpec,
                   IppiSize roiSize, Ipp8u* pMemInit ))
IPPAPI (IppStatus, ippiDCTInvInit_32f,
                   ( IppiDCTInvSpec_32f* pDCTSpec,
                   IppiSize roiSize, Ipp8u* pMemInit ))

/* /////////////////////////////////////////////////////////////////////////////
//                  DCT Buffer Size
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiDCTFwdGetSize, ippiDCTInvGetSize
//  Purpose:    Computes the size of the forward/inverse DCT context structure and the size
//              of the required work buffer (in bytes)
//  Parameters:
//     roiSize    Size of the ROI
//     pSizeSpec  Pointer to the size value of DCT context structure
//     pSizeInit  Pointer to the size value of the buffer for DCT initialization function
//     pSizeBuf   Pointer to the size value of the DCT external work buffer
//  Returns:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers is NULL
//     ippStsSizeErr          roiSize has a field with zero or negative value
*/

IPPAPI (IppStatus, ippiDCTFwdGetSize_32f,
                   ( IppiSize roiSize,
                   int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))
IPPAPI (IppStatus, ippiDCTInvGetSize_32f,
                   ( IppiSize roiSize,
                   int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))

/* /////////////////////////////////////////////////////////////////////////////
//                  DCT Transforms
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiDCTFwd, ippiDCTInv
//  Purpose:    Performs forward or inverse DCT of an image
//  Parameters:
//     pDCTSpec   Pointer to the DCT context structure
//     pSrc       Pointer to the source image
//     srcStep    Step through the source image
//     pDst       Pointer to the destination image
//     dstStep    Step through the destination image
//     pBuffer    Pointer to the work buffer
//  Returns:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers with the exception of
                              pBuffer is NULL
//     ippStsStepErr          srcStep or dstStep value is zero or negative
//     ippStsContextMatchErr  Invalid context structure
//     ippStsMemAllocErr      Memory allocation error
*/

IPPAPI (IppStatus, ippiDCTFwd_32f_C1R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep,
                     const IppiDCTFwdSpec_32f* pDCTSpec,
                     Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippiDCTInv_32f_C1R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep,
                     const IppiDCTInvSpec_32f* pDCTSpec,
                     Ipp8u* pBuffer ))

/* /////////////////////////////////////////////////////////////////////////////
//                  DFT Transforms
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiDFTFwd, ippiDFTInv
//  Purpose:    Performs forward or inverse DFT of an image
//  Parameters:
//     pDFTSpec    Pointer to the DFT context structure
//     pSrc        Pointer to source image
//     srcStep     Step through the source image
//     pDst        Pointer to the destination image
//     dstStep     Step through the destination image
//     pSrcDst     Pointer to the source/destination image (in-place)
//     srcDstStep  Step through the source/destination image (in-place)
//     pBuffer     Pointer to the external work buffer
//  Returns:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers with the exception of
                              pBuffer is NULL
//     ippStsStepErr          srcStep or dstStep value is zero or negative
//     ippStsContextMatchErr  Invalid context structure
//     ippStsMemAllocErr      Memory allocation error
*/

IPPAPI (IppStatus, ippiDFTFwd_CToC_32fc_C1R,
                   ( const Ipp32fc* pSrc, int srcStep,
                     Ipp32fc* pDst, int dstStep,
                     const IppiDFTSpec_C_32fc* pDFTSpec,
                     Ipp8u* pBuffer ))


IPPAPI (IppStatus, ippiDFTFwd_RToPack_32f_C1R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep,
                     const IppiDFTSpec_R_32f* pDFTSpec,
                     Ipp8u* pBuffer ))


IPPAPI (IppStatus, ippiDFTInv_CToC_32fc_C1R,
                   ( const Ipp32fc* pSrc, int srcStep,
                     Ipp32fc* pDst, int dstStep,
                     const IppiDFTSpec_C_32fc* pDFTSpec,
                     Ipp8u* pBuffer ))

IPPAPI (IppStatus, ippiDFTInv_PackToR_32f_C1R,
                   ( const Ipp32f* pSrc, int srcStep,
                     Ipp32f* pDst, int dstStep,
                     const IppiDFTSpec_R_32f* pDFTSpec,
                     Ipp8u* pBuffer ))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiDFTGetSize
//  Purpose:    Computes the size of the DFT context structure and the size
                of the required work buffer (in bytes)
//  Parameters:
//     roiSize    Size of the ROI
//     flag       Flag to choose the results normalization factors
//     hint       Option to select the algorithmic implementation of the transform
//                function
//     pSizeSpec  Pointer to the size value of DFT specification structure
//     pSizeInit  Pointer to the size value of the buffer for DFT initialization function
//     pSizeBuf   Pointer to the size value of the DFT external work buffer
//  Return:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers is NULL
//     ippStsFftOrderErr      Invalid roiSize
//     ippStsSizeErr          roiSize has a field with zero or negative value
//     ippStsFFTFlagErr       Incorrect normalization flag value
*/

IPPAPI (IppStatus, ippiDFTGetSize_C_32fc,
                   ( IppiSize roiSize, int flag, IppHintAlgorithm hint,
                   int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))
IPPAPI (IppStatus, ippiDFTGetSize_R_32f,
                   ( IppiSize roiSize, int flag, IppHintAlgorithm hint,
                   int* pSizeSpec, int* pSizeInit, int* pSizeBuf ))


/* /////////////////////////////////////////////////////////////////////////////
//                  DFT Context Functions
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:         ippiDFTInit
//  Purpose:      Initializes the DFT context structure
//  Parameters:
//     roiSize    Size of the ROI
//     flag       Flag to choose the results normalization factors
//     hint       Option to select the algorithmic implementation of the transform
//                function
//     pDFTSpec   Double pointer to the DFT context structure
//     pMemInit   Pointer to initialization buffer
//  Returns:
//     ippStsNoErr            No errors
//     ippStsNullPtrErr       One of the specified pointers is NULL
//     ippStsFftOrderErr      Invalid roiSize
//     ippStsSizeErr          roiSize has a field with zero or negative value
//     ippStsFFTFlagErr       Incorrect normalization flag value
*/

IPPAPI (IppStatus, ippiDFTInit_C_32fc,
                   ( IppiSize roiSize, int flag, IppHintAlgorithm hint,
                   IppiDFTSpec_C_32fc* pDFTSpec, Ipp8u* pMemInit ))
IPPAPI (IppStatus, ippiDFTInit_R_32f,
                   ( IppiSize roiSize, int flag, IppHintAlgorithm hint,
                   IppiDFTSpec_R_32f* pDFTSpec, Ipp8u* pMemInit ))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiAddC_32f_C1IR, ippiAddC_32f_C3IR, ippiAddC_32f_C4IR, ippiAddC_32f_AC4IR,
//        ippiSubC_32f_C1IR, ippiSubC_32f_C3IR, ippiSubC_32f_C4IR, ippiSubC_32f_AC4IR,
//        ippiMulC_32f_C1IR, ippiMulC_32f_C3IR, ippiMulC_32f_C4IR, ippiMulC_32f_AC4IR
//
//  Purpose:    Adds, subtracts, or multiplies pixel values of an image
//              and a constant, and places the results in the same image.
//
//  Returns:
//    ippStsNoErr              OK
//    ippStsNullPtrErr         Pointer is NULL
//    ippStsSizeErr            Width or height of an image is less than or equal to zero
//
//  Parameters:
//    value                    The constant value for the specified operation
//    pSrcDst                  Pointer to the image
//    srcDstStep               Step through the image
//    roiSize                  Size of the ROI
*/

IPPAPI(IppStatus, ippiMulC_32f_C1IR,  (Ipp32f value, Ipp32f* pSrcDst, int srcDstStep,
                                       IppiSize roiSize))


/* //////////////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiAddC_8u_C1IRSfs,  ippiAddC_8u_C3IRSfs,  ippiAddC_8u_C4IRSfs,   ippiAddC_8u_AC4IRSfs,
//        ippiAddC_16s_C1IRSfs, ippiAddC_16s_C3IRSfs, ippiAddC_16s_C4IRSfs,  ippiAddC_16s_AC4IRSfs,
//        ippiAddC_16u_C1IRSfs, ippiAddC_16u_C3IRSfs, ippiAddC_16u_C4IRSfs,  ippiAddC_16u_AC4IRSfs,
//        ippiSubC_8u_C1IRSfs,  ippiSubC_8u_C3IRSfs,  ippiSubC_8u_C4IRSfs,   ippiSubC_8u_AC4IRSfs,
//        ippiSubC_16s_C1IRSfs, ippiSubC_16s_C3IRSfs, ippiSubC_16s_C4IRSfs,  ippiSubC_16s_AC4IRSfs,
//        ippiSubC_16u_C1IRSfs, ippiSubC_16u_C3IRSfs, ippiSubC_16u_C4IRSfs,  ippiSubC_16u_AC4IRSfs,
//        ippiMulC_8u_C1IRSfs,  ippiMulC_8u_C3IRSfs,  ippiMulC_8u_C4IRSfs,   ippiMulC_8u_AC4IRSfs,
//        ippiMulC_16s_C1IRSfs, ippiMulC_16s_C3IRSfs, ippiMulC_16s_C4IRSfs,  ippiMulC_16s_AC4IRSfs
//        ippiMulC_16u_C1IRSfs, ippiMulC_16u_C3IRSfs, ippiMulC_16u_C4IRSfs,  ippiMulC_16u_AC4IRSfs
//
//  Purpose:    Adds, subtracts, or multiplies pixel values of an image and a constant
//              and places the scaled results in the same image.
//
//  Returns:
//    ippStsNoErr              OK
//    ippStsNullPtrErr         Pointer is NULL
//    ippStsSizeErr            Width or height of an image is less than or equal to zero
//
//  Parameters:
//    value                    Constant value (constant vector for multi-channel images)
//    pSrcDst                  Pointer to the image
//    srcDstStep               Step through the image
//    roiSize                  Size of the ROI
//    scaleFactor              Scale factor
*/

IPPAPI(IppStatus, ippiMulC_16s_C1IRSfs,  (Ipp16s value, Ipp16s* pSrcDst, int srcDstStep,
                                          IppiSize roiSize, int scaleFactor))


/* /////////////////////////////////////////////////////////////////////////////
//  Names:  ippiSqrDistanceNormGetBufferSize
//
//  Purpose:     Computes the size of the work buffer for the ippiSqrDistanceNorm functions.
//
//  Parameters:
//    srcRoiSize  - Size of the source ROI, in pixels.
//    tplRoiSize  - Size of the template ROI, in pixels.
//    algType     - Bit-field mask for the algorithm type definition. Possible values are the results of composition of the IppAlgType, IppiROIShape, and IppiNormOp values.
//                  Usage example: algType=(ippiROIFull|ippAlgFFT|ippiNormalized); - result image will be calculated for full-shaped ROI
//                  using 2D FFT and normalization applied.
//    pBufferSize - Pointer where to store the calculated buffer size (in bytes)
//  Return:
//    ippStsNoErr      - Ok.
//    ippStsSizeErr    - Error when :
//                           srcRoiSize or tplRoiSize is negative, or equal to zero.
//                           The value of srcRoiSize is less than the corresponding value of tplRoiSize.
//    ippStsAlgTypeErr - Error when :
//                           The result of the bitwise AND operation between the algType and ippAlgMask differs from the ippAlgAuto, ippAlgDirect, or ippAlgFFT values.
//                           The result of the bitwise AND operation between the algType and ippiROIMask differs from the ippiROIFull, ippiROISame, or ippiROIValid values.
//                           The result of the bitwise AND operation between the algType and ippiNormMask differs from the ippiNormNone or ippiNormalized values.
//    ippStsNullPtrErr - Error when the pBufferSize is NULL.
*/

IPPAPI( IppStatus, ippiSqrDistanceNormGetBufferSize, ( IppiSize srcRoiSize, IppiSize tplRoiSize, IppEnum algType, int* pBufferSize ))


/* ////////////////////////////////////////////////////////////////////////////
//  Names: ippiSqrDistanceNorm_32f_C1R
//         ippiSqrDistanceNorm_16u32f_C1R
//         ippiSqrDistanceNorm_8u32f_C1R
//         ippiSqrDistanceNorm_8u_C1RSfs
//  Purpose: Computes Euclidean distance between an image and a template.
//           The result image size depends on operation shape selected in algType mask as follows :
//             (Wa+Wb-1)*(Ha+Hb-1) for ippiROIFull mask,
//             (Wa)*(Ha)           for ippiROISame mask,
//             (Wa-Wb+1)*(Ha-Hb+1) for ippiROIValid mask,
//           where Wa*Ha and Wb*Hb are the sizes of the image and template , respectively.
//           Support of normalization operations (set the algType mask) :
//             ippiNormNone   - the squared Euclidean distances.
//             ippiNormalized - the normalized squared Euclidean distances.
//           If the IppAlgMask value in algType is equal to ippAlgAuto, the optimal algorithm is selected
//           automatically. For big data size, the function uses 2D FFT algorithm.
//  Parameters:
//    pSrc        - Pointer to the source image ROI.
//    srcStep     - Distance, in bytes, between the starting points of consecutive lines in the source image.
//    srcRoiSize  - Size of the source ROI, in pixels.
//    pTpl        - Pointer to the template image.
//    tplStep     - Distance, in bytes, between the starting points of consecutive lines in the template image.
//    tplRoiSize  - Size of the template ROI, in pixels.
//    pDst        - Pointer to the destination image ROI.
//    dstStep     - Distance, in bytes, between the starting points of consecutive lines in the destination image.
//    scaleFactor - Scale factor.
//    algType     - Bit-field mask for the algorithm type definition. Possible values are the results of composition of the IppAlgType, IppiROIShape, and IppiNormOp values.
//                  Usage example: algType=(ippiROIFull|ippiNormNone|ippAlgFFT); - result will be calculated for full-shaped ROI using 2D FFT without normalization.
//    pBuffer     - Pointer to the buffer for internal calculation.
//  Returns:
//    ippStsNoErr      OK.
//    ippStsNullPtrErr Error when any of the specified pointers is NULL.
//    ippStsStepErr    Error when the value of srcStep, tplStep, or dstStep is negative, or equal to zero.
//    ippStsSizeErr    Error when :
//                         srcRoiSize or tplRoiSize is negative, or equal to zero.
//                         The value of srcRoiSize is less than the corresponding value of the tplRoiSize.
//    ippStsAlgTypeErr Error when :
//                         The result of the bitwise AND operation between the algType and ippAlgMask differs from the ippAlgAuto, ippAlgDirect, or ippAlgFFT values.
//                         The result of the bitwise AND operation between the algType and ippiROIMask differs from the ippiROIFull, ippiROISame, or ippiROIValid values.
//                         The result of the bitwise AND operation between the algType and ippiNormMask differs from the ippiNormNone or ippiNormalized values.
*/
IPPAPI( IppStatus, ippiSqrDistanceNorm_32f_C1R, ( const Ipp32f* pSrc, int srcStep, IppiSize srcRoiSize, const Ipp32f* pTpl, int tplStep, IppiSize tplRoiSize,
         Ipp32f* pDst, int dstStep, IppEnum algType, Ipp8u* pBuffer ))

IPPAPI( IppStatus, ippiSqrDistanceNorm_8u32f_C1R, ( const Ipp8u* pSrc, int srcStep, IppiSize srcRoiSize, const Ipp8u* pTpl, int tplStep, IppiSize tplRoiSize,
         Ipp32f* pDst, int dstStep, IppEnum algType, Ipp8u* pBuffer ))


/* /////////////////////////////////////////////////////////////////////////////
//                      Resize Transform Functions
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetSize
//  Purpose:            Computes the size of Spec structure and temporal buffer for Resize transform
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    dstSize           Size of the output image (in pixels)
//    interpolation     Interpolation method
//    antialiasing      Supported values: 1- resizing with antialiasing, 0 - resizing without antialiasing
//    pSpecSize         Pointer to the size (in bytes) of the Spec structure
//    pInitBufSize      Pointer to the size (in bytes) of the temporal buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error in the following cases:
//                              -  if the source image size is less than a filter size of the chosen
//                                 interpolation method (except ippSuper),
//                              -  if one of the specified dimensions of the source image is less than
//                                 the corresponding dimension of the destination image (for ippSuper method only),
//                              -  if width or height of the source or destination image is negative,
//                              -  if one of the calculated sizes exceeds maximum 32 bit signed integer
//                                 positive value (the size of the one of the processed images is too large).
//                              -  if width or height of the source or destination image is negative.
//    ippStsInterpolationErr    Indicates an error if interpolation has an illegal value
//    ippStsNoAntialiasing      Indicates a warning if specified interpolation does not support antialiasing
//    ippStsNotSupportedModeErr Indicates an error if requested mode is currently not supported
//
//  Notes:
//    1. Supported interpolation methods are ippNearest, ippLinear, ippCubic, ippLanczos and ippSuper.
//    2. If antialiasing value is equal to 1, use the ippResizeAntialiasing<Filter>Init functions, otherwise, use ippResize<Filter>Init
//    3. The implemented interpolation algorithms have the following filter sizes: Nearest Neighbor 1x1,
//       Linear 2x2, Cubic 4x4, 2-lobed Lanczos 4x4.
*/

IPPAPI (IppStatus, ippiResizeGetSize_8u, (
    IppiSize srcSize, IppiSize dstSize,
    IppiInterpolationType interpolation, Ipp32u antialiasing,
    int* pSpecSize, int* pInitBufSize))

IPPAPI (IppStatus, ippiResizeGetSize_16u, (
    IppiSize srcSize, IppiSize dstSize,
    IppiInterpolationType interpolation, Ipp32u antialiasing,
    int* pSpecSize, int* pInitBufSize))

IPPAPI (IppStatus, ippiResizeGetSize_16s, (
    IppiSize srcSize, IppiSize dstSize,
    IppiInterpolationType interpolation, Ipp32u antialiasing,
    int* pSpecSize, Ipp32s* pInitBufSize))

IPPAPI (IppStatus, ippiResizeGetSize_32f, (
    IppiSize srcSize, IppiSize dstSize,
    IppiInterpolationType interpolation, Ipp32u antialiasing,
    int* pSpecSize, int* pInitBufSize))

IPPAPI (IppStatus, ippiResizeGetSize_64f, (
    IppiSize srcSize, IppiSize dstSize,
    IppiInterpolationType interpolation, Ipp32u antialiasing,
    int* pSpecSize, int* pInitBufSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetBufferSize
//  Purpose:            Computes the size of external buffer for Resize transform
//
//  Parameters:
//    pSpec             Pointer to the Spec structure for resize filter
//    dstSize           Size of the output image (in pixels)
//    numChannels       Number of channels, possible values are 1 or 3 or 4
//    pBufSize          Pointer to the size (in bytes) of the external buffer
//
//  Return Values:
//    ippStsNoErr           Indicates no error
//    ippStsNullPtrErr      Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation     Indicates a warning if width or height of output image is zero
//    ippStsContextMatchErr Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNumChannelErr   Indicates an error if numChannels has illegal value
//    ippStsSizeErr         Indicates an error condition in the following cases:
//                          - if width or height of the source image is negative,
//                          - if the calculated buffer size exceeds maximum 32 bit signed integer
//                            positive value (the processed image ROIs are too large ).
//    ippStsSizeWrn         Indicates a warning if the destination image size is more than
//                          the destination image origin size
*/

IPPAPI (IppStatus, ippiResizeGetBufferSize_8u, (
    IppiResizeSpec_32f* pSpec,
    IppiSize dstSize, Ipp32u numChannels,
    int* pBufSize))

IPPAPI (IppStatus, ippiResizeGetBufferSize_16u, (
    IppiResizeSpec_32f* pSpec,
    IppiSize dstSize, Ipp32u numChannels,
    int* pBufSize))

IPPAPI (IppStatus, ippiResizeGetBufferSize_16s, (
    IppiResizeSpec_32f* pSpec,
    IppiSize dstSize, Ipp32u numChannels,
    int* pBufSize))

IPPAPI (IppStatus, ippiResizeGetBufferSize_32f, (
    IppiResizeSpec_32f* pSpec,
    IppiSize dstSize, Ipp32u numChannels,
    int* pBufSize))

IPPAPI (IppStatus, ippiResizeGetBufferSize_64f, (
    IppiResizeSpec_64f* pSpec,
    IppiSize dstSize, Ipp32u numChannels,
    int* pBufSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetSrcOffset
//  Purpose:            Computes the offset of input image for Resize transform by tile processing
//
//  Parameters:
//    pSpec             Pointer to the Spec structure for resize filter
//    dstOffset         Offset of the tiled destination image respective
//                      to the destination image origin
//    srcOffset         Pointer to the offset of input image
//
//  Return Values:
//    ippStsNoErr           Indicates no error
//    ippStsNullPtrErr      Indicates an error if one of the specified pointers is NULL
//    ippStsContextMatchErr Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsOutOfRangeErr   Indicates an error if the destination image offset point is outside the
//                          destination image origin
*/
IPPAPI (IppStatus, ippiResizeGetSrcOffset_8u, (
    IppiResizeSpec_32f* pSpec,
    IppiPoint dstOffset, IppiPoint* srcOffset))

IPPAPI (IppStatus, ippiResizeGetSrcOffset_16u, (
    IppiResizeSpec_32f* pSpec,
    IppiPoint dstOffset, IppiPoint* srcOffset))

IPPAPI (IppStatus, ippiResizeGetSrcOffset_16s, (
    IppiResizeSpec_32f* pSpec,
    IppiPoint dstOffset, IppiPoint* srcOffset))

IPPAPI (IppStatus, ippiResizeGetSrcOffset_32f, (
    IppiResizeSpec_32f* pSpec,
    IppiPoint dstOffset, IppiPoint* srcOffset))

IPPAPI (IppStatus, ippiResizeGetSrcOffset_64f, (
    IppiResizeSpec_64f* pSpec,
    IppiPoint dstOffset, IppiPoint* srcOffset))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeNearestInit
//                      ippiResizeLinearInit
//                      ippiResizeCubicInit
//                      ippiResizeLanczosInit
//                      ippiResizeSuperInit
//
//  Purpose:            Initializes the Spec structure for the Resize transform
//                      by different interpolation methods
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    dstSize           Size of the output image (in pixels)
//    valueB            The first parameter (B) for specifying Cubic filters
//    valueC            The second parameter (C) for specifying Cubic filters
//    numLobes          The parameter for specifying Lanczos (2 or 3) or Hahn (3 or 4) filters
//    pInitBuf          Pointer to the temporal buffer for several filter initialization
//    pSpec             Pointer to the Spec structure for resize filter
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error in the following cases:
//                              -  if width or height of the source or destination image is negative,
//                              -  if the source image size is less than a filter size of the chosen
//                                 interpolation method (except ippiResizeSuperInit).
//                              -  if one of the specified dimensions of the source image is less than
//                                 the corresponding dimension of the destination image
//                                 (for ippiResizeSuperInit only).
//    ippStsNotSupportedModeErr Indicates an error if the requested mode is not supported.
//
//  Notes/References:
//    1. The equation shows the family of cubic filters:
//           ((12-9B-6C)*|x|^3 + (-18+12B+6C)*|x|^2                  + (6-2B)  ) / 6   for |x| < 1
//    K(x) = ((   -B-6C)*|x|^3 + (    6B+30C)*|x|^2 + (-12B-48C)*|x| + (8B+24C)) / 6   for 1 <= |x| < 2
//           0   elsewhere
//    Some values of (B,C) correspond to known cubic splines: Catmull-Rom (B=0,C=0.5), B-Spline (B=1,C=0) and other.
//      Mitchell, Don P.; Netravali, Arun N. (Aug. 1988). "Reconstruction filters in computer graphics"
//      http://www.mentallandscape.com/Papers_siggraph88.pdf
//
//    2. Hahn filter does not supported now.
//    3. The implemented interpolation algorithms have the following filter sizes: Nearest Neighbor 1x1,
//       Linear 2x2, Cubic 4x4, 2-lobed Lanczos 4x4, 3-lobed Lanczos 6x6.
*/


IPPAPI (IppStatus, ippiResizeLinearInit_8u, (
    IppiSize srcSize, IppiSize dstSize,
    IppiResizeSpec_32f* pSpec))

IPPAPI (IppStatus, ippiResizeLinearInit_16u, (
    IppiSize srcSize, IppiSize dstSize,
    IppiResizeSpec_32f* pSpec))

IPPAPI (IppStatus, ippiResizeLinearInit_16s, (
    IppiSize srcSize, IppiSize dstSize,
    IppiResizeSpec_32f* pSpec))

IPPAPI (IppStatus, ippiResizeLinearInit_32f, (
    IppiSize srcSize, IppiSize dstSize,
    IppiResizeSpec_32f* pSpec))

IPPAPI (IppStatus, ippiResizeLinearInit_64f, (
    IppiSize srcSize, IppiSize dstSize,
    IppiResizeSpec_64f* pSpec))

IPPAPI (IppStatus, ippiResizeCubicInit_8u, (
    IppiSize srcSize, IppiSize dstSize, Ipp32f valueB, Ipp32f valueC,
    IppiResizeSpec_32f* pSpec, Ipp8u* pInitBuf))

IPPAPI (IppStatus, ippiResizeCubicInit_16u, (
    IppiSize srcSize, IppiSize dstSize, Ipp32f valueB, Ipp32f valueC,
    IppiResizeSpec_32f* pSpec, Ipp8u* pInitBuf))

IPPAPI (IppStatus, ippiResizeCubicInit_16s, (
    IppiSize srcSize, IppiSize dstSize, Ipp32f valueB, Ipp32f valueC,
    IppiResizeSpec_32f* pSpec, Ipp8u* pInitBuf))

IPPAPI (IppStatus, ippiResizeCubicInit_32f, (
    IppiSize srcSize, IppiSize dstSize, Ipp32f valueB, Ipp32f valueC,
    IppiResizeSpec_32f* pSpec, Ipp8u* pInitBuf))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeNearest
//                      ippiResizeLinear
//                      ippiResizeCubic
//                      ippiResizeLanczos
//                      ippiResizeSuper
//
//  Purpose:            Changes an image size by different interpolation methods
//
//  Parameters:
//    pSrc              Pointer to the source image
//    srcStep           Distance (in bytes) between of consecutive lines in the source image
//    pDst              Pointer to the destination image
//    dstStep           Distance (in bytes) between of consecutive lines in the destination image
//    dstOffset         Offset of tiled image respectively destination image origin
//    dstSize           Size of the destination image (in pixels)
//    border            Type of the border
//    borderValue       Pointer to the constant value(s) if border type equals ippBorderConstant
//    pSpec             Pointer to the Spec structure for resize filter
//    pBuffer           Pointer to the work buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of output image is zero
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//    ippStsContextMatchErr     Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNotSupportedModeErr Indicates an error if requested mode is currently not supported
//    ippStsSizeErr             Indicates an error if width or height of the destination image
//                              is negative
//    ippStsStepErr             Indicates an error if the step value is not data type multiple
//    ippStsOutOfRangeErr       Indicates an error if the destination image offset point is outside the
//                              destination image origin
//    ippStsSizeWrn             Indicates a warning if the destination image size is more than
//                              the destination image origin size
//
//  Notes:
//    1. Supported border types are ippBorderInMemory and ippBorderReplicate
//       (except Nearest Neighbor and Super Sampling methods).
//    2. Hahn filter does not supported now.
*/

IPPAPI (IppStatus, ippiResizeLinear_8u_C1R, (
    const Ipp8u* pSrc, Ipp32s srcStep,
    Ipp8u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp8u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_8u_C3R, (
    const Ipp8u* pSrc, Ipp32s srcStep,
    Ipp8u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp8u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_8u_C4R, (
    const Ipp8u* pSrc, Ipp32s srcStep,
    Ipp8u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp8u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_16u_C1R, (
    const Ipp16u* pSrc, Ipp32s srcStep,
    Ipp16u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_16u_C3R, (
    const Ipp16u* pSrc, Ipp32s srcStep,
    Ipp16u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_16u_C4R, (
    const Ipp16u* pSrc, Ipp32s srcStep,
    Ipp16u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_16s_C1R, (
    const Ipp16s* pSrc, Ipp32s srcStep,
    Ipp16s* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16s* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_16s_C3R, (
    const Ipp16s* pSrc, Ipp32s srcStep,
    Ipp16s* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16s* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_16s_C4R, (
    const Ipp16s* pSrc, Ipp32s srcStep,
    Ipp16s* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16s* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_32f_C1R, (
    const Ipp32f* pSrc, Ipp32s srcStep,
    Ipp32f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp32f* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_32f_C3R, (
    const Ipp32f* pSrc, Ipp32s srcStep,
    Ipp32f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp32f* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_32f_C4R, (
    const Ipp32f* pSrc, Ipp32s srcStep,
    Ipp32f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp32f* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_64f_C1R, (
    const Ipp64f* pSrc, Ipp32s srcStep,
    Ipp64f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp64f* borderValue,
    IppiResizeSpec_64f* pSpec,
    Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_64f_C3R, (
    const Ipp64f* pSrc, Ipp32s srcStep,
    Ipp64f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp64f* borderValue,
    IppiResizeSpec_64f* pSpec,
    Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_64f_C4R, (
    const Ipp64f* pSrc, Ipp32s srcStep,
    Ipp64f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp64f* borderValue,
    IppiResizeSpec_64f* pSpec,
    Ipp8u* pBuffer))


IPPAPI (IppStatus, ippiResizeCubic_8u_C1R, (
    const Ipp8u* pSrc, Ipp32s srcStep,
    Ipp8u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp8u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_8u_C3R, (
    const Ipp8u* pSrc, Ipp32s srcStep,
    Ipp8u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp8u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_8u_C4R, (
    const Ipp8u* pSrc, Ipp32s srcStep,
    Ipp8u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp8u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_16u_C1R, (
    const Ipp16u* pSrc, Ipp32s srcStep,
    Ipp16u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_16u_C3R, (
    const Ipp16u* pSrc, Ipp32s srcStep,
    Ipp16u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_16u_C4R, (
    const Ipp16u* pSrc, Ipp32s srcStep,
    Ipp16u* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16u* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_16s_C1R, (
    const Ipp16s* pSrc, Ipp32s srcStep,
    Ipp16s* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16s* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_16s_C3R, (
    const Ipp16s* pSrc, Ipp32s srcStep,
    Ipp16s* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16s* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_16s_C4R, (
    const Ipp16s* pSrc, Ipp32s srcStep,
    Ipp16s* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp16s* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_32f_C1R, (
    const Ipp32f* pSrc, Ipp32s srcStep,
    Ipp32f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp32f* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_32f_C3R, (
    const Ipp32f* pSrc, Ipp32s srcStep,
    Ipp32f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp32f* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_32f_C4R, (
    const Ipp32f* pSrc, Ipp32s srcStep,
    Ipp32f* pDst, Ipp32s dstStep, IppiPoint dstOffset, IppiSize dstSize,
    IppiBorderType border, Ipp32f* borderValue,
    IppiResizeSpec_32f* pSpec, Ipp8u* pBuffer))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:      ippiThreshold_GT_8u_C1R
//              ippiThreshold_GT_8u_C3R
//              ippiThreshold_GT_8u_AC4R
//              ippiThreshold_GT_16s_C1R
//              ippiThreshold_GT_16s_C3R
//              ippiThreshold_GT_16s_AC4R
//              ippiThreshold_GT_32f_C1R
//              ippiThreshold_GT_32f_C3R
//              ippiThreshold_GT_32f_AC4R
//              ippiThreshold_GT_8u_C1IR
//              ippiThreshold_GT_8u_C3IR
//              ippiThreshold_GT_8u_AC4IR
//              ippiThreshold_GT_16s_C1IR
//              ippiThreshold_GT_16s_C3IR
//              ippiThreshold_GT_16s_AC4IR
//              ippiThreshold_GT_32f_C1IR
//              ippiThreshold_GT_32f_C3IR
//              ippiThreshold_GT_32f_AC4IR
//              ippiThreshold_GT_16u_C1R
//              ippiThreshold_GT_16u_C3R
//              ippiThreshold_GT_16u_AC4R
//              ippiThreshold_GT_16u_C1IR
//              ippiThreshold_GT_16u_C3IR
//              ippiThreshold_GT_16u_AC4IR
//
//  Purpose:   Performs threshold operation using the comparison "greater than"
//  Returns:
//   ippStsNoErr       OK
//   ippStsNullPtrErr  One of the pointers is NULL
//   ippStsSizeErr     roiSize has a field with zero or negative value
//   ippStsStepErr     One of the step values is zero or negative
//
//  Parameters:
//   pSrc       Pointer to the source image
//   srcStep    Step through the source image
//   pDst       Pointer to the destination image
//   dstStep    Step through the destination image
//   pSrcDst    Pointer to the source/destination image (in-place flavors)
//   srcDstStep Step through the source/destination image (in-place flavors)
//   roiSize    Size of the ROI
//   threshold  Threshold level value (array of values for multi-channel data)
*/
IPPAPI(IppStatus,ippiThreshold_GT_8u_C1R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, Ipp8u threshold))
IPPAPI(IppStatus,ippiThreshold_GT_16s_C1R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, Ipp16s threshold))
IPPAPI(IppStatus,ippiThreshold_GT_32f_C1R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f threshold))
IPPAPI(IppStatus,ippiThreshold_GT_8u_C3R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, const Ipp8u threshold[3]))
IPPAPI(IppStatus,ippiThreshold_GT_16s_C3R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, const Ipp16s threshold[3]))
IPPAPI(IppStatus,ippiThreshold_GT_32f_C3R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, const Ipp32f threshold[3]))
IPPAPI(IppStatus,ippiThreshold_GT_8u_C1IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp8u threshold))
IPPAPI(IppStatus,ippiThreshold_GT_16s_C1IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp16s threshold))
IPPAPI(IppStatus,ippiThreshold_GT_32f_C1IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp32f threshold))
IPPAPI(IppStatus,ippiThreshold_GT_8u_C3IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp8u threshold[3]))
IPPAPI(IppStatus,ippiThreshold_GT_16s_C3IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16s threshold[3]))
IPPAPI(IppStatus,ippiThreshold_GT_32f_C3IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp32f threshold[3]))
IPPAPI(IppStatus,ippiThreshold_GT_16u_C1R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, Ipp16u threshold))
IPPAPI(IppStatus,ippiThreshold_GT_16u_C3R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, const Ipp16u threshold[3]))
IPPAPI(IppStatus,ippiThreshold_GT_16u_C1IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp16u threshold))
IPPAPI(IppStatus,ippiThreshold_GT_16u_C3IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16u threshold[3]))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:      ippiThreshold_GTVal_8u_C1R
//              ippiThreshold_GTVal_8u_C3R
//              ippiThreshold_GTVal_8u_AC4R
//              ippiThreshold_GTVal_16s_C1R
//              ippiThreshold_GTVal_16s_C3R
//              ippiThreshold_GTVal_16s_AC4R
//              ippiThreshold_GTVal_32f_C1R
//              ippiThreshold_GTVal_32f_C3R
//              ippiThreshold_GTVal_32f_AC4R
//              ippiThreshold_GTVal_8u_C1IR
//              ippiThreshold_GTVal_8u_C3IR
//              ippiThreshold_GTVal_8u_AC4IR
//              ippiThreshold_GTVal_16s_C1IR
//              ippiThreshold_GTVal_16s_C3IR
//              ippiThreshold_GTVal_16s_AC4IR
//              ippiThreshold_GTVal_32f_C1IR
//              ippiThreshold_GTVal_32f_C3IR
//              ippiThreshold_GTVal_32f_AC4IR
//              ippiThreshold_GTVal_8u_C4R
//              ippiThreshold_GTVal_16s_C4R
//              ippiThreshold_GTVal_32f_C4R
//              ippiThreshold_GTVal_8u_C4IR
//              ippiThreshold_GTVal_16s_C4IR
//              ippiThreshold_GTVal_32f_C4IR
//              ippiThreshold_GTVal_16u_C1R
//              ippiThreshold_GTVal_16u_C3R
//              ippiThreshold_GTVal_16u_AC4R
//              ippiThreshold_GTVal_16u_C1IR
//              ippiThreshold_GTVal_16u_C3IR
//              ippiThreshold_GTVal_16u_AC4IR
//              ippiThreshold_GTVal_16u_C4R
//              ippiThreshold_GTVal_16u_C4IR
//
//  Purpose:  Performs thresholding of pixel values: pixels that are
//            greater than threshold, are set to a specified value
//  Returns:
//   ippStsNoErr       OK
//   ippStsNullPtrErr  One of the pointers is NULL
//   ippStsSizeErr     roiSize has a field with zero or negative value
//   ippStsStepErr     One of the step values is zero or negative
//
//  Parameters:
//   pSrc       Pointer to the source image
//   srcStep    Step through the source image
//   pDst       Pointer to the destination image
//   dstStep    Step through the destination image
//   pSrcDst    Pointer to the source/destination image (in-place flavors)
//   srcDstStep Step through the source/destination image (in-place flavors)
//   roiSize    Size of the ROI
//   threshold  Threshold level value (array of values for multi-channel data)
//   value      The output value (array or values for multi-channel data)
*/
IPPAPI(IppStatus,ippiThreshold_GTVal_8u_C1R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, Ipp8u threshold,
       Ipp8u value))
IPPAPI(IppStatus,ippiThreshold_GTVal_16s_C1R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, Ipp16s threshold,
       Ipp16s value))
IPPAPI(IppStatus,ippiThreshold_GTVal_32f_C1R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f threshold,
       Ipp32f value))
IPPAPI(IppStatus,ippiThreshold_GTVal_8u_C3R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, const Ipp8u threshold[3],
       const Ipp8u value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16s_C3R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, const Ipp16s threshold[3],
       const Ipp16s value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_32f_C3R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, const Ipp32f threshold[3],
       const Ipp32f value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_8u_C1IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp8u threshold, Ipp8u value))
IPPAPI(IppStatus,ippiThreshold_GTVal_16s_C1IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp16s threshold, Ipp16s value))
IPPAPI(IppStatus,ippiThreshold_GTVal_32f_C1IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp32f threshold, Ipp32f value))
IPPAPI(IppStatus,ippiThreshold_GTVal_8u_C3IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp8u threshold[3], const Ipp8u value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16s_C3IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16s threshold[3], const Ipp16s value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_32f_C3IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp32f threshold[3], const Ipp32f value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_8u_C4R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, const Ipp8u threshold[4],
       const Ipp8u value[4]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16s_C4R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, const Ipp16s threshold[4],
       const Ipp16s value[4]))
IPPAPI(IppStatus,ippiThreshold_GTVal_32f_C4R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, const Ipp32f threshold[4],
       const Ipp32f value[4]))
IPPAPI(IppStatus,ippiThreshold_GTVal_8u_C4IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp8u threshold[4], const Ipp8u value[4]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16s_C4IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16s threshold[4], const Ipp16s value[4]))
IPPAPI(IppStatus,ippiThreshold_GTVal_32f_C4IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp32f threshold[4], const Ipp32f value[4]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16u_C1R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, Ipp16u threshold,
       Ipp16u value))
IPPAPI(IppStatus,ippiThreshold_GTVal_16u_C3R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, const Ipp16u threshold[3],
       const Ipp16u value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16u_C1IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp16u threshold, Ipp16u value))
IPPAPI(IppStatus,ippiThreshold_GTVal_16u_C3IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16u threshold[3], const Ipp16u value[3]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16u_C4R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, const Ipp16u threshold[4],
       const Ipp16u value[4]))
IPPAPI(IppStatus,ippiThreshold_GTVal_16u_C4IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16u threshold[4], const Ipp16u value[4]))

/* ////////////////////////////////////////////////////////////////////////////
//  Names:      ippiThreshold_LTVal_8u_C1R
//              ippiThreshold_LTVal_8u_C3R
//              ippiThreshold_LTVal_8u_AC4R
//              ippiThreshold_LTVal_16s_C1R
//              ippiThreshold_LTVal_16s_C3R
//              ippiThreshold_LTVal_16s_AC4R
//              ippiThreshold_LTVal_32f_C1R
//              ippiThreshold_LTVal_32f_C3R
//              ippiThreshold_LTVal_32f_AC4R
//              ippiThreshold_LTVal_8u_C1IR
//              ippiThreshold_LTVal_8u_C3IR
//              ippiThreshold_LTVal_8u_AC4IR
//              ippiThreshold_LTVal_16s_C1IR
//              ippiThreshold_LTVal_16s_C3IR
//              ippiThreshold_LTVal_16s_AC4IR
//              ippiThreshold_LTVal_32f_C1IR
//              ippiThreshold_LTVal_32f_C3IR
//              ippiThreshold_LTVal_32f_AC4IR
//              ippiThreshold_LTVal_8u_C4R
//              ippiThreshold_LTVal_16s_C4R
//              ippiThreshold_LTVal_32f_C4R
//              ippiThreshold_LTVal_8u_C4IR
//              ippiThreshold_LTVal_16s_C4IR
//              ippiThreshold_LTVal_32f_C4IR
//              ippiThreshold_LTVal_16u_C1R
//              ippiThreshold_LTVal_16u_C3R
//              ippiThreshold_LTVal_16u_AC4R
//              ippiThreshold_LTVal_16u_C1IR
//              ippiThreshold_LTVal_16u_C3IR
//              ippiThreshold_LTVal_16u_AC4IR
//              ippiThreshold_LTVal_16u_C4R
//              ippiThreshold_LTVal_16u_C4IR
//
//  Purpose:  Performs thresholding of pixel values: pixels that are
//            less than threshold, are set to a specified value
//  Returns:
//   ippStsNoErr       OK
//   ippStsNullPtrErr  One of the pointers is NULL
//   ippStsSizeErr     roiSize has a field with zero or negative value
//   ippStsStepErr     One of the step values is zero or negative
//
//  Parameters:
//   pSrc       Pointer to the source image
//   srcStep    Step through the source image
//   pDst       Pointer to the destination image
//   dstStep    Step through the destination image
//   pSrcDst    Pointer to the source/destination image (in-place flavors)
//   srcDstStep Step through the source/destination image (in-place flavors)
//   roiSize    Size of the ROI
//   threshold  Threshold level value (array of values for multi-channel data)
//   value      The output value (array or values for multi-channel data)
*/
IPPAPI(IppStatus,ippiThreshold_LTVal_8u_C1R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, Ipp8u threshold,
       Ipp8u value))
IPPAPI(IppStatus,ippiThreshold_LTVal_16s_C1R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, Ipp16s threshold,
       Ipp16s value))
IPPAPI(IppStatus,ippiThreshold_LTVal_32f_C1R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f threshold,
       Ipp32f value))
IPPAPI(IppStatus,ippiThreshold_LTVal_8u_C3R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, const Ipp8u threshold[3],
       const Ipp8u value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16s_C3R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, const Ipp16s threshold[3],
       const Ipp16s value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_32f_C3R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, const Ipp32f threshold[3],
       const Ipp32f value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_8u_C1IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp8u threshold, Ipp8u value))
IPPAPI(IppStatus,ippiThreshold_LTVal_16s_C1IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp16s threshold, Ipp16s value))
IPPAPI(IppStatus,ippiThreshold_LTVal_32f_C1IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp32f threshold, Ipp32f value))
IPPAPI(IppStatus,ippiThreshold_LTVal_8u_C3IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp8u threshold[3], const Ipp8u value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16s_C3IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16s threshold[3], const Ipp16s value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_32f_C3IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp32f threshold[3], const Ipp32f value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_8u_C4R,(const Ipp8u* pSrc, int srcStep,
       Ipp8u* pDst, int dstStep, IppiSize roiSize, const Ipp8u threshold[4],
       const Ipp8u value[4]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16s_C4R,(const Ipp16s* pSrc, int srcStep,
       Ipp16s* pDst, int dstStep, IppiSize roiSize, const Ipp16s threshold[4],
       const Ipp16s value[4]))
IPPAPI(IppStatus,ippiThreshold_LTVal_32f_C4R,(const Ipp32f* pSrc, int srcStep,
       Ipp32f* pDst, int dstStep, IppiSize roiSize, const Ipp32f threshold[4],
       const Ipp32f value[4]))
IPPAPI(IppStatus,ippiThreshold_LTVal_8u_C4IR,(Ipp8u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp8u threshold[4], const Ipp8u value[4]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16s_C4IR,(Ipp16s* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16s threshold[4], const Ipp16s value[4]))
IPPAPI(IppStatus,ippiThreshold_LTVal_32f_C4IR,(Ipp32f* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp32f threshold[4], const Ipp32f value[4]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16u_C1R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, Ipp16u threshold,
       Ipp16u value))
IPPAPI(IppStatus,ippiThreshold_LTVal_16u_C3R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, const Ipp16u threshold[3],
       const Ipp16u value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16u_C1IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, Ipp16u threshold, Ipp16u value))
IPPAPI(IppStatus,ippiThreshold_LTVal_16u_C3IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16u threshold[3], const Ipp16u value[3]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16u_C4R,(const Ipp16u* pSrc, int srcStep,
       Ipp16u* pDst, int dstStep, IppiSize roiSize, const Ipp16u threshold[4],
       const Ipp16u value[4]))
IPPAPI(IppStatus,ippiThreshold_LTVal_16u_C4IR,(Ipp16u* pSrcDst, int srcDstStep,
       IppiSize roiSize, const Ipp16u threshold[4], const Ipp16u value[4]))


/*F/////////////////////////////////////////////////////////////////////////////////
//  Name: ippiComputeThreshold_Otsu_8u_C1R
//
//  Purpose: Calculate Otsu theshold value of images
//    Return:
//      ippStsNoErr              Ok
//      ippStsNullPtrErr         One of pointers is NULL
//      ippStsSizeErr            The width or height of images is less or equal zero
//      ippStsStepErr            The steps in images is less ROI
//    Parameters:
//      pSrc                     Pointer to image
//      srcStep                  Image step
//      roiSize                  Size of image ROI
//      pThreshold               Returned Otsu theshold value
//
//F*/

IPPAPI (IppStatus, ippiComputeThreshold_Otsu_8u_C1R, (const Ipp8u* pSrc, int srcStep,
        IppiSize roiSize, Ipp8u* pThreshold))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiFilterBoxBorderGetBufferSize
//  Purpose:            Computes the size of external buffer for FilterBoxBorder
//
//  Parameters:
//    roiSize      Maximum size of the destination image ROI.
//    maskSize     Size of the mask in pixels.
//    dataType     Data type of the image. Possible values are Ipp8u, Ipp16u, Ipp16s, or Ipp32f.
//    numChannels  Number of channels in the image. Possible values are 1, 3, or 4.
//    pBufferSize  Pointer to the size of the external work buffer.
//
//  Return Values:
//    ippStsNoErr Indicates no error.
//    ippStsSizeErr Indicates an error when roiSize is negative, or equal to zero.
//    ippStsMaskSizeErr Indicates an error when mask has an illegal value.
//    ippStsDataTypeErr Indicates an error when dataType has an illegal value.
//    ippStsNumChannelsError Indicates an error when numChannels has an illegal value.
*/
IPPAPI (IppStatus, ippiFilterBoxBorderGetBufferSize,(IppiSize roiSize, IppiSize maskSize, IppDataType dataType, int numChannels, int* pBufferSize))

/* ///////////////////////////////////////////////////////////////////////////
//  Name: ippiFilterBoxBorder_32f_<desc>R / ippiFilterBoxBorder_16u_<desc>R / ippiFilterBoxBorder_8u_<desc>R / ippiFilterBoxBorder_16s_<desc>R
//               <desc>  C1|C3|C4|AC4   (descriptor)
// Purpose:             Blurs an image using a simple box filter
// Parameters:
//   pSrc           Pointer to the source image.
//   srcStep        Distance in bytes between starting points of consecutive lines in the source image.
//   pDst           Pointer to the destination image.
//   dstStep        Distance in bytes between starting points of consecutive lines in the destination image.
//   dstRoiSize     Size of the destination ROI in pixels.
//   maskSize       Size of the mask in pixels.
//   border         Type of border. Possible values are:
//                     ippBorderConst Values of all border pixels are set to constant.
//                     ippBorderRepl Border is replicated from the edge pixels.
//                     ippBorderInMem Border is obtained from the source image pixels in memory.
//                     Mixed borders are also supported. They can be obtained by the bitwise operation OR between ippBorderRepl and ippBorderInMemTop, ippBorderInMemBottom, ippBorderInMemLeft, ippBorderInMemRight.
//   borderValue    Constant value to assign to pixels of the constant border. This parameter is applicable only to the ippBorderConst border type.
//   pBuffer        Pointer to the work buffer.
// Returns:
//   ippStsNoErr       Indicates no error.
//   ippStsNullPtrErr  Indicates an error when pSrc or pDst is NULL.
//   ippStsSizeErr     Indicates an error if roiSize has a field with zero or negative value.
//   ippStsMaskSizeErr Indicates an error if mask has an illegal value.
//   ippStsBorderErr   Indicates an error when border has an illegal value.
*/

IPPAPI(IppStatus, ippiFilterBoxBorder_32f_C1R,(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp32f* borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_32f_C3R,(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp32f borderValue[3],Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_32f_C4R,(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp32f borderValue[4],Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_16u_C1R,(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp16u* borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_16u_C3R,(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp16u borderValue[3],Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_16u_C4R,(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp16u borderValue[3],Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_16s_C1R,(const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep,IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp16s* borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_16s_C3R,(const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp16s borderValue[3],Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_16s_C4R,(const Ipp16s* pSrc, int srcStep, Ipp16s* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp16s borderValue[3],Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_8u_C1R,(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp8u* borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_8u_C3R,(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp8u borderValue[3],Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterBoxBorder_8u_C4R,(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep, IppiSize roiSize,
    IppiSize maskSize, IppiBorderType border,const Ipp8u borderValue[4],Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiFilterMedianBorderGetBufferSize
//
//  Purpose:            Computes the size of the external buffer for median filter with border
//
//  Parameters:
//   roiSize            Size of destination ROI in pixels.
//   maskSize           Size of filter mask.
//   dataType           Data type of the source an desination images.
//   numChannels        Number of channels in the images. Possible value is 1.
//   pBufferSize        Pointer to the size (in bytes) of the external work buffer.
//
//  Return Values:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when pBufferSize is NULL.
//   ippStsSizeErr      Indicates an error when roiSize has a field with negative or zero value.
//   ippStsMaskSizeErr  Indicates an error when maskSize has a field with negative, zero or even value.
//   ippStsDataTypeErr  Indicates an error when dataType has an illegal value.
//   ippStsNumChannelsErr Indicates an error when numChannels has an illegal value.
*/
IPPAPI(IppStatus, ippiFilterMedianBorderGetBufferSize,(IppiSize dstRoiSize, IppiSize maskSize,
    IppDataType dataType, int numChannels, int* pBufferSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiFilterMedianBorder_8u_C1R
//                      ippiFilterMedianBorder_16s_C1R
//                      ippiFilterMedianBorder_16u_C1R
//                      ippiFilterMedianBorder_32f_C1R
//  Purpose:            Perform median filtering of an image with border
//
//  Parameters:
//   pSrc               Pointer to the source image ROI.
//   srcStep            Distance in bytes between starting points of consecutive lines in the sorce image.
//   pDst               Pointer to the destination image ROI.
//   dstStep            Distance in bytes between starting points of consecutive lines in the destination image.
//   dstRoiSize         Size of destination ROI in pixels.
//   maskSize           Size of filter mask.
//   borderType         Type of border.
//   borderValue        Constant value to assign to pixels of the constant border. This parameter is applicable
//                      only to the ippBorderConst border type.
//   pBuffer            Pointer to the work buffer.
//
//  Return Values:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when pSrc, pDst or pBufferSize is NULL.
//   ippStsSizeErr      Indicates an error when roiSize has a field with negative or zero value.
//   ippStsMaskSizeErr  Indicates an error when maskSize has a field with negative, zero or even value.
//   ippStsNotEvenStepErr Indicated an error when one of the step values is not divisible by 4
//                      for floating-point images, or by 2 for short-integer images.
//   ippStsBorderErr    Indicates an error when borderType has illegal value.
*/

IPPAPI(IppStatus, ippiFilterMedianBorder_8u_C1R, (const Ipp8u* pSrc, int srcStep,
    Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterMedianBorder_16s_C1R, (const Ipp16s* pSrc, int srcStep,
    Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp16s borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterMedianBorder_16u_C1R, (const Ipp16u* pSrc, int srcStep,
    Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp16u borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterMedianBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
    Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiFilterMaxBorderGetBufferSize
//                      ippiFilterMinBorderGetBufferSize
//
//  Purpose:            Computes the size of the external buffer for median filter with border
//
//  Parameters:
//   roiSize            Size of destination ROI in pixels.
//   maskSize           Size of mask.
//   dataType           data type of source and destination images.
//   numChannels        Number of channels in the images. Possible values is 1.
//   pBufferSize        Pointer to the size (in bytes) of the external work buffer.
//
//  Return Values:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when pBufferSize is NULL.
//   ippStsSizeErr      Indicates an error when roiSize is negative, or equal to zero.
//   ippStsMaskSizeErr  Indicates an error when maskSize is negative, or equal to zero.
//   ippStsDataTypeErr  Indicates an error when dataType has an illegal value.
//   ippStsNumChannelsErr Indicates an error when numChannels has an illegal value.
*/
IPPAPI(IppStatus, ippiFilterMaxBorderGetBufferSize,(IppiSize dstRoiSize, IppiSize maskSize,
    IppDataType dataType, int numChannels, int* pBufferSize))
IPPAPI(IppStatus, ippiFilterMinBorderGetBufferSize,(IppiSize dstRoiSize, IppiSize maskSize,
    IppDataType dataType, int numChannels, int* pBufferSize))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:      ippiFilterMaxBorder_8u_C1R
//              ippiFilterMaxBorder_8u_C3R
//              ippiFilterMaxBorder_8u_AC4R
//              ippiFilterMaxBorder_8u_C4R
//              ippiFilterMaxBorder_16s_C1R
//              ippiFilterMaxBorder_16s_C3R
//              ippiFilterMaxBorder_16s_AC4R
//              ippiFilterMaxBorder_16s_C4R
//              ippiFilterMaxBorder_16u_C1R
//              ippiFilterMaxBorder_16u_C3R
//              ippiFilterMaxBorder_16u_AC4R
//              ippiFilterMaxBorder_16u_C4R
//              ippiFilterMaxBorder_32f_C1R
//              ippiFilterMaxBorder_32f_C3R
//              ippiFilterMaxBorder_32f_AC4R
//              ippiFilterMaxBorder_32f_C4R
//              ippiFilterMinBorder_8u_C1R
//              ippiFilterMinBorder_8u_C3R
//              ippiFilterMinBorder_8u_AC4R
//              ippiFilterMinBorder_8u_C4R
//              ippiFilterMinBorder_16s_C1R
//              ippiFilterMinBorder_16s_C3R
//              ippiFilterMinBorder_16s_AC4R
//              ippiFilterMinBorder_16s_C4R
//              ippiFilterMinBorder_16u_C1R
//              ippiFilterMinBorder_16u_C3R
//              ippiFilterMinBorder_16u_AC4R
//              ippiFilterMinBorder_16u_C4R
//              ippiFilterMinBorder_32f_C1R
//              ippiFilterMinBorder_32f_C3R
//              ippiFilterMinBorder_32f_AC4R
//              ippiFilterMinBorder_32f_C4R
//
// Purpose:    Max and Min Filter with Border
// Parameters:
//   pSrc               Pointer to the source image ROI.
//   srcStep            Distance in bytes between starting points of consecutive lines in the sorce image.
//   pDst               Pointer to the destination image ROI.
//   dstStep            Distance in bytes between starting points of consecutive lines in the destination image.
//   dstRoiSize         Size of destination ROI in pixels.
//   maskSize           Size of mask.
//   borderType         Type of border.
//   borderValue        Constant value to assign to pixels of the constant border. This parameter is applicable
//                      only to the ippBorderConst border type.
//   pBorderValue       Pointer to constant value to assign to pixels of the constant border. This parameter is applicable
//                      only to the ippBorderConst border type.
//   pBuffer            Pointer to the work buffer.
//
//  Return Values:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when pBuffer is NULL while it must be no NULL.
//   ippStsSizeErr      Indicates an error when roiSize is negative, or equal to zero.
//   ippStsStepErr      Indicates an error when srcStep or dstStep is negative, or equal to zero.
//   ippStsBorderErr    Indicates an error when borderType has illegal value.
*/
IPPAPI(IppStatus, ippiFilterMaxBorder_8u_C1R, (const Ipp8u* pSrc, int srcStep,
    Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_8u_C3R, (const Ipp8u* pSrc, int srcStep,
    Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp8u pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_8u_C4R, (const Ipp8u* pSrc, int srcStep,
    Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp8u pBorderValue[4], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_16s_C1R, (const Ipp16s* pSrc, int srcStep,
    Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp16s borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_16s_C3R, (const Ipp16s* pSrc, int srcStep,
    Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16s pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_16s_C4R, (const Ipp16s* pSrc, int srcStep,
    Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16s pBorderValue[4], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_16u_C1R, (const Ipp16u* pSrc, int srcStep,
    Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp16u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_16u_C3R, (const Ipp16u* pSrc, int srcStep,
    Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16u pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_16u_C4R, (const Ipp16u* pSrc, int srcStep,
    Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16u pBorderValue[4], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
    Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_32f_C3R, (const Ipp32f* pSrc, int srcStep,
    Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp32f pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMaxBorder_32f_C4R, (const Ipp32f* pSrc, int srcStep,
    Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp32f pBorderValue[4], Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterMinBorder_8u_C1R, (const Ipp8u* pSrc, int srcStep,
    Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_8u_C3R, (const Ipp8u* pSrc, int srcStep,
    Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp8u pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_8u_C4R, (const Ipp8u* pSrc, int srcStep,
    Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp8u pBorderValue[4], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_16s_C1R, (const Ipp16s* pSrc, int srcStep,
    Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp16s borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_16s_C3R, (const Ipp16s* pSrc, int srcStep,
    Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16s pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_16s_C4R, (const Ipp16s* pSrc, int srcStep,
    Ipp16s* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16s pBorderValue[4], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_16u_C1R, (const Ipp16u* pSrc, int srcStep,
    Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp16u borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_16u_C3R, (const Ipp16u* pSrc, int srcStep,
    Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16u pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_16u_C4R, (const Ipp16u* pSrc, int srcStep,
    Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp16u pBorderValue[4], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
    Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_32f_C3R, (const Ipp32f* pSrc, int srcStep,
    Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp32f pBorderValue[3], Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterMinBorder_32f_C4R, (const Ipp32f* pSrc, int srcStep,
    Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize,
    IppiBorderType borderType, const Ipp32f pBorderValue[4], Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//                     Bilateral filter function
*/

/* /////////////////////////////////////////////////////////////////////////////
//                     Bilateral filter functions with Border
// /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterBilateralBorderGetBufferSize
//  Purpose:    to define buffer size for bilateral filter
//  Parameters:
//   filter        Type of bilateral filter. Possible value is ippiFilterBilateralGauss.
//   dstRoiSize    Roi size (in pixels) of destination image what will be applied
//                 for processing.
//   radius        Radius of circular neighborhood what defines pixels for calculation.
//   dataType      Data type of the source and desination images. Possible values
//                 are ipp8u and ipp32f.
//   numChannels   Number of channels in the images. Possible values are 1 and 3.
//   distMethod    The type of method for definition of distance beetween pixel untensity.
//                 Possible value is ippDistNormL1.
//   pSpecSize     Pointer to the size (in bytes) of the spec.
//   pBufferSize   Pointer to the size (in bytes) of the external work buffer.
//  Return:
//    ippStsNoErr               OK
//    ippStsNullPtrErr          any pointer is NULL
//    ippStsSizeErr             size of dstRoiSize is less or equal 0
//    ippStsMaskSizeErr         radius is less or equal 0
//    ippStsNotSupportedModeErr filter or distMethod is not supported
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsNumChannelsErr      Indicates an error when numChannels has an illegal value.
*/
IPPAPI(IppStatus,ippiFilterBilateralBorderGetBufferSize,
       (IppiFilterBilateralType filter, IppiSize dstRoiSize,
       int radius, IppDataType dataType, int numChannels,
       IppiDistanceMethodType distMethodType, int *pSpecSize, int *pBufferSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterBilateralBorderInit
//  Purpose:    initialization of Spec for bilateral filter with border
//  Parameters:
//   filter           Type of bilateral filter. Possible value is ippiFilterBilateralGauss.
//   dstRoiSize       Roi size (in pixels) of destination image what will be applied
//                    for processing.
//   radius           Radius of circular neighborhood what defines pixels for calculation.
//   dataType      Data type of the source and desination images. Possible values
//                 are ipp8u and ipp32f.
//   numChannels   Number of channels in the images. Possible values are 1 and 3.
//   distMethodType   The type of method for definition of distance beetween pixel intensity.
//                    Possible value is ippDistNormL1.
//   valSquareSigma   square of Sigma for factor function for pixel intensity
//   posSquareSigma   square of Sigma for factor function for pixel position
//    pSpec           pointer to Spec
//  Return:
//    ippStsNoErr               OK
//    ippStsNullPtrErr          pointer ro Spec is NULL
//    ippStsSizeErr             size of dstRoiSize is less or equal 0
//    ippStsMaskSizeErr         radius is less or equal 0
//    ippStsNotSupportedModeErr filter or distMethod is not supported
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsNumChannelsErr      Indicates an error when numChannels has an illegal value.
//    ippStsBadArgErr           valSquareSigma or posSquareSigma is less or equal 0
*/
IPPAPI(IppStatus,ippiFilterBilateralBorderInit,
       (IppiFilterBilateralType filter, IppiSize dstRoiSize,
       int radius, IppDataType dataType, int numChannels,
       IppiDistanceMethodType distMethod, Ipp32f valSquareSigma,
       Ipp32f posSquareSigma, IppiFilterBilateralSpec *pSpec))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterBilateralBorder_8u_C1R
//              ippiFilterBilateralBorder_8u_C3R
//              ippiFilterBilateralBorder_32f_C1R
//              ippiFilterBilateralBorder_32f_C3R
//  Purpose:    bilateral filter
//  Parameters:
//    pSrc         Pointer to the source image
//    srcStep      Step through the source image
//    pDst         Pointer to the destination image
//    dstStep      Step through the destination image
//    dstRoiSize   Size of the destination ROI
//    borderType   Type of border.
//    borderValue  Pointer to constant value to assign to pixels of the constant border. This parameter is applicable
//                 only to the ippBorderConst border type. If this pointer is NULL than the constant value is equal 0.
//    pSpec        Pointer to filter spec
//    pBuffer      Pointer ro work buffer
//  Return:
//    ippStsNoErr           OK
//    ippStsNullPtrErr      pointer to Src, Dst, Spec or Buffer is NULL
//    ippStsSizeErr         size of dstRoiSize is less or equal 0
//    ippStsContextMatchErr filter Spec is not match
//    ippStsNotEvenStepErr  Indicated an error when one of the step values is not divisible by 4
//                          for floating-point images.
//    ippStsBorderErr       Indicates an error when borderType has illegal value.
*/
IPPAPI(IppStatus,ippiFilterBilateralBorder_8u_C1R,(const Ipp8u *pSrc, int srcStep,
       Ipp8u *pDst, int dstStep, IppiSize dstRoiSize,
       IppiBorderType borderType, Ipp8u *pBorderValue, IppiFilterBilateralSpec *pSpec,
       Ipp8u* pBuffer))
IPPAPI(IppStatus,ippiFilterBilateralBorder_8u_C3R,(const Ipp8u *pSrc, int srcStep,
       Ipp8u *pDst, int dstStep, IppiSize dstRoiSize,
       IppiBorderType borderType, Ipp8u *pBorderValue, IppiFilterBilateralSpec *pSpec,
       Ipp8u* pBuffer))
IPPAPI(IppStatus,ippiFilterBilateralBorder_32f_C1R,(const Ipp32f *pSrc, int srcStep,
       Ipp32f *pDst, int dstStep, IppiSize dstRoiSize,
       IppiBorderType borderType, Ipp32f *pBorderValue, IppiFilterBilateralSpec *pSpec,
       Ipp8u* pBuffer))
IPPAPI(IppStatus,ippiFilterBilateralBorder_32f_C3R,(const Ipp32f *pSrc, int srcStep,
       Ipp32f *pDst, int dstStep, IppiSize dstRoiSize,
       IppiBorderType borderType, Ipp32f *pBorderValue, IppiFilterBilateralSpec *pSpec,
       Ipp8u* pBuffer))


/* ////////////////////////////////////////////////////////////////////////////
//  Names:       ippiLUTPalette
//  Purpose:     intensity transformation of image using the palette lookup table pTable
//  Parameters:
//    pSrc       pointer to the source image
//    srcStep    line offset in input data in bytes
//    alphaValue constant alpha channel
//    pDst       pointer to the destination image
//    dstStep    line offset in output data in bytes
//    roiSize    size of source ROI in pixels
//    pTable     pointer to palette table of size 2^nBitSize or
//               array of pointers to each channel
//    nBitSize   number of valid bits in the source image
//               (range [1,8] for 8u source images and range [1,16] for 16u source images)
//  Returns:
//    ippStsNoErr         no errors
//    ippStsNullPtrErr    pSrc == NULL or pDst == NULL or pTable == NULL
//    ippStsSizeErr       width or height of ROI is less or equal zero
//    ippStsOutOfRangeErr nBitSize is out of range
//  Notes:
*/

IPPAPI(IppStatus, ippiLUTPalette_8u_C3R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
       IppiSize roiSize, const Ipp8u* const pTable[3], int nBitSize))

IPPAPI(IppStatus, ippiLUTPalette_8u_C4R, (const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
       IppiSize roiSize, const Ipp8u* const pTable[4], int nBitSize))


/* =============================================================================
							ippCV
============================================================================= */

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippcvGetLibVersion
//
//  Purpose:    getting of the library version
//
//  Returns:    the structure of information about  version of ippcv library
//
//  Parameters:
//
//  Notes:      not necessary to release the returned structure
*/
IPPAPI( const IppLibraryVersion*, ippcvGetLibVersion, (void) )


/****************************************************************************************\
*                                   Fixed Filters                                        *
\****************************************************************************************/


/* ///////////////////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiFilterSobelCrossGetBufferSize_8u16s_C1R,  ippiFilterLaplacianGetBufferSize_8u16s_C1R,
//          ippiFilterLowpassGetBufferSize_8u_C1R,        ippiFilterSobelCrossGetBufferSize_32f_C1R,
//          ippiFilterLaplacianGetBufferSize_32f_C1R,     ippiFilterLowpassGetBufferSize_32f_C1R
//
//
//  Purpose:    Perform convolution operation with fixed kernels 3x3 and 5x5
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width of the image is less or equal zero
//    ippStsMaskSizeErr        Wrong mask size
//
//  Parameters:
//    roiSize                  The image ROI size
//    mask                     The mask size
//    pBufferSize              The pointer to the buffer size
*/
IPPAPI(IppStatus, ippiFilterLaplacianGetBufferSize_8u16s_C1R,       (IppiSize roiSize, IppiMaskSize mask, int* pBufferSize))

IPPAPI(IppStatus, ippiFilterLaplacianGetBufferSize_32f_C1R,       (IppiSize roiSize, IppiMaskSize mask, int* pBufferSize))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiFilterSobelCrossBorder_8u16s_C1R,  ippiFilterLaplacianBorder_8u16s_C1R
//          ippiFilterLowpassBorder_8u_C1R,        ippiFilterSobelCrossBorder_32f_C1R
//          ippiFilterLowpassBorder_32f_C1R,       ippiFilterLaplacianBorder_32f_C1R
//
//  Purpose:    Perform convolution operation with fixed kernels 3x3 and 5x5
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsMaskSizeErr        Wrong mask size
//    ippStsBadArgErr          Wrong border type or zero divisor
//
//  Parameters:
//    pSrc                     The pointer to the source image
//    srcStep                  The step in the source image
//    pDst                     The pointer to the destination image
//    dstStep                  The step in the destination image
//    roiSize                  The image ROI size
//    mask                     The mask size
//    borderType               The type of the border
//    borderValue              The value for the constant border
//    pBuffer                  The pointer to the working buffer
//    divisor                  The value to divide output pixels by , (for integer functions)
//F*/

IPPAPI(IppStatus, ippiFilterLaplacianBorder_8u16s_C1R, (const Ipp8u* pSrc, int srcStep,
                                      Ipp16s* pDst, int dstStep, IppiSize roiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterLaplacianBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                      Ipp32f* pDst, int dstStep, IppiSize roiSize, IppiMaskSize mask,
                                      IppiBorderType borderType, Ipp32f  borderValue, Ipp8u* pBuffer))


/****************************************************************************************\
*                                   Separable Filters                                    *
\****************************************************************************************/


/* ///////////////////////////////////////////////////////////////////////////////////////////////////
//  Name:  ippiFilterRowBorderPipelineGetBufferSize_8u16s_C1R,   ippiFilterRowBorderPipelineGetBufferSize_8u16s_C3R
//         ippiFilterRowBorderPipelineGetBufferSize_16s_C1R,     ippiFilterRowBorderPipelineGetBufferSize_16s_C3R
//         ippiFilterRowBorderPipelineGetBufferSize_16u_C1R,     ippiFilterRowBorderPipelineGetBufferSize_16u_C3R
//         ippiFilterRowBorderPipelineGetBufferSize_Low_16s_C1R, ippiFilterRowBorderPipelineGetBufferSize_Low_16s_C3R
//         ippiFilterRowBorderPipelineGetBufferSize_32f_C1R,     ippiFilterRowBorderPipelineGetBufferSize_32f_C3R
//
//  Purpose:    Get size of external buffer.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width of the image or kernel size are less or equal zero
//
//  Parameters:
//    roiSize                  The image ROI size
//    kernelSize               The size of the kernel
//    pBufferSize              The pointer to the buffer size
*/

IPPAPI(IppStatus, ippiFilterRowBorderPipelineGetBufferSize_32f_C1R, (IppiSize roiSize, int kernelSize, int* pBufferSize))

IPPAPI(IppStatus, ippiFilterRowBorderPipelineGetBufferSize_32f_C3R, (IppiSize roiSize, int kernelSize, int* pBufferSize))


/*F///////////////////////////////////////////////////////////////////////////////////////////
//  Name:      ippiFilterRowBorderPipeline_8u16s_C1R,  ippiFilterRowBorderPipeline_8u16s_C3R
//             ippiFilterRowBorderPipeline_16s_C1R,    ippiFilterRowBorderPipeline_16s_C3R
//             ippiFilterRowBorderPipeline_16u_C1R,    ippiFilterRowBorderPipeline_16u_C3R
//             ippiFilterRowBorderPipeline_Low_16s_C1R, ippiFilterRowBorderPipeline_Low_16s_C3R
//             ippiFilterRowBorderPipeline_32f_C1R,    ippiFilterRowBorderPipeline_32f_C3R
//
//  Purpose:   Convolves source image rows with the row kernel
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsAnchorErr          The anchor outside the kernel
//    ippStsBadArgErr          Wrong border type or zero divisor
//
//  Parameters:
//    pSrc                     The pointer to the source image
//    srcStep                  The step in the source image
//    ppDst                    The double pointer to the destination image
//    roiSize                  The image ROI size
//    pKernel                  The pointer to the kernel
//    kernelSize               The size of the kernel
//    xAnchor                  The anchor value , (0<=xAnchor<kernelSize)
//    borderType               The type of the border
//    borderValue              The value for the constant border
//    divisor                  The value to divide output pixels by , (for integer functions)
//    pBuffer                  The pointer to the working buffer
//    Notes:                   The output is the doulble pointer to support the circle buffer
//F*/

IPPAPI(IppStatus, ippiFilterRowBorderPipeline_32f_C1R, (const Ipp32f* pSrc, int srcStep, Ipp32f** ppDst,
                                        IppiSize roiSize, const Ipp32f* pKernel, int kernelSize, int xAnchor,
                                        IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterRowBorderPipeline_32f_C3R, (const Ipp32f* pSrc, int srcStep, Ipp32f** ppDst,
                                        IppiSize roiSize, const Ipp32f* pKernel, int kernelSize, int xAnchor,
                                        IppiBorderType borderType, Ipp32f borderValue[3], Ipp8u* pBuffer))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiAdd_8u32f_C1IR,   ippiAdd_8s32f_C1IR,
//              ippiAdd_16u32f_C1IR,
//              ippiAdd_8u32f_C1IMR,  ippiAdd_8s32f_C1IMR,
//              ippiAdd_16u32f_C1IMR, ippiAdd_32f_C1IMR
//
//  Purpose:    Add image to accumulator.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            Step is too small to fit image.
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Arguments:
//    pSrc                     Pointer to source image
//    srcStep                  Step in the source image
//    pMask                    Pointer to mask
//    maskStep                 Step in the mask image
//    pSrcDst                  Pointer to accumulator image
//    srcDstStep               Step in the accumulator image
//    roiSize                  Image size
*/

IPPAPI(IppStatus, ippiAdd_8u32f_C1IR, (const Ipp8u*  pSrc, int srcStep,
                                       Ipp32f* pSrcDst, int srcDstStep,
                                       IppiSize roiSize ))

IPPAPI(IppStatus, ippiAdd_16u32f_C1IR, (const Ipp16u*  pSrc, int srcStep,
                                        Ipp32f* pSrcDst, int srcDstStep,
                                        IppiSize roiSize ))


IPPAPI(IppStatus, ippiAdd_8u32f_C1IMR,(const Ipp8u*  pSrc, int srcStep,
                                       const Ipp8u* pMask, int maskStep,
                                       Ipp32f* pSrcDst, int srcDstStep,
                                       IppiSize roiSize ))

IPPAPI(IppStatus, ippiAdd_16u32f_C1IMR,(const Ipp16u*  pSrc, int srcStep,
                                        const Ipp8u* pMask, int maskStep,
                                        Ipp32f* pSrcDst, int srcDstStep,
                                        IppiSize roiSize ))

IPPAPI(IppStatus, ippiAdd_32f_C1IMR,  (const Ipp32f* pSrc, int srcStep,
                                       const Ipp8u* pMask, int maskStep,
                                       Ipp32f* pSrcDst, int srcDstStep,
                                       IppiSize roiSize ))




/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiAddSquare_8u32f_C1IR,   ippiAddSquare_8s32f_C1IR,
//          ippiAddSquare_16u32f_C1IR,  ippiAddSquare_32f_C1IR,
//          ippiAddSquare_8u32f_C1IMR,  ippiAddSquare_8s32f_C1IMR,
//          ippiAddSquare_16u32f_C1IMR, ippiAddSquare_32f_C1IMR
//
//  Purpose:    Add squared image (i.e. multiplied by itself) to accumulator.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            Step is too small to fit image.
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Arguments:
//    pSrc                     Pointer to source image
//    srcStep                  Step in the source image
//    pMask                    Pointer to mask
//    maskStep                 Step in the mask image
//    pSrcDst                  Pointer to accumulator image
//    srcDstStep               Step in the accumulator image
//    roiSize                  Image size
*/

IPPAPI(IppStatus, ippiAddSquare_8u32f_C1IR, (const Ipp8u*  pSrc, int srcStep,
                                             Ipp32f* pSrcDst, int srcDstStep,
                                             IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddSquare_16u32f_C1IR, (const Ipp16u*  pSrc, int srcStep,
                                              Ipp32f* pSrcDst, int srcDstStep,
                                              IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddSquare_32f_C1IR,   (const Ipp32f* pSrc, int srcStep,
                                             Ipp32f* pSrcDst, int srcDstStep,
                                             IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddSquare_8u32f_C1IMR,(const Ipp8u* pSrc, int srcStep,
                                             const Ipp8u* pMask, int maskStep,
                                             Ipp32f* pSrcDst, int srcDstStep,
                                             IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddSquare_16u32f_C1IMR,(const Ipp16u* pSrc, int srcStep,
                                              const Ipp8u* pMask, int maskStep,
                                              Ipp32f* pSrcDst, int srcDstStep,
                                              IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddSquare_32f_C1IMR,  (const Ipp32f* pSrc, int srcStep,
                                             const Ipp8u* pMask, int maskStep,
                                             Ipp32f* pSrcDst, int srcDstStep,
                                             IppiSize roiSize ))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiAddProduct_8u32f_C1IR,   ippiAddProduct_8s32f_C1IR,
//        ippiAddProduct_16u32f_C1IR,  ippiAddProduct_32f_C1IR,
//        ippiAddProduct_8u32f_C1IMR,  ippiAddProduct_8s32f_C1IMR,
//        ippiAddProduct_16u32f_C1IMR, ippiAddProduct_32f_C1IMR
//
//  Purpose:  Add product of two images to accumulator.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            Step is too small to fit image.
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Arguments:
//    pSrc1                    Pointer to first source image
//    src1Step                 Step in the first source image
//    pSrc2                    Pointer to second source image
//    src2Step                 Step in the second source image
//    pMask                    Pointer to mask
//    maskStep                 Step in the mask image
//    pSrcDst                  Pointer to accumulator image
//    srcDstStep               Step in the accumulator image
//    roiSize                  Image size
*/

IPPAPI(IppStatus, ippiAddProduct_8u32f_C1IR, (const Ipp8u*  pSrc1, int src1Step,
                                              const Ipp8u*  pSrc2, int src2Step,
                                              Ipp32f* pSrcDst, int srcDstStep,
                                              IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddProduct_16u32f_C1IR, (const Ipp16u*  pSrc1, int src1Step,
                                               const Ipp16u*  pSrc2, int src2Step,
                                               Ipp32f* pSrcDst, int srcDstStep,
                                               IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddProduct_32f_C1IR,   (const Ipp32f* pSrc1, int src1Step,
                                              const Ipp32f* pSrc2, int src2Step,
                                              Ipp32f* pSrcDst, int srcDstStep,
                                              IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddProduct_8u32f_C1IMR,(const Ipp8u*  pSrc1, int src1Step,
                                              const Ipp8u*  pSrc2, int src2Step,
                                              const Ipp8u* pMask, int maskStep,
                                              Ipp32f* pSrcDst, int srcDstStep,
                                              IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddProduct_16u32f_C1IMR,(const Ipp16u*  pSrc1, int src1Step,
                                               const Ipp16u*  pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               Ipp32f* pSrcDst, int srcDstStep,
                                               IppiSize roiSize ))

IPPAPI(IppStatus, ippiAddProduct_32f_C1IMR,  (const Ipp32f* pSrc1, int src1Step,
                                              const Ipp32f* pSrc2, int src2Step,
                                              const Ipp8u* pMask, int maskStep,
                                              Ipp32f* pSrcDst, int srcDstStep,
                                              IppiSize roiSize ))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiAddWeighted_8u32f_C1IR,  ippiAddWeighted_8s32f_C1IR,
//        ippiAddWeighted_16u32f_C1IR, ippiAddWeighted_32f_C1IR,
//        ippiAddWeighted_8u32f_C1IMR, ippiAddWeighted_8s32f_C1IMR,
//        ippiAddWeighted_16u32f_C1IMR,ippiAddWeighted_32f_C1IMR
//        ippiAddWeighted_32f_C1R
//
//  Purpose:  Add image, multiplied by alpha, to accumulator, multiplied by (1 - alpha).
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            Step is too small to fit image.
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Arguments:
//    pSrc1                    Pointer to first source image
//    src1Step                 Step in the first source image
//    pSrc2                    Pointer to second source image
//    src2Step                 Step in the second source image
//    pMask                    Pointer to mask
//    maskStep                 Step in the mask image
//    pSrcDst                  Pointer to accumulator image
//    srcDstStep               Step in the accumulator image
//    pDst                     Pointer to destination image
//    dstStep                  Step in the destination image
//    roiSize                  Image size
//    alpha                    Weight of source image
*/

IPPAPI(IppStatus, ippiAddWeighted_8u32f_C1IR, (const Ipp8u*  pSrc, int srcStep,
                                               Ipp32f* pSrcDst, int srcDstStep,
                                               IppiSize roiSize, Ipp32f alpha ))

IPPAPI(IppStatus, ippiAddWeighted_16u32f_C1IR, (const Ipp16u*  pSrc, int srcStep,
                                                Ipp32f* pSrcDst, int srcDstStep,
                                                IppiSize roiSize, Ipp32f alpha ))

IPPAPI(IppStatus, ippiAddWeighted_32f_C1IR,   (const Ipp32f* pSrc, int srcStep,
                                               Ipp32f* pSrcDst, int srcDstStep,
                                               IppiSize roiSize, Ipp32f alpha ))

IPPAPI(IppStatus, ippiAddWeighted_8u32f_C1IMR,(const Ipp8u* pSrc, int srcStep,
                                               const Ipp8u* pMask, int maskStep,
                                               Ipp32f* pSrcDst, int srcDstStep,
                                               IppiSize roiSize, Ipp32f alpha ))

IPPAPI(IppStatus, ippiAddWeighted_16u32f_C1IMR,(const Ipp16u* pSrc, int srcStep,
                                                const Ipp8u* pMask, int maskStep,
                                                Ipp32f* pSrcDst, int srcDstStep,
                                                IppiSize roiSize, Ipp32f alpha ))

IPPAPI(IppStatus, ippiAddWeighted_32f_C1IMR,  (const Ipp32f* pSrc, int srcStep,
                                               const Ipp8u* pMask, int maskStep,
                                               Ipp32f* pSrcDst, int srcDstStep,
                                               IppiSize roiSize, Ipp32f alpha ))


/*F////////////////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiCopySubpixIntersect_8u_C1R,              ippiCopySubpixIntersect_16u_C1R,
//        ippiCopySubpixIntersect_8u16u_C1R_Sfs,       ippiCopySubpixIntersect_16u32f_C1R,
//        ippiCopySubpixIntersect_8u32f_C1R,           ippiCopySubpixIntersect_32f_C1R
//
//  Purpose:   finds intersection of centered window in the source image and copies
//             in to destination image with the border
//             border pixel are taken from the source image or replicated if they are outside it
//
//  Returns:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc                     Pointer to source image
//    srcStep                  Step in source image
//    srcRoiSize               Source image ROI size.
//    pDst                     Pointer to destination image
//    dstStep                  Step in destination image
//    dstRoiSize               Destination image ROI size.
//    point                    Center of dst window in src image (subpixel)
//    pMin                     Top left corner of dst filled part
//    pMax                     Bottom right corner of dst filled part
//    scaleFactor              Output scale factor, >= 0
//
//  Notes:                     For integer point.x or point.y pixels from the last row
//                             or column are not copied. Branches are possible.
//F*/

IPPAPI(IppStatus, ippiCopySubpixIntersect_8u_C1R,        (const Ipp8u* pSrc, int srcStep,
                  IppiSize srcRoiSize, Ipp8u* pDst, int dstStep, IppiSize dstRoiSize,
                  IppiPoint_32f point, IppiPoint *pMin, IppiPoint *pMax))

IPPAPI(IppStatus, ippiCopySubpixIntersect_8u32f_C1R,     (const Ipp8u* pSrc, int srcStep,
                  IppiSize srcRoiSize, Ipp32f* pDst, int dstStep, IppiSize dstRoiSize,
                  IppiPoint_32f point, IppiPoint *pMin, IppiPoint *pMax))

IPPAPI(IppStatus, ippiCopySubpixIntersect_32f_C1R,       (const Ipp32f* pSrc, int srcStep,
                  IppiSize srcRoiSize, Ipp32f* pDst, int dstStep, IppiSize dstRoiSize,
                  IppiPoint_32f point, IppiPoint *pMin, IppiPoint *pMax))


/****************************************************************************************\
*                                    Universal Pyramids                                  *
\****************************************************************************************/

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiPyramidGetSize
//
//  Purpose:    Computes the size of the structure for pyramids and the size
//              of the required work buffer (in bytes)
//
//  Arguments:
//    pPyrSize              Pointer to the size value of pyramid structure
//    pSizeBuf              Pointer to the size value of the pyramid external work buffer
//    level                 Maximal number pyramid level.
//    roiSize               Zero level image ROI size.
//    rate                  Neighbour levels ratio (1<rate<=10)
//
//  Return:
//    ippStsNoErr           Ok
//    ippStsNullPtrErr      One of pointers is NULL
//    ippStsSizeErr         roiSize has a field with zero or negative value.
//    ippStsBadArgErr       level is equal to or less than 0 or if rate has wrong value.
*/

IPPAPI( IppStatus, ippiPyramidGetSize,(int* pPyrSize, int* pBufSize, int level, IppiSize roiSize, Ipp32f rate))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiPyramidInit
//
//  Purpose:    Initializes structure for pyramids, calculates ROI for layers.
//
//  Arguments:
//    pPyr                  Pointer to the pointer to the pyramid structure.
//    level                 Maximal number pyramid level.
//    roiSize               Zero level image ROI size.
//    rate                  Neighbour levels ratio (1<rate<=10).
//    pPyrBuffer            Pointer to the buffer to initialize structure for pyramids.
//    pBuffer               Pointer to the work buffer.
//
//  Return:
//    ippStsNoErr           Ok
//    ippStsNullPtrErr      One of pointers is NULL
//    ippStsSizeErr         roiSize has a field with zero or negative value.
//    ippStsBadArgErr       level is equal to or less than 0 or if rate has wrong value.
*/

IPPAPI( IppStatus, ippiPyramidInit,( IppiPyramid** pPyr, int level, IppiSize roiSize, Ipp32f rate, Ipp8u* pPyrBuffer, Ipp8u* pBuffer))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiPyramidLayerDownGetSize_8u_C1R,   ippiPyramidLayerDownGetSize_8u_C3R
//              ippiPyramidLayerDownGetSize_16u_C1R,  ippiPyramidLayerDownGetSize_16u_C3R
//              ippiPyramidLayerDownGetSize_32f_C1R,  ippiPyramidLayerDownGetSize_32f_C3R
//              ippiPyramidLayerUpGetSize_8u_C1R,     ippiPyramidLayerUpGetSize_8u_C3R
//              ippiPyramidLayerUpGetSize_16u_C1R,    ippiPyramidLayerUpGetSize_16u_C3R
//              ippiPyramidLayerUpGetSizec_32f_C1R,   ippiPyramidLayerUpGetSize_32f_C3R
//
//  Purpose:    Calculates the size of structure for creating a lower(an upper) pyramid layer and the size
//              of the temporary  buffer (in bytes).
//
//  Arguments:
//    srcRoi                   Source image ROI size.
//    dstRoi                   Destination image ROI size.
//    rate                     Neighbour levels ratio (1<rate<=10)
//    kerSize                  Kernel size
//    pStateSize               Pointer to the size value of pyramid state structure.
//    pBufSize                 Pointer to the size value of the external work buffer.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsBadArgErr          Bad rate or kernel size
//
*/

IPPAPI(IppStatus, ippiPyramidLayerDownGetSize_8u_C1R, (IppiSize srcRoi, Ipp32f rate, int kerSize,
                         int* pStateSize, int* pBufSize))
IPPAPI(IppStatus, ippiPyramidLayerDownGetSize_16u_C1R,(IppiSize srcRoi, Ipp32f rate, int kerSize,
                         int* pStateSize, int* pBufSize))
IPPAPI(IppStatus, ippiPyramidLayerDownGetSize_32f_C1R,(IppiSize srcRoi, Ipp32f rate, int kerSize,
                         int* pStateSize, int* pBufSize))
IPPAPI(IppStatus, ippiPyramidLayerDownGetSize_8u_C3R, (IppiSize srcRoi, Ipp32f rate, int kerSize,
                         int* pStateSize, int* pBufSize))
IPPAPI(IppStatus, ippiPyramidLayerDownGetSize_16u_C3R,(IppiSize srcRoi, Ipp32f rate, int kerSize,
                         int* pStateSize, int* pBufSize))
IPPAPI(IppStatus, ippiPyramidLayerDownGetSize_32f_C3R,(IppiSize srcRoi, Ipp32f rate, int kerSize,
                         int* pStateSize, int* pBufSize))


IPPAPI(IppStatus, ippiPyramidLayerUpGetSize_8u_C1R, (IppiSize dstRoi, Ipp32f rate, int kerSize, int* pStateSize))
IPPAPI(IppStatus, ippiPyramidLayerUpGetSize_16u_C1R,(IppiSize dstRoi, Ipp32f rate, int kerSize, int* pStateSize))
IPPAPI(IppStatus, ippiPyramidLayerUpGetSize_32f_C1R,(IppiSize dstRoi, Ipp32f rate, int kerSize, int* pStateSize))
IPPAPI(IppStatus, ippiPyramidLayerUpGetSize_8u_C3R, (IppiSize dstRoi, Ipp32f rate, int kerSize, int* pStateSize))
IPPAPI(IppStatus, ippiPyramidLayerUpGetSize_16u_C3R,(IppiSize dstRoi, Ipp32f rate, int kerSize, int* pStateSize))
IPPAPI(IppStatus, ippiPyramidLayerUpGetSize_32f_C3R,(IppiSize dstRoi, Ipp32f rate, int kerSize, int* pStateSize))




/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiPyramidLayerDownInit_8u_C1R,   ippiPyramidLayerDownInit_8u_C3R
//              ippiPyramidLayerDownInit_16u_C1R,  ippiPyramidLayerDownInit_16u_C3R
//              ippiPyramidLayerDownInit_32f_C1R,  ippiPyramidLayerDownInit_32f_C3R
//              ippiPyramidLayerUpInit_8u_C1R,     ippiPyramidLayerUpInit_8u_C3R
//              ippiPyramidLayerUpInit_16u_C1R,    ippiPyramidLayerUpInit_16u_C3R
//              ippiPyramidLayerUpInit_32f_C1R,    ippiPyramidLayerUpInit_32f_C3R
//
//  Purpose:    Initializesa structure for creating a lower(an upper) pyramid layer.
//
//  Arguments:
//    ppState                  Pointer to the pointer to initialized pyramid state structure
//    srcRoi                   Source image ROI size.
//    dstRoi                   Destination image ROI size.
//    rate                     Neighbour levels ratio (1<rate<=10)
//    pKernel                  Separable symmetric kernel of odd length
//    kerSize                  Kernel size
//    mode                     IPPI_INTER_LINEAR - bilinear interpolation
//    StateBuf                 Pointer to the buffer to initialize state structure for pyramids.
//    pBuffer                  Pointer to the work buffer.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsBadArgErr          Bad mode, rate or kernel size
*/


IPPAPI(IppStatus, ippiPyramidLayerDownInit_8u_C1R, (IppiPyramidDownState_8u_C1R**  ppState, IppiSize srcRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf, Ipp8u* Buffer))
IPPAPI(IppStatus, ippiPyramidLayerDownInit_16u_C1R,(IppiPyramidDownState_16u_C1R** ppState, IppiSize srcRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf, Ipp8u* Buffer))
IPPAPI(IppStatus, ippiPyramidLayerDownInit_32f_C1R,(IppiPyramidDownState_32f_C1R** pState, IppiSize srcRoi,
                         Ipp32f rate, Ipp32f* pKernel, int kerSize, int mode, Ipp8u* StateBuf, Ipp8u* Buffer))
IPPAPI(IppStatus, ippiPyramidLayerDownInit_8u_C3R, (IppiPyramidDownState_8u_C3R**  ppState, IppiSize srcRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf, Ipp8u* Buffer))
IPPAPI(IppStatus, ippiPyramidLayerDownInit_16u_C3R,(IppiPyramidDownState_16u_C3R** ppState, IppiSize srcRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf, Ipp8u* Buffer))
IPPAPI(IppStatus, ippiPyramidLayerDownInit_32f_C3R,(IppiPyramidDownState_32f_C3R** ppState, IppiSize srcRoi,
                         Ipp32f rate, Ipp32f* pKernel, int kerSize, int mode, Ipp8u* StateBuf, Ipp8u* Buffer))


IPPAPI(IppStatus, ippiPyramidLayerUpInit_8u_C1R, (IppiPyramidUpState_8u_C1R**  ppState, IppiSize dstRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf))
IPPAPI(IppStatus, ippiPyramidLayerUpInit_16u_C1R,(IppiPyramidUpState_16u_C1R** ppState, IppiSize dstRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf))
IPPAPI(IppStatus, ippiPyramidLayerUpInit_32f_C1R,(IppiPyramidUpState_32f_C1R** ppState, IppiSize dstRoi,
                         Ipp32f rate, Ipp32f* pKernel, int kerSize, int mode, Ipp8u* StateBuf))
IPPAPI(IppStatus, ippiPyramidLayerUpInit_8u_C3R, (IppiPyramidUpState_8u_C3R**  ppState, IppiSize dstRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf))
IPPAPI(IppStatus, ippiPyramidLayerUpInit_16u_C3R,(IppiPyramidUpState_16u_C3R** ppState, IppiSize dstRoi,
                         Ipp32f rate, Ipp16s* pKernel, int kerSize, int mode, Ipp8u* StateBuf))
IPPAPI(IppStatus, ippiPyramidLayerUpInit_32f_C3R,(IppiPyramidUpState_32f_C3R** ppState, IppiSize dstRoi,
                         Ipp32f rate, Ipp32f* pKernel, int kerSize, int mode, Ipp8u* StateBuf))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiGetPyramidDownROI, ippiGetPyramidUpROI
//
//  Purpose:    Calculate possible size of destination ROI.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            Wrong src roi
//    ippStsBadArgErr          Wrong rate
//
//  Arguments:
//    srcRoi                   Source image ROI size.
//    pDstRoi                  Pointer to destination image ROI size (down).
//    pDstRoiMin               Pointer to minimal destination image ROI size (up).
//    pDstRoiMax               Pointer to maximal destination image ROI size (up).
//    rate                     Neighbour levels ratio (1<rate<=10)
//
//  Notes:                     For up case destination size belongs to interval
//                             max((int)((float)((src-1)*rate)),src+1)<=dst<=
//                             max((int)((float)(src)*rate)),src+1)
*/

IPPAPI(IppStatus, ippiGetPyramidDownROI,(IppiSize srcRoi, IppiSize *pDstRoi, Ipp32f rate))
IPPAPI(IppStatus, ippiGetPyramidUpROI,(IppiSize srcRoi, IppiSize *pDstRoiMin, IppiSize *pDstRoiMax, Ipp32f rate))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiPyramidLayerDown_8u_C1R, ippiPyramidLayerDown_16u_C1R, ippiPyramidLayerDown_32f_C1R
//              ippiPyramidLayerDown_8u_C3R, ippiPyramidLayerDown_16u_C3R, ippiPyramidLayerDown_32f_C3R
//              ippiPyramidLayerUp_8u_C1R,   ippiPyramidLayerUp_16u_C1R,   ippiPyramidLayerUp_32f_C1R
//              ippiPyramidLayerUp_8u_C3R,   ippiPyramidLayerUp_16u_C3R,   ippiPyramidLayerUp_32f_C3R
//
//  Purpose:    Perform downsampling/upsampling of the image with 5x5 gaussian.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of the specified pointers is NULL
//    ippStsSizeErr            The srcRoiSize or dstRoiSize has a fild with zero or negativ value
//    ippStsStepErr            The steps in images are too small
//    ippStsBadArgErr          pState->rate has wrong value
//    ippStsNotEvenStepErr     One of the step values is not divisibly by 4 for floating-point
//                             images, or by 2 for short-integer images.
//  Arguments:
//    pSrc                     Pointer to the source image
//    srcStep                  Step in byte through the source image
//    srcRoiSize               Size of the source image ROI in pixel.
//    dstRoiSize               Size of the destination image ROI in pixel.
//    pDst                     Pointer to destination image
//    dstStep                  Step in byte through the destination image
//    pState                   Pointer to the pyramid layer structure
*/

IPPAPI(IppStatus, ippiPyramidLayerDown_8u_C1R, (const Ipp8u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidDownState_8u_C1R* pState))
IPPAPI(IppStatus, ippiPyramidLayerDown_8u_C3R, (const Ipp8u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidDownState_8u_C3R* pState))
IPPAPI(IppStatus, ippiPyramidLayerDown_16u_C1R,(const Ipp16u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidDownState_16u_C1R* pState))
IPPAPI(IppStatus, ippiPyramidLayerDown_16u_C3R,(const Ipp16u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidDownState_16u_C3R* pState))
IPPAPI(IppStatus, ippiPyramidLayerDown_32f_C1R,(const Ipp32f* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidDownState_32f_C1R* pState))
IPPAPI(IppStatus, ippiPyramidLayerDown_32f_C3R,(const Ipp32f* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidDownState_32f_C3R* pState))

IPPAPI(IppStatus, ippiPyramidLayerUp_8u_C1R, (const Ipp8u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidUpState_8u_C1R* pState))
IPPAPI(IppStatus, ippiPyramidLayerUp_8u_C3R, (const Ipp8u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp8u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidUpState_8u_C3R* pState))
IPPAPI(IppStatus, ippiPyramidLayerUp_16u_C1R,(const Ipp16u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidUpState_16u_C1R* pState))
IPPAPI(IppStatus, ippiPyramidLayerUp_16u_C3R,(const Ipp16u* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp16u* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidUpState_16u_C3R* pState))
IPPAPI(IppStatus, ippiPyramidLayerUp_32f_C1R,(const Ipp32f* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidUpState_32f_C1R* pState))
IPPAPI(IppStatus, ippiPyramidLayerUp_32f_C3R,(const Ipp32f* pSrc, int srcStep, IppiSize srcRoiSize,
                         Ipp32f* pDst, int dstStep, IppiSize dstRoiSize, IppiPyramidUpState_32f_C3R* pState))


/****************************************************************************************\
*                                     Haar Classifier                                    *
\****************************************************************************************/


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:      ippiHaarClassifierInit  ippiTiltedHaarClassifierInit
//
//  Purpose:   Initializes memory for the stage of the Haar classifier.
//
//  Arguments:
//    pState                   The pointer to the pointer to the Haar classifier structure.
//    pFeature                 The pointer to the array of features.
//    pWeight                  The pointer to the array of feature weights.
//    pThreshold               The pointer to the array of classifier thresholds [length].
//    pVal1, pVal2             Pointers to arrays of classifier results [length].
//    pNum                     The pointer to the array of classifier lengths [length].
//    length                   The number of classifiers in the stage.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The number of classifiers or features is less or equal zero
//    ippStsbadArgErr          The bad feature rectangular
//    ippStsMemAllocErr        Memory allocation error
//
//  Notes:  For integer version feature weights pWeight are in Q0, classifier thresholds
//          pThreshold are in QT (see ApplyHaarClassifier), pVal1 and pVal2 are scale as
//          stage thresholds threshold of ApplyHaarClassifier function
*/
IPPAPI(IppStatus, ippiHaarClassifierInit_32f, (IppiHaarClassifier_32f *pState, const IppiRect* pFeature, const Ipp32f* pWeight,
                              const Ipp32f* pThreshold, const Ipp32f* pVal1, const Ipp32f* pVal2, const int* pNum, int length))
IPPAPI(IppStatus, ippiTiltedHaarClassifierInit_32f, (IppiHaarClassifier_32f *pState, const IppiRect* pFeature, const Ipp32f* pWeight,
                              const Ipp32f* pThreshold, const Ipp32f* pVal1, const Ipp32f* pVal2, const int* pNum, int length))


IPPAPI(IppStatus, ippiHaarClassifierGetSize, ( IppDataType dataType, IppiSize roiSize, const int* pNum, int length, int *pSize ))



/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:      ippiTiltHaarFeatures_32f,  ippiTiltHaarFeatures_32s
//  Purpose:   Tilts marked feature on -45 degree
//  Arguments:
//    pMask                    The mask of feature to tilt.
//    flag                     1 - left bottom  -45 degree
//                             0 - left top     +45 degree
//    pState                   The pointer to the Haar classifier structure.
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//  Notes:  The mask length is equal to the number of classifiers in the classifier
//          If pMask[i] != 0 i-th feature is tilted
//          Classifiers with tilted features require two input integral images and
//          can be used by rippiApplyMixedHaarClassifier functions
*/
IPPAPI(IppStatus, ippiTiltHaarFeatures_32f, (const Ipp8u *pMask, int flag, IppiHaarClassifier_32f *pState))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiGetHaarClassifierSize_32f,      ippiGetHaarClassifierSize_32s
//  Purpose:    Returns the size of the Haar classifier.
//  Arguments:
//    pState    Pointer to the Haar classifier structure.
//    pSize        Pointer to the returned value of Haar classifier size.
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
*/
IPPAPI(IppStatus, ippiGetHaarClassifierSize_32f, (IppiHaarClassifier_32f* pState, IppiSize* pSize))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:      ippiApplyHaarClassifier,    ippiApplyMixedHaarClassifier
//  Purpose:   Applies the stage of Haar classifiers to the image.
//  Arguments:
//    pSrc                     The pointer  to the source image of integrals.
//    srcStep                  The step in bytes through the source image.
//    pNorm                    The pointer  to the source image of norm factors.
//    normStep                 The step  in bytes through the image of norm factors.
//    pMask                    The pointer  to the source and destination image of classification decisions.
//    maskStep                 The step  in bytes through the image of classification decisions.
//    pPositive                The pointer to the number of positive decisions.
//    roiSize                  The size of source and destination images ROI in pixels.
//    threshold                The stage threshold value.
//    pState                   The pointer to the Haar classifier structure.
//    scaleFactor              Scale factor for classifier threshold*norm, <= 0
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The number of classifiers or features is less or equal zero
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
*/
IPPAPI(IppStatus, ippiApplyHaarClassifier_32f_C1R, (const Ipp32f* pSrc, int srcStep,
   const Ipp32f* pNorm, int normStep, Ipp8u* pMask, int maskStep, IppiSize roiSize, int *pPositive, Ipp32f threshold, IppiHaarClassifier_32f *pState))

IPPAPI(IppStatus, ippiApplyMixedHaarClassifier_32f_C1R, (const Ipp32f* pSrc, int srcStep, const Ipp32f* pTilt, int tiltStep,
   const Ipp32f* pNorm, int normStep, Ipp8u* pMask, int maskStep, IppiSize roiSize, int *pPositive, Ipp32f threshold, IppiHaarClassifier_32f *pState))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiMinEigenValGetBufferSize_8u32f_C1R, ippiMinEigenValGetBufferSize_32f_C1R
//
//  Purpose: Calculates size of temporary buffer, required to run one of MinEigenVal***
//           functions.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width is less or equal zero or bad window size
//
//  Parameters:
//    roiSize                  roiSize size in pixels
//    apertureSize             Linear size of derivative filter aperture
//    avgWindow                Linear size of averaging window
//    bufferSize               Output parameter. Calculated buffer size.
//F*/

IPPAPI(IppStatus, ippiMinEigenValGetBufferSize_8u32f_C1R, ( IppiSize roiSize, int apertureSize,
                                            int avgWindow, int* bufferSize ))

IPPAPI(IppStatus, ippiMinEigenValGetBufferSize_32f_C1R, ( IppiSize roiSize, int apertureSize,
                                            int avgWindow, int* bufferSize ))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiMinEigenVal_8u32f_C1R, ippiMinEigenVal_32f_C1R
//
//  Purpose: Calculate minimal eigen value of 2x2 autocorrelation gradient matrix
//           for every pixel. Pixels with relatively large minimal eigen values
//           are strong corners on the picture.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//                             or bad window size
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc                     Source image
//    srcStep                  Its step
//    pMinEigenVal             Image, filled with minimal eigen values for every pixel
//    minValStep               Its step
//    roiSize                  ROI size
//    kernType                 Kernel type (Scharr 3x3 or Sobel 3x3, 5x5)
//    apertureSize             Linear size of derivative filter aperture
//    avgWindow                Linear size of averaging window
//    pBuffer                  Preallocated temporary buffer, which size can be calculated
//                             using ippiMinEigenValGetSize function
//F*/

IPPAPI(IppStatus, ippiMinEigenVal_8u32f_C1R, ( const Ipp8u* pSrc, int srcStep,
                                               Ipp32f* pMinEigenVal, int minValStep,
                                               IppiSize roiSize, IppiKernelType kernType,
                                               int apertureSize, int avgWindow, Ipp8u* pBuffer ))

IPPAPI(IppStatus, ippiMinEigenVal_32f_C1R, ( const Ipp32f* pSrc, int srcStep,
                                             Ipp32f* pMinEigenVal, int minValStep,
                                             IppiSize roiSize, IppiKernelType kernType,
                                             int apertureSize, int avgWindow, Ipp8u* pBuffer ))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiMinMaxIndx_8u_C1R,   ippiMinMaxIndx_8s_C1R,
//        ippiMinMaxIndx_16u_C1R,  ippiMinMaxIndx_32f_C1R,
//        ippiMinMaxIndx_8u_C3CR,  ippiMinMaxIndx_8s_C3CR,
//        ippiMinMaxIndx_16u_C3CR, ippiMinMaxIndx_32f_C3CR,
//
//  Purpose:  Finds minimum and maximum values in the image and their coordinates
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc                     Pointer to image
//    srcStep                  Image step
//    roiSize                  Size of image ROI
//    coi                      Index of color channel (1..3) (if color image)
//    pMinVal                  Pointer to minimum value
//    pMaxVal                  Pointer to maximum value
//    pMinIndex                Minimum's coordinates
//    pMaxIndex                Maximum's coordinates
//
//  Notes:
//    Any of output parameters is optional
//F*/

IPPAPI(IppStatus, ippiMinMaxIndx_16u_C1R,( const Ipp16u* pSrc, int srcStep, IppiSize roiSize,
                                           Ipp32f* pMinVal, Ipp32f* pMaxVal,
                                           IppiPoint* pMinIndex, IppiPoint* pMaxIndex ))

IPPAPI(IppStatus, ippiMinMaxIndx_32f_C1R,( const Ipp32f* pSrc, int step, IppiSize roiSize,
                                           Ipp32f* pMinVal, Ipp32f* pMaxVal,
                                           IppiPoint* pMinIndex, IppiPoint* pMaxIndex ))


IPPAPI(IppStatus, ippiMinMaxIndx_8u_C1R,( const Ipp8u* pSrc, int srcStep, IppiSize roiSize,
                                          Ipp32f* pMinVal, Ipp32f* pMaxVal,
                                          IppiPoint* pMinIndex, IppiPoint* pMaxIndex ))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterGaussianGetBufferSize
//
//  Purpose:    Computes the size of the working buffer for the Gaussian filter
//
//  Return:
//     ippStsNoErr          Ok. Any other value indicates an error or a warning.
//     ippStsNullPtrErr     One of the pointers is NULL.
//     ippStsSizeErr        maxRoiSize  has a field with zero or negative value.
//     ippStsDataTypeErr    Indicates an error when dataType has an illegal value.
//     ippStsBadArgErr      Indicates an error if kernelSize is even or is less than 3.
//     ippStsChannelErr     Indicates an error when numChannels has an illegal value.
//
//  Arguments:
//     maxRoiSize           Maximal size of the image ROI in pixels.
//     kernelSize           Size of the Gaussian kernel (odd, greater or equal to 3).
//     dataType             Data type of the source and destination images.
//     numChannels          Number of channels in the images. Possible values are 1 and 3.
//     pSpecSize            Pointer to the computed size (in bytes) of the Gaussian
//                            specification structure.
//     pBufferSize          Pointer to the computed size (in bytes) of the external buffer.
*/
IPPAPI(IppStatus, ippiFilterGaussianGetBufferSize,(IppiSize maxRoiSize, Ipp32u KernelSize,
                  IppDataType dataType, int numChannels, int *pSpecSize, int* pBufferSize))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterGaussianInit
//
//  Purpose:    initialization of Spec for Gaussian filter
//
//  Return:
//     ippStsNoErr          Ok. Any other value indicates an error or a warning.
//     ippStsNullPtrErr     One of the pointers is NULL.
//     ippStsSizeErr        roiSize has a field with zero or negative value.
//     ippStsDataTypeErr    Indicates an error when borderType has an illegal value.
//     ippStsBadArgErr      kernelSize is even or is less than 3.
//     ippStsChannelErr     Indicates an error when numChannels has an illegal value.
//     ippStsBorderErr      Indicates an error condition if borderType has a illegal
//                           value.
//
//  Arguments:
//     roiSize              Size of the image ROI in pixels.
//     kernelSize           Size of the Gaussian kernel (odd, greater or equal to 3).
//     sigma                Standard deviation of the Gaussian kernel.
//     borderType           One of border supported types.
//     dataType             Data type of the source and destination images.
//     numChannels          Number of channels in the images. Possible values are 1 and 3.
//     pSpec                Pointer to the Spec.
//     pBuffer              Pointer to the buffer:
*/

IPPAPI(IppStatus, ippiFilterGaussianInit,(IppiSize roiSize, Ipp32u KernelSize, Ipp32f sigma,
                  IppiBorderType borderType, IppDataType dataType, int numChannels,
                  IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterGaussianBorder
//
//  Purpose:    Applies Gaussian filter with borders
//
//  Return:
//     ippStsNoErr      Ok. Any other value indicates an error or a warning.
//     ippStsNullPtrErr One of the specified pointers is NULL.
//     ippStsSizeErr    roiSize has a field with zero or negative value.
//     ippStsStepErr    Indicates an error condition if srcStep or dstStep is less
//                        than  roiSize.width * <pixelSize>.
//     ippStsNotEvenStepErr One of the step values is not divisible by 4 for floating-point images.
//     ippStsBadArgErr  kernelSize is less than 3 or sigma is less or equal than 0.
//
//  Arguments:
//     pSrc             Pointer to the source image ROI.
//     srcStep          Distance in bytes between starts of consecutive lines in the source image.
//     pDst             Pointer to the destination image ROI.
//     dstStep          Distance in bytes between starts of consecutive lines in the destination image.
//     roiSize          Size of the source and destination image ROI.
//     borderValue      Constant value to assign to pixels of the constant border. if border type equals ippBorderConstant
//     pSpec            Pointer to the Gaussian specification structure.
//     pBuffer          Pointer to the working buffer.
*/

IPPAPI(IppStatus, ippiFilterGaussianBorder_32f_C1R,(const Ipp32f* pSrc, int srcStep,
      Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f borderValue,
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussianBorder_16u_C1R,(const Ipp16u * pSrc, int srcStep,
      Ipp16u * pDst, int dstStep, IppiSize roiSize, Ipp16u borderValue,
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussianBorder_16s_C1R,(const Ipp16s* pSrc, int srcStep,
      Ipp16s* pDst, int dstStep, IppiSize roiSize, Ipp16s borderValue,
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussianBorder_8u_C1R,(const Ipp8u* pSrc, int srcStep,
      Ipp8u* pDst, int dstStep, IppiSize roiSize, Ipp8u borderValue,
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussianBorder_32f_C3R,(const Ipp32f* pSrc, int srcStep,
      Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussianBorder_16u_C3R,(const Ipp16u * pSrc, int srcStep,
      Ipp16u * pDst, int dstStep, IppiSize roiSize, Ipp16u borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussianBorder_16s_C3R,(const Ipp16s* pSrc, int srcStep,
      Ipp16s* pDst, int dstStep, IppiSize roiSize, Ipp16s borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussianBorder_8u_C3R,(const Ipp8u* pSrc, int srcStep,
      Ipp8u* pDst, int dstStep, IppiSize roiSize, Ipp8u borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))


/****************************************************************************************\
*                                   Image Integrals                                     *
\****************************************************************************************/


/* /////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiIntegral, ippiTiltedIntegral
//        ippiSqrIntegral, ippiTiltedSqrIntegral
//
//  Purpose:   calculates pixel sum on subimage
//
//  Returns:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc                     The pointer to source image
//    srcStep                  The step in source image
//    pDst                     The pointer to destination integral image
//    dstStep                  The step in destination image
//    pSq                      The pointer to destination square integral image
//    sqStep                   The step in destination image
//    roiSize                  The source and destination image ROI size.
//    val                      The value to add to pDst image pixels.
//    valSqr                   The value to add to pSq image pixels.
*/
IPPAPI(IppStatus, ippiIntegral_8u32s_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, IppiSize srcRoiSize, Ipp32s val))
IPPAPI(IppStatus, ippiIntegral_8u32f_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize srcRoiSize, Ipp32f val))
IPPAPI(IppStatus, ippiIntegral_32f_C1R,  (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize srcRoiSize))

IPPAPI(IppStatus, ippiTiltedIntegral_8u32s_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, IppiSize roiSize, Ipp32s val))
IPPAPI(IppStatus, ippiTiltedIntegral_8u32f_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f val))

IPPAPI(IppStatus, ippiSqrIntegral_8u32s_C1R,   (const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, Ipp32s* pSqr, int sqrStep, IppiSize roi, Ipp32s val, Ipp32s valSqr))
IPPAPI(IppStatus, ippiSqrIntegral_8u32s64f_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, Ipp64f* pSqr, int sqrStep, IppiSize roiSize, Ipp32s val, Ipp64f valSqr))
IPPAPI(IppStatus, ippiSqrIntegral_8u32f64f_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int dstStep, Ipp64f* pSqr, int sqrStep, IppiSize roiSize, Ipp32f val, Ipp64f valSqr))

IPPAPI(IppStatus, ippiTiltedSqrIntegral_8u32s_C1R,   (const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, Ipp32s* pSqr, int sqrStep, IppiSize roi, Ipp32s val, Ipp32s valSqr))
IPPAPI(IppStatus, ippiTiltedSqrIntegral_8u32s64f_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int dstStep, Ipp64f* pSqr, int sqrStep, IppiSize roiSize, Ipp32s val, Ipp64f valSqr))
IPPAPI(IppStatus, ippiTiltedSqrIntegral_8u32f64f_C1R,(const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int dstStep, Ipp64f* pSqr, int sqrStep, IppiSize roiSize, Ipp32f val, Ipp64f valSqr))



/* /////////////////////////////////////////////////////////////////////////////
//
//  Name:                 ippiHarrisCorner
//  Purpose:              Performs Harris corner detector operations
//
//  Parameters:
//    pSrc                Pointer to the source image
//    srcStep             Step through the source image
//    pDst                Pointer to the destination image
//    dstStep             Step through the destination image in pixels.
//    roiSize             Size of the source and destination image ROI.
//    filterType          Type of derivative operator
//    filterMask          Predefined mask of IppiMaskSize type.
//    avgWndSize          Size of a rectangle Window for computing an average value
//    k                   Harris Corner free coefficient
//    scale               Destination image scale factor
//    border              Type of the border
//    borderValue         Pointer to the constant value(s) if border type equals ippBorderConstant
//    pBuffer             Pointer to the work buffer
//
//  Return Values:
//    ippStsNoErr          Indicates no error.
//    ippStsNullPtrErr     Indicates an error condition if pSrc, pDst or pBufferSize is NULL.
//    ippStsSizeErr        Indicates an error in the two cases:
//                            if roiSize is negative, or equal to zero;
//                            if avgWndSize is equal to zero.
//    ippStsNotEvenStepErr Indicated an error when dstStep is not divisible by 4.
//    ippStsFilterTypeErr  Indicates an error when filterType has illegal value.
//    ippStsMaskSizeErr    Indicates an error condition if mask has an illegal value.
//    ippStsBorderErr      Indicates an error when borderType has illegal value.
//    ippStsNanArg         Indicates an error when one of k or scale is not a number
//    ippStsSizeWrn        Indicates a warning when avgWndSize is more
//                         than either image width or image height.
*/

IPPAPI(IppStatus, ippiHarrisCorner_8u32f_C1R, (
    const Ipp8u *pSrc, int srcStep, Ipp32f *pDst, int dstStep, IppiSize roiSize,
    IppiDifferentialKernel filterType, IppiMaskSize filterMask, Ipp32u avgWndSize, float k,
    float scale, IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiHarrisCorner_32f_C1R, (
    const Ipp32f *pSrc, int srcStep, Ipp32f *pDst, int dstStep, IppiSize roiSize,
    IppiDifferentialKernel filterType, IppiMaskSize filterMask, Ipp32u avgWndSize, float k,
    float scale, IppiBorderType borderType, Ipp32f borderValue, Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiHarrisCornerGetBufferSize
//
//  Purpose:            Computes the size of the external buffer for ippiHarrisCorner function
//
//  Parameters:
//   roiSize            Size of the source and destination ROI in pixels.
//   filterMask         Linear size of derivative filter aperture.
//   avgWndSize         Neighborhood block size for smoothing
//   dataType           Data type of the source image.
//   numChannels        Number of channels in the images. Possible values is 1.
//   pBufferSize        Pointer to the size (in bytes) of the external work buffer.
//
//  Return Values:
//   ippStsNoErr          Indicates no error.
//   ippStsNullPtrErr     Indicates an error when pBufferSize is NULL.
//   ippStsSizeErr        Indicates an error in the two cases:
//                          if roiSize is negative, or equal to zero;
//                          if avgWndSize is equal to zero.
//   ippStsFilterTypeErr  Indicates an error when filterType has illegal value.
//   ippStsMaskSizeErr    Indicates an error condition if mask has a wrong value.
//   ippStsDataTypeErr    Indicates an error when dataType has an illegal value.
//   ippStsNumChannelsErr Indicates an error when numChannels has an illegal value.
//   ippStsSizeWrn        Indicates a warning when avgWndSize is more
//                        than either image width or image height.
*/
IPPAPI(IppStatus, ippiHarrisCornerGetBufferSize, (IppiSize roiSize, IppiMaskSize filterMask,
                Ipp32u avgWndSize, IppDataType dataType, int numChannels, int* pBufferSize))


/*F/////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiRectStdDev_32s32f_C1R,           ippiRectStdDev_32f_C1R
//        ippiTiltedRectStdDev_32s32f_C1R,     ippiTiltedRectStdDev_32f_C1R
//        ippiRectStdDev_32s_C1RSfs,           ippiTiltedRectStdDev_32s_C1RSfs
//
//  Purpose:   Calculates standard deviation on rectangular window
//
//  Returns:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc                     The pointer to source image of integrals
//    srcStep                  The step in source image
//    pSqr                     The pointer to destination square integral image
//    sqrStep                  The step in destination image
//    pDst                     The pointer to destination image
//    dstStep                  The step in destination image
//    roiSize                  The destination image ROI size.
//    rect                     The rectangular window for standard deviation calculation.
//    scaleFactor              Output scale factor
//
//  Notes:
//F*/

IPPAPI(IppStatus, ippiRectStdDev_32f_C1R,  (const Ipp32f* pSrc, int srcStep,
                         const Ipp64f* pSqr, int sqrStep, Ipp32f* pDst, int dstStep,
                         IppiSize roiSize, IppiRect rect))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiAbsDiff_8u_C1R, ippiAbsDiff_8u_C3R, ippiAbsDiff_16u_C1R, ippiAbsDiff_32f_C1R,
//
//  Purpose: Calculate absolute difference between corresponding pixels of the two images
//           or between image pixels and scalar.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc1                    Source image
//    src1Step                 Its step
//    pSrc2                    Second source image
//    src2Step                 Its step
//    pDst                     Destination image
//    dstStep                  Its step
//    roiSize                  ROI size
//F*/

IPPAPI(IppStatus, ippiAbsDiff_8u_C1R, ( const Ipp8u* pSrc1, int src1Step,
                                        const Ipp8u* pSrc2, int src2Step,
                                        Ipp8u* pDst, int dstStep, IppiSize roiSize ))

IPPAPI(IppStatus, ippiAbsDiff_16u_C1R, ( const Ipp16u* pSrc1, int src1Step,
                                         const Ipp16u* pSrc2, int src2Step,
                                         Ipp16u* pDst, int dstStep, IppiSize roiSize ))

IPPAPI(IppStatus, ippiAbsDiff_32f_C1R, ( const Ipp32f* pSrc1, int src1Step,
                                         const Ipp32f* pSrc2, int src2Step,
                                         Ipp32f* pDst, int dstStep, IppiSize roiSize ))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiMean_8u_C1MR,  ippiMean_8s_C1MR,  ippiMean_16u_C1MR,  ippiMean_32f_C1MR,
//        ippiMean_8u_C3CMR, ippiMean_8s_C3CMR, ippiMean_16u_C3CMR, ippiMean_32f_C3CMR
//
//  Purpose:  Find mean value for selected region
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsCOIErr             COI index is illegal (coi<1 || coi>3)
//
//  Parameters:
//    pSrc                     Pointer to image
//    srcStep                  Image step
//    pMask                    Pointer to mask image
//    maskStep                 Step in the mask image
//    roiSize                  Size of image ROI
//    coi                      Index of color channel (1..3) (if color image)
//    pMean                    Returned mean value
//
//  Notes:
//F*/

IPPAPI( IppStatus, ippiMean_8u_C1MR, ( const Ipp8u* pSrc, int srcStep,
                                       const Ipp8u* pMask, int maskStep,
                                       IppiSize roiSize, Ipp64f* pMean ))

IPPAPI( IppStatus, ippiMean_8u_C3CMR, ( const Ipp8u* pSrc, int srcStep,
                                        const Ipp8u* pMask, int maskStep,
                                        IppiSize roiSize, int coi, Ipp64f* pMean ))


IPPAPI( IppStatus, ippiMean_16u_C1MR, ( const Ipp16u* pSrc, int srcStep,
                                        const Ipp8u* pMask, int maskStep,
                                        IppiSize roiSize, Ipp64f* pMean ))

IPPAPI( IppStatus, ippiMean_16u_C3CMR, ( const Ipp16u* pSrc, int srcStep,
                                         const Ipp8u* pMask, int maskStep,
                                         IppiSize roiSize, int coi, Ipp64f* pMean ))


IPPAPI( IppStatus, ippiMean_32f_C1MR, ( const Ipp32f* pSrc, int srcStep,
                                        const Ipp8u* pMask, int maskStep,
                                        IppiSize roiSize, Ipp64f* pMean ))

IPPAPI( IppStatus, ippiMean_32f_C3CMR, ( const Ipp32f* pSrc, int srcStep,
                                         const Ipp8u* pMask, int maskStep,
                                         IppiSize roiSize, int coi, Ipp64f* pMean ))



/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiMean_StdDev_8u_C1MR,   ippiMean_StdDev_8s_C1MR,
//        ippiMean_StdDev_16u_C1MR,  ippiMean_StdDev_32f_C1MR,
//        ippiMean_StdDev_8u_C3CMR,  ippiMean_StdDev_8s_C3CMR,
//        ippiMean_StdDev_16u_C3CMR, ippiMean_StdDev_32f_C3CMR
//
//  Purpose:  Find mean and standard deviation values for selected region
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc                     Pointer to image
//    srcStep                  Image step
//    pMask                    Pointer to mask image
//    maskStep                 Step in the mask image
//    roiSize                  Size of image ROI
//    coi                      Index of color channel (1..3) (if color image)
//    pMean                    Returned mean value
//    pStdDev                  Returned standard deviation
//
//  Notes:
//F*/

IPPAPI( IppStatus, ippiMean_StdDev_8u_C1MR, ( const Ipp8u* pSrc, int srcStep,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,
                                              Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_8u_C3CMR, ( const Ipp8u* pSrc, int srcStep,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize, int coi,
                                               Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_16u_C1MR, ( const Ipp16u* pSrc, int srcStep,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,
                                              Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_16u_C3CMR, ( const Ipp16u* pSrc, int srcStep,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize, int coi,
                                               Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_32f_C1MR, ( const Ipp32f* pSrc, int srcStep,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize,
                                               Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_32f_C3CMR, ( const Ipp32f* pSrc, int srcStep,
                                                const Ipp8u* pMask, int maskStep,
                                                IppiSize roiSize, int coi,
                                                Ipp64f* pMean, Ipp64f* pStdDev ))



/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiMean_StdDev_8u_C1R,   ippiMean_StdDev_8s_C1R,
//        ippiMean_StdDev_16u_C1R,  ippiMean_StdDev_32f_C1R,
//        ippiMean_StdDev_8u_C3CR,  ippiMean_StdDev_8s_C3CR,
//        ippiMean_StdDev_16u_C3CR, ippiMean_StdDev_32f_C3CR
//
//  Purpose:  Find mean and standard deviation values for selected region
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsCOIErr             COI index is illegal (coi<1 || coi>3)
//
//  Parameters:
//    pSrc                     Pointer to image
//    srcStep                  Image step
//    roiSize                  Size of image ROI
//    coi                      Index of color channel (1..3) (if color image)
//    pMean                    Returned mean value
//    pStdDev                  Returned standard deviation
//
//  Notes:
//F*/

IPPAPI( IppStatus, ippiMean_StdDev_8u_C1R, ( const Ipp8u* pSrc, int srcStep,
                                             IppiSize roiSize,
                                             Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_8u_C3CR, ( const Ipp8u* pSrc, int srcStep,
                                              IppiSize roiSize, int coi,
                                              Ipp64f* pMean, Ipp64f* pStdDev ))


IPPAPI( IppStatus, ippiMean_StdDev_16u_C1R, ( const Ipp16u* pSrc, int srcStep,
                                              IppiSize roiSize,
                                              Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_16u_C3CR, ( const Ipp16u* pSrc, int srcStep,
                                               IppiSize roiSize, int coi,
                                               Ipp64f* pMean, Ipp64f* pStdDev ))


IPPAPI( IppStatus, ippiMean_StdDev_32f_C1R, ( const Ipp32f* pSrc, int srcStep,
                                              IppiSize roiSize,
                                              Ipp64f* pMean, Ipp64f* pStdDev ))

IPPAPI( IppStatus, ippiMean_StdDev_32f_C3CR, ( const Ipp32f* pSrc, int srcStep,
                                               IppiSize roiSize, int coi,
                                               Ipp64f* pMean, Ipp64f* pStdDev ))



/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name: ippiMinMaxIndx_8u_C1MR,   ippiMinMaxIndx_8s_C1MR,
//        ippiMinMaxIndx_16u_C1MR,  ippiMinMaxIndx_32f_C1MR,
//        ippiMinMaxIndx_8u_C3CMR,  ippiMinMaxIndx_8s_C3CMR,
//        ippiMinMaxIndx_16u_C3CMR, ippiMinMaxIndx_32f_C3CMR,
//
//  Purpose:  Finds minimum and maximum values in the image and their coordinates
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc                     Pointer to image
//    srcStep                  Image step
//    pMask                    Pointer to mask image
//    maskStep                 Step in the mask image
//    roiSize                  Size of image ROI
//    coi                      Index of color channel (1..3) (if color image)
//    pMinVal                  Pointer to minimum value
//    pMaxVal                  Pointer to maximum value
//    pMinIndex                Minimum's coordinates
//    pMaxIndex                Maximum's coordinates
//
//  Notes:
//    Any of output parameters is optional
//F*/

IPPAPI(IppStatus, ippiMinMaxIndx_8u_C1MR,( const Ipp8u* pSrc, int srcStep,
                                           const Ipp8u* pMask, int maskStep,
                                           IppiSize roiSize,
                                           Ipp32f* pMinVal, Ipp32f* pMaxVal,
                                           IppiPoint* pMinIndex, IppiPoint* pMaxIndex ))

IPPAPI(IppStatus, ippiMinMaxIndx_16u_C1MR,( const Ipp16u* pSrc, int srcStep,
                                            const Ipp8u* pMask, int maskStep,
                                            IppiSize roiSize,
                                            Ipp32f* pMinVal, Ipp32f* pMaxVal,
                                            IppiPoint* pMinIndex, IppiPoint* pMaxIndex ))


IPPAPI( IppStatus, ippiMinMaxIndx_32f_C1MR, ( const Ipp32f* pSrc, int srcStep,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,
                                              Ipp32f* pMinVal, Ipp32f* pMaxVal,
                                              IppiPoint* pMinIndex, IppiPoint* pMaxIndex ))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Names: ippiNorm_Inf_8u_C1MR,       ippiNorm_Inf_8s_C1MR,
//         ippiNorm_Inf_16u_C1MR,      ippiNorm_Inf_32f_C1MR,
//         ippiNorm_Inf_8u_C3CMR,      ippiNorm_Inf_8s_C3CMR,
//         ippiNorm_Inf_16u_C3CMR,     ippiNorm_Inf_32f_C3CMR,
//         ippiNormDiff_Inf_8u_C1MR,   ippiNormDiff_Inf_8s_C1MR,
//         ippiNormDiff_Inf_16u_C1MR,  ippiNormDiff_Inf_32f_C1MR,
//         ippiNormDiff_Inf_8u_C3CMR,  ippiNormDiff_Inf_8s_C3CMR,
//         ippiNormDiff_Inf_16u_C3CMR, ippiNormDiff_Inf_32f_C3CMR,
//         ippiNormRel_Inf_8u_C1MR,    ippiNormRel_Inf_8s_C1MR,
//         ippiNormRel_Inf_16u_C1MR,   ippiNormRel_Inf_32f_C1MR,
//         ippiNormRel_Inf_8u_C3CMR,   ippiNormRel_Inf_8s_C3CMR,
//         ippiNormRel_Inf_16u_C3CMR,  ippiNormRel_Inf_32f_C3CMR,
//
//         ippiNorm_L1_8u_C1MR,        ippiNorm_L1_8s_C1MR,
//         ippiNorm_L1_16u_C1MR,       ippiNorm_L1_32f_C1MR,
//         ippiNorm_L1_8u_C3CMR,       ippiNorm_L1_8s_C3CMR,
//         ippiNorm_L1_16u_C3CMR,      ippiNorm_L1_32f_C3CMR,
//         ippiNormDiff_L1_8u_C1MR,    ippiNormDiff_L1_8s_C1MR,
//         ippiNormDiff_L1_16u_C1MR,   ippiNormDiff_L1_32f_C1MR,
//         ippiNormDiff_L1_8u_C3CMR,   ippiNormDiff_L1_8s_C3CMR,
//         ippiNormDiff_L1_16u_C3CMR,  ippiNormDiff_L1_32f_C3CMR,
//         ippiNormRel_L1_8u_C1MR,     ippiNormRel_L1_8s_C1MR,
//         ippiNormRel_L1_16u_C1MR,    ippiNormRel_L1_32f_C1MR,
//         ippiNormRel_L1_8u_C3CMR,    ippiNormRel_L1_8s_C3CMR,
//         ippiNormRel_L1_16u_C3CMR,   ippiNormRel_L1_32f_C3CMR,
//
//         ippiNorm_L2_8u_C1MR,        ippiNorm_L2_8s_C1MR,
//         ippiNorm_L2_16u_C1MR,       ippiNorm_L2_32f_C1MR,
//         ippiNorm_L2_8u_C3CMR,       ippiNorm_L2_8s_C3CMR,
//         ippiNorm_L2_16u_C3CMR,      ippiNorm_L2_32f_C3CMR,
//         ippiNormDiff_L2_8u_C1MR,    ippiNormDiff_L2_8s_C1MR,
//         ippiNormDiff_L2_16u_C1MR,   ippiNormDiff_L2_32f_C1MR,
//         ippiNormDiff_L2_8u_C3CMR,   ippiNormDiff_L2_8s_C3CMR,
//         ippiNormDiff_L2_16u_C3CMR,  ippiNormDiff_L2_32f_C3CMR,
//         ippiNormRel_L2_8u_C1MR,     ippiNormRel_L2_8s_C1MR,
//         ippiNormRel_L2_16u_C1MR,    ippiNormRel_L2_32f_C1MR,
//         ippiNormRel_L2_8u_C3CMR,    ippiNormRel_L2_8s_C3CMR,
//         ippiNormRel_L2_16u_C3CMR,   ippiNormRel_L2_32f_C3CMR
//
//  Purpose: Calculates ordinary, differential or relative norms of one or two images
//           in an arbitrary image region.
//
//  Returns:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//
//  Parameters:
//    pSrc, pSrc1              Pointers to source and mask images
//    pSrc2, pMask
//    srcStep, src1Step        Their steps
//    src2Step, maskStep
//    roiSize                  Their size or ROI size
//    coi                      COI index (1..3) (if 3-channel images)
//    pNorm                    The pointer to calculated norm
//
//  Notes:
//F*/

/* ///////////////////////////////// 8uC1 flavor ////////////////////////////////////// */

IPPAPI( IppStatus, ippiNorm_Inf_8u_C1MR, ( const Ipp8u* pSrc, int srcStep,
                                           const Ipp8u* pMask,int maskStep,
                                           IppiSize roiSize, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_Inf_16u_C1MR, ( const Ipp16u* pSrc, int srcStep,
                                            const Ipp8u* pMask,int maskStep,
                                            IppiSize roiSize, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_Inf_32f_C1MR, ( const Ipp32f* pSrc, int srcStep,
                                            const Ipp8u* pMask, int maskStep,
                                            IppiSize roiSize, Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNorm_Inf_8u_C3CMR, ( const Ipp8u* pSrc, int srcStep,
                                            const Ipp8u* pMask,int maskStep,
                                            IppiSize roiSize, int coi, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_Inf_16u_C3CMR, ( const Ipp16u* pSrc, int srcStep,
                                             const Ipp8u* pMask,int maskStep,
                                             IppiSize roiSize, int coi, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_Inf_32f_C3CMR, ( const Ipp32f* pSrc, int srcStep,
                                             const Ipp8u* pMask, int maskStep,
                                             IppiSize roiSize, int coi, Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNormDiff_Inf_8u_C1MR, ( const Ipp8u* pSrc1, int src1Step,
                                               const Ipp8u* pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormDiff_Inf_16u_C1MR, ( const Ipp16u* pSrc1, int src1Step,
                                                const Ipp16u* pSrc2, int src2Step,
                                                const Ipp8u* pMask, int maskStep,
                                                IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormDiff_Inf_32f_C1MR, ( const Ipp32f* pSrc1, int src1Step,
                                                const Ipp32f* pSrc2, int src2Step,
                                                const Ipp8u* pMask,  int maskStep,
                                                IppiSize roiSize,    Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNormDiff_Inf_8u_C3CMR, (const Ipp8u* pSrc1, int src1Step,
                                               const Ipp8u* pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize, int coi, Ipp64f* pNorm ))

IPPAPI( IppStatus, ippiNormDiff_Inf_16u_C3CMR, (const Ipp16u* pSrc1, int src1Step,
                                                const Ipp16u* pSrc2, int src2Step,
                                                const Ipp8u* pMask, int maskStep,
                                                IppiSize roiSize, int coi, Ipp64f* pNorm ))

IPPAPI( IppStatus, ippiNormDiff_Inf_32f_C3CMR, (const Ipp32f* pSrc1, int src1Step,
                                                const Ipp32f* pSrc2, int src2Step,
                                                const Ipp8u* pMask,  int maskStep,
                                                IppiSize roiSize, int coi, Ipp64f* pNorm ))


IPPAPI( IppStatus, ippiNormRel_Inf_8u_C1MR, ( const Ipp8u* pSrc1, int src1Step,
                                              const Ipp8u* pSrc2, int src2Step,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormRel_Inf_16u_C1MR, ( const Ipp16u* pSrc1, int src1Step,
                                               const Ipp16u* pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormRel_Inf_32f_C1MR, ( const Ipp32f* pSrc1, int src1Step,
                                               const Ipp32f* pSrc2, int src2Step,
                                               const Ipp8u* pMask,  int maskStep,
                                               IppiSize roiSize,    Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNorm_L1_8u_C1MR, ( const Ipp8u* pSrc, int srcStep,
                                          const Ipp8u* pMask,int maskStep,
                                          IppiSize roiSize, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L1_16u_C1MR, ( const Ipp16u* pSrc, int srcStep,
                                           const Ipp8u* pMask,int maskStep,
                                           IppiSize roiSize, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L1_32f_C1MR, ( const Ipp32f* pSrc, int srcStep,
                                           const Ipp8u* pMask, int maskStep,
                                           IppiSize roiSize, Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNorm_L1_8u_C3CMR, ( const Ipp8u* pSrc, int srcStep,
                                           const Ipp8u* pMask,int maskStep,
                                           IppiSize roiSize, int coi, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L1_16u_C3CMR, ( const Ipp16u* pSrc, int srcStep,
                                            const Ipp8u* pMask,int maskStep,
                                            IppiSize roiSize, int coi, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L1_32f_C3CMR, ( const Ipp32f* pSrc, int srcStep,
                                            const Ipp8u* pMask, int maskStep,
                                            IppiSize roiSize, int coi, Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNormDiff_L1_8u_C1MR, ( const Ipp8u* pSrc1, int src1Step,
                                              const Ipp8u* pSrc2, int src2Step,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormDiff_L1_16u_C1MR, ( const Ipp16u* pSrc1, int src1Step,
                                               const Ipp16u* pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormDiff_L1_32f_C1MR, ( const Ipp32f* pSrc1, int src1Step,
                                               const Ipp32f* pSrc2, int src2Step,
                                               const Ipp8u* pMask,  int maskStep,
                                               IppiSize roiSize,    Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNormDiff_L1_8u_C3CMR, ( const Ipp8u* pSrc1, int src1Step,
                                               const Ipp8u* pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize, int coi, Ipp64f* pNorm ))

IPPAPI( IppStatus, ippiNormDiff_L1_16u_C3CMR, ( const Ipp16u* pSrc1, int src1Step,
                                                const Ipp16u* pSrc2, int src2Step,
                                                const Ipp8u* pMask, int maskStep,
                                                IppiSize roiSize, int coi, Ipp64f* pNorm ))

IPPAPI( IppStatus, ippiNormDiff_L1_32f_C3CMR, ( const Ipp32f* pSrc1, int src1Step,
                                                const Ipp32f* pSrc2, int src2Step,
                                                const Ipp8u* pMask,  int maskStep,
                                                IppiSize roiSize, int coi, Ipp64f* pNorm ))


IPPAPI( IppStatus, ippiNormRel_L1_8u_C1MR, ( const Ipp8u* pSrc1, int src1Step,
                                             const Ipp8u* pSrc2, int src2Step,
                                             const Ipp8u* pMask, int maskStep,
                                             IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormRel_L1_16u_C1MR, ( const Ipp16u* pSrc1, int src1Step,
                                              const Ipp16u* pSrc2, int src2Step,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormRel_L1_32f_C1MR, ( const Ipp32f* pSrc1, int src1Step,
                                              const Ipp32f* pSrc2, int src2Step,
                                              const Ipp8u* pMask,  int maskStep,
                                              IppiSize roiSize,    Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNorm_L2_8u_C1MR, ( const Ipp8u* pSrc, int srcStep,
                                          const Ipp8u* pMask,int maskStep,
                                          IppiSize roiSize, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L2_16u_C1MR, ( const Ipp16u* pSrc, int srcStep,
                                           const Ipp8u* pMask,int maskStep,
                                           IppiSize roiSize, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L2_32f_C1MR, ( const Ipp32f* pSrc, int srcStep,
                                           const Ipp8u* pMask, int maskStep,
                                           IppiSize roiSize, Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNorm_L2_8u_C3CMR, ( const Ipp8u* pSrc, int srcStep,
                                           const Ipp8u* pMask,int maskStep,
                                           IppiSize roiSize, int coi, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L2_16u_C3CMR, ( const Ipp16u* pSrc, int srcStep,
                                            const Ipp8u* pMask,int maskStep,
                                            IppiSize roiSize, int coi, Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNorm_L2_32f_C3CMR, ( const Ipp32f* pSrc, int srcStep,
                                            const Ipp8u* pMask, int maskStep,
                                            IppiSize roiSize, int coi, Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNormDiff_L2_8u_C1MR, ( const Ipp8u* pSrc1, int src1Step,
                                              const Ipp8u* pSrc2, int src2Step,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormDiff_L2_16u_C1MR, ( const Ipp16u* pSrc1, int src1Step,
                                               const Ipp16u* pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormDiff_L2_32f_C1MR, ( const Ipp32f* pSrc1, int src1Step,
                                               const Ipp32f* pSrc2, int src2Step,
                                               const Ipp8u* pMask,  int maskStep,
                                               IppiSize roiSize,    Ipp64f* pNorm ) )


IPPAPI( IppStatus, ippiNormDiff_L2_8u_C3CMR, ( const Ipp8u* pSrc1, int src1Step,
                                               const Ipp8u* pSrc2, int src2Step,
                                               const Ipp8u* pMask, int maskStep,
                                               IppiSize roiSize, int coi, Ipp64f* pNorm ))

IPPAPI( IppStatus, ippiNormDiff_L2_16u_C3CMR, ( const Ipp16u* pSrc1, int src1Step,
                                                const Ipp16u* pSrc2, int src2Step,
                                                const Ipp8u* pMask, int maskStep,
                                                IppiSize roiSize, int coi, Ipp64f* pNorm ))

IPPAPI( IppStatus, ippiNormDiff_L2_32f_C3CMR, ( const Ipp32f* pSrc1, int src1Step,
                                                const Ipp32f* pSrc2, int src2Step,
                                                const Ipp8u* pMask,  int maskStep,
                                                IppiSize roiSize, int coi, Ipp64f* pNorm ))


IPPAPI( IppStatus, ippiNormRel_L2_8u_C1MR, ( const Ipp8u* pSrc1, int src1Step,
                                             const Ipp8u* pSrc2, int src2Step,
                                             const Ipp8u* pMask, int maskStep,
                                             IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormRel_L2_16u_C1MR, ( const Ipp16u* pSrc1, int src1Step,
                                              const Ipp16u* pSrc2, int src2Step,
                                              const Ipp8u* pMask, int maskStep,
                                              IppiSize roiSize,   Ipp64f* pNorm ) )

IPPAPI( IppStatus, ippiNormRel_L2_32f_C1MR, ( const Ipp32f* pSrc1, int src1Step,
                                              const Ipp32f* pSrc2, int src2Step,
                                              const Ipp8u* pMask,  int maskStep,
                                              IppiSize roiSize,    Ipp64f* pNorm ) )


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiCannyGetSize
//
//  Purpose: Calculates size of temporary buffer, required to run Canny function.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         Pointer bufferSize is NULL
//    ippStsSizeErr            roiSize has a field with zero or negative value
//
//  Parameters:
//    roiSize                  Size of image ROI in pixel
//    bufferSize               Pointer to the variable that returns the size of the temporary buffer
//F*/

IPPAPI( IppStatus, ippiCannyGetSize, ( IppiSize roiSize, int* bufferSize ))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiCanny_16s8u_C1IR,     ippiCanny_32f8u_C1IR
//
//  Purpose: Creates binary image of source's image edges,
//                using derivatives of the first order.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsBadArgErr          Bad thresholds
//
//  Parameters:
//    pSrcDx                   Pointers to the source image ( first derivatives  with respect to X )
//    srcDxStep                Step in bytes through the source image pSrcDx
//    pSrcDy                   Pointers to the source image ( first derivatives  with respect to Y )
//    srcDyStep                Step in bytes through the source image pSrcDy
//
//    roiSize                  Size of the source images ROI in pixels
//    lowThresh                Low threshold for edges detection
//    highThresh               Upper threshold for edges detection
//    pBuffer                  Pointer to the pre-allocated temporary buffer, which size can be
//                             calculated using ippiCannyGetSize function
//F*/

IPPAPI(IppStatus, ippiCanny_16s8u_C1R, ( Ipp16s* pSrcDx, int srcDxStep,
                                         Ipp16s* pSrcDy, int srcDyStep,
                                         Ipp8u*  pDstEdges, int dstEdgeStep,
                                         IppiSize roiSize,
                                         Ipp32f  lowThresh,
                                         Ipp32f  highThresh,
                                         Ipp8u*  pBuffer ))


/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiDistanceTransform_3x3_8u32f_C1R, ippiDistanceTransform_5x5_8u32f_C1R,
//           ippiDistanceTransform_3x3_8u16u_C1R, ippiDistanceTransform_5x5_8u16u_C1R,
//           ippiDistanceTransform_3x3_8u_C1R,    ippiDistanceTransform_5x5_8u_C1R,
//           ippiDistanceTransform_3x3_8u_C1IR,   ippiDistanceTransform_5x5_8u_C1IR,
//           ippiTrueDistanceTransform_8u32f_C1R
//
//  Purpose: For every non-zero pixel in the source image, the functions calculate
//           distance between that pixel and nearest zero pixel.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsCoeffErr           Zero mask coefficient
//
//  Parameters:
//    pSrc                     Source image
//    pSrcDst                  Pointer to the input and output image
//    srcStep                  Its step
//    pDst                     Output image with distances
//    dstStep                  Its step
//    roiSize                  ROI size
//    pMetrics                 Array that determines metrics used.
//    scaleFactor              Scale factor
//    pBuffer                  The pointer to the working buffer
//
//F*/

IPPAPI(IppStatus, ippiDistanceTransform_5x5_8u32f_C1R, ( const Ipp8u* pSrc, int srcStep,
                                                         Ipp32f* pDst, int dstStep,
                                                         IppiSize roiSize, Ipp32f* pMetrics ))

IPPAPI(IppStatus, ippiTrueDistanceTransform_8u32f_C1R, ( const Ipp8u* pSrc, int srcStep,
                                                         Ipp32f* pDst, int dstStep,
                                                         IppiSize roiSize, Ipp8u* pBuffer ))

IPPAPI(IppStatus, ippiDistanceTransform_3x3_8u_C1R, ( const Ipp8u* pSrc, int srcStep,
                                                      Ipp8u* pDst, int dstStep,
                                                      IppiSize roiSize, Ipp32s* pMetrics ))

IPPAPI(IppStatus, ippiDistanceTransform_3x3_8u32f_C1R, ( const Ipp8u* pSrc, int srcStep,
                                                         Ipp32f* pDst, int dstStep,
                                                         IppiSize roiSize, Ipp32f* pMetrics ))


/* ///////////////////////////////////////////////////////////////////////////////////////////////////
//  Name:  ippiTrueDistanceTransformGetBufferSize_8u32f_C1R
//
//  Purpose:    Get size of external buffer.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            Indicates an error condition if roiSize has a field with zero
//                             or negative value.
//
//  Parameters:
//    roiSize                  The image ROI size
//    pBufferSize              The pointer to the buffer size
*/

IPPAPI(IppStatus, ippiTrueDistanceTransformGetBufferSize_8u32f_C1R, (IppiSize roiSize, int* pBufferSize))


/****************************************************************************************\
*                                Morphological Operations                                *
\****************************************************************************************/

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiMorphologyBorderGetSize_16u_C1R,     ippiMorphologyBorderGetSize_16s_C1R,
//          ippiMorphologyBorderGetSize_1u_C1R,      ippiMorphologyBorderGetSize_8u_C1R,
//          ippiMorphologyBorderGetSize_8u_C3R,      ippiMorphologyBorderGetSize_8u_C4R,
//          ippiMorphologyBorderGetSize_32f_C1R,     ippiMorphologyBorderGetSize_32f_C3R
//          ippiMorphologyBorderGetSize_32f_C4R
//
//
//  Purpose:  Gets the size of the internal state or specification structure for morphological operations.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image, or width or height of the structuring
//                             element is less than,or equal to zero.
//
//  Parameters:
//    roiWidth                 Width of the image ROI in pixels.
//    pMask                    Pointer to the structuring element (mask).
//    maskSize                 Size of the structuring element.
//    pSize                    Pointer to the state structure length.
//    pSpecSize                Pointer to the specification structure size.
//    pBufferSize              Pointer to the buffer size value for the morphological initialization function.
*/

IPPAPI(IppStatus, ippiMorphologyBorderGetSize_8u_C1R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))
IPPAPI(IppStatus, ippiMorphologyBorderGetSize_8u_C3R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))
IPPAPI(IppStatus, ippiMorphologyBorderGetSize_8u_C4R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))
IPPAPI(IppStatus, ippiMorphologyBorderGetSize_32f_C1R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))
IPPAPI(IppStatus, ippiMorphologyBorderGetSize_32f_C3R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))
IPPAPI(IppStatus, ippiMorphologyBorderGetSize_32f_C4R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))
IPPAPI(IppStatus, ippiMorphologyBorderGetSize_16u_C1R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))
IPPAPI(IppStatus, ippiMorphologyBorderGetSize_16s_C1R,(IppiSize roiSize, IppiSize maskSize, int* pSpecSize, int* pBufferSize))

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiMorphologyBorderInit_16u_C1R,        ippiMorphologyBorderInit_16s_C1R
//          ippiMorphologyBorderInit_1u_C1R,         ippiMorphologyBorderInit_8u_C1R
//          ippiMorphologyBorderInit_8u_C3R,         ippiMorphologyBorderInit_8u_C4R
//          ippiMorphologyBorderInit_32f_C1R,        ippiMorphologyBorderInit_32f_C3R,
//          ippiMorphologyBorderInit_32f_C4R,
//
//  Purpose:  Initialize the internal state or specification structure for morphological operation.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image or width or height of the structuring
//                             element is less than, or equal to zero.
//    ippStsAnchorErr          Anchor point is outside the structuring element.
//
//  Parameters:
//    roiWidth                 Width of the image ROI in pixels.
//    pMask                    Pointer to the structuring element (mask).
//    maskSize                 Size of the structuring element.
//    anchor                   Anchor of the structuring element.
//    pState                   Pointer to the morphology state structure.
//    pMorphSpec               Pointer to the morphology specification structure.
//    pBuffer                  Pointer to the external work buffer.
*/

IPPAPI(IppStatus, ippiMorphologyBorderInit_16u_C1R,(IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiMorphologyBorderInit_16s_C1R,(IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiMorphologyBorderInit_8u_C1R,( IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer ))
IPPAPI(IppStatus, ippiMorphologyBorderInit_8u_C3R,( IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer ))
IPPAPI(IppStatus, ippiMorphologyBorderInit_8u_C4R,( IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer ))
IPPAPI(IppStatus, ippiMorphologyBorderInit_32f_C1R,( IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer ))
IPPAPI(IppStatus, ippiMorphologyBorderInit_32f_C3R,( IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer ))
IPPAPI(IppStatus, ippiMorphologyBorderInit_32f_C4R,( IppiSize roiSize, const Ipp8u* pMask, IppiSize maskSize,
                                              IppiMorphState* pMorphSpec, Ipp8u* pBuffer ))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiDilateBorder_8u_C1R,    ippiDilateBorder_8u_C3R,
//          ippiDilateBorder_8u_C4R,    ippiDilateBorder_32f_C1R,
//          ippiDilateBorder_32f_C3R,   ippiDilateBorder_32f_C4R
//
//          ippiErodeBorder_8u_C1R,     ippiErodeBorder_8u_C3R,
//          ippiErodeBorder_8u_C4R,     ippiErodeBorder_32f_C1R,
//          ippiErodeBorder_32f_C3R,    ippiErodeBorder_32f_C4R,
//
//          ippiDilateBorder_16u_C1R,            ippiDilateBorder_16s_C1R,
//          ippiDilateBorder_1u_C1R
//
//  Purpose:    Perform erosion/dilation of the image arbitrary shape structuring element.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            The ROI width or height is less than 1,
//                             or ROI width is bigger than ROI width in the state structure.
//    ippStsStepErr            Step is too small to fit the image.
//    ippStsNotEvenStepErr     Step is not multiple of the element.
//    ippStsBadArgErr          Incorrect border type.
//

//  Parameters:
//    pSrc                     Pointer to the source image.
//    srcStep                  Step in the source image.
//    pDst                     Pointer to the destination image.
//    dstStep                  Step in the destination image.
//    roiSize                  ROI size.
//    borderType               Type of border (ippBorderRepl now).
//    borderValue              Value for the constant border.
//    pState                   Pointer to the morphology state structure.
//    pMorphSpec               Pointer to the morphology specification structure.
//    pBuffer                  Pointer to the external work buffer.
*/
IPPAPI(IppStatus, ippiDilateBorder_8u_C1R, (const Ipp8u* pSrc, int srcStep,
                                   Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp8u borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilateBorder_8u_C3R, (const Ipp8u* pSrc, int srcStep,
                                   Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp8u borderValue[3], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilateBorder_8u_C4R, (const Ipp8u* pSrc, int srcStep,
                                   Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp8u borderValue[4], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilateBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                   Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp32f borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilateBorder_32f_C3R, (const Ipp32f* pSrc, int srcStep,
                                   Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp32f borderValue[3], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilateBorder_32f_C4R, (const Ipp32f* pSrc, int srcStep,
                                   Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp32f borderValue[4], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilateBorder_16u_C1R,(const Ipp16u* pSrc, int srcStep,
                                   Ipp16u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp16u borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilateBorder_16s_C1R,(const Ipp16s* pSrc, int srcStep,
                                   Ipp16s* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp16s borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiErodeBorder_8u_C1R, (const Ipp8u* pSrc, int srcStep,
                                   Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp8u borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErodeBorder_8u_C3R, (const Ipp8u* pSrc, int srcStep,
                                   Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp8u borderValue[3], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErodeBorder_8u_C4R, (const Ipp8u* pSrc, int srcStep,
                                   Ipp8u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp8u borderValue[4], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErodeBorder_32f_C1R, (const Ipp32f* pSrc, int srcStep,
                                   Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp32f borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErodeBorder_32f_C3R, (const Ipp32f* pSrc, int srcStep,
                                   Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp32f borderValue[3], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErodeBorder_32f_C4R, (const Ipp32f* pSrc, int srcStep,
                                   Ipp32f* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp32f borderValue[4], const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErodeBorder_16u_C1R,(const Ipp16u* pSrc, int srcStep,
                                   Ipp16u* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp16u borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErodeBorder_16s_C1R,(const Ipp16s* pSrc, int srcStep,
                                   Ipp16s* pDst, int dstStep, IppiSize roiSize,
                                   IppiBorderType borderType, Ipp16s borderValue, const IppiMorphState* pMorphSpec, Ipp8u* pBuffer))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:      ippiUpdateMotionHistory_8u32f_C1IR,  ippiUpdateMotionHistory_16u32f_C1IR
//             ippiUpdateMotionHistory_32f_C1IR
//
//  Purpose:   Sets motion history image (MHI) pixels to the current time stamp
//             when the corrensonding pixels in the silhoette image are non zero.
//             Else (silhouette pixels are zero) MHI pixels are
//             cleared if their values are too small (less than timestamp - mhiDuration),
//             i.e. they were updated far ago last time. Else MHI pixels remain unchanged.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsOutOfRangeErr      Maximal duration is negative
//
//  Arguments:
//    pSilhouette              The pointer to silhouette image
//    silhStep                 The step in silhouette image
//    pMHI                     The pointer to motion history image
//    mhiStep                  The step in mhi image
//    roiSize                  ROI size
//    timestamp                Current time stamp (milliseconds)
//    mhiDuration              Maximal duration of motion track (milliseconds)
*/

IPPAPI(IppStatus, ippiUpdateMotionHistory_8u32f_C1IR,
                        ( const Ipp8u* pSilhouette, int silhStep,
                          Ipp32f* pMHI, int mhiStep, IppiSize roiSize,
                          Ipp32f timestamp, Ipp32f mhiDuration ))


/*
//  Name:      ippiHoughLine_Region_8u32f_C1R
//
//
//  Purpose:   Perform Hough transform algorithm in defined region
//
//
//  Arguments:
//    pSrc                     Pointer to the source image ROI
//    srcStep                  Distance in bytes between starts of consecutive lines in the source image
//    roiSize                  The size of source ROI.
//    pLine                    Pointer to output array of detected lines
//    dstRoi                   Bottom left and top right corners of searched lines. All founded lines must be
//                             in this area. (line[n].rho>=dstRoi[0].rho && line[n].theta>=dstRoi[0].theta &&
//                                            line[n].rho<=dstRoi[1].rho && line[n].theta<=dstRoi[1].theta)
//    maxLineCount             Size of output array pLine in elements
//    pLineCount               Number of detected lines.  If founded more than maxLineCount lines than function returns "ippStsDstSizeLessExpected" status
//    delta                    Discretization step, delta.rho - distance resolution in pixels, delta.theta - angle resolution in radians
//    threshold                Threshold for a line is detected (if accumulator value > threshold)
//    pBuffer                  Pointer to the pre-allocated temporary buffer
//
//  Return:
//    ippStsNoErr               Ok
//    ippStsNullPtrErr          pSrc or pLine or pLineCount or pBuffer is NULL
//    ippStsStepErr             srcStep is not valid
//    ippStsSizeErr             roiSize has a field with zero or negative value or maxLineCount is zero or negative
//    ippStsBadArgErr           threshold or is less than or equal to zero or delta.rho less 0 or more ROI width+height or
//                              delta.theta less 0 or more PI or dstRoi[0].rho more dstRoi[1].rho or dstRoi[0].theta more dstRoi[1].theta
//    ippStsDstSizeLessExpected Ok, but lines detected more than maxLineCount
//
*/

IPPAPI(IppStatus, ippiHoughLine_Region_8u32f_C1R, (const Ipp8u* pSrc, int srcStep, IppiSize roiSize,
       IppPointPolar* pLine, IppPointPolar dstRoi[2], int maxLineCount, int* pLineCount, IppPointPolar delta, int threshold, Ipp8u* pBuffer))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:      ippiHoughLineGetSize_8u_C1R()
//
//
//  Purpose:   Calculate the size of temporary buffer for ippiHoughLine_8u32f_C1R function.
//
//  Arguments:
//    roiSize                  The size of source ROI.
//    delta                    Discretization step, delta.rho - distance resolution in pixels, delta.theta - angle resolution in radians
//    maxLineCount             The size of detected line buffer
//    pBufSize                 Pointer to the computed size of the temporary buffer
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         pBufSize is NULL
//    ippStsSizeErr            The roiSize or delta has a field with zero or negative value
//    ippStsOverflow           The size of buffer too big. Overflow.
//
*/

IPPAPI(IppStatus, ippiHoughLineGetSize_8u_C1R, (IppiSize roiSize, IppPointPolar delta, int maxLineCount, int* pBufSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiFilterSpeckleGetBufferSize
//
//  Purpose:            Computes the size of the external buffer for ippiFilterSpeckle function
//
//  Parameters:
//   dstRoiSize         Size of destination ROI in pixels.
//   dataType           Data type of the source and destination images.
//   numChannels        Number of channels in the images. Possible value is 1.
//   pBufferSize        Pointer to the size (in bytes) of the external work buffer.
//
//  Return Values:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when pBufferSize is NULL.
//   ippStsSizeErr      Indicates an error when roiSize is negative, or equal to zero.
//   ippStsDataTypeErr  Indicates an error when dataType has an illegal value.
//   ippStsNumChannelsErr Indicates an error when numChannels has an illegal value.
*/

IPPAPI(IppStatus, ippiMarkSpecklesGetBufferSize, (IppiSize roiSize,
    IppDataType dataType, int numChannels, int* pBufferSize))


/* /////////////////////////////////////////////////////////////////////////////
//
//  Name:                rippiFilterSpeckle
//  Purpose:             Performs filtering of small noise blobs (speckles)
//                       in the image
//
//  Parameters:
//    pSrcDst            Pointer to the source and destination image
//    srcDstStep         Step through the source and destination image
//    roiSize            Size of the source and destination image ROI.
//    speckleVal         The disparity value used to paint-off the speckles
//    maxSpeckleSize     The maximum speckle size to consider it a speckle.
//    maxPixDiff         Maximum difference between neighbor pixels to put them into the same blob.
//    norm               Specifies type of the norm to form the mask for marker propagation:
//                          ippiNormInf Infinity norm (8-connectivity);
//                          ippiNormL1  L1 norm (4-connectivity).
//    pBuffer            Pointer to the work buffer
//
//  Return Values:
//    ippStsNoErr        Indicates no error.
//    ippStsNullPtrErr   Indicates an error condition if pSrc, pDst or pBufferSize is NULL.
//    ippStsSizeErr      Indicates an error when roiSize is negative, or equal to zero.
//    ippStsNotEvenStepErr Indicated an error when one of the step values is not divisible by 4
//                       for floating-point images, or by 2 for short-integer images.
//    ippStsNanArg       Indicates an error when one of speckleVal or maxPixDiff is not a number
//    ippStsNormErr      Indicates an error when norm is incorrect or not supported
*/

IPPAPI(IppStatus, ippiMarkSpeckles_16s_C1IR, (Ipp16s *pSrcDst, int srcDstStep, IppiSize roiSize,
    Ipp16s speckleVal, int maxSpeckleSize, Ipp16s maxPixDiff, IppiNorm norm, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiMarkSpeckles_8u_C1IR, (Ipp8u *pSrcDst, int srcDstStep, IppiSize roiSize,
    Ipp8u speckleVal, int maxSpeckleSize, Ipp8u maxPixDiff, IppiNorm norm, Ipp8u* pBuffer))



/* =============================================================================
							ippCC
============================================================================= */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippccGetLibVersion
//  Purpose:    getting of the library version
//  Returns:    the structure of information about version
//              of ippCC library
//  Parameters:
//
//  Notes:      not necessary to release the returned structure
*/
IPPAPI( const IppLibraryVersion*, ippccGetLibVersion, (void) )


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiBGRToLab_8u_C3R    and ippiLabToBGR_8u_C3R
//              ippiBGRToLab_8u16u_C3R and ippiLabToBGR_16u8u_C3R
//  Purpose:    Converts an RGB image to CIE Lab color model and vice-versa
//  Parameters:
//    pSrc          Pointer to the source image ROI
//    srcStep       Step through the source  image (bytes)
//    pDst          Pointer to the destination image ROI
//    dstStep       Step through the destination image (bytes)
//    roiSize       Size of the ROI
//  Returns:
//     ippStsNullPtrErr  if src == NULL or dst == NULL
//     ippStsSizeErr     if imgSize.width <= 0 || imgSize.height <= 0
//     ippStsNoErr       otherwise
//  Reference:
//     Computer graphics: principles and practices. James D. Foley... [et al.]. 2nd ed.
//     Addison-Wesley, c1990.p.(584)
//
//    At first an RGB image is converted to the XYZ color model (see the function
//    ippRGBToXYZ_8u_C3R), then to the CIELab with the white point D65 and CIE chromaticity
//    coordinates of white point (xn,yn) = (0.312713, 0.329016)
//    L = 116. *((Y/Yn)^(1/3)) - 16    for Y/Yn >  0.008856
//    L = 903.3*(Y/Yn)                 for Y/Yn <= 0.008856
//    a = 500. * (f(X/Xn) - f(Y/Yn))
//    b = 200. * (f(Y/Yn) - f(Z/Zn))
//    where f(t)=t^(1/3)               for t >  0.008856
//    f(t)=7.787*t+16/116              for t <= 0.008856
//    These values are quantized and scaled to the 8-bit range of 0 to 255 for ippiBGRToLab_8u_C3R.
//    L =   L * 255. / 100.
//    a = (a + 128.)
//    b = (a + 128.)
//    and they are quantized and scaled to the 16-bit range of 0 to 65535 for ippiBGRToLab_8u16u_C3R
//    L =  L * 65535. / 100.
//    a = (a + 128.)* 255
//    b = (a + 128.)* 255
//    where:
//      normalizing multipliers
//    Yn = 1.0      * 255
//    Xn = 0.950455 * 255
//    Zn = 1.088753 * 255
//
//    L component values are in the range [0..100], a and b component values are
//    in the range [-128..127].
//
//    The CIELab to RGB conversion is performed as follows. At first
//    a Lab image is converted to the XYZ image
//      for ippiLabToBGR_8u_C3R
//    L =  L * 100./ 255.
//    a = (a - 128.)
//    b = (a - 128.)
//      or for ippiLabToBGR_16u8u_C3R
//    L =  L * 100./ 65535.
//    a = (a / 255 - 128.)
//    b = (b / 255 - 128.)
//    X = Xn * ( P + a / 500 )^3
//    Y = Yn * P^3
//    Z = Zn * ( P - b / 200 )^3
//    where P = (L + 16) / 116
//    Then the XYZ image is converted to the RGB color model (see the function
//     ippXYZToRGB_8u_C3R).
//
*/
IPPAPI(IppStatus, ippiBGRToLab_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst,int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiLabToBGR_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst,int dstStep, IppiSize roiSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiColorToGray
//  Purpose:    Converts an RGB image to gray scale (custom coefficients)
//  Parameters:
//     pSrc      Pointer to the source image , points to point(0,0)
//     pDst      Pointer to the destination image , points to point(0,0)
//     roiSize   Size of the ROI in pixels. Since the function performs point
//               operations (without a border), the ROI may be the whole image.
//     srcStep   Step in bytes through the source image to jump on the next line
//     dstStep   Step in bytes through the destination image to jump on the next line
//     coeffs[3] User-defined vector of coefficients.
//                 The sum of the coefficients should be less than or equal to 1
//  Returns:
//           ippStsNullPtrErr  pSrc == NULL, or pDst == NULL
//           ippStsSizeErr     roiSize has a field with zero or negative value
//           ippStsNoErr       No errors
//
//  The following equation is used to convert an RGB image to gray scale:
//
//   Y = coeffs[0] * R + coeffs[1] * G + coeffs[2] * B;
//
//
*/

IPPAPI(IppStatus,ippiColorToGray_8u_C3C1R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep,IppiSize roiSize,const Ipp32f coeffs[3]))
IPPAPI(IppStatus,ippiColorToGray_16u_C3C1R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep,IppiSize roiSize,const Ipp32f coeffs[3]))
IPPAPI(IppStatus,ippiColorToGray_32f_C3C1R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep,IppiSize roiSize,const Ipp32f coeffs[3]))

IPPAPI(IppStatus,ippiColorToGray_8u_AC4C1R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep,IppiSize roiSize,const Ipp32f coeffs[3]))
IPPAPI(IppStatus,ippiColorToGray_16u_AC4C1R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep,IppiSize roiSize,const Ipp32f coeffs[3]))
IPPAPI(IppStatus,ippiColorToGray_32f_AC4C1R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep,IppiSize roiSize,const Ipp32f coeffs[3]))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiRGBToGray
//  Purpose:    Converts an RGB image to gray scale (fixed coefficients)
//  Parameters:
//     pSrc     Pointer to the source image , points to point(0,0)
//     pDst     Pointer to the destination image , points to point(0,0)
//     roiSize  Size of the ROI in pixels. Since the function performs point
//          operations (without a border), the ROI may be the whole image.
//     srcStep  Step in bytes through the source image to jump on the next line
//     dstStep  Step in bytes through the destination image to jump on the next line
//  Returns:
//           ippStsNullPtrErr  pSrc == NULL, or pDst == NULL
//           ippStsSizeErr     roiSize has a field with zero or negative value
//           ippStsNoErr       No errors
//  Reference:
//      Jack Keith
//      Video Demystified: A Handbook for the Digital Engineer, 2nd ed.
//      1996.p.(82)
//
//  The transform coefficients of equation below correspond to the standard for
//  NTSC red, green and blue CRT phosphors (1953) that are standardized in the
//  ITU-R Recommendation BT. 601-2 (formerly CCIR Rec. 601-2).
//
//  The basic equation to compute non-linear video luma (monochrome) from non-linear
//  (gamma-corrected) RGB(R'G'B') is:
//
//  Y' = 0.299 * R' + 0.587 * G' + 0.114 * B';
//
//
*/

IPPAPI(IppStatus,ippiRGBToGray_8u_C3C1R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep,IppiSize roiSize))
IPPAPI(IppStatus,ippiRGBToGray_16u_C3C1R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep,IppiSize roiSize))
IPPAPI(IppStatus,ippiRGBToGray_32f_C3C1R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep,IppiSize roiSize))

IPPAPI(IppStatus,ippiRGBToGray_8u_AC4C1R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep,IppiSize roiSize))
IPPAPI(IppStatus,ippiRGBToGray_16u_AC4C1R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep,IppiSize roiSize))
IPPAPI(IppStatus,ippiRGBToGray_32f_AC4C1R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep,IppiSize roiSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiRGBToLUV,     ippiLUVToRGB
//  Purpose:    Converts an RGB image to the LUV color model and vice versa.
//  Parameters:
//    pSrc          Pointer to the source image ROI
//    srcStep       Step through the source image (bytes)
//    pDst          Pointer to the destination image ROI
//    dstStep       Step through the destination image (bytes)
//    dstRoiSize    size of the ROI
//  Returns:
//           ippStsNullPtrErr  pSrc == NULL, or pDst == NULL
//           ippStsStepErr     srcStep or dstStep is less than or equal to zero
//           ippStsSizeErr     roiSize has a field with zero or negative value
//           ippStsNoErr       No errors
//  Reference:
//     Computer Graphics: Principles and Practices. James D. Foley... [et al]. 2nd edition.
//     Addison-Wesley, 1990.p.(584)
//
//    At first an RGB image is converted to the XYZ format image (see the functions
//    ippiRGBToXYZ), then to the CIELUV with the white point D65 and CIE chromaticity
//    coordinates of white point (xn,yn) = (0.312713, 0.329016), and Yn = 1.0 - the luminance of white point.
//
//       L = 116. * (Y/Yn)**1/3. - 16.
//       U = 13. * L * ( u - un )
//       V = 13. * L * ( v - vn )
//      These are quantized and limited to the 8-bit range of 0 to 255.
//       L =   L * 255. / 100.
//       U = ( U + 134. ) * 255. / 354.
//       V = ( V + 140. ) * 255. / 256.
//       where:
//       u' = 4.*X / (X + 15.*Y + 3.*Z)
//       v' = 9.*Y / (X + 15.*Y + 3.*Z)
//       un = 4.*xn / ( -2.*xn + 12.*yn + 3. )
//       vn = 9.*yn / ( -2.*xn + 12.*yn + 3. ).
//       xn, yn is the CIE chromaticity coordinates of white point.
//       Yn = 255. is the luminance of white point.
//
//       The L component values are in the range [0..100], the U component values are
//       in the range [-134..220], and the V component values are in the range [-140..122].
//
//      The CIELUV to RGB conversion is performed as following. At first
//      a LUV image is converted to the XYZ image
//       L  =   L * 100./ 255.
//       U  = ( U * 354./ 255.) - 134.
//       V  = ( V * 256./ 255.) - 140.
//       u = U / ( 13.* L ) + un
//       v = V / ( 13.* L ) + vn
//       Y = (( L + 16. ) / 116. )**3.
//       Y *= Yn
//       X =  -9.* Y * u / (( u - 4.)* v - u * v )
//       Z = ( 9.*Y - 15*v*Y - v*X ) / 3. * v
//       where:
//       un = 4.*xn / ( -2.*xn + 12.*yn + 3. )
//       vn = 9.*yn / ( -2.*xn + 12.*yn + 3. ).
//       xn, yn is the CIE chromaticity coordinates of white point.
//       Yn = 255. is the luminance of white point.
//
//     Then the XYZ image is converted to the RGB image (see the functions
//     ippiXYZToRGB).
//
*/
IPPAPI(IppStatus, ippiRGBToLUV_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiLUVToRGB_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiRGBToLUV_16u_C3R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiLUVToRGB_16u_C3R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiRGBToLUV_16s_C3R,(const Ipp16s* pSrc,int srcStep,Ipp16s* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiLUVToRGB_16s_C3R,(const Ipp16s* pSrc,int srcStep,Ipp16s* pDst, int dstStep, IppiSize roiSize))

IPPAPI(IppStatus, ippiRGBToLUV_32f_C3R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiLUVToRGB_32f_C3R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep, IppiSize roiSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiRGBToXYZ,     ippiXYZToRGB
//  Purpose:    Converts an RGB image to the XYZ color model and vice versa.
//  Parameters:
//    pSrc          Pointer to the source image ROI
//    srcStep       Step through the source image (bytes)
//    pDst          Pointer to the destination image ROI
//    dstStep       Step through the destination image (bytes)
//    dstRoiSize    size of the ROI
//  Returns:
//           ippStsNullPtrErr  pSrc == NULL, or pDst == NULL
//           ippStsStepErr     srcStep or dstStep is less than or equal to zero
//           ippStsSizeErr     roiSize has a field with zero or negative value
//           ippStsNoErr       No errors
//  Reference:
//      David F. Rogers
//      Procedural Elements for Computer Graphics.
//      1985.
//
//  The basic equations to convert between Rec. 709 RGB (with its D65 white point) and CIE XYZ are:
//
//       X =  0.412453 * R + 0.35758 * G + 0.180423 * B
//       Y =  0.212671 * R + 0.71516 * G + 0.072169 * B
//       Z =  0.019334 * R + 0.119193* G + 0.950227 * B
//
//       R = 3.240479 * X - 1.53715  * Y  - 0.498535 * Z
//       G =-0.969256 * X + 1.875991 * Y  + 0.041556 * Z
//       B = 0.055648 * X - 0.204043 * Y  + 1.057311 * Z
//  Equations are given above in assumption that the X,Y,Z,R,G, and B
//   values are in the range [0..1].
//   Y, C1, C2, R, G, B - are scaled to the range:
//           [0..1]                      for the 32f depth,
//           [0..IPP_MAX_8u]             for the 8u depth,
//           [0..IPP_MAX_16u]            for the 16u depth,
//           [IPP_MIN_16s..IPP_MAX_16s]  for the 16s depth.
//
*/

IPPAPI(IppStatus, ippiRGBToXYZ_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiXYZToRGB_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiRGBToXYZ_16u_C3R,(const Ipp16u* pSrc, int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiXYZToRGB_16u_C3R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiRGBToXYZ_32f_C3R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiXYZToRGB_32f_C3R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep, IppiSize roiSize))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:        ippiRGBToHSV,   ippiHSVToRGB
//  Purpose:    Converts an RGB image to the HSV color model and vice versa
//  Parameters:
//     pSrc      Pointer to the source image , points to point(0,0)
//     pDst      Pointer to the destination image , points to point(0,0)
//     roiSize   Size of the ROI in pixels.
//     srcStep   Step in bytes through the source image to jump on the next line
//     dstStep   Step in bytes through the destination image to jump on the next line
//  Returns:
//           ippStsNullPtrErr  pSrc == NULL, or pDst == NULL
//           ippStsStepErr     srcStep or dstStep is less than or equal to zero
//           ippStsSizeErr     roiSize has a field with zero or negative value
//           ippStsNoErr       No errors
//  Reference:
//      David F.Rogers
//      Procedural Elements for Computer Graphics
//      1985.pp.(401-403)
//
//       H is the hue red at 0 degrees, which has range [0 .. 360 degrees],
//       S is the saturation,
//       V is the value
//       The RGB to HSV conversion algorithm in pseudo code:
//   Value:
//      V = max(R,G,B);
//   Saturation:
//      temp = min(R,G,B);
//      if V = 0 then // achromatic case
//          S = 0
//          H = 0
//      else // chromatics case
//          S = (V - temp)/V
//   Hue:
//      Cr = (V - R) / (V - temp)
//      Cg = (V - G) / (V - temp)
//      Cb = (V - B) / (V - temp)
//      if R = V then H = Cb - Cg
//      if G = V then H = 2 + Cr - Cb
//      if B = V then H = 4 + Cg - Cr
//      H = 60*H
//      if H < 0 then H = H + 360
//
//      The HSV to RGB conversion algorithm in pseudo code:
//      if S = 0 then
//         R = G = B = V
//      else
//          if H = 360 then
//              H = 0
//          else
//              H = H/60
//           I = floor(H)
//           F = H - I;
//           M = V * ( 1 - S);
//           N = V * ( 1 - S * F);
//           K = V * ( 1 - S * (1 - F));
//           if(I == 0)then{ R = V;G = K;B = M;}
//           if(I == 1)then{ R = N;G = V;B = M;}
//           if(I == 2)then{ R = M;G = V;B = K;}
//           if(I == 3)then{ R = M;G = N;B = V;}
//           if(I == 4)then{ R = K;G = M;B = V;}
//           if(I == 5)then{ R = V;G = M;B = N;}
//
//           in the range [0..IPP_MAX_8u ] for the 8u depth,
//           in the range [0..IPP_MAX_16u] for the 16u depth,
//
*/

IPPAPI(IppStatus, ippiRGBToHSV_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiHSVToRGB_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiRGBToHSV_16u_C3R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiHSVToRGB_16u_C3R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:     ippiRGBToHLS,     ippiHLSToRGB
//            ippiBGRToHLS,     ippiHLSToBGR
//  Purpose:    Converts an RGB(BGR) image to the HLS color model and vice versa
//  Parameters:
//     pSrc      Pointer to the source image , points to point(0,0)
//     pDst      Pointer to the destination image , points to point(0,0)
//     roiSize   Size of the ROI in pixels. Since the function performs point
//               operations (without a border), the ROI may be the whole image.
//     srcStep   Step in bytes through the source image to jump on the next line
//     dstStep   Step in bytes through the destination image to jump on the next line
//  Returns:
//           ippStsNullPtrErr  pSrc == NULL, or pDst == NULL
//           ippStsStepErr     srcStep or dstStep is less than or equal to zero
//           ippStsSizeErr     roiSize has a field with zero or negative value
//           ippStsNoErr       No errors
//  RGB and HLS values for the 32f data type should be in the range [0..1]
//  Reference:
//      David F.Rogers
//      Procedural Elements for Computer Graphics
//      1985.pp.(403-406)
//
//       H is the hue red at 0 degrees, which has range [0 .. 360 degrees],
//       L is the lightness,
//       S is the saturation,
//
//       The RGB to HLS conversion algorithm in pseudo code:
//   Lightness:
//      M1 = max(R,G,B); M2 = max(R,G,B); L = (M1+M2)/2
//   Saturation:
//      if M1 = M2 then // achromatic case
//          S = 0
//          H = 0
//      else // chromatics case
//          if L <= 0.5 then
//               S = (M1-M2) / (M1+M2)
//          else
//               S = (M1-M2) / (2-M1-M2)
//   Hue:
//      Cr = (M1-R) / (M1-M2)
//      Cg = (M1-G) / (M1-M2)
//      Cb = (M1-B) / (M1-M2)
//      if R = M2 then H = Cb - Cg
//      if G = M2 then H = 2 + Cr - Cb
//      if B = M2 then H = 4 + Cg - Cr
//      H = 60*H
//      if H < 0 then H = H + 360
//
//      The HSL to RGB conversion algorithm in pseudo code:
//      if L <= 0.5 then
//           M2 = L *(1 + S)
//      else
//           M2 = L + S - L * S
//      M1 = 2 * L - M2
//      if S = 0 then
//         R = G = B = L
//      else
//          h = H + 120
//          if h > 360 then
//              h = h - 360

//          if h  <  60 then
//              R = ( M1 + ( M2 - M1 ) * h / 60)
//          else if h < 180 then
//              R = M2
//          else if h < 240 then
//              R = M1 + ( M2 - M1 ) * ( 240 - h ) / 60
//          else
//              R = M1
//          h = H
//          if h  <  60 then
//              G = ( M1 + ( M2 - M1 ) * h / 60
//          else if h < 180 then
//              G = M2
//          else if h < 240 then
//              G = M1 + ( M2 - M1 ) * ( 240 - h ) / 60
//          else
//              G  = M1
//          h = H - 120
//          if h < 0 then
//              h += 360
//          if h  <  60 then
//              B = ( M1 + ( M2 - M1 ) * h / 60
//          else if h < 180 then
//              B = M2
//          else if h < 240 then
//              B = M1 + ( M2 - M1 ) * ( 240 - h ) / 60
//          else
//              B = M1
//
//    H,L,S,R,G,B are scaled to the range:
//             [0..1]                      for the 32f depth,
//             [0..IPP_MAX_8u]             for the 8u depth,
//             [0..IPP_MAX_16u]            for the 16u depth,
//             [IPP_MIN_16S..IPP_MAX_16s]  for the 16s depth.
//
*/

IPPAPI(IppStatus, ippiRGBToHLS_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiHLSToRGB_8u_C3R,(const Ipp8u* pSrc,int srcStep,Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiRGBToHLS_16u_C3R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiHLSToRGB_16u_C3R,(const Ipp16u* pSrc,int srcStep,Ipp16u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiRGBToHLS_32f_C3R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiHLSToRGB_32f_C3R,(const Ipp32f* pSrc,int srcStep,Ipp32f* pDst, int dstStep, IppiSize roiSize))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiRGBToYUV_8u_C3R,  ippiYUVToRGB_8u_C3R.ippiRGBToYUV_8u_C3R
//              ippiRGBToYUV_8u_AC4R, ippiYUVToRGB_8u_AC4R.
//              ippiRGBToYUV_8u_P3R,  ippiYUVToRGB_8u_P3R.
//              ippiRGBToYUV_8u_C3P3R,ippiYUVToRGB_8u_P3C3R.
//  Purpose:    Converts an RGB image to the YUV color model and vice versa.
//  Parameters:
//     pSrc   Pointer to the source image (for pixel-order data).An array of pointers
//            to separate source color planes (for plane-order data)
//     pDst   Pointer to the destination image (for pixel-order data).An array of
//            pointers to separate destination color planes (for of plane-order data)
//     roiSize   Size of ROI in pixels.
//     srcStep   Step in bytes through the source image to jump on the next line
//     dstStep   Step in bytes through the destination image to jump on the next line
//  Returns:
//           ippStsNullPtrErr  pSrc == NULL, or pDst == NULL
//           ippStsStepErr,    srcStep or dstStep is less than or equal to zero
//           ippStsSizeErr     roiSize has a field with zero or negative value
//           ippStsNoErr       No errors
//  Reference:
//      Jack Keith
//      Video Demystified: A Handbook for the Digital Engineer, 2nd ed.
//      1996.pp.(40-41)
//
//     The YUV color space is the basic color space used by the PAL, NTSC, and
//  SECAM composite color video standards.
//
//  These functions operate with gamma-corrected images.
//  The basic equations for conversion between gamma-corrected RGB(R'G'B')and YUV are:
//
//       Y' =  0.299*R' + 0.587*G' + 0.114*B'
//       U  = -0.147*R' - 0.289*G' + 0.436*B' = 0.492*(B' - Y' )
//       V  =  0.615*R' - 0.515*G' - 0.100*B' = 0.877*(R' - Y' )
//
//       R' = Y' + 1.140 * V
//       G' = Y' - 0.394 * U - 0.581 * V
//       B' = Y' + 2.032 * U
//
//     For digital RGB values in the range [0 .. 255], Y has a range [0..255],
//   U a range [-112 .. +112], and V a range [-157..+157].
//
//   These equations are usually scaled to simplify the implementation in an actual
//   NTSC or PAL digital encoder or decoder.
//
*/
IPPAPI(IppStatus, ippiRGBToYUV_8u_C3R,(const Ipp8u* pSrc, int srcStep , Ipp8u* pDst, int dstStep, IppiSize roiSize))
IPPAPI(IppStatus, ippiYUVToRGB_8u_C3R,(const Ipp8u* pSrc, int srcStep , Ipp8u* pDst, int dstStep, IppiSize roiSize))


#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_H__ */
