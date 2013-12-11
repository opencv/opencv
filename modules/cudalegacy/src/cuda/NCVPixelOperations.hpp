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

#ifndef _ncv_pixel_operations_hpp_
#define _ncv_pixel_operations_hpp_

#include <limits.h>
#include <float.h>
#include "opencv2/cudalegacy/NCV.hpp"

template<typename TBase> inline __host__ __device__ TBase _pixMaxVal();
template<> static inline __host__ __device__ Ncv8u  _pixMaxVal<Ncv8u>()  {return UCHAR_MAX;}
template<> static inline __host__ __device__ Ncv16u _pixMaxVal<Ncv16u>() {return USHRT_MAX;}
template<> static inline __host__ __device__ Ncv32u _pixMaxVal<Ncv32u>() {return  UINT_MAX;}
template<> static inline __host__ __device__ Ncv8s  _pixMaxVal<Ncv8s>()  {return  SCHAR_MAX;}
template<> static inline __host__ __device__ Ncv16s _pixMaxVal<Ncv16s>() {return  SHRT_MAX;}
template<> static inline __host__ __device__ Ncv32s _pixMaxVal<Ncv32s>() {return   INT_MAX;}
template<> static inline __host__ __device__ Ncv32f _pixMaxVal<Ncv32f>() {return   FLT_MAX;}
template<> static inline __host__ __device__ Ncv64f _pixMaxVal<Ncv64f>() {return   DBL_MAX;}

template<typename TBase> inline __host__ __device__ TBase _pixMinVal();
template<> static inline __host__ __device__ Ncv8u  _pixMinVal<Ncv8u>()  {return 0;}
template<> static inline __host__ __device__ Ncv16u _pixMinVal<Ncv16u>() {return 0;}
template<> static inline __host__ __device__ Ncv32u _pixMinVal<Ncv32u>() {return 0;}
template<> static inline __host__ __device__ Ncv8s  _pixMinVal<Ncv8s>()  {return SCHAR_MIN;}
template<> static inline __host__ __device__ Ncv16s _pixMinVal<Ncv16s>() {return SHRT_MIN;}
template<> static inline __host__ __device__ Ncv32s _pixMinVal<Ncv32s>() {return INT_MIN;}
template<> static inline __host__ __device__ Ncv32f _pixMinVal<Ncv32f>() {return FLT_MIN;}
template<> static inline __host__ __device__ Ncv64f _pixMinVal<Ncv64f>() {return DBL_MIN;}

template<typename Tvec> struct TConvVec2Base;
template<> struct TConvVec2Base<uchar1>  {typedef Ncv8u TBase;};
template<> struct TConvVec2Base<uchar3>  {typedef Ncv8u TBase;};
template<> struct TConvVec2Base<uchar4>  {typedef Ncv8u TBase;};
template<> struct TConvVec2Base<ushort1> {typedef Ncv16u TBase;};
template<> struct TConvVec2Base<ushort3> {typedef Ncv16u TBase;};
template<> struct TConvVec2Base<ushort4> {typedef Ncv16u TBase;};
template<> struct TConvVec2Base<uint1>   {typedef Ncv32u TBase;};
template<> struct TConvVec2Base<uint3>   {typedef Ncv32u TBase;};
template<> struct TConvVec2Base<uint4>   {typedef Ncv32u TBase;};
template<> struct TConvVec2Base<float1>  {typedef Ncv32f TBase;};
template<> struct TConvVec2Base<float3>  {typedef Ncv32f TBase;};
template<> struct TConvVec2Base<float4>  {typedef Ncv32f TBase;};
template<> struct TConvVec2Base<double1> {typedef Ncv64f TBase;};
template<> struct TConvVec2Base<double3> {typedef Ncv64f TBase;};
template<> struct TConvVec2Base<double4> {typedef Ncv64f TBase;};

#define NC(T)       (sizeof(T) / sizeof(TConvVec2Base<T>::TBase))

template<typename TBase, Ncv32u NC> struct TConvBase2Vec;
template<> struct TConvBase2Vec<Ncv8u, 1>  {typedef uchar1 TVec;};
template<> struct TConvBase2Vec<Ncv8u, 3>  {typedef uchar3 TVec;};
template<> struct TConvBase2Vec<Ncv8u, 4>  {typedef uchar4 TVec;};
template<> struct TConvBase2Vec<Ncv16u, 1> {typedef ushort1 TVec;};
template<> struct TConvBase2Vec<Ncv16u, 3> {typedef ushort3 TVec;};
template<> struct TConvBase2Vec<Ncv16u, 4> {typedef ushort4 TVec;};
template<> struct TConvBase2Vec<Ncv32u, 1> {typedef uint1 TVec;};
template<> struct TConvBase2Vec<Ncv32u, 3> {typedef uint3 TVec;};
template<> struct TConvBase2Vec<Ncv32u, 4> {typedef uint4 TVec;};
template<> struct TConvBase2Vec<Ncv32f, 1> {typedef float1 TVec;};
template<> struct TConvBase2Vec<Ncv32f, 3> {typedef float3 TVec;};
template<> struct TConvBase2Vec<Ncv32f, 4> {typedef float4 TVec;};
template<> struct TConvBase2Vec<Ncv64f, 1> {typedef double1 TVec;};
template<> struct TConvBase2Vec<Ncv64f, 3> {typedef double3 TVec;};
template<> struct TConvBase2Vec<Ncv64f, 4> {typedef double4 TVec;};

//TODO: consider using CUDA intrinsics to avoid branching
template<typename Tin> static inline __host__ __device__ void _TDemoteClampZ(Tin &a, Ncv8u &out) {out = (Ncv8u)CLAMP_0_255(a);};
template<typename Tin> static inline __host__ __device__ void _TDemoteClampZ(Tin &a, Ncv16u &out) {out = (Ncv16u)CLAMP(a, 0, USHRT_MAX);}
template<typename Tin> static inline __host__ __device__ void _TDemoteClampZ(Tin &a, Ncv32u &out) {out = (Ncv32u)CLAMP(a, 0, UINT_MAX);}
template<typename Tin> static inline __host__ __device__ void _TDemoteClampZ(Tin &a, Ncv32f &out) {out = (Ncv32f)a;}

//TODO: consider using CUDA intrinsics to avoid branching
template<typename Tin> static inline __host__ __device__ void _TDemoteClampNN(Tin &a, Ncv8u &out) {out = (Ncv8u)CLAMP_0_255(a+0.5f);}
template<typename Tin> static inline __host__ __device__ void _TDemoteClampNN(Tin &a, Ncv16u &out) {out = (Ncv16u)CLAMP(a+0.5f, 0, USHRT_MAX);}
template<typename Tin> static inline __host__ __device__ void _TDemoteClampNN(Tin &a, Ncv32u &out) {out = (Ncv32u)CLAMP(a+0.5f, 0, UINT_MAX);}
template<typename Tin> static inline __host__ __device__ void _TDemoteClampNN(Tin &a, Ncv32f &out) {out = (Ncv32f)a;}

template<typename Tout> inline Tout _pixMakeZero();
template<> static inline __host__ __device__ uchar1 _pixMakeZero<uchar1>() {return make_uchar1(0);}
template<> static inline __host__ __device__ uchar3 _pixMakeZero<uchar3>() {return make_uchar3(0,0,0);}
template<> static inline __host__ __device__ uchar4 _pixMakeZero<uchar4>() {return make_uchar4(0,0,0,0);}
template<> static inline __host__ __device__ ushort1 _pixMakeZero<ushort1>() {return make_ushort1(0);}
template<> static inline __host__ __device__ ushort3 _pixMakeZero<ushort3>() {return make_ushort3(0,0,0);}
template<> static inline __host__ __device__ ushort4 _pixMakeZero<ushort4>() {return make_ushort4(0,0,0,0);}
template<> static inline __host__ __device__ uint1 _pixMakeZero<uint1>() {return make_uint1(0);}
template<> static inline __host__ __device__ uint3 _pixMakeZero<uint3>() {return make_uint3(0,0,0);}
template<> static inline __host__ __device__ uint4 _pixMakeZero<uint4>() {return make_uint4(0,0,0,0);}
template<> static inline __host__ __device__ float1 _pixMakeZero<float1>() {return make_float1(0.f);}
template<> static inline __host__ __device__ float3 _pixMakeZero<float3>() {return make_float3(0.f,0.f,0.f);}
template<> static inline __host__ __device__ float4 _pixMakeZero<float4>() {return make_float4(0.f,0.f,0.f,0.f);}
template<> static inline __host__ __device__ double1 _pixMakeZero<double1>() {return make_double1(0.);}
template<> static inline __host__ __device__ double3 _pixMakeZero<double3>() {return make_double3(0.,0.,0.);}
template<> static inline __host__ __device__ double4 _pixMakeZero<double4>() {return make_double4(0.,0.,0.,0.);}

static inline __host__ __device__ uchar1 _pixMake(Ncv8u x) {return make_uchar1(x);}
static inline __host__ __device__ uchar3 _pixMake(Ncv8u x, Ncv8u y, Ncv8u z) {return make_uchar3(x,y,z);}
static inline __host__ __device__ uchar4 _pixMake(Ncv8u x, Ncv8u y, Ncv8u z, Ncv8u w) {return make_uchar4(x,y,z,w);}
static inline __host__ __device__ ushort1 _pixMake(Ncv16u x) {return make_ushort1(x);}
static inline __host__ __device__ ushort3 _pixMake(Ncv16u x, Ncv16u y, Ncv16u z) {return make_ushort3(x,y,z);}
static inline __host__ __device__ ushort4 _pixMake(Ncv16u x, Ncv16u y, Ncv16u z, Ncv16u w) {return make_ushort4(x,y,z,w);}
static inline __host__ __device__ uint1 _pixMake(Ncv32u x) {return make_uint1(x);}
static inline __host__ __device__ uint3 _pixMake(Ncv32u x, Ncv32u y, Ncv32u z) {return make_uint3(x,y,z);}
static inline __host__ __device__ uint4 _pixMake(Ncv32u x, Ncv32u y, Ncv32u z, Ncv32u w) {return make_uint4(x,y,z,w);}
static inline __host__ __device__ float1 _pixMake(Ncv32f x) {return make_float1(x);}
static inline __host__ __device__ float3 _pixMake(Ncv32f x, Ncv32f y, Ncv32f z) {return make_float3(x,y,z);}
static inline __host__ __device__ float4 _pixMake(Ncv32f x, Ncv32f y, Ncv32f z, Ncv32f w) {return make_float4(x,y,z,w);}
static inline __host__ __device__ double1 _pixMake(Ncv64f x) {return make_double1(x);}
static inline __host__ __device__ double3 _pixMake(Ncv64f x, Ncv64f y, Ncv64f z) {return make_double3(x,y,z);}
static inline __host__ __device__ double4 _pixMake(Ncv64f x, Ncv64f y, Ncv64f z, Ncv64f w) {return make_double4(x,y,z,w);}


template<typename Tin, typename Tout, Ncv32u CN> struct __pixDemoteClampZ_CN {static __host__ __device__ Tout _pixDemoteClampZ_CN(Tin &pix);};

template<typename Tin, typename Tout> struct __pixDemoteClampZ_CN<Tin, Tout, 1> {
static __host__ __device__ Tout _pixDemoteClampZ_CN(Tin &pix)
{
    Tout out;
    _TDemoteClampZ(pix.x, out.x);
    return out;
}};

template<typename Tin, typename Tout> struct __pixDemoteClampZ_CN<Tin, Tout, 3> {
static __host__ __device__ Tout _pixDemoteClampZ_CN(Tin &pix)
{
    Tout out;
    _TDemoteClampZ(pix.x, out.x);
    _TDemoteClampZ(pix.y, out.y);
    _TDemoteClampZ(pix.z, out.z);
    return out;
}};

template<typename Tin, typename Tout> struct __pixDemoteClampZ_CN<Tin, Tout, 4> {
static __host__ __device__ Tout _pixDemoteClampZ_CN(Tin &pix)
{
    Tout out;
    _TDemoteClampZ(pix.x, out.x);
    _TDemoteClampZ(pix.y, out.y);
    _TDemoteClampZ(pix.z, out.z);
    _TDemoteClampZ(pix.w, out.w);
    return out;
}};

template<typename Tin, typename Tout> static inline __host__ __device__ Tout _pixDemoteClampZ(Tin &pix)
{
    return __pixDemoteClampZ_CN<Tin, Tout, NC(Tin)>::_pixDemoteClampZ_CN(pix);
}


template<typename Tin, typename Tout, Ncv32u CN> struct __pixDemoteClampNN_CN {static __host__ __device__ Tout _pixDemoteClampNN_CN(Tin &pix);};

template<typename Tin, typename Tout> struct __pixDemoteClampNN_CN<Tin, Tout, 1> {
static __host__ __device__ Tout _pixDemoteClampNN_CN(Tin &pix)
{
    Tout out;
    _TDemoteClampNN(pix.x, out.x);
    return out;
}};

template<typename Tin, typename Tout> struct __pixDemoteClampNN_CN<Tin, Tout, 3> {
static __host__ __device__ Tout _pixDemoteClampNN_CN(Tin &pix)
{
    Tout out;
    _TDemoteClampNN(pix.x, out.x);
    _TDemoteClampNN(pix.y, out.y);
    _TDemoteClampNN(pix.z, out.z);
    return out;
}};

template<typename Tin, typename Tout> struct __pixDemoteClampNN_CN<Tin, Tout, 4> {
static __host__ __device__ Tout _pixDemoteClampNN_CN(Tin &pix)
{
    Tout out;
    _TDemoteClampNN(pix.x, out.x);
    _TDemoteClampNN(pix.y, out.y);
    _TDemoteClampNN(pix.z, out.z);
    _TDemoteClampNN(pix.w, out.w);
    return out;
}};

template<typename Tin, typename Tout> static inline __host__ __device__ Tout _pixDemoteClampNN(Tin &pix)
{
    return __pixDemoteClampNN_CN<Tin, Tout, NC(Tin)>::_pixDemoteClampNN_CN(pix);
}


template<typename Tin, typename Tout, typename Tw, Ncv32u CN> struct __pixScale_CN {static __host__ __device__ Tout _pixScale_CN(Tin &pix, Tw w);};

template<typename Tin, typename Tout, typename Tw> struct __pixScale_CN<Tin, Tout, Tw, 1> {
static __host__ __device__ Tout _pixScale_CN(Tin &pix, Tw w)
{
    Tout out;
    typedef typename TConvVec2Base<Tout>::TBase TBout;
    out.x = (TBout)(pix.x * w);
    return out;
}};

template<typename Tin, typename Tout, typename Tw> struct __pixScale_CN<Tin, Tout, Tw, 3> {
static __host__ __device__ Tout _pixScale_CN(Tin &pix, Tw w)
{
    Tout out;
    typedef typename TConvVec2Base<Tout>::TBase TBout;
    out.x = (TBout)(pix.x * w);
    out.y = (TBout)(pix.y * w);
    out.z = (TBout)(pix.z * w);
    return out;
}};

template<typename Tin, typename Tout, typename Tw> struct __pixScale_CN<Tin, Tout, Tw, 4> {
static __host__ __device__ Tout _pixScale_CN(Tin &pix, Tw w)
{
    Tout out;
    typedef typename TConvVec2Base<Tout>::TBase TBout;
    out.x = (TBout)(pix.x * w);
    out.y = (TBout)(pix.y * w);
    out.z = (TBout)(pix.z * w);
    out.w = (TBout)(pix.w * w);
    return out;
}};

template<typename Tin, typename Tout, typename Tw> static __host__ __device__ Tout _pixScale(Tin &pix, Tw w)
{
    return __pixScale_CN<Tin, Tout, Tw, NC(Tin)>::_pixScale_CN(pix, w);
}


template<typename Tin, typename Tout, Ncv32u CN> struct __pixAdd_CN {static __host__ __device__ Tout _pixAdd_CN(Tout &pix1, Tin &pix2);};

template<typename Tin, typename Tout> struct __pixAdd_CN<Tin, Tout, 1> {
static __host__ __device__ Tout _pixAdd_CN(Tout &pix1, Tin &pix2)
{
    Tout out;
    out.x = pix1.x + pix2.x;
    return out;
}};

template<typename Tin, typename Tout> struct __pixAdd_CN<Tin, Tout, 3> {
static __host__ __device__ Tout _pixAdd_CN(Tout &pix1, Tin &pix2)
{
    Tout out;
    out.x = pix1.x + pix2.x;
    out.y = pix1.y + pix2.y;
    out.z = pix1.z + pix2.z;
    return out;
}};

template<typename Tin, typename Tout> struct __pixAdd_CN<Tin, Tout, 4> {
static __host__ __device__ Tout _pixAdd_CN(Tout &pix1, Tin &pix2)
{
    Tout out;
    out.x = pix1.x + pix2.x;
    out.y = pix1.y + pix2.y;
    out.z = pix1.z + pix2.z;
    out.w = pix1.w + pix2.w;
    return out;
}};

template<typename Tin, typename Tout> static __host__ __device__ Tout _pixAdd(Tout &pix1, Tin &pix2)
{
    return __pixAdd_CN<Tin, Tout, NC(Tin)>::_pixAdd_CN(pix1, pix2);
}


template<typename Tin, typename Tout, Ncv32u CN> struct __pixDist_CN {static __host__ __device__ Tout _pixDist_CN(Tin &pix1, Tin &pix2);};

template<typename Tin, typename Tout> struct __pixDist_CN<Tin, Tout, 1> {
static __host__ __device__ Tout _pixDist_CN(Tin &pix1, Tin &pix2)
{
    return Tout(SQR(pix1.x - pix2.x));
}};

template<typename Tin, typename Tout> struct __pixDist_CN<Tin, Tout, 3> {
static __host__ __device__ Tout _pixDist_CN(Tin &pix1, Tin &pix2)
{
    return Tout(SQR(pix1.x - pix2.x) + SQR(pix1.y - pix2.y) + SQR(pix1.z - pix2.z));
}};

template<typename Tin, typename Tout> struct __pixDist_CN<Tin, Tout, 4> {
static __host__ __device__ Tout _pixDist_CN(Tin &pix1, Tin &pix2)
{
    return Tout(SQR(pix1.x - pix2.x) + SQR(pix1.y - pix2.y) + SQR(pix1.z - pix2.z) + SQR(pix1.w - pix2.w));
}};

template<typename Tin, typename Tout> static __host__ __device__ Tout _pixDist(Tin &pix1, Tin &pix2)
{
    return __pixDist_CN<Tin, Tout, NC(Tin)>::_pixDist_CN(pix1, pix2);
}


template <typename T> struct TAccPixWeighted;
template<> struct TAccPixWeighted<uchar1> {typedef double1 type;};
template<> struct TAccPixWeighted<uchar3> {typedef double3 type;};
template<> struct TAccPixWeighted<uchar4> {typedef double4 type;};
template<> struct TAccPixWeighted<ushort1> {typedef double1 type;};
template<> struct TAccPixWeighted<ushort3> {typedef double3 type;};
template<> struct TAccPixWeighted<ushort4> {typedef double4 type;};
template<> struct TAccPixWeighted<float1> {typedef double1 type;};
template<> struct TAccPixWeighted<float3> {typedef double3 type;};
template<> struct TAccPixWeighted<float4> {typedef double4 type;};

template<typename Tfrom> struct TAccPixDist {};
template<> struct TAccPixDist<uchar1> {typedef Ncv32u type;};
template<> struct TAccPixDist<uchar3> {typedef Ncv32u type;};
template<> struct TAccPixDist<uchar4> {typedef Ncv32u type;};
template<> struct TAccPixDist<ushort1> {typedef Ncv32u type;};
template<> struct TAccPixDist<ushort3> {typedef Ncv32u type;};
template<> struct TAccPixDist<ushort4> {typedef Ncv32u type;};
template<> struct TAccPixDist<float1> {typedef Ncv32f type;};
template<> struct TAccPixDist<float3> {typedef Ncv32f type;};
template<> struct TAccPixDist<float4> {typedef Ncv32f type;};

#endif //_ncv_pixel_operations_hpp_
