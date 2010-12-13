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

#ifndef __OPENCV_GPU_VECMATH_HPP__
#define __OPENCV_GPU_VECMATH_HPP__

#include "internal_shared.hpp"
#include "saturate_cast.hpp"

namespace cv
{
    namespace gpu
    {
        namespace device
        {
            template<typename T, int N> struct TypeVec;

            template<> struct TypeVec<uchar, 1> { typedef uchar vec_t; };
            template<> struct TypeVec<uchar1, 1> { typedef uchar1 vec_t; };
            template<> struct TypeVec<uchar, 2> { typedef uchar2 vec_t; };
            template<> struct TypeVec<uchar2, 2> { typedef uchar2 vec_t; };
            template<> struct TypeVec<uchar, 3> { typedef uchar3 vec_t; };
            template<> struct TypeVec<uchar3, 3> { typedef uchar3 vec_t; };
            template<> struct TypeVec<uchar, 4> { typedef uchar4 vec_t; };
            template<> struct TypeVec<uchar4, 4> { typedef uchar4 vec_t; };

            template<> struct TypeVec<char, 1> { typedef char vec_t; };
            template<> struct TypeVec<char1, 1> { typedef char1 vec_t; };
            template<> struct TypeVec<char, 2> { typedef char2 vec_t; };
            template<> struct TypeVec<char2, 2> { typedef char2 vec_t; };
            template<> struct TypeVec<char, 3> { typedef char3 vec_t; };
            template<> struct TypeVec<char3, 3> { typedef char3 vec_t; };
            template<> struct TypeVec<char, 4> { typedef char4 vec_t; };
            template<> struct TypeVec<char4, 4> { typedef char4 vec_t; };

            template<> struct TypeVec<ushort, 1> { typedef ushort vec_t; };
            template<> struct TypeVec<ushort1, 1> { typedef ushort1 vec_t; };
            template<> struct TypeVec<ushort, 2> { typedef ushort2 vec_t; };
            template<> struct TypeVec<ushort2, 2> { typedef ushort2 vec_t; };
            template<> struct TypeVec<ushort, 3> { typedef ushort3 vec_t; };
            template<> struct TypeVec<ushort3, 3> { typedef ushort3 vec_t; };
            template<> struct TypeVec<ushort, 4> { typedef ushort4 vec_t; };
            template<> struct TypeVec<ushort4, 4> { typedef ushort4 vec_t; };

            template<> struct TypeVec<short, 1> { typedef short vec_t; };
            template<> struct TypeVec<short1, 1> { typedef short1 vec_t; };
            template<> struct TypeVec<short, 2> { typedef short2 vec_t; };
            template<> struct TypeVec<short2, 2> { typedef short2 vec_t; };
            template<> struct TypeVec<short, 3> { typedef short3 vec_t; };
            template<> struct TypeVec<short3, 3> { typedef short3 vec_t; };
            template<> struct TypeVec<short, 4> { typedef short4 vec_t; };
            template<> struct TypeVec<short4, 4> { typedef short4 vec_t; };

            template<> struct TypeVec<uint, 1> { typedef uint vec_t; };
            template<> struct TypeVec<uint1, 1> { typedef uint1 vec_t; };
            template<> struct TypeVec<uint, 2> { typedef uint2 vec_t; };
            template<> struct TypeVec<uint2, 2> { typedef uint2 vec_t; };
            template<> struct TypeVec<uint, 3> { typedef uint3 vec_t; };
            template<> struct TypeVec<uint3, 3> { typedef uint3 vec_t; };
            template<> struct TypeVec<uint, 4> { typedef uint4 vec_t; };
            template<> struct TypeVec<uint4, 4> { typedef uint4 vec_t; };

            template<> struct TypeVec<int, 1> { typedef int vec_t; };
            template<> struct TypeVec<int1, 1> { typedef int1 vec_t; };
            template<> struct TypeVec<int, 2> { typedef int2 vec_t; };
            template<> struct TypeVec<int2, 2> { typedef int2 vec_t; };
            template<> struct TypeVec<int, 3> { typedef int3 vec_t; };
            template<> struct TypeVec<int3, 3> { typedef int3 vec_t; };
            template<> struct TypeVec<int, 4> { typedef int4 vec_t; };
            template<> struct TypeVec<int4, 4> { typedef int4 vec_t; };

            template<> struct TypeVec<float, 1> { typedef float vec_t; };
            template<> struct TypeVec<float1, 1> { typedef float1 vec_t; };
            template<> struct TypeVec<float, 2> { typedef float2 vec_t; };
            template<> struct TypeVec<float2, 2> { typedef float2 vec_t; };
            template<> struct TypeVec<float, 3> { typedef float3 vec_t; };
            template<> struct TypeVec<float3, 3> { typedef float3 vec_t; };
            template<> struct TypeVec<float, 4> { typedef float4 vec_t; };
            template<> struct TypeVec<float4, 4> { typedef float4 vec_t; };

            template<typename T> struct VecTraits;

            template<> struct VecTraits<uchar> 
            { 
                typedef uchar elem_t; 
                enum {cn=1};
                static __device__ __host__ uchar all(uchar v) {return v;}
                static __device__ __host__ uchar make(uchar x) {return x;}
            };
            template<> struct VecTraits<uchar1> 
            { 
                typedef uchar elem_t; 
                enum {cn=1};
                static __device__ __host__ uchar1 all(uchar v) {return make_uchar1(v);}
                static __device__ __host__ uchar1 make(uchar x) {return make_uchar1(x);}
            };
            template<> struct VecTraits<uchar2> 
            { 
                typedef uchar elem_t; 
                enum {cn=2}; 
                static __device__ __host__ uchar2 all(uchar v) {return make_uchar2(v, v);}
                static __device__ __host__ uchar2 make(uchar x, uchar y) {return make_uchar2(x, y);}
            };
            template<> struct VecTraits<uchar3> 
            { 
                typedef uchar elem_t; 
                enum {cn=3}; 
                static __device__ __host__ uchar3 all(uchar v) {return make_uchar3(v, v, v);}
                static __device__ __host__ uchar3 make(uchar x, uchar y, uchar z) {return make_uchar3(x, y, z);}
            };
            template<> struct VecTraits<uchar4> 
            { 
                typedef uchar elem_t; 
                enum {cn=4}; 
                static __device__ __host__ uchar4 all(uchar v) {return make_uchar4(v, v, v, v);}
                static __device__ __host__ uchar4 make(uchar x, uchar y, uchar z, uchar w) {return make_uchar4(x, y, z, w);}
            };

            template<> struct VecTraits<char> 
            { 
                typedef char elem_t; 
                enum {cn=1}; 
                static __device__ __host__ char all(char v) {return v;}
                static __device__ __host__ char make(char x) {return x;}
            };
            template<> struct VecTraits<char1> 
            { 
                typedef char elem_t; 
                enum {cn=1}; 
                static __device__ __host__ char1 all(char v) {return make_char1(v);}
                static __device__ __host__ char1 make(char x) {return make_char1(x);}
            };
            template<> struct VecTraits<char2> 
            { 
                typedef char elem_t; 
                enum {cn=2}; 
                static  __device__ __host__ char2 all(char v) {return make_char2(v, v);}
                static  __device__ __host__ char2 make(char x, char y) {return make_char2(x, y);}
            };
            template<> struct VecTraits<char3> 
            { 
                typedef char elem_t; 
                enum {cn=3}; 
                static __device__ __host__ char3 all(char v) {return make_char3(v, v, v);}
                static __device__ __host__ char3 make(char x, char y, char z) {return make_char3(x, y, z);}
            };
            template<> struct VecTraits<char4> 
            { 
                typedef char elem_t; 
                enum {cn=4}; 
                static __device__ __host__ char4 all(char v) {return make_char4(v, v, v, v);}
                static __device__ __host__ char4 make(char x, char y, char z, char w) {return make_char4(x, y, z, w);}
            };

            template<> struct VecTraits<ushort> 
            { 
                typedef ushort elem_t; 
                enum {cn=1}; 
                static __device__ __host__ ushort all(ushort v) {return v;}
                static __device__ __host__ ushort make(ushort x) {return x;}
            };
            template<> struct VecTraits<ushort1> 
            { 
                typedef ushort elem_t; 
                enum {cn=1}; 
                static __device__ __host__ ushort1 all(ushort v) {return make_ushort1(v);}
                static __device__ __host__ ushort1 make(ushort x) {return make_ushort1(x);}
            };
            template<> struct VecTraits<ushort2> 
            { 
                typedef ushort elem_t; 
                enum {cn=2}; 
                static __device__ __host__ ushort2 all(ushort v) {return make_ushort2(v, v);}
                static __device__ __host__ ushort2 make(ushort x, ushort y) {return make_ushort2(x, y);}
            };
            template<> struct VecTraits<ushort3> 
            { 
                typedef ushort elem_t; 
                enum {cn=3}; 
                static __device__ __host__ ushort3 all(ushort v) {return make_ushort3(v, v, v);}
                static __device__ __host__ ushort3 make(ushort x, ushort y, ushort z) {return make_ushort3(x, y, z);}
            };
            template<> struct VecTraits<ushort4> 
            { 
                typedef ushort elem_t; 
                enum {cn=4}; 
                static __device__ __host__ ushort4 all(ushort v) {return make_ushort4(v, v, v, v);}
                static __device__ __host__ ushort4 make(ushort x, ushort y, ushort z, ushort w) {return make_ushort4(x, y, z, w);}
            };

            template<> struct VecTraits<short> 
            { 
                typedef short elem_t; 
                enum {cn=1}; 
                static __device__ __host__ short all(short v) {return v;}
                static __device__ __host__ short make(short x) {return x;}
            };
            template<> struct VecTraits<short1> 
            { 
                typedef short elem_t; 
                enum {cn=1}; 
                static __device__ __host__ short1 all(short v) {return make_short1(v);}
                static __device__ __host__ short1 make(short x) {return make_short1(x);}
            };
            template<> struct VecTraits<short2> 
            { 
                typedef short elem_t; 
                enum {cn=2}; 
                static __device__ __host__ short2 all(short v) {return make_short2(v, v);}
                static __device__ __host__ short2 make(short x, short y) {return make_short2(x, y);}
            };
            template<> struct VecTraits<short3> 
            { 
                typedef short elem_t; 
                enum {cn=3}; 
                static __device__ __host__ short3 all(short v) {return make_short3(v, v, v);}
                static __device__ __host__ short3 make(short x, short y, short z) {return make_short3(x, y, z);}
            };
            template<> struct VecTraits<short4> 
            { 
                typedef short elem_t; 
                enum {cn=4}; 
                static __device__ __host__ short4 all(short v) {return make_short4(v, v, v, v);}
                static __device__ __host__ short4 make(short x, short y, short z, short w) {return make_short4(x, y, z, w);}
            };

            template<> struct VecTraits<uint> 
            { 
                typedef uint elem_t; 
                enum {cn=1}; 
                static __device__ __host__ uint all(uint v) {return v;}
                static __device__ __host__ uint make(uint x) {return x;}
            };
            template<> struct VecTraits<uint1> 
            { 
                typedef uint elem_t; 
                enum {cn=1}; 
                static __device__ __host__ uint1 all(uint v) {return make_uint1(v);}
                static __device__ __host__ uint1 make(uint x) {return make_uint1(x);}
            };
            template<> struct VecTraits<uint2> 
            { 
                typedef uint elem_t; 
                enum {cn=2}; 
                static __device__ __host__ uint2 all(uint v) {return make_uint2(v, v);}
                static __device__ __host__ uint2 make(uint x, uint y) {return make_uint2(x, y);}
            };
            template<> struct VecTraits<uint3> 
            { 
                typedef uint elem_t; 
                enum {cn=3}; 
                static __device__ __host__ uint3 all(uint v) {return make_uint3(v, v, v);}
                static __device__ __host__ uint3 make(uint x, uint y, uint z) {return make_uint3(x, y, z);}
            };
            template<> struct VecTraits<uint4> 
            { 
                typedef uint elem_t; 
                enum {cn=4}; 
                static __device__ __host__ uint4 all(uint v) {return make_uint4(v, v, v, v);}
                static __device__ __host__ uint4 make(uint x, uint y, uint z, uint w) {return make_uint4(x, y, z, w);}
            };

            template<> struct VecTraits<int> 
            { 
                typedef int elem_t; 
                enum {cn=1}; 
                static __device__ __host__ int all(int v) {return v;}
                static __device__ __host__ int make(int x) {return x;}
            };
            template<> struct VecTraits<int1> 
            { 
                typedef int elem_t; 
                enum {cn=1}; 
                static __device__ __host__ int1 all(int v) {return make_int1(v);}
                static __device__ __host__ int1 make(int x) {return make_int1(x);}
            };
            template<> struct VecTraits<int2> 
            { 
                typedef int elem_t; 
                enum {cn=2}; 
                static __device__ __host__ int2 all(int v) {return make_int2(v, v);}
                static __device__ __host__ int2 make(int x, int y) {return make_int2(x, y);}
            };
            template<> struct VecTraits<int3> 
            { 
                typedef int elem_t; 
                enum {cn=3}; 
                static __device__ __host__ int3 all(int v) {return make_int3(v, v, v);}
                static __device__ __host__ int3 make(int x, int y, int z) {return make_int3(x, y, z);}
            };
            template<> struct VecTraits<int4> 
            { 
                typedef int elem_t; 
                enum {cn=4}; 
                static __device__ __host__ int4 all(int v) {return make_int4(v, v, v, v);}
                static __device__ __host__ int4 make(int x, int y, int z, int w) {return make_int4(x, y, z, w);}
            };

            template<> struct VecTraits<float> 
            { 
                typedef float elem_t; 
                enum {cn=1}; 
                static __device__ __host__ float all(float v) {return v;}
                static __device__ __host__ float make(float x) {return x;}
            };
            template<> struct VecTraits<float1> 
            { 
                typedef float elem_t; 
                enum {cn=1}; 
                static __device__ __host__ float1 all(float v) {return make_float1(v);}
                static __device__ __host__ float1 make(float x) {return make_float1(x);}
            };
            template<> struct VecTraits<float2> 
            { 
                typedef float elem_t; 
                enum {cn=2}; 
                static __device__ __host__ float2 all(float v) {return make_float2(v, v);}
                static __device__ __host__ float2 make(float x, float y) {return make_float2(x, y);}
            };
            template<> struct VecTraits<float3> 
            { 
                typedef float elem_t; 
                enum {cn=3}; 
                static __device__ __host__ float3 all(float v) {return make_float3(v, v, v);}
                static __device__ __host__ float3 make(float x, float y, float z) {return make_float3(x, y, z);}
            };
            template<> struct VecTraits<float4> 
            { 
                typedef float elem_t;
                enum {cn=4}; 
                static __device__ __host__ float4 all(float v) {return make_float4(v, v, v, v);}
                static __device__ __host__ float4 make(float x, float y, float z, float w) {return make_float4(x, y, z, w);}
            };

            template <int cn, typename VecD> struct SatCast;
            template <typename VecD> struct SatCast<1, VecD>
            {
                template <typename VecS>
                static __device__ VecD cast(const VecS& v)
                {
                    typedef typename VecTraits<VecD>::elem_t D;
                    return VecTraits<VecD>::make(saturate_cast<D>(v.x));
                }
            };
            template <typename VecD> struct SatCast<2, VecD>
            {
                template <typename VecS>
                static __device__ VecD cast(const VecS& v)
                {
                    typedef typename VecTraits<VecD>::elem_t D;
                    return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y));
                }
            };
            template <typename VecD> struct SatCast<3, VecD>
            {
                template <typename VecS>
                static __device__ VecD cast(const VecS& v)
                {
                    typedef typename VecTraits<VecD>::elem_t D;
                    return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z));
                }
            };
            template <typename VecD> struct SatCast<4, VecD>
            {
                template <typename VecS>
                static __device__ VecD cast(const VecS& v)
                {
                    typedef typename VecTraits<VecD>::elem_t D;
                    return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z), saturate_cast<D>(v.w));
                }
            };

            template <typename VecD, typename VecS> static __device__ VecD saturate_cast_caller(const VecS& v)
            {
                return SatCast<VecTraits<VecD>::cn, VecD>::cast(v);
            }

            template<typename _Tp> static __device__ _Tp saturate_cast(const uchar1& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const char1& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const ushort1& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const short1& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const uint1& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const int1& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const float1& v) {return saturate_cast_caller<_Tp>(v);}

            template<typename _Tp> static __device__ _Tp saturate_cast(const uchar2& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const char2& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const ushort2& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const short2& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const uint2& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const int2& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const float2& v) {return saturate_cast_caller<_Tp>(v);}

            template<typename _Tp> static __device__ _Tp saturate_cast(const uchar3& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const char3& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const ushort3& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const short3& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const uint3& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const int3& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const float3& v) {return saturate_cast_caller<_Tp>(v);}

            template<typename _Tp> static __device__ _Tp saturate_cast(const uchar4& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const char4& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const ushort4& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const short4& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const uint4& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const int4& v) {return saturate_cast_caller<_Tp>(v);}
            template<typename _Tp> static __device__ _Tp saturate_cast(const float4& v) {return saturate_cast_caller<_Tp>(v);}

            static __device__  uchar1 operator+(const uchar1& a, const uchar1& b)
            {
                return make_uchar1(a.x + b.x);
            }
            static __device__  uchar1 operator-(const uchar1& a, const uchar1& b)
            {
                return make_uchar1(a.x - b.x);
            }
            static __device__  uchar1 operator*(const uchar1& a, const uchar1& b)
            {
                return make_uchar1(a.x * b.x);
            }
            static __device__  uchar1 operator/(const uchar1& a, const uchar1& b)
            {
                return make_uchar1(a.x / b.x);
            }
            static __device__ float1 operator*(const uchar1& a, float s)
            {
                return make_float1(a.x * s);
            }

            static __device__  uchar2 operator+(const uchar2& a, const uchar2& b)
            {
                return make_uchar2(a.x + b.x, a.y + b.y);
            }
            static __device__  uchar2 operator-(const uchar2& a, const uchar2& b)
            {
                return make_uchar2(a.x - b.x, a.y - b.y);
            }
            static __device__  uchar2 operator*(const uchar2& a, const uchar2& b)
            {
                return make_uchar2(a.x * b.x, a.y * b.y);
            }
            static __device__  uchar2 operator/(const uchar2& a, const uchar2& b)
            {
                return make_uchar2(a.x / b.x, a.y / b.y);
            }
            static __device__ float2 operator*(const uchar2& a, float s)
            {
                return make_float2(a.x * s, a.y * s);
            }

            static __device__  uchar3 operator+(const uchar3& a, const uchar3& b)
            {
                return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
            }
            static __device__  uchar3 operator-(const uchar3& a, const uchar3& b)
            {
                return make_uchar3(a.x - b.x, a.y - b.y, a.z - b.z);
            }
            static __device__  uchar3 operator*(const uchar3& a, const uchar3& b)
            {
                return make_uchar3(a.x * b.x, a.y * b.y, a.z * b.z);
            }
            static __device__  uchar3 operator/(const uchar3& a, const uchar3& b)
            {
                return make_uchar3(a.x / b.x, a.y / b.y, a.z / b.z);
            }
            static __device__ float3 operator*(const uchar3& a, float s)
            {
                return make_float3(a.x * s, a.y * s, a.z * s);
            }

            static __device__  uchar4 operator+(const uchar4& a, const uchar4& b)
            {
                return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
            static __device__  uchar4 operator-(const uchar4& a, const uchar4& b)
            {
                return make_uchar4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
            }
            static __device__  uchar4 operator*(const uchar4& a, const uchar4& b)
            {
                return make_uchar4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
            }
            static __device__  uchar4 operator/(const uchar4& a, const uchar4& b)
            {
                return make_uchar4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
            }
            static __device__ float4 operator*(const uchar4& a, float s)
            {
                return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
            }

            static __device__  char1 operator+(const char1& a, const char1& b)
            {
                return make_char1(a.x + b.x);
            }
            static __device__  char1 operator-(const char1& a, const char1& b)
            {
                return make_char1(a.x - b.x);
            }
            static __device__  char1 operator*(const char1& a, const char1& b)
            {
                return make_char1(a.x * b.x);
            }
            static __device__  char1 operator/(const char1& a, const char1& b)
            {
                return make_char1(a.x / b.x);
            }
            static __device__ float1 operator*(const char1& a, float s)
            {
                return make_float1(a.x * s);
            }

            static __device__  char2 operator+(const char2& a, const char2& b)
            {
                return make_char2(a.x + b.x, a.y + b.y);
            }
            static __device__  char2 operator-(const char2& a, const char2& b)
            {
                return make_char2(a.x - b.x, a.y - b.y);
            }
            static __device__  char2 operator*(const char2& a, const char2& b)
            {
                return make_char2(a.x * b.x, a.y * b.y);
            }
            static __device__  char2 operator/(const char2& a, const char2& b)
            {
                return make_char2(a.x / b.x, a.y / b.y);
            }
            static __device__ float2 operator*(const char2& a, float s)
            {
                return make_float2(a.x * s, a.y * s);
            }

            static __device__  char3 operator+(const char3& a, const char3& b)
            {
                return make_char3(a.x + b.x, a.y + b.y, a.z + b.z);
            }
            static __device__  char3 operator-(const char3& a, const char3& b)
            {
                return make_char3(a.x - b.x, a.y - b.y, a.z - b.z);
            }
            static __device__  char3 operator*(const char3& a, const char3& b)
            {
                return make_char3(a.x * b.x, a.y * b.y, a.z * b.z);
            }
            static __device__  char3 operator/(const char3& a, const char3& b)
            {
                return make_char3(a.x / b.x, a.y / b.y, a.z / b.z);
            }
            static __device__ float3 operator*(const char3& a, float s)
            {
                return make_float3(a.x * s, a.y * s, a.z * s);
            }

            static __device__  char4 operator+(const char4& a, const char4& b)
            {
                return make_char4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
            static __device__  char4 operator-(const char4& a, const char4& b)
            {
                return make_char4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
            }
            static __device__  char4 operator*(const char4& a, const char4& b)
            {
                return make_char4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
            }
            static __device__  char4 operator/(const char4& a, const char4& b)
            {
                return make_char4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
            }
            static __device__ float4 operator*(const char4& a, float s)
            {
                return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
            }

            static __device__  ushort1 operator+(const ushort1& a, const ushort1& b)
            {
                return make_ushort1(a.x + b.x);
            }
            static __device__  ushort1 operator-(const ushort1& a, const ushort1& b)
            {
                return make_ushort1(a.x - b.x);
            }
            static __device__  ushort1 operator*(const ushort1& a, const ushort1& b)
            {
                return make_ushort1(a.x * b.x);
            }
            static __device__  ushort1 operator/(const ushort1& a, const ushort1& b)
            {
                return make_ushort1(a.x / b.x);
            }
            static __device__ float1 operator*(const ushort1& a, float s)
            {
                return make_float1(a.x * s);
            }

            static __device__  ushort2 operator+(const ushort2& a, const ushort2& b)
            {
                return make_ushort2(a.x + b.x, a.y + b.y);
            }
            static __device__  ushort2 operator-(const ushort2& a, const ushort2& b)
            {
                return make_ushort2(a.x - b.x, a.y - b.y);
            }
            static __device__  ushort2 operator*(const ushort2& a, const ushort2& b)
            {
                return make_ushort2(a.x * b.x, a.y * b.y);
            }
            static __device__  ushort2 operator/(const ushort2& a, const ushort2& b)
            {
                return make_ushort2(a.x / b.x, a.y / b.y);
            }
            static __device__ float2 operator*(const ushort2& a, float s)
            {
                return make_float2(a.x * s, a.y * s);
            }

            static __device__  ushort3 operator+(const ushort3& a, const ushort3& b)
            {
                return make_ushort3(a.x + b.x, a.y + b.y, a.z + b.z);
            }
            static __device__  ushort3 operator-(const ushort3& a, const ushort3& b)
            {
                return make_ushort3(a.x - b.x, a.y - b.y, a.z - b.z);
            }
            static __device__  ushort3 operator*(const ushort3& a, const ushort3& b)
            {
                return make_ushort3(a.x * b.x, a.y * b.y, a.z * b.z);
            }
            static __device__  ushort3 operator/(const ushort3& a, const ushort3& b)
            {
                return make_ushort3(a.x / b.x, a.y / b.y, a.z / b.z);
            }
            static __device__ float3 operator*(const ushort3& a, float s)
            {
                return make_float3(a.x * s, a.y * s, a.z * s);
            }

            static __device__  ushort4 operator+(const ushort4& a, const ushort4& b)
            {
                return make_ushort4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
            static __device__  ushort4 operator-(const ushort4& a, const ushort4& b)
            {
                return make_ushort4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
            }
            static __device__  ushort4 operator*(const ushort4& a, const ushort4& b)
            {
                return make_ushort4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
            }
            static __device__  ushort4 operator/(const ushort4& a, const ushort4& b)
            {
                return make_ushort4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
            }
            static __device__ float4 operator*(const ushort4& a, float s)
            {
                return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
            }

            static __device__  short1 operator+(const short1& a, const short1& b)
            {
                return make_short1(a.x + b.x);
            }
            static __device__  short1 operator-(const short1& a, const short1& b)
            {
                return make_short1(a.x - b.x);
            }
            static __device__  short1 operator*(const short1& a, const short1& b)
            {
                return make_short1(a.x * b.x);
            }
            static __device__  short1 operator/(const short1& a, const short1& b)
            {
                return make_short1(a.x / b.x);
            }
            static __device__ float1 operator*(const short1& a, float s)
            {
                return make_float1(a.x * s);
            }

            static __device__  short2 operator+(const short2& a, const short2& b)
            {
                return make_short2(a.x + b.x, a.y + b.y);
            }
            static __device__  short2 operator-(const short2& a, const short2& b)
            {
                return make_short2(a.x - b.x, a.y - b.y);
            }
            static __device__  short2 operator*(const short2& a, const short2& b)
            {
                return make_short2(a.x * b.x, a.y * b.y);
            }
            static __device__  short2 operator/(const short2& a, const short2& b)
            {
                return make_short2(a.x / b.x, a.y / b.y);
            }
            static __device__ float2 operator*(const short2& a, float s)
            {
                return make_float2(a.x * s, a.y * s);
            }

            static __device__  short3 operator+(const short3& a, const short3& b)
            {
                return make_short3(a.x + b.x, a.y + b.y, a.z + b.z);
            }
            static __device__  short3 operator-(const short3& a, const short3& b)
            {
                return make_short3(a.x - b.x, a.y - b.y, a.z - b.z);
            }
            static __device__  short3 operator*(const short3& a, const short3& b)
            {
                return make_short3(a.x * b.x, a.y * b.y, a.z * b.z);
            }
            static __device__  short3 operator/(const short3& a, const short3& b)
            {
                return make_short3(a.x / b.x, a.y / b.y, a.z / b.z);
            }
            static __device__ float3 operator*(const short3& a, float s)
            {
                return make_float3(a.x * s, a.y * s, a.z * s);
            }

            static __device__  short4 operator+(const short4& a, const short4& b)
            {
                return make_short4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
            static __device__  short4 operator-(const short4& a, const short4& b)
            {
                return make_short4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
            }
            static __device__  short4 operator*(const short4& a, const short4& b)
            {
                return make_short4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
            }
            static __device__  short4 operator/(const short4& a, const short4& b)
            {
                return make_short4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
            }
            static __device__ float4 operator*(const short4& a, float s)
            {
                return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
            }

            static __device__  int1 operator+(const int1& a, const int1& b)
            {
                return make_int1(a.x + b.x);
            }
            static __device__  int1 operator-(const int1& a, const int1& b)
            {
                return make_int1(a.x - b.x);
            }
            static __device__  int1 operator*(const int1& a, const int1& b)
            {
                return make_int1(a.x * b.x);
            }
            static __device__  int1 operator/(const int1& a, const int1& b)
            {
                return make_int1(a.x / b.x);
            }
            static __device__ float1 operator*(const int1& a, float s)
            {
                return make_float1(a.x * s);
            }

            static __device__  int2 operator+(const int2& a, const int2& b)
            {
                return make_int2(a.x + b.x, a.y + b.y);
            }
            static __device__  int2 operator-(const int2& a, const int2& b)
            {
                return make_int2(a.x - b.x, a.y - b.y);
            }
            static __device__  int2 operator*(const int2& a, const int2& b)
            {
                return make_int2(a.x * b.x, a.y * b.y);
            }
            static __device__  int2 operator/(const int2& a, const int2& b)
            {
                return make_int2(a.x / b.x, a.y / b.y);
            }
            static __device__ float2 operator*(const int2& a, float s)
            {
                return make_float2(a.x * s, a.y * s);
            }

            static __device__  int3 operator+(const int3& a, const int3& b)
            {
                return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
            }
            static __device__  int3 operator-(const int3& a, const int3& b)
            {
                return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
            }
            static __device__  int3 operator*(const int3& a, const int3& b)
            {
                return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
            }
            static __device__  int3 operator/(const int3& a, const int3& b)
            {
                return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
            }
            static __device__ float3 operator*(const int3& a, float s)
            {
                return make_float3(a.x * s, a.y * s, a.z * s);
            }

            static __device__  int4 operator+(const int4& a, const int4& b)
            {
                return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
            static __device__  int4 operator-(const int4& a, const int4& b)
            {
                return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
            }
            static __device__  int4 operator*(const int4& a, const int4& b)
            {
                return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
            }
            static __device__  int4 operator/(const int4& a, const int4& b)
            {
                return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
            }
            static __device__ float4 operator*(const int4& a, float s)
            {
                return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
            }

            static __device__  float1 operator+(const float1& a, const float1& b)
            {
                return make_float1(a.x + b.x);
            }
            static __device__  float1 operator-(const float1& a, const float1& b)
            {
                return make_float1(a.x - b.x);
            }
            static __device__  float1 operator*(const float1& a, const float1& b)
            {
                return make_float1(a.x * b.x);
            }
            static __device__  float1 operator/(const float1& a, const float1& b)
            {
                return make_float1(a.x / b.x);
            }
            static __device__ float1 operator*(const float1& a, float s)
            {
                return make_float1(a.x * s);
            }

            static __device__  float2 operator+(const float2& a, const float2& b)
            {
                return make_float2(a.x + b.x, a.y + b.y);
            }
            static __device__  float2 operator-(const float2& a, const float2& b)
            {
                return make_float2(a.x - b.x, a.y - b.y);
            }
            static __device__  float2 operator*(const float2& a, const float2& b)
            {
                return make_float2(a.x * b.x, a.y * b.y);
            }
            static __device__  float2 operator/(const float2& a, const float2& b)
            {
                return make_float2(a.x / b.x, a.y / b.y);
            }
            static __device__ float2 operator*(const float2& a, float s)
            {
                return make_float2(a.x * s, a.y * s);
            }

            static __device__  float3 operator+(const float3& a, const float3& b)
            {
                return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
            }
            static __device__  float3 operator-(const float3& a, const float3& b)
            {
                return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
            }
            static __device__  float3 operator*(const float3& a, const float3& b)
            {
                return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
            }
            static __device__  float3 operator/(const float3& a, const float3& b)
            {
                return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
            }
            static __device__ float3 operator*(const float3& a, float s)
            {
                return make_float3(a.x * s, a.y * s, a.z * s);
            }

            static __device__  float4 operator+(const float4& a, const float4& b)
            {
                return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
            static __device__  float4 operator-(const float4& a, const float4& b)
            {
                return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
            }
            static __device__  float4 operator*(const float4& a, const float4& b)
            {
                return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
            }
            static __device__  float4 operator/(const float4& a, const float4& b)
            {
                return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
            }
            static __device__ float4 operator*(const float4& a, float s)
            {
                return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
            }
        }            
    }
}

#endif // __OPENCV_GPU_VECMATH_HPP__