/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#ifndef __OPENCV_CUDEV_UTIL_VEC_TRAITS_HPP__
#define __OPENCV_CUDEV_UTIL_VEC_TRAITS_HPP__

#include "../common.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// MakeVec

template<typename T, int CN> struct MakeVec;

#define CV_CUDEV_MAKE_VEC_INST(elem_type) \
    template<> struct MakeVec<elem_type, 1> { typedef elem_type      type; }; \
    template<> struct MakeVec<elem_type, 2> { typedef elem_type ## 2 type; }; \
    template<> struct MakeVec<elem_type, 3> { typedef elem_type ## 3 type; }; \
    template<> struct MakeVec<elem_type, 4> { typedef elem_type ## 4 type; };

CV_CUDEV_MAKE_VEC_INST(uchar)
CV_CUDEV_MAKE_VEC_INST(ushort)
CV_CUDEV_MAKE_VEC_INST(short)
CV_CUDEV_MAKE_VEC_INST(int)
CV_CUDEV_MAKE_VEC_INST(uint)
CV_CUDEV_MAKE_VEC_INST(float)
CV_CUDEV_MAKE_VEC_INST(double)

#undef CV_CUDEV_MAKE_VEC_INST

template<> struct MakeVec<schar, 1> { typedef schar type; };
template<> struct MakeVec<schar, 2> { typedef char2 type; };
template<> struct MakeVec<schar, 3> { typedef char3 type; };
template<> struct MakeVec<schar, 4> { typedef char4 type; };

template<> struct MakeVec<bool, 1> { typedef uchar  type; };
template<> struct MakeVec<bool, 2> { typedef uchar2 type; };
template<> struct MakeVec<bool, 3> { typedef uchar3 type; };
template<> struct MakeVec<bool, 4> { typedef uchar4 type; };

// VecTraits

template<typename T> struct VecTraits;

#define CV_CUDEV_VEC_TRAITS_INST(type) \
    template <> struct VecTraits<type> \
    { \
        typedef type elem_type; \
        enum {cn=1}; \
        __host__ __device__ __forceinline__ static type all(type v) {return v;} \
        __host__ __device__ __forceinline__ static type make(type x) {return x;} \
        __host__ __device__ __forceinline__ static type make(const type* v) {return *v;} \
    }; \
    template <> struct VecTraits<type ## 1> \
    { \
        typedef type elem_type; \
        enum {cn=1}; \
        __host__ __device__ __forceinline__ static type ## 1 all(type v) {return make_ ## type ## 1(v);} \
        __host__ __device__ __forceinline__ static type ## 1 make(type x) {return make_ ## type ## 1(x);} \
        __host__ __device__ __forceinline__ static type ## 1 make(const type* v) {return make_ ## type ## 1(*v);} \
    }; \
    template <> struct VecTraits<type ## 2> \
    { \
        typedef type elem_type; \
        enum {cn=2}; \
        __host__ __device__ __forceinline__ static type ## 2 all(type v) {return make_ ## type ## 2(v, v);} \
        __host__ __device__ __forceinline__ static type ## 2 make(type x, type y) {return make_ ## type ## 2(x, y);} \
        __host__ __device__ __forceinline__ static type ## 2 make(const type* v) {return make_ ## type ## 2(v[0], v[1]);} \
    }; \
    template <> struct VecTraits<type ## 3> \
    { \
        typedef type elem_type; \
        enum {cn=3}; \
        __host__ __device__ __forceinline__ static type ## 3 all(type v) {return make_ ## type ## 3(v, v, v);} \
        __host__ __device__ __forceinline__ static type ## 3 make(type x, type y, type z) {return make_ ## type ## 3(x, y, z);} \
        __host__ __device__ __forceinline__ static type ## 3 make(const type* v) {return make_ ## type ## 3(v[0], v[1], v[2]);} \
    }; \
    template <> struct VecTraits<type ## 4> \
    { \
        typedef type elem_type; \
        enum {cn=4}; \
        __host__ __device__ __forceinline__ static type ## 4 all(type v) {return make_ ## type ## 4(v, v, v, v);} \
        __host__ __device__ __forceinline__ static type ## 4 make(type x, type y, type z, type w) {return make_ ## type ## 4(x, y, z, w);} \
        __host__ __device__ __forceinline__ static type ## 4 make(const type* v) {return make_ ## type ## 4(v[0], v[1], v[2], v[3]);} \
    };

CV_CUDEV_VEC_TRAITS_INST(uchar)
CV_CUDEV_VEC_TRAITS_INST(ushort)
CV_CUDEV_VEC_TRAITS_INST(short)
CV_CUDEV_VEC_TRAITS_INST(int)
CV_CUDEV_VEC_TRAITS_INST(uint)
CV_CUDEV_VEC_TRAITS_INST(float)
CV_CUDEV_VEC_TRAITS_INST(double)

#undef CV_CUDEV_VEC_TRAITS_INST

template<> struct VecTraits<schar>
{
    typedef schar elem_type;
    enum {cn=1};
    __host__ __device__ __forceinline__ static schar all(schar v) {return v;}
    __host__ __device__ __forceinline__ static schar make(schar x) {return x;}
    __host__ __device__ __forceinline__ static schar make(const schar* x) {return *x;}
};
template<> struct VecTraits<char1>
{
    typedef schar elem_type;
    enum {cn=1};
    __host__ __device__ __forceinline__ static char1 all(schar v) {return make_char1(v);}
    __host__ __device__ __forceinline__ static char1 make(schar x) {return make_char1(x);}
    __host__ __device__ __forceinline__ static char1 make(const schar* v) {return make_char1(v[0]);}
};
template<> struct VecTraits<char2>
{
    typedef schar elem_type;
    enum {cn=2};
    __host__ __device__ __forceinline__ static char2 all(schar v) {return make_char2(v, v);}
    __host__ __device__ __forceinline__ static char2 make(schar x, schar y) {return make_char2(x, y);}
    __host__ __device__ __forceinline__ static char2 make(const schar* v) {return make_char2(v[0], v[1]);}
};
template<> struct VecTraits<char3>
{
    typedef schar elem_type;
    enum {cn=3};
    __host__ __device__ __forceinline__ static char3 all(schar v) {return make_char3(v, v, v);}
    __host__ __device__ __forceinline__ static char3 make(schar x, schar y, schar z) {return make_char3(x, y, z);}
    __host__ __device__ __forceinline__ static char3 make(const schar* v) {return make_char3(v[0], v[1], v[2]);}
};
template<> struct VecTraits<char4>
{
    typedef schar elem_type;
    enum {cn=4};
    __host__ __device__ __forceinline__ static char4 all(schar v) {return make_char4(v, v, v, v);}
    __host__ __device__ __forceinline__ static char4 make(schar x, schar y, schar z, schar w) {return make_char4(x, y, z, w);}
    __host__ __device__ __forceinline__ static char4 make(const schar* v) {return make_char4(v[0], v[1], v[2], v[3]);}
};

//! @}

}}

// DataType

namespace cv {

template <> class DataType<uint>
{
public:
    typedef uint         value_type;
    typedef value_type   work_type;
    typedef value_type   channel_type;
    typedef value_type   vec_type;
    enum { generic_type = 0,
           depth        = CV_32S,
           channels     = 1,
           fmt          = (int)'i',
           type         = CV_MAKE_TYPE(depth, channels)
         };
};

#define CV_CUDEV_DATA_TYPE_INST(_depth_type, _channel_num) \
    template <> class DataType< _depth_type ## _channel_num > \
    { \
    public: \
        typedef _depth_type ## _channel_num     value_type; \
        typedef value_type                      work_type; \
        typedef _depth_type                     channel_type; \
        typedef value_type                      vec_type; \
        enum { generic_type = 0, \
               depth        = DataType<channel_type>::depth, \
               channels     = _channel_num, \
               fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8), \
               type         = CV_MAKE_TYPE(depth, channels) \
             }; \
    };

CV_CUDEV_DATA_TYPE_INST(uchar, 1)
CV_CUDEV_DATA_TYPE_INST(uchar, 2)
CV_CUDEV_DATA_TYPE_INST(uchar, 3)
CV_CUDEV_DATA_TYPE_INST(uchar, 4)

CV_CUDEV_DATA_TYPE_INST(ushort, 1)
CV_CUDEV_DATA_TYPE_INST(ushort, 2)
CV_CUDEV_DATA_TYPE_INST(ushort, 3)
CV_CUDEV_DATA_TYPE_INST(ushort, 4)

CV_CUDEV_DATA_TYPE_INST(short, 1)
CV_CUDEV_DATA_TYPE_INST(short, 2)
CV_CUDEV_DATA_TYPE_INST(short, 3)
CV_CUDEV_DATA_TYPE_INST(short, 4)

CV_CUDEV_DATA_TYPE_INST(int, 1)
CV_CUDEV_DATA_TYPE_INST(int, 2)
CV_CUDEV_DATA_TYPE_INST(int, 3)
CV_CUDEV_DATA_TYPE_INST(int, 4)

CV_CUDEV_DATA_TYPE_INST(uint, 1)
CV_CUDEV_DATA_TYPE_INST(uint, 2)
CV_CUDEV_DATA_TYPE_INST(uint, 3)
CV_CUDEV_DATA_TYPE_INST(uint, 4)

CV_CUDEV_DATA_TYPE_INST(float, 1)
CV_CUDEV_DATA_TYPE_INST(float, 2)
CV_CUDEV_DATA_TYPE_INST(float, 3)
CV_CUDEV_DATA_TYPE_INST(float, 4)

CV_CUDEV_DATA_TYPE_INST(double, 1)
CV_CUDEV_DATA_TYPE_INST(double, 2)
CV_CUDEV_DATA_TYPE_INST(double, 3)
CV_CUDEV_DATA_TYPE_INST(double, 4)

#undef CV_CUDEV_DATA_TYPE_INST

template<> class DataType<char1>
{
public:
    typedef char1      value_type;
    typedef value_type work_type;
    typedef schar      channel_type;
    typedef value_type vec_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 1,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKE_TYPE(depth, channels)
         };
};

template<> class DataType<char2>
{
public:
    typedef char2      value_type;
    typedef value_type work_type;
    typedef schar      channel_type;
    typedef value_type vec_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 2,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKE_TYPE(depth, channels)
         };
};

template<> class DataType<char3>
{
public:
    typedef char3      value_type;
    typedef value_type work_type;
    typedef schar      channel_type;
    typedef value_type vec_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 3,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKE_TYPE(depth, channels)
         };
};

template<> class DataType<char4>
{
public:
    typedef char4      value_type;
    typedef value_type work_type;
    typedef schar      channel_type;
    typedef value_type vec_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 4,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKE_TYPE(depth, channels)
         };
};

}

#endif
