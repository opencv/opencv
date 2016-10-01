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

#ifndef OPENCV_CORE_TRAITS_HPP
#define OPENCV_CORE_TRAITS_HPP

#include "opencv2/core/cvdef.h"

namespace cv
{

//! @addtogroup core_basic
//! @{

/** @brief Template "trait" class for OpenCV primitive data types.

A primitive OpenCV data type is one of unsigned char, bool, signed char, unsigned short, signed
short, int, float, double, or a tuple of values of one of these types, where all the values in the
tuple have the same type. Any primitive type from the list can be defined by an identifier in the
form CV_\<bit-depth\>{U|S|F}C(\<number_of_channels\>), for example: uchar \~ CV_8UC1, 3-element
floating-point tuple \~ CV_32FC3, and so on. A universal OpenCV structure that is able to store a
single instance of such a primitive data type is Vec. Multiple instances of such a type can be
stored in a std::vector, Mat, Mat_, SparseMat, SparseMat_, or any other container that is able to
store Vec instances.

The DataType class is basically used to provide a description of such primitive data types without
adding any fields or methods to the corresponding classes (and it is actually impossible to add
anything to primitive C/C++ data types). This technique is known in C++ as class traits. It is not
DataType itself that is used but its specialized versions, such as:
@code
    template<> class DataType<uchar>
    {
        typedef uchar value_type;
        typedef int work_type;
        typedef uchar channel_type;
        enum { channel_type = CV_8U, channels = 1, fmt='u', type = CV_8U };
    };
    ...
    template<typename _Tp> DataType<std::complex<_Tp> >
    {
        typedef std::complex<_Tp> value_type;
        typedef std::complex<_Tp> work_type;
        typedef _Tp channel_type;
        // DataDepth is another helper trait class
        enum { depth = DataDepth<_Tp>::value, channels=2,
            fmt=(channels-1)*256+DataDepth<_Tp>::fmt,
            type=CV_MAKETYPE(depth, channels) };
    };
    ...
@endcode
The main purpose of this class is to convert compilation-time type information to an
OpenCV-compatible data type identifier, for example:
@code
    // allocates a 30x40 floating-point matrix
    Mat A(30, 40, DataType<float>::type);

    Mat B = Mat_<std::complex<double> >(3, 3);
    // the statement below will print 6, 2 , that is depth == CV_64F, channels == 2
    cout << B.depth() << ", " << B.channels() << endl;
@endcode
So, such traits are used to tell OpenCV which data type you are working with, even if such a type is
not native to OpenCV. For example, the matrix B initialization above is compiled because OpenCV
defines the proper specialized template class DataType\<complex\<_Tp\> \> . This mechanism is also
useful (and used in OpenCV this way) for generic algorithms implementations.
*/
template<typename _Tp> class DataType
{
public:
    typedef _Tp         value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 1,
           depth        = -1,
           channels     = 1,
           fmt          = 0,
           type = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<bool>
{
public:
    typedef bool        value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8U,
           channels     = 1,
           fmt          = (int)'u',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<uchar>
{
public:
    typedef uchar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8U,
           channels     = 1,
           fmt          = (int)'u',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<schar>
{
public:
    typedef schar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8S,
           channels     = 1,
           fmt          = (int)'c',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<char>
{
public:
    typedef schar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8S,
           channels     = 1,
           fmt          = (int)'c',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<ushort>
{
public:
    typedef ushort      value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_16U,
           channels     = 1,
           fmt          = (int)'w',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<short>
{
public:
    typedef short       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_16S,
           channels     = 1,
           fmt          = (int)'s',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<int>
{
public:
    typedef int         value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32S,
           channels     = 1,
           fmt          = (int)'i',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<float>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 1,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<double>
{
public:
    typedef double      value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_64F,
           channels     = 1,
           fmt          = (int)'d',
           type         = CV_MAKETYPE(depth, channels)
         };
};


/** @brief A helper class for cv::DataType

The class is specialized for each fundamental numerical data type supported by OpenCV. It provides
DataDepth<T>::value constant.
*/
template<typename _Tp> class DataDepth
{
public:
    enum
    {
        value = DataType<_Tp>::depth,
        fmt   = DataType<_Tp>::fmt
    };
};



template<int _depth> class TypeDepth
{
    enum { depth = CV_USRTYPE1 };
    typedef void value_type;
};

template<> class TypeDepth<CV_8U>
{
    enum { depth = CV_8U };
    typedef uchar value_type;
};

template<> class TypeDepth<CV_8S>
{
    enum { depth = CV_8S };
    typedef schar value_type;
};

template<> class TypeDepth<CV_16U>
{
    enum { depth = CV_16U };
    typedef ushort value_type;
};

template<> class TypeDepth<CV_16S>
{
    enum { depth = CV_16S };
    typedef short value_type;
};

template<> class TypeDepth<CV_32S>
{
    enum { depth = CV_32S };
    typedef int value_type;
};

template<> class TypeDepth<CV_32F>
{
    enum { depth = CV_32F };
    typedef float value_type;
};

template<> class TypeDepth<CV_64F>
{
    enum { depth = CV_64F };
    typedef double value_type;
};

//! @}

} // cv

#endif // OPENCV_CORE_TRAITS_HPP
