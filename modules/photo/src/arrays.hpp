/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "opencv2/core/base.hpp"

#ifndef __OPENCV_DENOISING_ARRAYS_HPP__
#define __OPENCV_DENOISING_ARRAYS_HPP__

template <class T>
struct Array2d
{
    T* a;
    int n1,n2;
    bool needToDeallocArray;

    Array2d(const Array2d& array2d):
        a(array2d.a), n1(array2d.n1), n2(array2d.n2), needToDeallocArray(false)
    {
        if (array2d.needToDeallocArray)
        {
            CV_Error(Error::BadDataPtr, "Copy constructor for self allocating arrays not supported");
        }
    }

    Array2d(T* _a, int _n1, int _n2):
        a(_a), n1(_n1), n2(_n2), needToDeallocArray(false)
    {
    }

    Array2d(int _n1, int _n2):
        n1(_n1), n2(_n2), needToDeallocArray(true)
    {
        a = new T[n1*n2];
    }

    ~Array2d()
    {
        if (needToDeallocArray)
            delete[] a;
    }

    T* operator [] (int i)
    {
        return a + i*n2;
    }

    inline T* row_ptr(int i)
    {
        return (*this)[i];
    }
};

template <class T>
struct Array3d
{
    T* a;
    int n1,n2,n3;
    bool needToDeallocArray;

    Array3d(T* _a, int _n1, int _n2, int _n3):
        a(_a), n1(_n1), n2(_n2), n3(_n3), needToDeallocArray(false)
    {
    }

    Array3d(int _n1, int _n2, int _n3):
        n1(_n1), n2(_n2), n3(_n3), needToDeallocArray(true)
    {
        a = new T[n1*n2*n3];
    }

    ~Array3d()
    {
        if (needToDeallocArray)
            delete[] a;
    }

    Array2d<T> operator [] (int i)
    {
        Array2d<T> array2d(a + i*n2*n3, n2, n3);
        return array2d;
    }

    inline T* row_ptr(int i1, int i2)
    {
        return a + i1*n2*n3 + i2*n3;
    }
};

template <class T>
struct Array4d
{
    T* a;
    int n1,n2,n3,n4;
    bool needToDeallocArray;
    int steps[4];

    void init_steps()
    {
        steps[0] = n2*n3*n4;
        steps[1] = n3*n4;
        steps[2] = n4;
        steps[3] = 1;
    }

    Array4d(T* _a, int _n1, int _n2, int _n3, int _n4) :
        a(_a), n1(_n1), n2(_n2), n3(_n3), n4(_n4), needToDeallocArray(false)
    {
        init_steps();
    }

    Array4d(int _n1, int _n2, int _n3, int _n4) :
        n1(_n1), n2(_n2), n3(_n3), n4(_n4), needToDeallocArray(true)
    {
        a = new T[n1*n2*n3*n4];
        init_steps();
    }

    ~Array4d()
    {
        if (needToDeallocArray)
            delete[] a;
    }

    Array3d<T> operator [] (int i)
    {
        Array3d<T> array3d(a + i*n2*n3*n4, n2, n3, n4);
        return array3d;
    }

    inline T* row_ptr(int i1, int i2, int i3)
    {
        return a + i1*n2*n3*n4 + i2*n3*n4 + i3*n4;
    }

    inline int step_size(int dimension)
    {
        return steps[dimension];
    }
};

#endif
