/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef OPENCV_FLANN_DATASET_H_
#define OPENCV_FLANN_DATASET_H_

#include <stdio.h>

#include "general.h"

namespace cvflann
{

/**
 * Class that implements a simple rectangular matrix stored in a memory buffer and
 * provides convenient matrix-like access using the [] operators.
 */
template <typename T>
class Matrix
{
public:
    typedef T type;

    size_t rows;
    size_t cols;
    size_t stride;
    T* data;

    Matrix() : rows(0), cols(0), stride(0), data(NULL)
    {
    }

    Matrix(T* data_, size_t rows_, size_t cols_, size_t stride_ = 0) :
        rows(rows_), cols(cols_),  stride(stride_), data(data_)
    {
        if (stride==0) stride = cols;
    }

    /**
     * Convenience function for deallocating the storage data.
     */
    FLANN_DEPRECATED void free()
    {
        fprintf(stderr, "The cvflann::Matrix<T>::free() method is deprecated "
                "and it does not do any memory deallocation any more.  You are"
                "responsible for deallocating the matrix memory (by doing"
                "'delete[] matrix.data' for example)");
    }

    /**
     * Operator that return a (pointer to a) row of the data.
     */
    T* operator[](size_t index) const
    {
        return data+index*stride;
    }
};


class UntypedMatrix
{
public:
    size_t rows;
    size_t cols;
    void* data;
    flann_datatype_t type;

    UntypedMatrix(void* data_, long rows_, long cols_) :
        rows(rows_), cols(cols_), data(data_)
    {
    }

    ~UntypedMatrix()
    {
    }


    template<typename T>
    Matrix<T> as()
    {
        return Matrix<T>((T*)data, rows, cols);
    }
};



}

#endif //OPENCV_FLANN_DATASET_H_
