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
//     and/or other GpuMaterials provided with the distribution.
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

#ifndef __OPENCV_GPU_MATRIX_OPERATIONS_HPP__
#define __OPENCV_GPU_MATRIX_OPERATIONS_HPP__

namespace cv
{

namespace gpu
{
///////////////////////////////////////////////////////////////////////
//////////////////////////////// CudaMem ////////////////////////////////
///////////////////////////////////////////////////////////////////////

inline CudaMem::CudaMem()  : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0) {}
inline CudaMem::CudaMem(int _rows, int _cols, int _type, int _alloc_type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0)
{
    if( _rows > 0 && _cols > 0 )
        create( _rows, _cols, _type, _alloc_type);
}

inline CudaMem::CudaMem(Size _size, int _type, int _alloc_type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0)
{
    if( _size.height > 0 && _size.width > 0 )
        create( _size.height, _size.width, _type, _alloc_type);
}

inline CudaMem::CudaMem(const CudaMem& m) : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), alloc_type(m.alloc_type)
{
    if( refcount )
        CV_XADD(refcount, 1);
}

inline CudaMem::CudaMem(const Mat& m, int _alloc_type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0)
{
    if( m.rows > 0 && m.cols > 0 )
        create( m.size(), m.type(), _alloc_type);

    Mat tmp = createMatHeader();
    m.copyTo(tmp);
}

inline CudaMem::~CudaMem()
{
    release();

}

inline CudaMem& CudaMem::operator = (const CudaMem& m)
{
    if( this != &m )
    {
        if( m.refcount )
            CV_XADD(m.refcount, 1);
        release();
        flags = m.flags;
        rows = m.rows; cols = m.cols;
        step = m.step; data = m.data;
        datastart = m.datastart;
        dataend = m.dataend;
        refcount = m.refcount;
        alloc_type = m.alloc_type;
    }
    return *this;
}

inline CudaMem CudaMem::clone() const
{
    CudaMem m(size(), type(), alloc_type);
    Mat to = m;
    Mat from = *this;
    from.copyTo(to);
    return m;
}

inline void CudaMem::create(Size _size, int _type, int _alloc_type) { create(_size.height, _size.width, _type, _alloc_type); }


//CCP void CudaMem::create(int _rows, int _cols, int _type, int _alloc_type);
//CPP void CudaMem::release();

inline Mat CudaMem::createMatHeader() const { return Mat(size(), type(), data); }
inline CudaMem::operator Mat() const { return createMatHeader(); }

inline CudaMem::operator GpuMat() const { return createGpuMatHeader(); }
//CPP GpuMat CudaMem::createGpuMatHeader() const;

inline bool CudaMem::isContinuous() const { return (flags & Mat::CONTINUOUS_FLAG) != 0; }
inline size_t CudaMem::elemSize() const { return CV_ELEM_SIZE(flags); }
inline size_t CudaMem::elemSize1() const { return CV_ELEM_SIZE1(flags); }
inline int CudaMem::type() const { return CV_MAT_TYPE(flags); }
inline int CudaMem::depth() const { return CV_MAT_DEPTH(flags); }
inline int CudaMem::channels() const { return CV_MAT_CN(flags); }
inline size_t CudaMem::step1() const { return step/elemSize1(); }
inline Size CudaMem::size() const { return Size(cols, rows); }
inline bool CudaMem::empty() const { return data == 0; }

} /* end of namespace gpu */

} /* end of namespace cv */

#endif /* __OPENCV_GPU_MATRIX_OPERATIONS_HPP__ */
