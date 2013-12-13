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

#include "precomp.hpp"
#include "opencl_kernels.hpp"

///////////////////////////////// UMat implementation ///////////////////////////////

namespace cv {

// it should be a prime number for the best hash function
enum { UMAT_NLOCKS = 31 };
static Mutex umatLocks[UMAT_NLOCKS];

UMatData::UMatData(const MatAllocator* allocator)
{
    prevAllocator = currAllocator = allocator;
    urefcount = refcount = 0;
    data = origdata = 0;
    size = 0;
    flags = 0;
    handle = 0;
    userdata = 0;
}

UMatData::~UMatData()
{
    prevAllocator = currAllocator = 0;
    urefcount = refcount = 0;
    data = origdata = 0;
    size = 0;
    flags = 0;
    handle = 0;
    userdata = 0;
}

void UMatData::lock()
{
    umatLocks[(size_t)(void*)this % UMAT_NLOCKS].lock();
}

void UMatData::unlock()
{
    umatLocks[(size_t)(void*)this % UMAT_NLOCKS].unlock();
}


MatAllocator* UMat::getStdAllocator()
{
    if( ocl::haveOpenCL() )
        return ocl::getOpenCLAllocator();
    return Mat::getStdAllocator();
}

void swap( UMat& a, UMat& b )
{
    std::swap(a.flags, b.flags);
    std::swap(a.dims, b.dims);
    std::swap(a.rows, b.rows);
    std::swap(a.cols, b.cols);
    std::swap(a.allocator, b.allocator);
    std::swap(a.u, b.u);
    std::swap(a.offset, b.offset);

    std::swap(a.size.p, b.size.p);
    std::swap(a.step.p, b.step.p);
    std::swap(a.step.buf[0], b.step.buf[0]);
    std::swap(a.step.buf[1], b.step.buf[1]);

    if( a.step.p == b.step.buf )
    {
        a.step.p = a.step.buf;
        a.size.p = &a.rows;
    }

    if( b.step.p == a.step.buf )
    {
        b.step.p = b.step.buf;
        b.size.p = &b.rows;
    }
}


static inline void setSize( UMat& m, int _dims, const int* _sz,
                            const size_t* _steps, bool autoSteps=false )
{
    CV_Assert( 0 <= _dims && _dims <= CV_MAX_DIM );
    if( m.dims != _dims )
    {
        if( m.step.p != m.step.buf )
        {
            fastFree(m.step.p);
            m.step.p = m.step.buf;
            m.size.p = &m.rows;
        }
        if( _dims > 2 )
        {
            m.step.p = (size_t*)fastMalloc(_dims*sizeof(m.step.p[0]) + (_dims+1)*sizeof(m.size.p[0]));
            m.size.p = (int*)(m.step.p + _dims) + 1;
            m.size.p[-1] = _dims;
            m.rows = m.cols = -1;
        }
    }

    m.dims = _dims;
    if( !_sz )
        return;

    size_t esz = CV_ELEM_SIZE(m.flags), total = esz;
    int i;
    for( i = _dims-1; i >= 0; i-- )
    {
        int s = _sz[i];
        CV_Assert( s >= 0 );
        m.size.p[i] = s;

        if( _steps )
            m.step.p[i] = i < _dims-1 ? _steps[i] : esz;
        else if( autoSteps )
        {
            m.step.p[i] = total;
            int64 total1 = (int64)total*s;
            if( (uint64)total1 != (size_t)total1 )
                CV_Error( CV_StsOutOfRange, "The total matrix size does not fit to \"size_t\" type" );
            total = (size_t)total1;
        }
    }

    if( _dims == 1 )
    {
        m.dims = 2;
        m.cols = 1;
        m.step[1] = esz;
    }
}

static void updateContinuityFlag(UMat& m)
{
    int i, j;
    for( i = 0; i < m.dims; i++ )
    {
        if( m.size[i] > 1 )
            break;
    }

    for( j = m.dims-1; j > i; j-- )
    {
        if( m.step[j]*m.size[j] < m.step[j-1] )
            break;
    }

    uint64 total = (uint64)m.step[0]*m.size[0];
    if( j <= i && total == (size_t)total )
        m.flags |= UMat::CONTINUOUS_FLAG;
    else
        m.flags &= ~UMat::CONTINUOUS_FLAG;
}


static void finalizeHdr(UMat& m)
{
    updateContinuityFlag(m);
    int d = m.dims;
    if( d > 2 )
        m.rows = m.cols = -1;
}


UMat Mat::getUMat(int accessFlags) const
{
    UMat hdr;
    if(!data)
        return hdr;
    UMatData* temp_u = u;
    if(!temp_u)
    {
        MatAllocator *a = allocator, *a0 = getStdAllocator();
        if(!a)
            a = a0;
        temp_u = a->allocate(dims, size.p, type(), data, step.p, accessFlags);
    }
    UMat::getStdAllocator()->allocate(temp_u, accessFlags);
    hdr.flags = flags;
    setSize(hdr, dims, size.p, step.p);
    finalizeHdr(hdr);
    hdr.u = temp_u;
    hdr.offset = data - datastart;
    return hdr;
}

void UMat::create(int d, const int* _sizes, int _type)
{
    int i;
    CV_Assert(0 <= d && d <= CV_MAX_DIM && _sizes);
    _type = CV_MAT_TYPE(_type);

    if( u && (d == dims || (d == 1 && dims <= 2)) && _type == type() )
    {
        if( d == 2 && rows == _sizes[0] && cols == _sizes[1] )
            return;
        for( i = 0; i < d; i++ )
            if( size[i] != _sizes[i] )
                break;
        if( i == d && (d > 1 || size[1] == 1))
            return;
    }

    release();
    if( d == 0 )
        return;
    flags = (_type & CV_MAT_TYPE_MASK) | MAGIC_VAL;
    setSize(*this, d, _sizes, 0, true);
    offset = 0;

    if( total() > 0 )
    {
        MatAllocator *a = allocator, *a0 = getStdAllocator();
        if(!a)
            a = a0;
        try
        {
            u = a->allocate(dims, size, _type, 0, step.p, 0);
            CV_Assert(u != 0);
        }
        catch(...)
        {
            if(a != a0)
                u = a0->allocate(dims, size, _type, 0, step.p, 0);
            CV_Assert(u != 0);
        }
        CV_Assert( step[dims-1] == (size_t)CV_ELEM_SIZE(flags) );
    }

    finalizeHdr(*this);
}

void UMat::copySize(const UMat& m)
{
    setSize(*this, m.dims, 0, 0);
    for( int i = 0; i < dims; i++ )
    {
        size[i] = m.size[i];
        step[i] = m.step[i];
    }
}


UMat::~UMat()
{
    release();
    if( step.p != step.buf )
        fastFree(step.p);
}

void UMat::deallocate()
{
    u->currAllocator->deallocate(u);
}


UMat::UMat(const UMat& m, const Range& _rowRange, const Range& _colRange)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), u(0), offset(0), size(&rows)
{
    CV_Assert( m.dims >= 2 );
    if( m.dims > 2 )
    {
        AutoBuffer<Range> rs(m.dims);
        rs[0] = _rowRange;
        rs[1] = _colRange;
        for( int i = 2; i < m.dims; i++ )
            rs[i] = Range::all();
        *this = m(rs);
        return;
    }

    *this = m;
    if( _rowRange != Range::all() && _rowRange != Range(0,rows) )
    {
        CV_Assert( 0 <= _rowRange.start && _rowRange.start <= _rowRange.end && _rowRange.end <= m.rows );
        rows = _rowRange.size();
        offset += step*_rowRange.start;
        flags |= SUBMATRIX_FLAG;
    }

    if( _colRange != Range::all() && _colRange != Range(0,cols) )
    {
        CV_Assert( 0 <= _colRange.start && _colRange.start <= _colRange.end && _colRange.end <= m.cols );
        cols = _colRange.size();
        offset += _colRange.start*elemSize();
        flags &= cols < m.cols ? ~CONTINUOUS_FLAG : -1;
        flags |= SUBMATRIX_FLAG;
    }

    if( rows == 1 )
        flags |= CONTINUOUS_FLAG;

    if( rows <= 0 || cols <= 0 )
    {
        release();
        rows = cols = 0;
    }
}


UMat::UMat(const UMat& m, const Rect& roi)
    : flags(m.flags), dims(2), rows(roi.height), cols(roi.width),
    allocator(m.allocator), u(m.u), offset(m.offset + roi.y*m.step[0]), size(&rows)
{
    CV_Assert( m.dims <= 2 );
    flags &= roi.width < m.cols ? ~CONTINUOUS_FLAG : -1;
    flags |= roi.height == 1 ? CONTINUOUS_FLAG : 0;

    size_t esz = CV_ELEM_SIZE(flags);
    offset += roi.x*esz;
    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols &&
              0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );
    if( u )
        CV_XADD(&(u->urefcount), 1);
    if( roi.width < m.cols || roi.height < m.rows )
        flags |= SUBMATRIX_FLAG;

    step[0] = m.step[0]; step[1] = esz;

    if( rows <= 0 || cols <= 0 )
    {
        release();
        rows = cols = 0;
    }
}


UMat::UMat(const UMat& m, const Range* ranges)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), u(0), offset(0), size(&rows)
{
    int i, d = m.dims;

    CV_Assert(ranges);
    for( i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        CV_Assert( r == Range::all() || (0 <= r.start && r.start < r.end && r.end <= m.size[i]) );
    }
    *this = m;
    for( i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        if( r != Range::all() && r != Range(0, size.p[i]))
        {
            size.p[i] = r.end - r.start;
            offset += r.start*step.p[i];
            flags |= SUBMATRIX_FLAG;
        }
    }
    updateContinuityFlag(*this);
}

UMat UMat::diag(int d) const
{
    CV_Assert( dims <= 2 );
    UMat m = *this;
    size_t esz = elemSize();
    int len;

    if( d >= 0 )
    {
        len = std::min(cols - d, rows);
        m.offset += esz*d;
    }
    else
    {
        len = std::min(rows + d, cols);
        m.offset -= step[0]*d;
    }
    CV_DbgAssert( len > 0 );

    m.size[0] = m.rows = len;
    m.size[1] = m.cols = 1;
    m.step[0] += (len > 1 ? esz : 0);

    if( m.rows > 1 )
        m.flags &= ~CONTINUOUS_FLAG;
    else
        m.flags |= CONTINUOUS_FLAG;

    if( size() != Size(1,1) )
        m.flags |= SUBMATRIX_FLAG;

    return m;
}

void UMat::locateROI( Size& wholeSize, Point& ofs ) const
{
    CV_Assert( dims <= 2 && step[0] > 0 );
    size_t esz = elemSize(), minstep;
    ptrdiff_t delta1 = (ptrdiff_t)offset, delta2 = (ptrdiff_t)u->size;

    if( delta1 == 0 )
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = (int)(delta1/step[0]);
        ofs.x = (int)((delta1 - step[0]*ofs.y)/esz);
        CV_DbgAssert( offset == (size_t)(ofs.y*step[0] + ofs.x*esz) );
    }
    minstep = (ofs.x + cols)*esz;
    wholeSize.height = (int)((delta2 - minstep)/step[0] + 1);
    wholeSize.height = std::max(wholeSize.height, ofs.y + rows);
    wholeSize.width = (int)((delta2 - step*(wholeSize.height-1))/esz);
    wholeSize.width = std::max(wholeSize.width, ofs.x + cols);
}


UMat& UMat::adjustROI( int dtop, int dbottom, int dleft, int dright )
{
    CV_Assert( dims <= 2 && step[0] > 0 );
    Size wholeSize; Point ofs;
    size_t esz = elemSize();
    locateROI( wholeSize, ofs );
    int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
    int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
    offset += (row1 - ofs.y)*step + (col1 - ofs.x)*esz;
    rows = row2 - row1; cols = col2 - col1;
    size.p[0] = rows; size.p[1] = cols;
    if( esz*cols == step[0] || rows == 1 )
        flags |= CONTINUOUS_FLAG;
    else
        flags &= ~CONTINUOUS_FLAG;
    return *this;
}


UMat UMat::reshape(int new_cn, int new_rows) const
{
    int cn = channels();
    UMat hdr = *this;

    if( dims > 2 && new_rows == 0 && new_cn != 0 && size[dims-1]*cn % new_cn == 0 )
    {
        hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
        hdr.step[dims-1] = CV_ELEM_SIZE(hdr.flags);
        hdr.size[dims-1] = hdr.size[dims-1]*cn / new_cn;
        return hdr;
    }

    CV_Assert( dims <= 2 );

    if( new_cn == 0 )
        new_cn = cn;

    int total_width = cols * cn;

    if( (new_cn > total_width || total_width % new_cn != 0) && new_rows == 0 )
        new_rows = rows * total_width / new_cn;

    if( new_rows != 0 && new_rows != rows )
    {
        int total_size = total_width * rows;
        if( !isContinuous() )
            CV_Error( CV_BadStep,
            "The matrix is not continuous, thus its number of rows can not be changed" );

        if( (unsigned)new_rows > (unsigned)total_size )
            CV_Error( CV_StsOutOfRange, "Bad new number of rows" );

        total_width = total_size / new_rows;

        if( total_width * new_rows != total_size )
            CV_Error( CV_StsBadArg, "The total number of matrix elements "
                                    "is not divisible by the new number of rows" );

        hdr.rows = new_rows;
        hdr.step[0] = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if( new_width * new_cn != total_width )
        CV_Error( CV_BadNumChannels,
        "The total width is not divisible by the new number of channels" );

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
    hdr.step[1] = CV_ELEM_SIZE(hdr.flags);
    return hdr;
}

UMat UMat::diag(const UMat& d)
{
    CV_Assert( d.cols == 1 || d.rows == 1 );
    int len = d.rows + d.cols - 1;
    UMat m(len, len, d.type(), Scalar(0));
    UMat md = m.diag();
    if( d.cols == 1 )
        d.copyTo(md);
    else
        transpose(d, md);
    return m;
}

int UMat::checkVector(int _elemChannels, int _depth, bool _requireContinuous) const
{
    return (depth() == _depth || _depth <= 0) &&
        (isContinuous() || !_requireContinuous) &&
        ((dims == 2 && (((rows == 1 || cols == 1) && channels() == _elemChannels) ||
                        (cols == _elemChannels && channels() == 1))) ||
        (dims == 3 && channels() == 1 && size.p[2] == _elemChannels && (size.p[0] == 1 || size.p[1] == 1) &&
         (isContinuous() || step.p[1] == step.p[2]*size.p[2])))
    ? (int)(total()*channels()/_elemChannels) : -1;
}


UMat UMat::cross(InputArray) const
{
    CV_Error(CV_StsNotImplemented, "");
    return UMat();
}


UMat UMat::reshape(int _cn, int _newndims, const int* _newsz) const
{
    if(_newndims == dims)
    {
        if(_newsz == 0)
            return reshape(_cn);
        if(_newndims == 2)
            return reshape(_cn, _newsz[0]);
    }

    CV_Error(CV_StsNotImplemented, "");
    // TBD
    return UMat();
}


Mat UMat::getMat(int accessFlags) const
{
    if(!u)
        return Mat();
    u->currAllocator->map(u, accessFlags | ACCESS_READ);
    CV_Assert(u->data != 0);
    Mat hdr(dims, size.p, type(), u->data + offset, step.p);
    hdr.flags = flags;
    hdr.u = u;
    hdr.datastart = u->data;
    hdr.data = hdr.datastart + offset;
    hdr.datalimit = hdr.dataend = u->data + u->size;
    CV_XADD(&hdr.u->refcount, 1);
    return hdr;
}

void* UMat::handle(int accessFlags) const
{
    if( !u )
        return 0;

    if ((accessFlags & ACCESS_WRITE) != 0)
        u->markHostCopyObsolete(true);

    // check flags: if CPU copy is newer, copy it back to GPU.
    if( u->deviceCopyObsolete() )
    {
        CV_Assert(u->refcount == 0);
        u->currAllocator->unmap(u);
    }
    return u->handle;
}

void UMat::ndoffset(size_t* ofs) const
{
    // offset = step[0]*ofs[0] + step[1]*ofs[1] + step[2]*ofs[2] + ...;
    size_t val = offset;
    for( int i = 0; i < dims; i++ )
    {
        size_t s = step.p[i];
        ofs[i] = val / s;
        val -= ofs[i]*s;
    }
}

void UMat::copyTo(OutputArray _dst) const
{
    int dtype = _dst.type();
    if( _dst.fixedType() && dtype != type() )
    {
        CV_Assert( channels() == CV_MAT_CN(dtype) );
        convertTo( _dst, dtype );
        return;
    }

    if( empty() )
    {
        _dst.release();
        return;
    }

    size_t i, sz[CV_MAX_DIM], srcofs[CV_MAX_DIM], dstofs[CV_MAX_DIM], esz = elemSize();
    for( i = 0; i < (size_t)dims; i++ )
        sz[i] = size.p[i];
    sz[dims-1] *= esz;
    ndoffset(srcofs);
    srcofs[dims-1] *= esz;

    _dst.create( dims, size.p, type() );
    if( _dst.kind() == _InputArray::UMAT )
    {
        UMat dst = _dst.getUMat();
        void* srchandle = handle(ACCESS_READ);
        void* dsthandle = dst.handle(ACCESS_WRITE);
        if( srchandle == dsthandle && dst.offset == offset )
            return;
        dst.ndoffset(dstofs);
        CV_Assert(u->currAllocator == dst.u->currAllocator);
        u->currAllocator->copy(u, dst.u, dims, sz, srcofs, step.p, dstofs, dst.step.p, false);
    }
    else
    {
        Mat dst = _dst.getMat();
        u->currAllocator->download(u, dst.data, dims, sz, srcofs, step.p, dst.step.p);
    }
}

void UMat::convertTo(OutputArray, int, double, double) const
{
    CV_Error(Error::StsNotImplemented, "");
}

UMat& UMat::setTo(InputArray _value, InputArray _mask)
{
    bool haveMask = !_mask.empty();
    int tp = type(), cn = CV_MAT_CN(tp);
    if( dims <= 2 && cn <= 4 && ocl::useOpenCL() )
    {
        Mat value = _value.getMat();
        CV_Assert( checkScalar(value, type(), _value.kind(), _InputArray::UMAT) );
        double buf[4];
        convertAndUnrollScalar(value, tp, (uchar*)buf, 1);

        char opts[1024];
        sprintf(opts, "-D dstT=%s", ocl::memopTypeToStr(tp));

        ocl::Kernel setK(haveMask ? "setMask" : "set", ocl::core::copyset_oclsrc, opts);
        if( !setK.empty() )
        {
            ocl::KernelArg scalararg(0, 0, 0, buf, CV_ELEM_SIZE(tp));
            UMat mask;

            if( haveMask )
            {
                mask = _mask.getUMat();
                CV_Assert( mask.size() == size() && mask.type() == CV_8U );
                ocl::KernelArg maskarg = ocl::KernelArg::ReadOnlyNoSize(mask);
                ocl::KernelArg dstarg = ocl::KernelArg::ReadWrite(*this);
                setK.args(maskarg, dstarg, scalararg);
            }
            else
            {
                ocl::KernelArg dstarg = ocl::KernelArg::WriteOnly(*this);
                setK.args(dstarg, scalararg);
            }

            size_t globalsize[] = { cols, rows };
            if( setK.run(2, globalsize, 0, false) )
                return *this;
        }
    }
    Mat m = getMat(haveMask ? ACCESS_RW : ACCESS_WRITE);
    m.setTo(_value, _mask);
    return *this;
}

UMat& UMat::operator = (const Scalar&)
{
    CV_Error(Error::StsNotImplemented, "");
    return *this;
}

}

/* End of file. */
