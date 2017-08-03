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
#include "opencl_kernels_core.hpp"

#include "bufferpool.impl.hpp"

/****************************************************************************************\
*                           [scaled] Identity matrix initialization                      *
\****************************************************************************************/

namespace cv {

void MatAllocator::map(UMatData*, int) const
{
}

void MatAllocator::unmap(UMatData* u) const
{
    if(u->urefcount == 0 && u->refcount == 0)
    {
        deallocate(u);
    }
}

void MatAllocator::download(UMatData* u, void* dstptr,
         int dims, const size_t sz[],
         const size_t srcofs[], const size_t srcstep[],
         const size_t dststep[]) const
{
    if(!u)
        return;
    int isz[CV_MAX_DIM];
    uchar* srcptr = u->data;
    for( int i = 0; i < dims; i++ )
    {
        CV_Assert( sz[i] <= (size_t)INT_MAX );
        if( sz[i] == 0 )
            return;
        if( srcofs )
            srcptr += srcofs[i]*(i <= dims-2 ? srcstep[i] : 1);
        isz[i] = (int)sz[i];
    }

    Mat src(dims, isz, CV_8U, srcptr, srcstep);
    Mat dst(dims, isz, CV_8U, dstptr, dststep);

    const Mat* arrays[] = { &src, &dst };
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs, 2);
    size_t planesz = it.size;

    for( size_t j = 0; j < it.nplanes; j++, ++it )
        memcpy(ptrs[1], ptrs[0], planesz);
}


void MatAllocator::upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                    const size_t dstofs[], const size_t dststep[],
                    const size_t srcstep[]) const
{
    if(!u)
        return;
    int isz[CV_MAX_DIM];
    uchar* dstptr = u->data;
    for( int i = 0; i < dims; i++ )
    {
        CV_Assert( sz[i] <= (size_t)INT_MAX );
        if( sz[i] == 0 )
            return;
        if( dstofs )
            dstptr += dstofs[i]*(i <= dims-2 ? dststep[i] : 1);
        isz[i] = (int)sz[i];
    }

    Mat src(dims, isz, CV_8U, (void*)srcptr, srcstep);
    Mat dst(dims, isz, CV_8U, dstptr, dststep);

    const Mat* arrays[] = { &src, &dst };
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs, 2);
    size_t planesz = it.size;

    for( size_t j = 0; j < it.nplanes; j++, ++it )
        memcpy(ptrs[1], ptrs[0], planesz);
}

void MatAllocator::copy(UMatData* usrc, UMatData* udst, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dstofs[], const size_t dststep[], bool /*sync*/) const
{
    CV_INSTRUMENT_REGION()

    if(!usrc || !udst)
        return;
    int isz[CV_MAX_DIM];
    uchar* srcptr = usrc->data;
    uchar* dstptr = udst->data;
    for( int i = 0; i < dims; i++ )
    {
        CV_Assert( sz[i] <= (size_t)INT_MAX );
        if( sz[i] == 0 )
            return;
        if( srcofs )
            srcptr += srcofs[i]*(i <= dims-2 ? srcstep[i] : 1);
        if( dstofs )
            dstptr += dstofs[i]*(i <= dims-2 ? dststep[i] : 1);
        isz[i] = (int)sz[i];
    }

    Mat src(dims, isz, CV_8U, srcptr, srcstep);
    Mat dst(dims, isz, CV_8U, dstptr, dststep);

    const Mat* arrays[] = { &src, &dst };
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs, 2);
    size_t planesz = it.size;

    for( size_t j = 0; j < it.nplanes; j++, ++it )
        memcpy(ptrs[1], ptrs[0], planesz);
}

BufferPoolController* MatAllocator::getBufferPoolController(const char* id) const
{
    (void)id;
    static DummyBufferPoolController dummy;
    return &dummy;
}

class StdMatAllocator : public MatAllocator
{
public:
    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data0, size_t* step, int /*flags*/, UMatUsageFlags /*usageFlags*/) const
    {
        size_t total = CV_ELEM_SIZE(type);
        for( int i = dims-1; i >= 0; i-- )
        {
            if( step )
            {
                if( data0 && step[i] != CV_AUTOSTEP )
                {
                    CV_Assert(total <= step[i]);
                    total = step[i];
                }
                else
                    step[i] = total;
            }
            total *= sizes[i];
        }
        uchar* data = data0 ? (uchar*)data0 : (uchar*)fastMalloc(total);
        UMatData* u = new UMatData(this);
        u->data = u->origdata = data;
        u->size = total;
        if(data0)
            u->flags |= UMatData::USER_ALLOCATED;

        return u;
    }

    bool allocate(UMatData* u, int /*accessFlags*/, UMatUsageFlags /*usageFlags*/) const
    {
        if(!u) return false;
        return true;
    }

    void deallocate(UMatData* u) const
    {
        if(!u)
            return;

        CV_Assert(u->urefcount == 0);
        CV_Assert(u->refcount == 0);
        if( !(u->flags & UMatData::USER_ALLOCATED) )
        {
            fastFree(u->origdata);
            u->origdata = 0;
        }
        delete u;
    }
};
namespace
{
    MatAllocator* volatile g_matAllocator = NULL;
}


MatAllocator* Mat::getDefaultAllocator()
{
    if (g_matAllocator == NULL)
    {
        cv::AutoLock lock(cv::getInitializationMutex());
        if (g_matAllocator == NULL)
        {
            g_matAllocator = getStdAllocator();
        }
    }
    return g_matAllocator;
}
void Mat::setDefaultAllocator(MatAllocator* allocator)
{
    g_matAllocator = allocator;
}
MatAllocator* Mat::getStdAllocator()
{
    CV_SINGLETON_LAZY_INIT(MatAllocator, new StdMatAllocator())
}

void swap( Mat& a, Mat& b )
{
    std::swap(a.flags, b.flags);
    std::swap(a.dims, b.dims);
    std::swap(a.rows, b.rows);
    std::swap(a.cols, b.cols);
    std::swap(a.data, b.data);
    std::swap(a.datastart, b.datastart);
    std::swap(a.dataend, b.dataend);
    std::swap(a.datalimit, b.datalimit);
    std::swap(a.allocator, b.allocator);
    std::swap(a.u, b.u);

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


static inline void setSize( Mat& m, int _dims, const int* _sz,
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

    size_t esz = CV_ELEM_SIZE(m.flags), esz1 = CV_ELEM_SIZE1(m.flags), total = esz;
    for( int i = _dims-1; i >= 0; i-- )
    {
        int s = _sz[i];
        CV_Assert( s >= 0 );
        m.size.p[i] = s;

        if( _steps )
        {
            if (_steps[i] % esz1 != 0)
            {
                CV_Error(Error::BadStep, "Step must be a multiple of esz1");
            }

            m.step.p[i] = i < _dims-1 ? _steps[i] : esz;
        }
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

static void updateContinuityFlag(Mat& m)
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

    uint64 t = (uint64)m.step[0]*m.size[0];
    if( j <= i && t == (size_t)t )
        m.flags |= Mat::CONTINUOUS_FLAG;
    else
        m.flags &= ~Mat::CONTINUOUS_FLAG;
}

static void finalizeHdr(Mat& m)
{
    updateContinuityFlag(m);
    int d = m.dims;
    if( d > 2 )
        m.rows = m.cols = -1;
    if(m.u)
        m.datastart = m.data = m.u->data;
    if( m.data )
    {
        m.datalimit = m.datastart + m.size[0]*m.step[0];
        if( m.size[0] > 0 )
        {
            m.dataend = m.ptr() + m.size[d-1]*m.step[d-1];
            for( int i = 0; i < d-1; i++ )
                m.dataend += (m.size[i] - 1)*m.step[i];
        }
        else
            m.dataend = m.datalimit;
    }
    else
        m.dataend = m.datalimit = 0;
}


void Mat::create(int d, const int* _sizes, int _type)
{
    int i;
    CV_Assert(0 <= d && d <= CV_MAX_DIM && _sizes);
    _type = CV_MAT_TYPE(_type);

    if( data && (d == dims || (d == 1 && dims <= 2)) && _type == type() )
    {
        if( d == 2 && rows == _sizes[0] && cols == _sizes[1] )
            return;
        for( i = 0; i < d; i++ )
            if( size[i] != _sizes[i] )
                break;
        if( i == d && (d > 1 || size[1] == 1))
            return;
    }

    int _sizes_backup[CV_MAX_DIM]; // #5991
    if (_sizes == (this->size.p))
    {
        for(i = 0; i < d; i++ )
            _sizes_backup[i] = _sizes[i];
        _sizes = _sizes_backup;
    }

    release();
    if( d == 0 )
        return;
    flags = (_type & CV_MAT_TYPE_MASK) | MAGIC_VAL;
    setSize(*this, d, _sizes, 0, true);

    if( total() > 0 )
    {
        MatAllocator *a = allocator, *a0 = getDefaultAllocator();
#ifdef HAVE_TGPU
        if( !a || a == tegra::getAllocator() )
            a = tegra::getAllocator(d, _sizes, _type);
#endif
        if(!a)
            a = a0;
        try
        {
            u = a->allocate(dims, size, _type, 0, step.p, 0, USAGE_DEFAULT);
            CV_Assert(u != 0);
        }
        catch(...)
        {
            if(a != a0)
                u = a0->allocate(dims, size, _type, 0, step.p, 0, USAGE_DEFAULT);
            CV_Assert(u != 0);
        }
        CV_Assert( step[dims-1] == (size_t)CV_ELEM_SIZE(flags) );
    }

    addref();
    finalizeHdr(*this);
}

void Mat::create(const std::vector<int>& _sizes, int _type)
{
    create((int)_sizes.size(), _sizes.data(), _type);
}

void Mat::copySize(const Mat& m)
{
    setSize(*this, m.dims, 0, 0);
    for( int i = 0; i < dims; i++ )
    {
        size[i] = m.size[i];
        step[i] = m.step[i];
    }
}

void Mat::deallocate()
{
    if(u)
    {
        UMatData* u_ = u;
        u = NULL;
        (u_->currAllocator ? u_->currAllocator : allocator ? allocator : getDefaultAllocator())->unmap(u_);
    }
}

Mat::Mat(const Mat& m, const Range& _rowRange, const Range& _colRange)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
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
        data += step*_rowRange.start;
        flags |= SUBMATRIX_FLAG;
    }

    if( _colRange != Range::all() && _colRange != Range(0,cols) )
    {
        CV_Assert( 0 <= _colRange.start && _colRange.start <= _colRange.end && _colRange.end <= m.cols );
        cols = _colRange.size();
        data += _colRange.start*elemSize();
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


Mat::Mat(const Mat& m, const Rect& roi)
    : flags(m.flags), dims(2), rows(roi.height), cols(roi.width),
    data(m.data + roi.y*m.step[0]),
    datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit),
    allocator(m.allocator), u(m.u), size(&rows)
{
    CV_Assert( m.dims <= 2 );
    flags &= roi.width < m.cols ? ~CONTINUOUS_FLAG : -1;
    flags |= roi.height == 1 ? CONTINUOUS_FLAG : 0;

    size_t esz = CV_ELEM_SIZE(flags);
    data += roi.x*esz;
    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols &&
              0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );
    if( u )
        CV_XADD(&u->refcount, 1);
    if( roi.width < m.cols || roi.height < m.rows )
        flags |= SUBMATRIX_FLAG;

    step[0] = m.step[0]; step[1] = esz;

    if( rows <= 0 || cols <= 0 )
    {
        release();
        rows = cols = 0;
    }
}


Mat::Mat(int _dims, const int* _sizes, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
{
    flags |= CV_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    setSize(*this, _dims, _sizes, _steps, true);
    finalizeHdr(*this);
}


Mat::Mat(const std::vector<int>& _sizes, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
{
    flags |= CV_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    setSize(*this, (int)_sizes.size(), _sizes.data(), _steps, true);
    finalizeHdr(*this);
}


Mat::Mat(const Mat& m, const Range* ranges)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
{
    int d = m.dims;

    CV_Assert(ranges);
    for( int i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        CV_Assert( r == Range::all() || (0 <= r.start && r.start < r.end && r.end <= m.size[i]) );
    }
    *this = m;
    for( int i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        if( r != Range::all() && r != Range(0, size.p[i]))
        {
            size.p[i] = r.end - r.start;
            data += r.start*step.p[i];
            flags |= SUBMATRIX_FLAG;
        }
    }
    updateContinuityFlag(*this);
}

Mat::Mat(const Mat& m, const std::vector<Range>& ranges)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
    datalimit(0), allocator(0), u(0), size(&rows)
{
    int d = m.dims;

    CV_Assert((int)ranges.size() == d);
    for (int i = 0; i < d; i++)
    {
        Range r = ranges[i];
        CV_Assert(r == Range::all() || (0 <= r.start && r.start < r.end && r.end <= m.size[i]));
    }
    *this = m;
    for (int i = 0; i < d; i++)
    {
        Range r = ranges[i];
        if (r != Range::all() && r != Range(0, size.p[i]))
        {
            size.p[i] = r.end - r.start;
            data += r.start*step.p[i];
            flags |= SUBMATRIX_FLAG;
        }
    }
    updateContinuityFlag(*this);
}

static Mat cvMatNDToMat(const CvMatND* m, bool copyData)
{
    Mat thiz;

    if( !m )
        return thiz;
    thiz.datastart = thiz.data = m->data.ptr;
    thiz.flags |= CV_MAT_TYPE(m->type);
    int _sizes[CV_MAX_DIM];
    size_t _steps[CV_MAX_DIM];

    int d = m->dims;
    for( int i = 0; i < d; i++ )
    {
        _sizes[i] = m->dim[i].size;
        _steps[i] = m->dim[i].step;
    }

    setSize(thiz, d, _sizes, _steps);
    finalizeHdr(thiz);

    if( copyData )
    {
        Mat temp(thiz);
        thiz.release();
        temp.copyTo(thiz);
    }

    return thiz;
}

static Mat cvMatToMat(const CvMat* m, bool copyData)
{
    Mat thiz;

    if( !m )
        return thiz;

    if( !copyData )
    {
        thiz.flags = Mat::MAGIC_VAL + (m->type & (CV_MAT_TYPE_MASK|CV_MAT_CONT_FLAG));
        thiz.dims = 2;
        thiz.rows = m->rows;
        thiz.cols = m->cols;
        thiz.datastart = thiz.data = m->data.ptr;
        size_t esz = CV_ELEM_SIZE(m->type), minstep = thiz.cols*esz, _step = m->step;
        if( _step == 0 )
            _step = minstep;
        thiz.datalimit = thiz.datastart + _step*thiz.rows;
        thiz.dataend = thiz.datalimit - _step + minstep;
        thiz.step[0] = _step; thiz.step[1] = esz;
    }
    else
    {
        thiz.datastart = thiz.dataend = thiz.data = 0;
        Mat(m->rows, m->cols, m->type, m->data.ptr, m->step).copyTo(thiz);
    }

    return thiz;
}


static Mat iplImageToMat(const IplImage* img, bool copyData)
{
    Mat m;

    if( !img )
        return m;

    m.dims = 2;
    CV_DbgAssert(CV_IS_IMAGE(img) && img->imageData != 0);

    int imgdepth = IPL2CV_DEPTH(img->depth);
    size_t esz;
    m.step[0] = img->widthStep;

    if(!img->roi)
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL);
        m.flags = Mat::MAGIC_VAL + CV_MAKETYPE(imgdepth, img->nChannels);
        m.rows = img->height;
        m.cols = img->width;
        m.datastart = m.data = (uchar*)img->imageData;
        esz = CV_ELEM_SIZE(m.flags);
    }
    else
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL || img->roi->coi != 0);
        bool selectedPlane = img->roi->coi && img->dataOrder == IPL_DATA_ORDER_PLANE;
        m.flags = Mat::MAGIC_VAL + CV_MAKETYPE(imgdepth, selectedPlane ? 1 : img->nChannels);
        m.rows = img->roi->height;
        m.cols = img->roi->width;
        esz = CV_ELEM_SIZE(m.flags);
        m.datastart = m.data = (uchar*)img->imageData +
            (selectedPlane ? (img->roi->coi - 1)*m.step*img->height : 0) +
            img->roi->yOffset*m.step[0] + img->roi->xOffset*esz;
    }
    m.datalimit = m.datastart + m.step.p[0]*m.rows;
    m.dataend = m.datastart + m.step.p[0]*(m.rows-1) + esz*m.cols;
    m.flags |= (m.cols*esz == m.step.p[0] || m.rows == 1 ? Mat::CONTINUOUS_FLAG : 0);
    m.step[1] = esz;

    if( copyData )
    {
        Mat m2 = m;
        m.release();
        if( !img->roi || !img->roi->coi ||
            img->dataOrder == IPL_DATA_ORDER_PLANE)
            m2.copyTo(m);
        else
        {
            int ch[] = {img->roi->coi - 1, 0};
            m.create(m2.rows, m2.cols, m2.type());
            mixChannels(&m2, 1, &m, 1, ch, 1);
        }
    }

    return m;
}

Mat Mat::diag(int d) const
{
    CV_Assert( dims <= 2 );
    Mat m = *this;
    size_t esz = elemSize();
    int len;

    if( d >= 0 )
    {
        len = std::min(cols - d, rows);
        m.data += esz*d;
    }
    else
    {
        len = std::min(rows + d, cols);
        m.data -= step[0]*d;
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

void Mat::pop_back(size_t nelems)
{
    CV_Assert( nelems <= (size_t)size.p[0] );

    if( isSubmatrix() )
        *this = rowRange(0, size.p[0] - (int)nelems);
    else
    {
        size.p[0] -= (int)nelems;
        dataend -= nelems*step.p[0];
        /*if( size.p[0] <= 1 )
        {
            if( dims <= 2 )
                flags |= CONTINUOUS_FLAG;
            else
                updateContinuityFlag(*this);
        }*/
    }
}


void Mat::push_back_(const void* elem)
{
    int r = size.p[0];
    if( isSubmatrix() || dataend + step.p[0] > datalimit )
        reserve( std::max(r + 1, (r*3+1)/2) );

    size_t esz = elemSize();
    memcpy(data + r*step.p[0], elem, esz);
    size.p[0] = r + 1;
    dataend += step.p[0];
    if( esz < step.p[0] )
        flags &= ~CONTINUOUS_FLAG;
}

void Mat::reserve(size_t nelems)
{
    const size_t MIN_SIZE = 64;

    CV_Assert( (int)nelems >= 0 );
    if( !isSubmatrix() && data + step.p[0]*nelems <= datalimit )
        return;

    int r = size.p[0];

    if( (size_t)r >= nelems )
        return;

    size.p[0] = std::max((int)nelems, 1);
    size_t newsize = total()*elemSize();

    if( newsize < MIN_SIZE )
        size.p[0] = (int)((MIN_SIZE + newsize - 1)*nelems/newsize);

    Mat m(dims, size.p, type());
    size.p[0] = r;
    if( r > 0 )
    {
        Mat mpart = m.rowRange(0, r);
        copyTo(mpart);
    }

    *this = m;
    size.p[0] = r;
    dataend = data + step.p[0]*r;
}

void Mat::reserveBuffer(size_t nbytes)
{
    size_t esz = 1;
    int mtype = CV_8UC1;
    if (!empty())
    {
        if (!isSubmatrix() && data + nbytes <= dataend)//Should it be datalimit?
            return;
        esz = elemSize();
        mtype = type();
    }

    size_t nelems = (nbytes - 1) / esz + 1;

#if SIZE_MAX > UINT_MAX
    CV_Assert(nelems <= size_t(INT_MAX)*size_t(INT_MAX));
    int newrows = nelems > size_t(INT_MAX) ? nelems > 0x400*size_t(INT_MAX) ? nelems > 0x100000 * size_t(INT_MAX) ? nelems > 0x40000000 * size_t(INT_MAX) ?
                  size_t(INT_MAX) : 0x40000000 : 0x100000 : 0x400 : 1;
#else
    int newrows = nelems > size_t(INT_MAX) ? 2 : 1;
#endif
    int newcols = (int)((nelems - 1) / newrows + 1);

    create(newrows, newcols, mtype);
}


void Mat::resize(size_t nelems)
{
    int saveRows = size.p[0];
    if( saveRows == (int)nelems )
        return;
    CV_Assert( (int)nelems >= 0 );

    if( isSubmatrix() || data + step.p[0]*nelems > datalimit )
        reserve(nelems);

    size.p[0] = (int)nelems;
    dataend += (size.p[0] - saveRows)*step.p[0];

    //updateContinuityFlag(*this);
}


void Mat::resize(size_t nelems, const Scalar& s)
{
    int saveRows = size.p[0];
    resize(nelems);

    if( size.p[0] > saveRows )
    {
        Mat part = rowRange(saveRows, size.p[0]);
        part = s;
    }
}

void Mat::push_back(const Mat& elems)
{
    int r = size.p[0], delta = elems.size.p[0];
    if( delta == 0 )
        return;
    if( this == &elems )
    {
        Mat tmp = elems;
        push_back(tmp);
        return;
    }
    if( !data )
    {
        *this = elems.clone();
        return;
    }

    size.p[0] = elems.size.p[0];
    bool eq = size == elems.size;
    size.p[0] = r;
    if( !eq )
        CV_Error(CV_StsUnmatchedSizes, "Pushed vector length is not equal to matrix row length");
    if( type() != elems.type() )
        CV_Error(CV_StsUnmatchedFormats, "Pushed vector type is not the same as matrix type");

    if( isSubmatrix() || dataend + step.p[0]*delta > datalimit )
        reserve( std::max(r + delta, (r*3+1)/2) );

    size.p[0] += delta;
    dataend += step.p[0]*delta;

    //updateContinuityFlag(*this);

    if( isContinuous() && elems.isContinuous() )
        memcpy(data + r*step.p[0], elems.data, elems.total()*elems.elemSize());
    else
    {
        Mat part = rowRange(r, r + delta);
        elems.copyTo(part);
    }
}


Mat cvarrToMat(const CvArr* arr, bool copyData,
               bool /*allowND*/, int coiMode, AutoBuffer<double>* abuf )
{
    if( !arr )
        return Mat();
    if( CV_IS_MAT_HDR_Z(arr) )
        return cvMatToMat((const CvMat*)arr, copyData);
    if( CV_IS_MATND(arr) )
        return cvMatNDToMat((const CvMatND*)arr, copyData );
    if( CV_IS_IMAGE(arr) )
    {
        const IplImage* iplimg = (const IplImage*)arr;
        if( coiMode == 0 && iplimg->roi && iplimg->roi->coi > 0 )
            CV_Error(CV_BadCOI, "COI is not supported by the function");
        return iplImageToMat(iplimg, copyData);
    }
    if( CV_IS_SEQ(arr) )
    {
        CvSeq* seq = (CvSeq*)arr;
        int total = seq->total, type = CV_MAT_TYPE(seq->flags), esz = seq->elem_size;
        if( total == 0 )
            return Mat();
        CV_Assert(total > 0 && CV_ELEM_SIZE(seq->flags) == esz);
        if(!copyData && seq->first->next == seq->first)
            return Mat(total, 1, type, seq->first->data);
        if( abuf )
        {
            abuf->allocate(((size_t)total*esz + sizeof(double)-1)/sizeof(double));
            double* bufdata = *abuf;
            cvCvtSeqToArray(seq, bufdata, CV_WHOLE_SEQ);
            return Mat(total, 1, type, bufdata);
        }

        Mat buf(total, 1, type);
        cvCvtSeqToArray(seq, buf.ptr(), CV_WHOLE_SEQ);
        return buf;
    }
    CV_Error(CV_StsBadArg, "Unknown array type");
    return Mat();
}

void Mat::locateROI( Size& wholeSize, Point& ofs ) const
{
    CV_Assert( dims <= 2 && step[0] > 0 );
    size_t esz = elemSize(), minstep;
    ptrdiff_t delta1 = data - datastart, delta2 = dataend - datastart;

    if( delta1 == 0 )
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = (int)(delta1/step[0]);
        ofs.x = (int)((delta1 - step[0]*ofs.y)/esz);
        CV_DbgAssert( data == datastart + ofs.y*step[0] + ofs.x*esz );
    }
    minstep = (ofs.x + cols)*esz;
    wholeSize.height = (int)((delta2 - minstep)/step[0] + 1);
    wholeSize.height = std::max(wholeSize.height, ofs.y + rows);
    wholeSize.width = (int)((delta2 - step*(wholeSize.height-1))/esz);
    wholeSize.width = std::max(wholeSize.width, ofs.x + cols);
}

Mat& Mat::adjustROI( int dtop, int dbottom, int dleft, int dright )
{
    CV_Assert( dims <= 2 && step[0] > 0 );
    Size wholeSize; Point ofs;
    size_t esz = elemSize();
    locateROI( wholeSize, ofs );
    int row1 = std::min(std::max(ofs.y - dtop, 0), wholeSize.height), row2 = std::max(0, std::min(ofs.y + rows + dbottom, wholeSize.height));
    int col1 = std::min(std::max(ofs.x - dleft, 0), wholeSize.width), col2 = std::max(0, std::min(ofs.x + cols + dright, wholeSize.width));
    if(row1 > row2)
        std::swap(row1, row2);
    if(col1 > col2)
        std::swap(col1, col2);

    data += (row1 - ofs.y)*step + (col1 - ofs.x)*esz;
    rows = row2 - row1; cols = col2 - col1;
    size.p[0] = rows; size.p[1] = cols;
    if( esz*cols == step[0] || rows == 1 )
        flags |= CONTINUOUS_FLAG;
    else
        flags &= ~CONTINUOUS_FLAG;
    return *this;
}

}

void cv::extractImageCOI(const CvArr* arr, OutputArray _ch, int coi)
{
    Mat mat = cvarrToMat(arr, false, true, 1);
    _ch.create(mat.dims, mat.size, mat.depth());
    Mat ch = _ch.getMat();
    if(coi < 0)
    {
        CV_Assert( CV_IS_IMAGE(arr) );
        coi = cvGetImageCOI((const IplImage*)arr)-1;
    }
    CV_Assert(0 <= coi && coi < mat.channels());
    int _pairs[] = { coi, 0 };
    mixChannels( &mat, 1, &ch, 1, _pairs, 1 );
}

void cv::insertImageCOI(InputArray _ch, CvArr* arr, int coi)
{
    Mat ch = _ch.getMat(), mat = cvarrToMat(arr, false, true, 1);
    if(coi < 0)
    {
        CV_Assert( CV_IS_IMAGE(arr) );
        coi = cvGetImageCOI((const IplImage*)arr)-1;
    }
    CV_Assert(ch.size == mat.size && ch.depth() == mat.depth() && 0 <= coi && coi < mat.channels());
    int _pairs[] = { 0, coi };
    mixChannels( &ch, 1, &mat, 1, _pairs, 1 );
}

namespace cv
{

Mat Mat::reshape(int new_cn, int new_rows) const
{
    int cn = channels();
    Mat hdr = *this;

    if( dims > 2 )
    {
        if( new_rows == 0 && new_cn != 0 && size[dims-1]*cn % new_cn == 0 )
        {
            hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
            hdr.step[dims-1] = CV_ELEM_SIZE(hdr.flags);
            hdr.size[dims-1] = hdr.size[dims-1]*cn / new_cn;
            return hdr;
        }
        if( new_rows > 0 )
        {
            int sz[] = { new_rows, (int)(total()/new_rows) };
            return reshape(new_cn, 2, sz);
        }
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

Mat Mat::diag(const Mat& d)
{
    CV_Assert( d.cols == 1 || d.rows == 1 );
    int len = d.rows + d.cols - 1;
    Mat m(len, len, d.type(), Scalar(0));
    Mat md = m.diag();
    if( d.cols == 1 )
        d.copyTo(md);
    else
        transpose(d, md);
    return m;
}

int Mat::checkVector(int _elemChannels, int _depth, bool _requireContinuous) const
{
    return data && (depth() == _depth || _depth <= 0) &&
        (isContinuous() || !_requireContinuous) &&
        ((dims == 2 && (((rows == 1 || cols == 1) && channels() == _elemChannels) ||
                        (cols == _elemChannels && channels() == 1))) ||
        (dims == 3 && channels() == 1 && size.p[2] == _elemChannels && (size.p[0] == 1 || size.p[1] == 1) &&
         (isContinuous() || step.p[1] == step.p[2]*size.p[2])))
    ? (int)(total()*channels()/_elemChannels) : -1;
}


void scalarToRawData(const Scalar& s, void* _buf, int type, int unroll_to)
{
    CV_INSTRUMENT_REGION()

    int i, depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(cn <= 4);
    switch(depth)
    {
    case CV_8U:
        {
        uchar* buf = (uchar*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<uchar>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_8S:
        {
        schar* buf = (schar*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<schar>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_16U:
        {
        ushort* buf = (ushort*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<ushort>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_16S:
        {
        short* buf = (short*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<short>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_32S:
        {
        int* buf = (int*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<int>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_32F:
        {
        float* buf = (float*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<float>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_64F:
        {
        double* buf = (double*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<double>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        break;
        }
    default:
        CV_Error(CV_StsUnsupportedFormat,"");
    }
}


/*************************************************************************************************\
                                        Input/Output Array
\*************************************************************************************************/

Mat _InputArray::getMat_(int i) const
{
    int k = kind();
    int accessFlags = flags & ACCESS_MASK;

    if( k == MAT )
    {
        const Mat* m = (const Mat*)obj;
        if( i < 0 )
            return *m;
        return m->row(i);
    }

    if( k == UMAT )
    {
        const UMat* m = (const UMat*)obj;
        if( i < 0 )
            return m->getMat(accessFlags);
        return m->getMat(accessFlags).row(i);
    }

    if( k == EXPR )
    {
        CV_Assert( i < 0 );
        return (Mat)*((const MatExpr*)obj);
    }

    if( k == MATX || k == STD_ARRAY )
    {
        CV_Assert( i < 0 );
        return Mat(sz, flags, obj);
    }

    if( k == STD_VECTOR )
    {
        CV_Assert( i < 0 );
        int t = CV_MAT_TYPE(flags);
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;

        return !v.empty() ? Mat(size(), t, (void*)&v[0]) : Mat();
    }

    if( k == STD_BOOL_VECTOR )
    {
        CV_Assert( i < 0 );
        int t = CV_8U;
        const std::vector<bool>& v = *(const std::vector<bool>*)obj;
        int j, n = (int)v.size();
        if( n == 0 )
            return Mat();
        Mat m(1, n, t);
        uchar* dst = m.data;
        for( j = 0; j < n; j++ )
            dst[j] = (uchar)v[j];
        return m;
    }

    if( k == NONE )
        return Mat();

    if( k == STD_VECTOR_VECTOR )
    {
        int t = type(i);
        const std::vector<std::vector<uchar> >& vv = *(const std::vector<std::vector<uchar> >*)obj;
        CV_Assert( 0 <= i && i < (int)vv.size() );
        const std::vector<uchar>& v = vv[i];

        return !v.empty() ? Mat(size(i), t, (void*)&v[0]) : Mat();
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& v = *(const std::vector<Mat>*)obj;
        CV_Assert( 0 <= i && i < (int)v.size() );

        return v[i];
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = (const Mat*)obj;
        CV_Assert( 0 <= i && i < sz.height );

        return v[i];
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *(const std::vector<UMat>*)obj;
        CV_Assert( 0 <= i && i < (int)v.size() );

        return v[i].getMat(accessFlags);
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call mapHost/unmapHost methods for ogl::Buffer object");
        return Mat();
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call download method for cuda::GpuMat object");
        return Mat();
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );

        const cuda::HostMem* cuda_mem = (const cuda::HostMem*)obj;

        return cuda_mem->createMatHeader();
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
    return Mat();
}

UMat _InputArray::getUMat(int i) const
{
    int k = kind();
    int accessFlags = flags & ACCESS_MASK;

    if( k == UMAT )
    {
        const UMat* m = (const UMat*)obj;
        if( i < 0 )
            return *m;
        return m->row(i);
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *(const std::vector<UMat>*)obj;
        CV_Assert( 0 <= i && i < (int)v.size() );

        return v[i];
    }

    if( k == MAT )
    {
        const Mat* m = (const Mat*)obj;
        if( i < 0 )
            return m->getUMat(accessFlags);
        return m->row(i).getUMat(accessFlags);
    }

    return getMat(i).getUMat(accessFlags);
}

void _InputArray::getMatVector(std::vector<Mat>& mv) const
{
    int k = kind();
    int accessFlags = flags & ACCESS_MASK;

    if( k == MAT )
    {
        const Mat& m = *(const Mat*)obj;
        int n = (int)m.size[0];
        mv.resize(n);

        for( int i = 0; i < n; i++ )
            mv[i] = m.dims == 2 ? Mat(1, m.cols, m.type(), (void*)m.ptr(i)) :
                Mat(m.dims-1, &m.size[1], m.type(), (void*)m.ptr(i), &m.step[1]);
        return;
    }

    if( k == EXPR )
    {
        Mat m = *(const MatExpr*)obj;
        int n = m.size[0];
        mv.resize(n);

        for( int i = 0; i < n; i++ )
            mv[i] = m.row(i);
        return;
    }

    if( k == MATX || k == STD_ARRAY )
    {
        size_t n = sz.height, esz = CV_ELEM_SIZE(flags);
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = Mat(1, sz.width, CV_MAT_TYPE(flags), (uchar*)obj + esz*sz.width*i);
        return;
    }

    if( k == STD_VECTOR )
    {
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;

        size_t n = size().width, esz = CV_ELEM_SIZE(flags);
        int t = CV_MAT_DEPTH(flags), cn = CV_MAT_CN(flags);
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = Mat(1, cn, t, (void*)(&v[0] + esz*i));
        return;
    }

    if( k == NONE )
    {
        mv.clear();
        return;
    }

    if( k == STD_VECTOR_VECTOR )
    {
        const std::vector<std::vector<uchar> >& vv = *(const std::vector<std::vector<uchar> >*)obj;
        int n = (int)vv.size();
        int t = CV_MAT_TYPE(flags);
        mv.resize(n);

        for( int i = 0; i < n; i++ )
        {
            const std::vector<uchar>& v = vv[i];
            mv[i] = Mat(size(i), t, (void*)&v[0]);
        }
        return;
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& v = *(const std::vector<Mat>*)obj;
        size_t n = v.size();
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = v[i];
        return;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = (const Mat*)obj;
        size_t n = sz.height;
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = v[i];
        return;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *(const std::vector<UMat>*)obj;
        size_t n = v.size();
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = v[i].getMat(accessFlags);
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _InputArray::getUMatVector(std::vector<UMat>& umv) const
{
    int k = kind();
    int accessFlags = flags & ACCESS_MASK;

    if( k == NONE )
    {
        umv.clear();
        return;
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& v = *(const std::vector<Mat>*)obj;
        size_t n = v.size();
        umv.resize(n);

        for( size_t i = 0; i < n; i++ )
            umv[i] = v[i].getUMat(accessFlags);
        return;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = (const Mat*)obj;
        size_t n = sz.height;
        umv.resize(n);

        for( size_t i = 0; i < n; i++ )
            umv[i] = v[i].getUMat(accessFlags);
        return;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *(const std::vector<UMat>*)obj;
        size_t n = v.size();
        umv.resize(n);

        for( size_t i = 0; i < n; i++ )
            umv[i] = v[i];
        return;
    }

    if( k == UMAT )
    {
        UMat& v = *(UMat*)obj;
        umv.resize(1);
        umv[0] = v;
        return;
    }
    if( k == MAT )
    {
        Mat& v = *(Mat*)obj;
        umv.resize(1);
        umv[0] = v.getUMat(accessFlags);
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

cuda::GpuMat _InputArray::getGpuMat() const
{
    int k = kind();

    if (k == CUDA_GPU_MAT)
    {
        const cuda::GpuMat* d_mat = (const cuda::GpuMat*)obj;
        return *d_mat;
    }

    if (k == CUDA_HOST_MEM)
    {
        const cuda::HostMem* cuda_mem = (const cuda::HostMem*)obj;
        return cuda_mem->createGpuMatHeader();
    }

    if (k == OPENGL_BUFFER)
    {
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call mapDevice/unmapDevice methods for ogl::Buffer object");
        return cuda::GpuMat();
    }

    if (k == NONE)
        return cuda::GpuMat();

    CV_Error(cv::Error::StsNotImplemented, "getGpuMat is available only for cuda::GpuMat and cuda::HostMem");
    return cuda::GpuMat();
}
void _InputArray::getGpuMatVector(std::vector<cuda::GpuMat>& gpumv) const
{
    int k = kind();
    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        gpumv = *(std::vector<cuda::GpuMat>*)obj;
    }
}
ogl::Buffer _InputArray::getOGlBuffer() const
{
    int k = kind();

    CV_Assert(k == OPENGL_BUFFER);

    const ogl::Buffer* gl_buf = (const ogl::Buffer*)obj;
    return *gl_buf;
}

int _InputArray::kind() const
{
    return flags & KIND_MASK;
}

int _InputArray::rows(int i) const
{
    return size(i).height;
}

int _InputArray::cols(int i) const
{
    return size(i).width;
}

Size _InputArray::size(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->size();
    }

    if( k == EXPR )
    {
        CV_Assert( i < 0 );
        return ((const MatExpr*)obj)->size();
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->size();
    }

    if( k == MATX || k == STD_ARRAY )
    {
        CV_Assert( i < 0 );
        return sz;
    }

    if( k == STD_VECTOR )
    {
        CV_Assert( i < 0 );
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;
        const std::vector<int>& iv = *(const std::vector<int>*)obj;
        size_t szb = v.size(), szi = iv.size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb/CV_ELEM_SIZE(flags)), 1);
    }

    if( k == STD_BOOL_VECTOR )
    {
        CV_Assert( i < 0 );
        const std::vector<bool>& v = *(const std::vector<bool>*)obj;
        return Size((int)v.size(), 1);
    }

    if( k == NONE )
        return Size();

    if( k == STD_VECTOR_VECTOR )
    {
        const std::vector<std::vector<uchar> >& vv = *(const std::vector<std::vector<uchar> >*)obj;
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );
        const std::vector<std::vector<int> >& ivv = *(const std::vector<std::vector<int> >*)obj;

        size_t szb = vv[i].size(), szi = ivv[i].size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb/CV_ELEM_SIZE(flags)), 1);
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );

        return vv[i].size();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return sz.height==0 ? Size() : Size(sz.height, 1);
        CV_Assert( i < sz.height );

        return vv[i].size();
    }

    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *(const std::vector<cuda::GpuMat>*)obj;
        if (i < 0)
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert(i < (int)vv.size());
        return vv[i].size();
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );

        return vv[i].size();
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        const ogl::Buffer* buf = (const ogl::Buffer*)obj;
        return buf->size();
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        const cuda::GpuMat* d_mat = (const cuda::GpuMat*)obj;
        return d_mat->size();
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );
        const cuda::HostMem* cuda_mem = (const cuda::HostMem*)obj;
        return cuda_mem->size();
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
    return Size();
}

int _InputArray::sizend(int* arrsz, int i) const
{
    int j, d=0, k = kind();

    if( k == NONE )
        ;
    else if( k == MAT )
    {
        CV_Assert( i < 0 );
        const Mat& m = *(const Mat*)obj;
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == UMAT )
    {
        CV_Assert( i < 0 );
        const UMat& m = *(const UMat*)obj;
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == STD_VECTOR_MAT && i >= 0 )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        CV_Assert( i < (int)vv.size() );
        const Mat& m = vv[i];
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == STD_ARRAY_MAT && i >= 0 )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert( i < sz.height );
        const Mat& m = vv[i];
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == STD_VECTOR_UMAT && i >= 0 )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        CV_Assert( i < (int)vv.size() );
        const UMat& m = vv[i];
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else
    {
        Size sz2d = size(i);
        d = 2;
        if(arrsz)
        {
            arrsz[0] = sz2d.height;
            arrsz[1] = sz2d.width;
        }
    }

    return d;
}

bool _InputArray::sameSize(const _InputArray& arr) const
{
    int k1 = kind(), k2 = arr.kind();
    Size sz1;

    if( k1 == MAT )
    {
        const Mat* m = ((const Mat*)obj);
        if( k2 == MAT )
            return m->size == ((const Mat*)arr.obj)->size;
        if( k2 == UMAT )
            return m->size == ((const UMat*)arr.obj)->size;
        if( m->dims > 2 )
            return false;
        sz1 = m->size();
    }
    else if( k1 == UMAT )
    {
        const UMat* m = ((const UMat*)obj);
        if( k2 == MAT )
            return m->size == ((const Mat*)arr.obj)->size;
        if( k2 == UMAT )
            return m->size == ((const UMat*)arr.obj)->size;
        if( m->dims > 2 )
            return false;
        sz1 = m->size();
    }
    else
        sz1 = size();
    if( arr.dims() > 2 )
        return false;
    return sz1 == arr.size();
}

int _InputArray::dims(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->dims;
    }

    if( k == EXPR )
    {
        CV_Assert( i < 0 );
        return ((const MatExpr*)obj)->a.dims;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->dims;
    }

    if( k == MATX || k == STD_ARRAY )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == STD_VECTOR || k == STD_BOOL_VECTOR )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == NONE )
        return 0;

    if( k == STD_VECTOR_VECTOR )
    {
        const std::vector<std::vector<uchar> >& vv = *(const std::vector<std::vector<uchar> >*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < (int)vv.size() );
        return 2;
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < (int)vv.size() );

        return vv[i].dims;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < sz.height );

        return vv[i].dims;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < (int)vv.size() );

        return vv[i].dims;
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
    return 0;
}

size_t _InputArray::total(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->total();
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->total();
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if( i < 0 )
            return vv.size();

        CV_Assert( i < (int)vv.size() );
        return vv[i].total();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return sz.height;

        CV_Assert( i < sz.height );
        return vv[i].total();
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        if( i < 0 )
            return vv.size();

        CV_Assert( i < (int)vv.size() );
        return vv[i].total();
    }

    return size(i).area();
}

int _InputArray::type(int i) const
{
    int k = kind();

    if( k == MAT )
        return ((const Mat*)obj)->type();

    if( k == UMAT )
        return ((const UMat*)obj)->type();

    if( k == EXPR )
        return ((const MatExpr*)obj)->type();

    if( k == MATX || k == STD_VECTOR || k == STD_ARRAY || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return CV_MAT_TYPE(flags);

    if( k == NONE )
        return -1;

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        if( vv.empty() )
        {
            CV_Assert((flags & FIXED_TYPE) != 0);
            return CV_MAT_TYPE(flags);
        }
        CV_Assert( i < (int)vv.size() );
        return vv[i >= 0 ? i : 0].type();
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if( vv.empty() )
        {
            CV_Assert((flags & FIXED_TYPE) != 0);
            return CV_MAT_TYPE(flags);
        }
        CV_Assert( i < (int)vv.size() );
        return vv[i >= 0 ? i : 0].type();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( sz.height == 0 )
        {
            CV_Assert((flags & FIXED_TYPE) != 0);
            return CV_MAT_TYPE(flags);
        }
        CV_Assert( i < sz.height );
        return vv[i >= 0 ? i : 0].type();
    }

    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *(const std::vector<cuda::GpuMat>*)obj;
        if (vv.empty())
        {
            CV_Assert((flags & FIXED_TYPE) != 0);
            return CV_MAT_TYPE(flags);
        }
        CV_Assert(i < (int)vv.size());
        return vv[i >= 0 ? i : 0].type();
    }

    if( k == OPENGL_BUFFER )
        return ((const ogl::Buffer*)obj)->type();

    if( k == CUDA_GPU_MAT )
        return ((const cuda::GpuMat*)obj)->type();

    if( k == CUDA_HOST_MEM )
        return ((const cuda::HostMem*)obj)->type();

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
    return 0;
}

int _InputArray::depth(int i) const
{
    return CV_MAT_DEPTH(type(i));
}

int _InputArray::channels(int i) const
{
    return CV_MAT_CN(type(i));
}

bool _InputArray::empty() const
{
    int k = kind();

    if( k == MAT )
        return ((const Mat*)obj)->empty();

    if( k == UMAT )
        return ((const UMat*)obj)->empty();

    if( k == EXPR )
        return false;

    if( k == MATX || k == STD_ARRAY )
        return false;

    if( k == STD_VECTOR )
    {
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;
        return v.empty();
    }

    if( k == STD_BOOL_VECTOR )
    {
        const std::vector<bool>& v = *(const std::vector<bool>*)obj;
        return v.empty();
    }

    if( k == NONE )
        return true;

    if( k == STD_VECTOR_VECTOR )
    {
        const std::vector<std::vector<uchar> >& vv = *(const std::vector<std::vector<uchar> >*)obj;
        return vv.empty();
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        return vv.empty();
    }

    if( k == STD_ARRAY_MAT )
    {
        return sz.height == 0;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        return vv.empty();
    }

    if( k == OPENGL_BUFFER )
        return ((const ogl::Buffer*)obj)->empty();

    if( k == CUDA_GPU_MAT )
        return ((const cuda::GpuMat*)obj)->empty();

    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *(const std::vector<cuda::GpuMat>*)obj;
        return vv.empty();
    }

    if( k == CUDA_HOST_MEM )
        return ((const cuda::HostMem*)obj)->empty();

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
    return true;
}

bool _InputArray::isContinuous(int i) const
{
    int k = kind();

    if( k == MAT )
        return i < 0 ? ((const Mat*)obj)->isContinuous() : true;

    if( k == UMAT )
        return i < 0 ? ((const UMat*)obj)->isContinuous() : true;

    if( k == EXPR || k == MATX || k == STD_VECTOR || k == STD_ARRAY ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return true;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return vv[i].isContinuous();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert(i < sz.height);
        return vv[i].isContinuous();
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return vv[i].isContinuous();
    }

    if( k == CUDA_GPU_MAT )
      return i < 0 ? ((const cuda::GpuMat*)obj)->isContinuous() : true;

    CV_Error(CV_StsNotImplemented, "Unknown/unsupported array type");
    return false;
}

bool _InputArray::isSubmatrix(int i) const
{
    int k = kind();

    if( k == MAT )
        return i < 0 ? ((const Mat*)obj)->isSubmatrix() : false;

    if( k == UMAT )
        return i < 0 ? ((const UMat*)obj)->isSubmatrix() : false;

    if( k == EXPR || k == MATX || k == STD_VECTOR || k == STD_ARRAY ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return false;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return vv[i].isSubmatrix();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert(i < sz.height);
        return vv[i].isSubmatrix();
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return vv[i].isSubmatrix();
    }

    CV_Error(CV_StsNotImplemented, "");
    return false;
}

size_t _InputArray::offset(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        const Mat * const m = ((const Mat*)obj);
        return (size_t)(m->ptr() - m->datastart);
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->offset;
    }

    if( k == EXPR || k == MATX || k == STD_VECTOR || k == STD_ARRAY ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return 0;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < (int)vv.size() );

        return (size_t)(vv[i].ptr() - vv[i].datastart);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < sz.height );
        return (size_t)(vv[i].ptr() - vv[i].datastart);
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return vv[i].offset;
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        const cuda::GpuMat * const m = ((const cuda::GpuMat*)obj);
        return (size_t)(m->data - m->datastart);
    }

    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *(const std::vector<cuda::GpuMat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return (size_t)(vv[i].data - vv[i].datastart);
    }

    CV_Error(Error::StsNotImplemented, "");
    return 0;
}

size_t _InputArray::step(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->step;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->step;
    }

    if( k == EXPR || k == MATX || k == STD_VECTOR || k == STD_ARRAY ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return 0;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < (int)vv.size() );
        return vv[i].step;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < sz.height );
        return vv[i].step;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *(const std::vector<UMat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return vv[i].step;
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        return ((const cuda::GpuMat*)obj)->step;
    }
    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *(const std::vector<cuda::GpuMat>*)obj;
        CV_Assert((size_t)i < vv.size());
        return vv[i].step;
    }

    CV_Error(Error::StsNotImplemented, "");
    return 0;
}

void _InputArray::copyTo(const _OutputArray& arr) const
{
    int k = kind();

    if( k == NONE )
        arr.release();
    else if( k == MAT || k == MATX || k == STD_VECTOR || k == STD_ARRAY || k == STD_BOOL_VECTOR )
    {
        Mat m = getMat();
        m.copyTo(arr);
    }
    else if( k == EXPR )
    {
        const MatExpr& e = *((MatExpr*)obj);
        if( arr.kind() == MAT )
            arr.getMatRef() = e;
        else
            Mat(e).copyTo(arr);
    }
    else if( k == UMAT )
        ((UMat*)obj)->copyTo(arr);
    else
        CV_Error(Error::StsNotImplemented, "");
}

void _InputArray::copyTo(const _OutputArray& arr, const _InputArray & mask) const
{
    int k = kind();

    if( k == NONE )
        arr.release();
    else if( k == MAT || k == MATX || k == STD_VECTOR || k == STD_ARRAY || k == STD_BOOL_VECTOR )
    {
        Mat m = getMat();
        m.copyTo(arr, mask);
    }
    else if( k == UMAT )
        ((UMat*)obj)->copyTo(arr, mask);
    else
        CV_Error(Error::StsNotImplemented, "");
}

bool _OutputArray::fixedSize() const
{
    return (flags & FIXED_SIZE) == FIXED_SIZE;
}

bool _OutputArray::fixedType() const
{
    return (flags & FIXED_TYPE) == FIXED_TYPE;
}

void _OutputArray::create(Size _sz, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((Mat*)obj)->size.operator()() == _sz);
        CV_Assert(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(_sz, mtype);
        return;
    }
    if( k == UMAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((UMat*)obj)->size.operator()() == _sz);
        CV_Assert(!fixedType() || ((UMat*)obj)->type() == mtype);
        ((UMat*)obj)->create(_sz, mtype);
        return;
    }
    if( k == CUDA_GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::GpuMat*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((cuda::GpuMat*)obj)->type() == mtype);
        ((cuda::GpuMat*)obj)->create(_sz, mtype);
        return;
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((ogl::Buffer*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((ogl::Buffer*)obj)->type() == mtype);
        ((ogl::Buffer*)obj)->create(_sz, mtype);
        return;
    }
    if( k == CUDA_HOST_MEM && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::HostMem*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((cuda::HostMem*)obj)->type() == mtype);
        ((cuda::HostMem*)obj)->create(_sz, mtype);
        return;
    }
    int sizes[] = {_sz.height, _sz.width};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int _rows, int _cols, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((Mat*)obj)->size.operator()() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(_rows, _cols, mtype);
        return;
    }
    if( k == UMAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((UMat*)obj)->size.operator()() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((UMat*)obj)->type() == mtype);
        ((UMat*)obj)->create(_rows, _cols, mtype);
        return;
    }
    if( k == CUDA_GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::GpuMat*)obj)->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((cuda::GpuMat*)obj)->type() == mtype);
        ((cuda::GpuMat*)obj)->create(_rows, _cols, mtype);
        return;
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((ogl::Buffer*)obj)->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((ogl::Buffer*)obj)->type() == mtype);
        ((ogl::Buffer*)obj)->create(_rows, _cols, mtype);
        return;
    }
    if( k == CUDA_HOST_MEM && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::HostMem*)obj)->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((cuda::HostMem*)obj)->type() == mtype);
        ((cuda::HostMem*)obj)->create(_rows, _cols, mtype);
        return;
    }
    int sizes[] = {_rows, _cols};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int d, const int* sizes, int mtype, int i,
                          bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    mtype = CV_MAT_TYPE(mtype);

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        Mat& m = *(Mat*)obj;
        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( d == 2 && m.dims == 2 && m.data &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }
        if(fixedSize())
        {
            CV_Assert(m.dims == d);
            for(int j = 0; j < d; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }
        m.create(d, sizes, mtype);
        return;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        UMat& m = *(UMat*)obj;
        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( d == 2 && m.dims == 2 && !m.empty() &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }
        if(fixedSize())
        {
            CV_Assert(m.dims == d);
            for(int j = 0; j < d; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }
        m.create(d, sizes, mtype);
        return;
    }

    if( k == MATX )
    {
        CV_Assert( i < 0 );
        int type0 = CV_MAT_TYPE(flags);
        CV_Assert( mtype == type0 || (CV_MAT_CN(mtype) == 1 && ((1 << type0) & fixedDepthMask) != 0) );
        CV_Assert( d == 2 && ((sizes[0] == sz.height && sizes[1] == sz.width) ||
                                 (allowTransposed && sizes[0] == sz.width && sizes[1] == sz.height)));
        return;
    }

    if( k == STD_ARRAY )
    {
        int type0 = CV_MAT_TYPE(flags);
        CV_Assert( mtype == type0 || (CV_MAT_CN(mtype) == 1 && ((1 << type0) & fixedDepthMask) != 0) );
        CV_Assert( d == 2 && sz.area() == sizes[0]*sizes[1]);
        return;
    }

    if( k == STD_VECTOR || k == STD_VECTOR_VECTOR )
    {
        CV_Assert( d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0]*sizes[1] == 0) );
        size_t len = sizes[0]*sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0;
        std::vector<uchar>* v = (std::vector<uchar>*)obj;

        if( k == STD_VECTOR_VECTOR )
        {
            std::vector<std::vector<uchar> >& vv = *(std::vector<std::vector<uchar> >*)obj;
            if( i < 0 )
            {
                CV_Assert(!fixedSize() || len == vv.size());
                vv.resize(len);
                return;
            }
            CV_Assert( i < (int)vv.size() );
            v = &vv[i];
        }
        else
            CV_Assert( i < 0 );

        int type0 = CV_MAT_TYPE(flags);
        CV_Assert( mtype == type0 || (CV_MAT_CN(mtype) == CV_MAT_CN(type0) && ((1 << type0) & fixedDepthMask) != 0) );

        int esz = CV_ELEM_SIZE(type0);
        CV_Assert(!fixedSize() || len == ((std::vector<uchar>*)v)->size() / esz);
        switch( esz )
        {
        case 1:
            ((std::vector<uchar>*)v)->resize(len);
            break;
        case 2:
            ((std::vector<Vec2b>*)v)->resize(len);
            break;
        case 3:
            ((std::vector<Vec3b>*)v)->resize(len);
            break;
        case 4:
            ((std::vector<int>*)v)->resize(len);
            break;
        case 6:
            ((std::vector<Vec3s>*)v)->resize(len);
            break;
        case 8:
            ((std::vector<Vec2i>*)v)->resize(len);
            break;
        case 12:
            ((std::vector<Vec3i>*)v)->resize(len);
            break;
        case 16:
            ((std::vector<Vec4i>*)v)->resize(len);
            break;
        case 24:
            ((std::vector<Vec6i>*)v)->resize(len);
            break;
        case 32:
            ((std::vector<Vec8i>*)v)->resize(len);
            break;
        case 36:
            ((std::vector<Vec<int, 9> >*)v)->resize(len);
            break;
        case 48:
            ((std::vector<Vec<int, 12> >*)v)->resize(len);
            break;
        case 64:
            ((std::vector<Vec<int, 16> >*)v)->resize(len);
            break;
        case 128:
            ((std::vector<Vec<int, 32> >*)v)->resize(len);
            break;
        case 256:
            ((std::vector<Vec<int, 64> >*)v)->resize(len);
            break;
        case 512:
            ((std::vector<Vec<int, 128> >*)v)->resize(len);
            break;
        default:
            CV_Error_(CV_StsBadArg, ("Vectors with element size %d are not supported. Please, modify OutputArray::create()\n", esz));
        }
        return;
    }

    if( k == NONE )
    {
        CV_Error(CV_StsNullPtr, "create() called for the missing output array" );
        return;
    }

    if( k == STD_VECTOR_MAT )
    {
        std::vector<Mat>& v = *(std::vector<Mat>*)obj;

        if( i < 0 )
        {
            CV_Assert( d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0]*sizes[1] == 0) );
            size_t len = sizes[0]*sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0, len0 = v.size();

            CV_Assert(!fixedSize() || len == len0);
            v.resize(len);
            if( fixedType() )
            {
                int _type = CV_MAT_TYPE(flags);
                for( size_t j = len0; j < len; j++ )
                {
                    if( v[j].type() == _type )
                        continue;
                    CV_Assert( v[j].empty() );
                    v[j].flags = (v[j].flags & ~CV_MAT_TYPE_MASK) | _type;
                }
            }
            return;
        }

        CV_Assert( i < (int)v.size() );
        Mat& m = v[i];

        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( d == 2 && m.dims == 2 && m.data &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }
        if(fixedSize())
        {
            CV_Assert(m.dims == d);
            for(int j = 0; j < d; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }

        m.create(d, sizes, mtype);
        return;
    }

    if( k == STD_ARRAY_MAT )
    {
        Mat* v = (Mat*)obj;

        if( i < 0 )
        {
            CV_Assert( d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0]*sizes[1] == 0) );
            size_t len = sizes[0]*sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0, len0 = sz.height;

            CV_Assert(len == len0);
            if( fixedType() )
            {
                int _type = CV_MAT_TYPE(flags);
                for( size_t j = len0; j < len; j++ )
                {
                    if( v[j].type() == _type )
                        continue;
                    CV_Assert( v[j].empty() );
                    v[j].flags = (v[j].flags & ~CV_MAT_TYPE_MASK) | _type;
                }
            }
            return;
        }

        CV_Assert( i < sz.height );
        Mat& m = v[i];

        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( d == 2 && m.dims == 2 && m.data &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }

        if(fixedSize())
        {
            CV_Assert(m.dims == d);
            for(int j = 0; j < d; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }

        m.create(d, sizes, mtype);
        return;
    }

    if( k == STD_VECTOR_UMAT )
    {
        std::vector<UMat>& v = *(std::vector<UMat>*)obj;

        if( i < 0 )
        {
            CV_Assert( d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0]*sizes[1] == 0) );
            size_t len = sizes[0]*sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0, len0 = v.size();

            CV_Assert(!fixedSize() || len == len0);
            v.resize(len);
            if( fixedType() )
            {
                int _type = CV_MAT_TYPE(flags);
                for( size_t j = len0; j < len; j++ )
                {
                    if( v[j].type() == _type )
                        continue;
                    CV_Assert( v[j].empty() );
                    v[j].flags = (v[j].flags & ~CV_MAT_TYPE_MASK) | _type;
                }
            }
            return;
        }

        CV_Assert( i < (int)v.size() );
        UMat& m = v[i];

        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( d == 2 && m.dims == 2 && m.u &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }
        if(fixedSize())
        {
            CV_Assert(m.dims == d);
            for(int j = 0; j < d; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }

        m.create(d, sizes, mtype);
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _OutputArray::createSameSize(const _InputArray& arr, int mtype) const
{
    int arrsz[CV_MAX_DIM], d = arr.sizend(arrsz);
    create(d, arrsz, mtype);
}

void _OutputArray::release() const
{
    CV_Assert(!fixedSize());

    int k = kind();

    if( k == MAT )
    {
        ((Mat*)obj)->release();
        return;
    }

    if( k == UMAT )
    {
        ((UMat*)obj)->release();
        return;
    }

    if( k == CUDA_GPU_MAT )
    {
        ((cuda::GpuMat*)obj)->release();
        return;
    }

    if( k == CUDA_HOST_MEM )
    {
        ((cuda::HostMem*)obj)->release();
        return;
    }

    if( k == OPENGL_BUFFER )
    {
        ((ogl::Buffer*)obj)->release();
        return;
    }

    if( k == NONE )
        return;

    if( k == STD_VECTOR )
    {
        create(Size(), CV_MAT_TYPE(flags));
        return;
    }

    if( k == STD_VECTOR_VECTOR )
    {
        ((std::vector<std::vector<uchar> >*)obj)->clear();
        return;
    }

    if( k == STD_VECTOR_MAT )
    {
        ((std::vector<Mat>*)obj)->clear();
        return;
    }

    if( k == STD_VECTOR_UMAT )
    {
        ((std::vector<UMat>*)obj)->clear();
        return;
    }
    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        ((std::vector<cuda::GpuMat>*)obj)->clear();
        return;
    }
    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _OutputArray::clear() const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert(!fixedSize());
        ((Mat*)obj)->resize(0);
        return;
    }

    release();
}

bool _OutputArray::needed() const
{
    return kind() != NONE;
}

Mat& _OutputArray::getMatRef(int i) const
{
    int k = kind();
    if( i < 0 )
    {
        CV_Assert( k == MAT );
        return *(Mat*)obj;
    }

    CV_Assert( k == STD_VECTOR_MAT || k == STD_ARRAY_MAT );

    if( k == STD_VECTOR_MAT )
    {
        std::vector<Mat>& v = *(std::vector<Mat>*)obj;
        CV_Assert( i < (int)v.size() );
        return v[i];
    }
    else
    {
        Mat* v = (Mat*)obj;
        CV_Assert( 0 <= i && i < sz.height );
        return v[i];
    }
}

UMat& _OutputArray::getUMatRef(int i) const
{
    int k = kind();
    if( i < 0 )
    {
        CV_Assert( k == UMAT );
        return *(UMat*)obj;
    }
    else
    {
        CV_Assert( k == STD_VECTOR_UMAT );
        std::vector<UMat>& v = *(std::vector<UMat>*)obj;
        CV_Assert( i < (int)v.size() );
        return v[i];
    }
}

cuda::GpuMat& _OutputArray::getGpuMatRef() const
{
    int k = kind();
    CV_Assert( k == CUDA_GPU_MAT );
    return *(cuda::GpuMat*)obj;
}
std::vector<cuda::GpuMat>& _OutputArray::getGpuMatVecRef() const
{
    int k = kind();
    CV_Assert(k == STD_VECTOR_CUDA_GPU_MAT);
    return *(std::vector<cuda::GpuMat>*)obj;
}

ogl::Buffer& _OutputArray::getOGlBufferRef() const
{
    int k = kind();
    CV_Assert( k == OPENGL_BUFFER );
    return *(ogl::Buffer*)obj;
}

cuda::HostMem& _OutputArray::getHostMemRef() const
{
    int k = kind();
    CV_Assert( k == CUDA_HOST_MEM );
    return *(cuda::HostMem*)obj;
}

void _OutputArray::setTo(const _InputArray& arr, const _InputArray & mask) const
{
    int k = kind();

    if( k == NONE )
        ;
    else if( k == MAT || k == MATX || k == STD_VECTOR || k == STD_ARRAY )
    {
        Mat m = getMat();
        m.setTo(arr, mask);
    }
    else if( k == UMAT )
        ((UMat*)obj)->setTo(arr, mask);
    else if( k == CUDA_GPU_MAT )
    {
        Mat value = arr.getMat();
        CV_Assert( checkScalar(value, type(), arr.kind(), _InputArray::CUDA_GPU_MAT) );
        ((cuda::GpuMat*)obj)->setTo(Scalar(Vec<double, 4>(value.ptr<double>())), mask);
    }
    else
        CV_Error(Error::StsNotImplemented, "");
}


void _OutputArray::assign(const UMat& u) const
{
    int k = kind();
    if (k == UMAT)
    {
        *(UMat*)obj = u;
    }
    else if (k == MAT)
    {
        u.copyTo(*(Mat*)obj); // TODO check u.getMat()
    }
    else if (k == MATX)
    {
        u.copyTo(getMat()); // TODO check u.getMat()
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::assign(const Mat& m) const
{
    int k = kind();
    if (k == UMAT)
    {
        m.copyTo(*(UMat*)obj); // TODO check m.getUMat()
    }
    else if (k == MAT)
    {
        *(Mat*)obj = m;
    }
    else if (k == MATX)
    {
        m.copyTo(getMat());
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


static _InputOutputArray _none;
InputOutputArray noArray() { return _none; }

}

/*************************************************************************************************\
                                        Matrix Operations
\*************************************************************************************************/

void cv::hconcat(const Mat* src, size_t nsrc, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    if( nsrc == 0 || !src )
    {
        _dst.release();
        return;
    }

    int totalCols = 0, cols = 0;
    for( size_t i = 0; i < nsrc; i++ )
    {
        CV_Assert( src[i].dims <= 2 &&
                   src[i].rows == src[0].rows &&
                   src[i].type() == src[0].type());
        totalCols += src[i].cols;
    }
    _dst.create( src[0].rows, totalCols, src[0].type());
    Mat dst = _dst.getMat();
    for( size_t i = 0; i < nsrc; i++ )
    {
        Mat dpart = dst(Rect(cols, 0, src[i].cols, src[i].rows));
        src[i].copyTo(dpart);
        cols += src[i].cols;
    }
}

void cv::hconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    CV_INSTRUMENT_REGION()

    Mat src[] = {src1.getMat(), src2.getMat()};
    hconcat(src, 2, dst);
}

void cv::hconcat(InputArray _src, OutputArray dst)
{
    CV_INSTRUMENT_REGION()

    std::vector<Mat> src;
    _src.getMatVector(src);
    hconcat(!src.empty() ? &src[0] : 0, src.size(), dst);
}

void cv::vconcat(const Mat* src, size_t nsrc, OutputArray _dst)
{
    CV_TRACE_FUNCTION_SKIP_NESTED()

    if( nsrc == 0 || !src )
    {
        _dst.release();
        return;
    }

    int totalRows = 0, rows = 0;
    for( size_t i = 0; i < nsrc; i++ )
    {
        CV_Assert(src[i].dims <= 2 &&
                  src[i].cols == src[0].cols &&
                  src[i].type() == src[0].type());
        totalRows += src[i].rows;
    }
    _dst.create( totalRows, src[0].cols, src[0].type());
    Mat dst = _dst.getMat();
    for( size_t i = 0; i < nsrc; i++ )
    {
        Mat dpart(dst, Rect(0, rows, src[i].cols, src[i].rows));
        src[i].copyTo(dpart);
        rows += src[i].rows;
    }
}

void cv::vconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    CV_INSTRUMENT_REGION()

    Mat src[] = {src1.getMat(), src2.getMat()};
    vconcat(src, 2, dst);
}

void cv::vconcat(InputArray _src, OutputArray dst)
{
    CV_INSTRUMENT_REGION()

    std::vector<Mat> src;
    _src.getMatVector(src);
    vconcat(!src.empty() ? &src[0] : 0, src.size(), dst);
}

//////////////////////////////////////// set identity ////////////////////////////////////////////

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_setIdentity( InputOutputArray _m, const Scalar& s )
{
    int type = _m.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type), kercn = cn, rowsPerWI = 1;
    int sctype = CV_MAKE_TYPE(depth, cn == 3 ? 4 : cn);
    if (ocl::Device::getDefault().isIntel())
    {
        rowsPerWI = 4;
        if (cn == 1)
        {
            kercn = std::min(ocl::predictOptimalVectorWidth(_m), 4);
            if (kercn != 4)
                kercn = 1;
        }
    }

    ocl::Kernel k("setIdentity", ocl::core::set_identity_oclsrc,
                  format("-D T=%s -D T1=%s -D cn=%d -D ST=%s -D kercn=%d -D rowsPerWI=%d",
                         ocl::memopTypeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::memopTypeToStr(depth), cn,
                         ocl::memopTypeToStr(sctype),
                         kercn, rowsPerWI));
    if (k.empty())
        return false;

    UMat m = _m.getUMat();
    k.args(ocl::KernelArg::WriteOnly(m, cn, kercn),
           ocl::KernelArg::Constant(Mat(1, 1, sctype, s)));

    size_t globalsize[2] = { (size_t)m.cols * cn / kercn, ((size_t)m.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::setIdentity( InputOutputArray _m, const Scalar& s )
{
    CV_INSTRUMENT_REGION()

    CV_Assert( _m.dims() <= 2 );

    CV_OCL_RUN(_m.isUMat(),
               ocl_setIdentity(_m, s))

    Mat m = _m.getMat();
    int rows = m.rows, cols = m.cols, type = m.type();

    if( type == CV_32FC1 )
    {
        float* data = m.ptr<float>();
        float val = (float)s[0];
        size_t step = m.step/sizeof(data[0]);

        for( int i = 0; i < rows; i++, data += step )
        {
            for( int j = 0; j < cols; j++ )
                data[j] = 0;
            if( i < cols )
                data[i] = val;
        }
    }
    else if( type == CV_64FC1 )
    {
        double* data = m.ptr<double>();
        double val = s[0];
        size_t step = m.step/sizeof(data[0]);

        for( int i = 0; i < rows; i++, data += step )
        {
            for( int j = 0; j < cols; j++ )
                data[j] = j == i ? val : 0;
        }
    }
    else
    {
        m = Scalar(0);
        m.diag() = s;
    }
}

//////////////////////////////////////////// trace ///////////////////////////////////////////

cv::Scalar cv::trace( InputArray _m )
{
    CV_INSTRUMENT_REGION()

    Mat m = _m.getMat();
    CV_Assert( m.dims <= 2 );
    int type = m.type();
    int nm = std::min(m.rows, m.cols);

    if( type == CV_32FC1 )
    {
        const float* ptr = m.ptr<float>();
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( int i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }

    if( type == CV_64FC1 )
    {
        const double* ptr = m.ptr<double>();
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( int i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }

    return cv::sum(m.diag());
}

////////////////////////////////////// transpose /////////////////////////////////////////

namespace cv
{

template<typename T> static void
transpose_( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz )
{
    int i=0, j, m = sz.width, n = sz.height;

    #if CV_ENABLE_UNROLLED
    for(; i <= m - 4; i += 4 )
    {
        T* d0 = (T*)(dst + dstep*i);
        T* d1 = (T*)(dst + dstep*(i+1));
        T* d2 = (T*)(dst + dstep*(i+2));
        T* d3 = (T*)(dst + dstep*(i+3));

        for( j = 0; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
            d1[j] = s0[1]; d1[j+1] = s1[1]; d1[j+2] = s2[1]; d1[j+3] = s3[1];
            d2[j] = s0[2]; d2[j+1] = s1[2]; d2[j+2] = s2[2]; d2[j+3] = s3[2];
            d3[j] = s0[3]; d3[j+1] = s1[3]; d3[j+2] = s2[3]; d3[j+3] = s3[3];
        }

        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0]; d1[j] = s0[1]; d2[j] = s0[2]; d3[j] = s0[3];
        }
    }
    #endif
    for( ; i < m; i++ )
    {
        T* d0 = (T*)(dst + dstep*i);
        j = 0;
        #if CV_ENABLE_UNROLLED
        for(; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
        }
        #endif
        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0];
        }
    }
}

template<typename T> static void
transposeI_( uchar* data, size_t step, int n )
{
    for( int i = 0; i < n; i++ )
    {
        T* row = (T*)(data + step*i);
        uchar* data1 = data + i*sizeof(T);
        for( int j = i+1; j < n; j++ )
            std::swap( row[j], *(T*)(data1 + step*j) );
    }
}

typedef void (*TransposeFunc)( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz );
typedef void (*TransposeInplaceFunc)( uchar* data, size_t step, int n );

#define DEF_TRANSPOSE_FUNC(suffix, type) \
static void transpose_##suffix( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz ) \
{ transpose_<type>(src, sstep, dst, dstep, sz); } \
\
static void transposeI_##suffix( uchar* data, size_t step, int n ) \
{ transposeI_<type>(data, step, n); }

DEF_TRANSPOSE_FUNC(8u, uchar)
DEF_TRANSPOSE_FUNC(16u, ushort)
DEF_TRANSPOSE_FUNC(8uC3, Vec3b)
DEF_TRANSPOSE_FUNC(32s, int)
DEF_TRANSPOSE_FUNC(16uC3, Vec3s)
DEF_TRANSPOSE_FUNC(32sC2, Vec2i)
DEF_TRANSPOSE_FUNC(32sC3, Vec3i)
DEF_TRANSPOSE_FUNC(32sC4, Vec4i)
DEF_TRANSPOSE_FUNC(32sC6, Vec6i)
DEF_TRANSPOSE_FUNC(32sC8, Vec8i)

static TransposeFunc transposeTab[] =
{
    0, transpose_8u, transpose_16u, transpose_8uC3, transpose_32s, 0, transpose_16uC3, 0,
    transpose_32sC2, 0, 0, 0, transpose_32sC3, 0, 0, 0, transpose_32sC4,
    0, 0, 0, 0, 0, 0, 0, transpose_32sC6, 0, 0, 0, 0, 0, 0, 0, transpose_32sC8
};

static TransposeInplaceFunc transposeInplaceTab[] =
{
    0, transposeI_8u, transposeI_16u, transposeI_8uC3, transposeI_32s, 0, transposeI_16uC3, 0,
    transposeI_32sC2, 0, 0, 0, transposeI_32sC3, 0, 0, 0, transposeI_32sC4,
    0, 0, 0, 0, 0, 0, 0, transposeI_32sC6, 0, 0, 0, 0, 0, 0, 0, transposeI_32sC8
};

#ifdef HAVE_OPENCL

static bool ocl_transpose( InputArray _src, OutputArray _dst )
{
    const ocl::Device & dev = ocl::Device::getDefault();
    const int TILE_DIM = 32, BLOCK_ROWS = 8;
    int type = _src.type(), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type),
        rowsPerWI = dev.isIntel() ? 4 : 1;

    UMat src = _src.getUMat();
    _dst.create(src.cols, src.rows, type);
    UMat dst = _dst.getUMat();

    String kernelName("transpose");
    bool inplace = dst.u == src.u;

    if (inplace)
    {
        CV_Assert(dst.cols == dst.rows);
        kernelName += "_inplace";
    }
    else
    {
        // check required local memory size
        size_t required_local_memory = (size_t) TILE_DIM*(TILE_DIM+1)*CV_ELEM_SIZE(type);
        if (required_local_memory > ocl::Device::getDefault().localMemSize())
            return false;
    }

    ocl::Kernel k(kernelName.c_str(), ocl::core::transpose_oclsrc,
                  format("-D T=%s -D T1=%s -D cn=%d -D TILE_DIM=%d -D BLOCK_ROWS=%d -D rowsPerWI=%d%s",
                         ocl::memopTypeToStr(type), ocl::memopTypeToStr(depth),
                         cn, TILE_DIM, BLOCK_ROWS, rowsPerWI, inplace ? " -D INPLACE" : ""));
    if (k.empty())
        return false;

    if (inplace)
        k.args(ocl::KernelArg::ReadWriteNoSize(dst), dst.rows);
    else
        k.args(ocl::KernelArg::ReadOnly(src),
               ocl::KernelArg::WriteOnlyNoSize(dst));

    size_t localsize[2]  = { TILE_DIM, BLOCK_ROWS };
    size_t globalsize[2] = { (size_t)src.cols, inplace ? ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI : (divUp((size_t)src.rows, TILE_DIM) * BLOCK_ROWS) };

    if (inplace && dev.isIntel())
    {
        localsize[0] = 16;
        localsize[1] = dev.maxWorkGroupSize() / localsize[0];
    }

    return k.run(2, globalsize, localsize, false);
}

#endif

#ifdef HAVE_IPP
static bool ipp_transpose( Mat &src, Mat &dst )
{
    CV_INSTRUMENT_REGION_IPP()

    int type = src.type();
    typedef IppStatus (CV_STDCALL * IppiTranspose)(const void * pSrc, int srcStep, void * pDst, int dstStep, IppiSize roiSize);
    typedef IppStatus (CV_STDCALL * IppiTransposeI)(const void * pSrcDst, int srcDstStep, IppiSize roiSize);
    IppiTranspose ippiTranspose = 0;
    IppiTransposeI ippiTranspose_I = 0;

    if (dst.data == src.data && dst.cols == dst.rows)
    {
        CV_SUPPRESS_DEPRECATED_START
        ippiTranspose_I =
            type == CV_8UC1 ? (IppiTransposeI)ippiTranspose_8u_C1IR :
            type == CV_8UC3 ? (IppiTransposeI)ippiTranspose_8u_C3IR :
            type == CV_8UC4 ? (IppiTransposeI)ippiTranspose_8u_C4IR :
            type == CV_16UC1 ? (IppiTransposeI)ippiTranspose_16u_C1IR :
            type == CV_16UC3 ? (IppiTransposeI)ippiTranspose_16u_C3IR :
            type == CV_16UC4 ? (IppiTransposeI)ippiTranspose_16u_C4IR :
            type == CV_16SC1 ? (IppiTransposeI)ippiTranspose_16s_C1IR :
            type == CV_16SC3 ? (IppiTransposeI)ippiTranspose_16s_C3IR :
            type == CV_16SC4 ? (IppiTransposeI)ippiTranspose_16s_C4IR :
            type == CV_32SC1 ? (IppiTransposeI)ippiTranspose_32s_C1IR :
            type == CV_32SC3 ? (IppiTransposeI)ippiTranspose_32s_C3IR :
            type == CV_32SC4 ? (IppiTransposeI)ippiTranspose_32s_C4IR :
            type == CV_32FC1 ? (IppiTransposeI)ippiTranspose_32f_C1IR :
            type == CV_32FC3 ? (IppiTransposeI)ippiTranspose_32f_C3IR :
            type == CV_32FC4 ? (IppiTransposeI)ippiTranspose_32f_C4IR : 0;
        CV_SUPPRESS_DEPRECATED_END
    }
    else
    {
        ippiTranspose =
            type == CV_8UC1 ? (IppiTranspose)ippiTranspose_8u_C1R :
            type == CV_8UC3 ? (IppiTranspose)ippiTranspose_8u_C3R :
            type == CV_8UC4 ? (IppiTranspose)ippiTranspose_8u_C4R :
            type == CV_16UC1 ? (IppiTranspose)ippiTranspose_16u_C1R :
            type == CV_16UC3 ? (IppiTranspose)ippiTranspose_16u_C3R :
            type == CV_16UC4 ? (IppiTranspose)ippiTranspose_16u_C4R :
            type == CV_16SC1 ? (IppiTranspose)ippiTranspose_16s_C1R :
            type == CV_16SC3 ? (IppiTranspose)ippiTranspose_16s_C3R :
            type == CV_16SC4 ? (IppiTranspose)ippiTranspose_16s_C4R :
            type == CV_32SC1 ? (IppiTranspose)ippiTranspose_32s_C1R :
            type == CV_32SC3 ? (IppiTranspose)ippiTranspose_32s_C3R :
            type == CV_32SC4 ? (IppiTranspose)ippiTranspose_32s_C4R :
            type == CV_32FC1 ? (IppiTranspose)ippiTranspose_32f_C1R :
            type == CV_32FC3 ? (IppiTranspose)ippiTranspose_32f_C3R :
            type == CV_32FC4 ? (IppiTranspose)ippiTranspose_32f_C4R : 0;
    }

    IppiSize roiSize = { src.cols, src.rows };
    if (ippiTranspose != 0)
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiTranspose, src.ptr(), (int)src.step, dst.ptr(), (int)dst.step, roiSize) >= 0)
            return true;
    }
    else if (ippiTranspose_I != 0)
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiTranspose_I, dst.ptr(), (int)dst.step, roiSize) >= 0)
            return true;
    }
    return false;
}
#endif

}


void cv::transpose( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION()

    int type = _src.type(), esz = CV_ELEM_SIZE(type);
    CV_Assert( _src.dims() <= 2 && esz <= 32 );

    CV_OCL_RUN(_dst.isUMat(),
               ocl_transpose(_src, _dst))

    Mat src = _src.getMat();
    if( src.empty() )
    {
        _dst.release();
        return;
    }

    _dst.create(src.cols, src.rows, src.type());
    Mat dst = _dst.getMat();

    // handle the case of single-column/single-row matrices, stored in STL vectors.
    if( src.rows != dst.cols || src.cols != dst.rows )
    {
        CV_Assert( src.size() == dst.size() && (src.cols == 1 || src.rows == 1) );
        src.copyTo(dst);
        return;
    }

    CV_IPP_RUN_FAST(ipp_transpose(src, dst))

    if( dst.data == src.data )
    {
        TransposeInplaceFunc func = transposeInplaceTab[esz];
        CV_Assert( func != 0 );
        CV_Assert( dst.cols == dst.rows );
        func( dst.ptr(), dst.step, dst.rows );
    }
    else
    {
        TransposeFunc func = transposeTab[esz];
        CV_Assert( func != 0 );
        func( src.ptr(), src.step, dst.ptr(), dst.step, src.size() );
    }
}


////////////////////////////////////// completeSymm /////////////////////////////////////////

void cv::completeSymm( InputOutputArray _m, bool LtoR )
{
    CV_INSTRUMENT_REGION()

    Mat m = _m.getMat();
    size_t step = m.step, esz = m.elemSize();
    CV_Assert( m.dims <= 2 && m.rows == m.cols );

    int rows = m.rows;
    int j0 = 0, j1 = rows;

    uchar* data = m.ptr();
    for( int i = 0; i < rows; i++ )
    {
        if( !LtoR ) j1 = i; else j0 = i+1;
        for( int j = j0; j < j1; j++ )
            memcpy(data + (i*step + j*esz), data + (j*step + i*esz), esz);
    }
}


cv::Mat cv::Mat::cross(InputArray _m) const
{
    Mat m = _m.getMat();
    int tp = type(), d = CV_MAT_DEPTH(tp);
    CV_Assert( dims <= 2 && m.dims <= 2 && size() == m.size() && tp == m.type() &&
        ((rows == 3 && cols == 1) || (cols*channels() == 3 && rows == 1)));
    Mat result(rows, cols, tp);

    if( d == CV_32F )
    {
        const float *a = (const float*)data, *b = (const float*)m.data;
        float* c = (float*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }
    else if( d == CV_64F )
    {
        const double *a = (const double*)data, *b = (const double*)m.data;
        double* c = (double*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }

    return result;
}


////////////////////////////////////////// reduce ////////////////////////////////////////////

namespace cv
{

template<typename T, typename ST, class Op> static void
reduceR_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    size.width *= srcmat.channels();
    AutoBuffer<WT> buffer(size.width);
    WT* buf = buffer;
    ST* dst = dstmat.ptr<ST>();
    const T* src = srcmat.ptr<T>();
    size_t srcstep = srcmat.step/sizeof(src[0]);
    int i;
    Op op;

    for( i = 0; i < size.width; i++ )
        buf[i] = src[i];

    for( ; --size.height; )
    {
        src += srcstep;
        i = 0;
        #if CV_ENABLE_UNROLLED
        for(; i <= size.width - 4; i += 4 )
        {
            WT s0, s1;
            s0 = op(buf[i], (WT)src[i]);
            s1 = op(buf[i+1], (WT)src[i+1]);
            buf[i] = s0; buf[i+1] = s1;

            s0 = op(buf[i+2], (WT)src[i+2]);
            s1 = op(buf[i+3], (WT)src[i+3]);
            buf[i+2] = s0; buf[i+3] = s1;
        }
        #endif
        for( ; i < size.width; i++ )
            buf[i] = op(buf[i], (WT)src[i]);
    }

    for( i = 0; i < size.width; i++ )
        dst[i] = (ST)buf[i];
}


template<typename T, typename ST, class Op> static void
reduceC_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    int cn = srcmat.channels();
    size.width *= cn;
    Op op;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = srcmat.ptr<T>(y);
        ST* dst = dstmat.ptr<ST>(y);
        if( size.width == cn )
            for( int k = 0; k < cn; k++ )
                dst[k] = src[k];
        else
        {
            for( int k = 0; k < cn; k++ )
            {
                WT a0 = src[k], a1 = src[k+cn];
                int i;
                for( i = 2*cn; i <= size.width - 4*cn; i += 4*cn )
                {
                    a0 = op(a0, (WT)src[i+k]);
                    a1 = op(a1, (WT)src[i+k+cn]);
                    a0 = op(a0, (WT)src[i+k+cn*2]);
                    a1 = op(a1, (WT)src[i+k+cn*3]);
                }

                for( ; i < size.width; i += cn )
                {
                    a0 = op(a0, (WT)src[i+k]);
                }
                a0 = op(a0, a1);
                dst[k] = (ST)a0;
            }
        }
    }
}

typedef void (*ReduceFunc)( const Mat& src, Mat& dst );

}

#define reduceSumR8u32s  reduceR_<uchar, int,   OpAdd<int> >
#define reduceSumR8u32f  reduceR_<uchar, float, OpAdd<int> >
#define reduceSumR8u64f  reduceR_<uchar, double,OpAdd<int> >
#define reduceSumR16u32f reduceR_<ushort,float, OpAdd<float> >
#define reduceSumR16u64f reduceR_<ushort,double,OpAdd<double> >
#define reduceSumR16s32f reduceR_<short, float, OpAdd<float> >
#define reduceSumR16s64f reduceR_<short, double,OpAdd<double> >
#define reduceSumR32f32f reduceR_<float, float, OpAdd<float> >
#define reduceSumR32f64f reduceR_<float, double,OpAdd<double> >
#define reduceSumR64f64f reduceR_<double,double,OpAdd<double> >

#define reduceMaxR8u  reduceR_<uchar, uchar, OpMax<uchar> >
#define reduceMaxR16u reduceR_<ushort,ushort,OpMax<ushort> >
#define reduceMaxR16s reduceR_<short, short, OpMax<short> >
#define reduceMaxR32f reduceR_<float, float, OpMax<float> >
#define reduceMaxR64f reduceR_<double,double,OpMax<double> >

#define reduceMinR8u  reduceR_<uchar, uchar, OpMin<uchar> >
#define reduceMinR16u reduceR_<ushort,ushort,OpMin<ushort> >
#define reduceMinR16s reduceR_<short, short, OpMin<short> >
#define reduceMinR32f reduceR_<float, float, OpMin<float> >
#define reduceMinR64f reduceR_<double,double,OpMin<double> >

#ifdef HAVE_IPP
static inline bool ipp_reduceSumC_8u16u16s32f_64f(const cv::Mat& srcmat, cv::Mat& dstmat)
{
    int sstep = (int)srcmat.step, stype = srcmat.type(),
            ddepth = dstmat.depth();

    IppiSize roisize = { srcmat.size().width, 1 };

    typedef IppStatus (CV_STDCALL * IppiSum)(const void * pSrc, int srcStep, IppiSize roiSize, Ipp64f* pSum);
    typedef IppStatus (CV_STDCALL * IppiSumHint)(const void * pSrc, int srcStep, IppiSize roiSize, Ipp64f* pSum, IppHintAlgorithm hint);
    IppiSum ippiSum = 0;
    IppiSumHint ippiSumHint = 0;

    if(ddepth == CV_64F)
    {
        ippiSum =
            stype == CV_8UC1 ? (IppiSum)ippiSum_8u_C1R :
            stype == CV_8UC3 ? (IppiSum)ippiSum_8u_C3R :
            stype == CV_8UC4 ? (IppiSum)ippiSum_8u_C4R :
            stype == CV_16UC1 ? (IppiSum)ippiSum_16u_C1R :
            stype == CV_16UC3 ? (IppiSum)ippiSum_16u_C3R :
            stype == CV_16UC4 ? (IppiSum)ippiSum_16u_C4R :
            stype == CV_16SC1 ? (IppiSum)ippiSum_16s_C1R :
            stype == CV_16SC3 ? (IppiSum)ippiSum_16s_C3R :
            stype == CV_16SC4 ? (IppiSum)ippiSum_16s_C4R : 0;
        ippiSumHint =
            stype == CV_32FC1 ? (IppiSumHint)ippiSum_32f_C1R :
            stype == CV_32FC3 ? (IppiSumHint)ippiSum_32f_C3R :
            stype == CV_32FC4 ? (IppiSumHint)ippiSum_32f_C4R : 0;
    }

    if(ippiSum)
    {
        for(int y = 0; y < srcmat.size().height; y++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippiSum, srcmat.ptr(y), sstep, roisize, dstmat.ptr<Ipp64f>(y)) < 0)
                return false;
        }
        return true;
    }
    else if(ippiSumHint)
    {
        for(int y = 0; y < srcmat.size().height; y++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippiSumHint, srcmat.ptr(y), sstep, roisize, dstmat.ptr<Ipp64f>(y), ippAlgHintAccurate) < 0)
                return false;
        }
        return true;
    }

    return false;
}

static inline void reduceSumC_8u16u16s32f_64f(const cv::Mat& srcmat, cv::Mat& dstmat)
{
    CV_IPP_RUN_FAST(ipp_reduceSumC_8u16u16s32f_64f(srcmat, dstmat));

    cv::ReduceFunc func = 0;

    if(dstmat.depth() == CV_64F)
    {
        int sdepth = CV_MAT_DEPTH(srcmat.type());
        func =
            sdepth == CV_8U ? (cv::ReduceFunc)cv::reduceC_<uchar, double,   cv::OpAdd<double> > :
            sdepth == CV_16U ? (cv::ReduceFunc)cv::reduceC_<ushort, double,   cv::OpAdd<double> > :
            sdepth == CV_16S ? (cv::ReduceFunc)cv::reduceC_<short, double,   cv::OpAdd<double> > :
            sdepth == CV_32F ? (cv::ReduceFunc)cv::reduceC_<float, double,   cv::OpAdd<double> > : 0;
    }
    CV_Assert(func);

    func(srcmat, dstmat);
}

#endif

#define reduceSumC8u32s  reduceC_<uchar, int,   OpAdd<int> >
#define reduceSumC8u32f  reduceC_<uchar, float, OpAdd<int> >
#define reduceSumC16u32f reduceC_<ushort,float, OpAdd<float> >
#define reduceSumC16s32f reduceC_<short, float, OpAdd<float> >
#define reduceSumC32f32f reduceC_<float, float, OpAdd<float> >
#define reduceSumC64f64f reduceC_<double,double,OpAdd<double> >

#ifdef HAVE_IPP
#define reduceSumC8u64f  reduceSumC_8u16u16s32f_64f
#define reduceSumC16u64f reduceSumC_8u16u16s32f_64f
#define reduceSumC16s64f reduceSumC_8u16u16s32f_64f
#define reduceSumC32f64f reduceSumC_8u16u16s32f_64f
#else
#define reduceSumC8u64f  reduceC_<uchar, double,OpAdd<int> >
#define reduceSumC16u64f reduceC_<ushort,double,OpAdd<double> >
#define reduceSumC16s64f reduceC_<short, double,OpAdd<double> >
#define reduceSumC32f64f reduceC_<float, double,OpAdd<double> >
#endif

#ifdef HAVE_IPP
#define REDUCE_OP(favor, optype, type1, type2) \
static inline bool ipp_reduce##optype##C##favor(const cv::Mat& srcmat, cv::Mat& dstmat) \
{ \
    if((srcmat.channels() == 1)) \
    { \
        int sstep = (int)srcmat.step; \
        typedef Ipp##favor IppType; \
        IppiSize roisize = ippiSize(srcmat.size().width, 1);\
        for(int y = 0; y < srcmat.size().height; y++)\
        {\
            if(CV_INSTRUMENT_FUN_IPP(ippi##optype##_##favor##_C1R, srcmat.ptr<IppType>(y), sstep, roisize, dstmat.ptr<IppType>(y)) < 0)\
                return false;\
        }\
        return true;\
    }\
    return false; \
} \
static inline void reduce##optype##C##favor(const cv::Mat& srcmat, cv::Mat& dstmat) \
{ \
    CV_IPP_RUN_FAST(ipp_reduce##optype##C##favor(srcmat, dstmat)); \
    cv::reduceC_ < type1, type2, cv::Op##optype < type2 > >(srcmat, dstmat); \
}
#endif

#ifdef HAVE_IPP
REDUCE_OP(8u, Max, uchar, uchar)
REDUCE_OP(16u, Max, ushort, ushort)
REDUCE_OP(16s, Max, short, short)
REDUCE_OP(32f, Max, float, float)
#else
#define reduceMaxC8u  reduceC_<uchar, uchar, OpMax<uchar> >
#define reduceMaxC16u reduceC_<ushort,ushort,OpMax<ushort> >
#define reduceMaxC16s reduceC_<short, short, OpMax<short> >
#define reduceMaxC32f reduceC_<float, float, OpMax<float> >
#endif
#define reduceMaxC64f reduceC_<double,double,OpMax<double> >

#ifdef HAVE_IPP
REDUCE_OP(8u, Min, uchar, uchar)
REDUCE_OP(16u, Min, ushort, ushort)
REDUCE_OP(16s, Min, short, short)
REDUCE_OP(32f, Min, float, float)
#else
#define reduceMinC8u  reduceC_<uchar, uchar, OpMin<uchar> >
#define reduceMinC16u reduceC_<ushort,ushort,OpMin<ushort> >
#define reduceMinC16s reduceC_<short, short, OpMin<short> >
#define reduceMinC32f reduceC_<float, float, OpMin<float> >
#endif
#define reduceMinC64f reduceC_<double,double,OpMin<double> >

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_reduce(InputArray _src, OutputArray _dst,
                       int dim, int op, int op0, int stype, int dtype)
{
    const int min_opt_cols = 128, buf_cols = 32;
    int sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype),
            ddepth = CV_MAT_DEPTH(dtype), ddepth0 = ddepth;
    const ocl::Device &defDev = ocl::Device::getDefault();
    bool doubleSupport = defDev.doubleFPConfig() > 0;

    size_t wgs = defDev.maxWorkGroupSize();
    bool useOptimized = 1 == dim && _src.cols() > min_opt_cols && (wgs >= buf_cols);

    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return false;

    if (op == CV_REDUCE_AVG)
    {
        if (sdepth < CV_32S && ddepth < CV_32S)
            ddepth = CV_32S;
    }

    const char * const ops[4] = { "OCL_CV_REDUCE_SUM", "OCL_CV_REDUCE_AVG",
                                  "OCL_CV_REDUCE_MAX", "OCL_CV_REDUCE_MIN" };
    int wdepth = std::max(ddepth, CV_32F);
    if (useOptimized)
    {
        size_t tileHeight = (size_t)(wgs / buf_cols);
        if (defDev.isIntel())
        {
            static const size_t maxItemInGroupCount = 16;
            tileHeight = min(tileHeight, defDev.localMemSize() / buf_cols / CV_ELEM_SIZE(CV_MAKETYPE(wdepth, cn)) / maxItemInGroupCount);
        }
        char cvt[3][40];
        cv::String build_opt = format("-D OP_REDUCE_PRE -D BUF_COLS=%d -D TILE_HEIGHT=%d -D %s -D dim=1"
                                            " -D cn=%d -D ddepth=%d"
                                            " -D srcT=%s -D bufT=%s -D dstT=%s"
                                            " -D convertToWT=%s -D convertToBufT=%s -D convertToDT=%s%s",
                                            buf_cols, tileHeight, ops[op], cn, ddepth,
                                            ocl::typeToStr(sdepth),
                                            ocl::typeToStr(ddepth),
                                            ocl::typeToStr(ddepth0),
                                            ocl::convertTypeStr(ddepth, wdepth, 1, cvt[0]),
                                            ocl::convertTypeStr(sdepth, ddepth, 1, cvt[1]),
                                            ocl::convertTypeStr(wdepth, ddepth0, 1, cvt[2]),
                                            doubleSupport ? " -D DOUBLE_SUPPORT" : "");
        ocl::Kernel k("reduce_horz_opt", ocl::core::reduce2_oclsrc, build_opt);
        if (k.empty())
            return false;
        UMat src = _src.getUMat();
        Size dsize(1, src.rows);
        _dst.create(dsize, dtype);
        UMat dst = _dst.getUMat();

        if (op0 == CV_REDUCE_AVG)
            k.args(ocl::KernelArg::ReadOnly(src),
                      ocl::KernelArg::WriteOnlyNoSize(dst), 1.0f / src.cols);
        else
            k.args(ocl::KernelArg::ReadOnly(src),
                      ocl::KernelArg::WriteOnlyNoSize(dst));

        size_t localSize[2] = { (size_t)buf_cols, (size_t)tileHeight};
        size_t globalSize[2] = { (size_t)buf_cols, (size_t)src.rows };
        return k.run(2, globalSize, localSize, false);
    }
    else
    {
        char cvt[2][40];
        cv::String build_opt = format("-D %s -D dim=%d -D cn=%d -D ddepth=%d"
                                      " -D srcT=%s -D dstT=%s -D dstT0=%s -D convertToWT=%s"
                                      " -D convertToDT=%s -D convertToDT0=%s%s",
                                      ops[op], dim, cn, ddepth, ocl::typeToStr(useOptimized ? ddepth : sdepth),
                                      ocl::typeToStr(ddepth), ocl::typeToStr(ddepth0),
                                      ocl::convertTypeStr(ddepth, wdepth, 1, cvt[0]),
                                      ocl::convertTypeStr(sdepth, ddepth, 1, cvt[0]),
                                      ocl::convertTypeStr(wdepth, ddepth0, 1, cvt[1]),
                                      doubleSupport ? " -D DOUBLE_SUPPORT" : "");

        ocl::Kernel k("reduce", ocl::core::reduce2_oclsrc, build_opt);
        if (k.empty())
            return false;

        UMat src = _src.getUMat();
        Size dsize(dim == 0 ? src.cols : 1, dim == 0 ? 1 : src.rows);
        _dst.create(dsize, dtype);
        UMat dst = _dst.getUMat();

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnly(src),
                temparg = ocl::KernelArg::WriteOnlyNoSize(dst);

        if (op0 == CV_REDUCE_AVG)
            k.args(srcarg, temparg, 1.0f / (dim == 0 ? src.rows : src.cols));
        else
            k.args(srcarg, temparg);

        size_t globalsize = std::max(dsize.width, dsize.height);
        return k.run(1, &globalsize, NULL, false);
    }
}

}

#endif

void cv::reduce(InputArray _src, OutputArray _dst, int dim, int op, int dtype)
{
    CV_INSTRUMENT_REGION()

    CV_Assert( _src.dims() <= 2 );
    int op0 = op;
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if( dtype < 0 )
        dtype = _dst.fixedType() ? _dst.type() : stype;
    dtype = CV_MAKETYPE(dtype >= 0 ? dtype : stype, cn);
    int ddepth = CV_MAT_DEPTH(dtype);

    CV_Assert( cn == CV_MAT_CN(dtype) );
    CV_Assert( op == CV_REDUCE_SUM || op == CV_REDUCE_MAX ||
               op == CV_REDUCE_MIN || op == CV_REDUCE_AVG );

    CV_OCL_RUN(_dst.isUMat(),
               ocl_reduce(_src, _dst, dim, op, op0, stype, dtype))

    Mat src = _src.getMat();
    _dst.create(dim == 0 ? 1 : src.rows, dim == 0 ? src.cols : 1, dtype);
    Mat dst = _dst.getMat(), temp = dst;

    if( op == CV_REDUCE_AVG )
    {
        op = CV_REDUCE_SUM;
        if( sdepth < CV_32S && ddepth < CV_32S )
        {
            temp.create(dst.rows, dst.cols, CV_32SC(cn));
            ddepth = CV_32S;
        }
    }

    ReduceFunc func = 0;
    if( dim == 0 )
    {
        if( op == CV_REDUCE_SUM )
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = GET_OPTIMIZED(reduceSumR8u32s);
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSumR8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSumR8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSumR16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSumR16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSumR16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSumR16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceSumR32f32f);
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumR32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumR64f64f;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMaxR8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMaxR32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxR64f;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMinR8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMinR32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinR64f;
        }
    }
    else
    {
        if(op == CV_REDUCE_SUM)
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = GET_OPTIMIZED(reduceSumC8u32s);
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSumC8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSumC8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSumC16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSumC16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSumC16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSumC16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceSumC32f32f);
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumC32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumC64f64f;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMaxC8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMaxC32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxC64f;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMinC8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMinC32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinC64f;
        }
    }

    if( !func )
        CV_Error( CV_StsUnsupportedFormat,
                  "Unsupported combination of input and output array formats" );

    func( src, temp );

    if( op0 == CV_REDUCE_AVG )
        temp.convertTo(dst, dst.type(), 1./(dim == 0 ? src.rows : src.cols));
}


//////////////////////////////////////// sort ///////////////////////////////////////////

namespace cv
{

template<typename T> static void sort_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    T* bptr;
    int n, len;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool inplace = src.data == dst.data;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;

    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
    }
    bptr = (T*)buf;

    for( int i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        if( sortRows )
        {
            T* dptr = dst.ptr<T>(i);
            if( !inplace )
            {
                const T* sptr = src.ptr<T>(i);
                memcpy(dptr, sptr, sizeof(T) * len);
            }
            ptr = dptr;
        }
        else
        {
            for( int j = 0; j < len; j++ )
                ptr[j] = src.ptr<T>(j)[i];
        }

        std::sort( ptr, ptr + len );
        if( sortDescending )
        {
            for( int j = 0; j < len/2; j++ )
                std::swap(ptr[j], ptr[len-1-j]);
        }

        if( !sortRows )
            for( int j = 0; j < len; j++ )
                dst.ptr<T>(j)[i] = ptr[j];
    }
}

#ifdef HAVE_IPP
typedef IppStatus (CV_STDCALL *IppSortFunc)(void  *pSrcDst, int    len, Ipp8u *pBuffer);

static IppSortFunc getSortFunc(int depth, bool sortDescending)
{
    if (!sortDescending)
        return depth == CV_8U ? (IppSortFunc)ippsSortRadixAscend_8u_I :
            depth == CV_16U ? (IppSortFunc)ippsSortRadixAscend_16u_I :
            depth == CV_16S ? (IppSortFunc)ippsSortRadixAscend_16s_I :
            depth == CV_32S ? (IppSortFunc)ippsSortRadixAscend_32s_I :
            depth == CV_32F ? (IppSortFunc)ippsSortRadixAscend_32f_I :
            depth == CV_64F ? (IppSortFunc)ippsSortRadixAscend_64f_I :
            0;
    else
        return depth == CV_8U ? (IppSortFunc)ippsSortRadixDescend_8u_I :
            depth == CV_16U ? (IppSortFunc)ippsSortRadixDescend_16u_I :
            depth == CV_16S ? (IppSortFunc)ippsSortRadixDescend_16s_I :
            depth == CV_32S ? (IppSortFunc)ippsSortRadixDescend_32s_I :
            depth == CV_32F ? (IppSortFunc)ippsSortRadixDescend_32f_I :
            depth == CV_64F ? (IppSortFunc)ippsSortRadixDescend_64f_I :
            0;
}

static bool ipp_sort(const Mat& src, Mat& dst, int flags)
{
    CV_INSTRUMENT_REGION_IPP()

    bool        sortRows        = (flags & 1) == CV_SORT_EVERY_ROW;
    bool        sortDescending  = (flags & CV_SORT_DESCENDING) != 0;
    bool        inplace         = (src.data == dst.data);
    int         depth           = src.depth();
    IppDataType type            = ippiGetDataType(depth);

    IppSortFunc ippsSortRadix_I = getSortFunc(depth, sortDescending);
    if(!ippsSortRadix_I)
        return false;

    if(sortRows)
    {
        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixGetBufferSize(src.cols, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        if(!inplace)
            src.copyTo(dst);

        for(int i = 0; i < dst.rows; i++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadix_I, (void*)dst.ptr(i), dst.cols, buffer) < 0)
                return false;
        }
    }
    else
    {
        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixGetBufferSize(src.rows, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        Mat  row(1, src.rows, src.type());
        Mat  srcSub;
        Mat  dstSub;
        Rect subRect(0,0,1,src.rows);

        for(int i = 0; i < src.cols; i++)
        {
            subRect.x = i;
            srcSub = Mat(src, subRect);
            dstSub = Mat(dst, subRect);
            srcSub.copyTo(row);

            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadix_I, (void*)row.ptr(), dst.rows, buffer) < 0)
                return false;

            row = row.reshape(1, dstSub.rows);
            row.copyTo(dstSub);
        }
    }

    return true;
}
#endif

template<typename _Tp> class LessThanIdx
{
public:
    LessThanIdx( const _Tp* _arr ) : arr(_arr) {}
    bool operator()(int a, int b) const { return arr[a] < arr[b]; }
    const _Tp* arr;
};

template<typename T> static void sortIdx_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    AutoBuffer<int> ibuf;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;

    CV_Assert( src.data != dst.data );

    int n, len;
    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
        ibuf.allocate(len);
    }
    T* bptr = (T*)buf;
    int* _iptr = (int*)ibuf;

    for( int i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        int* iptr = _iptr;

        if( sortRows )
        {
            ptr = (T*)(src.data + src.step*i);
            iptr = dst.ptr<int>(i);
        }
        else
        {
            for( int j = 0; j < len; j++ )
                ptr[j] = src.ptr<T>(j)[i];
        }
        for( int j = 0; j < len; j++ )
            iptr[j] = j;

        std::sort( iptr, iptr + len, LessThanIdx<T>(ptr) );
        if( sortDescending )
        {
            for( int j = 0; j < len/2; j++ )
                std::swap(iptr[j], iptr[len-1-j]);
        }

        if( !sortRows )
            for( int j = 0; j < len; j++ )
                dst.ptr<int>(j)[i] = iptr[j];
    }
}

#ifdef HAVE_IPP
#if !IPP_DISABLE_SORT_IDX
typedef IppStatus (CV_STDCALL *IppSortIndexFunc)(const void*  pSrc, Ipp32s srcStrideBytes, Ipp32s *pDstIndx, int len, Ipp8u *pBuffer);

static IppSortIndexFunc getSortIndexFunc(int depth, bool sortDescending)
{
    if (!sortDescending)
        return depth == CV_8U ? (IppSortIndexFunc)ippsSortRadixIndexAscend_8u :
            depth == CV_16U ? (IppSortIndexFunc)ippsSortRadixIndexAscend_16u :
            depth == CV_16S ? (IppSortIndexFunc)ippsSortRadixIndexAscend_16s :
            depth == CV_32S ? (IppSortIndexFunc)ippsSortRadixIndexAscend_32s :
            depth == CV_32F ? (IppSortIndexFunc)ippsSortRadixIndexAscend_32f :
            0;
    else
        return depth == CV_8U ? (IppSortIndexFunc)ippsSortRadixIndexDescend_8u :
            depth == CV_16U ? (IppSortIndexFunc)ippsSortRadixIndexDescend_16u :
            depth == CV_16S ? (IppSortIndexFunc)ippsSortRadixIndexDescend_16s :
            depth == CV_32S ? (IppSortIndexFunc)ippsSortRadixIndexDescend_32s :
            depth == CV_32F ? (IppSortIndexFunc)ippsSortRadixIndexDescend_32f :
            0;
}

static bool ipp_sortIdx( const Mat& src, Mat& dst, int flags )
{
    CV_INSTRUMENT_REGION_IPP()

    bool        sortRows        = (flags & 1) == SORT_EVERY_ROW;
    bool        sortDescending  = (flags & SORT_DESCENDING) != 0;
    int         depth           = src.depth();
    IppDataType type            = ippiGetDataType(depth);

    IppSortIndexFunc ippsSortRadixIndex = getSortIndexFunc(depth, sortDescending);
    if(!ippsSortRadixIndex)
        return false;

    if(sortRows)
    {
        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixIndexGetBufferSize(src.cols, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        for(int i = 0; i < src.rows; i++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadixIndex, (const void*)src.ptr(i), (Ipp32s)src.step[1], (Ipp32s*)dst.ptr(i), src.cols, buffer) < 0)
                return false;
        }
    }
    else
    {
        Mat  dstRow(1, dst.rows, dst.type());
        Mat  dstSub;
        Rect subRect(0,0,1,src.rows);

        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixIndexGetBufferSize(src.rows, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        Ipp32s srcStep = (Ipp32s)src.step[0];
        for(int i = 0; i < src.cols; i++)
        {
            subRect.x = i;
            dstSub = Mat(dst, subRect);

            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadixIndex, (const void*)src.ptr(0, i), srcStep, (Ipp32s*)dstRow.ptr(), src.rows, buffer) < 0)
                return false;

            dstRow = dstRow.reshape(1, dstSub.rows);
            dstRow.copyTo(dstSub);
        }
    }

    return true;
}
#endif
#endif

typedef void (*SortFunc)(const Mat& src, Mat& dst, int flags);
}

void cv::sort( InputArray _src, OutputArray _dst, int flags )
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    CV_Assert( src.dims <= 2 && src.channels() == 1 );
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();
    CV_IPP_RUN_FAST(ipp_sort(src, dst, flags));

    static SortFunc tab[] =
    {
        sort_<uchar>, sort_<schar>, sort_<ushort>, sort_<short>,
        sort_<int>, sort_<float>, sort_<double>, 0
    };
    SortFunc func = tab[src.depth()];
    CV_Assert( func != 0 );

    func( src, dst, flags );
}

void cv::sortIdx( InputArray _src, OutputArray _dst, int flags )
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    CV_Assert( src.dims <= 2 && src.channels() == 1 );
    Mat dst = _dst.getMat();
    if( dst.data == src.data )
        _dst.release();
    _dst.create( src.size(), CV_32S );
    dst = _dst.getMat();
#if !IPP_DISABLE_SORT_IDX
    CV_IPP_RUN_FAST(ipp_sortIdx(src, dst, flags));
#endif

    static SortFunc tab[] =
    {
        sortIdx_<uchar>, sortIdx_<schar>, sortIdx_<ushort>, sortIdx_<short>,
        sortIdx_<int>, sortIdx_<float>, sortIdx_<double>, 0
    };
    SortFunc func = tab[src.depth()];
    CV_Assert( func != 0 );
    func( src, dst, flags );
}


CV_IMPL void cvSetIdentity( CvArr* arr, CvScalar value )
{
    cv::Mat m = cv::cvarrToMat(arr);
    cv::setIdentity(m, value);
}


CV_IMPL CvScalar cvTrace( const CvArr* arr )
{
    return cv::trace(cv::cvarrToMat(arr));
}


CV_IMPL void cvTranspose( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.rows == dst.cols && src.cols == dst.rows && src.type() == dst.type() );
    transpose( src, dst );
}


CV_IMPL void cvCompleteSymm( CvMat* matrix, int LtoR )
{
    cv::Mat m = cv::cvarrToMat(matrix);
    cv::completeSymm( m, LtoR != 0 );
}


CV_IMPL void cvCrossProduct( const CvArr* srcAarr, const CvArr* srcBarr, CvArr* dstarr )
{
    cv::Mat srcA = cv::cvarrToMat(srcAarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( srcA.size() == dst.size() && srcA.type() == dst.type() );
    srcA.cross(cv::cvarrToMat(srcBarr)).copyTo(dst);
}


CV_IMPL void
cvReduce( const CvArr* srcarr, CvArr* dstarr, int dim, int op )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    if( dim < 0 )
        dim = src.rows > dst.rows ? 0 : src.cols > dst.cols ? 1 : dst.cols == 1;

    if( dim > 1 )
        CV_Error( CV_StsOutOfRange, "The reduced dimensionality index is out of range" );

    if( (dim == 0 && (dst.cols != src.cols || dst.rows != 1)) ||
        (dim == 1 && (dst.rows != src.rows || dst.cols != 1)) )
        CV_Error( CV_StsBadSize, "The output array size is incorrect" );

    if( src.channels() != dst.channels() )
        CV_Error( CV_StsUnmatchedFormats, "Input and output arrays must have the same number of channels" );

    cv::reduce(src, dst, dim, op, dst.type());
}


CV_IMPL CvArr*
cvRange( CvArr* arr, double start, double end )
{
    CvMat stub, *mat = (CvMat*)arr;
    int step;
    double val = start;

    if( !CV_IS_MAT(mat) )
        mat = cvGetMat( mat, &stub);

    int rows = mat->rows;
    int cols = mat->cols;
    int type = CV_MAT_TYPE(mat->type);
    double delta = (end-start)/(rows*cols);

    if( CV_IS_MAT_CONT(mat->type) )
    {
        cols *= rows;
        rows = 1;
        step = 1;
    }
    else
        step = mat->step / CV_ELEM_SIZE(type);

    if( type == CV_32SC1 )
    {
        int* idata = mat->data.i;
        int ival = cvRound(val), idelta = cvRound(delta);

        if( fabs(val - ival) < DBL_EPSILON &&
            fabs(delta - idelta) < DBL_EPSILON )
        {
            for( int i = 0; i < rows; i++, idata += step )
                for( int j = 0; j < cols; j++, ival += idelta )
                    idata[j] = ival;
        }
        else
        {
            for( int i = 0; i < rows; i++, idata += step )
                for( int j = 0; j < cols; j++, val += delta )
                    idata[j] = cvRound(val);
        }
    }
    else if( type == CV_32FC1 )
    {
        float* fdata = mat->data.fl;
        for( int i = 0; i < rows; i++, fdata += step )
            for( int j = 0; j < cols; j++, val += delta )
                fdata[j] = (float)val;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "The function only supports 32sC1 and 32fC1 datatypes" );

    return arr;
}


CV_IMPL void
cvSort( const CvArr* _src, CvArr* _dst, CvArr* _idx, int flags )
{
    cv::Mat src = cv::cvarrToMat(_src);

    if( _idx )
    {
        cv::Mat idx0 = cv::cvarrToMat(_idx), idx = idx0;
        CV_Assert( src.size() == idx.size() && idx.type() == CV_32S && src.data != idx.data );
        cv::sortIdx( src, idx, flags );
        CV_Assert( idx0.data == idx.data );
    }

    if( _dst )
    {
        cv::Mat dst0 = cv::cvarrToMat(_dst), dst = dst0;
        CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
        cv::sort( src, dst, flags );
        CV_Assert( dst0.data == dst.data );
    }
}


CV_IMPL int
cvKMeans2( const CvArr* _samples, int cluster_count, CvArr* _labels,
           CvTermCriteria termcrit, int attempts, CvRNG*,
           int flags, CvArr* _centers, double* _compactness )
{
    cv::Mat data = cv::cvarrToMat(_samples), labels = cv::cvarrToMat(_labels), centers;
    if( _centers )
    {
        centers = cv::cvarrToMat(_centers);

        centers = centers.reshape(1);
        data = data.reshape(1);

        CV_Assert( !centers.empty() );
        CV_Assert( centers.rows == cluster_count );
        CV_Assert( centers.cols == data.cols );
        CV_Assert( centers.depth() == data.depth() );
    }
    CV_Assert( labels.isContinuous() && labels.type() == CV_32S &&
        (labels.cols == 1 || labels.rows == 1) &&
        labels.cols + labels.rows - 1 == data.rows );

    double compactness = cv::kmeans(data, cluster_count, labels, termcrit, attempts,
                                    flags, _centers ? cv::_OutputArray(centers) : cv::_OutputArray() );
    if( _compactness )
        *_compactness = compactness;
    return 1;
}

///////////////////////////// n-dimensional matrices ////////////////////////////

namespace cv
{

Mat Mat::reshape(int _cn, int _newndims, const int* _newsz) const
{
    if(_newndims == dims)
    {
        if(_newsz == 0)
            return reshape(_cn);
        if(_newndims == 2)
            return reshape(_cn, _newsz[0]);
    }

    if (isContinuous())
    {
        CV_Assert(_cn >= 0 && _newndims > 0 && _newndims <= CV_MAX_DIM && _newsz);

        if (_cn == 0)
            _cn = this->channels();
        else
            CV_Assert(_cn <= CV_CN_MAX);

        size_t total_elem1_ref = this->total() * this->channels();
        size_t total_elem1 = _cn;

        AutoBuffer<int, 4> newsz_buf( (size_t)_newndims );

        for (int i = 0; i < _newndims; i++)
        {
            CV_Assert(_newsz[i] >= 0);

            if (_newsz[i] > 0)
                newsz_buf[i] = _newsz[i];
            else if (i < dims)
                newsz_buf[i] = this->size[i];
            else
                CV_Error(CV_StsOutOfRange, "Copy dimension (which has zero size) is not present in source matrix");

            total_elem1 *= (size_t)newsz_buf[i];
        }

        if (total_elem1 != total_elem1_ref)
            CV_Error(CV_StsUnmatchedSizes, "Requested and source matrices have different count of elements");

        Mat hdr = *this;
        hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((_cn-1) << CV_CN_SHIFT);
        setSize(hdr, _newndims, (int*)newsz_buf, NULL, true);

        return hdr;
    }

    CV_Error(CV_StsNotImplemented, "Reshaping of n-dimensional non-continuous matrices is not supported yet");
    // TBD
    return Mat();
}

Mat Mat::reshape(int _cn, const std::vector<int>& _newshape) const
{
    if(_newshape.empty())
    {
        CV_Assert(empty());
        return *this;
    }

    return reshape(_cn, (int)_newshape.size(), &_newshape[0]);
}


NAryMatIterator::NAryMatIterator()
    : arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
}

NAryMatIterator::NAryMatIterator(const Mat** _arrays, Mat* _planes, int _narrays)
: arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
    init(_arrays, _planes, 0, _narrays);
}

NAryMatIterator::NAryMatIterator(const Mat** _arrays, uchar** _ptrs, int _narrays)
    : arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
    init(_arrays, 0, _ptrs, _narrays);
}

void NAryMatIterator::init(const Mat** _arrays, Mat* _planes, uchar** _ptrs, int _narrays)
{
    CV_Assert( _arrays && (_ptrs || _planes) );
    int i, j, d1=0, i0 = -1, d = -1;

    arrays = _arrays;
    ptrs = _ptrs;
    planes = _planes;
    narrays = _narrays;
    nplanes = 0;
    size = 0;

    if( narrays < 0 )
    {
        for( i = 0; _arrays[i] != 0; i++ )
            ;
        narrays = i;
        CV_Assert(narrays <= 1000);
    }

    iterdepth = 0;

    for( i = 0; i < narrays; i++ )
    {
        CV_Assert(arrays[i] != 0);
        const Mat& A = *arrays[i];
        if( ptrs )
            ptrs[i] = A.data;

        if( !A.data )
            continue;

        if( i0 < 0 )
        {
            i0 = i;
            d = A.dims;

            // find the first dimensionality which is different from 1;
            // in any of the arrays the first "d1" step do not affect the continuity
            for( d1 = 0; d1 < d; d1++ )
                if( A.size[d1] > 1 )
                    break;
        }
        else
            CV_Assert( A.size == arrays[i0]->size );

        if( !A.isContinuous() )
        {
            CV_Assert( A.step[d-1] == A.elemSize() );
            for( j = d-1; j > d1; j-- )
                if( A.step[j]*A.size[j] < A.step[j-1] )
                    break;
            iterdepth = std::max(iterdepth, j);
        }
    }

    if( i0 >= 0 )
    {
        size = arrays[i0]->size[d-1];
        for( j = d-1; j > iterdepth; j-- )
        {
            int64 total1 = (int64)size*arrays[i0]->size[j-1];
            if( total1 != (int)total1 )
                break;
            size = (int)total1;
        }

        iterdepth = j;
        if( iterdepth == d1 )
            iterdepth = 0;

        nplanes = 1;
        for( j = iterdepth-1; j >= 0; j-- )
            nplanes *= arrays[i0]->size[j];
    }
    else
        iterdepth = 0;

    idx = 0;

    if( !planes )
        return;

    for( i = 0; i < narrays; i++ )
    {
        CV_Assert(arrays[i] != 0);
        const Mat& A = *arrays[i];

        if( !A.data )
        {
            planes[i] = Mat();
            continue;
        }

        planes[i] = Mat(1, (int)size, A.type(), A.data);
    }
}


NAryMatIterator& NAryMatIterator::operator ++()
{
    if( idx >= nplanes-1 )
        return *this;
    ++idx;

    if( iterdepth == 1 )
    {
        if( ptrs )
        {
            for( int i = 0; i < narrays; i++ )
            {
                if( !ptrs[i] )
                    continue;
                ptrs[i] = arrays[i]->data + arrays[i]->step[0]*idx;
            }
        }
        if( planes )
        {
            for( int i = 0; i < narrays; i++ )
            {
                if( !planes[i].data )
                    continue;
                planes[i].data = arrays[i]->data + arrays[i]->step[0]*idx;
            }
        }
    }
    else
    {
        for( int i = 0; i < narrays; i++ )
        {
            const Mat& A = *arrays[i];
            if( !A.data )
                continue;
            int _idx = (int)idx;
            uchar* data = A.data;
            for( int j = iterdepth-1; j >= 0 && _idx > 0; j-- )
            {
                int szi = A.size[j], t = _idx/szi;
                data += (_idx - t * szi)*A.step[j];
                _idx = t;
            }
            if( ptrs )
                ptrs[i] = data;
            if( planes )
                planes[i].data = data;
        }
    }

    return *this;
}

NAryMatIterator NAryMatIterator::operator ++(int)
{
    NAryMatIterator it = *this;
    ++*this;
    return it;
}

///////////////////////////////////////////////////////////////////////////
//                              MatConstIterator                         //
///////////////////////////////////////////////////////////////////////////

Point MatConstIterator::pos() const
{
    if( !m )
        return Point();
    CV_DbgAssert(m->dims <= 2);

    ptrdiff_t ofs = ptr - m->ptr();
    int y = (int)(ofs/m->step[0]);
    return Point((int)((ofs - y*m->step[0])/elemSize), y);
}

void MatConstIterator::pos(int* _idx) const
{
    CV_Assert(m != 0 && _idx);
    ptrdiff_t ofs = ptr - m->ptr();
    for( int i = 0; i < m->dims; i++ )
    {
        size_t s = m->step[i], v = ofs/s;
        ofs -= v*s;
        _idx[i] = (int)v;
    }
}

ptrdiff_t MatConstIterator::lpos() const
{
    if(!m)
        return 0;
    if( m->isContinuous() )
        return (ptr - sliceStart)/elemSize;
    ptrdiff_t ofs = ptr - m->ptr();
    int i, d = m->dims;
    if( d == 2 )
    {
        ptrdiff_t y = ofs/m->step[0];
        return y*m->cols + (ofs - y*m->step[0])/elemSize;
    }
    ptrdiff_t result = 0;
    for( i = 0; i < d; i++ )
    {
        size_t s = m->step[i], v = ofs/s;
        ofs -= v*s;
        result = result*m->size[i] + v;
    }
    return result;
}

void MatConstIterator::seek(ptrdiff_t ofs, bool relative)
{
    if( m->isContinuous() )
    {
        ptr = (relative ? ptr : sliceStart) + ofs*elemSize;
        if( ptr < sliceStart )
            ptr = sliceStart;
        else if( ptr > sliceEnd )
            ptr = sliceEnd;
        return;
    }

    int d = m->dims;
    if( d == 2 )
    {
        ptrdiff_t ofs0, y;
        if( relative )
        {
            ofs0 = ptr - m->ptr();
            y = ofs0/m->step[0];
            ofs += y*m->cols + (ofs0 - y*m->step[0])/elemSize;
        }
        y = ofs/m->cols;
        int y1 = std::min(std::max((int)y, 0), m->rows-1);
        sliceStart = m->ptr(y1);
        sliceEnd = sliceStart + m->cols*elemSize;
        ptr = y < 0 ? sliceStart : y >= m->rows ? sliceEnd :
            sliceStart + (ofs - y*m->cols)*elemSize;
        return;
    }

    if( relative )
        ofs += lpos();

    if( ofs < 0 )
        ofs = 0;

    int szi = m->size[d-1];
    ptrdiff_t t = ofs/szi;
    int v = (int)(ofs - t*szi);
    ofs = t;
    ptr = m->ptr() + v*elemSize;
    sliceStart = m->ptr();

    for( int i = d-2; i >= 0; i-- )
    {
        szi = m->size[i];
        t = ofs/szi;
        v = (int)(ofs - t*szi);
        ofs = t;
        sliceStart += v*m->step[i];
    }

    sliceEnd = sliceStart + m->size[d-1]*elemSize;
    if( ofs > 0 )
        ptr = sliceEnd;
    else
        ptr = sliceStart + (ptr - m->ptr());
}

void MatConstIterator::seek(const int* _idx, bool relative)
{
    int d = m->dims;
    ptrdiff_t ofs = 0;
    if( !_idx )
        ;
    else if( d == 2 )
        ofs = _idx[0]*m->size[1] + _idx[1];
    else
    {
        for( int i = 0; i < d; i++ )
            ofs = ofs*m->size[i] + _idx[i];
    }
    seek(ofs, relative);
}

//////////////////////////////// SparseMat ////////////////////////////////

template<typename T1, typename T2> void
convertData_(const void* _from, void* _to, int cn)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]);
}

template<typename T1, typename T2> void
convertScaleData_(const void* _from, void* _to, int cn, double alpha, double beta)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from*alpha + beta);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]*alpha + beta);
}

typedef void (*ConvertData)(const void* from, void* to, int cn);
typedef void (*ConvertScaleData)(const void* from, void* to, int cn, double alpha, double beta);

static ConvertData getConvertElem(int fromType, int toType)
{
    static ConvertData tab[][8] =
    {{ convertData_<uchar, uchar>, convertData_<uchar, schar>,
      convertData_<uchar, ushort>, convertData_<uchar, short>,
      convertData_<uchar, int>, convertData_<uchar, float>,
      convertData_<uchar, double>, 0 },

    { convertData_<schar, uchar>, convertData_<schar, schar>,
      convertData_<schar, ushort>, convertData_<schar, short>,
      convertData_<schar, int>, convertData_<schar, float>,
      convertData_<schar, double>, 0 },

    { convertData_<ushort, uchar>, convertData_<ushort, schar>,
      convertData_<ushort, ushort>, convertData_<ushort, short>,
      convertData_<ushort, int>, convertData_<ushort, float>,
      convertData_<ushort, double>, 0 },

    { convertData_<short, uchar>, convertData_<short, schar>,
      convertData_<short, ushort>, convertData_<short, short>,
      convertData_<short, int>, convertData_<short, float>,
      convertData_<short, double>, 0 },

    { convertData_<int, uchar>, convertData_<int, schar>,
      convertData_<int, ushort>, convertData_<int, short>,
      convertData_<int, int>, convertData_<int, float>,
      convertData_<int, double>, 0 },

    { convertData_<float, uchar>, convertData_<float, schar>,
      convertData_<float, ushort>, convertData_<float, short>,
      convertData_<float, int>, convertData_<float, float>,
      convertData_<float, double>, 0 },

    { convertData_<double, uchar>, convertData_<double, schar>,
      convertData_<double, ushort>, convertData_<double, short>,
      convertData_<double, int>, convertData_<double, float>,
      convertData_<double, double>, 0 },

    { 0, 0, 0, 0, 0, 0, 0, 0 }};

    ConvertData func = tab[CV_MAT_DEPTH(fromType)][CV_MAT_DEPTH(toType)];
    CV_Assert( func != 0 );
    return func;
}

static ConvertScaleData getConvertScaleElem(int fromType, int toType)
{
    static ConvertScaleData tab[][8] =
    {{ convertScaleData_<uchar, uchar>, convertScaleData_<uchar, schar>,
      convertScaleData_<uchar, ushort>, convertScaleData_<uchar, short>,
      convertScaleData_<uchar, int>, convertScaleData_<uchar, float>,
      convertScaleData_<uchar, double>, 0 },

    { convertScaleData_<schar, uchar>, convertScaleData_<schar, schar>,
      convertScaleData_<schar, ushort>, convertScaleData_<schar, short>,
      convertScaleData_<schar, int>, convertScaleData_<schar, float>,
      convertScaleData_<schar, double>, 0 },

    { convertScaleData_<ushort, uchar>, convertScaleData_<ushort, schar>,
      convertScaleData_<ushort, ushort>, convertScaleData_<ushort, short>,
      convertScaleData_<ushort, int>, convertScaleData_<ushort, float>,
      convertScaleData_<ushort, double>, 0 },

    { convertScaleData_<short, uchar>, convertScaleData_<short, schar>,
      convertScaleData_<short, ushort>, convertScaleData_<short, short>,
      convertScaleData_<short, int>, convertScaleData_<short, float>,
      convertScaleData_<short, double>, 0 },

    { convertScaleData_<int, uchar>, convertScaleData_<int, schar>,
      convertScaleData_<int, ushort>, convertScaleData_<int, short>,
      convertScaleData_<int, int>, convertScaleData_<int, float>,
      convertScaleData_<int, double>, 0 },

    { convertScaleData_<float, uchar>, convertScaleData_<float, schar>,
      convertScaleData_<float, ushort>, convertScaleData_<float, short>,
      convertScaleData_<float, int>, convertScaleData_<float, float>,
      convertScaleData_<float, double>, 0 },

    { convertScaleData_<double, uchar>, convertScaleData_<double, schar>,
      convertScaleData_<double, ushort>, convertScaleData_<double, short>,
      convertScaleData_<double, int>, convertScaleData_<double, float>,
      convertScaleData_<double, double>, 0 },

    { 0, 0, 0, 0, 0, 0, 0, 0 }};

    ConvertScaleData func = tab[CV_MAT_DEPTH(fromType)][CV_MAT_DEPTH(toType)];
    CV_Assert( func != 0 );
    return func;
}

enum { HASH_SIZE0 = 8 };

static inline void copyElem(const uchar* from, uchar* to, size_t elemSize)
{
    size_t i;
    for( i = 0; i + sizeof(int) <= elemSize; i += sizeof(int) )
        *(int*)(to + i) = *(const int*)(from + i);
    for( ; i < elemSize; i++ )
        to[i] = from[i];
}

static inline bool isZeroElem(const uchar* data, size_t elemSize)
{
    size_t i;
    for( i = 0; i + sizeof(int) <= elemSize; i += sizeof(int) )
        if( *(int*)(data + i) != 0 )
            return false;
    for( ; i < elemSize; i++ )
        if( data[i] != 0 )
            return false;
    return true;
}

SparseMat::Hdr::Hdr( int _dims, const int* _sizes, int _type )
{
    refcount = 1;

    dims = _dims;
    valueOffset = (int)alignSize(sizeof(SparseMat::Node) - MAX_DIM*sizeof(int) +
                                 dims*sizeof(int), CV_ELEM_SIZE1(_type));
    nodeSize = alignSize(valueOffset +
        CV_ELEM_SIZE(_type), (int)sizeof(size_t));

    int i;
    for( i = 0; i < dims; i++ )
        size[i] = _sizes[i];
    for( ; i < CV_MAX_DIM; i++ )
        size[i] = 0;
    clear();
}

void SparseMat::Hdr::clear()
{
    hashtab.clear();
    hashtab.resize(HASH_SIZE0);
    pool.clear();
    pool.resize(nodeSize);
    nodeCount = freeList = 0;
}


SparseMat::SparseMat(const Mat& m)
: flags(MAGIC_VAL), hdr(0)
{
    create( m.dims, m.size, m.type() );

    int i, idx[CV_MAX_DIM] = {0}, d = m.dims, lastSize = m.size[d - 1];
    size_t esz = m.elemSize();
    const uchar* dptr = m.ptr();

    for(;;)
    {
        for( i = 0; i < lastSize; i++, dptr += esz )
        {
            if( isZeroElem(dptr, esz) )
                continue;
            idx[d-1] = i;
            uchar* to = newNode(idx, hash(idx));
            copyElem( dptr, to, esz );
        }

        for( i = d - 2; i >= 0; i-- )
        {
            dptr += m.step[i] - m.size[i+1]*m.step[i+1];
            if( ++idx[i] < m.size[i] )
                break;
            idx[i] = 0;
        }
        if( i < 0 )
            break;
    }
}

void SparseMat::create(int d, const int* _sizes, int _type)
{
    CV_Assert( _sizes && 0 < d && d <= CV_MAX_DIM );
    for( int i = 0; i < d; i++ )
        CV_Assert( _sizes[i] > 0 );
    _type = CV_MAT_TYPE(_type);
    if( hdr && _type == type() && hdr->dims == d && hdr->refcount == 1 )
    {
        int i;
        for( i = 0; i < d; i++ )
            if( _sizes[i] != hdr->size[i] )
                break;
        if( i == d )
        {
            clear();
            return;
        }
    }
    int _sizes_backup[CV_MAX_DIM]; // #5991
    if (_sizes == hdr->size)
    {
        for(int i = 0; i < d; i++ )
            _sizes_backup[i] = _sizes[i];
        _sizes = _sizes_backup;
    }
    release();
    flags = MAGIC_VAL | _type;
    hdr = new Hdr(d, _sizes, _type);
}

void SparseMat::copyTo( SparseMat& m ) const
{
    if( hdr == m.hdr )
        return;
    if( !hdr )
    {
        m.release();
        return;
    }
    m.create( hdr->dims, hdr->size, type() );
    SparseMatConstIterator from = begin();
    size_t N = nzcount(), esz = elemSize();

    for( size_t i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        uchar* to = m.newNode(n->idx, n->hashval);
        copyElem( from.ptr, to, esz );
    }
}

void SparseMat::copyTo( Mat& m ) const
{
    CV_Assert( hdr );
    int ndims = dims();
    m.create( ndims, hdr->size, type() );
    m = Scalar(0);

    SparseMatConstIterator from = begin();
    size_t N = nzcount(), esz = elemSize();

    for( size_t i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        copyElem( from.ptr, (ndims > 1 ? m.ptr(n->idx) : m.ptr(n->idx[0])), esz);
    }
}


void SparseMat::convertTo( SparseMat& m, int rtype, double alpha ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);
    if( hdr == m.hdr && rtype != type()  )
    {
        SparseMat temp;
        convertTo(temp, rtype, alpha);
        m = temp;
        return;
    }

    CV_Assert(hdr != 0);
    if( hdr != m.hdr )
        m.create( hdr->dims, hdr->size, rtype );

    SparseMatConstIterator from = begin();
    size_t N = nzcount();

    if( alpha == 1 )
    {
        ConvertData cvtfunc = getConvertElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn );
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn, alpha, 0 );
        }
    }
}


void SparseMat::convertTo( Mat& m, int rtype, double alpha, double beta ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);

    CV_Assert( hdr );
    m.create( dims(), hdr->size, rtype );
    m = Scalar(beta);

    SparseMatConstIterator from = begin();
    size_t N = nzcount();

    if( alpha == 1 && beta == 0 )
    {
        ConvertData cvtfunc = getConvertElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn );
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn, alpha, beta );
        }
    }
}

void SparseMat::clear()
{
    if( hdr )
        hdr->clear();
}

uchar* SparseMat::ptr(int i0, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 1 );
    size_t h = hashval ? *hashval : hash(i0);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0 };
        return newNode( idx, h );
    }
    return NULL;
}

uchar* SparseMat::ptr(int i0, int i1, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1 };
        return newNode( idx, h );
    }
    return NULL;
}

uchar* SparseMat::ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1, i2 };
        return newNode( idx, h );
    }
    return NULL;
}

uchar* SparseMat::ptr(const int* idx, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                return &value<uchar>(elem);
        }
        nidx = elem->next;
    }

    return createMissing ? newNode(idx, h) : NULL;
}

void SparseMat::erase(int i0, int i1, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(int i0, int i1, int i2, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(const int* idx, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                break;
        }
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::resizeHashTab(size_t newsize)
{
    newsize = std::max(newsize, (size_t)8);
    if((newsize & (newsize-1)) != 0)
        newsize = (size_t)1 << cvCeil(std::log((double)newsize)/CV_LOG2);

    size_t hsize = hdr->hashtab.size();
    std::vector<size_t> _newh(newsize);
    size_t* newh = &_newh[0];
    for( size_t i = 0; i < newsize; i++ )
        newh[i] = 0;
    uchar* pool = &hdr->pool[0];
    for( size_t i = 0; i < hsize; i++ )
    {
        size_t nidx = hdr->hashtab[i];
        while( nidx )
        {
            Node* elem = (Node*)(pool + nidx);
            size_t next = elem->next;
            size_t newhidx = elem->hashval & (newsize - 1);
            elem->next = newh[newhidx];
            newh[newhidx] = nidx;
            nidx = next;
        }
    }
    hdr->hashtab = _newh;
}

uchar* SparseMat::newNode(const int* idx, size_t hashval)
{
    const int HASH_MAX_FILL_FACTOR=3;
    assert(hdr);
    size_t hsize = hdr->hashtab.size();
    if( ++hdr->nodeCount > hsize*HASH_MAX_FILL_FACTOR )
    {
        resizeHashTab(std::max(hsize*2, (size_t)8));
        hsize = hdr->hashtab.size();
    }

    if( !hdr->freeList )
    {
        size_t i, nsz = hdr->nodeSize, psize = hdr->pool.size(),
            newpsize = std::max(psize*3/2, 8*nsz);
        newpsize = (newpsize/nsz)*nsz;
        hdr->pool.resize(newpsize);
        uchar* pool = &hdr->pool[0];
        hdr->freeList = std::max(psize, nsz);
        for( i = hdr->freeList; i < newpsize - nsz; i += nsz )
            ((Node*)(pool + i))->next = i + nsz;
        ((Node*)(pool + i))->next = 0;
    }
    size_t nidx = hdr->freeList;
    Node* elem = (Node*)&hdr->pool[nidx];
    hdr->freeList = elem->next;
    elem->hashval = hashval;
    size_t hidx = hashval & (hsize - 1);
    elem->next = hdr->hashtab[hidx];
    hdr->hashtab[hidx] = nidx;

    int i, d = hdr->dims;
    for( i = 0; i < d; i++ )
        elem->idx[i] = idx[i];
    size_t esz = elemSize();
    uchar* p = &value<uchar>(elem);
    if( esz == sizeof(float) )
        *((float*)p) = 0.f;
    else if( esz == sizeof(double) )
        *((double*)p) = 0.;
    else
        memset(p, 0, esz);

    return p;
}


void SparseMat::removeNode(size_t hidx, size_t nidx, size_t previdx)
{
    Node* n = node(nidx);
    if( previdx )
    {
        Node* prev = node(previdx);
        prev->next = n->next;
    }
    else
        hdr->hashtab[hidx] = n->next;
    n->next = hdr->freeList;
    hdr->freeList = nidx;
    --hdr->nodeCount;
}


SparseMatConstIterator::SparseMatConstIterator(const SparseMat* _m)
: m((SparseMat*)_m), hashidx(0), ptr(0)
{
    if(!_m || !_m->hdr)
        return;
    SparseMat::Hdr& hdr = *m->hdr;
    const std::vector<size_t>& htab = hdr.hashtab;
    size_t i, hsize = htab.size();
    for( i = 0; i < hsize; i++ )
    {
        size_t nidx = htab[i];
        if( nidx )
        {
            hashidx = i;
            ptr = &hdr.pool[nidx] + hdr.valueOffset;
            return;
        }
    }
}

SparseMatConstIterator& SparseMatConstIterator::operator ++()
{
    if( !ptr || !m || !m->hdr )
        return *this;
    SparseMat::Hdr& hdr = *m->hdr;
    size_t next = ((const SparseMat::Node*)(ptr - hdr.valueOffset))->next;
    if( next )
    {
        ptr = &hdr.pool[next] + hdr.valueOffset;
        return *this;
    }
    size_t i = hashidx + 1, sz = hdr.hashtab.size();
    for( ; i < sz; i++ )
    {
        size_t nidx = hdr.hashtab[i];
        if( nidx )
        {
            hashidx = i;
            ptr = &hdr.pool[nidx] + hdr.valueOffset;
            return *this;
        }
    }
    hashidx = sz;
    ptr = 0;
    return *this;
}


double norm( const SparseMat& src, int normType )
{
    CV_INSTRUMENT_REGION()

    SparseMatConstIterator it = src.begin();

    size_t i, N = src.nzcount();
    normType &= NORM_TYPE_MASK;
    int type = src.type();
    double result = 0;

    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    if( type == CV_32F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result = std::max(result, std::abs((double)it.value<float>()));
            }
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result += std::abs(it.value<float>());
            }
        else
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                double v = it.value<float>();
                result += v*v;
            }
    }
    else if( type == CV_64F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result = std::max(result, std::abs(it.value<double>()));
            }
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result += std::abs(it.value<double>());
            }
        else
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                double v = it.value<double>();
                result += v*v;
            }
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );

    if( normType == NORM_L2 )
        result = std::sqrt(result);
    return result;
}

void minMaxLoc( const SparseMat& src, double* _minval, double* _maxval, int* _minidx, int* _maxidx )
{
    CV_INSTRUMENT_REGION()

    SparseMatConstIterator it = src.begin();
    size_t i, N = src.nzcount(), d = src.hdr ? src.hdr->dims : 0;
    int type = src.type();
    const int *minidx = 0, *maxidx = 0;

    if( type == CV_32F )
    {
        float minval = FLT_MAX, maxval = -FLT_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            CV_Assert(it.ptr);
            float v = it.value<float>();
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else if( type == CV_64F )
    {
        double minval = DBL_MAX, maxval = -DBL_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            CV_Assert(it.ptr);
            double v = it.value<double>();
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );

    if( _minidx && minidx )
        for( i = 0; i < d; i++ )
            _minidx[i] = minidx[i];
    if( _maxidx && maxidx )
        for( i = 0; i < d; i++ )
            _maxidx[i] = maxidx[i];
}


void normalize( const SparseMat& src, SparseMat& dst, double a, int norm_type )
{
    CV_INSTRUMENT_REGION()

    double scale = 1;
    if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( src, norm_type );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );

    src.convertTo( dst, -1, scale );
}

////////////////////// RotatedRect //////////////////////

RotatedRect::RotatedRect(const Point2f& _point1, const Point2f& _point2, const Point2f& _point3)
{
    Point2f _center = 0.5f * (_point1 + _point3);
    Vec2f vecs[2];
    vecs[0] = Vec2f(_point1 - _point2);
    vecs[1] = Vec2f(_point2 - _point3);
    // check that given sides are perpendicular
    CV_Assert( abs(vecs[0].dot(vecs[1])) / (norm(vecs[0]) * norm(vecs[1])) <= FLT_EPSILON );

    // wd_i stores which vector (0,1) or (1,2) will make the width
    // One of them will definitely have slope within -1 to 1
    int wd_i = 0;
    if( abs(vecs[1][1]) < abs(vecs[1][0]) ) wd_i = 1;
    int ht_i = (wd_i + 1) % 2;

    float _angle = atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0f / (float) CV_PI;
    float _width = (float) norm(vecs[wd_i]);
    float _height = (float) norm(vecs[ht_i]);

    center = _center;
    size = Size2f(_width, _height);
    angle = _angle;
}

void RotatedRect::points(Point2f pt[]) const
{
    double _angle = angle*CV_PI/180.;
    float b = (float)cos(_angle)*0.5f;
    float a = (float)sin(_angle)*0.5f;

    pt[0].x = center.x - a*size.height - b*size.width;
    pt[0].y = center.y + b*size.height - a*size.width;
    pt[1].x = center.x + a*size.height - b*size.width;
    pt[1].y = center.y - b*size.height - a*size.width;
    pt[2].x = 2*center.x - pt[0].x;
    pt[2].y = 2*center.y - pt[0].y;
    pt[3].x = 2*center.x - pt[1].x;
    pt[3].y = 2*center.y - pt[1].y;
}

Rect RotatedRect::boundingRect() const
{
    Point2f pt[4];
    points(pt);
    Rect r(cvFloor(std::min(std::min(std::min(pt[0].x, pt[1].x), pt[2].x), pt[3].x)),
           cvFloor(std::min(std::min(std::min(pt[0].y, pt[1].y), pt[2].y), pt[3].y)),
           cvCeil(std::max(std::max(std::max(pt[0].x, pt[1].x), pt[2].x), pt[3].x)),
           cvCeil(std::max(std::max(std::max(pt[0].y, pt[1].y), pt[2].y), pt[3].y)));
    r.width -= r.x - 1;
    r.height -= r.y - 1;
    return r;
}


Rect_<float> RotatedRect::boundingRect2f() const
{
    Point2f pt[4];
    points(pt);
    Rect_<float> r(Point_<float>(min(min(min(pt[0].x, pt[1].x), pt[2].x), pt[3].x), min(min(min(pt[0].y, pt[1].y), pt[2].y), pt[3].y)),
                   Point_<float>(max(max(max(pt[0].x, pt[1].x), pt[2].x), pt[3].x), max(max(max(pt[0].y, pt[1].y), pt[2].y), pt[3].y)));
    return r;
}

}

// glue

CvMatND::CvMatND(const cv::Mat& m)
{
    cvInitMatNDHeader(this, m.dims, m.size, m.type(), m.data );
    int i, d = m.dims;
    for( i = 0; i < d; i++ )
        dim[i].step = (int)m.step[i];
    type |= m.flags & cv::Mat::CONTINUOUS_FLAG;
}

_IplImage::_IplImage(const cv::Mat& m)
{
    CV_Assert( m.dims <= 2 );
    cvInitImageHeader(this, m.size(), cvIplDepth(m.flags), m.channels());
    cvSetData(this, m.data, (int)m.step[0]);
}

CvSparseMat* cvCreateSparseMat(const cv::SparseMat& sm)
{
    if( !sm.hdr || sm.hdr->dims > (int)cv::SparseMat::MAX_DIM)
        return 0;

    CvSparseMat* m = cvCreateSparseMat(sm.hdr->dims, sm.hdr->size, sm.type());

    cv::SparseMatConstIterator from = sm.begin();
    size_t i, N = sm.nzcount(), esz = sm.elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const cv::SparseMat::Node* n = from.node();
        uchar* to = cvPtrND(m, n->idx, 0, -2, 0);
        cv::copyElem(from.ptr, to, esz);
    }
    return m;
}

void CvSparseMat::copyToSparseMat(cv::SparseMat& m) const
{
    m.create( dims, &size[0], type );

    CvSparseMatIterator it;
    CvSparseNode* n = cvInitSparseMatIterator(this, &it);
    size_t esz = m.elemSize();

    for( ; n != 0; n = cvGetNextSparseNode(&it) )
    {
        const int* idx = CV_NODE_IDX(this, n);
        uchar* to = m.newNode(idx, m.hash(idx));
        cv::copyElem((const uchar*)CV_NODE_VAL(this, n), to, esz);
    }
}


/* End of file. */
