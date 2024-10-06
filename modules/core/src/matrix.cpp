// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "bufferpool.impl.hpp"

namespace cv {

std::string layoutToString(DataLayout layout)
{
    return
        layout == DATA_LAYOUT_ND ? "ND" :
        layout == DATA_LAYOUT_NCHW ? "NCHW" :
        layout == DATA_LAYOUT_NHWC ? "NHWC" :
        layout == DATA_LAYOUT_BLOCK ? "NC1HWC0" :
        layout == DATA_LAYOUT_NCDHW ? "NCDHW" :
        layout == DATA_LAYOUT_NDHWC ? "NDHWC" :
        layout == DATA_LAYOUT_PLANAR ? "PLANAR" :
        layout == DATA_LAYOUT_UNKNOWN ? "Unknown" : "???";
}

bool operator == (const MatShape& size1, const MatShape& size2)
{
    if (size1.dims != size2.dims)
        return false;
    if (size1.layout != size2.layout &&
        size1.layout != DATA_LAYOUT_UNKNOWN &&
        size2.layout != DATA_LAYOUT_UNKNOWN)
        return false;
    if (size1.layout == DATA_LAYOUT_BLOCK &&
        size2.layout == DATA_LAYOUT_BLOCK &&
        size1.C != size2.C)
        return false;
    for (int i = 0; i < size1.dims; i++) {
        if (size1.p[i] != size2.p[i])
            return false;
    }
    return true;
}

bool operator != (const MatShape& size1, const MatShape& size2)
{
    return !(size1 == size2);
}

/////////////////////////// MatShape ////////////////////////////////

MatShape MatShape::scalar()
{
    return MatShape(0);
}

void MatShape::clear()
{
    dims = -1;
    layout = DATA_LAYOUT_UNKNOWN;
    C = 0;
    for (int i = 0; i < MAX_DIMS; i++)
        p[i] = 0;
}

void MatShape::resize(size_t newSize, int value)
{
    CV_Assert(newSize < (size_t)MAX_DIMS);
    int old_dims = std::max(dims, 0);
    dims = (int)newSize;
    for (int i = old_dims; i < dims; i++)
        p[i] = value;
}

void MatShape::reserve(size_t)
{
    // no op; maybe need to add a check for overflow, but we check it anyway in other operations
}

void MatShape::assign(size_t newSize, int value)
{
    CV_Assert(newSize < (size_t)MAX_DIMS);
    dims = (int)newSize;
    for (int i = 0; i < dims; i++)
        p[i] = value;
}

void MatShape::assign(int newSize, int value)
{
    assign((size_t)newSize, value);
}

void MatShape::assign(const int* begin, const int* end)
{
    assign_(begin, end);
}

void MatShape::assign_(const int* begin, const int* end)
{
    ptrdiff_t newSize = end - begin;
    CV_Assert(0 <= newSize && newSize < (ptrdiff_t)MAX_DIMS);
    dims = (int)newSize;
    for (int i = 0; i < dims; i++)
        p[i] = begin[i];
}

int* MatShape::begin() { return p; }
const int* MatShape::begin() const { return p; }
int* MatShape::end() { return p + std::max(dims, 0); }
const int* MatShape::end() const { return p + std::max(dims, 0); }
int& MatShape::back() { return p[std::max(dims-1, 0)]; }
const int& MatShape::back() const { return p[std::max(dims-1, 0)]; }

void MatShape::push_back(int value)
{
    CV_Assert(dims+1 < MAX_DIMS);
    dims = std::max(dims+1, 1);
    p[dims-1] = value;
}

void MatShape::emplace_back(int value)
{
    push_back(value);
}

void MatShape::insert(int* where, int value)
{
    int old_dims = std::max(dims, 0);
    CV_Assert(old_dims+1 < MAX_DIMS);
    ptrdiff_t ofs = where - p;
    CV_Assert(0 <= ofs && ofs <= old_dims);
    dims = old_dims+1;
    for (int i = old_dims-1; i >= (int)ofs; i--)
        p[i+1] = p[i];
    p[ofs] = value;
}

void MatShape::insert(int* where, size_t count, int value)
{
    int old_dims = std::max(dims, 0);
    CV_Assert((size_t)(old_dims+count) < (size_t)MAX_DIMS);
    ptrdiff_t ofs = where - p;
    CV_Assert(0 <= ofs && ofs <= old_dims);
    dims = (int)(old_dims+count);
    for (int i = old_dims-1; i >= (int)ofs; i--)
        p[i+count] = p[i];
    for (int i = 0; i < (int)count; i++)
        p[i+ofs] = value;
}

void MatShape::insert(int* where, int count, int value)
{
    insert(where, (size_t)count, value);
}

void MatShape::insert(int* where, const int* begin, const int* end)
{
    insert_(where, begin, end);
}

void MatShape::insert_(int* where, const int* begin, const int* end)
{
    int old_dims = std::max(dims, 0);
    ptrdiff_t delta = end - begin;
    CV_Assert(0 <= delta && old_dims+delta < MAX_DIMS);
    ptrdiff_t ofs = where - p;
    CV_Assert(0 <= ofs && ofs <= old_dims);
    dims = (int)(old_dims+delta);
    for (int i = old_dims-1; i >= (int)ofs; i--)
        p[i+delta] = p[i];
    for (int i = 0; i < (int)delta; i++)
        p[i+ofs] = begin[i];
}

void MatShape::erase(int* where)
{
    CV_Assert(dims > 0);
    ptrdiff_t ofs = where - p;
    CV_Assert(0 <= ofs && ofs <= dims);
    if (ofs == dims)
        return;
    dims--;
    for (int i = (int)ofs+1; i <= dims; i++)
        p[i-1] = p[i];
}

size_t MatShape::total() const
{
    size_t result = 1;
    if (dims < 0)
        return 0;
    for (int i = 0; i < dims; i++)
        result *= p[i];
    return result;
}

std::string MatShape::str() const
{
    std::stringstream sstrm;
    if (empty()) {
        sstrm << "<empty>";
    } else if (dims == 0) {
        sstrm << "<scalar>";
    } else {
        sstrm << "[";
        for (int i = 0; i < dims; i++) {
            sstrm << (i > 0 ? " x " : "") << p[i];
        }
        sstrm << "]";
    }
    return sstrm.str();
}

static void finalizeBlockLayout(MatShape& size, int C=0)
{
    if (size.layout == DATA_LAYOUT_BLOCK) {
        CV_Assert(size.dims >= 4);
        int C0 = size.p[size.dims-1];
        CV_Assert(C0 > 1 && (C0 & (C0-1)) == 0);
        size.C = C > 0 ? C : size.p[1]*size.p[size.dims-1];
    } else {
        size.C = 0;
    }
    for (int i = std::max(size.dims, 0); i < MatShape::MAX_DIMS; i++)
        size.p[i] = 0;
    if (size.dims == 0)
        size.p[0] = 1;
}

MatShape::MatShape()
{
    clear();
}

MatShape::MatShape(size_t dims_, const int* size_, DataLayout layout_, int C_)
{
    layout = layout_;
    CV_Assert(dims_ <= (size_t)MAX_DIMS);
    dims = (int)dims_;
    for (int i = 0; i < dims; i++) {
        p[i] = size_ ? size_[i] : 0;
    }
    finalizeBlockLayout(*this, C_);
}

MatShape::MatShape(size_t dims_, int value, DataLayout layout_)
{
    layout = layout_;
    CV_Assert(dims_ <= (size_t)MAX_DIMS);
    dims = (int)dims_;
    for (int i = 0; i < dims; i++) {
        p[i] = value;
    }
    finalizeBlockLayout(*this, 0);
}

MatShape::MatShape(std::initializer_list<int> shape)
{
    layout = DATA_LAYOUT_UNKNOWN;
    CV_Assert(shape.size() <= (size_t)MAX_DIMS);
    dims = (int)shape.size();
    auto it = shape.begin();
    for (int i = 0; i < dims; i++, ++it) {
        p[i] = *it;
    }
    finalizeBlockLayout(*this, 0);
}

MatShape::MatShape(int dims_, int value, DataLayout layout_)
{
    layout = layout_;
    CV_Assert(dims_ <= MAX_DIMS);
    dims = dims_;
    for (int i = 0; i < dims; i++) {
        p[i] = value;
    }
    finalizeBlockLayout(*this, 0);
}

MatShape::MatShape(const std::vector<int>& shape_, DataLayout layout_, int C_)
{
    layout = layout_;
    size_t shape_size = shape_.size();
    CV_Assert(shape_size < (size_t)MAX_DIMS);
    dims = (int)shape_size;
    for (int i = 0; i < dims; i++) {
        p[i] = shape_[i];
    }
    finalizeBlockLayout(*this, C_);
}

MatShape::MatShape(const int* begin, const int* end, DataLayout layout_, int C_)
{
    layout = layout_;
    ptrdiff_t shape_size = end - begin;
    CV_Assert(0 <= shape_size && shape_size < MAX_DIMS);
    dims = (int)shape_size;
    for (int i = 0; i < dims; i++) {
        p[i] = begin[i];
    }
    finalizeBlockLayout(*this, C_);
}

MatShape::MatShape(const MatShape& shape)
{
    dims = shape.dims;
    layout = shape.layout;
    C = shape.C;
    for (int i = 0; i < MAX_DIMS; i++)
        p[i] = shape.p[i];
}

MatShape& MatShape::operator = (const MatShape& shape)
{
    if (this != &shape) {
        dims = shape.dims;
        layout = shape.layout;
        C = shape.C;
        for (int i = 0; i < MAX_DIMS; i++)
            p[i] = shape.p[i];
    }
    return *this;
}

bool MatShape::hasSymbols() const
{
    for (int i = 0; i < dims; i++) {
        if (p[i] < 0)
            return true;
    }
    return false;
}

MatShape MatShape::toBlock(int C0) const
{
    CV_Assert(dims >= 3);
    // C0 should be > 1 and be a power-of-2: 2, 4, 8, ...
    CV_Assert(C0 > 1 && (C0 & (C0-1)) == 0);
    CV_Assert(layout == DATA_LAYOUT_NCHW || layout == DATA_LAYOUT_NHWC);
    int c_idx = layout == DATA_LAYOUT_NCHW ? 1 : dims-1;

    MatShape newsize = *this;
    newsize.layout = DATA_LAYOUT_BLOCK;
    newsize.C = p[c_idx];
    newsize.p[newsize.dims++] = C0;
    newsize.p[c_idx] = (p[c_idx] + C0 - 1)/C0;

    return newsize;
}

MatShape MatShape::fromBlock(DataLayout newLayout) const
{
    CV_Assert(dims >= 4);
    CV_Assert(layout == DATA_LAYOUT_BLOCK);
    // C0 should be > 1 and be a power-of-2: 2, 4, 8, ...
    int C0 = p[dims-1];
    CV_Assert(C0 > 1 && (C0 & (C0-1)) == 0);
    CV_Assert(p[1] == (C + C0-1)/C0);
    CV_Assert(newLayout == DATA_LAYOUT_NCHW || newLayout == DATA_LAYOUT_NHWC);
    int c_idx = newLayout == DATA_LAYOUT_NCHW ? 1 : dims-2;

    MatShape newsize = *this;
    newsize.layout = newLayout;
    newsize.C = 0;
    newsize.p[c_idx] = C;
    newsize.dims--;

    return newsize;
}

MatShape MatShape::expand(const MatShape& another) const
{
    if (dims == 0)
        return another;
    if (another.dims == 0)
        return *this;

    if ((layout == DATA_LAYOUT_NCHW || layout == DATA_LAYOUT_NHWC) &&
        (another.layout == DATA_LAYOUT_NCHW || another.layout == DATA_LAYOUT_NHWC)) {
        CV_Assert(layout == another.layout);
    }
    // [TODO] support block layout
    CV_Assert(layout != DATA_LAYOUT_BLOCK && another.layout != DATA_LAYOUT_BLOCK);

    MatShape result;

    if (dims < 0 || another.dims < 0)
        return result;

    result = *this;
    result.dims = std::max(dims, another.dims);
    result.layout = layout == DATA_LAYOUT_UNKNOWN ? another.layout :
        layout == DATA_LAYOUT_ND && (another.layout == DATA_LAYOUT_NCHW ||
        another.layout == DATA_LAYOUT_NHWC) ? another.layout : layout;
    for (int i = result.dims-1; i >= 0; i--) {
        int i1 = i - (result.dims - dims);
        int i2 = i - (result.dims - another.dims);
        int sz1 = i1 < 0 ? 1 : p[i1];
        int sz2 = i2 < 0 ? 1 : another.p[i2];
        CV_Assert(sz1 == sz2 || sz1 == 1 || sz2 == 1);
        // [TODO] handle symbolic shapes
        result.p[i] = std::max(sz1, sz2);
    }
    return result;
}

MatShape::operator std::vector<int>() const
{
    if (dims < 0)
        return std::vector<int>(1, 0);
    return std::vector<int>(p, p + dims);
}

/////////////////////////// MatAllocator ////////////////////////////

void MatAllocator::map(UMatData*, AccessFlag) const
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
    CV_INSTRUMENT_REGION();

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
    CV_UNUSED(id);
    static DummyBufferPoolController dummy;
    return &dummy;
}

class StdMatAllocator CV_FINAL : public MatAllocator
{
public:
    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data0, size_t* step, AccessFlag /*flags*/, UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE
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

    bool allocate(UMatData* u, AccessFlag /*accessFlags*/, UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE
    {
        if(!u) return false;
        return true;
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
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

static
MatAllocator*& getDefaultAllocatorMatRef()
{
    static MatAllocator* g_matAllocator = Mat::getStdAllocator();
    return g_matAllocator;
}

MatAllocator* Mat::getDefaultAllocator()
{
    return getDefaultAllocatorMatRef();
}

void Mat::setDefaultAllocator(MatAllocator* allocator)
{
    getDefaultAllocatorMatRef() = allocator;
}

MatAllocator* Mat::getStdAllocator()
{
    CV_SINGLETON_LAZY_INIT(MatAllocator, new StdMatAllocator())
}

//==================================================================================================

bool MatSize::operator==(const MatSize& sz) const CV_NOEXCEPT
{
    int d = dims();
    int dsz = sz.dims();
    if( d != dsz )
        return false;
    if( d == 2 )
        return p[0] == sz.p[0] && p[1] == sz.p[1];

    for( int i = 0; i < d; i++ )
        if( p[i] != sz.p[i] )
            return false;
    return true;
}

void setSize( Mat& m, int _dims, const int* _sz, const size_t* _steps, bool autoSteps)
{
    CV_Assert( 0 <= _dims && _dims <= CV_MAX_DIM );
    if( m.dims != _dims )
    {
        if( m.step.p != m.step.buf && m.step.p != m.step.buf+1)
        {
            fastFree(m.step.p);
        }
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
        if( _dims > 2 )
        {
            m.step.p = (size_t*)fastMalloc(_dims*sizeof(m.step.p[0]) + (_dims+1)*sizeof(m.size.p[0]));
            m.size.p = (int*)(m.step.p + _dims) + 1;
            m.size.p[-1] = _dims;
            m.rows = m.cols = -1;
        }
    }

    m.dims = _dims;
    size_t esz = CV_ELEM_SIZE(m.flags), esz1 = CV_ELEM_SIZE1(m.flags), total = esz;
    if (_sz != 0) {
        for( int i = _dims-1; i >= 0; i-- )
        {
            int s = _sz[i];
            CV_Assert( s >= 0 );
            m.size.p[i] = s;

            if( _steps )
            {
                if (i < _dims-1)
                {
                    if (_steps[i] % esz1 != 0)
                    {
                        CV_Error_(Error::BadStep, ("Step %zu for dimension %d must be a multiple of esz1 %zu", _steps[i], i, esz1));
                    }

                    m.step.p[i] = _steps[i];
                }
                else
                {
                    m.step.p[i] = esz;
                }
            }
            else if( autoSteps )
            {
                m.step.p[i] = total;
                uint64 total1 = (uint64)total*s;
                if( (uint64)total1 != (size_t)total1 )
                    CV_Error( cv::Error::StsOutOfRange, "The total matrix size does not fit to \"size_t\" type" );
                total = (size_t)total1;
            }
        }
    }

    if( _dims < 2 )
    {
        m.cols = _dims >= 1 && _sz ? _sz[0] : 1;
        m.rows = 1;
        m.size.p = &m.cols;
        m.step.buf[0] = m.cols*esz;
        m.step.buf[1] = esz;
        m.step.p = &m.step.buf[1];
    }
}

int updateContinuityFlag(int flags, int dims, const int* size, const size_t* step)
{
    int i, j;
    if (dims <= 1)
        return flags | Mat::CONTINUOUS_FLAG;
    for( i = 0; i < dims; i++ )
    {
        if( size[i] > 1 )
            break;
    }

    uint64 t = (uint64)size[std::min(i, dims-1)]*CV_MAT_CN(flags);
    for( j = dims-1; j > i; j-- )
    {
        t *= size[j];
        if( step[j]*size[j] < step[j-1] )
            break;
    }

    if( j <= i && t == (uint64)(int)t )
        return flags | Mat::CONTINUOUS_FLAG;
    return flags & ~Mat::CONTINUOUS_FLAG;
}

void Mat::updateContinuityFlag()
{
    flags = cv::updateContinuityFlag(flags, dims, size.p, step.p);
}

void finalizeHdr(Mat& m)
{
    m.updateContinuityFlag();
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
            int lastdim = d > 0 ? d - 1 : 0;
            m.dataend = m.ptr() + m.size[lastdim]*m.step[lastdim];
            for( int i = 0; i < d-1; i++ )
                m.dataend += (m.size[i] - 1)*m.step[i];
        }
        else
            m.dataend = m.datalimit;
    }
    else
        m.dataend = m.datalimit = 0;
}

//======================================= Mat ======================================================

Mat::Mat() CV_NOEXCEPT
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{}

Mat::Mat(int _rows, int _cols, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_rows, _cols, _type);
}

Mat::Mat(int _rows, int _cols, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_rows, _cols, _type);
    *this = _s;
}

Mat::Mat(Size _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create( _sz.height, _sz.width, _type );
}

Mat::Mat(Size _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz.height, _sz.width, _type);
    *this = _s;
}

Mat::Mat(int _dims, const int* _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_dims, _sz, _type);
}

Mat::Mat(int _dims, const int* _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_dims, _sz, _type);
    *this = _s;
}

Mat::Mat(const std::vector<int>& _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz, _type);
}

Mat::Mat(const std::vector<int>& _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz, _type);
    *this = _s;
}

Mat::Mat(const MatShape& _shape, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_shape, _type);
}

Mat::Mat(std::initializer_list<int> _shape, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_shape, _type);
}

Mat::Mat(const MatShape& _shape, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_shape, _type);
    *this = _s;
}

Mat::Mat(std::initializer_list<int> _shape, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_shape, _type);
    *this = _s;
}

Mat::Mat(const Mat& m)
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), data(m.data),
      datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit), allocator(m.allocator),
      u(m.u), size(&rows), step(0)
{
    if( u )
        CV_XADD(&u->refcount, 1);
    if( m.dims <= 2 )
    {
        int _1d = m.dims <= 1;
        size.p = &rows + _1d;
        step.p = &step.buf[_1d];
        step.buf[0] = m.step.buf[0];
        step.buf[1] = m.step.buf[1];
    }
    else
    {
        dims = 0;
        copySize(m);
    }
}

Mat::Mat(int _rows, int _cols, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), dims(2), rows(_rows), cols(_cols),
      data((uchar*)_data), datastart((uchar*)_data), dataend(0), datalimit(0),
      allocator(0), u(0), size(&rows)
{
    CV_Assert(total() == 0 || data != NULL);

    size_t esz = CV_ELEM_SIZE(_type), esz1 = CV_ELEM_SIZE1(_type);
    size_t minstep = cols * esz;
    if( _step == AUTO_STEP )
    {
        _step = minstep;
    }
    else
    {
        CV_Assert( _step >= minstep );
        if (_step % esz1 != 0)
        {
            CV_Error(Error::BadStep, "Step must be a multiple of esz1");
        }
    }
    step.buf[0] = _step;
    step.buf[1] = esz;
    datalimit = datastart + _step * rows;
    dataend = datalimit - _step + minstep;
    updateContinuityFlag();
}

Mat::Mat(Size _sz, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), dims(2), rows(_sz.height), cols(_sz.width),
      data((uchar*)_data), datastart((uchar*)_data), dataend(0), datalimit(0),
      allocator(0), u(0), size(&rows)
{
    CV_Assert(total() == 0 || data != NULL);

    size_t esz = CV_ELEM_SIZE(_type), esz1 = CV_ELEM_SIZE1(_type);
    size_t minstep = cols*esz;
    if( _step == AUTO_STEP )
    {
        _step = minstep;
    }
    else
    {
        CV_CheckGE(_step, minstep, "");

        if (_step % esz1 != 0)
        {
            CV_Error(Error::BadStep, "Step must be a multiple of esz1");
        }
    }
    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step*rows;
    dataend = datalimit - _step + minstep;
    updateContinuityFlag();
}


Mat::~Mat()
{
    CV_Assert(dummy == 153 && step.buf[2] == 153);
    release();
    if( step.p != step.buf && step.p != step.buf+1 )
        fastFree(step.p);
}

Mat& Mat::operator=(const Mat& m)
{
    if( this != &m )
    {
        if( m.u )
            CV_XADD(&m.u->refcount, 1);
        release();
        flags = m.flags;

        if( dims <= 2 && m.dims <= 2 )
        {
            int _1d = m.dims < 2;
            dims = m.dims;
            rows = m.rows;
            cols = m.cols;
            step.p = &step.buf[_1d];
            step.buf[0] = m.step.buf[0];
            step.buf[1] = m.step.buf[1];
            size.p = &rows + _1d;
        }
        else
            copySize(m);
        data = m.data;
        datastart = m.datastart;
        dataend = m.dataend;
        datalimit = m.datalimit;
        allocator = m.allocator;
        u = m.u;
    }
    return *this;
}

Mat Mat::clone() const
{
    Mat m;
    copyTo(m);
    return m;
}

void Mat::assignTo( Mat& m, int _type ) const
{
    if( _type < 0 )
        m = *this;
    else
        convertTo(m, _type);
}

void Mat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    if( dims <= 2 && rows == _rows && cols == _cols && type() == _type && data )
        return;
    int sz[] = {_rows, _cols};
    create(2, sz, _type);
}

void Mat::create(Size _sz, int _type)
{
    create(_sz.height, _sz.width, _type);
}

void Mat::createSameSize(InputArray m, int type)
{
    _OutputArray(*this).createSameSize(m, type);
}

void Mat::fit(int _dims, const int* _sizes, int _type)
{
    size_t oldTotalBytes = u ? u->size : 0;
    size_t esz = CV_ELEM_SIZE(_type), newTotal = _dims >= 0;
    for (int i = 0; i < _dims; i++)
        newTotal *= _sizes[i];
    size_t newTotalBytes = newTotal*esz;
    if (newTotalBytes > 0 && (!isContinuous() ||
                              newTotalBytes > oldTotalBytes ||
                              data != datastart)) {
        create(_dims, _sizes, _type);
    } else {
        flags = (flags & ~Mat::TYPE_MASK) | CV_MAT_TYPE(_type);
        int _dummy_size = 0;
        setSize(*this, (_dims >= 0 ? _dims : 1), (_dims >= 0 ? _sizes : &_dummy_size), nullptr, true);
        finalizeHdr(*this);
    }
}

void Mat::fit(const std::vector<int>& _shape, int _type)
{
    fit((int)_shape.size(), _shape.data(), _type);
}

void Mat::fit(const MatShape& _shape, int _type)
{
    fit(_shape.dims, _shape.p, _type);
}

void Mat::fit(std::initializer_list<int> _shape, int _type)
{
    int new_shape[MatShape::MAX_DIMS];
    int new_ndims = (int)_shape.size();
    CV_Assert(new_ndims <= MatShape::MAX_DIMS);
    auto it = _shape.begin();
    for (int i = 0; i < new_ndims; i++, ++it)
        new_shape[i] = *it;
    fit(new_ndims, new_shape, _type);
}

void Mat::fit(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    int sz[] = {_rows, _cols};
    fit(2, sz, _type);
}

void Mat::fit(Size _sz, int _type)
{
    fit(_sz.height, _sz.width, _type);
}

void Mat::fitSameSize(InputArray m, int _type)
{
    int _sizes[CV_MAX_DIM];
    int _dims = m.sizend(_sizes);
    fit(_dims, _sizes, _type);
}

void Mat::addref()
{
    if( u )
        CV_XADD(&u->refcount, 1);
}

void Mat::release()
{
    if( u && CV_XADD(&u->refcount, -1) == 1 )
        deallocate();
    u = NULL;
    datastart = dataend = datalimit = data = 0;
    for(int i = 0; i < dims; i++)
        size.p[i] = 0;
#ifdef _DEBUG
    flags = MAGIC_VAL;
    if(step.p != step.buf && step.p != step.buf+1)
    {
        fastFree(step.p);
        step.p = step.buf;
        size.p = &rows;
    }
    dims = rows = cols = 0;
#endif
}

size_t Mat::step1(int i) const
{
    return step.p[i] / elemSize1();
}

bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

size_t Mat::total() const
{
    if( dims <= 2 )
        return (size_t)rows * cols;
    size_t p = 1;
    for( int i = 0; i < dims; i++ )
        p *= size[i];
    return p;
}

size_t Mat::total(int startDim, int endDim) const
{
    CV_Assert( 0 <= startDim && startDim <= endDim);
    size_t p = 1;
    int endDim_ = endDim <= dims ? endDim : dims;
    for( int i = startDim; i < endDim_; i++ )
        p *= size[i];
    return p;
}

MatShape Mat::shape() const
{
    return dims == 0 && data == 0 ? MatShape() : MatShape(dims, size.p);
}

Mat::Mat(Mat&& m) CV_NOEXCEPT
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), data(m.data),
      datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit), allocator(m.allocator),
      u(m.u), size(&rows)
{
    if (m.dims <= 2)  // move new step/size info
    {
        int _1d = dims <= 1;
        step.p = &step.buf[_1d];
        size.p = &rows + _1d;
        step.buf[0] = m.step.buf[0];
        step.buf[1] = m.step.buf[1];
    }
    else
    {
        CV_Assert(m.step.p != m.step.buf && m.step.p != m.step.buf+1);
        step.p = m.step.p;
        size.p = m.size.p;
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
    }
    m.flags = MAGIC_VAL; m.dims = m.rows = m.cols = 0;
    m.data = NULL; m.datastart = NULL; m.dataend = NULL; m.datalimit = NULL;
    m.allocator = NULL;
    m.u = NULL;
}


Mat& Mat::operator=(Mat&& m)
{
    if (this == &m)
      return *this;

    release();
    flags = m.flags; dims = m.dims; rows = m.rows; cols = m.cols; data = m.data;
    datastart = m.datastart; dataend = m.dataend; datalimit = m.datalimit; allocator = m.allocator;
    u = m.u;
    if (step.p != step.buf && step.p != step.buf+1) // release self step/size
    {
        fastFree(step.p);
    }
    step.p = step.buf;
    size.p = &rows;
    if (m.dims <= 2) // move new step/size info
    {
        int _1d = dims <= 1;
        step.buf[0] = m.step.buf[0];
        step.buf[1] = m.step.buf[1];
        step.p = &step.buf[_1d];
        size.p = &rows + _1d;
    }
    else
    {
        CV_Assert(m.step.p != m.step.buf && m.step.p != m.step.buf+1);
        step.p = m.step.p;
        size.p = m.size.p;
    }
    m.step.p = m.step.buf;
    m.size.p = &m.rows;
    m.flags = MAGIC_VAL; m.dims = m.rows = m.cols = 0;
    m.data = NULL; m.datastart = NULL; m.dataend = NULL; m.datalimit = NULL;
    m.allocator = NULL;
    m.u = NULL;
    return *this;
}


void Mat::create(int d0, const int* _sizes, int _type)
{
    int sz1 = 1, d = d0;
    int i;
    if (d == 0) {
        d = 1;
        _sizes = (const int*)&sz1;
    }
    CV_Assert(0 <= d && d <= CV_MAX_DIM && _sizes);
    _type = CV_MAT_TYPE(_type);

    if( data && d == dims && _type == type() )
    {
        for( i = 0; i < d; i++ )
            if( size[i] != _sizes[i] )
                break;
        if( i == d )
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
        if(!a)
            a = a0;
        try
        {
            u = a->allocate(dims, size, _type, 0, step.p, ACCESS_RW /* ignored */, USAGE_DEFAULT);
            CV_Assert(u != 0);
            allocator = a;
        }
        catch (...)
        {
            if (a == a0)
                throw;
            u = a0->allocate(dims, size, _type, 0, step.p, ACCESS_RW /* ignored */, USAGE_DEFAULT);
            CV_Assert(u != 0);
            allocator = a0;
        }
        CV_Assert( step[dims-1] == (size_t)CV_ELEM_SIZE(flags) );
    }

    addref();
    finalizeHdr(*this);
    dims = d0;
}

void Mat::create(const std::vector<int>& _sizes, int _type)
{
    create((int)_sizes.size(), _sizes.data(), _type);
}

void Mat::create(const MatShape& _shape, int _type)
{
    if (_shape.dims < 0) {
        release();
        return;
    }
    create(_shape.dims, _shape.p, _type);
}

void Mat::create(std::initializer_list<int> _shape, int _type)
{
    int new_shape[MatShape::MAX_DIMS];
    int new_ndims = (int)_shape.size();
    CV_Assert(new_ndims <= MatShape::MAX_DIMS);
    auto it = _shape.begin();
    for (int i = 0; i < new_ndims; i++, ++it)
        new_shape[i] = *it;

    create(new_ndims, new_shape, _type);
}

void Mat::copySize(const Mat& m)
{
    setSize(*this, m.dims, 0, 0);
    step.buf[0] = m.step.buf[0];
    step.buf[1] = m.step.buf[1];
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
    CV_Assert( m.dims >= 2 || (m.dims == 1 && (_rowRange == Range::all() || _rowRange == Range(0, m.rows))));
    if( m.dims > 2 )
    {
        AutoBuffer<Range> rs(m.dims);
        rs[0] = _rowRange;
        rs[1] = _colRange;
        for( int i = 2; i < m.dims; i++ )
            rs[i] = Range::all();
        *this = m(rs.data());
        return;
    }

    *this = m;
    try
    {
        if( _rowRange != Range::all() && _rowRange != Range(0,rows) )
        {
            CV_Assert( 0 <= _rowRange.start && _rowRange.start <= _rowRange.end
                       && _rowRange.end <= m.rows );
            rows = _rowRange.size();
            data += step*_rowRange.start;
            flags |= SUBMATRIX_FLAG;
        }

        if( _colRange != Range::all() && _colRange != Range(0,cols) )
        {
            CV_Assert( 0 <= _colRange.start && _colRange.start <= _colRange.end
                       && _colRange.end <= m.cols );
            cols = _colRange.size();
            data += _colRange.start*elemSize();
            flags |= SUBMATRIX_FLAG;
        }
    }
    catch(...)
    {
        release();
        throw;
    }

    updateContinuityFlag();

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

    size_t esz = CV_ELEM_SIZE(flags);
    data += roi.x*esz;
    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols &&
              0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );
    if( roi.width < m.cols || roi.height < m.rows )
        flags |= SUBMATRIX_FLAG;

    step[0] = m.step[0]; step[1] = esz;
    updateContinuityFlag();

    addref();
    if( rows <= 0 || cols <= 0 )
    {
        rows = cols = 0;
        release();
    }
}


Mat::Mat(int _dims, const int* _sizes, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
{
    flags |= CV_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    if (_dims > 0 || _data != nullptr)
        setSize(*this, _dims, _sizes, _steps, true);
    finalizeHdr(*this);
}


Mat::Mat(const std::vector<int>& _sizes, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
{
    flags |= CV_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    int _dims = (int)_sizes.size();
    if (_dims > 0 || _data != nullptr)
        setSize(*this, _dims, _sizes.data(), _steps, true);
    finalizeHdr(*this);
}

Mat::Mat(const MatShape& _shape, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
{
    flags |= CV_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    if (_shape.dims >= 0) {
        setSize(*this, _shape.dims, _shape.p, _steps, true);
    }
    else {
        CV_Assert(!data);
    }
    finalizeHdr(*this);
}

Mat::Mat(std::initializer_list<int> _shape, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows)
{
    int new_shape[MatShape::MAX_DIMS];
    int _dims = (int)_shape.size();
    CV_Assert(_dims <= MatShape::MAX_DIMS);
    auto it = _shape.begin();
    for (int i = 0; i < _dims; i++, ++it)
        new_shape[i] = *it;

    flags |= CV_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    if (_dims > 0 || _data != nullptr)
        setSize(*this, _dims, new_shape, _steps, true);
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
    updateContinuityFlag();
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
    updateContinuityFlag();
}


Mat Mat::diag(int d) const
{
    CV_Assert( dims == 2 );
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

    m.updateContinuityFlag();

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
    }
}


void Mat::push_back_(const void* elem)
{
    size_t r = size.p[0];
    if( isSubmatrix() || dataend + step.p[0] > datalimit )
        reserve( std::max(r + 1, (r*3+1)/2) );

    size_t esz = elemSize();
    memcpy(data + r*step.p[0], elem, esz);
    size.p[0] = int(r + 1);
    dataend += step.p[0];
    uint64 tsz = size.p[0];
    for( int i = 1; i < dims; i++ )
        tsz *= size.p[i];
    if( esz < step.p[0] || tsz != (uint64)(int)tsz )
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
    size_t r = size.p[0];
    size_t delta = elems.size.p[0];
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
    size.p[0] = int(r);
    if( !eq )
        CV_Error(cv::Error::StsUnmatchedSizes, "Pushed vector length is not equal to matrix row length");
    if( type() != elems.type() )
        CV_Error(cv::Error::StsUnmatchedFormats, "Pushed vector type is not the same as matrix type");

    if( isSubmatrix() || dataend + step.p[0]*delta > datalimit )
        reserve( std::max(r + delta, (r*3+1)/2) );

    size.p[0] += int(delta);
    dataend += step.p[0]*delta;

    //updateContinuityFlag(*this);

    if( isContinuous() && elems.isContinuous() )
        memcpy(data + r*step.p[0], elems.data, elems.total()*elems.elemSize());
    else
    {
        Mat part = rowRange(int(r), int(r + delta));
        elems.copyTo(part);
    }
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

    if (dims == 1) {
        data += (col1 - ofs.x)*(std::ptrdiff_t)esz;
        cols = col2 - col1;
        size.p[0] = cols;
    } else {
        data += (row1 - ofs.y)*(std::ptrdiff_t)step + (col1 - ofs.x)*(std::ptrdiff_t)esz;
        rows = row2 - row1; cols = col2 - col1;
        size.p[0] = rows; size.p[1] = cols;
        updateContinuityFlag();
    }
    return *this;
}

Mat Mat::reshape(int new_cn, int new_rows) const
{
    int cn = channels();
    Mat hdr = *this;

    if( new_cn == 0 )
        new_cn = cn;

    if( dims > 2 )
    {
        if( new_rows == 0 )
        {
            // special case: just change the number of channnels; retain the same shape,
            // except for the last, innermost dimension
            CV_Assert(size[dims-1]*cn % new_cn == 0);
            hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
            hdr.step[dims-1] = CV_ELEM_SIZE(hdr.flags);
            hdr.size[dims-1] = hdr.size[dims-1]*cn / new_cn;
            return hdr;
        }
        CV_Assert( new_rows > 0 );
        int sz[] = { new_rows, (int)(total()*cn/new_rows) };
        return reshape(new_cn, 2, sz);
    }

    int total_width = cols * cn;

    if( (new_cn > total_width || total_width % new_cn != 0) && new_rows == 0 )
        new_rows = rows * total_width / new_cn;

    if( new_rows != 0 && new_rows != rows )
    {
        int total_size = total_width * rows;
        if( !isContinuous() )
            CV_Error( cv::Error::BadStep,
            "The matrix is not continuous, thus its number of rows can not be changed" );

        total_width = total_size / new_rows;

        if( total_width * new_rows != total_size )
            CV_Error( cv::Error::StsBadArg, "The total number of matrix elements "
                                    "is not divisible by the new number of rows" );

        hdr.rows = new_rows;
        hdr.step.buf[0] = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if( new_width * new_cn != total_width )
        CV_Error( cv::Error::BadNumChannels,
        "The total width is not divisible by the new number of channels" );

    hdr.dims = 2;
    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
    hdr.step.buf[1] = CV_ELEM_SIZE(hdr.flags);
    hdr.step.p = &hdr.step.buf[0];
    hdr.size.p = &hdr.rows;
    return hdr;
}

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
        CV_Assert(_cn >= 0 && _newndims >= 0 && _newndims <= CV_MAX_DIM && (_newndims == 0 || _newsz != 0));

        if (_cn == 0)
            _cn = this->channels();
        else
            CV_Assert(_cn <= CV_CN_MAX);

        size_t total_elem1_ref = this->total() * this->channels();
        size_t total_elem1 = _cn;

        AutoBuffer<int, 4> newsz_buf( (size_t)_newndims );

        int m1_idx = -1;

        for (int i = 0; i < _newndims; i++)
        {
            if (_newsz[i] >= 0) {
                newsz_buf[i] = _newsz[i];
                total_elem1 *= (size_t)newsz_buf[i];
            } else {
                if (m1_idx >= 0)
                    CV_Error(cv::Error::StsBadSize, "More than one '-1' occured in the new shape");
                m1_idx = i;
            }
        }

        if (m1_idx >= 0) {
            if (total_elem1 == 0) {
                CV_Assert(total_elem1_ref == 0);
                total_elem1 = 1;
            }
            CV_Assert(total_elem1_ref % total_elem1 == 0);
            newsz_buf[m1_idx] = (int)(total_elem1_ref / total_elem1);
        } else if (total_elem1 != total_elem1_ref) {
            CV_Error(cv::Error::StsUnmatchedSizes, "Requested and source matrices have different count of elements");
        }

        Mat hdr = *this;
        hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((_cn-1) << CV_CN_SHIFT);
        setSize(hdr, _newndims, newsz_buf.data(), NULL, true);

        return hdr;
    }

    CV_Error(cv::Error::StsNotImplemented, "Reshaping of n-dimensional non-continuous matrices is not supported yet");
    // TBD
}

Mat Mat::reshape(int _cn, const std::vector<int>& _newshape) const
{
    int newdims = (int)_newshape.size();
    if(newdims == 0 && empty())
    {
        return *this;
    }

    return reshape(_cn, newdims, newdims > 0 ? &_newshape[0] : 0);
}

Mat Mat::reshape(int _cn, const MatShape& _newshape) const
{
    if (_newshape.dims < 0) {
        int newshape[] = {0};
        return reshape(_cn, 1, newshape);
    }
    return reshape(_cn, _newshape.dims, _newshape.p);
}

Mat Mat::reshape(int _cn, std::initializer_list<int> newshape_) const
{
    int newshape[MatShape::MAX_DIMS];
    size_t i, newshape_dims = newshape_.size();
    CV_Assert(newshape_dims <= (size_t)MatShape::MAX_DIMS);
    auto it = newshape_.begin();
    for (i = 0; i < newshape_dims; i++, ++it)
        newshape[i] = *it;
    return reshape(_cn, (int)newshape_dims, newshape);
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
        ((dims <= 2 && (((rows == 1 || cols == 1) && channels() == _elemChannels) ||
                        (cols == _elemChannels && channels() == 1))) ||
        (dims == 3 && channels() == 1 && size.p[2] == _elemChannels && (size.p[0] == 1 || size.p[1] == 1) &&
         (isContinuous() || step.p[1] == step.p[2]*size.p[2])))
    ? (int)(total()*channels()/_elemChannels) : -1;
}


static inline Size getContinuousSize_(int flags, int cols, int rows, int widthScale)
{
    int64 sz = (int64)cols * rows * widthScale;
    bool has_int_overflow = sz >= INT_MAX;
    bool isContiguous = (flags & Mat::CONTINUOUS_FLAG) != 0;
    return (isContiguous && !has_int_overflow)
            ? Size((int)sz, 1)
            : Size(cols * widthScale, rows);
}

Size getContinuousSize2D(Mat& m1, int widthScale)
{
    CV_CheckLE(m1.dims, 2, "");
    return getContinuousSize_(m1.flags,
                              m1.cols, m1.rows, widthScale);
}
Size getContinuousSize2D(Mat& m1, Mat& m2, int widthScale)
{
    CV_CheckLE(m1.dims, 2, "");
    CV_CheckLE(m2.dims, 2, "");
    const Size sz1 = m1.size();
    if (sz1 != m2.size())  // reshape all matrixes to the same size (#4159)
    {
        size_t total_sz = m1.total();
        CV_CheckEQ(total_sz, m2.total(), "");
        bool is_m1_vector = m1.cols == 1 || m1.rows == 1;
        bool is_m2_vector = m2.cols == 1 || m2.rows == 1;
        CV_Assert(is_m1_vector); CV_Assert(is_m2_vector);
        int total = (int)total_sz;  // vector-column
        bool isContiguous = ((m1.flags & m2.flags) & Mat::CONTINUOUS_FLAG) != 0;
        bool has_int_overflow = ((int64)total_sz * widthScale) >= INT_MAX;
        if (isContiguous && !has_int_overflow)
            total = 1; // vector-row
        m1 = m1.reshape(0, total);
        m2 = m2.reshape(0, total);
        CV_Assert(m1.cols == m2.cols && m1.rows == m2.rows);
        return Size(m1.cols * widthScale, m1.rows);
    }
    return getContinuousSize_(m1.flags & m2.flags,
                              m1.cols, m1.rows, widthScale);
}

Size getContinuousSize2D(Mat& m1, Mat& m2, Mat& m3, int widthScale)
{
    CV_CheckLE(m1.dims, 2, "");
    CV_CheckLE(m2.dims, 2, "");
    CV_CheckLE(m3.dims, 2, "");
    const Size sz1 = m1.size();
    if (sz1 != m2.size() || sz1 != m3.size())  // reshape all matrixes to the same size (#4159)
    {
        size_t total_sz = m1.total();
        CV_CheckEQ(total_sz, m2.total(), "");
        CV_CheckEQ(total_sz, m3.total(), "");
        bool is_m1_vector = m1.cols == 1 || m1.rows == 1;
        bool is_m2_vector = m2.cols == 1 || m2.rows == 1;
        bool is_m3_vector = m3.cols == 1 || m3.rows == 1;
        CV_Assert(is_m1_vector); CV_Assert(is_m2_vector); CV_Assert(is_m3_vector);
        int total = (int)total_sz;  // vector-column
        bool isContiguous = ((m1.flags & m2.flags & m3.flags) & Mat::CONTINUOUS_FLAG) != 0;
        bool has_int_overflow = ((int64)total_sz * widthScale) >= INT_MAX;
        if (isContiguous && !has_int_overflow)
            total = 1; // vector-row
        m1 = m1.reshape(0, total);
        m2 = m2.reshape(0, total);
        m3 = m3.reshape(0, total);
        CV_Assert(m1.cols == m2.cols && m1.rows == m2.rows && m1.cols == m3.cols && m1.rows == m3.rows);
        return Size(m1.cols * widthScale, m1.rows);
    }
    return getContinuousSize_(m1.flags & m2.flags & m3.flags,
                              m1.cols, m1.rows, widthScale);
}

} // cv::
