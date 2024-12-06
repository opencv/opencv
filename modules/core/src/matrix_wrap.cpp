// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/mat.hpp"

namespace cv {

/*************************************************************************************************\
                                        Input/Output Array
\*************************************************************************************************/

Mat _InputArray::getMat_(int i) const
{
    _InputArray::KindFlag k = kind();
    AccessFlag accessFlags = flags & ACCESS_MASK;

    if( k == MAT )
    {
        const Mat* m = getObj<Mat>();
        if( i < 0 )
            return *m;
        return m->row(i);
    }

    if( k == UMAT )
    {
        const UMat* m = getObj<UMat>();
        if( i < 0 )
            return m->getMat(accessFlags);
        return m->getMat(accessFlags).row(i);
    }

    if (k == MATX)
    {
        CV_Assert( i < 0 );
        return Mat(sz, flags, obj);
    }

    if( k == STD_VECTOR )
    {
        CV_Assert( i < 0 );
        int t = CV_MAT_TYPE(flags);
        const std::vector<uchar>& v = *getObj<std::vector<uchar>>();
        int v_size = size().width;

        return !v.empty() ? Mat(1, &v_size, t, (void*)&v[0]) : Mat();
    }

    if( k == STD_BOOL_VECTOR )
    {
        CV_Assert( i < 0 );
        int t = CV_8U;
        const std::vector<bool>& v = *getObj<std::vector<bool>>();
        int j, n = (int)v.size();
        if( n == 0 )
            return Mat();
        Mat m(1, &n, t);
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
        const std::vector<std::vector<uchar> >& vv = *getObj<std::vector<std::vector<uchar>>>();
        CV_Assert( 0 <= i && i < (int)vv.size() );
        const std::vector<uchar>& v = vv[i];
        int v_size = size(i).width;

        return !v.empty() ? Mat(1, &v_size, t, (void*)&v[0]) : Mat();
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& v = *getObj<std::vector<Mat>>();
        CV_Assert( 0 <= i && i < (int)v.size() );

        return v[i];
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = getObj<Mat>();
        CV_Assert( 0 <= i && i < sz.height );

        return v[i];
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *getObj<std::vector<UMat>>();
        CV_Assert( 0 <= i && i < (int)v.size() );

        return v[i].getMat(accessFlags);
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call mapHost/unmapHost methods for ogl::Buffer object");
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call download method for cuda::GpuMat object");
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );

        const cuda::HostMem* cuda_mem = getObj<cuda::HostMem>();

        return cuda_mem->createMatHeader();
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

UMat _InputArray::getUMat(int i) const
{
    _InputArray::KindFlag k = kind();
    AccessFlag accessFlags = flags & ACCESS_MASK;

    if( k == UMAT )
    {
        const UMat* m = getObj<UMat>();
        if( i < 0 )
            return *m;
        return m->row(i);
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *getObj<std::vector<UMat>>();
        CV_Assert( 0 <= i && i < (int)v.size() );

        return v[i];
    }

    if( k == MAT )
    {
        const Mat* m = getObj<Mat>();
        if( i < 0 )
            return m->getUMat(accessFlags);
        return m->row(i).getUMat(accessFlags);
    }

    return getMat(i).getUMat(accessFlags);
}

void _InputArray::getMatVector(std::vector<Mat>& mv) const
{
    _InputArray::KindFlag k = kind();
    AccessFlag accessFlags = flags & ACCESS_MASK;

    if( k == MAT )
    {
        const Mat& m = *getObj<Mat>();
        int n = (int)m.size[0];
        mv.resize(n);
        CV_Assert(m.dims >= 2);

        for( int i = 0; i < n; i++ )
            mv[i] = m.dims <= 2 ? Mat(1, m.cols, m.type(), (void*)m.ptr(i)) :
                Mat(m.dims-1, &m.size[1], m.type(), (void*)m.ptr(i), &m.step[1]);
        return;
    }

    if (k == MATX)
    {
        size_t n = sz.height, esz = CV_ELEM_SIZE(flags);
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = Mat(1, sz.width, CV_MAT_TYPE(flags), (uchar*)obj + esz*sz.width*i);
        return;
    }

    if( k == STD_VECTOR )
    {
        const std::vector<uchar>& v = *getObj<std::vector<uchar>>();

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
        const std::vector<std::vector<uchar> >& vv = *getObj<std::vector<std::vector<uchar>>>();
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
        const std::vector<Mat>& v = *getObj<std::vector<Mat>>();
        size_t n = v.size();
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = v[i];
        return;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = getObj<Mat>();
        size_t n = sz.height;
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = v[i];
        return;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *getObj<std::vector<UMat>>();
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
    _InputArray::KindFlag k = kind();
    AccessFlag accessFlags = flags & ACCESS_MASK;

    if( k == NONE )
    {
        umv.clear();
        return;
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& v = *getObj<std::vector<Mat>>();
        size_t n = v.size();
        umv.resize(n);

        for( size_t i = 0; i < n; i++ )
            umv[i] = v[i].getUMat(accessFlags);
        return;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = getObj<Mat>();
        size_t n = sz.height;
        umv.resize(n);

        for( size_t i = 0; i < n; i++ )
            umv[i] = v[i].getUMat(accessFlags);
        return;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& v = *getObj<std::vector<UMat>>();
        size_t n = v.size();
        umv.resize(n);

        for( size_t i = 0; i < n; i++ )
            umv[i] = v[i];
        return;
    }

    if( k == UMAT )
    {
        UMat& v = *getObj<UMat>();
        umv.resize(1);
        umv[0] = v;
        return;
    }
    if( k == MAT )
    {
        Mat& v = *getObj<Mat>();
        umv.resize(1);
        umv[0] = v.getUMat(accessFlags);
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

cuda::GpuMat _InputArray::getGpuMat() const
{
#ifdef HAVE_CUDA
    _InputArray::KindFlag k = kind();

    if (k == CUDA_GPU_MAT)
    {
        const cuda::GpuMat* d_mat = getObj<cuda::GpuMat>();
        return *d_mat;
    }

    if (k == CUDA_HOST_MEM)
    {
        const cuda::HostMem* cuda_mem = getObj<cuda::HostMem>();
        return cuda_mem->createGpuMatHeader();
    }

    if (k == OPENGL_BUFFER)
    {
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call mapDevice/unmapDevice methods for ogl::Buffer object");
    }

    if (k == NONE)
        return cuda::GpuMat();

    CV_Error(cv::Error::StsNotImplemented, "getGpuMat is available only for cuda::GpuMat and cuda::HostMem");
#else
    CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
}
void _InputArray::getGpuMatVector(std::vector<cuda::GpuMat>& gpumv) const
{
#ifdef HAVE_CUDA
    _InputArray::KindFlag k = kind();
    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        gpumv = *getObj<std::vector<cuda::GpuMat>>();
    }
#else
    CV_UNUSED(gpumv);
    CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
}
ogl::Buffer _InputArray::getOGlBuffer() const
{
    _InputArray::KindFlag k = kind();

    CV_Assert(k == OPENGL_BUFFER);

    const ogl::Buffer* gl_buf = getObj<ogl::Buffer>();
    return *gl_buf;
}

_InputArray::KindFlag _InputArray::kind() const
{
    KindFlag k = flags & KIND_MASK;
#if CV_VERSION_MAJOR < 5
    CV_DbgAssert(k != EXPR);
    CV_DbgAssert(k != STD_ARRAY);
#endif
    return k;
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
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        const Mat* m = getObj<Mat>();
        CV_Assert(m->dims <= 2);
        return Size(m->cols, m->rows);
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        const UMat* m = getObj<UMat>();
        CV_Assert(m->dims <= 2);
        return Size(m->cols, m->rows);
    }

    if (k == MATX)
    {
        CV_Assert( i < 0 );
        return sz;
    }

    if( k == STD_VECTOR )
    {
        CV_Assert( i < 0 );
        const std::vector<uchar>& v = *getObj<std::vector<uchar>>();
        const std::vector<int>& iv = *getObj<std::vector<int>>();
        size_t szb = v.size(), szi = iv.size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb/CV_ELEM_SIZE(flags)), 1);
    }

    if( k == STD_BOOL_VECTOR )
    {
        CV_Assert( i < 0 );
        const std::vector<bool>& v = *getObj<std::vector<bool>>();
        return Size((int)v.size(), 1);
    }

    if( k == NONE )
        return Size();

    if( k == STD_VECTOR_VECTOR )
    {
        const std::vector<std::vector<uchar> >& vv = *getObj<std::vector<std::vector<uchar>>>();
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );
        const std::vector<std::vector<int> >& ivv = *getObj<std::vector<std::vector<int>>>();

        size_t szb = vv[i].size(), szi = ivv[i].size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb/CV_ELEM_SIZE(flags)), 1);
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );

        return vv[i].size();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = getObj<Mat>();
        if( i < 0 )
            return sz.height==0 ? Size() : Size(sz.height, 1);
        CV_Assert( i < sz.height );

        return vv[i].size();
    }

    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
#ifdef HAVE_CUDA
        const std::vector<cuda::GpuMat>& vv = *getObj<std::vector<cuda::GpuMat>>();
        if (i < 0)
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert(i < (int)vv.size());
        return vv[i].size();
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );

        return vv[i].size();
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        const ogl::Buffer* buf = getObj<ogl::Buffer>();
        return buf->size();
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        const cuda::GpuMat* d_mat = getObj<cuda::GpuMat>();
        return d_mat->size();
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );
        const cuda::HostMem* cuda_mem = getObj<cuda::HostMem>();
        return cuda_mem->size();
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

int _InputArray::sizend(int* arrsz, int i) const
{
    int j, d = 0;
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        ;
    else if( k == MAT )
    {
        CV_Assert( i < 0 );
        const Mat& m = *getObj<Mat>();
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == UMAT )
    {
        CV_Assert( i < 0 );
        const UMat& m = *getObj<UMat>();
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == STD_VECTOR_MAT && i >= 0 )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        CV_Assert( i < (int)vv.size() );
        const Mat& m = vv[i];
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == STD_ARRAY_MAT && i >= 0 )
    {
        const Mat* vv = getObj<Mat>();
        CV_Assert( i < sz.height );
        const Mat& m = vv[i];
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == STD_VECTOR_UMAT && i >= 0 )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        CV_Assert( i < (int)vv.size() );
        const UMat& m = vv[i];
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if (k == STD_VECTOR && i < 0 )
    {
        Size sz2d = size();
        d = 1;
        if(arrsz)
        {
            arrsz[0] = sz2d.width;
        }
    }
    else
    {
        CV_CheckLE(dims(i), 2, "Not supported");
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

bool _InputArray::empty(int i) const
{
    _InputArray::KindFlag k = kind();
    if (i >= 0) {
        if (k == STD_VECTOR_MAT) {
            auto mv = reinterpret_cast<const std::vector<Mat>*>(obj);
            CV_Assert((size_t)i < mv->size());
            return mv->at(i).empty();
        }
        else if (k == STD_VECTOR_MAT) {
            auto umv = reinterpret_cast<const std::vector<UMat>*>(obj);
            CV_Assert((size_t)i < umv->size());
            return umv->at(i).empty();
        }
        else if (k == STD_VECTOR_VECTOR) {
            auto vv = reinterpret_cast<const std::vector<std::vector<int> >*>(obj);
            CV_Assert((size_t)i < vv->size());
            return vv->at(i).empty();
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
    return empty();
}

MatShape _InputArray::shape(int i) const
{
    int sizes[CV_MAX_DIM];
    int dims = sizend(sizes, i);

    if (dims == 0 && empty(i))
        return MatShape();
    return MatShape(dims, sizes);
}

bool _InputArray::sameSize(const _InputArray& arr) const
{
    _InputArray::KindFlag k1 = kind(), k2 = arr.kind();
    Size sz1;

    if( k1 == MAT )
    {
        const Mat* m = (getObj<Mat>());
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
        const UMat* m = (getObj<UMat>());
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
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return (getObj<Mat>())->dims;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return (getObj<UMat>())->dims;
    }

    if (k == MATX)
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == STD_VECTOR || k == STD_BOOL_VECTOR )
    {
        CV_Assert( i < 0 );
        return 1;
    }

    if( k == NONE )
        return 0;

    if( k == STD_VECTOR_VECTOR )
    {
        const std::vector<std::vector<uchar> >& vv = *getObj<std::vector<std::vector<uchar>>>();
        if( i < 0 )
            return 1;
        CV_Assert( i < (int)vv.size() );
        return 2;
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        if( i < 0 )
            return 1;
        CV_Assert( i < (int)vv.size() );

        return vv[i].dims;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = getObj<Mat>();
        if( i < 0 )
            return 1;
        CV_Assert( i < sz.height );

        return vv[i].dims;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
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
}

size_t _InputArray::total(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return (getObj<Mat>())->total();
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return (getObj<UMat>())->total();
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        if( i < 0 )
            return vv.size();

        CV_Assert( i < (int)vv.size() );
        return vv[i].total();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = getObj<Mat>();
        if( i < 0 )
            return sz.height;

        CV_Assert( i < sz.height );
        return vv[i].total();
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        if( i < 0 )
            return vv.size();

        CV_Assert( i < (int)vv.size() );
        return vv[i].total();
    }

    return size(i).area();
}

int _InputArray::type(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return (getObj<Mat>())->type();

    if( k == UMAT )
        return (getObj<UMat>())->type();

    if( k == MATX || k == STD_VECTOR || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return CV_MAT_TYPE(flags);

    if( k == NONE )
        return -1;

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
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
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
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
        const Mat* vv = getObj<Mat>();
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
#ifdef HAVE_CUDA
        const std::vector<cuda::GpuMat>& vv = *getObj<std::vector<cuda::GpuMat>>();
        if (vv.empty())
        {
            CV_Assert((flags & FIXED_TYPE) != 0);
            return CV_MAT_TYPE(flags);
        }
        CV_Assert(i < (int)vv.size());
        return vv[i >= 0 ? i : 0].type();
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }

    if( k == OPENGL_BUFFER )
        return (getObj<ogl::Buffer>())->type();

    if( k == CUDA_GPU_MAT )
        return (getObj<cuda::GpuMat>())->type();

    if( k == CUDA_HOST_MEM )
        return (getObj<cuda::HostMem>())->type();

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
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
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return (getObj<Mat>())->empty();

    if( k == UMAT )
        return (getObj<UMat>())->empty();

    if (k == MATX)
        return false;

    if( k == STD_VECTOR )
    {
        const std::vector<uchar>& v = *getObj<std::vector<uchar>>();
        return v.empty();
    }

    if( k == STD_BOOL_VECTOR )
    {
        const std::vector<bool>& v = *getObj<std::vector<bool>>();
        return v.empty();
    }

    if( k == NONE )
        return true;

    if( k == STD_VECTOR_VECTOR )
    {
        const std::vector<std::vector<uchar> >& vv = *getObj<std::vector<std::vector<uchar>>>();
        return vv.empty();
    }

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        return vv.empty();
    }

    if( k == STD_ARRAY_MAT )
    {
        return sz.height == 0;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        return vv.empty();
    }

    if( k == OPENGL_BUFFER )
        return (getObj<ogl::Buffer>())->empty();

    if( k == CUDA_GPU_MAT )
        return (getObj<cuda::GpuMat>())->empty();

    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *getObj<std::vector<cuda::GpuMat>>();
        return vv.empty();
    }

    if( k == CUDA_HOST_MEM )
        return (getObj<cuda::HostMem>())->empty();

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

bool _InputArray::isContinuous(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return i < 0 ? (getObj<Mat>())->isContinuous() : true;

    if( k == UMAT )
        return i < 0 ? (getObj<UMat>())->isContinuous() : true;

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return true;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return vv[i].isContinuous();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = getObj<Mat>();
        CV_Assert(i >= 0 && i < sz.height);
        return vv[i].isContinuous();
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return vv[i].isContinuous();
    }

    if( k == CUDA_GPU_MAT )
      return i < 0 ? (getObj<cuda::GpuMat>())->isContinuous() : true;

    CV_Error(cv::Error::StsNotImplemented, "Unknown/unsupported array type");
}

bool _InputArray::isSubmatrix(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return i < 0 ? (getObj<Mat>())->isSubmatrix() : false;

    if( k == UMAT )
        return i < 0 ? (getObj<UMat>())->isSubmatrix() : false;

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return false;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return vv[i].isSubmatrix();
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = getObj<Mat>();
        CV_Assert(i >= 0 && i < sz.height);
        return vv[i].isSubmatrix();
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return vv[i].isSubmatrix();
    }

    CV_Error(cv::Error::StsNotImplemented, "");
}

size_t _InputArray::offset(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        const Mat * const m = (getObj<Mat>());
        return (size_t)(m->ptr() - m->datastart);
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return (getObj<UMat>())->offset;
    }

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return 0;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        CV_Assert( i >= 0 && i < (int)vv.size() );

        return (size_t)(vv[i].ptr() - vv[i].datastart);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = getObj<Mat>();
        CV_Assert( i >= 0 && i < sz.height );
        return (size_t)(vv[i].ptr() - vv[i].datastart);
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return vv[i].offset;
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        const cuda::GpuMat * const m = (getObj<cuda::GpuMat>());
        return (size_t)(m->data - m->datastart);
    }

    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *getObj<std::vector<cuda::GpuMat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return (size_t)(vv[i].data - vv[i].datastart);
    }

    CV_Error(Error::StsNotImplemented, "");
}

size_t _InputArray::step(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return (getObj<Mat>())->step;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return (getObj<UMat>())->step;
    }

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return 0;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *getObj<std::vector<Mat>>();
        CV_Assert( i >= 0 && i < (int)vv.size() );
        return vv[i].step;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = getObj<Mat>();
        CV_Assert( i >= 0 && i < sz.height );
        return vv[i].step;
    }

    if( k == STD_VECTOR_UMAT )
    {
        const std::vector<UMat>& vv = *getObj<std::vector<UMat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return vv[i].step;
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        return (getObj<cuda::GpuMat>())->step;
    }
    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        const std::vector<cuda::GpuMat>& vv = *getObj<std::vector<cuda::GpuMat>>();
        CV_Assert(i >= 0 && (size_t)i < vv.size());
        return vv[i].step;
    }

    CV_Error(Error::StsNotImplemented, "");
}

void _InputArray::copyTo(const _OutputArray& arr) const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        arr.release();
    else if( k == MAT || k == MATX || k == STD_VECTOR || k == STD_BOOL_VECTOR )
    {
        Mat m = getMat();
        m.copyTo(arr);
    }
    else if( k == UMAT )
        (getObj<UMat>())->copyTo(arr);
#ifdef HAVE_CUDA
    else if (k == CUDA_GPU_MAT)
        (getObj<cuda::GpuMat>())->copyTo(arr);
#endif
    else
        CV_Error(Error::StsNotImplemented, "");
}

void _InputArray::copyTo(const _OutputArray& arr, const _InputArray & mask) const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        arr.release();
    else if( k == MAT || k == MATX || k == STD_VECTOR || k == STD_BOOL_VECTOR )
    {
        Mat m = getMat();
        m.copyTo(arr, mask);
    }
    else if( k == UMAT )
        (getObj<UMat>())->copyTo(arr, mask);
#ifdef HAVE_CUDA
    else if (k == CUDA_GPU_MAT)
        (getObj<cuda::GpuMat>())->copyTo(arr, mask);
#endif
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

void _OutputArray::create(Size _sz, int mtype, int i, bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    _InputArray::KindFlag k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<Mat>())->size.operator()() == _sz);
        CV_Assert(!fixedType() || (getObj<Mat>())->type() == mtype);
        (getObj<Mat>())->create(_sz, mtype);
        return;
    }
    if( k == UMAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<UMat>())->size.operator()() == _sz);
        CV_Assert(!fixedType() || (getObj<UMat>())->type() == mtype);
        (getObj<UMat>())->create(_sz, mtype);
        return;
    }
    if( k == CUDA_GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<cuda::GpuMat>())->size() == _sz);
        CV_Assert(!fixedType() || (getObj<cuda::GpuMat>())->type() == mtype);
#ifdef HAVE_CUDA
        (getObj<cuda::GpuMat>())->create(_sz, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<ogl::Buffer>())->size() == _sz);
        CV_Assert(!fixedType() || (getObj<ogl::Buffer>())->type() == mtype);
#ifdef HAVE_OPENGL
        (getObj<ogl::Buffer>())->create(_sz, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "OpenGL support is not enabled in this OpenCV build (missing HAVE_OPENGL)");
#endif
    }
    if( k == CUDA_HOST_MEM && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<cuda::HostMem>())->size() == _sz);
        CV_Assert(!fixedType() || (getObj<cuda::HostMem>())->type() == mtype);
#ifdef HAVE_CUDA
        (getObj<cuda::HostMem>())->create(_sz, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    int sizes[] = {_sz.height, _sz.width};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int _rows, int _cols, int mtype, int i, bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    _InputArray::KindFlag k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<Mat>())->size.operator()() == Size(_cols, _rows));
        CV_Assert(!fixedType() || (getObj<Mat>())->type() == mtype);
        (getObj<Mat>())->create(_rows, _cols, mtype);
        return;
    }
    if( k == UMAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<UMat>())->size.operator()() == Size(_cols, _rows));
        CV_Assert(!fixedType() || (getObj<UMat>())->type() == mtype);
        (getObj<UMat>())->create(_rows, _cols, mtype);
        return;
    }
    if( k == CUDA_GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<cuda::GpuMat>())->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || (getObj<cuda::GpuMat>())->type() == mtype);
#ifdef HAVE_CUDA
        (getObj<cuda::GpuMat>())->create(_rows, _cols, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<ogl::Buffer>())->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || (getObj<ogl::Buffer>())->type() == mtype);
#ifdef HAVE_OPENGL
        (getObj<ogl::Buffer>())->create(_rows, _cols, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "OpenGL support is not enabled in this OpenCV build (missing HAVE_OPENGL)");
#endif
    }
    if( k == CUDA_HOST_MEM && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || (getObj<cuda::HostMem>())->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || (getObj<cuda::HostMem>())->type() == mtype);
#ifdef HAVE_CUDA
        (getObj<cuda::HostMem>())->create(_rows, _cols, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    int sizes[] = {_rows, _cols};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int d, const int* sizes, int mtype, int i,
                          bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    int size0 = d > 0 ? sizes[0] : 1, size1 = d > 1 ? sizes[1] : 1;
    _InputArray::KindFlag k = kind();
    mtype = CV_MAT_TYPE(mtype);

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        Mat& m = *getObj<Mat>();
        CV_Assert(!(m.empty() && fixedType() && fixedSize()) && "Can't reallocate empty Mat with locked layout (probably due to misused 'const' modifier)");
        if (!m.empty() && d <= 2 && m.dims <= 2 &&
            m.type() == mtype &&
            ((m.rows == size0 && m.cols == size1) ||
            (allowTransposed && m.rows == size1 && m.cols == size0 && m.isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_DEPTH(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_CheckTypeEQ(m.type(), CV_MAT_TYPE(mtype), "Can't reallocate Mat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m.dims, d, "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m.size[j], sizes[j], "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
        }
        m.create(d, sizes, mtype);
        return;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        UMat& m = *getObj<UMat>();
        CV_Assert(!(m.empty() && fixedType() && fixedSize()) && "Can't reallocate empty UMat with locked layout (probably due to misused 'const' modifier)");
        if (!m.empty() && d <= 2 && m.dims <= 2 &&
            m.type() == mtype &&
            ((m.rows == size0 && m.cols == size1) ||
            (allowTransposed && m.rows == size1 && m.cols == size0 && m.isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_CheckTypeEQ(m.type(), CV_MAT_TYPE(mtype), "Can't reallocate UMat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m.dims, d, "Can't reallocate UMat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m.size[j], sizes[j], "Can't reallocate UMat with locked size (probably due to misused 'const' modifier)");
        }
        m.create(d, sizes, mtype);
        return;
    }

    if( k == MATX )
    {
        CV_Assert( i < 0 );
        int type0 = CV_MAT_TYPE(flags);
        CV_Assert( mtype == type0 || (CV_MAT_CN(mtype) == 1 && ((1 << type0) & fixedDepthMask) != 0) );
        CV_CheckLE(d, 2, "");
        Size requested_size(d == 2 ? sizes[1] : 1, d >= 1 ? sizes[0] : 1);
        if (sz.width == 1 || sz.height == 1)
        {
            // NB: 1D arrays assume allowTransposed=true (see #4159)
            int total_1d = std::max(sz.width, sz.height);
            CV_Check(requested_size, std::max(requested_size.width, requested_size.height) == total_1d, "");
        }
        else
        {
            if (!allowTransposed)
            {
                CV_CheckEQ(requested_size, sz, "");
            }
            else
            {
                CV_Check(requested_size,
                        (requested_size == sz || (requested_size.height == sz.width && requested_size.width == sz.height)),
                        "");
            }
        }
        return;
    }

    if( k == STD_VECTOR || k == STD_VECTOR_VECTOR )
    {
        CV_Assert( d <= 2 && (size0 == 1 || size1 == 1 || size0*size1 == 0) );
        size_t len = size0*size1 > 0 ? size0 + size1 - 1 : 0;
        std::vector<uchar>* v = getObj<std::vector<uchar>>();

        if( k == STD_VECTOR_VECTOR )
        {
            std::vector<std::vector<uchar> >& vv = *getObj<std::vector<std::vector<uchar>>>();
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
        case 20:
            ((std::vector<Vec<int, 5> >*)v)->resize(len);
            break;
        case 24:
            ((std::vector<Vec6i>*)v)->resize(len);
            break;
        case 28:
            ((std::vector<Vec<int, 7> >*)v)->resize(len);
            break;
        case 32:
            ((std::vector<Vec8i>*)v)->resize(len);
            break;
        case 36:
            ((std::vector<Vec<int, 9> >*)v)->resize(len);
            break;
        case 40:
            ((std::vector<Vec<int, 10> >*)v)->resize(len);
            break;
        case 44:
            ((std::vector<Vec<int, 11> >*)v)->resize(len);
            break;
        case 48:
            ((std::vector<Vec<int, 12> >*)v)->resize(len);
            break;
        case 52:
            ((std::vector<Vec<int, 13> >*)v)->resize(len);
            break;
        case 56:
            ((std::vector<Vec<int, 14> >*)v)->resize(len);
            break;
        case 60:
            ((std::vector<Vec<int, 15> >*)v)->resize(len);
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
            CV_Error_(cv::Error::StsBadArg, ("Vectors with element size %d are not supported. Please, modify OutputArray::create()\n", esz));
        }
        return;
    }

    if( k == NONE )
    {
        CV_Error(cv::Error::StsNullPtr, "create() called for the missing output array" );
    }

    if( k == STD_VECTOR_MAT )
    {
        std::vector<Mat>& v = *getObj<std::vector<Mat>>();

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
        Mat* v = getObj<Mat>();

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
        std::vector<UMat>& v = *getObj<std::vector<UMat>>();

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

    if ((k == CUDA_GPU_MAT || k == CUDA_HOST_MEM) && d <= 2 &&
        i < 0 && !allowTransposed && fixedDepthMask == 0)
    {
        create((d < 2 ? 1 : sizes[0]), (d < 1 ? 1 : sizes[d > 1]),
                mtype, i, allowTransposed, fixedDepthMask);
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _OutputArray::create(const MatShape& shape, int mtype, int i,
                          bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    if (shape.dims < 0) {
        release();
    } else {
        create(shape.dims, shape.p, mtype, i, allowTransposed, fixedDepthMask);
    }
}

void _OutputArray::createSameSize(const _InputArray& arr, int mtype) const
{
    int arrsz[CV_MAX_DIM], d = arr.sizend(arrsz);
    create(d, arrsz, mtype);
}

void _OutputArray::fit(int d, const int* sizes, int mtype, int i,
                       bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    int size0 = d > 0 ? sizes[0] : 1, size1 = d > 1 ? sizes[1] : 1;
    _InputArray::KindFlag k = kind();
    mtype = CV_MAT_TYPE(mtype);

    if( (k == MAT && i < 0) || (k == STD_VECTOR_MAT && i >= 0) )
    {
        Mat* m;
        if (k == MAT)
            m = getObj<Mat>();
        else {
            std::vector<Mat>& v = *getObj<std::vector<Mat>>();
            CV_Assert((size_t)i < v.size());
            m = &v[i];
        }
        CV_Assert(!(m->empty() && fixedType() && fixedSize()) && "Can't reallocate empty Mat with locked layout (probably due to misused 'const' modifier)");
        if (!m->empty() && d <= 2 && m->dims <= 2 &&
            m->type() == mtype &&
            ((m->rows == size0 && m->cols == size1) ||
             (allowTransposed && m->rows == size1 && m->cols == size0 && m->isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m->channels() && ((1 << CV_MAT_DEPTH(flags)) & fixedDepthMask) != 0 )
                mtype = m->type();
            else
                CV_CheckTypeEQ(m->type(), CV_MAT_TYPE(mtype), "Can't reallocate Mat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m->dims, d, "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m->size[j], sizes[j], "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
        }
        m->fit(d, sizes, mtype);
        return;
    }

    if( (k == UMAT && i < 0) || (k == STD_VECTOR_UMAT && i >= 0) )
    {
        UMat* m;
        if (k == UMAT)
            m = getObj<UMat>();
        else {
            std::vector<UMat>& v = *getObj<std::vector<UMat>>();
            CV_Assert((size_t)i < v.size());
            m = &v[i];
        }
        CV_Assert(!(m->empty() && fixedType() && fixedSize()) && "Can't reallocate empty Mat with locked layout (probably due to misused 'const' modifier)");
        if (!m->empty() && d <= 2 && m->dims <= 2 &&
            m->type() == mtype &&
            ((m->rows == size0 && m->cols == size1) ||
             (allowTransposed && m->rows == size1 && m->cols == size0 && m->isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m->channels() && ((1 << CV_MAT_DEPTH(flags)) & fixedDepthMask) != 0 )
                mtype = m->type();
            else
                CV_CheckTypeEQ(m->type(), CV_MAT_TYPE(mtype), "Can't reallocate Mat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m->dims, d, "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m->size[j], sizes[j], "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
        }
        m->fit(d, sizes, mtype);
        return;
    }

    create(d, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::release() const
{
    CV_Assert(!fixedSize());

    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        (getObj<Mat>())->release();
        return;
    }

    if( k == UMAT )
    {
        (getObj<UMat>())->release();
        return;
    }

    if( k == CUDA_GPU_MAT )
    {
#ifdef HAVE_CUDA
        (getObj<cuda::GpuMat>())->release();
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }

    if( k == CUDA_HOST_MEM )
    {
#ifdef HAVE_CUDA
        (getObj<cuda::HostMem>())->release();
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }

    if( k == OPENGL_BUFFER )
    {
#ifdef HAVE_OPENGL
        (getObj<ogl::Buffer>())->release();
        return;
#else
        CV_Error(Error::StsNotImplemented, "OpenGL support is not enabled in this OpenCV build (missing HAVE_OPENGL)");
#endif
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
        (getObj<std::vector<std::vector<uchar>>>())->clear();
        return;
    }

    if( k == STD_VECTOR_MAT )
    {
        (getObj<std::vector<Mat>>())->clear();
        return;
    }

    if( k == STD_VECTOR_UMAT )
    {
        (getObj<std::vector<UMat>>())->clear();
        return;
    }
    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
#ifdef HAVE_CUDA
        (getObj<std::vector<cuda::GpuMat>>())->clear();
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _OutputArray::clear() const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert(!fixedSize());
        (getObj<Mat>())->resize(0);
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
    _InputArray::KindFlag k = kind();
    if( i < 0 )
    {
        CV_Assert( k == MAT );
        return *getObj<Mat>();
    }

    CV_Assert( k == STD_VECTOR_MAT || k == STD_ARRAY_MAT );

    if( k == STD_VECTOR_MAT )
    {
        std::vector<Mat>& v = *getObj<std::vector<Mat>>();
        CV_Assert( i < (int)v.size() );
        return v[i];
    }
    else
    {
        Mat* v = getObj<Mat>();
        CV_Assert( 0 <= i && i < sz.height );
        return v[i];
    }
}

UMat& _OutputArray::getUMatRef(int i) const
{
    _InputArray::KindFlag k = kind();
    if( i < 0 )
    {
        CV_Assert( k == UMAT );
        return *getObj<UMat>();
    }
    else
    {
        CV_Assert( k == STD_VECTOR_UMAT );
        std::vector<UMat>& v = *getObj<std::vector<UMat>>();
        CV_Assert( i < (int)v.size() );
        return v[i];
    }
}

cuda::GpuMat& _OutputArray::getGpuMatRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert( k == CUDA_GPU_MAT );
    return *getObj<cuda::GpuMat>();
}
std::vector<cuda::GpuMat>& _OutputArray::getGpuMatVecRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert(k == STD_VECTOR_CUDA_GPU_MAT);
    return *getObj<std::vector<cuda::GpuMat>>();
}

ogl::Buffer& _OutputArray::getOGlBufferRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert( k == OPENGL_BUFFER );
    return *getObj<ogl::Buffer>();
}

cuda::HostMem& _OutputArray::getHostMemRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert( k == CUDA_HOST_MEM );
    return *getObj<cuda::HostMem>();
}

void _OutputArray::setTo(const _InputArray& arr, const _InputArray & mask) const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        ;
    else if (k == MAT || k == MATX || k == STD_VECTOR)
    {
        Mat m = getMat();
        m.setTo(arr, mask);
    }
    else if( k == UMAT )
        (getObj<UMat>())->setTo(arr, mask);
    else if( k == CUDA_GPU_MAT )
    {
#ifdef HAVE_CUDA
        Mat value = arr.getMat();
        CV_Assert( checkScalar(value, type(), arr.kind(), _InputArray::CUDA_GPU_MAT) );
        (getObj<cuda::GpuMat>())->setTo(Scalar(Vec<double, 4>(value.ptr<double>())), mask);
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    else
        CV_Error(Error::StsNotImplemented, "");
}

void _OutputArray::setZero() const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        ;
    else if (k == MAT || k == MATX || k == STD_VECTOR)
    {
        Mat m = getMat();
        m.setZero();
    }
    else
    {
        setTo(Scalar::all(0), noArray());
    }
}

void _OutputArray::assign(const UMat& u) const
{
    _InputArray::KindFlag k = kind();
    if (k == UMAT)
    {
        *getObj<UMat>() = u;
    }
    else if (k == MAT)
    {
        u.copyTo(*getObj<Mat>()); // TODO check u.getMat()
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
    _InputArray::KindFlag k = kind();
    if (k == UMAT)
    {
        m.copyTo(*getObj<UMat>()); // TODO check m.getUMat()
    }
    else if (k == MAT)
    {
        *getObj<Mat>() = m;
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


void _OutputArray::move(UMat& u) const
{
    if (fixedSize())
    {
        // TODO Performance warning
        assign(u);
        return;
    }
    int k = kind();
    if (k == UMAT)
    {
        *getObj<UMat>() = std::move(u);
    }
    else if (k == MAT)
    {
        u.copyTo(*getObj<Mat>()); // TODO check u.getMat()
        u.release();
    }
    else if (k == MATX)
    {
        u.copyTo(getMat()); // TODO check u.getMat()
        u.release();
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::move(Mat& m) const
{
    if (fixedSize())
    {
        // TODO Performance warning
        assign(m);
        return;
    }
    int k = kind();
    if (k == UMAT)
    {
        m.copyTo(*getObj<UMat>()); // TODO check m.getUMat()
        m.release();
    }
    else if (k == MAT)
    {
        *getObj<Mat>() = std::move(m);
    }
    else if (k == MATX)
    {
        m.copyTo(getMat());
        m.release();
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::assign(const std::vector<UMat>& v) const
{
    _InputArray::KindFlag k = kind();
    if (k == STD_VECTOR_UMAT)
    {
        std::vector<UMat>& this_v = *getObj<std::vector<UMat>>();
        CV_Assert(this_v.size() == v.size());

        for (size_t i = 0; i < v.size(); i++)
        {
            const UMat& m = v[i];
            UMat& this_m = this_v[i];
            if (this_m.u != NULL && this_m.u == m.u)
                continue; // same object (see dnn::Layer::forward_fallback)
            m.copyTo(this_m);
        }
    }
    else if (k == STD_VECTOR_MAT)
    {
        std::vector<Mat>& this_v = *getObj<std::vector<Mat>>();
        CV_Assert(this_v.size() == v.size());

        for (size_t i = 0; i < v.size(); i++)
        {
            const UMat& m = v[i];
            Mat& this_m = this_v[i];
            if (this_m.u != NULL && this_m.u == m.u)
                continue; // same object (see dnn::Layer::forward_fallback)
            m.copyTo(this_m);
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::assign(const std::vector<Mat>& v) const
{
    _InputArray::KindFlag k = kind();
    if (k == STD_VECTOR_UMAT)
    {
        std::vector<UMat>& this_v = *getObj<std::vector<UMat>>();
        CV_Assert(this_v.size() == v.size());

        for (size_t i = 0; i < v.size(); i++)
        {
            const Mat& m = v[i];
            UMat& this_m = this_v[i];
            if (this_m.u != NULL && this_m.u == m.u)
                continue; // same object (see dnn::Layer::forward_fallback)
            m.copyTo(this_m);
        }
    }
    else if (k == STD_VECTOR_MAT)
    {
        std::vector<Mat>& this_v = *getObj<std::vector<Mat>>();
        CV_Assert(this_v.size() == v.size());

        for (size_t i = 0; i < v.size(); i++)
        {
            const Mat& m = v[i];
            Mat& this_m = this_v[i];
            if (this_m.u != NULL && this_m.u == m.u)
                continue; // same object (see dnn::Layer::forward_fallback)
            m.copyTo(this_m);
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


static _InputOutputArray _none;
InputOutputArray noArray() { return _none; }

} // cv::
