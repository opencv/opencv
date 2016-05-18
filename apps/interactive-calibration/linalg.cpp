#include "linalg.hpp"

#ifdef USE_LAPACK

typedef int    integer;
#include <lapacke.h>

#include <cassert>
using namespace cv;

bool cvfork::solve(InputArray _src, const InputArray _src2arg, OutputArray _dst, int method )
    {
        bool result = true;
        Mat src = _src.getMat(), _src2 = _src2arg.getMat();
        int type = src.type();
        bool is_normal = (method & DECOMP_NORMAL) != 0;

        CV_Assert( type == _src2.type() && (type == CV_32F || type == CV_64F) );

        method &= ~DECOMP_NORMAL;
        CV_Assert( (method != DECOMP_LU && method != DECOMP_CHOLESKY) ||
            is_normal || src.rows == src.cols );

        double rcond=-1, s1=0, work1=0, *work=0, *s=0;
        float frcond=-1, fs1=0, fwork1=0, *fwork=0, *fs=0;
        integer m = src.rows, m_ = m, n = src.cols, mn = std::max(m,n),
            nm = std::min(m, n), nb = _src2.cols, lwork=-1, liwork=0, iwork1=0,
            lda = m, ldx = mn, info=0, rank=0, *iwork=0;
        int elem_size = CV_ELEM_SIZE(type);
        bool copy_rhs=false;
        int buf_size=0;
        AutoBuffer<uchar> buffer;
        uchar* ptr;
        char N[] = {'N', '\0'}, L[] = {'L', '\0'};

        Mat src2 = _src2;
        _dst.create( src.cols, src2.cols, src.type() );
        Mat dst = _dst.getMat();

        if( m <= n )
            is_normal = false;
        else if( is_normal )
            m_ = n;

        buf_size += (is_normal ? n*n : m*n)*elem_size;

        if( m_ != n || nb > 1 || !dst.isContinuous() )
        {
            copy_rhs = true;
            if( is_normal )
                buf_size += n*nb*elem_size;
            else
                buf_size += mn*nb*elem_size;
        }

        if( method == DECOMP_SVD || method == DECOMP_EIG )
        {
            integer nlvl = cvRound(std::log(std::max(std::min(m_,n)/25., 1.))/CV_LOG2) + 1;
            liwork = std::min(m_,n)*(3*std::max(nlvl,(integer)0) + 11);

            if( type == CV_32F )
                sgelsd_(&m_, &n, &nb, (float*)src.data, &lda, (float*)dst.data, &ldx,
                    &fs1, &frcond, &rank, &fwork1, &lwork, &iwork1, &info);
            else
                dgelsd_(&m_, &n, &nb, (double*)src.data, &lda, (double*)dst.data, &ldx,
                    &s1, &rcond, &rank, &work1, &lwork, &iwork1, &info );
            buf_size += nm*elem_size + (liwork + 1)*sizeof(integer);
        }
        else if( method == DECOMP_QR )
        {
            if( type == CV_32F )
                sgels_(N, &m_, &n, &nb, (float*)src.data, &lda,
                    (float*)dst.data, &ldx, &fwork1, &lwork, &info );
            else
                dgels_(N, &m_, &n, &nb, (double*)src.data, &lda,
                    (double*)dst.data, &ldx, &work1, &lwork, &info );
        }
        else if( method == DECOMP_LU )
        {
            buf_size += (n+1)*sizeof(integer);
        }
        else if( method == DECOMP_CHOLESKY )
            ;
        else
            CV_Error( Error::StsBadArg, "Unknown method" );
        assert(info == 0);

        lwork = cvRound(type == CV_32F ? (double)fwork1 : work1);
        buf_size += lwork*elem_size;
        buffer.allocate(buf_size);
        ptr = (uchar*)buffer;

        Mat at(n, m_, type, ptr);
        ptr += n*m_*elem_size;

        if( method == DECOMP_CHOLESKY || method == DECOMP_EIG )
            src.copyTo(at);
        else if( !is_normal )
            transpose(src, at);
        else
            mulTransposed(src, at, true);

        Mat xt;
        if( !is_normal )
        {
            if( copy_rhs )
            {
                Mat temp(nb, mn, type, ptr);
                ptr += nb*mn*elem_size;
                Mat bt = temp.colRange(0, m);
                xt = temp.colRange(0, n);
                transpose(src2, bt);
            }
            else
            {
                src2.copyTo(dst);
                xt = Mat(1, n, type, dst.data);
            }
        }
        else
        {
            if( copy_rhs )
            {
                xt = Mat(nb, n, type, ptr);
                ptr += nb*n*elem_size;
            }
            else
                xt = Mat(1, n, type, dst.data);
            // (a'*b)' = b'*a
            gemm( src2, src, 1, Mat(), 0, xt, GEMM_1_T );
        }

        lda = (int)(at.step ? at.step/elem_size : at.cols);
        ldx = (int)(xt.step ? xt.step/elem_size : (!is_normal && copy_rhs ? mn : n));

        if( method == DECOMP_SVD || method == DECOMP_EIG )
        {
            if( type == CV_32F )
            {
                fs = (float*)ptr;
                ptr += nm*elem_size;
                fwork = (float*)ptr;
                ptr += lwork*elem_size;
                iwork = (integer*)alignPtr(ptr, sizeof(integer));

                sgelsd_(&m_, &n, &nb, (float*)at.data, &lda, (float*)xt.data, &ldx,
                    fs, &frcond, &rank, fwork, &lwork, iwork, &info);
            }
            else
            {
                s = (double*)ptr;
                ptr += nm*elem_size;
                work = (double*)ptr;
                ptr += lwork*elem_size;
                iwork = (integer*)alignPtr(ptr, sizeof(integer));

                dgelsd_(&m_, &n, &nb, (double*)at.data, &lda, (double*)xt.data, &ldx,
                    s, &rcond, &rank, work, &lwork, iwork, &info);
            }
        }
        else if( method == DECOMP_QR )
        {
            if( type == CV_32F )
            {
                fwork = (float*)ptr;
                sgels_(N, &m_, &n, &nb, (float*)at.data, &lda,
                    (float*)xt.data, &ldx, fwork, &lwork, &info);
            }
            else
            {
                work = (double*)ptr;
                dgels_(N, &m_, &n, &nb, (double*)at.data, &lda,
                    (double*)xt.data, &ldx, work, &lwork, &info);
            }
        }
        else if( method == DECOMP_CHOLESKY || (method == DECOMP_LU && is_normal) )
        {
            if( type == CV_32F )
            {
                spotrf_(L, &n, (float*)at.data, &lda, &info);
                if(info==0)
                    spotrs_(L, &n, &nb, (float*)at.data, &lda, (float*)xt.data, &ldx, &info);
            }
            else
            {
                dpotrf_(L, &n, (double*)at.data, &lda, &info);
                if(info==0)
                    dpotrs_(L, &n, &nb, (double*)at.data, &lda, (double*)xt.data, &ldx, &info);
            }
        }
        else if( method == DECOMP_LU )
        {
            iwork = (integer*)alignPtr(ptr, sizeof(integer));
            if( type == CV_32F )
                sgesv_(&n, &nb, (float*)at.data, &lda, iwork, (float*)xt.data, &ldx, &info );
            else
                dgesv_(&n, &nb, (double*)at.data, &lda, iwork, (double*)xt.data, &ldx, &info );
        }
        else
            assert(0);
        result = info == 0;

        if( !result )
            dst = Scalar(0);
        else if( xt.data != dst.data )
            transpose( xt, dst );

        return result;
    }

static void _SVDcompute( const InputArray _aarr, OutputArray _w,
                         OutputArray _u, OutputArray _vt, int flags = 0)
{
    Mat a = _aarr.getMat(), u, vt;
    integer m = a.rows, n = a.cols, mn = std::max(m, n), nm = std::min(m, n);
    int type = a.type(), elem_size = (int)a.elemSize();
    bool compute_uv = _u.needed() || _vt.needed();

    if( flags & SVD::NO_UV )
    {
        _u.release();
        _vt.release();
        compute_uv = false;
    }

    if( compute_uv )
    {
        _u.create( (int)m, (int)((flags & SVD::FULL_UV) ? m : nm), type );
        _vt.create( (int)((flags & SVD::FULL_UV) ? n : nm), n, type );
        u = _u.getMat();
        vt = _vt.getMat();
    }

    _w.create(nm, 1, type, -1, true);

    Mat _a = a, w = _w.getMat();
    CV_Assert( w.isContinuous() );
    int work_ofs=0, iwork_ofs=0, buf_size = 0;
    bool temp_a = false;
    double u1=0, v1=0, work1=0;
    float uf1=0, vf1=0, workf1=0;
    integer lda, ldu, ldv, lwork=-1, iwork1=0, info=0;
    char mode[] = {compute_uv ? 'S' : 'N', '\0'};

    if( m != n && compute_uv && (flags & SVD::FULL_UV) )
        mode[0] = 'A';

    if( !(flags & SVD::MODIFY_A) )
    {
        if( mode[0] == 'N' || mode[0] == 'A' )
            temp_a = true;
        else if( compute_uv && (a.size() == vt.size() || a.size() == u.size()) && mode[0] == 'S' )
            mode[0] = 'O';
    }

    lda = a.cols;
    ldv = ldu = mn;

    if( type == CV_32F )
    {
        sgesdd_(mode, &n, &m, (float*)a.data, &lda, (float*)w.data,
                &vf1, &ldv, &uf1, &ldu, &workf1, &lwork, &iwork1, &info );
        lwork = cvRound(workf1);
    }
    else
    {
        dgesdd_(mode, &n, &m, (double*)a.data, &lda, (double*)w.data,
                &v1, &ldv, &u1, &ldu, &work1, &lwork, &iwork1, &info );
        lwork = cvRound(work1);
    }

    assert(info == 0);
    if( temp_a )
    {
        buf_size += n*m*elem_size;
    }
    work_ofs = buf_size;
    buf_size += lwork*elem_size;
    buf_size = alignSize(buf_size, sizeof(integer));
    iwork_ofs = buf_size;
    buf_size += 8*nm*sizeof(integer);

    AutoBuffer<uchar> buf(buf_size);
    uchar* buffer = (uchar*)buf;

    if( temp_a )
    {
        _a = Mat(a.rows, a.cols, type, buffer );
        a.copyTo(_a);
    }

    if( !(flags & SVD::MODIFY_A) && !temp_a )
    {
        if( compute_uv && a.size() == vt.size() )
        {
            a.copyTo(vt);
            _a = vt;
        }
        else if( compute_uv && a.size() == u.size() )
        {
            a.copyTo(u);
            _a = u;
        }
    }

    if( compute_uv )
    {
        ldv = (int)(vt.step ? vt.step/elem_size : vt.cols);
        ldu = (int)(u.step ? u.step/elem_size : u.cols);
    }

    lda = (int)(_a.step ? _a.step/elem_size : _a.cols);
    if( type == CV_32F )
    {
        sgesdd_(mode, &n, &m, _a.ptr<float>(), &lda, w.ptr<float>(),
                vt.data ? vt.ptr<float>() : (float*)&v1, &ldv,
                u.data ? u.ptr<float>() : (float*)&u1, &ldu,
                (float*)(buffer + work_ofs), &lwork,
                (integer*)(buffer + iwork_ofs), &info );
    }
    else
    {
        dgesdd_(mode, &n, &m, _a.ptr<double>(), &lda, w.ptr<double>(),
                vt.data ? vt.ptr<double>() : &v1, &ldv,
                u.data ? u.ptr<double>() : &u1, &ldu,
                (double*)(buffer + work_ofs), &lwork,
                (integer*)(buffer + iwork_ofs), &info );
    }
    CV_Assert(info >= 0);
    if(info != 0)
    {
        if( u.data )
            u = Scalar(0.);
        if( vt.data )
            vt = Scalar(0.);
        w = Scalar(0.);
    }
}
//////////////////////////////////////////////////////////
template<typename T1, typename T2, typename T3> static void
MatrAXPY( int m, int n, const T1* x, int dx,
          const T2* a, int inca, T3* y, int dy )
{
    int i, j;
    for( i = 0; i < m; i++, x += dx, y += dy )
    {
        T2 s = a[i*inca];
        for( j = 0; j <= n - 4; j += 4 )
        {
            T3 t0 = (T3)(y[j]   + s*x[j]);
            T3 t1 = (T3)(y[j+1] + s*x[j+1]);
            y[j]   = t0;
            y[j+1] = t1;
            t0 = (T3)(y[j+2] + s*x[j+2]);
            t1 = (T3)(y[j+3] + s*x[j+3]);
            y[j+2] = t0;
            y[j+3] = t1;
        }

        for( ; j < n; j++ )
            y[j] = (T3)(y[j] + s*x[j]);
    }
}
template<typename T> static void
SVBkSb( int m, int n, const T* w, int incw,
        const T* u, int ldu, int uT,
        const T* v, int ldv, int vT,
        const T* b, int ldb, int nb,
        T* x, int ldx, double* buffer, T eps )
{
    double threshold = 0;
    int udelta0 = uT ? ldu : 1, udelta1 = uT ? 1 : ldu;
    int vdelta0 = vT ? ldv : 1, vdelta1 = vT ? 1 : ldv;
    int i, j, nm = std::min(m, n);

    if( !b )
        nb = m;

    for( i = 0; i < n; i++ )
        for( j = 0; j < nb; j++ )
            x[i*ldx + j] = 0;

    for( i = 0; i < nm; i++ )
        threshold += w[i*incw];
    threshold *= eps;

    // v * inv(w) * uT * b
    for( i = 0; i < nm; i++, u += udelta0, v += vdelta0 )
    {
        double wi = w[i*incw];
        if( wi <= threshold )
            continue;
        wi = 1/wi;

        if( nb == 1 )
        {
            double s = 0;
            if( b )
                for( j = 0; j < m; j++ )
                    s += u[j*udelta1]*b[j*ldb];
            else
                s = u[0];
            s *= wi;

            for( j = 0; j < n; j++ )
                x[j*ldx] = (T)(x[j*ldx] + s*v[j*vdelta1]);
        }
        else
        {
            if( b )
            {
                for( j = 0; j < nb; j++ )
                    buffer[j] = 0;
                MatrAXPY( m, nb, b, ldb, u, udelta1, buffer, 0 );
                for( j = 0; j < nb; j++ )
                    buffer[j] *= wi;
            }
            else
            {
                for( j = 0; j < nb; j++ )
                    buffer[j] = u[j*udelta1]*wi;
            }
            MatrAXPY( n, nb, buffer, 0, v, vdelta1, x, ldx );
        }
    }
}

static void _backSubst( const InputArray _w, const InputArray _u, const InputArray _vt,
                     const InputArray _rhs, OutputArray _dst )
{
    Mat w = _w.getMat(), u = _u.getMat(), vt = _vt.getMat(), rhs = _rhs.getMat();
    int type = w.type(), esz = (int)w.elemSize();
    int m = u.rows, n = vt.cols, nb = rhs.data ? rhs.cols : m;
    AutoBuffer<double> buffer(nb);
    CV_Assert( u.data && vt.data && w.data );

    CV_Assert( rhs.data == 0 || (rhs.type() == type && rhs.rows == m) );

    _dst.create( n, nb, type );
    Mat dst = _dst.getMat();
    if( type == CV_32F )
        SVBkSb(m, n, (float*)w.data, 1, (float*)u.data, (int)(u.step/esz), false,
               (float*)vt.data, (int)(vt.step/esz), true, (float*)rhs.data, (int)(rhs.step/esz),
               nb, (float*)dst.data, (int)(dst.step/esz), buffer, 10*FLT_EPSILON );
    else if( type == CV_64F )
        SVBkSb(m, n, (double*)w.data, 1, (double*)u.data, (int)(u.step/esz), false,
               (double*)vt.data, (int)(vt.step/esz), true, (double*)rhs.data, (int)(rhs.step/esz),
               nb, (double*)dst.data, (int)(dst.step/esz), buffer, 2*DBL_EPSILON );
    else
        CV_Error( Error::StsUnsupportedFormat, "" );
}
///////////////////////////////////////////

#define Sf( y, x ) ((float*)(srcdata + y*srcstep))[x]
#define Sd( y, x ) ((double*)(srcdata + y*srcstep))[x]
#define Df( y, x ) ((float*)(dstdata + y*dststep))[x]
#define Dd( y, x ) ((double*)(dstdata + y*dststep))[x]

double cvfork::invert( InputArray _src, OutputArray _dst, int method )
{
    Mat src = _src.getMat();
    int type = src.type();

    CV_Assert(type == CV_32F || type == CV_64F);

    size_t esz = CV_ELEM_SIZE(type);
    int m = src.rows, n = src.cols;

    if( method == DECOMP_SVD )
    {
        int nm = std::min(m, n);

        AutoBuffer<uchar> _buf((m*nm + nm + nm*n)*esz + sizeof(double));
        uchar* buf = alignPtr((uchar*)_buf, (int)esz);
        Mat u(m, nm, type, buf);
        Mat w(nm, 1, type, u.ptr() + m*nm*esz);
        Mat vt(nm, n, type, w.ptr() + nm*esz);

        _SVDcompute(src, w, u, vt);
        _backSubst(w, u, vt, Mat(), _dst);

        return type == CV_32F ?
            (w.ptr<float>()[0] >= FLT_EPSILON ?
             w.ptr<float>()[n-1]/w.ptr<float>()[0] : 0) :
            (w.ptr<double>()[0] >= DBL_EPSILON ?
             w.ptr<double>()[n-1]/w.ptr<double>()[0] : 0);
    }
    return 0;
}

#endif //USE_LAPACK
