#include "clapack.h"


/* Subroutine */ int dgemv_(char *_trans, integer *_m, integer *_n, doublereal *
	_alpha, doublereal *a, integer *_lda, doublereal *x, integer *_incx, 
	doublereal *_beta, doublereal *y, integer *_incy)
{
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGEMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n matrix. */

/*  Arguments */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - DOUBLE PRECISION. */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading m by n part of the array A must */
/*           contain the matrix of coefficients. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           max( 1, m ). */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - DOUBLE PRECISION. */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry with BETA non-zero, the incremented array Y */
/*           must contain the vector y. On exit, Y is overwritten by the */
/*           updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */

/*     Test the input parameters. */

    char trans = lapack_toupper(_trans[0]);
    integer i, j, m = *_m, n = *_n, lda = *_lda, incx = *_incx, incy = *_incy;
    integer leny = trans == 'N' ? m : n, lenx = trans == 'N' ? n : m;
    real alpha = *_alpha, beta = *_beta;
    
    integer info = 0;
    if (trans != 'N' && trans != 'T' && trans != 'C')
        info = 1;
    else if (m < 0)
        info = 2;
    else if (n < 0)
        info = 3;
    else if (lda < max(1,m))
        info = 6;
    else if (incx == 0)
        info = 8;
    else if (incy == 0)
        info = 11;
    
    if (info != 0)
    {
        xerbla_("SGEMV ", &info);
        return 0;
    }
    
    if( incy < 0 )
        y -= incy*(leny - 1);
    if( incx < 0 )
        x -= incx*(lenx - 1);
    
    /*     Start the operations. In this version the elements of A are */
    /*     accessed sequentially with one pass through A. */
    
    if( beta != 1. )
    {
        if( incy == 1 )
        {
            if( beta == 0. )
                for( i = 0; i < leny; i++ )
                    y[i] = 0.;
            else
                for( i = 0; i < leny; i++ )
                    y[i] *= beta;
        }
        else
        {
            if( beta == 0. )
                for( i = 0; i < leny; i++ )
                    y[i*incy] = 0.;
            else
                for( i = 0; i < leny; i++ )
                    y[i*incy] *= beta;
        }
    }
    
    if( alpha == 0. )
        ;
    else if( trans == 'N' )
    {
        if( incy == 1 )
        {
            for( i = 0; i < n; i++, a += lda )
            {
                doublereal s = x[i*incx];
                if( s == 0. )
                    continue;
                s *= alpha;
                for( j = 0; j <= m - 2; j += 2 )
                {
                    doublereal t0 = y[j] + s*a[j];
                    doublereal t1 = y[j+1] + s*a[j+1];
                    y[j] = t0; y[j+1] = t1;
                }
                
                for( ; j < m; j++ )
                    y[j] += s*a[j];
            }
        }
        else
        {
            for( i = 0; i < n; i++, a += lda )
            {
                doublereal s = x[i*incx];
                if( s == 0. )
                    continue;
                s *= alpha;
                for( j = 0; j < m; j++ )
                    y[j*incy] += s*a[j];
            }
        }
    }
    else
    {
        if( incx == 1 )
        {
            for( i = 0; i < n; i++, a += lda )
            {
                doublereal s = 0;
                for( j = 0; j <= m - 2; j += 2 )
                    s += x[j]*a[j] + x[j+1]*a[j+1];
                for( ; j < m; j++ )
                    s += x[j]*a[j];
                y[i*incy] += alpha*s;
            }
        }
        else
        {
            for( i = 0; i < n; i++, a += lda )
            {
                doublereal s = 0;
                for( j = 0; j < m; j++ )
                    s += x[j*incx]*a[j];
                y[i*incy] += alpha*s;
            }
        }
    }
    
    return 0;

/*     End of DGEMV . */

} /* dgemv_ */
