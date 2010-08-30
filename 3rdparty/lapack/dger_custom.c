#include "clapack.h"


/* Subroutine */ int dger_(integer *_m, integer *_n, doublereal *_alpha, 
	doublereal *x, integer *_incx, doublereal *y, integer *_incy, 
	doublereal *a, integer *_lda)
{

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGER   performs the rank 1 operation */

/*     A := alpha*x*y' + A, */

/*  where alpha is a scalar, x is an m element vector, y is an n element */
/*  vector and A is an m by n matrix. */

/*  Arguments */
/*  ========== */

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

/*  X      - DOUBLE PRECISION array of dimension at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the m */
/*           element vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  Y      - DOUBLE PRECISION array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the n */
/*           element vector y. */
/*           Unchanged on exit. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading m by n part of the array A must */
/*           contain the matrix of coefficients. On exit, A is */
/*           overwritten by the updated matrix. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           max( 1, m ). */
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
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */

/*     Test the input parameters. */

    /* Function Body */
    integer i, j, m = *_m, n = *_n, incx = *_incx, incy = *_incy, lda = *_lda;
    doublereal alpha = *_alpha;
    integer info = 0;
    
    if (m < 0)
        info = 1;
    else if (n < 0)
        info = 2;
    else if (incx == 0)
        info = 5;
    else if (incy == 0)
        info = 7;
    else if (lda < max(1,m))
        info = 9;
    
    if (info != 0)
    {
        xerbla_("DGER  ", &info);
        return 0;
    }

    if (incx < 0)
        x -= (m-1)*incx;
    if (incy < 0)
        y -= (n-1)*incy;

    /*     Start the operations. In this version the elements of A are */
    /*     accessed sequentially with one pass through A. */
    
    if( alpha == 0 )
        ;
    else if( incx == 1 )
    {
        for( j = 0; j < n; j++, a += lda )
        {
            doublereal s = y[j*incy];
            if( s == 0 )
                continue;
            s *= alpha;
            
            for( i = 0; i <= m - 2; i += 2 )
            {
                doublereal t0 = a[i] + x[i]*s;
                doublereal t1 = a[i+1] + x[i+1]*s;
                a[i] = t0; a[i+1] = t1;
            }
            
            for( ; i < m; i++ )
                a[i] += x[i]*s;
        }
    }
    else
    {
        for( j = 0; j < n; j++, a += lda )
        {
            doublereal s = y[j*incy];
            if( s == 0 )
                continue;
            s *= alpha;
            
            for( i = 0; i < m; i++ )
                a[i] += x[i*incx]*s;
        }
    }

    return 0;

/*     End of DGER  . */

} /* dger_ */
