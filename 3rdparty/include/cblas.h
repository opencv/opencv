/* CLAPACK 3.0 BLAS wrapper macros and functions
 * Feb 5, 2000
 */

#ifndef __CBLAS_H
#define __CBLAS_H

#include "f2c.h"

#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable: 4244 4554)
#endif

#ifdef __cplusplus
extern "C" {
#endif

static __inline double r_lg10(real *x)
{
    return 0.43429448190325182765*log(*x);
}

static __inline double d_lg10(doublereal *x)
{
    return 0.43429448190325182765*log(*x);
}

static __inline double d_sign(doublereal *a, doublereal *b)
{
    double x = fabs(*a);
    return *b >= 0 ? x : -x;
}

static __inline double r_sign(real *a, real *b)
{
    double x = fabs((double)*a);
    return *b >= 0 ? x : -x;
}

extern const unsigned char lapack_toupper_tab[];
#define lapack_toupper(c) ((char)lapack_toupper_tab[(unsigned char)(c)])

extern const unsigned char lapack_lamch_tab[];
extern const doublereal lapack_dlamch_tab[];
extern const doublereal lapack_slamch_tab[];
    
static __inline logical lsame_(char *ca, char *cb)
{
    return lapack_toupper(ca[0]) == lapack_toupper(cb[0]);
}

static __inline doublereal dlamch_(char* cmach)
{
    return lapack_dlamch_tab[lapack_lamch_tab[(unsigned char)cmach[0]]];
}
    
static __inline doublereal slamch_(char* cmach)
{
    return lapack_slamch_tab[lapack_lamch_tab[(unsigned char)cmach[0]]];
}    
    
static __inline integer i_nint(real *x)
{
    return (integer)(*x >= 0 ? floor(*x + .5) : -floor(.5 - *x));
}

static __inline void exit_(integer *rc)
{
    exit(*rc);
}

static __inline double pow_dd(doublereal *ap, doublereal *bp)
{
    return pow(*ap, *bp);
}

logical slaisnan_(real *in1, real *in2);
logical dlaisnan_(doublereal *din1, doublereal *din2);

static __inline logical sisnan_(real *in1)
{
    return slaisnan_(in1, in1);
}

static __inline logical disnan_(doublereal *din1)
{
    return dlaisnan_(din1, din1);
}

char *F77_aloc(ftnlen, char*);

#ifdef __cplusplus
}
#endif

#endif /* __BLASWRAP_H */
