/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef __F2C_H__
#define __F2C_H__

#include <assert.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "cblas.h"
#include "lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#undef complex

typedef int integer;
typedef unsigned int uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef lapack_complex complex;
typedef lapack_doublecomplex doublecomplex;
typedef int logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;

#define TRUE_ (1)
#define FALSE_ (0)

#ifndef abs
#define abs(x) ((x) >= 0 ? (x) : -(x))
#endif
#define dabs(x) (double)abs(x)
#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) ((a) >= (b) ? (a) : (b))
#endif
#define dmin(a,b) (double)min(a,b)
#define dmax(a,b) (double)max(a,b)
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

static __inline double r_lg10(float *x)
{
    return 0.43429448190325182765*log(*x);
}

static __inline double d_lg10(double *x)
{
    return 0.43429448190325182765*log(*x);
}

static __inline double d_sign(double *a, double *b)
{
    double x = fabs(*a);
    return *b >= 0 ? x : -x;
}

static __inline double r_sign(float *a, float *b)
{
    double x = fabs((double)*a);
    return *b >= 0 ? x : -x;
}

static __inline int i_nint(float *x)
{
    return (int)(*x >= 0 ? floor(*x + .5) : -floor(.5 - *x));
}

int pow_ii(int *ap, int *bp);
double pow_di(double *ap, int *bp);
static __inline double pow_ri(float *ap, int *bp)
{
    double apd = *ap;
    return pow_di(&apd, bp);
}
static __inline double pow_dd(double *ap, double *bp)
{
    return pow(*ap, *bp);
}

static __inline void d_cnjg(doublecomplex *r, doublecomplex *z)
{
	double zi = z->i;
	r->r = z->r;
	r->i = -zi;
}

static __inline void r_cnjg(complex *r, complex *z)
{
	float zi = z->i;
	r->r = z->r;
	r->i = -zi;
}

static __inline int s_copy(char *a, char *b, int maxlen)
{
    strncpy(a, b, maxlen);
    a[maxlen] = '\0';
    return 0;
}

int s_cat(char *lp, char **rpp, int* rnp, int *np);
int s_cmp(char *a0, char *b0);
static __inline int i_len(char* s)
{
    return (int)strlen(s);
}

#ifdef __cplusplus
}
#endif

#endif
