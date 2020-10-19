/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef __F2C_H__
#define __F2C_H__

#include <assert.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>
/* needed for Windows Mobile */
#ifdef WINCE
#undef complex; 
#endif
#include <string.h>
#include <stdio.h>

#include "cblas.h"
#include "lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable: 4244 4554)
#endif

#undef complex

#define F2C_STR_MAX 64

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
#ifdef INTEGER_STAR_8	/* Adjust for integer*8. */
typedef long long longint;		/* system-dependent */
typedef unsigned long long ulongint;	/* system-dependent */
#define qbit_clear(a,b)	((a) & ~((ulongint)1 << (b)))
#define qbit_set(a,b)	((a) |  ((ulongint)1 << (b)))
#endif

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

typedef int flag;
typedef int ftnlen;
typedef int ftnint;

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

/*typedef long int Long;*/	/* No longer used; formerly in Namelist */

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#ifndef abs
#define abs(x) ((x) >= 0 ? (x) : -(x))
#endif
#define dabs(x) (doublereal)abs(x)
#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) ((a) >= (b) ? (a) : (b))
#endif
#define dmin(a,b) (doublereal)min(a,b)
#define dmax(a,b) (doublereal)max(a,b)
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

/* undef any lower-case symbols that your C compiler predefines, e.g.: */

#ifndef Skip_f2c_Undefs
#undef cray
#undef gcos
#undef mc68010
#undef mc68020
#undef mips
#undef pdp11
#undef sgi
#undef sparc
#undef sun
#undef sun2
#undef sun3
#undef sun4
#undef u370
#undef u3b
#undef u3b2
#undef u3b5
#undef unix
#undef vax
#endif

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

extern const unsigned char lapack_lamch_tab[];
    
static __inline int i_nint(float *x)
{
    return (int)(*x >= 0 ? floor(*x + .5) : -floor(.5 - *x));
}

static __inline void exit_(int *rc)
{
    exit(*rc);
}

int pow_ii(int *ap, int *bp);
double pow_ri(float *ap, int *bp);
double pow_di(double *ap, int *bp);

static __inline double pow_dd(double *ap, double *bp)
{
    return pow(*ap, *bp);
}

void d_cnjg(doublecomplex *r, doublecomplex *z);
void r_cnjg(complex *r, complex *z);

int s_copy(char *a, char *b);
int s_cat(char *lp, char **rpp, int* rnp, int *np);
int s_cmp(char *a0, char *b0);
int i_len(char*);

#ifdef __cplusplus
}
#endif

#endif
