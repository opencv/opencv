/* xerbla.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "clapack.h"


/* Table of constant values */

static integer c__1 = 1;

/* Subroutine */ int xerbla_(char *srname, integer *info)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ** On entry to \002,a,\002 parameter num"
	    "ber \002,i2,\002 had \002,\002an illegal value\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), i_len_trim(char *, ftnlen), do_fio(integer *, 
	    char *, ftnlen), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___1 = { 0, 6, 0, fmt_9999, 0 };



/*  -- LAPACK auxiliary routine (preliminary version) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  XERBLA  is an error handler for the LAPACK routines. */
/*  It is called by an LAPACK routine if an input parameter has an */
/*  invalid value.  A message is printed and execution stops. */

/*  Installers may consider modifying the STOP statement in order to */
/*  call system-specific exception-handling facilities. */

/*  Arguments */
/*  ========= */

/*  SRNAME  (input) CHARACTER*(*) */
/*          The name of the routine which called XERBLA. */

/*  INFO    (input) INTEGER */
/*          The position of the invalid parameter in the parameter list */
/*          of the calling routine. */

/* ===================================================================== */

/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

	printf("** On entry to %6s, parameter number %2i had an illegal value\n",
		srname, *info);


/*     End of XERBLA */

    return 0;
} /* xerbla_ */
