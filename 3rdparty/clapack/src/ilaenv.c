/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b IEEECK
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download IEEECK + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ieeeck.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ieeeck.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ieeeck.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER          FUNCTION IEEECK( ISPEC, ZERO, ONE )
//
//      .. Scalar Arguments ..
//      INTEGER            ISPEC
//      REAL               ONE, ZERO
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> IEEECK is called from the ILAENV to verify that Infinity and
//> possibly NaN arithmetic is safe (i.e. will not trap).
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ISPEC
//> \verbatim
//>          ISPEC is INTEGER
//>          Specifies whether to test just for inifinity arithmetic
//>          or whether to test for infinity and NaN arithmetic.
//>          = 0: Verify infinity arithmetic only.
//>          = 1: Verify infinity and NaN arithmetic.
//> \endverbatim
//>
//> \param[in] ZERO
//> \verbatim
//>          ZERO is REAL
//>          Must contain the value 0.0
//>          This is passed to prevent the compiler from optimizing
//>          away this code.
//> \endverbatim
//>
//> \param[in] ONE
//> \verbatim
//>          ONE is REAL
//>          Must contain the value 1.0
//>          This is passed to prevent the compiler from optimizing
//>          away this code.
//>
//>  RETURN VALUE:  INTEGER
//>          = 0:  Arithmetic failed to produce the correct answers
//>          = 1:  Arithmetic produced the correct answers
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
// =====================================================================
int ieeeck_(int *ispec, float *zero, float *one)
{
    // System generated locals
    int ret_val;

    // Local variables
    float nan1, nan2, nan3, nan4, nan5, nan6, neginf, posinf, negzro, newzro;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. Executable Statements ..
    ret_val = 1;
    posinf = *one / *zero;
    if (posinf <= *one) {
	ret_val = 0;
	return ret_val;
    }
    neginf = -(*one) / *zero;
    if (neginf >= *zero) {
	ret_val = 0;
	return ret_val;
    }
    negzro = *one / (neginf + *one);
    if (negzro != *zero) {
	ret_val = 0;
	return ret_val;
    }
    neginf = *one / negzro;
    if (neginf >= *zero) {
	ret_val = 0;
	return ret_val;
    }
    newzro = negzro + *zero;
    if (newzro != *zero) {
	ret_val = 0;
	return ret_val;
    }
    posinf = *one / newzro;
    if (posinf <= *one) {
	ret_val = 0;
	return ret_val;
    }
    neginf *= posinf;
    if (neginf >= *zero) {
	ret_val = 0;
	return ret_val;
    }
    posinf *= posinf;
    if (posinf <= *one) {
	ret_val = 0;
	return ret_val;
    }
    //
    //
    //
    //
    //    Return if we were only asked to check infinity arithmetic
    //
    if (*ispec == 0) {
	return ret_val;
    }
    nan1 = posinf + neginf;
    nan2 = posinf / neginf;
    nan3 = posinf / posinf;
    nan4 = posinf * *zero;
    nan5 = neginf * negzro;
    nan6 = nan5 * *zero;
    if (nan1 == nan1) {
	ret_val = 0;
	return ret_val;
    }
    if (nan2 == nan2) {
	ret_val = 0;
	return ret_val;
    }
    if (nan3 == nan3) {
	ret_val = 0;
	return ret_val;
    }
    if (nan4 == nan4) {
	ret_val = 0;
	return ret_val;
    }
    if (nan5 == nan5) {
	ret_val = 0;
	return ret_val;
    }
    if (nan6 == nan6) {
	ret_val = 0;
	return ret_val;
    }
    return ret_val;
} // ieeeck_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b ILAENV
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download ILAENV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ilaenv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ilaenv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ilaenv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER FUNCTION ILAENV( ISPEC, NAME, OPTS, N1, N2, N3, N4 )
//
//      .. Scalar Arguments ..
//      CHARACTER*( * )    NAME, OPTS
//      INTEGER            ISPEC, N1, N2, N3, N4
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> ILAENV is called from the LAPACK routines to choose problem-dependent
//> parameters for the local environment.  See ISPEC for a description of
//> the parameters.
//>
//> ILAENV returns an INTEGER
//> if ILAENV >= 0: ILAENV returns the value of the parameter specified by ISPEC
//> if ILAENV < 0:  if ILAENV = -k, the k-th argument had an illegal value.
//>
//> This version provides a set of parameters which should give good,
//> but not optimal, performance on many of the currently available
//> computers.  Users are encouraged to modify this subroutine to set
//> the tuning parameters for their particular machine using the option
//> and problem size information in the arguments.
//>
//> This routine will not function correctly if it is converted to all
//> lower case.  Converting it to all upper case is allowed.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ISPEC
//> \verbatim
//>          ISPEC is INTEGER
//>          Specifies the parameter to be returned as the value of
//>          ILAENV.
//>          = 1: the optimal blocksize; if this value is 1, an unblocked
//>               algorithm will give the best performance.
//>          = 2: the minimum block size for which the block routine
//>               should be used; if the usable block size is less than
//>               this value, an unblocked routine should be used.
//>          = 3: the crossover point (in a block routine, for N less
//>               than this value, an unblocked routine should be used)
//>          = 4: the number of shifts, used in the nonsymmetric
//>               eigenvalue routines (DEPRECATED)
//>          = 5: the minimum column dimension for blocking to be used;
//>               rectangular blocks must have dimension at least k by m,
//>               where k is given by ILAENV(2,...) and m by ILAENV(5,...)
//>          = 6: the crossover point for the SVD (when reducing an m by n
//>               matrix to bidiagonal form, if max(m,n)/min(m,n) exceeds
//>               this value, a QR factorization is used first to reduce
//>               the matrix to a triangular form.)
//>          = 7: the number of processors
//>          = 8: the crossover point for the multishift QR method
//>               for nonsymmetric eigenvalue problems (DEPRECATED)
//>          = 9: maximum size of the subproblems at the bottom of the
//>               computation tree in the divide-and-conquer algorithm
//>               (used by xGELSD and xGESDD)
//>          =10: ieee NaN arithmetic can be trusted not to trap
//>          =11: infinity arithmetic can be trusted not to trap
//>          12 <= ISPEC <= 16:
//>               xHSEQR or related subroutines,
//>               see IPARMQ for detailed explanation
//> \endverbatim
//>
//> \param[in] NAME
//> \verbatim
//>          NAME is CHARACTER*(*)
//>          The name of the calling subroutine, in either upper case or
//>          lower case.
//> \endverbatim
//>
//> \param[in] OPTS
//> \verbatim
//>          OPTS is CHARACTER*(*)
//>          The character options to the subroutine NAME, concatenated
//>          into a single character string.  For example, UPLO = 'U',
//>          TRANS = 'T', and DIAG = 'N' for a triangular routine would
//>          be specified as OPTS = 'UTN'.
//> \endverbatim
//>
//> \param[in] N1
//> \verbatim
//>          N1 is INTEGER
//> \endverbatim
//>
//> \param[in] N2
//> \verbatim
//>          N2 is INTEGER
//> \endverbatim
//>
//> \param[in] N3
//> \verbatim
//>          N3 is INTEGER
//> \endverbatim
//>
//> \param[in] N4
//> \verbatim
//>          N4 is INTEGER
//>          Problem dimensions for the subroutine NAME; these may not all
//>          be required.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date November 2019
//
//> \ingroup OTHERauxiliary
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The following conventions have been used when calling ILAENV from the
//>  LAPACK routines:
//>  1)  OPTS is a concatenation of all of the character options to
//>      subroutine NAME, in the same order that they appear in the
//>      argument list for NAME, even if they are not used in determining
//>      the value of the parameter specified by ISPEC.
//>  2)  The problem dimensions N1, N2, N3, N4 are specified in the order
//>      that they appear in the argument list for NAME.  N1 is used
//>      first, N2 second, and so on, and unused problem dimensions are
//>      passed a value of -1.
//>  3)  The parameter value returned by ILAENV is checked for validity in
//>      the calling subroutine.  For example, ILAENV is used to retrieve
//>      the optimal blocksize for STRTRI as follows:
//>
//>      NB = ILAENV( 1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1 )
//>      IF( NB.LE.1 ) NB = MAX( 1, N )
//> \endverbatim
//>
// =====================================================================
int ilaenv_(int *ispec, char *name__, char *opts, int *n1, int *n2, int *n3,
	int *n4)
{
    // Table of constant values
    int c__1 = 1;
    float c_b174 = 0.f;
    float c_b175 = 1.f;
    int c__0 = 0;

    // System generated locals
    int ret_val;

    // Local variables
    int twostage;
    int i__;
    char c1[1+1]={'\0'}, c2[2+1]={'\0'}, c3[3+1]={'\0'}, c4[2+1]={'\0'};
    int ic, nb, iz, nx;
    int cname;
    int nbmin;
    int sname;
    extern int ieeeck_(int *, float *, float *);
    char subnam[16+1]={'\0'};
    extern int iparmq_(int *, char *, char *, int *, int *, int *, int *);

    //
    // -- LAPACK auxiliary routine (version 3.9.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2019
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    switch (*ispec) {
	case 1:  goto L10;
	case 2:  goto L10;
	case 3:  goto L10;
	case 4:  goto L80;
	case 5:  goto L90;
	case 6:  goto L100;
	case 7:  goto L110;
	case 8:  goto L120;
	case 9:  goto L130;
	case 10:  goto L140;
	case 11:  goto L150;
	case 12:  goto L160;
	case 13:  goto L160;
	case 14:  goto L160;
	case 15:  goto L160;
	case 16:  goto L160;
    }
    //
    //    Invalid value for ISPEC
    //
    ret_val = -1;
    return ret_val;
L10:
    //
    //    Convert NAME to upper case if the first character is lower case.
    //
    ret_val = 1;
    s_copy(subnam, name__, (int)16);
    ic = *(unsigned char *)subnam;
    iz = 'Z';
    if (iz == 90 || iz == 122) {
	//
	//       ASCII character set
	//
	if (ic >= 97 && ic <= 122) {
	    *(unsigned char *)subnam = (char) (ic - 32);
	    for (i__ = 2; i__ <= 6; ++i__) {
		ic = *(unsigned char *)&subnam[i__ - 1];
		if (ic >= 97 && ic <= 122) {
		    *(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);
		}
// L20:
	    }
	}
    } else if (iz == 233 || iz == 169) {
	//
	//       EBCDIC character set
	//
	if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >= 162 &&
		ic <= 169) {
	    *(unsigned char *)subnam = (char) (ic + 64);
	    for (i__ = 2; i__ <= 6; ++i__) {
		ic = *(unsigned char *)&subnam[i__ - 1];
		if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >=
			162 && ic <= 169) {
		    *(unsigned char *)&subnam[i__ - 1] = (char) (ic + 64);
		}
// L30:
	    }
	}
    } else if (iz == 218 || iz == 250) {
	//
	//       Prime machines:  ASCII+128
	//
	if (ic >= 225 && ic <= 250) {
	    *(unsigned char *)subnam = (char) (ic - 32);
	    for (i__ = 2; i__ <= 6; ++i__) {
		ic = *(unsigned char *)&subnam[i__ - 1];
		if (ic >= 225 && ic <= 250) {
		    *(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);
		}
// L40:
	    }
	}
    }
    *(unsigned char *)c1 = *(unsigned char *)subnam;
    sname = *(unsigned char *)c1 == 'S' || *(unsigned char *)c1 == 'D';
    cname = *(unsigned char *)c1 == 'C' || *(unsigned char *)c1 == 'Z';
    if (! (cname || sname)) {
	return ret_val;
    }
    s_copy(c2, subnam + 1, (int)2);
    s_copy(c3, subnam + 3, (int)3);
    s_copy(c4, c3 + 1, (int)2);
    twostage = i_len(subnam) >= 11 && *(unsigned char *)&subnam[10] == '2';
    switch (*ispec) {
	case 1:  goto L50;
	case 2:  goto L60;
	case 3:  goto L70;
    }
L50:
    //
    //    ISPEC = 1:  block size
    //
    //    In these examples, separate code is provided for setting NB for
    //    real and complex.  We assume that NB will take the same value in
    //    single or double precision.
    //
    nb = 1;
    if (s_cmp(subnam + 1, "LAORH") == 0) {
	//
	//       This is for *LAORHR_GETRFNP routine
	//
	if (sname) {
	    nb = 32;
	} else {
	    nb = 32;
	}
    } else if (s_cmp(c2, "GE") == 0) {
	if (s_cmp(c3, "TRF") == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	} else if (s_cmp(c3, "QRF") == 0 || s_cmp(c3, "RQF") == 0 || s_cmp(c3,
		 "LQF") == 0 || s_cmp(c3, "QLF") == 0) {
	    if (sname) {
		nb = 32;
	    } else {
		nb = 32;
	    }
	} else if (s_cmp(c3, "QR ") == 0) {
	    if (*n3 == 1) {
		if (sname) {
		    //    M*N
		    if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
			nb = *n1;
		    } else {
			nb = 32768 / *n2;
		    }
		} else {
		    if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
			nb = *n1;
		    } else {
			nb = 32768 / *n2;
		    }
		}
	    } else {
		if (sname) {
		    nb = 1;
		} else {
		    nb = 1;
		}
	    }
	} else if (s_cmp(c3, "LQ ") == 0) {
	    if (*n3 == 2) {
		if (sname) {
		    //    M*N
		    if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
			nb = *n1;
		    } else {
			nb = 32768 / *n2;
		    }
		} else {
		    if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
			nb = *n1;
		    } else {
			nb = 32768 / *n2;
		    }
		}
	    } else {
		if (sname) {
		    nb = 1;
		} else {
		    nb = 1;
		}
	    }
	} else if (s_cmp(c3, "HRD") == 0) {
	    if (sname) {
		nb = 32;
	    } else {
		nb = 32;
	    }
	} else if (s_cmp(c3, "BRD") == 0) {
	    if (sname) {
		nb = 32;
	    } else {
		nb = 32;
	    }
	} else if (s_cmp(c3, "TRI") == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (s_cmp(c2, "PO") == 0) {
	if (s_cmp(c3, "TRF") == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (s_cmp(c2, "SY") == 0) {
	if (s_cmp(c3, "TRF") == 0) {
	    if (sname) {
		if (twostage) {
		    nb = 192;
		} else {
		    nb = 64;
		}
	    } else {
		if (twostage) {
		    nb = 192;
		} else {
		    nb = 64;
		}
	    }
	} else if (sname && s_cmp(c3, "TRD") == 0) {
	    nb = 32;
	} else if (sname && s_cmp(c3, "GST") == 0) {
	    nb = 64;
	}
    } else if (cname && s_cmp(c2, "HE") == 0) {
	if (s_cmp(c3, "TRF") == 0) {
	    if (twostage) {
		nb = 192;
	    } else {
		nb = 64;
	    }
	} else if (s_cmp(c3, "TRD") == 0) {
	    nb = 32;
	} else if (s_cmp(c3, "GST") == 0) {
	    nb = 64;
	}
    } else if (sname && s_cmp(c2, "OR") == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nb = 32;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nb = 32;
	    }
	}
    } else if (cname && s_cmp(c2, "UN") == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nb = 32;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nb = 32;
	    }
	}
    } else if (s_cmp(c2, "GB") == 0) {
	if (s_cmp(c3, "TRF") == 0) {
	    if (sname) {
		if (*n4 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    } else {
		if (*n4 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    }
	}
    } else if (s_cmp(c2, "PB") == 0) {
	if (s_cmp(c3, "TRF") == 0) {
	    if (sname) {
		if (*n2 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    } else {
		if (*n2 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    }
	}
    } else if (s_cmp(c2, "TR") == 0) {
	if (s_cmp(c3, "TRI") == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	} else if (s_cmp(c3, "EVC") == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (s_cmp(c2, "LA") == 0) {
	if (s_cmp(c3, "UUM") == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (sname && s_cmp(c2, "ST") == 0) {
	if (s_cmp(c3, "EBZ") == 0) {
	    nb = 1;
	}
    } else if (s_cmp(c2, "GG") == 0) {
	nb = 32;
	if (s_cmp(c3, "HD3") == 0) {
	    if (sname) {
		nb = 32;
	    } else {
		nb = 32;
	    }
	}
    }
    ret_val = nb;
    return ret_val;
L60:
    //
    //    ISPEC = 2:  minimum block size
    //
    nbmin = 2;
    if (s_cmp(c2, "GE") == 0) {
	if (s_cmp(c3, "QRF") == 0 || s_cmp(c3, "RQF") == 0 || s_cmp(c3, "LQF")
		 == 0 || s_cmp(c3, "QLF") == 0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	} else if (s_cmp(c3, "HRD") == 0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	} else if (s_cmp(c3, "BRD") == 0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	} else if (s_cmp(c3, "TRI") == 0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	}
    } else if (s_cmp(c2, "SY") == 0) {
	if (s_cmp(c3, "TRF") == 0) {
	    if (sname) {
		nbmin = 8;
	    } else {
		nbmin = 8;
	    }
	} else if (sname && s_cmp(c3, "TRD") == 0) {
	    nbmin = 2;
	}
    } else if (cname && s_cmp(c2, "HE") == 0) {
	if (s_cmp(c3, "TRD") == 0) {
	    nbmin = 2;
	}
    } else if (sname && s_cmp(c2, "OR") == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nbmin = 2;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nbmin = 2;
	    }
	}
    } else if (cname && s_cmp(c2, "UN") == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nbmin = 2;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nbmin = 2;
	    }
	}
    } else if (s_cmp(c2, "GG") == 0) {
	nbmin = 2;
	if (s_cmp(c3, "HD3") == 0) {
	    nbmin = 2;
	}
    }
    ret_val = nbmin;
    return ret_val;
L70:
    //
    //    ISPEC = 3:  crossover point
    //
    nx = 0;
    if (s_cmp(c2, "GE") == 0) {
	if (s_cmp(c3, "QRF") == 0 || s_cmp(c3, "RQF") == 0 || s_cmp(c3, "LQF")
		 == 0 || s_cmp(c3, "QLF") == 0) {
	    if (sname) {
		nx = 128;
	    } else {
		nx = 128;
	    }
	} else if (s_cmp(c3, "HRD") == 0) {
	    if (sname) {
		nx = 128;
	    } else {
		nx = 128;
	    }
	} else if (s_cmp(c3, "BRD") == 0) {
	    if (sname) {
		nx = 128;
	    } else {
		nx = 128;
	    }
	}
    } else if (s_cmp(c2, "SY") == 0) {
	if (sname && s_cmp(c3, "TRD") == 0) {
	    nx = 32;
	}
    } else if (cname && s_cmp(c2, "HE") == 0) {
	if (s_cmp(c3, "TRD") == 0) {
	    nx = 32;
	}
    } else if (sname && s_cmp(c2, "OR") == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nx = 128;
	    }
	}
    } else if (cname && s_cmp(c2, "UN") == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR") == 0 || s_cmp(c4, "RQ") == 0 || s_cmp(c4,
		    "LQ") == 0 || s_cmp(c4, "QL") == 0 || s_cmp(c4, "HR") ==
		    0 || s_cmp(c4, "TR") == 0 || s_cmp(c4, "BR") == 0) {
		nx = 128;
	    }
	}
    } else if (s_cmp(c2, "GG") == 0) {
	nx = 128;
	if (s_cmp(c3, "HD3") == 0) {
	    nx = 128;
	}
    }
    ret_val = nx;
    return ret_val;
L80:
    //
    //    ISPEC = 4:  number of shifts (used by xHSEQR)
    //
    ret_val = 6;
    return ret_val;
L90:
    //
    //    ISPEC = 5:  minimum column dimension (not used)
    //
    ret_val = 2;
    return ret_val;
L100:
    //
    //    ISPEC = 6:  crossover point for SVD (used by xGELSS and xGESVD)
    //
    ret_val = (int) ((float) min(*n1,*n2) * 1.6f);
    return ret_val;
L110:
    //
    //    ISPEC = 7:  number of processors (not used)
    //
    ret_val = 1;
    return ret_val;
L120:
    //
    //    ISPEC = 8:  crossover point for multishift (used by xHSEQR)
    //
    ret_val = 50;
    return ret_val;
L130:
    //
    //    ISPEC = 9:  maximum size of the subproblems at the bottom of the
    //                computation tree in the divide-and-conquer algorithm
    //                (used by xGELSD and xGESDD)
    //
    ret_val = 25;
    return ret_val;
L140:
    //
    //    ISPEC = 10: ieee NaN arithmetic can be trusted not to trap
    //
    //    ILAENV = 0
    ret_val = 1;
    if (ret_val == 1) {
	ret_val = ieeeck_(&c__1, &c_b174, &c_b175);
    }
    return ret_val;
L150:
    //
    //    ISPEC = 11: infinity arithmetic can be trusted not to trap
    //
    //    ILAENV = 0
    ret_val = 1;
    if (ret_val == 1) {
	ret_val = ieeeck_(&c__0, &c_b174, &c_b175);
    }
    return ret_val;
L160:
    //
    //    12 <= ISPEC <= 16: xHSEQR or related subroutines.
    //
    ret_val = iparmq_(ispec, name__, opts, n1, n2, n3, n4);
    return ret_val;
    //
    //    End of ILAENV
    //
} // ilaenv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b IPARMQ
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download IPARMQ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/iparmq.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/iparmq.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/iparmq.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER FUNCTION IPARMQ( ISPEC, NAME, OPTS, N, ILO, IHI, LWORK )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, ILO, ISPEC, LWORK, N
//      CHARACTER          NAME*( * ), OPTS*( * )
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>      This program sets problem and machine dependent parameters
//>      useful for xHSEQR and related subroutines for eigenvalue
//>      problems. It is called whenever
//>      IPARMQ is called with 12 <= ISPEC <= 16
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ISPEC
//> \verbatim
//>          ISPEC is INTEGER
//>              ISPEC specifies which tunable parameter IPARMQ should
//>              return.
//>
//>              ISPEC=12: (INMIN)  Matrices of order nmin or less
//>                        are sent directly to xLAHQR, the implicit
//>                        double shift QR algorithm.  NMIN must be
//>                        at least 11.
//>
//>              ISPEC=13: (INWIN)  Size of the deflation window.
//>                        This is best set greater than or equal to
//>                        the number of simultaneous shifts NS.
//>                        Larger matrices benefit from larger deflation
//>                        windows.
//>
//>              ISPEC=14: (INIBL) Determines when to stop nibbling and
//>                        invest in an (expensive) multi-shift QR sweep.
//>                        If the aggressive early deflation subroutine
//>                        finds LD converged eigenvalues from an order
//>                        NW deflation window and LD > (NW*NIBBLE)/100,
//>                        then the next QR sweep is skipped and early
//>                        deflation is applied immediately to the
//>                        remaining active diagonal block.  Setting
//>                        IPARMQ(ISPEC=14) = 0 causes TTQRE to skip a
//>                        multi-shift QR sweep whenever early deflation
//>                        finds a converged eigenvalue.  Setting
//>                        IPARMQ(ISPEC=14) greater than or equal to 100
//>                        prevents TTQRE from skipping a multi-shift
//>                        QR sweep.
//>
//>              ISPEC=15: (NSHFTS) The number of simultaneous shifts in
//>                        a multi-shift QR iteration.
//>
//>              ISPEC=16: (IACC22) IPARMQ is set to 0, 1 or 2 with the
//>                        following meanings.
//>                        0:  During the multi-shift QR/QZ sweep,
//>                            blocked eigenvalue reordering, blocked
//>                            Hessenberg-triangular reduction,
//>                            reflections and/or rotations are not
//>                            accumulated when updating the
//>                            far-from-diagonal matrix entries.
//>                        1:  During the multi-shift QR/QZ sweep,
//>                            blocked eigenvalue reordering, blocked
//>                            Hessenberg-triangular reduction,
//>                            reflections and/or rotations are
//>                            accumulated, and matrix-matrix
//>                            multiplication is used to update the
//>                            far-from-diagonal matrix entries.
//>                        2:  During the multi-shift QR/QZ sweep,
//>                            blocked eigenvalue reordering, blocked
//>                            Hessenberg-triangular reduction,
//>                            reflections and/or rotations are
//>                            accumulated, and 2-by-2 block structure
//>                            is exploited during matrix-matrix
//>                            multiplies.
//>                        (If xTRMM is slower than xGEMM, then
//>                        IPARMQ(ISPEC=16)=1 may be more efficient than
//>                        IPARMQ(ISPEC=16)=2 despite the greater level of
//>                        arithmetic work implied by the latter choice.)
//> \endverbatim
//>
//> \param[in] NAME
//> \verbatim
//>          NAME is CHARACTER string
//>               Name of the calling subroutine
//> \endverbatim
//>
//> \param[in] OPTS
//> \verbatim
//>          OPTS is CHARACTER string
//>               This is a concatenation of the string arguments to
//>               TTQRE.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>               N is the order of the Hessenberg matrix H.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>               It is assumed that H is already upper triangular
//>               in rows and columns 1:ILO-1 and IHI+1:N.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>               The amount of workspace available.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup OTHERauxiliary
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>       Little is known about how best to choose these parameters.
//>       It is possible to use different values of the parameters
//>       for each of CHSEQR, DHSEQR, SHSEQR and ZHSEQR.
//>
//>       It is probably best to choose different parameters for
//>       different matrices and different parameters at different
//>       times during the iteration, but this has not been
//>       implemented --- yet.
//>
//>
//>       The best choices of most of the parameters depend
//>       in an ill-understood way on the relative execution
//>       rate of xLAQR3 and xLAQR5 and on the nature of each
//>       particular eigenvalue problem.  Experiment may be the
//>       only practical way to determine which choices are most
//>       effective.
//>
//>       Following is a list of default values supplied by IPARMQ.
//>       These defaults may be adjusted in order to attain better
//>       performance in any particular computational environment.
//>
//>       IPARMQ(ISPEC=12) The xLAHQR vs xLAQR0 crossover point.
//>                        Default: 75. (Must be at least 11.)
//>
//>       IPARMQ(ISPEC=13) Recommended deflation window size.
//>                        This depends on ILO, IHI and NS, the
//>                        number of simultaneous shifts returned
//>                        by IPARMQ(ISPEC=15).  The default for
//>                        (IHI-ILO+1) <= 500 is NS.  The default
//>                        for (IHI-ILO+1) > 500 is 3*NS/2.
//>
//>       IPARMQ(ISPEC=14) Nibble crossover point.  Default: 14.
//>
//>       IPARMQ(ISPEC=15) Number of simultaneous shifts, NS.
//>                        a multi-shift QR iteration.
//>
//>                        If IHI-ILO+1 is ...
//>
//>                        greater than      ...but less    ... the
//>                        or equal to ...      than        default is
//>
//>                                0               30       NS =   2+
//>                               30               60       NS =   4+
//>                               60              150       NS =  10
//>                              150              590       NS =  **
//>                              590             3000       NS =  64
//>                             3000             6000       NS = 128
//>                             6000             infinity   NS = 256
//>
//>                    (+)  By default matrices of this order are
//>                         passed to the implicit double shift routine
//>                         xLAHQR.  See IPARMQ(ISPEC=12) above.   These
//>                         values of NS are used only in case of a rare
//>                         xLAHQR failure.
//>
//>                    (**) The asterisks (**) indicate an ad-hoc
//>                         function increasing from 10 to 64.
//>
//>       IPARMQ(ISPEC=16) Select structured matrix multiply.
//>                        (See ISPEC=16 above for details.)
//>                        Default: 3.
//> \endverbatim
//>
// =====================================================================
int iparmq_(int *ispec, char *name__, char *opts, int *n, int *ilo, int *ihi,
	int *lwork)
{
    // System generated locals
    int ret_val, i__1, i__2;
    float r__1;

    // Local variables
    int i__, ic, nh, ns, iz;
    char subnam[6+1]={'\0'};

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //
    // ================================================================
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    if (*ispec == 15 || *ispec == 13 || *ispec == 16) {
	//
	//       ==== Set the number simultaneous shifts ====
	//
	nh = *ihi - *ilo + 1;
	ns = 2;
	if (nh >= 30) {
	    ns = 4;
	}
	if (nh >= 60) {
	    ns = 10;
	}
	if (nh >= 150) {
	    // Computing MAX
	    r__1 = log((float) nh) / log(2.f);
	    i__1 = 10, i__2 = nh / i_nint(&r__1);
	    ns = max(i__1,i__2);
	}
	if (nh >= 590) {
	    ns = 64;
	}
	if (nh >= 3000) {
	    ns = 128;
	}
	if (nh >= 6000) {
	    ns = 256;
	}
	// Computing MAX
	i__1 = 2, i__2 = ns - ns % 2;
	ns = max(i__1,i__2);
    }
    if (*ispec == 12) {
	//
	//
	//       ===== Matrices of order smaller than NMIN get sent
	//       .     to xLAHQR, the classic double shift algorithm.
	//       .     This must be at least 11. ====
	//
	ret_val = 75;
    } else if (*ispec == 14) {
	//
	//       ==== INIBL: skip a multi-shift qr iteration and
	//       .    whenever aggressive early deflation finds
	//       .    at least (NIBBLE*(window size)/100) deflations. ====
	//
	ret_val = 14;
    } else if (*ispec == 15) {
	//
	//       ==== NSHFTS: The number of simultaneous shifts =====
	//
	ret_val = ns;
    } else if (*ispec == 13) {
	//
	//       ==== NW: deflation window size.  ====
	//
	if (nh <= 500) {
	    ret_val = ns;
	} else {
	    ret_val = ns * 3 / 2;
	}
    } else if (*ispec == 16) {
	//
	//       ==== IACC22: Whether to accumulate reflections
	//       .     before updating the far-from-diagonal elements
	//       .     and whether to use 2-by-2 block structure while
	//       .     doing it.  A small amount of work could be saved
	//       .     by making this choice dependent also upon the
	//       .     NH=IHI-ILO+1.
	//
	//
	//       Convert NAME to upper case if the first character is lower case.
	//
	ret_val = 0;
	s_copy(subnam, name__, (int)6);
	ic = *(unsigned char *)subnam;
	iz = 'Z';
	if (iz == 90 || iz == 122) {
	    //
	    //          ASCII character set
	    //
	    if (ic >= 97 && ic <= 122) {
		*(unsigned char *)subnam = (char) (ic - 32);
		for (i__ = 2; i__ <= 6; ++i__) {
		    ic = *(unsigned char *)&subnam[i__ - 1];
		    if (ic >= 97 && ic <= 122) {
			*(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);
		    }
		}
	    }
	} else if (iz == 233 || iz == 169) {
	    //
	    //          EBCDIC character set
	    //
	    if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >= 162
		    && ic <= 169) {
		*(unsigned char *)subnam = (char) (ic + 64);
		for (i__ = 2; i__ <= 6; ++i__) {
		    ic = *(unsigned char *)&subnam[i__ - 1];
		    if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 ||
			    ic >= 162 && ic <= 169) {
			*(unsigned char *)&subnam[i__ - 1] = (char) (ic + 64);
		    }
		}
	    }
	} else if (iz == 218 || iz == 250) {
	    //
	    //          Prime machines:  ASCII+128
	    //
	    if (ic >= 225 && ic <= 250) {
		*(unsigned char *)subnam = (char) (ic - 32);
		for (i__ = 2; i__ <= 6; ++i__) {
		    ic = *(unsigned char *)&subnam[i__ - 1];
		    if (ic >= 225 && ic <= 250) {
			*(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);
		    }
		}
	    }
	}
	if (s_cmp(subnam + 1, "GGHRD") == 0 || s_cmp(subnam + 1, "GGHD3") ==
		0) {
	    ret_val = 1;
	    if (nh >= 14) {
		ret_val = 2;
	    }
	} else if (s_cmp(subnam + 3, "EXC") == 0) {
	    if (nh >= 14) {
		ret_val = 1;
	    }
	    if (nh >= 14) {
		ret_val = 2;
	    }
	} else if (s_cmp(subnam + 1, "HSEQR") == 0 || s_cmp(subnam + 1, "LAQR"
		) == 0) {
	    if (ns >= 14) {
		ret_val = 1;
	    }
	    if (ns >= 14) {
		ret_val = 2;
	    }
	}
    } else {
	//       ===== invalid value of ispec =====
	ret_val = -1;
    }
    //
    //    ==== End of IPARMQ ====
    //
    return ret_val;
} // iparmq_

