#include "clapack.h"

/* Subroutine */ int slasd4_(integer *n, integer *i__, real *d__, real *z__, 
	real *delta, real *rho, real *sigma, real *work, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    real a, b, c__;
    integer j;
    real w, dd[3];
    integer ii;
    real dw, zz[3];
    integer ip1;
    real eta, phi, eps, tau, psi;
    integer iim1, iip1;
    real dphi, dpsi;
    integer iter;
    real temp, prew, sg2lb, sg2ub, temp1, temp2, dtiim, delsq, dtiip;
    integer niter;
    real dtisq;
    logical swtch;
    real dtnsq;
    extern /* Subroutine */ int slaed6_(integer *, logical *, real *, real *, 
	    real *, real *, real *, integer *);
    real delsq2;
    extern /* Subroutine */ int slasd5_(integer *, real *, real *, real *, 
	    real *, real *, real *);
    real dtnsq1;
    logical swtch3;
    extern doublereal slamch_(char *);
    logical orgati;
    real erretm, dtipsq, rhoinv;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This subroutine computes the square root of the I-th updated */
/*  eigenvalue of a positive symmetric rank-one modification to */
/*  a positive diagonal matrix whose entries are given as the squares */
/*  of the corresponding entries in the array d, and that */

/*         0 <= D(i) < D(j)  for  i < j */

/*  and that RHO > 0. This is arranged by the calling routine, and is */
/*  no loss in generality.  The rank-one modified system is thus */

/*         diag( D ) * diag( D ) +  RHO *  Z * Z_transpose. */

/*  where we assume the Euclidean norm of Z is 1. */

/*  The method consists of approximating the rational functions in the */
/*  secular equation by simpler interpolating rational functions. */

/*  Arguments */
/*  ========= */

/*  N      (input) INTEGER */
/*         The length of all arrays. */

/*  I      (input) INTEGER */
/*         The index of the eigenvalue to be computed.  1 <= I <= N. */

/*  D      (input) REAL array, dimension ( N ) */
/*         The original eigenvalues.  It is assumed that they are in */
/*         order, 0 <= D(I) < D(J)  for I < J. */

/*  Z      (input) REAL array, dimension (N) */
/*         The components of the updating vector. */

/*  DELTA  (output) REAL array, dimension (N) */
/*         If N .ne. 1, DELTA contains (D(j) - sigma_I) in its  j-th */
/*         component.  If N = 1, then DELTA(1) = 1.  The vector DELTA */
/*         contains the information necessary to construct the */
/*         (singular) eigenvectors. */

/*  RHO    (input) REAL */
/*         The scalar in the symmetric updating formula. */

/*  SIGMA  (output) REAL */
/*         The computed sigma_I, the I-th updated eigenvalue. */

/*  WORK   (workspace) REAL array, dimension (N) */
/*         If N .ne. 1, WORK contains (D(j) + sigma_I) in its  j-th */
/*         component.  If N = 1, then WORK( 1 ) = 1. */

/*  INFO   (output) INTEGER */
/*         = 0:  successful exit */
/*         > 0:  if INFO = 1, the updating process failed. */

/*  Internal Parameters */
/*  =================== */

/*  Logical variable ORGATI (origin-at-i?) is used for distinguishing */
/*  whether D(i) or D(i+1) is treated as the origin. */

/*            ORGATI = .true.    origin at i */
/*            ORGATI = .false.   origin at i+1 */

/*  Logical variable SWTCH3 (switch-for-3-poles?) is for noting */
/*  if we are working with THREE poles! */

/*  MAXIT is the maximum number of iterations allowed for each */
/*  eigenvalue. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ren-Cang Li, Computer Science Division, University of California */
/*     at Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Since this routine is called in an inner loop, we do no argument */
/*     checking. */

/*     Quick return for N=1 and 2. */

    /* Parameter adjustments */
    --work;
    --delta;
    --z__;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n == 1) {

/*        Presumably, I=1 upon entry */

	*sigma = sqrt(d__[1] * d__[1] + *rho * z__[1] * z__[1]);
	delta[1] = 1.f;
	work[1] = 1.f;
	return 0;
    }
    if (*n == 2) {
	slasd5_(i__, &d__[1], &z__[1], &delta[1], rho, sigma, &work[1]);
	return 0;
    }

/*     Compute machine epsilon */

    eps = slamch_("Epsilon");
    rhoinv = 1.f / *rho;

/*     The case I = N */

    if (*i__ == *n) {

/*        Initialize some basic variables */

	ii = *n - 1;
	niter = 1;

/*        Calculate initial guess */

	temp = *rho / 2.f;

/*        If ||Z||_2 is not one, then TEMP should be set to */
/*        RHO * ||Z||_2^2 / TWO */

	temp1 = temp / (d__[*n] + sqrt(d__[*n] * d__[*n] + temp));
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] = d__[j] + d__[*n] + temp1;
	    delta[j] = d__[j] - d__[*n] - temp1;
/* L10: */
	}

	psi = 0.f;
	i__1 = *n - 2;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / (delta[j] * work[j]);
/* L20: */
	}

	c__ = rhoinv + psi;
	w = c__ + z__[ii] * z__[ii] / (delta[ii] * work[ii]) + z__[*n] * z__[*
		n] / (delta[*n] * work[*n]);

	if (w <= 0.f) {
	    temp1 = sqrt(d__[*n] * d__[*n] + *rho);
	    temp = z__[*n - 1] * z__[*n - 1] / ((d__[*n - 1] + temp1) * (d__[*
		    n] - d__[*n - 1] + *rho / (d__[*n] + temp1))) + z__[*n] * 
		    z__[*n] / *rho;

/*           The following TAU is to approximate */
/*           SIGMA_n^2 - D( N )*D( N ) */

	    if (c__ <= temp) {
		tau = *rho;
	    } else {
		delsq = (d__[*n] - d__[*n - 1]) * (d__[*n] + d__[*n - 1]);
		a = -c__ * delsq + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*
			n];
		b = z__[*n] * z__[*n] * delsq;
		if (a < 0.f) {
		    tau = b * 2.f / (sqrt(a * a + b * 4.f * c__) - a);
		} else {
		    tau = (a + sqrt(a * a + b * 4.f * c__)) / (c__ * 2.f);
		}
	    }

/*           It can be proved that */
/*               D(N)^2+RHO/2 <= SIGMA_n^2 < D(N)^2+TAU <= D(N)^2+RHO */

	} else {
	    delsq = (d__[*n] - d__[*n - 1]) * (d__[*n] + d__[*n - 1]);
	    a = -c__ * delsq + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*n];
	    b = z__[*n] * z__[*n] * delsq;

/*           The following TAU is to approximate */
/*           SIGMA_n^2 - D( N )*D( N ) */

	    if (a < 0.f) {
		tau = b * 2.f / (sqrt(a * a + b * 4.f * c__) - a);
	    } else {
		tau = (a + sqrt(a * a + b * 4.f * c__)) / (c__ * 2.f);
	    }

/*           It can be proved that */
/*           D(N)^2 < D(N)^2+TAU < SIGMA(N)^2 < D(N)^2+RHO/2 */

	}

/*        The following ETA is to approximate SIGMA_n - D( N ) */

	eta = tau / (d__[*n] + sqrt(d__[*n] * d__[*n] + tau));

	*sigma = d__[*n] + eta;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - eta;
	    work[j] = d__[j] + d__[*i__] + eta;
/* L30: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (delta[j] * work[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L40: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / (delta[*n] * work[*n]);
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) * (
		dpsi + dphi);

	w = rhoinv + phi + psi;

/*        Test for convergence */

	if (dabs(w) <= eps * erretm) {
	    goto L240;
	}

/*        Calculate the new step */

	++niter;
	dtnsq1 = work[*n - 1] * delta[*n - 1];
	dtnsq = work[*n] * delta[*n];
	c__ = w - dtnsq1 * dpsi - dtnsq * dphi;
	a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
	b = dtnsq * dtnsq1 * w;
	if (c__ < 0.f) {
	    c__ = dabs(c__);
	}
	if (c__ == 0.f) {
	    eta = *rho - *sigma * *sigma;
	} else if (a >= 0.f) {
	    eta = (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) / (
		    c__ * 2.f);
	} else {
	    eta = b * 2.f / (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(
		    r__1))));
	}

/*        Note, eta should be positive if w is negative, and */
/*        eta should be negative otherwise. However, */
/*        if for some reason caused by roundoff, eta*w > 0, */
/*        we simply use one Newton step instead. This way */
/*        will guarantee eta*w < 0. */

	if (w * eta > 0.f) {
	    eta = -w / (dpsi + dphi);
	}
	temp = eta - dtnsq;
	if (temp > *rho) {
	    eta = *rho + dtnsq;
	}

	tau += eta;
	eta /= *sigma + sqrt(eta + *sigma * *sigma);
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] -= eta;
	    work[j] += eta;
/* L50: */
	}

	*sigma += eta;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L60: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / (work[*n] * delta[*n]);
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) * (
		dpsi + dphi);

	w = rhoinv + phi + psi;

/*        Main loop to update the values of the array   DELTA */

	iter = niter + 1;

	for (niter = iter; niter <= 20; ++niter) {

/*           Test for convergence */

	    if (dabs(w) <= eps * erretm) {
		goto L240;
	    }

/*           Calculate the new step */

	    dtnsq1 = work[*n - 1] * delta[*n - 1];
	    dtnsq = work[*n] * delta[*n];
	    c__ = w - dtnsq1 * dpsi - dtnsq * dphi;
	    a = (dtnsq + dtnsq1) * w - dtnsq1 * dtnsq * (dpsi + dphi);
	    b = dtnsq1 * dtnsq * w;
	    if (a >= 0.f) {
		eta = (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    } else {
		eta = b * 2.f / (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    }

/*           Note, eta should be positive if w is negative, and */
/*           eta should be negative otherwise. However, */
/*           if for some reason caused by roundoff, eta*w > 0, */
/*           we simply use one Newton step instead. This way */
/*           will guarantee eta*w < 0. */

	    if (w * eta > 0.f) {
		eta = -w / (dpsi + dphi);
	    }
	    temp = eta - dtnsq;
	    if (temp <= 0.f) {
		eta /= 2.f;
	    }

	    tau += eta;
	    eta /= *sigma + sqrt(eta + *sigma * *sigma);
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] -= eta;
		work[j] += eta;
/* L70: */
	    }

	    *sigma += eta;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.f;
	    psi = 0.f;
	    erretm = 0.f;
	    i__1 = ii;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / (work[j] * delta[j]);
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L80: */
	    }
	    erretm = dabs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    temp = z__[*n] / (work[*n] * delta[*n]);
	    phi = z__[*n] * temp;
	    dphi = temp * temp;
	    erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) * 
		    (dpsi + dphi);

	    w = rhoinv + phi + psi;
/* L90: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;
	goto L240;

/*        End for the case I = N */

    } else {

/*        The case for I < N */

	niter = 1;
	ip1 = *i__ + 1;

/*        Calculate initial guess */

	delsq = (d__[ip1] - d__[*i__]) * (d__[ip1] + d__[*i__]);
	delsq2 = delsq / 2.f;
	temp = delsq2 / (d__[*i__] + sqrt(d__[*i__] * d__[*i__] + delsq2));
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] = d__[j] + d__[*i__] + temp;
	    delta[j] = d__[j] - d__[*i__] - temp;
/* L100: */
	}

	psi = 0.f;
	i__1 = *i__ - 1;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / (work[j] * delta[j]);
/* L110: */
	}

	phi = 0.f;
	i__1 = *i__ + 2;
	for (j = *n; j >= i__1; --j) {
	    phi += z__[j] * z__[j] / (work[j] * delta[j]);
/* L120: */
	}
	c__ = rhoinv + psi + phi;
	w = c__ + z__[*i__] * z__[*i__] / (work[*i__] * delta[*i__]) + z__[
		ip1] * z__[ip1] / (work[ip1] * delta[ip1]);

	if (w > 0.f) {

/*           d(i)^2 < the ith sigma^2 < (d(i)^2+d(i+1)^2)/2 */

/*           We choose d(i) as origin. */

	    orgati = TRUE_;
	    sg2lb = 0.f;
	    sg2ub = delsq2;
	    a = c__ * delsq + z__[*i__] * z__[*i__] + z__[ip1] * z__[ip1];
	    b = z__[*i__] * z__[*i__] * delsq;
	    if (a > 0.f) {
		tau = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    } else {
		tau = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    }

/*           TAU now is an estimation of SIGMA^2 - D( I )^2. The */
/*           following, however, is the corresponding estimation of */
/*           SIGMA - D( I ). */

	    eta = tau / (d__[*i__] + sqrt(d__[*i__] * d__[*i__] + tau));
	} else {

/*           (d(i)^2+d(i+1)^2)/2 <= the ith sigma^2 < d(i+1)^2/2 */

/*           We choose d(i+1) as origin. */

	    orgati = FALSE_;
	    sg2lb = -delsq2;
	    sg2ub = 0.f;
	    a = c__ * delsq - z__[*i__] * z__[*i__] - z__[ip1] * z__[ip1];
	    b = z__[ip1] * z__[ip1] * delsq;
	    if (a < 0.f) {
		tau = b * 2.f / (a - sqrt((r__1 = a * a + b * 4.f * c__, dabs(
			r__1))));
	    } else {
		tau = -(a + sqrt((r__1 = a * a + b * 4.f * c__, dabs(r__1)))) 
			/ (c__ * 2.f);
	    }

/*           TAU now is an estimation of SIGMA^2 - D( IP1 )^2. The */
/*           following, however, is the corresponding estimation of */
/*           SIGMA - D( IP1 ). */

	    eta = tau / (d__[ip1] + sqrt((r__1 = d__[ip1] * d__[ip1] + tau, 
		    dabs(r__1))));
	}

	if (orgati) {
	    ii = *i__;
	    *sigma = d__[*i__] + eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] = d__[j] + d__[*i__] + eta;
		delta[j] = d__[j] - d__[*i__] - eta;
/* L130: */
	    }
	} else {
	    ii = *i__ + 1;
	    *sigma = d__[ip1] + eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] = d__[j] + d__[ip1] + eta;
		delta[j] = d__[j] - d__[ip1] - eta;
/* L140: */
	    }
	}
	iim1 = ii - 1;
	iip1 = ii + 1;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L150: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.f;
	phi = 0.f;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
/* L160: */
	}

	w = rhoinv + phi + psi;

/*        W is the value of the secular function with */
/*        its ii-th element removed. */

	swtch3 = FALSE_;
	if (orgati) {
	    if (w < 0.f) {
		swtch3 = TRUE_;
	    }
	} else {
	    if (w > 0.f) {
		swtch3 = TRUE_;
	    }
	}
	if (ii == 1 || ii == *n) {
	    swtch3 = FALSE_;
	}

	temp = z__[ii] / (work[ii] * delta[ii]);
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w += temp;
	erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) * 3.f 
		+ dabs(tau) * dw;

/*        Test for convergence */

	if (dabs(w) <= eps * erretm) {
	    goto L240;
	}

	if (w <= 0.f) {
	    sg2lb = dmax(sg2lb,tau);
	} else {
	    sg2ub = dmin(sg2ub,tau);
	}

/*        Calculate the new step */

	++niter;
	if (! swtch3) {
	    dtipsq = work[ip1] * delta[ip1];
	    dtisq = work[*i__] * delta[*i__];
	    if (orgati) {
/* Computing 2nd power */
		r__1 = z__[*i__] / dtisq;
		c__ = w - dtipsq * dw + delsq * (r__1 * r__1);
	    } else {
/* Computing 2nd power */
		r__1 = z__[ip1] / dtipsq;
		c__ = w - dtisq * dw - delsq * (r__1 * r__1);
	    }
	    a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
	    b = dtipsq * dtisq * w;
	    if (c__ == 0.f) {
		if (a == 0.f) {
		    if (orgati) {
			a = z__[*i__] * z__[*i__] + dtipsq * dtipsq * (dpsi + 
				dphi);
		    } else {
			a = z__[ip1] * z__[ip1] + dtisq * dtisq * (dpsi + 
				dphi);
		    }
		}
		eta = b / a;
	    } else if (a <= 0.f) {
		eta = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    } else {
		eta = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    }
	} else {

/*           Interpolation using THREE most relevant poles */

	    dtiim = work[iim1] * delta[iim1];
	    dtiip = work[iip1] * delta[iip1];
	    temp = rhoinv + psi + phi;
	    if (orgati) {
		temp1 = z__[iim1] / dtiim;
		temp1 *= temp1;
		c__ = temp - dtiip * (dpsi + dphi) - (d__[iim1] - d__[iip1]) *
			 (d__[iim1] + d__[iip1]) * temp1;
		zz[0] = z__[iim1] * z__[iim1];
		if (dpsi < temp1) {
		    zz[2] = dtiip * dtiip * dphi;
		} else {
		    zz[2] = dtiip * dtiip * (dpsi - temp1 + dphi);
		}
	    } else {
		temp1 = z__[iip1] / dtiip;
		temp1 *= temp1;
		c__ = temp - dtiim * (dpsi + dphi) - (d__[iip1] - d__[iim1]) *
			 (d__[iim1] + d__[iip1]) * temp1;
		if (dphi < temp1) {
		    zz[0] = dtiim * dtiim * dpsi;
		} else {
		    zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
		}
		zz[2] = z__[iip1] * z__[iip1];
	    }
	    zz[1] = z__[ii] * z__[ii];
	    dd[0] = dtiim;
	    dd[1] = delta[ii] * work[ii];
	    dd[2] = dtiip;
	    slaed6_(&niter, &orgati, &c__, dd, zz, &w, &eta, info);
	    if (*info != 0) {
		goto L240;
	    }
	}

/*        Note, eta should be positive if w is negative, and */
/*        eta should be negative otherwise. However, */
/*        if for some reason caused by roundoff, eta*w > 0, */
/*        we simply use one Newton step instead. This way */
/*        will guarantee eta*w < 0. */

	if (w * eta >= 0.f) {
	    eta = -w / dw;
	}
	if (orgati) {
	    temp1 = work[*i__] * delta[*i__];
	    temp = eta - temp1;
	} else {
	    temp1 = work[ip1] * delta[ip1];
	    temp = eta - temp1;
	}
	if (temp > sg2ub || temp < sg2lb) {
	    if (w < 0.f) {
		eta = (sg2ub - tau) / 2.f;
	    } else {
		eta = (sg2lb - tau) / 2.f;
	    }
	}

	tau += eta;
	eta /= *sigma + sqrt(*sigma * *sigma + eta);

	prew = w;

	*sigma += eta;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] += eta;
	    delta[j] -= eta;
/* L170: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L180: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.f;
	phi = 0.f;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
/* L190: */
	}

	temp = z__[ii] / (work[ii] * delta[ii]);
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w = rhoinv + phi + psi + temp;
	erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) * 3.f 
		+ dabs(tau) * dw;

	if (w <= 0.f) {
	    sg2lb = dmax(sg2lb,tau);
	} else {
	    sg2ub = dmin(sg2ub,tau);
	}

	swtch = FALSE_;
	if (orgati) {
	    if (-w > dabs(prew) / 10.f) {
		swtch = TRUE_;
	    }
	} else {
	    if (w > dabs(prew) / 10.f) {
		swtch = TRUE_;
	    }
	}

/*        Main loop to update the values of the array   DELTA and WORK */

	iter = niter + 1;

	for (niter = iter; niter <= 20; ++niter) {

/*           Test for convergence */

	    if (dabs(w) <= eps * erretm) {
		goto L240;
	    }

/*           Calculate the new step */

	    if (! swtch3) {
		dtipsq = work[ip1] * delta[ip1];
		dtisq = work[*i__] * delta[*i__];
		if (! swtch) {
		    if (orgati) {
/* Computing 2nd power */
			r__1 = z__[*i__] / dtisq;
			c__ = w - dtipsq * dw + delsq * (r__1 * r__1);
		    } else {
/* Computing 2nd power */
			r__1 = z__[ip1] / dtipsq;
			c__ = w - dtisq * dw - delsq * (r__1 * r__1);
		    }
		} else {
		    temp = z__[ii] / (work[ii] * delta[ii]);
		    if (orgati) {
			dpsi += temp * temp;
		    } else {
			dphi += temp * temp;
		    }
		    c__ = w - dtisq * dpsi - dtipsq * dphi;
		}
		a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
		b = dtipsq * dtisq * w;
		if (c__ == 0.f) {
		    if (a == 0.f) {
			if (! swtch) {
			    if (orgati) {
				a = z__[*i__] * z__[*i__] + dtipsq * dtipsq * 
					(dpsi + dphi);
			    } else {
				a = z__[ip1] * z__[ip1] + dtisq * dtisq * (
					dpsi + dphi);
			    }
			} else {
			    a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
			}
		    }
		    eta = b / a;
		} else if (a <= 0.f) {
		    eta = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1))
			    )) / (c__ * 2.f);
		} else {
		    eta = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, 
			    dabs(r__1))));
		}
	    } else {

/*              Interpolation using THREE most relevant poles */

		dtiim = work[iim1] * delta[iim1];
		dtiip = work[iip1] * delta[iip1];
		temp = rhoinv + psi + phi;
		if (swtch) {
		    c__ = temp - dtiim * dpsi - dtiip * dphi;
		    zz[0] = dtiim * dtiim * dpsi;
		    zz[2] = dtiip * dtiip * dphi;
		} else {
		    if (orgati) {
			temp1 = z__[iim1] / dtiim;
			temp1 *= temp1;
			temp2 = (d__[iim1] - d__[iip1]) * (d__[iim1] + d__[
				iip1]) * temp1;
			c__ = temp - dtiip * (dpsi + dphi) - temp2;
			zz[0] = z__[iim1] * z__[iim1];
			if (dpsi < temp1) {
			    zz[2] = dtiip * dtiip * dphi;
			} else {
			    zz[2] = dtiip * dtiip * (dpsi - temp1 + dphi);
			}
		    } else {
			temp1 = z__[iip1] / dtiip;
			temp1 *= temp1;
			temp2 = (d__[iip1] - d__[iim1]) * (d__[iim1] + d__[
				iip1]) * temp1;
			c__ = temp - dtiim * (dpsi + dphi) - temp2;
			if (dphi < temp1) {
			    zz[0] = dtiim * dtiim * dpsi;
			} else {
			    zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
			}
			zz[2] = z__[iip1] * z__[iip1];
		    }
		}
		dd[0] = dtiim;
		dd[1] = delta[ii] * work[ii];
		dd[2] = dtiip;
		slaed6_(&niter, &orgati, &c__, dd, zz, &w, &eta, info);
		if (*info != 0) {
		    goto L240;
		}
	    }

/*           Note, eta should be positive if w is negative, and */
/*           eta should be negative otherwise. However, */
/*           if for some reason caused by roundoff, eta*w > 0, */
/*           we simply use one Newton step instead. This way */
/*           will guarantee eta*w < 0. */

	    if (w * eta >= 0.f) {
		eta = -w / dw;
	    }
	    if (orgati) {
		temp1 = work[*i__] * delta[*i__];
		temp = eta - temp1;
	    } else {
		temp1 = work[ip1] * delta[ip1];
		temp = eta - temp1;
	    }
	    if (temp > sg2ub || temp < sg2lb) {
		if (w < 0.f) {
		    eta = (sg2ub - tau) / 2.f;
		} else {
		    eta = (sg2lb - tau) / 2.f;
		}
	    }

	    tau += eta;
	    eta /= *sigma + sqrt(*sigma * *sigma + eta);

	    *sigma += eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] += eta;
		delta[j] -= eta;
/* L200: */
	    }

	    prew = w;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.f;
	    psi = 0.f;
	    erretm = 0.f;
	    i__1 = iim1;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / (work[j] * delta[j]);
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L210: */
	    }
	    erretm = dabs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    dphi = 0.f;
	    phi = 0.f;
	    i__1 = iip1;
	    for (j = *n; j >= i__1; --j) {
		temp = z__[j] / (work[j] * delta[j]);
		phi += z__[j] * temp;
		dphi += temp * temp;
		erretm += phi;
/* L220: */
	    }

	    temp = z__[ii] / (work[ii] * delta[ii]);
	    dw = dpsi + dphi + temp * temp;
	    temp = z__[ii] * temp;
	    w = rhoinv + phi + psi + temp;
	    erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) * 
		    3.f + dabs(tau) * dw;
	    if (w * prew > 0.f && dabs(w) > dabs(prew) / 10.f) {
		swtch = ! swtch;
	    }

	    if (w <= 0.f) {
		sg2lb = dmax(sg2lb,tau);
	    } else {
		sg2ub = dmin(sg2ub,tau);
	    }

/* L230: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;

    }

L240:
    return 0;

/*     End of SLASD4 */

} /* slasd4_ */
