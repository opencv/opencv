/* dlaed4.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int dlaed4_(integer *n, integer *i__, doublereal *d__, 
	doublereal *z__, doublereal *delta, doublereal *rho, doublereal *dlam, 
	 integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    doublereal a, b, c__;
    integer j;
    doublereal w;
    integer ii;
    doublereal dw, zz[3];
    integer ip1;
    doublereal del, eta, phi, eps, tau, psi;
    integer iim1, iip1;
    doublereal dphi, dpsi;
    integer iter;
    doublereal temp, prew, temp1, dltlb, dltub, midpt;
    integer niter;
    logical swtch;
    extern /* Subroutine */ int dlaed5_(integer *, doublereal *, doublereal *, 
	     doublereal *, doublereal *, doublereal *), dlaed6_(integer *, 
	    logical *, doublereal *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, integer *);
    logical swtch3;
    extern doublereal dlamch_(char *);
    logical orgati;
    doublereal erretm, rhoinv;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This subroutine computes the I-th updated eigenvalue of a symmetric */
/*  rank-one modification to a diagonal matrix whose elements are */
/*  given in the array d, and that */

/*             D(i) < D(j)  for  i < j */

/*  and that RHO > 0.  This is arranged by the calling routine, and is */
/*  no loss in generality.  The rank-one modified system is thus */

/*             diag( D )  +  RHO *  Z * Z_transpose. */

/*  where we assume the Euclidean norm of Z is 1. */

/*  The method consists of approximating the rational functions in the */
/*  secular equation by simpler interpolating rational functions. */

/*  Arguments */
/*  ========= */

/*  N      (input) INTEGER */
/*         The length of all arrays. */

/*  I      (input) INTEGER */
/*         The index of the eigenvalue to be computed.  1 <= I <= N. */

/*  D      (input) DOUBLE PRECISION array, dimension (N) */
/*         The original eigenvalues.  It is assumed that they are in */
/*         order, D(I) < D(J)  for I < J. */

/*  Z      (input) DOUBLE PRECISION array, dimension (N) */
/*         The components of the updating vector. */

/*  DELTA  (output) DOUBLE PRECISION array, dimension (N) */
/*         If N .GT. 2, DELTA contains (D(j) - lambda_I) in its  j-th */
/*         component.  If N = 1, then DELTA(1) = 1. If N = 2, see DLAED5 */
/*         for detail. The vector DELTA contains the information necessary */
/*         to construct the eigenvectors by DLAED3 and DLAED9. */

/*  RHO    (input) DOUBLE PRECISION */
/*         The scalar in the symmetric updating formula. */

/*  DLAM   (output) DOUBLE PRECISION */
/*         The computed lambda_I, the I-th updated eigenvalue. */

/*  INFO   (output) INTEGER */
/*         = 0:  successful exit */
/*         > 0:  if INFO = 1, the updating process failed. */

/*  Internal Parameters */
/*  =================== */

/*  Logical variable ORGATI (origin-at-i?) is used for distinguishing */
/*  whether D(i) or D(i+1) is treated as the origin. */

/*            ORGATI = .true.    origin at i */
/*            ORGATI = .false.   origin at i+1 */

/*   Logical variable SWTCH3 (switch-for-3-poles?) is for noting */
/*   if we are working with THREE poles! */

/*   MAXIT is the maximum number of iterations allowed for each */
/*   eigenvalue. */

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
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Since this routine is called in an inner loop, we do no argument */
/*     checking. */

/*     Quick return for N=1 and 2. */

    /* Parameter adjustments */
    --delta;
    --z__;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n == 1) {

/*         Presumably, I=1 upon entry */

	*dlam = d__[1] + *rho * z__[1] * z__[1];
	delta[1] = 1.;
	return 0;
    }
    if (*n == 2) {
	dlaed5_(i__, &d__[1], &z__[1], &delta[1], rho, dlam);
	return 0;
    }

/*     Compute machine epsilon */

    eps = dlamch_("Epsilon");
    rhoinv = 1. / *rho;

/*     The case I = N */

    if (*i__ == *n) {

/*        Initialize some basic variables */

	ii = *n - 1;
	niter = 1;

/*        Calculate initial guess */

	midpt = *rho / 2.;

/*        If ||Z||_2 is not one, then TEMP should be set to */
/*        RHO * ||Z||_2^2 / TWO */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - midpt;
/* L10: */
	}

	psi = 0.;
	i__1 = *n - 2;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / delta[j];
/* L20: */
	}

	c__ = rhoinv + psi;
	w = c__ + z__[ii] * z__[ii] / delta[ii] + z__[*n] * z__[*n] / delta[*
		n];

	if (w <= 0.) {
	    temp = z__[*n - 1] * z__[*n - 1] / (d__[*n] - d__[*n - 1] + *rho) 
		    + z__[*n] * z__[*n] / *rho;
	    if (c__ <= temp) {
		tau = *rho;
	    } else {
		del = d__[*n] - d__[*n - 1];
		a = -c__ * del + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*n]
			;
		b = z__[*n] * z__[*n] * del;
		if (a < 0.) {
		    tau = b * 2. / (sqrt(a * a + b * 4. * c__) - a);
		} else {
		    tau = (a + sqrt(a * a + b * 4. * c__)) / (c__ * 2.);
		}
	    }

/*           It can be proved that */
/*               D(N)+RHO/2 <= LAMBDA(N) < D(N)+TAU <= D(N)+RHO */

	    dltlb = midpt;
	    dltub = *rho;
	} else {
	    del = d__[*n] - d__[*n - 1];
	    a = -c__ * del + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*n];
	    b = z__[*n] * z__[*n] * del;
	    if (a < 0.) {
		tau = b * 2. / (sqrt(a * a + b * 4. * c__) - a);
	    } else {
		tau = (a + sqrt(a * a + b * 4. * c__)) / (c__ * 2.);
	    }

/*           It can be proved that */
/*               D(N) < D(N)+TAU < LAMBDA(N) < D(N)+RHO/2 */

	    dltlb = 0.;
	    dltub = midpt;
	}

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - tau;
/* L30: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L40: */
	}
	erretm = abs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / delta[*n];
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8. + erretm - phi + rhoinv + abs(tau) * (dpsi 
		+ dphi);

	w = rhoinv + phi + psi;

/*        Test for convergence */

	if (abs(w) <= eps * erretm) {
	    *dlam = d__[*i__] + tau;
	    goto L250;
	}

	if (w <= 0.) {
	    dltlb = max(dltlb,tau);
	} else {
	    dltub = min(dltub,tau);
	}

/*        Calculate the new step */

	++niter;
	c__ = w - delta[*n - 1] * dpsi - delta[*n] * dphi;
	a = (delta[*n - 1] + delta[*n]) * w - delta[*n - 1] * delta[*n] * (
		dpsi + dphi);
	b = delta[*n - 1] * delta[*n] * w;
	if (c__ < 0.) {
	    c__ = abs(c__);
	}
	if (c__ == 0.) {
/*          ETA = B/A */
/*           ETA = RHO - TAU */
	    eta = dltub - tau;
	} else if (a >= 0.) {
	    eta = (a + sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (c__ 
		    * 2.);
	} else {
	    eta = b * 2. / (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))
		    );
	}

/*        Note, eta should be positive if w is negative, and */
/*        eta should be negative otherwise. However, */
/*        if for some reason caused by roundoff, eta*w > 0, */
/*        we simply use one Newton step instead. This way */
/*        will guarantee eta*w < 0. */

	if (w * eta > 0.) {
	    eta = -w / (dpsi + dphi);
	}
	temp = tau + eta;
	if (temp > dltub || temp < dltlb) {
	    if (w < 0.) {
		eta = (dltub - tau) / 2.;
	    } else {
		eta = (dltlb - tau) / 2.;
	    }
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] -= eta;
/* L50: */
	}

	tau += eta;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L60: */
	}
	erretm = abs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / delta[*n];
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8. + erretm - phi + rhoinv + abs(tau) * (dpsi 
		+ dphi);

	w = rhoinv + phi + psi;

/*        Main loop to update the values of the array   DELTA */

	iter = niter + 1;

	for (niter = iter; niter <= 30; ++niter) {

/*           Test for convergence */

	    if (abs(w) <= eps * erretm) {
		*dlam = d__[*i__] + tau;
		goto L250;
	    }

	    if (w <= 0.) {
		dltlb = max(dltlb,tau);
	    } else {
		dltub = min(dltub,tau);
	    }

/*           Calculate the new step */

	    c__ = w - delta[*n - 1] * dpsi - delta[*n] * dphi;
	    a = (delta[*n - 1] + delta[*n]) * w - delta[*n - 1] * delta[*n] * 
		    (dpsi + dphi);
	    b = delta[*n - 1] * delta[*n] * w;
	    if (a >= 0.) {
		eta = (a + sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (
			c__ * 2.);
	    } else {
		eta = b * 2. / (a - sqrt((d__1 = a * a - b * 4. * c__, abs(
			d__1))));
	    }

/*           Note, eta should be positive if w is negative, and */
/*           eta should be negative otherwise. However, */
/*           if for some reason caused by roundoff, eta*w > 0, */
/*           we simply use one Newton step instead. This way */
/*           will guarantee eta*w < 0. */

	    if (w * eta > 0.) {
		eta = -w / (dpsi + dphi);
	    }
	    temp = tau + eta;
	    if (temp > dltub || temp < dltlb) {
		if (w < 0.) {
		    eta = (dltub - tau) / 2.;
		} else {
		    eta = (dltlb - tau) / 2.;
		}
	    }
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] -= eta;
/* L70: */
	    }

	    tau += eta;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.;
	    psi = 0.;
	    erretm = 0.;
	    i__1 = ii;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / delta[j];
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L80: */
	    }
	    erretm = abs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    temp = z__[*n] / delta[*n];
	    phi = z__[*n] * temp;
	    dphi = temp * temp;
	    erretm = (-phi - psi) * 8. + erretm - phi + rhoinv + abs(tau) * (
		    dpsi + dphi);

	    w = rhoinv + phi + psi;
/* L90: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;
	*dlam = d__[*i__] + tau;
	goto L250;

/*        End for the case I = N */

    } else {

/*        The case for I < N */

	niter = 1;
	ip1 = *i__ + 1;

/*        Calculate initial guess */

	del = d__[ip1] - d__[*i__];
	midpt = del / 2.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - midpt;
/* L100: */
	}

	psi = 0.;
	i__1 = *i__ - 1;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / delta[j];
/* L110: */
	}

	phi = 0.;
	i__1 = *i__ + 2;
	for (j = *n; j >= i__1; --j) {
	    phi += z__[j] * z__[j] / delta[j];
/* L120: */
	}
	c__ = rhoinv + psi + phi;
	w = c__ + z__[*i__] * z__[*i__] / delta[*i__] + z__[ip1] * z__[ip1] / 
		delta[ip1];

	if (w > 0.) {

/*           d(i)< the ith eigenvalue < (d(i)+d(i+1))/2 */

/*           We choose d(i) as origin. */

	    orgati = TRUE_;
	    a = c__ * del + z__[*i__] * z__[*i__] + z__[ip1] * z__[ip1];
	    b = z__[*i__] * z__[*i__] * del;
	    if (a > 0.) {
		tau = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__, abs(
			d__1))));
	    } else {
		tau = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (
			c__ * 2.);
	    }
	    dltlb = 0.;
	    dltub = midpt;
	} else {

/*           (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1) */

/*           We choose d(i+1) as origin. */

	    orgati = FALSE_;
	    a = c__ * del - z__[*i__] * z__[*i__] - z__[ip1] * z__[ip1];
	    b = z__[ip1] * z__[ip1] * del;
	    if (a < 0.) {
		tau = b * 2. / (a - sqrt((d__1 = a * a + b * 4. * c__, abs(
			d__1))));
	    } else {
		tau = -(a + sqrt((d__1 = a * a + b * 4. * c__, abs(d__1)))) / 
			(c__ * 2.);
	    }
	    dltlb = -midpt;
	    dltub = 0.;
	}

	if (orgati) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] = d__[j] - d__[*i__] - tau;
/* L130: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] = d__[j] - d__[ip1] - tau;
/* L140: */
	    }
	}
	if (orgati) {
	    ii = *i__;
	} else {
	    ii = *i__ + 1;
	}
	iim1 = ii - 1;
	iip1 = ii + 1;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L150: */
	}
	erretm = abs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.;
	phi = 0.;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / delta[j];
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
	    if (w < 0.) {
		swtch3 = TRUE_;
	    }
	} else {
	    if (w > 0.) {
		swtch3 = TRUE_;
	    }
	}
	if (ii == 1 || ii == *n) {
	    swtch3 = FALSE_;
	}

	temp = z__[ii] / delta[ii];
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w += temp;
	erretm = (phi - psi) * 8. + erretm + rhoinv * 2. + abs(temp) * 3. + 
		abs(tau) * dw;

/*        Test for convergence */

	if (abs(w) <= eps * erretm) {
	    if (orgati) {
		*dlam = d__[*i__] + tau;
	    } else {
		*dlam = d__[ip1] + tau;
	    }
	    goto L250;
	}

	if (w <= 0.) {
	    dltlb = max(dltlb,tau);
	} else {
	    dltub = min(dltub,tau);
	}

/*        Calculate the new step */

	++niter;
	if (! swtch3) {
	    if (orgati) {
/* Computing 2nd power */
		d__1 = z__[*i__] / delta[*i__];
		c__ = w - delta[ip1] * dw - (d__[*i__] - d__[ip1]) * (d__1 * 
			d__1);
	    } else {
/* Computing 2nd power */
		d__1 = z__[ip1] / delta[ip1];
		c__ = w - delta[*i__] * dw - (d__[ip1] - d__[*i__]) * (d__1 * 
			d__1);
	    }
	    a = (delta[*i__] + delta[ip1]) * w - delta[*i__] * delta[ip1] * 
		    dw;
	    b = delta[*i__] * delta[ip1] * w;
	    if (c__ == 0.) {
		if (a == 0.) {
		    if (orgati) {
			a = z__[*i__] * z__[*i__] + delta[ip1] * delta[ip1] * 
				(dpsi + dphi);
		    } else {
			a = z__[ip1] * z__[ip1] + delta[*i__] * delta[*i__] * 
				(dpsi + dphi);
		    }
		}
		eta = b / a;
	    } else if (a <= 0.) {
		eta = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (
			c__ * 2.);
	    } else {
		eta = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__, abs(
			d__1))));
	    }
	} else {

/*           Interpolation using THREE most relevant poles */

	    temp = rhoinv + psi + phi;
	    if (orgati) {
		temp1 = z__[iim1] / delta[iim1];
		temp1 *= temp1;
		c__ = temp - delta[iip1] * (dpsi + dphi) - (d__[iim1] - d__[
			iip1]) * temp1;
		zz[0] = z__[iim1] * z__[iim1];
		zz[2] = delta[iip1] * delta[iip1] * (dpsi - temp1 + dphi);
	    } else {
		temp1 = z__[iip1] / delta[iip1];
		temp1 *= temp1;
		c__ = temp - delta[iim1] * (dpsi + dphi) - (d__[iip1] - d__[
			iim1]) * temp1;
		zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - temp1));
		zz[2] = z__[iip1] * z__[iip1];
	    }
	    zz[1] = z__[ii] * z__[ii];
	    dlaed6_(&niter, &orgati, &c__, &delta[iim1], zz, &w, &eta, info);
	    if (*info != 0) {
		goto L250;
	    }
	}

/*        Note, eta should be positive if w is negative, and */
/*        eta should be negative otherwise. However, */
/*        if for some reason caused by roundoff, eta*w > 0, */
/*        we simply use one Newton step instead. This way */
/*        will guarantee eta*w < 0. */

	if (w * eta >= 0.) {
	    eta = -w / dw;
	}
	temp = tau + eta;
	if (temp > dltub || temp < dltlb) {
	    if (w < 0.) {
		eta = (dltub - tau) / 2.;
	    } else {
		eta = (dltlb - tau) / 2.;
	    }
	}

	prew = w;

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] -= eta;
/* L180: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L190: */
	}
	erretm = abs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.;
	phi = 0.;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / delta[j];
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
/* L200: */
	}

	temp = z__[ii] / delta[ii];
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w = rhoinv + phi + psi + temp;
	erretm = (phi - psi) * 8. + erretm + rhoinv * 2. + abs(temp) * 3. + (
		d__1 = tau + eta, abs(d__1)) * dw;

	swtch = FALSE_;
	if (orgati) {
	    if (-w > abs(prew) / 10.) {
		swtch = TRUE_;
	    }
	} else {
	    if (w > abs(prew) / 10.) {
		swtch = TRUE_;
	    }
	}

	tau += eta;

/*        Main loop to update the values of the array   DELTA */

	iter = niter + 1;

	for (niter = iter; niter <= 30; ++niter) {

/*           Test for convergence */

	    if (abs(w) <= eps * erretm) {
		if (orgati) {
		    *dlam = d__[*i__] + tau;
		} else {
		    *dlam = d__[ip1] + tau;
		}
		goto L250;
	    }

	    if (w <= 0.) {
		dltlb = max(dltlb,tau);
	    } else {
		dltub = min(dltub,tau);
	    }

/*           Calculate the new step */

	    if (! swtch3) {
		if (! swtch) {
		    if (orgati) {
/* Computing 2nd power */
			d__1 = z__[*i__] / delta[*i__];
			c__ = w - delta[ip1] * dw - (d__[*i__] - d__[ip1]) * (
				d__1 * d__1);
		    } else {
/* Computing 2nd power */
			d__1 = z__[ip1] / delta[ip1];
			c__ = w - delta[*i__] * dw - (d__[ip1] - d__[*i__]) * 
				(d__1 * d__1);
		    }
		} else {
		    temp = z__[ii] / delta[ii];
		    if (orgati) {
			dpsi += temp * temp;
		    } else {
			dphi += temp * temp;
		    }
		    c__ = w - delta[*i__] * dpsi - delta[ip1] * dphi;
		}
		a = (delta[*i__] + delta[ip1]) * w - delta[*i__] * delta[ip1] 
			* dw;
		b = delta[*i__] * delta[ip1] * w;
		if (c__ == 0.) {
		    if (a == 0.) {
			if (! swtch) {
			    if (orgati) {
				a = z__[*i__] * z__[*i__] + delta[ip1] * 
					delta[ip1] * (dpsi + dphi);
			    } else {
				a = z__[ip1] * z__[ip1] + delta[*i__] * delta[
					*i__] * (dpsi + dphi);
			    }
			} else {
			    a = delta[*i__] * delta[*i__] * dpsi + delta[ip1] 
				    * delta[ip1] * dphi;
			}
		    }
		    eta = b / a;
		} else if (a <= 0.) {
		    eta = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1))))
			     / (c__ * 2.);
		} else {
		    eta = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__, 
			    abs(d__1))));
		}
	    } else {

/*              Interpolation using THREE most relevant poles */

		temp = rhoinv + psi + phi;
		if (swtch) {
		    c__ = temp - delta[iim1] * dpsi - delta[iip1] * dphi;
		    zz[0] = delta[iim1] * delta[iim1] * dpsi;
		    zz[2] = delta[iip1] * delta[iip1] * dphi;
		} else {
		    if (orgati) {
			temp1 = z__[iim1] / delta[iim1];
			temp1 *= temp1;
			c__ = temp - delta[iip1] * (dpsi + dphi) - (d__[iim1] 
				- d__[iip1]) * temp1;
			zz[0] = z__[iim1] * z__[iim1];
			zz[2] = delta[iip1] * delta[iip1] * (dpsi - temp1 + 
				dphi);
		    } else {
			temp1 = z__[iip1] / delta[iip1];
			temp1 *= temp1;
			c__ = temp - delta[iim1] * (dpsi + dphi) - (d__[iip1] 
				- d__[iim1]) * temp1;
			zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - 
				temp1));
			zz[2] = z__[iip1] * z__[iip1];
		    }
		}
		dlaed6_(&niter, &orgati, &c__, &delta[iim1], zz, &w, &eta, 
			info);
		if (*info != 0) {
		    goto L250;
		}
	    }

/*           Note, eta should be positive if w is negative, and */
/*           eta should be negative otherwise. However, */
/*           if for some reason caused by roundoff, eta*w > 0, */
/*           we simply use one Newton step instead. This way */
/*           will guarantee eta*w < 0. */

	    if (w * eta >= 0.) {
		eta = -w / dw;
	    }
	    temp = tau + eta;
	    if (temp > dltub || temp < dltlb) {
		if (w < 0.) {
		    eta = (dltub - tau) / 2.;
		} else {
		    eta = (dltlb - tau) / 2.;
		}
	    }

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] -= eta;
/* L210: */
	    }

	    tau += eta;
	    prew = w;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.;
	    psi = 0.;
	    erretm = 0.;
	    i__1 = iim1;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / delta[j];
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L220: */
	    }
	    erretm = abs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    dphi = 0.;
	    phi = 0.;
	    i__1 = iip1;
	    for (j = *n; j >= i__1; --j) {
		temp = z__[j] / delta[j];
		phi += z__[j] * temp;
		dphi += temp * temp;
		erretm += phi;
/* L230: */
	    }

	    temp = z__[ii] / delta[ii];
	    dw = dpsi + dphi + temp * temp;
	    temp = z__[ii] * temp;
	    w = rhoinv + phi + psi + temp;
	    erretm = (phi - psi) * 8. + erretm + rhoinv * 2. + abs(temp) * 3. 
		    + abs(tau) * dw;
	    if (w * prew > 0. && abs(w) > abs(prew) / 10.) {
		swtch = ! swtch;
	    }

/* L240: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;
	if (orgati) {
	    *dlam = d__[*i__] + tau;
	} else {
	    *dlam = d__[ip1] + tau;
	}

    }

L250:

    return 0;

/*     End of DLAED4 */

} /* dlaed4_ */
