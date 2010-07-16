/* slarrv.f -- translated by f2c (version 20061008).
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

static real c_b5 = 0.f;
static integer c__1 = 1;
static integer c__2 = 2;

/* Subroutine */ int slarrv_(integer *n, real *vl, real *vu, real *d__, real *
	l, real *pivmin, integer *isplit, integer *m, integer *dol, integer *
	dou, real *minrgp, real *rtol1, real *rtol2, real *w, real *werr, 
	real *wgap, integer *iblock, integer *indexw, real *gers, real *z__, 
	integer *ldz, integer *isuppz, real *work, integer *iwork, integer *
	info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2;
    logical L__1;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer minwsize, i__, j, k, p, q, miniwsize, ii;
    real gl;
    integer im, in;
    real gu, gap, eps, tau, tol, tmp;
    integer zto;
    real ztz;
    integer iend, jblk;
    real lgap;
    integer done;
    real rgap, left;
    integer wend, iter;
    real bstw;
    integer itmp1, indld;
    real fudge;
    integer idone;
    real sigma;
    integer iinfo, iindr;
    real resid;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    logical eskip;
    real right;
    integer nclus, zfrom;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *);
    real rqtol;
    integer iindc1, iindc2;
    extern /* Subroutine */ int slar1v_(integer *, integer *, integer *, real 
	    *, real *, real *, real *, real *, real *, real *, real *, 
	    logical *, integer *, real *, real *, integer *, integer *, real *
, real *, real *, real *);
    logical stp2ii;
    real lambda;
    integer ibegin, indeig;
    logical needbs;
    integer indlld;
    real sgndef, mingma;
    extern doublereal slamch_(char *);
    integer oldien, oldncl, wbegin;
    real spdiam;
    integer negcnt, oldcls;
    real savgap;
    integer ndepth;
    real ssigma;
    logical usedbs;
    integer iindwk, offset;
    real gaptol;
    extern /* Subroutine */ int slarrb_(integer *, real *, real *, integer *, 
	    integer *, real *, real *, integer *, real *, real *, real *, 
	    real *, integer *, real *, real *, integer *, integer *), slarrf_(
	    integer *, real *, real *, real *, integer *, integer *, real *, 
	    real *, real *, real *, real *, real *, real *, real *, real *, 
	    real *, real *, integer *);
    integer newcls, oldfst, indwrk, windex, oldlst;
    logical usedrq;
    integer newfst, newftt, parity, windmn, isupmn, newlst, windpl, zusedl, 
	    newsiz, zusedu, zusedw;
    real bstres, nrminv;
    logical tryrqc;
    integer isupmx;
    real rqcorr;
    extern /* Subroutine */ int slaset_(char *, integer *, integer *, real *, 
	    real *, real *, integer *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLARRV computes the eigenvectors of the tridiagonal matrix */
/*  T = L D L^T given L, D and APPROXIMATIONS to the eigenvalues of L D L^T. */
/*  The input eigenvalues should have been computed by SLARRE. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix.  N >= 0. */

/*  VL      (input) REAL */
/*  VU      (input) REAL */
/*          Lower and upper bounds of the interval that contains the desired */
/*          eigenvalues. VL < VU. Needed to compute gaps on the left or right */
/*          end of the extremal eigenvalues in the desired RANGE. */

/*  D       (input/output) REAL             array, dimension (N) */
/*          On entry, the N diagonal elements of the diagonal matrix D. */
/*          On exit, D may be overwritten. */

/*  L       (input/output) REAL             array, dimension (N) */
/*          On entry, the (N-1) subdiagonal elements of the unit */
/*          bidiagonal matrix L are in elements 1 to N-1 of L */
/*          (if the matrix is not splitted.) At the end of each block */
/*          is stored the corresponding shift as given by SLARRE. */
/*          On exit, L is overwritten. */

/*  PIVMIN  (in) DOUBLE PRECISION */
/*          The minimum pivot allowed in the Sturm sequence. */

/*  ISPLIT  (input) INTEGER array, dimension (N) */
/*          The splitting points, at which T breaks up into blocks. */
/*          The first block consists of rows/columns 1 to */
/*          ISPLIT( 1 ), the second of rows/columns ISPLIT( 1 )+1 */
/*          through ISPLIT( 2 ), etc. */

/*  M       (input) INTEGER */
/*          The total number of input eigenvalues.  0 <= M <= N. */

/*  DOL     (input) INTEGER */
/*  DOU     (input) INTEGER */
/*          If the user wants to compute only selected eigenvectors from all */
/*          the eigenvalues supplied, he can specify an index range DOL:DOU. */
/*          Or else the setting DOL=1, DOU=M should be applied. */
/*          Note that DOL and DOU refer to the order in which the eigenvalues */
/*          are stored in W. */
/*          If the user wants to compute only selected eigenpairs, then */
/*          the columns DOL-1 to DOU+1 of the eigenvector space Z contain the */
/*          computed eigenvectors. All other columns of Z are set to zero. */

/*  MINRGP  (input) REAL */

/*  RTOL1   (input) REAL */
/*  RTOL2   (input) REAL */
/*           Parameters for bisection. */
/*           An interval [LEFT,RIGHT] has converged if */
/*           RIGHT-LEFT.LT.MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) ) */

/*  W       (input/output) REAL             array, dimension (N) */
/*          The first M elements of W contain the APPROXIMATE eigenvalues for */
/*          which eigenvectors are to be computed.  The eigenvalues */
/*          should be grouped by split-off block and ordered from */
/*          smallest to largest within the block ( The output array */
/*          W from SLARRE is expected here ). Furthermore, they are with */
/*          respect to the shift of the corresponding root representation */
/*          for their block. On exit, W holds the eigenvalues of the */
/*          UNshifted matrix. */

/*  WERR    (input/output) REAL             array, dimension (N) */
/*          The first M elements contain the semiwidth of the uncertainty */
/*          interval of the corresponding eigenvalue in W */

/*  WGAP    (input/output) REAL             array, dimension (N) */
/*          The separation from the right neighbor eigenvalue in W. */

/*  IBLOCK  (input) INTEGER array, dimension (N) */
/*          The indices of the blocks (submatrices) associated with the */
/*          corresponding eigenvalues in W; IBLOCK(i)=1 if eigenvalue */
/*          W(i) belongs to the first block from the top, =2 if W(i) */
/*          belongs to the second block, etc. */

/*  INDEXW  (input) INTEGER array, dimension (N) */
/*          The indices of the eigenvalues within each block (submatrix); */
/*          for example, INDEXW(i)= 10 and IBLOCK(i)=2 imply that the */
/*          i-th eigenvalue W(i) is the 10-th eigenvalue in the second block. */

/*  GERS    (input) REAL             array, dimension (2*N) */
/*          The N Gerschgorin intervals (the i-th Gerschgorin interval */
/*          is (GERS(2*i-1), GERS(2*i)). The Gerschgorin intervals should */
/*          be computed from the original UNshifted matrix. */

/*  Z       (output) REAL             array, dimension (LDZ, max(1,M) ) */
/*          If INFO = 0, the first M columns of Z contain the */
/*          orthonormal eigenvectors of the matrix T */
/*          corresponding to the input eigenvalues, with the i-th */
/*          column of Z holding the eigenvector associated with W(i). */
/*          Note: the user must ensure that at least max(1,M) columns are */
/*          supplied in the array Z. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDZ >= 1, and if */
/*          JOBZ = 'V', LDZ >= max(1,N). */

/*  ISUPPZ  (output) INTEGER array, dimension ( 2*max(1,M) ) */
/*          The support of the eigenvectors in Z, i.e., the indices */
/*          indicating the nonzero elements in Z. The I-th eigenvector */
/*          is nonzero only in elements ISUPPZ( 2*I-1 ) through */
/*          ISUPPZ( 2*I ). */

/*  WORK    (workspace) REAL             array, dimension (12*N) */

/*  IWORK   (workspace) INTEGER array, dimension (7*N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */

/*          > 0:  A problem occured in SLARRV. */
/*          < 0:  One of the called subroutines signaled an internal problem. */
/*                Needs inspection of the corresponding parameter IINFO */
/*                for further information. */

/*          =-1:  Problem in SLARRB when refining a child's eigenvalues. */
/*          =-2:  Problem in SLARRF when computing the RRR of a child. */
/*                When a child is inside a tight cluster, it can be difficult */
/*                to find an RRR. A partial remedy from the user's point of */
/*                view is to make the parameter MINRGP smaller and recompile. */
/*                However, as the orthogonality of the computed vectors is */
/*                proportional to 1/MINRGP, the user should be aware that */
/*                he might be trading in precision when he decreases MINRGP. */
/*          =-3:  Problem in SLARRB when refining a single eigenvalue */
/*                after the Rayleigh correction was rejected. */
/*          = 5:  The Rayleigh Quotient Iteration failed to converge to */
/*                full accuracy in MAXITR steps. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Beresford Parlett, University of California, Berkeley, USA */
/*     Jim Demmel, University of California, Berkeley, USA */
/*     Inderjit Dhillon, University of Texas, Austin, USA */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Christof Voemel, University of California, Berkeley, USA */

/*  ===================================================================== */

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
/*     .. Executable Statements .. */
/*     .. */
/*     The first N entries of WORK are reserved for the eigenvalues */
    /* Parameter adjustments */
    --d__;
    --l;
    --isplit;
    --w;
    --werr;
    --wgap;
    --iblock;
    --indexw;
    --gers;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --isuppz;
    --work;
    --iwork;

    /* Function Body */
    indld = *n + 1;
    indlld = (*n << 1) + 1;
    indwrk = *n * 3 + 1;
    minwsize = *n * 12;
    i__1 = minwsize;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[i__] = 0.f;
/* L5: */
    }
/*     IWORK(IINDR+1:IINDR+N) hold the twist indices R for the */
/*     factorization used to compute the FP vector */
    iindr = 0;
/*     IWORK(IINDC1+1:IINC2+N) are used to store the clusters of the current */
/*     layer and the one above. */
    iindc1 = *n;
    iindc2 = *n << 1;
    iindwk = *n * 3 + 1;
    miniwsize = *n * 7;
    i__1 = miniwsize;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iwork[i__] = 0;
/* L10: */
    }
    zusedl = 1;
    if (*dol > 1) {
/*        Set lower bound for use of Z */
	zusedl = *dol - 1;
    }
    zusedu = *m;
    if (*dou < *m) {
/*        Set lower bound for use of Z */
	zusedu = *dou + 1;
    }
/*     The width of the part of Z that is used */
    zusedw = zusedu - zusedl + 1;
    slaset_("Full", n, &zusedw, &c_b5, &c_b5, &z__[zusedl * z_dim1 + 1], ldz);
    eps = slamch_("Precision");
    rqtol = eps * 2.f;

/*     Set expert flags for standard code. */
    tryrqc = TRUE_;
    if (*dol == 1 && *dou == *m) {
    } else {
/*        Only selected eigenpairs are computed. Since the other evalues */
/*        are not refined by RQ iteration, bisection has to compute to full */
/*        accuracy. */
	*rtol1 = eps * 4.f;
	*rtol2 = eps * 4.f;
    }
/*     The entries WBEGIN:WEND in W, WERR, WGAP correspond to the */
/*     desired eigenvalues. The support of the nonzero eigenvector */
/*     entries is contained in the interval IBEGIN:IEND. */
/*     Remark that if k eigenpairs are desired, then the eigenvectors */
/*     are stored in k contiguous columns of Z. */
/*     DONE is the number of eigenvectors already computed */
    done = 0;
    ibegin = 1;
    wbegin = 1;
    i__1 = iblock[*m];
    for (jblk = 1; jblk <= i__1; ++jblk) {
	iend = isplit[jblk];
	sigma = l[iend];
/*        Find the eigenvectors of the submatrix indexed IBEGIN */
/*        through IEND. */
	wend = wbegin - 1;
L15:
	if (wend < *m) {
	    if (iblock[wend + 1] == jblk) {
		++wend;
		goto L15;
	    }
	}
	if (wend < wbegin) {
	    ibegin = iend + 1;
	    goto L170;
	} else if (wend < *dol || wbegin > *dou) {
	    ibegin = iend + 1;
	    wbegin = wend + 1;
	    goto L170;
	}
/*        Find local spectral diameter of the block */
	gl = gers[(ibegin << 1) - 1];
	gu = gers[ibegin * 2];
	i__2 = iend;
	for (i__ = ibegin + 1; i__ <= i__2; ++i__) {
/* Computing MIN */
	    r__1 = gers[(i__ << 1) - 1];
	    gl = dmin(r__1,gl);
/* Computing MAX */
	    r__1 = gers[i__ * 2];
	    gu = dmax(r__1,gu);
/* L20: */
	}
	spdiam = gu - gl;
/*        OLDIEN is the last index of the previous block */
	oldien = ibegin - 1;
/*        Calculate the size of the current block */
	in = iend - ibegin + 1;
/*        The number of eigenvalues in the current block */
	im = wend - wbegin + 1;
/*        This is for a 1x1 block */
	if (ibegin == iend) {
	    ++done;
	    z__[ibegin + wbegin * z_dim1] = 1.f;
	    isuppz[(wbegin << 1) - 1] = ibegin;
	    isuppz[wbegin * 2] = ibegin;
	    w[wbegin] += sigma;
	    work[wbegin] = w[wbegin];
	    ibegin = iend + 1;
	    ++wbegin;
	    goto L170;
	}
/*        The desired (shifted) eigenvalues are stored in W(WBEGIN:WEND) */
/*        Note that these can be approximations, in this case, the corresp. */
/*        entries of WERR give the size of the uncertainty interval. */
/*        The eigenvalue approximations will be refined when necessary as */
/*        high relative accuracy is required for the computation of the */
/*        corresponding eigenvectors. */
	scopy_(&im, &w[wbegin], &c__1, &work[wbegin], &c__1);
/*        We store in W the eigenvalue approximations w.r.t. the original */
/*        matrix T. */
	i__2 = im;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w[wbegin + i__ - 1] += sigma;
/* L30: */
	}
/*        NDEPTH is the current depth of the representation tree */
	ndepth = 0;
/*        PARITY is either 1 or 0 */
	parity = 1;
/*        NCLUS is the number of clusters for the next level of the */
/*        representation tree, we start with NCLUS = 1 for the root */
	nclus = 1;
	iwork[iindc1 + 1] = 1;
	iwork[iindc1 + 2] = im;
/*        IDONE is the number of eigenvectors already computed in the current */
/*        block */
	idone = 0;
/*        loop while( IDONE.LT.IM ) */
/*        generate the representation tree for the current block and */
/*        compute the eigenvectors */
L40:
	if (idone < im) {
/*           This is a crude protection against infinitely deep trees */
	    if (ndepth > *m) {
		*info = -2;
		return 0;
	    }
/*           breadth first processing of the current level of the representation */
/*           tree: OLDNCL = number of clusters on current level */
	    oldncl = nclus;
/*           reset NCLUS to count the number of child clusters */
	    nclus = 0;

	    parity = 1 - parity;
	    if (parity == 0) {
		oldcls = iindc1;
		newcls = iindc2;
	    } else {
		oldcls = iindc2;
		newcls = iindc1;
	    }
/*           Process the clusters on the current level */
	    i__2 = oldncl;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		j = oldcls + (i__ << 1);
/*              OLDFST, OLDLST = first, last index of current cluster. */
/*                               cluster indices start with 1 and are relative */
/*                               to WBEGIN when accessing W, WGAP, WERR, Z */
		oldfst = iwork[j - 1];
		oldlst = iwork[j];
		if (ndepth > 0) {
/*                 Retrieve relatively robust representation (RRR) of cluster */
/*                 that has been computed at the previous level */
/*                 The RRR is stored in Z and overwritten once the eigenvectors */
/*                 have been computed or when the cluster is refined */
		    if (*dol == 1 && *dou == *m) {
/*                    Get representation from location of the leftmost evalue */
/*                    of the cluster */
			j = wbegin + oldfst - 1;
		    } else {
			if (wbegin + oldfst - 1 < *dol) {
/*                       Get representation from the left end of Z array */
			    j = *dol - 1;
			} else if (wbegin + oldfst - 1 > *dou) {
/*                       Get representation from the right end of Z array */
			    j = *dou;
			} else {
			    j = wbegin + oldfst - 1;
			}
		    }
		    scopy_(&in, &z__[ibegin + j * z_dim1], &c__1, &d__[ibegin]
, &c__1);
		    i__3 = in - 1;
		    scopy_(&i__3, &z__[ibegin + (j + 1) * z_dim1], &c__1, &l[
			    ibegin], &c__1);
		    sigma = z__[iend + (j + 1) * z_dim1];
/*                 Set the corresponding entries in Z to zero */
		    slaset_("Full", &in, &c__2, &c_b5, &c_b5, &z__[ibegin + j 
			    * z_dim1], ldz);
		}
/*              Compute DL and DLL of current RRR */
		i__3 = iend - 1;
		for (j = ibegin; j <= i__3; ++j) {
		    tmp = d__[j] * l[j];
		    work[indld - 1 + j] = tmp;
		    work[indlld - 1 + j] = tmp * l[j];
/* L50: */
		}
		if (ndepth > 0) {
/*                 P and Q are index of the first and last eigenvalue to compute */
/*                 within the current block */
		    p = indexw[wbegin - 1 + oldfst];
		    q = indexw[wbegin - 1 + oldlst];
/*                 Offset for the arrays WORK, WGAP and WERR, i.e., th P-OFFSET */
/*                 thru' Q-OFFSET elements of these arrays are to be used. */
/*                  OFFSET = P-OLDFST */
		    offset = indexw[wbegin] - 1;
/*                 perform limited bisection (if necessary) to get approximate */
/*                 eigenvalues to the precision needed. */
		    slarrb_(&in, &d__[ibegin], &work[indlld + ibegin - 1], &p, 
			     &q, rtol1, rtol2, &offset, &work[wbegin], &wgap[
			    wbegin], &werr[wbegin], &work[indwrk], &iwork[
			    iindwk], pivmin, &spdiam, &in, &iinfo);
		    if (iinfo != 0) {
			*info = -1;
			return 0;
		    }
/*                 We also recompute the extremal gaps. W holds all eigenvalues */
/*                 of the unshifted matrix and must be used for computation */
/*                 of WGAP, the entries of WORK might stem from RRRs with */
/*                 different shifts. The gaps from WBEGIN-1+OLDFST to */
/*                 WBEGIN-1+OLDLST are correctly computed in SLARRB. */
/*                 However, we only allow the gaps to become greater since */
/*                 this is what should happen when we decrease WERR */
		    if (oldfst > 1) {
/* Computing MAX */
			r__1 = wgap[wbegin + oldfst - 2], r__2 = w[wbegin + 
				oldfst - 1] - werr[wbegin + oldfst - 1] - w[
				wbegin + oldfst - 2] - werr[wbegin + oldfst - 
				2];
			wgap[wbegin + oldfst - 2] = dmax(r__1,r__2);
		    }
		    if (wbegin + oldlst - 1 < wend) {
/* Computing MAX */
			r__1 = wgap[wbegin + oldlst - 1], r__2 = w[wbegin + 
				oldlst] - werr[wbegin + oldlst] - w[wbegin + 
				oldlst - 1] - werr[wbegin + oldlst - 1];
			wgap[wbegin + oldlst - 1] = dmax(r__1,r__2);
		    }
/*                 Each time the eigenvalues in WORK get refined, we store */
/*                 the newly found approximation with all shifts applied in W */
		    i__3 = oldlst;
		    for (j = oldfst; j <= i__3; ++j) {
			w[wbegin + j - 1] = work[wbegin + j - 1] + sigma;
/* L53: */
		    }
		}
/*              Process the current node. */
		newfst = oldfst;
		i__3 = oldlst;
		for (j = oldfst; j <= i__3; ++j) {
		    if (j == oldlst) {
/*                    we are at the right end of the cluster, this is also the */
/*                    boundary of the child cluster */
			newlst = j;
		    } else if (wgap[wbegin + j - 1] >= *minrgp * (r__1 = work[
			    wbegin + j - 1], dabs(r__1))) {
/*                    the right relative gap is big enough, the child cluster */
/*                    (NEWFST,..,NEWLST) is well separated from the following */
			newlst = j;
		    } else {
/*                    inside a child cluster, the relative gap is not */
/*                    big enough. */
			goto L140;
		    }
/*                 Compute size of child cluster found */
		    newsiz = newlst - newfst + 1;
/*                 NEWFTT is the place in Z where the new RRR or the computed */
/*                 eigenvector is to be stored */
		    if (*dol == 1 && *dou == *m) {
/*                    Store representation at location of the leftmost evalue */
/*                    of the cluster */
			newftt = wbegin + newfst - 1;
		    } else {
			if (wbegin + newfst - 1 < *dol) {
/*                       Store representation at the left end of Z array */
			    newftt = *dol - 1;
			} else if (wbegin + newfst - 1 > *dou) {
/*                       Store representation at the right end of Z array */
			    newftt = *dou;
			} else {
			    newftt = wbegin + newfst - 1;
			}
		    }
		    if (newsiz > 1) {

/*                    Current child is not a singleton but a cluster. */
/*                    Compute and store new representation of child. */


/*                    Compute left and right cluster gap. */

/*                    LGAP and RGAP are not computed from WORK because */
/*                    the eigenvalue approximations may stem from RRRs */
/*                    different shifts. However, W hold all eigenvalues */
/*                    of the unshifted matrix. Still, the entries in WGAP */
/*                    have to be computed from WORK since the entries */
/*                    in W might be of the same order so that gaps are not */
/*                    exhibited correctly for very close eigenvalues. */
			if (newfst == 1) {
/* Computing MAX */
			    r__1 = 0.f, r__2 = w[wbegin] - werr[wbegin] - *vl;
			    lgap = dmax(r__1,r__2);
			} else {
			    lgap = wgap[wbegin + newfst - 2];
			}
			rgap = wgap[wbegin + newlst - 1];

/*                    Compute left- and rightmost eigenvalue of child */
/*                    to high precision in order to shift as close */
/*                    as possible and obtain as large relative gaps */
/*                    as possible */

			for (k = 1; k <= 2; ++k) {
			    if (k == 1) {
				p = indexw[wbegin - 1 + newfst];
			    } else {
				p = indexw[wbegin - 1 + newlst];
			    }
			    offset = indexw[wbegin] - 1;
			    slarrb_(&in, &d__[ibegin], &work[indlld + ibegin 
				    - 1], &p, &p, &rqtol, &rqtol, &offset, &
				    work[wbegin], &wgap[wbegin], &werr[wbegin]
, &work[indwrk], &iwork[iindwk], pivmin, &
				    spdiam, &in, &iinfo);
/* L55: */
			}

			if (wbegin + newlst - 1 < *dol || wbegin + newfst - 1 
				> *dou) {
/*                       if the cluster contains no desired eigenvalues */
/*                       skip the computation of that branch of the rep. tree */

/*                       We could skip before the refinement of the extremal */
/*                       eigenvalues of the child, but then the representation */
/*                       tree could be different from the one when nothing is */
/*                       skipped. For this reason we skip at this place. */
			    idone = idone + newlst - newfst + 1;
			    goto L139;
			}

/*                    Compute RRR of child cluster. */
/*                    Note that the new RRR is stored in Z */

/*                    SLARRF needs LWORK = 2*N */
			slarrf_(&in, &d__[ibegin], &l[ibegin], &work[indld + 
				ibegin - 1], &newfst, &newlst, &work[wbegin], 
				&wgap[wbegin], &werr[wbegin], &spdiam, &lgap, 
				&rgap, pivmin, &tau, &z__[ibegin + newftt * 
				z_dim1], &z__[ibegin + (newftt + 1) * z_dim1], 
				 &work[indwrk], &iinfo);
			if (iinfo == 0) {
/*                       a new RRR for the cluster was found by SLARRF */
/*                       update shift and store it */
			    ssigma = sigma + tau;
			    z__[iend + (newftt + 1) * z_dim1] = ssigma;
/*                       WORK() are the midpoints and WERR() the semi-width */
/*                       Note that the entries in W are unchanged. */
			    i__4 = newlst;
			    for (k = newfst; k <= i__4; ++k) {
				fudge = eps * 3.f * (r__1 = work[wbegin + k - 
					1], dabs(r__1));
				work[wbegin + k - 1] -= tau;
				fudge += eps * 4.f * (r__1 = work[wbegin + k 
					- 1], dabs(r__1));
/*                          Fudge errors */
				werr[wbegin + k - 1] += fudge;
/*                          Gaps are not fudged. Provided that WERR is small */
/*                          when eigenvalues are close, a zero gap indicates */
/*                          that a new representation is needed for resolving */
/*                          the cluster. A fudge could lead to a wrong decision */
/*                          of judging eigenvalues 'separated' which in */
/*                          reality are not. This could have a negative impact */
/*                          on the orthogonality of the computed eigenvectors. */
/* L116: */
			    }
			    ++nclus;
			    k = newcls + (nclus << 1);
			    iwork[k - 1] = newfst;
			    iwork[k] = newlst;
			} else {
			    *info = -2;
			    return 0;
			}
		    } else {

/*                    Compute eigenvector of singleton */

			iter = 0;

			tol = log((real) in) * 4.f * eps;

			k = newfst;
			windex = wbegin + k - 1;
/* Computing MAX */
			i__4 = windex - 1;
			windmn = max(i__4,1);
/* Computing MIN */
			i__4 = windex + 1;
			windpl = min(i__4,*m);
			lambda = work[windex];
			++done;
/*                    Check if eigenvector computation is to be skipped */
			if (windex < *dol || windex > *dou) {
			    eskip = TRUE_;
			    goto L125;
			} else {
			    eskip = FALSE_;
			}
			left = work[windex] - werr[windex];
			right = work[windex] + werr[windex];
			indeig = indexw[windex];
/*                    Note that since we compute the eigenpairs for a child, */
/*                    all eigenvalue approximations are w.r.t the same shift. */
/*                    In this case, the entries in WORK should be used for */
/*                    computing the gaps since they exhibit even very small */
/*                    differences in the eigenvalues, as opposed to the */
/*                    entries in W which might "look" the same. */
			if (k == 1) {
/*                       In the case RANGE='I' and with not much initial */
/*                       accuracy in LAMBDA and VL, the formula */
/*                       LGAP = MAX( ZERO, (SIGMA - VL) + LAMBDA ) */
/*                       can lead to an overestimation of the left gap and */
/*                       thus to inadequately early RQI 'convergence'. */
/*                       Prevent this by forcing a small left gap. */
/* Computing MAX */
			    r__1 = dabs(left), r__2 = dabs(right);
			    lgap = eps * dmax(r__1,r__2);
			} else {
			    lgap = wgap[windmn];
			}
			if (k == im) {
/*                       In the case RANGE='I' and with not much initial */
/*                       accuracy in LAMBDA and VU, the formula */
/*                       can lead to an overestimation of the right gap and */
/*                       thus to inadequately early RQI 'convergence'. */
/*                       Prevent this by forcing a small right gap. */
/* Computing MAX */
			    r__1 = dabs(left), r__2 = dabs(right);
			    rgap = eps * dmax(r__1,r__2);
			} else {
			    rgap = wgap[windex];
			}
			gap = dmin(lgap,rgap);
			if (k == 1 || k == im) {
/*                       The eigenvector support can become wrong */
/*                       because significant entries could be cut off due to a */
/*                       large GAPTOL parameter in LAR1V. Prevent this. */
			    gaptol = 0.f;
			} else {
			    gaptol = gap * eps;
			}
			isupmn = in;
			isupmx = 1;
/*                    Update WGAP so that it holds the minimum gap */
/*                    to the left or the right. This is crucial in the */
/*                    case where bisection is used to ensure that the */
/*                    eigenvalue is refined up to the required precision. */
/*                    The correct value is restored afterwards. */
			savgap = wgap[windex];
			wgap[windex] = gap;
/*                    We want to use the Rayleigh Quotient Correction */
/*                    as often as possible since it converges quadratically */
/*                    when we are close enough to the desired eigenvalue. */
/*                    However, the Rayleigh Quotient can have the wrong sign */
/*                    and lead us away from the desired eigenvalue. In this */
/*                    case, the best we can do is to use bisection. */
			usedbs = FALSE_;
			usedrq = FALSE_;
/*                    Bisection is initially turned off unless it is forced */
			needbs = ! tryrqc;
L120:
/*                    Check if bisection should be used to refine eigenvalue */
			if (needbs) {
/*                       Take the bisection as new iterate */
			    usedbs = TRUE_;
			    itmp1 = iwork[iindr + windex];
			    offset = indexw[wbegin] - 1;
			    r__1 = eps * 2.f;
			    slarrb_(&in, &d__[ibegin], &work[indlld + ibegin 
				    - 1], &indeig, &indeig, &c_b5, &r__1, &
				    offset, &work[wbegin], &wgap[wbegin], &
				    werr[wbegin], &work[indwrk], &iwork[
				    iindwk], pivmin, &spdiam, &itmp1, &iinfo);
			    if (iinfo != 0) {
				*info = -3;
				return 0;
			    }
			    lambda = work[windex];
/*                       Reset twist index from inaccurate LAMBDA to */
/*                       force computation of true MINGMA */
			    iwork[iindr + windex] = 0;
			}
/*                    Given LAMBDA, compute the eigenvector. */
			L__1 = ! usedbs;
			slar1v_(&in, &c__1, &in, &lambda, &d__[ibegin], &l[
				ibegin], &work[indld + ibegin - 1], &work[
				indlld + ibegin - 1], pivmin, &gaptol, &z__[
				ibegin + windex * z_dim1], &L__1, &negcnt, &
				ztz, &mingma, &iwork[iindr + windex], &isuppz[
				(windex << 1) - 1], &nrminv, &resid, &rqcorr, 
				&work[indwrk]);
			if (iter == 0) {
			    bstres = resid;
			    bstw = lambda;
			} else if (resid < bstres) {
			    bstres = resid;
			    bstw = lambda;
			}
/* Computing MIN */
			i__4 = isupmn, i__5 = isuppz[(windex << 1) - 1];
			isupmn = min(i__4,i__5);
/* Computing MAX */
			i__4 = isupmx, i__5 = isuppz[windex * 2];
			isupmx = max(i__4,i__5);
			++iter;
/*                    sin alpha <= |resid|/gap */
/*                    Note that both the residual and the gap are */
/*                    proportional to the matrix, so ||T|| doesn't play */
/*                    a role in the quotient */

/*                    Convergence test for Rayleigh-Quotient iteration */
/*                    (omitted when Bisection has been used) */

			if (resid > tol * gap && dabs(rqcorr) > rqtol * dabs(
				lambda) && ! usedbs) {
/*                       We need to check that the RQCORR update doesn't */
/*                       move the eigenvalue away from the desired one and */
/*                       towards a neighbor. -> protection with bisection */
			    if (indeig <= negcnt) {
/*                          The wanted eigenvalue lies to the left */
				sgndef = -1.f;
			    } else {
/*                          The wanted eigenvalue lies to the right */
				sgndef = 1.f;
			    }
/*                       We only use the RQCORR if it improves the */
/*                       the iterate reasonably. */
			    if (rqcorr * sgndef >= 0.f && lambda + rqcorr <= 
				    right && lambda + rqcorr >= left) {
				usedrq = TRUE_;
/*                          Store new midpoint of bisection interval in WORK */
				if (sgndef == 1.f) {
/*                             The current LAMBDA is on the left of the true */
/*                             eigenvalue */
				    left = lambda;
/*                             We prefer to assume that the error estimate */
/*                             is correct. We could make the interval not */
/*                             as a bracket but to be modified if the RQCORR */
/*                             chooses to. In this case, the RIGHT side should */
/*                             be modified as follows: */
/*                              RIGHT = MAX(RIGHT, LAMBDA + RQCORR) */
				} else {
/*                             The current LAMBDA is on the right of the true */
/*                             eigenvalue */
				    right = lambda;
/*                             See comment about assuming the error estimate is */
/*                             correct above. */
/*                              LEFT = MIN(LEFT, LAMBDA + RQCORR) */
				}
				work[windex] = (right + left) * .5f;
/*                          Take RQCORR since it has the correct sign and */
/*                          improves the iterate reasonably */
				lambda += rqcorr;
/*                          Update width of error interval */
				werr[windex] = (right - left) * .5f;
			    } else {
				needbs = TRUE_;
			    }
			    if (right - left < rqtol * dabs(lambda)) {
/*                             The eigenvalue is computed to bisection accuracy */
/*                             compute eigenvector and stop */
				usedbs = TRUE_;
				goto L120;
			    } else if (iter < 10) {
				goto L120;
			    } else if (iter == 10) {
				needbs = TRUE_;
				goto L120;
			    } else {
				*info = 5;
				return 0;
			    }
			} else {
			    stp2ii = FALSE_;
			    if (usedrq && usedbs && bstres <= resid) {
				lambda = bstw;
				stp2ii = TRUE_;
			    }
			    if (stp2ii) {
/*                          improve error angle by second step */
				L__1 = ! usedbs;
				slar1v_(&in, &c__1, &in, &lambda, &d__[ibegin]
, &l[ibegin], &work[indld + ibegin - 
					1], &work[indlld + ibegin - 1], 
					pivmin, &gaptol, &z__[ibegin + windex 
					* z_dim1], &L__1, &negcnt, &ztz, &
					mingma, &iwork[iindr + windex], &
					isuppz[(windex << 1) - 1], &nrminv, &
					resid, &rqcorr, &work[indwrk]);
			    }
			    work[windex] = lambda;
			}

/*                    Compute FP-vector support w.r.t. whole matrix */

			isuppz[(windex << 1) - 1] += oldien;
			isuppz[windex * 2] += oldien;
			zfrom = isuppz[(windex << 1) - 1];
			zto = isuppz[windex * 2];
			isupmn += oldien;
			isupmx += oldien;
/*                    Ensure vector is ok if support in the RQI has changed */
			if (isupmn < zfrom) {
			    i__4 = zfrom - 1;
			    for (ii = isupmn; ii <= i__4; ++ii) {
				z__[ii + windex * z_dim1] = 0.f;
/* L122: */
			    }
			}
			if (isupmx > zto) {
			    i__4 = isupmx;
			    for (ii = zto + 1; ii <= i__4; ++ii) {
				z__[ii + windex * z_dim1] = 0.f;
/* L123: */
			    }
			}
			i__4 = zto - zfrom + 1;
			sscal_(&i__4, &nrminv, &z__[zfrom + windex * z_dim1], 
				&c__1);
L125:
/*                    Update W */
			w[windex] = lambda + sigma;
/*                    Recompute the gaps on the left and right */
/*                    But only allow them to become larger and not */
/*                    smaller (which can only happen through "bad" */
/*                    cancellation and doesn't reflect the theory */
/*                    where the initial gaps are underestimated due */
/*                    to WERR being too crude.) */
			if (! eskip) {
			    if (k > 1) {
/* Computing MAX */
				r__1 = wgap[windmn], r__2 = w[windex] - werr[
					windex] - w[windmn] - werr[windmn];
				wgap[windmn] = dmax(r__1,r__2);
			    }
			    if (windex < wend) {
/* Computing MAX */
				r__1 = savgap, r__2 = w[windpl] - werr[windpl]
					 - w[windex] - werr[windex];
				wgap[windex] = dmax(r__1,r__2);
			    }
			}
			++idone;
		    }
/*                 here ends the code for the current child */

L139:
/*                 Proceed to any remaining child nodes */
		    newfst = j + 1;
L140:
		    ;
		}
/* L150: */
	    }
	    ++ndepth;
	    goto L40;
	}
	ibegin = iend + 1;
	wbegin = wend + 1;
L170:
	;
    }

    return 0;

/*     End of SLARRV */

} /* slarrv_ */
