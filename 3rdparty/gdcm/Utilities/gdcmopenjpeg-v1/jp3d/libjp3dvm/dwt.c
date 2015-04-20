/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyrigth (c) 2006, Mónica Díez, LPI-UVA, Spain
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  NOTE:
 *  This is a modified version of the openjpeg dwt.c file.
 *  Average speed improvement compared to the original file (measured on
 *  my own machine, a P4 running at 3.0 GHz):
 *  5x3 wavelets about 2 times faster
 *  9x7 wavelets about 3 times faster
 *  for both, encoding and decoding.
 *
 *  The better performance is caused by doing the 1-dimensional DWT
 *  within a temporary buffer where the data can be accessed sequential
 *  for both directions, horizontal and vertical. The 2d vertical DWT was
 *  the major bottleneck in the former version.
 *
 *  I have also removed the "Add Patrick" part because it is not longer
 *  needed.  
 *
 *  6/6/2005
 *  -Ive (aka Reiner Wahler)
 *  mail: ive@lilysoft.com
 */

#include "opj_includes.h"

/** @defgroup DWT DWT - Implementation of a discrete wavelet transform */
/*@{*/

/** @name Local static functions */
/*@{*/
unsigned int ops;
/**
Forward lazy transform (horizontal)
*/
static void dwt_deinterleave_h(int *a, int *b, int dn, int sn, int cas);
/**
Forward lazy transform (vertical)
*/
static void dwt_deinterleave_v(int *a, int *b, int dn, int sn, int x, int cas);
/**
Forward lazy transform (axial)
*/
static void dwt_deinterleave_z(int *a, int *b, int dn, int sn, int xy, int cas);
/**
Inverse lazy transform (horizontal)
*/
static void dwt_interleave_h(int *a, int *b, int dn, int sn, int cas);
/**
Inverse lazy transform (vertical)
*/
static void dwt_interleave_v(int *a, int *b, int dn, int sn, int x, int cas);
/**
Inverse lazy transform (axial)
*/
static void dwt_interleave_z(int *a, int *b, int dn, int sn, int xy, int cas);
/**
Forward 5-3 wavelet tranform in 1-D
*/
static void dwt_encode_53(int *a, int dn, int sn, int cas);
static void dwt_encode_97(int *a, int dn, int sn, int cas);
/**
Inverse 5-3 wavelet tranform in 1-D
*/
static void dwt_decode_53(int *a, int dn, int sn, int cas);
static void dwt_decode_97(int *a, int dn, int sn, int cas);
/**
Computing of wavelet transform L2 norms for arbitrary transforms
*/
static double dwt_calc_wtnorms(int orient, int level[3], int dwtid[3], opj_wtfilt_t *wtfiltx, opj_wtfilt_t *wtfilty, opj_wtfilt_t *wtfiltz);
/**
Encoding of quantification stepsize
*/
static void dwt_encode_stepsize(int stepsize, int numbps, opj_stepsize_t *bandno_stepsize);
/*@}*/

/*@}*/

#define S(i) a[(i)*2]
#define D(i) a[(1+(i)*2)]
#define S_(i) ((i)<0?S(0):((i)>=sn?S(sn-1):S(i)))
#define D_(i) ((i)<0?D(0):((i)>=dn?D(dn-1):D(i)))
/* new */
#define SS_(i) ((i)<0?S(0):((i)>=dn?S(dn-1):S(i)))
#define DD_(i) ((i)<0?D(0):((i)>=sn?D(sn-1):D(i)))

/* <summary>                                                              */
/* This table contains the norms of the 5-3 wavelets for different bands. */
/* </summary>                                                             */
static double dwt_norm[10][10][10][8];
static int flagnorm[10][10][10][8];

/*static const double dwt_norms[5][8][10] = {
	{//ResZ=1
		{1.000, 1.500, 2.750, 5.375, 10.68, 21.34, 42.67, 85.33, 170.7, 341.3},
		{1.038, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{1.038, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{.7186, .9218, 1.586, 3.043, 6.019, 12.01, 24.00, 47.97, 95.93}
	},{//ResZ=2
		{1.000, 1.8371, 2.750, 5.375, 10.68, 21.34, 42.67, 85.33, 170.7, 341.3},
		{1.2717, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{1.2717, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{.8803, .9218, 1.586, 3.043, 6.019, 12.01, 24.00, 47.97, 95.93},
		{1.2717},
		{.8803},
		{.8803},
		{.6093},
	},{ //ResZ=3
		{1.000, 1.8371, 4.5604, 5.375, 10.68, 21.34, 42.67, 85.33, 170.7, 341.3},
		{1.2717, 2.6403, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{1.2717, 2.6403, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{.8803, 1.5286, 1.586, 3.043, 6.019, 12.01, 24.00, 47.97, 95.93},
		{1.2717, 2.6403},
		{.8803, 1.5286},
		{.8803, 1.5286},
		{.6093, 0.8850},
	},{ //ResZ=4
		{1.000, 1.8371, 4.5604, 12.4614, 10.68, 21.34, 42.67, 85.33, 170.7, 341.3},
		{1.2717, 2.6403, 6.7691 , 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{1.2717, 2.6403, 6.7691 , 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
		{.8803, 1.5286, 3.6770 , 3.043, 6.019, 12.01, 24.00, 47.97, 95.93},
		{1.2717, 2.6403, 6.7691 },
		{.8803, 1.5286, 3.6770 },
		{.8803, 1.5286, 3.6770 },
		{.6093, 0.8850, 1.9974 },
	},{ //ResZ=5
		{1.000, 1.8371, 4.5604, 12.4614, 34.9025, 21.34, 42.67, 85.33, 170.7, 341.3},
		{1.2717, 2.6403, 6.7691 , 18.6304 , 11.33, 22.64, 45.25, 90.48, 180.9},
		{1.2717, 2.6403, 6.7691 , 18.6304, 11.33, 22.64, 45.25, 90.48, 180.9},
		{.8803, 1.5286, 3.6770 , 9.9446, 6.019, 12.01, 24.00, 47.97, 95.93},
		{1.2717, 2.6403, 6.7691, 18.6304},
		{.8803, 1.5286, 3.6770, 9.9446 },
		{.8803, 1.5286, 3.6770, 9.9446 },
		{.6093, 0.8850, 1.9974, 5.3083 },
	}
};*/

/* <summary>                                                              */
/* This table contains the norms of the 9-7 wavelets for different bands. */
/* </summary>                                                             */
/*static const double dwt_norms_real[5][8][10] = {
	{//ResZ==1
		{1.000, 1.9659, 4.1224, 8.4167, 16.9356, 33.9249, 67.8772, 135.7680, 271.5430, 543.0894},
		{1.0113, 1.9968, 4.1834, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{1.0113, 1.9968, 4.1834, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{0.5202, 0.9672, 2.0793, 4.3005, 8.6867, 17.4188, 34.8608, 69.7332, 139.4722}
	}, { //ResZ==2
		{1.000, 2.7564, 4.1224, 8.4167, 16.9356, 33.9249, 67.8772, 135.7680, 271.5430, 543.0894},
		{1.4179, 1.9968, 4.1834, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{1.4179, 1.9968, 4.1834, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{0.7294, 0.9672, 2.0793, 4.3005, 8.6867, 17.4188, 34.8608, 69.7332, 139.4722},
		{1.4179},
		{0.7294},
		{0.7294},
		{0.3752} //HHH
	},{ //ResZ==3
		{1.000, 2.7564, 8.3700, 8.4167, 16.9356, 33.9249, 67.8772, 135.7680, 271.5430, 543.0894},
		{1.4179, 4.0543, 4.1834, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{1.4179, 4.0543, 4.1834, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{0.7294, 1.9638, 2.0793, 4.3005, 8.6867, 17.4188, 34.8608, 69.7332, 139.4722},
		{1.4179, 4.0543},
		{0.7294, 1.9638},
		{0.7294, 1.9638},
		{0.3752, 0.9512} //HHH
	},{ //ResZ==4
		{1.000, 2.7564, 8.3700, 24.4183, 16.9356, 33.9249, 67.8772, 135.7680, 271.5430, 543.0894},
		{1.4179, 4.0543, 12.1366, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{1.4179, 4.0543, 12.1366, 8.5341, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{0.7294, 1.9638, 6.0323, 4.3005, 8.6867, 17.4188, 34.8608, 69.7332, 139.4722},
		{1.4179, 4.0543, 12.1366},
		{0.7294, 1.9638, 6.0323},
		{0.7294, 1.9638, 6.0323},
		{0.3752, 0.9512, 2.9982} //HHH
	},{ //ResZ==5
		{1.000, 2.7564, 8.3700, 24.4183, 69.6947, 33.9249, 67.8772, 135.7680, 271.5430, 543.0894},
		{1.4179, 4.0543, 12.1366, 35.1203, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{1.4179, 4.0543, 12.1366, 35.1203, 17.1667, 34.3852, 68.7967, 137.6065, 275.2196},
		{0.7294, 1.9638, 6.0323, 17.6977, 8.6867, 17.4188, 34.8608, 69.7332, 139.4722},
		{1.4179, 4.0543, 12.1366, 35.1203},
		{0.7294, 1.9638, 6.0323, 17.6977},
		{0.7294, 1.9638, 6.0323, 17.6977},
		{0.3752, 0.9512, 2.9982, 8.9182} //HHH
	}
};*/

static opj_atk_t atk_info_wt[] = {
	{0, 1, J3D_ATK_WS, J3D_ATK_IRR, 0, J3D_ATK_WS, 1.230174104, 4, {0}, {0}, {0}, {1,1,1,1}, {-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971}},/* WT 9-7 IRR*/
	{1, 0, J3D_ATK_WS, J3D_ATK_REV, 0, J3D_ATK_WS, 0, 2, {0}, {1,2}, {1,2}, {1,1}, {-1,1}},/* WT 5-3 REV*/
	{2, 0, J3D_ATK_ARB, J3D_ATK_REV, 0, J3D_ATK_CON, 0, 2, {0,0}, {0,1}, {0,1}, {1,1}, {{-1},{1}}}, /* WT 2-2 REV*/
	{3, 0, J3D_ATK_ARB, J3D_ATK_REV, 1, J3D_ATK_CON, 0, 3, {0,0,-1}, {0,1,2}, {0,1,2}, {1,1,3}, {{-1},{1},{1,0,-1}}}, /* WT 2-6 REV*/
	{4, 0, J3D_ATK_ARB, J3D_ATK_REV, 1, J3D_ATK_CON, 0, 3, {0,0,-2}, {0,1,6}, {0,1,32}, {1,1,5}, {{-1},{1},{-3,22,0,-22,3}}}, /* WT 2-10 REV*/
	{5, 1, J3D_ATK_ARB, J3D_ATK_IRR, 1, J3D_ATK_WS, 1, 7, {0}, {0}, {0}, {1,1,2,1,2,1,3},{{-1},{1.58613434206},{-0.460348209828, 0.460348209828},{0.25},{0.374213867768,-0.374213867768},{-1.33613434206},{0.29306717103,0,-0.29306717103}}}, /* WT 6-10 IRR*/
	{6, 1, J3D_ATK_ARB, J3D_ATK_IRR, 0, J3D_ATK_WS, 1, 11, {0}, {0}, {0}, {1,1,2,1,2,1,2,1,2,1,5},{{-1},{0,99715069105},{-1.00573127827, 1.00573127827},{-0.27040357631},{2.20509972343, -2.20509972343},{0.08059995736},
		{-1.62682532350, 1.62682532350},{0.52040357631},{0.60404664250, -0.60404664250},{-0.82775064841},{-0.06615812964, 0.29402137720, 0, -0.29402137720, 0.06615812964}}}, /* WT 10-18 IRR*/
	{7, 1, J3D_ATK_WS, J3D_ATK_IRR, 0, J3D_ATK_WS, 1, 2, {0}, {0}, {0}, {1,1}, {-0.5, 0.25}},	/* WT 5-3 IRR*/
	{8, 0, J3D_ATK_WS, J3D_ATK_REV, 0, J3D_ATK_WS, 0, 2, {0}, {4,4}, {8,8}, {2,2}, {{-9,1},{5,-1}}}		/* WT 13-7 REV*/
};
/* 
==========================================================
   local functions
==========================================================
*/

/* <summary>			                 */
/* Forward lazy transform (horizontal).  */
/* </summary>                            */ 
static void dwt_deinterleave_h(int *a, int *b, int dn, int sn, int cas) {
	int i;
    for (i=0; i<sn; i++) b[i]=a[2*i+cas];
    for (i=0; i<dn; i++) b[sn+i]=a[(2*i+1-cas)];
}

/* <summary>                             */  
/* Forward lazy transform (vertical).    */
/* </summary>                            */ 
static void dwt_deinterleave_v(int *a, int *b, int dn, int sn, int x, int cas) {
    int i;
    for (i=0; i<sn; i++) b[i*x]=a[2*i+cas];
    for (i=0; i<dn; i++) b[(sn+i)*x]=a[(2*i+1-cas)];
}

/* <summary>                             */  
/* Forward lazy transform (axial).       */
/* </summary>                            */ 
static void dwt_deinterleave_z(int *a, int *b, int dn, int sn, int xy, int cas) {
    int i;
    for (i=0; i<sn; i++) b[i*xy]=a[2*i+cas];
    for (i=0; i<dn; i++) b[(sn+i)*xy]=a[(2*i+1-cas)];
}

/* <summary>                             */
/* Inverse lazy transform (horizontal).  */
/* </summary>                            */
static void dwt_interleave_h(int *a, int *b, int dn, int sn, int cas) {
    int i;
    int *ai = NULL;
    int *bi = NULL;
    ai = a;
    bi = b + cas;
    for (i = 0; i < sn; i++) {
      *bi = *ai;  
	  bi += 2;  
	  ai++;
    }
    ai = a + sn;
    bi = b + 1 - cas;
    for (i = 0; i < dn; i++) {
      *bi = *ai;
	  bi += 2;
	  ai++;
    }
}

/* <summary>                             */  
/* Inverse lazy transform (vertical).    */
/* </summary>                            */ 
static void dwt_interleave_v(int *a, int *b, int dn, int sn, int x, int cas) {
    int i;
    int *ai = NULL;
    int *bi = NULL;
    ai = a;
    bi = b + cas;
    for (i = 0; i < sn; i++) {
      *bi = *ai;
	  bi += 2;
	  ai += x;
    }
    ai = a + (sn * x);
    bi = b + 1 - cas;
    for (i = 0; i < dn; i++) {
      *bi = *ai;
	  bi += 2;  
	  ai += x;
    }
}

/* <summary>                             */
/* Inverse lazy transform (axial).  */
/* </summary>                            */
static void dwt_interleave_z(int *a, int *b, int dn, int sn, int xy, int cas) {
    int i;
    int *ai = NULL;
    int *bi = NULL;
    ai = a;
    bi = b + cas;
    for (i = 0; i < sn; i++) {
      *bi = *ai;  
	  bi += 2;  
	  ai += xy;
    }
    ai = a + (sn * xy);
    bi = b + 1 - cas;
    for (i = 0; i < dn; i++) {
      *bi = *ai;
	  bi += 2;
	  ai += xy;
    }
}


/* <summary>                            */
/* Forward 5-3 or 9-7 wavelet tranform in 1-D. */
/* </summary>                           */
static void dwt_encode_53(int *a, int dn, int sn, int cas) {
	int i;

	if (!cas) {
		if ((dn > 0) || (sn > 1)) {	/* NEW :  CASE ONE ELEMENT */
			//for (i = 0; i < dn; i++) D(i) -= (S_(i) + S_(i + 1)) >> 1;
			//for (i = 0; i < sn; i++) S(i) += (D_(i - 1) + D_(i) + 2) >> 2;
			for (i = 0; i < dn; i++){
				D(i) -= (S_(i) + S_(i + 1)) >> 1;
				//ops += 2;
			}
			for (i = 0; i < sn; i++){
				S(i) += (D_(i - 1) + D_(i) + 2) >> 2;
				//ops += 3;
			}
		}
	} else {
		/*if (!sn && dn == 1)
			S(0) *= 2;
		else {
			for (i = 0; i < dn; i++) S(i) -= (DD_(i) + DD_(i - 1)) >> 1;
			for (i = 0; i < sn; i++) D(i) += (SS_(i) + SS_(i + 1) + 2) >> 2;
		}*/
		if (!sn && dn == 1){
			S(0) *= 2;
			//ops++;
		} else {
			for (i = 0; i < dn; i++){
				S(i) -= (DD_(i) + DD_(i - 1)) >> 1;
			//	ops += 2;
			}
			for (i = 0; i < sn; i++){
				D(i) += (SS_(i) + SS_(i + 1) + 2) >> 2;
			//	ops += 3;
			}
		}
	}
}
static void dwt_encode_97(int *a, int dn, int sn, int cas) {
	int i;

	if (!cas) {
			if ((dn > 0) || (sn > 1)) {	/* NEW :  CASE ONE ELEMENT */
				for (i = 0; i < dn; i++)
					D(i) -= fix_mul(S_(i) + S_(i + 1), 12993);
				for (i = 0; i < sn; i++)
					S(i) -= fix_mul(D_(i - 1) + D_(i), 434);
				for (i = 0; i < dn; i++)
					D(i) += fix_mul(S_(i) + S_(i + 1), 7233);
				for (i = 0; i < sn; i++)
					S(i) += fix_mul(D_(i - 1) + D_(i), 3633);
				for (i = 0; i < dn; i++)
					D(i) = fix_mul(D(i), 5038);	/*5038 */
				for (i = 0; i < sn; i++)
					S(i) = fix_mul(S(i), 6659);	/*6660 */
			}
		} else {
			if ((sn > 0) || (dn > 1)) {	/* NEW :  CASE ONE ELEMENT */
				for (i = 0; i < dn; i++)
					S(i) -= fix_mul(DD_(i) + DD_(i - 1), 12993);
				for (i = 0; i < sn; i++)
					D(i) -= fix_mul(SS_(i) + SS_(i + 1), 434);
				for (i = 0; i < dn; i++)
					S(i) += fix_mul(DD_(i) + DD_(i - 1), 7233);
				for (i = 0; i < sn; i++)
					D(i) += fix_mul(SS_(i) + SS_(i + 1), 3633);
				for (i = 0; i < dn; i++)
					S(i) = fix_mul(S(i), 5038);	/*5038 */
				for (i = 0; i < sn; i++)
					D(i) = fix_mul(D(i), 6659);	/*6660 */
			}
		}
}
/* <summary>                            */
/* Inverse 5-3 or 9-7 wavelet tranform in 1-D. */
/* </summary>                           */ 
static void dwt_decode_53(int *a, int dn, int sn, int cas) {
	int i;
	if (!cas) {
		if ((dn > 0) || (sn > 1)) { /* NEW :  CASE ONE ELEMENT */
			for (i = 0; i < sn; i++) S(i) -= (D_(i - 1) + D_(i) + 2) >> 2;
			for (i = 0; i < dn; i++) D(i) += (S_(i) + S_(i + 1)) >> 1;
		}
	} else {
		if (!sn  && dn == 1)          /* NEW :  CASE ONE ELEMENT */
			S(0) /= 2;
		else {
			for (i = 0; i < sn; i++) D(i) -= (SS_(i) + SS_(i + 1) + 2) >> 2;
			for (i = 0; i < dn; i++) S(i) += (DD_(i) + DD_(i - 1)) >> 1;
		}
	}
}
static void dwt_decode_97(int *a, int dn, int sn, int cas) {
	int i;

	if (!cas) {
		if ((dn > 0) || (sn > 1)) {	/* NEW :  CASE ONE ELEMENT */
			for (i = 0; i < sn; i++)
				S(i) = fix_mul(S(i), 10078);	/* 10076 */
			for (i = 0; i < dn; i++)
				D(i) = fix_mul(D(i), 13318);	/* 13320 */
			for (i = 0; i < sn; i++)
				S(i) -= fix_mul(D_(i - 1) + D_(i), 3633);
			for (i = 0; i < dn; i++)
				D(i) -= fix_mul(S_(i) + S_(i + 1), 7233);
			for (i = 0; i < sn; i++)
				S(i) += fix_mul(D_(i - 1) + D_(i), 434);
			for (i = 0; i < dn; i++)
				D(i) += fix_mul(S_(i) + S_(i + 1), 12994);	/* 12993 */
		}
	} else {
		if ((sn > 0) || (dn > 1)) {	/* NEW :  CASE ONE ELEMENT */
			for (i = 0; i < sn; i++)
				D(i) = fix_mul(D(i), 10078);	/* 10076 */
			for (i = 0; i < dn; i++)
				S(i) = fix_mul(S(i), 13318);	/* 13320 */
			for (i = 0; i < sn; i++)
				D(i) -= fix_mul(SS_(i) + SS_(i + 1), 3633);
			for (i = 0; i < dn; i++)
				S(i) -= fix_mul(DD_(i) + DD_(i - 1), 7233);
			for (i = 0; i < sn; i++)
				D(i) += fix_mul(SS_(i) + SS_(i + 1), 434);
			for (i = 0; i < dn; i++)
				S(i) += fix_mul(DD_(i) + DD_(i - 1), 12994);	/* 12993 */
		}
	}
}


/* <summary>                */
/* Get norm of arbitrary wavelet transform. */
/* </summary>               */
static int upandconv(double *nXPS, double *LPS, int lenXPS, int lenLPS) {
	/* Perform the convolution of the vectors. */
	int i,j;
	double *tmp = (double *)opj_malloc(2*lenXPS * sizeof(double));
	//Upsample
	memset(tmp, 0, 2*lenXPS*sizeof(double));
	for (i = 0; i < lenXPS; i++) {
		*(tmp + 2*i) = *(nXPS + i);
		*(nXPS + i) = 0;
	}
	//Convolution
	for (i = 0; i < 2*lenXPS; i++) {
		for (j = 0; j < lenLPS; j++) {
			*(nXPS+i+j) = *(nXPS+i+j) + *(tmp + i) * *(LPS + j);
			//fprintf(stdout,"*(tmp + %d) * *(LPS + %d) = %f * %f \n",i,j,*(tmp + i),*(LPS + j));
		}
	}
	free(tmp);
	return 2*lenXPS+lenLPS-1;
}

static double dwt_calc_wtnorms(int orient, int level[3], int dwtid[3], opj_wtfilt_t *wtfiltX,  opj_wtfilt_t *wtfiltY,  opj_wtfilt_t *wtfiltZ) {
	int i, lenLPS, lenHPS;
	double	Lx = 0, Ly= 0, Hx= 0, Hy= 0, Lz= 0, Hz= 0;
	double *nLPSx, *nHPSx,*nLPSy, *nHPSy,*nLPSz, *nHPSz;
	int levelx, levely, levelz;
	    
	levelx = (orient == 0) ? level[0]-1 : level[0];
	levely = (orient == 0) ? level[1]-1 : level[1];
	levelz = (orient == 0) ? level[2]-1 : level[2];
	
	//X axis
	lenLPS = wtfiltX->lenLPS;
	lenHPS = wtfiltX->lenHPS;
	for (i = 0; i < levelx; i++) {
		lenLPS *= 2;
		lenHPS *= 2;
		lenLPS += wtfiltX->lenLPS - 1;
		lenHPS += wtfiltX->lenLPS - 1;
	}
	nLPSx = (double *)opj_malloc(lenLPS * sizeof(double));
	nHPSx = (double *)opj_malloc(lenHPS * sizeof(double));

	memcpy(nLPSx, wtfiltX->LPS, wtfiltX->lenLPS * sizeof(double));
	memcpy(nHPSx, wtfiltX->HPS, wtfiltX->lenHPS * sizeof(double));
	lenLPS = wtfiltX->lenLPS;
	lenHPS = wtfiltX->lenHPS;
	for (i = 0; i < levelx; i++) {
		lenLPS = upandconv(nLPSx, wtfiltX->LPS, lenLPS, wtfiltX->lenLPS);
		lenHPS = upandconv(nHPSx, wtfiltX->LPS, lenHPS, wtfiltX->lenLPS);
	}
	for (i = 0; i < lenLPS; i++)
		Lx += nLPSx[i] * nLPSx[i];
	for (i = 0; i < lenHPS; i++)
		Hx += nHPSx[i] * nHPSx[i];
	Lx = sqrt(Lx);
	Hx = sqrt(Hx);
	free(nLPSx);
	free(nHPSx);
	
	//Y axis
	if (dwtid[0] != dwtid[1] || level[0] != level[1]){
		lenLPS = wtfiltY->lenLPS;
		lenHPS = wtfiltY->lenHPS;
		for (i = 0; i < levely; i++) {
			lenLPS *= 2;
			lenHPS *= 2;
			lenLPS += wtfiltY->lenLPS - 1;
			lenHPS += wtfiltY->lenLPS - 1;
		}
		nLPSy = (double *)opj_malloc(lenLPS * sizeof(double));
		nHPSy = (double *)opj_malloc(lenHPS * sizeof(double));

		memcpy(nLPSy, wtfiltY->LPS, wtfiltY->lenLPS * sizeof(double));
		memcpy(nHPSy, wtfiltY->HPS, wtfiltY->lenHPS * sizeof(double));
		lenLPS = wtfiltY->lenLPS;
		lenHPS = wtfiltY->lenHPS;
		for (i = 0; i < levely; i++) {
			lenLPS = upandconv(nLPSy, wtfiltY->LPS, lenLPS, wtfiltY->lenLPS);
			lenHPS = upandconv(nHPSy, wtfiltY->LPS, lenHPS, wtfiltY->lenLPS);
		}
		for (i = 0; i < lenLPS; i++)
			Ly += nLPSy[i] * nLPSy[i];
		for (i = 0; i < lenHPS; i++)
			Hy += nHPSy[i] * nHPSy[i];
		Ly = sqrt(Ly);
		Hy = sqrt(Hy);
		free(nLPSy);
		free(nHPSy);
	} else { 
		Ly = Lx;
		Hy = Hx;
	}
	//Z axis
	if (levelz >= 0) { 
		lenLPS = wtfiltZ->lenLPS;
		lenHPS = wtfiltZ->lenHPS;
		for (i = 0; i < levelz; i++) {
			lenLPS *= 2;
			lenHPS *= 2;
			lenLPS += wtfiltZ->lenLPS - 1;
			lenHPS += wtfiltZ->lenLPS - 1;
		}
		nLPSz = (double *)opj_malloc(lenLPS * sizeof(double));
		nHPSz = (double *)opj_malloc(lenHPS * sizeof(double));

		memcpy(nLPSz, wtfiltZ->LPS, wtfiltZ->lenLPS * sizeof(double));
		memcpy(nHPSz, wtfiltZ->HPS, wtfiltZ->lenHPS * sizeof(double));
		lenLPS = wtfiltZ->lenLPS;
		lenHPS = wtfiltZ->lenHPS;
		for (i = 0; i < levelz; i++) {
			lenLPS = upandconv(nLPSz, wtfiltZ->LPS, lenLPS, wtfiltZ->lenLPS);
			lenHPS = upandconv(nHPSz, wtfiltZ->LPS, lenHPS, wtfiltZ->lenLPS);
		}
		for (i = 0; i < lenLPS; i++)
			Lz += nLPSz[i] * nLPSz[i];
		for (i = 0; i < lenHPS; i++)
			Hz += nHPSz[i] * nHPSz[i];
		Lz = sqrt(Lz);
		Hz = sqrt(Hz);
		free(nLPSz);
		free(nHPSz);
	} else {
		Lz = 1.0; Hz = 1.0;
	}
	switch (orient) {
		case 0: 
			return Lx * Ly * Lz;
		case 1:
			return Lx * Hy * Lz;
		case 2:
			return Hx * Ly * Lz;
		case 3:
			return Hx * Hy * Lz;
		case 4:
			return Lx * Ly * Hz;
		case 5:
			return Lx * Hy * Hz;
		case 6: 
			return Hx * Ly * Hz;
		case 7:
			return Hx * Hy * Hz;
		default:
			return -1;
	}
	
}
static void dwt_getwtfilters(opj_wtfilt_t *wtfilt, int dwtid) {
	if (dwtid == 0) { //DWT 9-7 
			wtfilt->lenLPS = 7;		wtfilt->lenHPS = 9;
			wtfilt->LPS = (double *)opj_malloc(wtfilt->lenLPS * sizeof(double));
			wtfilt->HPS = (double *)opj_malloc(wtfilt->lenHPS * sizeof(double));
			wtfilt->LPS[0] = -0.091271763114;	wtfilt->HPS[0] = 0.026748757411;
			wtfilt->LPS[1] = -0.057543526228;	wtfilt->HPS[1] = 0.016864118443;
			wtfilt->LPS[2] = 0.591271763114;	wtfilt->HPS[2] = -0.078223266529;
			wtfilt->LPS[3] = 1.115087052457;	wtfilt->HPS[3] = -0.266864118443;
			wtfilt->LPS[4] = 0.591271763114;	wtfilt->HPS[4] = 0.602949018236;
			wtfilt->LPS[5] = -0.057543526228;	wtfilt->HPS[5] = -0.266864118443;
			wtfilt->LPS[6] = -0.091271763114;	wtfilt->HPS[6] = -0.078223266529;
												wtfilt->HPS[7] = 0.016864118443;
												wtfilt->HPS[8] = 0.026748757411;			
	} else if (dwtid == 1) { //DWT 5-3 
			wtfilt->lenLPS = 3;		wtfilt->lenHPS = 5;
			wtfilt->LPS = (double *)opj_malloc(wtfilt->lenLPS * sizeof(double));
			wtfilt->HPS = (double *)opj_malloc(wtfilt->lenHPS * sizeof(double));
			wtfilt->LPS[0] = 0.5;	wtfilt->HPS[0] = -0.125; 
			wtfilt->LPS[1] = 1;		wtfilt->HPS[1] = -0.25; 
			wtfilt->LPS[2] = 0.5;	wtfilt->HPS[2] = 0.75;
									wtfilt->HPS[3] = -0.25; 
									wtfilt->HPS[4] = -0.125;
	} else {
		fprintf(stdout,"[ERROR] Sorry, this wavelet hasn't been implemented so far ... Try another one :-)\n");
		exit(1);
	}
}
/* <summary>                            */
/* Encoding of quantization stepsize for each subband. */
/* </summary>                           */ 
static void dwt_encode_stepsize(int stepsize, int numbps, opj_stepsize_t *bandno_stepsize) {
	int p, n;
	p = int_floorlog2(stepsize) - 13;
	n = 11 - int_floorlog2(stepsize);
	bandno_stepsize->mant = (n < 0 ? stepsize >> -n : stepsize << n) & 0x7ff;
	bandno_stepsize->expn = numbps - p;
	//if J3D_CCP_QNTSTY_NOQNT --> stepsize = 8192.0 --> p = 0, n = -2 --> mant = 0; expn = (prec+gain)
	//else --> bandno_stepsize = (1<<(numbps - expn)) + (1<<(numbps - expn - 11)) * Ub
}

/* 
==========================================================
   DWT interface
==========================================================
*/
/* <summary>                            */
/* Forward 5-3 wavelet tranform in 3-D. */
/* </summary>                           */
void dwt_encode(opj_tcd_tilecomp_t * tilec, int dwtid[3]) {
	int i, j, k;
	int x, y, z;
	int w, h, wh, d;
	int level,levelx,levely,levelz,diff;
	int *a = NULL;
	int *aj = NULL;
	int *bj = NULL;
	int *cj = NULL;
	
	ops = 0;

	memset(flagnorm,0,8000*sizeof(int));
	w = tilec->x1-tilec->x0;
	h = tilec->y1-tilec->y0;
	d = tilec->z1-tilec->z0;
	wh = w * h;
	levelx = tilec->numresolution[0]-1;
	levely = tilec->numresolution[1]-1;
	levelz = tilec->numresolution[2]-1;
	level = int_max(levelx,int_max(levely,levelz));
	diff = tilec->numresolution[0] - tilec->numresolution[2];

	a = tilec->data;

	for (x = 0, y = 0, z = 0; (x < levelx) && (y < levely); x++, y++, z++) {
		int rw;			/* width of the resolution level computed                                                           */
		int rh;			/* heigth of the resolution level computed                                                          */
		int rd;			/* depth of the resolution level computed                                                          */
		int rw1;		/* width of the resolution level once lower than computed one                                       */
		int rh1;		/* height of the resolution level once lower than computed one                                      */
		int rd1;		/* depth of the resolution level once lower than computed one                                      */
		int cas_col;	/* 0 = non inversion on horizontal filtering 1 = inversion between low-pass and high-pass filtering */
		int cas_row;	/* 0 = non inversion on vertical filtering 1 = inversion between low-pass and high-pass filtering   */
		int cas_axl;	/* 0 = non inversion on axial filtering 1 = inversion between low-pass and high-pass filtering   */
		int dn, sn;
		
		rw = tilec->resolutions[level - x].x1 - tilec->resolutions[level - x].x0;
		rh = tilec->resolutions[level - y].y1 - tilec->resolutions[level - y].y0;
		rd = tilec->resolutions[level - z].z1 - tilec->resolutions[level - z].z0;
		rw1= tilec->resolutions[level - x - 1].x1 - tilec->resolutions[level - x - 1].x0;
		rh1= tilec->resolutions[level - y - 1].y1 - tilec->resolutions[level - y - 1].y0;
		rd1= tilec->resolutions[level - z - 1].z1 - tilec->resolutions[level - z - 1].z0;
		
		cas_col = tilec->resolutions[level - x].x0 % 2; /* 0 = non inversion on horizontal filtering 1 = inversion between low-pass and high-pass filtering */
		cas_row = tilec->resolutions[level - y].y0 % 2; /* 0 = non inversion on vertical filtering 1 = inversion between low-pass and high-pass filtering   */
		cas_axl = tilec->resolutions[level - z].z0 % 2;
	
		/*fprintf(stdout," x %d y %d z %d \n",x,y,z);
		fprintf(stdout," levelx %d levely %d levelz %d \n",levelx,levely,levelz);
		fprintf(stdout," z1 %d z0 %d\n",tilec->resolutions[level - z].z1,tilec->resolutions[level - z].z0);
		fprintf(stdout," rw %d rh %d rd %d \n rw1 %d rh1 %d rd1 %d \n",rw,rh,rd,rw1,rh1,rd1);*/

		for (i = 0; i < rd; i++) {
			
			cj = a + (i * wh);
			
			//Horizontal
			sn = rw1;
			dn = rw - rw1;
			bj = (int*)opj_malloc(rw * sizeof(int));
			if (dwtid[0] == 0) {
				for (j = 0; j < rh; j++) {
					aj = cj + j * w;
					for (k = 0; k < rw; k++)  bj[k] = aj[k];
					dwt_encode_97(bj, dn, sn, cas_row);
					dwt_deinterleave_h(bj, aj, dn, sn, cas_row);
				}
			} else if (dwtid[0] == 1) {
				for (j = 0; j < rh; j++) {
					aj = cj + j * w;
					for (k = 0; k < rw; k++)  bj[k] = aj[k];
					dwt_encode_53(bj, dn, sn, cas_row);
					dwt_deinterleave_h(bj, aj, dn, sn, cas_row);
				}
			} 
			opj_free(bj);

			//Vertical
			sn = rh1;
			dn = rh - rh1;
			bj = (int*)opj_malloc(rh * sizeof(int));
			if (dwtid[1] == 0) { /*DWT 9-7*/
				for (j = 0; j < rw; j++) {
					aj = cj + j;
					for (k = 0; k < rh; k++)  bj[k] = aj[k*w];
					dwt_encode_97(bj, dn, sn, cas_col);
					dwt_deinterleave_v(bj, aj, dn, sn, w, cas_col);
				}
            } else if (dwtid[1] == 1) { /*DWT 5-3*/
				for (j = 0; j < rw; j++) {
					aj = cj + j;
					for (k = 0; k < rh; k++)  bj[k] = aj[k*w];
					dwt_encode_53(bj, dn, sn, cas_col);
					dwt_deinterleave_v(bj, aj, dn, sn, w, cas_col);
				}
			} 
			opj_free(bj);
		}

		if (z < levelz){
			//Axial fprintf(stdout,"Axial DWT Transform %d %d %d\n",z,rd,rd1);
			sn = rd1;
			dn = rd - rd1;
			bj = (int*)opj_malloc(rd * sizeof(int));
			if (dwtid[2] == 0) {
                for (j = 0; j < (rw*rh); j++) {
					aj = a + j;
					for (k = 0; k < rd; k++)  bj[k] = aj[k*wh];
					dwt_encode_97(bj, dn, sn, cas_axl);
					dwt_deinterleave_z(bj, aj, dn, sn, wh, cas_axl);
				}
			} else if (dwtid[2] == 1) {
				for (j = 0; j < (rw*rh); j++) {
					aj = a + j;
					for (k = 0; k < rd; k++)  bj[k] = aj[k*wh];
					dwt_encode_53(bj, dn, sn, cas_axl);
					dwt_deinterleave_z(bj, aj, dn, sn, wh, cas_axl);
				}
			} 
			opj_free(bj);
		}
	}

	//fprintf(stdout,"[INFO] Ops: %d \n",ops);
}


/* <summary>                            */
/* Inverse 5-3 wavelet tranform in 3-D. */
/* </summary>                           */
void dwt_decode(opj_tcd_tilecomp_t * tilec, int stops[3], int dwtid[3]) {
	int i, j, k;
	int x, y, z;
	int w, h, wh, d;
	int level, levelx, levely, levelz, diff;
	int *a = NULL;
	int *aj = NULL;
	int *bj = NULL;
	int *cj = NULL;
	
	a = tilec->data;

	w = tilec->x1-tilec->x0;
	h = tilec->y1-tilec->y0;
	d = tilec->z1-tilec->z0;
	wh = w * h;
	levelx = tilec->numresolution[0]-1;
	levely = tilec->numresolution[1]-1;
	levelz = tilec->numresolution[2]-1;
	level = int_max(levelx,int_max(levely,levelz));
	diff = tilec->numresolution[0] - tilec->numresolution[2];
		
/* General lifting framework -- DCCS-LIWT */
	for (x = level - 1, y = level - 1, z = level - 1; (x >= stops[0]) && (y >= stops[1]); x--, y--, z--) {
		int rw;			/* width of the resolution level computed                                                           */
		int rh;			/* heigth of the resolution level computed                                                          */
		int rd;			/* depth of the resolution level computed                                                          */
		int rw1;		/* width of the resolution level once lower than computed one                                       */
		int rh1;		/* height of the resolution level once lower than computed one                                      */
		int rd1;		/* depth of the resolution level once lower than computed one                                      */
		int cas_col;	/* 0 = non inversion on horizontal filtering 1 = inversion between low-pass and high-pass filtering */
		int cas_row;	/* 0 = non inversion on vertical filtering 1 = inversion between low-pass and high-pass filtering   */
		int cas_axl;	/* 0 = non inversion on axial filtering 1 = inversion between low-pass and high-pass filtering   */
		int dn, sn;
		
		rw = tilec->resolutions[level - x].x1 - tilec->resolutions[level - x].x0;
		rh = tilec->resolutions[level - y].y1 - tilec->resolutions[level - y].y0;
		rd = tilec->resolutions[level - z].z1 - tilec->resolutions[level - z].z0;
		rw1= tilec->resolutions[level - x - 1].x1 - tilec->resolutions[level - x - 1].x0;
		rh1= tilec->resolutions[level - y - 1].y1 - tilec->resolutions[level - y - 1].y0;
		rd1= tilec->resolutions[level - z - 1].z1 - tilec->resolutions[level - z - 1].z0;
		
		cas_col = tilec->resolutions[level - x].x0 % 2; /* 0 = non inversion on horizontal filtering 1 = inversion between low-pass and high-pass filtering */
		cas_row = tilec->resolutions[level - y].y0 % 2; /* 0 = non inversion on vertical filtering 1 = inversion between low-pass and high-pass filtering   */
		cas_axl = tilec->resolutions[level - z].z0 % 2;
	
		/*fprintf(stdout," x %d y %d z %d \n",x,y,z);
		fprintf(stdout," levelx %d levely %d levelz %d \n",levelx,levely,levelz);
		fprintf(stdout," dwtid[0] %d [1] %d [2] %d \n",dwtid[0],dwtid[1],dwtid[2]);
		fprintf(stdout," rw %d rh %d rd %d \n rw1 %d rh1 %d rd1 %d \n",rw,rh,rd,rw1,rh1,rd1);
		fprintf(stdout,"IDWT Transform %d %d %d %d\n",level, z, rd,rd1);*/

		if (z >= stops[2] && rd != rd1) {
			//fprintf(stdout,"Axial Transform %d %d %d %d\n",levelz, z, rd,rd1);
			sn = rd1;
			dn = rd - rd1;
			bj = (int*)opj_malloc(rd * sizeof(int));
			if (dwtid[2] == 0) {
				for (j = 0; j < (rw*rh); j++) {
					aj = a + j;
					dwt_interleave_z(aj, bj, dn, sn, wh, cas_axl);
					dwt_decode_97(bj, dn, sn, cas_axl);
					for (k = 0; k < rd; k++)  aj[k * wh] = bj[k];
				}
			} else if (dwtid[2] == 1) {
				for (j = 0; j < (rw*rh); j++) {
					aj = a + j;
					dwt_interleave_z(aj, bj, dn, sn, wh, cas_axl);
					dwt_decode_53(bj, dn, sn, cas_axl);
					for (k = 0; k < rd; k++)  aj[k * wh] = bj[k];
				}
			} 
			opj_free(bj);
		}

		for (i = 0; i < rd; i++) {
			//Fetch corresponding slice for doing DWT-2D
 			cj = tilec->data + (i * wh);
			
			//Vertical
			sn = rh1;
			dn = rh - rh1;
			bj = (int*)opj_malloc(rh * sizeof(int));
			if (dwtid[1] == 0) {
				for (j = 0; j < rw; j++) {
					aj = cj + j;
					dwt_interleave_v(aj, bj, dn, sn, w, cas_col);
					dwt_decode_97(bj, dn, sn, cas_col);
					for (k = 0; k < rh; k++)  aj[k * w] = bj[k];
				}
			} else if (dwtid[1] == 1) {
				for (j = 0; j < rw; j++) {
					aj = cj + j;
					dwt_interleave_v(aj, bj, dn, sn, w, cas_col);
					dwt_decode_53(bj, dn, sn, cas_col);
					for (k = 0; k < rh; k++)  aj[k * w] = bj[k];
				}
			} 
			opj_free(bj);

			//Horizontal
			sn = rw1;
			dn = rw - rw1;
			bj = (int*)opj_malloc(rw * sizeof(int));
			if (dwtid[0]==0) {
				for (j = 0; j < rh; j++) {
					aj = cj + j*w;
					dwt_interleave_h(aj, bj, dn, sn, cas_row);
					dwt_decode_97(bj, dn, sn, cas_row);
					for (k = 0; k < rw; k++)  aj[k] = bj[k];
				}
			} else if (dwtid[0]==1) {
				for (j = 0; j < rh; j++) {
					aj = cj + j*w;
					dwt_interleave_h(aj, bj, dn, sn, cas_row);
					dwt_decode_53(bj, dn, sn, cas_row);
					for (k = 0; k < rw; k++)  aj[k] = bj[k];
				}
			} 
			opj_free(bj);
			
		}
	
	}

}


/* <summary>                          */
/* Get gain of wavelet transform. */
/* </summary>                         */
int dwt_getgain(int orient, int reversible) {
	if (reversible == 1) { 
		if (orient == 0)
			return 0;
		else if (orient == 1 || orient == 2 || orient == 4 )
			return 1;
		else if (orient == 3 || orient == 5 || orient == 6 )
			return 2;
		else 
			return 3;
	}
	//else if (reversible == 0){
	return 0;
}

/* <summary>                */
/* Get norm of wavelet transform. */
/* </summary>               */
double dwt_getnorm(int orient, int level[3], int dwtid[3]) {
	int levelx = level[0];
	int levely = level[1];
	int levelz = (level[2] < 0) ? 0 : level[2];
	double norm;

	if (flagnorm[levelx][levely][levelz][orient] == 1) {
		norm = dwt_norm[levelx][levely][levelz][orient];
		//fprintf(stdout,"[INFO] Level: %d %d %d Orient %d Dwt_norm: %f \n",level[0],level[1],level[2],orient,norm);
	} else {
		opj_wtfilt_t *wtfiltx =(opj_wtfilt_t *) opj_malloc(sizeof(opj_wtfilt_t));
		opj_wtfilt_t *wtfilty =(opj_wtfilt_t *) opj_malloc(sizeof(opj_wtfilt_t));
		opj_wtfilt_t *wtfiltz =(opj_wtfilt_t *) opj_malloc(sizeof(opj_wtfilt_t));
		//Fetch equivalent filters for each dimension
		dwt_getwtfilters(wtfiltx, dwtid[0]);
		dwt_getwtfilters(wtfilty, dwtid[1]);
		dwt_getwtfilters(wtfiltz, dwtid[2]);
		//Calculate the corresponding norm 
		norm = dwt_calc_wtnorms(orient, level, dwtid, wtfiltx, wtfilty, wtfiltz);
		//Save norm in array (no recalculation)
		dwt_norm[levelx][levely][levelz][orient] = norm;
		flagnorm[levelx][levely][levelz][orient] = 1;
		//Free reserved space
		opj_free(wtfiltx->LPS);	opj_free(wtfilty->LPS);	opj_free(wtfiltz->LPS);
		opj_free(wtfiltx->HPS);	opj_free(wtfilty->HPS);	opj_free(wtfiltz->HPS);
		opj_free(wtfiltx);		opj_free(wtfilty);		opj_free(wtfiltz);
		//fprintf(stdout,"[INFO] Dwtid: %d %d %d Level: %d %d %d Orient %d Norm: %f \n",dwtid[0],dwtid[1],dwtid[2],level[0],level[1],level[2],orient,norm);
	} 
	return norm;
}
/* <summary>								*/
/* Calculate explicit stepsizes for DWT.	*/
/* </summary>								*/
void dwt_calc_explicit_stepsizes(opj_tccp_t * tccp, int prec) { 
	int totnumbands, bandno, diff;
	
	assert(tccp->numresolution[0] >= tccp->numresolution[2]);	
	diff = tccp->numresolution[0] - tccp->numresolution[2];		/*if RESx=RESy != RESz */
	totnumbands = (7 * tccp->numresolution[0] - 6) - 4 * diff; /* 3-D */
		
	for (bandno = 0; bandno < totnumbands; bandno++) {
		double stepsize;
		int resno, level[3], orient, gain;

		/* Bandno:	0 - LLL 	1 - LHL 
					2 - HLL		3 - HHL
					4 - LLH		5 - LHH
					6 - HLH		7 - HHH	*/

		resno = (bandno == 0) ? 0 : ( (bandno <= 3 * diff) ? ((bandno - 1) / 3 + 1) : ((bandno + 4*diff - 1) / 7 + 1));
		orient = (bandno == 0) ? 0 : ( (bandno <= 3 * diff) ? ((bandno - 1) % 3 + 1) : ((bandno + 4*diff - 1) % 7 + 1));
		level[0] = tccp->numresolution[0] - 1 - resno;
		level[1] = tccp->numresolution[1] - 1 - resno;
		level[2] = tccp->numresolution[2] - 1 - resno;
	
		/* Gain:	0 - LLL 	1 - LHL 
					1 - HLL		2 - HHL
					1 - LLH		2 - LHH
					2 - HLH		3 - HHH		*/
		gain = (tccp->reversible == 0) ? 0 : ( (orient == 0) ? 0 : 
				( ((orient == 1) || (orient == 2) || (orient == 4)) ? 1 : 
						(((orient == 3) || (orient == 5) || (orient == 6)) ? 2 : 3)) );
				
		if (tccp->qntsty == J3D_CCP_QNTSTY_NOQNT) {
			stepsize = 1.0;
		} else {
			double norm = dwt_getnorm(orient,level,tccp->dwtid); //Fetch norms if irreversible transform (by the moment only I9.7)
			stepsize = (1 << (gain + 1)) / norm;
		}
		//fprintf(stdout,"[INFO] Bandno: %d Orient: %d Level: %d %d %d Stepsize: %f\n",bandno,orient,level[0],level[1],level[2],stepsize);
		dwt_encode_stepsize((int) floor(stepsize * 8192.0), prec + gain, &tccp->stepsizes[bandno]);
	}
}



