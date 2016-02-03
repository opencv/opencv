/*
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
#ifndef __T1_3D_H
#define __T1_3D_H
/**
@file t1_3d.h
@brief Implementation of the tier-1 coding (coding of code-block coefficients) (T1)

The functions in T1_3D.C have for goal to realize the tier-1 coding operation of 3D-EBCOT.
The functions in T1_3D.C are used by some function in TCD.C.
*/

/** @defgroup T1_3D T1_3D - Implementation of the tier-1 coding */
/*@{*/

/* ----------------------------------------------------------------------- */

/* Neighbourhood of 3D EBCOT (Significance context)*/
#define T1_3D_SIG_NE  0x00000001	/*< Context orientation : North-East direction */
#define T1_3D_SIG_SE  0x00000002	/*< Context orientation : South-East direction */
#define T1_3D_SIG_SW  0x00000004	/*< Context orientation : South-West direction */
#define T1_3D_SIG_NW  0x00000008	/* Context orientation : North-West direction */
#define T1_3D_SIG_N   0x00000010	/*< Context orientation : North direction */
#define T1_3D_SIG_E   0x00000020	/*< Context orientation : East direction */
#define T1_3D_SIG_S   0x00000040	/*< Context orientation : South direction */
#define T1_3D_SIG_W   0x00000080	/*< Context orientation : West direction */
#define T1_3D_SIG_FC  0x00000100	/*< Context orientation : Forward Central direction */		
#define T1_3D_SIG_BC  0x00000200	/*< Context orientation : Backward Central direction */	
#define T1_3D_SIG_FNE 0x00000400	/*< Context orientation : Forward North-East direction */	
#define T1_3D_SIG_FSE 0x00000800	/*< Context orientation : Forward South-East direction */	
#define T1_3D_SIG_FSW 0x00001000	/*< Context orientation : Forward South-West direction */	
#define T1_3D_SIG_FNW 0x00002000	/*< Context orientation : Forward North-West direction */	
#define T1_3D_SIG_FN  0x00004000	/*< Context orientation : Forward North direction */		
#define T1_3D_SIG_FE  0x00008000	/*< Context orientation : Forward East direction */		
#define T1_3D_SIG_FS  0x00010000	/*< Context orientation : Forward South direction */		
#define T1_3D_SIG_FW  0x00020000	/*< Context orientation : Forward West direction */		
#define T1_3D_SIG_BNE 0x00040000	/*< Context orientation : Backward North-East direction */	
#define T1_3D_SIG_BSE 0x00080000	/*< Context orientation : Backward South-East direction */	
#define T1_3D_SIG_BSW 0x00100000	/*< Context orientation : Backward South-West direction */	
#define T1_3D_SIG_BNW 0x00200000	/*< Context orientation : Backward North-West direction */	
#define T1_3D_SIG_BN  0x00400000	/*< Context orientation : Backward North direction */		
#define T1_3D_SIG_BE  0x00800000	/*< Context orientation : Backward East direction */		
#define T1_3D_SIG_BS  0x01000000	/*< Context orientation : Backward South direction */		
#define T1_3D_SIG_BW  0x02000000	/*< Context orientation : Backward West direction */		
#define T1_3D_SIG_COTH	(T1_3D_SIG_N|T1_3D_SIG_NE|T1_3D_SIG_E|T1_3D_SIG_SE|T1_3D_SIG_S|T1_3D_SIG_SW|T1_3D_SIG_W|T1_3D_SIG_NW)
#define T1_3D_SIG_BOTH	(T1_3D_SIG_BN|T1_3D_SIG_BNE|T1_3D_SIG_BE|T1_3D_SIG_BSE|T1_3D_SIG_BS|T1_3D_SIG_BSW|T1_3D_SIG_BW|T1_3D_SIG_BNW|T1_3D_SIG_BC)
#define T1_3D_SIG_FOTH  (T1_3D_SIG_FN|T1_3D_SIG_FNE|T1_3D_SIG_FE|T1_3D_SIG_FSE|T1_3D_SIG_FS|T1_3D_SIG_FSW|T1_3D_SIG_FW|T1_3D_SIG_FNW|T1_3D_SIG_FC)
#define T1_3D_SIG_OTH	(T1_3D_SIG_FOTH|T1_3D_SIG_BOTH|T1_3D_SIG_COTH)
#define T1_3D_SIG_PRIM	(T1_3D_SIG_N|T1_3D_SIG_E|T1_3D_SIG_S|T1_3D_SIG_W|T1_3D_SIG_FC|T1_3D_SIG_BC)

#define T1_3D_SGN_N		0x0400		
#define T1_3D_SGN_E		0x0800		
#define T1_3D_SGN_S		0x1000		
#define T1_3D_SGN_W		0x2000		
#define T1_3D_SGN_F		0x4000	
#define T1_3D_SGN_B		0x8000
#define T1_3D_SGN		(T1_3D_SGN_N|T1_3D_SGN_E|T1_3D_SGN_S|T1_3D_SGN_W|T1_3D_SGN_F|T1_3D_SGN_B)

#define T1_3D_SIG		0x0001  //Significance state
#define T1_3D_REFINE	0x0002  //Delayed significance
#define T1_3D_VISIT		0x0004  //First-pass membership

#define T1_3D_NUMCTXS_AGG	1
#define T1_3D_NUMCTXS_ZC	16
#define T1_3D_NUMCTXS_MAG	3
#define T1_3D_NUMCTXS_SC	6
#define T1_3D_NUMCTXS_UNI	1

#define T1_3D_CTXNO_AGG 0
#define T1_3D_CTXNO_ZC	(T1_3D_CTXNO_AGG+T1_3D_NUMCTXS_AGG) //1
#define T1_3D_CTXNO_MAG (T1_3D_CTXNO_ZC+T1_3D_NUMCTXS_ZC)	//17
#define T1_3D_CTXNO_SC	(T1_3D_CTXNO_MAG+T1_3D_NUMCTXS_MAG)	//20
#define T1_3D_CTXNO_UNI (T1_3D_CTXNO_SC+T1_3D_NUMCTXS_SC)	//26
#define T1_3D_NUMCTXS	(T1_3D_CTXNO_UNI+T1_3D_NUMCTXS_UNI) //27


/* ----------------------------------------------------------------------- */

/**
Tier-1 coding (coding of code-block coefficients)
*/
typedef struct opj_t1_3d {
	/** Codec context */
	opj_common_ptr cinfo;
	/** MQC component */
	opj_mqc_t *mqc;
	/** RAW component */
	opj_raw_t *raw;
	/** LUTs for decoding normalised MSE */
	int lut_nmsedec_sig[1 << T1_NMSEDEC_BITS];
	int lut_nmsedec_sig0[1 << T1_NMSEDEC_BITS];
	int lut_nmsedec_ref[1 << T1_NMSEDEC_BITS];
	int lut_nmsedec_ref0[1 << T1_NMSEDEC_BITS];
	/** Codeblock data */
	int data[T1_CBLKD][T1_CBLKH][T1_CBLKW];
	/** Context information for each voxel in codeblock */
	unsigned int flags[T1_CBLKD + 2][T1_CBLKH + 2][T1_CBLKH + 2];
	/** Voxel information (significance/visited/refined) */
	int flagSVR[T1_CBLKD + 2][T1_CBLKH + 2][T1_CBLKH + 2];
} opj_t1_3d_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Create a new T1_3D handle 
and initialize the look-up tables of the Tier-1 coder/decoder
@return Returns a new T1 handle if successful, returns NULL otherwise
@see t1_init_luts
*/
opj_t1_3d_t* t1_3d_create(opj_common_ptr cinfo);
/**
Destroy a previously created T1_3D handle
@param t1 T1_3D handle to destroy
*/
void t1_3d_destroy(opj_t1_3d_t *t1);
/**
Encode the code-blocks of a tile
@param t1 T1_3D handle
@param tile The tile to encode
@param tcp Tile coding parameters
*/
void t1_3d_encode_cblks(opj_t1_3d_t *t1, opj_tcd_tile_t *tile, opj_tcp_t *tcp);
/**
Decode the code-blocks of a tile
@param t1 T1_3D handle
@param tile The tile to decode
@param tcp Tile coding parameters
*/
void t1_3d_decode_cblks(opj_t1_3d_t *t1, opj_tcd_tile_t *tile, opj_tcp_t *tcp);
/**
Get weigths of MSE decoding
@param nmsedec The normalized MSE reduction
@param compno 
@param level 
@param orient
@param bpno
@param reversible
@param stepsize
@param numcomps
@param dwtid
returns MSE associated to decoding pass
double t1_3d_getwmsedec(int nmsedec, int compno, int levelxy, int levelz, int orient, int bpno, int reversible, double stepsize, int numcomps, int dwtid);
*/
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __T1_H */
