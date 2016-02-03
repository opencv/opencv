/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2006, Mónica Díez, LPI-UVA, Spain
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

#include "opj_includes.h"

/** @defgroup PI PI - Implementation of a packet iterator */
/*@{*/

/** @name Funciones locales */
/*@{*/

/**
Get next packet in layer-resolution-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true 
*/
static bool pi_next_lrcp(opj_pi_iterator_t * pi);
/**
Get next packet in resolution-layer-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true 
*/
static bool pi_next_rlcp(opj_pi_iterator_t * pi);
/**
Get next packet in resolution-precinct-component-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true 
*/
static bool pi_next_rpcl(opj_pi_iterator_t * pi);
/**
Get next packet in precinct-component-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true 
*/
static bool pi_next_pcrl(opj_pi_iterator_t * pi);
/**
Get next packet in component-precinct-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true 
*/
static bool pi_next_cprl(opj_pi_iterator_t * pi);

/*@}*/

/*@}*/

/* 
==========================================================
   local functions
==========================================================
*/

static bool pi_next_lrcp(opj_pi_iterator_t * pi) {
	opj_pi_comp_t *comp = NULL;
	opj_pi_resolution_t *res = NULL;
	long index = 0;

	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		res = &comp->resolutions[pi->resno];
		goto LABEL_SKIP;
	} else {
		pi->first = 0;
	}

	for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
		for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
			for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
				comp = &pi->comps[pi->compno];
				if (pi->resno >= comp->numresolution[0]) {
					continue;
				}
				res = &comp->resolutions[pi->resno];
				//for (pi->precno = 0; pi->precno < (res->prctno[0] * res->prctno[1]); pi->precno++) {
				for (pi->precno = 0; pi->precno < (res->prctno[0] * res->prctno[1] * res->prctno[2]); pi->precno++) {
					index = pi->layno * pi->step_l 
						+ pi->resno * pi->step_r 
						+ pi->compno * pi->step_c 
						+ pi->precno * pi->step_p;
					if (!pi->include[index]) {
						pi->include[index] = 1;
						return true;
					}
LABEL_SKIP:;

				}
			}
		}
	}
	
	return false;
}

static bool pi_next_rlcp(opj_pi_iterator_t * pi) {
	opj_pi_comp_t *comp = NULL;
	opj_pi_resolution_t *res = NULL;
	long index = 0;

	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		res = &comp->resolutions[pi->resno];
		goto LABEL_SKIP;
	} else {
		pi->first = 0;
	}

	for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
		for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
			for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
				comp = &pi->comps[pi->compno];
				if (pi->resno >= comp->numresolution[0]) {
					continue;
				}
				res = &comp->resolutions[pi->resno];
				//for (pi->precno = 0; pi->precno < (res->prctno[0] * res->prctno[1]); pi->precno++) {
				for (pi->precno = 0; pi->precno < (res->prctno[0] * res->prctno[1] * res->prctno[2]); pi->precno++) {
					index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
					if (!pi->include[index]) {
						pi->include[index] = 1;
						return true;
					}
LABEL_SKIP:;
				}
			}
		}
	}
	
	return false;
}

static bool pi_next_rpcl(opj_pi_iterator_t * pi) {
	opj_pi_comp_t *comp = NULL;
	opj_pi_resolution_t *res = NULL;
	long index = 0;

	if (!pi->first) {
		goto LABEL_SKIP;
	} else {
		int compno, resno;
		pi->first = 0;
		pi->dx = 0;
		pi->dy = 0;
		for (compno = 0; compno < pi->numcomps; compno++) {
			comp = &pi->comps[compno];
			for (resno = 0; resno < comp->numresolution[0]; resno++) {
				int dx, dy,dz;
				res = &comp->resolutions[resno];
				dx = comp->dx * (1 << (res->pdx + comp->numresolution[0] - 1 - resno));
				dy = comp->dy * (1 << (res->pdy + comp->numresolution[1] - 1 - resno));
				dz = comp->dz * (1 << (res->pdz + comp->numresolution[2] - 1 - resno));
				pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
				pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
				pi->dz = !pi->dz ? dz : int_min(pi->dz, dz);
			}
		}
	}

	for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
		for (pi->z = pi->tz0; pi->z < pi->tz1; pi->z += pi->dz - (pi->z % pi->dz)) {
			for (pi->y = pi->ty0; pi->y < pi->ty1; pi->y += pi->dy - (pi->y % pi->dy)) {
				for (pi->x = pi->tx0; pi->x < pi->tx1; pi->x += pi->dx - (pi->x % pi->dx)) {
					for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
						int levelnox, levelnoy, levelnoz;
						int trx0, try0, trz0;
						int trx1, try1, trz1;
						int rpx, rpy, rpz;
						int prci, prcj, prck;
						comp = &pi->comps[pi->compno];
						if (pi->resno >= comp->numresolution[0]) {
							continue;
						}
						res = &comp->resolutions[pi->resno];
						levelnox = comp->numresolution[0] - 1 - pi->resno;
						levelnoy = comp->numresolution[1] - 1 - pi->resno;
						levelnoz = comp->numresolution[2] - 1 - pi->resno;
						trx0 = int_ceildiv(pi->tx0, comp->dx << levelnox);
						try0 = int_ceildiv(pi->ty0, comp->dy << levelnoy);
						trz0 = int_ceildiv(pi->tz0, comp->dz << levelnoz);
						trx1 = int_ceildiv(pi->tx1, comp->dx << levelnox);
						try1 = int_ceildiv(pi->ty1, comp->dy << levelnoy);
						trz1 = int_ceildiv(pi->tz1, comp->dz << levelnoz);
						rpx = res->pdx + levelnox;
						rpy = res->pdy + levelnoy;
						rpz = res->pdz + levelnoz;
						if ((!(pi->x % (comp->dx << rpx) == 0) || (pi->x == pi->tx0 && (trx0 << levelnox) % (1 << rpx)))) {
							continue;
						}
						if ((!(pi->y % (comp->dy << rpy) == 0) || (pi->y == pi->ty0 && (try0 << levelnoy) % (1 << rpx)))) {
							continue;
						}
						if ((!(pi->z % (comp->dz << rpz) == 0) || (pi->z == pi->tz0 && (trz0 << levelnoz) % (1 << rpx)))) {
							continue;
						}
						if ((res->prctno[0]==0)||(res->prctno[1]==0)||(res->prctno[2]==0)) continue;
						
						if ((trx0==trx1)||(try0==try1)||(trz0==trz1)) continue;
						
						prci = int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelnox), res->pdx) 
							- int_floordivpow2(trx0, res->pdx);
						prcj = int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelnoy), res->pdy) 
							- int_floordivpow2(try0, res->pdy);
						prck = int_floordivpow2(int_ceildiv(pi->z, comp->dz << levelnoz), res->pdz) 
							- int_floordivpow2(trz0, res->pdz);
						pi->precno = prci + prcj * res->prctno[0] + prck * res->prctno[0] * res->prctno[1];
						for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
							index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
							if (!pi->include[index]) {
								pi->include[index] = 1;
								return true;
							}
	LABEL_SKIP:;
						}
					}
				}
			}
		}
	}
	
	return false;
}

static bool pi_next_pcrl(opj_pi_iterator_t * pi) {
	opj_pi_comp_t *comp = NULL;
	opj_pi_resolution_t *res = NULL;
	long index = 0;

	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		goto LABEL_SKIP;
	} else {
		int compno, resno;
		pi->first = 0;
		pi->dx = 0;
		pi->dy = 0;
		pi->dz = 0;
		for (compno = 0; compno < pi->numcomps; compno++) {
			comp = &pi->comps[compno];
			for (resno = 0; resno < comp->numresolution[0]; resno++) {
				int dx, dy, dz;
				res = &comp->resolutions[resno];
				dx = comp->dx * (1 << (res->pdx + comp->numresolution[0] - 1 - resno));
				dy = comp->dy * (1 << (res->pdy + comp->numresolution[1] - 1 - resno));
				dz = comp->dz * (1 << (res->pdy + comp->numresolution[2] - 1 - resno));
				pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
				pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
				pi->dz = !pi->dz ? dz : int_min(pi->dz, dz);
			}
		}
	}

for (pi->z = pi->tz0; pi->z < pi->tz1; pi->z += pi->dz - (pi->z % pi->dz)) {
	for (pi->y = pi->ty0; pi->y < pi->ty1; pi->y += pi->dy - (pi->y % pi->dy)) {
		for (pi->x = pi->tx0; pi->x < pi->tx1; pi->x += pi->dx - (pi->x % pi->dx)) {
			for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
				comp = &pi->comps[pi->compno];
				for (pi->resno = pi->poc.resno0; pi->resno < int_min(pi->poc.resno1, comp->numresolution[0]); pi->resno++) {
						int levelnox, levelnoy, levelnoz;
						int trx0, try0, trz0;
						int trx1, try1, trz1;
						int rpx, rpy, rpz;
						int prci, prcj, prck;
						comp = &pi->comps[pi->compno];
						if (pi->resno >= comp->numresolution[0]) {
							continue;
						}
						res = &comp->resolutions[pi->resno];
						levelnox = comp->numresolution[0] - 1 - pi->resno;
						levelnoy = comp->numresolution[1] - 1 - pi->resno;
						levelnoz = comp->numresolution[2] - 1 - pi->resno;
						trx0 = int_ceildiv(pi->tx0, comp->dx << levelnox);
						try0 = int_ceildiv(pi->ty0, comp->dy << levelnoy);
						trz0 = int_ceildiv(pi->tz0, comp->dz << levelnoz);
						trx1 = int_ceildiv(pi->tx1, comp->dx << levelnox);
						try1 = int_ceildiv(pi->ty1, comp->dy << levelnoy);
						trz1 = int_ceildiv(pi->tz1, comp->dz << levelnoz);
						rpx = res->pdx + levelnox;
						rpy = res->pdy + levelnoy;
						rpz = res->pdz + levelnoz;
						if ((!(pi->x % (comp->dx << rpx) == 0) || (pi->x == pi->tx0 && (trx0 << levelnox) % (1 << rpx)))) {
							continue;
						}
						if ((!(pi->y % (comp->dy << rpy) == 0) || (pi->y == pi->ty0 && (try0 << levelnoy) % (1 << rpx)))) {
							continue;
						}
						if ((!(pi->z % (comp->dz << rpz) == 0) || (pi->z == pi->tz0 && (trz0 << levelnoz) % (1 << rpx)))) {
							continue;
						}
						if ((res->prctno[0]==0)||(res->prctno[1]==0)||(res->prctno[2]==0)) continue;
						
						if ((trx0==trx1)||(try0==try1)||(trz0==trz1)) continue;
						
						prci = int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelnox), res->pdx) 
							- int_floordivpow2(trx0, res->pdx);
						prcj = int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelnoy), res->pdy) 
							- int_floordivpow2(try0, res->pdy);
						prck = int_floordivpow2(int_ceildiv(pi->z, comp->dz << levelnoz), res->pdz) 
							- int_floordivpow2(trz0, res->pdz);
						pi->precno = prci + prcj * res->prctno[0] + prck * res->prctno[0] * res->prctno[1];
						for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
							index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
							if (!pi->include[index]) {
								pi->include[index] = 1;
								return true;
							}	
LABEL_SKIP:;
					}
				}
			}
		}
	}
}
	
	return false;
}

static bool pi_next_cprl(opj_pi_iterator_t * pi) {
	opj_pi_comp_t *comp = NULL;
	opj_pi_resolution_t *res = NULL;
	long index = 0;

	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		goto LABEL_SKIP;
	} else {
		pi->first = 0;
	}

	for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
		int resno;
		comp = &pi->comps[pi->compno];
		pi->dx = 0;
		pi->dy = 0;
		for (resno = 0; resno < comp->numresolution[0]; resno++) {
			int dx, dy;
			res = &comp->resolutions[resno];
			dx = comp->dx * (1 << (res->pdx + comp->numresolution[0] - 1 - resno));
			dy = comp->dy * (1 << (res->pdy + comp->numresolution[0] - 1 - resno));
			pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
			pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
		}
	for (pi->z = pi->tz0; pi->z < pi->tz1; pi->z += pi->dz - (pi->z % pi->dz)) {
		for (pi->y = pi->ty0; pi->y < pi->ty1; pi->y += pi->dy - (pi->y % pi->dy)) {
			for (pi->x = pi->tx0; pi->x < pi->tx1; pi->x += pi->dx - (pi->x % pi->dx)) {
				for (pi->resno = pi->poc.resno0; pi->resno < int_min(pi->poc.resno1, comp->numresolution[0]); pi->resno++) {
						int levelnox, levelnoy, levelnoz;
						int trx0, try0, trz0;
						int trx1, try1, trz1;
						int rpx, rpy, rpz;
						int prci, prcj, prck;
						comp = &pi->comps[pi->compno];
						if (pi->resno >= comp->numresolution[0]) {
							continue;
						}
						res = &comp->resolutions[pi->resno];
						levelnox = comp->numresolution[0] - 1 - pi->resno;
						levelnoy = comp->numresolution[1] - 1 - pi->resno;
						levelnoz = comp->numresolution[2] - 1 - pi->resno;
						trx0 = int_ceildiv(pi->tx0, comp->dx << levelnox);
						try0 = int_ceildiv(pi->ty0, comp->dy << levelnoy);
						trz0 = int_ceildiv(pi->tz0, comp->dz << levelnoz);
						trx1 = int_ceildiv(pi->tx1, comp->dx << levelnox);
						try1 = int_ceildiv(pi->ty1, comp->dy << levelnoy);
						trz1 = int_ceildiv(pi->tz1, comp->dz << levelnoz);
						rpx = res->pdx + levelnox;
						rpy = res->pdy + levelnoy;
						rpz = res->pdz + levelnoz;
						if ((!(pi->x % (comp->dx << rpx) == 0) || (pi->x == pi->tx0 && (trx0 << levelnox) % (1 << rpx)))) {
							continue;
						}
						if ((!(pi->y % (comp->dy << rpy) == 0) || (pi->y == pi->ty0 && (try0 << levelnoy) % (1 << rpx)))) {
							continue;
						}
						if ((!(pi->z % (comp->dz << rpz) == 0) || (pi->z == pi->tz0 && (trz0 << levelnoz) % (1 << rpx)))) {
							continue;
						}
						if ((res->prctno[0]==0)||(res->prctno[1]==0)||(res->prctno[2]==0)) continue;
						
						if ((trx0==trx1)||(try0==try1)||(trz0==trz1)) continue;
						
						prci = int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelnox), res->pdx) 
							- int_floordivpow2(trx0, res->pdx);
						prcj = int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelnoy), res->pdy) 
							- int_floordivpow2(try0, res->pdy);
						prck = int_floordivpow2(int_ceildiv(pi->z, comp->dz << levelnoz), res->pdz) 
							- int_floordivpow2(trz0, res->pdz);
						pi->precno = prci + prcj * res->prctno[0] + prck * res->prctno[0] * res->prctno[1];
						for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
							index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
							if (!pi->include[index]) {
								pi->include[index] = 1;
								return true;
							}
	LABEL_SKIP:;
						}
					}
				}
			}
		}
	}
	
	return false;
}

/* 
==========================================================
   Packet iterator interface
==========================================================
*/

opj_pi_iterator_t *pi_create(opj_volume_t *volume, opj_cp_t *cp, int tileno) {
	int p, q, r;
	int compno, resno, pino;
	opj_pi_iterator_t *pi = NULL;
	opj_tcp_t *tcp = NULL;
	opj_tccp_t *tccp = NULL;
	size_t array_size;
	
	tcp = &cp->tcps[tileno];

	array_size = (tcp->numpocs + 1) * sizeof(opj_pi_iterator_t);
	pi = (opj_pi_iterator_t *) opj_malloc(array_size);
	if(!pi) {
		fprintf(stdout,"[ERROR] Malloc of opj_pi_iterator failed \n");
		return NULL;
	}
	
	for (pino = 0; pino < tcp->numpocs + 1; pino++) {	/* change */
		int maxres = 0;
		int maxprec = 0;
		p = tileno % cp->tw;
		q = tileno / cp->tw;
		r = tileno / (cp->tw * cp->th);

		pi[pino].tx0 = int_max(cp->tx0 + p * cp->tdx, volume->x0);
		pi[pino].ty0 = int_max(cp->ty0 + q * cp->tdy, volume->y0);
		pi[pino].tz0 = int_max(cp->tz0 + r * cp->tdz, volume->z0);
		pi[pino].tx1 = int_min(cp->tx0 + (p + 1) * cp->tdx, volume->x1);
		pi[pino].ty1 = int_min(cp->ty0 + (q + 1) * cp->tdy, volume->y1);
		pi[pino].tz1 = int_min(cp->tz0 + (r + 1) * cp->tdz, volume->z1);
		pi[pino].numcomps = volume->numcomps;

		array_size = volume->numcomps * sizeof(opj_pi_comp_t);
		pi[pino].comps = (opj_pi_comp_t *) opj_malloc(array_size);
		if(!pi[pino].comps) {
			fprintf(stdout,"[ERROR] Malloc of opj_pi_comp failed \n");
			pi_destroy(pi, cp, tileno);
			return NULL;
		}
		memset(pi[pino].comps, 0, array_size);
		
		for (compno = 0; compno < pi->numcomps; compno++) {
			int tcx0, tcx1, tcy0, tcy1, tcz0, tcz1;
			int i;
			opj_pi_comp_t *comp = &pi[pino].comps[compno];
			tccp = &tcp->tccps[compno];
			
			comp->dx = volume->comps[compno].dx;
			comp->dy = volume->comps[compno].dy;
			comp->dz = volume->comps[compno].dz;
			for (i = 0; i < 3; i++) {
				comp->numresolution[i] = tccp->numresolution[i];
				if (comp->numresolution[i] > maxres) {
					maxres = comp->numresolution[i];
				}
			}
			array_size = comp->numresolution[0] * sizeof(opj_pi_resolution_t);
			comp->resolutions =	(opj_pi_resolution_t *) opj_malloc(array_size);
			if(!comp->resolutions) {
				fprintf(stdout,"[ERROR] Malloc of opj_pi_resolution failed \n");
				pi_destroy(pi, cp, tileno);
				return NULL;
			}

			tcx0 = int_ceildiv(pi->tx0, comp->dx);
			tcy0 = int_ceildiv(pi->ty0, comp->dy);
			tcz0 = int_ceildiv(pi->tz0, comp->dz);
			tcx1 = int_ceildiv(pi->tx1, comp->dx);
			tcy1 = int_ceildiv(pi->ty1, comp->dy);
			tcz1 = int_ceildiv(pi->tz1, comp->dz);
			
			for (resno = 0; resno < comp->numresolution[0]; resno++) {
				int levelnox, levelnoy, levelnoz, diff;
				int rx0, ry0, rz0, rx1, ry1, rz1;
				int px0, py0, pz0, px1, py1, pz1;
				opj_pi_resolution_t *res = &comp->resolutions[resno];
				if (tccp->csty & J3D_CCP_CSTY_PRT) {
					res->pdx = tccp->prctsiz[0][resno];
					res->pdy = tccp->prctsiz[1][resno];
					res->pdz = tccp->prctsiz[2][resno];
				} else {
					res->pdx = 15;
					res->pdy = 15;
					res->pdz = 15;
				}
				levelnox = comp->numresolution[0] - 1 - resno;
				levelnoy = comp->numresolution[1] - 1 - resno;
                levelnoz = comp->numresolution[2] - 1 - resno;
				if (levelnoz < 0) levelnoz = 0; 
				diff = comp->numresolution[0] - comp->numresolution[2];

				rx0 = int_ceildivpow2(tcx0, levelnox);
				ry0 = int_ceildivpow2(tcy0, levelnoy);
				rz0 = int_ceildivpow2(tcz0, levelnoz);
				rx1 = int_ceildivpow2(tcx1, levelnox);
				ry1 = int_ceildivpow2(tcy1, levelnoy);
				rz1 = int_ceildivpow2(tcz1, levelnoz);
				px0 = int_floordivpow2(rx0, res->pdx) << res->pdx;
				py0 = int_floordivpow2(ry0, res->pdy) << res->pdy;
				pz0 = int_floordivpow2(rz0, res->pdz) << res->pdz;
				px1 = int_ceildivpow2(rx1, res->pdx) << res->pdx;
				py1 = int_ceildivpow2(ry1, res->pdy) << res->pdy;
				pz1 = int_ceildivpow2(rz1, res->pdz) << res->pdz;
				res->prctno[0] = (rx0==rx1)? 0 : ((px1 - px0) >> res->pdx);
				res->prctno[1] = (ry0==ry1)? 0 : ((py1 - py0) >> res->pdy);
				res->prctno[2] = (rz0==rz1)? 0 : ((pz1 - pz0) >> res->pdz);

				if (res->prctno[0]*res->prctno[1]*res->prctno[2] > maxprec) {
					maxprec = res->prctno[0]*res->prctno[1]*res->prctno[2];
				}
			}
		}
		
		tccp = &tcp->tccps[0];
		pi[pino].step_p = 1;
		pi[pino].step_c = maxprec * pi[pino].step_p;
		pi[pino].step_r = volume->numcomps * pi[pino].step_c;
		pi[pino].step_l = maxres * pi[pino].step_r;
		
		if (pino == 0) {
			array_size = volume->numcomps * maxres * tcp->numlayers * maxprec * sizeof(short int);
			pi[pino].include = (short int *) opj_malloc(array_size);
			if(!pi[pino].include) {
				fprintf(stdout,"[ERROR] Malloc of pi[pino].include failed \n");
				pi_destroy(pi, cp, tileno);
				return NULL;
			}
		}
		else {
			pi[pino].include = pi[pino - 1].include;
		}
		
		if (tcp->POC == 0) {
			pi[pino].first = 1;
			pi[pino].poc.resno0 = 0;
			pi[pino].poc.compno0 = 0;
			pi[pino].poc.layno1 = tcp->numlayers;
			pi[pino].poc.resno1 = maxres;
			pi[pino].poc.compno1 = volume->numcomps;
			pi[pino].poc.prg = tcp->prg;
		} else {
			pi[pino].first = 1;
			pi[pino].poc.resno0 = tcp->pocs[pino].resno0;
			pi[pino].poc.compno0 = tcp->pocs[pino].compno0;
			pi[pino].poc.layno1 = tcp->pocs[pino].layno1;
			pi[pino].poc.resno1 = tcp->pocs[pino].resno1;
			pi[pino].poc.compno1 = tcp->pocs[pino].compno1;
			pi[pino].poc.prg = tcp->pocs[pino].prg;
		}
	}
	
	return pi;
}

void pi_destroy(opj_pi_iterator_t *pi, opj_cp_t *cp, int tileno) {
	int compno, pino;
	opj_tcp_t *tcp = &cp->tcps[tileno];
	if(pi) {
		for (pino = 0; pino < tcp->numpocs + 1; pino++) {	
			if(pi[pino].comps) {
				for (compno = 0; compno < pi->numcomps; compno++) {
					opj_pi_comp_t *comp = &pi[pino].comps[compno];
					if(comp->resolutions) {
						opj_free(comp->resolutions);
					}
				}
				opj_free(pi[pino].comps);
			}
		}
		if(pi->include) {
			opj_free(pi->include);
		}
		opj_free(pi);
	}
}

bool pi_next(opj_pi_iterator_t * pi) {
	switch (pi->poc.prg) {
		case LRCP:
			return pi_next_lrcp(pi);
		case RLCP:
			return pi_next_rlcp(pi);
		case RPCL:
			return pi_next_rpcl(pi);
		case PCRL:
			return pi_next_pcrl(pi);
		case CPRL:
			return pi_next_cprl(pi);
	}
	
	return false;
}

