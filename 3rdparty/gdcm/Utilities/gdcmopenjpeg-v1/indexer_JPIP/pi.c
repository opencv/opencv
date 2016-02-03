/*
 * Copyright (c) 2001-2002, David Janssens
 * Copyright (c) 2003-2004, Yannick Verschueren
 * Copyright (c) 2003-2004, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#include "pi.h"
#include "int.h"
#include <stdlib.h>
#include <stdio.h>


/* <summary> */
/* Create a packet iterator.   */
/* </summary> */
pi_iterator_t *pi_create(j2k_image_t * img, j2k_cp_t * cp, int tileno)
{
	int p, q;
	int compno, resno, pino;
	int maxres = 0;
	pi_iterator_t *pi;
	j2k_tcp_t *tcp;
	j2k_tccp_t *tccp;

	tcp = &cp->tcps[tileno];
	pi = (pi_iterator_t *) malloc((tcp->numpocs + 1) * sizeof(pi_iterator_t));

	for (pino = 0; pino < tcp->numpocs + 1; pino++) {	/* change */
		p = tileno % cp->tw;
		q = tileno / cp->tw;

		pi[pino].tx0 = int_max(cp->tx0 + p * cp->tdx, img->x0);
		pi[pino].ty0 = int_max(cp->ty0 + q * cp->tdy, img->y0);
		pi[pino].tx1 = int_min(cp->tx0 + (p + 1) * cp->tdx, img->x1);
		pi[pino].ty1 = int_min(cp->ty0 + (q + 1) * cp->tdy, img->y1);
		pi[pino].numcomps = img->numcomps;
		pi[pino].comps = (pi_comp_t *) malloc(img->numcomps * sizeof(pi_comp_t));

		for (compno = 0; compno < pi->numcomps; compno++) {
			int tcx0, tcy0, tcx1, tcy1;
			pi_comp_t *comp = &pi[pino].comps[compno];
			tccp = &tcp->tccps[compno];
			comp->dx = img->comps[compno].dx;
			comp->dy = img->comps[compno].dy;
			comp->numresolutions = tccp->numresolutions;
			comp->resolutions =
				(pi_resolution_t *) malloc(comp->numresolutions *
																	 sizeof(pi_resolution_t));
			tcx0 = int_ceildiv(pi->tx0, comp->dx);
			tcy0 = int_ceildiv(pi->ty0, comp->dy);
			tcx1 = int_ceildiv(pi->tx1, comp->dx);
			tcy1 = int_ceildiv(pi->ty1, comp->dy);
			if (comp->numresolutions > maxres) {
				maxres = comp->numresolutions;
			}
			for (resno = 0; resno < comp->numresolutions; resno++) {
				int levelno;
				int rx0, ry0, rx1, ry1;
				int px0, py0, px1, py1;
				pi_resolution_t *res = &comp->resolutions[resno];
				if (tccp->csty & J2K_CCP_CSTY_PRT) {
					res->pdx = tccp->prcw[resno];
					res->pdy = tccp->prch[resno];
				} else {
					res->pdx = 15;
					res->pdy = 15;
				}
				levelno = comp->numresolutions - 1 - resno;
				rx0 = int_ceildivpow2(tcx0, levelno);
				ry0 = int_ceildivpow2(tcy0, levelno);
				rx1 = int_ceildivpow2(tcx1, levelno);
				ry1 = int_ceildivpow2(tcy1, levelno);
				px0 = int_floordivpow2(rx0, res->pdx) << res->pdx;
				py0 = int_floordivpow2(ry0, res->pdy) << res->pdy;
				px1 = int_ceildivpow2(rx1, res->pdx) << res->pdx;
				py1 = int_ceildivpow2(ry1, res->pdy) << res->pdy;
				res->pw = (px1 - px0) >> res->pdx;
				res->ph = (py1 - py0) >> res->pdy;
			}
		}
		
		tccp = &tcp->tccps[0];
		pi[pino].step_p=1;
		pi[pino].step_c=100*pi[pino].step_p;
		pi[pino].step_r=img->numcomps*pi[pino].step_c;
		pi[pino].step_l=maxres*pi[pino].step_r;
		
		if (pino==0)
		  pi[pino].include=(short int*)calloc(img->numcomps*maxres*tcp->numlayers*100,sizeof(short int));
		else
		  pi[pino].include=pi[pino-1].include;

		/*if (pino == tcp->numpocs) {*/
		  if (tcp->POC == 0) {
			pi[pino].first = 1;
			pi[pino].poc.resno0 = 0;
			pi[pino].poc.compno0 = 0;
			pi[pino].poc.layno1 = tcp->numlayers;
			pi[pino].poc.resno1 = maxres;
			pi[pino].poc.compno1 = img->numcomps;
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

/* <summary> */
/* Get next packet in layer=resolution-component-precinct order.   */
/* </summary> */
int pi_next_lrcp(pi_iterator_t * pi)
{
	pi_comp_t *comp;
	pi_resolution_t *res;

	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		res = &comp->resolutions[pi->resno];
		goto skip;
	} else {
		pi->first = 0;
	}
	for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
		for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1;
				 pi->resno++) {
			for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1;
					 pi->compno++) {
				comp = &pi->comps[pi->compno];
				if (pi->resno >= comp->numresolutions) {

					continue;
				}
				res = &comp->resolutions[pi->resno];
				for (pi->precno = 0; pi->precno < res->pw * res->ph; pi->precno++) {
				  if (!pi->include[pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p]){
				    pi->include[pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p] = 1;
				    return 1;
					}
				skip:;
				}
			}
		}
	}
	return 0;
}

/* <summary> */
/* Get next packet in resolution-layer-component-precinct order.   */
/* </summary> */
int pi_next_rlcp(pi_iterator_t * pi)
{
	pi_comp_t *comp;
	pi_resolution_t *res;
	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		res = &comp->resolutions[pi->resno];
		goto skip;
	} else {
		pi->first = 0;
	}
	for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
		for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
			for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1;
					 pi->compno++) {
				comp = &pi->comps[pi->compno];
				if (pi->resno >= comp->numresolutions) {
					continue;
				}
				res = &comp->resolutions[pi->resno];
				for (pi->precno = 0; pi->precno < res->pw * res->ph; pi->precno++) {
				  if (!pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p]){
				    pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p] = 1;
				    return 1;
				  }
				skip:;
				}
			}
		}
	}
	return 0;
}

/* <summary> */
/* Get next packet in resolution-precinct-component-layer order.   */
/* </summary> */
int pi_next_rpcl(pi_iterator_t * pi)
{
	pi_comp_t *comp;
	pi_resolution_t *res;
	if (!pi->first) {
		goto skip;
	} else {
		int compno, resno;
		pi->first = 0;
		pi->dx = 0;
		pi->dy = 0;
		for (compno = 0; compno < pi->numcomps; compno++) {
			comp = &pi->comps[compno];
			for (resno = 0; resno < comp->numresolutions; resno++) {
				int dx, dy;
				res = &comp->resolutions[resno];
				dx =
					comp->dx * (1 << (res->pdx + comp->numresolutions - 1 - resno));
				dy =
					comp->dy * (1 << (res->pdy + comp->numresolutions - 1 - resno));
				pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
				pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
			}
		}
	}
	for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
		for (pi->y = pi->ty0; pi->y < pi->ty1;
				 pi->y += pi->dy - (pi->y % pi->dy)) {
			for (pi->x = pi->tx0; pi->x < pi->tx1;
					 pi->x += pi->dx - (pi->x % pi->dx)) {
				for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1;
						 pi->compno++) {
					int levelno;
					int trx0, try0;
					int rpx, rpy;
					int prci, prcj;
					comp = &pi->comps[pi->compno];
					if (pi->resno >= comp->numresolutions) {
						continue;
					}
					res = &comp->resolutions[pi->resno];
					levelno = comp->numresolutions - 1 - pi->resno;
					trx0 = int_ceildiv(pi->tx0, comp->dx << levelno);
					try0 = int_ceildiv(pi->ty0, comp->dy << levelno);
					rpx = res->pdx + levelno;
					rpy = res->pdy + levelno;
					if (!
							(pi->x % (comp->dx << rpx) == 0
							 || (pi->x == pi->tx0 && (trx0 << levelno) % (1 << rpx)))) {
						continue;
					}
					if (!
							(pi->y % (comp->dy << rpy) == 0
							 || (pi->y == pi->ty0 && (try0 << levelno) % (1 << rpx)))) {
						continue;
					}
					prci =
						int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelno),
														 res->pdx) - int_floordivpow2(trx0, res->pdx);
					prcj =
						int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelno),
														 res->pdy) - int_floordivpow2(try0, res->pdy);
					pi->precno = prci + prcj * res->pw;
					for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
					  if (!pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p]){
					    pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p] = 1;
					    return 1;
						}
					skip:;
					}
				}
			}
		}
	}
	return 0;
}

/* <summary> */
/* Get next packet in precinct-component-resolution-layer order.   */
/* </summary> */
int pi_next_pcrl(pi_iterator_t * pi)
{
	pi_comp_t *comp;
	pi_resolution_t *res;
	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		goto skip;
	} else {
		int compno, resno;
		pi->first = 0;
		pi->dx = 0;
		pi->dy = 0;
		for (compno = 0; compno < pi->numcomps; compno++) {
			comp = &pi->comps[compno];
			for (resno = 0; resno < comp->numresolutions; resno++) {
				int dx, dy;
				res = &comp->resolutions[resno];
				dx =
					comp->dx * (1 << (res->pdx + comp->numresolutions - 1 - resno));
				dy =
					comp->dy * (1 << (res->pdy + comp->numresolutions - 1 - resno));
				pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
				pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
			}
		}
	}
	for (pi->y = pi->ty0; pi->y < pi->ty1;
			 pi->y += pi->dy - (pi->y % pi->dy)) {
		for (pi->x = pi->tx0; pi->x < pi->tx1;
				 pi->x += pi->dx - (pi->x % pi->dx)) {
			for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1;
					 pi->compno++) {
				comp = &pi->comps[pi->compno];
				for (pi->resno = pi->poc.resno0;
						 pi->resno < int_min(pi->poc.resno1, comp->numresolutions);
						 pi->resno++) {
					int levelno;
					int trx0, try0;
					int rpx, rpy;
					int prci, prcj;
					res = &comp->resolutions[pi->resno];
					levelno = comp->numresolutions - 1 - pi->resno;
					trx0 = int_ceildiv(pi->tx0, comp->dx << levelno);
					try0 = int_ceildiv(pi->ty0, comp->dy << levelno);
					rpx = res->pdx + levelno;
					rpy = res->pdy + levelno;
					if (!
							(pi->x % (comp->dx << rpx) == 0
							 || (pi->x == pi->tx0 && (trx0 << levelno) % (1 << rpx)))) {
						continue;
					}
					if (!
							(pi->y % (comp->dy << rpy) == 0
							 || (pi->y == pi->ty0 && (try0 << levelno) % (1 << rpx)))) {
						continue;
					}
					prci =
						int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelno),
														 res->pdx) - int_floordivpow2(trx0, res->pdx);
					prcj =
						int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelno),
														 res->pdy) - int_floordivpow2(try0, res->pdy);
					pi->precno = prci + prcj * res->pw;
					for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
					  if (! pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p]){
					    pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p] = 1;
							return 1;
						}
					skip:;
					}
				}
			}
		}
	}
	return 0;
}

/* <summary> */
/* Get next packet in component-precinct-resolution-layer order.   */
/* </summary> */
int pi_next_cprl(pi_iterator_t * pi)
{
	pi_comp_t *comp;
	pi_resolution_t *res;
	if (!pi->first) {
		comp = &pi->comps[pi->compno];
		goto skip;
	} else {
		pi->first = 0;
	}
	for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1;
			 pi->compno++) {
		int resno;
		comp = &pi->comps[pi->compno];
		pi->dx = 0;
		pi->dy = 0;
		for (resno = 0; resno < comp->numresolutions; resno++) {
			int dx, dy;
			res = &comp->resolutions[resno];
			dx = comp->dx * (1 << (res->pdx + comp->numresolutions - 1 - resno));
			dy = comp->dy * (1 << (res->pdy + comp->numresolutions - 1 - resno));
			pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
			pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
		}
		for (pi->y = pi->ty0; pi->y < pi->ty1;
				 pi->y += pi->dy - (pi->y % pi->dy)) {
			for (pi->x = pi->tx0; pi->x < pi->tx1;
					 pi->x += pi->dx - (pi->x % pi->dx)) {
				for (pi->resno = pi->poc.resno0;
						 pi->resno < int_min(pi->poc.resno1, comp->numresolutions);
						 pi->resno++) {
					int levelno;
					int trx0, try0;
					int rpx, rpy;
					int prci, prcj;
					res = &comp->resolutions[pi->resno];
					levelno = comp->numresolutions - 1 - pi->resno;
					trx0 = int_ceildiv(pi->tx0, comp->dx << levelno);
					try0 = int_ceildiv(pi->ty0, comp->dy << levelno);
					rpx = res->pdx + levelno;
					rpy = res->pdy + levelno;
					if (!
							(pi->x % (comp->dx << rpx) == 0
							 || (pi->x == pi->tx0 && (trx0 << levelno) % (1 << rpx)))) {
						continue;
					}
					if (!
							(pi->y % (comp->dy << rpy) == 0
							 || (pi->y == pi->ty0 && (try0 << levelno) % (1 << rpx)))) {
						continue;
					}
					prci =
						int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelno),
														 res->pdx) - int_floordivpow2(trx0, res->pdx);
					prcj =
						int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelno),
														 res->pdy) - int_floordivpow2(try0, res->pdy);
					pi->precno = prci + prcj * res->pw;
					for (pi->layno = 0; pi->layno < pi->poc.layno1; pi->layno++) {
					  if (! pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p]){
					    pi->include[pi->layno*pi->step_l+pi->resno*pi->step_r+pi->compno*pi->step_c+pi->precno*pi->step_p] = 1;
					    return 1;
						}
					skip:;
					}
				}
			}
		}
	}
	return 0;
}

/* <summary> */
/* Get next packet.   */
/* </summary> */
int pi_next(pi_iterator_t * pi)
{
	switch (pi->poc.prg) {
	case 0:
		return pi_next_lrcp(pi);
	case 1:
		return pi_next_rlcp(pi);
	case 2:
		return pi_next_rpcl(pi);
	case 3:
		return pi_next_pcrl(pi);
	case 4:
		return pi_next_cprl(pi);
	}
	return 0;
}
