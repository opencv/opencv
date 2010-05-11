/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2002 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 * 
 * JasPer License Version 2.0
 * 
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 * 
 * All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 * 
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 * 
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 * 
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 * 
 * __END_OF_JASPER_LICENSE__
 */

/*
 * MQ Arithmetic Coder
 *
 * $Id: jpc_mqcod.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include "jasper/jas_malloc.h"

#include "jpc_mqcod.h"

/******************************************************************************\
* Data.
\******************************************************************************/

/* MQ coder per-state information. */

jpc_mqstate_t jpc_mqstates[47 * 2] = {
	{0x5601, 0, &jpc_mqstates[ 2], &jpc_mqstates[ 3]},
	{0x5601, 1, &jpc_mqstates[ 3], &jpc_mqstates[ 2]},
	{0x3401, 0, &jpc_mqstates[ 4], &jpc_mqstates[12]},
	{0x3401, 1, &jpc_mqstates[ 5], &jpc_mqstates[13]},
	{0x1801, 0, &jpc_mqstates[ 6], &jpc_mqstates[18]},
	{0x1801, 1, &jpc_mqstates[ 7], &jpc_mqstates[19]},
	{0x0ac1, 0, &jpc_mqstates[ 8], &jpc_mqstates[24]},
	{0x0ac1, 1, &jpc_mqstates[ 9], &jpc_mqstates[25]},
	{0x0521, 0, &jpc_mqstates[10], &jpc_mqstates[58]},
	{0x0521, 1, &jpc_mqstates[11], &jpc_mqstates[59]},
	{0x0221, 0, &jpc_mqstates[76], &jpc_mqstates[66]},
	{0x0221, 1, &jpc_mqstates[77], &jpc_mqstates[67]},
	{0x5601, 0, &jpc_mqstates[14], &jpc_mqstates[13]},
	{0x5601, 1, &jpc_mqstates[15], &jpc_mqstates[12]},
	{0x5401, 0, &jpc_mqstates[16], &jpc_mqstates[28]},
	{0x5401, 1, &jpc_mqstates[17], &jpc_mqstates[29]},
	{0x4801, 0, &jpc_mqstates[18], &jpc_mqstates[28]},
	{0x4801, 1, &jpc_mqstates[19], &jpc_mqstates[29]},
	{0x3801, 0, &jpc_mqstates[20], &jpc_mqstates[28]},
	{0x3801, 1, &jpc_mqstates[21], &jpc_mqstates[29]},
	{0x3001, 0, &jpc_mqstates[22], &jpc_mqstates[34]},
	{0x3001, 1, &jpc_mqstates[23], &jpc_mqstates[35]},
	{0x2401, 0, &jpc_mqstates[24], &jpc_mqstates[36]},
	{0x2401, 1, &jpc_mqstates[25], &jpc_mqstates[37]},
	{0x1c01, 0, &jpc_mqstates[26], &jpc_mqstates[40]},
	{0x1c01, 1, &jpc_mqstates[27], &jpc_mqstates[41]},
	{0x1601, 0, &jpc_mqstates[58], &jpc_mqstates[42]},
	{0x1601, 1, &jpc_mqstates[59], &jpc_mqstates[43]},
	{0x5601, 0, &jpc_mqstates[30], &jpc_mqstates[29]},
	{0x5601, 1, &jpc_mqstates[31], &jpc_mqstates[28]},
	{0x5401, 0, &jpc_mqstates[32], &jpc_mqstates[28]},
	{0x5401, 1, &jpc_mqstates[33], &jpc_mqstates[29]},
	{0x5101, 0, &jpc_mqstates[34], &jpc_mqstates[30]},
	{0x5101, 1, &jpc_mqstates[35], &jpc_mqstates[31]},
	{0x4801, 0, &jpc_mqstates[36], &jpc_mqstates[32]},
	{0x4801, 1, &jpc_mqstates[37], &jpc_mqstates[33]},
	{0x3801, 0, &jpc_mqstates[38], &jpc_mqstates[34]},
	{0x3801, 1, &jpc_mqstates[39], &jpc_mqstates[35]},
	{0x3401, 0, &jpc_mqstates[40], &jpc_mqstates[36]},
	{0x3401, 1, &jpc_mqstates[41], &jpc_mqstates[37]},
	{0x3001, 0, &jpc_mqstates[42], &jpc_mqstates[38]},
	{0x3001, 1, &jpc_mqstates[43], &jpc_mqstates[39]},
	{0x2801, 0, &jpc_mqstates[44], &jpc_mqstates[38]},
	{0x2801, 1, &jpc_mqstates[45], &jpc_mqstates[39]},
	{0x2401, 0, &jpc_mqstates[46], &jpc_mqstates[40]},
	{0x2401, 1, &jpc_mqstates[47], &jpc_mqstates[41]},
	{0x2201, 0, &jpc_mqstates[48], &jpc_mqstates[42]},
	{0x2201, 1, &jpc_mqstates[49], &jpc_mqstates[43]},
	{0x1c01, 0, &jpc_mqstates[50], &jpc_mqstates[44]},
	{0x1c01, 1, &jpc_mqstates[51], &jpc_mqstates[45]},
	{0x1801, 0, &jpc_mqstates[52], &jpc_mqstates[46]},
	{0x1801, 1, &jpc_mqstates[53], &jpc_mqstates[47]},
	{0x1601, 0, &jpc_mqstates[54], &jpc_mqstates[48]},
	{0x1601, 1, &jpc_mqstates[55], &jpc_mqstates[49]},
	{0x1401, 0, &jpc_mqstates[56], &jpc_mqstates[50]},
	{0x1401, 1, &jpc_mqstates[57], &jpc_mqstates[51]},
	{0x1201, 0, &jpc_mqstates[58], &jpc_mqstates[52]},
	{0x1201, 1, &jpc_mqstates[59], &jpc_mqstates[53]},
	{0x1101, 0, &jpc_mqstates[60], &jpc_mqstates[54]},
	{0x1101, 1, &jpc_mqstates[61], &jpc_mqstates[55]},
	{0x0ac1, 0, &jpc_mqstates[62], &jpc_mqstates[56]},
	{0x0ac1, 1, &jpc_mqstates[63], &jpc_mqstates[57]},
	{0x09c1, 0, &jpc_mqstates[64], &jpc_mqstates[58]},
	{0x09c1, 1, &jpc_mqstates[65], &jpc_mqstates[59]},
	{0x08a1, 0, &jpc_mqstates[66], &jpc_mqstates[60]},
	{0x08a1, 1, &jpc_mqstates[67], &jpc_mqstates[61]},
	{0x0521, 0, &jpc_mqstates[68], &jpc_mqstates[62]},
	{0x0521, 1, &jpc_mqstates[69], &jpc_mqstates[63]},
	{0x0441, 0, &jpc_mqstates[70], &jpc_mqstates[64]},
	{0x0441, 1, &jpc_mqstates[71], &jpc_mqstates[65]},
	{0x02a1, 0, &jpc_mqstates[72], &jpc_mqstates[66]},
	{0x02a1, 1, &jpc_mqstates[73], &jpc_mqstates[67]},
	{0x0221, 0, &jpc_mqstates[74], &jpc_mqstates[68]},
	{0x0221, 1, &jpc_mqstates[75], &jpc_mqstates[69]},
	{0x0141, 0, &jpc_mqstates[76], &jpc_mqstates[70]},
	{0x0141, 1, &jpc_mqstates[77], &jpc_mqstates[71]},
	{0x0111, 0, &jpc_mqstates[78], &jpc_mqstates[72]},
	{0x0111, 1, &jpc_mqstates[79], &jpc_mqstates[73]},
	{0x0085, 0, &jpc_mqstates[80], &jpc_mqstates[74]},
	{0x0085, 1, &jpc_mqstates[81], &jpc_mqstates[75]},
	{0x0049, 0, &jpc_mqstates[82], &jpc_mqstates[76]},
	{0x0049, 1, &jpc_mqstates[83], &jpc_mqstates[77]},
	{0x0025, 0, &jpc_mqstates[84], &jpc_mqstates[78]},
	{0x0025, 1, &jpc_mqstates[85], &jpc_mqstates[79]},
	{0x0015, 0, &jpc_mqstates[86], &jpc_mqstates[80]},
	{0x0015, 1, &jpc_mqstates[87], &jpc_mqstates[81]},
	{0x0009, 0, &jpc_mqstates[88], &jpc_mqstates[82]},
	{0x0009, 1, &jpc_mqstates[89], &jpc_mqstates[83]},
	{0x0005, 0, &jpc_mqstates[90], &jpc_mqstates[84]},
	{0x0005, 1, &jpc_mqstates[91], &jpc_mqstates[85]},
	{0x0001, 0, &jpc_mqstates[90], &jpc_mqstates[86]},
	{0x0001, 1, &jpc_mqstates[91], &jpc_mqstates[87]},
	{0x5601, 0, &jpc_mqstates[92], &jpc_mqstates[92]},
	{0x5601, 1, &jpc_mqstates[93], &jpc_mqstates[93]},
};
