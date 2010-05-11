/*
 * Copyright (c) 1999-2000, Image Power, Inc. and the University of
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
 * Command Line Option Parsing Library
 *
 * $Id: jas_getopt.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <string.h>

#include "jasper/jas_getopt.h"
#include "jasper/jas_math.h"

/******************************************************************************\
* Global data.
\******************************************************************************/

int jas_optind = 0;
int jas_opterr = 1;
char *jas_optarg = 0;

/******************************************************************************\
* Code.
\******************************************************************************/

static jas_opt_t *jas_optlookup(jas_opt_t *opts, char *name)
{
	jas_opt_t *opt;

	for (opt = opts; opt->id >= 0 && opt->name; ++opt) {
		if (!strcmp(opt->name, name)) {
			return opt;
		}
	}
	return 0;
}

int jas_getopt(int argc, char **argv, jas_opt_t *opts)
{
	char *cp;
	int id;
	int hasarg;
	jas_opt_t *opt;
	char *s;

	if (!jas_optind) {
		jas_optind = JAS_MIN(1, argc);
	}
	while (jas_optind < argc) {
		s = cp = argv[jas_optind];
		if (*cp == '-') {
			/* We are processing an option. */
			++jas_optind;
			if (*++cp == '-') {
				/* We are processing a long option. */
				++cp;
				if (*cp == '\0') {
					/* This is the end of the options. */
					return JAS_GETOPT_EOF;
				}
				if (!(opt = jas_optlookup(opts, cp))) {
					if (jas_opterr) {
						jas_eprintf("unknown long option %s\n", s);
					}
					return JAS_GETOPT_ERR;
				}
				hasarg = (opt->flags & JAS_OPT_HASARG) != 0;
				id = opt->id;
			} else {
				/* We are processing a short option. */
				if (strlen(cp) != 1 ||
				  !(opt = jas_optlookup(opts, cp))) {
					if (jas_opterr) {
						jas_eprintf("unknown short option %s\n", s);
					}
					return JAS_GETOPT_ERR;
				}
				hasarg = (opt->flags & JAS_OPT_HASARG) != 0;
				id = opt->id;
			}
			if (hasarg) {
				/* The option has an argument. */
				if (jas_optind >= argc) {
					if (jas_opterr) {
						jas_eprintf("missing argument for option %s\n", s);
					}
					return JAS_GETOPT_ERR;
				}
				jas_optarg = argv[jas_optind];
				++jas_optind;
			} else {
				/* The option does not have an argument. */
				jas_optarg = 0;
			}
			return id;
		} else {
			/* We are not processing an option. */
			return JAS_GETOPT_EOF;
		}
	}
	return JAS_GETOPT_EOF;
}
