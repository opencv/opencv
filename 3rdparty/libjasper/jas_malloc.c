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
 * Memory Allocator
 *
 * $Id: jas_malloc.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

/* We need the prototype for memset. */
#include <string.h>
#include <limits.h>
#include <errno.h>
#if !defined _WIN32 || defined __MINGW__ || defined __MINGW32__
#include <stdint.h>
#endif

#include "jasper/jas_malloc.h"

/******************************************************************************\
* Code.
\******************************************************************************/

#if defined(DEBUG_MEMALLOC)
#include "../../../local/src/memalloc.c"
#endif

#if !defined(DEBUG_MEMALLOC)

#define MEMALLOC_ALIGNMENT	32
#define MEMALLOC_ALIGN2
#undef MEMALLOC_ALIGN2

void *jas_malloc(size_t size)
{
#if defined(MEMALLOC_ALIGN2)
    void *ptr;
abort();
    if (posix_memalign(&ptr, MEMALLOC_ALIGNMENT, size)) {
        return 0;
    }
    return ptr;
#endif
    return malloc(size);
}

void jas_free(void *ptr)
{
    free(ptr);
}

void *jas_realloc(void *ptr, size_t size)
{
    return ptr ? realloc(ptr, size) : malloc(size);
}

void *jas_realloc2(void *ptr, size_t nmemb, size_t size)
{
    if (!ptr)
        return jas_alloc2(nmemb, size);
    if (nmemb && SIZE_MAX / nmemb < size) {
        errno = ENOMEM;
        return NULL;
    }
    return jas_realloc(ptr, nmemb * size);

}

void *jas_alloc2(size_t nmemb, size_t size)
{
    if (nmemb && SIZE_MAX / nmemb < size) {
        errno = ENOMEM;
        return NULL;
    }

    return jas_malloc(nmemb * size);
}

void *jas_alloc3(size_t a, size_t b, size_t c)
{
    size_t n;

    if (a && SIZE_MAX / a < b) {
        errno = ENOMEM;
        return NULL;
    }

    return jas_alloc2(a*b, c);
}

void *jas_calloc(size_t nmemb, size_t size)
{
    void *ptr;

    ptr = jas_alloc2(nmemb, size);
    if (ptr)
        memset(ptr, 0, nmemb*size);
    return ptr;
}

#endif
