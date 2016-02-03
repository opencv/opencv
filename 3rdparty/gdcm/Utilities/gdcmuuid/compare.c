/* vi: set sw=4 ts=4: */
/*
 * compare.c --- compare whether or not two UUID's are the same
 *
 * Returns 0 if the two UUID's are different, and 1 if they are the same.
 *
 * Copyright (C) 1996, 1997 Theodore Ts'o.
 *
 * %Begin-Header%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, and the entire permission notice in its entirety,
 *    including the disclaimer of warranties.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ALL OF
 * WHICH ARE HEREBY DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF NOT ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 * %End-Header%
 */

#include "uuidP.h"
#include <string.h>

#define UUCMP(u1,u2) if (u1 != u2) return((u1 < u2) ? -1 : 1);

int uuid_compare(const uuid_t uu1, const uuid_t uu2)
{
  struct uuid  uuid1, uuid2;

  uuid_unpack(uu1, &uuid1);
  uuid_unpack(uu2, &uuid2);

  UUCMP(uuid1.time_low, uuid2.time_low);
  UUCMP(uuid1.time_mid, uuid2.time_mid);
  UUCMP(uuid1.time_hi_and_version, uuid2.time_hi_and_version);
  UUCMP(uuid1.clock_seq, uuid2.clock_seq);
  return memcmp(uuid1.node, uuid2.node, 6);
}
