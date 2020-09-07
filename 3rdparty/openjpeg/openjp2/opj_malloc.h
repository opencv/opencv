/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2007, Callum Lerwick <seg@haxxed.com>
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
#ifndef OPJ_MALLOC_H
#define OPJ_MALLOC_H

#include <stddef.h>
/**
@file opj_malloc.h
@brief Internal functions

The functions in opj_malloc.h are internal utilities used for memory management.
*/

/** @defgroup MISC MISC - Miscellaneous internal functions */
/*@{*/

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
Allocate an uninitialized memory block
@param size Bytes to allocate
@return Returns a void pointer to the allocated space, or NULL if there is insufficient memory available
*/
void * opj_malloc(size_t size);

/**
Allocate a memory block with elements initialized to 0
@param numOfElements  Blocks to allocate
@param sizeOfElements Bytes per block to allocate
@return Returns a void pointer to the allocated space, or NULL if there is insufficient memory available
*/
void * opj_calloc(size_t numOfElements, size_t sizeOfElements);

/**
Allocate memory aligned to a 16 byte boundary
@param size Bytes to allocate
@return Returns a void pointer to the allocated space, or NULL if there is insufficient memory available
*/
void * opj_aligned_malloc(size_t size);
void * opj_aligned_realloc(void *ptr, size_t size);
void opj_aligned_free(void* ptr);

/**
Allocate memory aligned to a 32 byte boundary
@param size Bytes to allocate
@return Returns a void pointer to the allocated space, or NULL if there is insufficient memory available
*/
void * opj_aligned_32_malloc(size_t size);
void * opj_aligned_32_realloc(void *ptr, size_t size);

/**
Reallocate memory blocks.
@param m Pointer to previously allocated memory block
@param s New size in bytes
@return Returns a void pointer to the reallocated (and possibly moved) memory block
*/
void * opj_realloc(void * m, size_t s);

/**
Deallocates or frees a memory block.
@param m Previously allocated memory block to be freed
*/
void opj_free(void * m);

#if defined(__GNUC__) && !defined(OPJ_SKIP_POISON)
#pragma GCC poison malloc calloc realloc free
#endif

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_MALLOC_H */

