/*
 * copyright (c) 2006 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file libavutil/mem.h
 * memory handling functions
 */

#ifndef AVUTIL_MEM_H
#define AVUTIL_MEM_H

#include "common.h"

#if AV_GCC_VERSION_AT_LEAST(3,1)
    #define av_malloc_attrib __attribute__((__malloc__))
#else
    #define av_malloc_attrib
#endif

#if (!defined(__ICC) || __ICC > 1100) && AV_GCC_VERSION_AT_LEAST(4,3)
    #define av_alloc_size(n) __attribute__((alloc_size(n)))
#else
    #define av_alloc_size(n)
#endif

/**
 * Allocates a block of \p size bytes with alignment suitable for all
 * memory accesses (including vectors if available on the CPU).
 * @param size Size in bytes for the memory block to be allocated.
 * @return Pointer to the allocated block, NULL if the block cannot
 * be allocated.
 * @see av_mallocz()
 */
void *av_malloc(unsigned int size) av_malloc_attrib av_alloc_size(1);

/**
 * Allocates or reallocates a block of memory.
 * If \p ptr is NULL and \p size > 0, allocates a new block. If \p
 * size is zero, frees the memory block pointed to by \p ptr.
 * @param size Size in bytes for the memory block to be allocated or
 * reallocated.
 * @param ptr Pointer to a memory block already allocated with
 * av_malloc(z)() or av_realloc() or NULL.
 * @return Pointer to a newly reallocated block or NULL if the block
 * cannot be reallocated or the function is used to free the memory block.
 * @see av_fast_realloc()
 */
void *av_realloc(void *ptr, unsigned int size) av_alloc_size(2);

/**
 * Frees a memory block which has been allocated with av_malloc(z)() or
 * av_realloc().
 * @param ptr Pointer to the memory block which should be freed.
 * @note ptr = NULL is explicitly allowed.
 * @note It is recommended that you use av_freep() instead.
 * @see av_freep()
 */
void av_free(void *ptr);

/**
 * Allocates a block of \p size bytes with alignment suitable for all
 * memory accesses (including vectors if available on the CPU) and
 * zeroes all the bytes of the block.
 * @param size Size in bytes for the memory block to be allocated.
 * @return Pointer to the allocated block, NULL if it cannot be allocated.
 * @see av_malloc()
 */
void *av_mallocz(unsigned int size) av_malloc_attrib av_alloc_size(1);

/**
 * Duplicates the string \p s.
 * @param s string to be duplicated
 * @return Pointer to a newly allocated string containing a
 * copy of \p s or NULL if the string cannot be allocated.
 */
char *av_strdup(const char *s) av_malloc_attrib;

/**
 * Frees a memory block which has been allocated with av_malloc(z)() or
 * av_realloc() and set the pointer pointing to it to NULL.
 * @param ptr Pointer to the pointer to the memory block which should
 * be freed.
 * @see av_free()
 */
void av_freep(void *ptr);

#endif /* AVUTIL_MEM_H */
