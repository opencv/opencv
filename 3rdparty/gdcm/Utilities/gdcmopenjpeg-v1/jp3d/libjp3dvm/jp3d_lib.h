/*
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
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
#ifndef __J3D_LIB_H
#define __J3D_LIB_H
/**
@file jp3d_lib.h
@brief Internal functions

The functions in JP3D_LIB.C are internal utilities mainly used for memory management.
*/

/** @defgroup MISC MISC - Miscellaneous internal functions */
/*@{*/

/** @name Funciones generales */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
Difference in successive opj_clock() calls tells you the elapsed time
@return Returns time in seconds
*/
double opj_clock();

/**
Allocate a memory block with elements initialized to 0
@param size Bytes to allocate
@return Returns a void pointer to the allocated space, or NULL if there is insufficient memory available
*/
void* opj_malloc( size_t size );

/**
Reallocate memory blocks.
@param memblock Pointer to previously allocated memory block
@param size New size in bytes
@return Returns a void pointer to the reallocated (and possibly moved) memory block
*/
void* opj_realloc( void *memblock, size_t size );

/**
Deallocates or frees a memory block.
@param memblock Previously allocated memory block to be freed
*/
void opj_free( void *memblock );

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __J3D_LIB_H */

