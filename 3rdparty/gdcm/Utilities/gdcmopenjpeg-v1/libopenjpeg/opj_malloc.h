/*
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
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
#ifndef __OPJ_MALLOC_H
#define __OPJ_MALLOC_H
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
#ifdef ALLOC_PERF_OPT
void * OPJ_CALLCONV opj_malloc(size_t size);
#else
#define opj_malloc(size) malloc(size)
#endif

/**
Allocate a memory block with elements initialized to 0
@param num Blocks to allocate
@param size Bytes per block to allocate
@return Returns a void pointer to the allocated space, or NULL if there is insufficient memory available
*/
#ifdef ALLOC_PERF_OPT
void * OPJ_CALLCONV opj_calloc(size_t _NumOfElements, size_t _SizeOfElements);
#else
#define opj_calloc(num, size) calloc(num, size)
#endif

/**
Allocate memory aligned to a 16 byte boundry
@param size Bytes to allocate
@return Returns a void pointer to the allocated space, or NULL if there is insufficient memory available
*/
/* FIXME: These should be set with cmake tests, but we're currently not requiring use of cmake */
#ifdef _WIN32
	/* Someone should tell the mingw people that their malloc.h ought to provide _mm_malloc() */
	#ifdef __GNUC__
		#include <mm_malloc.h>
		#define HAVE_MM_MALLOC
	#else /* MSVC, Intel C++ */
		#include <malloc.h>
		#ifdef _mm_malloc
			#define HAVE_MM_MALLOC
		#endif
	#endif
#else /* Not _WIN32 */
	#if defined(__sun)
		#define HAVE_MEMALIGN
	/* Linux x86_64 and OSX always align allocations to 16 bytes */
	#elif !defined(__amd64__) && !defined(__APPLE__)	
		#define HAVE_MEMALIGN
		#include <malloc.h>			
	#endif
#endif

#define opj_aligned_malloc(size) malloc(size)
#define opj_aligned_free(m) free(m)

#ifdef HAVE_MM_MALLOC
	#undef opj_aligned_malloc
	#define opj_aligned_malloc(size) _mm_malloc(size, 16)
	#undef opj_aligned_free
	#define opj_aligned_free(m) _mm_free(m)
#endif

#ifdef HAVE_MEMALIGN
	extern void* memalign(size_t, size_t);
	#undef opj_aligned_malloc
	#define opj_aligned_malloc(size) memalign(16, (size))
	#undef opj_aligned_free
	#define opj_aligned_free(m) free(m)
#endif

#ifdef HAVE_POSIX_MEMALIGN
	#undef opj_aligned_malloc
	extern int posix_memalign(void**, size_t, size_t);

	static INLINE void* __attribute__ ((malloc)) opj_aligned_malloc(size_t size){
		void* mem = NULL;
		posix_memalign(&mem, 16, size);
		return mem;
	}
	#undef opj_aligned_free
	#define opj_aligned_free(m) free(m)
#endif

#ifdef ALLOC_PERF_OPT
	#undef opj_aligned_malloc
	#define opj_aligned_malloc(size) opj_malloc(size)
	#undef opj_aligned_free
	#define opj_aligned_free(m) opj_free(m)
#endif

/**
Reallocate memory blocks.
@param m Pointer to previously allocated memory block
@param s New size in bytes
@return Returns a void pointer to the reallocated (and possibly moved) memory block
*/
#ifdef ALLOC_PERF_OPT
void * OPJ_CALLCONV opj_realloc(void * m, size_t s);
#else
#define opj_realloc(m, s) realloc(m, s)
#endif

/**
Deallocates or frees a memory block.
@param m Previously allocated memory block to be freed
*/
#ifdef ALLOC_PERF_OPT
void OPJ_CALLCONV opj_free(void * m);
#else
#define opj_free(m) free(m)
#endif

#ifdef __GNUC__
#pragma GCC poison malloc calloc realloc free
#endif

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __OPJ_MALLOC_H */

