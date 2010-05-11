/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdlib.h>
#include <stdio.h>


namespace flann
{

/**
 * Allocates (using C's malloc) a generic type T.
 *
 * Params:
 *     count = number of instances to allocate.
 * Returns: pointer (of type T*) to memory buffer
 */
template <typename T>
T* allocate(size_t count = 1)
{
	T* mem = (T*) ::malloc(sizeof(T)*count);
	return mem;
}


/**
 * Pooled storage allocator
 *
 * The following routines allow for the efficient allocation of storage in
 * small chunks from a specified pool.  Rather than allowing each structure
 * to be freed individually, an entire pool of storage is freed at once.
 * This method has two advantages over just using malloc() and free().  First,
 * it is far more efficient for allocating small objects, as there is
 * no overhead for remembering all the information needed to free each
 * object or consolidating fragmented memory.  Second, the decision about
 * how long to keep an object is made at the time of allocation, and there
 * is no need to track down all the objects to free them.
 *
 */

const size_t     WORDSIZE=16;
const  size_t     BLOCKSIZE=8192;

class PooledAllocator
{
	/* We maintain memory alignment to word boundaries by requiring that all
		allocations be in multiples of the machine wordsize.  */
	  /* Size of machine word in bytes.  Must be power of 2. */
	/* Minimum number of bytes requested at a time from	the system.  Must be multiple of WORDSIZE. */


	int 	remaining;  /* Number of bytes left in current block of storage. */
	void*	base;     /* Pointer to base of current block of storage. */
	void*	loc;      /* Current location in block to next allocate memory. */
	int 	blocksize;


public:
	int 	usedMemory;
	int 	wastedMemory;

	/**
		Default constructor. Initializes a new pool.
	*/
	PooledAllocator(int blocksize = BLOCKSIZE)
	{
    	this->blocksize = blocksize;
		remaining = 0;
		base = NULL;

		usedMemory = 0;
		wastedMemory = 0;
	}

	/**
	 * Destructor. Frees all the memory allocated in this pool.
	 */
 	~PooledAllocator()
	{
		void *prev;

		while (base != NULL) {
			prev = *((void **) base);  /* Get pointer to prev block. */
			::free(base);
			base = prev;
		}
	}

	/**
	 * Returns a pointer to a piece of new memory of the given size in bytes
	 * allocated from the pool.
	 */
	void* malloc(int size)
	{
		int blocksize;

		/* Round size up to a multiple of wordsize.  The following expression
			only works for WORDSIZE that is a power of 2, by masking last bits of
			incremented size to zero.
		*/
		size = (size + (WORDSIZE - 1)) & ~(WORDSIZE - 1);

		/* Check whether a new block must be allocated.  Note that the first word
			of a block is reserved for a pointer to the previous block.
		*/
		if (size > remaining) {

			wastedMemory += remaining;

		/* Allocate new storage. */
			blocksize = (size + sizeof(void*) + (WORDSIZE-1) > BLOCKSIZE) ?
						size + sizeof(void*) + (WORDSIZE-1) : BLOCKSIZE;

			// use the standard C malloc to allocate memory
			void* m = ::malloc(blocksize);
			if (!m) {
                fprintf(stderr,"Failed to allocate memory.");
                exit(1);
			}

			/* Fill first word of new block with pointer to previous block. */
			((void **) m)[0] = base;
			base = m;

			int shift = 0;
			//int shift = (WORDSIZE - ( (((size_t)m) + sizeof(void*)) & (WORDSIZE-1))) & (WORDSIZE-1);

			remaining = blocksize - sizeof(void*) - shift;
			loc = ((char*)m + sizeof(void*) + shift);
		}
		void* rloc = loc;
		loc = (char*)loc + size;
		remaining -= size;

		usedMemory += size;

		return rloc;
	}

	/**
	 * Allocates (using this pool) a generic type T.
	 *
	 * Params:
	 *     count = number of instances to allocate.
	 * Returns: pointer (of type T*) to memory buffer
	 */
    template <typename T>
	T* allocate(size_t count = 1)
	{
		T* mem = (T*) this->malloc(sizeof(T)*count);
		return mem;
	}

};

}

#endif //ALLOCATOR_H
