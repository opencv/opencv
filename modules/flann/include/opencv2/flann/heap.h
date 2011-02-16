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

#ifndef _OPENCV_HEAP_H_
#define _OPENCV_HEAP_H_


#include <algorithm>

namespace cvflann
{

/**
 * Priority Queue Implementation
 *
 * The priority queue is implemented with a heap.  A heap is a complete
 * (full) binary tree in which each parent is less than both of its
 * children, but the order of the children is unspecified.
 * Note that a heap uses 1-based indexing to allow for power-of-2
 * location of parents and children.  We ignore element 0 of Heap array.
 */
template <typename T>
class Heap {

	/**
	* Storage array for the heap.
	* Type T must be comparable.
	*/
	T* heap;
    int length;

	/**
	 * Number of element in the heap
	 */
	int count;



public:
	/**
	 * Constructor.
	 *
	 * Params:
	 *     size = heap size
	 */

	Heap(int size)
	{
        length = size+1;
		heap = new T[length];  // heap uses 1-based indexing
		count = 0;
	}


	/**
	 * Destructor.
	 *
	 */
	~Heap()
	{
		delete[] heap;
	}

	/**
	 *
	 * Returns: heap size
	 */
	int size()
	{
		return count;
	}

	/**
	 * Tests if the heap is empty
	 *
	 * Returns: true is heap empty, false otherwise
	 */
	bool empty()
	{
		return size()==0;
	}

	/**
	 * Clears the heap.
	 */
	void clear()
	{
		count = 0;
	}


	/**
	 * Insert a new element in the heap.
	 *
	 * We select the next empty leaf node, and then keep moving any larger
	 * parents down until the right location is found to store this element.
	 *
	 * Params:
	 *     value = the new element to be inserted in the heap
	 */
	void insert(T value)
	{
		/* If heap is full, then return without adding this element. */
		if (count == length-1) {
			return;
		}

		int loc = ++(count);   /* Remember 1-based indexing. */

		/* Keep moving parents down until a place is found for this node. */
		int par = loc / 2;                 /* Location of parent. */
		while (par > 0  && value < heap[par]) {
			heap[loc] = heap[par];     /* Move parent down to loc. */
			loc = par;
			par = loc / 2;
		}
		/* Insert the element at the determined location. */
		heap[loc] = value;
	}



	/**
 	 * Returns the node of minimum value from the heap (top of the heap).
	 *
	 * Params:
	 *     value = out parameter used to return the min element
	 * Returns: false if heap empty
	 */
	bool popMin(T& value)
	{
		if (count == 0) {
 			return false;
		}

		/* Switch first node with last. */
        std::swap(heap[1],heap[count]);

		count -= 1;
		heapify(1);      /* Move new node 1 to right position. */

		value = heap[count + 1];
		return true;  /* Return old last node. */
	}


	/**
	 * Reorganizes the heap (a parent is smaller than its children)
	 * starting with a node.
	 *
	 * Params:
	 *     parent = node form which to start heap reorganization.
	 */
	void heapify(int parent)
	{
		int minloc = parent;

		/* Check the left child */
		int left = 2 * parent;
		if (left <= count && heap[left] < heap[parent]) {
			minloc = left;
		}

		/* Check the right child */
		int right = left + 1;
		if (right <= count && heap[right] < heap[minloc]) {
			minloc = right;
		}

		/* If a child was smaller, than swap parent with it and Heapify. */
		if (minloc != parent) {
            std::swap(heap[parent],heap[minloc]);
			heapify(minloc);
		}
	}

};

} // namespace cvflann

#endif //_OPENCV_HEAP_H_
