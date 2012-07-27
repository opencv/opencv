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

#ifndef OPENCV_FLANN_HEAP_H_
#define OPENCV_FLANN_HEAP_H_

#include <algorithm>
#include <vector>

namespace cvflann
{

/**
 * Priority Queue Implementation
 *
 * The priority queue is implemented with a heap.  A heap is a complete
 * (full) binary tree in which each parent is less than both of its
 * children, but the order of the children is unspecified.
 */
template <typename T>
class Heap
{

    /**
     * Storage array for the heap.
     * Type T must be comparable.
     */
    std::vector<T> heap;
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
     *     sz = heap size
     */

    Heap(int sz)
    {
        length = sz;
        heap.reserve(length);
        count = 0;
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
        heap.clear();
        count = 0;
    }

    struct CompareT
    {
        bool operator()(const T& t_1, const T& t_2) const
        {
            return t_2 < t_1;
        }
    };

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
        if (count == length) {
            return;
        }

        heap.push_back(value);
        static CompareT compareT;
        std::push_heap(heap.begin(), heap.end(), compareT);
        ++count;
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

        value = heap[0];
        static CompareT compareT;
        std::pop_heap(heap.begin(), heap.end(), compareT);
        heap.pop_back();
        --count;

        return true;  /* Return old last node. */
    }
};

}

#endif //OPENCV_FLANN_HEAP_H_
