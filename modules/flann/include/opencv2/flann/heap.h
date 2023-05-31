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

//! @cond IGNORED

#include <algorithm>
#include <vector>

#include <unordered_map>

namespace cvflann
{

// TODO: Define x > y operator and use std::greater<T> instead
template <typename T>
struct greater
{
    bool operator()(const T& x, const T& y) const
    {
        return y < x;
    }
};

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
public:
    /**
     * \brief Constructs a heap with a pre-allocated capacity
     *
     * \param capacity heap maximum capacity
     */
    Heap(const int capacity)
    {
        reserve(capacity);
    }

    /**
     * \brief Move-constructs a heap from an external vector
     *
     * \param vec external vector
     */
    Heap(std::vector<T>&& vec)
        : heap(std::move(vec))
    {
        std::make_heap(heap.begin(), heap.end(), greater<T>());
    }

    /**
     *
     * \returns heap size
     */
    int size() const
    {
        return (int)heap.size();
    }

    /**
     *
     * \returns heap capacity
     */
    int capacity() const
    {
        return (int)heap.capacity();
    }

    /**
     * \brief Tests if the heap is empty
     *
     * \returns true is heap empty, false otherwise
     */
    bool empty()
    {
        return heap.empty();
    }

    /**
     * \brief Clears the heap.
     */
    void clear()
    {
        heap.clear();
    }

    /**
     * \brief Sets the heap maximum capacity.
     *
     * \param capacity heap maximum capacity
     */
    void reserve(const int capacity)
    {
        heap.reserve(capacity);
    }

    /**
     * \brief Inserts a new element in the heap.
     *
     * We select the next empty leaf node, and then keep moving any larger
     * parents down until the right location is found to store this element.
     *
     * \param value the new element to be inserted in the heap
     */
    void insert(T value)
    {
        /* If heap is full, then return without adding this element. */
        if (size() == capacity()) {
            return;
        }

        heap.push_back(value);
        std::push_heap(heap.begin(), heap.end(), greater<T>());
    }

    /**
     * \brief Returns the node of minimum value from the heap (top of the heap).
     *
     * \param[out] value parameter used to return the min element
     * \returns false if heap empty
     */
    bool popMin(T& value)
    {
        if (empty()) {
            return false;
        }

        value = heap[0];
        std::pop_heap(heap.begin(), heap.end(), greater<T>());
        heap.pop_back();

        return true;  /* Return old last node. */
    }

    /**
     * \brief Returns a shared heap for the given memory pool ID.
     *
     * It constructs the heap if it does not already exists.
     *
     * \param poolId a user-chosen hashable ID for identifying the heap.
     *     For thread-safe operations, using current thread ID is a good choice.
     * \param capacity heap maximum capacity
     * \param iterThreshold remove heaps that were not reused for more than specified iterations count
     *        if iterThreshold value is less 2, it will be internally adjusted to twice the number of CPU threads
     * \returns pointer to the heap
     */
    template <typename HashableT>
    static cv::Ptr<Heap<T>> getPooledInstance(
        const HashableT& poolId, const int capacity, int iterThreshold = 0)
    {
        static cv::Mutex mutex;
        const cv::AutoLock lock(mutex);

        struct HeapMapValueType {
            cv::Ptr<Heap<T>> heapPtr;
            int iterCounter;
        };
        typedef std::unordered_map<HashableT, HeapMapValueType> HeapMapType;

        static HeapMapType heapsPool;
        typename HeapMapType::iterator heapIt = heapsPool.find(poolId);

        if (heapIt == heapsPool.end())
        {
            // Construct the heap as it does not already exists
            HeapMapValueType heapAndTimePair = {cv::makePtr<Heap<T>>(capacity), 0};
            const std::pair<typename HeapMapType::iterator, bool>& emplaceResult = heapsPool.emplace(poolId, std::move(heapAndTimePair));
            CV_CheckEQ(static_cast<int>(emplaceResult.second), 1, "Failed to insert the heap into its memory pool");
            heapIt = emplaceResult.first;
        }
        else
        {
            CV_CheckEQ(heapIt->second.heapPtr.use_count(), 1, "Cannot modify a heap that is currently accessed by another caller");
            heapIt->second.heapPtr->clear();
            heapIt->second.heapPtr->reserve(capacity);
            heapIt->second.iterCounter = 0;
        }

        if (iterThreshold <= 1) {
            iterThreshold = 2 * cv::getNumThreads();
        }

        // Remove heaps that were not reused for more than given iterThreshold
        typename HeapMapType::iterator cleanupIt = heapsPool.begin();
        while (cleanupIt != heapsPool.end())
        {
            if (cleanupIt->second.iterCounter++ > iterThreshold)
            {
                CV_Assert(cleanupIt != heapIt);
                cleanupIt = heapsPool.erase(cleanupIt);
                continue;
            }
            ++cleanupIt;
        }

        return heapIt->second.heapPtr;
    }
};

}

//! @endcond

#endif //OPENCV_FLANN_HEAP_H_
