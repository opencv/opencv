// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2013-2016, The Regents of The University of Michigan.
//
// This software was developed in the APRIL Robotics Lab under the
// direction of Edwin Olson, ebolson@umich.edu. This software may be
// available under alternative licensing terms; contact the address above.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the Regents of The University of Michigan.

#include "../../precomp.hpp"
#include "zmaxheap.hpp"


//                 0
//         1               2
//      3     4        5       6
//     7 8   9 10    11 12   13 14
//
// Children of node i:  2*i+1, 2*i+2
// Parent of node i: (i-1) / 2
//
// Heap property: a parent is greater than (or equal to) its children.

#define MIN_CAPACITY 16
namespace cv {
namespace aruco {
struct zmaxheap
{
    size_t el_sz;

    int size;
    int alloc;

    float *values;
    char *data;

    void (*swap)(zmaxheap_t *heap, int a, int b);
};

static inline void _swap_default(zmaxheap_t *heap, int a, int b)
{
    float t = heap->values[a];
    heap->values[a] = heap->values[b];
    heap->values[b] = t;

    cv::AutoBuffer<char> tmp(heap->el_sz);
    memcpy(tmp.data(), &heap->data[a*heap->el_sz], heap->el_sz);
    memcpy(&heap->data[a*heap->el_sz], &heap->data[b*heap->el_sz], heap->el_sz);
    memcpy(&heap->data[b*heap->el_sz], tmp.data(), heap->el_sz);
}

static inline void _swap_pointer(zmaxheap_t *heap, int a, int b)
{
    float t = heap->values[a];
    heap->values[a] = heap->values[b];
    heap->values[b] = t;

    void **pp = (void**) heap->data;
    void *tmp = pp[a];
    pp[a] = pp[b];
    pp[b] = tmp;
}


zmaxheap_t *zmaxheap_create(size_t el_sz)
{
    zmaxheap_t *heap = (zmaxheap_t*)calloc(1, sizeof(zmaxheap_t));
    heap->el_sz = el_sz;

    heap->swap = _swap_default;

    if (el_sz == sizeof(void*))
        heap->swap = _swap_pointer;

    return heap;
}

void zmaxheap_destroy(zmaxheap_t *heap)
{
    free(heap->values);
    free(heap->data);
    memset(heap, 0, sizeof(zmaxheap_t));
    free(heap);
}

static void _zmaxheap_ensure_capacity(zmaxheap_t *heap, int capacity)
{
    if (heap->alloc >= capacity)
        return;

    int newcap = heap->alloc;

    while (newcap < capacity) {
        if (newcap < MIN_CAPACITY) {
            newcap = MIN_CAPACITY;
            continue;
        }

        newcap *= 2;
    }

    heap->values = (float*)realloc(heap->values, newcap * sizeof(float));
    heap->data = (char*)realloc(heap->data, newcap * heap->el_sz);
    heap->alloc = newcap;
}

void zmaxheap_add(zmaxheap_t *heap, void *p, float v)
{
    _zmaxheap_ensure_capacity(heap, heap->size + 1);

    int idx = heap->size;

    heap->values[idx] = v;
    memcpy(&heap->data[idx*heap->el_sz], p, heap->el_sz);

    heap->size++;

    while (idx > 0) {

        int parent = (idx - 1) / 2;

        // we're done!
        if (heap->values[parent] >= v)
            break;

        // else, swap and recurse upwards.
        heap->swap(heap, idx, parent);
        idx = parent;
    }
}

// Removes the item in the heap at the given index.  Returns 1 if the
// item existed. 0 Indicates an invalid idx (heap is smaller than
// idx). This is mostly intended to be used by zmaxheap_remove_max.
static int zmaxheap_remove_index(zmaxheap_t *heap, int idx, void *p, float *v)
{
    if (idx >= heap->size)
        return 0;

    // copy out the requested element from the heap.
    if (v != NULL)
        *v = heap->values[idx];
    if (p != NULL)
        memcpy(p, &heap->data[idx*heap->el_sz], heap->el_sz);

    heap->size--;

    // If this element is already the last one, then there's nothing
    // for us to do.
    if (idx == heap->size)
        return 1;

    // copy last element to first element. (which probably upsets
    // the heap property).
    heap->values[idx] = heap->values[heap->size];
    memcpy(&heap->data[idx*heap->el_sz], &heap->data[heap->el_sz * heap->size], heap->el_sz);

    // now fix the heap. Note, as we descend, we're "pushing down"
    // the same node the entire time. Thus, while the index of the
    // parent might change, the parent_score doesn't.
    int parent = idx;
    float parent_score = heap->values[idx];

    // descend, fixing the heap.
    while (parent < heap->size) {

        int left = 2*parent + 1;
        int right = left + 1;

//            assert(parent_score == heap->values[parent]);

        float left_score = (left < heap->size) ? heap->values[left] : -INFINITY;
        float right_score = (right < heap->size) ? heap->values[right] : -INFINITY;

        // put the biggest of (parent, left, right) as the parent.

        // already okay?
        if (parent_score >= left_score && parent_score >= right_score)
            break;

        // if we got here, then one of the children is bigger than the parent.
        if (left_score >= right_score) {
            CV_Assert(left < heap->size);
            heap->swap(heap, parent, left);
            parent = left;
        } else {
            // right_score can't be less than left_score if right_score is -INFINITY.
            CV_Assert(right < heap->size);
            heap->swap(heap, parent, right);
            parent = right;
        }
    }

    return 1;
}

int zmaxheap_remove_max(zmaxheap_t *heap, void *p, float *v)
{
    return zmaxheap_remove_index(heap, 0, p, v);
}

}}
