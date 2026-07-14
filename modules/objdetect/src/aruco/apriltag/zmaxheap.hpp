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
#ifndef _OPENCV_ZMAXHEAP_HPP_
#define _OPENCV_ZMAXHEAP_HPP_

namespace cv {
namespace aruco {
typedef struct zmaxheap zmaxheap_t;

typedef struct zmaxheap_iterator zmaxheap_iterator_t;
struct zmaxheap_iterator {
    zmaxheap_t *heap;
    int in, out;
};

zmaxheap_t *zmaxheap_create(size_t el_sz);

void zmaxheap_destroy(zmaxheap_t *heap);

void zmaxheap_add(zmaxheap_t *heap, void *p, float v);

// returns 0 if the heap is empty, so you can do
// while (zmaxheap_remove_max(...)) { }
int zmaxheap_remove_max(zmaxheap_t *heap, void *p, float *v);

}}
#endif
