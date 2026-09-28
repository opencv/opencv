// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// N-dimensional strided copy engine shared by the data-movement functions
// (transposeND, flipND, concatND, splitND, tileND, sliceND).
//
// Each of those functions only builds a pair of strided views (source and destination)
// and hands them to nd::copy(). The engine normalizes the views, collapses contiguous
// axes, picks an inner kernel (memcpy, fill, strided gather, or blocked 2D transpose)
// and runs everything through parallel_for_.
//
// Channels always stay inside the element (esz == elemSize()), so every function built
// on top of this works for any depth and any number of channels, including ROIs.

#ifndef OPENCV_CORE_ND_COPY_HPP
#define OPENCV_CORE_ND_COPY_HPP

namespace cv { namespace nd {

enum { MAX_VIEW_DIMS = CV_MAX_DIM };

struct View
{
    uchar* data = nullptr;
    int dims = 0;
    size_t esz = 0;                    // element size in bytes, channels included
    int size[MAX_VIEW_DIMS];
    ptrdiff_t step[MAX_VIEW_DIMS];     // in bytes; 0 = broadcast, negative = reversed
};

View viewOf(const Mat& m);

// The i-th axis of the result is the order[i]-th axis of v.
View permute(const View& v, const int* order);

// Reverse the direction of the given axis.
void flip(View& v, int axis);

// Keep `count` elements along the axis, starting at `start` and moving by `step` (may be negative).
void slice(View& v, int axis, int start, int step, int count);

// Copy src into dst element by element. Both views must have the same dims, sizes and esz.
// The views may overlap; in that case the source is copied into a temporary buffer first.
void copy(const View& src, const View& dst);

// Same as copy(), but for n independent pairs executed in one parallel pass.
void copyBatch(const View* src, const View* dst, int n);

// 2D block transpose kernel for the given element size, or nullptr if there is none.
// dst(i, j) = src(j, i) for i < sz.width, j < sz.height. Defined in matrix_transform.cpp.
typedef void (*TransposeFunc)(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz);
TransposeFunc getTransposeFunc(size_t esz);

}} // namespace cv::nd

#endif
