// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// broadcastOp: an op-agnostic driver for broadcasting element-wise traversal.
//
// It takes a flat list of operand Mats (it does NOT distinguish inputs from outputs), computes
// the numpy-broadcast iteration space over all of them (channels = innermost dim), partitions it
// into tasks, runs them with parallel_for_, and for each tile passes the per-operand slices
// THROUGH to a body lambda. Everything semantic - which array is the output, which kernels run,
// temp buffers - lives in `body`, which typically runs a prepared EwProgram.
//
// Hierarchy:  cv::expression()  ->  broadcastOp()  ->  body lambda (runs the frozen program).
//   - cv::expression: parse + compile + prep (shapes, ctx, buffer sizes, nstripes via opCost).
//   - broadcastOp: geometry (collapse + inner 2D tile) + partition + parallel_for_; O(ndims) tile
//     addressing via EwTileCursor.
//   - body: per-thread scratch (locals) + per-tile loop re-pointing program args at the slices.

#ifndef OPENCV_EW_BROADCAST_HPP
#define OPENCV_EW_BROADCAST_HPP

#include "ew_op.hpp"
#include <functional>

namespace cv { namespace ew {

// One operand's slice for the current tile: base pointer + steps in ELEMENTS.
//   stepx in {0,1}: 1 = contiguous along width, 0 = broadcast-scalar along width.
//   stepy: step between the `height` rows (0 = broadcast along height).
// ptr is non-const so the body can write the operand(s) it treats as outputs.
struct EwSlice
{
    void*  ptr   = nullptr;
    size_t stepy = 0;
    size_t stepx = 0;
};

// One 2D tile, built by broadcastOp and handed to the body. `slices[k]` corresponds to
// arrays[k] (same order). The body reads `width`/`height` and the per-operand slices; it owns
// all interpretation (which array is the output, what kernels to run).
struct EwTile
{
    int width  = 0;          // innermost tile extent (elements)
    int height = 0;          // 2nd-innermost extent (1 unless a 2D tile is handed out)
    int narrays = 0;
    const EwSlice* slices = nullptr;   // [narrays], valid for the duration of the body call
};

// Drive a broadcasting element-wise traversal.
//   arrays[0 .. narrays-1] : uniform operands (inputs AND outputs, undistinguished). The
//       iteration space is the numpy-broadcast of all their shapes (channels innermost).
//   body  : invoked once per tile with the per-operand slices for that tile; it runs the prepared
//       program. broadcastOp drives parallel_for_ internally and decodes each tile's addresses in
//       O(ndims), so the body never deals with iteration/threading - just one tile at a time.
//       (Per-thread scratch, e.g. temp buffers or a scalar register pattern, are locals in the
//       body; declared per call they are inherently thread-safe.)
//   expandChannels : how channel counts are presented to the body.
//       true  => channels are an explicit innermost iteration dim, so the body always sees
//                single-channel (scalar-wise) data. Channel 1<->N broadcast is then handled
//                geometrically (stepx 0 on the channel axis). Simple and always correct; the
//                channel dim folds into width automatically when all operands share it. This is
//                what cv::expression uses initially.
//       false => channels stay folded into the element (esz = full elemSize); the body is
//                responsible for them (e.g. deinterleave tricks for the efficient C1<->CN path).
//                Only (n,n)/(n,1)/(1,n) channel combinations ever occur - never (n,m).
//   nstripes : parallel_for_ work hint. 0 => derive from the shapes assuming ~100 cycles/element
//       (broadcastOp can't see the body's cost; cv::expression passes a real value via opCost).
//
// For a cv::Mat the innermost axis (channels / last dim) is always contiguous, so after collapse
// every operand's innermost stride is inherently in {0,1} - there is no gather case and nothing
// to materialize. broadcastOp keeps the unit-stride run as the innermost (tile width) and sends
// any gapped outer axis to stepy (a 2D tile, height>1), so cropped ROIs need no special handling.
void broadcastOp(Mat* arrays, size_t narrays,
                 const std::function<void(const EwTile&)>& body,
                 bool expandChannels = false,
                 double nstripes = 0.);

}} // namespace cv::ew

#endif // OPENCV_EW_BROADCAST_HPP
