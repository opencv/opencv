// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// broadcastOp: an op-agnostic driver for broadcasting element-wise traversal.
//
// It takes a flat list of operand Mats (it does NOT distinguish inputs from outputs), computes
// the numpy-broadcast iteration space over all of them (channels = innermost dim), partitions it
// into tasks, runs them with parallel_for_, and for each tile passes the per-operand slices
// THROUGH to a body lambda. Everything semantic - which array is the output, which kernels run,
// temp buffers - lives in `body`, which typically runs a prepared TExpr.
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

// NOTE: `cv::BroadcastOp` (struct + the `broadcastOp()` shorthand) now lives in core's public
// mat.hpp (included via opencv2/core.hpp from ew_op.hpp) as part of integrating the engine into core.
// Its implementation BroadcastOp::run still lives in ew_broadcast.cpp for now. This header remains so
// existing `#include "ew_broadcast.hpp"` sites keep working; it adds nothing beyond the core header.

#endif // OPENCV_EW_BROADCAST_HPP
