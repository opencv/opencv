// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 3: the executor. Runs a frozen EwProgram over a set of input Mats, producing the
// broadcast result(s). Shape inference, broadcast preparation, dim collapsing and all
// allocation happen once per call, OUTSIDE the parallel loop; the parallel workers only
// re-point args at their tile and run the instruction list.

#ifndef OPENCV_EW_EXEC_HPP
#define OPENCV_EW_EXEC_HPP

#include "ew_op.hpp"

namespace cv { namespace ew {

// ----- manual program builders (stand-ins for the future engine-backed cv::add etc.) -----

// Compose a binary arith op (OP_ADD / OP_SUB / OP_MUL / OP_DIV) for ANY (depth0, depth1, rdepth):
// cast operands to a common type, op direct-or-wide-then-cast (see .cpp). MUL/DIV compute in the
// float work type and cast down. makeAddProgram is a thin OP_ADD wrapper.
// maskDepth != EW_DEPTH_NONE adds a write-mask input (#2): the result lands in a temp and a final
// OP_COPY_MASK overwrites only the masked subset of the (pre-existing) output, preserving the rest
// (dst = mask ? result : dst, matching cv::add/... with a mask).
// scale != 1 (MUL/DIV only) rides the mul/div instruction's params[0] - mul: a*b*scale, div:
// a*scale/b - exactly like cv::multiply/divide's scale argument.
void makeBinaryArithProgram(EwProgram& p, ElemwiseOp op, int depth0, int depth1, int rdepth,
                            int maskDepth = EW_DEPTH_NONE, double scale = 1.0);
void makeAddProgram(EwProgram& p, int depth0, int depth1, int rdepth);

// addWeighted(a,alpha,b,beta,gamma) = a*alpha + b*beta + gamma, built as two fused convert_scale
// MACs + an add (+ a final cast when rdepth != working type). Exercises the L1 fragmentation path.
void makeAddWeightedProgram(EwProgram& p, int depth0, int depth1, int rdepth,
                            double alpha, double beta, double gamma);

}} // namespace cv::ew

#endif // OPENCV_EW_EXEC_HPP
