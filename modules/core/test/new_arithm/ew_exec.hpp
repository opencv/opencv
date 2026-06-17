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

// Execute a compiled program.
//   inputs.size() must equal program.ninputs.
//   outputs is resized to program.noutputs and (re)allocated to the broadcast result shape.
void exec(const EwProgram& program,
          const std::vector<Mat>& inputs,
          std::vector<Mat>& outputs);

// ----- tiny manual program builders (placeholder until the Layer-2 compiler lands) -----

// Single binary op:  out = op(in0, in1)
EwProgram makeBinaryProgram(ElemwiseOp op, int depth0, int depth1, int rdepth);

// Single unary op:   out = op(in0)     (e.g. OP_CAST)
EwProgram makeUnaryProgram(ElemwiseOp op, int depth0, int rdepth);

// Compose add for ANY (depth0, depth1, rdepth) out of the cast + add kernels:
//   - cast operands to a common input type C (= depth0 if equal, else numpy-ish promotion);
//   - if a direct add kernel (C,C,rdepth) exists, use it; else add in a safe wide type and
//     cast the result down to rdepth.
EwProgram makeAddProgram(int depth0, int depth1, int rdepth);

}} // namespace cv::ew

#endif // OPENCV_EW_EXEC_HPP
