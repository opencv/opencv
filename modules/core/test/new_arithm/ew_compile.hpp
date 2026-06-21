// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 2: the graph compiler. Takes a small expression graph (built by hand for now, by the
// cv::expression parser later) and lowers it into a frozen EwProgram: it infers types, inserts
// OP_CAST nodes wherever an exact kernel is missing or a different output type is requested,
// and packs the intermediate results into a minimal set of reusable temp buffers (liveness).
//
// The graph is a flat SSA-style pool of nodes; children are created before their parents, so
// node order is a valid evaluation order. A node is one of:
//   - NODE_INPUT  : the i-th external input (depth supplied at compile time)
//   - NODE_CONST  : a cv::Scalar literal (its depth adapts to the consuming op)
//   - NODE_OP     : op(args...); OP_CAST additionally carries a target depth in `castDepth`.

#ifndef OPENCV_EW_COMPILE_HPP
#define OPENCV_EW_COMPILE_HPP

#include "ew_op.hpp"

namespace cv { namespace ew {

enum EwNodeKind { NODE_INPUT, NODE_CONST, NODE_OP };

struct EwNode
{
    EwNodeKind kind = NODE_OP;

    // NODE_OP
    ElemwiseOp op = OP_NOP;
    int args[3] = { -1, -1, -1 };
    int nargs = 0;
    int castDepth = EW_DEPTH_NONE;   // OP_CAST / explicitly-typed ops: requested result depth

    // NODE_INPUT
    int inputIndex = -1;

    // NODE_CONST
    Scalar cval;
    int cchannels = 1;               // # of meaningful channels in cval (<= 4)

    // filled in by the compiler
    int depth = EW_DEPTH_NONE;       // inferred natural result depth
};

// Expression graph: a node pool plus the list of output node ids (one per result tensor).
struct EwGraph
{
    std::vector<EwNode> nodes;
    int ninputs = 0;
    std::vector<int> outputs;

    // builder helpers (return the new node's id)
    int input(int idx);
    int constant(const Scalar& v, int channels = 1);
    int unary(ElemwiseOp op, int a);
    int binary(ElemwiseOp op, int a, int b);
    int ternary(ElemwiseOp op, int a, int b, int c);
    int cast(int a, int depth);
    void output(int node) { outputs.push_back(node); }
};

// Compile a graph into a frozen program. inputDepths[i] is the depth of input #i.
void compile(const EwGraph& g, EwProgram& p, const int* inputDepths, size_t ninputs);

}} // namespace cv::ew

#endif // OPENCV_EW_COMPILE_HPP
