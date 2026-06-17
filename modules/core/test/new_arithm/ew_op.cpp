// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 0: op metadata that does not need to be inline (kept out of the header).

#include "ew_op.hpp"

namespace cv { namespace ew {

const char* opName(ElemwiseOp op)
{
    switch (op)
    {
    case OP_NOP:           return "nop";
    case OP_NEG:           return "neg";
    case OP_ABS:           return "abs";
    case OP_NOT:           return "not";
    case OP_SQRT:          return "sqrt";
    case OP_EXP:           return "exp";
    case OP_LOG:           return "log";
    case OP_SIN:           return "sin";
    case OP_COS:           return "cos";
    case OP_TANH:          return "tanh";
    case OP_ERF:           return "erf";
    case OP_RELU:          return "relu";
    case OP_CAST:          return "cast";
    case OP_ADD:           return "add";
    case OP_SUB:           return "sub";
    case OP_MUL:           return "mul";
    case OP_DIV:           return "div";
    case OP_POW:           return "pow";
    case OP_MIN:           return "min";
    case OP_MAX:           return "max";
    case OP_ABSDIFF:       return "absdiff";
    case OP_AND:           return "and";
    case OP_OR:            return "or";
    case OP_XOR:           return "xor";
    case OP_CMP_EQ:        return "cmp_eq";
    case OP_CMP_NE:        return "cmp_ne";
    case OP_CMP_LT:        return "cmp_lt";
    case OP_CMP_LE:        return "cmp_le";
    case OP_CMP_GT:        return "cmp_gt";
    case OP_CMP_GE:        return "cmp_ge";
    case OP_CLAMP:         return "clamp";
    case OP_SELECT:        return "select";
    case OP_CONVERT_SCALE: return "convert_scale";
    default:               return "?";
    }
}

// --- op category (declared in ew_op.hpp) --------------------------------------------------
ElemwiseCategory opCategory(ElemwiseOp op)
{
    switch (op)
    {
    case OP_AND: case OP_OR: case OP_XOR: case OP_NOT:
        return CAT_BITWISE;
    case OP_CMP_EQ: case OP_CMP_NE: case OP_CMP_LT:
    case OP_CMP_LE: case OP_CMP_GT: case OP_CMP_GE:
        return CAT_COMPARE;
    case OP_SQRT: case OP_EXP: case OP_LOG:
    case OP_SIN: case OP_COS: case OP_TANH: case OP_ERF: case OP_RELU:
        return CAT_MATH;
    case OP_CAST: case OP_CONVERT_SCALE:
        return CAT_CAST;
    case OP_SELECT:
        return CAT_SELECT;
    default:
        return CAT_ARITH;   // add/sub/mul/div/pow/min/max/absdiff/neg/abs/clamp
    }
}

}} // namespace cv::ew
