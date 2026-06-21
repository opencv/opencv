// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Prototype of the new element-wise expression engine (lives in tests for now).
// Layer 0: the low-level contract shared by kernels, the graph compiler and the executor.
//
// Design notes (agreed):
//  - Universal arity: every instruction is {fptr, arg0, arg1, arg2, result}; unused
//    operands reference the reserved "none" arg slot (index 0) -> {nullptr, 0, 0}.
//  - A kernel processes one 2D tile of a single slice. The two tile dimensions are not
//    necessarily (y, x): the executor may map e.g. (x, channel) onto (height, width).
//  - Broadcasting at the lowest level is expressed via per-operand y/x steps; a step of 0
//    re-reads the same row/element (scalar / row / column broadcast).
//  - Steps are measured in ELEMENTS, not bytes. dst is contiguous in x (dst.stepx == 1).
//  - No untyped `void* params`: scalars/constants/scale/offset/bounds are all regular
//    (typically broadcast) args.

#ifndef OPENCV_EW_OP_HPP
#define OPENCV_EW_OP_HPP

#include "opencv2/core.hpp"
#include <array>
#include <vector>

namespace cv { namespace ew {

// Sentinel depth for an unused operand (note: CV_8U == 0 is a *valid* depth,
// so the "no operand" marker must be negative).
enum { EW_DEPTH_NONE = -1 };

// Each op carries a fixed numerical value whose high bits encode its arity, so the arity
// can be recovered with a shift and no lookup table:  arity = (op >> OP_ARITY_SHIFT) & 7.
// The ops are grouped into contiguous arity blocks (unary = 1<<10, binary = 2<<10,
// ternary = 3<<10); within a block the low bits are just a running index.
enum { OP_ARITY_SHIFT = 10 };
enum
{
    OP_UNARY_BASE   = 1 << OP_ARITY_SHIFT,   // 0x400
    OP_BINARY_BASE  = 2 << OP_ARITY_SHIFT,   // 0x800
    OP_TERNARY_BASE = 3 << OP_ARITY_SHIFT    // 0xC00
};

// The single enumeration of element-wise operations, used both in the IR and by the
// kernel dispatcher.
enum ElemwiseOp
{
    OP_NOP = 0,

    // ---------------- unary (arity 1) ----------------
    OP_NEG = OP_UNARY_BASE, OP_ABS, OP_NOT,
    OP_SQRT, OP_EXP, OP_LOG,
    OP_SIN, OP_COS, OP_TANH, OP_ERF, OP_RELU,
    OP_CAST,            // saturating type conversion, no scaling

    // ---------------- binary (arity 2) ----------------
    OP_ADD = OP_BINARY_BASE, OP_SUB, OP_MUL, OP_DIV, OP_POW,
    OP_MIN, OP_MAX, OP_ABSDIFF,
    OP_AND, OP_OR, OP_XOR,
    // compare -> mask (result depth given explicitly, e.g. CV_Bool/CV_8U)
    OP_CMP_EQ, OP_CMP_NE, OP_CMP_LT, OP_CMP_LE, OP_CMP_GT, OP_CMP_GE,
    // partial-output write: dst = (mask != 0) ? src : dst (unmasked output PRESERVED). arg0 = src
    // (data), arg1 = mask (1 byte: bool/u8/s8). Used to apply an op's mask: the op computes into a
    // temp, copyMask overwrites only the masked subset of the (pre-existing) output - matching
    // cv::add/... with a mask.
    OP_COPY_MASK,

    // ---------------- ternary (arity 3) ----------------
    OP_CLAMP = OP_TERNARY_BASE,   // clamp(x, lo, hi)
    OP_SELECT,                    // select(mask, a, b)  (a.k.a. where)
    OP_CONVERT_SCALE              // cast<rdepth>(src*scale + offset); scale/offset may be tensors
};

// Arity from the encoding above (0 for OP_NOP). No table to keep in lock-step with the enum.
inline int opArity(ElemwiseOp op) { return ((int)op >> OP_ARITY_SHIFT) & 7; }

// Operation category — the graph compiler's type-inference rules differ per category.
enum ElemwiseCategory
{
    CAT_ARITH = 0,  // numeric, result type follows promotion rules
    CAT_BITWISE,    // integer-only, same-type
    CAT_COMPARE,    // produces a mask, result depth is explicit
    CAT_MATH,       // transcendental, float domain
    CAT_CAST,       // type conversion (with/without scaling)
    CAT_SELECT      // data-routing (select/where)
};

// Op metadata (implemented in ew_op.cpp).
const char*      opName(ElemwiseOp op);
ElemwiseCategory opCategory(ElemwiseOp op);

// ---------------------------------------------------------------------------
// Steps for one operand, in elemsize1 units, one entry per shape dim (parallel to a
// MatShape). A 0 entry means broadcast along that axis. Heap-free, like MatShape.
// ---------------------------------------------------------------------------
typedef std::array<size_t, MatShape::MAX_DIMS> EwSteps;

// ---------------------------------------------------------------------------
// The low-level kernel contract.
//
// Processes a width x height tile of one slice. Each source operand carries its own
// (stepy, stepx) in elements; a 0 step means broadcast along that axis. stepx is restricted
// to {0,1} (1 = contiguous, 0 = broadcast-scalar along x): the general strided/gather case is
// excluded and the executor guarantees the invariant (materializing a contiguous copy of any
// rare non-unit-innermost-stride input). The result is contiguous in x (dst stepx == 1).
// Returns >= 0 on success, < 0 (a CV_HAL_ERROR_* code) to let the caller fall back.
//
// The trailing `ctx` is an optional, per-instruction adapter context built once before the
// parallel loop (see EwInsn::ctx). Generic kernels ignore it (it is nullptr). "Adapter"
// kernels reinterpret it to carry precomputed state - e.g. a multi-channel-scalar register
// pattern (array-op-scalar), a wrapped core BinaryFunc + scale params (convert / convert_scale),
// or a result mask (compare -> 0/1 vs 0/255). An adapter does no arithmetic itself: it only
// unpacks ctx and forwards to the real kernel.
// ---------------------------------------------------------------------------
typedef int (*ElemwiseFunc)(
    const void* src0, size_t step0y, size_t step0x,
    const void* src1, size_t step1y, size_t step1x,
    const void* src2, size_t step2y, size_t step2x,
    void*       dst,  size_t dstepy,
    int width, int height, void* ctx);

// Dispatcher: returns the best kernel for (op, operand depths, result depth), or nullptr
// if the exact combination is not provided (the compiler then inserts OP_CAST nodes and
// retries with a supported working type). Unused operand depths are EW_DEPTH_NONE.
ElemwiseFunc getElemwiseFunc(ElemwiseOp op, int depth0, int depth1, int depth2, int rdepth);

// T + T -> R
ElemwiseFunc getAddFunc(int T, int R);

// T / T -> R (work float). `checked` => guard divide-by-zero -> 0 (both-integer inputs); else a/0
// -> inf (any float input). The caller decides from the ORIGINAL input types (not the common T).
ElemwiseFunc getDivFunc(int T, int R, bool checked);

// ---------------------------------------------------------------------------
// Adapter context: the optional trailing void* of ElemwiseFunc. A small fixed-size POD built
// on the fly by the executor before the parallel loop and discriminated there by op category;
// each adapter kernel casts the void* to the variant it expects. Large payloads (e.g. an
// expanded multi-channel-scalar register pattern) live in a separate AutoBuffer and are only
// pointed to from here, so EwCtx stays small with no heap ownership (no std::shared_ptr).
// ---------------------------------------------------------------------------

// Mirror of core's internal BinaryFunc. The return type is not part of a function's symbol
// name, so an engine-side declaration using this type links to cv::getConvertFunc /
// cv::getConvertScaleFunc even though core declares them returning its own BinaryFunc.
typedef void (*EwBinaryFunc)(const uchar* src1, size_t step1,
                             const uchar* src2, size_t step2,
                             uchar* dst, size_t step, Size sz, void* params);

struct EwCtx
{
    union
    {
        // CAT_CAST: wrap a core convert / convert-scale BinaryFunc (steps in bytes, Size tile).
        struct { EwBinaryFunc fn; int sesz1, desz1; double scale[2]; } cvt;
        // array-op-scalar: the specialized inner kernel + the cyclic C-channel register pattern.
        // The pattern lives in a separate per-call AutoBuffer; here we only point into it.
        struct { ElemwiseFunc fn; const void* pattern; } sc;
        // CAT_COMPARE: result mask AND-ed onto the native compare (1 => 0/1, 0xFF => 0/255).
        struct { int mask; } cmp;
    };
};

// ---------------------------------------------------------------------------
// Runtime operand descriptor: what a kernel actually receives per slice.
// depth/channels are carried for assertions/debugging only; the kernel itself is already
// type-specialized via the function pointer.
// ---------------------------------------------------------------------------
struct EwArg
{
    const void* ptr = nullptr;
    size_t stepy = 0;
    size_t stepx = 0;
    int depth = EW_DEPTH_NONE;   // debug
    int channels = 0;            // debug
};

// ---------------------------------------------------------------------------
// Static (shape-independent) description of an arg slot in a compiled program.
// ---------------------------------------------------------------------------
enum ArgKind
{
    ARG_NONE = 0,   // the reserved empty operand (slot 0)
    ARG_INPUT,
    ARG_CONST,
    ARG_TEMP,
    ARG_OUTPUT
};

struct EwArgInfo
{
    ArgKind kind = ARG_NONE;
    int depth = EW_DEPTH_NONE;
    int channels = 0;               // for ARG_CONST: # of meaningful channels in cval (0/1 => single broadcast value)
    int index = -1;                 // input#/output#/temp-id depending on kind
    // Constant value(s), ARG_CONST only: a per-channel cv::Scalar (up to 4 channels). That
    // covers ~all real constants; anything larger/per-element is passed as a real input tensor.
    Scalar cval;
};

// One compiled instruction: resolved kernel + the op (kept for sanity checks) + indices
// into the program's arg table.
struct EwInsn
{
    ElemwiseFunc fptr = nullptr;
    ElemwiseOp op = OP_NOP;
    int arg0 = 0, arg1 = 0, arg2 = 0, result = 0;
};

// A frozen program: everything that does NOT depend on the concrete shapes of a call.
// Shapes, strides and the physical tile/output memory are computed per-call in the executor.
//
// Heap-free for the common case: to eventually back cv::add() the whole program is (re)built on
// every call, so its containers must not allocate. AutoBuffer keeps typical element-wise programs
// (a handful of insns/slots) entirely inline on the stack and only falls back to the heap for
// unusually large expressions. AutoBuffer is copyable, so EwProgram is still returned by value.
//
// NB: a default-constructed AutoBuffer reports size()==fixed_size (it is meant to be sized up
// front, not grown from empty). The constructor calls allocate(0) on each container to reset the
// size to 0, after which push_back/resize/size() behave exactly like std::vector.
struct EwProgram
{
    AutoBuffer<EwInsn, 16>     prog;          // resolved instructions, in execution order
    AutoBuffer<EwArgInfo, 16>  arginfo;       // slot 0 is always ARG_NONE
    int ninputs = 0;
    int noutputs = 0;
    int ntemps = 0;
    int nbuffers = 0;                         // distinct physical temp buffers after liveness
    AutoBuffer<int, 16>        bufferOfTemp;   // temp-id -> physical buffer id

    EwProgram();
    void clear();

    // Execute a compiled program.
    void exec(const Mat* inputs, Mat* outputs);
};

}} // namespace cv::ew

#endif // OPENCV_EW_OP_HPP
