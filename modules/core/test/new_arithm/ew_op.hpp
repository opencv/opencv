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
enum TOp
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
inline int opArity(TOp op) { return ((int)op >> OP_ARITY_SHIFT) & 7; }

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
const char*      opName(TOp op);
ElemwiseCategory opCategory(TOp op);

// numpy-ish arithmetic promotion of two depths, OpenCV same-type preference and INTEGER-PRESERVING
// (mixing two integers stays integer; a float operand pulls it to f32/f64). EW_DEPTH_NONE on one
// side returns the other. Used both by emit() for type inference and as the rdepth == -1 auto rule.
int promoteArith(int a, int b);

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
// The trailing `params` points at the instruction's scalar parameter block (EwInsn::params,
// a cv::Scalar's 4 doubles): mul/div read params[0] as a scale (1.0 = none). Ops with no scalar
// parameter ignore it. It is never null (the executor always passes the instruction's block).
// ---------------------------------------------------------------------------
typedef int (*ElemwiseFunc)(
    const void* src0, size_t step0y, size_t step0x,
    const void* src1, size_t step1y, size_t step1x,
    const void* src2, size_t step2y, size_t step2x,
    void*       dst,  size_t dstepy,
    int width, int height, const double* params);

// Dispatcher: returns the best kernel for (op, operand depths, result depth), or nullptr
// if the exact combination is not provided (the compiler then inserts OP_CAST nodes and
// retries with a supported working type). Unused operand depths are EW_DEPTH_NONE.
ElemwiseFunc getElemwiseFunc(TOp op, int depth0, int depth1, int depth2, int rdepth);

// T + T -> R
ElemwiseFunc getAddFunc(int T, int R);

// T / T -> R (work float). `checked` => guard divide-by-zero -> 0 (both-integer inputs); else a/0
// -> inf (any float input). The caller decides from the ORIGINAL input types (not the common T).
ElemwiseFunc getDivFunc(int T, int R, bool checked);

// ---------------------------------------------------------------------------
// Calling convention of an instruction's kernel. Most ops are EW_FN_ELEMWISE (the broadcasting
// ElemwiseFunc above). Type conversions (OP_CAST / OP_CONVERT_SCALE) reuse core's convert
// kernels directly as EW_FN_BINARY: no broadcast, byte steps, a Size tile, scale/offset passed
// through the instruction's params block - so no adapter/context layer is needed.
// ---------------------------------------------------------------------------
enum EwFuncKind { EW_FN_ELEMWISE = 0, EW_FN_BINARY = 1 };

// Mirror of core's internal BinaryFunc. The return type is not part of a function's symbol
// name, so an engine-side declaration using this type links to cv::getConvertFunc /
// cv::getConvertScaleFunc even though core declares them returning its own BinaryFunc. For the
// scale variant `params` is read as a double[2] {scale, offset}; plain convert ignores it.
typedef void (*EwBinaryFunc)(const uchar* src1, size_t step1,
                             const uchar* src2, size_t step2,
                             uchar* dst, size_t step, Size sz, void* params);

// An element-wise expression as ONE flat program (the analogue of cv::MatExpr for element-wise
// ops): an arg table (`arginfo`, the typed operands) + an instruction list (`prog`). There is no
// separate high-level graph - the program IS the representation. It is built directly, exactly the
// way the hand builders (makeXxx in ew_exec) and the cv::expression parser do it:
//   - declare operands with addInput()/addConst()/addOutput() (and addTemp() for intermediates);
//   - append operations with addInsn() (a single op, you pick the operand/result slots) or, when
//     you want automatic type inference + cast insertion, with emit() (it derives the result depth,
//     inserts OP_CAST where an exact kernel is missing, allocates the result temp, returns its slot).
// The operand types are known at build time (inputs carry their depth), so the whole program is
// typed as it is built. compile() is then a cheap finalize pass: bind each instruction's kernel
// (resolveInsnKernel) and pack the temps into a minimal set of physical buffers (liveness). Any
// info compile() needs PER ARG (e.g. liveness intervals) is transient - it lives in local arrays
// inside compile() and is gone when it returns; `Arg` stays the small permanent operand record.
//
// Heap-free for the common case: to eventually back cv::add() the program is (re)built on every
// call, so its containers must not allocate for typical (small) expressions. AutoBuffer keeps a
// handful of insns/slots inline on the stack and only spills to the heap for large expressions; it
// is copyable, so TExpr is still returned/passed by value.
//
// NB: a default-constructed AutoBuffer reports size()==fixed_size (meant to be sized up front, not
// grown from empty). clear() calls allocate(0) on each container to reset the size to 0, after
// which push_back/resize/size() behave exactly like std::vector.
struct TExpr
{
    // Static (shape-independent) classification of an arg slot.
    enum ArgKind
    {
        NONE = 0,   // the reserved empty operand (slot 0)
        INPUT,
        CONST,
        TEMP,
        OUTPUT
    };

    struct Arg
    {
        ArgKind kind = NONE;
        int depth = EW_DEPTH_NONE;  // EW_DEPTH_NONE on a CONST = "flexible" (emit() types it per use)
        int channels = 0;           // for CONST: # of meaningful channels in cval (0/1 => single broadcast value)
        int index = -1;             // input#/output#/temp-id depending on kind
        // Constant value(s), CONST only: a per-channel cv::Scalar (up to 4 channels). That covers
        // ~all real constants; anything larger/per-element is passed as a real input tensor.
        Scalar cval;
    };

    // One compiled instruction: the op + arg-table indices + a resolved kernel. The kernel uses one
    // of two calling conventions selected by `fnkind`: EW_FN_ELEMWISE uses `fptr` (the broadcasting
    // ElemwiseFunc), EW_FN_BINARY uses `bfptr` (a core convert BinaryFunc, for OP_CAST/CONVERT_SCALE).
    // `params` is the per-instruction scalar block (passed to the kernel by pointer): mul/div scale
    // in params[0]; convert_scale {scale, offset} in params[0..1]. Default (scale 1, offset 0) is the
    // identity, so ops with no scalar parameter need not set it. fptr/bfptr are null until compile().
    struct Insn
    {
        ElemwiseFunc fptr = nullptr;        // EW_FN_ELEMWISE kernel
        EwBinaryFunc bfptr = nullptr;       // EW_FN_BINARY  kernel (core convert / convert-scale)
        TOp op = OP_NOP;
        signed char fnkind = EW_FN_ELEMWISE;
        int arg0 = 0, arg1 = 0, arg2 = 0, result = 0;
        Scalar params = Scalar(1);          // op scalars; params[0]=scale defaults to 1 (identity)
    };

    AutoBuffer<Insn, 16>   prog;             // instructions, in execution order (kernels bound by compile())
    AutoBuffer<Arg, 16>    arginfo;          // slot 0 is always NONE
    int ninputs = 0;
    int noutputs = 0;
    int ntemps = 0;
    int nbuffers = 0;                        // distinct physical temp buffers after liveness
    int nconsts = 0;                         // # materialized CONST slots (set by compile()) - sizes
                                             // the const store; nconsts==0 enables exec()'s fast path
    AutoBuffer<int, 16>    bufferOfTemp;     // temp-id -> physical buffer id
    AutoBuffer<int, 8>     bufEszPrefix;     // [nbuffers+1] prefix sums of each physical temp buffer's
                                             // elem size (set by compile()); [nbuffers] = temp bytes
                                             // per output element. Lets exec() skip per-call recompute.

    TExpr();
    void clear();                            // reset to an empty program (slot 0 = NONE)

    // ---- operand / instruction builders (return the new slot / instruction index) ----
    int  addInput(int depth);
    int  addConst(int depth, const Scalar& v, int channels = 1);   // depth EW_DEPTH_NONE => flexible
    int  addTemp(int depth);
    int  addOutput(int depth);
    int  addInsn(TOp op, int a0, int a1, int a2, int result, const Scalar& params = Scalar(1));

    // Append `op` over the given operand slots with automatic type inference: derive the result
    // depth (promotion / castDepth), pick a direct kernel if one exists else compute in a working
    // type and cast, materializing flexible CONST operands at the chosen type. Returns the slot
    // holding the result. `castDepth` forces the result depth (OP_CAST / explicitly-typed ops).
    int  emit(TOp op, const int* args, int nargs, int castDepth = EW_DEPTH_NONE);

    // Declare a result tensor fed by `rootSlot`. If `rootSlot` is a TEMP produced by one
    // instruction and used nowhere else, that instruction is redirected to write the new OUTPUT
    // slot directly (no copy); otherwise an identity OP_CAST copies it. Returns the OUTPUT slot.
    int  output(int rootSlot);

    // Finalize: bind every instruction's kernel and pack temps into physical buffers (liveness).
    void compile();

    // Execute the compiled program over a set of input Mats, producing the broadcast result(s).
    void exec(const Mat* inputs, Mat* outputs);
};

// Resolve the kernel + calling convention for one instruction from its op and operand/result
// depths: sets ins.fnkind and the matching function pointer (fptr or bfptr). Returns false (and
// leaves the pointers null) when no kernel exists, so the builder can fall back to a wider type.
bool resolveInsnKernel(TExpr::Insn& ins, int depth0, int depth1, int depth2, int rdepth);

}} // namespace cv::ew

#endif // OPENCV_EW_OP_HPP
