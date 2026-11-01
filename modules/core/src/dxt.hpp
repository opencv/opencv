// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Shared, ISA-independent types of the 1D DFT/DCT engine (dxt.cpp + dxt.simd.hpp).
//
// The engine runs a 1D transform in three groups of passes:
//   A (preproc):  table-driven gather from the user input (or from an interleaved temp) into
//                 split re[]/im[] workspace, fused with the first twiddle-free radix-r0 stage,
//                 r0 in {1,2,3,4,5,8} (vectorized for r0 > 1, scalar table gather for r0 == 1).
//   B (stages):   mixed-radix decimation-in-time stages with growing span, direction-free
//                 (forward twiddles only), out-of-place (src pair -> dst pair, ping-pong) or
//                 in-place when the plan says it is legal. Vectorized with universal intrinsics.
//                 For long transforms the leading stages run chunk by chunk (depth-first) so
//                 that their data stays in L2.
//   C (postproc): interleave / CCS packing / DCT ordering with fused scaling into the user output.
//                 For C2C (and even C2R) the last radix-4/8 stage writes the interleaved, scaled
//                 output itself (fused last stage), saving one pass over the data.
//
// Inverse transforms use the conjugation trick: conj() on the gather, im_scale = -scale on output.
// Buffers keep fixed semantics (re is always re).
//
// Plans are immutable after construction and shared read-only by all rows of a batch; the
// workspace is a separate, per-executor, mutable buffer allocated ONCE by the owner (never per
// call). The owner's workspace can live in the plan's single allocation (build(..., true)):
// one malloc per cv::dft() call instead of three, which also matters for glibc, whose heap
// trimming (and the page faults that follow on every call) is triggered by two freed MB-sized
// blocks but not by one twice as big. The tables of the power-of-two lengths up to 1024 are
// shared, static tables (see dxt.cpp).

#ifndef OPENCV_CORE_SRC_DXT_HPP
#define OPENCV_CORE_SRC_DXT_HPP

#include "opencv2/core/utility.hpp"

namespace cv {

enum DftKind
{
    DFT_KIND_C2C = 0,   // complex -> complex (forward or inverse)
    DFT_KIND_R2C,       // real -> CCS-packed (or "half complex") spectrum, forward only
    DFT_KIND_C2R,       // CCS-packed (or full complex) spectrum -> real, inverse only
    DFT_KIND_DCT,       // DCT-II, even n only
    DFT_KIND_IDCT       // DCT-III, even n only
};

enum DftStageMode
{
    DFT_STAGE_SCALAR = 0,   // scalar reference loop (span < any vector width, no SIMD for this depth, ...)
    DFT_STAGE_FULL = 1,     // span >= VL: vector loop with back-off clamp (brief, section 5)
    DFT_STAGE_SIMD128 = 2   // 128-bit lanes < span < VL (wide ISAs only): the same loop on 128-bit vectors
};

enum { DFT_MAX_STAGES = 40 };

struct DftStage
{
    int radix;          // 3, 4, 5, 8 or an odd number >= 7
    int span;           // L: length of the sub-DFTs combined by this stage
    int ngroups;        // G = nc / (radix*span)
    int nsteps;         // number of (vector) iterations of the flat loop, precomputed by the plan
    int steps_per_group; // nsteps / ngroups (the driver re-derives nsteps for chunked runs)
    int mode;           // DftStageMode
    int tw_stride;      // per-leg twiddle table length: span, plus one cache line of padding when
                        // span*esz is a multiple of 4 KB (the legs must not share one cache set)
    // Per-stage twiddles, split layout, leg-major: tw_re[(q-1)*tw_stride + j] = cos(-2*pi*q*j/(radix*span)),
    // q in [1, radix), j in [0, span). Type T of the plan.
    const void* tw_re;
    const void* tw_im;
    // radix-odd (>=7) only: cs[m] = cos(2*pi*m/radix), sn[m] = sin(2*pi*m/radix), m in [0, radix);
    // the butterfly addresses them with the cyclic index m = p*k mod radix
    const void* cs;
    const void* sn;
};

struct DftPlan
{
    int kind;           // DftKind
    int n;              // logical transform length (real length for R2C/C2R/DCT/IDCT)
    int nc;             // complex length actually run through the stages (n/2 for even real kinds)
    int depth;          // CV_32F or CV_64F
    int vl;             // vector lanes of the kernels this plan was built for (DFTKernels::vlanes)
    int first_radix;    // 1, 2, 3, 4, 5 or 8: the fused first stage done in preproc
    bool real_input;    // R2C with odd n: gather real samples, im = 0
    bool pingpong;      // stages must alternate between two workspace pairs
    bool need_tmp;      // C2R / DCT / IDCT: a temp of 2*nc elements (interleaved complex or the Makhoul-permuted reals)
    bool fused_last;    // the last stage is radix 4 or 8 and writes the interleaved scaled output itself (C2C, even C2R)
    int nchunk_stages;  // leading stages run chunk by chunk (chunk = stages[nchunk_stages].span); 0 = none
    int nstages;
    DftStage stages[DFT_MAX_STAGES];
    // first_radix == 1: 2*nc element indices into the (real-typed) source (re index, im index);
    // first_radix > 1: nc/first_radix output positions (g*first_radix) of the input columns j.
    // Points into `buf` or into the static tables.
    const int* itab;
    AutoBuffer<double> buf;     // ONE block: generated tables (type T, 8-byte aligned), itab and,
                                // when requested, the owner's workspace; the stage/rtw tables of
                                // 2^k lengths <= 1024 are static and not stored here
    uchar* workspace;           // ws_bytes inside `buf` when built with with_workspace, else 0
    const void* rtw_re;         // W_n^k = exp(-2*pi*i*k/n), k in [0, n/2]; R2C/C2R/DCT/IDCT with even n
    const void* rtw_im;
    const void* dct_re;         // s*W_4n^k, k in [0, n/2]; DCT/IDCT (s = DCT normalization)
    const void* dct_im;
    size_t ws_bytes;            // workspace size (incl. 64 bytes for alignment)
    size_t im_ofs;              // byte offset of im from re inside a pair: nc*esz plus one cache line when
                                // nc*esz is a multiple of 4 KB (keeps the re and im streams out of one cache set)
    size_t pair1_ofs;           // byte offset of the second (re, im) pair or of the temp, from the aligned base
    size_t scratch_ofs;         // byte offset of the radix-odd scratch area

    DftPlan() : kind(0), n(0), nc(0), depth(0), vl(0), first_radix(1), real_input(false),
                pingpong(false), need_tmp(false), fused_last(false), nchunk_stages(0), nstages(0),
                itab(0), workspace(0), rtw_re(0), rtw_im(0), dct_re(0), dct_im(0), ws_bytes(0),
                im_ofs(0), pair1_ofs(0), scratch_ofs(0) {}

    // Builds the plan for a 1D transform of the given kind (DftKind) and length.
    //
    // IMPORTANT: `vl` and `vl128` (vector lanes of the widest and of the 128-bit vectors) MUST
    // come from the dispatched kernel table (DFTKernels::vlanes, ::vlanes128), never from
    // VTraits<> in dxt.cpp, which is compiled for the baseline ISA and reports the wrong lane
    // count on AVX2/AVX512/RVV builds. Stage modes and step counts depend on them.
    // with_workspace: also reserve ws_bytes for the owner's workspace in the same block (`workspace`).
    void build(int kind, int n, int depth, int vl, int vl128, bool with_workspace = false);

private:
    // fills the twiddle tables (stage tables, W_n^k, DCT) at `dst`; ntw = total number of T entries
    template<typename T> void fillTables(T* dst, size_t ntw, bool need_rtw, bool need_dct, bool static_stages, bool static_rtw);

    DftPlan(const DftPlan&);            // tables hold raw pointers into `buf`: no copies
    DftPlan& operator=(const DftPlan&);
};

// (plan, src, src_step (in elements), re, im, conj)
typedef void (*DftPreprocFunc)(const DftPlan& plan, const void* src, size_t sstep,
                               void* re, void* im, bool conj);
// (stage, src_re, src_im, dst_re, dst_im, scratch); src == dst allowed only when !plan.pingpong
typedef void (*DftStageFunc)(const DftStage& st, const void* sre, const void* sim,
                             void* dre, void* dim, void* scratch);
// fused last stage: (stage, src_re, src_im, interleaved dst, re_scale, im_scale)
typedef void (*DftLastStageFunc)(const DftStage& st, const void* sre, const void* sim,
                                 void* dst, double re_scale, double im_scale);

struct DFTKernels
{
    // group A
    DftPreprocFunc preprocRadix0;   // gather + deinterleave only
    DftPreprocFunc preprocRadix2;   // + fused twiddle-free radix-2
    DftPreprocFunc preprocRadix3;   // + fused twiddle-free radix-3 (odd nc)
    DftPreprocFunc preprocRadix4;   // + fused twiddle-free radix-4
    DftPreprocFunc preprocRadix5;   // + fused twiddle-free radix-5 (odd nc)
    DftPreprocFunc preprocRadix8;   // + fused twiddle-free radix-8
    // C2R: CCS (or full complex, complex_input) spectrum -> interleaved "2Z" temp (2*nc elements)
    void (*preprocCCS)(const DftPlan& plan, const void* src, void* tmp, bool complex_input);
    // DCT: Makhoul permutation of the (possibly strided) input into the temp
    void (*preprocDCT)(const DftPlan& plan, const void* src, size_t sstep, void* tmp);
    // IDCT: pre-twiddle + CCS untangle -> interleaved temp
    void (*preprocIDCT)(const DftPlan& plan, const void* src, size_t sstep, void* tmp);
    // group B
    DftStageFunc radix3, radix4, radix5, radix8, radixOdd;
    DftLastStageFunc radix4Last;    // radix-4 stage + interleave + scale into the user output
    DftLastStageFunc radix8Last;    // the same for a radix-8 last stage
    // group C
    void (*postprocDFT)(const DftPlan& plan, const void* re, const void* im, void* dst,
                        double re_scale, double im_scale);          // split -> interleaved complex
    void (*postprocRealDFT)(const DftPlan& plan, const void* re, const void* im, void* dst,
                            double scale, bool complex_output);     // R2C untangle + CCS/complex packing
    void (*postprocReal)(const DftPlan& plan, const void* re, void* dst, double scale); // re only
    void (*postprocDCT)(const DftPlan& plan, const void* re, const void* im, void* dst, size_t dstep);
    void (*postprocIDCT)(const DftPlan& plan, const void* re, const void* im, void* dst, size_t dstep);
    int vlanes;     // VTraits<v_T>::vlanes() of the ISA these pointers belong to (1 = no SIMD)
    int vlanes128;  // lanes of the 128-bit vectors used by DFT_STAGE_SIMD128 (== vlanes when not wider)
};

}

#endif
