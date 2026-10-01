// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// cv::reduceND: reduction of an n-dimensional array over any set of axes.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cv {

namespace {

enum
{
    RND_MAX_DIMS = CV_MAX_DIM + 1,   // + the channel axis
    RND_BLOCK = 1024,                // elements converted per call when the source needs conversion
    RND_VCHUNK = 1024,               // innermost-axis chunk in the vertical kernel
    RND_SERIAL_WORK = 64*1024,       // below this (source elements) everything runs in one thread
    RND_STRIPE_WORK = 64*1024
};

//////////////////////////////// vector traits ////////////////////////////////

template<typename T> struct VecT { enum { ok = 0 }; typedef T vtype; };
#if (CV_SIMD || CV_SIMD_SCALABLE)
#define CV_RND_VEC(T, V) template<> struct VecT<T> { enum { ok = 1 }; typedef V vtype; };
CV_RND_VEC(uchar, v_uint8)
CV_RND_VEC(schar, v_int8)
CV_RND_VEC(ushort, v_uint16)
CV_RND_VEC(short, v_int16)
CV_RND_VEC(unsigned, v_uint32)
CV_RND_VEC(int, v_int32)
CV_RND_VEC(float, v_float32)
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
CV_RND_VEC(double, v_float64)
#endif
#undef CV_RND_VEC
#endif

//////////////////////////////// operations ////////////////////////////////
// apply() folds one element into an accumulator, combine() merges two accumulators,
// finalize() turns an accumulator into the result. The vector versions are only used
// for the accumulator types that have them (see VecT).

template<typename A> struct OpSum
{
    static A init() { return A(0); }
    static A apply(A a, A x) { return a + x; }
    static A combine(A a, A b) { return a + b; }
    static A finalize(A a, int64) { return a; }
    template<typename V> static V vapply(const V& a, const V& x) { return v_add(a, x); }
    template<typename V> static V vcombine(const V& a, const V& b) { return v_add(a, b); }
};

template<typename A> struct OpAvg : OpSum<A>
{
    static A finalize(A a, int64 n) { return n > 0 ? A(a/(double)n) : A(0); }
};

template<typename A> struct OpLogSum : OpSum<A>
{
    static A finalize(A a, int64) { return A(std::log(a)); }
};

template<typename A> struct OpSum2 : OpSum<A>
{
    static A apply(A a, A x) { return a + x*x; }
    template<typename V> static V vapply(const V& a, const V& x) { return v_add(a, v_mul(x, x)); }
};

template<typename A> struct OpL2 : OpSum2<A>
{
    static A finalize(A a, int64) { return A(std::sqrt(a)); }
};

template<typename A> struct OpL1 : OpSum<A>
{
    static A apply(A a, A x) { return a + (x < A(0) ? -x : x); }
    template<typename V> static V vapply(const V& a, const V& x) { return v_add(a, v_abs(x)); }
};

template<typename A> struct OpProd
{
    static A init() { return A(1); }
    static A apply(A a, A x) { return a*x; }
    static A combine(A a, A b) { return a*b; }
    static A finalize(A a, int64) { return a; }
    template<typename V> static V vapply(const V& a, const V& x) { return v_mul(a, x); }
    template<typename V> static V vcombine(const V& a, const V& b) { return v_mul(a, b); }
};

template<typename A> struct OpMax
{
    static A init() { return std::numeric_limits<A>::has_infinity ? -std::numeric_limits<A>::infinity()
                                                                   : std::numeric_limits<A>::lowest(); }
    static A apply(A a, A x) { return a < x ? x : a; }
    static A combine(A a, A b) { return a < b ? b : a; }
    static A finalize(A a, int64) { return a; }
    template<typename V> static V vapply(const V& a, const V& x) { return v_max(a, x); }
    template<typename V> static V vcombine(const V& a, const V& b) { return v_max(a, b); }
};

template<typename A> struct OpMin
{
    static A init() { return std::numeric_limits<A>::has_infinity ? std::numeric_limits<A>::infinity()
                                                                   : std::numeric_limits<A>::max(); }
    static A apply(A a, A x) { return x < a ? x : a; }
    static A combine(A a, A b) { return b < a ? b : a; }
    static A finalize(A a, int64) { return a; }
    template<typename V> static V vapply(const V& a, const V& x) { return v_min(a, x); }
    template<typename V> static V vcombine(const V& a, const V& b) { return v_min(a, b); }
};

// second pass of log-sum-exp: sum(exp(x - m)), then m + log(sum), where m is the maximum
template<typename A> struct OpSumExpShifted
{
    static A init() { return A(0); }
    static A apply(A a, A x, A m) { return a + A(std::exp(x - m)); }
    static A combine(A a, A b) { return a + b; }
    static A finalize(A a, A m) { return std::isinf(m) ? m : m + A(std::log(a)); }
};

//////////////////////////////// row kernels ////////////////////////////////

// horizontal: fold n contiguous elements into a scalar accumulator
template<class Op, typename A, bool vec = (VecT<A>::ok != 0)> struct HRow
{
    static A run(A acc, const A* x, int64 n)
    {
        for (int64 i = 0; i < n; i++)
            acc = Op::apply(acc, x[i]);
        return acc;
    }
};

// vertical: acc[i] = apply(acc[i], x[i])
template<class Op, typename A, bool vec = (VecT<A>::ok != 0)> struct VRow
{
    static void run(A* acc, const A* x, int64 n)
    {
        for (int64 i = 0; i < n; i++)
            acc[i] = Op::apply(acc[i], x[i]);
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)
template<class Op, typename A> struct HRow<Op, A, true>
{
    static A run(A acc, const A* x, int64 n)
    {
        typedef typename VecT<A>::vtype V;
        const int L = VTraits<V>::vlanes();
        int64 i = 0;
        if (n >= 2*L)
        {
            A buf[VTraits<V>::max_nlanes];
            for (int k = 0; k < L; k++)
                buf[k] = Op::init();
            V a0 = vx_load(buf), a1 = a0;
            for (; i + 2*L <= n; i += 2*L)
            {
                a0 = Op::vapply(a0, vx_load(x + i));
                a1 = Op::vapply(a1, vx_load(x + i + L));
            }
            v_store(buf, Op::vcombine(a0, a1));
            for (int k = 0; k < L; k++)
                acc = Op::combine(acc, buf[k]);
        }
        for (; i < n; i++)
            acc = Op::apply(acc, x[i]);
        return acc;
    }
};

template<class Op, typename A> struct VRow<Op, A, true>
{
    static void run(A* acc, const A* x, int64 n)
    {
        typedef typename VecT<A>::vtype V;
        const int L = VTraits<V>::vlanes();
        int64 i = 0;
        for (; i + L <= n; i += L)
            v_store(acc + i, Op::vapply(vx_load(acc + i), vx_load(x + i)));
        for (; i < n; i++)
            acc[i] = Op::apply(acc[i], x[i]);
    }
};
#endif

//////////////////////////////// plan ////////////////////////////////

struct Plan
{
    // kept axes (outer first); ksstep in bytes, kdstep in result elements
    int nk;
    int ksize[RND_MAX_DIMS];
    ptrdiff_t ksstep[RND_MAX_DIMS], kdstep[RND_MAX_DIMS];
    // reduced axes (outer first), steps in bytes
    int nr;
    int rsize[RND_MAX_DIMS];
    ptrdiff_t rsstep[RND_MAX_DIMS];
    bool innerReduced;      // the innermost axis is reduced (horizontal mode)

    const uchar* src;
    int sdepth, adepth;     // source depth and accumulator depth
    size_t sesz1;
    int64 count;            // reduced elements per output
    int64 nout;             // number of outputs
    BinaryFunc cvt;         // source -> accumulator conversion, or nullptr
};

// iterator over the flat index space of some axes, tracking a byte offset and an element offset
struct Iter
{
    int n = 0;
    const int* size = nullptr;
    const ptrdiff_t* s = nullptr;
    const ptrdiff_t* d = nullptr;
    int64 idx[RND_MAX_DIMS];
    ptrdiff_t soff = 0, doff = 0;

    void init(int n_, const int* size_, const ptrdiff_t* s_, const ptrdiff_t* d_, int64 flat)
    {
        n = n_; size = size_; s = s_; d = d_;
        soff = doff = 0;
        for (int k = n - 1; k >= 0; k--)
        {
            int64 q = flat / size[k];
            idx[k] = flat - q*size[k];
            soff += (ptrdiff_t)idx[k]*s[k];
            if (d)
                doff += (ptrdiff_t)idx[k]*d[k];
            flat = q;
        }
    }

    void next()
    {
        for (int k = n - 1; k >= 0; k--)
        {
            soff += s[k];
            if (d)
                doff += d[k];
            if (++idx[k] < size[k])
                return;
            soff -= (ptrdiff_t)size[k]*s[k];
            if (d)
                doff -= (ptrdiff_t)size[k]*d[k];
            idx[k] = 0;
        }
    }
};

// Returns `len` contiguous accumulator-typed elements taken from the source at `p` with byte step `step`.
template<typename A> static const A* loadRow(const Plan& pl, const uchar* p, ptrdiff_t step, int64 len,
                                             A* buf, uchar* tmp)
{
    if (step != (ptrdiff_t)pl.sesz1)
    {
        for (int64 i = 0; i < len; i++, p += step)
            memcpy(tmp + i*pl.sesz1, p, pl.sesz1);
        p = tmp;
    }
    if (!pl.cvt)
        return (const A*)p;
    pl.cvt(p, 0, 0, 0, (uchar*)buf, 0, Size((int)len, 1), 0);
    return buf;
}

// Row loop with conversion blocking: calls f(x, len) on consecutive pieces of the row.
template<typename A, typename F> static void forRow(const Plan& pl, const uchar* p, ptrdiff_t step,
                                                    int64 n, A* buf, uchar* tmp, F f)
{
    const bool direct = !pl.cvt && step == (ptrdiff_t)pl.sesz1;
    const int64 blk = direct ? n : (int64)RND_BLOCK;
    for (int64 e = 0; e < n; e += blk)
    {
        int64 len = std::min(blk, n - e);
        f(loadRow<A>(pl, p + e*step, step, len, buf, tmp), len);
    }
}

//////////////////////////////// drivers ////////////////////////////////

// Parallel-split bookkeeping shared by both modes: nsplit partial results per output.
struct Work
{
    int64 nitems, nsplit;
    double nstripes;
    bool serial;
};

// Horizontal mode: outputs are the kept axes, each one folds `count` elements; the flat reduced
// index space [0, count) is split into nsplit ranges when there are too few outputs.
template<typename A, class Op, class OpM> static void runHorizontal(const Plan& pl, const Work& w, A* res,
                                                                   const A* mbuf, A* partial)
{
    const int nr = pl.nr;
    const int64 n = pl.rsize[nr - 1];                 // innermost reduced axis
    const ptrdiff_t rstep = pl.rsstep[nr - 1];
    const int64 nsplit = w.nsplit;

    auto body = [&](const Range& range)
    {
        AutoBuffer<A> bufA(RND_BLOCK);
        AutoBuffer<uchar> bufT(RND_BLOCK*pl.sesz1);
        Iter ko, ro;
        int64 it = range.start;
        if (nsplit == 1)
            ko.init(pl.nk, pl.ksize, pl.ksstep, pl.kdstep, it);
        while (it < range.end)
        {
            const int64 o = it / nsplit, s = it - o*nsplit;
            if (nsplit > 1)
                ko.init(pl.nk, pl.ksize, pl.ksstep, pl.kdstep, o);
            const uchar* base = pl.src + ko.soff;
            const A m = mbuf ? mbuf[ko.doff] : A(0);
            int64 a = s*pl.count/nsplit, b = (s + 1)*pl.count/nsplit;
            A acc = Op::init();
            if (a < b)
            {
                ro.init(nr - 1, pl.rsize, pl.rsstep, nullptr, a / n);
                int64 e = a % n;
                while (a < b)
                {
                    int64 len = std::min(n - e, b - a);
                    forRow<A>(pl, base + ro.soff + e*rstep, rstep, len, bufA.data(), bufT.data(),
                              [&](const A* x, int64 l) { acc = OpM::run(acc, x, l, m); });
                    a += len;
                    e = 0;
                    ro.next();
                }
            }
            if (nsplit == 1)
            {
                res[ko.doff] = OpM::finalize(acc, pl.count, m);
                ko.next();
            }
            else
                partial[s*pl.nout + ko.doff] = acc;
            it++;
        }
    };

    if (w.serial)
        body(Range(0, (int)w.nitems));
    else
        parallel_for_(Range(0, (int)w.nitems), body, w.nstripes);
}

// Vertical mode: the innermost axis is kept; each item accumulates whole rows of a chunk of it.
template<typename A, class Op, class OpM> static void runVertical(const Plan& pl, const Work& w, A* res,
                                                                 const A* mbuf, A* partial)
{
    const int nk = pl.nk;
    const int64 m = pl.ksize[nk - 1];
    const ptrdiff_t kstep = pl.ksstep[nk - 1], kdstep = pl.kdstep[nk - 1];
    const int64 nch = (m + RND_VCHUNK - 1)/RND_VCHUNK;
    const int64 nsplit = w.nsplit;
    int64 rcount = 1;
    for (int i = 0; i < pl.nr; i++)
        rcount *= pl.rsize[i];

    auto body = [&](const Range& range)
    {
        AutoBuffer<A> bufA(RND_BLOCK), accBuf(RND_VCHUNK);
        AutoBuffer<uchar> bufT(RND_BLOCK*pl.sesz1);
        A* acc = accBuf.data();
        Iter ko, ro;
        for (int64 it = range.start; it < range.end; it++)
        {
            int64 s = it % nsplit, q = it / nsplit;
            int64 ch = q % nch, o = q / nch;
            ko.init(nk - 1, pl.ksize, pl.ksstep, pl.kdstep, o);
            const int64 e0 = ch*RND_VCHUNK, len = std::min((int64)RND_VCHUNK, m - e0);
            const ptrdiff_t soff = ko.soff + (ptrdiff_t)e0*kstep, doff = ko.doff + (ptrdiff_t)e0*kdstep;
            const A* mrow = mbuf ? mbuf + doff : nullptr;   // kdstep == 1 for the innermost kept axis
            for (int64 i = 0; i < len; i++)
                acc[i] = Op::init();

            int64 a = s*rcount/nsplit, b = (s + 1)*rcount/nsplit;
            if (a < b)
            {
                ro.init(pl.nr, pl.rsize, pl.rsstep, nullptr, a);
                for (; a < b; a++, ro.next())
                {
                    const uchar* p = pl.src + soff + ro.soff;
                    const A* x = loadRow<A>(pl, p, kstep, len, bufA.data(), bufT.data());
                    OpM::vrun(acc, x, len, mrow);
                }
            }
            if (nsplit == 1)
                for (int64 i = 0; i < len; i++)
                    res[doff + i] = OpM::finalize(acc[i], pl.count, mrow ? mrow[i] : A(0));
            else
                for (int64 i = 0; i < len; i++)
                    partial[s*pl.nout + doff + i] = acc[i];
        }
    };

    if (w.serial)
        body(Range(0, (int)w.nitems));
    else
        parallel_for_(Range(0, (int)w.nitems), body, w.nstripes);
}

// Adapters that give the plain operations and the shifted-exp pass one interface.
template<typename A, class Op> struct Plain
{
    static A run(A acc, const A* x, int64 n, A) { return HRow<Op, A>::run(acc, x, n); }
    static void vrun(A* acc, const A* x, int64 n, const A*) { VRow<Op, A>::run(acc, x, n); }
    static A finalize(A a, int64 count, A) { return Op::finalize(a, count); }
};

template<typename A> struct ShiftedExp
{
    typedef OpSumExpShifted<A> Op;
    static A run(A acc, const A* x, int64 n, A m)
    {
        for (int64 i = 0; i < n; i++)
            acc = Op::apply(acc, x[i], m);
        return acc;
    }
    static void vrun(A* acc, const A* x, int64 n, const A* m)
    {
        for (int64 i = 0; i < n; i++)
            acc[i] = Op::apply(acc[i], x[i], m[i]);
    }
    static A finalize(A a, int64, A m) { return Op::finalize(a, m); }
};

template<typename A, class Op, class OpM> static void runPass(const Plan& pl, A* res, const A* mbuf)
{
    const int nthreads = getNumThreads();
    const double work = (double)pl.nout*(double)std::max(pl.count, (int64)1);
    Work w;
    w.serial = work < RND_SERIAL_WORK || nthreads <= 1;
    w.nsplit = 1;
    int64 items;
    if (pl.innerReduced)
        items = pl.nout;
    else
    {
        int64 nouter = pl.nout/pl.ksize[pl.nk - 1];
        items = nouter*((pl.ksize[pl.nk - 1] + RND_VCHUNK - 1)/RND_VCHUNK);
    }
    // too few outputs to keep all the threads busy: split the reduction as well
    const int64 want = 4*(int64)nthreads;
    if (!w.serial && items < want && pl.count > 1)
    {
        int64 rows = pl.innerReduced ? pl.count/4096 : pl.count/16;
        w.nsplit = std::max((int64)1, std::min((want + items - 1)/items, std::min(rows, (int64)256)));
        w.nsplit = std::min(w.nsplit, std::max((int64)1, (int64)(4 << 20)/std::max(pl.nout, (int64)1)));
    }
    w.nitems = items*w.nsplit;
    w.nstripes = std::min((double)w.nitems, std::max(1.0, work/RND_STRIPE_WORK));
    CV_Assert(w.nitems <= INT_MAX);

    AutoBuffer<A> partial(w.nsplit > 1 ? (size_t)(w.nsplit*pl.nout) : 1);
    if (pl.innerReduced)
        runHorizontal<A, Op, OpM>(pl, w, res, mbuf, partial.data());
    else
        runVertical<A, Op, OpM>(pl, w, res, mbuf, partial.data());

    if (w.nsplit > 1)
    {
        for (int64 o = 0; o < pl.nout; o++)
        {
            A acc = partial[o];
            for (int64 s = 1; s < w.nsplit; s++)
                acc = Op::combine(acc, partial[s*pl.nout + o]);
            res[o] = OpM::finalize(acc, pl.count, mbuf ? mbuf[o] : A(0));
        }
    }
}

template<typename A> static void fillResult(A* res, int64 n, A v)
{
    for (int64 i = 0; i < n; i++)
        res[i] = v;
}

template<typename A> static void runOp(const Plan& pl, int op, A* res)
{
    if (pl.count == 0)
    {
        // empty reduction: the identity of the operation
        A v = op == REDUCE_MAX ? OpMax<A>::init() : op == REDUCE_MIN ? OpMin<A>::init() :
              op == REDUCE_PROD ? A(1) :
              op == REDUCE_LOG_SUM || op == REDUCE_LOG_SUM_EXP ? OpMax<A>::init() : A(0);
        fillResult(res, pl.nout, v);
        return;
    }
    switch (op)
    {
    case REDUCE_SUM:   runPass<A, OpSum<A>, Plain<A, OpSum<A> > >(pl, res, nullptr); break;
    case REDUCE_AVG:   runPass<A, OpAvg<A>, Plain<A, OpAvg<A> > >(pl, res, nullptr); break;
    case REDUCE_SUM2:  runPass<A, OpSum2<A>, Plain<A, OpSum2<A> > >(pl, res, nullptr); break;
    case REDUCE_L1:    runPass<A, OpL1<A>, Plain<A, OpL1<A> > >(pl, res, nullptr); break;
    case REDUCE_L2:    runPass<A, OpL2<A>, Plain<A, OpL2<A> > >(pl, res, nullptr); break;
    case REDUCE_PROD:  runPass<A, OpProd<A>, Plain<A, OpProd<A> > >(pl, res, nullptr); break;
    case REDUCE_LOG_SUM: runPass<A, OpLogSum<A>, Plain<A, OpLogSum<A> > >(pl, res, nullptr); break;
    case REDUCE_LOG_SUM_EXP:
    {
        AutoBuffer<A> mbuf((size_t)pl.nout);
        runPass<A, OpMax<A>, Plain<A, OpMax<A> > >(pl, mbuf.data(), nullptr);
        runPass<A, OpSumExpShifted<A>, ShiftedExp<A> >(pl, res, mbuf.data());
        break;
    }
    default:
        CV_Error(Error::StsBadArg, "reduceND: unknown reduction operation");
    }
}

template<typename A> static void runMinMax(const Plan& pl, int op, A* res)
{
    if (pl.count == 0)
        fillResult(res, pl.nout, op == REDUCE_MAX ? OpMax<A>::init() : OpMin<A>::init());
    else if (op == REDUCE_MAX)
        runPass<A, OpMax<A>, Plain<A, OpMax<A> > >(pl, res, nullptr);
    else
        runPass<A, OpMin<A>, Plain<A, OpMin<A> > >(pl, res, nullptr);
}

static void dispatch(const Plan& pl, int op, uchar* res)
{
    const bool minmax = op == REDUCE_MAX || op == REDUCE_MIN;
    switch (pl.adepth)
    {
    case CV_32F:
        if (minmax) runMinMax<float>(pl, op, (float*)res); else runOp<float>(pl, op, (float*)res);
        break;
    case CV_64F:
        if (minmax) runMinMax<double>(pl, op, (double*)res); else runOp<double>(pl, op, (double*)res);
        break;
    case CV_8U:  runMinMax<uchar>(pl, op, (uchar*)res); break;
    case CV_8S:  runMinMax<schar>(pl, op, (schar*)res); break;
    case CV_16U: runMinMax<ushort>(pl, op, (ushort*)res); break;
    case CV_16S: runMinMax<short>(pl, op, (short*)res); break;
    case CV_32U: runMinMax<unsigned>(pl, op, (unsigned*)res); break;
    case CV_32S: runMinMax<int>(pl, op, (int*)res); break;
    case CV_64U: runMinMax<uint64_t>(pl, op, (uint64_t*)res); break;
    case CV_64S: runMinMax<int64_t>(pl, op, (int64_t*)res); break;
    default:
        CV_Error(Error::StsInternal, "reduceND: unexpected accumulator depth");
    }
}

// accumulator depth for the given source/destination depths and operation
static int accumulatorDepth(int sdepth, int ddepth, int op)
{
    if (op == REDUCE_MAX || op == REDUCE_MIN)
    {
        if (sdepth == CV_16F || sdepth == CV_16BF)
            return CV_32F;
        return sdepth == CV_Bool ? CV_8U : sdepth;
    }
    const bool wide = sdepth == CV_32S || sdepth == CV_32U || sdepth == CV_64S || sdepth == CV_64U ||
                      sdepth == CV_64F || ddepth == CV_64F;
    return wide ? CV_64F : CV_32F;
}

} // namespace

void reduceND(InputArray _src, OutputArray _dst, const std::vector<int>& axes_, int op,
              bool keepdims, int dtype)
{
    CV_INSTRUMENT_REGION();

    CV_Check(op, op >= REDUCE_SUM && op <= REDUCE_LOG_SUM_EXP, "reduceND: unknown reduction operation");

    Mat src = _src.getMat();
    const int dims = src.dims, cn = src.channels();
    const int sdepth = src.depth();
    int ddepth = dtype < 0 ? sdepth : CV_MAT_DEPTH(dtype);
    if ((op == REDUCE_MAX || op == REDUCE_MIN) && sdepth == CV_Bool && ddepth != CV_Bool)
        CV_Error(Error::StsBadArg, "reduceND: the output of max/min over bool must be bool");

    // which axes are reduced; empty means all of them
    bool reduced[CV_MAX_DIM] = {false};
    if (axes_.empty())
        std::fill(reduced, reduced + dims, true);
    for (int a : axes_)
    {
        CV_CheckGE(a, -dims, "reduceND: axis is out of range");
        CV_CheckLT(a, dims, "reduceND: axis is out of range");
        int ax = a < 0 ? a + dims : a;
        CV_Check(a, !reduced[ax], "reduceND: duplicate axis");
        reduced[ax] = true;
    }

    int keepShape[CV_MAX_DIM], outShape[CV_MAX_DIM], nout = 0;
    for (int i = 0; i < dims; i++)
    {
        keepShape[i] = reduced[i] ? 1 : src.size[i];
        if (!reduced[i] || keepdims)
            outShape[nout++] = keepShape[i];
    }
    const int dtype_ = CV_MAKETYPE(ddepth, cn);
    _dst.create(nout, outShape, dtype_);
    Mat dst = _dst.getMat();

    // The results are computed into a contiguous accumulator-typed buffer laid out like the
    // keepdims output (channels innermost), then converted into dst if needed.
    Plan pl;
    pl.src = src.data;
    pl.sdepth = sdepth == CV_Bool ? CV_8U : sdepth;
    pl.adepth = accumulatorDepth(sdepth, ddepth, op);
    pl.sesz1 = CV_ELEM_SIZE1(sdepth);
    pl.cvt = pl.sdepth == pl.adepth ? nullptr : getConvertFunc(pl.sdepth, pl.adepth);
    CV_Assert(pl.sdepth == pl.adepth || pl.cvt);

    // gather the axes (+ the channel axis), outer first
    int n = 0;
    int size[RND_MAX_DIMS];
    ptrdiff_t sstep[RND_MAX_DIMS], dstep[RND_MAX_DIMS];
    bool red[RND_MAX_DIMS];
    ptrdiff_t dcur = cn;
    ptrdiff_t dsteps[CV_MAX_DIM];
    for (int i = dims - 1; i >= 0; i--)
    {
        dsteps[i] = dcur;
        dcur *= keepShape[i];
    }
    pl.count = 1;
    for (int i = 0; i < dims; i++)
    {
        if (reduced[i])
            pl.count *= src.size[i];
        size[n] = src.size[i];
        sstep[n] = (ptrdiff_t)src.step[i];
        dstep[n] = reduced[i] ? 0 : dsteps[i];
        red[n] = reduced[i];
        n++;
    }
    if (cn > 1)
    {
        size[n] = cn;
        sstep[n] = (ptrdiff_t)pl.sesz1;
        dstep[n] = 1;
        red[n] = false;
        n++;
    }

    pl.nout = (int64)dst.total()*cn;
    const bool direct = pl.adepth == ddepth && dst.isContinuous();
    Mat resMat;
    uchar* res = dst.data;
    if (!direct)
    {
        resMat.create(1, (int)std::max(pl.nout, (int64)1), CV_MAKETYPE(pl.adepth, 1));
        res = resMat.data;
    }
    if (pl.nout == 0)
        return;

    // drop size-1 axes, then merge neighbours of the same kind that are contiguous in both views
    int m = 0;
    for (int i = 0; i < n; i++)
    {
        if (size[i] == 0)
        {
            // empty source along this axis
            if (red[i])
            {
                pl.count = 0;
                continue;
            }
            return;
        }
        if (size[i] == 1)
            continue;
        if (m > 0 && red[m-1] == red[i] && sstep[m-1] == sstep[i]*size[i] &&
            dstep[m-1] == dstep[i]*size[i])
        {
            size[m-1] *= size[i];
            sstep[m-1] = sstep[i];
            dstep[m-1] = dstep[i];
        }
        else
        {
            size[m] = size[i];
            sstep[m] = sstep[i];
            dstep[m] = dstep[i];
            red[m] = red[i];
            m++;
        }
    }
    if (m == 0)
    {
        // a single element
        size[0] = 1;
        sstep[0] = (ptrdiff_t)pl.sesz1;
        dstep[0] = 1;
        red[0] = false;
        m = 1;
    }
    pl.innerReduced = red[m-1];
    pl.nk = pl.nr = 0;
    for (int i = 0; i < m; i++)
    {
        if (red[i])
        {
            pl.rsize[pl.nr] = size[i];
            pl.rsstep[pl.nr] = sstep[i];
            pl.nr++;
        }
        else
        {
            pl.ksize[pl.nk] = size[i];
            pl.ksstep[pl.nk] = sstep[i];
            pl.kdstep[pl.nk] = dstep[i];
            pl.nk++;
        }
    }
    if (pl.nk == 0)
    {
        pl.ksize[0] = 1;
        pl.ksstep[0] = 0;
        pl.kdstep[0] = 0;
        pl.nk = 1;
    }

    dispatch(pl, op, res);

    if (!direct)
    {
        Mat r(dst.dims, dst.size.p, CV_MAKETYPE(pl.adepth, cn), resMat.data);
        if (dst.dims == 0)
            r = Mat(0, nullptr, CV_MAKETYPE(pl.adepth, cn), resMat.data);
        r.convertTo(dst, ddepth);
    }
}

} // namespace cv
