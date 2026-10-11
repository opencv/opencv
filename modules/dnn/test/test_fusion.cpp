// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include "../src/adjacency_graph.hpp"
#include "../src/layers/cpu_kernels/fusion_apply.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;

static std::string _tf(const std::string& filename, bool required = true)
{
    return findDataFile(std::string("dnn/onnx/") + filename, required);
}

static const std::vector<const float*> kNoBufs;

// A standalone graph from one layer's math. Only tests build graphs this way;
// the pass always goes through a shared arena and fusion::extract.
static Ptr<AdjacencyGraph> graphOf(const LayerMath& m)
{
    AdjacencyGraphBuilder b;
    const int in = b.internNode(FusionEltwiseOp::INPUT, {});
    const int root = fusion::instantiate(b, in, m);
    if (root < 0)
        return Ptr<AdjacencyGraph>();
    return fusion::extract(b.graph(), root, std::vector<Mat>());
}

// A layer states its math through the table registered for its type, not a virtual,
// so tests ask for it exactly the way the pass does.
static bool unfoldOf(const Ptr<Layer>& l, LayerMath& m, const ConstOperand& side)
{
    const FusionOps* ops = fusionOpsFor(l.get());
    return ops && ops->unfold && ops->unfold(l.get(), m, side);
}

static float eval1(const LayerMath& r, float x)
{
    Ptr<AdjacencyGraph> g = graphOf(r);
    CV_Assert(g);
    return fusion::evalElement(*g, x, kNoBufs, 0);
}

TEST(Fusion, IdenticalMathCollapsesToTheSameNode)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath r;
    const int zero = r.constant(0.f);
    r.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, zero);

    EXPECT_EQ(fusion::instantiate(arena, in, r), fusion::instantiate(arena, in, r));
    EXPECT_EQ(fusion::instantiate(arena, in, r), fusion::instantiate(arena, in, r));
    EXPECT_EQ(3u, arena.size());

    AdjacencyGraphBuilder b;
    const int i2 = b.internNode(FusionEltwiseOp::INPUT, {});
    const int k = b.internNode(FusionEltwiseOp::CONST, {}, 7.f);
    EXPECT_EQ(b.internNode(FusionEltwiseOp::ADD, {i2, k}), b.internNode(FusionEltwiseOp::ADD, {k, i2}));
    EXPECT_NE(b.internNode(FusionEltwiseOp::SUB, {i2, k}), b.internNode(FusionEltwiseOp::SUB, {k, i2}));
}

TEST(Fusion, ConeIsBoundedIndependentlyOfArenaSize)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath a, b;
    const int zero = a.constant(0.f);
    a.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, zero);
    fusion::detail::gelu(b);
    const int rootA = fusion::instantiate(arena, in, a);
    const int rootB = fusion::instantiate(arena, in, b);
    ASSERT_GE(rootA, 0);
    ASSERT_GE(rootB, 0);

    std::vector<char> scratch;
    EXPECT_EQ(3, fusion::detail::markLive(arena.graph(), rootA, scratch));
    EXPECT_EQ(9, fusion::detail::markLive(arena.graph(), rootB, scratch));
    EXPECT_GT(arena.size(), (size_t)9);
}

TEST(Fusion, ExtractionYieldsAStandaloneGraph)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath a, b;
    const int zero = a.constant(0.f);
    a.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, zero);
    fusion::detail::gelu(b);
    fusion::instantiate(arena, in, a);
    const int rootB = fusion::instantiate(arena, in, b);

    Ptr<AdjacencyGraph> g = fusion::extract(arena.graph(), rootB, std::vector<Mat>());
    ASSERT_TRUE(g);
    EXPECT_EQ(9u, g->size());
    EXPECT_EQ(FusionEltwiseOp::INPUT, g->nodes()[0].op);
    EXPECT_EQ(FusionEltwiseOp::MUL, g->nodes()[g->outputNode()].op);
    EXPECT_NEAR(0.5f * 1.5f * (1.f + std::erf(1.5f * 0.70710678118654752440f)),
                fusion::evalElement(*g, 1.5f, kNoBufs, 0), 1e-5);

    EXPECT_FALSE(fusion::extract(arena.graph(), -1, std::vector<Mat>()));
    EXPECT_FALSE(fusion::extract(arena.graph(), (int)arena.size(), std::vector<Mat>()));
}

TEST(Fusion, OverLimitConeIsRefusedNotEvaluated)
{
    AdjacencyGraphBuilder arena;
    int cur = arena.internNode(FusionEltwiseOp::INPUT, {});
    std::vector<char> scratch;
    int steps = 0;
    while (fusion::detail::markLive(arena.graph(), cur, scratch) <= FUSION_MAX_EXPR_NODES && steps < 200) {
        LayerMath r;
        fusion::detail::gelu(r);
        const int next = fusion::instantiate(arena, cur, r);
        ASSERT_GE(next, 0);
        cur = next;
        steps++;
    }
    ASSERT_GT(fusion::detail::markLive(arena.graph(), cur, scratch), FUSION_MAX_EXPR_NODES);
    EXPECT_FALSE(fusion::extract(arena.graph(), cur, std::vector<Mat>()));
}

TEST(Fusion, MathMatchesClosedForm)
{
    const float xs[] = { -3.f, -0.5f, 0.f, 0.25f, 1.f, 4.f };
    LayerMath r;
    for (float x : xs) {
        r = LayerMath();
        r.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, r.constant(0.f));
        EXPECT_FLOAT_EQ(std::max(x, 0.f), eval1(r, x)) << "relu " << x;

        r = LayerMath();
        r.clamp(LayerMath::INPUT_VALUE, 0.f, 6.f);
        EXPECT_FLOAT_EQ(std::min(std::max(x, 0.f), 6.f), eval1(r, x)) << "clip " << x;

        r = LayerMath(); fusion::detail::sigmoid(r);
        EXPECT_NEAR(1.f / (1.f + std::exp(-x)), eval1(r, x), 1e-5) << "sigmoid " << x;

        r = LayerMath(); fusion::detail::gelu(r);
        EXPECT_NEAR(0.5f * x * (1.f + std::erf(x * 0.70710678118654752440f)), eval1(r, x), 1e-5)
            << "gelu " << x;

        r = LayerMath();
        r.unary(FusionEltwiseOp::TANH, LayerMath::INPUT_VALUE);
        EXPECT_NEAR(std::tanh(x), eval1(r, x), 1e-6) << "tanh " << x;

        r = LayerMath();
        const int scaled = r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.constant(2.f));
        r.unary(FusionEltwiseOp::EXP, r.binary(FusionEltwiseOp::ADD, scaled, r.constant(5.f)));
        EXPECT_NEAR(std::exp(2.f * x + 5.f), eval1(r, x), 1e-2) << "scaled exp " << x;
    }
}

TEST(Fusion, DeclaredMathMatchesTheActivation)
{
    const float kSqrt2Pi = 0.7978845834732056f;
    const float kCoef    = 0.044714998453855515f * kSqrt2Pi;
    const float kSeluA   = 1.67326319217681884765625f;
    const float kSeluG   = 1.05070102214813232421875f;

    const float xs[] = { -3.f, -0.5f, 0.f, 0.25f, 1.f, 4.f };
    for (float x : xs) {
        struct { const char* type; float expected; } cases[] = {
            { "Swish",       x / (1.f + std::exp(-x)) },
            { "ELU",         x >= 0.f ? x : std::exp(x) - 1.f },
            { "AbsVal",      std::abs(x) },
            { "HardSwish",   x * std::max(0.f, std::min(1.f, x / 6.f + 0.5f)) },
            { "Softsign",    x / (1.f + std::abs(x)) },
            { "HardSigmoid", std::max(0.f, std::min(1.f, 0.2f * x + 0.5f)) },
            { "Celu",        std::max(0.f, x) + std::min(0.f, std::expm1(x)) },
            { "Selu",        kSeluG * (x > 0.f ? x : kSeluA * std::expm1(x)) },
            { "GeluApproximation",
              0.5f * x * (1.f + std::tanh(x * (kSqrt2Pi + kCoef * x * x))) },
        };

        for (const auto& c : cases) {
            LayerParams lp;
            Ptr<Layer> l = LayerFactory::createLayerInstance(c.type, lp);
            ASSERT_TRUE(l) << c.type;
            l->inputs.assign(1, Arg(1));

            LayerMath m;
            ConstOperand side;
            ASSERT_TRUE(unfoldOf(l, m, side)) << c.type << " declared no math";
            EXPECT_NEAR(c.expected, eval1(m, x), 1e-5) << c.type << " at x=" << x;
        }
    }
}

TEST(Fusion, CeluWithNegativeAlphaIsRefused)
{
    LayerParams lp;
    lp.set("alpha", -1.f);
    Ptr<Layer> l = LayerFactory::createLayerInstance("Celu", lp);
    ASSERT_TRUE(l);
    l->inputs.assign(1, Arg(1));

    LayerMath m;
    ConstOperand side;
    EXPECT_FALSE(unfoldOf(l, m, side));
}

TEST(Fusion, EmptyMathIsRefused)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    EXPECT_EQ(-1, fusion::instantiate(arena, in, LayerMath()));
    EXPECT_EQ(1u, arena.size());
}

TEST(Fusion, ReversedSubIsRefused)
{
    LayerParams lp;
    lp.set("operation", "sub");
    Ptr<Layer> sub = NaryEltwiseLayer::create(lp);
    ASSERT_TRUE(sub);
    sub->inputs.assign(2, Arg());

    LayerMath r;
    ConstOperand vs;
    vs.count = 1;
    vs.consts[0].value = 3.f;
    vs.flowIsFirstInput = false;
    EXPECT_FALSE(unfoldOf(sub, r, vs));

    vs.flowIsFirstInput = true;
    r = LayerMath();
    ASSERT_TRUE(unfoldOf(sub, r, vs));
    EXPECT_FLOAT_EQ(-1.f, eval1(r, 2.f));
}

TEST(Fusion, VariadicNaryIsRefused)
{
    LayerParams lp;
    lp.set("operation", "sum");
    Ptr<Layer> sum = NaryEltwiseLayer::create(lp);
    ASSERT_TRUE(sum);

    LayerMath r;
    ConstOperand vs;
    vs.count = 1;
    vs.consts[0].value = 3.f;

    sum->inputs.assign(3, Arg());
    EXPECT_FALSE(unfoldOf(sum, r, vs));

    r = LayerMath();
    sum->inputs.assign(2, Arg());
    EXPECT_TRUE(unfoldOf(sum, r, vs));
}

TEST(Fusion, ClipWithOneDynamicBoundIsRefused)
{
    LayerParams lp;
    Ptr<Layer> clip = ClipLayer::create(lp);
    ASSERT_TRUE(clip);

    LayerMath r;
    ConstOperand vs;
    vs.count = 2;
    vs.consts[0].value = 2.f;
    vs.consts[1].value = 0.f;

    clip->inputs = { Arg(1), Arg(2) };
    EXPECT_FALSE(unfoldOf(clip, r, vs));

    // Clip(x, "", max): the omitted min is an empty Arg, not a missing one
    r = LayerMath();
    clip->inputs = { Arg(1), Arg(0), Arg(2) };
    EXPECT_FALSE(unfoldOf(clip, r, vs));

    r = LayerMath();
    vs.consts[1].value = 6.f;
    clip->inputs = { Arg(1), Arg(2), Arg(3) };
    ASSERT_TRUE(unfoldOf(clip, r, vs));
    EXPECT_FLOAT_EQ(2.f, eval1(r, 1.f));
    EXPECT_FLOAT_EQ(6.f, eval1(r, 9.f));
}

// Every layer now states which kernel computes its own math, so this goes through
// the real path: the layer fills LayerMath, and PreparedFusion picks the kernel up.
TEST(Fusion, LayersDeclareTheirOwnKernel)
{
    struct { const char* name; const char* type; int nInputs; } cases[] = {
        { "sigmoid", "Sigmoid", 1 },
        { "gelu",    "Gelu",    1 },
        { "tanh",    "TanH",    1 },
        { "relu",    "ReLU",    1 },
    };

    for (const auto& c : cases) {
        LayerParams lp;
        Ptr<Layer> l = LayerFactory::createLayerInstance(c.type, lp);
        ASSERT_TRUE(l) << c.name;
        l->inputs.assign(c.nInputs, Arg(1));

        LayerMath m;
        ConstOperand side;
        ASSERT_TRUE(unfoldOf(l, m, side)) << c.name;
        EXPECT_TRUE(m.kernel.fn != nullptr) << c.name << ": no kernel declared";

        Ptr<AdjacencyGraph> expr = graphOf(m);
        ASSERT_TRUE(expr) << c.name;
        expr->kernel = m.kernel;

        PreparedFusion pf;
        const bool took = pf.take(expr);
        EXPECT_TRUE(took) << c.name;
        EXPECT_TRUE(pf.activationFn != nullptr) << c.name << ": fell to the interpreter";
    }
}

// Clip and NaryEltwise are not ElementWiseLayers, so they declare explicitly rather
// than through the wrapper. They must end up on the same fast path.
TEST(Fusion, NonElementwiseLayersDeclareToo)
{
    LayerParams lp;
    Ptr<Layer> clip = ClipLayer::create(lp);
    ASSERT_TRUE(clip);
    clip->inputs = { Arg(1), Arg(2), Arg(3) };
    LayerMath cm;
    ConstOperand cs;
    cs.count = 2; cs.consts[0].value = 0.f; cs.consts[1].value = 6.f;
    ASSERT_TRUE(unfoldOf(clip, cm, cs));
    EXPECT_TRUE(cm.kernel.fn != nullptr) << "clip declared no kernel";
    EXPECT_EQ(2, cm.kernel.nparams);

    LayerParams np;
    np.set("operation", "max");
    Ptr<Layer> mx = NaryEltwiseLayer::create(np);
    ASSERT_TRUE(mx);
    mx->inputs.assign(2, Arg(1));
    LayerMath nm;
    ConstOperand ns;
    ns.count = 1; ns.consts[0].value = 0.f;
    ASSERT_TRUE(unfoldOf(mx, nm, ns));
    EXPECT_TRUE(nm.kernel.fn != nullptr) << "Max(x,0) declared no kernel";
}

TEST(Fusion, SubclassOfElementWiseLayerIsRegistered)
{
    LayerParams lp;
    Mat slope(1, 4, CV_32F);
    float* s = slope.ptr<float>();
    s[0] = 0.1f; s[1] = 0.2f; s[2] = 0.3f; s[3] = 0.4f;
    lp.blobs.push_back(slope);

    Ptr<Layer> prelu = ChannelsPReLULayer::create(lp);
    ASSERT_TRUE(prelu);
    prelu->inputs.assign(1, Arg(1));

    const FusionOps* ops = fusionOpsFor(prelu.get());
    ASSERT_TRUE(ops && ops->unfold) << "fusionOpsFor() missed the concrete type";

    ASSERT_TRUE(ops->ownedBuffers);
    std::vector<Mat> owned;
    ASSERT_TRUE(ops->ownedBuffers(prelu.get(), owned));
    ASSERT_EQ(1u, owned.size());

    LayerMath m;
    ConstOperand side;
    side.count = 1;
    side.consts[0].bufferId = 0;
    ASSERT_TRUE(ops->unfold(prelu.get(), m, side));
}

TEST(Fusion, ApplyTakesKernelPathThenAffinePath)
{
    LayerMath r;
    r.setKernel(cv::dnn::getActivationFunc(ACTIV_CLIP), { 0.f, 6.f });
    r.clamp(LayerMath::INPUT_VALUE, 0.f, 6.f);
    Ptr<AdjacencyGraph> ce = graphOf(r);
    ce->kernel = r.kernel;
    PreparedFusion kern;
    ASSERT_TRUE(kern.take(ce));
    ASSERT_TRUE(kern.activationFn != nullptr);

    int n = 5;
    Mat y(1, &n, CV_32F);
    const float src[] = { -2.f, 0.f, 3.f, 6.f, 9.f };
    std::copy(src, src + n, y.ptr<float>());
    kern.run(y);
    const float want[] = { 0.f, 0.f, 3.f, 6.f, 6.f };
    for (int i = 0; i < n; i++)
        EXPECT_FLOAT_EQ(want[i], y.ptr<float>()[i]) << "clip i=" << i;

    // A scalar scale and shift has no kernel but is taken and run as a row loop.
    r = LayerMath();
    const int scaled = r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.constant(2.f));
    r.binary(FusionEltwiseOp::ADD, scaled, r.constant(5.f));
    PreparedFusion affine;
    ASSERT_TRUE(affine.take(graphOf(r)));
    EXPECT_TRUE(affine.activationFn == nullptr);

    int sz[] = { 257, 384 };
    Mat z(2, sz, CV_32F);
    const int total = (int)z.total();
    for (int i = 0; i < total; i++)
        z.ptr<float>()[i] = (float)(i % 100);
    affine.run(z);
    for (int i = 0; i < total; i += 997)
        EXPECT_FLOAT_EQ(2.f * (float)(i % 100) + 5.f, z.ptr<float>()[i]) << "affine i=" << i;
}

TEST(Fusion, ExpressionWithoutKernelOrAffineFormIsRefused)
{
    // Evaluating it one value at a time would be slower than keeping it as its own layer.
    LayerMath r;
    r.unary(FusionEltwiseOp::SQRT, LayerMath::INPUT_VALUE);
    PreparedFusion f;
    EXPECT_FALSE(f.take(graphOf(r)));
    EXPECT_FALSE(f.expr);
}

TEST(Fusion, PerChannelConstIndexesTheLastAxis)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath r;
    r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.perChannelConstant(1));
    const int root = fusion::instantiate(arena, in, r);
    ASSERT_GE(root, 0);

    int one = 1, three = 3;
    Mat b0(1, &one, CV_32F);
    b0.ptr<float>()[0] = 1.f;
    Mat b1(1, &three, CV_32F);
    b1.ptr<float>()[0] = 2.f;
    b1.ptr<float>()[1] = 3.f;
    b1.ptr<float>()[2] = 4.f;

    std::vector<Mat> tooFew(1, b0);
    EXPECT_FALSE(fusion::extract(arena.graph(), root, tooFew));

    std::vector<Mat> bufs;
    bufs.push_back(b0);
    bufs.push_back(b1);
    Ptr<AdjacencyGraph> expr = fusion::extract(arena.graph(), root, bufs);
    ASSERT_TRUE(expr);
    bool seen = false;
    for (const FusionNode& nd : expr->nodes()) {
        if (nd.op == FusionEltwiseOp::PER_CHANNEL_CONST) {
            EXPECT_EQ(1, nd.constBufferId);
            seen = true;
        }
    }
    EXPECT_TRUE(seen);

    PreparedFusion fa;
    ASSERT_TRUE(fa.take(expr));
    EXPECT_TRUE(fa.activationFn == nullptr);

    int sz[] = { 2, 3 };
    Mat y(2, sz, CV_32F);
    for (int i = 0; i < 6; i++)
        y.ptr<float>()[i] = 1.f;
    fa.run(y);
    const float want[] = { 2.f, 3.f, 4.f, 2.f, 3.f, 4.f };
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(want[i], y.ptr<float>()[i]) << "i=" << i;
}

TEST(Fusion, PerChannelBufferIsRefusedByTheInterpreter)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath r;
    r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.perChannelConstant(0));
    const int root = fusion::instantiate(arena, in, r);
    ASSERT_GE(root, 0);

    int three = 3;
    Mat b(1, &three, CV_32F);
    b.ptr<float>()[0] = 2.f;
    b.ptr<float>()[1] = 3.f;
    b.ptr<float>()[2] = 4.f;
    const std::vector<Mat> bufs(1, b);

    Ptr<AdjacencyGraph> lastAxis = fusion::extract(arena.graph(), root, bufs);
    ASSERT_TRUE(lastAxis);
    PreparedFusion plain;
    EXPECT_TRUE(plain.take(lastAxis));

    Ptr<AdjacencyGraph> perChannel =
        fusion::extract(arena.graph(), root, bufs, std::vector<uchar>(1, 1));
    ASSERT_TRUE(perChannel);
    PreparedFusion channel;
    EXPECT_FALSE(channel.take(perChannel));
}

TEST(Fusion, SharedRootKeepsEachChainsOwnBuffers)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});

    LayerMath r;
    r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.perChannelConstant(0));

    const int rootA = fusion::instantiate(arena, in, r);
    const int rootB = fusion::instantiate(arena, in, r);
    ASSERT_GE(rootA, 0);
    EXPECT_EQ(rootA, rootB);

    int three = 3;
    Mat ba(1, &three, CV_32F), bb(1, &three, CV_32F);
    for (int i = 0; i < 3; i++) { ba.ptr<float>()[i] = 2.f; bb.ptr<float>()[i] = 10.f; }

    Ptr<AdjacencyGraph> ea = fusion::extract(arena.graph(), rootA, std::vector<Mat>(1, ba));
    Ptr<AdjacencyGraph> eb = fusion::extract(arena.graph(), rootB, std::vector<Mat>(1, bb));
    ASSERT_TRUE(ea);
    ASSERT_TRUE(eb);

    ASSERT_EQ(1u, ea->constBufs.size());
    ASSERT_EQ(1u, eb->constBufs.size());
    EXPECT_FLOAT_EQ(2.f,  ea->constBufs[0].ptr<float>()[0]);
    EXPECT_FLOAT_EQ(10.f, eb->constBufs[0].ptr<float>()[0]);

    PreparedFusion fa, fb;
    ASSERT_TRUE(fa.take(ea));
    ASSERT_TRUE(fb.take(eb));

    int sz[] = { 1, 3 };
    Mat ya(2, sz, CV_32F), yb(2, sz, CV_32F);
    for (int i = 0; i < 3; i++) { ya.ptr<float>()[i] = 1.f; yb.ptr<float>()[i] = 1.f; }
    fa.run(ya);
    fb.run(yb);
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(2.f,  ya.ptr<float>()[i]) << "A i=" << i;
        EXPECT_FLOAT_EQ(10.f, yb.ptr<float>()[i]) << "B i=" << i;
    }
}

TEST(Fusion, AffineWithScaleAndShiftBuffers)
{
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath r;
    const int shift = r.perChannelConstant(1);  // created first, so it lands on the left of the ADD
    const int scaled = r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.perChannelConstant(0));
    r.binary(FusionEltwiseOp::ADD, shift, scaled);
    const int root = fusion::instantiate(arena, in, r);
    ASSERT_GE(root, 0);

    // 37 is not a multiple of any SIMD width, so every row also runs the scalar tail.
    const int nch = 37, rows = 5;
    Mat s(1, &nch, CV_32F), b(1, &nch, CV_32F);
    for (int c = 0; c < nch; c++) { s.ptr<float>()[c] = 0.5f + c; b.ptr<float>()[c] = c - 3.f; }
    Ptr<AdjacencyGraph> expr = fusion::extract(arena.graph(), root, std::vector<Mat>{ s, b });
    ASSERT_TRUE(expr);
    PreparedFusion f;
    ASSERT_TRUE(f.take(expr));

    int sz[] = { rows, nch };
    Mat y(2, sz, CV_32F);
    for (int i = 0; i < rows * nch; i++)
        y.ptr<float>()[i] = 0.25f * i;
    f.run(y);
    for (int i = 0; i < rows * nch; i++) {
        const int c = i % nch;
        EXPECT_NEAR(0.25f * i * (0.5f + c) + (c - 3.f), y.ptr<float>()[i], 1e-3) << "i=" << i;
    }
}

TEST(Fusion, AffineFoldsSubAndShiftBeforeScale)
{
    // (x + b[c]) * s[c] folds to x * s[c] + b[c] * s[c].
    AdjacencyGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath r;
    const int shifted = r.binary(FusionEltwiseOp::ADD, LayerMath::INPUT_VALUE, r.perChannelConstant(1));
    r.binary(FusionEltwiseOp::MUL, shifted, r.perChannelConstant(0));
    const int root = fusion::instantiate(arena, in, r);
    ASSERT_GE(root, 0);

    const int nch = 37, rows = 3;
    Mat s(1, &nch, CV_32F), b(1, &nch, CV_32F);
    for (int c = 0; c < nch; c++) { s.ptr<float>()[c] = 1.f + 0.1f * c; b.ptr<float>()[c] = 2.f - c; }
    Ptr<AdjacencyGraph> expr = fusion::extract(arena.graph(), root, std::vector<Mat>{ s, b });
    ASSERT_TRUE(expr);
    PreparedFusion f;
    ASSERT_TRUE(f.take(expr));

    int sz[] = { rows, nch };
    Mat y(2, sz, CV_32F);
    for (int i = 0; i < rows * nch; i++)
        y.ptr<float>()[i] = (float)(i % 11);
    f.run(y);
    for (int i = 0; i < rows * nch; i++) {
        const int c = i % nch;
        EXPECT_NEAR(((float)(i % 11) + (2.f - c)) * (1.f + 0.1f * c), y.ptr<float>()[i], 1e-3) << "i=" << i;
    }

    // (x - 3) * 0.5 with scalars is taken; c - x is not affine in this form and is refused.
    LayerMath sub;
    const int minus = sub.binary(FusionEltwiseOp::SUB, LayerMath::INPUT_VALUE, sub.constant(3.f));
    sub.binary(FusionEltwiseOp::MUL, minus, sub.constant(0.5f));
    PreparedFusion fs;
    ASSERT_TRUE(fs.take(graphOf(sub)));
    int n = 19;
    Mat z(1, &n, CV_32F);
    for (int i = 0; i < n; i++)
        z.ptr<float>()[i] = (float)i;
    fs.run(z);
    for (int i = 0; i < n; i++)
        EXPECT_FLOAT_EQ((i - 3.f) * 0.5f, z.ptr<float>()[i]) << "sub i=" << i;

    LayerMath rev;
    rev.binary(FusionEltwiseOp::SUB, rev.constant(3.f), LayerMath::INPUT_VALUE);
    PreparedFusion fr;
    EXPECT_FALSE(fr.take(graphOf(rev)));
}

TEST(Fusion, ShapeOfStaticInputFoldsToConstant)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/dynamic_reshape.onnx"));
    net.setInputShape("0", MatShape({1, 2, 3, 4}));
    Mat inp(std::vector<int>{1, 2, 3, 4}, CV_32F, Scalar(1));
    net.setInput(inp);
    Mat out = net.forward();
    EXPECT_EQ(MatShape({1, 12, 2}), out.shape());

    std::vector<String> types;
    net.getLayerTypes(types);
    EXPECT_EQ(types.end(), std::find(types.begin(), types.end(), "Shape"));
}

TEST(Fusion, FoldedShapeRejectsOtherInputShape)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/dynamic_reshape.onnx"));
    net.setInputShape("0", MatShape({1, 2, 3, 4}));
    Mat inp(std::vector<int>{1, 2, 3, 4}, CV_32F, Scalar(1));
    net.setInput(inp);
    net.forward();
    Mat other(std::vector<int>{1, 2, 3, 8}, CV_32F, Scalar(1));
    EXPECT_THROW(net.setInput(other), cv::Exception);
}

TEST(Fusion, ShapeOfUnpinnedInputStaysDynamic)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/dynamic_reshape.onnx"));
    Mat inp(std::vector<int>{1, 2, 3, 4}, CV_32F, Scalar(1));
    net.setInput(inp);
    Mat out = net.forward();
    EXPECT_EQ(MatShape({1, 12, 2}), out.shape());

    std::vector<String> types;
    net.getLayerTypes(types);
    EXPECT_NE(types.end(), std::find(types.begin(), types.end(), "Shape"));

    Mat other(std::vector<int>{1, 2, 3, 8}, CV_32F, Scalar(1));
    net.setInput(other);
    out = net.forward();
    EXPECT_EQ(MatShape({1, 24, 2}), out.shape());
}

//! How many layers of @p type the fused program still holds.
static int fusedCount(Net& net, const char* type)
{
    Ptr<Graph> g = net.getMainGraph();
    CV_Assert(g);
    int n = 0;
    for (const Ptr<LayerInfo>& l : g->prog())
        n += l && l->type == type;
    return n;
}

TEST(Fusion, ConvTakesTheResidualAdd)
{
    const std::string model = findDataFile("dnn/onnx/models/depthwiseconv_add.onnx");
    Net net = readNetFromONNX(model, ENGINE_OPENCV);
    ASSERT_TRUE(net.getMainGraph());

    const int shape[] = {1, 8, 32, 32};
    Mat input(4, shape, CV_32F, Scalar(1.f));
    net.setInput(input);
    net.forward();

    EXPECT_EQ(fusedCount(net, "NaryEltwise"), 1) << "the residual Add was not absorbed";
}

TEST(Fusion, SharedProjectionsCollapseToOneGemm)
{
    const std::string model =
        findDataFile("dnn/onnx/models/attention_shared_shape_reshape.onnx");
    Net net = readNetFromONNX(model, ENGINE_OPENCV);
    ASSERT_TRUE(net.getMainGraph());

    const int shape[] = {1, 4, 8};
    Mat input(3, shape, CV_32F, Scalar(1.f));
    net.setInput(input);
    net.forward();

    EXPECT_EQ(fusedCount(net, "Gemm"), 1);
    EXPECT_EQ(fusedCount(net, "Slice2"), 3);
}

// A projection without bias must still fuse, with zeros for its part of the QKV bias.
static void checkAttentionFusesWithMissingBias(const std::string& name, int nfused = 1)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + name + ".onnx"), ENGINE_OPENCV);
    ASSERT_TRUE(net.getMainGraph());

    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + name + ".npy")));
    Mat out = net.forward();

    EXPECT_EQ(nfused, fusedCount(net, "Attention")) << name;
    normAssert(blobFromNPY(findDataFile("dnn/onnx/data/output_" + name + ".npy")), out,
               name.c_str());
}

TEST(Fusion, AttentionWithoutProjectionBias)
{
    checkAttentionFusesWithMissingBias("attention_no_qkv_bias");
}

TEST(Fusion, AttentionWithoutKeyBias)
{
    checkAttentionFusesWithMissingBias("attention_no_k_bias");
}

TEST(Fusion, ClipAttentionWithoutKeyBias)
{
    checkAttentionFusesWithMissingBias("clip_attention_no_k_bias");
}

TEST(Fusion, CombinedQKVAttentionWithoutBias)
{
    checkAttentionFusesWithMissingBias("attention_combined_qkv_no_bias");
}

// A scalar K bias does not fit the K part of the QKV bias, so the block stays unfused.
TEST(Fusion, AttentionWithWrongLengthBiasIsNotFused)
{
    checkAttentionFusesWithMissingBias("attention_scalar_k_bias", 0);
}

// readQKVGrouping()/evalDims() must read qkv_batch/qkv_seq from a real Shape/Gather/Mul/Concat
// chain feeding the QKV Reshape, not just accept it as a layer param (as the AttentionLayer-level
// tests in test_layers.cpp do).
static void checkAttentionQKVGrouping(const std::string& name)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + name + ".onnx"), ENGINE_OPENCV);
    ASSERT_TRUE(net.getMainGraph());

    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + name + ".npy")));
    Mat out = net.forward();

    EXPECT_EQ(1, fusedCount(net, "Attention")) << name;
    normAssert(blobFromNPY(findDataFile("dnn/onnx/data/output_" + name + ".npy")), out, name.c_str());
}

TEST(Fusion, AttentionQKVGroupingSeqMerge)
{
    checkAttentionQKVGrouping("attention_qkv_grouping_seq_merge");
}

TEST(Fusion, AttentionQKVGroupingBatchMerge)
{
    checkAttentionQKVGrouping("attention_qkv_grouping_batch_merge");
}

static void checkBatchNormFoldsIntoConv(const std::string& name)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + name + ".onnx"), ENGINE_OPENCV);
    ASSERT_TRUE(net.getMainGraph());

    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + name + ".npy")));
    Mat out = net.forward();

    EXPECT_EQ(0, fusedCount(net, "BatchNorm2")) << name << ": the fold did not fire";
    normAssert(blobFromNPY(findDataFile("dnn/onnx/data/output_" + name + ".npy")), out,
               name.c_str());
}

TEST(Fusion, BatchNormFoldsIntoDenseConv)
{
    checkBatchNormFoldsIntoConv("batchnorm_conv_dense");
}

TEST(Fusion, BatchNormFoldsIntoDepthwiseConv)
{
    checkBatchNormFoldsIntoConv("batchnorm_conv_depthwise");
}

TEST(Fusion, BatchNormFoldsIntoGroupedConv)
{
    checkBatchNormFoldsIntoConv("batchnorm_conv_grouped");
}

TEST(Fusion, BatchNormFoldsIntoPrepacked1x1Conv)
{
    checkBatchNormFoldsIntoConv("batchnorm_conv_1x1_mlas");
}

// 3 input channels: the conv runs on the plain input with its own weight copy.
TEST(Fusion, BatchNormFoldsIntoSmallCinConv)
{
    checkBatchNormFoldsIntoConv("batchnorm_conv_small_cin");
}

// ConvTranspose2d(8->16) -> Add(residual): the Add fuses into TransformLayout.
TEST(Fusion, TransformLayoutAddFusedMatchesUnfusedComputation)
{
    Net net = readNetFromONNX(_tf("models/deconv_transform_add.onnx"));
    ASSERT_FALSE(net.empty());

    std::vector<std::string> preTypes;
    net.getLayerTypes(preTypes);
    ASSERT_NE(std::find(preTypes.begin(), preTypes.end(), std::string("NaryEltwise")), preTypes.end())
        << "test model must start with a separate NaryEltwise layer for this test to mean anything";

    Mat input(std::vector<int>{1, 8, 6, 6}, CV_32F);
    theRNG().fill(input, RNG::UNIFORM, -1.0, 1.0);
    Mat residual(std::vector<int>{1, 16, 12, 12}, CV_32F);
    theRNG().fill(residual, RNG::UNIFORM, -1.0, 1.0);

    net.setInput(input, "input");
    net.setInput(residual, "residual");

    // Comes back already converted from block layout.
    Mat deconvOut;
    net.forward(deconvOut, "/deconv/ConvTranspose_output_0");

    Mat fusedOut = net.forward();

    // The fusion should have run: no NaryEltwise left in the graph.
    std::vector<std::string> postTypes;
    net.getLayerTypes(postTypes);
    EXPECT_EQ(std::find(postTypes.begin(), postTypes.end(), std::string("NaryEltwise")), postTypes.end())
        << "NaryEltwise should have been absorbed into TransformLayout by the fusion pass";

    Mat unfusedOut;
    cv::add(deconvOut, residual, unfusedOut);

    normAssert(unfusedOut, fusedOut, "fused vs unfused TransformLayout+Add", 1e-5, 1e-5);
}

// ConvTranspose2d(8->12): C0=8 doesn't divide 12, so the last block is partial.
TEST(Fusion, TransformLayoutAddPartialChannelBlockMatchesUnfused)
{
    Net net = readNetFromONNX(_tf("models/deconv_transform_add_partial.onnx"));
    ASSERT_FALSE(net.empty());

    Mat input(std::vector<int>{1, 8, 6, 6}, CV_32F);
    theRNG().fill(input, RNG::UNIFORM, -1.0, 1.0);
    Mat residual(std::vector<int>{1, 12, 12, 12}, CV_32F);
    theRNG().fill(residual, RNG::UNIFORM, -1.0, 1.0);

    net.setInput(input, "input");
    net.setInput(residual, "residual");

    Mat deconvOut;
    net.forward(deconvOut, "/deconv/ConvTranspose_output_0");
    Mat fusedOut = net.forward();

    std::vector<std::string> postTypes;
    net.getLayerTypes(postTypes);
    EXPECT_EQ(std::find(postTypes.begin(), postTypes.end(), std::string("NaryEltwise")), postTypes.end())
        << "NaryEltwise should have been absorbed into TransformLayout by the fusion pass";

    Mat unfusedOut;
    cv::add(deconvOut, residual, unfusedOut);

    normAssert(unfusedOut, fusedOut, "fused vs unfused, partial channel block", 1e-5, 1e-5);
}

// Residual [1,16,1,1] broadcasts over the ConvTranspose output. Shapes aren't known
// at fusion time, so the fused layer must fall back to a broadcasting add.
TEST(Fusion, TransformLayoutAddBroadcastResidualMatchesUnfused)
{
    Net net = readNetFromONNX(_tf("models/deconv_transform_add_broadcast.onnx"));
    ASSERT_FALSE(net.empty());

    Mat input(std::vector<int>{1, 8, 6, 6}, CV_32F);
    theRNG().fill(input, RNG::UNIFORM, -1.0, 1.0);
    Mat residual(std::vector<int>{1, 16, 1, 1}, CV_32F);
    theRNG().fill(residual, RNG::UNIFORM, -1.0, 1.0);

    net.setInput(input, "input");
    net.setInput(residual, "residual");

    Mat deconvOut;
    net.forward(deconvOut, "/deconv/ConvTranspose_output_0");
    Mat fusedOut = net.forward();

    Mat residualFull;
    broadcast(residual, deconvOut.shape(), residualFull);
    Mat unfusedOut;
    cv::add(deconvOut, residualFull, unfusedOut);

    normAssert(unfusedOut, fusedOut, "fused vs unfused, broadcast residual", 1e-5, 1e-5);
}

// Same values as formula_weights() in opencv_extra's generate_onnx_models.py.
static Mat formulaWeights(const std::vector<int>& shape, int salt = 1)
{
    Mat m(shape, CV_32F);
    float* p = m.ptr<float>();
    for (size_t i = 0; i < m.total(); i++)
        p[i] = (float)((int)((i * 37 + salt) % 61) - 30) / 256.f;
    return m;
}

static Mat randomInput(const std::vector<int>& shape)
{
    Mat m(shape, CV_32F);
    theRNG().fill(m, RNG::UNIFORM, -1.0, 1.0);
    return m;
}

static Net loadModel(const std::string& name)
{
    return readNetFromONNX(_tf("models/" + name + ".onnx"), ENGINE_OPENCV);
}

// LayerScale after a MatMul folds into the Gemm weights, but not when the MatMul result is also an output.
TEST(Fusion, LayerScaleFoldsIntoGemmWeights)
{
    Net net = loadModel("matmul_channel_affine");
    Mat x = randomInput({2, 5, 16}), s = randomInput({2, 5, 16});
    net.setInput(x, "X");
    net.setInput(s, "S");
    std::vector<Mat> outs;
    net.forward(outs, std::vector<String>{"Y", "proj", "Y_shared"});
    ASSERT_EQ(3u, outs.size());

    const Mat W = formulaWeights({16, 24}), b = formulaWeights({24}, 2), beta = formulaWeights({24}, 4);
    Mat gamma = formulaWeights({24}, 3) + 1;
    const float *bp = b.ptr<float>(), *gp = gamma.ptr<float>(), *betap = beta.ptr<float>();
    Mat px = x.reshape(1, 10) * W, ps = s.reshape(1, 10) * W;
    Mat refY(10, 24, CV_32F), refProj(10, 24, CV_32F), refShared(10, 24, CV_32F);
    for (int i = 0; i < 10; i++)
        for (int n = 0; n < 24; n++)
        {
            refY.at<float>(i, n) = (px.at<float>(i, n) + bp[n]) * gp[n] + betap[n];
            refProj.at<float>(i, n) = ps.at<float>(i, n) + bp[n];
            refShared.at<float>(i, n) = refProj.at<float>(i, n) * gp[n];
        }
    const std::vector<int> shape{2, 5, 24};
    normAssert(refY.reshape(1, shape), outs[0], "folded");
    normAssert(refProj.reshape(1, shape), outs[1], "shared MatMul result");
    normAssert(refShared.reshape(1, shape), outs[2], "not folded");

    // One Gemm holds W with column n scaled by gamma[n], the other plain W.
    Mat scaledW = W.clone();
    for (int n = 0; n < 24; n++)
    {
        Mat col = scaledW.col(n);
        col *= gp[n];
    }
    int scaled = 0, plain = 0;
    for (const auto& l : net.getMainGraph()->prog())
    {
        if (!l || l->type != "Gemm" || l->blobs.empty() || l->blobs[0].total() != W.total())
            continue;
        Mat B = l->blobs[0].rows == W.rows ? l->blobs[0] : Mat(l->blobs[0].t());
        scaled += cvtest::norm(B, scaledW, NORM_INF) < 1e-6;
        plain += cvtest::norm(B, W, NORM_INF) < 1e-6;
    }
    EXPECT_EQ(1, scaled) << "LayerScale was not folded into the Gemm weights";
    EXPECT_EQ(1, plain) << "the Gemm whose result is shared must keep its weights";
}

// Direct NCHW convolution without groups, accumulated in double; pads are {top, left, bottom, right}.
static Mat refConv(const Mat& x, const Mat& w, const Mat& b, Vec2i strides, Vec4i pads,
                   Vec2i dilations = Vec2i(1, 1))
{
    const int N = x.size[0], C = x.size[1], H = x.size[2], W = x.size[3];
    const int K = w.size[0], kh = w.size[2], kw = w.size[3];
    CV_Assert(w.size[1] == C);
    const int OH = (H + pads[0] + pads[2] - (kh - 1) * dilations[0] - 1) / strides[0] + 1;
    const int OW = (W + pads[1] + pads[3] - (kw - 1) * dilations[1] - 1) / strides[1] + 1;
    Mat y(std::vector<int>{N, K, OH, OW}, CV_32F);
    const float *xp = x.ptr<float>(), *wp = w.ptr<float>();
    float* yp = y.ptr<float>();
    for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++)
    for (int oy = 0; oy < OH; oy++)
    for (int ox = 0; ox < OW; ox++)
    {
        double s = b.empty() ? 0. : b.ptr<float>()[k];
        for (int c = 0; c < C; c++)
        for (int ky = 0; ky < kh; ky++)
        for (int kx = 0; kx < kw; kx++)
        {
            const int iy = oy * strides[0] - pads[0] + ky * dilations[0];
            const int ix = ox * strides[1] - pads[1] + kx * dilations[1];
            if (0 <= iy && iy < H && 0 <= ix && ix < W)
                s += (double)xp[((n * C + c) * H + iy) * W + ix] * wp[((k * C + c) * kh + ky) * kw + kx];
        }
        yp[((n * K + k) * OH + oy) * OW + ox] = (float)s;
    }
    return y;
}

static Mat silu(const Mat& x)
{
    Mat y = x.clone();
    float* p = y.ptr<float>();
    for (size_t i = 0; i < y.total(); i++)
        p[i] = p[i] / (1.f + std::exp(-p[i]));
    return y;
}

static Mat relu(const Mat& x)
{
    Mat y = x.clone();
    float* p = y.ptr<float>();
    for (size_t i = 0; i < y.total(); i++)
        p[i] = std::max(p[i], 0.f);
    return y;
}

// A conv with at most 4 input channels reads the plain input, so this RGB model needs no layout conversion.
TEST(Fusion, SmallInputChannelConvNeedsNoLayoutConversion)
{
#ifdef HAVE_MLAS
    Net net = loadModel("convolution");
    net.setInput(randomInput({1, 3, 10, 10}));
    net.forward();
    EXPECT_EQ(0, fusedCount(net, "TransformLayout"));
#else
    throw SkipTestException("the plain-input conv path needs MLAS");
#endif
}

// Convs with 3 input channels: stride 2 + ReLU, and dilation 2 without bias.
TEST(Fusion, SmallInputChannelConvMatchesReference)
{
    Net net = loadModel("conv_small_cin");
    Mat x = randomInput({2, 3, 17, 15});
    net.setInput(x);
    std::vector<Mat> outs;
    net.forward(outs, std::vector<String>{"Y_s2", "Y_d2"});
    ASSERT_EQ(2u, outs.size());

    normAssert(relu(refConv(x, formulaWeights({16, 3, 3, 3}, 7), formulaWeights({16}, 8),
                            Vec2i(2, 2), Vec4i(1, 1, 1, 1))),
               outs[0], "stride 2 + ReLU");
    normAssert(refConv(x, formulaWeights({8, 3, 3, 3}, 9), Mat(), Vec2i(1, 1), Vec4i(2, 2, 2, 2), Vec2i(2, 2)),
               outs[1], "dilation 2");
}

// A 1x9 kernel also has 9 taps but must not take the 3x3 path; the 3x3 conv uses stride (2, 1) and (2, 2).
TEST(Fusion, NineTapAndStrided3x3Conv)
{
    Net net = loadModel("conv_9tap_and_strided");
    Mat x = randomInput({1, 16, 12, 24});
    net.setInput(x);
    std::vector<Mat> outs;
    net.forward(outs, std::vector<String>{"Y_1x9", "Y_3x3", "Y_3x3_s2"});
    ASSERT_EQ(3u, outs.size());

    const Mat w = formulaWeights({32, 16, 3, 3}, 5), b = formulaWeights({32}, 6);
    const Vec4i pads(0, 1, 1, 2);
    normAssert(refConv(x, formulaWeights({16, 16, 1, 9}), Mat(), Vec2i(1, 1), Vec4i(0, 4, 0, 4)),
               outs[0], "1x9 kernel");
    normAssert(silu(refConv(x, w, b, Vec2i(2, 1), pads)), outs[1], "3x3 kernel, stride (2, 1)");
    normAssert(refConv(x, w, b, Vec2i(2, 2), pads), outs[2], "3x3 kernel, stride (2, 2)");
}

}} // namespace opencv_test
