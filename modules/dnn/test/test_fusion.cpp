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

TEST(Fusion, ApplyTakesKernelPathThenInterpreterPath)
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

    r = LayerMath();
    r.unary(FusionEltwiseOp::SQRT, LayerMath::INPUT_VALUE);
    PreparedFusion interp;
    ASSERT_TRUE(interp.take(graphOf(r)));
    EXPECT_TRUE(interp.activationFn == nullptr);

    int big = (1 << 16) + 17;
    Mat z(1, &big, CV_32F);
    for (int i = 0; i < big; i++)
        z.ptr<float>()[i] = (float)(i % 100);
    interp.run(z);
    for (int i = 0; i < big; i += 997)
        EXPECT_NEAR(std::sqrt((float)(i % 100)), z.ptr<float>()[i], 1e-5) << "sqrt i=" << i;
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

}} // namespace opencv_test
