// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#ifdef HAVE_CUDA
#include <opencv2/core/cuda.hpp>
#endif
#include <cstring>

namespace opencv_test { namespace {

static LayerParams attentionWorkspaceParams(int valueHidden, bool rotary, Mat& weights, Mat& bias)
{
    const int channels = 32;
    int hidden[] = {channels, channels, valueHidden};
    LayerParams params;
    params.set("num_heads", 2);
    params.set("qkv_hidden_sizes", DictValue::arrayInt(hidden, 3));
    params.set("do_rotary", rotary);
    weights.create(channels, 2 * channels + valueHidden, CV_32F);
    bias.create(MatShape{2 * channels + valueHidden}, CV_32F);
    theRNG().fill(weights, RNG::UNIFORM, -.1f, .1f);
    theRNG().fill(bias, RNG::UNIFORM, -.1f, .1f);
    return params;
}

typedef testing::TestWithParam<tuple<bool, bool, bool> > Layer_AttentionWorkspace;

TEST_P(Layer_AttentionWorkspace, memoryPlan)
{
    const bool dynamicWeights = get<0>(GetParam()), hasMask = get<1>(GetParam()), rotary = get<2>(GetParam());
    Mat weights, bias;
    LayerParams params = attentionWorkspaceParams(48, rotary, weights, bias);
    if (!dynamicWeights)
        params.blobs = {weights, bias};
    Ptr<Layer> layer = AttentionLayer::create(params);
#ifdef HAVE_MLAS
    size_t previous = 0;
#endif
    for (int seq : {1024, 2048})
    {
        std::vector<MatShape> inputs{MatShape{2, seq, 32}}, outputs, internals;
        if (dynamicWeights)
        {
            inputs.push_back(weights.shape());
            inputs.push_back(bias.shape());
        }
        if (hasMask)
            inputs.push_back(MatShape{seq, seq});
        layer->getMemoryShapes(inputs, 1, outputs, internals);
        ASSERT_EQ(size_t(rotary ? 4 : 3), internals.size());
        size_t elements = 0;
        for (const MatShape& shape : internals)
            elements += total(shape);
#ifdef HAVE_MLAS
        if (!hasMask)
        {
            // Linear workspace growth, including QKV and optional RoPE.
            EXPECT_LT(elements, size_t(2 * seq * 128));
            if (previous)
            {
                EXPECT_LE(elements, 2 * previous);
            }
            previous = elements;
        }
        else
#endif
        {
            EXPECT_EQ(MatShape({4, seq, seq}), internals[1]);
            EXPECT_EQ(MatShape({4, seq, 24}), internals[2]);
        }
    }
}

typedef testing::TestWithParam<bool> Layer_AttentionWorkspaceShapes;

// the same net runs inputs of different lengths one after another
TEST_P(Layer_AttentionWorkspaceShapes, changingInputShape)
{
    const bool rotary = GetParam();
    Mat weights, bias;
    LayerParams params = attentionWorkspaceParams(48, rotary, weights, bias);
    params.blobs = {weights, bias};
    Net net;
    net.addLayerToPrev("attention", "Attention", CV_32F, params);
    for (int seq : {17, 257, 1, 17})
    {
        SCOPED_TRACE(testing::Message() << "sequence=" << seq);
        Mat input(MatShape{2, seq, 32}, CV_32F);
        theRNG().fill(input, RNG::UNIFORM, -1.f, 1.f);
        std::vector<Mat> inputs{input};
        std::vector<Mat> expected;
        runLayer(AttentionLayer::create(params), inputs, expected);
        net.setInput(input);
        Mat actual = net.forward();
        ASSERT_EQ(expected[0].shape(), actual.shape());
        EXPECT_EQ(0, std::memcmp(expected[0].data, actual.data, expected[0].total() * expected[0].elemSize()));
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_AttentionWorkspace, testing::Combine(
    testing::Bool(), testing::Bool(), testing::Bool()));
INSTANTIATE_TEST_CASE_P(/**/, Layer_AttentionWorkspaceShapes, testing::Bool());

namespace {
struct TiledAttnOnnxBuf
{
    std::vector<uchar> b;
    void uvar(uint64_t v)
    {
        while (v >= 0x80) { b.push_back((uchar)((v & 0x7f) | 0x80)); v >>= 7; }
        b.push_back((uchar)v);
    }
    void tag(uint32_t f, uint32_t w) { uvar((uint64_t(f) << 3) | w); }
    void i64(uint32_t f, int64_t v) { tag(f, 0); uvar((uint64_t)v); }
    void i32(uint32_t f, int32_t v) { tag(f, 0); uvar((uint64_t)(int64_t)v); }
    void bytes(uint32_t f, const void* p, size_t n)
    {
        tag(f, 2);
        uvar(n);
        const uchar* q = static_cast<const uchar*>(p);
        b.insert(b.end(), q, q + n);
    }
    void str(uint32_t f, const std::string& s) { bytes(f, s.data(), s.size()); }
    void sub(uint32_t f, const TiledAttnOnnxBuf& c) { bytes(f, c.b.data(), c.b.size()); }
};

static TiledAttnOnnxBuf tiledValueInfoF32(const std::string& name, const std::vector<int>& dims)
{
    TiledAttnOnnxBuf shape;
    for (int d : dims) { TiledAttnOnnxBuf dm; dm.i64(1, d); shape.sub(1, dm); }
    TiledAttnOnnxBuf tt; tt.i32(1, 1); tt.sub(2, shape);
    TiledAttnOnnxBuf tp; tp.sub(1, tt);
    TiledAttnOnnxBuf vi; vi.str(1, name); vi.sub(2, tp);
    return vi;
}

static TiledAttnOnnxBuf tiledTensorF32(const std::string& name, const Mat& m)
{
    CV_Assert(m.type() == CV_32F && m.isContinuous());
    TiledAttnOnnxBuf t;
    for (int i = 0; i < m.dims; ++i) t.i64(1, m.size[i]);
    t.i32(2, 1);
    t.str(8, name);
    t.bytes(9, m.data, m.total() * sizeof(float));
    return t;
}

static TiledAttnOnnxBuf tiledTensorI64(const std::string& name, const std::vector<int64_t>& vs)
{
    TiledAttnOnnxBuf t;
    t.i64(1, (int64_t)vs.size());
    t.i32(2, 7);
    t.str(8, name);
    t.bytes(9, vs.data(), vs.size() * sizeof(int64_t));
    return t;
}

static TiledAttnOnnxBuf tiledAttrInt(const std::string& name, int64_t v)
{
    TiledAttnOnnxBuf a; a.str(1, name); a.i64(3, v); a.i32(20, 2);
    return a;
}

static TiledAttnOnnxBuf tiledAttrInts(const std::string& name, const std::vector<int64_t>& vs)
{
    TiledAttnOnnxBuf a; a.str(1, name);
    for (int64_t v : vs) a.i64(8, v);
    a.i32(20, 7);
    return a;
}

static TiledAttnOnnxBuf tiledNode(const std::string& op, const std::vector<std::string>& inps,
                                  const std::vector<std::string>& outs,
                                  const std::vector<TiledAttnOnnxBuf>& attrs = {},
                                  const std::string& domain = "")
{
    TiledAttnOnnxBuf n;
    for (const auto& s : inps) n.str(1, s);
    for (const auto& s : outs) n.str(2, s);
    n.str(4, op);
    for (const auto& a : attrs) n.sub(5, a);
    if (!domain.empty()) n.str(7, domain);
    return n;
}

// Decomposed BERT self-attention model (fuses to Attention with optional dynamic mask M)
static std::vector<uchar> buildFusedAttentionOnnx(int B, int S, int H, int Dh,
                                                  const Mat& W, const Mat& bias,
                                                  const std::vector<int>& maskDims)
{
    const int C = H * Dh;
    Mat Wq = W.colRange(0, C).clone();
    Mat Wk = W.colRange(C, 2 * C).clone();
    Mat Wv = W.colRange(2 * C, 3 * C).clone();
    Mat Bq = Mat(std::vector<int>{C}, CV_32F, const_cast<float*>(bias.ptr<float>())).clone();
    Mat Bk = Mat(std::vector<int>{C}, CV_32F, const_cast<float*>(bias.ptr<float>() + C)).clone();
    Mat Bv = Mat(std::vector<int>{C}, CV_32F, const_cast<float*>(bias.ptr<float>() + 2 * C)).clone();
    Mat scaleTensor(std::vector<int>{1}, CV_32F, Scalar(std::sqrt((float)Dh)));

    std::vector<TiledAttnOnnxBuf> nodes = {
        tiledNode("MatMul", {"X", "Wq"}, {"Qm"}),
        tiledNode("Add", {"Qm", "Bq"}, {"Qa"}),
        tiledNode("Reshape", {"Qa", "qkv_shape"}, {"Qr"}),
        tiledNode("Transpose", {"Qr"}, {"Qt"}, { tiledAttrInts("perm", {0, 2, 1, 3}) }),
        tiledNode("MatMul", {"X", "Wk"}, {"Km"}),
        tiledNode("Add", {"Km", "Bk"}, {"Ka"}),
        tiledNode("Reshape", {"Ka", "qkv_shape"}, {"Kr"}),
        tiledNode("Transpose", {"Kr"}, {"Kt"}, { tiledAttrInts("perm", {0, 2, 3, 1}) }),
        tiledNode("MatMul", {"X", "Wv"}, {"Vm"}),
        tiledNode("Add", {"Vm", "Bv"}, {"Va"}),
        tiledNode("Reshape", {"Va", "qkv_shape"}, {"Vr"}),
        tiledNode("Transpose", {"Vr"}, {"Vt"}, { tiledAttrInts("perm", {0, 2, 1, 3}) }),
        tiledNode("MatMul", {"Qt", "Kt"}, {"Scores"}),
        tiledNode("Div", {"Scores", "Scale"}, {"Scaled"})
    };
    std::string smIn = "Scaled";
    if (!maskDims.empty())
    {
        nodes.push_back(tiledNode("Add", {"Scaled", "M"}, {"Masked"}));
        smIn = "Masked";
    }
    nodes.push_back(tiledNode("Softmax", {smIn}, {"Probs"}, { tiledAttrInt("axis", -1) }));
    nodes.push_back(tiledNode("MatMul", {"Probs", "Vt"}, {"Ctx"}));
    nodes.push_back(tiledNode("Transpose", {"Ctx"}, {"CtxT"}, { tiledAttrInts("perm", {0, 2, 1, 3}) }));
    nodes.push_back(tiledNode("Reshape", {"CtxT", "out_shape"}, {"Y"}));

    TiledAttnOnnxBuf g;
    for (const auto& n : nodes) g.sub(1, n);
    g.str(2, "g");
    for (const auto& t : std::vector<TiledAttnOnnxBuf>{
            tiledTensorF32("Wq", Wq), tiledTensorF32("Bq", Bq),
            tiledTensorF32("Wk", Wk), tiledTensorF32("Bk", Bk),
            tiledTensorF32("Wv", Wv), tiledTensorF32("Bv", Bv),
            tiledTensorF32("Scale", scaleTensor),
            tiledTensorI64("qkv_shape", {B, S, H, Dh}),
            tiledTensorI64("out_shape", {B, S, C}) })
        g.sub(5, t);
    g.sub(11, tiledValueInfoF32("X", {B, S, C}));
    if (!maskDims.empty())
        g.sub(11, tiledValueInfoF32("M", maskDims));
    g.sub(12, tiledValueInfoF32("Y", {B, S, C}));

    TiledAttnOnnxBuf m;
    m.i64(1, 8);
    m.str(2, "test");
    m.sub(7, g);
    TiledAttnOnnxBuf op; op.str(1, ""); op.i64(2, 18);
    m.sub(8, op);
    return m.b;
}
} // namespace

typedef testing::TestWithParam<tuple<Backend, Target> > Layer_AttentionTiledCUDA;
TEST_P(Layer_AttentionTiledCUDA, MultiTileAccuracyAndSubquadraticWorkspace)
{
    const Backend backendId = get<0>(GetParam());
    const Target targetId = get<1>(GetParam());
    if (backendId != DNN_BACKEND_CUDA)
        throw SkipTestException("Tests sequence-tiled CUDA AttentionOp accuracy and VRAM bound");

    const bool fp16 = targetId == DNN_TARGET_CUDA_FP16;
    const double l1 = fp16 ? 1.5e-2 : 1e-5;
    const double lInf = fp16 ? 6e-2 : 1e-4;

    // 1. Multi-tile accuracy (S = 520 > 2 * 256, non-multiple tail 8) across unmasked and 4 mask layouts
    {
        const int B = 2, S = 520, H = 4, Dh = 16, C = H * Dh;
        Mat X({B, S, C}, CV_32F), W(C, 3 * C, CV_32F), bias(std::vector<int>{3 * C}, CV_32F);
        theRNG().fill(X, RNG::UNIFORM, -0.5f, 0.5f);
        theRNG().fill(W, RNG::UNIFORM, -0.2f, 0.2f);
        theRNG().fill(bias, RNG::UNIFORM, -0.1f, 0.1f);

        const std::vector<std::vector<int> > maskShapes = {
            {},
            {1, 1, 1, S},
            {1, 1, S, S},
            {B, 1, S, S},
            {B, H, S, S}
        };
        for (size_t mi = 0; mi < maskShapes.size(); ++mi)
        {
            const std::vector<int>& mShape = maskShapes[mi];
            std::vector<uchar> buf = buildFusedAttentionOnnx(B, S, H, Dh, W, bias, mShape);

            Mat mask;
            if (!mShape.empty())
            {
                mask.create(mShape, CV_32F);
                theRNG().fill(mask, RNG::UNIFORM, -2.f, 0.f);
            }

            Net cpuNet = readNetFromONNX(buf);
            cpuNet.setPreferableBackend(DNN_BACKEND_OPENCV);
            cpuNet.setPreferableTarget(DNN_TARGET_CPU);
            cpuNet.setInput(X, "X");
            if (!mShape.empty()) cpuNet.setInput(mask, "M");
            Mat ref = cpuNet.forward().clone();

            Net cudaNet = readNetFromONNX(buf);
            cudaNet.setPreferableBackend(backendId);
            cudaNet.setPreferableTarget(targetId);
            cudaNet.setInput(X, "X");
            if (!mShape.empty()) cudaNet.setInput(mask, "M");
            Mat out = cudaNet.forward();
            normAssert(ref, out, format("mask_case=%zu", mi).c_str(), l1, lInf);
        }
    }

#ifdef HAVE_CUDA
    // 2. Sub-quadratic workspace VRAM bound at B=2, H=12, S=2048, D=768 (stock dense scores need 384 MiB alone)
    if (targetId == DNN_TARGET_CUDA)
    {
        const int B = 2, H = 12, Dh = 64, C = H * Dh;
        Mat W(C, 3 * C, CV_32F), bias(std::vector<int>{3 * C}, CV_32F);
        theRNG().fill(W, RNG::UNIFORM, -0.05f, 0.05f);
        theRNG().fill(bias, RNG::UNIFORM, -0.05f, 0.05f);

        // Warmup CUDA/cuBLAS handles on S=16 before sampling device memory
        {
            Net warmup = readNetFromONNX(buildFusedAttentionOnnx(1, 16, H, Dh, W, bias, {}));
            warmup.setPreferableBackend(backendId);
            warmup.setPreferableTarget(targetId);
            warmup.setInput(Mat({1, 16, C}, CV_32F, Scalar(0)), "X");
            warmup.forward();
        }

        cv::cuda::DeviceInfo devInfo;
        const size_t freeBefore = devInfo.freeMemory();
        Net bigNet = readNetFromONNX(buildFusedAttentionOnnx(B, 2048, H, Dh, W, bias, {}));
        bigNet.setPreferableBackend(backendId);
        bigNet.setPreferableTarget(targetId);
        Mat bigX({B, 2048, C}, CV_32F, Scalar(0.01f));
        bigNet.setInput(bigX, "X");
        Mat bigOut = bigNet.forward();
        const size_t freeAfter = devInfo.freeMemory();
        const size_t usedBytes = (freeBefore > freeAfter) ? (freeBefore - freeAfter) : 0;
        // Tiled workspace + IO + weights is ~150 MiB; stock dense [B*H, 2048, 2048] allocates >500 MiB
        EXPECT_LT(usedBytes, size_t(280ULL * 1024 * 1024))
            << "CUDA Attention workspace grew quadratically: used " << (usedBytes >> 20) << " MiB";
    }
#endif
}
INSTANTIATE_TEST_CASE_P(/**/, Layer_AttentionTiledCUDA, dnnBackendsAndTargets());

}} // namespace
