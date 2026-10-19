// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <cmath>

namespace opencv_test { namespace {

static std::vector<int> shapeOf(const Mat& m) { return std::vector<int>(m.size.p, m.size.p + m.dims); }

static std::vector<int> broadcastShape(const std::vector<Mat>& ms)
{
    int dims = 0;
    for (const Mat& m : ms) dims = std::max(dims, m.dims);
    std::vector<int> out(dims, 1);
    for (const Mat& m : ms)
        for (int i = 0; i < m.dims; i++)
            if (m.size[i] != 1)
                out[dims - m.dims + i] = m.size[i];
    return out;
}

static double getElem(const Mat& m, size_t i)
{
    switch (m.depth())
    {
    case CV_Bool: case CV_8U: return m.ptr<uchar>()[i];
    case CV_8S: return m.ptr<schar>()[i];
    case CV_16U: return m.ptr<ushort>()[i];
    case CV_16S: return m.ptr<short>()[i];
    case CV_32S: return m.ptr<int>()[i];
    case CV_64S: return (double)m.ptr<int64_t>()[i];
    case CV_32F: return m.ptr<float>()[i];
    case CV_64F: return m.ptr<double>()[i];
    }
    CV_Error(Error::StsBadArg, "unsupported depth");
}

static int64_t getInt(const Mat& m, size_t i)
{
    switch (m.depth())
    {
    case CV_8U: return m.ptr<uchar>()[i];
    case CV_8S: return m.ptr<schar>()[i];
    case CV_16U: return m.ptr<ushort>()[i];
    case CV_16S: return m.ptr<short>()[i];
    case CV_32S: return m.ptr<int>()[i];
    case CV_64S: return m.ptr<int64_t>()[i];
    }
    CV_Error(Error::StsBadArg, "unsupported depth");
}

static size_t srcIndex(const Mat& m, const std::vector<int>& oshape, size_t oidx)
{
    size_t idx = 0, mul = 1;
    for (int i = (int)oshape.size() - 1; i >= 0; i--)
    {
        int c = (int)(oidx % oshape[i]);
        oidx /= oshape[i];
        int j = i - ((int)oshape.size() - m.dims);
        if (j >= 0)
        {
            if (m.size[j] != 1)
                idx += c*mul;
            mul *= m.size[j];
        }
    }
    return idx;
}

static Mat runNary(const std::string& op, const std::vector<Mat>& inputs, int outType)
{
    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.name = "testLayer";
    lp.set("operation", op);
    Ptr<Layer> layer = LayerFactory::createLayerInstance("NaryEltwise", lp);
    std::vector<Mat> inps(inputs), outs{Mat(broadcastShape(inputs), outType)}, internals;
    layer->finalize(inps, outs);
    layer->forward(inps, outs, internals);
    return outs[0];
}

static double refNary(const std::string& op, const std::vector<double>& x, const std::vector<int64_t>& xi,
                      bool isInt, int depth)
{
    if (op == "add" || op == "sum") { double s = 0; for (double v : x) s += v; return s; }
    if (op == "mean") { double s = 0; for (double v : x) s += v; return s/x.size(); }
    if (op == "sub") return x[0] - x[1];
    if (op == "mul") return x[0]*x[1];
    if (op == "div") return x[0]/x[1];
    if (op == "max") return *std::max_element(x.begin(), x.end());
    if (op == "min") return *std::min_element(x.begin(), x.end());
    if (op == "pow") return std::pow(x[0], x[1]);
    if (op == "equal") return x[0] == x[1];
    if (op == "greater") return x[0] > x[1];
    if (op == "greaterorequal") return x[0] >= x[1];
    if (op == "less") return x[0] < x[1];
    if (op == "lessorequal") return x[0] <= x[1];
    if (op == "where") return x[0] != 0 ? x[1] : x[2];
    if (op == "and") return (x[0] != 0) && (x[1] != 0);
    if (op == "or") return (x[0] != 0) || (x[1] != 0);
    if (op == "xor") return (x[0] != 0) != (x[1] != 0);
    if (isInt && op == "bitwise_and") return (double)(xi[0] & xi[1]);
    if (isInt && op == "bitwise_or") return (double)(xi[0] | xi[1]);
    if (isInt && op == "bitwise_xor") return (double)(xi[0] ^ xi[1]);
    CV_Error(Error::StsBadArg, "unknown op " + op + " for depth " + std::to_string(depth));
}

static void checkNary(const std::string& op, const std::vector<Mat>& inputs, int outType, const std::string& info)
{
    Mat out = runNary(op, inputs, outType);
    const std::vector<int> oshape = broadcastShape(inputs);
    ASSERT_EQ(oshape, shapeOf(out)) << info;
    const int depth = inputs.back().depth();
    const bool isInt = depth != CV_32F && depth != CV_64F && depth != CV_Bool;
    const double tol = CV_MAT_DEPTH(outType) == CV_32F ? 1e-6 : CV_MAT_DEPTH(outType) == CV_64F ? 1e-12 : 0;
    for (size_t i = 0; i < out.total(); i++)
    {
        std::vector<double> x;
        std::vector<int64_t> xi;
        for (const Mat& m : inputs)
        {
            size_t j = srcIndex(m, oshape, i);
            x.push_back(getElem(m, j));
            xi.push_back(isInt && m.depth() != CV_Bool ? getInt(m, j) : 0);
        }
        double e = refNary(op, x, xi, isInt, depth);
        if (CV_MAT_DEPTH(outType) == CV_32F)
            e = (float)e;
        double v = getElem(out, i);
        if (std::isnan(e) && std::isnan(v))
            continue;
        ASSERT_LE(std::abs(e - v), tol*(std::abs(e) + 1)) << info << " at " << i << ": expected " << e << ", got " << v;
    }
}

static Mat randomMat(RNG& rng, const std::vector<int>& shape, int type, double lo, double hi)
{
    Mat m(shape, type);
    if (m.total() == 0)
        return m;
    if (type == CV_Bool)
    {
        Mat u(1, (int)m.total(), CV_8U, m.data);
        rng.fill(u, RNG::UNIFORM, 0, 2);
        return m;
    }
    Mat tmp(shape, CV_64F);
    rng.fill(tmp, RNG::UNIFORM, lo, hi);
    if (type != CV_32F && type != CV_64F)
        for (size_t i = 0; i < tmp.total(); i++)
            tmp.ptr<double>()[i] = std::floor(tmp.ptr<double>()[i]);
    tmp.convertTo(m, type);
    return m;
}

typedef std::vector<std::vector<int> > Shapes;
static const Shapes binaryShapes[] = {
    {{}, {}}, {{1}, {1}}, {{5}, {}}, {{2, 3, 4, 5}, {2, 3, 4, 5}}, {{2, 3, 4, 5}, {1, 3, 1, 1}},
    {{2, 3, 4, 5}, {5}}, {{2, 1, 4, 1}, {1, 3, 1, 5}}, {{3, 4, 5}, {3, 4, 1}}, {{}, {3, 4}}
};

TEST(DNN_NaryEltwiseCore, float_ops)
{
    RNG& rng = theRNG();
    for (int type : {CV_32F, CV_64F})
        for (const Shapes& sh : binaryShapes)
            for (const char* op : {"add", "sub", "mul", "div", "max", "min", "mean", "sum", "pow"})
            {
                const bool pow = std::string(op) == "pow";
                Mat a = randomMat(rng, sh[0], type, pow ? 0.1 : -10, pow ? 3 : 10);
                Mat b = randomMat(rng, sh[1], type, pow ? -3 : 0.5, pow ? 3 : 10);
                checkNary(op, {a, b}, type, cv::format("%s %s %s %s", op, typeToString(type).c_str(),
                                                        MatShape(sh[0]).str().c_str(), MatShape(sh[1]).str().c_str()));
            }
}

TEST(DNN_NaryEltwiseCore, nary_ops)
{
    RNG& rng = theRNG();
    for (int type : {CV_32F, CV_64F})
        for (const char* op : {"sum", "mean", "max", "min"})
        {
            Mat a = randomMat(rng, {2, 3, 4}, type, -5, 5), b = randomMat(rng, {3, 1}, type, -5, 5);
            Mat c = randomMat(rng, {1, 1, 4}, type, -5, 5), d = randomMat(rng, {}, type, -5, 5);
            checkNary(op, {a, b, c}, type, std::string(op) + " x3");
            checkNary(op, {a, b, c, d}, type, std::string(op) + " x4");
        }
    Mat ia = randomMat(rng, {2, 3, 4}, CV_32S, -50, 50), ib = randomMat(rng, {3, 4}, CV_32S, -50, 50);
    Mat ic = randomMat(rng, {4}, CV_32S, -50, 50);
    checkNary("max", {ia, ib, ic}, CV_32S, "int max x3");
    checkNary("min", {ia, ib, ic}, CV_32S, "int min x3");
}

TEST(DNN_NaryEltwiseCore, comparisons)
{
    RNG& rng = theRNG();
    for (int type : {CV_32F, CV_64F, CV_8U, CV_8S, CV_16S, CV_32S, CV_64S})
        for (const Shapes& sh : binaryShapes)
            for (const char* op : {"equal", "greater", "greaterorequal", "less", "lessorequal"})
            {
                // small ranges, so that "equal" is often true
                Mat a = randomMat(rng, sh[0], type, 0, 4), b = randomMat(rng, sh[1], type, 0, 4);
                checkNary(op, {a, b}, CV_Bool, cv::format("%s %s", op, typeToString(type).c_str()));
            }
}

TEST(DNN_NaryEltwiseCore, where_logic_bitwise)
{
    RNG& rng = theRNG();
    for (const Shapes& sh : binaryShapes)
    {
        Mat c = randomMat(rng, sh[0], CV_Bool, 0, 2);
        for (int type : {CV_32F, CV_64F, CV_8U, CV_32S, CV_64S})
        {
            Mat x = randomMat(rng, sh[1], type, -100, 100), y = randomMat(rng, sh[0], type, -100, 100);
            checkNary("where", {c, x, y}, type, "where " + typeToString(type) + " " + MatShape(sh[0]).str() + " " +
                      MatShape(sh[1]).str());
        }
        Mat p = randomMat(rng, sh[0], CV_Bool, 0, 2), q = randomMat(rng, sh[1], CV_Bool, 0, 2);
        for (const char* op : {"and", "or", "xor"})
            checkNary(op, {p, q}, CV_Bool, op);
        for (int type : {CV_8U, CV_16S, CV_32S, CV_64S})
        {
            Mat a = randomMat(rng, sh[0], type, 0, 120), b = randomMat(rng, sh[1], type, 0, 120);
            for (const char* op : {"bitwise_and", "bitwise_or", "bitwise_xor", "max", "min"})
                checkNary(op, {a, b}, type, cv::format("%s %s", op, typeToString(type).c_str()));
        }
    }
}

// integer arithmetic wraps around in the layer
TEST(DNN_NaryEltwiseCore, int_add_wraps)
{
    Mat a({4}, CV_8U, Scalar(200)), b({4}, CV_8U, Scalar(100));
    Mat out = runNary("add", {a, b}, CV_8U);
    EXPECT_EQ(44, (int)out.ptr<uchar>()[0]);
}

static Mat runActivation(const std::string& type, const LayerParams& params, const Mat& input)
{
    LayerParams lp = params;
    lp.type = type;
    lp.name = "testLayer";
    Ptr<Layer> layer = LayerFactory::createLayerInstance(type, lp);
    std::vector<Mat> inps{input}, outs{Mat(input.size, input.type())}, internals;
    layer->finalize(inps, outs);
    layer->forward(inps, outs, internals);
    return outs[0];
}

TEST(DNN_ActivationCore, accuracy)
{
    RNG& rng = theRNG();
    struct Case { const char* type; std::function<void(LayerParams&)> setup; std::function<double(double)> ref; double lo, hi; };
    std::vector<Case> cases = {
        {"Sqrt", [](LayerParams&) {}, [](double x) { return std::sqrt(x); }, 0, 10},
        {"AbsVal", [](LayerParams&) {}, [](double x) { return std::abs(x); }, -5, 5},
        {"Reciprocal", [](LayerParams&) {}, [](double x) { return 1/x; }, 0.1, 5},
        {"Log", [](LayerParams&) {}, [](double x) { return std::log(x); }, 0.01, 10},
        {"TanH", [](LayerParams&) {}, [](double x) { return std::tanh(x); }, -4, 4},
        {"Erf", [](LayerParams&) {}, [](double x) { return std::erf(x); }, -3, 3},
        {"ReLU", [](LayerParams&) {}, [](double x) { return std::max(x, 0.); }, -3, 3},
        {"ReLU", [](LayerParams& p) { p.set("negative_slope", 0.1f); }, [](double x) { return x > 0 ? x : 0.1f*x; }, -3, 3},
        {"Exp", [](LayerParams&) {}, [](double x) { return std::exp(x); }, -5, 5},
        {"Exp", [](LayerParams& p) { p.set("base", 2.f); p.set("scale", 0.5f); p.set("shift", -1.f); },
                [](double x) { return std::pow(2., 0.5*x - 1.); }, -5, 5},
        {"Power", [](LayerParams& p) { p.set("power", 2.f); p.set("scale", 0.5f); p.set("shift", 1.f); },
                  [](double x) { return std::pow(0.5*x + 1, 2.); }, -3, 3},
        {"Power", [](LayerParams& p) { p.set("power", 1.f); p.set("scale", -2.f); p.set("shift", 0.25f); },
                  [](double x) { return -2*x + 0.25; }, -3, 3},
        {"Power", [](LayerParams& p) { p.set("power", 0.5f); }, [](double x) { return std::sqrt(x); }, 0, 9},
    };
    for (const Case& c : cases)
        for (int type : {CV_32F, CV_64F})
            for (std::vector<int> shape : {std::vector<int>{}, std::vector<int>{7}, std::vector<int>{2, 3, 20, 20}})
            {
                LayerParams lp;
                c.setup(lp);
                Mat x = randomMat(rng, shape, type, c.lo, c.hi);
                Mat y = runActivation(c.type, lp, x);
                ASSERT_EQ(shapeOf(x), shapeOf(y)) << c.type;
                for (size_t i = 0; i < x.total(); i++)
                {
                    double e = c.ref(getElem(x, i)), v = getElem(y, i);
                    ASSERT_LE(std::abs(e - v), 2e-5*(std::abs(e) + 1))
                        << c.type << " " << typeToString(type) << " " << MatShape(shape).str() << " x=" << getElem(x, i);
                }
            }
}

}} // namespace
