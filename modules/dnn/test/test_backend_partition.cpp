// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/graph_partition.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;

static const int CPU = DNN_BACKEND_OPENCV;
static const int GPU = DNN_BACKEND_CUDA;

static Ptr<LayerInfo> makeOp(const std::string& type, const std::vector<int>& inputs,
                             const std::vector<int>& outputs)
{
    Ptr<LayerInfo> op = makePtr<LayerInfo>();
    op->type = type;
    op->name = type + "_" + std::to_string(outputs.empty() ? 0 : outputs[0]);
    for (int idx : inputs)
        op->inputs.push_back(Arg(idx));
    for (int idx : outputs)
        op->outputs.push_back(Arg(idx));
    return op;
}

static std::vector<Ptr<LayerInfo> > makeChain(size_t n)
{
    std::vector<Ptr<LayerInfo> > prog;
    for (size_t i = 0; i < n; i++)
        prog.push_back(makeOp("Op", {(int)i + 1}, {(int)i + 2}));
    return prog;
}

TEST(DNN_Partition, short_accelerator_runs_move_to_cpu)
{
    std::vector<Ptr<LayerInfo> > prog = makeChain(8);
    std::vector<uchar> hostOp(prog.size(), 0);
    std::vector<int> placement = {GPU, GPU, CPU, GPU, GPU, GPU, CPU, GPU};
    demoteShortRuns(prog, hostOp, placement, 3);
    EXPECT_EQ(placement, std::vector<int>({CPU, CPU, CPU, GPU, GPU, GPU, CPU, CPU}));
}

TEST(DNN_Partition, host_ops_neither_break_nor_extend_a_run)
{
    std::vector<Ptr<LayerInfo> > prog = makeChain(5);
    prog[3].release();
    std::vector<uchar> hostOp = {0, 1, 0, 0, 0};
    std::vector<int> placement = {GPU, CPU, GPU, CPU, CPU};
    demoteShortRuns(prog, hostOp, placement, 2);
    EXPECT_EQ(placement, std::vector<int>({GPU, CPU, GPU, CPU, CPU}));
    demoteShortRuns(prog, hostOp, placement, 3);
    EXPECT_EQ(placement, std::vector<int>({CPU, CPU, CPU, CPU, CPU}));
}

TEST(DNN_Partition, fully_accelerated_graph_is_kept)
{
    std::vector<Ptr<LayerInfo> > prog = makeChain(2);
    std::vector<uchar> hostOp(prog.size(), 0);
    std::vector<int> placement = {GPU, GPU};
    demoteShortRuns(prog, hostOp, placement, 10);
    EXPECT_EQ(placement, std::vector<int>({GPU, GPU}));
}

TEST(DNN_Partition, segments_and_transitions)
{
    // args: 1 graph input, 2 constant weights, 7 graph output
    std::vector<Ptr<LayerInfo> > prog = {
        makeOp("Conv2", {1, 2}, {3}),
        makeOp("Shape", {3}, {5}),
        makeOp("ReLU", {3}, {4}),
        makeOp("Resize2", {4, 5}, {6}),
        makeOp("Conv2", {6, 2}, {7}),
    };
    std::vector<uchar> hostOp = {0, 1, 0, 0, 0};
    std::vector<int> placement = {GPU, CPU, GPU, CPU, GPU};
    std::vector<uchar> constArg(8, 0);
    constArg[2] = 1;

    std::vector<GraphSegment> segments;
    std::vector<ArgTransition> transitions;
    buildSegments(prog, hostOp, placement, {Arg(7)}, constArg, segments, transitions);

    ASSERT_EQ(segments.size(), 3u);
    EXPECT_EQ(segments[0].backend, GPU); EXPECT_EQ(segments[0].first, 0); EXPECT_EQ(segments[0].last, 2); EXPECT_EQ(segments[0].nops, 2);
    EXPECT_EQ(segments[1].backend, CPU); EXPECT_EQ(segments[1].first, 3); EXPECT_EQ(segments[1].last, 3);
    EXPECT_EQ(segments[2].backend, GPU); EXPECT_EQ(segments[2].first, 4); EXPECT_EQ(segments[2].last, 4);

    // the host Shape op reads only a header and the constant is uploaded once, so neither is a transition
    ASSERT_EQ(transitions.size(), 4u);
    EXPECT_EQ(transitions[0].arg.idx, 1); EXPECT_EQ(transitions[0].from, CPU); EXPECT_EQ(transitions[0].to, GPU); EXPECT_EQ(transitions[0].consumer, 0);
    EXPECT_EQ(transitions[1].arg.idx, 4); EXPECT_EQ(transitions[1].from, GPU); EXPECT_EQ(transitions[1].to, CPU); EXPECT_EQ(transitions[1].consumer, 3);
    EXPECT_EQ(transitions[2].arg.idx, 6); EXPECT_EQ(transitions[2].from, CPU); EXPECT_EQ(transitions[2].to, GPU); EXPECT_EQ(transitions[2].consumer, 4);
    EXPECT_EQ(transitions[3].arg.idx, 7); EXPECT_EQ(transitions[3].from, GPU); EXPECT_EQ(transitions[3].to, CPU); EXPECT_EQ(transitions[3].consumer, -1);
}

static Ptr<Layer> createNoExec(const Ptr<LayerInfo>&, void*)
{
    return Ptr<Layer>();
}

// Registers an executor for the lifetime of a test, for a type that has none of its own.
struct ScopedExec
{
    ScopedExec(const String& type_, int backend_, LayerFactory::ExecConstructor ctor, int minOpset, int maxOpset)
        : type(type_), backend(backend_)
    {
        if (LayerFactory::isExecSupported(type, backend))
            throw SkipTestException(type + " has a real executor for this backend; the test needs an op without one");
        LayerFactory::registerExec(type, backend, ctor, minOpset, maxOpset);
    }
    ~ScopedExec() { LayerFactory::unregisterExec(type, backend); }
    String type;
    int backend;
};

TEST(DNN_LayerFactory, exec_table_is_per_backend_and_checks_opset)
{
    const String type = "PartitionTestOp";
    {
        ScopedExec reg(type, GPU, createNoExec, 7, 13);
        EXPECT_TRUE(LayerFactory::isExecSupported(type, GPU));
        EXPECT_FALSE(LayerFactory::isExecSupported(type, GPU, 6));
        EXPECT_TRUE(LayerFactory::isExecSupported(type, GPU, 7));
        EXPECT_TRUE(LayerFactory::isExecSupported(type, GPU, 13));
        EXPECT_FALSE(LayerFactory::isExecSupported(type, GPU, 14));
        EXPECT_FALSE(LayerFactory::isExecSupported(type, CPU, 7));

        LayerFactory::registerExec(type, GPU, createNoExec);  // replaces the entry, with no opset limit
        EXPECT_TRUE(LayerFactory::isExecSupported(type, GPU, 100));
    }
    EXPECT_FALSE(LayerFactory::isExecSupported(type, GPU));
}

// A CUDA exec that always fails, standing in for an unsupported case or a kernel that does not compile.
class FailingCudaExec CV_FINAL : public Layer
{
public:
    static int attempts;
    static Ptr<Layer> create(const Ptr<LayerInfo>& info, void*)
    {
        Ptr<FailingCudaExec> e = makePtr<FailingCudaExec>();
        e->name = info->name;
        e->type = info->type;
        e->inputs = info->inputs;
        e->outputs = info->outputs;
        e->netimpl = info->netimpl;
        return e;
    }
    void forwardCUDA(InputArrayOfArrays, OutputArrayOfArrays, void*) CV_OVERRIDE
    {
        attempts++;
        CV_Error(Error::StsNotImplemented, "this test exec always fails");
    }
};
int FailingCudaExec::attempts = 0;

class DNN_Partition_CUDA : public ::testing::Test
{
public:
    void SetUp() CV_OVERRIDE
    {
        if (getAvailableTargets(DNN_BACKEND_CUDA).empty())
            throw SkipTestException("CUDA backend is not available");
        model = findDataFile("dnn/onnx/models/xfeat.onnx");
        cpuNet = readNetFromONNX(model);
        cudaNet = readNetFromONNX(model);
        cudaNet.setPreferableBackend(DNN_BACKEND_CUDA);
        cudaNet.setPreferableTarget(DNN_TARGET_CUDA);
        cudaNet.enableTF32(false);
        FailingCudaExec::attempts = 0;
    }

    static std::vector<Mat> run(Net& net, const Mat& inp)
    {
        net.setInput(inp);
        std::vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        return outs;
    }

    // a fresh input every call, so a device copy that was not refreshed shows up as a mismatch
    void compareWithCPU()
    {
        Mat inp(std::vector<int>{1, 1, 640, 640}, CV_32F);
        randu(inp, 0.f, 1.f);
        std::vector<Mat> ref = run(cpuNet, inp);
        std::vector<Mat> out = run(cudaNet, inp);
        ASSERT_EQ(ref.size(), out.size());
        for (size_t k = 0; k < ref.size(); k++)
            normAssert(ref[k], out[k], cv::format("output #%zu", k).c_str(), 1e-4, 1e-3);
    }

    // backend of every op of @p type in the CUDA net's main graph
    std::vector<int> placementOf(const std::string& type) const
    {
        std::vector<int> backends;
        Ptr<Graph> graph = cudaNet.getMainGraph();
        for (size_t i = 0; i < graph->prog().size(); i++) {
            const Ptr<LayerInfo>& op = graph->prog()[i];
            if (op && (type.empty() || op->type == type))
                backends.push_back(graph->opBackend((int)i));
        }
        return backends;
    }

    std::string model;
    Net cpuNet, cudaNet;
};

// xfeat.onnx has a Pad between two CUDA runs; Pad2 has no CUDA exec of its own.
TEST_F(DNN_Partition_CUDA, mixed_graph_matches_cpu)
{
    compareWithCPU();
    compareWithCPU();
    std::vector<int> all = placementOf("");
    EXPECT_GT(std::count(all.begin(), all.end(), GPU), 0);
    if (!LayerFactory::isExecSupported("Pad2", GPU)) {
        EXPECT_GT(std::count(all.begin(), all.end(), CPU), 0);
        for (int b : placementOf("Pad2"))
            EXPECT_EQ(b, CPU);
    }
}

TEST_F(DNN_Partition_CUDA, failing_op_moves_to_cpu_once)
{
    ScopedExec reg("Pad2", GPU, FailingCudaExec::create, 0, INT_MAX);
    compareWithCPU();
    const int attemptsAfterFirst = FailingCudaExec::attempts;
    EXPECT_GT(attemptsAfterFirst, 0);
    compareWithCPU();
    EXPECT_EQ(FailingCudaExec::attempts, attemptsAfterFirst);
    std::vector<int> pad = placementOf("Pad2");
    ASSERT_FALSE(pad.empty());
    for (int b : pad)
        EXPECT_EQ(b, CPU);
}

TEST_F(DNN_Partition_CUDA, exec_outside_its_opset_range_is_not_used)
{
    ScopedExec reg("Pad2", GPU, FailingCudaExec::create, 12, INT_MAX);  // xfeat.onnx is opset 11
    compareWithCPU();
    EXPECT_EQ(FailingCudaExec::attempts, 0);
    std::vector<int> pad = placementOf("Pad2");
    ASSERT_FALSE(pad.empty());
    for (int b : pad)
        EXPECT_EQ(b, CPU);
}

}} // namespace opencv_test
