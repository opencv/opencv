// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test {

struct GemmParam_t {
    int shapeIn0[2]; // Mat A
    int shapeIn1[2]; // Mat B
    double declared_flops;
};

// MatA * MatB = MatC
static const GemmParam_t testGemmConfigs[] = {
        {{ 128 , 128 }, { 128 , 128 },  6291456.},
        {{ 128 , 128 }, { 128 , 1024 },  50331648.},
        {{ 128 , 128 }, { 128 , 4096 },  201326592.},
        {{ 128 , 1024 }, { 1024 , 128 },  50331648.},
        {{ 128 , 1024 }, { 1024 , 1024 },  402653184.},
        {{ 128 , 1024 }, { 1024 , 4096 },  1610612736.},
        {{ 128 , 4096 }, { 4096 , 128 },  201326592.},
        {{ 128 , 4096 }, { 4096 , 1024 },  1610612736.},
        {{ 128 , 4096 }, { 4096 , 4096 },  6442450944.},
        {{ 1024 , 128 }, { 128 , 128 },  50331648.},
        {{ 1024 , 128 }, { 128 , 1024 },  402653184.},
        {{ 1024 , 128 }, { 128 , 4096 },  1610612736.},
        {{ 1024 , 1024 }, { 1024 , 128 },  402653184.},
        {{ 1024 , 1024 }, { 1024 , 1024 },  3221225472.},
        {{ 1024 , 1024 }, { 1024 , 4096 },  12884901888.},
        {{ 1024 , 4096 }, { 4096 , 128 },  1610612736.},
        {{ 1024 , 4096 }, { 4096 , 1024 },  12884901888.},
        {{ 1024 , 4096 }, { 4096 , 4096 },  51539607552.},
        {{ 4096 , 128 }, { 128 , 128 },  201326592.},
        {{ 4096 , 128 }, { 128 , 1024 },  1610612736.},
        {{ 4096 , 128 }, { 128 , 4096 },  6442450944.},
        {{ 4096 , 1024 }, { 1024 , 128 },  1610612736.},
        {{ 4096 , 1024 }, { 1024 , 1024 },  12884901888.},
        {{ 4096 , 1024 }, { 1024 , 4096 },  51539607552.},
        {{ 4096 , 4096 }, { 4096 , 128 },  6442450944.},
        {{ 4096 , 4096 }, { 4096 , 1024 },  51539607552.},
        {{ 4096 , 4096 }, { 4096 , 4096 },  206158430208.},
};

struct GemmParamID
{
    enum {
        GEMM_0 = 0,
        GEMM_LAST = sizeof(testGemmConfigs) / sizeof(testGemmConfigs[0])
    };

    int val_;
    GemmParamID(int val = 0) : val_(val) {}
    operator int() const { return val_; }
    static ::testing::internal::ParamGenerator<GemmParamID> all()
    {
        enum { NUM = (int)GEMM_LAST };
        GemmParamID v_[NUM]; for (int i = 0; i < NUM; ++i) { v_[i] = GemmParamID(i); } // reduce generated code size
        return ::testing::ValuesIn(v_, v_ + NUM);
    }
};

static inline void PrintTo(const GemmParamID& v, std::ostream* os)
{
    CV_Assert((int)v >= 0); CV_Assert((int)v < GemmParamID::GEMM_LAST);
    const GemmParam_t& p = testGemmConfigs[(int)v];

    *os << "GFLOPS=" << cv::format("%.3f", p.declared_flops * 1e-9)
        << ", Mat0 = {" << p.shapeIn0[0] << ", " << p.shapeIn0[1]
        << "} , Mat1 = {" << p.shapeIn1[0] << ", " << p.shapeIn1[1]<<"}";
}

typedef tuple<GemmParamID, tuple<Backend, Target> > GemmTestParam_t;
typedef TestBaseWithParam<GemmTestParam_t> Gemm;

PERF_TEST_P_(Gemm, gemm2D)
{
    int test_id = (int)get<0>(GetParam());
    ASSERT_GE(test_id, 0);
    ASSERT_LT(test_id, GemmParamID::GEMM_LAST);
    const GemmParam_t& params = testGemmConfigs[test_id];
    double declared_flops = params.declared_flops;

    MatShape inputShape0 = {params.shapeIn0[0], params.shapeIn0[1]};
    MatShape inputShape1 = {params.shapeIn1[0], params.shapeIn1[1]};

    Backend backendId = get<0>(get<1>(GetParam()));
    Target targetId = get<1>(get<1>(GetParam()));

    Mat input0(2, &inputShape0[0], CV_32F);
    Mat input1(2, &inputShape1[0], CV_32F);

    randu(input0, -1.0f, 1.0f);
    randu(input1, -1.0f, 1.0f);

    LayerParams lp;

    lp.set("bias_term", false);
    lp.type = "InnerProduct";
    lp.name = "testLayer";

    Net net;
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    // warmup
    std::vector<String> inpNames(2);
    inpNames[0] = "a";
    inpNames[1] = "b";
    net.setInputsNames(inpNames);
    net.setInput(input0, inpNames[0]);
    net.setInput(input1, inpNames[1]);

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    Mat output = net.forward();

    std::vector<MatShape> inputShapes;
    inputShapes.push_back(inputShape0);
    inputShapes.push_back(inputShape1);

    size_t weightsMemory = 0, blobsMemory = 0;
    net.getMemoryConsumption(inputShapes, weightsMemory, blobsMemory);
    int64 flops = net.getFLOPS(inputShapes);
    CV_Assert(flops > 0);

    std::cout
            << "Mat0=" << divUp(input0.total() * input0.elemSize(), 1u<<10) << " Kb " << inputShape0
            << "    Mat1=" << divUp(input1.total() * input1.elemSize(), 1u<<10) << " Kb " << inputShape1
            << "    OUT=" << divUp(output.total() * output.elemSize(), 1u<<10) << " Kb " << shape(output)
            << "    MFLOPS=" << flops * 1e-6 << std::endl;

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }
    EXPECT_NEAR(flops, declared_flops, declared_flops * 1e-6);
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Gemm, Combine(
        GemmParamID::all(),
        dnnBackendsAndTargets(false, false)  // defined in ../test/test_common.hpp
));

} // namespace
