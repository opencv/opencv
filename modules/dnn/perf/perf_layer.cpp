// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test {

struct Layer_Slice : public TestBaseWithParam<tuple<Backend, Target> >
{
    template<int DIMS>
    void test_slice(const int* inputShape, const int* begin, const int* end)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat input(DIMS, inputShape, CV_32FC1, Scalar::all(0));
        for (int i = 0; i < (int)input.total(); ++i)
            input.ptr<float>()[i] = (float)(i & 4095);

        std::vector<Range> range(DIMS);
        for (int i = 0; i < DIMS; ++i)
            range[i] = Range(begin[i], end[i]);

        Net net;
        LayerParams lp;
        lp.type = "Slice";
        lp.name = "testLayer";
        lp.set("begin", DictValue::arrayInt<int*>((int*)&begin[0], DIMS));
        lp.set("end", DictValue::arrayInt<int*>((int*)&end[0], DIMS));
        net.addLayerToPrev(lp.name, lp.type, lp);

        // warmup
        {
            net.setInput(input);
            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();

            EXPECT_GT(cv::norm(out, NORM_INF), 0);
#if 0
            //normAssert(out, input(range));
            cout << input(range).clone().reshape(1, 1) << endl;
            cout << out.reshape(1, 1) << endl;
#endif
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }
};

static std::set<std::string> nary_eltwise_cuda_deny_ops = {"equal", "greater", "less", "mean", "pow", "sub"};

struct Layer_NaryEltwise : public TestBaseWithParam<tuple<Backend, Target> >
{
    void test_layer(const std::vector<int>& a_shape, const std::vector<int>& b_shape, const String op, bool isRef = false)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        if (!isRef && backendId == DNN_BACKEND_CUDA)
        {
            if (a_shape.size() != b_shape.size())
                throw SkipTestException("The test is skipped because inputs with different shape size are not supported.");

            for(int i = 0; i < a_shape.size(); i++)
                if (a_shape[i] != b_shape[i] && a_shape[i] != 1 && b_shape[i] != 1)
                    throw SkipTestException("The test is skipped because inputs are not supported.");

            if (nary_eltwise_cuda_deny_ops.find(op) != nary_eltwise_cuda_deny_ops.end())
                throw SkipTestException("The operator '" + op + "' is skipped because is not support with cuda currently.");
        }
        Mat a(a_shape, CV_32FC1);
        Mat b(b_shape, CV_32FC1);

        Scalar mean = 0.f;
        Scalar std = 1.f;
        randn(a, mean, std);
        randn(b, mean, std);


        Net net;
        LayerParams lp;
        if (isRef)
            lp.type = "Eltwise";
        else
            lp.type = "NaryEltwise";
        lp.name = "testLayer";
        lp.set("operation", op);
        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 1, id, 1);

        // warmup
        {
            std::vector<String> inpNames(2);
            inpNames[0] = "a";
            inpNames[1] = "b";
            net.setInputsNames(inpNames);
            net.setInput(a, inpNames[0]);
            net.setInput(b, inpNames[1]);

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }

    int N = 8;
    int C = 256;
    int H = 128;
    int W = 100;
};


PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_add)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "add");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_div)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "div");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_ref_div)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "div", true);
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_equal)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "equal");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_greater)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "greater");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_less)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "less");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_max)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "max");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_ref_max)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "max", true);
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_mean)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "mean");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_min)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "min");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_ref_min)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "min", true);
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_mul)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "mul");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_ref_mul)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "prod", true);
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_pow)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "pow");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_sub)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "sub");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_sum)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "sum");
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_ref_sum)
{
    test_layer({N, C, H, W}, {N, C, H, W}, "sum", true);
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_C_sum)
{
    test_layer({N, C, H, W}, {C, 1, 1}, "sum");
}

PERF_TEST_P_(Layer_NaryEltwise, NHWC_C)
{
    test_layer({N, H, W, C}, {1, C}, "sum");
}

PERF_TEST_P_(Layer_NaryEltwise, NHWC_H)
{
    test_layer({N, H, W, C}, {1, H, 1, 1}, "sum");
}

PERF_TEST_P_(Layer_Slice, YOLOv4_tiny_1)
{
    const int inputShape[4] = {1, 64, 104, 104};
    const int begin[] = {0, 32, 0, 0};
    const int end[] = {1, 64, 104, 104};
    test_slice<4>(inputShape, begin, end);
}

PERF_TEST_P_(Layer_Slice, YOLOv4_tiny_2)
{
    const int inputShape[4] = {1, 128, 52, 52};
    const int begin[] = {0, 64, 0, 0};
    const int end[] = {1, 128, 52, 52};
    test_slice<4>(inputShape, begin, end);
}

PERF_TEST_P_(Layer_Slice, YOLOv4_tiny_3)
{
    const int inputShape[4] = {1, 256, 26, 26};
    const int begin[] = {0, 128, 0, 0};
    const int end[] = {1, 256, 26, 26};
    test_slice<4>(inputShape, begin, end);
}


PERF_TEST_P_(Layer_Slice, FastNeuralStyle_eccv16)
{
    const int inputShape[4] = {1, 128, 80, 100};
    const int begin[] = {0, 0, 2, 2};
    const int end[] = {1, 128, 76, 96};
    test_slice<4>(inputShape, begin, end);
}

using Layer_Scatter = TestBaseWithParam<tuple<std::vector<int>, std::string, int, tuple<Backend, Target>>>;
PERF_TEST_P_(Layer_Scatter, scatter) {
    std::vector<int> shape = get<0>(GetParam());
    std::string reduction = get<1>(GetParam());
    int axis = get<2>(GetParam());
    int backend_id = get<0>(get<3>(GetParam()));
    int target_id = get<1>(get<3>(GetParam()));

    Mat data(shape, CV_32FC1);
    Mat indices(shape, CV_32FC1);
    Mat updates(shape, CV_32FC1);

    randn(data, 0.f, 1.f);
    randu(indices, 0, shape[axis]);
    randn(updates, 0.f, 1.f);

    indices.convertTo(indices, CV_32SC1, 1, -1);

    Net net;
    LayerParams lp;
    lp.type = "Scatter";
    lp.name = "testLayer";
    lp.set("reduction", reduction);
    lp.set("axis", axis);

    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 0, id, 0);
    net.connect(0, 1, id, 1);
    net.connect(0, 2, id, 2);

    // warmup
    {
        std::vector<String> input_names{"data", "indices", "updates"};
        net.setInputsNames(input_names);
        net.setInput(data, input_names[0]);
        net.setInput(indices, input_names[1]);
        net.setInput(updates, input_names[2]);

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);
        Mat out = net.forward();
    }

    // perf
    TEST_CYCLE()
    {
        Mat res = net.forward();
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Scatter, Combine(
    Values(std::vector<int>{2, 128, 64, 50}),
    Values(std::string("none"), std::string("add")),
    Values(0), // use Values(0, 1, 2, 3) for more details
    dnnBackendsAndTargets(/* withInferenceEngine= */ false,
                          /* withHalide= */          false,
                          /* withCpuOCV= */          true,
                          /* withVkCom= */           false,
                          /* withCUDA= */            false,
                          /* withNgraph= */          false,
                          /* withWebnn= */           false,
                          /* withCann= */            false) // only test on CPU
));

using Layer_ScatterND = TestBaseWithParam<tuple<std::vector<int>, std::string, tuple<Backend, Target>>>;
PERF_TEST_P_(Layer_ScatterND, scatterND) {
    std::vector<int> shape = get<0>(GetParam());
    std::string reduction = get<1>(GetParam());
    int backend_id = get<0>(get<2>(GetParam()));
    int target_id = get<1>(get<2>(GetParam()));

    std::vector<int> indices_shape(shape);
    indices_shape.push_back(int(shape.size()));
    Mat data(shape, CV_32FC1);
    Mat indices(indices_shape, CV_32FC1);
    Mat updates(shape, CV_32FC1);

    randn(data, 0.f, 1.f);
    randn(updates, 0.f, 1.f);

    // Create indices such that indices[n_i, c_j, h_k, w_l, :4] = [i, j, k, l]
    std::vector<int> current_index_tuple(shape.size());
    int total = data.total();
    std::vector<int> indices_step;
    for (int i = 0; i < indices.dims; i++)
    {
        int step = indices.step.p[i] / sizeof(float);
        indices_step.push_back(step);
    }
    int t, j, idx, offset_at_idx, offset;
    auto *indices_ptr = indices.ptr<float>();
    for (int i = 0; i < total; i++)
    {
        t = i;
        for (j = shape.size() - 1; j >= 0; j--)
        {
            idx = t / shape[j];
            offset_at_idx = (int)(t - idx * shape[j]);
            current_index_tuple[j] = offset_at_idx;
            t = idx;
        }

        offset = 0;
        for (j = 0; j < shape.size(); j++)
            offset += current_index_tuple[j] * indices_step[j];

        for (j = 0; j < shape.size(); j++)
            indices_ptr[offset + j] = current_index_tuple[j];
    }

    Net net;
    LayerParams lp;
    lp.type = "ScatterND";
    lp.name = "testLayer";
    lp.set("reduction", reduction);

    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 0, id, 0);
    net.connect(0, 1, id, 1);
    net.connect(0, 2, id, 2);

    // warmup
    {
        std::vector<String> input_names{"data", "indices", "updates"};
        net.setInputsNames(input_names);
        net.setInput(data, input_names[0]);
        net.setInput(indices, input_names[1]);
        net.setInput(updates, input_names[2]);

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);
        Mat out = net.forward();
    }

    TEST_CYCLE()
    {
        Mat res = net.forward();
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_ScatterND, Combine(
    Values(std::vector<int>{2, 128, 64, 50}),
    Values(std::string("none"), std::string("add")),
    dnnBackendsAndTargets(/* withInferenceEngine= */ false,
                          /* withHalide= */          false,
                          /* withCpuOCV= */          true,
                          /* withVkCom= */           false,
                          /* withCUDA= */            false,
                          /* withNgraph= */          false,
                          /* withWebnn= */           false,
                          /* withCann= */            false) // only test on CPU
));

struct Layer_LayerNorm : public TestBaseWithParam<tuple<Backend, Target> >
{
    void test_layer(const std::vector<int>& x_shape)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat x(x_shape, CV_32FC1);
        Mat scale(x_shape.back(), 1, CV_32FC1);
        Mat b(x_shape.back(), 1, CV_32FC1);

        randu(x, 0.f, 1.f);
        randu(scale, 0.f, 1.f);
        randu(b, 0.f, 1.f);


        Net net;
        LayerParams lp;
        lp.type = "LayerNormalization";
        lp.name = "testLayer";
        lp.set("axis", 2);
        lp.set("hasBias", true);
        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);
        net.connect(0, 1, id, 1);
        net.connect(0, 2, id, 2);

        // warmup
        {
            std::vector<String> inpNames(3);
            inpNames[0] = "x";
            inpNames[1] = "scale";
            inpNames[2] = "b";
            net.setInputsNames(inpNames);
            net.setInput(x, inpNames[0]);
            net.setInput(scale, inpNames[1]);
            net.setInput(b, inpNames[2]);

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }

    int N = 1;
    int H = 50;
    int W = 768;
};

PERF_TEST_P_(Layer_LayerNorm, LayerNorm)
{
    test_layer({N, H ,W});
}

struct Layer_LayerNormExpanded : public TestBaseWithParam<tuple<Backend, Target> >
{
    void test_layer(const std::vector<int>& x_shape)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat x(x_shape, CV_32FC1);
        Mat scale(1, x_shape.back(), CV_32FC1); // transpose to pass shape check
        Mat b(1, x_shape.back(), CV_32FC1);     // transpose to pass shape check

        randu(x, 0.f, 1.f);
        randu(scale, 0.f, 1.f);
        randu(b, 0.f, 1.f);

        // sub graph structure:
        //   -> ReduceMean ->     -> Pow(2) -> ReduceMean -> Add(epsilon) -> Sqrt ->
        // x                  Sub                                                    Div -> Mul(scale) -> Add(bias)
        //   --------------->     ------------------------------------------------->

        Net net;

        LayerParams lp_rm;
        lp_rm.type = "Reduce";
        lp_rm.name = "reducemean1";
        lp_rm.set("reduce", "AVE");
        std::vector<int> deleteDims(1, x_shape.back());
        lp_rm.set("deleted_dims", DictValue::arrayInt(&deleteDims[0], deleteDims.size()));
        std::vector<int> targetDims(x_shape.begin(), x_shape.end());
        targetDims[x_shape.size() - 1] = 1;
        lp_rm.set("target_dims", DictValue::arrayInt(&targetDims[0], targetDims.size()));
        int id_rm = net.addLayerToPrev(lp_rm.name, lp_rm.type, lp_rm);
        net.connect(0, 0, id_rm, 0);

        LayerParams lp_sub;
        lp_sub.type = "NaryEltwise";
        lp_sub.name = "sub1";
        lp_sub.set("operation", "sub");
        int id_sub = net.addLayer(lp_sub.name, lp_sub.type, lp_sub);
        net.connect(0, 0, id_sub, 0);
        net.connect(id_rm, 0, id_sub, 1);

        Mat pow_const(1, 1, CV_32FC1);
        pow_const.at<float>(0) = 2.f;
        LayerParams lp_pow_const;
        lp_pow_const.type = "Const";
        lp_pow_const.name = "const1";
        lp_pow_const.blobs.push_back(pow_const);
        int id_pow_const = net.addLayer(lp_pow_const.name, lp_pow_const.type, lp_pow_const);
        LayerParams lp_pow;
        lp_pow.type = "NaryEltwise";
        lp_pow.name = "pow1";
        lp_pow.set("operation", "pow");
        int id_pow = net.addLayer(lp_pow.name, lp_pow.type, lp_pow);
        net.connect(id_sub, 0, id_pow, 0);
        net.connect(id_pow_const, 0, id_pow, 1);

        LayerParams lp_rm1;
        lp_rm1.type = "Reduce";
        lp_rm1.name = "reducemean2";
        lp_rm1.set("reduce", "AVE");
        lp_rm1.set("deleted_dims", DictValue::arrayInt(&deleteDims[0], deleteDims.size()));
        lp_rm1.set("target_dims", DictValue::arrayInt(&targetDims[0], targetDims.size()));
        int id_rm1 = net.addLayer(lp_rm1.name, lp_rm1.type, lp_rm1);
        net.connect(id_pow, 0, id_rm1, 0);

        Mat add_const(1, 1, CV_32F);
        add_const.at<float>(0) = 1e-5;
        LayerParams lp_add_const;
        lp_add_const.type = "Const";
        lp_add_const.name = "const2";
        lp_add_const.blobs.push_back(add_const);
        int id_add_const = net.addLayer(lp_add_const.name, lp_add_const.type, lp_add_const);
        LayerParams lp_add;
        lp_add.type = "NaryEltwise";
        lp_add.name = "add1";
        lp_add.set("operation", "add");
        int id_add = net.addLayer(lp_add.name, lp_add.type, lp_add);
        net.connect(id_rm1, 0, id_add, 0);
        net.connect(id_add_const, 0, id_add, 1);

        LayerParams lp_sqrt;
        lp_sqrt.type = "Sqrt";
        lp_sqrt.name = "sqrt1";
        int id_sqrt = net.addLayer(lp_sqrt.name, lp_sqrt.type, lp_sqrt);
        net.connect(id_add, 0, id_sqrt, 0);

        LayerParams lp_div;
        lp_div.type = "NaryEltwise";
        lp_div.name = "div1";
        lp_div.set("operation", "div");
        int id_div = net.addLayer(lp_div.name, lp_div.type, lp_div);
        net.connect(id_sub, 0, id_div, 0);
        net.connect(id_sqrt, 0, id_div, 1);

        LayerParams lp_mul;
        lp_mul.type = "NaryEltwise";
        lp_mul.name = "mul1";
        lp_mul.set("operation", "mul");
        int id_mul = net.addLayer(lp_mul.name, lp_mul.type, lp_mul);
        net.connect(id_div, 0, id_mul, 0);
        net.connect(0, 1, id_mul, 1);

        LayerParams lp_add1;
        lp_add1.type = "NaryEltwise";
        lp_add1.name = "add2";
        lp_add1.set("operation", "add");
        int id_add1 = net.addLayer(lp_add1.name, lp_add1.type, lp_add1);
        net.connect(id_mul, 0, id_add1, 0);
        net.connect(0, 2, id_add1, 1);

        // warmup
        {
            std::vector<String> inpNames(3);
            inpNames[0] = "x";
            inpNames[1] = "scale";
            inpNames[2] = "b";
            net.setInputsNames(inpNames);
            net.setInput(x, inpNames[0]);
            net.setInput(scale, inpNames[1]);
            net.setInput(b, inpNames[2]);

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }

    int N = 1;
    int H = 50;
    int W = 768;
};

PERF_TEST_P_(Layer_LayerNormExpanded, DISABLED_LayerNormExpanded)
{
    test_layer({N, H ,W});
}

struct Layer_GatherElements : public TestBaseWithParam<tuple<Backend, Target> >
{
    void test_layer(const std::vector<int>& data_shape, const std::vector<int>& indices_shape, int axis = 0)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat data(data_shape, CV_32FC1);
        Mat indices(indices_shape, CV_32FC1);

        randu(data, 0.f, 1.f);
        randu(indices, 0, data_shape[axis]);

        Net net;
        LayerParams lp;
        lp.type = "GatherElements";
        lp.name = "testLayer";
        lp.set("axis", axis);
        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);
        net.connect(0, 1, id, 1);

        // warmup
        {
            std::vector<String> inpNames(3);
            inpNames[0] = "data";
            inpNames[1] = "indices";
            net.setInputsNames(inpNames);
            net.setInput(data, inpNames[0]);
            net.setInput(indices, inpNames[1]);

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }
};

PERF_TEST_P_(Layer_GatherElements, GatherElements)
{
    test_layer({2700, 1, 2914}, {2700, 1, 81}, 2);
}

struct Layer_InstanceNorm : public TestBaseWithParam<tuple<Backend, Target> >
{
    void test_layer(const std::vector<int>& x_shape)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat x(x_shape, CV_32FC1);
        Mat scale(x_shape[1], 1, CV_32FC1);
        Mat b(x_shape[1], 1, CV_32FC1);

        randu(x, 0.f, 1.f);
        randu(scale, 0.f, 1.f);
        randu(b, 0.f, 1.f);

        Net net;
        LayerParams lp;
        lp.type = "InstanceNormalization";
        lp.name = "testLayer";
        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);
        net.connect(0, 1, id, 1);
        net.connect(0, 2, id, 2);

        // warmup
        {
            std::vector<String> inpNames{"x", "scale", "b"};
            net.setInputsNames(inpNames);
            net.setInput(x, inpNames[0]);
            net.setInput(scale, inpNames[1]);
            net.setInput(b, inpNames[2]);

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }

    int N = 2;
    int C = 64;
    int H = 180;
    int W = 240;
};

PERF_TEST_P_(Layer_InstanceNorm, InstanceNorm)
{
    test_layer({N, C, H, W});
}

struct Layer_Attention : public TestBaseWithParam<tuple<Backend, Target>> {
    void test_layer(const std::vector<int> x_shape, const std::vector<int> qkv_hidden_sizes, const int num_heads) {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        auto qk_hidden_size = qkv_hidden_sizes[0];
        auto v_hidden_size = qkv_hidden_sizes[2];

        auto input_hidden_size = x_shape[2];
        auto hidden_size = qk_hidden_size + qk_hidden_size + v_hidden_size;

        Mat x(x_shape, CV_32F);
        Mat weight(std::vector<int>{input_hidden_size, hidden_size}, CV_32F);
        Mat bias(std::vector<int>{hidden_size}, CV_32F);

        randu(x, 0.f, 1.f);
        randu(weight, 0.f, 1.f);
        randu(bias, 0.f, 1.f);

        LayerParams lp;
        lp.type = "Attention";
        lp.name = "testLayer";
        lp.set("num_heads", num_heads);
        lp.set("qkv_hidden_sizes", DictValue::arrayInt(qkv_hidden_sizes.data(), qkv_hidden_sizes.size()));

        Net net;
        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);
        net.connect(0, 1, id, 1);
        net.connect(0, 2, id, 2);

        {
            std::vector<std::string> input_names{"x", "weight", "bias"};
            net.setInputsNames(input_names);
            net.setInput(x, input_names[0]);
            net.setInput(weight, input_names[1]);
            net.setInput(bias, input_names[2]);

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat out = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }
};

PERF_TEST_P_(Layer_Attention, VisionTransformer) {
    test_layer({1, 197, 768}, {768, 768, 768}, 12);
}

struct Layer_GroupNorm : public TestBaseWithParam<tuple<Backend, Target> >
{
    void test_layer(const std::vector<int>& x_shape, int num_groups)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat x(x_shape, CV_32FC1);
        Mat scale(x_shape[1], 1, CV_32FC1);
        Mat b(x_shape[1], 1, CV_32FC1);

        randu(x, 0.f, 1.f);
        randu(scale, 0.f, 1.f);
        randu(b, 0.f, 1.f);

        Net net;
        LayerParams lp;
        lp.type = "GroupNormalization";
        lp.name = "testLayer";
        lp.set("num_groups", num_groups);

        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);
        net.connect(0, 1, id, 1);
        net.connect(0, 2, id, 2);

        // warmup
        {
            std::vector<String> inpNames{"x", "scale", "b"};
            net.setInputsNames(inpNames);
            net.setInput(x, inpNames[0]);
            net.setInput(scale, inpNames[1]);
            net.setInput(b, inpNames[2]);

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }

    int N = 2;
    int C = 64;
    int H = 180;
    int W = 240;
    int num_groups = 16;
};

PERF_TEST_P_(Layer_GroupNorm, GroupNorm)
{
    test_layer({N, C, H, W}, num_groups);
}


INSTANTIATE_TEST_CASE_P(/**/, Layer_Slice, dnnBackendsAndTargets(false, false));
INSTANTIATE_TEST_CASE_P(/**/, Layer_NaryEltwise, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));
#ifdef HAVE_CUDA
INSTANTIATE_TEST_CASE_P(CUDA, Layer_NaryEltwise, testing::Values(std::make_tuple(DNN_BACKEND_CUDA, DNN_TARGET_CUDA)));
#endif
#ifdef HAVE_VULKAN
INSTANTIATE_TEST_CASE_P(VULKAN, Layer_NaryEltwise, testing::Values(std::make_tuple(DNN_BACKEND_VKCOM, DNN_TARGET_VULKAN)));
#endif
INSTANTIATE_TEST_CASE_P(/**/, Layer_LayerNorm, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));
INSTANTIATE_TEST_CASE_P(/**/, Layer_LayerNormExpanded, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));
INSTANTIATE_TEST_CASE_P(/**/, Layer_GatherElements, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));
INSTANTIATE_TEST_CASE_P(/**/, Layer_InstanceNorm, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));
INSTANTIATE_TEST_CASE_P(/**/, Layer_Attention, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));
INSTANTIATE_TEST_CASE_P(/**/, Layer_GroupNorm, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));

typedef TestBaseWithParam<tuple<Vec4i, int, bool, tuple<Backend, Target> > > Layer_FullyConnected;
PERF_TEST_P_(Layer_FullyConnected, fc)
{
    std::vector<int> inpShape;
    inpShape.reserve(4);
    for (int i = 0; i < 4; ++i) {
        int dim = get<0>(GetParam())[i];
        if (dim == 0)
            break;
        inpShape.push_back(dim);
    }
    Mat input(inpShape, CV_32F);
    randn(input, 0, 1);

    int axis = input.dims - 1;
    int outDims = get<1>(GetParam());
    bool isMatMul = get<2>(GetParam());
    int backendId = get<0>(get<3>(GetParam()));
    int targetId = get<1>(get<3>(GetParam()));

    if (inpShape.size() == 4 && inpShape[0] == 5 && inpShape[1] == 16 && inpShape[2] == 512 && inpShape[3] == 128 && outDims >= 512)
        applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    std::vector<int> weightShape;
    if (isMatMul) {
        weightShape = inpShape;
        weightShape[weightShape.size() - 2] = outDims;
    } else {
        weightShape = {outDims, (int)input.total(axis, input.dims)};
    }
    Mat weights(weightShape, CV_32F);
    randn(weights, 0, 1);

    LayerParams lp;
    lp.set("axis", input.dims - 1);
    lp.set("is_matmul", weights.dims > 2);
    lp.set("bias_term", false);
    lp.set("num_output", (int)weights.total(0, weights.dims - 1));
    lp.blobs.resize(1, weights);

    Net net;
    net.addLayerToPrev("matmul", "InnerProduct", lp);

    net.setInput(input);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    // warmup
    Mat output = net.forward();

    TEST_CYCLE()
    {
        net.forward();
    }
    SANITY_CHECK_NOTHING();
}
INSTANTIATE_TEST_CASE_P(/**/, Layer_FullyConnected, Combine(
    Values(                // input size
        Vec4i(5, 512, 384),
        Vec4i(5, 16, 512, 128)
    ),
    Values(256, 512, 1024),  // output dimension
    testing::Bool(),         // is_matmul
    dnnBackendsAndTargets()
));

typedef TestBaseWithParam<tuple<std::vector<int>, int, tuple<Backend, Target> > > Layer_Softmax;
PERF_TEST_P_(Layer_Softmax, softmax_3d) {
    std::vector<int> shape = get<0>(GetParam());
    int axis = get<1>(GetParam());
    int backendId = get<0>(get<2>(GetParam()));
    int targetId = get<1>(get<2>(GetParam()));

    Mat data(shape, CV_32FC1);
    Scalar mean = 0.f;
    Scalar std = 1.f;
    randn(data, mean, std);

    Net net;
    LayerParams lp;
    lp.type = "Softmax";
    lp.name = "testLayer";
    lp.set("axis", axis);

    net.addLayerToPrev(lp.name, lp.type, lp);
    // warmup
    {
        net.setInput(data);
        net.setPreferableBackend(backendId);
        net.setPreferableTarget(targetId);
        Mat out = net.forward();
    }

    TEST_CYCLE() {
        Mat res = net.forward();
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Softmax, Combine(
    Values(                // input size
            std::vector<int>({16, 50, 50}),
            std::vector<int>({16, 197, 197}),
            std::vector<int>({16, 1024, 1024})
    ),
    Values(0, 1, 2),  // axis
    dnnBackendsAndTargets(/* withInferenceEngine= */ false,
                          /* withHalide= */          false,
                          /* withCpuOCV= */          true,
                          /* withVkCom= */           false,
                          /* withCUDA= */            false,
                          /* withNgraph= */          false,
                          /* withWebnn= */           false,
                          /* withCann= */            false) // only test on CPU
));

struct Layer_Elementwise : public TestBaseWithParam<tuple<Backend, Target>> {
    void test_layer(const std::string &op_type, const std::vector<int> &input_shape) {
        int backend_id = get<0>(GetParam());
        int target_id = get<1>(GetParam());

        Mat input(input_shape, CV_32F);
        randu(input, 0.f, 1.f);

        LayerParams lp;
        lp.type = op_type;
        lp.name = cv::format("PerfLayer/%s", op_type.c_str());

        Net net;
        net.addLayerToPrev(lp.name, lp.type, lp);

        // Warmup
        {
            net.setInput(input);
            net.setPreferableBackend(backend_id);
            net.setPreferableTarget(target_id);
            net.forward();
        }

        TEST_CYCLE() {
            net.forward();
        }

        SANITY_CHECK_NOTHING();
    }

    int N = 2;
    int C = 32;
    int H = 416;
    int W = 416;
};

PERF_TEST_P_(Layer_Elementwise, Gelu) {
    test_layer("Gelu", std::vector<int>{1, 50, 3072});
}
PERF_TEST_P_(Layer_Elementwise, Swish) {
    test_layer("Swish", std::vector<int>{N, C, H, W});
}
PERF_TEST_P_(Layer_Elementwise, Mish) {
    test_layer("Mish", std::vector<int>{N, C, H, W});
}
PERF_TEST_P_(Layer_Elementwise, Elu) {
    test_layer("ELU", std::vector<int>{N, C, H, W});
}
PERF_TEST_P_(Layer_Elementwise, Celu) {
    test_layer("Celu", std::vector<int>{N, C, H, W});
}
PERF_TEST_P_(Layer_Elementwise, Selu) {
    test_layer("Selu", std::vector<int>{N, C, H, W});
}
PERF_TEST_P_(Layer_Elementwise, HardSwish) {
    test_layer("HardSwish", std::vector<int>{N, C, H, W});
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Elementwise,
                        dnnBackendsAndTargets(/* withInferenceEngine= */ true,
                                              /* withHalide= */          false,
                                              /* withCpuOCV= */          true,
                                              /* withVkCom= */           false,
                                              /* withCUDA= */            true,
                                              /* withNgraph= */          true,
                                              /* withWebnn= */           false,
                                              /* withCann= */            false));

} // namespace
