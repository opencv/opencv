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

struct Layer_NaryEltwise : public TestBaseWithParam<tuple<Backend, Target> >
{
    void test_layer(const std::vector<int>& a_shape, const std::vector<int>& b_shape)
    {
//        int backendId = get<0>(GetParam());
//        int targetId = get<1>(GetParam());

        Mat a(a_shape, CV_32FC1);
        Mat b(b_shape, CV_32FC1);

        for (int i = 0; i < (int)a.total(); ++i)
            a.ptr<float>()[i] = (float)(i & 4095);

        for (int i = 0; i < (int)b.total(); ++i)
            b.ptr<float>()[i] = (float)(i ^ 4095);

        Net net;
        LayerParams lp;
        lp.type = "NaryEltwise";
        lp.name = "testLayer";
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

//            net.setPreferableBackend(backendId);
//            net.setPreferableTarget(targetId);
            Mat out = net.forward();
        }

        TEST_CYCLE()
        {
            Mat res = net.forward();
        }

        SANITY_CHECK_NOTHING();
    }

    void test_ref_layer(const std::vector<int>& a_shape, const std::vector<int>& b_shape)
    {
//        int backendId = get<0>(GetParam());
//        int targetId = get<1>(GetParam());

        Mat a(a_shape, CV_32FC1);
        Mat b(b_shape, CV_32FC1);

        for (int i = 0; i < (int)a.total(); ++i)
            a.ptr<float>()[i] = (float)(i & 4095);

        for (int i = 0; i < (int)b.total(); ++i)
            b.ptr<float>()[i] = (float)(i ^ 4095);

        Net net;
        LayerParams lp;
        lp.type = "Eltwise";
        lp.name = "testLayer";
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

//            net.setPreferableBackend(backendId);
//            net.setPreferableTarget(targetId);
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



PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW)
{
    test_layer({N, C, H, W}, {N, C, H, W});
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_NCHW_ref)
{
    test_ref_layer({N, C, H, W}, {N, C, H, W});
}

PERF_TEST_P_(Layer_NaryEltwise, NCHW_C)
{
    test_layer({N, C, H, W}, {C, 1, 1});
}

PERF_TEST_P_(Layer_NaryEltwise, NHWC_C)
{
    test_layer({N, H, W, C}, {1, C});
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

INSTANTIATE_TEST_CASE_P(/**/, Layer_Slice, dnnBackendsAndTargets(false, false));
INSTANTIATE_TEST_CASE_P(/**/, Layer_NaryEltwise, testing::Values(std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)));

} // namespace
