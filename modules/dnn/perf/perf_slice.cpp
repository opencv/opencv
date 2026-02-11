// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

struct Layer_Slice_Test : public TestBaseWithParam<tuple<Backend, Target>>
{
    void test_slice(const std::vector<int>& input_shape, int axis, int begin, int end, int step = 1)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat data(input_shape, CV_32FC1);
        randu(data, 0.f, 1.f);

        Net net;
        LayerParams lp;
        lp.type = "Slice";
        lp.name = "testLayer";
        lp.set("axis", axis);

        std::vector<int> begins(input_shape.size(), 0);
        std::vector<int> ends = input_shape;
        std::vector<int> steps(input_shape.size(), 1);

        begins[axis] = begin;
        ends[axis] = end;
        steps[axis] = step;

        lp.set("begin", DictValue::arrayInt(&begins[0], begins.size()));
        lp.set("end", DictValue::arrayInt(&ends[0], ends.size()));
        if (step != 1) {
            lp.set("steps", DictValue::arrayInt(&steps[0], steps.size()));
        }

        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);

        net.setInputsNames({"data"});

        // warmup
        {
            net.setInput(data, "data");
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

PERF_TEST_P_(Layer_Slice_Test, Slice_Contiguous_Axis0)
{
    test_slice({64, 128, 128}, 0, 10, 54);
}

PERF_TEST_P_(Layer_Slice_Test, Slice_Contiguous_Axis2)
{
    test_slice({64, 128, 128}, 2, 10, 118);
}

PERF_TEST_P_(Layer_Slice_Test, Slice_Small_Middle)
{
    test_slice({32, 64, 32}, 1, 20, 40);
}

PERF_TEST_P_(Layer_Slice_Test, Slice_Strided_Axis0_Step2)
{
    // Strided slice on outer axis.
    // [64, 128, 128] -> [0:64:2, ...]
    test_slice({64, 128, 128}, 0, 0, 64, 2);
}

PERF_TEST_P_(Layer_Slice_Test, Slice_Strided_Axis2_Step2)
{
    // Strided slice on inner axis.
    // [64, 128, 128] -> [..., 0:128:2]
    test_slice({64, 128, 128}, 2, 0, 128, 2);
}


INSTANTIATE_TEST_CASE_P(/**/, Layer_Slice_Test, dnnBackendsAndTargets(false, false, true, false, false, false, false, false));

} // namespace opencv_test
