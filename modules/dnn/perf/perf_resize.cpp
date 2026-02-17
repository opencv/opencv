// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

struct Layer_Resize : public TestBaseWithParam<tuple<Backend, Target>>
{
    void test_layer(const std::vector<int>& inpShape, int outH, int outW, const String& interp)
    {
        int backendId = get<0>(GetParam());
        int targetId = get<1>(GetParam());

        Mat input(inpShape, CV_32FC1);
        randu(input, 0.f, 1.f);

        Net net;
        LayerParams lp;
        lp.type = "Resize";
        lp.name = "testLayer";
        lp.set("interpolation", interp);
        lp.set("width", outW);
        lp.set("height", outH);

        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);

        // warmup
        {
            net.setInputsNames({"data"});
            net.setInput(input, "data");
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

PERF_TEST_P_(Layer_Resize, Resize_Upsample_Linear)
{
    // N=4, C=64, H=64, W=64 -> 128x128 (x2 upsample)
    // Common in segmentation/detection heads
    test_layer({4, 64, 64, 64}, 128, 128, "opencv_linear");
}

PERF_TEST_P_(Layer_Resize, Resize_Downsample_Nearest)
{
    // N=4, C=128, H=128, W=128 -> 64x64 (x0.5 downsample)
    test_layer({4, 128, 128, 128}, 64, 64, "nearest");
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Resize, dnnBackendsAndTargets());

} // namespace opencv_test
