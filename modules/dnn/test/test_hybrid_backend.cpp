#include "test_precomp.hpp"
#include <opencv2/dnn.hpp>

namespace opencv_test {

TEST(DNN_Hybrid, MixedCPUAndCUDAExecution)
{
    cv::dnn::Net net;

    // Input
    cv::Mat input = cv::Mat::ones(1, 3, 8, 8, CV_32F);
    net.setInput(input);

    // CUDA supported layer (Conv)
    cv::dnn::LayerParams conv;
    conv.type = "Convolution";
    conv.set("kernel_size", 1);
    conv.set("num_output", 3);
    conv.blobs.push_back(cv::Mat::ones(3, 3, CV_32F));
    conv.blobs.push_back(cv::Mat::zeros(3, 1, CV_32F));

    int conv1 = net.addLayer("conv1", "Convolution", conv);

    // CUDA unsupported layer (Shape)
    cv::dnn::LayerParams shape;
    int shapeId = net.addLayer("shape", "Shape", shape);
    net.connect(conv1, 0, shapeId, 0);

    // CUDA supported layer again
    int conv2 = net.addLayer("conv2", "Convolution", conv);
    net.connect(shapeId, 0, conv2, 0);

    // Enable hybrid
    net.enableHybridBackend(true);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Forward should succeed
    EXPECT_NO_THROW(net.forward());
}

} // namespace opencv_test

TEST(DNN_Hybrid, ExcessiveTransitionsDisableHybrid)
{
    cv::dnn::Net net;

    cv::Mat input = cv::Mat::ones(1, 3, 8, 8, CV_32F);
    net.setInput(input);

    cv::dnn::LayerParams conv;
    conv.type = "Convolution";
    conv.set("kernel_size", 1);
    conv.set("num_output", 3);
    conv.blobs.push_back(cv::Mat::ones(3, 3, CV_32F));
    conv.blobs.push_back(cv::Mat::zeros(3, 1, CV_32F));

    int prev = net.addLayer("conv0", "Convolution", conv);

    // Alternate CPU-only layers to force transitions
    for (int i = 0; i < 4; i++)
    {
        cv::dnn::LayerParams shape;
        int sid = net.addLayer("shape_" + std::to_string(i), "Shape", shape);
        net.connect(prev, 0, sid, 0);

        int cid = net.addLayer("conv_" + std::to_string(i), "Convolution", conv);
        net.connect(sid, 0, cid, 0);

        prev = cid;
    }

    net.enableHybridBackend(true);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setHybridTransitionLimit(1); // very strict

    // Should fallback safely
    EXPECT_NO_THROW(net.forward());
}
