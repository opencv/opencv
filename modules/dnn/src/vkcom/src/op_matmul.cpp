// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/op_matmul.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define KSTRIP_LEN 32
#define BLOCK_SIZE 64

OpMatMul::OpMatMul(std::vector<Mat>& matBlobs, const int _M, const int _K, const int _N) : M(_M), K(_K), N(_N)
{
    // Convert Weight to GPU Tensor.
    type_ = kOpTypeMatMul;
    CV_Assert(matBlobs.empty() || matBlobs.size() == 1);

    if (matBlobs.size() == 1)
    {
        Tensor weightTensor;
        CV_Assert(matBlobs[0].isContinuous() && matBlobs[0].type() == CV_32F);
        std::vector<int> matShape = shape(matBlobs[0]);
        weightTensor.reshape((const char*)matBlobs[0].data, matShape); // This code will copy the src data from Mat to VkBuffer.

        weightTensorPtr = makePtr<Tensor>(weightTensor);
    }
}

void OpMatMul::firstForward()
{
    if (!firstForwardFinsh)
    {
        config.local_size_x = BLOCK_SIZE;
        config.local_size_y = BLOCK_SIZE;
        config.local_size_z = 1;

        computeGroupCount();
        firstForwardFinsh = true;
    }
    else
        return;
}

bool OpMatMul::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    CV_Assert((ins.size() == 1 || ins.size() == 2) && outs.size() == 1);
    Shape inputShape = ins[0].getShape();
    Shape outputShape = outs[0].getShape();
    CV_Assert(inputShape.size() == outputShape.size());

    CV_Assert(inputShape.size() == 2 || inputShape.size() == 4);

    if (inputShape.size() == 2)
    {
        batch = 0;
        Hi = inputShape[0];
        Wi = inputShape[1];

        H0 = outputShape[0];
        W0 = outputShape[1];
    }
    else if (inputShape.size() == 4)
    {
        batch = inputShape[kShapeIdxBatch];
        Hi = inputShape[kShapeIdxHeight];
        Wi = inputShape[kShapeIdxWidth];

        H0 = outputShape[kShapeIdxHeight];
        W0 = outputShape[kShapeIdxWidth];
    }

    firstForward();

    int KStrip = K/KSTRIP_LEN;
    int KStripRemain = K - KStrip * KSTRIP_LEN;
    std::vector<int> param = {M, K, N, KStrip, KStripRemain};

    std::vector<int> shape = {(int)param.size()};
    Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), shape, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    std::string key = "gemm_spv";
    destTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // weight
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // out
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
    };

    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(key, destTypes);
    Ptr<Descriptor> desSet = pipeline->createSet();
    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();

    VkCommandBuffer cmdBufferReal = cmdBuffer->get();
    desSet->writeTensor(ins[0], 0);

    if (weightTensorPtr)
        desSet->writeTensor(*weightTensorPtr, 1);
    else
    {
        CV_Assert(ins.size() == 2);
        desSet->writeTensor(ins[1], 1);
    }

    desSet->writeTensor(outs[0], 2);
    desSet->writeTensor(paramTensor, 3); // TODO change the parameter from pushconstance to buffer.

    cmdBuffer->beginRecord();
    pipeline->bind(cmdBufferReal, desSet->get());
    vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
    cmdBuffer->endRecord();

    cmdPoolPtr->submitAndWait(cmdBufferReal);

    return true;
}

bool OpMatMul::computeGroupCount()
{
    group_x_ = alignSize(M, BLOCK_SIZE) / BLOCK_SIZE;
    group_y_ = alignSize(N, BLOCK_SIZE) / BLOCK_SIZE;
    group_z_ = 1;

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
