// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/op_conv.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN
#define BLOCK_SIZE 64

#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535

OpConv::OpConv(const Mat& weightBlob, const std::vector<float>& biasvec, int _activType, const int _ngroups, const int _K,
               const int _C, const int _Hk, const int _Wk, const int _stride_h, const int _stride_w,
               const int _dilation_h, const int _dilation_w, const int _pad_left, const int _pad_top):
               activ((FusedActivationType)_activType), ngroups(_ngroups), K(_K), C(_C), Hk(_Hk), Wk(_Wk), stride_h(_stride_h), stride_w(_stride_w),
               dilation_h(_dilation_h), dilation_w(_dilation_w), pad_left(_pad_left), pad_top(_pad_top)
{
    type_ = kOpTypeConv;
    CV_Assert(!weightBlob.empty());

    Kg = K/ngroups, Cg = max(C/ngroups, 1);
    ksize = Hk * Wk;
    CgHkWk = Cg * ksize;
    fast_1x1 = ksize == 1 && stride_w == 1 && stride_h == 1 && pad_top == 0 && pad_left == 0;

    if (ngroups > 1 && ngroups == K && ngroups == C)
    {
        if (Hk == 3 && Wk == 3)
            shader_name = "conv_depthwise_3x3_spv";
        else
            shader_name = "conv_depthwise_spv";

        shaderType = kConvShaderTypeDepthWise;
        STRIP_LEN = 16;
    }
    else if (fast_1x1) // 1x1
    {
        shader_name = "conv_1x1_fast_spv";
        shaderType = kConvShaderTypeGeneric;
        STRIP_LEN = 32;
    }
    else
    {
        shader_name = "conv_implicit_gemm_spv";
        shaderType = kConvShaderTypeGeneric;
        STRIP_LEN = 32;
    }

    CgHkWk_aligned = alignSize(CgHkWk, STRIP_LEN);
    // repack the weight. The shape is from [K, C, Hk, Wk] to [ngroups, Ceil(K/group), Align(Cg*Hk*Wk, STRIP_LEN)]
    if (shaderType == kConvShaderTypeGeneric)
    {
        std::vector<int> repackWeightShape = {ngroups, Kg, CgHkWk_aligned};

        Mat repackWeight = Mat(repackWeightShape, CV_32FC1, Scalar_<float>(0.0f));
        float* weightsBufPtr = repackWeight.ptr<float>();
        const float* srcWeight = weightBlob.ptr<float>();
        const size_t wstep = weightBlob.step1(); // Hk*Wk*Cg

        // Pack the weight.
        parallel_for_(Range(0, ngroups * Kg), [&](const Range& r0){
        for (int gki = r0.start; gki < r0.end; gki++)
        {
            const float* wptr = srcWeight + gki * wstep;
            float* packed_wptr = weightsBufPtr + gki * CgHkWk_aligned;

            memcpy(packed_wptr, wptr, sizeof(wptr[0]) * CgHkWk);
        }});

        // Create weightTensor
        Tensor weightTensor;
        CV_Assert(repackWeight.isContinuous() && weightBlob.type() == CV_32F);
        weightTensor.reshape((const char*)repackWeight.data, repackWeightShape);
        weightTensorPtr = makePtr<Tensor>(weightTensor);
    }
    else
    {
        // Create weightTensor
        Tensor weightTensor;
        CV_Assert(weightBlob.isContinuous() && weightBlob.type() == CV_32F);
        std::vector<int> matShape = shape(weightBlob);
        weightTensor.reshape((const char*)weightBlob.data, matShape); // This code will copy the src data from Mat to VkBuffer.

        weightTensorPtr = makePtr<Tensor>(weightTensor);
    }

    if (!biasvec.empty())
    {
        int biasAlignedSize = alignSize(biasvec.size(), BLOCK_SIZE);
        std::vector<int> biasShape = {biasAlignedSize};

        biasCopy.resize(biasAlignedSize, 0.f);

        for (int i = 0; i < biasvec.size(); i++)
        {
            biasCopy[i] = biasvec[i];
        }

        Tensor biasTensor;
        biasTensor.reshape((const char*)biasCopy.data(), biasShape); // This code will copy the src data from Mat to VkBuffer.

        biasTensorPtr = makePtr<Tensor>(biasTensor);
    }
    else
    {
        std::vector<int> shape = {K};
        Tensor bias(0, shape);
        biasTensorPtr = makePtr<Tensor>(bias);
    }
}

void OpConv::firstForward()
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

OpConv::~OpConv()
{
}

bool OpConv::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    CV_Assert(ins.size() == 1 && outs.size() == 1);
    Shape inputShape = ins[0].getShape();
    Shape outputShape = outs[0].getShape();
    CV_Assert(inputShape.size() == outputShape.size());

    batch = inputShape[kShapeIdxBatch];
    Hi = inputShape[kShapeIdxHeight];
    Wi = inputShape[kShapeIdxWidth];

    H0 = outputShape[kShapeIdxHeight];
    W0 = outputShape[kShapeIdxWidth];

    firstForward();

    std::vector<int> param = {Hi, Wi,
                              H0, W0,
                              stride_h, stride_w,
                              pad_top, pad_left,
                              Hk, Wk,
                              dilation_h, dilation_w,
                              Kg, Cg,
                              ngroups,
                              CgHkWk_aligned,
                              (int)activ,
                              0, 0};

    std::vector<int> shape = {(int)param.size()};
    destTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // bias
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // weight
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
    };

    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(shader_name, destTypes);
    Ptr<Descriptor> desSet = pipeline->createSet();
    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();

    VkCommandBuffer cmdBufferReal = cmdBuffer->get();
    desSet->writeTensor(ins[0], 0);
    desSet->writeTensor(*biasTensorPtr, 1);
    desSet->writeTensor(*weightTensorPtr, 2);
    desSet->writeTensor(outs[0], 3);

    if (shaderType == kConvShaderTypeGeneric)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int g = 0; g < ngroups; g++)
            {
                param[17] = b;
                param[18] = g;
                Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), shape, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
                desSet->writeTensor(paramTensor, 4);

                cmdBuffer->beginRecord();
                pipeline->bind(cmdBufferReal, desSet->get());
                vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
                cmdBuffer->endRecord();

                cmdPoolPtr->submitAndWait(cmdBufferReal);
            }
        }
    }
    else if (shaderType == kConvShaderTypeDepthWise)
    {
        for (int b = 0; b < batch; b++)
        {
            param[17] = b;
            Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), shape, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            desSet->writeTensor(paramTensor, 4);

            cmdBuffer->beginRecord();
            pipeline->bind(cmdBufferReal, desSet->get());
            vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
            cmdBuffer->endRecord();

            cmdPoolPtr->submitAndWait(cmdBufferReal);
        }
    }

    return true;
}

bool OpConv::computeGroupCount()
{
    int outplan = H0 * W0;
    if (shaderType == kConvShaderTypeDepthWise)
    {
        group_x_ = alignSize(outplan, config.local_size_x) / config.local_size_x;
        group_y_ = K;
        group_z_ = 1;
        return true;
    }
    else if (shaderType == kConvShaderTypeGeneric)
    {
        group_x_ = alignSize(Kg, config.local_size_x) / config.local_size_x;
        group_y_ = alignSize(outplan, config.local_size_y) / config.local_size_y;
        group_z_ = 1;
    }
    else if (shaderType == kConvShaderTest)
    {
        group_x_ = 1;
        group_y_ = 1;
        group_z_ = 1;
    }
    else
        CV_Error(cv::Error::StsNotImplemented, "shader type is not supported at compute GroupCount.");

    CV_Assert(group_x_ <= MAX_GROUP_COUNT_X);
    CV_Assert(group_y_ <= MAX_GROUP_COUNT_Y);
    CV_Assert(group_z_ <= MAX_GROUP_COUNT_Z);

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
