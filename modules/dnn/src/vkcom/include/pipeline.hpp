// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_PIPELINE_VULKAN_HPP
#define OPENCV_PIPELINE_VULKAN_HPP

#include "../../precomp.hpp"
#include "tensor.hpp"
#include <map>
#include <queue>

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif // HAVE_VULKAN

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

class Pipeline;
class Descriptor
{
public:
    static Ptr<Descriptor> create(const VkDescriptorPool& pool, const VkDescriptorSet& set,
                                  Pipeline* _pipeline);
    ~Descriptor();

    void writeTensor(Tensor tensor, int bindIndex);
    // the buffer is bond to the device VkMemory.
    void writeBuffer(VkBuffer buffer, int bindIndex, size_t size, VkDeviceSize offset = 0);

    VkDescriptorSet get() const
    {
        return desSet;
    }

private:
    friend class Pipeline;
    Descriptor(const VkDescriptorPool& pool, const VkDescriptorSet& set, Pipeline* _pipeline);

    VkDescriptorPool desPool;
    VkDescriptorSet desSet;
    Pipeline* pipeline;
    // If is true, the deconstruct will release the instance, otherwise, re-use it.
    bool needRelease = true;
};

class Pipeline
{
public:
    static Ptr<Pipeline> create(const uint32_t* spv, size_t length, const std::vector<VkDescriptorType>& bufferTypes,
                                VkPipelineCache& cache, const std::vector<uint32_t>& localSize = std::vector<uint32_t>());
    ~Pipeline();

    VkPipeline get() const
    {
        return pipelineVK;
    }

    Ptr<Descriptor> createSet();

    void bind(VkCommandBuffer buffer, VkDescriptorSet descriptorSet) const;

    inline VkDescriptorType argType(int index) const
    {
        return bufferTypes[index];
    }

    // To save the descriptor that can be reused.
    std::queue<std::pair<VkDescriptorPool, VkDescriptorSet> > descriptorPairQueue;
private:
    Pipeline(const uint32_t* spv, size_t length, const std::vector<VkDescriptorType>& bufferTypes,
             VkPipelineCache& cache, const std::vector<uint32_t>& localSize = std::vector<uint32_t>());

    VkPipeline pipelineVK;
    VkPipelineLayout pipelineLayout;
    std::vector<VkDescriptorPoolSize> desPoolSize;
    VkDescriptorSetLayout setLayout;
    std::vector<VkDescriptorType> bufferTypes;
};

class PipelineFactory
{
public:
    static Ptr<PipelineFactory> create();

    // Try to retrieve the Pipeline from pipelineCreated, create a new pipeline instance if not found.
    Ptr<Pipeline> getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                const std::vector<uint32_t>& localSize = std::vector<uint32_t>());
    ~PipelineFactory();
    void reset();

    void operator=(const PipelineFactory &) = delete;
    PipelineFactory(PipelineFactory &other) = delete;
private:
    PipelineFactory();
    mutable std::map<std::string, Ptr<Pipeline> > pipelineCreated;
    VkPipelineCache pipelineCache;
};

#endif // HAVE_VULKAN
}}} // namespace cv::dnn::vkcom
#endif //OPENCV_PIPELINE_VULKAN_HPP
