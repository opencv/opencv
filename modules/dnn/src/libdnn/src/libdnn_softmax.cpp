#include "../../precomp.hpp"
#include <vector>
#include "common.hpp"
#include "libdnn.hpp"
#include "opencl_kernels_dnn.hpp"

namespace greentea
{

#ifdef HAVE_OPENCL
template<typename Dtype>
LibDNNSoftmax<Dtype>::LibDNNSoftmax(LibDNNSoftmaxConfig config)
{
    softmax_axis_ = config.axis;
    channels_ = config.channels;

    inner_num_ = 1;
    outer_num_ = 1;
    count_ = 1;
    int_tp scale_sz = 1;
    for (int_tp i = softmax_axis_ + 1; i < config.in_shape.size(); i++)
        inner_num_ *= config.in_shape[i];
    use_slm_ = (config.in_shape[softmax_axis_] * inner_num_ + inner_num_ * 17) <= 8192;
    for (int_tp i = 0; i < softmax_axis_; i++)
        outer_num_ *= config.in_shape[i];
    count_ = inner_num_ + outer_num_;

    std::vector<int_tp> scale_dims = config.in_shape;
    scale_dims[softmax_axis_] = use_slm_ ? 1 : 17;
    for (int_tp i = 0; i < scale_dims.size(); i++)
        scale_sz *= scale_dims[i];

    ocl::Context ctx = ocl::Context::getDefault();
    cl_int err;
    scale_data_ = reinterpret_cast<Dtype*>(clCreateBuffer((cl_context)ctx.ptr(),
                                                          CL_MEM_READ_WRITE,
                                                          sizeof(Dtype) * scale_sz,
                                                          NULL, &err));
    CHECK_EQ(err, CL_SUCCESS) << "Failed to create scale buffer.";
}

template<typename Dtype>
LibDNNSoftmax<Dtype>::~LibDNNSoftmax()
{
    if (scale_data_)
    {
        clReleaseMemObject((cl_mem)scale_data_);
    }
}

template<typename Dtype>
bool LibDNNSoftmax<Dtype>::Forward(const Dtype* bottom_data, Dtype* top_data)
{
    bool ret = true;
    ocl::Queue queue = ocl::Queue::getDefault();
    if (ocl::Device::getDefault().intelSubgroupsSupport() && inner_num_ < 128)
    {
        String opts = " -cl-no-subgroup-ifp ";
        ocl::Kernel oclk_softmax_forward_kernel;
        if (use_slm_)
            oclk_softmax_forward_kernel.create(CL_KERNEL_SELECT("softmax_forward_slm"),
                                               cv::ocl::dnn::softmax_loss_oclsrc, opts);
        else
            oclk_softmax_forward_kernel.create(CL_KERNEL_SELECT("softmax_forward"),
                                               cv::ocl::dnn::softmax_loss_oclsrc, opts);

        size_t global_size[] = { 256, outer_num_, 1 };
        size_t local_size[] = { 256, 1, 1 };
        cl_uint argIdx = 0;

        if (use_slm_)
        {
            oclk_softmax_forward_kernel.set(argIdx++, outer_num_);
            oclk_softmax_forward_kernel.set(argIdx++, channels_);
            oclk_softmax_forward_kernel.set(argIdx++, inner_num_);
            oclk_softmax_forward_kernel.set(argIdx++, (cl_mem) scale_data_);
            oclk_softmax_forward_kernel.set(argIdx++, (cl_mem) bottom_data);
            oclk_softmax_forward_kernel.set(argIdx++, (cl_mem) top_data);
            clSetKernelArg((cl_kernel)oclk_softmax_forward_kernel.ptr(), argIdx++, channels_ * inner_num_* sizeof(Dtype), NULL);
            clSetKernelArg((cl_kernel)oclk_softmax_forward_kernel.ptr(), argIdx++, inner_num_* sizeof(Dtype), NULL);
            clSetKernelArg((cl_kernel)oclk_softmax_forward_kernel.ptr(), argIdx++, 16 * inner_num_* sizeof(Dtype), NULL);

            OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                             (cl_kernel)oclk_softmax_forward_kernel.ptr(), 3,
                                             NULL, global_size, local_size, 0, NULL, NULL));
        }
        else
        {
            oclk_softmax_forward_kernel.set(argIdx++, outer_num_);
            oclk_softmax_forward_kernel.set(argIdx++, channels_);
            oclk_softmax_forward_kernel.set(argIdx++, inner_num_);
            oclk_softmax_forward_kernel.set(argIdx++, (cl_mem) scale_data_);
            oclk_softmax_forward_kernel.set(argIdx++, (cl_mem) bottom_data);
            oclk_softmax_forward_kernel.set(argIdx++, (cl_mem) top_data);
            OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                             (cl_kernel)oclk_softmax_forward_kernel.ptr(), 3,
                                             NULL, global_size, local_size, 0, NULL, NULL));
        }
    }
    else
    {
        ret = false;
    }
    return ret;
}

template class LibDNNSoftmax<float>;
template class LibDNNSoftmax<double>;
#endif // HAVE_OPENCL

}  // namespace greentea
