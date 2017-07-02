#include "../../precomp.hpp"
#include <string>
#include <vector>
#include "common.hpp"
#include "libdnn.hpp"
#include "opencl_kernels_dnn.hpp"

namespace greentea
{

#ifdef HAVE_OPENCL
template<typename Dtype>
LibDNNPool<Dtype>::LibDNNPool(LibDNNPoolConfig config)
{
    int_tp dims = config.in_shape.size();
    int_tp spatial_dims = config.kernel.size();

    batch_size_ = config.in_shape[0];
    channels_ = config.channels;
    pool_method_ = config.pool_method;

    for (int_tp i = 0; i < spatial_dims; ++i)
    {
        kernel_shape_.push_back(config.kernel[i]);
        pad_.push_back(config.pad[i]);
        stride_.push_back(config.stride[i]);
        im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
        im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
    }

    kernel_h_ = kernel_shape_[0];
    kernel_w_ = kernel_shape_[1];
    stride_h_ = stride_[0];
    stride_w_ = stride_[1];
    pad_h_ = pad_[0];
    pad_w_ = pad_[1];
    height_ = im_in_shape_[0];
    width_ = im_in_shape_[1];
    pooled_height_ = im_out_shape_[0];
    pooled_width_ = im_out_shape_[1];

    count_ = 1;
    for (int_tp i = 0; i < config.out_shape.size(); ++i)
    {
        count_ *= config.out_shape[i];
    }

    mask_idx_ = NULL;
}

template<typename Dtype>
LibDNNPool<Dtype>::~LibDNNPool()
{
    if (mask_idx_)
    {
        clReleaseMemObject((cl_mem)mask_idx_);
    }
}

template<typename Dtype>
bool LibDNNPool<Dtype>::Forward(const Dtype *bottom_data,
                                Dtype *top_data,
                                Dtype *top_mask)
{
    bool ret = true;
    ocl::Queue queue = ocl::Queue::getDefault();
    size_t global[] = { 128 * 128 };
    size_t local[] = { 128 };
    cl_uint argIdx = 0;

    // support 2D case
    switch (pool_method_)
    {
    case LIBDNN_POOLING_METHOD_MAX:
        {
            if (top_mask == NULL && mask_idx_ == NULL)
            {
                AllocateMemory((void **)&mask_idx_,  sizeof(int_tp) * count_, CL_MEM_READ_WRITE);
            }
            ocl::Kernel oclk_max_pool_forward(CL_KERNEL_SELECT("max_pool_forward"), cv::ocl::dnn::dnn_pooling_oclsrc);

            argIdx = 0;
            oclk_max_pool_forward.set(argIdx++, count_);
            oclk_max_pool_forward.set(argIdx++, (cl_mem) bottom_data);
            oclk_max_pool_forward.set(argIdx++, batch_size_);
            oclk_max_pool_forward.set(argIdx++, channels_);
            oclk_max_pool_forward.set(argIdx++, height_);
            oclk_max_pool_forward.set(argIdx++, width_);
            oclk_max_pool_forward.set(argIdx++, pooled_height_);
            oclk_max_pool_forward.set(argIdx++, pooled_width_);
            oclk_max_pool_forward.set(argIdx++, kernel_h_);
            oclk_max_pool_forward.set(argIdx++, kernel_w_);
            oclk_max_pool_forward.set(argIdx++, stride_h_);
            oclk_max_pool_forward.set(argIdx++, stride_w_);
            oclk_max_pool_forward.set(argIdx++, pad_h_);
            oclk_max_pool_forward.set(argIdx++, pad_w_);
            oclk_max_pool_forward.set(argIdx++, (cl_mem) top_data);
            oclk_max_pool_forward.set(argIdx++, mask_idx_ == NULL ? 0 : 1);
            oclk_max_pool_forward.set(argIdx++, (cl_mem) mask_idx_);
            oclk_max_pool_forward.set(argIdx++, (cl_mem) top_mask);

            oclk_max_pool_forward.run(1, global, local, false);
        }
        break;
    case LIBDNN_POOLING_METHOD_AVE:
        {
            ocl::Kernel oclk_ave_pool_forward(CL_KERNEL_SELECT("ave_pool_forward"), cv::ocl::dnn::dnn_pooling_oclsrc);

            argIdx = 0;
            oclk_ave_pool_forward.set(argIdx++, count_);
            oclk_ave_pool_forward.set(argIdx++, (cl_mem) bottom_data);
            oclk_ave_pool_forward.set(argIdx++, batch_size_);
            oclk_ave_pool_forward.set(argIdx++, channels_);
            oclk_ave_pool_forward.set(argIdx++, height_);
            oclk_ave_pool_forward.set(argIdx++, width_);
            oclk_ave_pool_forward.set(argIdx++, pooled_height_);
            oclk_ave_pool_forward.set(argIdx++, pooled_width_);
            oclk_ave_pool_forward.set(argIdx++, kernel_h_);
            oclk_ave_pool_forward.set(argIdx++, kernel_w_);
            oclk_ave_pool_forward.set(argIdx++, stride_h_);
            oclk_ave_pool_forward.set(argIdx++, stride_w_);
            oclk_ave_pool_forward.set(argIdx++, pad_h_);
            oclk_ave_pool_forward.set(argIdx++, pad_w_);
            oclk_ave_pool_forward.set(argIdx++, (cl_mem) top_data);

            oclk_ave_pool_forward.run(1, global, local, false);
        }
        break;
    case LIBDNN_POOLING_METHOD_STO:
        {
            ocl::Kernel oclk_sto_pool_forward(CL_KERNEL_SELECT("sto_pool_forward_test"), cv::ocl::dnn::dnn_pooling_oclsrc);

            argIdx = 0;
            oclk_sto_pool_forward.set(argIdx++, count_);
            oclk_sto_pool_forward.set(argIdx++, (cl_mem) bottom_data);
            oclk_sto_pool_forward.set(argIdx++, batch_size_);
            oclk_sto_pool_forward.set(argIdx++, channels_);
            oclk_sto_pool_forward.set(argIdx++, height_);
            oclk_sto_pool_forward.set(argIdx++, width_);
            oclk_sto_pool_forward.set(argIdx++, pooled_height_);
            oclk_sto_pool_forward.set(argIdx++, pooled_width_);
            oclk_sto_pool_forward.set(argIdx++, kernel_h_);
            oclk_sto_pool_forward.set(argIdx++, kernel_w_);
            oclk_sto_pool_forward.set(argIdx++, stride_h_);
            oclk_sto_pool_forward.set(argIdx++, stride_w_);
            oclk_sto_pool_forward.set(argIdx++, (cl_mem)top_data);

            oclk_sto_pool_forward.run(1, global, local, false);
        }
        break;
    default:
        {
            ret = false;
            LOG(FATAL)<< "Unknown pooling method.";
        }
    }
    return ret;
}

template class LibDNNPool<float>;
template class LibDNNPool<double>;
#endif // HAVE_OPENCL

}  // namespace greentea
