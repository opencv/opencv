#include "../../precomp.hpp"
#include "common.hpp"
#include "libdnn.hpp"
#include "opencl_kernels_dnn.hpp"

namespace greentea
{

#ifdef HAVE_OPENCL
template<typename Dtype>
LibDNNLRN<Dtype>::LibDNNLRN(LibDNNLRNConfig config)
{
    lrn_type_ = config.lrn_type;
    phase_test_ = config.phase_test;
    size_ = config.local_size;
    CHECK_EQ(size_ % 2, 1)<< "LRN only supports odd values for local_size";
    alpha_ = config.alpha;
    beta_ = config.beta;
    k_ = config.k;
    norm_by_size_ = config.norm_by_size;
    num_ = config.batch_size;
    channels_ = config.channels;
    height_ = config.height;
    width_ = config.width;
}

template<typename Dtype>
bool LibDNNLRN<Dtype>::Forward(const Dtype* bottom_data, Dtype* top_data)
{
    bool ret = true;
    switch (lrn_type_)
    {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
        CrossChannelForward_gpu(bottom_data, top_data);
        break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
        //TODO
        //WithinChannelForward(bottom_data, top_data);
        ret = false;
        break;
    default:
        ret = false;
        LOG(FATAL)<< "Unknown normalization region.";
    }
    return ret;
}

template<typename Dtype>
void LibDNNLRN<Dtype>::CrossChannelForward_gpu(const Dtype* bottom_data,
                                               Dtype* top_data)
{
    ocl::Queue queue = ocl::Queue::getDefault();

    CHECK_EQ(phase_test_, true) << "Only support forward inference.";

    cl_uint argIdx = 0;
    int_tp n_threads = num_ * height_ * width_;
    size_t global_work_size_[1] = {(size_t)n_threads};
    String opts = " -cl-no-subgroup-ifp ";
    ocl::Kernel oclk_lrn_fill;
    oclk_lrn_fill.create(CL_KERNEL_SELECT("lrn_full_no_scale"), cv::ocl::dnn::dnn_lrn_oclsrc, opts);

    oclk_lrn_fill.set(argIdx++, n_threads);
    oclk_lrn_fill.set(argIdx++, (cl_mem) bottom_data);
    oclk_lrn_fill.set(argIdx++, num_);
    oclk_lrn_fill.set(argIdx++, channels_);
    oclk_lrn_fill.set(argIdx++, height_);
    oclk_lrn_fill.set(argIdx++, width_);
    oclk_lrn_fill.set(argIdx++, size_);
    int size_norm_factor = norm_by_size_ ? size_ : 1;
    oclk_lrn_fill.set(argIdx++, alpha_ / size_norm_factor);
    oclk_lrn_fill.set(argIdx++, k_);
    oclk_lrn_fill.set(argIdx++, (cl_mem) top_data);
    oclk_lrn_fill.set(argIdx++, -beta_);
    oclk_lrn_fill.run(1, global_work_size_, NULL, false);
}

template class LibDNNLRN<float>;
template class LibDNNLRN<double>;
#endif // HAVE_OPENCL

}  // namespace greentea
