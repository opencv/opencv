#include "../../precomp.hpp"
#include "common.hpp"
#include "libdnn.hpp"
#include "greentea_math_functions.hpp"

namespace greentea
{

#ifdef HAVE_OPENCL
template<typename Dtype>
LibDNNInnerProduct<Dtype>::LibDNNInnerProduct(LibDNNInnerProductConfig config)
{
    bias_term_  = config.bias_term;
    transpose_  = config.transpose;
    N_ = num_output_ = config.num_output;
    M_ = config.M;
    K_ = config.K;
    phase_test_ = config.phase_test;
    image_copied_ = false;
    weight_image_ = NULL;

    // Set up the bias multiplier
    if (bias_term_)
    {
        int_tp flags = 0;
        flags = CL_MEM_READ_WRITE;
        AllocateMemory((void **)&bias_multiplier_,  sizeof(Dtype) * M_, flags);
        greentea_gpu_set(0, M_, Dtype(1), (cl_mem) bias_multiplier_, 0);
    }
}

template<typename Dtype>
bool LibDNNInnerProduct<Dtype>::Forward(const Dtype* bottom_data,
                                        const Dtype* weight,
                                        const Dtype* bias,
                                        Dtype* top_data)
{
    if (M_ == 1)
    {
        greentea_gpu_gemv<Dtype>(0, CblasNoTrans, N_,
                                 K_, (Dtype) 1., (cl_mem) weight, 0,
                                 (cl_mem) bottom_data, 0, (Dtype) 0.,
                                 (cl_mem) top_data, 0);
        if (bias_term_)
            greentea_gpu_axpy<Dtype>(0, N_,
                                     1,
                                     (cl_mem) bias, 0,
                                     (cl_mem) top_data, 0);
    }
    else
    {
        size_t max_image_size = std::min(ocl::Device::getDefault().image2DMaxWidth(),
                                         ocl::Device::getDefault().image2DMaxHeight());
        if (M_ <= max_image_size &&
            N_ <= max_image_size &&
            K_ <= max_image_size &&
            std::is_same<Dtype, float>::value &&
            ocl::Device::getDefault().intelSubgroupsSupport())
        {

            if (phase_test_ == false || image_copied_ == false)
            {
                int height = !transpose_ ? N_ : K_;
                int width = !transpose_ ? K_ : N_;
                int padded_height = !transpose_ ? height : (height + ((height & 7) ? 1 : 0));
                int padded_width = !transpose_ ? width : (width + ((width & 7) ? 1 : 0));
                greentea_gpu_gemm_copy_buffer_to_image(0,
                                                       (cl_mem *)&weight_image_, (cl_mem) weight, 0,
                                                       false, !transpose_,
                                                       true, padded_height, padded_width,
                                                       height, width, (int)0, NULL, NULL);
                image_copied_ = true;
            }
            greentea_gpu_gemm<Dtype>(0, CblasNoTrans,
                                     transpose_ ? CblasNoTrans : CblasTrans,
                                     M_, N_, K_, (Dtype) 1.,
                                     (cl_mem) bottom_data, 0, (cl_mem) weight_image_, 0,
                                     (Dtype) 0., (cl_mem) top_data, 0, false, true);
        } else
            greentea_gpu_gemm<Dtype>(0, CblasNoTrans,
                                     transpose_ ? CblasNoTrans : CblasTrans,
                                     M_, N_, K_, (Dtype) 1.,
                                     (cl_mem) bottom_data, 0, (cl_mem) weight, 0,
                                     (Dtype) 0., (cl_mem) top_data, 0, false, false);

        if (bias_term_)
            greentea_gpu_gemm<Dtype>(0, CblasNoTrans,
                                     CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                     (cl_mem) bias_multiplier_, 0,
                                     (cl_mem) bias, 0,
                                     (Dtype) 1., (cl_mem) top_data, 0, false, false);
    }
    return true;
}

template class LibDNNInnerProduct<float>;
template class LibDNNInnerProduct<double>;
#endif // HAVE_OPENCL

}  // namespace greentea
