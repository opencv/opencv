/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "../../precomp.hpp"
#include "common.hpp"

#ifdef HAVE_OPENCL
#include "greentea_math_functions.hpp"

#include <mutex>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark.hpp"
#include "opencl_kernels_dnn.hpp"

using namespace cv;
namespace greentea {

struct gemm_callback_arg {
    std::vector<cl_event> evs;
    std::vector<cl_mem> imgs;
};

static void CL_CALLBACK gemm_callback (cl_event event,
                                       cl_int event_command_exec_status,
                                       void *user_data)
{
    struct gemm_callback_arg *arg = (struct gemm_callback_arg *) user_data;
    for(int i = 0; i < arg->evs.size(); i++)
    {
        clReleaseEvent(arg->evs[i]);
    }

    for(int i = 0; i < arg->imgs.size(); i++)
    {
        clReleaseMemObject(arg->imgs[i]);
    }
    delete arg;
}

// Create and copy buffer to image for GEMM's matrix A and B.
// Will return image to caller if the input image is NULL. Otherwise,
// will use the image directly. It's caller's responsibility to
// release the created image.
void greentea_gpu_gemm_copy_buffer_to_image(int_tp ctx_id,
                                            cl_mem *image, cl_mem buffer, int offset,
                                            bool is_matrix_a, bool transpose,
                                            bool padding, int padded_height,
                                            int padded_width, int height,
                                            int width, int wait_list_size,
                                            cl_event *wait_list,
                                            cl_event *event)
{
    ocl::Context ctx = ocl::Context::getDefault();
    ocl::Queue queue = ocl::Queue::getDefault();
    cl_image_desc desc;
    cl_image_format format;

    memset(&desc, 0, sizeof(desc));
    if (!is_matrix_a && transpose)
    {
        // For matrix B with transpose, we need to handle them differently.
        // As we can't use the sub group block read to get a row easily,
        // we have to use CL_FLOAT type with read_imagef to get the row.
        cl_int err;
        format.image_channel_data_type = CL_FLOAT;
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        if ( width % 4 == 0 )
        {
            desc.image_width = width / 4;
            format.image_channel_order = CL_RGBA;
        }
        else
        {
            desc.image_width = width;
            format.image_channel_order = CL_R;
        }
        desc.image_height = height;
        // if (offB == 0 && (desc.image_width % 4) == 0 && N > 8 && K > 8)
        //  desc.mem_object = buffer;
        if (*image == NULL)
        {
            *image = clCreateImage((cl_context)ctx.ptr(),
                                   CL_MEM_READ_WRITE,
                                   &format,
                                   &desc,
                                   NULL,
                                   &err);
            OCL_CHECK(err);
        }
        // if (!desc.mem_object) {
        size_t origin[] = {0, 0, 0};
        size_t region[] = {(size_t)desc.image_width,
                           (size_t)desc.image_height, 1};
        OCL_CHECK(clEnqueueCopyBufferToImage((cl_command_queue)queue.ptr(),
                                             buffer, *image, sizeof(float) * offset,
                                             origin, region, wait_list_size,
                                             wait_list, event));
        // }
        return;
    }

    if (*image == NULL)
    {
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        format.image_channel_data_type = CL_UNSIGNED_INT8;
        format.image_channel_order = CL_RGBA;
        if (!padding)
        {
            //if (width % 4 == 0 && offset == 0 && height > 8 && width > 8)
            //  desc.buffer = buffer;
            desc.image_width = width;
            desc.image_height = height;
        }
        else
        {
            desc.image_width = padded_width;
            desc.image_height = padded_height;
        }
        cl_int err;
        *image = clCreateImage((cl_context)ctx.ptr(),
                               desc.buffer ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE,
                               &format,
                               &desc,
                               NULL,
                               &err);
        OCL_CHECK(err);
    }
    if (!padding && desc.buffer != NULL)
        return;
    if (!padding && desc.buffer == NULL)
    {
        // copy without padding.
        size_t origin[] = {0, 0, 0};
        size_t region[] = {(size_t)width, (size_t)height, 1};
        OCL_CHECK(clEnqueueCopyBufferToImage((cl_command_queue)queue.ptr(),
                                              buffer, *image, sizeof(float) * offset,
                                              origin, region, wait_list_size, wait_list, event));
        return;
    }
    ocl::Kernel oclk_gemm_copy("gemm_buffer_copy_image_float", cv::ocl::dnn::gemm_image_oclsrc);

    size_t global_copy[2];
    global_copy[0] = padding ? padded_width : width;
    global_copy[1] = padding ? padded_height : height;
    oclk_gemm_copy.set(0, (cl_mem) buffer);
    oclk_gemm_copy.set(1, (cl_mem) *image);
    oclk_gemm_copy.set(2, offset);
    oclk_gemm_copy.set(3, width);
    oclk_gemm_copy.set(4, height);
    OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                     (cl_kernel)oclk_gemm_copy.ptr(),
                                     2, NULL, global_copy, NULL,
                                     wait_list_size, wait_list,
                                     event));
}

// #define GEMM_PROFILING
#ifdef GEMM_PROFILING
#define START_TIMER(n) \
    clFinish(ctx.get_queue().handle().get()); \
    start[n] = getTickCount();

#define STOP_TIMER(n) \
    clFinish(ctx.get_queue().handle().get()); \
    end[n] = getTickCount();
#else
#define START_TIMER(n)
#define STOP_TIMER(n)
#endif

enum gemm_type_t
{
    GEMM_TYPE_NONE = 0,
    GEMM_TYPE_FAST_IMAGE_32_1,
    GEMM_TYPE_FAST_IMAGE_32_2,
    GEMM_TYPE_FAST_BUFFER,
    GEMM_TYPE_MAX
};

static void greentea_gpu_fast_image_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                                         const CBLAS_TRANSPOSE TransB, const int_tp M,
                                         const int_tp N, const int_tp K, const float alpha,
                                         const cl_mem A, const int_tp offA, const cl_mem B,
                                         const int_tp offB, const float beta, cl_mem C,
                                         const int_tp offC, bool is_image_a, bool is_image_b,
                                         enum gemm_type_t gemm_type)
{
    CHECK_EQ(gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 || gemm_type == GEMM_TYPE_FAST_IMAGE_32_2, true)
             << "Invalid fast image gemm type." << std::endl;

    if (is_image_a)
        CHECK_EQ(offA, 0) << "Invalid input image offset." << std::endl;

    if (is_image_b)
        CHECK_EQ(offB, 0) << "Invalid input image offset." << std::endl;

    #ifdef GEMM_PROFILING
    int64 start[4], end[4];
    for (int i = 0; i < 4; i++)
        start[i] = end[i] = 0;
    #endif
    uint32_t widthA = (TransA == CblasNoTrans) ? K : M;
    uint32_t heightA = (TransA == CblasNoTrans) ? M : K;
    uint32_t widthB = (TransB == CblasNoTrans) ? N : K;
    uint32_t heightB = (TransB == CblasNoTrans) ? K : N;
    // To fix the edge problem casued by the sub group block read.
    // we have to pad the image if it's not multiple of tile.
    // just padding one line is enough as the sub group block read
    // will clamp to edge according to the spec.
    uint32_t padded_k = K + ((K & 7) ? 1 : 0);
    uint32_t imageA_w = (TransA == CblasNoTrans) ? padded_k : M;
    uint32_t imageA_h = (TransA == CblasNoTrans) ? M : padded_k;
    uint32_t imageB_w = (TransB == CblasNoTrans) ? N : padded_k;
    uint32_t imageB_h = (TransB == CblasNoTrans) ? padded_k : N;
    ocl::Context ctx = ocl::Context::getDefault();

    cl_mem ImA = NULL;
    cl_mem ImB = NULL;

    cl_event ev[5];
    cl_uint ev_idx = 0;
    memset(ev, 0, sizeof(cl_event) * 5);
    struct gemm_callback_arg * arg = new gemm_callback_arg;
    if (TransB == CblasNoTrans)
    {
        bool padding_A = false;
        bool padding_B = false;

        if (!is_image_a && !is_image_b)
        {
            if (M * K < N * K)
                padding_B = true;
            else
                padding_A = true;
        }

        START_TIMER(0);
        if (!is_image_a)
        {
            greentea_gpu_gemm_copy_buffer_to_image(ctx_id, &ImA, A, offA,
                                                   true, TransA != CblasNoTrans,
                                                   padding_A, imageA_h, imageA_w,
                                                   heightA, widthA, 0, NULL, &ev[ev_idx]);
            if (ev[ev_idx] != NULL)
                ev_idx++;
        }

        STOP_TIMER(0);
        START_TIMER(1);

        if (!is_image_b)
        {
            greentea_gpu_gemm_copy_buffer_to_image(ctx_id, &ImB, B, offB,
                                                   false, false,
                                                   padding_B, imageB_h, imageB_w,
                                                   heightB, widthB, 0, NULL, &ev[ev_idx]);
            if (ev[ev_idx] != NULL)
                ev_idx++;
        }
        STOP_TIMER(1);
    }
    else
    {
        // We will use normal read_imagef to read image B when B has transpose.
        // thus we don't need to pad image A at all.
        START_TIMER(2);
        if (!is_image_a)
        {
            bool padding;
            padding = !is_image_b;
            greentea_gpu_gemm_copy_buffer_to_image(ctx_id, &ImA, A, offA,
                                                   true, TransA != CblasNoTrans,
                                                   padding, imageA_h, imageA_w,
                                                   heightA, widthA, 0, NULL, &ev[ev_idx]);
            if (ev[ev_idx] != NULL)
            ev_idx++;
        }
        STOP_TIMER(2);
    }
    if (!is_image_a)
        arg->imgs.push_back(ImA);
    else
        ImA = A;
    if (!is_image_b)
        arg->imgs.push_back(ImB);
    else
        ImB = B;

    ocl::Kernel oclk_gemm_float;
    std::string kernel_name("gemm_");
    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1)
        kernel_name += "32_1_";
    else
        kernel_name += "32_2_";

    if (TransA == CblasNoTrans)
        kernel_name += "N";
    else
        kernel_name += "T";

    if (TransB == CblasNoTrans)
        kernel_name += "N_";
    else
    {
        kernel_name += "T_";
        if (is_image_b)
        {
            if (K % 4 == 0)
                kernel_name += "VEC4_";
            else
                kernel_name += "SCALAR_";
        }
        else
        {
            kernel_name += "BUFFER_";
        }
    }

    if (alpha == 1)
        kernel_name += "1_";
    else
        kernel_name += "0_";

    if (beta == 0)
        kernel_name += "0";
    else
        kernel_name += "1";
    kernel_name += "_float";

    String opts = "";
    oclk_gemm_float.create(kernel_name.c_str(), cv::ocl::dnn::gemm_image_oclsrc, opts);

    size_t global[2];

    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1)
        global[0] = (size_t)( N + 7 ) & ~7;
    else
        global[0] = (size_t)( (N / 2 ) + 7 ) ^ ~7;

    global[1]  = (size_t)(M + 31) / 32;
    const size_t local[] = {8, 1};

    cl_uint arg_idx = 0;
    oclk_gemm_float.set(arg_idx++, (cl_mem) ImA);
    if (TransB == CblasNoTrans || is_image_b)
        oclk_gemm_float.set(arg_idx++, (cl_mem) ImB);
    else
    {
        oclk_gemm_float.set(arg_idx++, (cl_mem) B);
        oclk_gemm_float.set(arg_idx++, offB);
    }
    oclk_gemm_float.set(arg_idx++, (cl_mem) C);
    oclk_gemm_float.set(arg_idx++, offC);
    oclk_gemm_float.set(arg_idx++, M);
    oclk_gemm_float.set(arg_idx++, N);
    oclk_gemm_float.set(arg_idx++, alpha);
    oclk_gemm_float.set(arg_idx++, beta);
    oclk_gemm_float.set(arg_idx++, padded_k);
    if (TransB != CblasNoTrans)
        oclk_gemm_float.set(arg_idx++, K);

    cl_event *wait_list = NULL;
    if (ev_idx != 0)
        wait_list = &ev[0];
    START_TIMER(3);
    OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)ocl::Queue::getDefault().ptr(),
                                     (cl_kernel)oclk_gemm_float.ptr(), 2, NULL,
                                     global, local, ev_idx,
                                     wait_list, &ev[ev_idx]));
    STOP_TIMER(3);
    #ifdef GEMM_PROFILING
    double elapsed[4], total_elapsed;
    for ( int i = 0; i < 4; i++ )
    {
        elapsed[i] = (double)end[i] - (double)start[i];
        total_elapsed += elapsed[i];
    }
    printf("kernel name %s \n", kernel_name.c_str());
    printf("gemm %d %d %d %f %f %d %d %f %f %f %f %fGFLOPS %f GFLOPS\n",
            M, K, N, alpha, beta, TransA == CblasNoTrans, TransB == CblasNoTrans,
            elapsed[0] / 1000., elapsed[1] / 1000., elapsed[2] / 1000.,
            elapsed[3] / 1000.,
            M * N * ( 2*K - 1. ) / ( elapsed[3] * 1e3 ),
            M * N * ( 2 * K - 1.) / ( total_elapsed * 1e3 ) );
    #endif
    arg->evs.assign(ev, ev + ev_idx + 1);
    clSetEventCallback(ev[ev_idx], CL_COMPLETE, &gemm_callback, (void*)arg);
}

static void greentea_gpu_fast_buffer_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                                          const CBLAS_TRANSPOSE TransB, const int_tp M,
                                          const int_tp N, const int_tp K, const float alpha,
                                          const cl_mem A, const int_tp offA, const cl_mem B,
                                          const int_tp offB, const float beta, cl_mem C,
                                          const int_tp offC, enum gemm_type_t gemm_type)
{
    CHECK_EQ(gemm_type == GEMM_TYPE_FAST_BUFFER, true)
             << "Invalid fast buffer gemm type." << std::endl;

#ifdef GEMM_PROFILING
    struct timeval start[1], end[1];
    start[0] = end[0];
#endif

    cl_event ev = NULL;

    ocl::Context ctx = ocl::Context::getDefault();
    size_t sub_group_size = 8;
    bool is_small_batch = (M == 2 || M == 4 || M == 8);
    ocl::Kernel oclk_gemm_float;
    std::string kernel_name("gemm_buffer_");
    if (TransA == CblasNoTrans && TransB == CblasNoTrans)
    {
        kernel_name += "NN_float";
    }
    else if (TransA == CblasNoTrans && TransB != CblasNoTrans)
    {
        if (M == 2)
            kernel_name +="NT_M_2_float";
        else if (M == 4)
            kernel_name +="NT_M_4_float";
        else if (M == 8)
            kernel_name +="NT_M_8_float";
        else
            kernel_name += "NT_float";
    }
    else if(TransA != CblasNoTrans && TransB == CblasNoTrans)
    {
        kernel_name += "TN_float";
    }
    else
    {
        kernel_name += "TT_float";
    }
    String opts = "";
    oclk_gemm_float.create(kernel_name.c_str(), cv::ocl::dnn::gemm_buffer_oclsrc, opts);
    size_t local[2] = {};
    size_t global[2] = {};
    if (TransA == CblasNoTrans && TransB != CblasNoTrans && is_small_batch )
    {
        if (M == 8)
            local[0] = 16;
        else if (M == 4)
            local[0] = 32;
        else
            local[0] = 64;
        local[1] = 1;

        if (M == 8)
            global[0] = N * local[0];
        else
            global[0] = (N + 3) / 4 * local[0];
        global[1] = 1;
    }
    else
    {
        size_t lx = sub_group_size;
        size_t ly = (TransB != CblasNoTrans && TransA == CblasNoTrans) ? 16 : 4;
        int dx = (TransB != CblasNoTrans && TransA == CblasNoTrans) ? 1 : 4;
        int dy = 8;
        size_t gx = (size_t)(N + dx - 1) / dx;
        size_t gy = (size_t)(M + dy - 1) / dy;
        global[0] = (gx + lx - 1) / lx * lx;
        global[1] = (gy + ly - 1) / ly * ly;
        local[0] = lx;
        local[1] = ly;
    }

    cl_uint arg_idx = 0;
    oclk_gemm_float.set(arg_idx++, (cl_mem) A);
    oclk_gemm_float.set(arg_idx++, offA);
    oclk_gemm_float.set(arg_idx++, (cl_mem) B);
    oclk_gemm_float.set(arg_idx++, offB);
    oclk_gemm_float.set(arg_idx++, (cl_mem) C);
    oclk_gemm_float.set(arg_idx++, offC);
    oclk_gemm_float.set(arg_idx++, M);
    oclk_gemm_float.set(arg_idx++, N);
    oclk_gemm_float.set(arg_idx++, K);
    oclk_gemm_float.set(arg_idx++, alpha);
    oclk_gemm_float.set(arg_idx++, beta);

    START_TIMER(0);
    if (TransB == CblasNoTrans || TransA != CblasNoTrans)
    {
        int stride = 256;
        for (int start_index = 0; start_index < K; start_index += stride)
        {
            oclk_gemm_float.set(arg_idx, start_index);
            OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)ocl::Queue::getDefault().ptr(),
                                             (cl_kernel)oclk_gemm_float.ptr(), 2, NULL,
                                             global, local, 0,
                                             NULL, &ev));
        }
    }
    else
    {
        OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)ocl::Queue::getDefault().ptr(),
                                         (cl_kernel)oclk_gemm_float.ptr(), 2, NULL,
                                         global, local, 0,
                                         NULL, &ev));
    }
    STOP_TIMER(0);
    clReleaseEvent(ev);

#ifdef GEMM_PROFILING
    double total_elapsed;
    total_elapsed = (end[0].tv_sec - start[0].tv_sec) * 1e6 + (end[0].tv_usec - start[0].tv_usec);
    printf("kernel name %s \n", kernel_name.c_str());
    printf("gemm %d %d %d %f %f %d %d %f %fGFLOPS\n",
            M, K, N, alpha, beta, TransA == CblasNoTrans, TransB == CblasNoTrans,
            total_elapsed / 1000., M * N * ( 2 * K - 1.) / ( total_elapsed * 1e3 ) );
#endif
}

template<typename Dtype>
static void greentea_gpu_gemm_common(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                                     const CBLAS_TRANSPOSE TransB, const int_tp M,
                                     const int_tp N, const int_tp K, const Dtype alpha,
                                     const cl_mem A, const int_tp offA, const cl_mem B,
                                     const int_tp offB, const Dtype beta, cl_mem C,
                                     const int_tp offC, bool is_image_a, bool is_image_b,
                                     gemm_type_t gemm_type)
{

    ocl::Context ctx = ocl::Context::getDefault();

    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 ||
        gemm_type == GEMM_TYPE_FAST_IMAGE_32_2)
    {
        greentea_gpu_fast_image_gemm(ctx_id, TransA, TransB, M, N, K,
                                     alpha, A, offA, B, offB, beta, C,
                                     offC, is_image_a, is_image_b,
                                     gemm_type);
    }
    else if (gemm_type == GEMM_TYPE_FAST_BUFFER)
    {
        greentea_gpu_fast_buffer_gemm(ctx_id, TransA, TransB, M, N, K,
                                      alpha, A, offA, B, offB, beta, C,
                                      offC, gemm_type);
    }
}

static void auto_tune_gemm(int ctx_id, const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB,
                           gemm_type_t *tuned_gemm_types,
                           bool use_fast_gemm_image)
{
    ocl::Context ctx = ocl::Context::getDefault();
    int M = 1024;
    int K = 512;
    int N = 1024;
    cl_int err;
    cl_mem A = clCreateBuffer((cl_context)ctx.ptr(), CL_MEM_ALLOC_HOST_PTR, M * K * sizeof(float), NULL, &err);
    OCL_CHECK(err);
    cl_mem B = clCreateBuffer((cl_context)ctx.ptr(), CL_MEM_ALLOC_HOST_PTR, K * N * sizeof(float), NULL, &err);
    OCL_CHECK(err);
    cl_mem C = clCreateBuffer((cl_context)ctx.ptr(), CL_MEM_ALLOC_HOST_PTR, M * N * sizeof(float), NULL, &err);
    OCL_CHECK(err);

    std::vector<gemm_type_t> gemm_tests;

    if (use_fast_gemm_image)
        gemm_tests.push_back(GEMM_TYPE_FAST_IMAGE_32_1);
    gemm_tests.push_back(GEMM_TYPE_FAST_BUFFER);

    // warm up.
    for ( int i = 0; i < gemm_tests.size(); i++ )
    {
        greentea_gpu_gemm_common(ctx_id, TransA, TransB, M, N, K,
                                 1.0f, A, 0, B, 0, 0.0f, C, 0, false, false,
                                 gemm_tests[i]);
    }
    double fastest_time = 1e10;
    int fastest_index = -1;
    clFinish((cl_command_queue)ocl::Queue::getDefault().ptr());
    for ( int i = 0; i < gemm_tests.size(); i++ )
    {
        int64 start, end;
        start = getTickCount();
        greentea_gpu_gemm_common(ctx_id, TransA, TransB, M, N, K,
                                 1.0f, A, 0, B, 0, 0.0f, C, 0, false, false,
                                 gemm_tests[i]);
        clFinish((cl_command_queue)ocl::Queue::getDefault().ptr());
        end = getTickCount();
        double elapsed = (double)end - (double)start;
        if (elapsed < fastest_time)
        {
            fastest_time = elapsed;
            fastest_index = i;
        }
    }
    clReleaseMemObject(A);
    clReleaseMemObject(B);
    clReleaseMemObject(C);

    if (fastest_index >= 0)
    {
        tuned_gemm_types[ctx_id] = gemm_tests[fastest_index];
#ifdef GEMM_PROFILING
        printf("The tuned GEMM kernel get %f GFLOPS with kernel type %d.\n",
               M*N*(2*(double)K-1)/(fastest_time * 1e3),
               tuned_gemm_types[ctx_id]);
#endif
    }
}

static gemm_type_t tuned_gemm_nn_types_with_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_nt_types_with_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tn_types_with_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tt_types_with_image[16] = {GEMM_TYPE_NONE};

static gemm_type_t tuned_gemm_nn_types_without_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_nt_types_without_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tn_types_without_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tt_types_without_image[16] = {GEMM_TYPE_NONE};

static void auto_tune_gemm_all(int ctx_id, bool use_fast_gemm_image)
{
    if(use_fast_gemm_image)
    {
        auto_tune_gemm(ctx_id, CblasNoTrans, CblasNoTrans, tuned_gemm_nn_types_with_image, true);
        auto_tune_gemm(ctx_id, CblasNoTrans, CblasTrans, tuned_gemm_nt_types_with_image, true);
        auto_tune_gemm(ctx_id, CblasTrans, CblasNoTrans, tuned_gemm_tn_types_with_image, true);
        auto_tune_gemm(ctx_id, CblasTrans, CblasTrans, tuned_gemm_tt_types_with_image, true);
    }
    else
    {
        auto_tune_gemm(ctx_id, CblasNoTrans, CblasNoTrans, tuned_gemm_nn_types_without_image, false);
        auto_tune_gemm(ctx_id, CblasNoTrans, CblasTrans, tuned_gemm_nt_types_without_image, false);
        auto_tune_gemm(ctx_id, CblasTrans, CblasNoTrans, tuned_gemm_tn_types_without_image, false);
        auto_tune_gemm(ctx_id, CblasTrans, CblasTrans, tuned_gemm_tt_types_without_image, false);
    }
}

static std::mutex auto_tune_gemm_mutex;

template<typename Dtype>
void greentea_gpu_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC, bool is_image_a, bool is_image_b)
{
    ocl::Device dev = ocl::Device::getDefault();
    bool use_fast_gemm_image = false;
    bool use_fast_gemm_buffer = false;

    if (dev.type() == CL_DEVICE_TYPE_CPU)
    {
        LOG(FATAL) << "Not Support CPU device";
        return;
    }

    if (dev.type() == CL_DEVICE_TYPE_GPU &&
        std::is_same<Dtype, float>::value)
    {
        // Check whether can/should we use the fast gemm driver.
        // There are the following considerations/restrications:
        // 1. The fast gemm kernel is using image which has a size limitation.
        // 2. The fast gemm kernel is using the intel sub group extension.
        // 3. Currently, only the IGC compiler (the driver version is 16.xxx)
        //    can get better performance with the fast gemm.
        // Cap at 1 MB to capture faulty OpenCL implementations (nVidia)
        bool has_sub_group_ext = dev.extensions().find("cl_intel_subgroups") != std::string::npos;
        if (has_sub_group_ext)
        {
            size_t max_image_size = std::min(dev.image2DMaxWidth(),
                                             dev.image2DMaxHeight());
            if (M <= max_image_size &&
                K <= max_image_size &&
                N <= max_image_size)
            {
                use_fast_gemm_image = true;
            }
            use_fast_gemm_buffer = true;
        }
    }

    gemm_type_t preferred_gemm_type = GEMM_TYPE_FAST_BUFFER;

    if (0)  //disable auto tune
    {
        std::lock_guard<std::mutex> lock(auto_tune_gemm_mutex);
        if (use_fast_gemm_image)
        {
            if (tuned_gemm_nn_types_with_image[ctx_id] == GEMM_TYPE_NONE)
            {
                auto_tune_gemm_all(ctx_id, true);
            }

            if (TransA == CblasNoTrans && TransB == CblasNoTrans)
                preferred_gemm_type = tuned_gemm_nn_types_with_image[ctx_id];
            else if (TransA == CblasTrans && TransB == CblasNoTrans)
                preferred_gemm_type = tuned_gemm_tn_types_with_image[ctx_id];
            else if (TransA == CblasNoTrans && TransB == CblasTrans)
                preferred_gemm_type = tuned_gemm_nt_types_with_image[ctx_id];
            else if (TransA == CblasTrans && TransB == CblasTrans)
                preferred_gemm_type = tuned_gemm_tt_types_with_image[ctx_id];
        }
        else if (use_fast_gemm_buffer)
        {
            if (tuned_gemm_nn_types_without_image[ctx_id] == GEMM_TYPE_NONE)
            {
                auto_tune_gemm_all(ctx_id, false);
            }

            if (TransA == CblasNoTrans && TransB == CblasNoTrans)
                preferred_gemm_type = tuned_gemm_nn_types_without_image[ctx_id];
            else if (TransA == CblasTrans && TransB == CblasNoTrans)
                preferred_gemm_type = tuned_gemm_tn_types_without_image[ctx_id];
            else if (TransA == CblasNoTrans && TransB == CblasTrans)
                preferred_gemm_type = tuned_gemm_nt_types_without_image[ctx_id];
            else if (TransA == CblasTrans && TransB == CblasTrans)
                preferred_gemm_type = tuned_gemm_tt_types_without_image[ctx_id];
        }
    }

    CHECK_EQ(use_fast_gemm_image || (!is_image_a && !is_image_b), true)
    << "Invalid GEMM parameters.";

    if (is_image_a || is_image_b)
        preferred_gemm_type = GEMM_TYPE_FAST_IMAGE_32_1;

    greentea_gpu_gemm_common(ctx_id, TransA, TransB, M, N, K, alpha, A, offA,
                             B, offB, beta, C, offC, is_image_a, is_image_b,
                             preferred_gemm_type);
}

template void greentea_gpu_gemm<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int_tp M, const int_tp N,
                                       const int_tp K, const float alpha,
                                       const cl_mem A, const int_tp offA,
                                       const cl_mem B, const int_tp offB,
                                       const float beta, cl_mem C,
                                       const int_tp offC,
                                       const bool is_image_a,
                                       const bool is_image_b);
template void greentea_gpu_gemm<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int_tp M, const int_tp N,
                                        const int_tp K, const double alpha,
                                        const cl_mem A, const int_tp offA,
                                        const cl_mem B, const int_tp offB,
                                        const double beta, cl_mem C,
                                        const int_tp offC,
                                        const bool is_image_a,
                                        const bool is_image_b);

template void greentea_gpu_gemm_common<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int_tp M, const int_tp N,
                                       const int_tp K, const float alpha,
                                       const cl_mem A, const int_tp offA,
                                       const cl_mem B, const int_tp offB,
                                       const float beta, cl_mem C,
                                       const int_tp offC,
                                       const bool is_image_a,
                                       const bool is_image_b,
                                       const gemm_type_t);
template void greentea_gpu_gemm_common<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int_tp M, const int_tp N,
                                        const int_tp K, const double alpha,
                                        const cl_mem A, const int_tp offA,
                                        const cl_mem B, const int_tp offB,
                                        const double beta, cl_mem C,
                                        const int_tp offC,
                                        const bool is_image_a,
                                        const bool is_image_b,
                                        const gemm_type_t);

template<typename Dtype>
void greentea_gpu_gemv(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int_tp M, const int_tp N, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem x,
                       const int_tp offx, const Dtype beta, cl_mem y,
                       const int_tp offy)
{
    ocl::Context ctx = ocl::Context::getDefault();

    if (ocl::Device::getDefault().type() == CL_DEVICE_TYPE_CPU)
    {
        LOG(FATAL) << "Not Support CPU device";
    }
    else
    {
        if (std::is_same<Dtype, float>::value && TransA == CblasNoTrans)
        {
            ocl::Kernel k(CL_KERNEL_SELECT("matvec_mul4"), cv::ocl::dnn::matvec_mul_oclsrc);
            uint row_size = M;
            uint col_size = N;
            size_t localsize = 128;
            size_t globalsize = row_size / 4 * localsize;

            uint argId = 0;
            k.set(argId++, (cl_mem) A);
            k.set(argId++, offA);
            k.set(argId++, cl_uint(col_size));
            k.set(argId++, cl_uint(col_size%4));
            k.set(argId++, (cl_mem) x);
            k.set(argId++, offx);
            k.set(argId++, alpha);
            k.set(argId++, beta);
            k.set(argId++, (cl_mem) y);
            k.set(argId++, offy);
            clSetKernelArg((cl_kernel)k.ptr(), argId++, localsize * sizeof(cl_float4), NULL);

            clEnqueueNDRangeKernel((cl_command_queue)ocl::Queue::getDefault().ptr(),
                                   (cl_kernel)k.ptr(), 1,
                                   NULL,
                                   &globalsize,
                                   &localsize, 0, NULL,
                                   NULL);
            if ((row_size % 4) != 0)
            {
                ocl::Kernel k_1(CL_KERNEL_SELECT("matvec_mul1"), cv::ocl::dnn::matvec_mul_oclsrc);
                size_t localsize = 128;
                size_t globalsize = row_size % 4 * localsize;
                uint row_offset = row_size - (row_size % 4);

                uint argId = 0;
                k_1.set(argId++, (cl_mem) A);
                k_1.set(argId++, offA);
                k_1.set(argId++, cl_uint(col_size));
                k_1.set(argId++, cl_uint(row_offset));
                k_1.set(argId++, cl_uint(col_size%4));
                k_1.set(argId++, (cl_mem) x);
                k_1.set(argId++, offx);
                k_1.set(argId++, alpha);
                k_1.set(argId++, beta);
                k_1.set(argId++, (cl_mem) y);
                k_1.set(argId++, offy);
                clSetKernelArg((cl_kernel)k_1.ptr(), argId++, localsize * sizeof(cl_float), NULL);

                clEnqueueNDRangeKernel((cl_command_queue)ocl::Queue::getDefault().ptr(),
                                       (cl_kernel)k_1.ptr(), 1,
                                       NULL,
                                       &globalsize,
                                       &localsize, 0, NULL,
                                       NULL);
            }
        }
        else
        {
            /* FIXME add implementation here */
        }
    }
}

template void greentea_gpu_gemv<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const int_tp M, const int_tp N,
                                       const float alpha, const cl_mem A,
                                       const int_tp offA, const cl_mem x,
                                       const int_tp offx, const float beta,
                                       cl_mem y, const int_tp offy);
template void greentea_gpu_gemv<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const int_tp M, const int_tp N,
                                        const double alpha, const cl_mem A,
                                        const int_tp offA, const cl_mem x,
                                        const int_tp offx, const double beta,
                                        cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_axpy(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       const cl_mem X, const int_tp offX, cl_mem Y,
                       const int_tp offY)
{
    ocl::Context ctx = ocl::Context::getDefault();

    if (ocl::Device::getDefault().type() == CL_DEVICE_TYPE_CPU)
    {
        LOG(FATAL) << "Not Support CPU device";
    }
    else
    {
        ocl::Kernel oclk_axpy(CL_KERNEL_SELECT("axpy"), cv::ocl::dnn::math_oclsrc);
        size_t global[] = { 128 * 128 };
        size_t local[] = { 128 };

        cl_uint argIdx = 0;
        oclk_axpy.set(argIdx++, N);
        oclk_axpy.set(argIdx++, alpha);
        oclk_axpy.set(argIdx++, (cl_mem) X);
        oclk_axpy.set(argIdx++, offX);
        oclk_axpy.set(argIdx++, (cl_mem) Y);
        oclk_axpy.set(argIdx++, offY);

        oclk_axpy.run(1, global, local, false);
    }
}

template void greentea_gpu_axpy<float>(const int_tp ctx_id, const int_tp N,
                                       const float alpha, const cl_mem X,
                                       const int_tp offX, cl_mem Y,
                                       const int_tp offY);
template void greentea_gpu_axpy<double>(const int_tp ctx_id, const int_tp N,
                                        const double alpha, const cl_mem X,
                                        const int_tp offX, cl_mem Y,
                                        const int_tp offY);

template<typename Dtype>
void greentea_gpu_set(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                      cl_mem Y, const int_tp offY) {
    // OpenCL Version >= 1.2 approach
    // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
    //                  Y, &alpha, sizeof(Dtype),
    //                  offY, N, 0, NULL, NULL);

    // OpenCL Version < 1.2 fallback
    ocl::Kernel oclk_fill(CL_KERNEL_SELECT("fill"), cv::ocl::dnn::fillbuffer_oclsrc);
    size_t global[] = { 128 * 128 };
    size_t local[] = { 128 };

    cl_uint argIdx = 0;
    oclk_fill.set(argIdx++, N);
    oclk_fill.set(argIdx++, alpha);
    oclk_fill.set(argIdx++, (cl_mem) Y);
    oclk_fill.set(argIdx++, offY);

    oclk_fill.run(1, global, local, false);
}

template void greentea_gpu_set<int_tp>(const int_tp ctx_id, const int_tp N,
                                       const int_tp alpha, cl_mem Y,
                                       const int_tp offY);
template void greentea_gpu_set<float>(const int_tp ctx_id, const int_tp N,
                                      const float alpha, cl_mem Y,
                                      const int_tp offY);
template void greentea_gpu_set<double>(const int_tp ctx_id, const int_tp N,
                                       const double alpha, cl_mem Y,
                                       const int_tp offY);
}  // namespace greentea
#endif  // HAVE_OPENCL
