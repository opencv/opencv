/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Jin Ma, jin@multicorewareinc.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

namespace cv
{
    namespace ocl
    {
        namespace stereoCSBP
        {
            static inline int divUp(int total, int grain)
            {
                return (total + grain - 1) / grain;
            }
            static string get_kernel_name(string kernel_name, int data_type)
            {
                stringstream idxStr;
                if(data_type == CV_16S)
                    idxStr << "0";
                else
                    idxStr << "1";
                kernel_name += idxStr.str();

                return kernel_name;
            }
            using cv::ocl::StereoConstantSpaceBP;
            //////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////init_data_cost//////////////////////////////////
            //////////////////////////////////////////////////////////////////////////////////
            static void init_data_cost_caller(const oclMat &left, const oclMat &right, oclMat &temp,
                StereoConstantSpaceBP &rthis,
                int msg_step, int h, int w, int level)
            {
                Context  *clCxt = left.clCxt;
                int data_type = rthis.msg_type;
                int channels = left.oclchannels();

                string kernelName = get_kernel_name("init_data_cost_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                //size_t blockSize = 256;
                size_t localThreads[]  = {32, 8 ,1};
                size_t globalThreads[] = {divUp(w, localThreads[0]) *localThreads[0],
                    divUp(h, localThreads[1]) *localThreads[1],
                    1
                };

                int cdisp_step1 = msg_step * h;
                openCLVerifyKernel(clCxt, kernel,  localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem),  (void *)&temp.data));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_mem),  (void *)&left.data));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_mem),  (void *)&right.data));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_int),  (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_int),  (void *)&w));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int),  (void *)&level));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_int),  (void *)&channels));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int),  (void *)&msg_step));
                openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_float), (void *)&rthis.data_weight));
                openCLSafeCall(clSetKernelArg(kernel, 9, sizeof(cl_float), (void *)&rthis.max_data_term));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&cdisp_step1));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_int), (void *)&rthis.min_disp_th));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_int), (void *)&left.step));
                openCLSafeCall(clSetKernelArg(kernel, 13, sizeof(cl_int), (void *)&rthis.ndisp));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 2, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }

            static void init_data_cost_reduce_caller(const oclMat &left, const oclMat &right, oclMat &temp,
                StereoConstantSpaceBP &rthis,
                int msg_step, int h, int w, int level)
            {

                Context  *clCxt = left.clCxt;
                int data_type = rthis.msg_type;
                int channels = left.oclchannels();
                int win_size = (int)std::pow(2.f, level);

                string kernelName = get_kernel_name("init_data_cost_reduce_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                const int threadsNum = 256;
                //size_t blockSize = threadsNum;
                size_t localThreads[3]  = {(size_t)win_size, 1, (size_t)threadsNum / win_size};
                size_t globalThreads[3] = { w *localThreads[0],
                    h * divUp(rthis.ndisp, localThreads[2]) *localThreads[1], 1 * localThreads[2]
                };

                int local_mem_size = threadsNum * sizeof(float);
                int cdisp_step1 = msg_step * h;

                openCLVerifyKernel(clCxt, kernel, localThreads);

                openCLSafeCall(clSetKernelArg(kernel, 0,  sizeof(cl_mem),  (void *)&temp.data));
                openCLSafeCall(clSetKernelArg(kernel, 1,  sizeof(cl_mem),  (void *)&left.data));
                openCLSafeCall(clSetKernelArg(kernel, 2,  sizeof(cl_mem),  (void *)&right.data));
                openCLSafeCall(clSetKernelArg(kernel, 3,  local_mem_size,  (void *)NULL));
                openCLSafeCall(clSetKernelArg(kernel, 4,  sizeof(cl_int),  (void *)&level));
                openCLSafeCall(clSetKernelArg(kernel, 5,  sizeof(cl_int),  (void *)&left.rows));
                openCLSafeCall(clSetKernelArg(kernel, 6,  sizeof(cl_int),  (void *)&left.cols));
                openCLSafeCall(clSetKernelArg(kernel, 7,  sizeof(cl_int),  (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 8,  sizeof(cl_int),  (void *)&win_size));
                openCLSafeCall(clSetKernelArg(kernel, 9,  sizeof(cl_int),  (void *)&channels));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int),  (void *)&rthis.ndisp));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_int),  (void *)&left.step));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_float), (void *)&rthis.data_weight));
                openCLSafeCall(clSetKernelArg(kernel, 13, sizeof(cl_float), (void *)&rthis.max_data_term));
                openCLSafeCall(clSetKernelArg(kernel, 14, sizeof(cl_int),  (void *)&rthis.min_disp_th));
                openCLSafeCall(clSetKernelArg(kernel, 15, sizeof(cl_int),  (void *)&cdisp_step1));
                openCLSafeCall(clSetKernelArg(kernel, 16, sizeof(cl_int),  (void *)&msg_step));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 3, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));
                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }

            static void get_first_initial_local_caller(uchar *data_cost_selected, uchar *disp_selected_pyr,
                oclMat &temp, StereoConstantSpaceBP &rthis,
                int h, int w, int nr_plane, int msg_step)
            {
                Context  *clCxt = temp.clCxt;
                int data_type = rthis.msg_type;

                string kernelName = get_kernel_name("get_first_k_initial_local_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                //size_t blockSize = 256;
                size_t localThreads[]  = {32, 8 ,1};
                size_t globalThreads[] = { roundUp(w, localThreads[0]), roundUp(h, localThreads[1]), 1 };

                int disp_step = msg_step * h;
                openCLVerifyKernel(clCxt, kernel, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data_cost_selected));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&disp_selected_pyr));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&temp.data));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&w));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&nr_plane));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&msg_step));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&disp_step));
                openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&rthis.ndisp));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 2, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }
            static void get_first_initial_global_caller(uchar *data_cost_selected, uchar *disp_selected_pyr,
                oclMat &temp, StereoConstantSpaceBP &rthis,
                int h, int w, int nr_plane, int msg_step)
            {
                Context  *clCxt = temp.clCxt;
                int data_type = rthis.msg_type;

                string kernelName = get_kernel_name("get_first_k_initial_global_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                //size_t blockSize = 256;
                size_t localThreads[]  = {32, 8, 1};
                size_t globalThreads[] = {divUp(w, localThreads[0]) *localThreads[0],
                    divUp(h, localThreads[1]) *localThreads[1],
                    1
                };

                int disp_step = msg_step * h;
                openCLVerifyKernel(clCxt, kernel, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data_cost_selected));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&disp_selected_pyr));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&temp.data));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&w));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&nr_plane));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&msg_step));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&disp_step));
                openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&rthis.ndisp));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 2, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }

            static void init_data_cost(const oclMat &left, const oclMat &right, oclMat &temp, StereoConstantSpaceBP &rthis,
                uchar *disp_selected_pyr, uchar *data_cost_selected,
                size_t msg_step, int h, int w, int level, int nr_plane)
            {

                if(level <= 1)
                    init_data_cost_caller(left, right, temp, rthis, msg_step, h, w, level);
                else
                    init_data_cost_reduce_caller(left, right, temp, rthis, msg_step, h, w, level);

                if(rthis.use_local_init_data_cost == true)
                {
                    get_first_initial_local_caller(data_cost_selected, disp_selected_pyr, temp, rthis, h, w, nr_plane, msg_step);
                }
                else
                {
                    get_first_initial_global_caller(data_cost_selected, disp_selected_pyr, temp, rthis, h, w,
                        nr_plane, msg_step);
                }
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////compute_data_cost//////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////
            static void compute_data_cost_caller(uchar *disp_selected_pyr, uchar *data_cost,
                StereoConstantSpaceBP &rthis, int msg_step1,
                int msg_step2, const oclMat &left, const oclMat &right, int h,
                int w, int h2, int level, int nr_plane)
            {
                Context  *clCxt = left.clCxt;
                int channels = left.oclchannels();
                int data_type = rthis.msg_type;

                string kernelName = get_kernel_name("compute_data_cost_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                size_t localThreads[]  = { 32, 8, 1 };
                size_t globalThreads[] = { roundUp(w, localThreads[0]), roundUp(h, localThreads[1]), 1 };

                int disp_step1 = msg_step1 * h;
                int disp_step2 = msg_step2 * h2;
                openCLVerifyKernel(clCxt, kernel, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0,  sizeof(cl_mem),  (void *)&disp_selected_pyr));
                openCLSafeCall(clSetKernelArg(kernel, 1,  sizeof(cl_mem),  (void *)&data_cost));
                openCLSafeCall(clSetKernelArg(kernel, 2,  sizeof(cl_mem),  (void *)&left.data));
                openCLSafeCall(clSetKernelArg(kernel, 3,  sizeof(cl_mem),  (void *)&right.data));
                openCLSafeCall(clSetKernelArg(kernel, 4,  sizeof(cl_int),  (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 5,  sizeof(cl_int),  (void *)&w));
                openCLSafeCall(clSetKernelArg(kernel, 6,  sizeof(cl_int),  (void *)&level));
                openCLSafeCall(clSetKernelArg(kernel, 7,  sizeof(cl_int),  (void *)&nr_plane));
                openCLSafeCall(clSetKernelArg(kernel, 8,  sizeof(cl_int),  (void *)&channels));
                openCLSafeCall(clSetKernelArg(kernel, 9,  sizeof(cl_int),  (void *)&msg_step1));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int),  (void *)&msg_step2));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_int),  (void *)&disp_step1));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_int),  (void *)&disp_step2));
                openCLSafeCall(clSetKernelArg(kernel, 13, sizeof(cl_float), (void *)&rthis.data_weight));
                openCLSafeCall(clSetKernelArg(kernel, 14, sizeof(cl_float), (void *)&rthis.max_data_term));
                openCLSafeCall(clSetKernelArg(kernel, 15, sizeof(cl_int),  (void *)&left.step));
                openCLSafeCall(clSetKernelArg(kernel, 16, sizeof(cl_int),  (void *)&rthis.min_disp_th));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 2, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }
            static void compute_data_cost_reduce_caller(uchar *disp_selected_pyr, uchar *data_cost,
                StereoConstantSpaceBP &rthis, int msg_step1,
                int msg_step2, const oclMat &left, const oclMat &right, int h,
                int w, int h2, int level, int nr_plane)
            {
                Context  *clCxt = left.clCxt;
                int data_type = rthis.msg_type;
                int channels = left.oclchannels();
                int win_size = (int)std::pow(2.f, level);

                string kernelName = get_kernel_name("compute_data_cost_reduce_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                const size_t threadsNum = 256;
                //size_t blockSize = threadsNum;
                size_t localThreads[3]  = { (size_t)win_size, 1, (size_t)threadsNum / win_size };
                size_t globalThreads[3] = { w *localThreads[0],
                    h * divUp(nr_plane, localThreads[2]) *localThreads[1], 1 * localThreads[2]
                };

                int disp_step1 = msg_step1 * h;
                int disp_step2 = msg_step2 * h2;
                size_t local_mem_size = threadsNum * sizeof(float);
                openCLVerifyKernel(clCxt, kernel, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0,  sizeof(cl_mem),  (void *)&disp_selected_pyr));
                openCLSafeCall(clSetKernelArg(kernel, 1,  sizeof(cl_mem),  (void *)&data_cost));
                openCLSafeCall(clSetKernelArg(kernel, 2,  sizeof(cl_mem),  (void *)&left.data));
                openCLSafeCall(clSetKernelArg(kernel, 3,  sizeof(cl_mem),  (void *)&right.data));
                openCLSafeCall(clSetKernelArg(kernel, 4, local_mem_size,   (void *)NULL));
                openCLSafeCall(clSetKernelArg(kernel, 5,  sizeof(cl_int),  (void *)&level));
                openCLSafeCall(clSetKernelArg(kernel, 6,  sizeof(cl_int),  (void *)&left.rows));
                openCLSafeCall(clSetKernelArg(kernel, 7,  sizeof(cl_int),  (void *)&left.cols));
                openCLSafeCall(clSetKernelArg(kernel, 8,  sizeof(cl_int),  (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 9,  sizeof(cl_int),  (void *)&nr_plane));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int),  (void *)&channels));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_int),  (void *)&win_size));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_int),  (void *)&msg_step1));
                openCLSafeCall(clSetKernelArg(kernel, 13, sizeof(cl_int),  (void *)&msg_step2));
                openCLSafeCall(clSetKernelArg(kernel, 14, sizeof(cl_int),  (void *)&disp_step1));
                openCLSafeCall(clSetKernelArg(kernel, 15, sizeof(cl_int),  (void *)&disp_step2));
                openCLSafeCall(clSetKernelArg(kernel, 16, sizeof(cl_float), (void *)&rthis.data_weight));
                openCLSafeCall(clSetKernelArg(kernel, 17, sizeof(cl_float), (void *)&rthis.max_data_term));
                openCLSafeCall(clSetKernelArg(kernel, 18, sizeof(cl_int),  (void *)&left.step));
                openCLSafeCall(clSetKernelArg(kernel, 19, sizeof(cl_int),  (void *)&rthis.min_disp_th));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 3, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }
            static void compute_data_cost(uchar *disp_selected_pyr, uchar *data_cost, StereoConstantSpaceBP &rthis,
                int msg_step1, int msg_step2, const oclMat &left, const oclMat &right, int h, int w,
                int h2, int level, int nr_plane)
            {
                if(level <= 1)
                    compute_data_cost_caller(disp_selected_pyr, data_cost, rthis, msg_step1, msg_step2,
                    left, right, h, w, h2, level, nr_plane);
                else
                    compute_data_cost_reduce_caller(disp_selected_pyr, data_cost, rthis,  msg_step1, msg_step2,
                    left, right, h, w, h2, level, nr_plane);
            }
            ////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////////////init message//////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////
            static void init_message(uchar *u_new, uchar *d_new, uchar *l_new, uchar *r_new,
                uchar *u_cur, uchar *d_cur, uchar *l_cur, uchar *r_cur,
                uchar *disp_selected_pyr_new, uchar *disp_selected_pyr_cur,
                uchar *data_cost_selected, uchar *data_cost, oclMat &temp, StereoConstantSpaceBP rthis,
                size_t msg_step1, size_t msg_step2, int h, int w, int nr_plane,
                int h2, int w2, int nr_plane2)
            {
                Context  *clCxt = temp.clCxt;
                int data_type = rthis.msg_type;

                string kernelName = get_kernel_name("init_message_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                //size_t blockSize = 256;
                size_t localThreads[]  = {32, 8, 1};
                size_t globalThreads[] = { roundUp(w, localThreads[0]), roundUp(h, localThreads[1]), 1 };

                int disp_step1 = msg_step1 * h;
                int disp_step2 = msg_step2 * h2;
                openCLVerifyKernel(clCxt, kernel, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0,  sizeof(cl_mem), (void *)&u_new));
                openCLSafeCall(clSetKernelArg(kernel, 1,  sizeof(cl_mem), (void *)&d_new));
                openCLSafeCall(clSetKernelArg(kernel, 2,  sizeof(cl_mem), (void *)&l_new));
                openCLSafeCall(clSetKernelArg(kernel, 3,  sizeof(cl_mem), (void *)&r_new));
                openCLSafeCall(clSetKernelArg(kernel, 4,  sizeof(cl_mem), (void *)&u_cur));
                openCLSafeCall(clSetKernelArg(kernel, 5,  sizeof(cl_mem), (void *)&d_cur));
                openCLSafeCall(clSetKernelArg(kernel, 6,  sizeof(cl_mem), (void *)&l_cur));
                openCLSafeCall(clSetKernelArg(kernel, 7,  sizeof(cl_mem), (void *)&r_cur));
                openCLSafeCall(clSetKernelArg(kernel, 8,  sizeof(cl_mem), (void *)&temp.data));
                openCLSafeCall(clSetKernelArg(kernel, 9,  sizeof(cl_mem), (void *)&disp_selected_pyr_new));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&disp_selected_pyr_cur));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&data_cost_selected));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&data_cost));
                openCLSafeCall(clSetKernelArg(kernel, 13, sizeof(cl_int), (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 14, sizeof(cl_int), (void *)&w));
                openCLSafeCall(clSetKernelArg(kernel, 15, sizeof(cl_int), (void *)&nr_plane));
                openCLSafeCall(clSetKernelArg(kernel, 16, sizeof(cl_int), (void *)&h2));
                openCLSafeCall(clSetKernelArg(kernel, 17, sizeof(cl_int), (void *)&w2));
                openCLSafeCall(clSetKernelArg(kernel, 18, sizeof(cl_int), (void *)&nr_plane2));
                openCLSafeCall(clSetKernelArg(kernel, 19, sizeof(cl_int), (void *)&disp_step1));
                openCLSafeCall(clSetKernelArg(kernel, 20, sizeof(cl_int), (void *)&disp_step2));
                openCLSafeCall(clSetKernelArg(kernel, 21, sizeof(cl_int), (void *)&msg_step1));
                openCLSafeCall(clSetKernelArg(kernel, 22, sizeof(cl_int), (void *)&msg_step2));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 2, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }
            ////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////calc_all_iterations////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////////////////////////////
            static void calc_all_iterations_caller(uchar *u, uchar *d, uchar *l, uchar *r, uchar *data_cost_selected,
                uchar *disp_selected_pyr, oclMat &temp, StereoConstantSpaceBP rthis,
                int msg_step, int h, int w, int nr_plane, int i)
            {
                Context  *clCxt = temp.clCxt;
                int data_type = rthis.msg_type;

                string kernelName = get_kernel_name("compute_message_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);
                size_t localThreads[]  = {32, 8, 1};
                size_t globalThreads[] = {divUp(w, (localThreads[0]) << 1) *localThreads[0],
                    divUp(h, localThreads[1]) *localThreads[1],
                    1
                };

                int disp_step = msg_step * h;
                openCLVerifyKernel(clCxt, kernel, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0,  sizeof(cl_mem),  (void *)&u));
                openCLSafeCall(clSetKernelArg(kernel, 1,  sizeof(cl_mem),  (void *)&d));
                openCLSafeCall(clSetKernelArg(kernel, 2,  sizeof(cl_mem),  (void *)&l));
                openCLSafeCall(clSetKernelArg(kernel, 3,  sizeof(cl_mem),  (void *)&r));
                openCLSafeCall(clSetKernelArg(kernel, 4,  sizeof(cl_mem),  (void *)&data_cost_selected));
                openCLSafeCall(clSetKernelArg(kernel, 5,  sizeof(cl_mem),  (void *)&disp_selected_pyr));
                openCLSafeCall(clSetKernelArg(kernel, 6,  sizeof(cl_mem),  (void *)&temp.data));
                openCLSafeCall(clSetKernelArg(kernel, 7,  sizeof(cl_int),  (void *)&h));
                openCLSafeCall(clSetKernelArg(kernel, 8,  sizeof(cl_int),  (void *)&w));
                openCLSafeCall(clSetKernelArg(kernel, 9,  sizeof(cl_int),  (void *)&nr_plane));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int),  (void *)&i));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_float), (void *)&rthis.max_disc_term));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_int),  (void *)&disp_step));
                openCLSafeCall(clSetKernelArg(kernel, 13, sizeof(cl_int),  (void *)&msg_step));
                openCLSafeCall(clSetKernelArg(kernel, 14, sizeof(cl_float), (void *)&rthis.disc_single_jump));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 2, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }
            static void calc_all_iterations(uchar *u, uchar *d, uchar *l, uchar *r, uchar *data_cost_selected,
                uchar *disp_selected_pyr, oclMat &temp, StereoConstantSpaceBP rthis,
                int msg_step, int h, int w, int nr_plane)
            {
                for(int t = 0; t < rthis.iters; t++)
                    calc_all_iterations_caller(u, d, l, r, data_cost_selected, disp_selected_pyr, temp, rthis,
                    msg_step, h, w, nr_plane, t & 1);
            }

            ///////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////compute_disp////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            static void compute_disp(uchar *u, uchar *d, uchar *l, uchar *r, uchar *data_cost_selected,
                uchar *disp_selected_pyr, StereoConstantSpaceBP &rthis, size_t msg_step,
                oclMat &disp, int nr_plane)
            {
                Context  *clCxt = disp.clCxt;
                int data_type = rthis.msg_type;

                string kernelName = get_kernel_name("compute_disp_", data_type);

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereocsbp, kernelName);

                //size_t blockSize = 256;
                size_t localThreads[]  = { 32, 8, 1 };
                size_t globalThreads[] = { roundUp(disp.cols, localThreads[0]), roundUp(disp.rows, localThreads[1]), 1 };

                int step_size = disp.step / disp.elemSize();
                int disp_step = disp.rows * msg_step;
                openCLVerifyKernel(clCxt, kernel, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0,  sizeof(cl_mem), (void *)&u));
                openCLSafeCall(clSetKernelArg(kernel, 1,  sizeof(cl_mem), (void *)&d));
                openCLSafeCall(clSetKernelArg(kernel, 2,  sizeof(cl_mem), (void *)&l));
                openCLSafeCall(clSetKernelArg(kernel, 3,  sizeof(cl_mem), (void *)&r));
                openCLSafeCall(clSetKernelArg(kernel, 4,  sizeof(cl_mem), (void *)&data_cost_selected));
                openCLSafeCall(clSetKernelArg(kernel, 5,  sizeof(cl_mem), (void *)&disp_selected_pyr));
                openCLSafeCall(clSetKernelArg(kernel, 6,  sizeof(cl_mem), (void *)&disp.data));
                openCLSafeCall(clSetKernelArg(kernel, 7,  sizeof(cl_int), (void *)&step_size));
                openCLSafeCall(clSetKernelArg(kernel, 8,  sizeof(cl_int), (void *)&disp.cols));
                openCLSafeCall(clSetKernelArg(kernel, 9,  sizeof(cl_int), (void *)&disp.rows));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&nr_plane));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_int), (void *)&msg_step));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_int), (void *)&disp_step));
                openCLSafeCall(clEnqueueNDRangeKernel(*(cl_command_queue*)getClCommandQueuePtr(), kernel, 2, NULL,
                    globalThreads, localThreads, 0, NULL, NULL));

                clFinish(*(cl_command_queue*)getClCommandQueuePtr());
                openCLSafeCall(clReleaseKernel(kernel));
            }
        }
    }
}
namespace
{
    const float DEFAULT_MAX_DATA_TERM = 30.0f;
    const float DEFAULT_DATA_WEIGHT = 1.0f;
    const float DEFAULT_MAX_DISC_TERM = 160.0f;
    const float DEFAULT_DISC_SINGLE_JUMP = 10.0f;
}

void cv::ocl::StereoConstantSpaceBP::estimateRecommendedParams(int width, int height, int &ndisp, int &iters, int &levels, int &nr_plane)
{
    ndisp = (int) ((float) width / 3.14f);
    if ((ndisp & 1) != 0)
        ndisp++;

    int mm = ::max(width, height);
    iters = mm / 100 + ((mm > 1200) ? - 4 : 4);

    levels = (int)::log(static_cast<double>(mm)) * 2 / 3;
    if (levels == 0) levels++;

    nr_plane = (int) ((float) ndisp / std::pow(2.0, levels + 1));
}

cv::ocl::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp_, int iters_, int levels_, int nr_plane_,
    int msg_type_)

    : ndisp(ndisp_), iters(iters_), levels(levels_), nr_plane(nr_plane_),
    max_data_term(DEFAULT_MAX_DATA_TERM), data_weight(DEFAULT_DATA_WEIGHT),
    max_disc_term(DEFAULT_MAX_DISC_TERM), disc_single_jump(DEFAULT_DISC_SINGLE_JUMP), min_disp_th(0),
    msg_type(msg_type_), use_local_init_data_cost(true)
{
    CV_Assert(msg_type_ == CV_32F || msg_type_ == CV_16S);
}


cv::ocl::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp_, int iters_, int levels_, int nr_plane_,
    float max_data_term_, float data_weight_, float max_disc_term_, float disc_single_jump_,
    int min_disp_th_, int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_), nr_plane(nr_plane_),
    max_data_term(max_data_term_), data_weight(data_weight_),
    max_disc_term(max_disc_term_), disc_single_jump(disc_single_jump_), min_disp_th(min_disp_th_),
    msg_type(msg_type_), use_local_init_data_cost(true)
{
    CV_Assert(msg_type_ == CV_32F || msg_type_ == CV_16S);
}

template<class T>
static void csbp_operator(StereoConstantSpaceBP &rthis, oclMat u[2], oclMat d[2], oclMat l[2], oclMat r[2],
    oclMat disp_selected_pyr[2], oclMat &data_cost, oclMat &data_cost_selected,
    oclMat &temp, oclMat &out, const oclMat &left, const oclMat &right, oclMat &disp)
{
    CV_DbgAssert(0 < rthis.ndisp && 0 < rthis.iters && 0 < rthis.levels && 0 < rthis.nr_plane
        && left.rows == right.rows && left.cols == right.cols && left.type() == right.type());

    CV_Assert(rthis.levels <= 8 && (left.type() == CV_8UC1 || left.type() == CV_8UC3));

    const Scalar zero = Scalar::all(0);

    ////////////////////////////////////Init///////////////////////////////////////////////////
    int rows = left.rows;
    int cols = left.cols;

    rthis.levels = min(rthis.levels, int(log((double)rthis.ndisp) / log(2.0)));
    int levels = rthis.levels;

    AutoBuffer<int> buf(levels * 4);

    int *cols_pyr = buf;
    int *rows_pyr = cols_pyr + levels;
    int *nr_plane_pyr = rows_pyr + levels;
    int *step_pyr = nr_plane_pyr + levels;

    cols_pyr[0] = cols;
    rows_pyr[0] = rows;
    nr_plane_pyr[0] = rthis.nr_plane;

    const int n = 64;
    step_pyr[0] = alignSize(cols * sizeof(T), n) / sizeof(T);
    for (int i = 1; i < levels; i++)
    {
        cols_pyr[i] = cols_pyr[i - 1]  / 2;
        rows_pyr[i] = rows_pyr[i - 1]/ 2;

        nr_plane_pyr[i] = nr_plane_pyr[i - 1] * 2;

        step_pyr[i] = alignSize(cols_pyr[i] * sizeof(T), n) / sizeof(T);
    }

    Size msg_size(step_pyr[0], rows * nr_plane_pyr[0]);
    Size data_cost_size(step_pyr[0], rows * nr_plane_pyr[0] * 2);

    u[0].create(msg_size, DataType<T>::type);
    d[0].create(msg_size, DataType<T>::type);
    l[0].create(msg_size, DataType<T>::type);
    r[0].create(msg_size, DataType<T>::type);

    u[1].create(msg_size, DataType<T>::type);
    d[1].create(msg_size, DataType<T>::type);
    l[1].create(msg_size, DataType<T>::type);
    r[1].create(msg_size, DataType<T>::type);

    disp_selected_pyr[0].create(msg_size, DataType<T>::type);
    disp_selected_pyr[1].create(msg_size, DataType<T>::type);

    data_cost.create(data_cost_size, DataType<T>::type);
    data_cost_selected.create(msg_size, DataType<T>::type);

    Size temp_size = data_cost_size;
    if (data_cost_size.width * data_cost_size.height < step_pyr[0] * rows_pyr[levels - 1] * rthis.ndisp)
        temp_size = Size(step_pyr[0], rows_pyr[levels - 1] * rthis.ndisp);

    temp.create(temp_size, DataType<T>::type);
    temp = zero;

    ///////////////////////////////// Compute////////////////////////////////////////////////

    //csbp::load_constants(rthis.ndisp, rthis.max_data_term, rthis.data_weight,
    //   rthis.max_disc_term, rthis.disc_single_jump, rthis.min_disp_th, left, right, temp);

    l[0] = zero;
    d[0] = zero;
    r[0] = zero;
    u[0] = zero;
    disp_selected_pyr[0] = zero;

    l[1] = zero;
    d[1] = zero;
    r[1] = zero;
    u[1] = zero;
    disp_selected_pyr[1] = zero;

    data_cost = zero;

    data_cost_selected = zero;

    int cur_idx = 0;

    for (int i = levels - 1; i >= 0; i--)
    {
        if (i == levels - 1)
        {
            cv::ocl::stereoCSBP::init_data_cost(left, right, temp, rthis, disp_selected_pyr[cur_idx].data,
                data_cost_selected.data, step_pyr[0], rows_pyr[i], cols_pyr[i],
                i, nr_plane_pyr[i]);
        }
        else
        {
            cv::ocl::stereoCSBP::compute_data_cost(
                disp_selected_pyr[cur_idx].data, data_cost.data, rthis, step_pyr[0],
                step_pyr[0], left, right, rows_pyr[i], cols_pyr[i], rows_pyr[i + 1], i,
                nr_plane_pyr[i + 1]);

            int new_idx = (cur_idx + 1) & 1;

            cv::ocl::stereoCSBP::init_message(u[new_idx].data, d[new_idx].data, l[new_idx].data, r[new_idx].data,
                u[cur_idx].data, d[cur_idx].data, l[cur_idx].data, r[cur_idx].data,
                disp_selected_pyr[new_idx].data, disp_selected_pyr[cur_idx].data,
                data_cost_selected.data, data_cost.data, temp, rthis, step_pyr[0],
                step_pyr[0], rows_pyr[i], cols_pyr[i], nr_plane_pyr[i], rows_pyr[i + 1],
                cols_pyr[i + 1], nr_plane_pyr[i + 1]);
            cur_idx = new_idx;
        }
        cv::ocl::stereoCSBP::calc_all_iterations(u[cur_idx].data, d[cur_idx].data, l[cur_idx].data, r[cur_idx].data,
            data_cost_selected.data, disp_selected_pyr[cur_idx].data, temp,
            rthis, step_pyr[0], rows_pyr[i], cols_pyr[i], nr_plane_pyr[i]);
    }

    if (disp.empty())
        disp.create(rows, cols, CV_16S);

    out = ((disp.type() == CV_16S) ? disp : (out.create(rows, cols, CV_16S), out));
    out = zero;

    stereoCSBP::compute_disp(u[cur_idx].data, d[cur_idx].data, l[cur_idx].data, r[cur_idx].data,
        data_cost_selected.data, disp_selected_pyr[cur_idx].data, rthis, step_pyr[0],
        out, nr_plane_pyr[0]);
    if (disp.type() != CV_16S)
        out.convertTo(disp, disp.type());
}


typedef void (*csbp_operator_t)(StereoConstantSpaceBP &rthis, oclMat u[2], oclMat d[2], oclMat l[2], oclMat r[2],
    oclMat disp_selected_pyr[2], oclMat &data_cost, oclMat &data_cost_selected,
    oclMat &temp, oclMat &out, const oclMat &left, const oclMat &right, oclMat &disp);

const static csbp_operator_t operators[] = {0, 0, 0, csbp_operator<short>, 0, csbp_operator<float>, 0, 0};

void cv::ocl::StereoConstantSpaceBP::operator()(const oclMat &left, const oclMat &right, oclMat &disp)
{

    CV_Assert(msg_type == CV_32F || msg_type == CV_16S);
    operators[msg_type](*this, u, d, l, r, disp_selected_pyr, data_cost, data_cost_selected, temp, out,
        left, right, disp);
}
