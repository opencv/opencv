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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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
#include <vector>

using namespace cv;
using namespace cv::ocl;
using namespace std;

////////////////////////////////////////////////////////////////////////
///////////////// stereoBP /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

#if !defined (HAVE_OPENCL)

namespace cv
{
    namespace ocl
    {
        void cv::ocl::StereoBeliefPropagation::estimateRecommendedParams(int, int, int &, int &, int &)
        {
            throw_nogpu();
        }

        cv::ocl::StereoBeliefPropagation::StereoBeliefPropagation(int, int, int, int)
        {
            throw_nogpu();
        }

        cv::ocl::StereoBeliefPropagation::StereoBeliefPropagation(int, int, int, float, float, float, float, int)
        {
            throw_nogpu();
        }

        void cv::ocl::StereoBeliefPropagation::operator()(const oclMat &, const oclMat &, oclMat &)
        {
            throw_nogpu();
        }

        void cv::ocl::StereoBeliefPropagation::operator()(const oclMat &, oclMat &)
        {
            throw_nogpu();
        }
    }
}

#else /* !defined (HAVE_OPENCL) */

namespace cv
{
    namespace ocl
    {

        ///////////////////////////OpenCL kernel strings///////////////////////////
        extern const char *stereobp;
    }

}
namespace cv
{
    namespace ocl
    {
        namespace stereoBP
        {
            //////////////////////////////////////////////////////////////////////////
            //////////////////////////////common////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////
            typedef struct
            {
                int   cndisp;
                float cmax_data_term;
                float cdata_weight;
                float cmax_disc_term;
                float cdisc_single_jump;
            } con_struct_t;

            cl_mem cl_con_struct =  NULL;
            void load_constants(Context *clCxt, int ndisp, float max_data_term, float data_weight,
                                float max_disc_term, float disc_single_jump)
            {
                con_struct_t *con_struct = new con_struct_t;
                con_struct -> cndisp            = ndisp;
                con_struct -> cmax_data_term    = max_data_term;
                con_struct -> cdata_weight      = data_weight;
                con_struct -> cmax_disc_term    = max_data_term;
                con_struct -> cdisc_single_jump = disc_single_jump;

                cl_con_struct = load_constant(clCxt->impl->clContext, clCxt->impl->clCmdQueue, (void *)con_struct,
                                              sizeof(con_struct_t));

                delete con_struct;
            }
            void release_constants()
            {
                openCLFree(cl_con_struct);
            }
            static inline int divUp(int total, int grain)
            {
                return (total + grain - 1) / grain;
            }
            /////////////////////////////////////////////////////////////////////////////
            ///////////////////////////comp data////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////
            void  comp_data_call(const oclMat &left, const oclMat &right, oclMat &data, int disp,                                float cmax_data_term, float cdata_weight)
            {
                Context  *clCxt = left.clCxt;
                int channels = left.channels();
                int data_type = data.type();

                string kernelName = "comp_data_";
                stringstream idxStr;
                if(data_type == CV_16S)
                    idxStr << "0";
                else
                    idxStr << "1";
                kernelName += idxStr.str();

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobp, kernelName);

                size_t blockSize = 32;
                size_t localThreads[]  = {32, 8};
                size_t globalThreads[] = {divUp(left.cols, localThreads[0]) * localThreads[0],
                                          divUp(left.rows, localThreads[1]) * localThreads[1]
                                         };

                openCLVerifyKernel(clCxt, kernel, &blockSize, globalThreads, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&left.data));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&left.rows));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&left.cols));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&left.step));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&right.data));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&right.step));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&data.data));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&data.cols));
                openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&data.step));
                openCLSafeCall(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&cl_con_struct));
                //openCLSafeCall(clSetKernelArg(kernel,12,sizeof(cl_int),(void*)&disp));
                //openCLSafeCall(clSetKernelArg(kernel,13,sizeof(cl_float),(void*)&cmax_data_term));
                //openCLSafeCall(clSetKernelArg(kernel,14,sizeof(cl_float),(void*)&cdata_weight));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&channels));

                openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 2, NULL,
                                                      globalThreads, localThreads, 0, NULL, NULL));

                clFinish(clCxt->impl->clCmdQueue);
                openCLSafeCall(clReleaseKernel(kernel));
            }
            ///////////////////////////////////////////////////////////////////////////////////
            /////////////////////////data set down////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////
            void data_step_down_call(int dst_cols, int dst_rows, int src_rows,
                                     const oclMat &src, oclMat &dst, int disp)
            {
                Context  *clCxt = src.clCxt;
                int data_type = src.type();

                string kernelName = "data_step_down_";
                stringstream idxStr;
                if(data_type == CV_16S)
                    idxStr << "0";
                else
                    idxStr << "1";
                kernelName += idxStr.str();

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobp, kernelName);

                size_t blockSize = 32;
                size_t localThreads[]  = {32, 8};
                size_t globalThreads[] = {divUp(dst_cols, localThreads[0]) * localThreads[0],
                                          divUp(dst_rows, localThreads[1]) * localThreads[1]
                                         };

                openCLVerifyKernel(clCxt, kernel, &blockSize, globalThreads, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.data));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&src_rows));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&src.cols));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dst.data));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&dst_rows));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&dst_cols));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&dst.cols));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&disp));

                openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 2, NULL,
                                                      globalThreads, localThreads, 0, NULL, NULL));

                clFinish(clCxt->impl->clCmdQueue);
                openCLSafeCall(clReleaseKernel(kernel));
            }
            /////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////live up message////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////
            void level_up_message_call(int dst_idx, int dst_cols, int dst_rows, int src_rows,
                                       oclMat &src, oclMat &dst, int ndisp)
            {
                Context  *clCxt = src.clCxt;
                int data_type = src.type();

                string kernelName = "level_up_message_";
                stringstream idxStr;
                if(data_type == CV_16S)
                    idxStr << "0";
                else
                    idxStr << "1";
                kernelName += idxStr.str();

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobp, kernelName);

                size_t blockSize = 32;
                size_t localThreads[]  = {32, 8};
                size_t globalThreads[] = {divUp(dst_cols, localThreads[0]) * localThreads[0],
                                          divUp(dst_rows, localThreads[1]) * localThreads[1]
                                         };

                openCLVerifyKernel(clCxt, kernel, &blockSize, globalThreads, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.data));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&src_rows));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&src.step));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dst.data));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&dst_rows));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&dst_cols));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&dst.step));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&ndisp));

                openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 2, NULL,
                                                      globalThreads, localThreads, 0, NULL, NULL));

                clFinish(clCxt->impl->clCmdQueue);
                openCLSafeCall(clReleaseKernel(kernel));
            }
            void level_up_messages_calls(int dst_idx, int dst_cols, int dst_rows, int src_rows,
                                         oclMat *mus, oclMat *mds, oclMat *mls, oclMat *mrs,
                                         int ndisp)
            {
                int src_idx = (dst_idx + 1) & 1;

                level_up_message_call(dst_idx, dst_cols, dst_rows, src_rows,
                                      mus[src_idx], mus[dst_idx], ndisp);

                level_up_message_call(dst_idx, dst_cols, dst_rows, src_rows,
                                      mds[src_idx], mds[dst_idx], ndisp);

                level_up_message_call(dst_idx, dst_cols, dst_rows, src_rows,
                                      mls[src_idx], mls[dst_idx], ndisp);

                level_up_message_call(dst_idx, dst_cols, dst_rows, src_rows,
                                      mrs[src_idx], mrs[dst_idx], ndisp);
            }
            //////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////cals_all_iterations_call///////////////////////////
            /////////////////////////////////////////////////////////////////////////////////
            void calc_all_iterations_call(int cols, int rows, oclMat &u, oclMat &d,
                                          oclMat &l, oclMat &r, oclMat &data,
                                          int t, int cndisp, float cmax_disc_term,
                                          float cdisc_single_jump)
            {
                Context  *clCxt = l.clCxt;
                int data_type = u.type();

                string kernelName = "one_iteration_";
                stringstream idxStr;
                if(data_type == CV_16S)
                    idxStr << "0";
                else
                    idxStr << "1";
                kernelName += idxStr.str();

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobp, kernelName);

                size_t blockSize = 32;
                size_t localThreads[]  = {32, 8};
                size_t globalThreads[] = {divUp(cols, (localThreads[0] << 1)) * (localThreads[0] << 1),
                                          divUp(rows, localThreads[1]) * localThreads[1]
                                         };

                openCLVerifyKernel(clCxt, kernel, &blockSize, globalThreads, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&u.data));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&u.step));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&u.cols));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&data.data));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&data.step));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&data.cols));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&d.data));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&l.data));
                openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&r.data));
                openCLSafeCall(clSetKernelArg(kernel, 9, sizeof(cl_int), (void *)&t));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&cols));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_int), (void *)&rows));
                openCLSafeCall(clSetKernelArg(kernel, 12, sizeof(cl_int), (void *)&cndisp));
                openCLSafeCall(clSetKernelArg(kernel, 13, sizeof(cl_float), (void *)&cmax_disc_term));
                openCLSafeCall(clSetKernelArg(kernel, 14, sizeof(cl_float), (void *)&cdisc_single_jump));

                openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 2, NULL,
                                                      globalThreads, localThreads, 0, NULL, NULL));

                clFinish(clCxt->impl->clCmdQueue);
                openCLSafeCall(clReleaseKernel(kernel));
            }

            void calc_all_iterations_calls(int cols, int rows, int iters, oclMat &u,
                                           oclMat &d, oclMat &l, oclMat &r,
                                           oclMat &data, int cndisp, float cmax_disc_term,
                                           float cdisc_single_jump)
            {
                for(int t = 0; t < iters; ++t)
                    calc_all_iterations_call(cols, rows, u, d, l, r, data, t, cndisp,
                                             cmax_disc_term, cdisc_single_jump);
            }
            ///////////////////////////////////////////////////////////////////////////////
            ///////////////////////output///////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            void output_call(const oclMat &u, const oclMat &d, const oclMat l, const oclMat &r,
                             const oclMat &data, oclMat &disp, int ndisp)
            {
                Context  *clCxt = u.clCxt;
                int data_type = u.type();

                string kernelName = "output_";
                stringstream idxStr;
                if(data_type == CV_16S)
                    idxStr << "0";
                else
                    idxStr << "1";
                kernelName += idxStr.str();

                cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobp, kernelName);

                size_t blockSize = 32;
                size_t localThreads[]  = {32, 8};
                size_t globalThreads[] = {divUp(disp.cols, localThreads[0]) * localThreads[0],
                                          divUp(disp.rows, localThreads[1]) * localThreads[1]
                                         };

                openCLVerifyKernel(clCxt, kernel, &blockSize, globalThreads, localThreads);
                openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&u.data));
                openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&u.step));
                openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&u.cols));
                openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d.data));
                openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&l.data));
                openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&r.data));
                openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&data.data));
                openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&disp.data));
                openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&disp.rows));
                openCLSafeCall(clSetKernelArg(kernel, 9, sizeof(cl_int), (void *)&disp.cols));
                openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&disp.step));
                openCLSafeCall(clSetKernelArg(kernel, 11, sizeof(cl_int), (void *)&ndisp));

                openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 2, NULL,
                                                      globalThreads, localThreads, 0, NULL, NULL));

                clFinish(clCxt->impl->clCmdQueue);
                openCLSafeCall(clReleaseKernel(kernel));

            }
        }
    }
}
namespace
{
    const float DEFAULT_MAX_DATA_TERM = 10.0f;
    const float DEFAULT_DATA_WEIGHT = 0.07f;
    const float DEFAULT_MAX_DISC_TERM = 1.7f;
    const float DEFAULT_DISC_SINGLE_JUMP = 1.0f;

    template<typename T>
    void print_gpu_mat(const oclMat &mat)
    {
        T *data_1 = new T[mat.rows * mat.cols * mat.channels()];
        Context  *clCxt = mat.clCxt;
        int status = clEnqueueReadBuffer(clCxt -> impl-> clCmdQueue, (cl_mem)mat.data, CL_TRUE, 0,
                                         mat.rows * mat.cols * mat.channels() * sizeof(T), data_1, 0, NULL, NULL);

        if(status != CL_SUCCESS)
            cout << "error " << status << endl;

        cout << ".........................................................." << endl;
        cout << "elemSize() " << mat.elemSize() << endl;
        cout << "elemSize() " << mat.elemSize1() << endl;
        cout << "channels: " << mat.channels() << endl;
        cout << "rows: " << mat.rows << endl;
        cout << "cols: " << mat.cols << endl;

        for(int i = 0; i < 30; i++)
        {
            for(int j = 0; j < 30; j++)
            {
                cout << (int)data_1[i * mat.cols * mat.channels() + j] << " ";
            }
            cout << endl;
        }
    }
}

void cv::ocl::StereoBeliefPropagation::estimateRecommendedParams(int width, int height, int &ndisp, int &iters, int &levels)
{
    ndisp = width / 4;
    if ((ndisp & 1) != 0)
        ndisp++;

    int mm = ::max(width, height);
    iters = mm / 100 + 2;

    levels = (int)(::log(static_cast<double>(mm)) + 1) * 4 / 5;
    if (levels == 0) levels++;
}

cv::ocl::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp_, int iters_, int levels_, int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_),
      max_data_term(DEFAULT_MAX_DATA_TERM), data_weight(DEFAULT_DATA_WEIGHT),
      max_disc_term(DEFAULT_MAX_DISC_TERM), disc_single_jump(DEFAULT_DISC_SINGLE_JUMP),
      msg_type(msg_type_), datas(levels_)
{
}

cv::ocl::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp_, int iters_, int levels_, float max_data_term_, float data_weight_, float max_disc_term_, float disc_single_jump_, int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_),
      max_data_term(max_data_term_), data_weight(data_weight_),
      max_disc_term(max_disc_term_), disc_single_jump(disc_single_jump_),
      msg_type(msg_type_), datas(levels_)
{
}

namespace
{
    class StereoBeliefPropagationImpl
    {
    public:
        StereoBeliefPropagationImpl(StereoBeliefPropagation &rthis_,
                                    oclMat &u_, oclMat &d_, oclMat &l_, oclMat &r_,
                                    oclMat &u2_, oclMat &d2_, oclMat &l2_, oclMat &r2_,
                                    vector<oclMat>& datas_, oclMat &out_)
            : rthis(rthis_), u(u_), d(d_), l(l_), r(r_), u2(u2_), d2(d2_), l2(l2_), r2(r2_), datas(datas_), out(out_),
              zero(Scalar::all(0)), scale(rthis_.msg_type == CV_32F ? 1.0f : 10.0f)
        {
            CV_Assert(0 < rthis.ndisp && 0 < rthis.iters && 0 < rthis.levels);
            CV_Assert(rthis.msg_type == CV_32F || rthis.msg_type == CV_16S);
            CV_Assert(rthis.msg_type == CV_32F || (1 << (rthis.levels - 1)) * scale * rthis.max_data_term < numeric_limits<short>::max());
        }

        void operator()(const oclMat &left, const oclMat &right, oclMat &disp)
        {
            CV_Assert(left.size() == right.size() && left.type() == right.type());
            CV_Assert(left.type() == CV_8UC1 || left.type() == CV_8UC3 || left.type() == CV_8UC4);

            rows = left.rows;
            cols = left.cols;

            int divisor = (int)pow(2.f, rthis.levels - 1.0f);
            int lowest_cols = cols / divisor;
            int lowest_rows = rows / divisor;
            const int min_image_dim_size = 2;
            CV_Assert(min(lowest_cols, lowest_rows) > min_image_dim_size);

            init();

            datas[0].create(rows * rthis.ndisp, cols, rthis.msg_type);
            datas[0].setTo(Scalar_<short>::all(0));

            cv::ocl::stereoBP::comp_data_call(left, right, datas[0], rthis.ndisp, rthis.max_data_term, scale * rthis.data_weight);

            calcBP(disp);
        }

        void operator()(const oclMat &data, oclMat &disp)
        {
            CV_Assert((data.type() == rthis.msg_type) && (data.rows % rthis.ndisp == 0));

            rows = data.rows / rthis.ndisp;
            cols = data.cols;

            int divisor = (int)pow(2.f, rthis.levels - 1.0f);
            int lowest_cols = cols / divisor;
            int lowest_rows = rows / divisor;
            const int min_image_dim_size = 2;
            CV_Assert(min(lowest_cols, lowest_rows) > min_image_dim_size);

            init();

            datas[0] = data;

            calcBP(disp);
        }
    private:
        void init()
        {
            u.create(rows * rthis.ndisp, cols, rthis.msg_type);
            d.create(rows * rthis.ndisp, cols, rthis.msg_type);
            l.create(rows * rthis.ndisp, cols, rthis.msg_type);
            r.create(rows * rthis.ndisp, cols, rthis.msg_type);

            if (rthis.levels & 1)
            {
                //can clear less area
                u = zero;
                d = zero;
                l = zero;
                r = zero;
            }

            if (rthis.levels > 1)
            {
                int less_rows = (rows + 1) / 2;
                int less_cols = (cols + 1) / 2;

                u2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);
                d2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);
                l2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);
                r2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);

                if ((rthis.levels & 1) == 0)
                {
                    u2 = zero;
                    d2 = zero;
                    l2 = zero;
                    r2 = zero;
                }
            }

            cv::ocl::stereoBP::load_constants(u.clCxt, rthis.ndisp, rthis.max_data_term, scale * rthis.data_weight,
                                              scale * rthis.max_disc_term, scale * rthis.disc_single_jump);

            datas.resize(rthis.levels);

            cols_all.resize(rthis.levels);
            rows_all.resize(rthis.levels);

            cols_all[0] = cols;
            rows_all[0] = rows;
        }

        void calcBP(oclMat &disp)
        {
            using namespace cv::ocl::stereoBP;

            for (int i = 1; i < rthis.levels; ++i)
            {
                cols_all[i] = (cols_all[i-1] + 1) / 2;
                rows_all[i] = (rows_all[i-1] + 1) / 2;

                datas[i].create(rows_all[i] * rthis.ndisp, cols_all[i], rthis.msg_type);
                datas[i].setTo(Scalar_<short>::all(0));

                data_step_down_call(cols_all[i], rows_all[i], rows_all[i-1],
                                    datas[i-1], datas[i], rthis.ndisp);
            }

            oclMat mus[] = {u, u2};
            oclMat mds[] = {d, d2};
            oclMat mrs[] = {r, r2};
            oclMat mls[] = {l, l2};

            int mem_idx = (rthis.levels & 1) ? 0 : 1;

            for (int i = rthis.levels - 1; i >= 0; --i)
            {
                // for lower level we have already computed messages by setting to zero
                if (i != rthis.levels - 1)
                    level_up_messages_calls(mem_idx, cols_all[i], rows_all[i], rows_all[i+1],
                                            mus, mds, mls, mrs, rthis.ndisp);

                calc_all_iterations_calls(cols_all[i], rows_all[i], rthis.iters, mus[mem_idx],
                                          mds[mem_idx], mls[mem_idx], mrs[mem_idx], datas[i],
                                          rthis.ndisp, scale * rthis.max_disc_term,
                                          scale * rthis.disc_single_jump);

                mem_idx = (mem_idx + 1) & 1;
            }

            if (disp.empty())
                disp.create(rows, cols, CV_16S);

            out = ((disp.type() == CV_16S) ? disp : (out.create(rows, cols, CV_16S), out));
            out = zero;

            output_call(u, d, l, r, datas.front(), out, rthis.ndisp);


            if (disp.type() != CV_16S)
                out.convertTo(disp, disp.type());

            release_constants();
        }

        StereoBeliefPropagation &rthis;

        oclMat &u;
        oclMat &d;
        oclMat &l;
        oclMat &r;

        oclMat &u2;
        oclMat &d2;
        oclMat &l2;
        oclMat &r2;

        vector<oclMat>& datas;
        oclMat &out;

        const Scalar zero;
        const float scale;

        int rows, cols;

        vector<int> cols_all, rows_all;
    };
}

void cv::ocl::StereoBeliefPropagation::operator()(const oclMat &left, const oclMat &right, oclMat &disp)
{
    ::StereoBeliefPropagationImpl impl(*this, u, d, l, r, u2, d2, l2, r2, datas, out);
    impl(left, right, disp);
}

void cv::ocl::StereoBeliefPropagation::operator()(const oclMat &data, oclMat &disp)
{
    ::StereoBeliefPropagationImpl impl(*this, u, d, l, r, u2, d2, l2, r2, datas, out);
    impl(data, disp);
}
#endif /* !defined (HAVE_OPENCL) */
