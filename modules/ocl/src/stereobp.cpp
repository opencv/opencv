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
//    Peng Xiao,   pengxiao@outlook.com
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

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
            static void load_constants(int ndisp, float max_data_term, float data_weight,
                                float max_disc_term, float disc_single_jump)
            {
                con_struct_t *con_struct = new con_struct_t;
                con_struct -> cndisp            = ndisp;
                con_struct -> cmax_data_term    = max_data_term;
                con_struct -> cdata_weight      = data_weight;
                con_struct -> cmax_disc_term    = max_disc_term;
                con_struct -> cdisc_single_jump = disc_single_jump;

                Context* clCtx = Context::getContext();
                cl_context clContext = *(cl_context*)(clCtx->getOpenCLContextPtr());
                cl_command_queue clCmdQueue = *(cl_command_queue*)(clCtx->getOpenCLCommandQueuePtr());
                cl_con_struct = load_constant(clContext, clCmdQueue, (void *)con_struct,
                                              sizeof(con_struct_t));

                delete con_struct;
            }
            static void release_constants()
            {
                openCLFree(cl_con_struct);
            }

            /////////////////////////////////////////////////////////////////////////////
            ///////////////////////////comp data////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////
            static void  comp_data_call(const oclMat &left, const oclMat &right, oclMat &data, int /*disp*/,
                float /*cmax_data_term*/, float /*cdata_weight*/)
            {
                Context  *clCxt = left.clCxt;
                int channels = left.oclchannels();
                int data_type = data.type();

                String kernelName = "comp_data";

                std::vector<std::pair<size_t , const void *> > args;

                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&left.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&left.rows));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&left.cols));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&left.step));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&right.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&right.step));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&data.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&data.step));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&cl_con_struct));

                size_t gt[3] = {left.cols, left.rows, 1}, lt[3] = {16, 16, 1};

                const int OPT_SIZE = 50;
                char cn_opt [OPT_SIZE] = "";
                sprintf( cn_opt, "%s -D CN=%d",
                    (data_type == CV_16S ? "-D T_SHORT":"-D T_FLOAT"),
                    channels
                    );
                openCLExecuteKernel(clCxt, &stereobp, kernelName, gt, lt, args, -1, -1, cn_opt);
            }
            ///////////////////////////////////////////////////////////////////////////////////
            /////////////////////////data set down////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////
            static void data_step_down_call(int dst_cols, int dst_rows, int src_rows,
                const oclMat &src, oclMat &dst, int disp)
            {
                Context  *clCxt = src.clCxt;
                int data_type = src.type();

                String kernelName = "data_step_down";

                std::vector<std::pair<size_t , const void *> > args;

                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_rows));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_rows));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_cols));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.step));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.step));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&disp));

                size_t gt[3] = {dst_cols, dst_rows, 1}, lt[3] = {16, 16, 1};
                const char* t_opt  = data_type == CV_16S ? "-D T_SHORT":"-D T_FLOAT";
                openCLExecuteKernel(clCxt, &stereobp, kernelName, gt, lt, args, -1, -1, t_opt);
            }
            /////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////live up message////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////
            static void level_up_message_call(int dst_cols, int dst_rows, int src_rows,
                oclMat &src, oclMat &dst, int ndisp)
            {
                Context  *clCxt = src.clCxt;
                int data_type = src.type();

                String kernelName = "level_up_message";
                std::vector<std::pair<size_t , const void *> > args;

                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_rows));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.step));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_rows));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_cols));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.step));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&ndisp));

                size_t gt[3] = {dst_cols, dst_rows, 1}, lt[3] = {16, 16, 1};
                const char* t_opt  = data_type == CV_16S ? "-D T_SHORT":"-D T_FLOAT";
                openCLExecuteKernel(clCxt, &stereobp, kernelName, gt, lt, args, -1, -1, t_opt);
            }
            static void level_up_messages_calls(int dst_idx, int dst_cols, int dst_rows, int src_rows,
                                         oclMat *mus, oclMat *mds, oclMat *mls, oclMat *mrs,
                                         int ndisp)
            {
                int src_idx = (dst_idx + 1) & 1;

                level_up_message_call(dst_cols, dst_rows, src_rows,
                                      mus[src_idx], mus[dst_idx], ndisp);

                level_up_message_call(dst_cols, dst_rows, src_rows,
                                      mds[src_idx], mds[dst_idx], ndisp);

                level_up_message_call(dst_cols, dst_rows, src_rows,
                                      mls[src_idx], mls[dst_idx], ndisp);

                level_up_message_call(dst_cols, dst_rows, src_rows,
                                      mrs[src_idx], mrs[dst_idx], ndisp);
            }
            //////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////cals_all_iterations_call///////////////////////////
            /////////////////////////////////////////////////////////////////////////////////
            static void calc_all_iterations_call(int cols, int rows, oclMat &u, oclMat &d,
                oclMat &l, oclMat &r, oclMat &data,
                int t, int cndisp, float cmax_disc_term,
                float cdisc_single_jump)
            {
                Context  *clCxt = l.clCxt;
                int data_type = u.type();

                String kernelName = "one_iteration";

                std::vector<std::pair<size_t , const void *> > args;

                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&u.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&u.step));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&data.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&data.step));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&d.data));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&l.data));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&r.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&t));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cols));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&rows));
                args.push_back( std::make_pair( sizeof(cl_float) , (void *)&cmax_disc_term));
                args.push_back( std::make_pair( sizeof(cl_float) , (void *)&cdisc_single_jump));

                size_t gt[3] = {cols, rows, 1}, lt[3] = {16, 16, 1};
                char opt[80] = "";
                sprintf(opt, "-D %s -D CNDISP=%d", data_type == CV_16S ? "T_SHORT":"T_FLOAT", cndisp);
                openCLExecuteKernel(clCxt, &stereobp, kernelName, gt, lt, args, -1, -1, opt);
            }

            static void calc_all_iterations_calls(int cols, int rows, int iters, oclMat &u,
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
            static void output_call(const oclMat &u, const oclMat &d, const oclMat l, const oclMat &r,
                const oclMat &data, oclMat &disp, int ndisp)
            {
                Context  *clCxt = u.clCxt;
                int data_type = u.type();

                String kernelName = "output";

                std::vector<std::pair<size_t , const void *> > args;

                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&u.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&u.step));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&d.data));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&l.data));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&r.data));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&data.data));
                args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&disp.data));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&disp.rows));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&disp.cols));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&disp.step));
                args.push_back( std::make_pair( sizeof(cl_int) , (void *)&ndisp));

                size_t gt[3] = {disp.cols, disp.rows, 1}, lt[3] = {16, 16, 1};
                const char* t_opt  = data_type == CV_16S ? "-D T_SHORT":"-D T_FLOAT";
                openCLExecuteKernel(clCxt, &stereobp, kernelName, gt, lt, args, -1, -1, t_opt);
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
                                    std::vector<oclMat> &datas_, oclMat &out_)
            : rthis(rthis_), u(u_), d(d_), l(l_), r(r_), u2(u2_), d2(d2_), l2(l2_), r2(r2_), datas(datas_), out(out_),
              zero(Scalar::all(0)), scale(rthis_.msg_type == CV_32F ? 1.0f : 10.0f)
        {
            CV_Assert(0 < rthis.ndisp && 0 < rthis.iters && 0 < rthis.levels);
            CV_Assert(rthis.msg_type == CV_32F || rthis.msg_type == CV_16S);
            CV_Assert(rthis.msg_type == CV_32F || (1 << (rthis.levels - 1)) * scale * rthis.max_data_term < std::numeric_limits<short>::max());
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

            cv::ocl::stereoBP::load_constants(rthis.ndisp, rthis.max_data_term, scale * rthis.data_weight,
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
                cols_all[i] = (cols_all[i - 1] + 1) / 2;
                rows_all[i] = (rows_all[i - 1] + 1) / 2;

                datas[i].create(rows_all[i] * rthis.ndisp, cols_all[i], rthis.msg_type);
                datas[i].setTo(Scalar_<short>::all(0));

                data_step_down_call(cols_all[i], rows_all[i], rows_all[i - 1],
                                    datas[i - 1], datas[i], rthis.ndisp);
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
                    level_up_messages_calls(mem_idx, cols_all[i], rows_all[i], rows_all[i + 1],
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
        StereoBeliefPropagationImpl& operator=(const StereoBeliefPropagationImpl&);

        StereoBeliefPropagation &rthis;

        oclMat &u;
        oclMat &d;
        oclMat &l;
        oclMat &r;

        oclMat &u2;
        oclMat &d2;
        oclMat &l2;
        oclMat &r2;

        std::vector<oclMat> &datas;
        oclMat &out;

        const Scalar zero;
        const float scale;

        int rows, cols;

        std::vector<int> cols_all, rows_all;
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
