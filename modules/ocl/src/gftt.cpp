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
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@outlook.com
//
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
// This software is provided by the copyright holders and contributors as is and
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

// currently sort procedure on the host is more efficient
static bool use_cpu_sorter = true;

// compact structure for corners
struct DefCorner
{
    float eig;  //eigenvalue of corner
    short x;    //x coordinate of corner point
    short y;    //y coordinate of corner point
} ;

// compare procedure for corner
//it is used for sort on the host side
struct DefCornerCompare
{
    bool operator()(const DefCorner a, const DefCorner b) const
    {
        return a.eig > b.eig;
    }
};

// sort corner point using opencl bitonicosrt implementation
static void sortCorners_caller(oclMat& corners, const int count)
{
    Context * cxt = Context::getContext();
    int     GS = count/2;
    int     LS = min(255,GS);
    size_t  globalThreads[3] = {(size_t)GS, 1, 1};
    size_t  localThreads[3]  = {(size_t)LS, 1, 1};

    // 2^numStages should be equal to count or the output is invalid
    int numStages = 0;
    for(int i = count; i > 1; i >>= 1)
    {
        ++numStages;
    }
    const int argc = 4;
    std::vector< std::pair<size_t, const void *> > args(argc);
    std::string kernelname = "sortCorners_bitonicSort";
    args[0] = std::make_pair(sizeof(cl_mem), (void *)&corners.data);
    args[1] = std::make_pair(sizeof(cl_int), (void *)&count);
    for(int stage = 0; stage < numStages; ++stage)
    {
        args[2] = std::make_pair(sizeof(cl_int), (void *)&stage);
        for(int passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
        {
            args[3] = std::make_pair(sizeof(cl_int), (void *)&passOfStage);
            openCLExecuteKernel(cxt, &imgproc_gftt, kernelname, globalThreads, localThreads, args, -1, -1);
        }
    }
}

// find corners on matrix and put it into array
static void findCorners_caller(
    const oclMat&   eig_mat,        //input matrix worth eigenvalues
    oclMat&         eigMinMax,      //input with min and max values of eigenvalues
    const float     qualityLevel,
    const oclMat&   mask,
    oclMat&         corners,        //output array with detected corners
    oclMat&         counter)        //output value with number of detected corners, have to be 0 before call
{
    string  opt;
    Context * cxt = Context::getContext();

    std::vector< std::pair<size_t, const void*> > args;

    const int mask_strip = mask.step / mask.elemSize1();

    args.push_back(make_pair( sizeof(cl_mem),   (void*)&(eig_mat.data)));

    int src_pitch = (int)eig_mat.step;
    args.push_back(make_pair( sizeof(cl_int),   (void*)&src_pitch ));
    args.push_back(make_pair( sizeof(cl_mem),   (void*)&mask.data ));
    args.push_back(make_pair( sizeof(cl_mem),   (void*)&corners.data ));
    args.push_back(make_pair( sizeof(cl_int),   (void*)&mask_strip));
    args.push_back(make_pair( sizeof(cl_mem),   (void*)&eigMinMax.data ));
    args.push_back(make_pair( sizeof(cl_float), (void*)&qualityLevel ));
    args.push_back(make_pair( sizeof(cl_int),   (void*)&eig_mat.rows ));
    args.push_back(make_pair( sizeof(cl_int),   (void*)&eig_mat.cols ));
    args.push_back(make_pair( sizeof(cl_int),   (void*)&corners.cols ));
    args.push_back(make_pair( sizeof(cl_mem),   (void*)&counter.data ));

    size_t globalThreads[3] = {(size_t)eig_mat.cols, (size_t)eig_mat.rows, 1};
    size_t localThreads[3]  = {16, 16, 1};
    if(!mask.empty())
        opt += " -D WITH_MASK=1";

     openCLExecuteKernel(cxt, &imgproc_gftt, "findCorners", globalThreads, localThreads, args, -1, -1, opt.c_str());
}


static void minMaxEig_caller(const oclMat &src, oclMat &dst, oclMat & tozero)
{
    size_t groupnum = src.clCxt->getDeviceInfo().maxComputeUnits;
    CV_Assert(groupnum != 0);

    int dbsize = groupnum * 2 * src.elemSize();
    ensureSizeIsEnough(1, dbsize, CV_8UC1, dst);

    cl_mem dst_data = reinterpret_cast<cl_mem>(dst.data);

    int vElemSize = src.elemSize1();
    int src_step = src.step / vElemSize, src_offset = src.offset / vElemSize;
    int total = src.size().area();

    {
        // first parallel pass
        vector<pair<size_t , const void *> > args;
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&src_step));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&src_offset));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&src.rows ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&src.cols ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&total));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&groupnum));
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst_data ));
        size_t globalThreads[3] = {(size_t)groupnum * 256, 1, 1};
        size_t localThreads[3] = {256, 1, 1};
        openCLExecuteKernel(src.clCxt, &arithm_minMax, "arithm_op_minMax", globalThreads, localThreads,
                            args, -1, -1, "-D T=float -D DEPTH_5 -D vlen=1");
    }

    {
        // run final "serial" kernel to find accumulate results from threads and reset corner counter
        vector<pair<size_t , const void *> > args;
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst_data ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&groupnum ));
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&tozero.data ));
        size_t globalThreads[3] = {1, 1, 1};
        size_t localThreads[3] = {1, 1, 1};
        openCLExecuteKernel(src.clCxt, &imgproc_gftt, "arithm_op_minMax_final", globalThreads, localThreads,
                            args, -1, -1);
    }
}

void cv::ocl::GoodFeaturesToTrackDetector_OCL::operator ()(const oclMat& image, oclMat& corners, const oclMat& mask)
{
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()));

    ensureSizeIsEnough(image.size(), CV_32F, eig_);

    if (useHarrisDetector)
        cornerHarris_dxdy(image, eig_, Dx_, Dy_, blockSize, 3, harrisK);
    else
        cornerMinEigenVal_dxdy(image, eig_, Dx_, Dy_, blockSize, 3);

    ensureSizeIsEnough(1,1, CV_32SC1, counter_);

    // find max eigenvalue and reset detected counters
    minMaxEig_caller(eig_,eig_minmax_,counter_);

    // allocate buffer for kernels
    int corner_array_size = std::max(1024, static_cast<int>(image.size().area() * 0.05));

    if(!use_cpu_sorter)
    {   // round to 2^n
        unsigned int n=1;
        for(n=1;n<(unsigned int)corner_array_size;n<<=1) ;
        corner_array_size = (int)n;

        ensureSizeIsEnough(1, corner_array_size , CV_32FC2, tmpCorners_);

        // set to 0 to be able use bitonic sort on whole 2^n array
        tmpCorners_.setTo(0);
    }
    else
    {
        ensureSizeIsEnough(1, corner_array_size , CV_32FC2, tmpCorners_);
    }

    int total = tmpCorners_.cols; // by default the number of corner is full array
    vector<DefCorner>   tmp(tmpCorners_.cols); // input buffer with corner for HOST part of algorithm

    //find points with high eigenvalue and put it into the output array
    findCorners_caller(
        eig_,
        eig_minmax_,
        static_cast<float>(qualityLevel),
        mask,
        tmpCorners_,
        counter_);

    if(!use_cpu_sorter)
    {// sort detected corners on deivce side
        sortCorners_caller(tmpCorners_, corner_array_size);
    }
    else
    {// send non-blocking request to read real non-zero number of corners to sort it on the HOST side
        openCLVerifyCall(clEnqueueReadBuffer(getClCommandQueue(counter_.clCxt), (cl_mem)counter_.data, CL_FALSE, 0,sizeof(int), &total, 0, NULL, NULL));
    }

    //blocking read whole corners array (sorted or not sorted)
    openCLReadBuffer(tmpCorners_.clCxt,(cl_mem)tmpCorners_.data,&tmp[0],tmpCorners_.cols*sizeof(DefCorner));

    if (total == 0)
    {// check for trivial case
        corners.release();
        return;
    }

    if(use_cpu_sorter)
    {// sort detected corners on cpu side.
        tmp.resize(total);
        cv::sort(tmp,DefCornerCompare());
    }

    //estimate maximal size of final output array
    int total_max = maxCorners > 0 ? std::min(maxCorners, total) : total;
    int D2 = (int)ceil(minDistance * minDistance);
    // allocate output buffer
    vector<Point2f> tmp2;
    tmp2.reserve(total_max);


    if (minDistance < 1)
    {// we have not distance restriction. then just copy with conversion maximal allowed points into output array
        for(int i=0;i<total_max && tmp[i].eig>0.0f;++i)
        {
            tmp2.push_back(Point2f(tmp[i].x,tmp[i].y));
        }
    }
    else
    {// we have distance restriction. then start coping to output array from the first element and check distance for each next one
        const int cell_size = cvRound(minDistance);
        const int grid_width = (image.cols + cell_size - 1) / cell_size;
        const int grid_height = (image.rows + cell_size - 1) / cell_size;

        std::vector< std::vector<Point2i> > grid(grid_width * grid_height);

        for (int i = 0; i < total ; ++i)
        {
            DefCorner p = tmp[i];

            if(p.eig<=0.0f)
                break; // condition to stop that is needed for GPU bitonic sort usage.

            bool good = true;

            int x_cell = static_cast<int>(p.x / cell_size);
            int y_cell = static_cast<int>(p.y / cell_size);

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for (int yy = y1; yy <= y2; yy++)
            {
                for (int xx = x1; xx <= x2; xx++)
                {
                    vector<Point2i>& m = grid[yy * grid_width + xx];
                    if (m.empty())
                        continue;
                    for(size_t j = 0; j < m.size(); j++)
                    {
                        int dx = p.x - m[j].x;
                        int dy = p.y - m[j].y;

                        if (dx * dx + dy * dy < D2)
                        {
                            good = false;
                            goto break_out_;
                        }
                    }
                }
            }

            break_out_:

            if(good)
            {
                grid[y_cell * grid_width + x_cell].push_back(Point2i(p.x,p.y));

                tmp2.push_back(Point2f(p.x,p.y));

                if (maxCorners > 0 && tmp2.size() == static_cast<size_t>(maxCorners))
                    break;
            }
        }

    }
    int final_size = static_cast<int>(tmp2.size());
    if(final_size>0)
        corners.upload(Mat(1, final_size, CV_32FC2, &tmp2[0]));
    else
        corners.release();
}
void cv::ocl::GoodFeaturesToTrackDetector_OCL::downloadPoints(const oclMat &points, vector<Point2f> &points_v)
{
    CV_DbgAssert(points.type() == CV_32FC2);
    points_v.resize(points.cols);
    openCLSafeCall(clEnqueueReadBuffer(
        *(cl_command_queue*)getClCommandQueuePtr(),
        reinterpret_cast<cl_mem>(points.data),
        CL_TRUE,
        0,
        points.cols * sizeof(Point2f),
        &points_v[0],
        0,
        NULL,
        NULL));
}
