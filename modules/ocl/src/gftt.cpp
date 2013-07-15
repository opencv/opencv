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
//     and/or other oclMaterials provided with the distribution.
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
#include <iomanip>
#include "precomp.hpp"

using namespace cv;
using namespace cv::ocl;

static bool use_cpu_sorter = true;

namespace cv
{
    namespace ocl
    {
        ///////////////////////////OpenCL kernel strings///////////////////////////
        extern const char *imgproc_gftt;
    }
}

namespace
{
enum SortMethod
{
    CPU_STL,
    BITONIC,
    SELECTION
};

const int GROUP_SIZE = 256;

template<SortMethod method>
struct Sorter
{
    //typedef EigType;
};

//TODO(pengx): optimize GPU sorter's performance thus CPU sorter is removed.
template<>
struct Sorter<CPU_STL>
{
    typedef oclMat EigType;
    static cv::Mutex cs;
    static Mat mat_eig;

    //prototype
    static int clfloat2Gt(cl_float2 pt1, cl_float2 pt2)
    {
        float v1 = mat_eig.at<float>(cvRound(pt1.s[1]), cvRound(pt1.s[0]));
        float v2 = mat_eig.at<float>(cvRound(pt2.s[1]), cvRound(pt2.s[0]));
        return v1 > v2;
    }
    static void sortCorners_caller(const EigType& eig_tex, oclMat& corners, const int count)
    {
        cv::AutoLock lock(cs);
        //temporarily use STL's sort function
        Mat mat_corners = corners;
        mat_eig = eig_tex;
        std::sort(mat_corners.begin<cl_float2>(), mat_corners.begin<cl_float2>() + count, clfloat2Gt);
        corners = mat_corners;
    }
};
cv::Mutex Sorter<CPU_STL>::cs;
cv::Mat   Sorter<CPU_STL>::mat_eig;

template<>
struct Sorter<BITONIC>
{
    typedef TextureCL EigType;

    static void sortCorners_caller(const EigType& eig_tex, oclMat& corners, const int count)
    {
        Context * cxt = Context::getContext();
        size_t globalThreads[3] = {count / 2, 1, 1};
        size_t localThreads[3]  = {GROUP_SIZE, 1, 1};

        // 2^numStages should be equal to count or the output is invalid
        int numStages = 0;
        for(int i = count; i > 1; i >>= 1)
        {
            ++numStages;
        }
        const int argc = 5;
        std::vector< std::pair<size_t, const void *> > args(argc);
        String kernelname = "sortCorners_bitonicSort";
        args[0] = std::make_pair(sizeof(cl_mem), (void *)&eig_tex);
        args[1] = std::make_pair(sizeof(cl_mem), (void *)&corners.data);
        args[2] = std::make_pair(sizeof(cl_int), (void *)&count);
        for(int stage = 0; stage < numStages; ++stage)
        {
            args[3] = std::make_pair(sizeof(cl_int), (void *)&stage);
            for(int passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
            {
                args[4] = std::make_pair(sizeof(cl_int), (void *)&passOfStage);
                openCLExecuteKernel(cxt, &imgproc_gftt, kernelname, globalThreads, localThreads, args, -1, -1);
            }
        }
    }
};

template<>
struct Sorter<SELECTION>
{
    typedef TextureCL EigType;

    static void sortCorners_caller(const EigType& eig_tex, oclMat& corners, const int count)
    {
        Context * cxt = Context::getContext();

        size_t globalThreads[3] = {count, 1, 1};
        size_t localThreads[3]  = {GROUP_SIZE, 1, 1};

        std::vector< std::pair<size_t, const void *> > args;
        //local
        String kernelname = "sortCorners_selectionSortLocal";
        int lds_size = GROUP_SIZE * sizeof(cl_float2);
        args.push_back( std::make_pair( sizeof(cl_mem), (void*)&eig_tex) );
        args.push_back( std::make_pair( sizeof(cl_mem), (void*)&corners.data) );
        args.push_back( std::make_pair( sizeof(cl_int), (void*)&count) );
        args.push_back( std::make_pair( lds_size,       (void*)NULL) );

        openCLExecuteKernel(cxt, &imgproc_gftt, kernelname, globalThreads, localThreads, args, -1, -1);

        //final
        kernelname = "sortCorners_selectionSortFinal";
        args.pop_back();
        openCLExecuteKernel(cxt, &imgproc_gftt, kernelname, globalThreads, localThreads, args, -1, -1);
    }
};

int findCorners_caller(
    const TextureCL& eig,
    const float threshold,
    const oclMat& mask,
    oclMat& corners,
    const int max_count)
{
    std::vector<int> k;
    Context * cxt = Context::getContext();

    std::vector< std::pair<size_t, const void*> > args;
    String kernelname = "findCorners";

    const int mask_strip = mask.step / mask.elemSize1();

    oclMat g_counter(1, 1, CV_32SC1);
    g_counter.setTo(0);

    args.push_back(std::make_pair( sizeof(cl_mem),   (void*)&eig  ));
    args.push_back(std::make_pair( sizeof(cl_mem),   (void*)&mask.data ));
    args.push_back(std::make_pair( sizeof(cl_mem),   (void*)&corners.data ));
    args.push_back(std::make_pair( sizeof(cl_int),   (void*)&mask_strip));
    args.push_back(std::make_pair( sizeof(cl_float), (void*)&threshold ));
    args.push_back(std::make_pair( sizeof(cl_int), (void*)&eig.rows ));
    args.push_back(std::make_pair( sizeof(cl_int), (void*)&eig.cols ));
    args.push_back(std::make_pair( sizeof(cl_int), (void*)&max_count ));
    args.push_back(std::make_pair( sizeof(cl_mem), (void*)&g_counter.data ));

    size_t globalThreads[3] = {eig.cols, eig.rows, 1};
    size_t localThreads[3]  = {16, 16, 1};

    const char * opt = mask.empty() ? "" : "-D WITH_MASK";
    openCLExecuteKernel(cxt, &imgproc_gftt, kernelname, globalThreads, localThreads, args, -1, -1, opt);
    return std::min(Mat(g_counter).at<int>(0), max_count);
}
}//unnamed namespace

void cv::ocl::GoodFeaturesToTrackDetector_OCL::operator ()(const oclMat& image, oclMat& corners, const oclMat& mask)
{
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()));

    CV_DbgAssert(support_image2d());

    ensureSizeIsEnough(image.size(), CV_32F, eig_);

    if (useHarrisDetector)
        cornerMinEigenVal_dxdy(image, eig_, Dx_, Dy_, blockSize, 3, harrisK);
    else
        cornerMinEigenVal_dxdy(image, eig_, Dx_, Dy_, blockSize, 3);

    double maxVal = 0;
    minMax_buf(eig_, 0, &maxVal, oclMat(), minMaxbuf_);

    ensureSizeIsEnough(1, std::max(1000, static_cast<int>(image.size().area() * 0.05)), CV_32FC2, tmpCorners_);

    Ptr<TextureCL> eig_tex = bindTexturePtr(eig_);
    int total = findCorners_caller(
        *eig_tex,
        static_cast<float>(maxVal * qualityLevel),
        mask,
        tmpCorners_,
        tmpCorners_.cols);

    if (total == 0)
    {
        corners.release();
        return;
    }
    if(use_cpu_sorter)
    {
        Sorter<CPU_STL>::sortCorners_caller(eig_, tmpCorners_, total);
    }
    else
    {
        //if total is power of 2
        if(((total - 1) & (total)) == 0)
        {
            Sorter<BITONIC>::sortCorners_caller(*eig_tex, tmpCorners_, total);
        }
        else
        {
            Sorter<SELECTION>::sortCorners_caller(*eig_tex, tmpCorners_, total);
        }
    }

    if (minDistance < 1)
    {
        Rect roi_range(0, 0, maxCorners > 0 ? std::min(maxCorners, total) : total, 1);
        tmpCorners_(roi_range).copyTo(corners);
    }
    else
    {
        std::vector<Point2f> tmp(total);
        downloadPoints(tmpCorners_, tmp);

        std::vector<Point2f> tmp2;
        tmp2.reserve(total);

        const int cell_size = cvRound(minDistance);
        const int grid_width = (image.cols + cell_size - 1) / cell_size;
        const int grid_height = (image.rows + cell_size - 1) / cell_size;

        std::vector< std::vector<Point2f> > grid(grid_width * grid_height);

        for (int i = 0; i < total; ++i)
        {
            Point2f p = tmp[i];

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
                    std::vector<Point2f>& m = grid[yy * grid_width + xx];

                    if (!m.empty())
                    {
                        for(size_t j = 0; j < m.size(); j++)
                        {
                            float dx = p.x - m[j].x;
                            float dy = p.y - m[j].y;

                            if (dx * dx + dy * dy < minDistance * minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if(good)
            {
                grid[y_cell * grid_width + x_cell].push_back(p);

                tmp2.push_back(p);

                if (maxCorners > 0 && tmp2.size() == static_cast<size_t>(maxCorners))
                    break;
            }
        }

        corners.upload(Mat(1, static_cast<int>(tmp2.size()), CV_32FC2, &tmp2[0]));
    }
}
void cv::ocl::GoodFeaturesToTrackDetector_OCL::downloadPoints(const oclMat &points, std::vector<Point2f> &points_v)
{
    CV_DbgAssert(points.type() == CV_32FC2);
    points_v.resize(points.cols);
    openCLSafeCall(clEnqueueReadBuffer(
        *reinterpret_cast<cl_command_queue*>(getoclCommandQueue()),
        reinterpret_cast<cl_mem>(points.data),
        CL_TRUE,
        0,
        points.cols * sizeof(Point2f),
        &points_v[0],
        0,
        NULL,
        NULL));
}
