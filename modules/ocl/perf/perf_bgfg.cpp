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
//    Jin Ma, jin@multicorewareinc.com
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
#include "precomp.hpp"

///////////// MOG////////////////////////
PERFTEST(MOG)
{
    const string inputFile[] = {"../../../samples/gpu/768x576.avi", "../../../samples/gpu/1920x1080.avi"};
    int cn[] = {1, 3};

    float learningRate[] = {0.0f, 0.01f};

    for(unsigned int idx = 0; idx < sizeof(inputFile)/sizeof(string); idx++)
    {
        cv::VideoCapture cap(inputFile[idx]);
        ASSERT_TRUE(cap.isOpened());

        cv::ocl::oclMat frame_ocl;
        cv::Mat frame;
        cv::Mat frame_init;
        cap >> frame;
        ASSERT_FALSE(frame.empty());
        frame_init = frame;

        int nframe = 5;
        std::vector<cv::Mat> frame_buffer_init;
        std::vector<cv::Mat> frame_buffer;
        std::vector<cv::ocl::oclMat> frame_buffer_ocl;
        std::vector<cv::Mat> foreground_buf_ocl;
        std::vector<cv::Mat> foreground_buf_cpu;

        for(int i = 0; i < nframe; i++)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            frame_buffer_init.push_back(frame);
        }

        for(unsigned int i = 0; i < sizeof(learningRate)/sizeof(float); i++)
        {
            for(unsigned int j = 0; j < sizeof(cn)/sizeof(int); j++)
            {
                SUBTEST << frame.cols << 'x' << frame.rows << ".avi; "<<"channels: "<<cn[j]<<"; learningRate: "<<learningRate[i];

                frame_buffer = frame_buffer_init;
                frame = frame_init;
                cv::Mat temp;
                if(cn[j] != 3)
                {
                    if(cn[j] == 1)
                        cv::cvtColor(frame, temp, cv::COLOR_RGB2GRAY);
                    else
                        cv::cvtColor(frame, temp, cv::COLOR_RGB2RGBA);

                    cv::swap(temp, frame);
                }

                cv::BackgroundSubtractorMOG mog_cpu;
                cv::Mat foreground_cpu;

                mog_cpu(frame, foreground_cpu, learningRate[i]);

                foreground_cpu.release();

                cv::ocl::oclMat d_frame(frame);
                cv::ocl::MOG_OCL d_mog;
                cv::ocl::oclMat foreground_ocl;

                d_mog(d_frame, foreground_ocl, learningRate[i]);
                foreground_ocl.release();
                d_mog.release();

                for(int iter = 0; iter < nframe; iter++)
                {
                    cv::Mat temp1;
                    if(cn[j] != 3)
                    {
                        if(cn[j] == 1)
                            cv::cvtColor(frame_buffer[iter], temp1, cv::COLOR_RGB2GRAY);
                        else
                            cv::cvtColor(frame_buffer[iter], temp1, cv::COLOR_RGB2RGBA);

                        cv::swap(temp1, frame_buffer[iter]);
                    }
                    temp1.release();
                    cv::ocl::oclMat temp2;
                    temp2.upload(frame_buffer[iter]);
                    frame_buffer_ocl.push_back(temp2);
                }

                CPU_ON;
                foreground_buf_cpu.clear();
                for(int iter = 0; iter < nframe; iter++)
                {
                    mog_cpu(frame_buffer[iter], foreground_cpu, learningRate[i]);
                    foreground_buf_cpu.push_back(foreground_cpu);
                }
                CPU_OFF;


                WARMUP_ON;
                d_mog(d_frame, foreground_ocl, learningRate[i]);
                d_mog.release();
                WARMUP_OFF;

                foreground_ocl.release();

                GPU_ON;
                for(int iter = 0; iter < nframe; iter++)
                {
                    d_mog(frame_buffer_ocl[iter], foreground_ocl, learningRate[i]);
                }
                d_mog.release();
                GPU_OFF;

                foreground_ocl.release();

                frame_buffer_ocl.clear();

                GPU_FULL_ON;
                foreground_buf_ocl.clear();
                frame_buffer_ocl.clear();
                for(int iter = 0; iter < nframe; iter++)
                {
                    cv::ocl::oclMat temp;
                    temp.upload(frame_buffer[iter]);
                    frame_buffer_ocl.push_back(temp);

                    d_mog(frame_buffer_ocl[iter], foreground_ocl, learningRate[i]);

                    cv::Mat temp1;
                    foreground_ocl.download(temp1);
                    foreground_buf_ocl.push_back(temp1);
                }
                d_mog.release();
                GPU_FULL_OFF;

                for(int iter = 0; iter < nframe; iter++)
                    TestSystem::instance().ExpectedMatNear(foreground_buf_ocl[iter], foreground_buf_cpu[iter], 0.0);

                foreground_buf_ocl.clear();
                foreground_buf_cpu.clear();
                frame_buffer_ocl.clear();
                frame_buffer.clear();
            }
        }

        cap.release();
    }
}

///////////// MOG2////////////////////////
PERFTEST(MOG2)
{
    const string inputFile[] = {"../../../samples/gpu/768x576.avi", "../../../samples/gpu/1920x1080.avi"};
    int cn[] = {1, 3, 4};

    for(unsigned int idx = 0; idx < sizeof(inputFile)/sizeof(string); idx++)
    {
        cv::VideoCapture cap(inputFile[idx]);
        ASSERT_TRUE(cap.isOpened());

        cv::ocl::oclMat frame_ocl;
        cv::Mat frame;
        cv::Mat frame_init;
        cap >> frame;
        ASSERT_FALSE(frame.empty());
        frame_init = frame;

        int nframe = 5;
        std::vector<cv::Mat> frame_buffer_init;
        std::vector<cv::Mat> frame_buffer;
        std::vector<cv::ocl::oclMat> frame_buffer_ocl;
        std::vector<cv::Mat> foreground_buf_ocl;
        std::vector<cv::Mat> foreground_buf_cpu;

        for(int i = 0; i < nframe; i++)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            frame_buffer_init.push_back(frame);
        }

        for(unsigned int j = 0; j < sizeof(cn)/sizeof(int); j++)
        {
            SUBTEST << frame.cols << 'x' << frame.rows << ".avi; "<<"channels: "<<cn[j];

            frame_buffer = frame_buffer_init;
            frame = frame_init;
            cv::Mat temp;
            if(cn[j] != 3)
            {
                if(cn[j] == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_RGB2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_RGB2RGBA);

                cv::swap(temp, frame);
            }

            cv::BackgroundSubtractorMOG2 mog_cpu;
            mog_cpu.set("detectShadows", false);
            cv::Mat foreground_cpu;

            mog_cpu(frame, foreground_cpu);

            foreground_cpu.release();

            cv::ocl::oclMat d_frame(frame);
            cv::ocl::MOG2_OCL d_mog;
            cv::ocl::oclMat foreground_ocl;

            d_mog(d_frame, foreground_ocl);
            foreground_ocl.release();
            d_mog.release();

            for(int iter = 0; iter < nframe; iter++)
            {
                cv::Mat temp1;
                if(cn[j] != 3)
                {
                    if(cn[j] == 1)
                        cv::cvtColor(frame_buffer[iter], temp1, cv::COLOR_RGB2GRAY);
                    else
                        cv::cvtColor(frame_buffer[iter], temp1, cv::COLOR_RGB2RGBA);

                    cv::swap(temp1, frame_buffer[iter]);
                }
                temp1.release();
                cv::ocl::oclMat temp2;
                temp2.upload(frame_buffer[iter]);
                frame_buffer_ocl.push_back(temp2);
            }

            CPU_ON;
            foreground_buf_cpu.clear();
            for(int iter = 0; iter < nframe; iter++)
            {
                mog_cpu(frame_buffer[iter], foreground_cpu);
                foreground_buf_cpu.push_back(foreground_cpu);
            }
            CPU_OFF;


            WARMUP_ON;
            d_mog(d_frame, foreground_ocl);
            d_mog.release();
            WARMUP_OFF;

            foreground_ocl.release();

            GPU_ON;
            for(int iter = 0; iter < nframe; iter++)
            {
                d_mog(frame_buffer_ocl[iter], foreground_ocl);
            }
            d_mog.release();
            GPU_OFF;

            foreground_ocl.release();

            frame_buffer_ocl.clear();

            GPU_FULL_ON;
            foreground_buf_ocl.clear();
            frame_buffer_ocl.clear();
            for(int iter = 0; iter < nframe; iter++)
            {
                cv::ocl::oclMat temp;
                temp.upload(frame_buffer[iter]);
                frame_buffer_ocl.push_back(temp);

                d_mog(frame_buffer_ocl[iter], foreground_ocl);

                cv::Mat temp1;
                foreground_ocl.download(temp1);
                foreground_buf_ocl.push_back(temp1);
            }
            d_mog.release();
            GPU_FULL_OFF;

            for(int iter = 0; iter < nframe; iter++)
                TestSystem::instance().ExpectedMatNear(foreground_buf_ocl[iter], foreground_buf_cpu[iter], 0.0);

            foreground_buf_ocl.clear();
            foreground_buf_cpu.clear();
            frame_buffer_ocl.clear();
            frame_buffer.clear();
        }
        cap.release();
    }
}

///////////// MOG2GetBackgroundImage////////////////////////
PERFTEST(MOG2GetBackgroundImage)
{
    const string inputFile[] = {"../../../samples/gpu/768x576.avi", "../../../samples/gpu/1920x1080.avi"};
    int cn[] = {3};
    
    for(unsigned int idx = 0; idx < sizeof(inputFile)/sizeof(string); idx++)
    {
        cv::VideoCapture cap(inputFile[idx]);
        ASSERT_TRUE(cap.isOpened());

        cv::ocl::oclMat frame_ocl;
        cv::Mat frame;
        cv::Mat frame_init;
        cap >> frame;
        ASSERT_FALSE(frame.empty());
        frame_init = frame;

        int nframe = 5;
        std::vector<cv::Mat> frame_buffer_init;
        std::vector<cv::Mat> frame_buffer;
        std::vector<cv::ocl::oclMat> frame_buffer_ocl;
        std::vector<cv::Mat> foreground_buf_ocl;
        std::vector<cv::Mat> background_buf_ocl;
        std::vector<cv::Mat> foreground_buf_cpu;
        std::vector<cv::Mat> background_buf_cpu;

        for(int i = 0; i < nframe; i++)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            frame_buffer_init.push_back(frame);
        }

        for(unsigned int j = 0; j < sizeof(cn)/sizeof(int); j++)
        {
            SUBTEST << frame.cols << 'x' << frame.rows << ".avi; "<<"channels: "<<cn[j];

            frame_buffer = frame_buffer_init;
            frame = frame_init;
            cv::Mat temp;
            if(cn[j] != 3)
            {
                if(cn[j] == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_RGB2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_RGB2RGBA);

                cv::swap(temp, frame);
            }

            cv::BackgroundSubtractorMOG2 mog_cpu;
            cv::Mat foreground_cpu;
            cv::Mat background_cpu;

            mog_cpu(frame, foreground_cpu);
            mog_cpu.getBackgroundImage(background_cpu);

            foreground_cpu.release();
            background_cpu.release();

            cv::ocl::oclMat d_frame(frame);
            cv::ocl::MOG2_OCL d_mog;
            cv::ocl::oclMat foreground_ocl;
            cv::ocl::oclMat background_ocl;

            d_mog(d_frame, foreground_ocl);
            d_mog.getBackgroundImage(background_ocl);
            foreground_ocl.release();
            background_ocl.release();

            for(int iter = 0; iter < nframe; iter++)
            {
                cv::Mat temp1;
                if(cn[j] != 3)
                {
                    if(cn[j] == 1)
                        cv::cvtColor(frame_buffer[iter], temp1, cv::COLOR_RGB2GRAY);
                    else
                        cv::cvtColor(frame_buffer[iter], temp1, cv::COLOR_RGB2RGBA);

                    cv::swap(temp1, frame_buffer[iter]);
                }
                temp1.release();
                cv::ocl::oclMat temp2;
                temp2.upload(frame_buffer[iter]);
                frame_buffer_ocl.push_back(temp2);
            }

            CPU_ON;
            foreground_buf_cpu.clear();
            background_buf_cpu.clear();
            for(int iter = 0; iter < nframe; iter++)
            {
                mog_cpu(frame_buffer[iter], foreground_cpu);
                foreground_buf_cpu.push_back(foreground_cpu);
            }
            mog_cpu.getBackgroundImage(background_cpu);
            background_buf_cpu.push_back(background_cpu);
            CPU_OFF;

            WARMUP_ON;
            d_mog(d_frame, foreground_ocl);
            d_mog.release();
            WARMUP_OFF;

            foreground_ocl.release();

            GPU_ON;
            for(int iter = 0; iter < nframe; iter++)
            {
                d_mog(frame_buffer_ocl[iter], foreground_ocl);
            }
            d_mog.getBackgroundImage(background_ocl);
            d_mog.release();
            background_ocl.release();
            GPU_OFF;

            foreground_ocl.release();

            frame_buffer_ocl.clear();

            GPU_FULL_ON;
            background_buf_ocl.clear();
            foreground_buf_ocl.clear();
            frame_buffer_ocl.clear();
            for(int iter = 0; iter < nframe; iter++)
            {
                cv::ocl::oclMat temp;
                temp.upload(frame_buffer[iter]);
                frame_buffer_ocl.push_back(temp);

                d_mog(frame_buffer_ocl[iter], foreground_ocl);

                cv::Mat temp1;
                foreground_ocl.download(temp1);
                foreground_buf_ocl.push_back(temp1);
            }
            d_mog.getBackgroundImage(background_ocl);
            background_buf_ocl.push_back(cv::Mat(background_ocl));
            d_mog.release();
            background_ocl.release();
            GPU_FULL_OFF;

            for(int iter = 0; iter < 1; iter++)
                TestSystem::instance().ExpectedMatNear(background_buf_ocl[iter], background_buf_cpu[iter], 0.0);

            foreground_buf_ocl.clear();
            foreground_buf_cpu.clear();
            frame_buffer_ocl.clear();
            frame_buffer.clear();
        }
    }
}
