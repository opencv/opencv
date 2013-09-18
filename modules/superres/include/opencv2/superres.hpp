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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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

#ifndef __OPENCV_SUPERRES_HPP__
#define __OPENCV_SUPERRES_HPP__

#include "opencv2/core.hpp"

namespace cv
{
    namespace superres
    {
        CV_EXPORTS bool initModule_superres();

        class CV_EXPORTS FrameSource
        {
        public:
            virtual ~FrameSource();

            virtual void nextFrame(OutputArray frame) = 0;
            virtual void reset() = 0;
        };

        CV_EXPORTS Ptr<FrameSource> createFrameSource_Empty();

        CV_EXPORTS Ptr<FrameSource> createFrameSource_Video(const String& fileName);
        CV_EXPORTS Ptr<FrameSource> createFrameSource_Video_GPU(const String& fileName);

        CV_EXPORTS Ptr<FrameSource> createFrameSource_Camera(int deviceId = 0);

        class CV_EXPORTS SuperResolution : public cv::Algorithm, public FrameSource
        {
        public:
            void setInput(const Ptr<FrameSource>& frameSource);

            void nextFrame(OutputArray frame);
            void reset();

            virtual void collectGarbage();

        protected:
            SuperResolution();

            virtual void initImpl(Ptr<FrameSource>& frameSource) = 0;
            virtual void processImpl(Ptr<FrameSource>& frameSource, OutputArray output) = 0;

        private:
            Ptr<FrameSource> frameSource_;
            bool firstCall_;
        };

        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        // Dennis Mitzel, Thomas Pock, Thomas Schoenemann, Daniel Cremers. Video Super Resolution using Duality Based TV-L1 Optical Flow.
        CV_EXPORTS Ptr<SuperResolution> createSuperResolution_BTVL1();
        CV_EXPORTS Ptr<SuperResolution> createSuperResolution_BTVL1_GPU();
        CV_EXPORTS Ptr<SuperResolution> createSuperResolution_BTVL1_OCL();
    }
}

#endif // __OPENCV_SUPERRES_HPP__
