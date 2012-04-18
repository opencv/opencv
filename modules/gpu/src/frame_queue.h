/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

#ifndef __FRAME_QUEUE_H__
#define __FRAME_QUEUE_H__

#include "precomp.hpp"
#include "thread_wrappers.h"

#ifdef HAVE_CUDA

namespace cv { namespace gpu
{
    namespace detail
    {
        class FrameQueue
        {
        public:
            static const int MaximumSize = 20; // MAX_FRM_CNT;

            FrameQueue();

            void endDecode() { endOfDecode_ = true; }
            bool isEndOfDecode() const { return endOfDecode_ != 0;}

            // Spins until frame becomes available or decoding gets canceled. 
            // If the requested frame is available the method returns true.
            // If decoding was interupted before the requested frame becomes
            // available, the method returns false.
            bool waitUntilFrameAvailable(int pictureIndex);

            void enqueue(const CUVIDPARSERDISPINFO* picParams);

            // Deque the next frame.
            // Parameters:
            //      displayInfo - New frame info gets placed into this object.
            // Returns:
            //      true, if a new frame was returned,
            //      false, if the queue was empty and no new frame could be returned. 
            bool dequeue(CUVIDPARSERDISPINFO& displayInfo);

            void releaseFrame(const CUVIDPARSERDISPINFO& picParams) { isFrameInUse_[picParams.picture_index] = false; }

        private:
            FrameQueue(const FrameQueue&);
            FrameQueue& operator =(const FrameQueue&);

            bool isInUse(int pictureIndex) const { return isFrameInUse_[pictureIndex] != 0; }

            CriticalSection criticalSection_;

            volatile int isFrameInUse_[MaximumSize];
            volatile int endOfDecode_;

            int framesInQueue_;
            int readPosition_;
            CUVIDPARSERDISPINFO displayQueue_[MaximumSize];
        };
    }
}}

#endif // HAVE_CUDA

#endif // __FRAME_QUEUE_H__
