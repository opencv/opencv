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

#include "frame_queue.h"

cv::gpu::detail::FrameQueue::FrameQueue() :
    endOfDecode_(0),
    framesInQueue_(0),
    readPosition_(0)
{
    std::memset(displayQueue_, 0, sizeof(displayQueue_));
    std::memset((void*)isFrameInUse_, 0, sizeof(isFrameInUse_));
}

bool cv::gpu::detail::FrameQueue::waitUntilFrameAvailable(int pictureIndex)
{
    while (isInUse(pictureIndex))
    {
        // Decoder is getting too far ahead from display
        Thread::sleep(1);

        if (isEndOfDecode())
            return false;
    }

    return true;
}

void cv::gpu::detail::FrameQueue::enqueue(const CUVIDPARSERDISPINFO* picParams)
{
    // Mark the frame as 'in-use' so we don't re-use it for decoding until it is no longer needed
    // for display
    isFrameInUse_[picParams->picture_index] = true;

    // Wait until we have a free entry in the display queue (should never block if we have enough entries)
    do
    {
        bool isFramePlaced = false;

        {
            CriticalSection::AutoLock autoLock(criticalSection_);

            if (framesInQueue_ < MaximumSize)
            {
                int writePosition = (readPosition_ + framesInQueue_) % MaximumSize;
                displayQueue_[writePosition] = *picParams;
                framesInQueue_++;
                isFramePlaced = true;
            }
        }

        if (isFramePlaced) // Done
            break;

        // Wait a bit
        Thread::sleep(1);
    } while (!isEndOfDecode());
}

bool cv::gpu::detail::FrameQueue::dequeue(CUVIDPARSERDISPINFO& displayInfo)
{
    CriticalSection::AutoLock autoLock(criticalSection_);

    if (framesInQueue_ > 0)
    {
        int entry = readPosition_;
        displayInfo = displayQueue_[entry];
        readPosition_ = (entry + 1) % MaximumSize;
        framesInQueue_--;
        return true;
    }

    return false;
}
