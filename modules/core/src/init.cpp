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

#include <iostream>

namespace cv {
namespace core {

void OpenCV_Init()
{
    std::cout << "OpenCV_Init() called" << std::endl; // TODO For testing only, will be removed after precommit tests
}


struct CallbackEntry
{
    void (*cleanupFn)(void* ctx);
    void *ctx;

    CallbackEntry(void (*_cleanupFn)(void*), void* _ctx)
        : cleanupFn(_cleanupFn), ctx(_ctx)
    {
        // nothing
    }
};

static cv::Mutex g_shutdownCallbacksMutex;
static std::vector<CallbackEntry*> g_shutdownCallbacks;

void OpenCV_Shutdown()
{
    std::cout << "OpenCV_Shutdown() called" << std::endl; // TODO For testing only, will be removed after precommit tests

    cv::AutoLock lock(g_shutdownCallbacksMutex);
    for (size_t i = 0; i < g_shutdownCallbacks.size(); i++)
    {
        CallbackEntry* e = g_shutdownCallbacks[i];
        if (e && e->cleanupFn)
        {
            CallbackEntry saved = *e;
            e->cleanupFn = NULL;
            // TODO lock.unlock();
            saved.cleanupFn(saved.ctx);
            // TODO lock.relock();
        }
        g_shutdownCallbacks[i] = NULL;
        delete e;
    }
}

void* OpenCV_RegisterShutdownCallback(void (*cleanupFn)(void* ctx), void* ctx)
{
    CallbackEntry* e = new CallbackEntry(cleanupFn, ctx);
    cv::AutoLock lock(g_shutdownCallbacksMutex);
    g_shutdownCallbacks.push_back(e);
    return e;
}

bool OpenCV_UnRegisterShutdownCallback(void* callbackHandle)
{
    CV_Assert(callbackHandle != NULL);
    cv::AutoLock lock(g_shutdownCallbacksMutex);
    CallbackEntry* e = (CallbackEntry*)callbackHandle;
    e->cleanupFn = NULL;
    return true;
}

}
}
