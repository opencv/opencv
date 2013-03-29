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

#include "thread_wrappers.h"

#if defined(HAVE_CUDA) && defined(HAVE_NVCUVID)

#ifdef WIN32
    #define NOMINMAX
    #include <windows.h>
#else
    #include <pthread.h>
    #include <unistd.h>
#endif

#ifdef WIN32
    class cv::gpu::detail::CriticalSection::Impl
    {
    public:
        Impl()
        {
            InitializeCriticalSection(&criticalSection_);
        }

        ~Impl()
        {
            DeleteCriticalSection(&criticalSection_);
        }

        void enter()
        {
            EnterCriticalSection(&criticalSection_);
        }

        void leave()
        {
            LeaveCriticalSection(&criticalSection_);
        }

    private:
        CRITICAL_SECTION criticalSection_;
    };
#else
    class cv::gpu::detail::CriticalSection::Impl
    {
    public:
        Impl()
        {
            pthread_mutexattr_t mutex_attribute;
            pthread_mutexattr_init(&mutex_attribute);
            pthread_mutexattr_settype(&mutex_attribute, PTHREAD_MUTEX_RECURSIVE);
            pthread_mutex_init(&mutex_, 0);
            pthread_mutexattr_destroy(&mutex_attribute);
        }

        ~Impl()
        {
            pthread_mutex_destroy(&mutex_);
        }

        void enter()
        {
            pthread_mutex_lock(&mutex_);
        }

        void leave()
        {
            pthread_mutex_unlock(&mutex_);
        }

    private:
        pthread_mutex_t mutex_;
    };
#endif

cv::gpu::detail::CriticalSection::CriticalSection() :
    impl_(new Impl)
{
}

cv::gpu::detail::CriticalSection::~CriticalSection()
{
}

void cv::gpu::detail::CriticalSection::enter()
{
    impl_->enter();
}

void cv::gpu::detail::CriticalSection::leave()
{
    impl_->leave();
}

#ifdef WIN32
    namespace
    {
        struct UserData
        {
            void (*func)(void* userData);
            void* param;
        };

        DWORD WINAPI WinThreadFunction(LPVOID lpParam)
        {
            UserData* userData = static_cast<UserData*>(lpParam);

            userData->func(userData->param);

            return 0;
        }
    }

    class cv::gpu::detail::Thread::Impl
    {
    public:
        Impl(void (*func)(void* userData), void* userData)
        {
            userData_.func = func;
            userData_.param = userData;

            thread_ = CreateThread(
                NULL,                   // default security attributes
                0,                      // use default stack size
                WinThreadFunction,      // thread function name
                &userData_,             // argument to thread function
                0,                      // use default creation flags
                &threadId_);            // returns the thread identifier
        }

        ~Impl()
        {
            CloseHandle(thread_);
        }

        void wait()
        {
            WaitForSingleObject(thread_, INFINITE);
        }

    private:
        UserData userData_;
        HANDLE thread_;
        DWORD threadId_;
    };
#else
    namespace
    {
        struct UserData
        {
            void (*func)(void* userData);
            void* param;
        };

        void* PThreadFunction(void* lpParam)
        {
            UserData* userData = static_cast<UserData*>(lpParam);

            userData->func(userData->param);

            return 0;
        }
    }

    class cv::gpu::detail::Thread::Impl
    {
    public:
        Impl(void (*func)(void* userData), void* userData)
        {
            userData_.func = func;
            userData_.param = userData;

            pthread_create(&thread_, NULL, PThreadFunction, &userData_);
        }

        ~Impl()
        {
            pthread_detach(thread_);
        }

        void wait()
        {
            pthread_join(thread_, NULL);
        }

    private:
        pthread_t thread_;
        UserData userData_;
    };
#endif

cv::gpu::detail::Thread::Thread(void (*func)(void* userData), void* userData) :
    impl_(new Impl(func, userData))
{
}

cv::gpu::detail::Thread::~Thread()
{
}

void cv::gpu::detail::Thread::wait()
{
    impl_->wait();
}

void cv::gpu::detail::Thread::sleep(int ms)
{
#ifdef WIN32
    ::Sleep(ms);
#else
    ::usleep(ms * 1000);
#endif
}

#endif // HAVE_CUDA
