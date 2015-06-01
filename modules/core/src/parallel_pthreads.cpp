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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

#if defined HAVE_PTHREADS && HAVE_PTHREADS

#include <algorithm>
#include <pthread.h>

namespace cv
{

class ThreadManager;

enum ForThreadState
{
    eFTNotStarted = 0,
    eFTStarted = 1,
    eFTToStop = 2,
    eFTStoped = 3
};

enum ThreadManagerPoolState
{
    eTMNotInited = 0,
    eTMFailedToInit = 1,
    eTMInited = 2,
    eTMSingleThreaded = 3
};

struct work_load
{
    work_load()
    {
        clear();
    }

    work_load(const cv::Range& range, const cv::ParallelLoopBody& body, int nstripes)
    {
        set(range, body, nstripes);
    }

    void set(const cv::Range& range, const cv::ParallelLoopBody& body, int nstripes)
    {
        m_body = &body;
        m_range = &range;
        m_nstripes = nstripes;
        m_blocks_count = ((m_range->end - m_range->start - 1)/m_nstripes) + 1;
    }

    const cv::ParallelLoopBody* m_body;
    const cv::Range*            m_range;
    int                         m_nstripes;
    unsigned int                m_blocks_count;

    void clear()
    {
        m_body = 0;
        m_range = 0;
        m_nstripes = 0;
        m_blocks_count = 0;
    }
};

class ForThread
{
public:

    ForThread(): m_task_start(false), m_parent(0), m_state(eFTNotStarted), m_id(0)
    {
    }

    //called from manager thread
    bool init(size_t id, ThreadManager* parent);

    //called from manager thread
    void run();

    //called from manager thread
    void stop();

    ~ForThread();

private:

    //called from worker thread
    static void* thread_loop_wrapper(void* thread_object);

    //called from worker thread
    void execute();

    //called from worker thread
    void thread_body();

    pthread_t       m_posix_thread;
    pthread_mutex_t m_thread_mutex;
    pthread_cond_t  m_cond_thread_task;
    bool            m_task_start;

    ThreadManager*  m_parent;
    ForThreadState  m_state;
    size_t          m_id;
};

class ThreadManager
{
public:
    friend class ForThread;

    static ThreadManager& instance()
    {
        if(!m_instance.ptr)
        {
            pthread_mutex_lock(&m_manager_access_mutex);

            if(!m_instance.ptr)
            {
                m_instance.ptr = new ThreadManager();
            }

            pthread_mutex_unlock(&m_manager_access_mutex);
        }

        return *m_instance.ptr;
    }


    static void stop()
    {
        ThreadManager& manager = instance();

        if(manager.m_pool_state == eTMInited)
        {
            for(size_t i = 0; i < manager.m_num_threads; ++i)
            {
                manager.m_threads[i].stop();
            }
        }

        manager.m_pool_state = eTMNotInited;
    }

    void run(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes);

    size_t getNumOfThreads();

    void setNumOfThreads(size_t n);

private:

    struct ptr_holder
    {
        ThreadManager* ptr;

        ptr_holder(): ptr(NULL) { }

        ~ptr_holder()
        {
            if(ptr)
            {
                delete ptr;
            }
        }
    };

    ThreadManager();

    ~ThreadManager();

    void wait_complete();

    void notify_complete();

    bool initPool();

    size_t defaultNumberOfThreads();

    std::vector<ForThread> m_threads;
    size_t m_num_threads;

    pthread_mutex_t m_manager_task_mutex;
    pthread_cond_t  m_cond_thread_task_complete;
    bool            m_task_complete;

    unsigned int m_task_position;
    unsigned int m_num_of_completed_tasks;

    static pthread_mutex_t m_manager_access_mutex;
    static ptr_holder m_instance;

    static const char m_env_name[];
    static const unsigned int m_default_number_of_threads;

    work_load m_work_load;

    struct work_thread_t
    {
        work_thread_t(): value(false) { }
        bool value;
    };

    cv::TLSData<work_thread_t> m_is_work_thread;

    ThreadManagerPoolState m_pool_state;
};

#ifndef PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP
#define PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP PTHREAD_RECURSIVE_MUTEX_INITIALIZER
#endif

pthread_mutex_t ThreadManager::m_manager_access_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

ThreadManager::ptr_holder ThreadManager::m_instance;
const char ThreadManager::m_env_name[] = "OPENCV_FOR_THREADS_NUM";

#ifdef ANDROID
// many modern phones/tables have 4-core CPUs. Let's use no more
// than 2 threads by default not to overheat the devices
const unsigned int ThreadManager::m_default_number_of_threads = 2;
#else
const unsigned int ThreadManager::m_default_number_of_threads = 8;
#endif

ForThread::~ForThread()
{
    if(m_state == eFTStarted)
    {
        stop();

        pthread_mutex_destroy(&m_thread_mutex);

        pthread_cond_destroy(&m_cond_thread_task);
    }
}

bool ForThread::init(size_t id, ThreadManager* parent)
{
    m_id = id;

    m_parent = parent;

    int res = 0;

    res |= pthread_mutex_init(&m_thread_mutex, NULL);

    res |= pthread_cond_init(&m_cond_thread_task, NULL);

    if(!res)
    {
        res = pthread_create(&m_posix_thread, NULL, thread_loop_wrapper, (void*)this);
    }


    return res == 0;
}

void ForThread::stop()
{
    if(m_state == eFTStarted)
    {
        m_state = eFTToStop;

        run();

        pthread_join(m_posix_thread, NULL);
    }

    m_state = eFTStoped;
}

void ForThread::run()
{
    pthread_mutex_lock(&m_thread_mutex);

    m_task_start = true;

    pthread_cond_signal(&m_cond_thread_task);

    pthread_mutex_unlock(&m_thread_mutex);
}

void* ForThread::thread_loop_wrapper(void* thread_object)
{
    ((ForThread*)thread_object)->thread_body();
    return 0;
}

void ForThread::execute()
{
    unsigned int m_current_pos = CV_XADD(&m_parent->m_task_position, 1);

    work_load& load = m_parent->m_work_load;

    while(m_current_pos < load.m_blocks_count)
    {
        int start = load.m_range->start + m_current_pos*load.m_nstripes;
        int end = std::min(start + load.m_nstripes, load.m_range->end);

        load.m_body->operator()(cv::Range(start, end));

        m_current_pos = CV_XADD(&m_parent->m_task_position, 1);
    }
}

void ForThread::thread_body()
{
    m_parent->m_is_work_thread.get()->value = true;

    pthread_mutex_lock(&m_thread_mutex);

    m_state = eFTStarted;

    while(m_state == eFTStarted)
    {
        //to handle spurious wakeups
        while( !m_task_start && m_state != eFTToStop )
            pthread_cond_wait(&m_cond_thread_task, &m_thread_mutex);

        if(m_state == eFTStarted)
        {
            execute();

            m_task_start = false;

            m_parent->notify_complete();
        }
    }

    pthread_mutex_unlock(&m_thread_mutex);
}

ThreadManager::ThreadManager(): m_num_threads(0), m_task_complete(false), m_num_of_completed_tasks(0), m_pool_state(eTMNotInited)
{
    int res = 0;

    res |= pthread_mutex_init(&m_manager_task_mutex, NULL);

    res |= pthread_cond_init(&m_cond_thread_task_complete, NULL);

    if(!res)
    {
        setNumOfThreads(defaultNumberOfThreads());

        m_task_position = 0;
    }
    else
    {
        m_num_threads = 1;
        m_pool_state = eTMFailedToInit;
        m_task_position = 0;

        //print error;
    }
}

ThreadManager::~ThreadManager()
{
    stop();

    pthread_mutex_destroy(&m_manager_task_mutex);

    pthread_cond_destroy(&m_cond_thread_task_complete);

    pthread_mutex_destroy(&m_manager_access_mutex);
}

void ThreadManager::run(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes)
{
    bool is_work_thread = m_is_work_thread.get()->value;

    if( (getNumOfThreads() > 1) && !is_work_thread &&
        (range.end - range.start > 1) && (nstripes <= 0 || nstripes >= 1.5) )
    {
        int res = pthread_mutex_trylock(&m_manager_access_mutex);

        if(!res)
        {
            if(initPool())
            {
                double min_stripes = double(range.end - range.start)/(4*m_threads.size());

                nstripes = std::max(nstripes, min_stripes);

                pthread_mutex_lock(&m_manager_task_mutex);

                m_num_of_completed_tasks = 0;

                m_task_position = 0;

                m_task_complete = false;

                m_work_load.set(range, body, std::ceil(nstripes));

                for(size_t i = 0; i < m_threads.size(); ++i)
                {
                    m_threads[i].run();
                }

                wait_complete();
            }
            else
            {
                //print error
                body(range);
            }
        }
        else
        {
            body(range);
        }
    }
    else
    {
        body(range);
    }
}

void ThreadManager::wait_complete()
{
    //to handle spurious wakeups
    while(!m_task_complete)
        pthread_cond_wait(&m_cond_thread_task_complete, &m_manager_task_mutex);

    pthread_mutex_unlock(&m_manager_task_mutex);

    pthread_mutex_unlock(&m_manager_access_mutex);
}

void ThreadManager::notify_complete()
{

    unsigned int comp = CV_XADD(&m_num_of_completed_tasks, 1);

    if(comp == (m_num_threads - 1))
    {
        pthread_mutex_lock(&m_manager_task_mutex);

        m_task_complete = true;

        pthread_cond_signal(&m_cond_thread_task_complete);

        pthread_mutex_unlock(&m_manager_task_mutex);
    }
}

bool ThreadManager::initPool()
{
    if(m_pool_state != eTMNotInited || m_num_threads == 1)
        return true;

    m_threads.resize(m_num_threads);

    bool res = true;

    for(size_t i = 0; i < m_threads.size(); ++i)
    {
        res |= m_threads[i].init(i, this);
    }

    if(res)
    {
        m_pool_state = eTMInited;
    }
    else
    {
        //TODO: join threads?
        m_pool_state = eTMFailedToInit;
    }

    return res;
}

size_t ThreadManager::getNumOfThreads()
{
    return m_num_threads;
}

void ThreadManager::setNumOfThreads(size_t n)
{
    int res = pthread_mutex_lock(&m_manager_access_mutex);

    if(!res)
    {
        if(n == 0)
        {
            n = defaultNumberOfThreads();
        }

        if(n != m_num_threads && m_pool_state != eTMFailedToInit)
        {
            if(m_pool_state == eTMInited)
            {
                stop();
                m_threads.clear();
            }

            m_num_threads = n;

            if(m_num_threads == 1)
            {
                m_pool_state = eTMSingleThreaded;
            }
            else
            {
                m_pool_state = eTMNotInited;
            }
        }

        pthread_mutex_unlock(&m_manager_access_mutex);
    }
}

size_t ThreadManager::defaultNumberOfThreads()
{
    unsigned int result = m_default_number_of_threads;

    char * env = getenv(m_env_name);

    if(env != NULL)
    {
        sscanf(env, "%u", &result);

        result = std::max(1u, result);
        //do we need upper limit of threads number?
    }

    return result;
}

void parallel_for_pthreads(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes);
size_t parallel_pthreads_get_threads_num();
void parallel_pthreads_set_threads_num(int num);

size_t parallel_pthreads_get_threads_num()
{
    return ThreadManager::instance().getNumOfThreads();
}

void parallel_pthreads_set_threads_num(int num)
{
    if(num < 0)
    {
        ThreadManager::instance().setNumOfThreads(0);
    }
    else
    {
        ThreadManager::instance().setNumOfThreads(size_t(num));
    }
}

void parallel_for_pthreads(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes)
{
    ThreadManager::instance().run(range, body, nstripes);
}

}

#endif
