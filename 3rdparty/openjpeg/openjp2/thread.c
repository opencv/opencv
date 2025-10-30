/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2016, Even Rouault
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>

#ifdef MUTEX_win32

/* Some versions of x86_64-w64-mingw32-gc -m32 resolve InterlockedCompareExchange() */
/* as __sync_val_compare_and_swap_4 but fails to link it. As this protects against */
/* a rather unlikely race, skip it */
#if !(defined(__MINGW32__) && defined(__i386__))
#define HAVE_INTERLOCKED_COMPARE_EXCHANGE 1
#endif

#include <windows.h>
#include <process.h>

#include "opj_includes.h"

OPJ_BOOL OPJ_CALLCONV opj_has_thread_support(void)
{
    return OPJ_TRUE;
}

int OPJ_CALLCONV opj_get_num_cpus(void)
{
    SYSTEM_INFO info;
    DWORD dwNum;
    GetSystemInfo(&info);
    dwNum = info.dwNumberOfProcessors;
    if (dwNum < 1) {
        return 1;
    }
    return (int)dwNum;
}

struct opj_mutex_t {
    CRITICAL_SECTION cs;
};

opj_mutex_t* opj_mutex_create(void)
{
    opj_mutex_t* mutex = (opj_mutex_t*) opj_malloc(sizeof(opj_mutex_t));
    if (!mutex) {
        return NULL;
    }
    InitializeCriticalSectionAndSpinCount(&(mutex->cs), 4000);
    return mutex;
}

void opj_mutex_lock(opj_mutex_t* mutex)
{
    EnterCriticalSection(&(mutex->cs));
}

void opj_mutex_unlock(opj_mutex_t* mutex)
{
    LeaveCriticalSection(&(mutex->cs));
}

void opj_mutex_destroy(opj_mutex_t* mutex)
{
    if (!mutex) {
        return;
    }
    DeleteCriticalSection(&(mutex->cs));
    opj_free(mutex);
}

struct opj_cond_waiter_list_t {
    HANDLE hEvent;
    struct opj_cond_waiter_list_t* next;
};
typedef struct opj_cond_waiter_list_t opj_cond_waiter_list_t;

struct opj_cond_t {
    opj_mutex_t             *internal_mutex;
    opj_cond_waiter_list_t  *waiter_list;
};

static DWORD TLSKey = 0;
static volatile LONG inTLSLockedSection = 0;
static volatile int TLSKeyInit = OPJ_FALSE;

opj_cond_t* opj_cond_create(void)
{
    opj_cond_t* cond = (opj_cond_t*) opj_malloc(sizeof(opj_cond_t));
    if (!cond) {
        return NULL;
    }

    /* Make sure that the TLS key is allocated in a thread-safe way */
    /* We cannot use a global mutex/critical section since its creation itself would not be */
    /* thread-safe, so use InterlockedCompareExchange trick */
    while (OPJ_TRUE) {

#if HAVE_INTERLOCKED_COMPARE_EXCHANGE
        if (InterlockedCompareExchange(&inTLSLockedSection, 1, 0) == 0)
#endif
        {
            if (!TLSKeyInit) {
                TLSKey = TlsAlloc();
                TLSKeyInit = OPJ_TRUE;
            }
#if HAVE_INTERLOCKED_COMPARE_EXCHANGE
            InterlockedCompareExchange(&inTLSLockedSection, 0, 1);
#endif
            break;
        }
    }

    if (TLSKey == TLS_OUT_OF_INDEXES) {
        opj_free(cond);
        return NULL;
    }
    cond->internal_mutex = opj_mutex_create();
    if (cond->internal_mutex == NULL) {
        opj_free(cond);
        return NULL;
    }
    cond->waiter_list = NULL;
    return cond;
}

void opj_cond_wait(opj_cond_t* cond, opj_mutex_t* mutex)
{
    opj_cond_waiter_list_t* item;
    HANDLE hEvent = (HANDLE) TlsGetValue(TLSKey);
    if (hEvent == NULL) {
        hEvent = CreateEvent(NULL, /* security attributes */
                             0,    /* manual reset = no */
                             0,    /* initial state = unsignaled */
                             NULL  /* no name */);
        assert(hEvent);

        TlsSetValue(TLSKey, hEvent);
    }

    /* Insert the waiter into the waiter list of the condition */
    opj_mutex_lock(cond->internal_mutex);

    item = (opj_cond_waiter_list_t*)opj_malloc(sizeof(opj_cond_waiter_list_t));
    assert(item != NULL);

    item->hEvent = hEvent;
    item->next = cond->waiter_list;

    cond->waiter_list = item;

    opj_mutex_unlock(cond->internal_mutex);

    /* Release the client mutex before waiting for the event being signaled */
    opj_mutex_unlock(mutex);

    /* Ideally we would check that we do not get WAIT_FAILED but it is hard */
    /* to report a failure. */
    WaitForSingleObject(hEvent, INFINITE);

    /* Reacquire the client mutex */
    opj_mutex_lock(mutex);
}

void opj_cond_signal(opj_cond_t* cond)
{
    opj_cond_waiter_list_t* psIter;

    /* Signal the first registered event, and remove it from the list */
    opj_mutex_lock(cond->internal_mutex);

    psIter = cond->waiter_list;
    if (psIter != NULL) {
        SetEvent(psIter->hEvent);
        cond->waiter_list = psIter->next;
        opj_free(psIter);
    }

    opj_mutex_unlock(cond->internal_mutex);
}

void opj_cond_destroy(opj_cond_t* cond)
{
    if (!cond) {
        return;
    }
    opj_mutex_destroy(cond->internal_mutex);
    assert(cond->waiter_list == NULL);
    opj_free(cond);
}

struct opj_thread_t {
    opj_thread_fn thread_fn;
    void* user_data;
    HANDLE hThread;
};

static unsigned int __stdcall opj_thread_callback_adapter(void *info)
{
    opj_thread_t* thread = (opj_thread_t*) info;
    HANDLE hEvent = NULL;

    thread->thread_fn(thread->user_data);

    /* Free the handle possible allocated by a cond */
    while (OPJ_TRUE) {
        /* Make sure TLSKey is not being created just at that moment... */
#if HAVE_INTERLOCKED_COMPARE_EXCHANGE
        if (InterlockedCompareExchange(&inTLSLockedSection, 1, 0) == 0)
#endif
        {
            if (TLSKeyInit) {
                hEvent = (HANDLE) TlsGetValue(TLSKey);
            }
#if HAVE_INTERLOCKED_COMPARE_EXCHANGE
            InterlockedCompareExchange(&inTLSLockedSection, 0, 1);
#endif
            break;
        }
    }
    if (hEvent) {
        CloseHandle(hEvent);
    }

    return 0;
}

opj_thread_t* opj_thread_create(opj_thread_fn thread_fn, void* user_data)
{
    opj_thread_t* thread;

    assert(thread_fn);

    thread = (opj_thread_t*) opj_malloc(sizeof(opj_thread_t));
    if (!thread) {
        return NULL;
    }
    thread->thread_fn = thread_fn;
    thread->user_data = user_data;

    thread->hThread = (HANDLE)_beginthreadex(NULL, 0,
                      opj_thread_callback_adapter, thread, 0, NULL);

    if (thread->hThread == NULL) {
        opj_free(thread);
        return NULL;
    }
    return thread;
}

void opj_thread_join(opj_thread_t* thread)
{
    WaitForSingleObject(thread->hThread, INFINITE);
    CloseHandle(thread->hThread);

    opj_free(thread);
}

#elif MUTEX_pthread

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

/* Moved after all system includes, and in particular pthread.h, so as to */
/* avoid poisoning issuing with malloc() use in pthread.h with ulibc (#1013) */
#include "opj_includes.h"

OPJ_BOOL OPJ_CALLCONV opj_has_thread_support(void)
{
    return OPJ_TRUE;
}

int OPJ_CALLCONV opj_get_num_cpus(void)
{
#ifdef _SC_NPROCESSORS_ONLN
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    return 1;
#endif
}

struct opj_mutex_t {
    pthread_mutex_t mutex;
};

opj_mutex_t* opj_mutex_create(void)
{
    opj_mutex_t* mutex = (opj_mutex_t*) opj_calloc(1U, sizeof(opj_mutex_t));
    if (mutex != NULL) {
        if (pthread_mutex_init(&mutex->mutex, NULL) != 0) {
            opj_free(mutex);
            mutex = NULL;
        }
    }
    return mutex;
}

void opj_mutex_lock(opj_mutex_t* mutex)
{
    pthread_mutex_lock(&(mutex->mutex));
}

void opj_mutex_unlock(opj_mutex_t* mutex)
{
    pthread_mutex_unlock(&(mutex->mutex));
}

void opj_mutex_destroy(opj_mutex_t* mutex)
{
    if (!mutex) {
        return;
    }
    pthread_mutex_destroy(&(mutex->mutex));
    opj_free(mutex);
}

struct opj_cond_t {
    pthread_cond_t cond;
};

opj_cond_t* opj_cond_create(void)
{
    opj_cond_t* cond = (opj_cond_t*) opj_malloc(sizeof(opj_cond_t));
    if (!cond) {
        return NULL;
    }
    if (pthread_cond_init(&(cond->cond), NULL) != 0) {
        opj_free(cond);
        return NULL;
    }
    return cond;
}

void opj_cond_wait(opj_cond_t* cond, opj_mutex_t* mutex)
{
    pthread_cond_wait(&(cond->cond), &(mutex->mutex));
}

void opj_cond_signal(opj_cond_t* cond)
{
    int ret = pthread_cond_signal(&(cond->cond));
    (void)ret;
    assert(ret == 0);
}

void opj_cond_destroy(opj_cond_t* cond)
{
    if (!cond) {
        return;
    }
    pthread_cond_destroy(&(cond->cond));
    opj_free(cond);
}


struct opj_thread_t {
    opj_thread_fn thread_fn;
    void* user_data;
    pthread_t thread;
};

static void* opj_thread_callback_adapter(void* info)
{
    opj_thread_t* thread = (opj_thread_t*) info;
    thread->thread_fn(thread->user_data);
    return NULL;
}

opj_thread_t* opj_thread_create(opj_thread_fn thread_fn, void* user_data)
{
    pthread_attr_t attr;
    opj_thread_t* thread;

    assert(thread_fn);

    thread = (opj_thread_t*) opj_malloc(sizeof(opj_thread_t));
    if (!thread) {
        return NULL;
    }
    thread->thread_fn = thread_fn;
    thread->user_data = user_data;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    if (pthread_create(&(thread->thread), &attr,
                       opj_thread_callback_adapter, (void *) thread) != 0) {
        opj_free(thread);
        return NULL;
    }
    return thread;
}

void opj_thread_join(opj_thread_t* thread)
{
    void* status;
    pthread_join(thread->thread, &status);

    opj_free(thread);
}

#else
/* Stub implementation */

#include "opj_includes.h"

OPJ_BOOL OPJ_CALLCONV opj_has_thread_support(void)
{
    return OPJ_FALSE;
}

int OPJ_CALLCONV opj_get_num_cpus(void)
{
    return 1;
}

opj_mutex_t* opj_mutex_create(void)
{
    return NULL;
}

void opj_mutex_lock(opj_mutex_t* mutex)
{
    (void) mutex;
}

void opj_mutex_unlock(opj_mutex_t* mutex)
{
    (void) mutex;
}

void opj_mutex_destroy(opj_mutex_t* mutex)
{
    (void) mutex;
}

opj_cond_t* opj_cond_create(void)
{
    return NULL;
}

void opj_cond_wait(opj_cond_t* cond, opj_mutex_t* mutex)
{
    (void) cond;
    (void) mutex;
}

void opj_cond_signal(opj_cond_t* cond)
{
    (void) cond;
}

void opj_cond_destroy(opj_cond_t* cond)
{
    (void) cond;
}

opj_thread_t* opj_thread_create(opj_thread_fn thread_fn, void* user_data)
{
    (void) thread_fn;
    (void) user_data;
    return NULL;
}

void opj_thread_join(opj_thread_t* thread)
{
    (void) thread;
}

#endif

typedef struct {
    int key;
    void* value;
    opj_tls_free_func opj_free_func;
} opj_tls_key_val_t;

struct opj_tls_t {
    opj_tls_key_val_t* key_val;
    int                key_val_count;
};

static opj_tls_t* opj_tls_new(void)
{
    return (opj_tls_t*) opj_calloc(1, sizeof(opj_tls_t));
}

static void opj_tls_destroy(opj_tls_t* tls)
{
    int i;
    if (!tls) {
        return;
    }
    for (i = 0; i < tls->key_val_count; i++) {
        if (tls->key_val[i].opj_free_func) {
            tls->key_val[i].opj_free_func(tls->key_val[i].value);
        }
    }
    opj_free(tls->key_val);
    opj_free(tls);
}

void* opj_tls_get(opj_tls_t* tls, int key)
{
    int i;
    for (i = 0; i < tls->key_val_count; i++) {
        if (tls->key_val[i].key == key) {
            return tls->key_val[i].value;
        }
    }
    return NULL;
}

OPJ_BOOL opj_tls_set(opj_tls_t* tls, int key, void* value,
                     opj_tls_free_func opj_free_func)
{
    opj_tls_key_val_t* new_key_val;
    int i;

    if (tls->key_val_count == INT_MAX) {
        return OPJ_FALSE;
    }
    for (i = 0; i < tls->key_val_count; i++) {
        if (tls->key_val[i].key == key) {
            if (tls->key_val[i].opj_free_func) {
                tls->key_val[i].opj_free_func(tls->key_val[i].value);
            }
            tls->key_val[i].value = value;
            tls->key_val[i].opj_free_func = opj_free_func;
            return OPJ_TRUE;
        }
    }
    new_key_val = (opj_tls_key_val_t*) opj_realloc(tls->key_val,
                  ((size_t)tls->key_val_count + 1U) * sizeof(opj_tls_key_val_t));
    if (!new_key_val) {
        return OPJ_FALSE;
    }
    tls->key_val = new_key_val;
    new_key_val[tls->key_val_count].key = key;
    new_key_val[tls->key_val_count].value = value;
    new_key_val[tls->key_val_count].opj_free_func = opj_free_func;
    tls->key_val_count ++;
    return OPJ_TRUE;
}


typedef struct {
    opj_job_fn          job_fn;
    void               *user_data;
} opj_worker_thread_job_t;

typedef struct {
    opj_thread_pool_t   *tp;
    opj_thread_t        *thread;
    int                  marked_as_waiting;

    opj_mutex_t         *mutex;
    opj_cond_t          *cond;
} opj_worker_thread_t;

typedef enum {
    OPJWTS_OK,
    OPJWTS_STOP,
    OPJWTS_ERROR
} opj_worker_thread_state;

struct opj_job_list_t {
    opj_worker_thread_job_t* job;
    struct opj_job_list_t* next;
};
typedef struct opj_job_list_t opj_job_list_t;

struct opj_worker_thread_list_t {
    opj_worker_thread_t* worker_thread;
    struct opj_worker_thread_list_t* next;
};
typedef struct opj_worker_thread_list_t opj_worker_thread_list_t;

struct opj_thread_pool_t {
    opj_worker_thread_t*             worker_threads;
    int                              worker_threads_count;
    opj_cond_t*                      cond;
    opj_mutex_t*                     mutex;
    volatile opj_worker_thread_state state;
    opj_job_list_t*                  job_queue;
    volatile int                     pending_jobs_count;
    opj_worker_thread_list_t*        waiting_worker_thread_list;
    int                              waiting_worker_thread_count;
    opj_tls_t*                       tls;
    int                              signaling_threshold;
};

static OPJ_BOOL opj_thread_pool_setup(opj_thread_pool_t* tp, int num_threads);
static opj_worker_thread_job_t* opj_thread_pool_get_next_job(
    opj_thread_pool_t* tp,
    opj_worker_thread_t* worker_thread,
    OPJ_BOOL signal_job_finished);

opj_thread_pool_t* opj_thread_pool_create(int num_threads)
{
    opj_thread_pool_t* tp;

    tp = (opj_thread_pool_t*) opj_calloc(1, sizeof(opj_thread_pool_t));
    if (!tp) {
        return NULL;
    }
    tp->state = OPJWTS_OK;

    if (num_threads <= 0) {
        tp->tls = opj_tls_new();
        if (!tp->tls) {
            opj_free(tp);
            tp = NULL;
        }
        return tp;
    }

    tp->mutex = opj_mutex_create();
    if (!tp->mutex) {
        opj_free(tp);
        return NULL;
    }
    if (!opj_thread_pool_setup(tp, num_threads)) {
        opj_thread_pool_destroy(tp);
        return NULL;
    }
    return tp;
}

static void opj_worker_thread_function(void* user_data)
{
    opj_worker_thread_t* worker_thread;
    opj_thread_pool_t* tp;
    opj_tls_t* tls;
    OPJ_BOOL job_finished = OPJ_FALSE;

    worker_thread = (opj_worker_thread_t*) user_data;
    tp = worker_thread->tp;
    tls = opj_tls_new();

    while (OPJ_TRUE) {
        opj_worker_thread_job_t* job = opj_thread_pool_get_next_job(tp, worker_thread,
                                       job_finished);
        if (job == NULL) {
            break;
        }

        if (job->job_fn) {
            job->job_fn(job->user_data, tls);
        }
        opj_free(job);
        job_finished = OPJ_TRUE;
    }

    opj_tls_destroy(tls);
}

static OPJ_BOOL opj_thread_pool_setup(opj_thread_pool_t* tp, int num_threads)
{
    int i;
    OPJ_BOOL bRet = OPJ_TRUE;

    assert(num_threads > 0);

    tp->cond = opj_cond_create();
    if (tp->cond == NULL) {
        return OPJ_FALSE;
    }

    tp->worker_threads = (opj_worker_thread_t*) opj_calloc((size_t)num_threads,
                         sizeof(opj_worker_thread_t));
    if (tp->worker_threads == NULL) {
        return OPJ_FALSE;
    }
    tp->worker_threads_count = num_threads;

    for (i = 0; i < num_threads; i++) {
        tp->worker_threads[i].tp = tp;

        tp->worker_threads[i].mutex = opj_mutex_create();
        if (tp->worker_threads[i].mutex == NULL) {
            tp->worker_threads_count = i;
            bRet = OPJ_FALSE;
            break;
        }

        tp->worker_threads[i].cond = opj_cond_create();
        if (tp->worker_threads[i].cond == NULL) {
            opj_mutex_destroy(tp->worker_threads[i].mutex);
            tp->worker_threads_count = i;
            bRet = OPJ_FALSE;
            break;
        }

        tp->worker_threads[i].marked_as_waiting = OPJ_FALSE;

        tp->worker_threads[i].thread = opj_thread_create(opj_worker_thread_function,
                                       &(tp->worker_threads[i]));
        if (tp->worker_threads[i].thread == NULL) {
            opj_mutex_destroy(tp->worker_threads[i].mutex);
            opj_cond_destroy(tp->worker_threads[i].cond);
            tp->worker_threads_count = i;
            bRet = OPJ_FALSE;
            break;
        }
    }

    /* Wait all threads to be started */
    /* printf("waiting for all threads to be started\n"); */
    opj_mutex_lock(tp->mutex);
    while (tp->waiting_worker_thread_count < tp->worker_threads_count) {
        opj_cond_wait(tp->cond, tp->mutex);
    }
    opj_mutex_unlock(tp->mutex);
    /* printf("all threads started\n"); */

    if (tp->state == OPJWTS_ERROR) {
        bRet = OPJ_FALSE;
    }

    return bRet;
}

/*
void opj_waiting()
{
    printf("waiting!\n");
}
*/

static opj_worker_thread_job_t* opj_thread_pool_get_next_job(
    opj_thread_pool_t* tp,
    opj_worker_thread_t* worker_thread,
    OPJ_BOOL signal_job_finished)
{
    while (OPJ_TRUE) {
        opj_job_list_t* top_job_iter;

        opj_mutex_lock(tp->mutex);

        if (signal_job_finished) {
            signal_job_finished = OPJ_FALSE;
            tp->pending_jobs_count --;
            /*printf("tp=%p, remaining jobs: %d\n", tp, tp->pending_jobs_count);*/
            if (tp->pending_jobs_count <= tp->signaling_threshold) {
                opj_cond_signal(tp->cond);
            }
        }

        if (tp->state == OPJWTS_STOP) {
            opj_mutex_unlock(tp->mutex);
            return NULL;
        }
        top_job_iter = tp->job_queue;
        if (top_job_iter) {
            opj_worker_thread_job_t* job;
            tp->job_queue = top_job_iter->next;

            job = top_job_iter->job;
            opj_mutex_unlock(tp->mutex);
            opj_free(top_job_iter);
            return job;
        }

        /* opj_waiting(); */
        if (!worker_thread->marked_as_waiting) {
            opj_worker_thread_list_t* item;

            worker_thread->marked_as_waiting = OPJ_TRUE;
            tp->waiting_worker_thread_count ++;
            assert(tp->waiting_worker_thread_count <= tp->worker_threads_count);

            item = (opj_worker_thread_list_t*) opj_malloc(sizeof(opj_worker_thread_list_t));
            if (item == NULL) {
                tp->state = OPJWTS_ERROR;
                opj_cond_signal(tp->cond);

                opj_mutex_unlock(tp->mutex);
                return NULL;
            }

            item->worker_thread = worker_thread;
            item->next = tp->waiting_worker_thread_list;
            tp->waiting_worker_thread_list = item;
        }

        /* printf("signaling that worker thread is ready\n"); */
        opj_cond_signal(tp->cond);

        opj_mutex_lock(worker_thread->mutex);
        opj_mutex_unlock(tp->mutex);

        /* printf("waiting for job\n"); */
        opj_cond_wait(worker_thread->cond, worker_thread->mutex);

        opj_mutex_unlock(worker_thread->mutex);
        /* printf("got job\n"); */
    }
}

OPJ_BOOL opj_thread_pool_submit_job(opj_thread_pool_t* tp,
                                    opj_job_fn job_fn,
                                    void* user_data)
{
    opj_worker_thread_job_t* job;
    opj_job_list_t* item;

    if (tp->mutex == NULL) {
        job_fn(user_data, tp->tls);
        return OPJ_TRUE;
    }

    job = (opj_worker_thread_job_t*)opj_malloc(sizeof(opj_worker_thread_job_t));
    if (job == NULL) {
        return OPJ_FALSE;
    }
    job->job_fn = job_fn;
    job->user_data = user_data;

    item = (opj_job_list_t*) opj_malloc(sizeof(opj_job_list_t));
    if (item == NULL) {
        opj_free(job);
        return OPJ_FALSE;
    }
    item->job = job;

    opj_mutex_lock(tp->mutex);

    tp->signaling_threshold = 100 * tp->worker_threads_count;
    while (tp->pending_jobs_count > tp->signaling_threshold) {
        /* printf("%d jobs enqueued. Waiting\n", tp->pending_jobs_count); */
        opj_cond_wait(tp->cond, tp->mutex);
        /* printf("...%d jobs enqueued.\n", tp->pending_jobs_count); */
    }

    item->next = tp->job_queue;
    tp->job_queue = item;
    tp->pending_jobs_count ++;

    if (tp->waiting_worker_thread_list) {
        opj_worker_thread_t* worker_thread;
        opj_worker_thread_list_t* next;
        opj_worker_thread_list_t* to_opj_free;

        worker_thread = tp->waiting_worker_thread_list->worker_thread;

        assert(worker_thread->marked_as_waiting);
        worker_thread->marked_as_waiting = OPJ_FALSE;

        next = tp->waiting_worker_thread_list->next;
        to_opj_free = tp->waiting_worker_thread_list;
        tp->waiting_worker_thread_list = next;
        tp->waiting_worker_thread_count --;

        opj_mutex_lock(worker_thread->mutex);
        opj_mutex_unlock(tp->mutex);
        opj_cond_signal(worker_thread->cond);
        opj_mutex_unlock(worker_thread->mutex);

        opj_free(to_opj_free);
    } else {
        opj_mutex_unlock(tp->mutex);
    }

    return OPJ_TRUE;
}

void opj_thread_pool_wait_completion(opj_thread_pool_t* tp,
                                     int max_remaining_jobs)
{
    if (tp->mutex == NULL) {
        return;
    }

    if (max_remaining_jobs < 0) {
        max_remaining_jobs = 0;
    }
    opj_mutex_lock(tp->mutex);
    tp->signaling_threshold = max_remaining_jobs;
    while (tp->pending_jobs_count > max_remaining_jobs) {
        /*printf("tp=%p, jobs before wait = %d, max_remaining_jobs = %d\n", tp, tp->pending_jobs_count, max_remaining_jobs);*/
        opj_cond_wait(tp->cond, tp->mutex);
        /*printf("tp=%p, jobs after wait = %d\n", tp, tp->pending_jobs_count);*/
    }
    opj_mutex_unlock(tp->mutex);
}

int opj_thread_pool_get_thread_count(opj_thread_pool_t* tp)
{
    return tp->worker_threads_count;
}

void opj_thread_pool_destroy(opj_thread_pool_t* tp)
{
    if (!tp) {
        return;
    }
    if (tp->cond) {
        int i;
        opj_thread_pool_wait_completion(tp, 0);

        opj_mutex_lock(tp->mutex);
        tp->state = OPJWTS_STOP;
        opj_mutex_unlock(tp->mutex);

        for (i = 0; i < tp->worker_threads_count; i++) {
            opj_mutex_lock(tp->worker_threads[i].mutex);
            opj_cond_signal(tp->worker_threads[i].cond);
            opj_mutex_unlock(tp->worker_threads[i].mutex);
            opj_thread_join(tp->worker_threads[i].thread);
            opj_cond_destroy(tp->worker_threads[i].cond);
            opj_mutex_destroy(tp->worker_threads[i].mutex);
        }

        opj_free(tp->worker_threads);

        while (tp->waiting_worker_thread_list != NULL) {
            opj_worker_thread_list_t* next = tp->waiting_worker_thread_list->next;
            opj_free(tp->waiting_worker_thread_list);
            tp->waiting_worker_thread_list = next;
        }

        opj_cond_destroy(tp->cond);
    }
    opj_mutex_destroy(tp->mutex);
    opj_tls_destroy(tp->tls);
    opj_free(tp);
}
