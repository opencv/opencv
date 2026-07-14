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

#ifndef THREAD_H
#define THREAD_H

#include "openjpeg.h"

/**
@file thread.h
@brief Thread API

The functions in thread.c have for goal to manage mutex, conditions, thread
creation and thread pools that accept jobs.
*/

/** @defgroup THREAD THREAD - Mutex, conditions, threads and thread pools */
/*@{*/

/** @name Mutex */
/*@{*/

/** Opaque type for a mutex */
typedef struct opj_mutex_t opj_mutex_t;

/** Creates a mutex.
 * @return the mutex or NULL in case of error (can for example happen if the library
 * is built without thread support)
 */
opj_mutex_t* opj_mutex_create(void);

/** Lock/acquire the mutex.
 * @param mutex the mutex to acquire.
 */
void opj_mutex_lock(opj_mutex_t* mutex);

/** Unlock/release the mutex.
 * @param mutex the mutex to release.
 */
void opj_mutex_unlock(opj_mutex_t* mutex);

/** Destroy a mutex
 * @param mutex the mutex to destroy.
 */
void opj_mutex_destroy(opj_mutex_t* mutex);

/*@}*/

/** @name Condition */
/*@{*/

/** Opaque type for a condition */
typedef struct opj_cond_t opj_cond_t;

/** Creates a condition.
 * @return the condition or NULL in case of error (can for example happen if the library
 * is built without thread support)
 */
opj_cond_t* opj_cond_create(void);

/** Wait for the condition to be signaled.
 * The semantics is the same as the POSIX pthread_cond_wait.
 * The provided mutex *must* be acquired before calling this function, and
 * released afterwards.
 * The mutex will be released by this function while it must wait for the condition
 * and reacquired afterwards.
 * In some particular situations, the function might return even if the condition is not signaled
 * with opj_cond_signal(), hence the need to check with an application level
 * mechanism.
 *
 * Waiting thread :
 * \code
 *    opj_mutex_lock(mutex);
 *    while( !some_application_level_condition )
 *    {
 *        opj_cond_wait(cond, mutex);
 *    }
 *    opj_mutex_unlock(mutex);
 * \endcode
 *
 * Signaling thread :
 * \code
 *    opj_mutex_lock(mutex);
 *    some_application_level_condition = TRUE;
 *    opj_cond_signal(cond);
 *    opj_mutex_unlock(mutex);
 * \endcode
 *
 * @param cond the condition to wait.
 * @param mutex the mutex (in acquired state before calling this function)
 */
void opj_cond_wait(opj_cond_t* cond, opj_mutex_t* mutex);

/** Signal waiting threads on a condition.
 * One of the thread waiting with opj_cond_wait() will be waken up.
 * It is strongly advised that this call is done with the mutex that is used
 * by opj_cond_wait(), in a acquired state.
 * @param cond the condition to signal.
 */
void opj_cond_signal(opj_cond_t* cond);

/** Destroy a condition
 * @param cond the condition to destroy.
 */
void opj_cond_destroy(opj_cond_t* cond);

/*@}*/

/** @name Thread */
/*@{*/

/** Opaque type for a thread handle */
typedef struct opj_thread_t opj_thread_t;

/** User function to execute in a thread
 * @param user_data user data provided with opj_thread_create()
 */
typedef void (*opj_thread_fn)(void* user_data);

/** Creates a new thread.
 * @param thread_fn Function to run in the new thread.
 * @param user_data user data provided to the thread function. Might be NULL.
 * @return a thread handle or NULL in case of failure (can for example happen if the library
 * is built without thread support)
 */
opj_thread_t* opj_thread_create(opj_thread_fn thread_fn, void* user_data);

/** Wait for a thread to be finished and release associated resources to the
 * thread handle.
 * @param thread the thread to wait for being finished.
 */
void opj_thread_join(opj_thread_t* thread);

/*@}*/

/** @name Thread local storage */
/*@{*/
/** Opaque type for a thread local storage */
typedef struct opj_tls_t opj_tls_t;

/** Get a thread local value corresponding to the provided key.
 * @param tls thread local storage handle
 * @param key key whose value to retrieve.
 * @return value associated with the key, or NULL is missing.
 */
void* opj_tls_get(opj_tls_t* tls, int key);

/** Type of the function used to free a TLS value */
typedef void (*opj_tls_free_func)(void* value);

/** Set a thread local value corresponding to the provided key.
 * @param tls thread local storage handle
 * @param key key whose value to set.
 * @param value value to set (may be NULL).
 * @param free_func function to call currently installed value.
 * @return OPJ_TRUE if successful.
 */
OPJ_BOOL opj_tls_set(opj_tls_t* tls, int key, void* value,
                     opj_tls_free_func free_func);

/*@}*/

/** @name Thread pool */
/*@{*/

/** Opaque type for a thread pool */
typedef struct opj_thread_pool_t opj_thread_pool_t;

/** Create a new thread pool.
 * num_thread must nominally be >= 1 to create a real thread pool. If num_threads
 * is negative or null, then a dummy thread pool will be created. All functions
 * operating on the thread pool will work, but job submission will be run
 * synchronously in the calling thread.
 *
 * @param num_threads the number of threads to allocate for this thread pool.
 * @return a thread pool handle, or NULL in case of failure (can for example happen if the library
 * is built without thread support)
 */
opj_thread_pool_t* opj_thread_pool_create(int num_threads);

/** User function to execute in a thread
 * @param user_data user data provided with opj_thread_create()
 * @param tls handle to thread local storage
 */
typedef void (*opj_job_fn)(void* user_data, opj_tls_t* tls);


/** Submit a new job to be run by one of the thread in the thread pool.
 * The job ( thread_fn, user_data ) will be added in the queue of jobs managed
 * by the thread pool, and run by the first thread that is no longer busy.
 *
 * @param tp the thread pool handle.
 * @param job_fn Function to run. Must not be NULL.
 * @param user_data User data provided to thread_fn.
 * @return OPJ_TRUE if the job was successfully submitted.
 */
OPJ_BOOL opj_thread_pool_submit_job(opj_thread_pool_t* tp, opj_job_fn job_fn,
                                    void* user_data);

/** Wait that no more than max_remaining_jobs jobs are remaining in the queue of
 * the thread pool. The aim of this function is to avoid submitting too many
 * jobs while the thread pool cannot cope fast enough with them, which would
 * result potentially in out-of-memory situations with too many job descriptions
 * being queued.
 *
 * @param tp the thread pool handle
 * @param max_remaining_jobs maximum number of jobs allowed to be queued without waiting.
 */
void opj_thread_pool_wait_completion(opj_thread_pool_t* tp,
                                     int max_remaining_jobs);

/** Return the number of threads associated with the thread pool.
 *
 * @param tp the thread pool handle.
 * @return number of threads associated with the thread pool.
 */
int opj_thread_pool_get_thread_count(opj_thread_pool_t* tp);

/** Destroy a thread pool.
 * @param tp the thread pool handle.
 */
void opj_thread_pool_destroy(opj_thread_pool_t* tp);

/*@}*/

/*@}*/

#endif /* THREAD_H */
