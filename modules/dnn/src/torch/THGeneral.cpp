#include "../precomp.hpp"
#if defined(ENABLE_TORCH_IMPORTER) && ENABLE_TORCH_IMPORTER
#include <opencv2/core.hpp>

#if defined(TH_DISABLE_HEAP_TRACKING)
#elif (defined(__unix) || defined(_WIN32))
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

#include "THGeneral.h"

extern "C"
{

#ifndef TH_HAVE_THREAD
#define TH_THREAD
#else
#define TH_THREAD __thread
#endif

/* Torch Error Handling */
static void defaultTorchErrorHandlerFunction(const char *msg, void*)
{
  CV_Error(cv::Error::StsError, cv::String("Torch Error: ") + msg);
}

static TH_THREAD void (*torchErrorHandlerFunction)(const char *msg, void *data) = defaultTorchErrorHandlerFunction;
static TH_THREAD void *torchErrorHandlerData;

void _THError(const char *file, const int line, const char *fmt, ...)
{
  char msg[2048];
  va_list args;

  /* vasprintf not standard */
  /* vsnprintf: how to handle if does not exists? */
  va_start(args, fmt);
  int n = vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  if(n < 2048) {
    snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
  }

  (*torchErrorHandlerFunction)(msg, torchErrorHandlerData);
}

void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...) {
  char msg[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);
  _THError(file, line, "Assertion `%s' failed. %s", exp, msg);
}

void THSetErrorHandler( void (*torchErrorHandlerFunction_)(const char *msg, void *data), void *data )
{
  if(torchErrorHandlerFunction_)
    torchErrorHandlerFunction = torchErrorHandlerFunction_;
  else
    torchErrorHandlerFunction = defaultTorchErrorHandlerFunction;
  torchErrorHandlerData = data;
}

/* Torch Arg Checking Handling */
static void defaultTorchArgErrorHandlerFunction(int argNumber, const char *msg, void*)
{
  if(msg)
    CV_Error(cv::Error::StsError, cv::format("Torch invalid argument %d: %s", argNumber, msg));
  else
    CV_Error(cv::Error::StsError, cv::format("Invalid argument %d", argNumber));
}

static TH_THREAD void (*torchArgErrorHandlerFunction)(int argNumber, const char *msg, void *data) = defaultTorchArgErrorHandlerFunction;
static TH_THREAD void *torchArgErrorHandlerData;

void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...)
{
  if(!condition) {
    char msg[2048];
    va_list args;

    /* vasprintf not standard */
    /* vsnprintf: how to handle if does not exists? */
    va_start(args, fmt);
    int n = vsnprintf(msg, 2048, fmt, args);
    va_end(args);

    if(n < 2048) {
      snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
    }

    (*torchArgErrorHandlerFunction)(argNumber, msg, torchArgErrorHandlerData);
  }
}

void THSetArgErrorHandler( void (*torchArgErrorHandlerFunction_)(int argNumber, const char *msg, void *data), void *data )
{
  if(torchArgErrorHandlerFunction_)
    torchArgErrorHandlerFunction = torchArgErrorHandlerFunction_;
  else
    torchArgErrorHandlerFunction = defaultTorchArgErrorHandlerFunction;
  torchArgErrorHandlerData = data;
}

static TH_THREAD void (*torchGCFunction)(void *data) = NULL;
static TH_THREAD void *torchGCData;
static TH_THREAD long torchHeapSize = 0;
static TH_THREAD long torchHeapSizeSoftMax = 300000000; // 300MB, adjusted upward dynamically

/* Optional hook for integrating with a garbage-collected frontend.
 *
 * If torch is running with a garbage-collected frontend (e.g. Lua),
 * the GC isn't aware of TH-allocated memory so may not know when it
 * needs to run. These hooks trigger the GC to run in two cases:
 *
 * (1) When a memory allocation (malloc, realloc, ...) fails
 * (2) When the total TH-allocated memory hits a dynamically-adjusted
 *     soft maximum.
 */
void THSetGCHandler( void (*torchGCFunction_)(void *data), void *data )
{
  torchGCFunction = torchGCFunction_;
  torchGCData = data;
}

static long getAllocSize(void *ptr) {
#if defined(TH_DISABLE_HEAP_TRACKING)
  return 0;
#elif defined(__unix)
  return malloc_usable_size(ptr);
#elif defined(__APPLE__)
  return malloc_size(ptr);
#elif defined(_WIN32)
  return _msize(ptr);
#else
  return 0;
#endif
}

/* (1) if the torch-allocated heap size exceeds the soft max, run GC
 * (2) if post-GC heap size exceeds 80% of the soft max, increase the
 *     soft max by 40%
 */
static void maybeTriggerGC() {
  if(torchGCFunction && torchHeapSize > torchHeapSizeSoftMax) {
    torchGCFunction(torchGCData);
    if(torchHeapSize > torchHeapSizeSoftMax * 0.8) {
      torchHeapSizeSoftMax = torchHeapSizeSoftMax * 1.4;
    }
  }
}

// hooks into the TH heap tracking
void THHeapUpdate(long size) {
  torchHeapSize += size;
  if (size > 0)
    maybeTriggerGC();
}

static void* THAllocInternal(long size)
{
  void *ptr;

  if (size > 5120)
  {
#if (defined(__unix) || defined(__APPLE__)) && (!defined(DISABLE_POSIX_MEMALIGN))
    if (posix_memalign(&ptr, 64, size) != 0)
      ptr = NULL;
/*
#elif defined(_WIN32)
    ptr = _aligned_malloc(size, 64);
*/
#else
    ptr = malloc(size);
#endif
  }
  else
  {
    ptr = malloc(size);
  }

  THHeapUpdate(getAllocSize(ptr));
  return ptr;
}

void* THAlloc(long size)
{
  void *ptr;

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  if(size == 0)
    return NULL;

  ptr = THAllocInternal(size);

  if(!ptr && torchGCFunction) {
    torchGCFunction(torchGCData);
    ptr = THAllocInternal(size);
  }

  if(!ptr)
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", size/1073741824);

  return ptr;
}

void* THRealloc(void *ptr, long size)
{
  if(!ptr)
    return(THAlloc(size));

  if(size == 0)
  {
    THFree(ptr);
    return NULL;
  }

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  THHeapUpdate(-getAllocSize(ptr));
  void *newptr = realloc(ptr, size);

  if(!newptr && torchGCFunction) {
    torchGCFunction(torchGCData);
    newptr = realloc(ptr, size);
  }
  THHeapUpdate(getAllocSize(newptr ? newptr : ptr));

  if(!newptr)
    THError("$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);

  return newptr;
}

void THFree(void *ptr)
{
  THHeapUpdate(-getAllocSize(ptr));
  free(ptr);
}

}
#endif
