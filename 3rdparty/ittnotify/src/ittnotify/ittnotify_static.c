/*
  Copyright (C) 2005-2019 Intel Corporation

  SPDX-License-Identifier: GPL-2.0-only OR BSD-3-Clause
*/

#define INTEL_NO_MACRO_BODY
#define INTEL_ITTNOTIFY_API_PRIVATE
#include "ittnotify_config.h"

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#if !defined(PATH_MAX)
#define PATH_MAX 512
#endif
#else /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
#include <limits.h>
#include <dlfcn.h>
#include <errno.h>
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "ittnotify.h"
#include "legacy/ittnotify.h"

#include "disable_warnings.h"

static const char api_version[] = API_VERSION "\0\n@(#) $Revision$\n";

#define _N_(n) ITT_JOIN(INTEL_ITTNOTIFY_PREFIX,n)

#ifndef HAS_CPP_ATTR
#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define HAS_CPP_ATTR(X) __has_cpp_attribute(X)
#else
#define HAS_CPP_ATTR(X) 0
#endif
#endif

#ifndef HAS_C_ATTR
#if defined(__STDC__) && defined(__has_c_attribute)
#define HAS_C_ATTR(X) __has_c_attribute(X)
#else
#define HAS_C_ATTR(X) 0
#endif
#endif

#ifndef HAS_GNU_ATTR
#if defined(__has_attribute)
#define HAS_GNU_ATTR(X) __has_attribute(X)
#else
#define HAS_GNU_ATTR(X) 0
#endif
#endif

#ifndef ITT_ATTRIBUTE_FALLTHROUGH
#if (HAS_CPP_ATTR(fallthrough) || HAS_C_ATTR(fallthrough)) && (__cplusplus >= 201703L || _MSVC_LANG >= 201703L)
#define ITT_ATTRIBUTE_FALLTHROUGH [[fallthrough]]
#elif HAS_CPP_ATTR(gnu::fallthrough)
#define ITT_ATTRIBUTE_FALLTHROUGH [[gnu::fallthrough]]
#elif HAS_CPP_ATTR(clang::fallthrough)
#define ITT_ATTRIBUTE_FALLTHROUGH [[clang::fallthrough]]
#elif HAS_GNU_ATTR(fallthrough) && !__INTEL_COMPILER
#define ITT_ATTRIBUTE_FALLTHROUGH __attribute__((fallthrough))
#else
#define ITT_ATTRIBUTE_FALLTHROUGH
#endif
#endif

#if ITT_OS==ITT_OS_WIN
static const char* ittnotify_lib_name = "libittnotify.dll";
#elif ITT_OS==ITT_OS_LINUX || ITT_OS==ITT_OS_FREEBSD || ITT_OS==ITT_OS_OPENBSD
static const char* ittnotify_lib_name = "libittnotify.so";
#elif ITT_OS==ITT_OS_MAC
static const char* ittnotify_lib_name = "libittnotify.dylib";
#else
#error Unsupported or unknown OS.
#endif

#ifdef __ANDROID__
#include <android/log.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/limits.h>

#ifdef ITT_ANDROID_LOG
    #define ITT_ANDROID_LOG_TAG   "INTEL_VTUNE_USERAPI"
    #define ITT_ANDROID_LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, ITT_ANDROID_LOG_TAG, __VA_ARGS__))
    #define ITT_ANDROID_LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, ITT_ANDROID_LOG_TAG, __VA_ARGS__))
    #define ITT_ANDROID_LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR,ITT_ANDROID_LOG_TAG, __VA_ARGS__))
    #define ITT_ANDROID_LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG,ITT_ANDROID_LOG_TAG, __VA_ARGS__))
#else
    #define ITT_ANDROID_LOGI(...)
    #define ITT_ANDROID_LOGW(...)
    #define ITT_ANDROID_LOGE(...)
    #define ITT_ANDROID_LOGD(...)
#endif

/* default location of userapi collector on Android */
#define ANDROID_ITTNOTIFY_DEFAULT_PATH_MASK(x)  "/data/data/com.intel.vtune/perfrun/lib" \
                                                #x "/runtime/libittnotify.so"

#if ITT_ARCH==ITT_ARCH_IA32 || ITT_ARCH==ITT_ARCH_ARM
#define ANDROID_ITTNOTIFY_DEFAULT_PATH  ANDROID_ITTNOTIFY_DEFAULT_PATH_MASK(32)
#else
#define ANDROID_ITTNOTIFY_DEFAULT_PATH  ANDROID_ITTNOTIFY_DEFAULT_PATH_MASK(64)
#endif

#endif


#ifndef LIB_VAR_NAME
#if ITT_ARCH==ITT_ARCH_IA32 || ITT_ARCH==ITT_ARCH_ARM
#define LIB_VAR_NAME INTEL_LIBITTNOTIFY32
#else
#define LIB_VAR_NAME INTEL_LIBITTNOTIFY64
#endif
#endif /* LIB_VAR_NAME */

#define ITT_MUTEX_INIT_AND_LOCK(p) {                                 \
    if (PTHREAD_SYMBOLS)                                             \
    {                                                                \
        if (!p.mutex_initialized)                                    \
        {                                                            \
            if (__itt_interlocked_compare_exchange(&p.atomic_counter, 1, 0) == 0) \
            {                                                        \
                __itt_mutex_init(&p.mutex);                          \
                p.mutex_initialized = 1;                             \
            }                                                        \
            else                                                     \
                while (!p.mutex_initialized)                         \
                    __itt_thread_yield();                            \
        }                                                            \
        __itt_mutex_lock(&p.mutex);                                  \
    }                                                                \
}

#define ITT_MUTEX_DESTROY(p) {                                       \
    if (PTHREAD_SYMBOLS)                                             \
    {                                                                \
        if (p.mutex_initialized)                                     \
        {                                                            \
            if (__itt_interlocked_compare_exchange(&p.atomic_counter, 0, 1) == 1) \
            {                                                        \
                __itt_mutex_destroy(&p.mutex);                       \
                p.mutex_initialized = 0;                             \
            }                                                        \
        }                                                            \
    }                                                                \
}

#define ITT_MODULE_OBJECT_VERSION 1

typedef int (__itt_init_ittlib_t)(const char*, __itt_group_id);

/* this define used to control initialization function name. */
#ifndef __itt_init_ittlib_name
ITT_EXTERN_C int _N_(init_ittlib)(const char*, __itt_group_id);
static __itt_init_ittlib_t* __itt_init_ittlib_ptr = _N_(init_ittlib);
#define __itt_init_ittlib_name __itt_init_ittlib_ptr
#endif /* __itt_init_ittlib_name */

typedef void (__itt_fini_ittlib_t)(void);

/* this define used to control finalization function name. */
#ifndef __itt_fini_ittlib_name
ITT_EXTERN_C void _N_(fini_ittlib)(void);
static __itt_fini_ittlib_t* __itt_fini_ittlib_ptr = _N_(fini_ittlib);
#define __itt_fini_ittlib_name __itt_fini_ittlib_ptr
#endif /* __itt_fini_ittlib_name */

extern __itt_global _N_(_ittapi_global);

/* building pointers to imported funcs */
#undef ITT_STUBV
#undef ITT_STUB
#define ITT_STUB(api,type,name,args,params,ptr,group,format)   \
static type api ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)) args;\
typedef type api ITT_JOIN(_N_(name),_t) args;                  \
ITT_EXTERN_C_BEGIN ITT_JOIN(_N_(name),_t)* ITTNOTIFY_NAME(name) = ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)); ITT_EXTERN_C_END \
static type api ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)) args \
{                                                              \
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL) \
        __itt_init_ittlib_name(NULL, __itt_group_all);         \
    if (ITTNOTIFY_NAME(name) && ITTNOTIFY_NAME(name) != ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init))) \
        return ITTNOTIFY_NAME(name) params;                    \
    else                                                       \
        return (type)0;                                        \
}

#define ITT_STUBV(api,type,name,args,params,ptr,group,format)  \
static type api ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)) args;\
typedef type api ITT_JOIN(_N_(name),_t) args;                  \
ITT_EXTERN_C_BEGIN ITT_JOIN(_N_(name),_t)* ITTNOTIFY_NAME(name) = ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)); ITT_EXTERN_C_END \
static type api ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)) args \
{                                                              \
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL) \
        __itt_init_ittlib_name(NULL, __itt_group_all);         \
    if (ITTNOTIFY_NAME(name) && ITTNOTIFY_NAME(name) != ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init))) \
        ITTNOTIFY_NAME(name) params;                           \
    else                                                       \
        return;                                                \
}

#undef __ITT_INTERNAL_INIT
#include "ittnotify_static.h"

#undef ITT_STUB
#undef ITT_STUBV
#define ITT_STUB(api,type,name,args,params,ptr,group,format)   \
static type api ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)) args;\
typedef type api ITT_JOIN(_N_(name),_t) args;                  \
ITT_EXTERN_C_BEGIN ITT_JOIN(_N_(name),_t)* ITTNOTIFY_NAME(name) = ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)); ITT_EXTERN_C_END

#define ITT_STUBV(api,type,name,args,params,ptr,group,format)  \
static type api ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)) args;\
typedef type api ITT_JOIN(_N_(name),_t) args;                  \
ITT_EXTERN_C_BEGIN ITT_JOIN(_N_(name),_t)* ITTNOTIFY_NAME(name) = ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)); ITT_EXTERN_C_END

#define __ITT_INTERNAL_INIT
#include "ittnotify_static.h"
#undef __ITT_INTERNAL_INIT

ITT_GROUP_LIST(group_list);

#pragma pack(push, 8)

typedef struct ___itt_group_alias
{
    const char*    env_var;
    __itt_group_id groups;
} __itt_group_alias;

static __itt_group_alias group_alias[] = {
    { "KMP_FOR_TPROFILE", (__itt_group_id)(__itt_group_control | __itt_group_thread | __itt_group_sync  | __itt_group_mark) },
    { "KMP_FOR_TCHECK",   (__itt_group_id)(__itt_group_control | __itt_group_thread | __itt_group_sync  | __itt_group_fsync | __itt_group_mark | __itt_group_suppress) },
    { NULL,               (__itt_group_none) },
    { api_version,        (__itt_group_none) } /* !!! Just to avoid unused code elimination !!! */
};

#pragma pack(pop)

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#if _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4054) /* warning C4054: 'type cast' : from function pointer 'XXX' to data pointer 'void *' */
#endif
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

static __itt_api_info api_list[] = {
/* Define functions with static implementation */
#undef ITT_STUB
#undef ITT_STUBV
#define ITT_STUB(api,type,name,args,params,nameindll,group,format) { ITT_TO_STR(ITT_JOIN(__itt_,nameindll)), (void**)(void*)&ITTNOTIFY_NAME(name), (void*)(size_t)&ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)), (void*)(size_t)&ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)), (__itt_group_id)(group)},
#define ITT_STUBV ITT_STUB
#define __ITT_INTERNAL_INIT
#include "ittnotify_static.h"
#undef __ITT_INTERNAL_INIT
/* Define functions without static implementation */
#undef ITT_STUB
#undef ITT_STUBV
#define ITT_STUB(api,type,name,args,params,nameindll,group,format) {ITT_TO_STR(ITT_JOIN(__itt_,nameindll)), (void**)(void*)&ITTNOTIFY_NAME(name), (void*)(size_t)&ITT_VERSIONIZE(ITT_JOIN(_N_(name),_init)), NULL, (__itt_group_id)(group)},
#define ITT_STUBV ITT_STUB
#include "ittnotify_static.h"
    {NULL, NULL, NULL, NULL, __itt_group_none}
};

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#if _MSC_VER
#pragma warning(pop)
#endif
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/* static part descriptor which handles. all notification api attributes. */
__itt_global _N_(_ittapi_global) = {
    ITT_MAGIC,                                     /* identification info */
    ITT_MAJOR, ITT_MINOR, API_VERSION_BUILD,       /* version info */
    0,                                             /* api_initialized */
    0,                                             /* mutex_initialized */
    0,                                             /* atomic_counter */
    MUTEX_INITIALIZER,                             /* mutex */
    NULL,                                          /* dynamic library handle */
    NULL,                                          /* error_handler */
    NULL,                                          /* dll_path_ptr */
    (__itt_api_info*)&api_list,                    /* api_list_ptr */
    NULL,                                          /* next __itt_global */
    NULL,                                          /* thread_list */
    NULL,                                          /* domain_list */
    NULL,                                          /* string_list */
    __itt_collection_uninitialized,                /* collection state */
    NULL,                                          /* counter_list */
    0,                                             /* ipt_collect_events */
    NULL,                                          /* histogram_list */
    NULL                                           /* counter_metadata_list */
};

typedef void (__itt_api_init_t)(__itt_global*, __itt_group_id);
typedef void (__itt_api_fini_t)(__itt_global*);

static __itt_domain dummy_domain;
/* ========================================================================= */

#ifdef ITT_NOTIFY_EXT_REPORT
ITT_EXTERN_C void _N_(error_handler)(__itt_error_code, va_list args);
#endif /* ITT_NOTIFY_EXT_REPORT */

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#if _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4055) /* warning C4055: 'type cast' : from data pointer 'void *' to function pointer 'XXX' */
#endif
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

static void __itt_report_error(int code, ...)
{
    va_list args;
    va_start(args, code);
    if (_N_(_ittapi_global).error_handler != NULL)
    {
        __itt_error_handler_t* handler = (__itt_error_handler_t*)(size_t)_N_(_ittapi_global).error_handler;
        handler((__itt_error_code)code, args);
    }
#ifdef ITT_NOTIFY_EXT_REPORT
    _N_(error_handler)((__itt_error_code)code, args);
#endif /* ITT_NOTIFY_EXT_REPORT */
    va_end(args);
}

static int __itt_is_collector_available(void);

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#if _MSC_VER
#pragma warning(pop)
#endif
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static __itt_domain* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(domain_createW),_init))(const wchar_t* name)
{
    __itt_domain *h_tail = NULL, *h = NULL;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
        if (ITTNOTIFY_NAME(domain_createW) && ITTNOTIFY_NAME(domain_createW) != ITT_VERSIONIZE(ITT_JOIN(_N_(domain_createW),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(domain_createW)(name);
        }
        else
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return &dummy_domain;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).domain_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameW != NULL && !wcscmp(h->nameW, name)) break;
        }
        if (h == NULL)
        {
            NEW_DOMAIN_W(&_N_(_ittapi_global), h, h_tail, name);
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return h;
}

static __itt_domain* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(domain_createA),_init))(const char* name)
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
static __itt_domain* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(domain_create),_init))(const char* name)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
{
    __itt_domain *h_tail = NULL, *h = NULL;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        if (ITTNOTIFY_NAME(domain_createA) && ITTNOTIFY_NAME(domain_createA) != ITT_VERSIONIZE(ITT_JOIN(_N_(domain_createA),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(domain_createA)(name);
        }
#else
        if (ITTNOTIFY_NAME(domain_create) && ITTNOTIFY_NAME(domain_create) != ITT_VERSIONIZE(ITT_JOIN(_N_(domain_create),_init)))
        {
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(domain_create)(name);
        }
#endif
        else
        {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#else
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif
            return &dummy_domain;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).domain_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameA != NULL && !__itt_fstrcmp(h->nameA, name)) break;
        }
        if (h == NULL)
        {
            NEW_DOMAIN_A(&_N_(_ittapi_global), h, h_tail, name);
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return h;
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(module_load_with_sections),_init))(__itt_module_object* module_obj)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(module_load_with_sections) && ITTNOTIFY_NAME(module_load_with_sections) != ITT_VERSIONIZE(ITT_JOIN(_N_(module_load_with_sections),_init)))
    {
        if(module_obj != NULL)
        {
            module_obj->version = ITT_MODULE_OBJECT_VERSION;
            ITTNOTIFY_NAME(module_load_with_sections)(module_obj);
        }
    }
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(module_unload_with_sections),_init))(__itt_module_object* module_obj)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(module_unload_with_sections) && ITTNOTIFY_NAME(module_unload_with_sections) != ITT_VERSIONIZE(ITT_JOIN(_N_(module_unload_with_sections),_init)))
    {
        if(module_obj != NULL)
        {
            module_obj->version = ITT_MODULE_OBJECT_VERSION;
            ITTNOTIFY_NAME(module_unload_with_sections)(module_obj);
        }
    }
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static __itt_string_handle* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(string_handle_createW),_init))(const wchar_t* name)
{
    __itt_string_handle *h_tail = NULL, *h = NULL;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
        if (ITTNOTIFY_NAME(string_handle_createW) && ITTNOTIFY_NAME(string_handle_createW) != ITT_VERSIONIZE(ITT_JOIN(_N_(string_handle_createW),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(string_handle_createW)(name);
        }
        else
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).string_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->strW != NULL && !wcscmp(h->strW, name)) break;
        }
        if (h == NULL)
        {
            NEW_STRING_HANDLE_W(&_N_(_ittapi_global), h, h_tail, name);
        }
    }
    __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return h;
}

static __itt_string_handle* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(string_handle_createA),_init))(const char* name)
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
static __itt_string_handle* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(string_handle_create),_init))(const char* name)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
{
    __itt_string_handle *h_tail = NULL, *h = NULL;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        if (ITTNOTIFY_NAME(string_handle_createA) && ITTNOTIFY_NAME(string_handle_createA) != ITT_VERSIONIZE(ITT_JOIN(_N_(string_handle_createA),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(string_handle_createA)(name);
        }
#else
        if (ITTNOTIFY_NAME(string_handle_create) && ITTNOTIFY_NAME(string_handle_create) != ITT_VERSIONIZE(ITT_JOIN(_N_(string_handle_create),_init)))
        {
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(string_handle_create)(name);
        }
#endif
        else
        {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#else
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).string_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->strA != NULL && !__itt_fstrcmp(h->strA, name)) break;
        }
        if (h == NULL)
        {
            NEW_STRING_HANDLE_A(&_N_(_ittapi_global), h, h_tail, name);
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return h;
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createW),_init))(const wchar_t *name, const wchar_t *domain)
{
    __itt_counter_info_t *h_tail = NULL, *h = NULL;
    __itt_metadata_type type = __itt_metadata_u64;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
        if (ITTNOTIFY_NAME(counter_createW) && ITTNOTIFY_NAME(counter_createW) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createW),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_createW)(name, domain);
        }
        else
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).counter_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameW != NULL && h->type == (int)type && !wcscmp(h->nameW, name) && ((h->domainW == NULL && domain == NULL) ||
                (h->domainW != NULL && domain != NULL && !wcscmp(h->domainW, domain)))) break;

        }
        if (h == NULL)
        {
            NEW_COUNTER_W(&_N_(_ittapi_global), h, h_tail, name, domain, type);
        }
    }
    __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_counter)h;
}

static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createA),_init))(const char *name, const char *domain)
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create),_init))(const char *name, const char *domain)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
{
    __itt_counter_info_t *h_tail = NULL, *h = NULL;
    __itt_metadata_type type = __itt_metadata_u64;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        if (ITTNOTIFY_NAME(counter_createA) && ITTNOTIFY_NAME(counter_createA) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createA),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_createA)(name, domain);
        }
#else
        if (ITTNOTIFY_NAME(counter_create) && ITTNOTIFY_NAME(counter_create) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create),_init)))
        {
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_create)(name, domain);
        }
#endif
        else
        {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#else
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).counter_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameA != NULL && h->type == (int)type && !__itt_fstrcmp(h->nameA, name) && ((h->domainA == NULL && domain == NULL) ||
                (h->domainA != NULL && domain != NULL && !__itt_fstrcmp(h->domainA, domain)))) break;
        }
        if (h == NULL)
        {
            NEW_COUNTER_A(&_N_(_ittapi_global), h, h_tail, name, domain, type);
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_counter)h;
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_typedW),_init))(const wchar_t *name, const wchar_t *domain, __itt_metadata_type type)
{
    __itt_counter_info_t *h_tail = NULL, *h = NULL;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
        if (ITTNOTIFY_NAME(counter_create_typedW) && ITTNOTIFY_NAME(counter_create_typedW) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_typedW),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_create_typedW)(name, domain, type);
        }
        else
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).counter_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameW != NULL && h->type == (int)type && !wcscmp(h->nameW, name) && ((h->domainW == NULL && domain == NULL) ||
                (h->domainW != NULL && domain != NULL && !wcscmp(h->domainW, domain)))) break;

        }
        if (h == NULL)
        {
            NEW_COUNTER_W(&_N_(_ittapi_global), h, h_tail, name, domain, type);
        }
    }
    __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_counter)h;
}

static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_typedA),_init))(const char *name, const char *domain, __itt_metadata_type type)
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_typed),_init))(const char *name, const char *domain, __itt_metadata_type type)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
{
    __itt_counter_info_t *h_tail = NULL, *h = NULL;

    if (name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        if (ITTNOTIFY_NAME(counter_create_typedA) && ITTNOTIFY_NAME(counter_create_typedA) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_typedA),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_create_typedA)(name, domain, type);
        }
#else
        if (ITTNOTIFY_NAME(counter_create_typed) && ITTNOTIFY_NAME(counter_create_typed) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_typed),_init)))
        {
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_create_typed)(name, domain, type);
        }
#endif
        else
        {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#else
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).counter_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameA != NULL && h->type == (int)type && !__itt_fstrcmp(h->nameA, name) && ((h->domainA == NULL && domain == NULL) ||
                (h->domainA != NULL && domain != NULL && !__itt_fstrcmp(h->domainA, domain)))) break;
        }
        if (h == NULL)
        {
            NEW_COUNTER_A(&_N_(_ittapi_global), h, h_tail, name, domain, type);
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_counter)h;
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static __itt_histogram* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(histogram_createW),_init))(const __itt_domain* domain, const wchar_t* name, __itt_metadata_type x_type, __itt_metadata_type y_type)
{
    __itt_histogram *h_tail = NULL, *h = NULL;

    if (domain == NULL || name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
        if (ITTNOTIFY_NAME(histogram_createW) && ITTNOTIFY_NAME(histogram_createW) != ITT_VERSIONIZE(ITT_JOIN(_N_(histogram_createW),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(histogram_createW)(domain, name, x_type, y_type);
        }
        else
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).histogram_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->domain == NULL) continue;
            else if (h->domain == domain && h->nameW != NULL && !wcscmp(h->nameW, name)) break;
        }
        if (h == NULL)
        {
            NEW_HISTOGRAM_W(&_N_(_ittapi_global), h, h_tail, domain, name, x_type, y_type);
        }
    }
    __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_histogram*)h;
}

static __itt_histogram* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(histogram_createA),_init))(const __itt_domain* domain, const char* name, __itt_metadata_type x_type, __itt_metadata_type y_type)
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
static __itt_histogram* ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(histogram_create),_init))(const __itt_domain* domain, const char* name, __itt_metadata_type x_type, __itt_metadata_type y_type)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
{
    __itt_histogram *h_tail = NULL, *h = NULL;

    if (domain == NULL || name == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        if (ITTNOTIFY_NAME(histogram_createA) && ITTNOTIFY_NAME(histogram_createA) != ITT_VERSIONIZE(ITT_JOIN(_N_(histogram_createA),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(histogram_createA)(domain, name, x_type, y_type);
        }
#else
        if (ITTNOTIFY_NAME(histogram_create) && ITTNOTIFY_NAME(histogram_create) != ITT_VERSIONIZE(ITT_JOIN(_N_(histogram_create),_init)))
        {
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(histogram_create)(domain, name, x_type, y_type);
        }
#endif
        else
        {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#else
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).histogram_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->domain == NULL) continue;
            else if (h->domain == domain && h->nameA != NULL && !__itt_fstrcmp(h->nameA, name)) break;
        }
        if (h == NULL)
        {
            NEW_HISTOGRAM_A(&_N_(_ittapi_global), h, h_tail, domain, name, x_type, y_type);
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_histogram*)h;
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createW_v3),_init))(const __itt_domain* domain, const wchar_t* name, __itt_metadata_type type)
{
    __itt_counter_info_t *h_tail = NULL, *h = NULL;

    if (name == NULL || domain == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
        if (ITTNOTIFY_NAME(counter_createW_v3) && ITTNOTIFY_NAME(counter_createW_v3) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createW_v3),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_createW_v3)(domain, name, type);
        }
        else
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).counter_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameW != NULL  && h->type == (int)type && !wcscmp(h->nameW, name) && ((h->domainW == NULL && domain->nameW == NULL) ||
                (h->domainW != NULL && domain->nameW != NULL && !wcscmp(h->domainW, domain->nameW)))) break;

        }
        if (h == NULL)
        {
            NEW_COUNTER_W(&_N_(_ittapi_global),h,h_tail,name,domain->nameW,type);
        }
    }
    __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_counter)h;
}

static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createA_v3),_init))(const __itt_domain* domain, const char* name, __itt_metadata_type type)
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
static __itt_counter ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_v3),_init))(const __itt_domain* domain, const char* name, __itt_metadata_type type)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
{
    __itt_counter_info_t *h_tail = NULL, *h = NULL;

    if (name == NULL || domain == NULL)
    {
        return NULL;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        if (ITTNOTIFY_NAME(counter_createA_v3) && ITTNOTIFY_NAME(counter_createA_v3) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_createA_v3),_init)))
        {
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_createA_v3)(domain, name, type);
        }
#else
        if (ITTNOTIFY_NAME(counter_create_v3) && ITTNOTIFY_NAME(counter_create_v3) != ITT_VERSIONIZE(ITT_JOIN(_N_(counter_create_v3),_init)))
        {
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            return ITTNOTIFY_NAME(counter_create_v3)(domain, name, type);
        }
#endif
        else
        {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#else
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif
            return NULL;
        }
    }
    if (__itt_is_collector_available())
    {
        for (h_tail = NULL, h = _N_(_ittapi_global).counter_list; h != NULL; h_tail = h, h = h->next)
        {
            if (h->nameA != NULL  && h->type == (int)type && !__itt_fstrcmp(h->nameA, name) && ((h->domainA == NULL && domain->nameA == NULL) ||
                (h->domainA != NULL && domain->nameA != NULL && !__itt_fstrcmp(h->domainA, domain->nameA)))) break;
        }
        if (h == NULL)
        {
            NEW_COUNTER_A(&_N_(_ittapi_global),h,h_tail,name,domain->nameA,type);
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return (__itt_counter)h;
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(bind_context_metadata_to_counter),_init))(__itt_counter counter, size_t length, __itt_context_metadata* metadata)
{
    __itt_counter_metadata *h_tail = NULL, *h = NULL;

    if (counter == NULL || length == 0 || metadata == NULL)
    {
        return;
    }

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).api_initialized)
    {
        if (ITTNOTIFY_NAME(bind_context_metadata_to_counter) && ITTNOTIFY_NAME(bind_context_metadata_to_counter) != ITT_VERSIONIZE(ITT_JOIN(_N_(bind_context_metadata_to_counter),_init)))
        {
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
            ITTNOTIFY_NAME(bind_context_metadata_to_counter)(counter, length, metadata);
        }
        else
        {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
            __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#else
            if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif
            return;
        }
    }
    if (__itt_is_collector_available())
    {
        size_t item;
        char* str_valueA = NULL;
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        wchar_t* str_valueW = NULL;
#endif
        unsigned long long value = 0;
        __itt_context_type type = __itt_context_unknown;

        for (item = 0; item < length; item++)
        {
            type = metadata[item].type;
            for (h_tail = NULL, h = _N_(_ittapi_global).counter_metadata_list; h != NULL; h_tail = h, h = h->next)
            {
                if (h->counter != NULL && h->counter == counter && h->type == type) break;
            }
            if (h == NULL && counter != NULL && type != __itt_context_unknown)
            {
                if (type == __itt_context_nameA || type == __itt_context_deviceA || type == __itt_context_unitsA || type == __itt_context_pci_addrA)
                {
                    str_valueA = (char*)(metadata[item].value);
                    NEW_COUNTER_METADATA_STR_A(&_N_(_ittapi_global),h,h_tail,counter,type,str_valueA);
                }
#if ITT_PLATFORM==ITT_PLATFORM_WIN
                else if (type == __itt_context_nameW || type == __itt_context_deviceW || type == __itt_context_unitsW || type == __itt_context_pci_addrW)
                {
                    str_valueW = (wchar_t*)(metadata[item].value);
                    NEW_COUNTER_METADATA_STR_W(&_N_(_ittapi_global),h,h_tail,counter,type,str_valueW);
                }
#endif
                else if (type >= __itt_context_tid && type <= __itt_context_cpu_cycles_flag)
                {
                    value = *(unsigned long long*)(metadata[item].value);
                    NEW_COUNTER_METADATA_NUM(&_N_(_ittapi_global),h,h_tail,counter,type,value);
                }
            }
        }
    }
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
}

/* -------------------------------------------------------------------------- */

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(pause),_init))(void)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(pause) && ITTNOTIFY_NAME(pause) != ITT_VERSIONIZE(ITT_JOIN(_N_(pause),_init)))
    {
        ITTNOTIFY_NAME(pause)();
    }
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(resume),_init))(void)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(resume) && ITTNOTIFY_NAME(resume) != ITT_VERSIONIZE(ITT_JOIN(_N_(resume),_init)))
    {
        ITTNOTIFY_NAME(resume)();
    }
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(pause_scoped),_init))(__itt_collection_scope scope)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(pause_scoped) && ITTNOTIFY_NAME(pause_scoped) != ITT_VERSIONIZE(ITT_JOIN(_N_(pause_scoped),_init)))
    {
        ITTNOTIFY_NAME(pause_scoped)(scope);
    }
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(resume_scoped),_init))(__itt_collection_scope scope)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(resume_scoped) && ITTNOTIFY_NAME(resume_scoped) != ITT_VERSIONIZE(ITT_JOIN(_N_(resume_scoped),_init)))
    {
        ITTNOTIFY_NAME(resume_scoped)(scope);
    }
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_nameW),_init))(const wchar_t* name)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(thread_set_nameW) && ITTNOTIFY_NAME(thread_set_nameW) != ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_nameW),_init)))
    {
        ITTNOTIFY_NAME(thread_set_nameW)(name);
    }
}

static int ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thr_name_setW),_init))(const wchar_t* name, int namelen)
{
    (void)namelen;
    ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_nameW),_init))(name);
    return 0;
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_nameA),_init))(const char* name)
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_name),_init))(const char* name)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
#if ITT_PLATFORM==ITT_PLATFORM_WIN
    if (ITTNOTIFY_NAME(thread_set_nameA) && ITTNOTIFY_NAME(thread_set_nameA) != ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_nameA),_init)))
    {
        ITTNOTIFY_NAME(thread_set_nameA)(name);
    }
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
    if (ITTNOTIFY_NAME(thread_set_name) && ITTNOTIFY_NAME(thread_set_name) != ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_name),_init)))
    {
        ITTNOTIFY_NAME(thread_set_name)(name);
    }
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
static int ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thr_name_setA),_init))(const char* name, int namelen)
{
    (void)namelen;
    ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_nameA),_init))(name);
    return 0;
}
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
static int ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thr_name_set),_init))(const char* name, int namelen)
{
    (void)namelen;
    ITT_VERSIONIZE(ITT_JOIN(_N_(thread_set_name),_init))(name);
    return 0;
}
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thread_ignore),_init))(void)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    if (ITTNOTIFY_NAME(thread_ignore) && ITTNOTIFY_NAME(thread_ignore) != ITT_VERSIONIZE(ITT_JOIN(_N_(thread_ignore),_init)))
    {
        ITTNOTIFY_NAME(thread_ignore)();
    }
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(thr_ignore),_init))(void)
{
    ITT_VERSIONIZE(ITT_JOIN(_N_(thread_ignore),_init))();
}

static void ITTAPI ITT_VERSIONIZE(ITT_JOIN(_N_(enable_attach),_init))(void)
{
#ifdef __ANDROID__
    /*
     * if LIB_VAR_NAME env variable were set before then stay previous value
     * else set default path
    */
    setenv(ITT_TO_STR(LIB_VAR_NAME), ANDROID_ITTNOTIFY_DEFAULT_PATH, 0);
#endif
}

/* -------------------------------------------------------------------------- */

static const char* __itt_fsplit(const char* s, const char* sep, const char** out, int* len)
{
    int i;
    int j;

    if (!s || !sep || !out || !len)
        return NULL;

    for (i = 0; s[i]; i++)
    {
        int b = 0;
        for (j = 0; sep[j]; j++)
            if (s[i] == sep[j])
            {
                b = 1;
                break;
            }
        if (!b)
            break;
    }

    if (!s[i])
        return NULL;

    *len = 0;
    *out = &s[i];

    for (; s[i]; i++, (*len)++)
    {
        int b = 0;
        for (j = 0; sep[j]; j++)
            if (s[i] == sep[j])
            {
                b = 1;
                break;
            }
        if (b)
            break;
    }

    for (; s[i]; i++)
    {
        int b = 0;
        for (j = 0; sep[j]; j++)
            if (s[i] == sep[j])
            {
                b = 1;
                break;
            }
        if (!b)
            break;
    }

    return &s[i];
}

/* This function return value of env variable that placed into static buffer.
 * !!! The same static buffer is used for subsequent calls. !!!
 * This was done to avoid dynamic allocation for few calls.
 * Actually we need this function only four times.
 */
static const char* __itt_get_env_var(const char* name)
{
#define MAX_ENV_VALUE_SIZE 4086
    static char  env_buff[MAX_ENV_VALUE_SIZE];
    static char* env_value = (char*)env_buff;

    if (name != NULL)
    {
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        size_t max_len = MAX_ENV_VALUE_SIZE - (size_t)(env_value - env_buff);
        DWORD rc = GetEnvironmentVariableA(name, env_value, (DWORD)max_len);
        if (rc >= max_len)
            __itt_report_error(__itt_error_env_too_long, name, (size_t)rc - 1, (size_t)(max_len - 1));
        else if (rc > 0)
        {
            const char* ret = (const char*)env_value;
            env_value += rc + 1;
            return ret;
        }
        else
        {
            /* If environment variable is empty, GetEnvironmentVariables()
             * returns zero (number of characters (not including terminating null),
             * and GetLastError() returns ERROR_SUCCESS. */
            DWORD err = GetLastError();
            if (err == ERROR_SUCCESS)
                return env_value;

            if (err != ERROR_ENVVAR_NOT_FOUND)
                __itt_report_error(__itt_error_cant_read_env, name, (int)err);
        }
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
        char* env = getenv(name);
        if (env != NULL)
        {
            size_t len = __itt_fstrnlen(env, MAX_ENV_VALUE_SIZE);
            size_t max_len = MAX_ENV_VALUE_SIZE - (size_t)(env_value - env_buff);
            if (len < max_len)
            {
                const char* ret = (const char*)env_value;
                __itt_fstrcpyn(env_value, max_len, env, len + 1);
                env_value += len + 1;
                return ret;
            } else
                __itt_report_error(__itt_error_env_too_long, name, (size_t)len, (size_t)(max_len - 1));
        }
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
    }
    return NULL;
}

static const char* __itt_get_lib_name(void)
{
    const char* lib_name = __itt_get_env_var(ITT_TO_STR(LIB_VAR_NAME));
    return lib_name;
}

/* Avoid clashes with std::min */
#define __itt_min(a,b) ((a) < (b) ? (a) : (b))

static __itt_group_id __itt_get_groups(void)
{
    int i;
    __itt_group_id res = __itt_group_none;
    const char* var_name  = "INTEL_ITTNOTIFY_GROUPS";
    const char* group_str = __itt_get_env_var(var_name);

    if (group_str != NULL)
    {
        int len;
        char gr[255];
        const char* chunk;
        while ((group_str = __itt_fsplit(group_str, ",; ", &chunk, &len)) != NULL)
        {
            int min_len = __itt_min(len, (int)(sizeof(gr) - 1));
            __itt_fstrcpyn(gr, sizeof(gr) - 1, chunk,  min_len);
            gr[min_len] = 0;

            for (i = 0; group_list[i].name != NULL; i++)
            {
                if (!__itt_fstrcmp(gr, group_list[i].name))
                {
                    res = (__itt_group_id)(res | group_list[i].id);
                    break;
                }
            }
        }
        /* TODO: !!! Workaround for bug with warning for unknown group !!!
         * Should be fixed in new initialization scheme.
         * Now the following groups should be set always. */
        for (i = 0; group_list[i].id != __itt_group_none; i++)
            if (group_list[i].id != __itt_group_all &&
                group_list[i].id > __itt_group_splitter_min &&
                group_list[i].id < __itt_group_splitter_max)
                res = (__itt_group_id)(res | group_list[i].id);
        return res;
    }
    else
    {
        for (i = 0; group_alias[i].env_var != NULL; i++)
            if (__itt_get_env_var(group_alias[i].env_var) != NULL)
                return group_alias[i].groups;
    }

    return res;
}

#undef __itt_min

static int __itt_lib_version(lib_t lib)
{
    if (lib == NULL)
        return 0;
    if (__itt_get_proc(lib, "__itt_api_init"))
        return 2;
    if (__itt_get_proc(lib, "__itt_api_version"))
        return 1;
    return 0;
}

/* It's not used right now! Comment it out to avoid warnings.
static void __itt_reinit_all_pointers(void)
{
    register int i;
    // Fill all pointers with initial stubs
    for (i = 0; _N_(_ittapi_global).api_list_ptr[i].name != NULL; i++)
        *_N_(_ittapi_global).api_list_ptr[i].func_ptr = _N_(_ittapi_global).api_list_ptr[i].init_func;
}
*/

static void __itt_nullify_all_pointers(void)
{
    int i;
    /* Nulify all pointers except domain_create, string_handle_create  and counter_create */
    for (i = 0; _N_(_ittapi_global).api_list_ptr[i].name != NULL; i++)
        *_N_(_ittapi_global).api_list_ptr[i].func_ptr = _N_(_ittapi_global).api_list_ptr[i].null_func;
}

static int __itt_is_collector_available(void)
{
    int is_available;

    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    if (_N_(_ittapi_global).state == __itt_collection_uninitialized)
    {
        _N_(_ittapi_global).state = (NULL == __itt_get_lib_name()) ? __itt_collection_collector_absent : __itt_collection_collector_exists;
    }
    is_available = (_N_(_ittapi_global).state == __itt_collection_collector_exists ||
        _N_(_ittapi_global).state == __itt_collection_init_successful);
    __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    return is_available;
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#if _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4054) /* warning C4054: 'type cast' : from function pointer 'XXX' to data pointer 'void *' */
#pragma warning(disable: 4055) /* warning C4055: 'type cast' : from data pointer 'void *' to function pointer 'XXX' */
#endif
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

ITT_EXTERN_C void _N_(fini_ittlib)(void)
{
    __itt_api_fini_t* __itt_api_fini_ptr = NULL;
    static volatile TIDT current_thread = 0;

    if (_N_(_ittapi_global).api_initialized)
    {
        ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
        if (_N_(_ittapi_global).api_initialized)
        {
            if (current_thread == 0)
            {
                if (PTHREAD_SYMBOLS) current_thread = __itt_thread_id();
                if (_N_(_ittapi_global).lib != NULL)
                {
                    __itt_api_fini_ptr = (__itt_api_fini_t*)(size_t)__itt_get_proc(_N_(_ittapi_global).lib, "__itt_api_fini");
                }
                if (__itt_api_fini_ptr)
                {
                    __itt_api_fini_ptr(&_N_(_ittapi_global));
                }

                __itt_nullify_all_pointers();

 /* TODO: !!! not safe !!! don't support unload so far.
  *             if (_N_(_ittapi_global).lib != NULL)
  *                 __itt_unload_lib(_N_(_ittapi_global).lib);
  *             _N_(_ittapi_global).lib = NULL;
  */
                _N_(_ittapi_global).api_initialized = 0;
                current_thread = 0;
            }
        }
        if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    }
}

/* !!! this function should be called under mutex lock !!! */
static void __itt_free_allocated_resources(void)
{
    __itt_string_handle* current_string = _N_(_ittapi_global).string_list;
    while (current_string != NULL)
    {
        __itt_string_handle* tmp = current_string->next;
        free((char*)current_string->strA);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        free((wchar_t*)current_string->strW);
#endif
        free(current_string);
        current_string = tmp;
    }
    _N_(_ittapi_global).string_list = NULL;

    __itt_domain* current_domain = _N_(_ittapi_global).domain_list;
    while (current_domain != NULL)
    {
        __itt_domain* tmp = current_domain->next;
        free((char*)current_domain->nameA);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        free((wchar_t*)current_domain->nameW);
#endif
        free(current_domain);
        current_domain = tmp;
    }
    _N_(_ittapi_global).domain_list = NULL;

    __itt_counter_info_t* current_couter = _N_(_ittapi_global).counter_list;
    while (current_couter != NULL)
    {
        __itt_counter_info_t* tmp = current_couter->next;
        free((char*)current_couter->nameA);
        free((char*)current_couter->domainA);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        free((wchar_t*)current_couter->nameW);
        free((wchar_t*)current_couter->domainW);
#endif
        free(current_couter);
        current_couter = tmp;
    }
    _N_(_ittapi_global).counter_list = NULL;

    __itt_histogram* current_histogram = _N_(_ittapi_global).histogram_list;
    while (current_histogram != NULL)
    {
        __itt_histogram* tmp = current_histogram->next;
        free((char*)current_histogram->nameA);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        free((wchar_t*)current_histogram->nameW);
#endif
        free(current_histogram);
        current_histogram = tmp;
    }
    _N_(_ittapi_global).histogram_list = NULL;

    __itt_counter_metadata* current_counter_metadata = _N_(_ittapi_global).counter_metadata_list;
    while (current_counter_metadata != NULL)
    {
        __itt_counter_metadata* tmp = current_counter_metadata->next;
        free((char*)current_counter_metadata->str_valueA);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
        free((wchar_t*)current_counter_metadata->str_valueW);
#endif
        free(current_counter_metadata);
        current_counter_metadata = tmp;
    }
    _N_(_ittapi_global).counter_metadata_list = NULL;
}

ITT_EXTERN_C int _N_(init_ittlib)(const char* lib_name, __itt_group_id init_groups)
{
    int i;
    __itt_group_id groups;
#ifdef ITT_COMPLETE_GROUP
    __itt_group_id zero_group = __itt_group_none;
#endif /* ITT_COMPLETE_GROUP */
    static volatile TIDT current_thread = 0;

    if (!_N_(_ittapi_global).api_initialized)
    {
#ifndef ITT_SIMPLE_INIT
        ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
#endif /* ITT_SIMPLE_INIT */

        if (!_N_(_ittapi_global).api_initialized)
        {
            if (current_thread == 0)
            {
                if (PTHREAD_SYMBOLS) current_thread = __itt_thread_id();
                if (lib_name == NULL)
                {
                    lib_name = __itt_get_lib_name();
                }
                groups = __itt_get_groups();
                if (DL_SYMBOLS && (groups != __itt_group_none || lib_name != NULL))
                {
                    _N_(_ittapi_global).lib = __itt_load_lib((lib_name == NULL) ? ittnotify_lib_name : lib_name);

                    if (_N_(_ittapi_global).lib != NULL)
                    {
                        _N_(_ittapi_global).state = __itt_collection_init_successful;
                        __itt_api_init_t* __itt_api_init_ptr;
                        int lib_version = __itt_lib_version(_N_(_ittapi_global).lib);

                        switch (lib_version)
                        {
                        case 0:
                            groups = __itt_group_legacy;
                            ITT_ATTRIBUTE_FALLTHROUGH;
                        case 1:
                            /* Fill all pointers from dynamic library */
                            for (i = 0; _N_(_ittapi_global).api_list_ptr[i].name != NULL; i++)
                            {
                                if (_N_(_ittapi_global).api_list_ptr[i].group & groups & init_groups)
                                {
                                    *_N_(_ittapi_global).api_list_ptr[i].func_ptr = (void*)__itt_get_proc(_N_(_ittapi_global).lib, _N_(_ittapi_global).api_list_ptr[i].name);
                                    if (*_N_(_ittapi_global).api_list_ptr[i].func_ptr == NULL)
                                    {
                                        /* Restore pointers for function with static implementation */
                                        *_N_(_ittapi_global).api_list_ptr[i].func_ptr = _N_(_ittapi_global).api_list_ptr[i].null_func;
                                        __itt_report_error(__itt_error_no_symbol, lib_name, _N_(_ittapi_global).api_list_ptr[i].name);
#ifdef ITT_COMPLETE_GROUP
                                        zero_group = (__itt_group_id)(zero_group | _N_(_ittapi_global).api_list_ptr[i].group);
#endif /* ITT_COMPLETE_GROUP */
                                    }
                                }
                                else
                                    *_N_(_ittapi_global).api_list_ptr[i].func_ptr = _N_(_ittapi_global).api_list_ptr[i].null_func;
                            }

                            if (groups == __itt_group_legacy)
                            {
                                /* Compatibility with legacy tools */
                                ITTNOTIFY_NAME(thread_ignore)  = ITTNOTIFY_NAME(thr_ignore);
#if ITT_PLATFORM==ITT_PLATFORM_WIN
                                ITTNOTIFY_NAME(sync_createA)   = ITTNOTIFY_NAME(sync_set_nameA);
                                ITTNOTIFY_NAME(sync_createW)   = ITTNOTIFY_NAME(sync_set_nameW);
#else  /* ITT_PLATFORM!=ITT_PLATFORM_WIN */
                                ITTNOTIFY_NAME(sync_create)    = ITTNOTIFY_NAME(sync_set_name);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
                                ITTNOTIFY_NAME(sync_prepare)   = ITTNOTIFY_NAME(notify_sync_prepare);
                                ITTNOTIFY_NAME(sync_cancel)    = ITTNOTIFY_NAME(notify_sync_cancel);
                                ITTNOTIFY_NAME(sync_acquired)  = ITTNOTIFY_NAME(notify_sync_acquired);
                                ITTNOTIFY_NAME(sync_releasing) = ITTNOTIFY_NAME(notify_sync_releasing);
                            }

#ifdef ITT_COMPLETE_GROUP
                            for (i = 0; _N_(_ittapi_global).api_list_ptr[i].name != NULL; i++)
                                if (_N_(_ittapi_global).api_list_ptr[i].group & zero_group)
                                    *_N_(_ittapi_global).api_list_ptr[i].func_ptr = _N_(_ittapi_global).api_list_ptr[i].null_func;
#endif /* ITT_COMPLETE_GROUP */
                            break;
                        case 2:
                            __itt_api_init_ptr = (__itt_api_init_t*)(size_t)__itt_get_proc(_N_(_ittapi_global).lib, "__itt_api_init");
                            if (__itt_api_init_ptr)
                                __itt_api_init_ptr(&_N_(_ittapi_global), init_groups);
                            break;
                        }
                    }
                    else
                    {
                        _N_(_ittapi_global).state = __itt_collection_init_fail;
                        __itt_free_allocated_resources();
                        __itt_nullify_all_pointers();

                        __itt_report_error(__itt_error_no_module, lib_name,
#if ITT_PLATFORM==ITT_PLATFORM_WIN
                            __itt_system_error()
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
                            dlerror()
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
                        );
                    }
                }
                else
                {
                    _N_(_ittapi_global).state = __itt_collection_collector_absent;
                    __itt_nullify_all_pointers();
                }
                _N_(_ittapi_global).api_initialized = 1;
                current_thread = 0;
                /* !!! Just to avoid unused code elimination !!! */
                if (__itt_fini_ittlib_ptr == _N_(fini_ittlib)) current_thread = 0;
            }
        }

#ifndef ITT_SIMPLE_INIT
        if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
#endif /* ITT_SIMPLE_INIT */
    }

    /* Evaluating if any function ptr is non empty and it's in init_groups */
    for (i = 0; _N_(_ittapi_global).api_list_ptr[i].name != NULL; i++)
    {
        if (*_N_(_ittapi_global).api_list_ptr[i].func_ptr != _N_(_ittapi_global).api_list_ptr[i].null_func &&
            _N_(_ittapi_global).api_list_ptr[i].group & init_groups)
        {
            return 1;
        }
    }
    return 0;
}

ITT_EXTERN_C __itt_error_handler_t* _N_(set_error_handler)(__itt_error_handler_t* handler)
{
    __itt_error_handler_t* prev = (__itt_error_handler_t*)(size_t)_N_(_ittapi_global).error_handler;
    _N_(_ittapi_global).error_handler = (void*)(size_t)handler;
    return prev;
}

#if ITT_PLATFORM==ITT_PLATFORM_WIN
#if _MSC_VER
#pragma warning(pop)
#endif
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** __itt_mark_pt_region functions marks region of interest
 * region parameter defines different regions.
 * 0 <= region < 8 */

#if defined(ITT_API_IPT_SUPPORT) && (ITT_PLATFORM==ITT_PLATFORM_WIN || ITT_PLATFORM==ITT_PLATFORM_POSIX) && !defined(__ANDROID__)
void __itt_pt_mark(__itt_pt_region region);
void __itt_pt_mark_event(__itt_pt_region region);
#endif

ITT_EXTERN_C void _N_(mark_pt_region_begin)(__itt_pt_region region)
{
#if defined(ITT_API_IPT_SUPPORT) && (ITT_PLATFORM==ITT_PLATFORM_WIN || ITT_PLATFORM==ITT_PLATFORM_POSIX) && !defined(__ANDROID__)
    if (_N_(_ittapi_global).ipt_collect_events == 1)
    {
        __itt_pt_mark_event(2*region);
    }
    else
    {
        __itt_pt_mark(2*region);
    }
#else
    (void)region;
#endif
}

ITT_EXTERN_C void _N_(mark_pt_region_end)(__itt_pt_region region)
{
#if defined(ITT_API_IPT_SUPPORT) && (ITT_PLATFORM==ITT_PLATFORM_WIN || ITT_PLATFORM==ITT_PLATFORM_POSIX) && !defined(__ANDROID__)
    if (_N_(_ittapi_global).ipt_collect_events == 1)
    {
        __itt_pt_mark_event(2*region + 1);
    }
    else
    {
        __itt_pt_mark(2*region + 1);
    }
#else
     (void)region;
#endif
}

ITT_EXTERN_C __itt_collection_state (_N_(get_collection_state))(void)
{
    if (!_N_(_ittapi_global).api_initialized && _N_(_ittapi_global).thread_list == NULL)
    {
        __itt_init_ittlib_name(NULL, __itt_group_all);
    }
    return _N_(_ittapi_global).state;
}

/* !!! should be called from the library destructor !!!
 * this function destroys the mutex and frees resources
 * allocated by ITT API static part
 */
ITT_EXTERN_C void (_N_(release_resources))(void)
{
    ITT_MUTEX_INIT_AND_LOCK(_N_(_ittapi_global));
    __itt_free_allocated_resources();
    if (PTHREAD_SYMBOLS) __itt_mutex_unlock(&_N_(_ittapi_global).mutex);
    ITT_MUTEX_DESTROY(_N_(_ittapi_global));
}
