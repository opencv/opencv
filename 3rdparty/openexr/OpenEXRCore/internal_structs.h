/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_PRIVATE_STRUCTS_H
#define OPENEXR_PRIVATE_STRUCTS_H

#include "openexr_config.h"
#include "internal_attr.h"

#ifdef ILMTHREAD_THREADING_ENABLED
#    ifdef _WIN32
#        include <windows.h>
#        include <synchapi.h>
#    else
#        include <pthread.h>
#    endif
#endif

#ifndef __STDC_FORMAT_MACROS
#    define __STDC_FORMAT_MACROS
#endif

#include <inttypes.h>

#ifdef _MSC_VER
#    ifndef PRId64
#        define PRId64 "I64d"
#    endif
#    ifndef PRIu64
#        define PRIu64 "I64u"
#    endif
#endif

/* for testing, we include a bunch of internal stuff into the unit tests which are in c++ */
#ifdef __cplusplus
#    include <atomic>
using atomic_uintptr_t = std::atomic_uintptr_t;
#else
/* msvc, from version 19.31, evaluate __has_include(<stdatomic.h>) to true but
 * doesn't actually support it yet. Ignoring msvc for now, once we know minimal
 * version supporting it, we can compare against _MSC_VER. */
#    if !defined(_MSC_VER)
#        if defined __has_include
#            if __has_include(<stdatomic.h>)
#                define EXR_HAS_STD_ATOMICS 1
#            endif
#        endif
#    endif

#    ifdef EXR_HAS_STD_ATOMICS
#        include <stdatomic.h>
#    elif defined(_MSC_VER)
/* msvc w/ c11 support is only very new, until we know what the preprocessor checks are, provide defaults */
#        include <windows.h>
/* yeah, yeah, might be a 32-bit pointer, but if we make it the same, we
 * can write less since we know support is coming (eventually) */
typedef uint64_t atomic_uintptr_t;
#    else
#        error OS unimplemented support for atomics
#    endif
#endif

struct _internal_exr_part
{
    int part_index;
    exr_storage_t
        storage_mode; /**< Part of the file version flag declaring scanlines or tiled mode */

    exr_attribute_list_t attributes;

    /*  required attributes  */
    exr_attribute_t* channels;
    exr_attribute_t* compression;
    exr_attribute_t* dataWindow;
    exr_attribute_t* displayWindow;
    exr_attribute_t* lineOrder;
    exr_attribute_t* pixelAspectRatio;
    exr_attribute_t* screenWindowCenter;
    exr_attribute_t* screenWindowWidth;

    /** required for tiled files (@see storage_mode) */
    exr_attribute_t* tiles;

    /** required for deep or multipart files */
    exr_attribute_t* name;
    exr_attribute_t* type;
    exr_attribute_t* version;
    exr_attribute_t* chunkCount;
    /* in the file layout doc, but not required any more? */
    exr_attribute_t* maxSamplesPerPixel;

    /* copy commonly accessed required attributes to local struct memory for quick access */
    exr_attr_box2i_t  data_window;
    exr_attr_box2i_t  display_window;
    exr_compression_t comp_type;
    exr_lineorder_t   lineorder;

    int32_t zip_compression_level;
    float   dwa_compression_level;

    int32_t  num_tile_levels_x;
    int32_t  num_tile_levels_y;
    int32_t* tile_level_tile_count_x;
    int32_t* tile_level_tile_count_y;
    int32_t* tile_level_tile_size_x;
    int32_t* tile_level_tile_size_y;

    uint64_t unpacked_size_per_chunk;
    int16_t  lines_per_chunk;
    int16_t  chan_has_line_sampling;

    int32_t          chunk_count;
    uint64_t         chunk_table_offset;
    atomic_uintptr_t chunk_table;
};

enum _INTERNAL_EXR_READ_MODE
{
    EXR_MUST_READ_ALL    = 0,
    EXR_ALLOW_SHORT_READ = 1
};

enum _INTERNAL_EXR_CONTEXT_MODE
{
    EXR_CONTEXT_READ          = 0,
    EXR_CONTEXT_WRITE         = 1,
    EXR_CONTEXT_UPDATE_HEADER = 2,
    EXR_CONTEXT_WRITING_DATA  = 3,
    EXR_CONTEXT_WRITE_FINISHED
};

struct _internal_exr_context
{
    uint8_t mode;
    uint8_t version;
    uint8_t max_name_length;

    uint8_t is_singlepart_tiled;
    uint8_t has_nonimage_data;
    uint8_t is_multipart;

    uint8_t strict_header;
    uint8_t silent_header;

    exr_attr_string_t filename;
    exr_attr_string_t tmp_filename;

    exr_result_t (*do_read) (
        const struct _internal_exr_context* file,
        void*,
        uint64_t,
        uint64_t*,
        int64_t*,
        enum _INTERNAL_EXR_READ_MODE);
    exr_result_t (*do_write) (
        struct _internal_exr_context* file, const void*, uint64_t, uint64_t*);

    exr_result_t (*standard_error) (
        const struct _internal_exr_context* ctxt, exr_result_t code);
    exr_result_t (*report_error) (
        const struct _internal_exr_context* ctxt,
        exr_result_t                        code,
        const char*                         msg);
    exr_result_t (*print_error) (
        const struct _internal_exr_context* ctxt,
        exr_result_t                        code,
        const char*                         msg,
        ...) EXR_PRINTF_FUNC_ATTRIBUTE;

    exr_error_handler_cb_t error_handler_fn;

    exr_memory_allocation_func_t alloc_fn;
    exr_memory_free_func_t       free_fn;

    int max_image_w;
    int max_image_h;
    int max_tile_w;
    int max_tile_h;

    int   default_zip_level;
    float default_dwa_quality;

    void*                         real_user_data;
    void*                         user_data;
    exr_destroy_stream_func_ptr_t destroy_fn;

    int64_t             file_size;
    exr_read_func_ptr_t read_fn;

    exr_write_func_ptr_t write_fn;
    /* used when writing under a mutex, is there a better way? */
    uint64_t output_file_offset;
    int      cur_output_part;
    int      last_output_chunk;
    int      output_chunk_count;

    /** all files have at least one part */
    int num_parts;

    struct _internal_exr_part first_part;
    /* cheap array of one */
    struct _internal_exr_part*  init_part;
    struct _internal_exr_part** parts;

    exr_attribute_list_t custom_handlers;

    /* mostly needed for writing, but used during read to ensure
     * custom attribute handlers are safe */
#ifdef ILMTHREAD_THREADING_ENABLED
#    ifdef _WIN32
    CRITICAL_SECTION mutex;
#    else
    pthread_mutex_t mutex;
#    endif
#endif
    uint8_t disable_chunk_reconstruct;
    uint8_t legacy_header;
    uint8_t _pad[6];
};

#define EXR_CTXT(c) ((struct _internal_exr_context*) (c))
#define EXR_CCTXT(c) ((const struct _internal_exr_context*) (c))

#define EXR_CONST_CAST(t, v) ((t) (uintptr_t) v)

static inline void
internal_exr_lock (const struct _internal_exr_context* c)
{
#ifdef ILMTHREAD_THREADING_ENABLED
    struct _internal_exr_context* nonc =
        EXR_CONST_CAST (struct _internal_exr_context*, c);
#    ifdef _WIN32
    EnterCriticalSection (&nonc->mutex);
#    else
    pthread_mutex_lock (&nonc->mutex);
#    endif
#endif
}

static inline void
internal_exr_unlock (const struct _internal_exr_context* c)
{
#ifdef ILMTHREAD_THREADING_ENABLED
    struct _internal_exr_context* nonc =
        EXR_CONST_CAST (struct _internal_exr_context*, c);
#    ifdef _WIN32
    LeaveCriticalSection (&nonc->mutex);
#    else
    pthread_mutex_unlock (&nonc->mutex);
#    endif
#endif
}

#define EXR_LOCK(c) internal_exr_lock ((const struct _internal_exr_context*) c)
#define EXR_UNLOCK(c)                                                          \
    internal_exr_unlock ((const struct _internal_exr_context*) c)

#define EXR_LOCK_WRITE(c)                                                      \
    if (c->mode == EXR_CONTEXT_WRITE) internal_exr_lock (c)

#define EXR_UNLOCK_WRITE(c)                                                    \
    if (c->mode == EXR_CONTEXT_WRITE) internal_exr_unlock (c)

#define EXR_RETURN_WRITE(c)                                                    \
    ((c->mode == EXR_CONTEXT_WRITE) ? internal_exr_unlock (c) : ((void) 0))

#define EXR_UNLOCK_AND_RETURN_PCTXT(v) ((void) (EXR_UNLOCK (pctxt)), v)
#define EXR_UNLOCK_WRITE_AND_RETURN_PCTXT(v)                                   \
    ((void) (EXR_RETURN_WRITE (pctxt)), v)

#define INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR(c)                                 \
    struct _internal_exr_context* pctxt = EXR_CTXT (c);                        \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG

#define INTERN_EXR_PROMOTE_CONST_CONTEXT_OR_ERROR(c)                           \
    const struct _internal_exr_context* pctxt = EXR_CCTXT (c);                 \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG

#define EXR_PROMOTE_LOCKED_CONTEXT_OR_ERROR(c)                                 \
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (c);                                   \
    EXR_LOCK (c)

#define EXR_PROMOTE_CONST_CONTEXT_OR_ERROR(c)                                  \
    INTERN_EXR_PROMOTE_CONST_CONTEXT_OR_ERROR (c);                             \
    EXR_LOCK_WRITE (pctxt)

#define EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR(c, pi)                    \
    struct _internal_exr_context* pctxt = EXR_CTXT (c);                        \
    struct _internal_exr_part*    part;                                        \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG;                            \
    EXR_LOCK (pctxt);                                                          \
    if (pi < 0 || pi >= pctxt->num_parts)                                      \
        return (                                                               \
            (void) (EXR_UNLOCK (pctxt)),                                       \
            pctxt->print_error (                                               \
                pctxt,                                                         \
                EXR_ERR_ARGUMENT_OUT_OF_RANGE,                                 \
                "Part index (%d) out of range",                                \
                pi));                                                          \
    part = pctxt->parts[pi]

#define EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR(c, pi)                     \
    const struct _internal_exr_context* pctxt = EXR_CCTXT (c);                 \
    const struct _internal_exr_part*    part;                                  \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG;                            \
    EXR_LOCK_WRITE (pctxt);                                                    \
    if (pi < 0 || pi >= pctxt->num_parts)                                      \
        return (                                                               \
            (void) (EXR_RETURN_WRITE (pctxt)),                                 \
            pctxt->print_error (                                               \
                pctxt,                                                         \
                EXR_ERR_ARGUMENT_OUT_OF_RANGE,                                 \
                "Part index (%d) out of range",                                \
                pi));                                                          \
    part = pctxt->parts[pi]

#define EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR_NO_LOCK(c, pi)             \
    const struct _internal_exr_context* pctxt = EXR_CCTXT (c);                 \
    const struct _internal_exr_part*    part;                                  \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG;                            \
    if (pi < 0 || pi >= pctxt->num_parts)                                      \
        return (                                                               \
            (void) (EXR_RETURN_WRITE (pctxt)),                                 \
            pctxt->print_error (                                               \
                pctxt,                                                         \
                EXR_ERR_ARGUMENT_OUT_OF_RANGE,                                 \
                "Part index (%d) out of range",                                \
                pi));                                                          \
    part = pctxt->parts[pi]

#define EXR_PROMOTE_CONST_CONTEXT_OR_ERROR_NO_PART_NO_LOCK(c, pi)              \
    const struct _internal_exr_context* pctxt = EXR_CCTXT (c);                 \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG;                            \
    if (pi < 0 || pi >= pctxt->num_parts)                                      \
    return (                                                                   \
        (void) (EXR_RETURN_WRITE (pctxt)),                                     \
        pctxt->print_error (                                                   \
            pctxt,                                                             \
            EXR_ERR_ARGUMENT_OUT_OF_RANGE,                                     \
            "Part index (%d) out of range",                                    \
            pi))

#define EXR_PROMOTE_READ_CONST_CONTEXT_AND_PART_OR_ERROR(c, pi)                \
    const struct _internal_exr_context* pctxt = EXR_CCTXT (c);                 \
    const struct _internal_exr_part*    part;                                  \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG;                            \
    if (pctxt->mode != EXR_CONTEXT_READ)                                       \
        return pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_READ);           \
    if (pi < 0 || pi >= pctxt->num_parts)                                      \
        return pctxt->print_error (                                            \
            pctxt,                                                             \
            EXR_ERR_ARGUMENT_OUT_OF_RANGE,                                     \
            "Part index (%d) out of range",                                    \
            pi);                                                               \
    part = pctxt->parts[pi]

#define EXR_PROMOTE_READ_CONST_CONTEXT_OR_ERROR_NO_PART(c, pi)                 \
    const struct _internal_exr_context* pctxt = EXR_CCTXT (c);                 \
    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG;                            \
    if (pctxt->mode != EXR_CONTEXT_READ)                                       \
        return pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_READ);           \
    if (pi < 0 || pi >= pctxt->num_parts)                                      \
    return pctxt->print_error (                                                \
        pctxt,                                                                 \
        EXR_ERR_ARGUMENT_OUT_OF_RANGE,                                         \
        "Part index (%d) out of range",                                        \
        pi)

void internal_exr_update_default_handlers (exr_context_initializer_t* inits);

exr_result_t internal_exr_add_part (
    struct _internal_exr_context*, struct _internal_exr_part**, int* new_index);
void internal_exr_revert_add_part (
    struct _internal_exr_context*, struct _internal_exr_part**, int* new_index);

exr_result_t internal_exr_context_restore_handlers (
    struct _internal_exr_context* ctxt, exr_result_t rv);
exr_result_t internal_exr_alloc_context (
    struct _internal_exr_context**   out,
    const exr_context_initializer_t* initializers,
    enum _INTERNAL_EXR_CONTEXT_MODE  mode,
    size_t                           extra_data);
void internal_exr_destroy_context (struct _internal_exr_context* ctxt);

#endif /* OPENEXR_PRIVATE_STRUCTS_H */
