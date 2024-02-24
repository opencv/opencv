/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_BASE_H
#define OPENEXR_BASE_H

#include "openexr_config.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** @brief Retrieve the current library version. The @p extra string is for
 * custom installs, and is a static string, do not free the returned
 * pointer.
 */
EXR_EXPORT void
exr_get_library_version (int* maj, int* min, int* patch, const char** extra);

/** 
 * @defgroup SafetyChecks Controls for internal safety checks
 * @{
 */

/** @brief Limit the size of image allowed to be parsed or created by
 * the library.
 *
 * This is used as a safety check against corrupt files, but can also
 * serve to avoid potential issues on machines which have very
 * constrained RAM.
 *
 * These values are among the only globals in the core layer of
 * OpenEXR. The intended use is for applications to define a global
 * default, which will be combined with the values provided to the
 * individual context creation routine. The values are used to check
 * against parsed header values. This adds some level of safety from
 * memory overruns where a corrupt file given to the system may cause
 * a large allocation to happen, enabling buffer overruns or other
 * potential security issue.
 *
 * These global values are combined with the values in
 * \ref exr_context_initializer_t using the following rules:
 *
 * 1. negative values are ignored.
 *
 * 2. if either value has a positive (non-zero) value, and the other
 *    has 0, the positive value is preferred.
 *
 * 3. If both are positive (non-zero), the minimum value is used.
 *
 * 4. If both values are 0, this disables the constrained size checks.
 *
 * This function does not fail.
 */
EXR_EXPORT void exr_set_default_maximum_image_size (int w, int h);

/** @brief Retrieve the global default maximum image size.
 *
 * This function does not fail.
 */
EXR_EXPORT void exr_get_default_maximum_image_size (int* w, int* h);

/** @brief Limit the size of an image tile allowed to be parsed or
 * created by the library.
 *
 * Similar to image size, this places constraints on the maximum tile
 * size as a safety check against bad file data
 *
 * This is used as a safety check against corrupt files, but can also
 * serve to avoid potential issues on machines which have very
 * constrained RAM
 *
 * These values are among the only globals in the core layer of
 * OpenEXR. The intended use is for applications to define a global
 * default, which will be combined with the values provided to the
 * individual context creation routine. The values are used to check
 * against parsed header values. This adds some level of safety from
 * memory overruns where a corrupt file given to the system may cause
 * a large allocation to happen, enabling buffer overruns or other
 * potential security issue.
 *
 * These global values are combined with the values in
 * \ref exr_context_initializer_t using the following rules:
 *
 * 1. negative values are ignored.
 *
 * 2. if either value has a positive (non-zero) value, and the other
 *    has 0, the positive value is preferred.
 *
 * 3. If both are positive (non-zero), the minimum value is used.
 *
 * 4. If both values are 0, this disables the constrained size checks.
 *
 * This function does not fail.
 */
EXR_EXPORT void exr_set_default_maximum_tile_size (int w, int h);

/** @brief Retrieve the global maximum tile size.
 *
 * This function does not fail.
 */
EXR_EXPORT void exr_get_default_maximum_tile_size (int* w, int* h);

/** @} */

/**
 * @defgroup CompressionDefaults Provides default compression settings
 * @{
 */

/** @brief Assigns a default zip compression level.
 *
 * This value may be controlled separately on each part, but this
 * global control determines the initial value.
 */
EXR_EXPORT void exr_set_default_zip_compression_level (int l);

/** @brief Retrieve the global default zip compression value
 */
EXR_EXPORT void exr_get_default_zip_compression_level (int* l);

/** @brief Assigns a default DWA compression quality level.
 *
 * This value may be controlled separately on each part, but this
 * global control determines the initial value.
 */
EXR_EXPORT void exr_set_default_dwa_compression_quality (float q);

/** @brief Retrieve the global default dwa compression quality
 */
EXR_EXPORT void exr_get_default_dwa_compression_quality (float* q);

/** @} */

/**
 * @defgroup MemoryAllocators Provides global control over memory allocators
 * @{
 */

/** @brief Function pointer used to hold a malloc-like routine.
 *
 * Providing these to a context will override what memory is used to
 * allocate the context itself, as well as any allocations which
 * happen during processing of a file or stream. This can be used by
 * systems which provide rich malloc tracking routines to override the
 * internal allocations performed by the library.
 *
 * This function is expected to allocate and return a new memory
 * handle, or `NULL` if allocation failed (which the library will then
 * handle and return an out-of-memory error).
 *
 * If one is provided, both should be provided.
 * @sa exr_memory_free_func_t
 */
typedef void* (*exr_memory_allocation_func_t) (size_t bytes);

/** @brief Function pointer used to hold a free-like routine.
 *
 * Providing these to a context will override what memory is used to
 * allocate the context itself, as well as any allocations which
 * happen during processing of a file or stream. This can be used by
 * systems which provide rich malloc tracking routines to override the
 * internal allocations performed by the library.
 *
 * This function is expected to return memory to the system, ala free
 * from the C library.
 *
 * If providing one, probably need to provide both routines.
 * @sa exr_memory_allocation_func_t
 */
typedef void (*exr_memory_free_func_t) (void* ptr);

/** @brief Allow the user to override default allocator used internal
 * allocations necessary for files, attributes, and other temporary
 * memory.
 *
 * These routines may be overridden when creating a specific context,
 * however this provides global defaults such that the default can be
 * applied.
 *
 * If either pointer is 0, the appropriate malloc/free routine will be
 * substituted.
 *
 * This function does not fail.
 */
EXR_EXPORT void exr_set_default_memory_routines (
    exr_memory_allocation_func_t alloc_func, exr_memory_free_func_t free_func);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_BASE_H */
