/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_INTERNAL_ATTR_H
#define OPENEXR_INTERNAL_ATTR_H

#include "openexr_attr.h"
#include "openexr_context.h"

/** 
 * @defgroup InternalAttributeFunctions Functions for manipulating attributes
 *
 * The functions are currently internal to the library and are not
 * exposed to the outside. This is done primarily to strengthen the
 * contract around const-ness which then implies when it is safe (or
 * not) to use an exr context in a threaded manner.
 *
 * NB: These functions are not tagged with internal_ as a prefix like
 * other internal functions are such that if it is deemed useful to
 * expose them publicly in the future, it is easier to do so.
 *
 * @{
 * @}
 */

#include "internal_channel_list.h"
#include "internal_float_vector.h"
#include "internal_opaque.h"
#include "internal_preview.h"
#include "internal_string.h"
#include "internal_string_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

int internal_exr_is_standard_type (const char* typen);

/** @addtogroup InternalAttributeFunctions
 * @{
 */

typedef struct exr_attribute_list
{
    int num_attributes; /**< Number of attribute entries in the list */
    int num_alloced; /**< Allocation count. if > 0, attribute list owns pointer */
    exr_attribute_t** entries; /**< Creation order list of attributes */
    exr_attribute_t**
        sorted_entries; /**< Sorted order list of attributes for fast lookup */
} exr_attribute_list_t;

/** Initialize a list to an empty attribute list */
exr_result_t exr_attr_list_init (exr_context_t ctxt, exr_attribute_list_t* l);

/** Free memory for all the owned attributes in the list as well as the list itself */
exr_result_t
exr_attr_list_destroy (exr_context_t ctxt, exr_attribute_list_t* l);

/** Compute the number of bytes required to store this attribute list in a file */
exr_result_t exr_attr_list_compute_size (
    exr_context_t ctxt, exr_attribute_list_t* l, uint64_t* out);

/** Find an attribute in the list by name */
exr_result_t exr_attr_list_find_by_name (
    exr_const_context_t   ctxt,
    exr_attribute_list_t* l,
    const char*           name,
    exr_attribute_t**     out);

/** @brief Adds a new attribute to the list with a name and a (string) type
 *
 * if data_len > 0, will allocate extra memory as part of the
 * attribute block which allows one to do things like pre-allocate the
 * string storage space for a string attribute, or similar. If this is
 * specified, data_ptr must be provided to receive the memory
 * location. The responsibility is transferred to the caller to know
 * not to free this returned memory.
 *
 */
exr_result_t exr_attr_list_add_by_type (
    exr_context_t         ctxt,
    exr_attribute_list_t* l,
    const char*           name,
    const char*           type,
    int32_t               data_len,
    uint8_t**             data_ptr,
    exr_attribute_t**     attr);

/** @brief Adds a new attribute to the list with a name and a built-in type
 *
 * if data_len > 0, will allocate extra memory as part of the
 * attribute block which allows one to do things like pre-allocate the
 * string storage space for a string attribute, or similar. If this is
 * specified, data_ptr must be provided to receive the memory
 * location. The responsibility is transferred to the caller to know
 * not to free this returned memory.
 *
 */
exr_result_t exr_attr_list_add (
    exr_context_t         ctxt,
    exr_attribute_list_t* l,
    const char*           name,
    exr_attribute_type_t  type,
    int32_t               data_len,
    uint8_t**             data_ptr,
    exr_attribute_t**     attr);

/** @brief Adds a new attribute to the list with a static name (no
 * allocation) and a built-in type
 *
 * if data_len > 0, will allocate extra memory as part of the
 * attribute block which allows one to do things like pre-allocate the
 * string storage space for a string attribute, or similar. If this is
 * specified, data_ptr must be provided to receive the memory
 * location. The responsibility is transferred to the caller to know
 * not to free this returned memory.
 *
 */
exr_result_t exr_attr_list_add_static_name (
    exr_context_t         ctxt,
    exr_attribute_list_t* l,
    const char*           name,
    exr_attribute_type_t  type,
    int32_t               data_len,
    uint8_t**             data_ptr,
    exr_attribute_t**     attr);

/** Removes an attribute from the list and frees any associated memory */
exr_result_t exr_attr_list_remove (
    exr_context_t ctxt, exr_attribute_list_t* l, exr_attribute_t* attr);

/**
 * @}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_INTERNAL_ATTR_H */
