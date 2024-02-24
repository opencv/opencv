/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_DEBUG_H
#define OPENEXR_DEBUG_H

#include "openexr_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** Debug function: print to stdout the parts and attributes of the
 * context @p c
 */
EXR_EXPORT exr_result_t
exr_print_context_info (exr_const_context_t c, int verbose);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_DEBUG_H */
