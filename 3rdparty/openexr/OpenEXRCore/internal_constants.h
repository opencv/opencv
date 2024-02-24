/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_PRIV_CONSTANTS_H
#define OPENEXR_PRIV_CONSTANTS_H

#define EXR_REQ_CHANNELS_STR "channels"
#define EXR_REQ_COMP_STR "compression"
#define EXR_REQ_DATA_STR "dataWindow"
#define EXR_REQ_DISP_STR "displayWindow"
#define EXR_REQ_LO_STR "lineOrder"
#define EXR_REQ_PAR_STR "pixelAspectRatio"
#define EXR_REQ_SCR_WC_STR "screenWindowCenter"
#define EXR_REQ_SCR_WW_STR "screenWindowWidth"
#define EXR_REQ_TILES_STR "tiles"
/* exr 2.0 req attr */
#define EXR_REQ_NAME_STR "name"
#define EXR_REQ_TYPE_STR "type"
#define EXR_REQ_VERSION_STR "version"
#define EXR_REQ_CHUNK_COUNT_STR "chunkCount"
/* this is in the file layout / technical info document
 * but not actually used anywhere...
 * #define REQ_MSS_COUNT_STR "maxSamplesPerPixel"
 */

#define EXR_SHORTNAME_MAXLEN 31
#define EXR_LONGNAME_MAXLEN 255

#endif /* OPENEXR_PRIV_CONSTANTS_H */
