// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Data-types common to the mux and demux libraries.
//
// Author: Urvang (urvang@google.com)

#ifndef WEBP_WEBP_MUX_TYPES_H_
#define WEBP_WEBP_MUX_TYPES_H_

#include <stdlib.h>  // free()
#include <string.h>  // memset()
#include "./types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#if !(defined(__cplusplus) || defined(c_plusplus))
typedef enum WebPFeatureFlags WebPFeatureFlags;
typedef enum WebPMuxAnimDispose WebPMuxAnimDispose;
#endif

// VP8X Feature Flags.
enum WebPFeatureFlags {
  FRAGMENTS_FLAG  = 0x00000001,
  ANIMATION_FLAG  = 0x00000002,
  XMP_FLAG        = 0x00000004,
  EXIF_FLAG       = 0x00000008,
  ALPHA_FLAG      = 0x00000010,
  ICCP_FLAG       = 0x00000020
};

// Dispose method (animation only). Indicates how the area used by the current
// frame is to be treated before rendering the next frame on the canvas.
enum WebPMuxAnimDispose {
  WEBP_MUX_DISPOSE_NONE,       // Do not dispose.
  WEBP_MUX_DISPOSE_BACKGROUND  // Dispose to background color.
};

// Data type used to describe 'raw' data, e.g., chunk data
// (ICC profile, metadata) and WebP compressed image data.
typedef struct WebPData WebPData;
struct WebPData {
  const uint8_t* bytes;
  size_t size;
};

// Initializes the contents of the 'webp_data' object with default values.
static WEBP_INLINE void WebPDataInit(WebPData* webp_data) {
  if (webp_data != NULL) {
    memset(webp_data, 0, sizeof(*webp_data));
  }
}

// Clears the contents of the 'webp_data' object by calling free(). Does not
// deallocate the object itself.
static WEBP_INLINE void WebPDataClear(WebPData* webp_data) {
  if (webp_data != NULL) {
    free((void*)webp_data->bytes);
    WebPDataInit(webp_data);
  }
}

// Allocates necessary storage for 'dst' and copies the contents of 'src'.
// Returns true on success.
static WEBP_INLINE int WebPDataCopy(const WebPData* src, WebPData* dst) {
  if (src == NULL || dst == NULL) return 0;
  WebPDataInit(dst);
  if (src->bytes != NULL && src->size != 0) {
    dst->bytes = (uint8_t*)malloc(src->size);
    if (dst->bytes == NULL) return 0;
    memcpy((void*)dst->bytes, src->bytes, src->size);
    dst->size = src->size;
  }
  return 1;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_WEBP_MUX_TYPES_H_ */
