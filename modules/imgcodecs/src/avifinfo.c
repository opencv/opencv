// Copyright (c) 2021, Alliance for Open Media. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#include "avifinfo.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

//------------------------------------------------------------------------------

// Status returned when reading the content of a box (or file).
typedef enum {
  kFound,     // Input correctly parsed and information retrieved.
  kNotFound,  // Input correctly parsed but information is missing or elsewhere.
  kTruncated,  // Input correctly parsed until missing bytes to continue.
  kAborted,  // Input correctly parsed until stopped to avoid timeout or crash.
  kInvalid,  // Input incorrectly parsed.
} AvifInfoInternalStatus;

static AvifInfoStatus AvifInfoInternalConvertStatus(AvifInfoInternalStatus s) {
  return (s == kFound)                         ? kAvifInfoOk
         : (s == kNotFound || s == kTruncated) ? kAvifInfoNotEnoughData
         : (s == kAborted)                     ? kAvifInfoTooComplex
                                               : kAvifInfoInvalidFile;
}

// uint32_t is used everywhere in this file. It is unlikely to be insufficient
// to parse AVIF headers.
#define AVIFINFO_MAX_SIZE UINT32_MAX
// Be reasonable. Avoid timeouts and out-of-memory.
#define AVIFINFO_MAX_NUM_BOXES 4096
// AvifInfoInternalFeatures uses uint8_t to store values.
#define AVIFINFO_MAX_VALUE UINT8_MAX
// Maximum number of stored associations. Past that, they are skipped.
#define AVIFINFO_MAX_TILES 16
#define AVIFINFO_MAX_PROPS 32
#define AVIFINFO_MAX_FEATURES 8
#define AVIFINFO_UNDEFINED 0

// Reads an unsigned integer from 'input' with most significant bits first.
// 'input' must be at least 'num_bytes'-long.
static uint32_t AvifInfoInternalReadBigEndian(const uint8_t* input,
                                              uint32_t num_bytes) {
  uint32_t value = 0;
  for (uint32_t i = 0; i < num_bytes; ++i) {
    value = (value << 8) | input[i];
  }
  return value;
}

//------------------------------------------------------------------------------
// Convenience macros.

#if defined(AVIFINFO_LOG_ERROR)  // Toggle to log encountered issues.
static void AvifInfoInternalLogError(const char* file, int line,
                                     AvifInfoInternalStatus status) {
  const char* kStr[] = {"Found", "NotFound", "Truncated", "Invalid", "Aborted"};
  fprintf(stderr, "  %s:%d: %s\n", file, line, kStr[status]);
  // Set a breakpoint here to catch the first detected issue.
}
#define AVIFINFO_RETURN(check_status)                               \
  do {                                                              \
    const AvifInfoInternalStatus status_checked = (check_status);   \
    if (status_checked != kFound && status_checked != kNotFound) {  \
      AvifInfoInternalLogError(__FILE__, __LINE__, status_checked); \
    }                                                               \
    return status_checked;                                          \
  } while (0)
#else
#define AVIFINFO_RETURN(check_status) \
  do {                                \
    return (check_status);            \
  } while (0)
#endif

#define AVIFINFO_CHECK(check_condition, check_status)      \
  do {                                                     \
    if (!(check_condition)) AVIFINFO_RETURN(check_status); \
  } while (0)
#define AVIFINFO_CHECK_STATUS_IS(check_status, expected_status)            \
  do {                                                                     \
    const AvifInfoInternalStatus status_returned = (check_status);         \
    AVIFINFO_CHECK(status_returned == (expected_status), status_returned); \
  } while (0)
#define AVIFINFO_CHECK_FOUND(check_status) \
  AVIFINFO_CHECK_STATUS_IS((check_status), kFound)
#define AVIFINFO_CHECK_NOT_FOUND(check_status) \
  AVIFINFO_CHECK_STATUS_IS((check_status), kNotFound)

//------------------------------------------------------------------------------
// Streamed input struct and helper functions.

typedef struct {
  void* stream;        // User-defined data.
  read_stream_t read;  // Used to fetch more bytes from the 'stream'.
  skip_stream_t skip;  // Used to advance the position in the 'stream'.
                       // Fallback to 'read' if 'skip' is null.
} AvifInfoInternalStream;

// Reads 'num_bytes' from the 'stream'. They are available at '*data'.
// 'num_bytes' must be greater than zero.
static AvifInfoInternalStatus AvifInfoInternalRead(
    AvifInfoInternalStream* stream, uint32_t num_bytes, const uint8_t** data) {
  *data = stream->read(stream->stream, num_bytes);
  AVIFINFO_CHECK(*data != NULL, kTruncated);
  return kFound;
}

// Skips 'num_bytes' from the 'stream'. 'num_bytes' can be zero.
static AvifInfoInternalStatus AvifInfoInternalSkip(
    AvifInfoInternalStream* stream, uint32_t num_bytes) {
  // Avoid a call to the user-defined function for nothing.
  if (num_bytes > 0) {
    if (stream->skip == NULL) {
      const uint8_t* unused;
      while (num_bytes > AVIFINFO_MAX_NUM_READ_BYTES) {
        AVIFINFO_CHECK_FOUND(
            AvifInfoInternalRead(stream, AVIFINFO_MAX_NUM_READ_BYTES, &unused));
        num_bytes -= AVIFINFO_MAX_NUM_READ_BYTES;
      }
      return AvifInfoInternalRead(stream, num_bytes, &unused);
    }
    stream->skip(stream->stream, num_bytes);
  }
  return kFound;
}

//------------------------------------------------------------------------------
// Features are parsed into temporary property associations.

typedef struct {
  uint8_t tile_item_id;
  uint8_t parent_item_id;
} AvifInfoInternalTile;  // Tile item id <-> parent item id associations.

typedef struct {
  uint8_t property_index;
  uint8_t item_id;
} AvifInfoInternalProp;  // Property index <-> item id associations.

typedef struct {
  uint8_t property_index;
  uint32_t width, height;
} AvifInfoInternalDimProp;  // Property <-> features associations.

typedef struct {
  uint8_t property_index;
  uint8_t bit_depth, num_channels;
} AvifInfoInternalChanProp;  // Property <-> features associations.

typedef struct {
  uint8_t has_primary_item;  // True if "pitm" was parsed.
  uint8_t has_alpha;         // True if an alpha "auxC" was parsed.
  uint8_t primary_item_id;
  AvifInfoFeatures primary_item_features;  // Deduced from the data below.
  uint8_t data_was_skipped;  // True if some loops/indices were skipped.

  uint8_t num_tiles;
  AvifInfoInternalTile tiles[AVIFINFO_MAX_TILES];
  uint8_t num_props;
  AvifInfoInternalProp props[AVIFINFO_MAX_PROPS];
  uint8_t num_dim_props;
  AvifInfoInternalDimProp dim_props[AVIFINFO_MAX_FEATURES];
  uint8_t num_chan_props;
  AvifInfoInternalChanProp chan_props[AVIFINFO_MAX_FEATURES];
} AvifInfoInternalFeatures;

// Generates the features of a given 'target_item_id' from internal features.
static AvifInfoInternalStatus AvifInfoInternalGetItemFeatures(
    AvifInfoInternalFeatures* f, uint32_t target_item_id, uint32_t tile_depth) {
  for (uint32_t prop_item = 0; prop_item < f->num_props; ++prop_item) {
    if (f->props[prop_item].item_id != target_item_id) continue;
    const uint32_t property_index = f->props[prop_item].property_index;

    // Retrieve the width and height of the primary item if not already done.
    if (target_item_id == f->primary_item_id &&
        (f->primary_item_features.width == AVIFINFO_UNDEFINED ||
         f->primary_item_features.height == AVIFINFO_UNDEFINED)) {
      for (uint32_t i = 0; i < f->num_dim_props; ++i) {
        if (f->dim_props[i].property_index != property_index) continue;
        f->primary_item_features.width = f->dim_props[i].width;
        f->primary_item_features.height = f->dim_props[i].height;
        if (f->primary_item_features.bit_depth != AVIFINFO_UNDEFINED &&
            f->primary_item_features.num_channels != AVIFINFO_UNDEFINED) {
          return kFound;
        }
        break;
      }
    }
    // Retrieve the bit depth and number of channels of the target item if not
    // already done.
    if (f->primary_item_features.bit_depth == AVIFINFO_UNDEFINED ||
        f->primary_item_features.num_channels == AVIFINFO_UNDEFINED) {
      for (uint32_t i = 0; i < f->num_chan_props; ++i) {
        if (f->chan_props[i].property_index != property_index) continue;
        f->primary_item_features.bit_depth = f->chan_props[i].bit_depth;
        f->primary_item_features.num_channels = f->chan_props[i].num_channels;
        if (f->primary_item_features.width != AVIFINFO_UNDEFINED &&
            f->primary_item_features.height != AVIFINFO_UNDEFINED) {
          return kFound;
        }
        break;
      }
    }
  }

  // Check for the bit_depth and num_channels in a tile if not yet found.
  for (uint32_t tile = 0; tile < f->num_tiles && tile_depth < 3; ++tile) {
    if (f->tiles[tile].parent_item_id != target_item_id) continue;
    AVIFINFO_CHECK_NOT_FOUND(AvifInfoInternalGetItemFeatures(
        f, f->tiles[tile].tile_item_id, tile_depth + 1));
  }
  AVIFINFO_RETURN(kNotFound);
}

// Generates the 'f->primary_item_features' from the AvifInfoInternalFeatures.
// Returns kNotFound if there is not enough information.
static AvifInfoInternalStatus AvifInfoInternalGetPrimaryItemFeatures(
    AvifInfoInternalFeatures* f) {
  // Nothing to do without the primary item ID.
  AVIFINFO_CHECK(f->has_primary_item, kNotFound);
  // Early exit.
  AVIFINFO_CHECK(f->num_dim_props > 0 && f->num_chan_props, kNotFound);
  AVIFINFO_CHECK_FOUND(
      AvifInfoInternalGetItemFeatures(f, f->primary_item_id, /*tile_depth=*/0));

  // "auxC" is parsed before the "ipma" properties so it is known now, if any.
  if (f->has_alpha) ++f->primary_item_features.num_channels;
  return kFound;
}

//------------------------------------------------------------------------------
// Box header parsing and various size checks.

typedef struct {
  uint32_t size;          // In bytes.
  uint8_t type[4];        // Four characters.
  uint32_t version;       // 0 or actual version if this is a full box.
  uint32_t flags;         // 0 or actual value if this is a full box.
  uint32_t content_size;  // 'size' minus the header size.
} AvifInfoInternalBox;

// Reads the header of a 'box' starting at the beginning of a 'stream'.
// 'num_remaining_bytes' is the remaining size of the container of the 'box'
// (either the file size itself or the content size of the parent of the 'box').
static AvifInfoInternalStatus AvifInfoInternalParseBox(
    AvifInfoInternalStream* stream, uint32_t num_remaining_bytes,
    uint32_t* num_parsed_boxes, AvifInfoInternalBox* box) {
  const uint8_t* data;
  // See ISO/IEC 14496-12:2012(E) 4.2
  uint32_t box_header_size = 8;  // box 32b size + 32b type (at least)
  AVIFINFO_CHECK(box_header_size <= num_remaining_bytes, kInvalid);
  AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 8, &data));
  box->size = AvifInfoInternalReadBigEndian(data, sizeof(uint32_t));
  memcpy(box->type, data + 4, 4);
  // 'box->size==1' means 64-bit size should be read after the box type.
  // 'box->size==0' means this box extends to all remaining bytes.
  if (box->size == 1) {
    box_header_size += 8;
    AVIFINFO_CHECK(box_header_size <= num_remaining_bytes, kInvalid);
    AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 8, &data));
    // Stop the parsing if any box has a size greater than 4GB.
    AVIFINFO_CHECK(AvifInfoInternalReadBigEndian(data, sizeof(uint32_t)) == 0,
                   kAborted);
    // Read the 32 least-significant bits.
    box->size = AvifInfoInternalReadBigEndian(data + 4, sizeof(uint32_t));
  } else if (box->size == 0) {
    box->size = num_remaining_bytes;
  }
  AVIFINFO_CHECK(box->size >= box_header_size, kInvalid);
  AVIFINFO_CHECK(box->size <= num_remaining_bytes, kInvalid);

  const int has_fullbox_header =
      !memcmp(box->type, "meta", 4) || !memcmp(box->type, "pitm", 4) ||
      !memcmp(box->type, "ipma", 4) || !memcmp(box->type, "ispe", 4) ||
      !memcmp(box->type, "pixi", 4) || !memcmp(box->type, "iref", 4) ||
      !memcmp(box->type, "auxC", 4);
  if (has_fullbox_header) box_header_size += 4;
  AVIFINFO_CHECK(box->size >= box_header_size, kInvalid);
  box->content_size = box->size - box_header_size;
  // Avoid timeouts. The maximum number of parsed boxes is arbitrary.
  ++*num_parsed_boxes;
  AVIFINFO_CHECK(*num_parsed_boxes < AVIFINFO_MAX_NUM_BOXES, kAborted);

  box->version = 0;
  box->flags = 0;
  if (has_fullbox_header) {
    AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 4, &data));
    box->version = AvifInfoInternalReadBigEndian(data, 1);
    box->flags = AvifInfoInternalReadBigEndian(data + 1, 3);
    // See AV1 Image File Format (AVIF) 8.1
    // at https://aomediacodec.github.io/av1-avif/#avif-boxes (available when
    // https://github.com/AOMediaCodec/av1-avif/pull/170 is merged).
    const uint32_t is_parsable =
        (!memcmp(box->type, "meta", 4) && box->version <= 0) ||
        (!memcmp(box->type, "pitm", 4) && box->version <= 1) ||
        (!memcmp(box->type, "ipma", 4) && box->version <= 1) ||
        (!memcmp(box->type, "ispe", 4) && box->version <= 0) ||
        (!memcmp(box->type, "pixi", 4) && box->version <= 0) ||
        (!memcmp(box->type, "iref", 4) && box->version <= 1) ||
        (!memcmp(box->type, "auxC", 4) && box->version <= 0);
    // Instead of considering this file as invalid, skip unparsable boxes.
    if (!is_parsable) memcpy(box->type, "\0skp", 4);  // \0 so not a valid type
  }
  return kFound;
}

//------------------------------------------------------------------------------

// Parses a 'stream' of an "ipco" box into 'features'.
// "ispe" is used for width and height, "pixi" and "av1C" are used for bit depth
// and number of channels, and "auxC" is used for alpha.
static AvifInfoInternalStatus ParseIpco(AvifInfoInternalStream* stream,
                                        uint32_t num_remaining_bytes,
                                        uint32_t* num_parsed_boxes,
                                        AvifInfoInternalFeatures* features) {
  uint32_t box_index = 1;  // 1-based index. Used for iterating over properties.
  do {
    AvifInfoInternalBox box;
    AVIFINFO_CHECK_FOUND(AvifInfoInternalParseBox(stream, num_remaining_bytes,
                                                  num_parsed_boxes, &box));

    if (!memcmp(box.type, "ispe", 4)) {
      // See ISO/IEC 23008-12:2017(E) 6.5.3.2
      const uint8_t* data;
      AVIFINFO_CHECK(box.content_size >= 8, kInvalid);
      AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 8, &data));
      const uint32_t width = AvifInfoInternalReadBigEndian(data + 0, 4);
      const uint32_t height = AvifInfoInternalReadBigEndian(data + 4, 4);
      AVIFINFO_CHECK(width != 0 && height != 0, kInvalid);
      if (features->num_dim_props < AVIFINFO_MAX_FEATURES &&
          box_index <= AVIFINFO_MAX_VALUE) {
        features->dim_props[features->num_dim_props].property_index = box_index;
        features->dim_props[features->num_dim_props].width = width;
        features->dim_props[features->num_dim_props].height = height;
        ++features->num_dim_props;
      } else {
        features->data_was_skipped = 1;
      }
      AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size - 8));
    } else if (!memcmp(box.type, "pixi", 4)) {
      // See ISO/IEC 23008-12:2017(E) 6.5.6.2
      const uint8_t* data;
      AVIFINFO_CHECK(box.content_size >= 1, kInvalid);
      AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 1, &data));
      const uint32_t num_channels = AvifInfoInternalReadBigEndian(data + 0, 1);
      AVIFINFO_CHECK(num_channels >= 1, kInvalid);
      AVIFINFO_CHECK(box.content_size >= 1 + num_channels, kInvalid);
      AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 1, &data));
      const uint32_t bit_depth = AvifInfoInternalReadBigEndian(data, 1);
      AVIFINFO_CHECK(bit_depth >= 1, kInvalid);
      for (uint32_t i = 1; i < num_channels; ++i) {
        AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 1, &data));
        // Bit depth should be the same for all channels.
        AVIFINFO_CHECK(AvifInfoInternalReadBigEndian(data, 1) == bit_depth,
                       kInvalid);
        AVIFINFO_CHECK(i <= 32, kAborted);  // Be reasonable.
      }
      if (features->num_chan_props < AVIFINFO_MAX_FEATURES &&
          box_index <= AVIFINFO_MAX_VALUE && bit_depth <= AVIFINFO_MAX_VALUE &&
          num_channels <= AVIFINFO_MAX_VALUE) {
        features->chan_props[features->num_chan_props].property_index =
            box_index;
        features->chan_props[features->num_chan_props].bit_depth = bit_depth;
        features->chan_props[features->num_chan_props].num_channels =
            num_channels;
        ++features->num_chan_props;
      } else {
        features->data_was_skipped = 1;
      }
      AVIFINFO_CHECK_FOUND(
          AvifInfoInternalSkip(stream, box.content_size - (1 + num_channels)));
    } else if (!memcmp(box.type, "av1C", 4)) {
      // See AV1 Codec ISO Media File Format Binding 2.3.1
      // at https://aomediacodec.github.io/av1-isobmff/#av1c
      // Only parse the necessary third byte. Assume that the others are valid.
      const uint8_t* data;
      AVIFINFO_CHECK(box.content_size >= 3, kInvalid);
      AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 3, &data));
      const int high_bitdepth = (data[2] & 0x40) != 0;
      const int twelve_bit = (data[2] & 0x20) != 0;
      const int monochrome = (data[2] & 0x10) != 0;
      if (twelve_bit) {
        AVIFINFO_CHECK(high_bitdepth, kInvalid);
      }
      if (features->num_chan_props < AVIFINFO_MAX_FEATURES &&
          box_index <= AVIFINFO_MAX_VALUE) {
        features->chan_props[features->num_chan_props].property_index =
            box_index;
        features->chan_props[features->num_chan_props].bit_depth =
            high_bitdepth ? twelve_bit ? 12 : 10 : 8;
        features->chan_props[features->num_chan_props].num_channels =
            monochrome ? 1 : 3;
        ++features->num_chan_props;
      } else {
        features->data_was_skipped = 1;
      }
      AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size - 3));
    } else if (!memcmp(box.type, "auxC", 4)) {
      // See AV1 Image File Format (AVIF) 4
      // at https://aomediacodec.github.io/av1-avif/#auxiliary-images
      const char* kAlphaStr = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";
      const uint32_t kAlphaStrLength = 44;  // Includes terminating character.
      if (box.content_size >= kAlphaStrLength) {
        const uint8_t* data;
        AVIFINFO_CHECK_FOUND(
            AvifInfoInternalRead(stream, kAlphaStrLength, &data));
        const char* const aux_type = (const char*)data;
        if (strcmp(aux_type, kAlphaStr) == 0) {
          // Note: It is unlikely but it is possible that this alpha plane does
          //       not belong to the primary item or a tile. Ignore this issue.
          features->has_alpha = 1;
        }
        AVIFINFO_CHECK_FOUND(
            AvifInfoInternalSkip(stream, box.content_size - kAlphaStrLength));
      } else {
        AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size));
      }
    } else {
      AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size));
    }
    ++box_index;
    num_remaining_bytes -= box.size;
  } while (num_remaining_bytes > 0);
  AVIFINFO_RETURN(kNotFound);
}

// Parses a 'stream' of an "iprp" box into 'features'. The "ipco" box contain
// the properties which are linked to items by the "ipma" box.
static AvifInfoInternalStatus ParseIprp(AvifInfoInternalStream* stream,
                                        uint32_t num_remaining_bytes,
                                        uint32_t* num_parsed_boxes,
                                        AvifInfoInternalFeatures* features) {
  do {
    AvifInfoInternalBox box;
    AVIFINFO_CHECK_FOUND(AvifInfoInternalParseBox(stream, num_remaining_bytes,
                                                  num_parsed_boxes, &box));

    if (!memcmp(box.type, "ipco", 4)) {
      AVIFINFO_CHECK_NOT_FOUND(
          ParseIpco(stream, box.content_size, num_parsed_boxes, features));
    } else if (!memcmp(box.type, "ipma", 4)) {
      // See ISO/IEC 23008-12:2017(E) 9.3.2
      uint32_t num_read_bytes = 4;
      const uint8_t* data;
      AVIFINFO_CHECK(box.content_size >= num_read_bytes, kInvalid);
      AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 4, &data));
      const uint32_t entry_count = AvifInfoInternalReadBigEndian(data, 4);
      const uint32_t id_num_bytes = (box.version < 1) ? 2 : 4;
      const uint32_t index_num_bytes = (box.flags & 1) ? 2 : 1;
      const uint32_t essential_bit_mask = (box.flags & 1) ? 0x8000 : 0x80;

      for (uint32_t entry = 0; entry < entry_count; ++entry) {
        if (entry >= AVIFINFO_MAX_PROPS ||
            features->num_props >= AVIFINFO_MAX_PROPS) {
          features->data_was_skipped = 1;
          break;
        }
        num_read_bytes += id_num_bytes + 1;
        AVIFINFO_CHECK(box.content_size >= num_read_bytes, kInvalid);
        AVIFINFO_CHECK_FOUND(
            AvifInfoInternalRead(stream, id_num_bytes + 1, &data));
        const uint32_t item_id =
            AvifInfoInternalReadBigEndian(data, id_num_bytes);
        const uint32_t association_count =
            AvifInfoInternalReadBigEndian(data + id_num_bytes, 1);

        uint32_t property;
        for (property = 0; property < association_count; ++property) {
          if (property >= AVIFINFO_MAX_PROPS ||
              features->num_props >= AVIFINFO_MAX_PROPS) {
            features->data_was_skipped = 1;
            break;
          }
          num_read_bytes += index_num_bytes;
          AVIFINFO_CHECK(box.content_size >= num_read_bytes, kInvalid);
          AVIFINFO_CHECK_FOUND(
              AvifInfoInternalRead(stream, index_num_bytes, &data));
          const uint32_t value =
              AvifInfoInternalReadBigEndian(data, index_num_bytes);
          // const int essential = (value & essential_bit_mask);  // Unused.
          const uint32_t property_index = (value & ~essential_bit_mask);
          if (property_index <= AVIFINFO_MAX_VALUE &&
              item_id <= AVIFINFO_MAX_VALUE) {
            features->props[features->num_props].property_index =
                property_index;
            features->props[features->num_props].item_id = item_id;
            ++features->num_props;
          } else {
            features->data_was_skipped = 1;
          }
        }
        if (property < association_count) break;  // Do not read garbage.
      }

      // If all features are available now, do not look further.
      AVIFINFO_CHECK_NOT_FOUND(
          AvifInfoInternalGetPrimaryItemFeatures(features));

      // Mostly if 'data_was_skipped'.
      AVIFINFO_CHECK_FOUND(
          AvifInfoInternalSkip(stream, box.content_size - num_read_bytes));
    } else {
      AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size));
    }
    num_remaining_bytes -= box.size;
  } while (num_remaining_bytes != 0);
  AVIFINFO_RETURN(kNotFound);
}

//------------------------------------------------------------------------------

// Parses a 'stream' of an "iref" box into 'features'.
// The "dimg" boxes contain links between tiles and their parent items, which
// can be used to infer bit depth and number of channels for the primary item
// when the latter does not have these properties.
static AvifInfoInternalStatus ParseIref(AvifInfoInternalStream* stream,
                                        uint32_t num_remaining_bytes,
                                        uint32_t* num_parsed_boxes,
                                        AvifInfoInternalFeatures* features) {
  do {
    AvifInfoInternalBox box;
    AVIFINFO_CHECK_FOUND(AvifInfoInternalParseBox(stream, num_remaining_bytes,
                                                  num_parsed_boxes, &box));

    if (!memcmp(box.type, "dimg", 4)) {
      // See ISO/IEC 14496-12:2015(E) 8.11.12.2
      const uint32_t num_bytes_per_id = (box.version == 0) ? 2 : 4;
      uint32_t num_read_bytes = num_bytes_per_id + 2;
      const uint8_t* data;
      AVIFINFO_CHECK(box.content_size >= num_read_bytes, kInvalid);
      AVIFINFO_CHECK_FOUND(
          AvifInfoInternalRead(stream, num_bytes_per_id + 2, &data));
      const uint32_t from_item_id =
          AvifInfoInternalReadBigEndian(data, num_bytes_per_id);
      const uint32_t reference_count =
          AvifInfoInternalReadBigEndian(data + num_bytes_per_id, 2);

      for (uint32_t i = 0; i < reference_count; ++i) {
        if (i >= AVIFINFO_MAX_TILES) {
          features->data_was_skipped = 1;
          break;
        }
        num_read_bytes += num_bytes_per_id;
        AVIFINFO_CHECK(box.content_size >= num_read_bytes, kInvalid);
        AVIFINFO_CHECK_FOUND(
            AvifInfoInternalRead(stream, num_bytes_per_id, &data));
        const uint32_t to_item_id =
            AvifInfoInternalReadBigEndian(data, num_bytes_per_id);
        if (from_item_id <= AVIFINFO_MAX_VALUE &&
            to_item_id <= AVIFINFO_MAX_VALUE &&
            features->num_tiles < AVIFINFO_MAX_TILES) {
          features->tiles[features->num_tiles].tile_item_id = to_item_id;
          features->tiles[features->num_tiles].parent_item_id = from_item_id;
          ++features->num_tiles;
        } else {
          features->data_was_skipped = 1;
        }
      }

      // If all features are available now, do not look further.
      AVIFINFO_CHECK_NOT_FOUND(
          AvifInfoInternalGetPrimaryItemFeatures(features));

      // Mostly if 'data_was_skipped'.
      AVIFINFO_CHECK_FOUND(
          AvifInfoInternalSkip(stream, box.content_size - num_read_bytes));
    } else {
      AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size));
    }
    num_remaining_bytes -= box.size;
  } while (num_remaining_bytes > 0);
  AVIFINFO_RETURN(kNotFound);
}

//------------------------------------------------------------------------------

// Parses a 'stream' of a "meta" box. It looks for the primary item ID in the
// "pitm" box and recurses into other boxes to find its 'features'.
static AvifInfoInternalStatus ParseMeta(AvifInfoInternalStream* stream,
                                        uint32_t num_remaining_bytes,
                                        uint32_t* num_parsed_boxes,
                                        AvifInfoInternalFeatures* features) {
  do {
    AvifInfoInternalBox box;
    AVIFINFO_CHECK_FOUND(AvifInfoInternalParseBox(stream, num_remaining_bytes,
                                                  num_parsed_boxes, &box));

    if (!memcmp(box.type, "pitm", 4)) {
      // See ISO/IEC 14496-12:2015(E) 8.11.4.2
      const uint32_t num_bytes_per_id = (box.version == 0) ? 2 : 4;
      const uint8_t* data;
      AVIFINFO_CHECK(num_bytes_per_id <= num_remaining_bytes, kInvalid);
      AVIFINFO_CHECK_FOUND(
          AvifInfoInternalRead(stream, num_bytes_per_id, &data));
      const uint32_t primary_item_id =
          AvifInfoInternalReadBigEndian(data, num_bytes_per_id);
      AVIFINFO_CHECK(primary_item_id <= AVIFINFO_MAX_VALUE, kAborted);
      features->has_primary_item = 1;
      features->primary_item_id = primary_item_id;
      AVIFINFO_CHECK_FOUND(
          AvifInfoInternalSkip(stream, box.content_size - num_bytes_per_id));
    } else if (!memcmp(box.type, "iprp", 4)) {
      AVIFINFO_CHECK_NOT_FOUND(
          ParseIprp(stream, box.content_size, num_parsed_boxes, features));
    } else if (!memcmp(box.type, "iref", 4)) {
      AVIFINFO_CHECK_NOT_FOUND(
          ParseIref(stream, box.content_size, num_parsed_boxes, features));
    } else {
      AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size));
    }
    num_remaining_bytes -= box.size;
  } while (num_remaining_bytes != 0);
  // According to ISO/IEC 14496-12:2012(E) 8.11.1.1 there is at most one "meta".
  AVIFINFO_RETURN(features->data_was_skipped ? kAborted : kInvalid);
}

//------------------------------------------------------------------------------

// Parses a file 'stream'. The file type is checked through the "ftyp" box.
static AvifInfoInternalStatus ParseFtyp(AvifInfoInternalStream* stream) {
  AvifInfoInternalBox box;
  uint32_t num_parsed_boxes = 0;
  AVIFINFO_CHECK_FOUND(AvifInfoInternalParseBox(stream, AVIFINFO_MAX_SIZE,
                                                &num_parsed_boxes, &box));
  AVIFINFO_CHECK(!memcmp(box.type, "ftyp", 4), kInvalid);
  // Iterate over brands. See ISO/IEC 14496-12:2012(E) 4.3.1
  AVIFINFO_CHECK(box.content_size >= 8, kInvalid);  // major_brand,minor_version
  for (uint32_t i = 0; i + 4 <= box.content_size; i += 4) {
    const uint8_t* data;
    AVIFINFO_CHECK_FOUND(AvifInfoInternalRead(stream, 4, &data));
    if (i == 4) continue;  // Skip minor_version.
    if (!memcmp(data, "avif", 4) || !memcmp(data, "avis", 4)) {
      AVIFINFO_CHECK_FOUND(
          AvifInfoInternalSkip(stream, box.content_size - (i + 4)));
      return kFound;
    }
    AVIFINFO_CHECK(i <= 32 * 4, kAborted);  // Be reasonable.
  }
  AVIFINFO_RETURN(kInvalid);  // No AVIF brand no good.
}

// Parses a file 'stream'. 'features' are extracted from the "meta" box.
static AvifInfoInternalStatus ParseFile(AvifInfoInternalStream* stream,
                                        uint32_t* num_parsed_boxes,
                                        AvifInfoInternalFeatures* features) {
  while (1) {
    AvifInfoInternalBox box;
    AVIFINFO_CHECK_FOUND(AvifInfoInternalParseBox(stream, AVIFINFO_MAX_SIZE,
                                                  num_parsed_boxes, &box));
    if (!memcmp(box.type, "meta", 4)) {
      return ParseMeta(stream, box.content_size, num_parsed_boxes, features);
    } else {
      AVIFINFO_CHECK_FOUND(AvifInfoInternalSkip(stream, box.content_size));
    }
  }
  AVIFINFO_RETURN(kInvalid);  // No "meta" no good.
}

//------------------------------------------------------------------------------
// Helpers for converting the fixed-size input public API to the streamed one.

typedef struct {
  const uint8_t* data;
  size_t data_size;
} AvifInfoInternalForward;

static const uint8_t* AvifInfoInternalForwardRead(void* stream,
                                                  size_t num_bytes) {
  AvifInfoInternalForward* forward = (AvifInfoInternalForward*)stream;
  if (num_bytes > forward->data_size) return NULL;
  const uint8_t* data = forward->data;
  forward->data += num_bytes;
  forward->data_size -= num_bytes;
  return data;
}

static void AvifInfoInternalForwardSkip(void* stream, size_t num_bytes) {
  AvifInfoInternalForward* forward = (AvifInfoInternalForward*)stream;
  if (num_bytes > forward->data_size) num_bytes = forward->data_size;
  forward->data += num_bytes;
  forward->data_size -= num_bytes;
}

//------------------------------------------------------------------------------
// Fixed-size input public API

AvifInfoStatus AvifInfoIdentify(const uint8_t* data, size_t data_size) {
  AvifInfoInternalForward stream;
  stream.data = data;
  stream.data_size = data_size;
  // Forward null 'data' as a null 'stream' to handle it the same way.
  return AvifInfoIdentifyStream(
      (void*)&stream, (data == NULL) ? NULL : AvifInfoInternalForwardRead,
      AvifInfoInternalForwardSkip);
}

AvifInfoStatus AvifInfoGetFeatures(const uint8_t* data, size_t data_size,
                                   AvifInfoFeatures* features) {
  AvifInfoInternalForward stream;
  stream.data = data;
  stream.data_size = data_size;
  return AvifInfoGetFeaturesStream(
      (void*)&stream, (data == NULL) ? NULL : AvifInfoInternalForwardRead,
      AvifInfoInternalForwardSkip, features);
}

//------------------------------------------------------------------------------
// Streamed input API

AvifInfoStatus AvifInfoIdentifyStream(void* stream, read_stream_t read,
                                      skip_stream_t skip) {
  if (read == NULL) return kAvifInfoNotEnoughData;

  AvifInfoInternalStream internal_stream;
  internal_stream.stream = stream;
  internal_stream.read = read;
  internal_stream.skip = skip;  // Fallbacks to 'read' if null.
  return AvifInfoInternalConvertStatus(ParseFtyp(&internal_stream));
}

AvifInfoStatus AvifInfoGetFeaturesStream(void* stream, read_stream_t read,
                                         skip_stream_t skip,
                                         AvifInfoFeatures* features) {
  if (features != NULL) memset(features, 0, sizeof(*features));
  if (read == NULL) return kAvifInfoNotEnoughData;

  AvifInfoInternalStream internal_stream;
  internal_stream.stream = stream;
  internal_stream.read = read;
  internal_stream.skip = skip;  // Fallbacks to 'read' if null.
  uint32_t num_parsed_boxes = 0;
  AvifInfoInternalFeatures internal_features;
  memset(&internal_features, AVIFINFO_UNDEFINED, sizeof(internal_features));

  // Go through all relevant boxes sequentially.
  const AvifInfoInternalStatus status =
      ParseFile(&internal_stream, &num_parsed_boxes, &internal_features);
  if (status == kFound && features != NULL) {
    memcpy(features, &internal_features.primary_item_features,
           sizeof(*features));
  }
  return AvifInfoInternalConvertStatus(status);
}
