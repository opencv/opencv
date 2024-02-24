/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_ATTR_H
#define OPENEXR_ATTR_H

#include "openexr_context.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** 
 * @defgroup AttributeTypes Attribute/metadata value types and struct declarations
 *
 * @brief These are a group of enum values defining valid values for
 * some attributes and then associated structs for other types.
 *
 * Some of these types will be directly representable/storable in
 * the file, some not. There is some overlap here with Imath, and they
 * should be kept in the same order for compatibility. However do note
 * that these are just the raw data, and no useful functions are
 * declared at this layer, that is what Imath is for.
 *
 * @{
 */

/** Enum declaring allowed values for \c uint8_t value stored in built-in compression type. */
typedef enum
{
    EXR_COMPRESSION_NONE  = 0,
    EXR_COMPRESSION_RLE   = 1,
    EXR_COMPRESSION_ZIPS  = 2,
    EXR_COMPRESSION_ZIP   = 3,
    EXR_COMPRESSION_PIZ   = 4,
    EXR_COMPRESSION_PXR24 = 5,
    EXR_COMPRESSION_B44   = 6,
    EXR_COMPRESSION_B44A  = 7,
    EXR_COMPRESSION_DWAA  = 8,
    EXR_COMPRESSION_DWAB  = 9,
    EXR_COMPRESSION_LAST_TYPE /**< Invalid value, provided for range checking. */
} exr_compression_t;

/** Enum declaring allowed values for \c uint8_t value stored in built-in env map type. */
typedef enum
{
    EXR_ENVMAP_LATLONG = 0,
    EXR_ENVMAP_CUBE    = 1,
    EXR_ENVMAP_LAST_TYPE /**< Invalid value, provided for range checking. */
} exr_envmap_t;

/** Enum declaring allowed values for \c uint8_t value stored in \c lineOrder type. */
typedef enum
{
    EXR_LINEORDER_INCREASING_Y = 0,
    EXR_LINEORDER_DECREASING_Y = 1,
    EXR_LINEORDER_RANDOM_Y     = 2,
    EXR_LINEORDER_LAST_TYPE /**< Invalid value, provided for range checking. */
} exr_lineorder_t;

/** Enum declaring allowed values for part type. */
typedef enum
{
    EXR_STORAGE_SCANLINE = 0,  /**< Corresponds to type of \c scanlineimage. */
    EXR_STORAGE_TILED,         /**< Corresponds to type of \c tiledimage. */
    EXR_STORAGE_DEEP_SCANLINE, /**< Corresponds to type of \c deepscanline. */
    EXR_STORAGE_DEEP_TILED,    /**< Corresponds to type of \c deeptile. */
    EXR_STORAGE_LAST_TYPE /**< Invalid value, provided for range checking. */
} exr_storage_t;

/** @brief Enum representing what type of tile information is contained. */
typedef enum
{
    EXR_TILE_ONE_LEVEL     = 0, /**< Single level of image data. */
    EXR_TILE_MIPMAP_LEVELS = 1, /**< Mipmapped image data. */
    EXR_TILE_RIPMAP_LEVELS = 2, /**< Ripmapped image data. */
    EXR_TILE_LAST_TYPE /**< Invalid value, provided for range checking. */
} exr_tile_level_mode_t;

/** @brief Enum representing how to scale positions between levels. */
typedef enum
{
    EXR_TILE_ROUND_DOWN = 0,
    EXR_TILE_ROUND_UP   = 1,
    EXR_TILE_ROUND_LAST_TYPE
} exr_tile_round_mode_t;

/** @brief Enum capturing the underlying data type on a channel. */
typedef enum
{
    EXR_PIXEL_UINT  = 0,
    EXR_PIXEL_HALF  = 1,
    EXR_PIXEL_FLOAT = 2,
    EXR_PIXEL_LAST_TYPE
} exr_pixel_type_t;

/* /////////////////////////////////////// */
/* First set of structs are data where we can read directly with no allocation needed... */

/* Most are naturally aligned, but force some of these
 * structs to be tightly packed
 */
#pragma pack(push, 1)

/** @brief Struct to hold color chromaticities to interpret the tristimulus color values in the image data. */
typedef struct
{
    float red_x;
    float red_y;
    float green_x;
    float green_y;
    float blue_x;
    float blue_y;
    float white_x;
    float white_y;
} exr_attr_chromaticities_t;

/** @brief Struct to hold keycode information. */
typedef struct
{
    int32_t film_mfc_code;
    int32_t film_type;
    int32_t prefix;
    int32_t count;
    int32_t perf_offset;
    int32_t perfs_per_frame;
    int32_t perfs_per_count;
} exr_attr_keycode_t;

/** @brief struct to hold a 32-bit floating-point 3x3 matrix. */
typedef struct
{
    float m[9];
} exr_attr_m33f_t;

/** @brief struct to hold a 64-bit floating-point 3x3 matrix. */
typedef struct
{
    double m[9];
} exr_attr_m33d_t;

/** @brief Struct to hold a 32-bit floating-point 4x4 matrix. */
typedef struct
{
    float m[16];
} exr_attr_m44f_t;

/** @brief Struct to hold a 64-bit floating-point 4x4 matrix. */
typedef struct
{
    double m[16];
} exr_attr_m44d_t;

/** @brief Struct to hold an integer ratio value. */
typedef struct
{
    int32_t  num;
    uint32_t denom;
} exr_attr_rational_t;

/** @brief Struct to hold timecode information. */
typedef struct
{
    uint32_t time_and_flags;
    uint32_t user_data;
} exr_attr_timecode_t;

/** @brief Struct to hold a 2-element integer vector. */
typedef struct
{
    union
    {
        struct
        {
            int32_t x, y;
        };
        int32_t arr[2];
    };
} exr_attr_v2i_t;

/** @brief Struct to hold a 2-element 32-bit float vector. */
typedef struct
{
    union
    {
        struct
        {
            float x, y;
        };
        float arr[2];
    };
} exr_attr_v2f_t;

/** @brief Struct to hold a 2-element 64-bit float vector. */
typedef struct
{
    union
    {
        struct
        {
            double x, y;
        };
        double arr[2];
    };
} exr_attr_v2d_t;

/** @brief Struct to hold a 3-element integer vector. */
typedef struct
{
    union
    {
        struct
        {
            int32_t x, y, z;
        };
        int32_t arr[3];
    };
} exr_attr_v3i_t;

/** @brief Struct to hold a 3-element 32-bit float vector. */
typedef struct
{
    union
    {
        struct
        {
            float x, y, z;
        };
        float arr[3];
    };
} exr_attr_v3f_t;

/** @brief Struct to hold a 3-element 64-bit float vector. */
typedef struct
{
    union
    {
        struct
        {
            double x, y, z;
        };
        double arr[3];
    };
} exr_attr_v3d_t;

/** @brief Struct to hold an integer box/region definition. */
typedef struct
{
    exr_attr_v2i_t min;
    exr_attr_v2i_t max;
} exr_attr_box2i_t;

/** @brief Struct to hold a floating-point box/region definition. */
typedef struct
{
    exr_attr_v2f_t min;
    exr_attr_v2f_t max;
} exr_attr_box2f_t;

/** @brief Struct holding base tiledesc attribute type defined in spec
 *
 * NB: This is in a tightly packed area so it can be read directly, be
 * careful it doesn't become padded to the next \c uint32_t boundary.
 */
typedef struct
{
    uint32_t x_size;
    uint32_t y_size;
    uint8_t  level_and_round;
} exr_attr_tiledesc_t;

/** @brief Macro to access type of tiling from packed structure. */
#define EXR_GET_TILE_LEVEL_MODE(tiledesc)                                      \
    ((exr_tile_level_mode_t) (((tiledesc).level_and_round) & 0xF))
/** @brief Macro to access the rounding mode of tiling from packed structure. */
#define EXR_GET_TILE_ROUND_MODE(tiledesc)                                      \
    ((exr_tile_round_mode_t) ((((tiledesc).level_and_round) >> 4) & 0xF))
/** @brief Macro to pack the tiling type and rounding mode into packed structure. */
#define EXR_PACK_TILE_LEVEL_ROUND(lvl, mode)                                   \
    ((uint8_t) ((((uint8_t) ((mode) &0xF) << 4)) | ((uint8_t) ((lvl) &0xF))))

#pragma pack(pop)

/* /////////////////////////////////////// */
/* Now structs that involve heap allocation to store data. */

/** Storage for a string. */
typedef struct
{
    int32_t length;
    /** If this is non-zero, the string owns the data, if 0, is a const ref to a static string. */
    int32_t alloc_size;

    const char* str;
} exr_attr_string_t;

/** Storage for a string vector. */
typedef struct
{
    int32_t n_strings;
    /** If this is non-zero, the string vector owns the data, if 0, is a const ref. */
    int32_t alloc_size;

    const exr_attr_string_t* strings;
} exr_attr_string_vector_t;

/** Float vector storage struct. */
typedef struct
{
    int32_t length;
    /** If this is non-zero, the float vector owns the data, if 0, is a const ref. */
    int32_t alloc_size;

    const float* arr;
} exr_attr_float_vector_t;

/** Hint for lossy compression methods about how to treat values
 * (logarithmic or linear), meaning a human sees values like R, G, B,
 * luminance difference between 0.1 and 0.2 as about the same as 1.0
 * to 2.0 (logarithmic), where chroma coordinates are closer to linear
 * (0.1 and 0.2 is about the same difference as 1.0 and 1.1).
 */
typedef enum
{
    EXR_PERCEPTUALLY_LOGARITHMIC = 0,
    EXR_PERCEPTUALLY_LINEAR      = 1
} exr_perceptual_treatment_t;

/** Individual channel information. */
typedef struct
{
    exr_attr_string_t name;
    /** Data representation for these pixels: uint, half, float. */
    exr_pixel_type_t pixel_type;
    /** Possible values are 0 and 1 per docs exr_perceptual_treatment_t. */
    uint8_t p_linear;
    uint8_t reserved[3];
    int32_t x_sampling;
    int32_t y_sampling;
} exr_attr_chlist_entry_t;

/** List of channel information (sorted alphabetically). */
typedef struct
{
    int num_channels;
    int num_alloced;

    const exr_attr_chlist_entry_t* entries;
} exr_attr_chlist_t;

/** @brief Struct to define attributes of an embedded preview image. */
typedef struct
{
    uint32_t width;
    uint32_t height;
    /** If this is non-zero, the preview owns the data, if 0, is a const ref. */
    size_t alloc_size;

    const uint8_t* rgba;
} exr_attr_preview_t;

/** Custom storage structure for opaque data.
 *
 * Handlers for opaque types can be registered, then when a
 * non-builtin type is encountered with a registered handler, the
 * function pointers to unpack/pack it will be set up.
 *
 * @sa exr_register_attr_type_handler
 */
typedef struct
{
    int32_t size;
    int32_t unpacked_size;
    /** If this is non-zero, the struct owns the data, if 0, is a const ref. */
    int32_t packed_alloc_size;
    uint8_t pad[4];

    void* packed_data;

    /** When an application wants to have custom data, they can store
     * an unpacked form here which will be requested to be destroyed
     * upon destruction of the attribute.
     */
    void* unpacked_data;

    /** An application can register an attribute handler which then
     * fills in these function pointers. This allows a user to delay
     * the expansion of the custom type until access is desired, and
     * similarly, to delay the packing of the data until write time.
     */
    exr_result_t (*unpack_func_ptr) (
        exr_context_t ctxt,
        const void*   data,
        int32_t       attrsize,
        int32_t*      outsize,
        void**        outbuffer);
    exr_result_t (*pack_func_ptr) (
        exr_context_t ctxt,
        const void*   data,
        int32_t       datasize,
        int32_t*      outsize,
        void*         outbuffer);
    void (*destroy_unpacked_func_ptr) (
        exr_context_t ctxt, void* data, int32_t attrsize);
} exr_attr_opaquedata_t;

/* /////////////////////////////////////// */

/** @brief Built-in/native attribute type enum.
 *
 * This will enable us to do a tagged type struct to generically store
 * attributes.
 */
typedef enum
{
    EXR_ATTR_UNKNOWN =
        0,          /**< Type indicating an error or uninitialized attribute. */
    EXR_ATTR_BOX2I, /**< Integer region definition. @see exr_attr_box2i_t. */
    EXR_ATTR_BOX2F, /**< Float region definition. @see exr_attr_box2f_t. */
    EXR_ATTR_CHLIST, /**< Definition of channels in file @see exr_chlist_entry. */
    EXR_ATTR_CHROMATICITIES, /**< Values to specify color space of colors in file @see exr_attr_chromaticities_t. */
    EXR_ATTR_COMPRESSION,    /**< ``uint8_t`` declaring compression present. */
    EXR_ATTR_DOUBLE,         /**< Double precision floating point number. */
    EXR_ATTR_ENVMAP,         /**< ``uint8_t`` declaring environment map type. */
    EXR_ATTR_FLOAT, /**< Normal (4 byte) precision floating point number. */
    EXR_ATTR_FLOAT_VECTOR, /**< List of normal (4 byte) precision floating point numbers. */
    EXR_ATTR_INT,     /**< 32-bit signed integer value. */
    EXR_ATTR_KEYCODE, /**< Struct recording keycode @see exr_attr_keycode_t. */
    EXR_ATTR_LINEORDER, /**< ``uint8_t`` declaring scanline ordering. */
    EXR_ATTR_M33F,      /**< 9 32-bit floats representing a 3x3 matrix. */
    EXR_ATTR_M33D,      /**< 9 64-bit floats representing a 3x3 matrix. */
    EXR_ATTR_M44F,      /**< 16 32-bit floats representing a 4x4 matrix. */
    EXR_ATTR_M44D,      /**< 16 64-bit floats representing a 4x4 matrix. */
    EXR_ATTR_PREVIEW, /**< 2 ``unsigned ints`` followed by 4 x w x h ``uint8_t`` image. */
    EXR_ATTR_RATIONAL, /**< \c int followed by ``unsigned int`` */
    EXR_ATTR_STRING,   /**< ``int`` (length) followed by char string data. */
    EXR_ATTR_STRING_VECTOR, /**< 0 or more text strings (int + string). number is based on attribute size. */
    EXR_ATTR_TILEDESC, /**< 2 ``unsigned ints`` ``xSize``, ``ySize`` followed by mode. */
    EXR_ATTR_TIMECODE, /**< 2 ``unsigned ints`` time and flags, user data. */
    EXR_ATTR_V2I,      /**< Pair of 32-bit integers. */
    EXR_ATTR_V2F,      /**< Pair of 32-bit floats. */
    EXR_ATTR_V2D,      /**< Pair of 64-bit floats. */
    EXR_ATTR_V3I,      /**< Set of 3 32-bit integers. */
    EXR_ATTR_V3F,      /**< Set of 3 32-bit floats. */
    EXR_ATTR_V3D,      /**< Set of 3 64-bit floats. */
    EXR_ATTR_OPAQUE,   /**< User/unknown provided type. */
    EXR_ATTR_LAST_KNOWN_TYPE
} exr_attribute_type_t;

/** @brief Storage, name and type information for an attribute.
 *
 * Attributes (metadata) for the file cause a surprising amount of
 * overhead. It is not uncommon for a production-grade EXR to have
 * many attributes. As such, the attribute struct is designed in a
 * slightly more complicated manner. It is optimized to have the
 * storage for that attribute: the struct itself, the name, the type,
 * and the data all allocated as one block. Further, the type and
 * standard names may use a static string to avoid allocating space
 * for those as necessary with the pointers pointing to static strings
 * (not to be freed). Finally, small values are optimized for.
 */
typedef struct
{
    /** Name of the attribute. */
    const char* name;
    /** String type name of the attribute. */
    const char* type_name;
    /** Length of name string (short flag is 31 max, long allows 255). */
    uint8_t name_length;
    /** Length of type string (short flag is 31 max, long allows 255). */
    uint8_t type_name_length;

    uint8_t pad[2];

    /** Enum of the attribute type. */
    exr_attribute_type_t type;

    /** Union of pointers of different types that can be used to type
     * pun to an appropriate type for builtins. Do note that while
     * this looks like a big thing, it is only the size of a single
     * pointer.  These are all pointers into some other data block
     * storing the value you want, with the exception of the pod types
     * which are just put in place (i.e. small value optimization).
     *
     * The attribute type \c type should directly correlate to one
     * of these entries.
     */
    union
    {
        // NB: not pointers for POD types
        uint8_t uc;
        double  d;
        float   f;
        int32_t i;

        exr_attr_box2i_t*          box2i;
        exr_attr_box2f_t*          box2f;
        exr_attr_chlist_t*         chlist;
        exr_attr_chromaticities_t* chromaticities;
        exr_attr_keycode_t*        keycode;
        exr_attr_float_vector_t*   floatvector;
        exr_attr_m33f_t*           m33f;
        exr_attr_m33d_t*           m33d;
        exr_attr_m44f_t*           m44f;
        exr_attr_m44d_t*           m44d;
        exr_attr_preview_t*        preview;
        exr_attr_rational_t*       rational;
        exr_attr_string_t*         string;
        exr_attr_string_vector_t*  stringvector;
        exr_attr_tiledesc_t*       tiledesc;
        exr_attr_timecode_t*       timecode;
        exr_attr_v2i_t*            v2i;
        exr_attr_v2f_t*            v2f;
        exr_attr_v2d_t*            v2d;
        exr_attr_v3i_t*            v3i;
        exr_attr_v3f_t*            v3f;
        exr_attr_v3d_t*            v3d;
        exr_attr_opaquedata_t*     opaque;
        uint8_t*                   rawptr;
    };
} exr_attribute_t;

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_ATTR_H */
