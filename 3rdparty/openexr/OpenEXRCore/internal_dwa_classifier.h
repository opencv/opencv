//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMF_INTERNAL_DWA_HELPERS_H_HAS_BEEN_INCLUDED
#    error "only include internal_dwa_helpers.h"
#endif

/**************************************/

typedef struct _Classifier
{
    const char*      _suffix;
    CompressorScheme _scheme;
    exr_pixel_type_t _type;
    int              _cscIdx;
    uint16_t         _caseInsensitive;
    uint16_t         _stringStatic;
} Classifier;

#define DWA_CLASSIFIER_FALSE 0
#define DWA_CLASSIFIER_TRUE 1

// clang-format off

static Classifier sDefaultChannelRules[] = {
    {"R",  LOSSY_DCT, EXR_PIXEL_HALF,   0, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"R",  LOSSY_DCT, EXR_PIXEL_FLOAT,  0, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"G",  LOSSY_DCT, EXR_PIXEL_HALF,   1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"G",  LOSSY_DCT, EXR_PIXEL_FLOAT,  1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"B",  LOSSY_DCT, EXR_PIXEL_HALF,   2, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"B",  LOSSY_DCT, EXR_PIXEL_FLOAT,  2, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"Y",  LOSSY_DCT, EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"Y",  LOSSY_DCT, EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"BY", LOSSY_DCT, EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"BY", LOSSY_DCT, EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"RY", LOSSY_DCT, EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"RY", LOSSY_DCT, EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"A",  RLE,       EXR_PIXEL_UINT,  -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"A",  RLE,       EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE},
    {"A",  RLE,       EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_FALSE, DWA_CLASSIFIER_TRUE}};

static Classifier sLegacyChannelRules[] = {
    {"r",     LOSSY_DCT, EXR_PIXEL_HALF,   0, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"r",     LOSSY_DCT, EXR_PIXEL_FLOAT,  0, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"red",   LOSSY_DCT, EXR_PIXEL_HALF,   0, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"red",   LOSSY_DCT, EXR_PIXEL_FLOAT,  0, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"g",     LOSSY_DCT, EXR_PIXEL_HALF,   1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"g",     LOSSY_DCT, EXR_PIXEL_FLOAT,  1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"grn",   LOSSY_DCT, EXR_PIXEL_HALF,   1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"grn",   LOSSY_DCT, EXR_PIXEL_FLOAT,  1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"green", LOSSY_DCT, EXR_PIXEL_HALF,   1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"green", LOSSY_DCT, EXR_PIXEL_FLOAT,  1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"b",     LOSSY_DCT, EXR_PIXEL_HALF,   2, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"b",     LOSSY_DCT, EXR_PIXEL_FLOAT,  2, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"blu",   LOSSY_DCT, EXR_PIXEL_HALF,   2, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"blu",   LOSSY_DCT, EXR_PIXEL_FLOAT,  2, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"blue",  LOSSY_DCT, EXR_PIXEL_HALF,   2, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"blue",  LOSSY_DCT, EXR_PIXEL_FLOAT,  2, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"y",     LOSSY_DCT, EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"y",     LOSSY_DCT, EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"by",    LOSSY_DCT, EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"by",    LOSSY_DCT, EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"ry",    LOSSY_DCT, EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"ry",    LOSSY_DCT, EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"a",     RLE,       EXR_PIXEL_UINT,  -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"a",     RLE,       EXR_PIXEL_HALF,  -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE},
    {"a",     RLE,       EXR_PIXEL_FLOAT, -1, DWA_CLASSIFIER_TRUE, DWA_CLASSIFIER_TRUE}};

// clang-format on

static inline void
Classifier_destroy (void (*free_fn) (void*), Classifier* p)
{
    if (p->_suffix && !p->_stringStatic)
        free_fn (EXR_CONST_CAST (void*, p->_suffix));
}

static exr_result_t
Classifier_read (
    void* (*alloc_fn) (size_t),
    Classifier*     out,
    const uint8_t** ptr,
    size_t*         size)
{
    const uint8_t* curin = *ptr;
    size_t         len   = 0;
    uint8_t        value;
    uint8_t        type;

    if (*size <= 3) return EXR_ERR_CORRUPT_CHUNK;

    {
        // maximum length of string plus one byte for terminating NULL
        char  suffix[128 + 1];
        char* mem;
        memset (suffix, 0, 128 + 1);
        for (; len < 128 + 1; ++len)
        {
            if (len > (*size - 3)) return EXR_ERR_CORRUPT_CHUNK;
            if (curin[len] == '\0') break;
            suffix[len] = (char) curin[len];
        }
        if (len == 128 + 1) return EXR_ERR_CORRUPT_CHUNK;
        // account for extra byte for nil terminator
        len += 1;

        mem = alloc_fn (len);
        if (!mem) return EXR_ERR_OUT_OF_MEMORY;

        memcpy (mem, suffix, len);
        out->_suffix       = mem;
        out->_stringStatic = DWA_CLASSIFIER_FALSE;
    }

    if (*size < len + 2 * sizeof (uint8_t)) return EXR_ERR_CORRUPT_CHUNK;

    curin += len;

    value = curin[0];
    type  = curin[1];

    curin += 2;

    *ptr = curin;
    *size -= len + 2 * sizeof (uint8_t);

    out->_cscIdx = (int) (value >> 4) - 1;
    if (out->_cscIdx < -1 || out->_cscIdx >= 3) return EXR_ERR_CORRUPT_CHUNK;

    out->_scheme = (CompressorScheme) ((value >> 2) & 3);
    if (out->_scheme >= NUM_COMPRESSOR_SCHEMES) return EXR_ERR_CORRUPT_CHUNK;

    out->_caseInsensitive =
        (value & 1 ? DWA_CLASSIFIER_TRUE : DWA_CLASSIFIER_FALSE);

    if (type >= EXR_PIXEL_LAST_TYPE) return EXR_ERR_CORRUPT_CHUNK;

    out->_type = (exr_pixel_type_t) type;
    return EXR_ERR_SUCCESS;
}

static inline int
Classifier_match (
    const Classifier* me, const char* suffix, const exr_pixel_type_t type)
{
    if (me->_type != type) return DWA_CLASSIFIER_FALSE;
#ifdef _MSC_VER
    if (me->_caseInsensitive) return _stricmp (suffix, me->_suffix) == 0;
#else
    if (me->_caseInsensitive) return strcasecmp (suffix, me->_suffix) == 0;
#endif

    return strcmp (suffix, me->_suffix) == 0;
}

static inline uint64_t
Classifier_size (const Classifier* me)
{
    return strlen (me->_suffix) + 1 + 2 * sizeof (uint8_t);
}

static inline uint64_t
Classifier_write (const Classifier* me, uint8_t** ptr)
{
    uint8_t* outptr    = *ptr;
    uint8_t  value     = 0;
    uint64_t sizeBytes = strlen (me->_suffix) + 1;

    memcpy (outptr, me->_suffix, sizeBytes);
    outptr += sizeBytes;

    value |= ((uint8_t) (me->_cscIdx + 1) & 15) << 4;
    value |= ((uint8_t) me->_scheme & 3) << 2;
    value |= (uint8_t) me->_caseInsensitive & 1;

    outptr[0] = value;
    outptr[1] = (uint8_t) me->_type;
    outptr += 2;
    *ptr = outptr;
    return sizeBytes + 2;
}

static inline const char*
Classifier_find_suffix (const char* channel_name)
{
    const char* suffix = strrchr (channel_name, '.');
    if (suffix) { suffix += 1; }
    else { suffix = channel_name; }
    return suffix;
}
