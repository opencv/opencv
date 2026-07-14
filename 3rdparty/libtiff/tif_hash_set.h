/**********************************************************************
 * $Id$
 *
 * Name:     tif_hash_set.h
 * Project:  TIFF - Common Portability Library
 * Purpose:  Hash set functions.
 * Author:   Even Rouault, <even dot rouault at spatialys.com>
 *
 **********************************************************************
 * Copyright (c) 2008-2009, Even Rouault <even dot rouault at spatialys.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************/

#ifndef TIFF_HASH_SET_H_INCLUDED
#define TIFF_HASH_SET_H_INCLUDED

#include <stdbool.h>

/**
 * \file tif_hash_set.h
 *
 * Hash set implementation.
 *
 * An hash set is a data structure that holds elements that are unique
 * according to a comparison function. Operations on the hash set, such as
 * insertion, removal or lookup, are supposed to be fast if an efficient
 * "hash" function is provided.
 */

#ifdef __cplusplus
extern "C"
{
#endif

    /* Types */

    /** Opaque type for a hash set */
    typedef struct _TIFFHashSet TIFFHashSet;

    /** TIFFHashSetHashFunc */
    typedef unsigned long (*TIFFHashSetHashFunc)(const void *elt);

    /** TIFFHashSetEqualFunc */
    typedef bool (*TIFFHashSetEqualFunc)(const void *elt1, const void *elt2);

    /** TIFFHashSetFreeEltFunc */
    typedef void (*TIFFHashSetFreeEltFunc)(void *elt);

    /* Functions */

    TIFFHashSet *TIFFHashSetNew(TIFFHashSetHashFunc fnHashFunc,
                                TIFFHashSetEqualFunc fnEqualFunc,
                                TIFFHashSetFreeEltFunc fnFreeEltFunc);

    void TIFFHashSetDestroy(TIFFHashSet *set);

    int TIFFHashSetSize(const TIFFHashSet *set);

#ifdef notused
    void TIFFHashSetClear(TIFFHashSet *set);

    /** TIFFHashSetIterEltFunc */
    typedef int (*TIFFHashSetIterEltFunc)(void *elt, void *user_data);

    void TIFFHashSetForeach(TIFFHashSet *set, TIFFHashSetIterEltFunc fnIterFunc,
                            void *user_data);
#endif

    bool TIFFHashSetInsert(TIFFHashSet *set, void *elt);

    void *TIFFHashSetLookup(TIFFHashSet *set, const void *elt);

    bool TIFFHashSetRemove(TIFFHashSet *set, const void *elt);

#ifdef notused
    bool TIFFHashSetRemoveDeferRehash(TIFFHashSet *set, const void *elt);
#endif

#ifdef __cplusplus
}
#endif

#endif /* TIFF_HASH_SET_H_INCLUDED */
