/**********************************************************************
 *
 * Name:     tif_hash_set.c
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

#include "tif_config.h"

#include "tif_hash_set.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/** List element structure. */
typedef struct _TIFFList TIFFList;

/** List element structure. */
struct _TIFFList
{
    /*! Pointer to the data object. Should be allocated and freed by the
     * caller.
     * */
    void *pData;
    /*! Pointer to the next element in list. NULL, if current element is the
     * last one.
     */
    struct _TIFFList *psNext;
};

struct _TIFFHashSet
{
    TIFFHashSetHashFunc fnHashFunc;
    TIFFHashSetEqualFunc fnEqualFunc;
    TIFFHashSetFreeEltFunc fnFreeEltFunc;
    TIFFList **tabList;
    int nSize;
    int nIndiceAllocatedSize;
    int nAllocatedSize;
    TIFFList *psRecyclingList;
    int nRecyclingListSize;
    bool bRehash;
#ifdef HASH_DEBUG
    int nCollisions;
#endif
};

static const int anPrimes[] = {
    53,        97,        193,       389,       769,       1543,     3079,
    6151,      12289,     24593,     49157,     98317,     196613,   393241,
    786433,    1572869,   3145739,   6291469,   12582917,  25165843, 50331653,
    100663319, 201326611, 402653189, 805306457, 1610612741};

/************************************************************************/
/*                    TIFFHashSetHashPointer()                          */
/************************************************************************/

/**
 * Hash function for an arbitrary pointer
 *
 * @param elt the arbitrary pointer to hash
 *
 * @return the hash value of the pointer
 */

static unsigned long TIFFHashSetHashPointer(const void *elt)
{
    return (unsigned long)(uintptr_t)((void *)(elt));
}

/************************************************************************/
/*                   TIFFHashSetEqualPointer()                          */
/************************************************************************/

/**
 * Equality function for arbitrary pointers
 *
 * @param elt1 the first arbitrary pointer to compare
 * @param elt2 the second arbitrary pointer to compare
 *
 * @return true if the pointers are equal
 */

static bool TIFFHashSetEqualPointer(const void *elt1, const void *elt2)
{
    return elt1 == elt2;
}

/************************************************************************/
/*                          TIFFHashSetNew()                             */
/************************************************************************/

/**
 * Creates a new hash set
 *
 * The hash function must return a hash value for the elements to insert.
 * If fnHashFunc is NULL, TIFFHashSetHashPointer will be used.
 *
 * The equal function must return if two elements are equal.
 * If fnEqualFunc is NULL, TIFFHashSetEqualPointer will be used.
 *
 * The free function is used to free elements inserted in the hash set,
 * when the hash set is destroyed, when elements are removed or replaced.
 * If fnFreeEltFunc is NULL, elements inserted into the hash set will not be
 * freed.
 *
 * @param fnHashFunc hash function. May be NULL.
 * @param fnEqualFunc equal function. May be NULL.
 * @param fnFreeEltFunc element free function. May be NULL.
 *
 * @return a new hash set
 */

TIFFHashSet *TIFFHashSetNew(TIFFHashSetHashFunc fnHashFunc,
                            TIFFHashSetEqualFunc fnEqualFunc,
                            TIFFHashSetFreeEltFunc fnFreeEltFunc)
{
    TIFFHashSet *set = (TIFFHashSet *)malloc(sizeof(TIFFHashSet));
    if (set == NULL)
        return NULL;
    set->fnHashFunc = fnHashFunc ? fnHashFunc : TIFFHashSetHashPointer;
    set->fnEqualFunc = fnEqualFunc ? fnEqualFunc : TIFFHashSetEqualPointer;
    set->fnFreeEltFunc = fnFreeEltFunc;
    set->nSize = 0;
    set->tabList = (TIFFList **)(calloc(53, sizeof(TIFFList *)));
    if (set->tabList == NULL)
    {
        free(set);
        return NULL;
    }
    set->nIndiceAllocatedSize = 0;
    set->nAllocatedSize = 53;
    set->psRecyclingList = NULL;
    set->nRecyclingListSize = 0;
    set->bRehash = false;
#ifdef HASH_DEBUG
    set->nCollisions = 0;
#endif
    return set;
}

/************************************************************************/
/*                          TIFFHashSetSize()                            */
/************************************************************************/

/**
 * Returns the number of elements inserted in the hash set
 *
 * Note: this is not the internal size of the hash set
 *
 * @param set the hash set
 *
 * @return the number of elements in the hash set
 */

int TIFFHashSetSize(const TIFFHashSet *set)
{
    assert(set != NULL);
    return set->nSize;
}

/************************************************************************/
/*                       TIFFHashSetGetNewListElt()                      */
/************************************************************************/

static TIFFList *TIFFHashSetGetNewListElt(TIFFHashSet *set)
{
    if (set->psRecyclingList)
    {
        TIFFList *psRet = set->psRecyclingList;
        psRet->pData = NULL;
        set->nRecyclingListSize--;
        set->psRecyclingList = psRet->psNext;
        return psRet;
    }

    return (TIFFList *)malloc(sizeof(TIFFList));
}

/************************************************************************/
/*                       TIFFHashSetReturnListElt()                      */
/************************************************************************/

static void TIFFHashSetReturnListElt(TIFFHashSet *set, TIFFList *psList)
{
    if (set->nRecyclingListSize < 128)
    {
        psList->psNext = set->psRecyclingList;
        set->psRecyclingList = psList;
        set->nRecyclingListSize++;
    }
    else
    {
        free(psList);
    }
}

/************************************************************************/
/*                   TIFFHashSetClearInternal()                          */
/************************************************************************/

static void TIFFHashSetClearInternal(TIFFHashSet *set, bool bFinalize)
{
    assert(set != NULL);
    for (int i = 0; i < set->nAllocatedSize; i++)
    {
        TIFFList *cur = set->tabList[i];
        while (cur)
        {
            if (set->fnFreeEltFunc)
                set->fnFreeEltFunc(cur->pData);
            TIFFList *psNext = cur->psNext;
            if (bFinalize)
                free(cur);
            else
                TIFFHashSetReturnListElt(set, cur);
            cur = psNext;
        }
        set->tabList[i] = NULL;
    }
    set->bRehash = false;
}

/************************************************************************/
/*                         TIFFListDestroy()                            */
/************************************************************************/

/**
 * Destroy a list. Caller responsible for freeing data objects contained in
 * list elements.
 *
 * @param psList pointer to list head.
 *
 */

static void TIFFListDestroy(TIFFList *psList)
{
    TIFFList *psCurrent = psList;

    while (psCurrent)
    {
        TIFFList *const psNext = psCurrent->psNext;
        free(psCurrent);
        psCurrent = psNext;
    }
}

/************************************************************************/
/*                        TIFFHashSetDestroy()                          */
/************************************************************************/

/**
 * Destroys an allocated hash set.
 *
 * This function also frees the elements if a free function was
 * provided at the creation of the hash set.
 *
 * @param set the hash set
 */

void TIFFHashSetDestroy(TIFFHashSet *set)
{
    if (set)
    {
        TIFFHashSetClearInternal(set, true);
        free(set->tabList);
        TIFFListDestroy(set->psRecyclingList);
        free(set);
    }
}

#ifdef notused
/************************************************************************/
/*                        TIFFHashSetClear()                             */
/************************************************************************/

/**
 * Clear all elements from a hash set.
 *
 * This function also frees the elements if a free function was
 * provided at the creation of the hash set.
 *
 * @param set the hash set
 */

void TIFFHashSetClear(TIFFHashSet *set)
{
    TIFFHashSetClearInternal(set, false);
    set->nIndiceAllocatedSize = 0;
    set->nAllocatedSize = 53;
#ifdef HASH_DEBUG
    set->nCollisions = 0;
#endif
    set->nSize = 0;
}

/************************************************************************/
/*                       TIFFHashSetForeach()                           */
/************************************************************************/

/**
 * Walk through the hash set and runs the provided function on all the
 * elements
 *
 * This function is provided the user_data argument of TIFFHashSetForeach.
 * It must return true to go on the walk through the hash set, or FALSE to
 * make it stop.
 *
 * Note : the structure of the hash set must *NOT* be modified during the
 * walk.
 *
 * @param set the hash set.
 * @param fnIterFunc the function called on each element.
 * @param user_data the user data provided to the function.
 */

void TIFFHashSetForeach(TIFFHashSet *set, TIFFHashSetIterEltFunc fnIterFunc,
                        void *user_data)
{
    assert(set != NULL);
    if (!fnIterFunc)
        return;

    for (int i = 0; i < set->nAllocatedSize; i++)
    {
        TIFFList *cur = set->tabList[i];
        while (cur)
        {
            if (!fnIterFunc(cur->pData, user_data))
                return;

            cur = cur->psNext;
        }
    }
}
#endif

/************************************************************************/
/*                        TIFFHashSetRehash()                           */
/************************************************************************/

static bool TIFFHashSetRehash(TIFFHashSet *set)
{
    int nNewAllocatedSize = anPrimes[set->nIndiceAllocatedSize];
    TIFFList **newTabList =
        (TIFFList **)(calloc(nNewAllocatedSize, sizeof(TIFFList *)));
    if (newTabList == NULL)
        return false;
#ifdef HASH_DEBUG
    TIFFDebug("TIFFHASH",
              "hashSet=%p, nSize=%d, nCollisions=%d, "
              "fCollisionRate=%.02f",
              set, set->nSize, set->nCollisions,
              set->nCollisions * 100.0 / set->nSize);
    set->nCollisions = 0;
#endif
    for (int i = 0; i < set->nAllocatedSize; i++)
    {
        TIFFList *cur = set->tabList[i];
        while (cur)
        {
            const unsigned long nNewHashVal =
                set->fnHashFunc(cur->pData) % nNewAllocatedSize;
#ifdef HASH_DEBUG
            if (newTabList[nNewHashVal])
                set->nCollisions++;
#endif
            TIFFList *psNext = cur->psNext;
            cur->psNext = newTabList[nNewHashVal];
            newTabList[nNewHashVal] = cur;
            cur = psNext;
        }
    }
    free(set->tabList);
    set->tabList = newTabList;
    set->nAllocatedSize = nNewAllocatedSize;
    set->bRehash = false;
    return true;
}

/************************************************************************/
/*                        TIFFHashSetFindPtr()                          */
/************************************************************************/

static void **TIFFHashSetFindPtr(TIFFHashSet *set, const void *elt)
{
    const unsigned long nHashVal = set->fnHashFunc(elt) % set->nAllocatedSize;
    TIFFList *cur = set->tabList[nHashVal];
    while (cur)
    {
        if (set->fnEqualFunc(cur->pData, elt))
            return &cur->pData;
        cur = cur->psNext;
    }
    return NULL;
}

/************************************************************************/
/*                         TIFFHashSetInsert()                          */
/************************************************************************/

/**
 * Inserts an element into a hash set.
 *
 * If the element was already inserted in the hash set, the previous
 * element is replaced by the new element. If a free function was provided,
 * it is used to free the previously inserted element
 *
 * @param set the hash set
 * @param elt the new element to insert in the hash set
 *
 * @return true if success. If false is returned, elt has not been inserted,
 * but TIFFHashSetInsert() will have run the free function if provided.
 */

bool TIFFHashSetInsert(TIFFHashSet *set, void *elt)
{
    assert(set != NULL);
    void **pElt = TIFFHashSetFindPtr(set, elt);
    if (pElt)
    {
        if (set->fnFreeEltFunc)
            set->fnFreeEltFunc(*pElt);

        *pElt = elt;
        return true;
    }

    if (set->nSize >= 2 * set->nAllocatedSize / 3 ||
        (set->bRehash && set->nIndiceAllocatedSize > 0 &&
         set->nSize <= set->nAllocatedSize / 2))
    {
        set->nIndiceAllocatedSize++;
        if (!TIFFHashSetRehash(set))
        {
            set->nIndiceAllocatedSize--;
            if (set->fnFreeEltFunc)
                set->fnFreeEltFunc(elt);
            return false;
        }
    }

    const unsigned long nHashVal = set->fnHashFunc(elt) % set->nAllocatedSize;
#ifdef HASH_DEBUG
    if (set->tabList[nHashVal])
        set->nCollisions++;
#endif

    TIFFList *new_elt = TIFFHashSetGetNewListElt(set);
    if (new_elt == NULL)
    {
        if (set->fnFreeEltFunc)
            set->fnFreeEltFunc(elt);
        return false;
    }
    new_elt->pData = elt;
    new_elt->psNext = set->tabList[nHashVal];
    set->tabList[nHashVal] = new_elt;
    set->nSize++;

    return true;
}

/************************************************************************/
/*                        TIFFHashSetLookup()                           */
/************************************************************************/

/**
 * Returns the element found in the hash set corresponding to the element to
 * look up The element must not be modified.
 *
 * @param set the hash set
 * @param elt the element to look up in the hash set
 *
 * @return the element found in the hash set or NULL
 */

void *TIFFHashSetLookup(TIFFHashSet *set, const void *elt)
{
    assert(set != NULL);
    void **pElt = TIFFHashSetFindPtr(set, elt);
    if (pElt)
        return *pElt;

    return NULL;
}

/************************************************************************/
/*                     TIFFHashSetRemoveInternal()                      */
/************************************************************************/

static bool TIFFHashSetRemoveInternal(TIFFHashSet *set, const void *elt,
                                      bool bDeferRehash)
{
    assert(set != NULL);
    if (set->nIndiceAllocatedSize > 0 && set->nSize <= set->nAllocatedSize / 2)
    {
        set->nIndiceAllocatedSize--;
        if (bDeferRehash)
            set->bRehash = true;
        else
        {
            if (!TIFFHashSetRehash(set))
            {
                set->nIndiceAllocatedSize++;
                return false;
            }
        }
    }

    int nHashVal = (int)(set->fnHashFunc(elt) % set->nAllocatedSize);
    TIFFList *cur = set->tabList[nHashVal];
    TIFFList *prev = NULL;
    while (cur)
    {
        if (set->fnEqualFunc(cur->pData, elt))
        {
            if (prev)
                prev->psNext = cur->psNext;
            else
                set->tabList[nHashVal] = cur->psNext;

            if (set->fnFreeEltFunc)
                set->fnFreeEltFunc(cur->pData);

            TIFFHashSetReturnListElt(set, cur);
#ifdef HASH_DEBUG
            if (set->tabList[nHashVal])
                set->nCollisions--;
#endif
            set->nSize--;
            return true;
        }
        prev = cur;
        cur = cur->psNext;
    }
    return false;
}

/************************************************************************/
/*                         TIFFHashSetRemove()                          */
/************************************************************************/

/**
 * Removes an element from a hash set
 *
 * @param set the hash set
 * @param elt the new element to remove from the hash set
 *
 * @return true if the element was in the hash set
 */

bool TIFFHashSetRemove(TIFFHashSet *set, const void *elt)
{
    return TIFFHashSetRemoveInternal(set, elt, false);
}

#ifdef notused
/************************************************************************/
/*                     TIFFHashSetRemoveDeferRehash()                   */
/************************************************************************/

/**
 * Removes an element from a hash set.
 *
 * This will defer potential rehashing of the set to later calls to
 * TIFFHashSetInsert() or TIFFHashSetRemove().
 *
 * @param set the hash set
 * @param elt the new element to remove from the hash set
 *
 * @return true if the element was in the hash set
 */

bool TIFFHashSetRemoveDeferRehash(TIFFHashSet *set, const void *elt)
{
    return TIFFHashSetRemoveInternal(set, elt, true);
}
#endif
