/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _CV_LIST_H_
#define _CV_LIST_H_

#include <stdlib.h>
#include <assert.h>

#define CV_FORCE_INLINE CV_INLINE

#if !defined(_LIST_INLINE)
#define _LIST_INLINE CV_FORCE_INLINE
#endif /*_LIST_INLINE*/

#if defined DECLARE_LIST
#if defined _MSC_VER && _MSC_VER >= 1200
    #pragma warning("DECLARE_LIST macro is already defined!")
#endif
#endif /*DECLARE_LIST*/

static const long default_size = 10;
static const long default_inc_size = 10;

struct _pos
{
    void* m_pos;
#ifdef _DEBUG
    struct _list* m_list;
#endif /*_DEBUG*/
};
typedef struct _pos CVPOS;
struct _list
{
    void* m_buffer;
    void* m_first_buffer;
    long m_buf_size; /* The size of the buffer */
    long m_size; /* The number of elements */
    CVPOS m_head;
    CVPOS m_tail;
    CVPOS m_head_free;
};

typedef struct _list _CVLIST;

#define DECLARE_LIST(type, prefix)\
    /* Basic element of a list*/\
    struct prefix##element_##type\
    {\
        struct prefix##element_##type* m_prev;\
        struct prefix##element_##type* m_next;\
        type m_data;\
    };\
    typedef struct prefix##element_##type ELEMENT_##type;\
    /* Initialization and destruction*/\
    _LIST_INLINE _CVLIST* prefix##create_list_##type(long);\
    _LIST_INLINE void prefix##destroy_list_##type(_CVLIST*);\
    /* Access functions*/\
    _LIST_INLINE CVPOS prefix##get_head_pos_##type(_CVLIST*);\
    _LIST_INLINE CVPOS prefix##get_tail_pos_##type(_CVLIST*);\
    _LIST_INLINE type* prefix##get_next_##type(CVPOS*);\
    _LIST_INLINE type* prefix##get_prev_##type(CVPOS*);\
    _LIST_INLINE int prefix##is_pos_##type(CVPOS pos);\
    /* Modification functions*/\
    _LIST_INLINE void prefix##clear_list_##type(_CVLIST*);\
    _LIST_INLINE CVPOS prefix##add_head_##type(_CVLIST*, type*);\
    _LIST_INLINE CVPOS prefix##add_tail_##type(_CVLIST*, type*);\
    _LIST_INLINE void prefix##remove_head_##type(_CVLIST*);\
    _LIST_INLINE void prefix##remove_tail_##type(_CVLIST*);\
    _LIST_INLINE CVPOS prefix##insert_before_##type(_CVLIST*, CVPOS, type*);\
    _LIST_INLINE CVPOS prefix##insert_after_##type(_CVLIST*, CVPOS, type*);\
    _LIST_INLINE void prefix##remove_at_##type(_CVLIST*, CVPOS);\
    _LIST_INLINE void prefix##set_##type(CVPOS, type*);\
    _LIST_INLINE type* prefix##get_##type(CVPOS);\
    /* Statistics functions*/\
    _LIST_INLINE int prefix##get_count_##type(_CVLIST*);

/* This macro finds a space for a new element and puts in into 'element' pointer */
#define INSERT_NEW(element_type, l, element)\
    l->m_size++;\
    if(l->m_head_free.m_pos != NULL)\
    {\
        element = (element_type*)(l->m_head_free.m_pos);\
        if(element->m_next != NULL)\
        {\
            element->m_next->m_prev = NULL;\
            l->m_head_free.m_pos = element->m_next;\
        }\
        else\
        {\
            l->m_head_free.m_pos = NULL;\
        }\
    }\
    else\
    {\
        if(l->m_buf_size < l->m_size && l->m_head_free.m_pos == NULL)\
        {\
            *(void**)l->m_buffer = cvAlloc(l->m_buf_size*sizeof(element_type) + sizeof(void*));\
            l->m_buffer = *(void**)l->m_buffer;\
            *(void**)l->m_buffer = NULL;\
            element = (element_type*)((char*)l->m_buffer + sizeof(void*));\
        }\
        else\
        {\
            element = (element_type*)((char*)l->m_buffer + sizeof(void*)) + l->m_size - 1;\
        }\
    }

/* This macro adds 'element' to the list of free elements*/
#define INSERT_FREE(element_type, l, element)\
    if(l->m_head_free.m_pos != NULL)\
    {\
        ((element_type*)l->m_head_free.m_pos)->m_prev = element;\
    }\
    element->m_next = ((element_type*)l->m_head_free.m_pos);\
    l->m_head_free.m_pos = element;


/*#define GET_FIRST_FREE(l) ((ELEMENT_##type*)(l->m_head_free.m_pos))*/

#define IMPLEMENT_LIST(type, prefix)\
_CVLIST* prefix##create_list_##type(long size)\
{\
    _CVLIST* pl = (_CVLIST*)cvAlloc(sizeof(_CVLIST));\
    pl->m_buf_size = size > 0 ? size : default_size;\
    pl->m_first_buffer = cvAlloc(pl->m_buf_size*sizeof(ELEMENT_##type) + sizeof(void*));\
    pl->m_buffer = pl->m_first_buffer;\
    *(void**)pl->m_buffer = NULL;\
    pl->m_size = 0;\
    pl->m_head.m_pos = NULL;\
    pl->m_tail.m_pos = NULL;\
    pl->m_head_free.m_pos = NULL;\
    return pl;\
}\
void prefix##destroy_list_##type(_CVLIST* l)\
{\
    void* cur = l->m_first_buffer;\
    void* next;\
    while(cur)\
    {\
        next = *(void**)cur;\
        cvFree(&cur);\
        cur = next;\
    }\
    cvFree(&l);\
}\
CVPOS prefix##get_head_pos_##type(_CVLIST* l)\
{\
    return l->m_head;\
}\
CVPOS prefix##get_tail_pos_##type(_CVLIST* l)\
{\
    return l->m_tail;\
}\
type* prefix##get_next_##type(CVPOS* pos)\
{\
    if(pos->m_pos)\
    {\
        ELEMENT_##type* element = (ELEMENT_##type*)(pos->m_pos);\
        pos->m_pos = element->m_next;\
        return &element->m_data;\
    }\
    else\
    {\
        return NULL;\
    }\
}\
type* prefix##get_prev_##type(CVPOS* pos)\
{\
    if(pos->m_pos)\
    {\
        ELEMENT_##type* element = (ELEMENT_##type*)(pos->m_pos);\
        pos->m_pos = element->m_prev;\
        return &element->m_data;\
    }\
    else\
    {\
        return NULL;\
    }\
}\
int prefix##is_pos_##type(CVPOS pos)\
{\
    return !!pos.m_pos;\
}\
void prefix##clear_list_##type(_CVLIST* l)\
{\
    l->m_head.m_pos = NULL;\
    l->m_tail.m_pos = NULL;\
    l->m_size = 0;\
    l->m_head_free.m_pos = NULL;\
}\
CVPOS prefix##add_head_##type(_CVLIST* l, type* data)\
{\
    ELEMENT_##type* element;\
    INSERT_NEW(ELEMENT_##type, l, element);\
    element->m_prev = NULL;\
    element->m_next = (ELEMENT_##type*)(l->m_head.m_pos);\
    memcpy(&(element->m_data), data, sizeof(*data));\
    if(element->m_next)\
    {\
        element->m_next->m_prev = element;\
    }\
    else\
    {\
        l->m_tail.m_pos = element;\
    }\
    l->m_head.m_pos = element;\
    return l->m_head;\
}\
CVPOS prefix##add_tail_##type(_CVLIST* l, type* data)\
{\
    ELEMENT_##type* element;\
    INSERT_NEW(ELEMENT_##type, l, element);\
    element->m_next = NULL;\
    element->m_prev = (ELEMENT_##type*)(l->m_tail.m_pos);\
    memcpy(&(element->m_data), data, sizeof(*data));\
    if(element->m_prev)\
    {\
        element->m_prev->m_next = element;\
    }\
    else\
    {\
        l->m_head.m_pos = element;\
    }\
    l->m_tail.m_pos = element;\
    return l->m_tail;\
}\
void prefix##remove_head_##type(_CVLIST* l)\
{\
    ELEMENT_##type* element = ((ELEMENT_##type*)(l->m_head.m_pos));\
    if(element->m_next != NULL)\
    {\
        element->m_next->m_prev = NULL;\
    }\
    l->m_head.m_pos = element->m_next;\
    INSERT_FREE(ELEMENT_##type, l, element);\
    l->m_size--;\
}\
void prefix##remove_tail_##type(_CVLIST* l)\
{\
    ELEMENT_##type* element = ((ELEMENT_##type*)(l->m_tail.m_pos));\
    if(element->m_prev != NULL)\
    {\
        element->m_prev->m_next = NULL;\
    }\
    l->m_tail.m_pos = element->m_prev;\
    INSERT_FREE(ELEMENT_##type, l, element);\
    l->m_size--;\
}\
CVPOS prefix##insert_after_##type(_CVLIST* l, CVPOS pos, type* data)\
{\
    ELEMENT_##type* element;\
    ELEMENT_##type* before;\
    CVPOS newpos;\
    INSERT_NEW(ELEMENT_##type, l, element);\
    memcpy(&(element->m_data), data, sizeof(*data));\
    before = (ELEMENT_##type*)pos.m_pos;\
    element->m_prev = before;\
    element->m_next = before->m_next;\
    before->m_next = element;\
    if(element->m_next != NULL)\
        element->m_next->m_prev = element;\
    else\
        l->m_tail.m_pos = element;\
    newpos.m_pos = element;\
    return newpos;\
}\
CVPOS prefix##insert_before_##type(_CVLIST* l, CVPOS pos, type* data)\
{\
    ELEMENT_##type* element;\
    ELEMENT_##type* after;\
    CVPOS newpos;\
    INSERT_NEW(ELEMENT_##type, l, element);\
    memcpy(&(element->m_data), data, sizeof(*data));\
    after = (ELEMENT_##type*)pos.m_pos;\
    element->m_prev = after->m_prev;\
    element->m_next = after;\
    after->m_prev = element;\
    if(element->m_prev != NULL)\
        element->m_prev->m_next = element;\
    else\
        l->m_head.m_pos = element;\
    newpos.m_pos = element;\
    return newpos;\
}\
void prefix##remove_at_##type(_CVLIST* l, CVPOS pos)\
{\
    ELEMENT_##type* element = ((ELEMENT_##type*)pos.m_pos);\
    if(element->m_prev != NULL)\
    {\
        element->m_prev->m_next = element->m_next;\
    }\
    else\
    {\
        l->m_head.m_pos = element->m_next;\
    }\
    if(element->m_next != NULL)\
    {\
        element->m_next->m_prev = element->m_prev;\
    }\
    else\
    {\
        l->m_tail.m_pos = element->m_prev;\
    }\
    INSERT_FREE(ELEMENT_##type, l, element);\
    l->m_size--;\
}\
void prefix##set_##type(CVPOS pos, type* data)\
{\
    ELEMENT_##type* element = ((ELEMENT_##type*)(pos.m_pos));\
    memcpy(&(element->m_data), data, sizeof(*data));\
}\
type* prefix##get_##type(CVPOS pos)\
{\
    ELEMENT_##type* element = ((ELEMENT_##type*)(pos.m_pos));\
    return &(element->m_data);\
}\
int prefix##get_count_##type(_CVLIST* list)\
{\
    return list->m_size;\
}

#define DECLARE_AND_IMPLEMENT_LIST(type, prefix)\
    DECLARE_LIST(type, prefix)\
    IMPLEMENT_LIST(type, prefix)

typedef struct __index
{
    int value;
    float rho, theta;
}
_index;

DECLARE_LIST( _index, h_ )

#endif/*_CV_LIST_H_*/
