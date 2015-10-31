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
#include "precomp.hpp"

/* default alignment for dynamic data strucutures, resided in storages. */
#define  CV_STRUCT_ALIGN    ((int)sizeof(double))

/* default storage block size */
#define  CV_STORAGE_BLOCK_SIZE   ((1<<16) - 128)

#define ICV_FREE_PTR(storage)  \
    ((schar*)(storage)->top + (storage)->block_size - (storage)->free_space)

#define ICV_ALIGNED_SEQ_BLOCK_SIZE  \
    (int)cvAlign(sizeof(CvSeqBlock), CV_STRUCT_ALIGN)

CV_INLINE int
cvAlignLeft( int size, int align )
{
    return size & -align;
}

#define CV_GET_LAST_ELEM( seq, block ) \
    ((block)->data + ((block)->count - 1)*((seq)->elem_size))

#define CV_SWAP_ELEMS(a,b,elem_size)  \
{                                     \
    int k;                            \
    for( k = 0; k < elem_size; k++ )  \
    {                                 \
        char t0 = (a)[k];             \
        char t1 = (b)[k];             \
        (a)[k] = t1;                  \
        (b)[k] = t0;                  \
    }                                 \
}

#define ICV_SHIFT_TAB_MAX 32
static const schar icvPower2ShiftTab[] =
{
    0, 1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, 4,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5
};

/****************************************************************************************\
*            Functions for manipulating memory storage - list of memory blocks           *
\****************************************************************************************/

/* Initialize allocated storage: */
static void
icvInitMemStorage( CvMemStorage* storage, int block_size )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    if( block_size <= 0 )
        block_size = CV_STORAGE_BLOCK_SIZE;

    block_size = cvAlign( block_size, CV_STRUCT_ALIGN );
    assert( sizeof(CvMemBlock) % CV_STRUCT_ALIGN == 0 );

    memset( storage, 0, sizeof( *storage ));
    storage->signature = CV_STORAGE_MAGIC_VAL;
    storage->block_size = block_size;
}


/* Create root memory storage: */
CV_IMPL CvMemStorage*
cvCreateMemStorage( int block_size )
{
    CvMemStorage* storage = (CvMemStorage *)cvAlloc( sizeof( CvMemStorage ));
    icvInitMemStorage( storage, block_size );
    return storage;
}


/* Create child memory storage: */
CV_IMPL CvMemStorage *
cvCreateChildMemStorage( CvMemStorage * parent )
{
    if( !parent )
        CV_Error( CV_StsNullPtr, "" );

    CvMemStorage* storage = cvCreateMemStorage(parent->block_size);
    storage->parent = parent;

    return storage;
}


/* Release all blocks of the storage (or return them to parent, if any): */
static void
icvDestroyMemStorage( CvMemStorage* storage )
{
    int k = 0;

    CvMemBlock *block;
    CvMemBlock *dst_top = 0;

    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    if( storage->parent )
        dst_top = storage->parent->top;

    for( block = storage->bottom; block != 0; k++ )
    {
        CvMemBlock *temp = block;

        block = block->next;
        if( storage->parent )
        {
            if( dst_top )
            {
                temp->prev = dst_top;
                temp->next = dst_top->next;
                if( temp->next )
                    temp->next->prev = temp;
                dst_top = dst_top->next = temp;
            }
            else
            {
                dst_top = storage->parent->bottom = storage->parent->top = temp;
                temp->prev = temp->next = 0;
                storage->free_space = storage->block_size - sizeof( *temp );
            }
        }
        else
        {
            cvFree( &temp );
        }
    }

    storage->top = storage->bottom = 0;
    storage->free_space = 0;
}


/* Release memory storage: */
CV_IMPL void
cvReleaseMemStorage( CvMemStorage** storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    CvMemStorage* st = *storage;
    *storage = 0;
    if( st )
    {
        icvDestroyMemStorage( st );
        cvFree( &st );
    }
}


/* Clears memory storage (return blocks to the parent, if any): */
CV_IMPL void
cvClearMemStorage( CvMemStorage * storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    if( storage->parent )
        icvDestroyMemStorage( storage );
    else
    {
        storage->top = storage->bottom;
        storage->free_space = storage->bottom ? storage->block_size - sizeof(CvMemBlock) : 0;
    }
}


/* Moves stack pointer to next block.
   If no blocks, allocate new one and link it to the storage: */
static void
icvGoNextMemBlock( CvMemStorage * storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    if( !storage->top || !storage->top->next )
    {
        CvMemBlock *block;

        if( !(storage->parent) )
        {
            block = (CvMemBlock *)cvAlloc( storage->block_size );
        }
        else
        {
            CvMemStorage *parent = storage->parent;
            CvMemStoragePos parent_pos;

            cvSaveMemStoragePos( parent, &parent_pos );
            icvGoNextMemBlock( parent );

            block = parent->top;
            cvRestoreMemStoragePos( parent, &parent_pos );

            if( block == parent->top )  /* the single allocated block */
            {
                assert( parent->bottom == block );
                parent->top = parent->bottom = 0;
                parent->free_space = 0;
            }
            else
            {
                /* cut the block from the parent's list of blocks */
                parent->top->next = block->next;
                if( block->next )
                    block->next->prev = parent->top;
            }
        }

        /* link block */
        block->next = 0;
        block->prev = storage->top;

        if( storage->top )
            storage->top->next = block;
        else
            storage->top = storage->bottom = block;
    }

    if( storage->top->next )
        storage->top = storage->top->next;
    storage->free_space = storage->block_size - sizeof(CvMemBlock);
    assert( storage->free_space % CV_STRUCT_ALIGN == 0 );
}


/* Remember memory storage position: */
CV_IMPL void
cvSaveMemStoragePos( const CvMemStorage * storage, CvMemStoragePos * pos )
{
    if( !storage || !pos )
        CV_Error( CV_StsNullPtr, "" );

    pos->top = storage->top;
    pos->free_space = storage->free_space;
}


/* Restore memory storage position: */
CV_IMPL void
cvRestoreMemStoragePos( CvMemStorage * storage, CvMemStoragePos * pos )
{
    if( !storage || !pos )
        CV_Error( CV_StsNullPtr, "" );
    if( pos->free_space > storage->block_size )
        CV_Error( CV_StsBadSize, "" );

    /*
    // this breaks icvGoNextMemBlock, so comment it off for now
    if( storage->parent && (!pos->top || pos->top->next) )
    {
        CvMemBlock* save_bottom;
        if( !pos->top )
            save_bottom = 0;
        else
        {
            save_bottom = storage->bottom;
            storage->bottom = pos->top->next;
            pos->top->next = 0;
            storage->bottom->prev = 0;
        }
        icvDestroyMemStorage( storage );
        storage->bottom = save_bottom;
    }*/

    storage->top = pos->top;
    storage->free_space = pos->free_space;

    if( !storage->top )
    {
        storage->top = storage->bottom;
        storage->free_space = storage->top ? storage->block_size - sizeof(CvMemBlock) : 0;
    }
}


/* Allocate continuous buffer of the specified size in the storage: */
CV_IMPL void*
cvMemStorageAlloc( CvMemStorage* storage, size_t size )
{
    schar *ptr = 0;
    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    if( size > INT_MAX )
        CV_Error( CV_StsOutOfRange, "Too large memory block is requested" );

    assert( storage->free_space % CV_STRUCT_ALIGN == 0 );

    if( (size_t)storage->free_space < size )
    {
        size_t max_free_space = cvAlignLeft(storage->block_size - sizeof(CvMemBlock), CV_STRUCT_ALIGN);
        if( max_free_space < size )
            CV_Error( CV_StsOutOfRange, "requested size is negative or too big" );

        icvGoNextMemBlock( storage );
    }

    ptr = ICV_FREE_PTR(storage);
    assert( (size_t)ptr % CV_STRUCT_ALIGN == 0 );
    storage->free_space = cvAlignLeft(storage->free_space - (int)size, CV_STRUCT_ALIGN );

    return ptr;
}


CV_IMPL CvString
cvMemStorageAllocString( CvMemStorage* storage, const char* ptr, int len )
{
    CvString str;
    memset(&str, 0, sizeof(CvString));

    str.len = len >= 0 ? len : (int)strlen(ptr);
    str.ptr = (char*)cvMemStorageAlloc( storage, str.len + 1 );
    memcpy( str.ptr, ptr, str.len );
    str.ptr[str.len] = '\0';

    return str;
}


/****************************************************************************************\
*                               Sequence implementation                                  *
\****************************************************************************************/

/* Create empty sequence: */
CV_IMPL CvSeq *
cvCreateSeq( int seq_flags, size_t header_size, size_t elem_size, CvMemStorage* storage )
{
    CvSeq *seq = 0;

    if( !storage )
        CV_Error( CV_StsNullPtr, "" );
    if( header_size < sizeof( CvSeq ) || elem_size <= 0 )
        CV_Error( CV_StsBadSize, "" );

    /* allocate sequence header */
    seq = (CvSeq*)cvMemStorageAlloc( storage, header_size );
    memset( seq, 0, header_size );

    seq->header_size = (int)header_size;
    seq->flags = (seq_flags & ~CV_MAGIC_MASK) | CV_SEQ_MAGIC_VAL;
    {
        int elemtype = CV_MAT_TYPE(seq_flags);
        int typesize = CV_ELEM_SIZE(elemtype);

        if( elemtype != CV_SEQ_ELTYPE_GENERIC && elemtype != CV_USRTYPE1 &&
            typesize != 0 && typesize != (int)elem_size )
            CV_Error( CV_StsBadSize,
            "Specified element size doesn't match to the size of the specified element type "
            "(try to use 0 for element type)" );
    }
    seq->elem_size = (int)elem_size;
    seq->storage = storage;

    cvSetSeqBlockSize( seq, (int)((1 << 10)/elem_size) );

    return seq;
}


/* adjusts <delta_elems> field of sequence. It determines how much the sequence
   grows if there are no free space inside the sequence buffers */
CV_IMPL void
cvSetSeqBlockSize( CvSeq *seq, int delta_elements )
{
    int elem_size;
    int useful_block_size;

    if( !seq || !seq->storage )
        CV_Error( CV_StsNullPtr, "" );
    if( delta_elements < 0 )
        CV_Error( CV_StsOutOfRange, "" );

    useful_block_size = cvAlignLeft(seq->storage->block_size - sizeof(CvMemBlock) -
                                    sizeof(CvSeqBlock), CV_STRUCT_ALIGN);
    elem_size = seq->elem_size;

    if( delta_elements == 0 )
    {
        delta_elements = (1 << 10) / elem_size;
        delta_elements = MAX( delta_elements, 1 );
    }
    if( delta_elements * elem_size > useful_block_size )
    {
        delta_elements = useful_block_size / elem_size;
        if( delta_elements == 0 )
            CV_Error( CV_StsOutOfRange, "Storage block size is too small "
                                        "to fit the sequence elements" );
    }

    seq->delta_elems = delta_elements;
}


/* Find a sequence element by its index: */
CV_IMPL schar*
cvGetSeqElem( const CvSeq *seq, int index )
{
    CvSeqBlock *block;
    int count, total = seq->total;

    if( (unsigned)index >= (unsigned)total )
    {
        index += index < 0 ? total : 0;
        index -= index >= total ? total : 0;
        if( (unsigned)index >= (unsigned)total )
            return 0;
    }

    block = seq->first;
    if( index + index <= total )
    {
        while( index >= (count = block->count) )
        {
            block = block->next;
            index -= count;
        }
    }
    else
    {
        do
        {
            block = block->prev;
            total -= block->count;
        }
        while( index < total );
        index -= total;
    }

    return block->data + index * seq->elem_size;
}


/* Calculate index of a sequence element: */
CV_IMPL int
cvSeqElemIdx( const CvSeq* seq, const void* _element, CvSeqBlock** _block )
{
    const schar *element = (const schar *)_element;
    int elem_size;
    int id = -1;
    CvSeqBlock *first_block;
    CvSeqBlock *block;

    if( !seq || !element )
        CV_Error( CV_StsNullPtr, "" );

    block = first_block = seq->first;
    elem_size = seq->elem_size;

    for( ;; )
    {
        if( (unsigned)(element - block->data) < (unsigned) (block->count * elem_size) )
        {
            if( _block )
                *_block = block;
            if( elem_size <= ICV_SHIFT_TAB_MAX && (id = icvPower2ShiftTab[elem_size - 1]) >= 0 )
                id = (int)((size_t)(element - block->data) >> id);
            else
                id = (int)((size_t)(element - block->data) / elem_size);
            id += block->start_index - seq->first->start_index;
            break;
        }
        block = block->next;
        if( block == first_block )
            break;
    }

    return id;
}


CV_IMPL int
cvSliceLength( CvSlice slice, const CvSeq* seq )
{
    int total = seq->total;
    int length = slice.end_index - slice.start_index;

    if( length != 0 )
    {
        if( slice.start_index < 0 )
            slice.start_index += total;
        if( slice.end_index <= 0 )
            slice.end_index += total;

        length = slice.end_index - slice.start_index;
    }

    while( length < 0 )
        length += total;
    if( length > total )
        length = total;

    return length;
}


/* Copy all sequence elements into single continuous array: */
CV_IMPL void*
cvCvtSeqToArray( const CvSeq *seq, void *array, CvSlice slice )
{
    int elem_size, total;
    CvSeqReader reader;
    char *dst = (char*)array;

    if( !seq || !array )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = seq->elem_size;
    total = cvSliceLength( slice, seq )*elem_size;

    if( total == 0 )
        return 0;

    cvStartReadSeq( seq, &reader, 0 );
    cvSetSeqReaderPos( &reader, slice.start_index, 0 );

    do
    {
        int count = (int)(reader.block_max - reader.ptr);
        if( count > total )
            count = total;

        memcpy( dst, reader.ptr, count );
        dst += count;
        reader.block = reader.block->next;
        reader.ptr = reader.block->data;
        reader.block_max = reader.ptr + reader.block->count*elem_size;
        total -= count;
    }
    while( total > 0 );

    return array;
}


/* Construct a sequence from an array without copying any data.
   NB: The resultant sequence cannot grow beyond its initial size: */
CV_IMPL CvSeq*
cvMakeSeqHeaderForArray( int seq_flags, int header_size, int elem_size,
                         void *array, int total, CvSeq *seq, CvSeqBlock * block )
{
    CvSeq* result = 0;

    if( elem_size <= 0 || header_size < (int)sizeof( CvSeq ) || total < 0 )
        CV_Error( CV_StsBadSize, "" );

    if( !seq || ((!array || !block) && total > 0) )
        CV_Error( CV_StsNullPtr, "" );

    memset( seq, 0, header_size );

    seq->header_size = header_size;
    seq->flags = (seq_flags & ~CV_MAGIC_MASK) | CV_SEQ_MAGIC_VAL;
    {
        int elemtype = CV_MAT_TYPE(seq_flags);
        int typesize = CV_ELEM_SIZE(elemtype);

        if( elemtype != CV_SEQ_ELTYPE_GENERIC &&
            typesize != 0 && typesize != elem_size )
            CV_Error( CV_StsBadSize,
            "Element size doesn't match to the size of predefined element type "
            "(try to use 0 for sequence element type)" );
    }
    seq->elem_size = elem_size;
    seq->total = total;
    seq->block_max = seq->ptr = (schar *) array + total * elem_size;

    if( total > 0 )
    {
        seq->first = block;
        block->prev = block->next = block;
        block->start_index = 0;
        block->count = total;
        block->data = (schar *) array;
    }

    result = seq;

    return result;
}


/* The function allocates space for at least one more sequence element.
   If there are free sequence blocks (seq->free_blocks != 0)
   they are reused, otherwise the space is allocated in the storage: */
static void
icvGrowSeq( CvSeq *seq, int in_front_of )
{
    CvSeqBlock *block;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );
    block = seq->free_blocks;

    if( !block )
    {
        int elem_size = seq->elem_size;
        int delta_elems = seq->delta_elems;
        CvMemStorage *storage = seq->storage;

        if( seq->total >= delta_elems*4 )
            cvSetSeqBlockSize( seq, delta_elems*2 );

        if( !storage )
            CV_Error( CV_StsNullPtr, "The sequence has NULL storage pointer" );

        /* If there is a free space just after last allocated block
           and it is big enough then enlarge the last block.
           This can happen only if the new block is added to the end of sequence: */
        if( (size_t)(ICV_FREE_PTR(storage) - seq->block_max) < CV_STRUCT_ALIGN &&
            storage->free_space >= seq->elem_size && !in_front_of )
        {
            int delta = storage->free_space / elem_size;

            delta = MIN( delta, delta_elems ) * elem_size;
            seq->block_max += delta;
            storage->free_space = cvAlignLeft((int)(((schar*)storage->top + storage->block_size) -
                                              seq->block_max), CV_STRUCT_ALIGN );
            return;
        }
        else
        {
            int delta = elem_size * delta_elems + ICV_ALIGNED_SEQ_BLOCK_SIZE;

            /* Try to allocate <delta_elements> elements: */
            if( storage->free_space < delta )
            {
                int small_block_size = MAX(1, delta_elems/3)*elem_size +
                                       ICV_ALIGNED_SEQ_BLOCK_SIZE;
                /* try to allocate smaller part */
                if( storage->free_space >= small_block_size + CV_STRUCT_ALIGN )
                {
                    delta = (storage->free_space - ICV_ALIGNED_SEQ_BLOCK_SIZE)/seq->elem_size;
                    delta = delta*seq->elem_size + ICV_ALIGNED_SEQ_BLOCK_SIZE;
                }
                else
                {
                    icvGoNextMemBlock( storage );
                    assert( storage->free_space >= delta );
                }
            }

            block = (CvSeqBlock*)cvMemStorageAlloc( storage, delta );
            block->data = (schar*)cvAlignPtr( block + 1, CV_STRUCT_ALIGN );
            block->count = delta - ICV_ALIGNED_SEQ_BLOCK_SIZE;
            block->prev = block->next = 0;
        }
    }
    else
    {
        seq->free_blocks = block->next;
    }

    if( !(seq->first) )
    {
        seq->first = block;
        block->prev = block->next = block;
    }
    else
    {
        block->prev = seq->first->prev;
        block->next = seq->first;
        block->prev->next = block->next->prev = block;
    }

    /* For free blocks the <count> field means
     * total number of bytes in the block.
     *
     * For used blocks it means current number
     * of sequence elements in the block:
     */
    assert( block->count % seq->elem_size == 0 && block->count > 0 );

    if( !in_front_of )
    {
        seq->ptr = block->data;
        seq->block_max = block->data + block->count;
        block->start_index = block == block->prev ? 0 :
            block->prev->start_index + block->prev->count;
    }
    else
    {
        int delta = block->count / seq->elem_size;
        block->data += block->count;

        if( block != block->prev )
        {
            assert( seq->first->start_index == 0 );
            seq->first = block;
        }
        else
        {
            seq->block_max = seq->ptr = block->data;
        }

        block->start_index = 0;

        for( ;; )
        {
            block->start_index += delta;
            block = block->next;
            if( block == seq->first )
                break;
        }
    }

    block->count = 0;
}

/* Recycle a sequence block: */
static void
icvFreeSeqBlock( CvSeq *seq, int in_front_of )
{
    CvSeqBlock *block = seq->first;

    assert( (in_front_of ? block : block->prev)->count == 0 );

    if( block == block->prev )  /* single block case */
    {
        block->count = (int)(seq->block_max - block->data) + block->start_index * seq->elem_size;
        block->data = seq->block_max - block->count;
        seq->first = 0;
        seq->ptr = seq->block_max = 0;
        seq->total = 0;
    }
    else
    {
        if( !in_front_of )
        {
            block = block->prev;
            assert( seq->ptr == block->data );

            block->count = (int)(seq->block_max - seq->ptr);
            seq->block_max = seq->ptr = block->prev->data +
                block->prev->count * seq->elem_size;
        }
        else
        {
            int delta = block->start_index;

            block->count = delta * seq->elem_size;
            block->data -= block->count;

            /* Update start indices of sequence blocks: */
            for( ;; )
            {
                block->start_index -= delta;
                block = block->next;
                if( block == seq->first )
                    break;
            }

            seq->first = block->next;
        }

        block->prev->next = block->next;
        block->next->prev = block->prev;
    }

    assert( block->count > 0 && block->count % seq->elem_size == 0 );
    block->next = seq->free_blocks;
    seq->free_blocks = block;
}


/****************************************************************************************\
*                             Sequence Writer implementation                             *
\****************************************************************************************/

/* Initialize sequence writer: */
CV_IMPL void
cvStartAppendToSeq( CvSeq *seq, CvSeqWriter * writer )
{
    if( !seq || !writer )
        CV_Error( CV_StsNullPtr, "" );

    memset( writer, 0, sizeof( *writer ));
    writer->header_size = sizeof( CvSeqWriter );

    writer->seq = seq;
    writer->block = seq->first ? seq->first->prev : 0;
    writer->ptr = seq->ptr;
    writer->block_max = seq->block_max;
}


/* Initialize sequence writer: */
CV_IMPL void
cvStartWriteSeq( int seq_flags, int header_size,
                 int elem_size, CvMemStorage * storage, CvSeqWriter * writer )
{
    if( !storage || !writer )
        CV_Error( CV_StsNullPtr, "" );

    CvSeq* seq = cvCreateSeq( seq_flags, header_size, elem_size, storage );
    cvStartAppendToSeq( seq, writer );
}


/* Update sequence header: */
CV_IMPL void
cvFlushSeqWriter( CvSeqWriter * writer )
{
    if( !writer )
        CV_Error( CV_StsNullPtr, "" );

    CvSeq* seq = writer->seq;
    seq->ptr = writer->ptr;

    if( writer->block )
    {
        int total = 0;
        CvSeqBlock *first_block = writer->seq->first;
        CvSeqBlock *block = first_block;

        writer->block->count = (int)((writer->ptr - writer->block->data) / seq->elem_size);
        assert( writer->block->count > 0 );

        do
        {
            total += block->count;
            block = block->next;
        }
        while( block != first_block );

        writer->seq->total = total;
    }
}


/* Calls icvFlushSeqWriter and finishes writing process: */
CV_IMPL CvSeq *
cvEndWriteSeq( CvSeqWriter * writer )
{
    if( !writer )
        CV_Error( CV_StsNullPtr, "" );

    cvFlushSeqWriter( writer );
    CvSeq* seq = writer->seq;

    /* Truncate the last block: */
    if( writer->block && writer->seq->storage )
    {
        CvMemStorage *storage = seq->storage;
        schar *storage_block_max = (schar *) storage->top + storage->block_size;

        assert( writer->block->count > 0 );

        if( (unsigned)((storage_block_max - storage->free_space)
            - seq->block_max) < CV_STRUCT_ALIGN )
        {
            storage->free_space = cvAlignLeft((int)(storage_block_max - seq->ptr), CV_STRUCT_ALIGN);
            seq->block_max = seq->ptr;
        }
    }

    writer->ptr = 0;
    return seq;
}


/* Create new sequence block: */
CV_IMPL void
cvCreateSeqBlock( CvSeqWriter * writer )
{
    if( !writer || !writer->seq )
        CV_Error( CV_StsNullPtr, "" );

    CvSeq* seq = writer->seq;

    cvFlushSeqWriter( writer );

    icvGrowSeq( seq, 0 );

    writer->block = seq->first->prev;
    writer->ptr = seq->ptr;
    writer->block_max = seq->block_max;
}


/****************************************************************************************\
*                               Sequence Reader implementation                           *
\****************************************************************************************/

/* Initialize sequence reader: */
CV_IMPL void
cvStartReadSeq( const CvSeq *seq, CvSeqReader * reader, int reverse )
{
    CvSeqBlock *first_block;
    CvSeqBlock *last_block;

    if( reader )
    {
        reader->seq = 0;
        reader->block = 0;
        reader->ptr = reader->block_max = reader->block_min = 0;
    }

    if( !seq || !reader )
        CV_Error( CV_StsNullPtr, "" );

    reader->header_size = sizeof( CvSeqReader );
    reader->seq = (CvSeq*)seq;

    first_block = seq->first;

    if( first_block )
    {
        last_block = first_block->prev;
        reader->ptr = first_block->data;
        reader->prev_elem = CV_GET_LAST_ELEM( seq, last_block );
        reader->delta_index = seq->first->start_index;

        if( reverse )
        {
            schar *temp = reader->ptr;

            reader->ptr = reader->prev_elem;
            reader->prev_elem = temp;

            reader->block = last_block;
        }
        else
        {
            reader->block = first_block;
        }

        reader->block_min = reader->block->data;
        reader->block_max = reader->block_min + reader->block->count * seq->elem_size;
    }
    else
    {
        reader->delta_index = 0;
        reader->block = 0;

        reader->ptr = reader->prev_elem = reader->block_min = reader->block_max = 0;
    }
}


/* Change the current reading block
 * to the previous or to the next:
 */
CV_IMPL void
cvChangeSeqBlock( void* _reader, int direction )
{
    CvSeqReader* reader = (CvSeqReader*)_reader;

    if( !reader )
        CV_Error( CV_StsNullPtr, "" );

    if( direction > 0 )
    {
        reader->block = reader->block->next;
        reader->ptr = reader->block->data;
    }
    else
    {
        reader->block = reader->block->prev;
        reader->ptr = CV_GET_LAST_ELEM( reader->seq, reader->block );
    }
    reader->block_min = reader->block->data;
    reader->block_max = reader->block_min + reader->block->count * reader->seq->elem_size;
}


/* Return the current reader position: */
CV_IMPL int
cvGetSeqReaderPos( CvSeqReader* reader )
{
    int elem_size;
    int index = -1;

    if( !reader || !reader->ptr )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = reader->seq->elem_size;
    if( elem_size <= ICV_SHIFT_TAB_MAX && (index = icvPower2ShiftTab[elem_size - 1]) >= 0 )
        index = (int)((reader->ptr - reader->block_min) >> index);
    else
        index = (int)((reader->ptr - reader->block_min) / elem_size);

    index += reader->block->start_index - reader->delta_index;

    return index;
}


/* Set reader position to given position,
 * either absolute or relative to the
 *  current one:
 */
CV_IMPL void
cvSetSeqReaderPos( CvSeqReader* reader, int index, int is_relative )
{
    CvSeqBlock *block;
    int elem_size, count, total;

    if( !reader || !reader->seq )
        CV_Error( CV_StsNullPtr, "" );

    total = reader->seq->total;
    elem_size = reader->seq->elem_size;

    if( !is_relative )
    {
        if( index < 0 )
        {
            if( index < -total )
                CV_Error( CV_StsOutOfRange, "" );
            index += total;
        }
        else if( index >= total )
        {
            index -= total;
            if( index >= total )
                CV_Error( CV_StsOutOfRange, "" );
        }

        block = reader->seq->first;
        if( index >= (count = block->count) )
        {
            if( index + index <= total )
            {
                do
                {
                    block = block->next;
                    index -= count;
                }
                while( index >= (count = block->count) );
            }
            else
            {
                do
                {
                    block = block->prev;
                    total -= block->count;
                }
                while( index < total );
                index -= total;
            }
        }
        reader->ptr = block->data + index * elem_size;
        if( reader->block != block )
        {
            reader->block = block;
            reader->block_min = block->data;
            reader->block_max = block->data + block->count * elem_size;
        }
    }
    else
    {
        schar* ptr = reader->ptr;
        index *= elem_size;
        block = reader->block;

        if( index > 0 )
        {
            while( ptr + index >= reader->block_max )
            {
                int delta = (int)(reader->block_max - ptr);
                index -= delta;
                reader->block = block = block->next;
                reader->block_min = ptr = block->data;
                reader->block_max = block->data + block->count*elem_size;
            }
            reader->ptr = ptr + index;
        }
        else
        {
            while( ptr + index < reader->block_min )
            {
                int delta = (int)(ptr - reader->block_min);
                index += delta;
                reader->block = block = block->prev;
                reader->block_min = block->data;
                reader->block_max = ptr = block->data + block->count*elem_size;
            }
            reader->ptr = ptr + index;
        }
    }
}


/* Push element onto the sequence: */
CV_IMPL schar*
cvSeqPush( CvSeq *seq, const void *element )
{
    schar *ptr = 0;
    size_t elem_size;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = seq->elem_size;
    ptr = seq->ptr;

    if( ptr >= seq->block_max )
    {
        icvGrowSeq( seq, 0 );

        ptr = seq->ptr;
        assert( ptr + elem_size <= seq->block_max /*&& ptr == seq->block_min */  );
    }

    if( element )
        memcpy( ptr, element, elem_size );
    seq->first->prev->count++;
    seq->total++;
    seq->ptr = ptr + elem_size;

    return ptr;
}


/* Pop last element off of the sequence: */
CV_IMPL void
cvSeqPop( CvSeq *seq, void *element )
{
    schar *ptr;
    int elem_size;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );
    if( seq->total <= 0 )
        CV_Error( CV_StsBadSize, "" );

    elem_size = seq->elem_size;
    seq->ptr = ptr = seq->ptr - elem_size;

    if( element )
        memcpy( element, ptr, elem_size );
    seq->ptr = ptr;
    seq->total--;

    if( --(seq->first->prev->count) == 0 )
    {
        icvFreeSeqBlock( seq, 0 );
        assert( seq->ptr == seq->block_max );
    }
}


/* Push element onto the front of the sequence: */
CV_IMPL schar*
cvSeqPushFront( CvSeq *seq, const void *element )
{
    schar* ptr = 0;
    int elem_size;
    CvSeqBlock *block;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = seq->elem_size;
    block = seq->first;

    if( !block || block->start_index == 0 )
    {
        icvGrowSeq( seq, 1 );

        block = seq->first;
        assert( block->start_index > 0 );
    }

    ptr = block->data -= elem_size;

    if( element )
        memcpy( ptr, element, elem_size );
    block->count++;
    block->start_index--;
    seq->total++;

    return ptr;
}


/* Shift out first element of the sequence: */
CV_IMPL void
cvSeqPopFront( CvSeq *seq, void *element )
{
    int elem_size;
    CvSeqBlock *block;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );
    if( seq->total <= 0 )
        CV_Error( CV_StsBadSize, "" );

    elem_size = seq->elem_size;
    block = seq->first;

    if( element )
        memcpy( element, block->data, elem_size );
    block->data += elem_size;
    block->start_index++;
    seq->total--;

    if( --(block->count) == 0 )
        icvFreeSeqBlock( seq, 1 );
}

/* Insert new element in middle of sequence: */
CV_IMPL schar*
cvSeqInsert( CvSeq *seq, int before_index, const void *element )
{
    int elem_size;
    int block_size;
    CvSeqBlock *block;
    int delta_index;
    int total;
    schar* ret_ptr = 0;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );

    total = seq->total;
    before_index += before_index < 0 ? total : 0;
    before_index -= before_index > total ? total : 0;

    if( (unsigned)before_index > (unsigned)total )
        CV_Error( CV_StsOutOfRange, "" );

    if( before_index == total )
    {
        ret_ptr = cvSeqPush( seq, element );
    }
    else if( before_index == 0 )
    {
        ret_ptr = cvSeqPushFront( seq, element );
    }
    else
    {
        elem_size = seq->elem_size;

        if( before_index >= total >> 1 )
        {
            schar *ptr = seq->ptr + elem_size;

            if( ptr > seq->block_max )
            {
                icvGrowSeq( seq, 0 );

                ptr = seq->ptr + elem_size;
                assert( ptr <= seq->block_max );
            }

            delta_index = seq->first->start_index;
            block = seq->first->prev;
            block->count++;
            block_size = (int)(ptr - block->data);

            while( before_index < block->start_index - delta_index )
            {
                CvSeqBlock *prev_block = block->prev;

                memmove( block->data + elem_size, block->data, block_size - elem_size );
                block_size = prev_block->count * elem_size;
                memcpy( block->data, prev_block->data + block_size - elem_size, elem_size );
                block = prev_block;

                /* Check that we don't fall into an infinite loop: */
                assert( block != seq->first->prev );
            }

            before_index = (before_index - block->start_index + delta_index) * elem_size;
            memmove( block->data + before_index + elem_size, block->data + before_index,
                     block_size - before_index - elem_size );

            ret_ptr = block->data + before_index;

            if( element )
                memcpy( ret_ptr, element, elem_size );
            seq->ptr = ptr;
        }
        else
        {
            block = seq->first;

            if( block->start_index == 0 )
            {
                icvGrowSeq( seq, 1 );

                block = seq->first;
            }

            delta_index = block->start_index;
            block->count++;
            block->start_index--;
            block->data -= elem_size;

            while( before_index > block->start_index - delta_index + block->count )
            {
                CvSeqBlock *next_block = block->next;

                block_size = block->count * elem_size;
                memmove( block->data, block->data + elem_size, block_size - elem_size );
                memcpy( block->data + block_size - elem_size, next_block->data, elem_size );
                block = next_block;

                /* Check that we don't fall into an infinite loop: */
                assert( block != seq->first );
            }

            before_index = (before_index - block->start_index + delta_index) * elem_size;
            memmove( block->data, block->data + elem_size, before_index - elem_size );

            ret_ptr = block->data + before_index - elem_size;

            if( element )
                memcpy( ret_ptr, element, elem_size );
        }

        seq->total = total + 1;
    }

    return ret_ptr;
}


/* Removes element from sequence: */
CV_IMPL void
cvSeqRemove( CvSeq *seq, int index )
{
    schar *ptr;
    int elem_size;
    int block_size;
    CvSeqBlock *block;
    int delta_index;
    int total, front = 0;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );

    total = seq->total;

    index += index < 0 ? total : 0;
    index -= index >= total ? total : 0;

    if( (unsigned) index >= (unsigned) total )
        CV_Error( CV_StsOutOfRange, "Invalid index" );

    if( index == total - 1 )
    {
        cvSeqPop( seq, 0 );
    }
    else if( index == 0 )
    {
        cvSeqPopFront( seq, 0 );
    }
    else
    {
        block = seq->first;
        elem_size = seq->elem_size;
        delta_index = block->start_index;
        while( block->start_index - delta_index + block->count <= index )
            block = block->next;

        ptr = block->data + (index - block->start_index + delta_index) * elem_size;

        front = index < total >> 1;
        if( !front )
        {
            block_size = block->count * elem_size - (int)(ptr - block->data);

            while( block != seq->first->prev )  /* while not the last block */
            {
                CvSeqBlock *next_block = block->next;

                memmove( ptr, ptr + elem_size, block_size - elem_size );
                memcpy( ptr + block_size - elem_size, next_block->data, elem_size );
                block = next_block;
                ptr = block->data;
                block_size = block->count * elem_size;
            }

            memmove( ptr, ptr + elem_size, block_size - elem_size );
            seq->ptr -= elem_size;
        }
        else
        {
            ptr += elem_size;
            block_size = (int)(ptr - block->data);

            while( block != seq->first )
            {
                CvSeqBlock *prev_block = block->prev;

                memmove( block->data + elem_size, block->data, block_size - elem_size );
                block_size = prev_block->count * elem_size;
                memcpy( block->data, prev_block->data + block_size - elem_size, elem_size );
                block = prev_block;
            }

            memmove( block->data + elem_size, block->data, block_size - elem_size );
            block->data += elem_size;
            block->start_index++;
        }

        seq->total = total - 1;
        if( --block->count == 0 )
            icvFreeSeqBlock( seq, front );
    }
}


/* Add several elements to the beginning or end of a sequence: */
CV_IMPL void
cvSeqPushMulti( CvSeq *seq, const void *_elements, int count, int front )
{
    char *elements = (char *) _elements;

    if( !seq )
        CV_Error( CV_StsNullPtr, "NULL sequence pointer" );
    if( count < 0 )
        CV_Error( CV_StsBadSize, "number of removed elements is negative" );

    int elem_size = seq->elem_size;

    if( !front )
    {
        while( count > 0 )
        {
            int delta = (int)((seq->block_max - seq->ptr) / elem_size);

            delta = MIN( delta, count );
            if( delta > 0 )
            {
                seq->first->prev->count += delta;
                seq->total += delta;
                count -= delta;
                delta *= elem_size;
                if( elements )
                {
                    memcpy( seq->ptr, elements, delta );
                    elements += delta;
                }
                seq->ptr += delta;
            }

            if( count > 0 )
                icvGrowSeq( seq, 0 );
        }
    }
    else
    {
        CvSeqBlock* block = seq->first;

        while( count > 0 )
        {
            int delta;

            if( !block || block->start_index == 0 )
            {
                icvGrowSeq( seq, 1 );

                block = seq->first;
                assert( block->start_index > 0 );
            }

            delta = MIN( block->start_index, count );
            count -= delta;
            block->start_index -= delta;
            block->count += delta;
            seq->total += delta;
            delta *= elem_size;
            block->data -= delta;

            if( elements )
                memcpy( block->data, elements + count*elem_size, delta );
        }
    }
}


/* Remove several elements from the end of sequence: */
CV_IMPL void
cvSeqPopMulti( CvSeq *seq, void *_elements, int count, int front )
{
    char *elements = (char *) _elements;

    if( !seq )
        CV_Error( CV_StsNullPtr, "NULL sequence pointer" );
    if( count < 0 )
        CV_Error( CV_StsBadSize, "number of removed elements is negative" );

    count = MIN( count, seq->total );

    if( !front )
    {
        if( elements )
            elements += count * seq->elem_size;

        while( count > 0 )
        {
            int delta = seq->first->prev->count;

            delta = MIN( delta, count );
            assert( delta > 0 );

            seq->first->prev->count -= delta;
            seq->total -= delta;
            count -= delta;
            delta *= seq->elem_size;
            seq->ptr -= delta;

            if( elements )
            {
                elements -= delta;
                memcpy( elements, seq->ptr, delta );
            }

            if( seq->first->prev->count == 0 )
                icvFreeSeqBlock( seq, 0 );
        }
    }
    else
    {
        while( count > 0 )
        {
            int delta = seq->first->count;

            delta = MIN( delta, count );
            assert( delta > 0 );

            seq->first->count -= delta;
            seq->total -= delta;
            count -= delta;
            seq->first->start_index += delta;
            delta *= seq->elem_size;

            if( elements )
            {
                memcpy( elements, seq->first->data, delta );
                elements += delta;
            }

            seq->first->data += delta;
            if( seq->first->count == 0 )
                icvFreeSeqBlock( seq, 1 );
        }
    }
}


/* Remove all elements from a sequence: */
CV_IMPL void
cvClearSeq( CvSeq *seq )
{
    if( !seq )
        CV_Error( CV_StsNullPtr, "" );
    cvSeqPopMulti( seq, 0, seq->total );
}


CV_IMPL CvSeq*
cvSeqSlice( const CvSeq* seq, CvSlice slice, CvMemStorage* storage, int copy_data )
{
    CvSeq* subseq = 0;
    int elem_size, count, length;
    CvSeqReader reader;
    CvSeqBlock *block, *first_block = 0, *last_block = 0;

    if( !CV_IS_SEQ(seq) )
        CV_Error( CV_StsBadArg, "Invalid sequence header" );

    if( !storage )
    {
        storage = seq->storage;
        if( !storage )
            CV_Error( CV_StsNullPtr, "NULL storage pointer" );
    }

    elem_size = seq->elem_size;
    length = cvSliceLength( slice, seq );
    if( slice.start_index < 0 )
        slice.start_index += seq->total;
    else if( slice.start_index >= seq->total )
        slice.start_index -= seq->total;
    if( (unsigned)length > (unsigned)seq->total ||
        ((unsigned)slice.start_index >= (unsigned)seq->total && length != 0) )
        CV_Error( CV_StsOutOfRange, "Bad sequence slice" );

    subseq = cvCreateSeq( seq->flags, seq->header_size, elem_size, storage );

    if( length > 0 )
    {
        cvStartReadSeq( seq, &reader, 0 );
        cvSetSeqReaderPos( &reader, slice.start_index, 0 );
        count = (int)((reader.block_max - reader.ptr)/elem_size);

        do
        {
            int bl = MIN( count, length );

            if( !copy_data )
            {
                block = (CvSeqBlock*)cvMemStorageAlloc( storage, sizeof(*block) );
                if( !first_block )
                {
                    first_block = subseq->first = block->prev = block->next = block;
                    block->start_index = 0;
                }
                else
                {
                    block->prev = last_block;
                    block->next = first_block;
                    last_block->next = first_block->prev = block;
                    block->start_index = last_block->start_index + last_block->count;
                }
                last_block = block;
                block->data = reader.ptr;
                block->count = bl;
                subseq->total += bl;
            }
            else
                cvSeqPushMulti( subseq, reader.ptr, bl, 0 );
            length -= bl;
            reader.block = reader.block->next;
            reader.ptr = reader.block->data;
            count = reader.block->count;
        }
        while( length > 0 );
    }

    return subseq;
}


// Remove slice from the middle of the sequence.
// !!! TODO !!! Implement more efficient algorithm
CV_IMPL void
cvSeqRemoveSlice( CvSeq* seq, CvSlice slice )
{
    int total, length;

    if( !CV_IS_SEQ(seq) )
        CV_Error( CV_StsBadArg, "Invalid sequence header" );

    length = cvSliceLength( slice, seq );
    total = seq->total;

    if( slice.start_index < 0 )
        slice.start_index += total;
    else if( slice.start_index >= total )
        slice.start_index -= total;

    if( (unsigned)slice.start_index >= (unsigned)total )
        CV_Error( CV_StsOutOfRange, "start slice index is out of range" );

    slice.end_index = slice.start_index + length;

    if ( slice.start_index == slice.end_index )
        return;

    if( slice.end_index < total )
    {
        CvSeqReader reader_to, reader_from;
        int elem_size = seq->elem_size;

        cvStartReadSeq( seq, &reader_to );
        cvStartReadSeq( seq, &reader_from );

        if( slice.start_index > total - slice.end_index )
        {
            int i, count = seq->total - slice.end_index;
            cvSetSeqReaderPos( &reader_to, slice.start_index );
            cvSetSeqReaderPos( &reader_from, slice.end_index );

            for( i = 0; i < count; i++ )
            {
                memcpy( reader_to.ptr, reader_from.ptr, elem_size );
                CV_NEXT_SEQ_ELEM( elem_size, reader_to );
                CV_NEXT_SEQ_ELEM( elem_size, reader_from );
            }

            cvSeqPopMulti( seq, 0, slice.end_index - slice.start_index );
        }
        else
        {
            int i, count = slice.start_index;
            cvSetSeqReaderPos( &reader_to, slice.end_index );
            cvSetSeqReaderPos( &reader_from, slice.start_index );

            for( i = 0; i < count; i++ )
            {
                CV_PREV_SEQ_ELEM( elem_size, reader_to );
                CV_PREV_SEQ_ELEM( elem_size, reader_from );

                memcpy( reader_to.ptr, reader_from.ptr, elem_size );
            }

            cvSeqPopMulti( seq, 0, slice.end_index - slice.start_index, 1 );
        }
    }
    else
    {
        cvSeqPopMulti( seq, 0, total - slice.start_index );
        cvSeqPopMulti( seq, 0, slice.end_index - total, 1 );
    }
}


// Insert a sequence into the middle of another sequence:
// !!! TODO !!! Implement more efficient algorithm
CV_IMPL void
cvSeqInsertSlice( CvSeq* seq, int index, const CvArr* from_arr )
{
    CvSeqReader reader_to, reader_from;
    int i, elem_size, total, from_total;
    CvSeq from_header, *from = (CvSeq*)from_arr;
    CvSeqBlock block;

    if( !CV_IS_SEQ(seq) )
        CV_Error( CV_StsBadArg, "Invalid destination sequence header" );

    if( !CV_IS_SEQ(from))
    {
        CvMat* mat = (CvMat*)from;
        if( !CV_IS_MAT(mat))
            CV_Error( CV_StsBadArg, "Source is not a sequence nor matrix" );

        if( !CV_IS_MAT_CONT(mat->type) || (mat->rows != 1 && mat->cols != 1) )
            CV_Error( CV_StsBadArg, "The source array must be 1d coninuous vector" );

        from = cvMakeSeqHeaderForArray( CV_SEQ_KIND_GENERIC, sizeof(from_header),
                                                 CV_ELEM_SIZE(mat->type),
                                                 mat->data.ptr, mat->cols + mat->rows - 1,
                                                 &from_header, &block );
    }

    if( seq->elem_size != from->elem_size )
        CV_Error( CV_StsUnmatchedSizes,
        "Source and destination sequence element sizes are different." );

    from_total = from->total;

    if( from_total == 0 )
        return;

    total = seq->total;
    index += index < 0 ? total : 0;
    index -= index > total ? total : 0;

    if( (unsigned)index > (unsigned)total )
        CV_Error( CV_StsOutOfRange, "" );

    elem_size = seq->elem_size;

    if( index < (total >> 1) )
    {
        cvSeqPushMulti( seq, 0, from_total, 1 );

        cvStartReadSeq( seq, &reader_to );
        cvStartReadSeq( seq, &reader_from );
        cvSetSeqReaderPos( &reader_from, from_total );

        for( i = 0; i < index; i++ )
        {
            memcpy( reader_to.ptr, reader_from.ptr, elem_size );
            CV_NEXT_SEQ_ELEM( elem_size, reader_to );
            CV_NEXT_SEQ_ELEM( elem_size, reader_from );
        }
    }
    else
    {
        cvSeqPushMulti( seq, 0, from_total );

        cvStartReadSeq( seq, &reader_to );
        cvStartReadSeq( seq, &reader_from );
        cvSetSeqReaderPos( &reader_from, total );
        cvSetSeqReaderPos( &reader_to, seq->total );

        for( i = 0; i < total - index; i++ )
        {
            CV_PREV_SEQ_ELEM( elem_size, reader_to );
            CV_PREV_SEQ_ELEM( elem_size, reader_from );
            memcpy( reader_to.ptr, reader_from.ptr, elem_size );
        }
    }

    cvStartReadSeq( from, &reader_from );
    cvSetSeqReaderPos( &reader_to, index );

    for( i = 0; i < from_total; i++ )
    {
        memcpy( reader_to.ptr, reader_from.ptr, elem_size );
        CV_NEXT_SEQ_ELEM( elem_size, reader_to );
        CV_NEXT_SEQ_ELEM( elem_size, reader_from );
    }
}

// Sort the sequence using user-specified comparison function.
// The semantics is similar to qsort() function.
// The code is based on BSD system qsort():
//    * Copyright (c) 1992, 1993
//    *  The Regents of the University of California.  All rights reserved.
//    *
//    * Redistribution and use in source and binary forms, with or without
//    * modification, are permitted provided that the following conditions
//    * are met:
//    * 1. Redistributions of source code must retain the above copyright
//    *    notice, this list of conditions and the following disclaimer.
//    * 2. Redistributions in binary form must reproduce the above copyright
//    *    notice, this list of conditions and the following disclaimer in the
//    *    documentation and/or other materials provided with the distribution.
//    * 3. All advertising materials mentioning features or use of this software
//    *    must display the following acknowledgement:
//    *  This product includes software developed by the University of
//    *  California, Berkeley and its contributors.
//    * 4. Neither the name of the University nor the names of its contributors
//    *    may be used to endorse or promote products derived from this software
//    *    without specific prior written permission.
//    *
//    * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
//    * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//    * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
//    * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
//    * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//    * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//    * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
//    * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//    * SUCH DAMAGE.

typedef struct CvSeqReaderPos
{
    CvSeqBlock* block;
    schar* ptr;
    schar* block_min;
    schar* block_max;
}
CvSeqReaderPos;

#define CV_SAVE_READER_POS( reader, pos )   \
{                                           \
    (pos).block = (reader).block;           \
    (pos).ptr = (reader).ptr;               \
    (pos).block_min = (reader).block_min;   \
    (pos).block_max = (reader).block_max;   \
}

#define CV_RESTORE_READER_POS( reader, pos )\
{                                           \
    (reader).block = (pos).block;           \
    (reader).ptr = (pos).ptr;               \
    (reader).block_min = (pos).block_min;   \
    (reader).block_max = (pos).block_max;   \
}

inline schar*
icvMed3( schar* a, schar* b, schar* c, CvCmpFunc cmp_func, void* aux )
{
    return cmp_func(a, b, aux) < 0 ?
      (cmp_func(b, c, aux) < 0 ? b : cmp_func(a, c, aux) < 0 ? c : a)
     :(cmp_func(b, c, aux) > 0 ? b : cmp_func(a, c, aux) < 0 ? a : c);
}

CV_IMPL void
cvSeqSort( CvSeq* seq, CvCmpFunc cmp_func, void* aux )
{
    int elem_size;
    int isort_thresh = 7;
    CvSeqReader left, right;
    int sp = 0;

    struct
    {
        CvSeqReaderPos lb;
        CvSeqReaderPos ub;
    }
    stack[48];

    if( !CV_IS_SEQ(seq) )
        CV_Error( !seq ? CV_StsNullPtr : CV_StsBadArg, "Bad input sequence" );

    if( !cmp_func )
        CV_Error( CV_StsNullPtr, "Null compare function" );

    if( seq->total <= 1 )
        return;

    elem_size = seq->elem_size;
    isort_thresh *= elem_size;

    cvStartReadSeq( seq, &left, 0 );
    right = left;
    CV_SAVE_READER_POS( left, stack[0].lb );
    CV_PREV_SEQ_ELEM( elem_size, right );
    CV_SAVE_READER_POS( right, stack[0].ub );

    while( sp >= 0 )
    {
        CV_RESTORE_READER_POS( left, stack[sp].lb );
        CV_RESTORE_READER_POS( right, stack[sp].ub );
        sp--;

        for(;;)
        {
            int i, n, m;
            CvSeqReader ptr, ptr2;

            if( left.block == right.block )
                n = (int)(right.ptr - left.ptr) + elem_size;
            else
            {
                n = cvGetSeqReaderPos( &right );
                n = (n - cvGetSeqReaderPos( &left ) + 1)*elem_size;
            }

            if( n <= isort_thresh )
            {
            insert_sort:
                ptr = ptr2 = left;
                CV_NEXT_SEQ_ELEM( elem_size, ptr );
                CV_NEXT_SEQ_ELEM( elem_size, right );
                while( ptr.ptr != right.ptr )
                {
                    ptr2.ptr = ptr.ptr;
                    if( ptr2.block != ptr.block )
                    {
                        ptr2.block = ptr.block;
                        ptr2.block_min = ptr.block_min;
                        ptr2.block_max = ptr.block_max;
                    }
                    while( ptr2.ptr != left.ptr )
                    {
                        schar* cur = ptr2.ptr;
                        CV_PREV_SEQ_ELEM( elem_size, ptr2 );
                        if( cmp_func( ptr2.ptr, cur, aux ) <= 0 )
                            break;
                        CV_SWAP_ELEMS( ptr2.ptr, cur, elem_size );
                    }
                    CV_NEXT_SEQ_ELEM( elem_size, ptr );
                }
                break;
            }
            else
            {
                CvSeqReader left0, left1, right0, right1;
                CvSeqReader tmp0, tmp1;
                schar *m1, *m2, *m3, *pivot;
                int swap_cnt = 0;
                int l, l0, l1, r, r0, r1;

                left0 = tmp0 = left;
                right0 = right1 = right;
                n /= elem_size;

                if( n > 40 )
                {
                    int d = n / 8;
                    schar *p1, *p2, *p3;
                    p1 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, d, 1 );
                    p2 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, d, 1 );
                    p3 = tmp0.ptr;
                    m1 = icvMed3( p1, p2, p3, cmp_func, aux );
                    cvSetSeqReaderPos( &tmp0, (n/2) - d*3, 1 );
                    p1 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, d, 1 );
                    p2 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, d, 1 );
                    p3 = tmp0.ptr;
                    m2 = icvMed3( p1, p2, p3, cmp_func, aux );
                    cvSetSeqReaderPos( &tmp0, n - 1 - d*3 - n/2, 1 );
                    p1 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, d, 1 );
                    p2 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, d, 1 );
                    p3 = tmp0.ptr;
                    m3 = icvMed3( p1, p2, p3, cmp_func, aux );
                }
                else
                {
                    m1 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, n/2, 1 );
                    m2 = tmp0.ptr;
                    cvSetSeqReaderPos( &tmp0, n - 1 - n/2, 1 );
                    m3 = tmp0.ptr;
                }

                pivot = icvMed3( m1, m2, m3, cmp_func, aux );
                left = left0;
                if( pivot != left.ptr )
                {
                    CV_SWAP_ELEMS( pivot, left.ptr, elem_size );
                    pivot = left.ptr;
                }
                CV_NEXT_SEQ_ELEM( elem_size, left );
                left1 = left;

                for(;;)
                {
                    while( left.ptr != right.ptr && (r = cmp_func(left.ptr, pivot, aux)) <= 0 )
                    {
                        if( r == 0 )
                        {
                            if( left1.ptr != left.ptr )
                                CV_SWAP_ELEMS( left1.ptr, left.ptr, elem_size );
                            swap_cnt = 1;
                            CV_NEXT_SEQ_ELEM( elem_size, left1 );
                        }
                        CV_NEXT_SEQ_ELEM( elem_size, left );
                    }

                    while( left.ptr != right.ptr && (r = cmp_func(right.ptr,pivot, aux)) >= 0 )
                    {
                        if( r == 0 )
                        {
                            if( right1.ptr != right.ptr )
                                CV_SWAP_ELEMS( right1.ptr, right.ptr, elem_size );
                            swap_cnt = 1;
                            CV_PREV_SEQ_ELEM( elem_size, right1 );
                        }
                        CV_PREV_SEQ_ELEM( elem_size, right );
                    }

                    if( left.ptr == right.ptr )
                    {
                        r = cmp_func(left.ptr, pivot, aux);
                        if( r == 0 )
                        {
                            if( left1.ptr != left.ptr )
                                CV_SWAP_ELEMS( left1.ptr, left.ptr, elem_size );
                            swap_cnt = 1;
                            CV_NEXT_SEQ_ELEM( elem_size, left1 );
                        }
                        if( r <= 0 )
                        {
                            CV_NEXT_SEQ_ELEM( elem_size, left );
                        }
                        else
                        {
                            CV_PREV_SEQ_ELEM( elem_size, right );
                        }
                        break;
                    }

                    CV_SWAP_ELEMS( left.ptr, right.ptr, elem_size );
                    CV_NEXT_SEQ_ELEM( elem_size, left );
                    r = left.ptr == right.ptr;
                    CV_PREV_SEQ_ELEM( elem_size, right );
                    swap_cnt = 1;
                    if( r )
                        break;
                }

                if( swap_cnt == 0 )
                {
                    left = left0, right = right0;
                    goto insert_sort;
                }

                l = cvGetSeqReaderPos( &left );
                if( l == 0 )
                    l = seq->total;
                l0 = cvGetSeqReaderPos( &left0 );
                l1 = cvGetSeqReaderPos( &left1 );
                if( l1 == 0 )
                    l1 = seq->total;

                n = MIN( l - l1, l1 - l0 );
                if( n > 0 )
                {
                    tmp0 = left0;
                    tmp1 = left;
                    cvSetSeqReaderPos( &tmp1, 0-n, 1 );
                    for( i = 0; i < n; i++ )
                    {
                        CV_SWAP_ELEMS( tmp0.ptr, tmp1.ptr, elem_size );
                        CV_NEXT_SEQ_ELEM( elem_size, tmp0 );
                        CV_NEXT_SEQ_ELEM( elem_size, tmp1 );
                    }
                }

                r = cvGetSeqReaderPos( &right );
                r0 = cvGetSeqReaderPos( &right0 );
                r1 = cvGetSeqReaderPos( &right1 );
                m = MIN( r0 - r1, r1 - r );
                if( m > 0 )
                {
                    tmp0 = left;
                    tmp1 = right0;
                    cvSetSeqReaderPos( &tmp1, 1-m, 1 );
                    for( i = 0; i < m; i++ )
                    {
                        CV_SWAP_ELEMS( tmp0.ptr, tmp1.ptr, elem_size );
                        CV_NEXT_SEQ_ELEM( elem_size, tmp0 );
                        CV_NEXT_SEQ_ELEM( elem_size, tmp1 );
                    }
                }

                n = l - l1;
                m = r1 - r;
                if( n > 1 )
                {
                    if( m > 1 )
                    {
                        if( n > m )
                        {
                            sp++;
                            CV_SAVE_READER_POS( left0, stack[sp].lb );
                            cvSetSeqReaderPos( &left0, n - 1, 1 );
                            CV_SAVE_READER_POS( left0, stack[sp].ub );
                            left = right = right0;
                            cvSetSeqReaderPos( &left, 1 - m, 1 );
                        }
                        else
                        {
                            sp++;
                            CV_SAVE_READER_POS( right0, stack[sp].ub );
                            cvSetSeqReaderPos( &right0, 1 - m, 1 );
                            CV_SAVE_READER_POS( right0, stack[sp].lb );
                            left = right = left0;
                            cvSetSeqReaderPos( &right, n - 1, 1 );
                        }
                    }
                    else
                    {
                        left = right = left0;
                        cvSetSeqReaderPos( &right, n - 1, 1 );
                    }
                }
                else if( m > 1 )
                {
                    left = right = right0;
                    cvSetSeqReaderPos( &left, 1 - m, 1 );
                }
                else
                    break;
            }
        }
    }
}


CV_IMPL schar*
cvSeqSearch( CvSeq* seq, const void* _elem, CvCmpFunc cmp_func,
             int is_sorted, int* _idx, void* userdata )
{
    schar* result = 0;
    const schar* elem = (const schar*)_elem;
    int idx = -1;
    int i, j;

    if( _idx )
        *_idx = idx;

    if( !CV_IS_SEQ(seq) )
        CV_Error( !seq ? CV_StsNullPtr : CV_StsBadArg, "Bad input sequence" );

    if( !elem )
        CV_Error( CV_StsNullPtr, "Null element pointer" );

    int elem_size = seq->elem_size;
    int total = seq->total;

    if( total == 0 )
        return 0;

    if( !is_sorted )
    {
        CvSeqReader reader;
        cvStartReadSeq( seq, &reader, 0 );

        if( cmp_func )
        {
            for( i = 0; i < total; i++ )
            {
                if( cmp_func( elem, reader.ptr, userdata ) == 0 )
                    break;
                CV_NEXT_SEQ_ELEM( elem_size, reader );
            }
        }
        else if( (elem_size & (sizeof(int)-1)) == 0 )
        {
            for( i = 0; i < total; i++ )
            {
                for( j = 0; j < elem_size; j += sizeof(int) )
                {
                    if( *(const int*)(reader.ptr + j) != *(const int*)(elem + j) )
                        break;
                }
                if( j == elem_size )
                    break;
                CV_NEXT_SEQ_ELEM( elem_size, reader );
            }
        }
        else
        {
            for( i = 0; i < total; i++ )
            {
                for( j = 0; j < elem_size; j++ )
                {
                    if( reader.ptr[j] != elem[j] )
                        break;
                }
                if( j == elem_size )
                    break;
                CV_NEXT_SEQ_ELEM( elem_size, reader );
            }
        }

        idx = i;
        if( i < total )
            result = reader.ptr;
    }
    else
    {
        if( !cmp_func )
            CV_Error( CV_StsNullPtr, "Null compare function" );

        i = 0, j = total;

        while( j > i )
        {
            int k = (i+j)>>1, code;
            schar* ptr = cvGetSeqElem( seq, k );
            code = cmp_func( elem, ptr, userdata );
            if( !code )
            {
                result = ptr;
                idx = k;
                if( _idx )
                    *_idx = idx;
                return result;
            }
            if( code < 0 )
                j = k;
            else
                i = k+1;
        }
        idx = j;
    }

    if( _idx )
        *_idx = idx;

    return result;
}


CV_IMPL void
cvSeqInvert( CvSeq* seq )
{
    CvSeqReader left_reader, right_reader;
    int elem_size;
    int i, count;

    cvStartReadSeq( seq, &left_reader, 0 );
    cvStartReadSeq( seq, &right_reader, 1 );
    elem_size = seq->elem_size;
    count = seq->total >> 1;

    for( i = 0; i < count; i++ )
    {
        CV_SWAP_ELEMS( left_reader.ptr, right_reader.ptr, elem_size );
        CV_NEXT_SEQ_ELEM( elem_size, left_reader );
        CV_PREV_SEQ_ELEM( elem_size, right_reader );
    }
}


typedef struct CvPTreeNode
{
    struct CvPTreeNode* parent;
    schar* element;
    int rank;
}
CvPTreeNode;


// This function splits the input sequence or set into one or more equivalence classes.
// is_equal(a,b,...) returns non-zero if the two sequence elements
// belong to the same class.  The function returns sequence of integers -
// 0-based class indexes for each element.
//
// The algorithm is described in "Introduction to Algorithms"
// by Cormen, Leiserson and Rivest, chapter "Data structures for disjoint sets"
CV_IMPL  int
cvSeqPartition( const CvSeq* seq, CvMemStorage* storage, CvSeq** labels,
                CvCmpFunc is_equal, void* userdata )
{
    CvSeq* result = 0;
    CvMemStorage* temp_storage = 0;
    int class_idx = 0;

    CvSeqWriter writer;
    CvSeqReader reader, reader0;
    CvSeq* nodes;
    int i, j;
    int is_set;

    if( !labels )
        CV_Error( CV_StsNullPtr, "" );

    if( !seq || !is_equal )
        CV_Error( CV_StsNullPtr, "" );

    if( !storage )
        storage = seq->storage;

    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    is_set = CV_IS_SET(seq);

    temp_storage = cvCreateChildMemStorage( storage );

    nodes = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPTreeNode), temp_storage );

    cvStartReadSeq( seq, &reader );
    memset( &writer, 0, sizeof(writer));
    cvStartAppendToSeq( nodes, &writer );

    // Initial O(N) pass. Make a forest of single-vertex trees.
    for( i = 0; i < seq->total; i++ )
    {
        CvPTreeNode node = { 0, 0, 0 };
        if( !is_set || CV_IS_SET_ELEM( reader.ptr ))
            node.element = reader.ptr;
        CV_WRITE_SEQ_ELEM( node, writer );
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    cvEndWriteSeq( &writer );

    // Because in the next loop we will iterate
    // through all the sequence nodes each time,
    // we do not need to initialize reader every time:
    cvStartReadSeq( nodes, &reader );
    cvStartReadSeq( nodes, &reader0 );

    // The main O(N^2) pass. Merge connected components.
    for( i = 0; i < nodes->total; i++ )
    {
        CvPTreeNode* node = (CvPTreeNode*)(reader0.ptr);
        CvPTreeNode* root = node;
        CV_NEXT_SEQ_ELEM( nodes->elem_size, reader0 );

        if( !node->element )
            continue;

        // find root
        while( root->parent )
            root = root->parent;

        for( j = 0; j < nodes->total; j++ )
        {
            CvPTreeNode* node2 = (CvPTreeNode*)reader.ptr;

            if( node2->element && node2 != node &&
                is_equal( node->element, node2->element, userdata ))
            {
                CvPTreeNode* root2 = node2;

                // unite both trees
                while( root2->parent )
                    root2 = root2->parent;

                if( root2 != root )
                {
                    if( root->rank > root2->rank )
                        root2->parent = root;
                    else
                    {
                        root->parent = root2;
                        root2->rank += root->rank == root2->rank;
                        root = root2;
                    }
                    assert( root->parent == 0 );

                    // Compress path from node2 to the root:
                    while( node2->parent )
                    {
                        CvPTreeNode* temp = node2;
                        node2 = node2->parent;
                        temp->parent = root;
                    }

                    // Compress path from node to the root:
                    node2 = node;
                    while( node2->parent )
                    {
                        CvPTreeNode* temp = node2;
                        node2 = node2->parent;
                        temp->parent = root;
                    }
                }
            }

            CV_NEXT_SEQ_ELEM( sizeof(*node), reader );
        }
    }

    // Final O(N) pass (Enumerate classes)
    // Reuse reader one more time
    result = cvCreateSeq( 0, sizeof(CvSeq), sizeof(int), storage );
    cvStartAppendToSeq( result, &writer );

    for( i = 0; i < nodes->total; i++ )
    {
        CvPTreeNode* node = (CvPTreeNode*)reader.ptr;
        int idx = -1;

        if( node->element )
        {
            while( node->parent )
                node = node->parent;
            if( node->rank >= 0 )
                node->rank = ~class_idx++;
            idx = ~node->rank;
        }

        CV_NEXT_SEQ_ELEM( sizeof(*node), reader );
        CV_WRITE_SEQ_ELEM( idx, writer );
    }

    cvEndWriteSeq( &writer );

    if( labels )
        *labels = result;

    cvReleaseMemStorage( &temp_storage );
    return class_idx;
}


/****************************************************************************************\
*                                      Set implementation                                *
\****************************************************************************************/

/* Creates empty set: */
CV_IMPL CvSet*
cvCreateSet( int set_flags, int header_size, int elem_size, CvMemStorage * storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );
    if( header_size < (int)sizeof( CvSet ) ||
        elem_size < (int)sizeof(void*)*2 ||
        (elem_size & (sizeof(void*)-1)) != 0 )
        CV_Error( CV_StsBadSize, "" );

    CvSet* set = (CvSet*) cvCreateSeq( set_flags, header_size, elem_size, storage );
    set->flags = (set->flags & ~CV_MAGIC_MASK) | CV_SET_MAGIC_VAL;

    return set;
}


/* Add new element to the set: */
CV_IMPL int
cvSetAdd( CvSet* set, CvSetElem* element, CvSetElem** inserted_element )
{
    int id = -1;
    CvSetElem *free_elem;

    if( !set )
        CV_Error( CV_StsNullPtr, "" );

    if( !(set->free_elems) )
    {
        int count = set->total;
        int elem_size = set->elem_size;
        schar *ptr;
        icvGrowSeq( (CvSeq *) set, 0 );

        set->free_elems = (CvSetElem*) (ptr = set->ptr);
        for( ; ptr + elem_size <= set->block_max; ptr += elem_size, count++ )
        {
            ((CvSetElem*)ptr)->flags = count | CV_SET_ELEM_FREE_FLAG;
            ((CvSetElem*)ptr)->next_free = (CvSetElem*)(ptr + elem_size);
        }
        assert( count <= CV_SET_ELEM_IDX_MASK+1 );
        ((CvSetElem*)(ptr - elem_size))->next_free = 0;
        set->first->prev->count += count - set->total;
        set->total = count;
        set->ptr = set->block_max;
    }

    free_elem = set->free_elems;
    set->free_elems = free_elem->next_free;

    id = free_elem->flags & CV_SET_ELEM_IDX_MASK;
    if( element )
        memcpy( free_elem, element, set->elem_size );

    free_elem->flags = id;
    set->active_count++;

    if( inserted_element )
        *inserted_element = free_elem;

    return id;
}


/* Remove element from a set given element index: */
CV_IMPL void
cvSetRemove( CvSet* set, int index )
{
    CvSetElem* elem = cvGetSetElem( set, index );
    if( elem )
        cvSetRemoveByPtr( set, elem );
    else if( !set )
        CV_Error( CV_StsNullPtr, "" );
}


/* Remove all elements from a set: */
CV_IMPL void
cvClearSet( CvSet* set )
{
    cvClearSeq( (CvSeq*)set );
    set->free_elems = 0;
    set->active_count = 0;
}


/****************************************************************************************\
*                                 Graph  implementation                                  *
\****************************************************************************************/

/* Create a new graph: */
CV_IMPL CvGraph *
cvCreateGraph( int graph_type, int header_size,
               int vtx_size, int edge_size, CvMemStorage * storage )
{
    CvGraph *graph = 0;
    CvSet *edges = 0;
    CvSet *vertices = 0;

    if( header_size < (int) sizeof( CvGraph     )
    ||  edge_size   < (int) sizeof( CvGraphEdge )
    ||  vtx_size    < (int) sizeof( CvGraphVtx  )
    ){
        CV_Error( CV_StsBadSize, "" );
    }

    vertices = cvCreateSet( graph_type, header_size, vtx_size, storage );
    edges = cvCreateSet( CV_SEQ_KIND_GENERIC | CV_SEQ_ELTYPE_GRAPH_EDGE,
                                  sizeof( CvSet ), edge_size, storage );

    graph = (CvGraph*)vertices;
    graph->edges = edges;

    return graph;
}


/* Remove all vertices and edges from a graph: */
CV_IMPL void
cvClearGraph( CvGraph * graph )
{
    if( !graph )
        CV_Error( CV_StsNullPtr, "" );

    cvClearSet( graph->edges );
    cvClearSet( (CvSet*)graph );
}


/* Add a vertex to a graph: */
CV_IMPL int
cvGraphAddVtx( CvGraph* graph, const CvGraphVtx* _vertex, CvGraphVtx** _inserted_vertex )
{
    CvGraphVtx *vertex = 0;
    int index = -1;

    if( !graph )
        CV_Error( CV_StsNullPtr, "" );

    vertex = (CvGraphVtx*)cvSetNew((CvSet*)graph);
    if( vertex )
    {
        if( _vertex )
            memcpy( vertex + 1, _vertex + 1, graph->elem_size - sizeof(CvGraphVtx) );
        vertex->first = 0;
        index = vertex->flags;
    }

    if( _inserted_vertex )
        *_inserted_vertex = vertex;

    return index;
}


/* Remove a vertex from the graph together with its incident edges: */
CV_IMPL int
cvGraphRemoveVtxByPtr( CvGraph* graph, CvGraphVtx* vtx )
{
    int count = -1;

    if( !graph || !vtx )
        CV_Error( CV_StsNullPtr, "" );

    if( !CV_IS_SET_ELEM(vtx))
        CV_Error( CV_StsBadArg, "The vertex does not belong to the graph" );

    count = graph->edges->active_count;
    for( ;; )
    {
        CvGraphEdge *edge = vtx->first;
        if( !edge )
            break;
        cvGraphRemoveEdgeByPtr( graph, edge->vtx[0], edge->vtx[1] );
    }
    count -= graph->edges->active_count;
    cvSetRemoveByPtr( (CvSet*)graph, vtx );

    return count;
}


/* Remove a vertex from the graph together with its incident edges: */
CV_IMPL int
cvGraphRemoveVtx( CvGraph* graph, int index )
{
    int count = -1;
    CvGraphVtx *vtx = 0;

    if( !graph )
        CV_Error( CV_StsNullPtr, "" );

    vtx = cvGetGraphVtx( graph, index );
    if( !vtx )
        CV_Error( CV_StsBadArg, "The vertex is not found" );

    count = graph->edges->active_count;
    for( ;; )
    {
        CvGraphEdge *edge = vtx->first;
        count++;

        if( !edge )
            break;
        cvGraphRemoveEdgeByPtr( graph, edge->vtx[0], edge->vtx[1] );
    }
    count -= graph->edges->active_count;
    cvSetRemoveByPtr( (CvSet*)graph, vtx );

    return count;
}


/* Find a graph edge given pointers to the ending vertices: */
CV_IMPL CvGraphEdge*
cvFindGraphEdgeByPtr( const CvGraph* graph,
                      const CvGraphVtx* start_vtx,
                      const CvGraphVtx* end_vtx )
{
    int ofs = 0;

    if( !graph || !start_vtx || !end_vtx )
        CV_Error( CV_StsNullPtr, "" );

    if( start_vtx == end_vtx )
        return 0;

    if( !CV_IS_GRAPH_ORIENTED( graph ) &&
        (start_vtx->flags & CV_SET_ELEM_IDX_MASK) > (end_vtx->flags & CV_SET_ELEM_IDX_MASK) )
    {
        const CvGraphVtx* t;
        CV_SWAP( start_vtx, end_vtx, t );
    }

    CvGraphEdge* edge = start_vtx->first;
    for( ; edge; edge = edge->next[ofs] )
    {
        ofs = start_vtx == edge->vtx[1];
        assert( ofs == 1 || start_vtx == edge->vtx[0] );
        if( edge->vtx[1] == end_vtx )
            break;
    }

    return edge;
}


/* Find an edge in the graph given indices of the ending vertices: */
CV_IMPL CvGraphEdge *
cvFindGraphEdge( const CvGraph* graph, int start_idx, int end_idx )
{
    CvGraphVtx *start_vtx;
    CvGraphVtx *end_vtx;

    if( !graph )
        CV_Error( CV_StsNullPtr, "graph pointer is NULL" );

    start_vtx = cvGetGraphVtx( graph, start_idx );
    end_vtx = cvGetGraphVtx( graph, end_idx );

    return cvFindGraphEdgeByPtr( graph, start_vtx, end_vtx );
}


/* Given two vertices, return the edge
 * connecting them, creating it if it
 * did not already exist:
 */
CV_IMPL int
cvGraphAddEdgeByPtr( CvGraph* graph,
                     CvGraphVtx* start_vtx, CvGraphVtx* end_vtx,
                     const CvGraphEdge* _edge,
                     CvGraphEdge ** _inserted_edge )
{
    CvGraphEdge *edge = 0;
    int result = -1;
    int delta;

    if( !graph )
        CV_Error( CV_StsNullPtr, "graph pointer is NULL" );

    if( !CV_IS_GRAPH_ORIENTED( graph ) &&
        (start_vtx->flags & CV_SET_ELEM_IDX_MASK) > (end_vtx->flags & CV_SET_ELEM_IDX_MASK) )
    {
        CvGraphVtx* t;
        CV_SWAP( start_vtx, end_vtx, t );
    }

    edge = cvFindGraphEdgeByPtr( graph, start_vtx, end_vtx );
    if( edge )
    {
        result = 0;
        if( _inserted_edge )
            *_inserted_edge = edge;
        return result;
    }

    if( start_vtx == end_vtx )
        CV_Error( start_vtx ? CV_StsBadArg : CV_StsNullPtr,
        "vertex pointers coinside (or set to NULL)" );

    edge = (CvGraphEdge*)cvSetNew( (CvSet*)(graph->edges) );
    assert( edge->flags >= 0 );

    edge->vtx[0] = start_vtx;
    edge->vtx[1] = end_vtx;
    edge->next[0] = start_vtx->first;
    edge->next[1] = end_vtx->first;
    start_vtx->first = end_vtx->first = edge;

    delta = graph->edges->elem_size - sizeof(*edge);
    if( _edge )
    {
        if( delta > 0 )
            memcpy( edge + 1, _edge + 1, delta );
        edge->weight = _edge->weight;
    }
    else
    {
        if( delta > 0 )
            memset( edge + 1, 0, delta );
        edge->weight = 1.f;
    }

    result = 1;

    if( _inserted_edge )
        *_inserted_edge = edge;

    return result;
}

/* Given two vertices, return the edge
 * connecting them, creating it if it
 * did not already exist:
 */
CV_IMPL int
cvGraphAddEdge( CvGraph* graph,
                int start_idx, int end_idx,
                const CvGraphEdge* _edge,
                CvGraphEdge ** _inserted_edge )
{
    CvGraphVtx *start_vtx;
    CvGraphVtx *end_vtx;

    if( !graph )
        CV_Error( CV_StsNullPtr, "" );

    start_vtx = cvGetGraphVtx( graph, start_idx );
    end_vtx = cvGetGraphVtx( graph, end_idx );

    return cvGraphAddEdgeByPtr( graph, start_vtx, end_vtx, _edge, _inserted_edge );
}


/* Remove the graph edge connecting two given vertices: */
CV_IMPL void
cvGraphRemoveEdgeByPtr( CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx )
{
    int ofs, prev_ofs;
    CvGraphEdge *edge, *next_edge, *prev_edge;

    if( !graph || !start_vtx || !end_vtx )
        CV_Error( CV_StsNullPtr, "" );

    if( start_vtx == end_vtx )
        return;

    if( !CV_IS_GRAPH_ORIENTED( graph ) &&
        (start_vtx->flags & CV_SET_ELEM_IDX_MASK) > (end_vtx->flags & CV_SET_ELEM_IDX_MASK) )
    {
        CvGraphVtx* t;
        CV_SWAP( start_vtx, end_vtx, t );
    }

    for( ofs = prev_ofs = 0, prev_edge = 0, edge = start_vtx->first; edge != 0;
         prev_ofs = ofs, prev_edge = edge, edge = edge->next[ofs] )
    {
        ofs = start_vtx == edge->vtx[1];
        assert( ofs == 1 || start_vtx == edge->vtx[0] );
        if( edge->vtx[1] == end_vtx )
            break;
    }

    if( !edge )
        return;

    next_edge = edge->next[ofs];
    if( prev_edge )
        prev_edge->next[prev_ofs] = next_edge;
    else
        start_vtx->first = next_edge;

    for( ofs = prev_ofs = 0, prev_edge = 0, edge = end_vtx->first; edge != 0;
         prev_ofs = ofs, prev_edge = edge, edge = edge->next[ofs] )
    {
        ofs = end_vtx == edge->vtx[1];
        assert( ofs == 1 || end_vtx == edge->vtx[0] );
        if( edge->vtx[0] == start_vtx )
            break;
    }

    assert( edge != 0 );

    next_edge = edge->next[ofs];
    if( prev_edge )
        prev_edge->next[prev_ofs] = next_edge;
    else
        end_vtx->first = next_edge;

    cvSetRemoveByPtr( graph->edges, edge );
}


/* Remove the graph edge connecting two given vertices: */
CV_IMPL void
cvGraphRemoveEdge( CvGraph* graph, int start_idx, int end_idx )
{
    CvGraphVtx *start_vtx;
    CvGraphVtx *end_vtx;

    if( !graph )
        CV_Error( CV_StsNullPtr, "" );

    start_vtx = cvGetGraphVtx( graph, start_idx );
    end_vtx = cvGetGraphVtx( graph, end_idx );

    cvGraphRemoveEdgeByPtr( graph, start_vtx, end_vtx );
}


/* Count number of edges incident to a given vertex: */
CV_IMPL int
cvGraphVtxDegreeByPtr( const CvGraph* graph, const CvGraphVtx* vertex )
{
    CvGraphEdge *edge;
    int count;

    if( !graph || !vertex )
        CV_Error( CV_StsNullPtr, "" );

    for( edge = vertex->first, count = 0; edge; )
    {
        count++;
        edge = CV_NEXT_GRAPH_EDGE( edge, vertex );
    }

    return count;
}


/* Count number of edges incident to a given vertex: */
CV_IMPL int
cvGraphVtxDegree( const CvGraph* graph, int vtx_idx )
{
    CvGraphVtx *vertex;
    CvGraphEdge *edge;
    int count;

    if( !graph )
        CV_Error( CV_StsNullPtr, "" );

    vertex = cvGetGraphVtx( graph, vtx_idx );
    if( !vertex )
        CV_Error( CV_StsObjectNotFound, "" );

    for( edge = vertex->first, count = 0; edge; )
    {
        count++;
        edge = CV_NEXT_GRAPH_EDGE( edge, vertex );
    }

    return count;
}


typedef struct CvGraphItem
{
    CvGraphVtx* vtx;
    CvGraphEdge* edge;
}
CvGraphItem;


static  void
icvSeqElemsClearFlags( CvSeq* seq, int offset, int clear_mask )
{
    CvSeqReader reader;
    int i, total, elem_size;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = seq->elem_size;
    total = seq->total;

    if( (unsigned)offset > (unsigned)elem_size )
        CV_Error( CV_StsBadArg, "" );

    cvStartReadSeq( seq, &reader );

    for( i = 0; i < total; i++ )
    {
        int* flag_ptr = (int*)(reader.ptr + offset);
        *flag_ptr &= ~clear_mask;

        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }
}


static  schar*
icvSeqFindNextElem( CvSeq* seq, int offset, int mask,
                    int value, int* start_index )
{
    schar* elem_ptr = 0;

    CvSeqReader reader;
    int total, elem_size, index;

    if( !seq || !start_index )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = seq->elem_size;
    total = seq->total;
    index = *start_index;

    if( (unsigned)offset > (unsigned)elem_size )
        CV_Error( CV_StsBadArg, "" );

    if( total == 0 )
        return 0;

    if( (unsigned)index >= (unsigned)total )
    {
        index %= total;
        index += index < 0 ? total : 0;
    }

    cvStartReadSeq( seq, &reader );

    if( index != 0 )
        cvSetSeqReaderPos( &reader, index );

    for( index = 0; index < total; index++ )
    {
        int* flag_ptr = (int*)(reader.ptr + offset);
        if( (*flag_ptr & mask) == value )
            break;

        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }

    if( index < total )
    {
        elem_ptr = reader.ptr;
        *start_index = index;
    }

    return  elem_ptr;
}

#define CV_FIELD_OFFSET( field, structtype ) ((int)(size_t)&((structtype*)0)->field)

CV_IMPL CvGraphScanner*
cvCreateGraphScanner( CvGraph* graph, CvGraphVtx* vtx, int mask )
{
    if( !graph )
        CV_Error( CV_StsNullPtr, "Null graph pointer" );

    CV_Assert( graph->storage != 0 );

    CvGraphScanner* scanner = (CvGraphScanner*)cvAlloc( sizeof(*scanner) );
    memset( scanner, 0, sizeof(*scanner));

    scanner->graph = graph;
    scanner->mask = mask;
    scanner->vtx = vtx;
    scanner->index = vtx == 0 ? 0 : -1;

    CvMemStorage* child_storage = cvCreateChildMemStorage( graph->storage );

    scanner->stack = cvCreateSeq( 0, sizeof(CvSet),
                       sizeof(CvGraphItem), child_storage );

    icvSeqElemsClearFlags( (CvSeq*)graph,
                                    CV_FIELD_OFFSET( flags, CvGraphVtx),
                                    CV_GRAPH_ITEM_VISITED_FLAG|
                                    CV_GRAPH_SEARCH_TREE_NODE_FLAG );

    icvSeqElemsClearFlags( (CvSeq*)(graph->edges),
                                    CV_FIELD_OFFSET( flags, CvGraphEdge),
                                    CV_GRAPH_ITEM_VISITED_FLAG );

    return scanner;
}


CV_IMPL void
cvReleaseGraphScanner( CvGraphScanner** scanner )
{
    if( !scanner )
        CV_Error( CV_StsNullPtr, "Null double pointer to graph scanner" );

    if( *scanner )
    {
        if( (*scanner)->stack )
            cvReleaseMemStorage( &((*scanner)->stack->storage));
        cvFree( scanner );
    }
}


CV_IMPL int
cvNextGraphItem( CvGraphScanner* scanner )
{
    int code = -1;
    CvGraphVtx* vtx;
    CvGraphVtx* dst;
    CvGraphEdge* edge;
    CvGraphItem item;

    if( !scanner || !(scanner->stack))
        CV_Error( CV_StsNullPtr, "Null graph scanner" );

    dst = scanner->dst;
    vtx = scanner->vtx;
    edge = scanner->edge;

    for(;;)
    {
        for(;;)
        {
            if( dst && !CV_IS_GRAPH_VERTEX_VISITED(dst) )
            {
                scanner->vtx = vtx = dst;
                edge = vtx->first;
                dst->flags |= CV_GRAPH_ITEM_VISITED_FLAG;

                if((scanner->mask & CV_GRAPH_VERTEX))
                {
                    scanner->vtx = vtx;
                    scanner->edge = vtx->first;
                    scanner->dst = 0;
                    code = CV_GRAPH_VERTEX;
                    return code;
                }
            }

            while( edge )
            {
                dst = edge->vtx[vtx == edge->vtx[0]];

                if( !CV_IS_GRAPH_EDGE_VISITED(edge) )
                {
                    // Check that the edge is outgoing:
                    if( !CV_IS_GRAPH_ORIENTED( scanner->graph ) || dst != edge->vtx[0] )
                    {
                        edge->flags |= CV_GRAPH_ITEM_VISITED_FLAG;

                        if( !CV_IS_GRAPH_VERTEX_VISITED(dst) )
                        {
                            item.vtx = vtx;
                            item.edge = edge;

                            vtx->flags |= CV_GRAPH_SEARCH_TREE_NODE_FLAG;

                            cvSeqPush( scanner->stack, &item );

                            if( scanner->mask & CV_GRAPH_TREE_EDGE )
                            {
                                code = CV_GRAPH_TREE_EDGE;
                                scanner->vtx = vtx;
                                scanner->dst = dst;
                                scanner->edge = edge;
                                return code;
                            }
                            break;
                        }
                        else
                        {
                            if( scanner->mask & (CV_GRAPH_BACK_EDGE|
                                                 CV_GRAPH_CROSS_EDGE|
                                                 CV_GRAPH_FORWARD_EDGE) )
                            {
                                code = (dst->flags & CV_GRAPH_SEARCH_TREE_NODE_FLAG) ?
                                       CV_GRAPH_BACK_EDGE :
                                       (edge->flags & CV_GRAPH_FORWARD_EDGE_FLAG) ?
                                       CV_GRAPH_FORWARD_EDGE : CV_GRAPH_CROSS_EDGE;
                                edge->flags &= ~CV_GRAPH_FORWARD_EDGE_FLAG;
                                if( scanner->mask & code )
                                {
                                    scanner->vtx = vtx;
                                    scanner->dst = dst;
                                    scanner->edge = edge;
                                    return code;
                                }
                            }
                        }
                    }
                    else if( (dst->flags & (CV_GRAPH_ITEM_VISITED_FLAG|
                             CV_GRAPH_SEARCH_TREE_NODE_FLAG)) ==
                             (CV_GRAPH_ITEM_VISITED_FLAG|
                             CV_GRAPH_SEARCH_TREE_NODE_FLAG))
                    {
                        edge->flags |= CV_GRAPH_FORWARD_EDGE_FLAG;
                    }
                }

                edge = CV_NEXT_GRAPH_EDGE( edge, vtx );
            }

            if( !edge ) /* need to backtrack */
            {
                if( scanner->stack->total == 0 )
                {
                    if( scanner->index >= 0 )
                        vtx = 0;
                    else
                        scanner->index = 0;
                    break;
                }
                cvSeqPop( scanner->stack, &item );
                vtx = item.vtx;
                vtx->flags &= ~CV_GRAPH_SEARCH_TREE_NODE_FLAG;
                edge = item.edge;
                dst = 0;

                if( scanner->mask & CV_GRAPH_BACKTRACKING )
                {
                    scanner->vtx = vtx;
                    scanner->edge = edge;
                    scanner->dst = edge->vtx[vtx == edge->vtx[0]];
                    code = CV_GRAPH_BACKTRACKING;
                    return code;
                }
            }
        }

        if( !vtx )
        {
            vtx = (CvGraphVtx*)icvSeqFindNextElem( (CvSeq*)(scanner->graph),
                  CV_FIELD_OFFSET( flags, CvGraphVtx ), CV_GRAPH_ITEM_VISITED_FLAG|INT_MIN,
                  0, &(scanner->index) );

            if( !vtx )
            {
                code = CV_GRAPH_OVER;
                break;
            }
        }

        dst = vtx;
        if( scanner->mask & CV_GRAPH_NEW_TREE )
        {
            scanner->dst = dst;
            scanner->edge = 0;
            scanner->vtx = 0;
            code = CV_GRAPH_NEW_TREE;
            break;
        }
    }

    return code;
}


CV_IMPL CvGraph*
cvCloneGraph( const CvGraph* graph, CvMemStorage* storage )
{
    int* flag_buffer = 0;
    CvGraphVtx** ptr_buffer = 0;
    CvGraph* result = 0;

    int i, k;
    int vtx_size, edge_size;
    CvSeqReader reader;

    if( !CV_IS_GRAPH(graph))
        CV_Error( CV_StsBadArg, "Invalid graph pointer" );

    if( !storage )
        storage = graph->storage;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    vtx_size = graph->elem_size;
    edge_size = graph->edges->elem_size;

    flag_buffer = (int*)cvAlloc( graph->total*sizeof(flag_buffer[0]));
    ptr_buffer = (CvGraphVtx**)cvAlloc( graph->total*sizeof(ptr_buffer[0]));
    result = cvCreateGraph( graph->flags, graph->header_size,
                                     vtx_size, edge_size, storage );
    memcpy( result + sizeof(CvGraph), graph + sizeof(CvGraph),
            graph->header_size - sizeof(CvGraph));

    // Pass 1.  Save flags, copy vertices:
    cvStartReadSeq( (CvSeq*)graph, &reader );
    for( i = 0, k = 0; i < graph->total; i++ )
    {
        if( CV_IS_SET_ELEM( reader.ptr ))
        {
            CvGraphVtx* vtx = (CvGraphVtx*)reader.ptr;
            CvGraphVtx* dstvtx = 0;
            cvGraphAddVtx( result, vtx, &dstvtx );
            flag_buffer[k] = dstvtx->flags = vtx->flags;
            vtx->flags = k;
            ptr_buffer[k++] = dstvtx;
        }
        CV_NEXT_SEQ_ELEM( vtx_size, reader );
    }

    // Pass 2.  Copy edges:
    cvStartReadSeq( (CvSeq*)graph->edges, &reader );
    for( i = 0; i < graph->edges->total; i++ )
    {
        if( CV_IS_SET_ELEM( reader.ptr ))
        {
            CvGraphEdge* edge = (CvGraphEdge*)reader.ptr;
            CvGraphEdge* dstedge = 0;
            CvGraphVtx* new_org = ptr_buffer[edge->vtx[0]->flags];
            CvGraphVtx* new_dst = ptr_buffer[edge->vtx[1]->flags];
            cvGraphAddEdgeByPtr( result, new_org, new_dst, edge, &dstedge );
            dstedge->flags = edge->flags;
        }
        CV_NEXT_SEQ_ELEM( edge_size, reader );
    }

    // Pass 3.  Restore flags:
    cvStartReadSeq( (CvSeq*)graph, &reader );
    for( i = 0, k = 0; i < graph->edges->total; i++ )
    {
        if( CV_IS_SET_ELEM( reader.ptr ))
        {
            CvGraphVtx* vtx = (CvGraphVtx*)reader.ptr;
            vtx->flags = flag_buffer[k++];
        }
        CV_NEXT_SEQ_ELEM( vtx_size, reader );
    }

    cvFree( &flag_buffer );
    cvFree( &ptr_buffer );

    if( cvGetErrStatus() < 0 )
        result = 0;

    return result;
}


/****************************************************************************************\
*                                 Working with sequence tree                             *
\****************************************************************************************/

// Gather pointers to all the sequences, accessible from the <first>, to the single sequence.
CV_IMPL CvSeq*
cvTreeToNodeSeq( const void* first, int header_size, CvMemStorage* storage )
{
    CvSeq* allseq = 0;
    CvTreeNodeIterator iterator;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    allseq = cvCreateSeq( 0, header_size, sizeof(first), storage );

    if( first )
    {
        cvInitTreeNodeIterator( &iterator, first, INT_MAX );

        for(;;)
        {
            void* node = cvNextTreeNode( &iterator );
            if( !node )
                break;
            cvSeqPush( allseq, &node );
        }
    }



    return allseq;
}


typedef struct CvTreeNode
{
    int       flags;         /* micsellaneous flags */
    int       header_size;   /* size of sequence header */
    struct    CvTreeNode* h_prev; /* previous sequence */
    struct    CvTreeNode* h_next; /* next sequence */
    struct    CvTreeNode* v_prev; /* 2nd previous sequence */
    struct    CvTreeNode* v_next; /* 2nd next sequence */
}
CvTreeNode;



// Insert contour into tree given certain parent sequence.
// If parent is equal to frame (the most external contour),
// then added contour will have null pointer to parent:
CV_IMPL void
cvInsertNodeIntoTree( void* _node, void* _parent, void* _frame )
{
    CvTreeNode* node = (CvTreeNode*)_node;
    CvTreeNode* parent = (CvTreeNode*)_parent;

    if( !node || !parent )
        CV_Error( CV_StsNullPtr, "" );

    node->v_prev = _parent != _frame ? parent : 0;
    node->h_next = parent->v_next;

    assert( parent->v_next != node );

    if( parent->v_next )
        parent->v_next->h_prev = node;
    parent->v_next = node;
}


// Remove contour from tree, together with the contour's children:
CV_IMPL void
cvRemoveNodeFromTree( void* _node, void* _frame )
{
    CvTreeNode* node = (CvTreeNode*)_node;
    CvTreeNode* frame = (CvTreeNode*)_frame;

    if( !node )
        CV_Error( CV_StsNullPtr, "" );

    if( node == frame )
        CV_Error( CV_StsBadArg, "frame node could not be deleted" );

    if( node->h_next )
        node->h_next->h_prev = node->h_prev;

    if( node->h_prev )
        node->h_prev->h_next = node->h_next;
    else
    {
        CvTreeNode* parent = node->v_prev;
        if( !parent )
            parent = frame;

        if( parent )
        {
            assert( parent->v_next == node );
            parent->v_next = node->h_next;
        }
    }
}


CV_IMPL void
cvInitTreeNodeIterator( CvTreeNodeIterator* treeIterator,
                        const void* first, int max_level )
{
    if( !treeIterator || !first )
        CV_Error( CV_StsNullPtr, "" );

    if( max_level < 0 )
        CV_Error( CV_StsOutOfRange, "" );

    treeIterator->node = (void*)first;
    treeIterator->level = 0;
    treeIterator->max_level = max_level;
}


CV_IMPL void*
cvNextTreeNode( CvTreeNodeIterator* treeIterator )
{
    CvTreeNode* prevNode = 0;
    CvTreeNode* node;
    int level;

    if( !treeIterator )
        CV_Error( CV_StsNullPtr, "NULL iterator pointer" );

    prevNode = node = (CvTreeNode*)treeIterator->node;
    level = treeIterator->level;

    if( node )
    {
        if( node->v_next && level+1 < treeIterator->max_level )
        {
            node = node->v_next;
            level++;
        }
        else
        {
            while( node->h_next == 0 )
            {
                node = node->v_prev;
                if( --level < 0 )
                {
                    node = 0;
                    break;
                }
            }
            node = node && treeIterator->max_level != 0 ? node->h_next : 0;
        }
    }

    treeIterator->node = node;
    treeIterator->level = level;
    return prevNode;
}


CV_IMPL void*
cvPrevTreeNode( CvTreeNodeIterator* treeIterator )
{
    CvTreeNode* prevNode = 0;
    CvTreeNode* node;
    int level;

    if( !treeIterator )
        CV_Error( CV_StsNullPtr, "" );

    prevNode = node = (CvTreeNode*)treeIterator->node;
    level = treeIterator->level;

    if( node )
    {
        if( !node->h_prev )
        {
            node = node->v_prev;
            if( --level < 0 )
                node = 0;
        }
        else
        {
            node = node->h_prev;

            while( node->v_next && level < treeIterator->max_level )
            {
                node = node->v_next;
                level++;

                while( node->h_next )
                    node = node->h_next;
            }
        }
    }

    treeIterator->node = node;
    treeIterator->level = level;
    return prevNode;
}

namespace cv
{

////////////////////////////////////////////////////////////////////////////////

schar*  seqPush( CvSeq* seq, const void* element )
{
    return cvSeqPush(seq, element);
}

schar*  seqPushFront( CvSeq* seq, const void* element )
{
    return cvSeqPushFront(seq, element);
}

void  seqPop( CvSeq* seq, void* element )
{
    cvSeqPop(seq, element);
}

void  seqPopFront( CvSeq* seq, void* element )
{
    cvSeqPopFront(seq, element);
}

void  seqRemove( CvSeq* seq, int index )
{
    cvSeqRemove(seq, index);
}

void  clearSeq( CvSeq* seq )
{
    cvClearSeq(seq);
}

schar*  getSeqElem( const CvSeq* seq, int index )
{
    return cvGetSeqElem(seq, index);
}

void  seqRemoveSlice( CvSeq* seq, CvSlice slice )
{
    return cvSeqRemoveSlice(seq, slice);
}

void  seqInsertSlice( CvSeq* seq, int before_index, const CvArr* from_arr )
{
    cvSeqInsertSlice(seq, before_index, from_arr);
}

}

/* End of file. */
