/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#define CV_USE_SYSTEM_MALLOC 1

namespace cv
{

static void* OutOfMemoryError(size_t size)
{
    CV_Error_(CV_StsNoMem, ("Failed to allocate %lu bytes", (unsigned long)size));
    return 0;
}

#if CV_USE_SYSTEM_MALLOC

#if defined WIN32 || defined _WIN32
void deleteThreadAllocData() {}
#endif

void* fastMalloc( size_t size )
{
    uchar* udata = (uchar*)malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
    if(!udata)
        return OutOfMemoryError(size);
    uchar** adata = alignPtr((uchar**)udata + 1, CV_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void fastFree(void* ptr)
{
    if(ptr)
    {
        uchar* udata = ((uchar**)ptr)[-1];
        CV_DbgAssert(udata < (uchar*)ptr &&
               ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*)+CV_MALLOC_ALIGN));
        free(udata);
    }
}

#else //CV_USE_SYSTEM_MALLOC

#if 0
#define SANITY_CHECK(block) \
    CV_Assert(((size_t)(block) & (MEM_BLOCK_SIZE-1)) == 0 && \
        (unsigned)(block)->binIdx <= (unsigned)MAX_BIN && \
        (block)->signature == MEM_BLOCK_SIGNATURE)
#else
#define SANITY_CHECK(block)
#endif

#define STAT(stmt)

#ifdef WIN32
#if (_WIN32_WINNT >= 0x0602)
#include <synchapi.h>
#endif

struct CriticalSection
{
    CriticalSection()
    {
#if (_WIN32_WINNT >= 0x0600)
        InitializeCriticalSectionEx(&cs, 1000, 0);
#else
        InitializeCriticalSection(&cs);
#endif
    }
    ~CriticalSection() { DeleteCriticalSection(&cs); }
    void lock() { EnterCriticalSection(&cs); }
    void unlock() { LeaveCriticalSection(&cs); }
    bool trylock() { return TryEnterCriticalSection(&cs) != 0; }

    CRITICAL_SECTION cs;
};

void* SystemAlloc(size_t size)
{
    void* ptr = malloc(size);
    return ptr ? ptr : OutOfMemoryError(size);
}

void SystemFree(void* ptr, size_t)
{
    free(ptr);
}
#else //WIN32

#include <sys/mman.h>

struct CriticalSection
{
    CriticalSection() { pthread_mutex_init(&mutex, 0); }
    ~CriticalSection() { pthread_mutex_destroy(&mutex); }
    void lock() { pthread_mutex_lock(&mutex); }
    void unlock() { pthread_mutex_unlock(&mutex); }
    bool trylock() { return pthread_mutex_trylock(&mutex) == 0; }

    pthread_mutex_t mutex;
};

void* SystemAlloc(size_t size)
{
    #ifndef MAP_ANONYMOUS
    #define MAP_ANONYMOUS MAP_ANON
    #endif
    void* ptr = 0;
    ptr = mmap(ptr, size, (PROT_READ | PROT_WRITE), MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    return ptr != MAP_FAILED ? ptr : OutOfMemoryError(size);
}

void SystemFree(void* ptr, size_t size)
{
    munmap(ptr, size);
}
#endif //WIN32

struct AutoLock
{
    AutoLock(CriticalSection& _cs) : cs(&_cs) { cs->lock(); }
    ~AutoLock() { cs->unlock(); }
    CriticalSection* cs;
};

const size_t MEM_BLOCK_SIGNATURE = 0x01234567;
const int MEM_BLOCK_SHIFT = 14;
const size_t MEM_BLOCK_SIZE = 1 << MEM_BLOCK_SHIFT;
const size_t HDR_SIZE = 128;
const size_t MAX_BLOCK_SIZE = MEM_BLOCK_SIZE - HDR_SIZE;
const int MAX_BIN = 28;

static const int binSizeTab[MAX_BIN+1] =
{ 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256, 320, 384, 480, 544, 672, 768,
896, 1056, 1328, 1600, 2688, 4048, 5408, 8128, 16256 };

struct MallocTables
{
    void initBinTab()
    {
        int i, j = 0, n;
        for( i = 0; i <= MAX_BIN; i++ )
        {
            n = binSizeTab[i]>>3;
            for( ; j <= n; j++ )
                binIdx[j] = (uchar)i;
        }
    }
    int bin(size_t size)
    {
        assert( size <= MAX_BLOCK_SIZE );
        return binIdx[(size + 7)>>3];
    }

    MallocTables()
    {
        initBinTab();
    }

    uchar binIdx[MAX_BLOCK_SIZE/8+1];
};

MallocTables mallocTables;

struct Node
{
    Node* next;
};

struct ThreadData;

struct Block
{
    Block(Block* _next)
    {
        signature = MEM_BLOCK_SIGNATURE;
        prev = 0;
        next = _next;
        privateFreeList = publicFreeList = 0;
        bumpPtr = endPtr = 0;
        objSize = 0;
        threadData = 0;
        data = (uchar*)this + HDR_SIZE;
    }

    ~Block() {}

    void init(Block* _prev, Block* _next, int _objSize, ThreadData* _threadData)
    {
        prev = _prev;
        if(prev)
            prev->next = this;
        next = _next;
        if(next)
            next->prev = this;
        objSize = _objSize;
        binIdx = mallocTables.bin(objSize);
        threadData = _threadData;
        privateFreeList = publicFreeList = 0;
        bumpPtr = data;
        int nobjects = MAX_BLOCK_SIZE/objSize;
        endPtr = bumpPtr + nobjects*objSize;
        almostEmptyThreshold = (nobjects + 1)/2;
        allocated = 0;
    }

    bool isFilled() const { return allocated > almostEmptyThreshold; }

    size_t signature;
    Block* prev;
    Block* next;
    Node* privateFreeList;
    Node* publicFreeList;
    uchar* bumpPtr;
    uchar* endPtr;
    uchar* data;
    ThreadData* threadData;
    int objSize;
    int binIdx;
    int allocated;
    int almostEmptyThreshold;
    CriticalSection cs;
};

struct BigBlock
{
    BigBlock(int bigBlockSize, BigBlock* _next)
    {
        first = alignPtr((Block*)(this+1), MEM_BLOCK_SIZE);
        next = _next;
        nblocks = (int)(((char*)this + bigBlockSize - (char*)first)/MEM_BLOCK_SIZE);
        Block* p = 0;
        for( int i = nblocks-1; i >= 0; i-- )
            p = ::new((uchar*)first + i*MEM_BLOCK_SIZE) Block(p);
    }

    ~BigBlock()
    {
        for( int i = nblocks-1; i >= 0; i-- )
            ((Block*)((uchar*)first+i*MEM_BLOCK_SIZE))->~Block();
    }

    BigBlock* next;
    Block* first;
    int nblocks;
};

struct BlockPool
{
    BlockPool(int _bigBlockSize=1<<20) : pool(0), bigBlockSize(_bigBlockSize)
    {
    }

    ~BlockPool()
    {
        AutoLock lock(cs);
        while( pool )
        {
            BigBlock* nextBlock = pool->next;
            pool->~BigBlock();
            SystemFree(pool, bigBlockSize);
            pool = nextBlock;
        }
    }

    Block* alloc()
    {
        AutoLock lock(cs);
        Block* block;
        if( !freeBlocks )
        {
            BigBlock* bblock = ::new(SystemAlloc(bigBlockSize)) BigBlock(bigBlockSize, pool);
            assert( bblock != 0 );
            freeBlocks = bblock->first;
            pool = bblock;
        }
        block = freeBlocks;
        freeBlocks = freeBlocks->next;
        if( freeBlocks )
            freeBlocks->prev = 0;
        STAT(stat.bruttoBytes += MEM_BLOCK_SIZE);
        return block;
    }

    void free(Block* block)
    {
        AutoLock lock(cs);
        block->prev = 0;
        block->next = freeBlocks;
        freeBlocks = block;
        STAT(stat.bruttoBytes -= MEM_BLOCK_SIZE);
    }

    CriticalSection cs;
    Block* freeBlocks;
    BigBlock* pool;
    int bigBlockSize;
    int blocksPerBigBlock;
};

BlockPool mallocPool;

enum { START=0, FREE=1, GC=2 };

struct ThreadData
{
    ThreadData() { for(int i = 0; i <= MAX_BIN; i++) bins[i][START] = bins[i][FREE] = bins[i][GC] = 0; }
    ~ThreadData()
    {
        // mark all the thread blocks as abandoned or even release them
        for( int i = 0; i <= MAX_BIN; i++ )
        {
            Block *bin = bins[i][START], *block = bin;
            bins[i][START] = bins[i][FREE] = bins[i][GC] = 0;
            if( block )
            {
                do
                {
                    Block* next = block->next;
                    int allocated = block->allocated;
                    {
                    AutoLock lock(block->cs);
                    block->next = block->prev = 0;
                    block->threadData = 0;
                    Node *node = block->publicFreeList;
                    for( ; node != 0; node = node->next )
                        allocated--;
                    }
                    if( allocated == 0 )
                        mallocPool.free(block);
                    block = next;
                }
                while( block != bin );
            }
        }
    }

    void moveBlockToFreeList( Block* block )
    {
        int i = block->binIdx;
        Block*& freePtr = bins[i][FREE];
        CV_DbgAssert( block->next->prev == block && block->prev->next == block );
        if( block != freePtr )
        {
            Block*& gcPtr = bins[i][GC];
            if( gcPtr == block )
                gcPtr = block->next;
            if( block->next != block )
            {
                block->prev->next = block->next;
                block->next->prev = block->prev;
            }
            block->next = freePtr->next;
            block->prev = freePtr;
            freePtr = block->next->prev = block->prev->next = block;
        }
    }

    Block* bins[MAX_BIN+1][3];

#ifdef WIN32
#ifdef WINCE
#   define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif //WINCE

    static DWORD tlsKey;
    static ThreadData* get()
    {
        ThreadData* data;
        if( tlsKey == TLS_OUT_OF_INDEXES )
            tlsKey = TlsAlloc();
        data = (ThreadData*)TlsGetValue(tlsKey);
        if( !data )
        {
            data = new ThreadData;
            TlsSetValue(tlsKey, data);
        }
        return data;
    }
#else //WIN32
    static void deleteData(void* data)
    {
        delete (ThreadData*)data;
    }

    static pthread_key_t tlsKey;
    static ThreadData* get()
    {
        ThreadData* data;
        if( !tlsKey )
            pthread_key_create(&tlsKey, deleteData);
        data = (ThreadData*)pthread_getspecific(tlsKey);
        if( !data )
        {
            data = new ThreadData;
            pthread_setspecific(tlsKey, data);
        }
        return data;
    }
#endif //WIN32
};

#ifdef WIN32
DWORD ThreadData::tlsKey = TLS_OUT_OF_INDEXES;

void deleteThreadAllocData()
{
    if( ThreadData::tlsKey != TLS_OUT_OF_INDEXES )
        delete (ThreadData*)TlsGetValue( ThreadData::tlsKey );
}

#else //WIN32
pthread_key_t ThreadData::tlsKey = 0;
#endif //WIN32

#if 0
static void checkList(ThreadData* tls, int idx)
{
    Block* block = tls->bins[idx][START];
    if( !block )
    {
        CV_DbgAssert( tls->bins[idx][FREE] == 0 && tls->bins[idx][GC] == 0 );
    }
    else
    {
        bool gcInside = false;
        bool freeInside = false;
        do
        {
            if( tls->bins[idx][FREE] == block )
                freeInside = true;
            if( tls->bins[idx][GC] == block )
                gcInside = true;
            block = block->next;
        }
        while( block != tls->bins[idx][START] );
        CV_DbgAssert( gcInside && freeInside );
    }
}
#else
#define checkList(tls, idx)
#endif

void* fastMalloc( size_t size )
{
    if( size > MAX_BLOCK_SIZE )
    {
        size_t size1 = size + sizeof(uchar*)*2 + MEM_BLOCK_SIZE;
        uchar* udata = (uchar*)SystemAlloc(size1);
        uchar** adata = alignPtr((uchar**)udata + 2, MEM_BLOCK_SIZE);
        adata[-1] = udata;
        adata[-2] = (uchar*)size1;
        return adata;
    }

    {
    ThreadData* tls = ThreadData::get();
    int idx = mallocTables.bin(size);
    Block*& startPtr = tls->bins[idx][START];
    Block*& gcPtr = tls->bins[idx][GC];
    Block*& freePtr = tls->bins[idx][FREE], *block = freePtr;
    checkList(tls, idx);
    size = binSizeTab[idx];
    STAT(
        stat.nettoBytes += size;
        stat.mallocCalls++;
        );
    uchar* data = 0;

    for(;;)
    {
        if( block )
        {
            // try to find non-full block
            for(;;)
            {
                CV_DbgAssert( block->next->prev == block && block->prev->next == block );
                if( block->bumpPtr )
                {
                    data = block->bumpPtr;
                    if( (block->bumpPtr += size) >= block->endPtr )
                        block->bumpPtr = 0;
                    break;
                }

                if( block->privateFreeList )
                {
                    data = (uchar*)block->privateFreeList;
                    block->privateFreeList = block->privateFreeList->next;
                    break;
                }

                if( block == startPtr )
                    break;
                block = block->next;
            }
#if 0
            avg_k += _k;
            avg_nk++;
            if( avg_nk == 1000 )
            {
                printf("avg search iters per 1e3 allocs = %g\n", (double)avg_k/avg_nk );
                avg_k = avg_nk = 0;
            }
#endif

            freePtr = block;
            if( !data )
            {
                block = gcPtr;
                for( int k = 0; k < 2; k++ )
                {
                    SANITY_CHECK(block);
                    CV_DbgAssert( block->next->prev == block && block->prev->next == block );
                    if( block->publicFreeList )
                    {
                        {
                        AutoLock lock(block->cs);
                        block->privateFreeList = block->publicFreeList;
                        block->publicFreeList = 0;
                        }
                        Node* node = block->privateFreeList;
                        for(;node != 0; node = node->next)
                            --block->allocated;
                        data = (uchar*)block->privateFreeList;
                        block->privateFreeList = block->privateFreeList->next;
                        gcPtr = block->next;
                        if( block->allocated+1 <= block->almostEmptyThreshold )
                            tls->moveBlockToFreeList(block);
                        break;
                    }
                    block = block->next;
                }
                if( !data )
                    gcPtr = block;
            }
        }

        if( data )
            break;
        block = mallocPool.alloc();
        block->init(startPtr ? startPtr->prev : block, startPtr ? startPtr : block, (int)size, tls);
        if( !startPtr )
            startPtr = gcPtr = freePtr = block;
        checkList(tls, block->binIdx);
        SANITY_CHECK(block);
    }

    ++block->allocated;
    return data;
    }
}

void fastFree( void* ptr )
{
    if( ((size_t)ptr & (MEM_BLOCK_SIZE-1)) == 0 )
    {
        if( ptr != 0 )
        {
            void* origPtr = ((void**)ptr)[-1];
            size_t sz = (size_t)((void**)ptr)[-2];
            SystemFree( origPtr, sz );
        }
        return;
    }

    {
    ThreadData* tls = ThreadData::get();
    Node* node = (Node*)ptr;
    Block* block = (Block*)((size_t)ptr & -(int)MEM_BLOCK_SIZE);
    assert( block->signature == MEM_BLOCK_SIGNATURE );

    if( block->threadData == tls )
    {
        STAT(
        stat.nettoBytes -= block->objSize;
        stat.freeCalls++;
        float ratio = (float)stat.nettoBytes/stat.bruttoBytes;
        if( stat.minUsageRatio > ratio )
            stat.minUsageRatio = ratio;
        );

        SANITY_CHECK(block);

        bool prevFilled = block->isFilled();
        --block->allocated;
        if( !block->isFilled() && (block->allocated == 0 || prevFilled) )
        {
            if( block->allocated == 0 )
            {
                int idx = block->binIdx;
                Block*& startPtr = tls->bins[idx][START];
                Block*& freePtr = tls->bins[idx][FREE];
                Block*& gcPtr = tls->bins[idx][GC];

                if( block == block->next )
                {
                    CV_DbgAssert( startPtr == block && freePtr == block && gcPtr == block );
                    startPtr = freePtr = gcPtr = 0;
                }
                else
                {
                    if( freePtr == block )
                        freePtr = block->next;
                    if( gcPtr == block )
                        gcPtr = block->next;
                    if( startPtr == block )
                        startPtr = block->next;
                    block->prev->next = block->next;
                    block->next->prev = block->prev;
                }
                mallocPool.free(block);
                checkList(tls, idx);
                return;
            }

            tls->moveBlockToFreeList(block);
        }
        node->next = block->privateFreeList;
        block->privateFreeList = node;
    }
    else
    {
        AutoLock lock(block->cs);
        SANITY_CHECK(block);

        node->next = block->publicFreeList;
        block->publicFreeList = node;
        if( block->threadData == 0 )
        {
            // take ownership of the abandoned block.
            // note that it can happen at the same time as
            // ThreadData::deleteData() marks the blocks as abandoned,
            // so this part of the algorithm needs to be checked for data races
            int idx = block->binIdx;
            block->threadData = tls;
            Block*& startPtr = tls->bins[idx][START];

            if( startPtr )
            {
                block->next = startPtr;
                block->prev = startPtr->prev;
                block->next->prev = block->prev->next = block;
            }
            else
                startPtr = tls->bins[idx][FREE] = tls->bins[idx][GC] = block;
        }
    }
    }
}

#endif //CV_USE_SYSTEM_MALLOC

}

CV_IMPL void cvSetMemoryManager( CvAllocFunc, CvFreeFunc, void * )
{
    CV_Error( -1, "Custom memory allocator is not supported" );
}

CV_IMPL void* cvAlloc( size_t size )
{
    return cv::fastMalloc( size );
}

CV_IMPL void cvFree_( void* ptr )
{
    cv::fastFree( ptr );
}


/* End of file. */
