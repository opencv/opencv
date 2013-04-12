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

//#define CV_USE_SYSTEM_MALLOC 1

namespace cv
{

static void* OutOfMemoryError(size_t size)
{
    CV_Error_(CV_StsNoMem, ("Failed to allocate %lu bytes", (unsigned long)size));
    return 0;
}

#if defined WIN32 || defined _WIN32
static void (*disposeThreadData)() = NULL;

void deleteThreadAllocData() 
{
    if (disposeThreadData != NULL)
        disposeThreadData();
}

inline void registerThreadDataDisposer(void (*_disposeThreadData)())
{
    disposeThreadData = _disposeThreadData;
}
#endif

#ifdef WIN32

#include <windows.h>

struct CriticalSection
{
    CriticalSection() { InitializeCriticalSection(&cs); }
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

#else

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

#endif

struct FastLock
{
    FastLock(CriticalSection& _cs) : cs(&_cs) { cs->lock(); }
    ~FastLock() { cs->unlock(); }
    CriticalSection* cs;
};

///////////////////////////////// System Alloc/Free /////////////////////////////////

struct NaiveAllocator
{
    static void* allocate(size_t size, void* userdata = NULL)
    {
        uchar* udata = (uchar*)malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
        if(!udata)
            return OutOfMemoryError(size);
        uchar** adata = alignPtr((uchar**)udata + 1, CV_MALLOC_ALIGN);
        adata[-1] = udata;
        return adata;
    }

    static int deallocate(void* ptr, void* userdata = NULL)
    {
        if (ptr)
        {
            uchar* udata = ((uchar**)ptr)[-1];
            CV_DbgAssert(udata < (uchar*)ptr &&
                ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*) + CV_MALLOC_ALIGN));
            free(udata);
        }

        return 0;
    }
};

//////////////////////////////////// Memory Pool ////////////////////////////////////

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
        data = (uchar*)this + HDR_SIZE;
        threadData = 0;
        objSize = binIdx = allocated = 0;
    }

    void init(Block* _prev, Block* _next, size_t _objSize, ThreadData* _threadData)
    {
        prev = _prev;
        if (prev)
            prev->next = this;
        next = _next;
        if (next)
            next->prev = this;
        privateFreeList = publicFreeList = 0;
        bumpPtr = data;
        int nObjects = maxBlockSize / _objSize;
        endPtr = bumpPtr + nObjects * _objSize;
        threadData = _threadData;
        objSize = _objSize;
        binIdx = getBinIdx(objSize);
        allocated = 0;
        almostEmptyThreshold = (nObjects + 1) / 2;
    }

    bool isFilled() const 
    { 
        return allocated > almostEmptyThreshold; 
    }

    // Do not change the order of the following members! 
    // Otherwise sizeof(Block) may change.
    Block* prev;
    Block* next;
    Node* privateFreeList;
    Node* publicFreeList;
    uchar* bumpPtr;
    uchar* endPtr;
    uchar* data;
    ThreadData* threadData;
    size_t objSize;
    unsigned int signature;
    int binIdx;
    int allocated;
    int almostEmptyThreshold;
    CriticalSection cs;

    ////////////////////////// Static Members and Methods //////////////////////////
    
    static void initBlockSize(size_t blockSize)
    {
        // make sure blockSize can be represented as 2^n and larger than HDR_SIZE * 2
        assert(!(blockSize & (blockSize - 1)) && blockSize > HDR_SIZE * 2);

        memBlockSize = blockSize;
        maxBlockSize = memBlockSize - HDR_SIZE;

        maxBin = 0;
        while (blockSize)
        {
            maxBin++;
            blockSize >>= 1;
        }
        maxBin -= 3;

        delete[] binSizeTable;
        binSizeTable = new int[maxBin];
        for (int i = maxBin - 2; i >= 0; i--)
        {
            binSizeTable[i] = 1 << (i + 3);
        }
        binSizeTable[maxBin - 1] = maxBlockSize;

        delete[] binIdxTable;
        binIdxTable = new int[(maxBlockSize >> 3) + 1];
        int j = 0;
        for (int i = 0; i < maxBin; i++)
        {
            int n = binSizeTable[i] >> 3;
            for (; j <= n; j++)
                binIdxTable[j] = i;
        }
    }

    static int getBinIdx(size_t size)
    {
        assert(size <= maxBlockSize);
        return binIdxTable[(size + 7) >> 3];
    }

    static const unsigned int MEM_BLOCK_SIGNATURE = 0x01234567;
    static const size_t HDR_SIZE;
    static size_t memBlockSize;
    static size_t maxBlockSize;
    static int maxBin;
    static int* binSizeTable;
    static int* binIdxTable;
};
const size_t Block::HDR_SIZE = sizeof(Block);
size_t Block::memBlockSize = 0;
size_t Block::maxBlockSize = 0;
int Block::maxBin = 0;
int* Block::binSizeTable = 0;
int* Block::binIdxTable = 0;

#if 0
#define SANITY_CHECK(block) \
    CV_Assert(((size_t)(block) & (Block::memBlockSize - 1)) == 0 && \
    (unsigned)(block)->binIdx < (unsigned)Block::maxBin && \
    (block)->signature == Block::MEM_BLOCK_SIGNATURE)
#else
#define SANITY_CHECK(block)
#endif

#define STAT(stmt)

struct BigBlock
{
    BigBlock(BigBlock* _next)
    {
        first = alignPtr((Block*)(this + 1), Block::memBlockSize);
        next = _next;
        nBlocks = (int)(((char*)this + bigBlockSize - (char*)first) / Block::memBlockSize);
        Block* p = 0;
        for (int i = nBlocks - 1; i >= 0; i--)
            p = ::new((uchar*)first + i * Block::memBlockSize) Block(p);
    }

    ~BigBlock()
    {
        for (int i = nBlocks - 1; i >= 0; i--)
            ((Block*)((uchar*)first + i * Block::memBlockSize))->~Block();
    }

    BigBlock* next;
    Block* first;
    int nBlocks;

    ////////////////////////// Static Members and Methods //////////////////////////

    static size_t bigBlockSize;
};
size_t BigBlock::bigBlockSize = 0;

struct BlockPool
{
    BlockPool() : pool(0)
    {
    }

    ~BlockPool()
    {
        clear();
    }

    void clear()
    {
        FastLock lock(cs);

        while (pool)
        {
            BigBlock* nextBlock = pool->next;
            pool->~BigBlock();
            SystemFree(pool, BigBlock::bigBlockSize);
            pool = nextBlock;
        }
    }

    Block* alloc()
    {
        FastLock lock(cs);

        if (!freeBlocks)
        {
            BigBlock* bblock = ::new(SystemAlloc(BigBlock::bigBlockSize)) BigBlock(pool);
            assert(bblock != 0);
            freeBlocks = bblock->first;
            pool = bblock;
        }

        Block* block = freeBlocks;
        freeBlocks = freeBlocks->next;
        if (freeBlocks)
            freeBlocks->prev = 0;
        STAT(stat.bruttoBytes += Block::memBlockSize);
        return block;
    }

    void free(Block* block)
    {
        FastLock lock(cs);

        block->prev = 0;
        block->next = freeBlocks;
        freeBlocks = block;
        STAT(stat.bruttoBytes -= Block::memBlockSize);
    }

    CriticalSection cs;
    Block* freeBlocks;
    BigBlock* pool;
};

enum { START = 0, FREE = 1, GC = 2 };

struct ThreadData
{
    ThreadData(BlockPool* _blockPool) 
    { 
        bins = new Block**[Block::maxBin];
        for (int i = Block::maxBin - 1; i >= 0; i--)
            bins[i] = new Block*[3];

        for (int i = 0; i < Block::maxBin; i++)
            bins[i][START] = bins[i][FREE] = bins[i][GC] = 0; 

        blockPool = _blockPool;
    }

    ThreadData(const ThreadData& other)
    {
        bins = new Block**[Block::maxBin];
        for (int i = Block::maxBin - 1; i >= 0; i--)
            bins[i] = new Block*[3];

        for (int i = 0; i < Block::maxBin; i++)
        {
            bins[i][START] = other.bins[i][START];
            bins[i][FREE] = other.bins[i][FREE];
            bins[i][GC] = other.bins[i][GC];
        }

        blockPool = other.blockPool;
    }

    ThreadData& operator=(const ThreadData& other)
    {
        if (this == &other)
            return *this;

        for (int i = 0; i < Block::maxBin; i++)
        {
            bins[i][START] = other.bins[i][START];
            bins[i][FREE] = other.bins[i][FREE];
            bins[i][GC] = other.bins[i][GC];
        }

        blockPool = other.blockPool;

        return *this;
    }

    ~ThreadData()
    {
        // mark all the thread blocks as abandoned or even release them
        for (int i = 0; i < Block::maxBin; i++)
        {
            Block *bin = bins[i][START], *block = bin;
            bins[i][START] = bins[i][FREE] = bins[i][GC] = 0;

            if (block)
            {
                do
                {
                    Block* next = block->next;
                    int allocated = block->allocated;

                    {
                        FastLock lock(block->cs);

                        block->next = block->prev = 0;
                        block->threadData = 0;
                        Node *node = block->publicFreeList;
                        for (; node != 0; node = node->next)
                            allocated--;
                    }

                    if (allocated == 0)
                        blockPool->free(block);
                    block = next;
                }
                while (block != bin);
            }
        }

        for (int i = 0; i < Block::maxBin; i++)
            delete[] bins[i];
        delete[] bins;
    }

    void moveBlockToFreeList(Block* block)
    {
        int idx = block->binIdx;
        Block*& freePtr = bins[idx][FREE];
        CV_DbgAssert(block->next->prev == block && block->prev->next == block);

        if (block != freePtr)
        {
            Block*& gcPtr = bins[idx][GC];
            if (gcPtr == block)
                gcPtr = block->next;
            if (block->next != block)
            {
                block->prev->next = block->next;
                block->next->prev = block->prev;
            }
            block->next = freePtr->next;
            block->prev = freePtr;
            freePtr = block->next->prev = block->prev->next = block;
        }
    }

    Block*** bins;
    BlockPool* blockPool;

    ////////////////////////// Static Members and Methods //////////////////////////

#ifdef WIN32

#ifndef TLS_OUT_OF_INDEXES
#define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif

    static DWORD tlsKey;

    static void deleteData()
    {
        if (tlsKey != TLS_OUT_OF_INDEXES)
            delete (ThreadData*)TlsGetValue(tlsKey);
    }

    friend struct StaticConstructor;
    struct StaticConstructor 
    {
        StaticConstructor()
        {
            registerThreadDataDisposer(deleteData);
        }
    };
    static StaticConstructor staticConstructor;

    static ThreadData* get(BlockPool* _blockPool)
    {
        ThreadData* data;
        if (tlsKey == TLS_OUT_OF_INDEXES)
            tlsKey = TlsAlloc();
        data = (ThreadData*)TlsGetValue(tlsKey);
        if (!data)
        {
            data = new ThreadData(_blockPool);
            TlsSetValue(tlsKey, data);
        }
        return data;
    }

#else //WIN32

    static pthread_key_t tlsKey;

    static void deleteData(void* data)
    {
        delete (ThreadData*)data;
    }

    static ThreadData* get()
    {
        ThreadData* data;
        if (!tlsKey)
            pthread_key_create(&tlsKey, deleteData);
        data = (ThreadData*)pthread_getspecific(tlsKey);
        if (!data)
        {
            data = new ThreadData;
            pthread_setspecific(tlsKey, data);
        }
        return data;
    }

#endif
};
#ifdef WIN32
DWORD ThreadData::tlsKey = TLS_OUT_OF_INDEXES;
ThreadData::StaticConstructor ThreadData::staticConstructor;
#else
pthread_key_t ThreadData::tlsKey = 0;
#endif

#if 0
static void checkList(ThreadData* tls, int idx)
{
    Block* block = tls->bins[idx][START];
    if (!block)
    {
        CV_DbgAssert(tls->bins[idx][FREE] == 0 && tls->bins[idx][GC] == 0);
    }
    else
    {
        bool gcInside = false;
        bool freeInside = false;
        do
        {
            if (tls->bins[idx][FREE] == block)
                freeInside = true;
            if (tls->bins[idx][GC] == block)
                gcInside = true;
            block = block->next;
        }
        while (block != tls->bins[idx][START]);
        CV_DbgAssert(gcInside && freeInside);
    }
}
#else
#define checkList(tls, idx)
#endif

struct MemPoolAllocator
{
    static size_t align(size_t num)
    {
        size_t result = 1;

        while (result < num)
        {
            result <<= 1;
        }

        return result;
    }

    static void init(size_t blockSize = 16256)
    {
        if (blockSize <= Block::HDR_SIZE)
        {
            CV_Error(-1, "BlockSize is too small.");
            return;
        }

        blockPool.clear();
        size_t actualBlockSize = align(blockSize + Block::HDR_SIZE);
        Block::initBlockSize(actualBlockSize);
        BigBlock::bigBlockSize = actualBlockSize << 2;
    };

    static BlockPool blockPool;

    static void* allocate(size_t size, void* userdata = NULL)
    {
        if (size > Block::maxBlockSize)
        {
            size_t actualSize = size + sizeof(uchar*) * 2 + Block::memBlockSize;
            uchar* udata = (uchar*)SystemAlloc(actualSize);
            uchar** adata = alignPtr((uchar**)udata + 2, Block::memBlockSize);
            adata[-1] = udata;
            adata[-2] = (uchar*)actualSize;
            return adata;
        }

        {
            ThreadData* tls = ThreadData::get(&blockPool);
            int idx = Block::getBinIdx(size);
            Block*& startPtr = tls->bins[idx][START];
            Block*& gcPtr = tls->bins[idx][GC];
            Block*& freePtr = tls->bins[idx][FREE], *block = freePtr;
            checkList(tls, idx);
            size = Block::binSizeTable[idx];
            STAT(
                stat.nettoBytes += size;
                stat.mallocCalls++;
            );
            uchar* data = 0;

            for (;;)
            {
                if (block)
                {
                    // try to find non-full block
                    for (;;)
                    {
                        CV_DbgAssert(block->next->prev == block && block->prev->next == block);

                        if (block->bumpPtr)
                        {
                            data = block->bumpPtr;
                            if ((block->bumpPtr += size) >= block->endPtr)
                                block->bumpPtr = 0;
                            break;
                        }

                        if (block->privateFreeList)
                        {
                            data = (uchar*)block->privateFreeList;
                            block->privateFreeList = block->privateFreeList->next;
                            break;
                        }

                        if (block == startPtr)
                            break;
                        block = block->next;
                    }

#if 0
                    avg_k += _k;
                    avg_nk++;
                    if( avg_nk == 1000 )
                    {
                        printf("avg search iters per 1e3 allocs = %g\n", (double)avg_k / avg_nk );
                        avg_k = avg_nk = 0;
                    }
#endif

                    freePtr = block;
                    if (!data)
                    {
                        block = gcPtr;
                        for (int k = 0; k < 2; k++)
                        {
                            SANITY_CHECK(block);
                            CV_DbgAssert(block->next->prev == block && block->prev->next == block);

                            if (block->publicFreeList)
                            {
                                {
                                    FastLock lock(block->cs);

                                    block->privateFreeList = block->publicFreeList;
                                    block->publicFreeList = 0;
                                }

                                Node* node = block->privateFreeList;
                                for (; node != 0; node = node->next)
                                    --block->allocated;
                                data = (uchar*)block->privateFreeList;
                                block->privateFreeList = block->privateFreeList->next;
                                gcPtr = block->next;
                                if (block->allocated + 1 <= block->almostEmptyThreshold)
                                    tls->moveBlockToFreeList(block);
                                break;
                            }
                            block = block->next;
                        }
                        if (!data)
                            gcPtr = block;
                    }
                }

                if (data)
                    break;
                block = blockPool.alloc();
                block->init(startPtr ? startPtr->prev : block, startPtr ? startPtr : block, (int)size, tls);
                if (!startPtr)
                    startPtr = gcPtr = freePtr = block;

                checkList(tls, block->binIdx);
                SANITY_CHECK(block);
            }

            ++block->allocated;
            return data;
        }
    }

    static int deallocate(void* ptr, void* userdata = NULL)
    {
        if (((size_t)ptr & (Block::memBlockSize - 1)) == 0)
        {
            if (ptr != 0)
            {
                void* origPtr = ((void**)ptr)[-1];
                size_t sz = (size_t)((void**)ptr)[-2];
                SystemFree(origPtr, sz);
            }
            return 0;
        }

        {
            ThreadData* tls = ThreadData::get(&blockPool);
            Node* node = (Node*)ptr;
            Block* block = (Block*)((size_t)ptr & -(int)Block::memBlockSize);
            assert(block->signature == Block::MEM_BLOCK_SIGNATURE);

            if (block->threadData == tls)
            {
                STAT(
                    stat.nettoBytes -= block->objSize;
                    stat.freeCalls++;
                    float ratio = (float)stat.nettoBytes / stat.bruttoBytes;
                    if (stat.minUsageRatio > ratio)
                        stat.minUsageRatio = ratio;
                );
                SANITY_CHECK(block);

                bool prevFilled = block->isFilled();
                --block->allocated;
                if (!block->isFilled() && (block->allocated == 0 || prevFilled))
                {
                    if (block->allocated == 0)
                    {
                        int idx = block->binIdx;
                        Block*& startPtr = tls->bins[idx][START];
                        Block*& freePtr = tls->bins[idx][FREE];
                        Block*& gcPtr = tls->bins[idx][GC];

                        if (block == block->next)
                        {
                            CV_DbgAssert(startPtr == block && freePtr == block && gcPtr == block);
                            startPtr = freePtr = gcPtr = 0;
                        }
                        else
                        {
                            if (freePtr == block)
                                freePtr = block->next;
                            if (gcPtr == block)
                                gcPtr = block->next;
                            if (startPtr == block)
                                startPtr = block->next;
                            block->prev->next = block->next;
                            block->next->prev = block->prev;
                        }
                        blockPool.free(block);
                        checkList(tls, idx);
                        return 0;
                    }
                    tls->moveBlockToFreeList(block);
                }
                node->next = block->privateFreeList;
                block->privateFreeList = node;
            }
            else
            {
                FastLock lock(block->cs);
                SANITY_CHECK(block);

                node->next = block->publicFreeList;
                block->publicFreeList = node;
                if (block->threadData == 0)
                {
                    // take ownership of the abandoned block.
                    // note that it can happen at the same time as
                    // ThreadData::deleteData() marks the blocks as abandoned,
                    // so this part of the algorithm needs to be checked for data races
                    int idx = block->binIdx;
                    block->threadData = tls;
                    Block*& startPtr = tls->bins[idx][START];

                    if (startPtr)
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

        return 0;
    }
};
BlockPool MemPoolAllocator::blockPool;

/////////////////////////// Alloc/Free Strategy Management ///////////////////////////

static CvAllocFunc cvAllocFuncTable[3] = { NaiveAllocator::allocate, MemPoolAllocator::allocate };
static CvFreeFunc cvFreeFuncTable[3] = { NaiveAllocator::deallocate, MemPoolAllocator::deallocate };
static void* cvUserDataTable[3] = { };
static uchar selectedIdx = 0;

inline void* fastMalloc(size_t size)
{
    uchar* mem = (uchar*)cvAllocFuncTable[selectedIdx](size + 1, cvUserDataTable[selectedIdx]);
    *mem = selectedIdx;
    return mem + 1;
}

inline void fastFree(void* ptr)
{
    uchar* mem = (uchar*)ptr - 1;
    uchar idx = *mem;
    cvFreeFuncTable[idx](mem, cvUserDataTable[idx]);
}
}

CV_IMPL void* cvAlloc(size_t size)
{
    return cv::fastMalloc(size);
}

CV_IMPL void cvFree_(void* ptr)
{
    return cv::fastFree(ptr);
}

static cv::CriticalSection cs;
static bool hasSetMemoryPool = false;
static bool hasSetUserDefinedMemoryManager = false;

CV_IMPL void cvTurnOnMemoryPool(size_t blockSize)
{
    cs.lock();

    if (hasSetMemoryPool)
    {
        CV_Error(-1, "Turning on memory pool for more than one time is not supported.");
        cs.unlock();
        return;
    }

    hasSetMemoryPool = true;
    cv::MemPoolAllocator::init(blockSize);
    cv::selectedIdx = 1;

    cs.unlock();
}

CV_IMPL void cvTurnOffMemoryPool()
{
    cs.lock();

    cv::selectedIdx = 0;

    cs.unlock();
}

CV_IMPL void cvSetMemoryManager(CvAllocFunc allocFunc, CvFreeFunc freeFunc, void* userData)
{
    if (allocFunc == NULL && freeFunc == NULL)
        return;

    if ((allocFunc == NULL && freeFunc != NULL) ||
        (allocFunc != NULL && freeFunc == NULL))
    {
        CV_Error(-1, "You need to provide both the allocator and the dellocator.");
        return;
    }

    cs.lock();

    if (hasSetUserDefinedMemoryManager)
    {
        CV_Error(-1, "Setting user defined memory manager for more than one time is not supported.");
        cs.unlock();
        return;
    }

    hasSetUserDefinedMemoryManager = true;
    cv::cvAllocFuncTable[2] = allocFunc;
    cv::cvFreeFuncTable[2] = freeFunc;
    cv::cvUserDataTable[2] = userData;

    cs.unlock();
}

CV_IMPL void cvRemoveMemoryManager()
{
    cs.lock();

    cv::selectedIdx = 0;

    cs.unlock();
}

/* End of file. */
