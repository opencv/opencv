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
void (*disposeThreadData)() = NULL;

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

#if CV_USE_SYSTEM_MALLOC

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

#else

//#if 0
//#define SANITY_CHECK(block) \
//    CV_Assert(((size_t)(block) & (MEM_BLOCK_SIZE-1)) == 0 && \
//        (unsigned)(block)->binIdx <= (unsigned)MAX_BIN && \
//        (block)->signature == MEM_BLOCK_SIGNATURE)
//#else
#define SANITY_CHECK(block)
//#endif

#define STAT(stmt)

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

struct FastLock
{
    FastLock(CriticalSection& _cs) : cs(&_cs) { cs->lock(); }
    ~FastLock() { cs->unlock(); }
    CriticalSection* cs;
};

struct MemPoolConfig
{
public:
    MemPoolConfig()
    {
        maxBin = 0;
        binSizeTable = binIdxTable = 0;

        setBlockSize(1 << 14);
    }

    void setBlockSize(size_t blockSize)
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

    int getBinIdx(size_t size)
    {
        assert(size <= MAX_BLOCK_SIZE);
        return binIdxTable[(size + 7) >> 3];
    }

    static const size_t MEM_BLOCK_SIGNATURE = 0x01234567;
    static const size_t HDR_SIZE = 128;

    size_t memBlockSize;
    size_t maxBlockSize;

    int maxBin;
    int* binSizeTable;
    int* binIdxTable;

private:
    MemPoolConfig(const MemPoolConfig& m) {}
    MemPoolConfig& operator=(const MemPoolConfig& m) { return *this; }
};

MemPoolConfig config;

struct Node
{
    Node* next;
};

struct ThreadData;

struct Block
{
    Block(Block* _next)
    {
        signature = config.MEM_BLOCK_SIGNATURE;
        prev = 0;
        next = _next;
        privateFreeList = publicFreeList = 0;
        bumpPtr = endPtr = 0;
        objSize = 0;
        threadData = 0;
        data = (uchar*)this + config.HDR_SIZE;
    }

    ~Block() {}

    void init(Block* _prev, Block* _next, size_t _objSize, ThreadData* _threadData)
    {
        prev = _prev;
        if (prev)
            prev->next = this;
        next = _next;
        if (next)
            next->prev = this;
        objSize = _objSize;
        binIdx = config.getBinIdx(objSize);
        threadData = _threadData;
        privateFreeList = publicFreeList = 0;
        bumpPtr = data;
        size_t nObjects = config.maxBlockSize / objSize;
        endPtr = bumpPtr + nObjects * objSize;
        almostEmptyThreshold = (nObjects + 1) / 2;
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
    size_t objSize;
    int binIdx;
    int allocated;
    size_t almostEmptyThreshold;
    CriticalSection cs;
};

struct BigBlock
{
    BigBlock(size_t _bigBlockSize, BigBlock* _next)
    {
        first = alignPtr((Block*)(this + 1), config.memBlockSize);
        next = _next;
        nBlocks = (int)(((char*)this + _bigBlockSize - (char*)first) / config.memBlockSize);
        Block* p = 0;
        for (int i = nBlocks - 1; i >= 0; i--)
            p = ::new((uchar*)first + i * config.memBlockSize) Block(p);
    }

    ~BigBlock()
    {
        for (int i = nBlocks - 1; i >= 0; i--)
            ((Block*)((uchar*)first + i * config.memBlockSize))->~Block();
    }

    BigBlock* next;
    Block* first;
    int nBlocks;
};

struct BlockPool
{
    BlockPool(size_t _bigBlockSize) : pool(0), bigBlockSize(_bigBlockSize)
    {
    }

    ~BlockPool()
    {
        FastLock lock(cs);

        while (pool)
        {
            BigBlock* nextBlock = pool->next;
            pool->~BigBlock();
            SystemFree(pool, bigBlockSize);
            pool = nextBlock;
        }
    }

    Block* alloc()
    {
        FastLock lock(cs);

        if (!freeBlocks)
        {
            BigBlock* bblock = ::new(SystemAlloc(bigBlockSize)) BigBlock(bigBlockSize, pool);
            assert(bblock != 0);
            freeBlocks = bblock->first;
            pool = bblock;
        }

        Block* block = freeBlocks;
        freeBlocks = freeBlocks->next;
        if (freeBlocks)
            freeBlocks->prev = 0;
        STAT(stat.bruttoBytes += config.memBlockSize);
        return block;
    }

    void free(Block* block)
    {
        FastLock lock(cs);

        block->prev = 0;
        block->next = freeBlocks;
        freeBlocks = block;
        STAT(stat.bruttoBytes -= config.memBlockSize);
    }

    CriticalSection cs;
    Block* freeBlocks;
    BigBlock* pool;
    size_t bigBlockSize;
};

BlockPool blockPool(1 << 20);

enum { START = 0, FREE = 1, GC = 2 };

struct ThreadData
{
    ThreadData() 
    { 
        bins = new Block**[config.maxBin];
        for (int i = config.maxBin - 1; i >= 0; i--)
            bins[i] = new Block*[3];

        for (int i = 0; i < config.maxBin; i++)
            bins[i][START] = bins[i][FREE] = bins[i][GC] = 0; 
    }

    ThreadData(const ThreadData& other)
    {
        bins = new Block**[config.maxBin];
        for (int i = config.maxBin - 1; i >= 0; i--)
            bins[i] = new Block*[3];

        for (int i = 0; i < config.maxBin; i++)
        {
            bins[i][START] = other.bins[i][START];
            bins[i][FREE] = other.bins[i][FREE];
            bins[i][GC] = other.bins[i][GC];
        }
    }

    ThreadData& operator=(const ThreadData& other)
    {
        if (this == &other)
            return *this;

        for (int i = 0; i < config.maxBin; i++)
        {
            bins[i][START] = other.bins[i][START];
            bins[i][FREE] = other.bins[i][FREE];
            bins[i][GC] = other.bins[i][GC];
        }

        return *this;
    }

    ~ThreadData()
    {
        // mark all the thread blocks as abandoned or even release them
        for (int i = 0; i < config.maxBin; i++)
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
                        blockPool.free(block);
                    block = next;
                }
                while (block != bin);
            }
        }

        for (int i = 0; i < config.maxBlockSize; i++)
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

#ifdef WIN32 || defined _WIN32

#ifdef WINCE
#   define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif //WINCE

    static void deleteData()
    {
        if (ThreadData::tlsKey != TLS_OUT_OF_INDEXES)
            delete (ThreadData*)TlsGetValue(ThreadData::tlsKey);
    }

    static DWORD tlsKey;
    static ThreadData* get()
    {
        registerThreadDataDisposer(deleteData);

        ThreadData* data;
        if (tlsKey == TLS_OUT_OF_INDEXES)
            tlsKey = TlsAlloc();
        data = (ThreadData*)TlsGetValue(tlsKey);
        if (!data)
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

#endif //WIN32
};

#ifdef WIN32 || defined _WIN32
DWORD ThreadData::tlsKey = TLS_OUT_OF_INDEXES;
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

void* fastMalloc(size_t size)
{
    if (size > config.maxBlockSize)
    {
        size_t actualSize = size + sizeof(uchar*) * 2 + config.memBlockSize;
        uchar* udata = (uchar*)SystemAlloc(actualSize);
        uchar** adata = alignPtr((uchar**)udata + 2, config.memBlockSize);
        adata[-1] = udata;
        adata[-2] = (uchar*)actualSize;
        return adata;
    }

    {
    ThreadData* tls = ThreadData::get();
    int idx = config.getBinIdx(size);
    Block*& startPtr = tls->bins[idx][START];
    Block*& gcPtr = tls->bins[idx][GC];
    Block*& freePtr = tls->bins[idx][FREE], *block = freePtr;
    checkList(tls, idx);
    size = config.binSizeTable[idx];
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

void fastFree(void* ptr)
{
    if (((size_t)ptr & (config.memBlockSize - 1)) == 0)
    {
        if (ptr != 0)
        {
            void* origPtr = ((void**)ptr)[-1];
            size_t sz = (size_t)((void**)ptr)[-2];
            SystemFree(origPtr, sz);
        }
        return;
    }

    {
    ThreadData* tls = ThreadData::get();
    Node* node = (Node*)ptr;
    Block* block = (Block*)((size_t)ptr & -(int)config.memBlockSize);
    assert(block->signature == MEM_BLOCK_SIGNATURE);

    if (block->threadData == tls)
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
                return;
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
}

#endif //CV_USE_SYSTEM_MALLOC

}

CvAllocFunc cvAllocFunc = cv::fastMalloc;
CvFreeFunc cvFreeFunc = cv::fastFree;

CV_IMPL void* cvAlloc(size_t size)
{
    return cvAllocFunc(size);
}

CV_IMPL void cvFree_(void* ptr)
{
    cvFreeFunc(ptr);
}

CV_IMPL void cvSetMemoryManager(CvAllocFunc alloc_func, CvFreeFunc free_func)
{
    if (alloc_func != NULL)
        cvAllocFunc = alloc_func;

    if (free_func != NULL)
        cvFreeFunc = free_func;

    CV_Error( -1, "Custom memory allocator is not supported" );
}

/* End of file. */
