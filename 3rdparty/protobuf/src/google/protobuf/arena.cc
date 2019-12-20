// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <google/protobuf/arena.h>

#include <algorithm>
#include <limits>


#ifdef ADDRESS_SANITIZER
#include <sanitizer/asan_interface.h>
#endif  // ADDRESS_SANITIZER

#include <google/protobuf/stubs/port.h>

namespace google {
static const size_t kMinCleanupListElements = 8;
static const size_t kMaxCleanupListElements = 64;  // 1kB on 64-bit.

namespace protobuf {
namespace internal {


google::protobuf::internal::SequenceNumber ArenaImpl::lifecycle_id_generator_;
#if defined(GOOGLE_PROTOBUF_NO_THREADLOCAL)
ArenaImpl::ThreadCache& ArenaImpl::thread_cache() {
  static internal::ThreadLocalStorage<ThreadCache>* thread_cache_ =
      new internal::ThreadLocalStorage<ThreadCache>();
  return *thread_cache_->Get();
}
#elif defined(PROTOBUF_USE_DLLS)
ArenaImpl::ThreadCache& ArenaImpl::thread_cache() {
  static GOOGLE_THREAD_LOCAL ThreadCache thread_cache_ = { -1, NULL };
  return thread_cache_;
}
#else
GOOGLE_THREAD_LOCAL ArenaImpl::ThreadCache ArenaImpl::thread_cache_ = {-1, NULL};
#endif

void ArenaImpl::Init() {
  lifecycle_id_ = lifecycle_id_generator_.GetNext();
  google::protobuf::internal::NoBarrier_Store(&hint_, 0);
  google::protobuf::internal::NoBarrier_Store(&threads_, 0);

  if (initial_block_) {
    // Thread which calls Init() owns the first block. This allows the
    // single-threaded case to allocate on the first block without having to
    // perform atomic operations.
    InitBlock(initial_block_, &thread_cache(), options_.initial_block_size);
    ThreadInfo* info = NewThreadInfo(initial_block_);
    info->next = NULL;
    google::protobuf::internal::NoBarrier_Store(&threads_,
                                  reinterpret_cast<google::protobuf::internal::AtomicWord>(info));
    google::protobuf::internal::NoBarrier_Store(&space_allocated_,
                                  options_.initial_block_size);
    CacheBlock(initial_block_);
  } else {
    google::protobuf::internal::NoBarrier_Store(&space_allocated_, 0);
  }
}

ArenaImpl::~ArenaImpl() {
  // Have to do this in a first pass, because some of the destructors might
  // refer to memory in other blocks.
  CleanupList();
  FreeBlocks();
}

uint64 ArenaImpl::Reset() {
  // Have to do this in a first pass, because some of the destructors might
  // refer to memory in other blocks.
  CleanupList();
  uint64 space_allocated = FreeBlocks();
  Init();

  return space_allocated;
}

ArenaImpl::Block* ArenaImpl::NewBlock(void* me, Block* my_last_block,
                                      size_t min_bytes) {
  size_t size;
  if (my_last_block != NULL) {
    // Double the current block size, up to a limit.
    size = std::min(2 * my_last_block->size, options_.max_block_size);
  } else {
    size = options_.start_block_size;
  }
  // Verify that min_bytes + kHeaderSize won't overflow.
  GOOGLE_CHECK_LE(min_bytes, std::numeric_limits<size_t>::max() - kHeaderSize);
  size = std::max(size, kHeaderSize + min_bytes);

  Block* b = reinterpret_cast<Block*>(options_.block_alloc(size));
  InitBlock(b, me, size);
  google::protobuf::internal::NoBarrier_AtomicIncrement(&space_allocated_, size);
  return b;
}

void ArenaImpl::InitBlock(Block* b, void *me, size_t size) {
  b->pos = kHeaderSize;
  b->size = size;
  b->owner = me;
  b->next = NULL;
#ifdef ADDRESS_SANITIZER
  // Poison the rest of the block for ASAN. It was unpoisoned by the underlying
  // malloc but it's not yet usable until we return it as part of an allocation.
  ASAN_POISON_MEMORY_REGION(
      reinterpret_cast<char*>(b) + b->pos, b->size - b->pos);
#endif  // ADDRESS_SANITIZER
}

ArenaImpl::CleanupChunk* ArenaImpl::ExpandCleanupList(CleanupChunk* cleanup,
                                                      Block* b) {
  size_t size = cleanup ? cleanup->size * 2 : kMinCleanupListElements;
  size = std::min(size, kMaxCleanupListElements);
  size_t bytes = internal::AlignUpTo8(CleanupChunk::SizeOf(size));
  if (b->avail() < bytes) {
    b = GetBlock(bytes);
  }
  CleanupChunk* list =
      reinterpret_cast<CleanupChunk*>(AllocFromBlock(b, bytes));
  list->next = b->thread_info->cleanup;
  list->size = size;
  list->len = 0;
  b->thread_info->cleanup = list;
  return list;
}

inline GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
void ArenaImpl::AddCleanupInBlock(
    Block* b, void* elem, void (*func)(void*)) {
  CleanupChunk* cleanup = b->thread_info->cleanup;
  if (cleanup == NULL || cleanup->len == cleanup->size) {
    cleanup = ExpandCleanupList(cleanup, b);
  }

  CleanupNode* node = &cleanup->nodes[cleanup->len++];

  node->elem = elem;
  node->cleanup = func;
}

void ArenaImpl::AddCleanup(void* elem, void (*cleanup)(void*)) {
  return AddCleanupInBlock(GetBlock(0), elem, cleanup);
}

void* ArenaImpl::AllocateAligned(size_t n) {
  GOOGLE_DCHECK_EQ(internal::AlignUpTo8(n), n);  // Must be already aligned.

  return AllocFromBlock(GetBlock(n), n);
}

void* ArenaImpl::AllocateAlignedAndAddCleanup(size_t n,
                                              void (*cleanup)(void*)) {
  GOOGLE_DCHECK_EQ(internal::AlignUpTo8(n), n);  // Must be already aligned.

  Block* b = GetBlock(n);
  void* mem = AllocFromBlock(b, n);
  AddCleanupInBlock(b, mem, cleanup);
  return mem;
}

inline GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
ArenaImpl::Block* ArenaImpl::GetBlock(size_t n) {
  Block* my_block = NULL;

  // If this thread already owns a block in this arena then try to use that.
  // This fast path optimizes the case where multiple threads allocate from the
  // same arena.
  ThreadCache* tc = &thread_cache();
  if (tc->last_lifecycle_id_seen == lifecycle_id_) {
    my_block = tc->last_block_used_;
    if (my_block->avail() >= n) {
      return my_block;
    }
  }

  // Check whether we own the last accessed block on this arena.
  // This fast path optimizes the case where a single thread uses multiple
  // arenas.
  Block* b = reinterpret_cast<Block*>(google::protobuf::internal::Acquire_Load(&hint_));
  if (b != NULL && b->owner == tc) {
    my_block = b;
    if (my_block->avail() >= n) {
      return my_block;
    }
  }
  return GetBlockSlow(tc, my_block, n);
}

inline GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
void* ArenaImpl::AllocFromBlock(Block* b, size_t n) {
  GOOGLE_DCHECK_EQ(internal::AlignUpTo8(b->pos), b->pos);  // Must be already aligned.
  GOOGLE_DCHECK_EQ(internal::AlignUpTo8(n), n);  // Must be already aligned.
  GOOGLE_DCHECK_GE(b->avail(), n);
  size_t p = b->pos;
  b->pos = p + n;
#ifdef ADDRESS_SANITIZER
  ASAN_UNPOISON_MEMORY_REGION(reinterpret_cast<char*>(b) + p, n);
#endif  // ADDRESS_SANITIZER
  return reinterpret_cast<char*>(b) + p;
}

ArenaImpl::Block* ArenaImpl::GetBlockSlow(void* me, Block* my_full_block,
                                          size_t n) {
  ThreadInfo* info =
      my_full_block ? my_full_block->thread_info : GetThreadInfo(me, n);
  GOOGLE_DCHECK(info != NULL);
  Block* b = info->head;
  if (b->avail() < n) {
    Block* new_b = NewBlock(me, b, n);
    new_b->thread_info = info;
    new_b->next = b;
    info->head = new_b;
    b = new_b;
  }
  CacheBlock(b);
  return b;
}

uint64 ArenaImpl::SpaceAllocated() const {
  return google::protobuf::internal::NoBarrier_Load(&space_allocated_);
}

uint64 ArenaImpl::SpaceUsed() const {
  ThreadInfo* info =
      reinterpret_cast<ThreadInfo*>(google::protobuf::internal::Acquire_Load(&threads_));
  uint64 space_used = 0;

  for ( ; info; info = info->next) {
    // Remove the overhead of the ThreadInfo itself.
    space_used -= sizeof(ThreadInfo);
    for (Block* b = info->head; b; b = b->next) {
      space_used += (b->pos - kHeaderSize);
    }
  }

  return space_used;
}

uint64 ArenaImpl::FreeBlocks() {
  uint64 space_allocated = 0;
  // By omitting an Acquire barrier we ensure that any user code that doesn't
  // properly synchronize Reset() or the destructor will throw a TSAN warning.
  ThreadInfo* info =
      reinterpret_cast<ThreadInfo*>(google::protobuf::internal::NoBarrier_Load(&threads_));

  while (info) {
    // This is inside the block we are freeing, so we need to read it now.
    ThreadInfo* next_info = info->next;
    for (Block* b = info->head; b; ) {
      // This is inside the block we are freeing, so we need to read it now.
      Block* next_block = b->next;
      space_allocated += (b->size);

#ifdef ADDRESS_SANITIZER
      // This memory was provided by the underlying allocator as unpoisoned, so
      // return it in an unpoisoned state.
      ASAN_UNPOISON_MEMORY_REGION(reinterpret_cast<char*>(b), b->size);
#endif  // ADDRESS_SANITIZER

      if (b != initial_block_) {
        options_.block_dealloc(b, b->size);
      }

      b = next_block;
    }
    info = next_info;
  }

  return space_allocated;
}

void ArenaImpl::CleanupList() {
  // By omitting an Acquire barrier we ensure that any user code that doesn't
  // properly synchronize Reset() or the destructor will throw a TSAN warning.
  ThreadInfo* info =
      reinterpret_cast<ThreadInfo*>(google::protobuf::internal::NoBarrier_Load(&threads_));

  for ( ; info; info = info->next) {
    CleanupChunk* list = info->cleanup;
    while (list) {
      size_t n = list->len;
      CleanupNode* node = &list->nodes[list->len - 1];
      for (size_t i = 0; i < n; i++, node--) {
        node->cleanup(node->elem);
      }
      list = list->next;
    }
  }
}

ArenaImpl::ThreadInfo* ArenaImpl::NewThreadInfo(Block* b) {
  GOOGLE_DCHECK(FindThreadInfo(b->owner) == NULL);
  ThreadInfo* info =
      reinterpret_cast<ThreadInfo*>(AllocFromBlock(b, sizeof(ThreadInfo)));
  b->thread_info = info;
  info->owner = b->owner;
  info->head = b;
  info->cleanup = NULL;
  return info;
}

ArenaImpl::ThreadInfo* ArenaImpl::FindThreadInfo(void* me) {
  ThreadInfo* info =
      reinterpret_cast<ThreadInfo*>(google::protobuf::internal::Acquire_Load(&threads_));
  for ( ; info; info = info->next) {
    if (info->owner == me) {
      return info;
    }
  }

  return NULL;
}

ArenaImpl::ThreadInfo* ArenaImpl::GetThreadInfo(void* me, size_t n) {
  ThreadInfo* info = FindThreadInfo(me);

  if (!info) {
    // This thread doesn't have any ThreadInfo, which also means it doesn't have
    // any blocks yet.  So we'll allocate its first block now.
    Block* b = NewBlock(me, NULL, sizeof(ThreadInfo) + n);
    info = NewThreadInfo(b);

    google::protobuf::internal::AtomicWord head;
    do {
      head = google::protobuf::internal::NoBarrier_Load(&threads_);
      info->next = reinterpret_cast<ThreadInfo*>(head);
    } while (google::protobuf::internal::Release_CompareAndSwap(
                 &threads_, head, reinterpret_cast<google::protobuf::internal::AtomicWord>(info)) != head);
  }

  return info;
}

}  // namespace internal

void Arena::CallDestructorHooks() {
  uint64 space_allocated = impl_.SpaceAllocated();
  // Call the reset hook
  if (on_arena_reset_ != NULL) {
    on_arena_reset_(this, hooks_cookie_, space_allocated);
  }

  // Call the destruction hook
  if (on_arena_destruction_ != NULL) {
    on_arena_destruction_(this, hooks_cookie_, space_allocated);
  }
}

void Arena::OnArenaAllocation(const std::type_info* allocated_type,
                              size_t n) const {
  if (on_arena_allocation_ != NULL) {
    on_arena_allocation_(allocated_type, n, hooks_cookie_);
  }
}

}  // namespace protobuf
}  // namespace google
