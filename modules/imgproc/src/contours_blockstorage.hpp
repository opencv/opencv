// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CONTOURS_BLOCKSTORAGE_HPP
#define OPENCV_CONTOURS_BLOCKSTORAGE_HPP

#include "precomp.hpp"

#include <array>

namespace cv {

// BLOCK_SIZE_ELEM - number of elements in a block
// STATIC_CAPACITY_BYTES - static memory in bytes for preallocated blocks
template <typename T, size_t BLOCK_SIZE_ELEM = 1024, size_t STATIC_CAPACITY_BYTES = 4096>
class BlockStorage {
    public:
        using value_type = T;
        typedef struct {value_type data[BLOCK_SIZE_ELEM];} block_type;

        BlockStorage()
        {
            const size_t minDynamicBlocks = !staticBlocksCount ? 1 : 0;
            for(size_t i = 0 ; i<minDynamicBlocks ; ++i)
                dynamicBlocks.push_back(new block_type);
        }
        BlockStorage(const BlockStorage&) = delete;
        BlockStorage(BlockStorage&&) = default;
        ~BlockStorage() {
            for(const auto & block : dynamicBlocks) {
                delete block;
            }
        }
        BlockStorage& operator=(const BlockStorage&) = delete;
        BlockStorage& operator=(BlockStorage&&) = default;

        void clear(void) {
            const size_t minDynamicBlocks = !staticBlocksCount ? 1 : 0;
            for(size_t i = minDynamicBlocks, count = dynamicBlocks.size() ; i<count ; ++i ) {
                delete dynamicBlocks[i];
            }
            dynamicBlocks.resize(minDynamicBlocks);
            sz = 0;
        }

        void push_back(const value_type& value) {
            const size_t blockIndex = sz / BLOCK_SIZE_ELEM;
            const size_t currentBlocksCount = staticBlocksCount+dynamicBlocks.size();
            if (blockIndex == currentBlocksCount)
                dynamicBlocks.push_back(new block_type);
            block_type& cur_block =
                (blockIndex < staticBlocksCount) ? staticBlocks[blockIndex] :
                *dynamicBlocks[blockIndex-staticBlocksCount];
            cur_block.data[sz % BLOCK_SIZE_ELEM] = value;
            ++sz;
        }

        size_t size() const { return sz; }

        const value_type& at(size_t index) const {
            const size_t blockIndex = index / BLOCK_SIZE_ELEM;
            const block_type& cur_block =
                (blockIndex < staticBlocksCount) ? staticBlocks[blockIndex] :
                *dynamicBlocks[blockIndex-staticBlocksCount];
            return cur_block.data[index % BLOCK_SIZE_ELEM];
        }
        value_type& at(size_t index) {
            const size_t blockIndex = index / BLOCK_SIZE_ELEM;
            block_type& cur_block =
                (blockIndex < staticBlocksCount) ? staticBlocks[blockIndex] :
                *dynamicBlocks[blockIndex-staticBlocksCount];
            return cur_block.data[index % BLOCK_SIZE_ELEM];
        }
        const value_type& operator[](size_t index) const {return at(index);}
        value_type& operator[](size_t index) {return at(index);}
    public:
        friend class RangeIterator;
        class RangeIterator
        {
            public:
                RangeIterator(const BlockStorage* _owner, size_t _first, size_t _last)
                             :owner(_owner),remaining(_last-_first),
                              blockIndex(_first/BLOCK_SIZE_ELEM),offset(_first%BLOCK_SIZE_ELEM) {
                }
            private:
                const BlockStorage* owner = nullptr;
                size_t remaining = 0;
                size_t blockIndex = 0;
                size_t offset = 0;
            public:
                bool done(void) const {return !remaining;}
                std::pair<const value_type*, size_t> operator*(void) const {return get();}
                std::pair<const value_type*, size_t> get(void) const {
                    const block_type& cur_block =
                        (blockIndex < owner->staticBlocksCount) ? owner->staticBlocks[blockIndex] :
                        *owner->dynamicBlocks[blockIndex-owner->staticBlocksCount];
                    const value_type* rangeStart = cur_block.data+offset;
                    const size_t rangeLength = std::min(remaining, BLOCK_SIZE_ELEM-offset);
                    return std::make_pair(rangeStart, rangeLength);
                }
                RangeIterator& operator++() {
                    std::pair<const value_type*, size_t> range = get();
                    remaining -= range.second;
                    offset = 0;
                    ++blockIndex;
                    return *this;
                }
        };
        RangeIterator getRangeIterator(size_t first, size_t last) const {
          return RangeIterator(this, first, last);
        }
    private:
        std::array<block_type, STATIC_CAPACITY_BYTES/(BLOCK_SIZE_ELEM*sizeof(value_type))> staticBlocks;
        const size_t staticBlocksCount = STATIC_CAPACITY_BYTES/(BLOCK_SIZE_ELEM*sizeof(value_type));
        std::vector<block_type*> dynamicBlocks;
        size_t sz = 0;
};

}  // namespace cv

#endif  // OPENCV_CONTOURS_BLOCKSTORAGE_HPP
