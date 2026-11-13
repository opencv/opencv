/*M*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

/***********************************************************************
 * Author: Vincent Rabaud
 *************************************************************************/

#ifndef OPENCV_FLANN_DYNAMIC_BITSET_H_
#define OPENCV_FLANN_DYNAMIC_BITSET_H_

//! @cond IGNORED

#ifndef FLANN_USE_BOOST
#  define FLANN_USE_BOOST 0
#endif
//#define FLANN_USE_BOOST 1
#if FLANN_USE_BOOST
#include <boost/dynamic_bitset.hpp>
typedef boost::dynamic_bitset<> DynamicBitset;
#else

#include <limits.h>

#include "dist.h"

namespace cvflann {

/** Class re-implementing the boost version of it
 * This helps not depending on boost, it also does not do the bound checks
 * and has a way to reset a block for speed
 */
class DynamicBitset
{
public:
    /** default constructor
     */
    DynamicBitset() : size_(0)
    {
    }

    /** only constructor we use in our code
     * @param sz the size of the bitset (in bits)
     */
    DynamicBitset(size_t sz)
    {
        resize(sz);
        reset();
    }

    /** Sets all the bits to 0
     */
    void clear()
    {
        std::fill(bitset_.begin(), bitset_.end(), 0);
    }

    /** @brief checks if the bitset is empty
     * @return true if the bitset is empty
     */
    bool empty() const
    {
        return bitset_.empty();
    }

    /** set all the bits to 0
     */
    void reset()
    {
        std::fill(bitset_.begin(), bitset_.end(), 0);
    }

    /** @brief set one bit to 0
     */
    void reset(size_t index)
    {
        bitset_[index / cell_bit_size_] &= ~(size_t(1) << (index % cell_bit_size_));
    }

    /** @brief sets a specific bit to 0, and more bits too
     * This function is useful when resetting a given set of bits so that the
     * whole bitset ends up being 0: if that's the case, we don't care about setting
     * other bits to 0
     */
    void reset_block(size_t index)
    {
        bitset_[index / cell_bit_size_] = 0;
    }

    /** resize the bitset so that it contains at least sz bits
     */
    void resize(size_t sz)
    {
        size_ = sz;
        bitset_.resize(sz / cell_bit_size_ + 1);
    }

    /** set a bit to true
     * @param index the index of the bit to set to 1
     */
    void set(size_t index)
    {
        bitset_[index / cell_bit_size_] |= size_t(1) << (index % cell_bit_size_);
    }

    /** gives the number of contained bits
     */
    size_t size() const
    {
        return size_;
    }

    /** check if a bit is set
     * @param index the index of the bit to check
     * @return true if the bit is set
     */
    bool test(size_t index) const
    {
        return (bitset_[index / cell_bit_size_] & (size_t(1) << (index % cell_bit_size_))) != 0;
    }

private:
    std::vector<size_t> bitset_;
    size_t size_;
    static const unsigned int cell_bit_size_ = CHAR_BIT * sizeof(size_t);
};

} // namespace cvflann

#endif

#include <vector>

namespace cvflann {

/** The dataset points one search has already checked.
 * Every search on a thread reuses the same bits: a search marks the points it checks, and the destructor clears
 * only those marks. That costs far less than allocating and zeroing one bit per dataset point for every query.
 * A search that starts while another one runs on the same thread gets bits of its own.
 */
class CheckedSet
{
public:
    explicit CheckedSet(size_t size) : bits_(&own_bits_), touched_(&own_touched_), overflowed_(false)
    {
        if (depth() == 0)
        {
            bits_ = &threadBits();
            touched_ = &threadTouched();
        }
        if (bits_->size() < size)
            bits_->resize(size);
        max_touched_ = size / 64 + 64;
        ++depth();
    }
    ~CheckedSet()
    {
        if (overflowed_)
        {
            bits_->reset();
        }
        else
        {
            for (size_t i = 0; i < touched_->size(); ++i)
                bits_->reset((*touched_)[i]);
        }
        touched_->clear();
        --depth();
    }
    bool test(size_t index) const { return bits_->test(index); }
    void set(size_t index)
    {
        if (bits_->test(index))
            return;
        bits_->set(index);
        if (!overflowed_)
        {
            if (touched_->size() < max_touched_)
                touched_->push_back(index);
            else
                overflowed_ = true;
        }
    }

private:
    CheckedSet(const CheckedSet&);
    CheckedSet& operator=(const CheckedSet&);
    static DynamicBitset& threadBits() { thread_local DynamicBitset bits; return bits; }
    static std::vector<size_t>& threadTouched() { thread_local std::vector<size_t> touched; return touched; }
    static int& depth() { thread_local int searches = 0; return searches; }

    DynamicBitset own_bits_;
    std::vector<size_t> own_touched_;
    DynamicBitset* bits_;
    std::vector<size_t>* touched_;
    size_t max_touched_;
    bool overflowed_;
};

} // namespace cvflann

//! @endcond

#endif // OPENCV_FLANN_DYNAMIC_BITSET_H_
