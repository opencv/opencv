/***********************************************************************
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

#ifndef OPENCV_FLANN_LSH_TABLE_H_
#define OPENCV_FLANN_LSH_TABLE_H_

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits.h>
// TODO as soon as we use C++0x, use the code in USE_UNORDERED_MAP
#ifdef __GXX_EXPERIMENTAL_CXX0X__
#  define USE_UNORDERED_MAP 1
#else
#  define USE_UNORDERED_MAP 0
#endif
#if USE_UNORDERED_MAP
#include <unordered_map>
#else
#include <map>
#endif
#include <math.h>
#include <stddef.h>

#include "dynamic_bitset.h"
#include "matrix.h"

namespace cvflann
{

namespace lsh
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** What is stored in an LSH bucket
 */
typedef uint32_t FeatureIndex;
/** The id from which we can get a bucket back in an LSH table
 */
typedef unsigned int BucketKey;

/** A bucket in an LSH table
 */
typedef std::vector<FeatureIndex> Bucket;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** POD for stats about an LSH table
 */
struct LshStats
{
    std::vector<unsigned int> bucket_sizes_;
    size_t n_buckets_;
    size_t bucket_size_mean_;
    size_t bucket_size_median_;
    size_t bucket_size_min_;
    size_t bucket_size_max_;
    size_t bucket_size_std_dev;
    /** Each contained vector contains three value: beginning/end for interval, number of elements in the bin
     */
    std::vector<std::vector<unsigned int> > size_histogram_;
};

/** Overload the << operator for LshStats
 * @param out the streams
 * @param stats the stats to display
 * @return the streams
 */
inline std::ostream& operator <<(std::ostream& out, const LshStats& stats)
{
    int w = 20;
    out << "Lsh Table Stats:\n" << std::setw(w) << std::setiosflags(std::ios::right) << "N buckets : "
    << stats.n_buckets_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "mean size : "
    << std::setiosflags(std::ios::left) << stats.bucket_size_mean_ << "\n" << std::setw(w)
    << std::setiosflags(std::ios::right) << "median size : " << stats.bucket_size_median_ << "\n" << std::setw(w)
    << std::setiosflags(std::ios::right) << "min size : " << std::setiosflags(std::ios::left)
    << stats.bucket_size_min_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "max size : "
    << std::setiosflags(std::ios::left) << stats.bucket_size_max_;

    // Display the histogram
    out << std::endl << std::setw(w) << std::setiosflags(std::ios::right) << "histogram : "
    << std::setiosflags(std::ios::left);
    for (std::vector<std::vector<unsigned int> >::const_iterator iterator = stats.size_histogram_.begin(), end =
             stats.size_histogram_.end(); iterator != end; ++iterator) out << (*iterator)[0] << "-" << (*iterator)[1] << ": " << (*iterator)[2] << ",  ";

    return out;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Lsh hash table. As its key is a sub-feature, and as usually
 * the size of it is pretty small, we keep it as a continuous memory array.
 * The value is an index in the corpus of features (we keep it as an unsigned
 * int for pure memory reasons, it could be a size_t)
 */
template<typename ElementType>
class LshTable
{
public:
    /** A container of all the feature indices. Optimized for space
     */
#if USE_UNORDERED_MAP
    typedef std::unordered_map<BucketKey, Bucket> BucketsSpace;
#else
    typedef std::map<BucketKey, Bucket> BucketsSpace;
#endif

    /** A container of all the feature indices. Optimized for speed
     */
    typedef std::vector<Bucket> BucketsSpeed;

    /** Default constructor
     */
    LshTable()
    {
    }

    /** Default constructor
     * Create the mask and allocate the memory
     * @param feature_size is the size of the feature (considered as a ElementType[])
     * @param key_size is the number of bits that are turned on in the feature
     */
    LshTable(unsigned int /*feature_size*/, unsigned int /*key_size*/)
    {
        std::cerr << "LSH is not implemented for that type" << std::endl;
        assert(0);
    }

    /** Add a feature to the table
     * @param value the value to store for that feature
     * @param feature the feature itself
     */
    void add(unsigned int value, const ElementType* feature)
    {
        // Add the value to the corresponding bucket
        BucketKey key = (lsh::BucketKey)getKey(feature);

        switch (speed_level_) {
        case kArray:
            // That means we get the buckets from an array
            buckets_speed_[key].push_back(value);
            break;
        case kBitsetHash:
            // That means we can check the bitset for the presence of a key
            key_bitset_.set(key);
            buckets_space_[key].push_back(value);
            break;
        case kHash:
        {
            // That means we have to check for the hash table for the presence of a key
            buckets_space_[key].push_back(value);
            break;
        }
        }
    }

    /** Add a set of features to the table
     * @param dataset the values to store
     */
    void add(Matrix<ElementType> dataset)
    {
#if USE_UNORDERED_MAP
        buckets_space_.rehash((buckets_space_.size() + dataset.rows) * 1.2);
#endif
        // Add the features to the table
        for (unsigned int i = 0; i < dataset.rows; ++i) add(i, dataset[i]);
        // Now that the table is full, optimize it for speed/space
        optimize();
    }

    /** Get a bucket given the key
     * @param key
     * @return
     */
    inline const Bucket* getBucketFromKey(BucketKey key) const
    {
        // Generate other buckets
        switch (speed_level_) {
        case kArray:
            // That means we get the buckets from an array
            return &buckets_speed_[key];
            break;
        case kBitsetHash:
            // That means we can check the bitset for the presence of a key
            if (key_bitset_.test(key)) return &buckets_space_.find(key)->second;
            else return 0;
            break;
        case kHash:
        {
            // That means we have to check for the hash table for the presence of a key
            BucketsSpace::const_iterator bucket_it, bucket_end = buckets_space_.end();
            bucket_it = buckets_space_.find(key);
            // Stop here if that bucket does not exist
            if (bucket_it == bucket_end) return 0;
            else return &bucket_it->second;
            break;
        }
        }
        return 0;
    }

    /** Compute the sub-signature of a feature
     */
    size_t getKey(const ElementType* /*feature*/) const
    {
        std::cerr << "LSH is not implemented for that type" << std::endl;
        assert(0);
        return 1;
    }

    /** Get statistics about the table
     * @return
     */
    LshStats getStats() const;

private:
    /** defines the speed fo the implementation
     * kArray uses a vector for storing data
     * kBitsetHash uses a hash map but checks for the validity of a key with a bitset
     * kHash uses a hash map only
     */
    enum SpeedLevel
    {
        kArray, kBitsetHash, kHash
    };

    /** Initialize some variables
     */
    void initialize(size_t key_size)
    {
        const size_t key_size_lower_bound = 1;
        //a value (size_t(1) << key_size) must fit the size_t type so key_size has to be strictly less than size of size_t
        const size_t key_size_upper_bound = std::min(sizeof(BucketKey) * CHAR_BIT + 1, sizeof(size_t) * CHAR_BIT);
        if (key_size < key_size_lower_bound || key_size >= key_size_upper_bound)
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Invalid key_size (=%d). Valid values for your system are %d <= key_size < %d.", (int)key_size, (int)key_size_lower_bound, (int)key_size_upper_bound));
        }

        speed_level_ = kHash;
        key_size_ = (unsigned)key_size;
    }

    /** Optimize the table for speed/space
     */
    void optimize()
    {
        // If we are already using the fast storage, no need to do anything
        if (speed_level_ == kArray) return;

        // Use an array if it will be more than half full
        if (buckets_space_.size() > ((size_t(1) << key_size_) / 2)) {
            speed_level_ = kArray;
            // Fill the array version of it
            buckets_speed_.resize(size_t(1) << key_size_);
            for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket) buckets_speed_[key_bucket->first] = key_bucket->second;

            // Empty the hash table
            buckets_space_.clear();
            return;
        }

        // If the bitset is going to use less than 10% of the RAM of the hash map (at least 1 size_t for the key and two
        // for the vector) or less than 512MB (key_size_ <= 30)
        if (((std::max(buckets_space_.size(), buckets_speed_.size()) * CHAR_BIT * 3 * sizeof(BucketKey)) / 10
             >= (size_t(1) << key_size_)) || (key_size_ <= 32)) {
            speed_level_ = kBitsetHash;
            key_bitset_.resize(size_t(1) << key_size_);
            key_bitset_.reset();
            // Try with the BucketsSpace
            for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket) key_bitset_.set(key_bucket->first);
        }
        else {
            speed_level_ = kHash;
            key_bitset_.clear();
        }
    }

    /** The vector of all the buckets if they are held for speed
     */
    BucketsSpeed buckets_speed_;

    /** The hash table of all the buckets in case we cannot use the speed version
     */
    BucketsSpace buckets_space_;

    /** What is used to store the data */
    SpeedLevel speed_level_;

    /** If the subkey is small enough, it will keep track of which subkeys are set through that bitset
     * That is just a speedup so that we don't look in the hash table (which can be mush slower that checking a bitset)
     */
    DynamicBitset key_bitset_;

    /** The size of the sub-signature in bits
     */
    unsigned int key_size_;

    // Members only used for the unsigned char specialization
    /** The mask to apply to a feature to get the hash key
     * Only used in the unsigned char case
     */
    std::vector<size_t> mask_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialization for unsigned char

template<>
inline LshTable<unsigned char>::LshTable(unsigned int feature_size, unsigned int subsignature_size)
{
    initialize(subsignature_size);
    // Allocate the mask
    mask_ = std::vector<size_t>((size_t)ceil((float)(feature_size * sizeof(char)) / (float)sizeof(size_t)), 0);

    // A bit brutal but fast to code
    std::vector<size_t> indices(feature_size * CHAR_BIT);
    for (size_t i = 0; i < feature_size * CHAR_BIT; ++i) indices[i] = i;
    std::random_shuffle(indices.begin(), indices.end());

    // Generate a random set of order of subsignature_size_ bits
    for (unsigned int i = 0; i < key_size_; ++i) {
        size_t index = indices[i];

        // Set that bit in the mask
        size_t divisor = CHAR_BIT * sizeof(size_t);
        size_t idx = index / divisor; //pick the right size_t index
        mask_[idx] |= size_t(1) << (index % divisor); //use modulo to find the bit offset
    }

    // Set to 1 if you want to display the mask for debug
#if 0
    {
        size_t bcount = 0;
        BOOST_FOREACH(size_t mask_block, mask_){
            out << std::setw(sizeof(size_t) * CHAR_BIT / 4) << std::setfill('0') << std::hex << mask_block
                << std::endl;
            bcount += __builtin_popcountll(mask_block);
        }
        out << "bit count : " << std::dec << bcount << std::endl;
        out << "mask size : " << mask_.size() << std::endl;
        return out;
    }
#endif
}

/** Return the Subsignature of a feature
 * @param feature the feature to analyze
 */
template<>
inline size_t LshTable<unsigned char>::getKey(const unsigned char* feature) const
{
    // no need to check if T is dividable by sizeof(size_t) like in the Hamming
    // distance computation as we have a mask
    const size_t* feature_block_ptr = reinterpret_cast<const size_t*> ((const void*)feature);

    // Figure out the subsignature of the feature
    // Given the feature ABCDEF, and the mask 001011, the output will be
    // 000CEF
    size_t subsignature = 0;
    size_t bit_index = 1;

    for (std::vector<size_t>::const_iterator pmask_block = mask_.begin(); pmask_block != mask_.end(); ++pmask_block) {
        // get the mask and signature blocks
        size_t feature_block = *feature_block_ptr;
        size_t mask_block = *pmask_block;
        while (mask_block) {
            // Get the lowest set bit in the mask block
            size_t lowest_bit = mask_block & (-(ptrdiff_t)mask_block);
            // Add it to the current subsignature if necessary
            subsignature += (feature_block & lowest_bit) ? bit_index : 0;
            // Reset the bit in the mask block
            mask_block ^= lowest_bit;
            // increment the bit index for the subsignature
            bit_index <<= 1;
        }
        // Check the next feature block
        ++feature_block_ptr;
    }
    return subsignature;
}

template<>
inline LshStats LshTable<unsigned char>::getStats() const
{
    LshStats stats;
    stats.bucket_size_mean_ = 0;
    if ((buckets_speed_.empty()) && (buckets_space_.empty())) {
        stats.n_buckets_ = 0;
        stats.bucket_size_median_ = 0;
        stats.bucket_size_min_ = 0;
        stats.bucket_size_max_ = 0;
        return stats;
    }

    if (!buckets_speed_.empty()) {
        for (BucketsSpeed::const_iterator pbucket = buckets_speed_.begin(); pbucket != buckets_speed_.end(); ++pbucket) {
            stats.bucket_sizes_.push_back((lsh::FeatureIndex)pbucket->size());
            stats.bucket_size_mean_ += pbucket->size();
        }
        stats.bucket_size_mean_ /= buckets_speed_.size();
        stats.n_buckets_ = buckets_speed_.size();
    }
    else {
        for (BucketsSpace::const_iterator x = buckets_space_.begin(); x != buckets_space_.end(); ++x) {
            stats.bucket_sizes_.push_back((lsh::FeatureIndex)x->second.size());
            stats.bucket_size_mean_ += x->second.size();
        }
        stats.bucket_size_mean_ /= buckets_space_.size();
        stats.n_buckets_ = buckets_space_.size();
    }

    std::sort(stats.bucket_sizes_.begin(), stats.bucket_sizes_.end());

    //  BOOST_FOREACH(int size, stats.bucket_sizes_)
    //          std::cout << size << " ";
    //  std::cout << std::endl;
    stats.bucket_size_median_ = stats.bucket_sizes_[stats.bucket_sizes_.size() / 2];
    stats.bucket_size_min_ = stats.bucket_sizes_.front();
    stats.bucket_size_max_ = stats.bucket_sizes_.back();

    // TODO compute mean and std
    /*float mean, stddev;
       stats.bucket_size_mean_ = mean;
       stats.bucket_size_std_dev = stddev;*/

    // Include a histogram of the buckets
    unsigned int bin_start = 0;
    unsigned int bin_end = 20;
    bool is_new_bin = true;
    for (std::vector<unsigned int>::iterator iterator = stats.bucket_sizes_.begin(), end = stats.bucket_sizes_.end(); iterator
         != end; )
        if (*iterator < bin_end) {
            if (is_new_bin) {
                stats.size_histogram_.push_back(std::vector<unsigned int>(3, 0));
                stats.size_histogram_.back()[0] = bin_start;
                stats.size_histogram_.back()[1] = bin_end - 1;
                is_new_bin = false;
            }
            ++stats.size_histogram_.back()[2];
            ++iterator;
        }
        else {
            bin_start += 20;
            bin_end += 20;
            is_new_bin = true;
        }

    return stats;
}

// End the two namespaces
}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* OPENCV_FLANN_LSH_TABLE_H_ */
