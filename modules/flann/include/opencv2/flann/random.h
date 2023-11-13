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

#ifndef OPENCV_FLANN_RANDOM_H
#define OPENCV_FLANN_RANDOM_H

//! @cond IGNORED

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace cvflann
{

inline int rand()
{
#ifndef OPENCV_FLANN_USE_STD_RAND
#   if INT_MAX == RAND_MAX
    int v = cv::theRNG().next() & INT_MAX;
#   else
    int v = cv::theRNG().uniform(0, RAND_MAX + 1);
#   endif
#else
    int v = std::rand();
#endif // OPENCV_FLANN_USE_STD_RAND
    return v;
}

/**
 * Seeds the random number generator
 *  @param seed Random seed
 */
inline void seed_random(unsigned int seed)
{
#ifndef OPENCV_FLANN_USE_STD_RAND
    cv::theRNG() = cv::RNG(seed);
#else
    std::srand(seed);
#endif
}

/*
 * Generates a random double value.
 */
/**
 * Generates a random double value.
 * @param high Upper limit
 * @param low Lower limit
 * @return Random double value
 */
inline double rand_double(double high = 1.0, double low = 0)
{
    return low + ((high-low) * (rand() / (RAND_MAX + 1.0)));
}

/**
 * Generates a random integer value.
 * @param high Upper limit
 * @param low Lower limit
 * @return Random integer value
 */
inline int rand_int(int high = RAND_MAX, int low = 0)
{
    return low + (int) ( double(high-low) * (rand() / (RAND_MAX + 1.0)));
}

/**
 * Random number generator that returns a distinct number from
 * the [0,n) interval each time.
 */
class UniqueRandom
{
    std::vector<int> vals_;
    int size_;
    int counter_;

public:
    /**
     * Constructor.
     * @param n Size of the interval from which to generate
     */
    UniqueRandom(int n)
    {
        init(n);
    }

    /**
     * Initializes the number generator.
     * @param n the size of the interval from which to generate random numbers.
     */
    void init(int n)
    {
        // create and initialize an array of size n
        vals_.resize(n);
        size_ = n;
        for (int i = 0; i < size_; ++i) vals_[i] = i;

        // shuffle the elements in the array
#ifndef OPENCV_FLANN_USE_STD_RAND
        cv::randShuffle(vals_);
#else
        std::random_shuffle(vals_.begin(), vals_.end());
#endif

        counter_ = 0;
    }

    /**
     * Return a distinct random integer in greater or equal to 0 and less
     * than 'n' on each call. It should be called maximum 'n' times.
     * Returns: a random integer
     */
    int next()
    {
        if (counter_ == size_) {
            return -1;
        }
        else {
            return vals_[counter_++];
        }
    }
};

}

//! @endcond

#endif //OPENCV_FLANN_RANDOM_H
