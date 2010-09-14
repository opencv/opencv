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

#ifndef RANDOM_H
#define RANDOM_H

#include <algorithm>
#include <cstdlib>
#include <cassert>

using namespace std;

namespace cvflann
{

/**
 * Seeds the random number generator
 */
void seed_random(unsigned int seed);

/*
 * Generates a random double value.
 */
double rand_double(double high = 1.0, double low=0);

/*
 * Generates a random integer value.
 */
int rand_int(int high = RAND_MAX, int low = 0);


/**
 * Random number generator that returns a distinct number from
 * the [0,n) interval each time.
 *
 * TODO: improve on this to use a generator function instead of an
 * array of randomly permuted numbers
 */
class UniqueRandom
{
	int* vals;
    int size;
	int counter;

public:
	/**
	 * Constructor.
	 * Params:
	 *     n = the size of the interval from which to generate
	 *     		random numbers.
	 */
	UniqueRandom(int n) : vals(NULL) {
		init(n);
	}

	~UniqueRandom()
	{
		delete[] vals;
	}

	/**
	 * Initializes the number generator.
	 * Params:
	 * 		n = the size of the interval from which to generate
	 *     		random numbers.
	 */
	void init(int n)
	{
    	// create and initialize an array of size n
		if (vals == NULL || n!=size) {
            delete[] vals;
	        size = n;
            vals = new int[size];
    	}
    	for(int i=0;i<size;++i) {
			vals[i] = i;
		}

		// shuffle the elements in the array
        // Fisher-Yates shuffle
		for (int i=size;i>0;--i) {
// 			int rand = cast(int) (drand48() * n);
			int rnd = rand_int(i);
			assert(rnd >=0 && rnd < i);
			swap(vals[i-1], vals[rnd]);
		}

		counter = 0;
	}

	/**
	 * Return a distinct random integer in greater or equal to 0 and less
	 * than 'n' on each call. It should be called maximum 'n' times.
	 * Returns: a random integer
	 */
	int next() {
		if (counter==size) {
			return -1;
		} else {
			return vals[counter++];
		}
	}
};

}

#endif //RANDOM_H
