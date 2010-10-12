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

#ifndef NNINDEX_H
#define NNINDEX_H

#include <string>

#include "opencv2/flann/general.h"
#include "opencv2/flann/matrix.h"

using namespace std;

namespace cvflann
{


template <typename ELEM_TYPE>
class ResultSet;

/**
* Nearest-neighbour index base class
*/
template <typename ELEM_TYPE>
class NNIndex
{
public:

	virtual ~NNIndex() {};

	/**
	Method responsible with building the index.
	*/
	virtual void buildIndex() = 0;

	/**
	Saves the index to a stream
	*/
	virtual void saveIndex(FILE* stream) = 0;

	/**
	Loads the index from a stream
	*/
	virtual void loadIndex(FILE* stream) = 0;

	/**
	Method that searches for nearest-neighbors
	*/
	virtual void findNeighbors(ResultSet<ELEM_TYPE>& result, const ELEM_TYPE* vec, const SearchParams& searchParams) = 0;

	/**
	Number of features in this index.
	*/
	virtual size_t size() const = 0;

	/**
	The length of each vector in this index.
	*/
	virtual size_t veclen() const = 0;

	/**
	The amount of memory (in bytes) this index uses.
	*/
	virtual int usedMemory() const = 0;

	/**
	* Algorithm name
	*/
	virtual flann_algorithm_t getType() const = 0;

	/**
	 * Returns the parameters used for the index
	 */
	virtual const IndexParams* getParameters() const = 0;

};


} // namespace cvflann

#endif //NNINDEX_H
