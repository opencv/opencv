/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE NNIndexGOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef _OPENCV_SAVING_H_
#define _OPENCV_SAVING_H_

#include "opencv2/flann/general.h"
#include "opencv2/flann/nn_index.h"
#include <cstdio>
#include <cstring>

namespace cvflann
{
template <typename T> struct Datatype {};
template<> struct Datatype<char> { static flann_datatype_t type() { return INT8; } };
template<> struct Datatype<short> { static flann_datatype_t type() { return INT16; } };
template<> struct Datatype<int> { static flann_datatype_t type() { return INT32; } };
template<> struct Datatype<unsigned char> { static flann_datatype_t type() { return UINT8; } };
template<> struct Datatype<unsigned short> { static flann_datatype_t type() { return UINT16; } };
template<> struct Datatype<unsigned int> { static flann_datatype_t type() { return UINT32; } };
template<> struct Datatype<float> { static flann_datatype_t type() { return FLOAT32; } };
template<> struct Datatype<double> { static flann_datatype_t type() { return FLOAT64; } };


CV_EXPORTS const char* FLANN_SIGNATURE();
CV_EXPORTS const char* FLANN_VERSION();

/**
 * Structure representing the index header.
 */
struct CV_EXPORTS IndexHeader
{
	char signature[16];
	char version[16];
	flann_datatype_t data_type;
	flann_algorithm_t index_type;
	int rows;
	int cols;
};

/**
 * Saves index header to stream
 *
 * @param stream - Stream to save to
 * @param index - The index to save
 */
template<typename ELEM_TYPE>
void save_header(FILE* stream, const NNIndex<ELEM_TYPE>& index)
{
	IndexHeader header;
	memset(header.signature, 0 , sizeof(header.signature));
	strcpy(header.signature, FLANN_SIGNATURE());
	memset(header.version, 0 , sizeof(header.version));
	strcpy(header.version, FLANN_VERSION());
	header.data_type = Datatype<ELEM_TYPE>::type();
	header.index_type = index.getType();
	header.rows = (int)index.size();
	header.cols = index.veclen();

	std::fwrite(&header, sizeof(header),1,stream);
}


/**
 *
 * @param stream - Stream to load from
 * @return Index header
 */
CV_EXPORTS IndexHeader load_header(FILE* stream);


template<typename T>
void save_value(FILE* stream, const T& value, int count = 1)
{
	fwrite(&value, sizeof(value),count, stream);
}


template<typename T>
void load_value(FILE* stream, T& value, int count = 1)
{
	int read_cnt = fread(&value, sizeof(value),count, stream);
	if (read_cnt!=count) {
		throw FLANNException("Cannot read from file");
	}
}

} // namespace cvflann

#endif /* _OPENCV_SAVING_H_ */
