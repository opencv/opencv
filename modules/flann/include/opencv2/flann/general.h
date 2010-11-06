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

#ifndef _OPENCV_GENERAL_H_
#define _OPENCV_GENERAL_H_

#ifdef __cplusplus

#include <stdexcept>
#include <cassert>
#include "opencv2/flann/object_factory.h"

namespace cvflann {

#undef ARRAY_LEN
#define ARRAY_LEN(a) (sizeof(a)/sizeof(a[0]))

/* Nearest neighbour index algorithms */
enum flann_algorithm_t {
	LINEAR = 0,
	KDTREE = 1,
	KMEANS = 2,
	COMPOSITE = 3,
	SAVED = 254,
	AUTOTUNED = 255
};

enum flann_centers_init_t {
	CENTERS_RANDOM = 0,
	CENTERS_GONZALES = 1,
	CENTERS_KMEANSPP = 2
};

enum flann_log_level_t {
	LOG_NONE = 0,
	LOG_FATAL = 1,
	LOG_ERROR = 2,
	LOG_WARN = 3,
	LOG_INFO = 4
};

enum flann_distance_t {
	EUCLIDEAN = 1,
	MANHATTAN = 2,
	MINKOWSKI = 3,
	MAX_DIST   = 4,
	HIK       = 5,
	HELLINGER = 6,
	CS        = 7,
	CHI_SQUARE = 7,
	KL        = 8,
	KULLBACK_LEIBLER        = 8
};

enum flann_datatype_t {
	INT8 = 0,
	INT16 = 1,
	INT32 = 2,
	INT64 = 3,
	UINT8 = 4,
	UINT16 = 5,
	UINT32 = 6,
	UINT64 = 7,
	FLOAT32 = 8,
	FLOAT64 = 9
};

template <typename ELEM_TYPE>
struct DistType
{
	typedef ELEM_TYPE type;
};

template <>
struct DistType<unsigned char>
{
	typedef float type;
};

template <>
struct DistType<int>
{
	typedef float type;
};


class FLANNException : public std::runtime_error {
 public:
   FLANNException(const char* message) : std::runtime_error(message) { }

   FLANNException(const std::string& message) : std::runtime_error(message) { }
 };


struct CV_EXPORTS IndexParams {
protected:
	IndexParams(flann_algorithm_t algorithm_) : algorithm(algorithm_) {};
	
public:
	virtual ~IndexParams() {}
	virtual flann_algorithm_t getIndexType() const = 0;

	virtual void print() const = 0;

	flann_algorithm_t algorithm;
};


typedef ObjectFactory<IndexParams, flann_algorithm_t> ParamsFactory;
CV_EXPORTS ParamsFactory& ParamsFactory_instance();

struct CV_EXPORTS SearchParams {
	SearchParams(int checks_ = 32) :
		checks(checks_) {};

	int checks;
};

} // namespace cvflann

#endif

#endif  /* _OPENCV_GENERAL_H_ */
