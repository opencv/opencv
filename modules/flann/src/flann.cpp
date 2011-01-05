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
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include "precomp.hpp"

namespace cvflann
{
// ----------------------- dist.cpp ---------------------------

/** Global variable indicating the distance metric
 * to be used.
 */
flann_distance_t flann_distance_type_ = EUCLIDEAN;
flann_distance_t flann_distance_type() { return flann_distance_type_; }

/**
 * Zero iterator that emulates a zero feature.
 */
ZeroIterator<float> zero_;
ZeroIterator<float>& zero() { return zero_; }

/**
 * Order of Minkowski distance to use.
 */
int flann_minkowski_order_;
int flann_minkowski_order() { return flann_minkowski_order_; }


double euclidean_dist(const unsigned char* first1, const unsigned char* last1, unsigned char* first2, double acc)
{
	double distsq = acc;
	double diff0, diff1, diff2, diff3;
	const unsigned char* lastgroup = last1 - 3;

	while (first1 < lastgroup) {
		diff0 = first1[0] - first2[0];
		diff1 = first1[1] - first2[1];
		diff2 = first1[2] - first2[2];
		diff3 = first1[3] - first2[3];
		distsq += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
		first1 += 4;
		first2 += 4;
	}
	while (first1 < last1) {
		diff0 = *first1++ - *first2++;
		distsq += diff0 * diff0;
	}
	return distsq;
}

// ----------------------- index_testing.cpp ---------------------------

int countCorrectMatches(int* neighbors, int* groundTruth, int n)
{
    int count = 0;
    for (int i=0;i<n;++i) {
        for (int k=0;k<n;++k) {
            if (neighbors[i]==groundTruth[k]) {
                count++;
                break;
            }
        }
    }
    return count;
}

// ----------------------- logger().cpp ---------------------------

Logger logger_;

Logger& logger() { return logger_; }

int Logger::log(int level, const char* fmt, ...)
{
    if (level > logLevel ) return -1;

    int ret;
    va_list arglist;
    va_start(arglist, fmt);
    ret = vfprintf(stream, fmt, arglist);
    va_end(arglist);

    return ret;
}

int Logger::log(int level, const char* fmt, va_list arglist)
{
    if (level > logLevel ) return -1;

    int ret;
    ret = vfprintf(stream, fmt, arglist);

    return ret;
}


#define LOG_METHOD(NAME,LEVEL) \
    int Logger::NAME(const char* fmt, ...) \
    { \
        int ret; \
        va_list ap; \
        va_start(ap, fmt); \
        ret = log(LEVEL, fmt, ap); \
        va_end(ap); \
        return ret; \
    }


LOG_METHOD(fatal, FLANN_LOG_FATAL)
LOG_METHOD(error, FLANN_LOG_ERROR)
LOG_METHOD(warn, FLANN_LOG_WARN)
LOG_METHOD(info, FLANN_LOG_INFO)

// ----------------------- random.cpp ---------------------------

void seed_random(unsigned int seed)
{
    srand(seed);
}

double rand_double(double high, double low)
{
    return low + ((high-low) * (std::rand() / (RAND_MAX + 1.0)));
}


int rand_int(int high, int low)
{
    return low + (int) ( double(high-low) * (std::rand() / (RAND_MAX + 1.0)));
}

// ----------------------- saving.cpp ---------------------------

const char FLANN_SIGNATURE_[] = "FLANN_INDEX";
const char FLANN_VERSION_[] = "1.5.0";

const char* FLANN_SIGNATURE() { return FLANN_SIGNATURE_; }
const char* FLANN_VERSION() { return FLANN_VERSION_; }

IndexHeader load_header(FILE* stream)
{
	IndexHeader header;
	size_t read_size = fread(&header,sizeof(header),1,stream);

	if (read_size!=1) {
		throw FLANNException("Invalid index file, cannot read");
	}

	if (strcmp(header.signature,FLANN_SIGNATURE())!=0) {
		throw FLANNException("Invalid index file, wrong signature");
	}

	return header;

}

// ----------------------- flann.cpp ---------------------------


void log_verbosity(int level)
{
    if (level>=0) {
        logger().setLevel(level);
    }
}

void set_distance_type(flann_distance_t distance_type, int order)
{
	flann_distance_type_ = distance_type;
	flann_minkowski_order_ = order;
}


static ParamsFactory the_factory;

ParamsFactory& ParamsFactory_instance()
{
    return the_factory;
}

class StaticInit
{
public:
	StaticInit()
	{
		ParamsFactory_instance().register_<LinearIndexParams>(LINEAR);
		ParamsFactory_instance().register_<KDTreeIndexParams>(KDTREE);
		ParamsFactory_instance().register_<KMeansIndexParams>(KMEANS);
		ParamsFactory_instance().register_<CompositeIndexParams>(COMPOSITE);
		ParamsFactory_instance().register_<AutotunedIndexParams>(AUTOTUNED);
//		ParamsFactory::instance().register_<SavedIndexParams>(SAVED);
	}
};
StaticInit __init;


} // namespace cvflann



