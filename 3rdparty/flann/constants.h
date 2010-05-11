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

#ifndef CONSTANTS_H
#define CONSTANTS_H


const double FLANN_VERSION = 1.20;

/* Nearest neighbor index algorithms */
enum flann_algorithm_t {
	LINEAR = 0,
	KDTREE = 1,
	KMEANS = 2,
	COMPOSITE = 3,
	SAVED = 254,
	AUTOTUNED = 255,
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
	MINKOWSKI = 3
};

#endif  // CONSTANTS_H
