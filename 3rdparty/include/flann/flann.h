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

#ifndef FLANN_H
#define FLANN_H


#include "constants.h"


#ifdef WIN32
/* win32 dll export/import directives */
#ifdef flann_EXPORTS
#define LIBSPEC __declspec(dllexport)
#else
#define LIBSPEC __declspec(dllimport)
#endif
#else
/* unix needs nothing */
#define LIBSPEC
#endif





struct FLANNParameters {
	flann_algorithm_t algorithm; // the algorithm to use (see constants.h)

	int checks;                // how many leafs (features) to check in one search
    float cb_index;            // cluster boundary index. Used when searching the kmeans tree
	int trees;                 // number of randomized trees to use (for kdtree)
	int branching;             // branching factor (for kmeans tree)
	int iterations;            // max iterations to perform in one kmeans cluetering (kmeans tree)
	flann_centers_init_t centers_init;          // algorithm used for picking the initial cluetr centers for kmeans tree
	float target_precision;    // precision desired (used for autotuning, -1 otherwise)
	float build_weight;        // build tree time weighting factor
	float memory_weight;       // index memory weigthing factor
    float sample_fraction;     // what fraction of the dataset to use for autotuning

    flann_log_level_t log_level;             // determines the verbosity of each flann function
	char* log_destination;     // file where the output should go, NULL for the console
	long random_seed;          // random seed to use
};



typedef void* FLANN_INDEX; // deprecated
typedef void* flann_index_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
Sets the log level used for all flann functions (unless
specified in FLANNParameters for each call

Params:
    level = verbosity level (defined in constants.h)
*/
LIBSPEC void flann_log_verbosity(int level);


/**
 * Sets the distance type to use throughout FLANN.
 * If distance type specified is MINKOWSKI, the second argument
 * specifies which order the minkowski distance should have.
 */
LIBSPEC void flann_set_distance_type(flann_distance_t distance_type, int order);


/**
Builds and returns an index. It uses autotuning if the target_precision field of index_params
is between 0 and 1, or the parameters specified if it's -1.

Params:
    dataset = pointer to a data set stored in row major order
    rows = number of rows (features) in the dataset
    cols = number of columns in the dataset (feature dimensionality)
    speedup = speedup over linear search, estimated if using autotuning, output parameter
    index_params = index related parameters
    flann_params = generic flann parameters

Returns: the newly created index or a number <0 for error
*/
LIBSPEC FLANN_INDEX flann_build_index(float* dataset,
									  int rows,
									  int cols,
									  float* speedup,
									  struct FLANNParameters* flann_params);





/**
 * Saves the index to a file. Only the index is saved into the file, the dataset corresponding to the index is not saved.
 *
 * @param index_id The index that should be saved
 * @param filename The filename the index should be saved to
 * @return Returns 0 on success, negative value on error.
 */
LIBSPEC int flann_save_index(FLANN_INDEX index_id,
							 char* filename);


/**
 * Loads an index from a file.
 *
 * @param filename File to load the index from.
 * @param dataset The dataset corresponding to the index.
 * @param rows Dataset tors
 * @param cols Dataset columns
 * @return
 */
LIBSPEC FLANN_INDEX flann_load_index(char* filename,
									 float* dataset,
									 int rows,
									 int cols);



/**
Builds an index and uses it to find nearest neighbors.

Params:
    dataset = pointer to a data set stored in row major order
    rows = number of rows (features) in the dataset
    cols = number of columns in the dataset (feature dimensionality)
    testset = pointer to a query set stored in row major order
    trows = number of rows (features) in the query dataset (same dimensionality as features in the dataset)
    indices = pointer to matrix for the indices of the nearest neighbors of the testset features in the dataset
            (must have trows number of rows and nn number of columns)
    nn = how many nearest neighbors to return
    index_params = index related parameters
    flann_params = generic flann parameters

Returns: zero or -1 for error
*/
LIBSPEC int flann_find_nearest_neighbors(float* dataset,
										 int rows,
										 int cols,
										 float* testset,
										 int trows,
										 int* indices,
										 float* dists,
										 int nn,
										 struct FLANNParameters* flann_params);



/**
Searches for nearest neighbors using the index provided

Params:
    index_id = the index (constructed previously using flann_build_index).
    testset = pointer to a query set stored in row major order
    trows = number of rows (features) in the query dataset (same dimensionality as features in the dataset)
    indices = pointer to matrix for the indices of the nearest neighbors of the testset features in the dataset
            (must have trows number of rows and nn number of columns)
    nn = how many nearest neighbors to return
    checks = number of checks to perform before the search is stopped
    flann_params = generic flann parameters

Returns: zero or a number <0 for error
*/
LIBSPEC int flann_find_nearest_neighbors_index(FLANN_INDEX index_id,
											   float* testset,
											   int trows,
											   int* indices,
											   float* dists,
											   int nn,
											   int checks,
											   struct FLANNParameters* flann_params);



/**
 * Performs an radius search using an already constructed index.
 *
 * In case of radius search, instead of always returning a predetermined
 * number of nearest neighbours (for example the 10 nearest neighbours), the
 * search will return all the neighbours found within a search radius
 * of the query point.
 *
 * The check parameter in the function below sets the level of approximation
 * for the search by only visiting "checks" number of features in the index
 * (the same way as for the KNN search). A lower value for checks will give
 * a higher search speedup at the cost of potentially not returning all the
 * neighbours in the specified radius.
 */
LIBSPEC int flann_radius_search(FLANN_INDEX index_ptr, /* the index */
										float* query,	/* query point */
										int* indices, /* array for storing the indices found (will be modified) */
										float* dists, /* similar, but for storing distances */
										int max_nn,  /* size of arrays indices and dists */
										float radius, /* search radius (squared radius for euclidian metric) */
										int checks,  /* number of features to check, sets the level of approximation */
										FLANNParameters* flann_params);


/**
Deletes an index and releases the memory used by it.

Params:
    index_id = the index (constructed previously using flann_build_index).
    flann_params = generic flann parameters

Returns: zero or a number <0 for error
*/
LIBSPEC int flann_free_index(FLANN_INDEX index_id,
		                     struct FLANNParameters* flann_params);

/**
Clusters the features in the dataset using a hierarchical kmeans clustering approach.
This is significantly faster than using a flat kmeans clustering for a large number
of clusters.

Params:
    dataset = pointer to a data set stored in row major order
    rows = number of rows (features) in the dataset
    cols = number of columns in the dataset (feature dimensionality)
    clusters = number of cluster to compute
    result = memory buffer where the output cluster centers are storred
    index_params = used to specify the kmeans tree parameters (branching factor, max number of iterations to use)
    flann_params = generic flann parameters

Returns: number of clusters computed or a number <0 for error. This number can be different than the number of clusters requested, due to the
    way hierarchical clusters are computed. The number of clusters returned will be the highest number of the form
    (branch_size-1)*K+1 smaller than the number of clusters requested.
*/

LIBSPEC int flann_compute_cluster_centers(float* dataset,
										  int rows,
										  int cols,
										  int clusters,
										  float* result,
										  struct FLANNParameters* flann_params);


#ifdef __cplusplus
};


#include "flann.hpp"

#endif


#endif /*FLANN_H*/
