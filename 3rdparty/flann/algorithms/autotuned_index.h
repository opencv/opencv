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

#ifndef AUTOTUNEDINDEX_H_
#define AUTOTUNEDINDEX_H_

#include "constants.h"
#include "nn_index.h"
#include "ground_truth.h"
#include "index_testing.h"

namespace cvflann
{

class AutotunedIndex : public NNIndex
{
	NNIndex* bestIndex;
	IndexParams* bestParams;
	SearchParams bestSearchParams;

    Matrix<float>* sampledDataset;
    Matrix<float>* testDataset;
    Matrix<int>* gt_matches;

    float speedup;

	/**
	 * The dataset used by this index
	 */
    const Matrix<float> dataset;

    /**
     * Index parameters
     */
    const AutotunedIndexParams& params;

    /**
    * Number of features in the dataset.
    */
    int size_;

    /**
    * Length of each feature.
    */
    int veclen_;


public:

    AutotunedIndex(const Matrix<float>& inputData, const AutotunedIndexParams& params_ = AutotunedIndexParams() ) :
    	dataset(inputData), params(params_)
	{
        size_ = dataset.rows;
        veclen_ = dataset.cols;

        bestIndex = NULL;

	}

    virtual ~AutotunedIndex()
    {
    	delete bestIndex;
    	delete bestParams;
    };

	/**
		Method responsible with building the index.
	*/
	virtual void buildIndex()
	{
		bestParams = estimateBuildParams();
		bestIndex = bestParams->createIndex(dataset);
		bestIndex->buildIndex();
		speedup = estimateSearchParams(bestSearchParams);
	}

    /**
        Saves the index to a stream
    */
    virtual void saveIndex(FILE* stream)
    {
    	bestIndex->saveIndex(stream);
    }

    /**
        Loads the index from a stream
    */
    virtual void loadIndex(FILE* stream)
    {
    	bestIndex->loadIndex(stream);
    }

	/**
		Method that searches for nearest-neighbors
	*/
	virtual void findNeighbors(ResultSet& result, const float* vec, const SearchParams& /*searchParams*/)
	{
		bestIndex->findNeighbors(result, vec, bestSearchParams);
	}

	/**
		Number of features in this index.
	*/
	virtual int size() const
	{
		return bestIndex->size();
	}

	/**
		The length of each vector in this index.
	*/
	virtual int veclen() const
	{
		return bestIndex->veclen();
	}

	/**
	 The amount of memory (in bytes) this index uses.
	*/
 	virtual int usedMemory() const
 	{
 		return bestIndex->usedMemory();
 	}

    /**
    * Algorithm name
    */
    virtual flann_algorithm_t getType() const
    {
    	return bestIndex->getType();
    }

    /**
      Estimates the search parameters required in order to get a certain precision.
      If testset is not given it uses cross-validation.
    */
//    virtual Params estimateSearchParams(float precision, Dataset<float>* testset = NULL)
//    {
//        Params params;
//
//        return params;
//    }
private:

    struct CostData {
        float searchTimeCost;
        float buildTimeCost;
        float timeCost;
        float memoryCost;
        float totalCost;
    };

    typedef pair<CostData, KDTreeIndexParams> KDTreeCostData;
    typedef pair<CostData, KMeansIndexParams> KMeansCostData;


    void evaluate_kmeans(CostData& cost, const KMeansIndexParams& kmeans_params)
    {
        StartStopTimer t;
        int checks;
        const int nn = 1;

        logger.info("KMeansTree using params: max_iterations=%d, branching=%d\n", kmeans_params.iterations, kmeans_params.branching);
        KMeansIndex kmeans(*sampledDataset, kmeans_params);
        // measure index build time
        t.start();
        kmeans.buildIndex();
        t.stop();
        float buildTime = (float)t.value;

        // measure search time
        float searchTime = test_index_precision(kmeans, *sampledDataset, *testDataset, *gt_matches, params.target_precision, checks, nn);;

        float datasetMemory = (float)(sampledDataset->rows*sampledDataset->cols*sizeof(float));
        cost.memoryCost = (kmeans.usedMemory()+datasetMemory)/datasetMemory;
        cost.searchTimeCost = searchTime;
        cost.buildTimeCost = buildTime;
        cost.timeCost = (buildTime*params.build_weight+searchTime);
        logger.info("KMeansTree buildTime=%g, searchTime=%g, timeCost=%g, buildTimeFactor=%g\n",buildTime, searchTime, cost.timeCost, params.build_weight);
    }


     void evaluate_kdtree(CostData& cost, const KDTreeIndexParams& kdtree_params)
    {
        StartStopTimer t;
        int checks;
        const int nn = 1;

        logger.info("KDTree using params: trees=%d\n",kdtree_params.trees);
        KDTreeIndex kdtree(*sampledDataset, kdtree_params);

        t.start();
        kdtree.buildIndex();
        t.stop();
        float buildTime = (float)t.value;

        //measure search time
        float searchTime = test_index_precision(kdtree, *sampledDataset, *testDataset, *gt_matches, params.target_precision, checks, nn);

        float datasetMemory = (float)(sampledDataset->rows*sampledDataset->cols*sizeof(float));
        cost.memoryCost = (kdtree.usedMemory()+datasetMemory)/datasetMemory;
        cost.searchTimeCost = searchTime;
        cost.buildTimeCost = buildTime;
        cost.timeCost = (buildTime*params.build_weight+searchTime);
        logger.info("KDTree buildTime=%g, searchTime=%g, timeCost=%g\n",buildTime, searchTime, cost.timeCost);
    }


//    struct KMeansSimpleDownhillFunctor {
//
//        Autotune& autotuner;
//        KMeansSimpleDownhillFunctor(Autotune& autotuner_) : autotuner(autotuner_) {};
//
//        float operator()(int* params) {
//
//            float maxFloat = numeric_limits<float>::max();
//
//            if (params[0]<2) return maxFloat;
//            if (params[1]<0) return maxFloat;
//
//            CostData c;
//            c.params["algorithm"] = KMEANS;
//            c.params["centers-init"] = CENTERS_RANDOM;
//            c.params["branching"] = params[0];
//            c.params["max-iterations"] = params[1];
//
//            autotuner.evaluate_kmeans(c);
//
//            return c.timeCost;
//
//        }
//    };
//
//    struct KDTreeSimpleDownhillFunctor {
//
//        Autotune& autotuner;
//        KDTreeSimpleDownhillFunctor(Autotune& autotuner_) : autotuner(autotuner_) {};
//
//        float operator()(int* params) {
//            float maxFloat = numeric_limits<float>::max();
//
//            if (params[0]<1) return maxFloat;
//
//            CostData c;
//            c.params["algorithm"] = KDTREE;
//            c.params["trees"] = params[0];
//
//            autotuner.evaluate_kdtree(c);
//
//            return c.timeCost;
//
//        }
//    };



    KMeansCostData optimizeKMeans()
    {
        logger.info("KMEANS, Step 1: Exploring parameter space\n");

        // explore kmeans parameters space using combinations of the parameters below
        int maxIterations[] = { 1, 5, 10, 15 };
        int branchingFactors[] = { 16, 32, 64, 128, 256 };

        int kmeansParamSpaceSize = ARRAY_LEN(maxIterations)*ARRAY_LEN(branchingFactors);

        vector<KMeansCostData> kmeansCosts(kmeansParamSpaceSize);

//        CostData* kmeansCosts = new CostData[kmeansParamSpaceSize];

        // evaluate kmeans for all parameter combinations
        int cnt = 0;
        for (size_t i=0; i<ARRAY_LEN(maxIterations); ++i) {
            for (size_t j=0; j<ARRAY_LEN(branchingFactors); ++j) {

            	kmeansCosts[cnt].second.centers_init = CENTERS_RANDOM;
            	kmeansCosts[cnt].second.branching = branchingFactors[j];
            	kmeansCosts[cnt].second.iterations = maxIterations[j];

                evaluate_kmeans(kmeansCosts[cnt].first, kmeansCosts[cnt].second);

                int k = cnt;
                // order by time cost
                while (k>0 && kmeansCosts[k].first.timeCost < kmeansCosts[k-1].first.timeCost) {
                    swap(kmeansCosts[k],kmeansCosts[k-1]);
                    --k;
                }
                ++cnt;
            }
        }

//         logger.info("KMEANS, Step 2: simplex-downhill optimization\n");
//
//         const int n = 2;
//         // choose initial simplex points as the best parameters so far
//         int kmeansNMPoints[n*(n+1)];
//         float kmeansVals[n+1];
//         for (int i=0;i<n+1;++i) {
//             kmeansNMPoints[i*n] = (int)kmeansCosts[i].params["branching"];
//             kmeansNMPoints[i*n+1] = (int)kmeansCosts[i].params["max-iterations"];
//             kmeansVals[i] = kmeansCosts[i].timeCost;
//         }
//         KMeansSimpleDownhillFunctor kmeans_cost_func(*this);
//         // run optimization
//         optimizeSimplexDownhill(kmeansNMPoints,n,kmeans_cost_func,kmeansVals);
//         // store results
//         for (int i=0;i<n+1;++i) {
//             kmeansCosts[i].params["branching"] = kmeansNMPoints[i*2];
//             kmeansCosts[i].params["max-iterations"] = kmeansNMPoints[i*2+1];
//             kmeansCosts[i].timeCost = kmeansVals[i];
//         }

        float optTimeCost = kmeansCosts[0].first.timeCost;
        // recompute total costs factoring in the memory costs
        for (int i=0;i<kmeansParamSpaceSize;++i) {
            kmeansCosts[i].first.totalCost = (kmeansCosts[i].first.timeCost/optTimeCost + params.memory_weight * kmeansCosts[i].first.memoryCost);

            int k = i;
            while (k>0 && kmeansCosts[k].first.totalCost < kmeansCosts[k-1].first.totalCost) {
                swap(kmeansCosts[k],kmeansCosts[k-1]);
                k--;
            }
        }
        // display the costs obtained
        for (int i=0;i<kmeansParamSpaceSize;++i) {
            logger.info("KMeans, branching=%d, iterations=%d, time_cost=%g[%g] (build=%g, search=%g), memory_cost=%g, cost=%g\n",
                kmeansCosts[i].second.branching, kmeansCosts[i].second.iterations,
            kmeansCosts[i].first.timeCost,kmeansCosts[i].first.timeCost/optTimeCost,
            kmeansCosts[i].first.buildTimeCost, kmeansCosts[i].first.searchTimeCost,
            kmeansCosts[i].first.memoryCost,kmeansCosts[i].first.totalCost);
        }

        return kmeansCosts[0];
    }


    KDTreeCostData optimizeKDTree()
    {

        logger.info("KD-TREE, Step 1: Exploring parameter space\n");

        // explore kd-tree parameters space using the parameters below
        int testTrees[] = { 1, 4, 8, 16, 32 };

        size_t kdtreeParamSpaceSize = ARRAY_LEN(testTrees);
        vector<KDTreeCostData> kdtreeCosts(kdtreeParamSpaceSize);

        // evaluate kdtree for all parameter combinations
        int cnt = 0;
        for (size_t i=0; i<ARRAY_LEN(testTrees); ++i) {
        	kdtreeCosts[cnt].second.trees = testTrees[i];

            evaluate_kdtree(kdtreeCosts[cnt].first, kdtreeCosts[cnt].second);

            int k = cnt;
            // order by time cost
            while (k>0 && kdtreeCosts[k].first.timeCost < kdtreeCosts[k-1].first.timeCost) {
                swap(kdtreeCosts[k],kdtreeCosts[k-1]);
                --k;
            }
            ++cnt;
        }

//         logger.info("KD-TREE, Step 2: simplex-downhill optimization\n");
//
//         const int n = 1;
//         // choose initial simplex points as the best parameters so far
//         int kdtreeNMPoints[n*(n+1)];
//         float kdtreeVals[n+1];
//         for (int i=0;i<n+1;++i) {
//             kdtreeNMPoints[i] = (int)kdtreeCosts[i].params["trees"];
//             kdtreeVals[i] = kdtreeCosts[i].timeCost;
//         }
//         KDTreeSimpleDownhillFunctor kdtree_cost_func(*this);
//         // run optimization
//         optimizeSimplexDownhill(kdtreeNMPoints,n,kdtree_cost_func,kdtreeVals);
//         // store results
//         for (int i=0;i<n+1;++i) {
//             kdtreeCosts[i].params["trees"] = kdtreeNMPoints[i];
//             kdtreeCosts[i].timeCost = kdtreeVals[i];
//         }

        float optTimeCost = kdtreeCosts[0].first.timeCost;
        // recompute costs for kd-tree factoring in memory cost
        for (size_t i=0;i<kdtreeParamSpaceSize;++i) {
            kdtreeCosts[i].first.totalCost = (kdtreeCosts[i].first.timeCost/optTimeCost + params.memory_weight * kdtreeCosts[i].first.memoryCost);

            int k = i;
            while (k>0 && kdtreeCosts[k].first.totalCost < kdtreeCosts[k-1].first.totalCost) {
                swap(kdtreeCosts[k],kdtreeCosts[k-1]);
                k--;
            }
        }
        // display costs obtained
        for (size_t i=0;i<kdtreeParamSpaceSize;++i) {
            logger.info("kd-tree, trees=%d, time_cost=%g[%g] (build=%g, search=%g), memory_cost=%g, cost=%g\n",
            kdtreeCosts[i].second.trees,kdtreeCosts[i].first.timeCost,kdtreeCosts[i].first.timeCost/optTimeCost,
            kdtreeCosts[i].first.buildTimeCost, kdtreeCosts[i].first.searchTimeCost,
            kdtreeCosts[i].first.memoryCost,kdtreeCosts[i].first.totalCost);
        }

        return kdtreeCosts[0];
    }

    /**
        Chooses the best nearest-neighbor algorithm and estimates the optimal
        parameters to use when building the index (for a given precision).
        Returns a dictionary with the optimal parameters.
    */
    IndexParams* estimateBuildParams()
    {
        int sampleSize = int(params.sample_fraction*dataset.rows);
        int testSampleSize = min(sampleSize/10, 1000);

        logger.info("Entering autotuning, dataset size: %d, sampleSize: %d, testSampleSize: %d\n",dataset.rows, sampleSize, testSampleSize);

        // For a very small dataset, it makes no sense to build any fancy index, just
        // use linear search
        if (testSampleSize<10) {
            logger.info("Choosing linear, dataset too small\n");
            return new LinearIndexParams();
        }

        // We use a fraction of the original dataset to speedup the autotune algorithm
        sampledDataset = dataset.sample(sampleSize);
        // We use a cross-validation approach, first we sample a testset from the dataset
        testDataset = sampledDataset->sample(testSampleSize,true);

        // We compute the ground truth using linear search
        logger.info("Computing ground truth... \n");
        gt_matches = new Matrix<int>(testDataset->rows, 1);
        StartStopTimer t;
        t.start();
        compute_ground_truth(*sampledDataset, *testDataset, *gt_matches, 0);
        t.stop();
        float bestCost = (float)t.value;
        IndexParams* bestParams = new LinearIndexParams();

        // Start parameter autotune process
        logger.info("Autotuning parameters...\n");


        KMeansCostData kmeansCost = optimizeKMeans();

        if (kmeansCost.first.totalCost<bestCost) {
            bestParams = new KMeansIndexParams(kmeansCost.second);
            bestCost = kmeansCost.first.totalCost;
        }

        KDTreeCostData kdtreeCost = optimizeKDTree();

        if (kdtreeCost.first.totalCost<bestCost) {
            bestParams = new KDTreeIndexParams(kdtreeCost.second);
            bestCost = kdtreeCost.first.totalCost;
        }

        // free the memory used by the datasets we sampled
        delete sampledDataset;
        delete testDataset;
        delete gt_matches;

        return bestParams;
    }



    /**
        Estimates the search time parameters needed to get the desired precision.
        Precondition: the index is built
        Postcondition: the searchParams will have the optimum params set, also the speedup obtained over linear search.
    */
    float estimateSearchParams(SearchParams& searchParams)
    {
        const int nn = 1;
        const long SAMPLE_COUNT = 1000;

        assert(bestIndex!=NULL);   // must have a valid index

        float speedup = 0;

        int samples = min(dataset.rows/10, SAMPLE_COUNT);
        if (samples>0) {
            Matrix<float>* testDataset = dataset.sample(samples);

            logger.info("Computing ground truth\n");

            // we need to compute the ground truth first
            Matrix<int> gt_matches(testDataset->rows,1);
            StartStopTimer t;
            t.start();
            compute_ground_truth(dataset, *testDataset, gt_matches,1);
            t.stop();
            float linear = (float)t.value;

            int checks;
            logger.info("Estimating number of checks\n");

            float searchTime;
            float cb_index;
            if (bestIndex->getType() == KMEANS) {

                logger.info("KMeans algorithm, estimating cluster border factor\n");
                KMeansIndex* kmeans = (KMeansIndex*)bestIndex;
                float bestSearchTime = -1;
                float best_cb_index = -1;
                int best_checks = -1;
                for (cb_index = 0;cb_index<1.1f; cb_index+=0.2f) {
                    kmeans->set_cb_index(cb_index);
                    searchTime = test_index_precision(*kmeans, dataset, *testDataset, gt_matches, params.target_precision, checks, nn, 1);
                    if (searchTime<bestSearchTime || bestSearchTime == -1) {
                        bestSearchTime = searchTime;
                        best_cb_index = cb_index;
                        best_checks = checks;
                    }
                }
                searchTime = bestSearchTime;
                cb_index = best_cb_index;
                checks = best_checks;

                kmeans->set_cb_index(best_cb_index);
                logger.info("Optimum cb_index: %g\n",cb_index);
                ((KMeansIndexParams*)bestParams)->cb_index = cb_index;
            }
            else {
                searchTime = test_index_precision(*bestIndex, dataset, *testDataset, gt_matches, params.target_precision, checks, nn, 1);
            }

            logger.info("Required number of checks: %d \n",checks);;
            searchParams.checks = checks;
            delete testDataset;

            speedup = linear/searchTime;
        }

        return speedup;
    }



};

}

#endif /* AUTOTUNEDINDEX_H_ */
