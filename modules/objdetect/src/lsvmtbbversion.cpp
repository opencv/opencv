#include "precomp.hpp"

#ifdef HAVE_TBB
#include "_lsvm_tbbversion.h"

/*
// Task class
*/
class ScoreComputation : public tbb::task
{
private:
    const CvLSVMFilterObject **filters;
    const int n;
    const CvLSVMFeaturePyramid *H;
    const float b;
    const int maxXBorder;
    const int maxYBorder;
    const float scoreThreshold;
    const int kLevels;
    const int *procLevels;
public:
    float **score;
    CvPoint ***points;
    CvPoint ****partsDisplacement;
    int *kPoints;
public:
    ScoreComputation(const CvLSVMFilterObject **_filters, int _n, 
                     const CvLSVMFeaturePyramid *_H,
                     float _b, int _maxXBorder, int _maxYBorder,
                     float _scoreThreshold, int _kLevels, const int *_procLevels,
                     float **_score, CvPoint ***_points, int *_kPoints,
                     CvPoint ****_partsDisplacement) :
    n(_n), b(_b), maxXBorder(_maxXBorder), 
        maxYBorder(_maxYBorder), scoreThreshold(_scoreThreshold),
        kLevels(_kLevels), score(_score), points(_points), kPoints(_kPoints),
        partsDisplacement(_partsDisplacement)
    {
        filters = _filters;
        H = _H;
        procLevels = _procLevels;
    };

    task* execute()
    {
        int i, level, partsLevel, res;
        for (i = 0; i < kLevels; i++)
        {
            level = procLevels[i];
            partsLevel = level - H->lambda;
            res = thresholdFunctionalScoreFixedLevel(
                filters, n, H, level, b,
                maxXBorder, maxYBorder, scoreThreshold, &(score[partsLevel]), 
                points[partsLevel], &(kPoints[partsLevel]), 
                partsDisplacement[partsLevel]);
            if (res != LATENT_SVM_OK)
            {
                continue;
            }
        }
        return NULL;
    }
};

/*
// Computation score function using TBB tasks
//
// API
// int tbbTasksThresholdFunctionalScore(const CvLSVMFilterObject **filters, const int n, 
                                        const CvLSVMFeatureMap *H, const float b,
                                        const int maxXBorder, const int maxYBorder,
                                        const float scoreThreshold,
                                        int *kLevels, int **procLevels,
                                        const int threadsNum,
                                        float **score, CvPoint ***points, 
                                        int *kPoints,
                                        CvPoint ****partsDisplacement);
// INPUT
// filters           - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// kLevels           - array that contains number of levels processed 
                       by each thread
// procLevels        - array that contains lists of levels processed 
                       by each thread
// threadsNum        - the number of created threads
// OUTPUT
// score             - score function values that exceed threshold
// points            - the set of root filter positions (in the block space)
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
//
*/
int tbbTasksThresholdFunctionalScore(const CvLSVMFilterObject **filters, const int n, 
                                     const CvLSVMFeaturePyramid *H, const float b,
                                     const int maxXBorder, const int maxYBorder,
                                     const float scoreThreshold,
                                     int *kLevels, int **procLevels,
                                     const int threadsNum,
                                     float **score, CvPoint ***points, 
                                     int *kPoints,
                                     CvPoint ****partsDisplacement)
{
    tbb::task_list tasks;
    int i;
    for (i = 0; i < threadsNum; i++)
    {
        ScoreComputation& sc = 
            *new(tbb::task::allocate_root()) ScoreComputation(filters, n, H, b,
            maxXBorder, maxYBorder, scoreThreshold, kLevels[i], procLevels[i], 
            score, points, kPoints, partsDisplacement);
        tasks.push_back(sc);
    }
    tbb::task::spawn_root_and_wait(tasks);
    return LATENT_SVM_OK;
};
#endif

