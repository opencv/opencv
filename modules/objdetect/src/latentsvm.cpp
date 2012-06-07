#include "precomp.hpp"
#include "_latentsvm.h"
#include "_lsvm_matching.h"

/*
// Transformation filter displacement from the block space
// to the space of pixels at the initial image
//
// API
// int convertPoints(int countLevel, CvPoint *points, int *levels,
                  CvPoint **partsDisplacement, int kPoints, int n);
// INPUT
// countLevel        - the number of levels in the feature pyramid
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// partsDisplacement - displacement of part filters (in the block space)
// kPoints           - number of root filter positions
// n                 - number of part filters
// initialImageLevel - level that contains features for initial image
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// points            - the set of root filter positions (in the space of pixels)
// partsDisplacement - displacement of part filters (in the space of pixels)
// RESULT
// Error status
*/
int convertPoints(int /*countLevel*/, int lambda,
                  int initialImageLevel,
                  CvPoint *points, int *levels,
                  CvPoint **partsDisplacement, int kPoints, int n,
                  int maxXBorder,
                  int maxYBorder)
{
    int i, j, bx, by;
    float step, scale;
    step = powf( 2.0f, 1.0f / ((float)lambda) );

    computeBorderSize(maxXBorder, maxYBorder, &bx, &by);

    for (i = 0; i < kPoints; i++)
    {
        // scaling factor for root filter
        scale = SIDE_LENGTH * powf(step, (float)(levels[i] - initialImageLevel));
        points[i].x = (int)((points[i].x - bx + 1) * scale);
        points[i].y = (int)((points[i].y - by + 1) * scale);

        // scaling factor for part filters
        scale = SIDE_LENGTH * powf(step, (float)(levels[i] - lambda - initialImageLevel));
        for (j = 0; j < n; j++)
        {
            partsDisplacement[i][j].x = (int)((partsDisplacement[i][j].x -
                                               2 * bx + 1) * scale);
            partsDisplacement[i][j].y = (int)((partsDisplacement[i][j].y -
                                               2 * by + 1) * scale);
        }
    }
    return LATENT_SVM_OK;
}

/*
// Elimination boxes that are outside the image boudaries
//
// API
// int clippingBoxes(int width, int height,
                     CvPoint *points, int kPoints);
// INPUT
// width             - image wediht
// height            - image heigth
// points            - a set of points (coordinates of top left or
                       bottom right corners)
// kPoints           - points number
// OUTPUT
// points            - updated points (if coordinates less than zero then
                       set zero coordinate, if coordinates more than image
                       size then set coordinates equal image size)
// RESULT
// Error status
*/
int clippingBoxes(int width, int height,
                  CvPoint *points, int kPoints)
{
    int i;
    for (i = 0; i < kPoints; i++)
    {
        if (points[i].x > width - 1)
        {
            points[i].x = width - 1;
        }
        if (points[i].x < 0)
        {
            points[i].x = 0;
        }
        if (points[i].y > height - 1)
        {
            points[i].y = height - 1;
        }
        if (points[i].y < 0)
        {
            points[i].y = 0;
        }
    }
    return LATENT_SVM_OK;
}

/*
// Creation feature pyramid with nullable border
//
// API
// featurePyramid* createFeaturePyramidWithBorder(const IplImage *image,
                                                  int maxXBorder, int maxYBorder);

// INPUT
// image             - initial image
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// RESULT
// Feature pyramid with nullable border
*/
CvLSVMFeaturePyramid* createFeaturePyramidWithBorder(IplImage *image,
                                               int maxXBorder, int maxYBorder)
{
    int opResult;
    int bx, by;
    int level;
    CvLSVMFeaturePyramid *H;

    // Obtaining feature pyramid
    opResult = getFeaturePyramid(image, &H);

    if (opResult != LATENT_SVM_OK)
    {
        freeFeaturePyramidObject(&H);
        return NULL;
    } /* if (opResult != LATENT_SVM_OK) */

    // Addition nullable border for each feature map
    // the size of the border for root filters
    computeBorderSize(maxXBorder, maxYBorder, &bx, &by);
    for (level = 0; level < H->numLevels; level++)
    {
        addNullableBorder(H->pyramid[level], bx, by);
    }
    return H;
}

/*
// Computation of the root filter displacement and values of score function
//
// API
// int searchObject(const featurePyramid *H, const filterObject **all_F, int n,
                    float b,
                    int maxXBorder,
                     int maxYBorder,
                     CvPoint **points, int **levels, int *kPoints, float *score,
                     CvPoint ***partsDisplacement);
// INPUT
// image             - initial image for searhing object
// all_F             - the set of filters (the first element is root filter,
                       other elements - part filters)
// n                 - the number of part filters
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// points            - positions (x, y) of the upper-left corner
                       of root filter frame
// levels            - levels that correspond to each position
// kPoints           - number of positions
// score             - value of the score function
// partsDisplacement - part filters displacement for each position
                       of the root filter
// RESULT
// Error status
*/
int searchObject(const CvLSVMFeaturePyramid *H, const CvLSVMFilterObject **all_F,
                 int n, float b,
                 int maxXBorder,
                 int maxYBorder,
                 CvPoint **points, int **levels, int *kPoints, float *score,
                 CvPoint ***partsDisplacement)
{
    int opResult;

    // Matching
    opResult = maxFunctionalScore(all_F, n, H, b, maxXBorder, maxYBorder,
                                  score, points, levels,
                                  kPoints, partsDisplacement);
    if (opResult != LATENT_SVM_OK)
    {
        return LATENT_SVM_SEARCH_OBJECT_FAILED;
    }

    // Transformation filter displacement from the block space
    // to the space of pixels at the initial image
    // that settles at the level number LAMBDA
    convertPoints(H->numLevels, LAMBDA, LAMBDA, (*points),
                  (*levels), (*partsDisplacement), (*kPoints), n,
                  maxXBorder, maxYBorder);

    return LATENT_SVM_OK;
}

/*
// Computation right bottom corners coordinates of bounding boxes
//
// API
// int estimateBoxes(CvPoint *points, int *levels, int kPoints,
                     int sizeX, int sizeY, CvPoint **oppositePoints);
// INPUT
// points            - left top corners coordinates of bounding boxes
// levels            - levels of feature pyramid where points were found
// (sizeX, sizeY)    - size of root filter
// OUTPUT
// oppositePoins     - right bottom corners coordinates of bounding boxes
// RESULT
// Error status
*/
static int estimateBoxes(CvPoint *points, int *levels, int kPoints,
                  int sizeX, int sizeY, CvPoint **oppositePoints)
{
    int i;
    float step;

    step = powf( 2.0f, 1.0f / ((float)(LAMBDA)));

    *oppositePoints = (CvPoint *)malloc(sizeof(CvPoint) * kPoints);
    for (i = 0; i < kPoints; i++)
    {
        getOppositePoint(points[i], sizeX, sizeY, step, levels[i] - LAMBDA, &((*oppositePoints)[i]));
    }
    return LATENT_SVM_OK;
}

/*
// Computation of the root filter displacement and values of score function
//
// API
// int searchObjectThreshold(const featurePyramid *H,
                             const filterObject **all_F, int n,
                             float b,
                             int maxXBorder, int maxYBorder,
                             float scoreThreshold,
                             CvPoint **points, int **levels, int *kPoints,
                             float **score, CvPoint ***partsDisplacement);
// INPUT
// H                 - feature pyramid
// all_F             - the set of filters (the first element is root filter,
                       other elements - part filters)
// n                 - the number of part filters
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// points            - positions (x, y) of the upper-left corner
                       of root filter frame
// levels            - levels that correspond to each position
// kPoints           - number of positions
// score             - values of the score function
// partsDisplacement - part filters displacement for each position
                       of the root filter
// RESULT
// Error status
*/
int searchObjectThreshold(const CvLSVMFeaturePyramid *H,
                          const CvLSVMFilterObject **all_F, int n,
                          float b,
                          int maxXBorder, int maxYBorder,
                          float scoreThreshold,
                          CvPoint **points, int **levels, int *kPoints,
                          float **score, CvPoint ***partsDisplacement,
                          int numThreads)
{
    int opResult;


    // Matching
#ifdef HAVE_TBB
    if (numThreads <= 0)
    {
        opResult = LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT;
        return opResult;
    }
    opResult = tbbThresholdFunctionalScore(all_F, n, H, b, maxXBorder, maxYBorder,
                                           scoreThreshold, numThreads, score,
                                           points, levels, kPoints,
                                           partsDisplacement);
#else
    opResult = thresholdFunctionalScore(all_F, n, H, b,
                                        maxXBorder, maxYBorder,
                                        scoreThreshold,
                                        score, points, levels,
                                        kPoints, partsDisplacement);

  (void)numThreads;
#endif
    if (opResult != LATENT_SVM_OK)
    {
        return LATENT_SVM_SEARCH_OBJECT_FAILED;
    }

    // Transformation filter displacement from the block space
    // to the space of pixels at the initial image
    // that settles at the level number LAMBDA
    convertPoints(H->numLevels, LAMBDA, LAMBDA, (*points),
                  (*levels), (*partsDisplacement), (*kPoints), n,
                  maxXBorder, maxYBorder);

    return LATENT_SVM_OK;
}

/*
// Compute opposite point for filter box
//
// API
// int getOppositePoint(CvPoint point,
                        int sizeX, int sizeY,
                        float step, int degree,
                        CvPoint *oppositePoint);

// INPUT
// point             - coordinates of filter top left corner
                       (in the space of pixels)
// (sizeX, sizeY)    - filter dimension in the block space
// step              - scaling factor
// degree            - degree of the scaling factor
// OUTPUT
// oppositePoint     - coordinates of filter bottom corner
                       (in the space of pixels)
// RESULT
// Error status
*/
int getOppositePoint(CvPoint point,
                     int sizeX, int sizeY,
                     float step, int degree,
                     CvPoint *oppositePoint)
{
    float scale;
    scale = SIDE_LENGTH * powf(step, (float)degree);
    oppositePoint->x = (int)(point.x + sizeX * scale);
    oppositePoint->y = (int)(point.y + sizeY * scale);
    return LATENT_SVM_OK;
}


/*
// Drawing root filter boxes
//
// API
// int showRootFilterBoxes(const IplImage *image,
                           const filterObject *filter,
                           CvPoint *points, int *levels, int kPoints,
                           CvScalar color, int thickness,
                           int line_type, int shift);
// INPUT
// image             - initial image
// filter            - root filter object
// points            - a set of points
// levels            - levels of feature pyramid
// kPoints           - number of points
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// window contained initial image and filter boxes
// RESULT
// Error status
*/
int showRootFilterBoxes(IplImage *image,
                        const CvLSVMFilterObject *filter,
                        CvPoint *points, int *levels, int kPoints,
                        CvScalar color, int thickness,
                        int line_type, int shift)
{
    int i;
    float step;
    CvPoint oppositePoint;
    step = powf( 2.0f, 1.0f / ((float)LAMBDA));

    for (i = 0; i < kPoints; i++)
    {
        // Drawing rectangle for filter
        getOppositePoint(points[i], filter->sizeX, filter->sizeY,
                         step, levels[i] - LAMBDA, &oppositePoint);
        cvRectangle(image, points[i], oppositePoint,
                    color, thickness, line_type, shift);
    }
#ifdef HAVE_OPENCV_HIGHGUI
    cvShowImage("Initial image", image);
#endif
    return LATENT_SVM_OK;
}

/*
// Drawing part filter boxes
//
// API
// int showPartFilterBoxes(const IplImage *image,
                           const filterObject *filter,
                           CvPoint *points, int *levels, int kPoints,
                           CvScalar color, int thickness,
                           int line_type, int shift);
// INPUT
// image             - initial image
// filters           - a set of part filters
// n                 - number of part filters
// partsDisplacement - a set of points
// levels            - levels of feature pyramid
// kPoints           - number of foot filter positions
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// window contained initial image and filter boxes
// RESULT
// Error status
*/
int showPartFilterBoxes(IplImage *image,
                        const CvLSVMFilterObject **filters,
                        int n, CvPoint **partsDisplacement,
                        int *levels, int kPoints,
                        CvScalar color, int thickness,
                        int line_type, int shift)
{
    int i, j;
    float step;
    CvPoint oppositePoint;

    step = powf( 2.0f, 1.0f / ((float)LAMBDA));

    for (i = 0; i < kPoints; i++)
    {
        for (j = 0; j < n; j++)
        {
            // Drawing rectangles for part filters
            getOppositePoint(partsDisplacement[i][j],
                             filters[j + 1]->sizeX, filters[j + 1]->sizeY,
                             step, levels[i] - 2 * LAMBDA, &oppositePoint);
            cvRectangle(image, partsDisplacement[i][j], oppositePoint,
                        color, thickness, line_type, shift);
        }
    }
#ifdef HAVE_OPENCV_HIGHGUI
    cvShowImage("Initial image", image);
#endif
    return LATENT_SVM_OK;
}

/*
// Drawing boxes
//
// API
// int showBoxes(const IplImage *img,
                 const CvPoint *points, const CvPoint *oppositePoints, int kPoints,
                 CvScalar color, int thickness, int line_type, int shift);
// INPUT
// img               - initial image
// points            - top left corner coordinates
// oppositePoints    - right bottom corner coordinates
// kPoints           - points number
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// RESULT
// Error status
*/
int showBoxes(IplImage *img,
              const CvPoint *points, const CvPoint *oppositePoints, int kPoints,
              CvScalar color, int thickness, int line_type, int shift)
{
    int i;
    for (i = 0; i < kPoints; i++)
    {
        cvRectangle(img, points[i], oppositePoints[i],
                    color, thickness, line_type, shift);
    }
#ifdef HAVE_OPENCV_HIGHGUI
    cvShowImage("Initial image", img);
#endif
    return LATENT_SVM_OK;
}

/*
// Computation maximum filter size for each dimension
//
// API
// int getMaxFilterDims(const filterObject **filters, int kComponents,
                        const int *kPartFilters,
                        unsigned int *maxXBorder, unsigned int *maxYBorder);
// INPUT
// filters           - a set of filters (at first root filter, then part filters
                       and etc. for all components)
// kComponents       - number of components
// kPartFilters      - number of part filters for each component
// OUTPUT
// maxXBorder        - maximum of filter size at the horizontal dimension
// maxYBorder        - maximum of filter size at the vertical dimension
// RESULT
// Error status
*/
int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
                     const int *kPartFilters,
                     unsigned int *maxXBorder, unsigned int *maxYBorder)
{
    int i, componentIndex;
    *maxXBorder = filters[0]->sizeX;
    *maxYBorder = filters[0]->sizeY;
    componentIndex = kPartFilters[0] + 1;
    for (i = 1; i < kComponents; i++)
    {
        if ((unsigned)filters[componentIndex]->sizeX > *maxXBorder)
        {
            *maxXBorder = filters[componentIndex]->sizeX;
        }
        if ((unsigned)filters[componentIndex]->sizeY > *maxYBorder)
        {
            *maxYBorder = filters[componentIndex]->sizeY;
        }
        componentIndex += (kPartFilters[i] + 1);
    }
    return LATENT_SVM_OK;
}

/*
// Computation root filters displacement and values of score function
//
// API
// int searchObjectThresholdSomeComponents(const featurePyramid *H,
                                           const filterObject **filters,
                                           int kComponents, const int *kPartFilters,
                                           const float *b, float scoreThreshold,
                                           CvPoint **points, CvPoint **oppPoints,
                                           float **score, int *kPoints);
// INPUT
// H                 - feature pyramid
// filters           - filters (root filter then it's part filters, etc.)
// kComponents       - root filters number
// kPartFilters      - array of part filters number for each component
// b                 - array of linear terms
// scoreThreshold    - score threshold
// OUTPUT
// points            - root filters displacement (top left corners)
// oppPoints         - root filters displacement (bottom right corners)
// score             - array of score values
// kPoints           - number of boxes
// RESULT
// Error status
*/
int searchObjectThresholdSomeComponents(const CvLSVMFeaturePyramid *H,
                                        const CvLSVMFilterObject **filters,
                                        int kComponents, const int *kPartFilters,
                                        const float *b, float scoreThreshold,
                                        CvPoint **points, CvPoint **oppPoints,
                                        float **score, int *kPoints,
                                        int numThreads)
{
    int error = 0;
    int i, j, s, f, componentIndex;
    unsigned int maxXBorder, maxYBorder;
    CvPoint **pointsArr, **oppPointsArr, ***partsDisplacementArr;
    float **scoreArr;
    int *kPointsArr, **levelsArr;

    // Allocation memory
    pointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
    oppPointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
    scoreArr = (float **)malloc(sizeof(float *) * kComponents);
    kPointsArr = (int *)malloc(sizeof(int) * kComponents);
    levelsArr = (int **)malloc(sizeof(int *) * kComponents);
    partsDisplacementArr = (CvPoint ***)malloc(sizeof(CvPoint **) * kComponents);

    // Getting maximum filter dimensions
    error = getMaxFilterDims(filters, kComponents, kPartFilters, &maxXBorder, &maxYBorder);
    componentIndex = 0;
    *kPoints = 0;
    // For each component perform searching
    for (i = 0; i < kComponents; i++)
    {
#ifdef HAVE_TBB
        error = searchObjectThreshold(H, &(filters[componentIndex]), kPartFilters[i],
            b[i], maxXBorder, maxYBorder, scoreThreshold,
            &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]),
            &(scoreArr[i]), &(partsDisplacementArr[i]), numThreads);
        if (error != LATENT_SVM_OK)
        {
            // Release allocated memory
            free(pointsArr);
            free(oppPointsArr);
            free(scoreArr);
            free(kPointsArr);
            free(levelsArr);
            free(partsDisplacementArr);
            return LATENT_SVM_SEARCH_OBJECT_FAILED;
        }
#else
    (void)numThreads;
        searchObjectThreshold(H, &(filters[componentIndex]), kPartFilters[i],
            b[i], maxXBorder, maxYBorder, scoreThreshold,
            &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]),
            &(scoreArr[i]), &(partsDisplacementArr[i]));
#endif
        estimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i],
            filters[componentIndex]->sizeX, filters[componentIndex]->sizeY, &(oppPointsArr[i]));
        componentIndex += (kPartFilters[i] + 1);
        *kPoints += kPointsArr[i];
    }

    *points = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    *oppPoints = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    *score = (float *)malloc(sizeof(float) * (*kPoints));
    s = 0;
    for (i = 0; i < kComponents; i++)
    {
        f = s + kPointsArr[i];
        for (j = s; j < f; j++)
        {
            (*points)[j].x = pointsArr[i][j - s].x;
            (*points)[j].y = pointsArr[i][j - s].y;
            (*oppPoints)[j].x = oppPointsArr[i][j - s].x;
            (*oppPoints)[j].y = oppPointsArr[i][j - s].y;
            (*score)[j] = scoreArr[i][j - s];
        }
        s = f;
    }

    // Release allocated memory
    for (i = 0; i < kComponents; i++)
    {
        free(pointsArr[i]);
        free(oppPointsArr[i]);
        free(scoreArr[i]);
        free(levelsArr[i]);
        for (j = 0; j < kPointsArr[i]; j++)
        {
            free(partsDisplacementArr[i][j]);
        }
        free(partsDisplacementArr[i]);
    }
    free(pointsArr);
    free(oppPointsArr);
    free(scoreArr);
    free(kPointsArr);
    free(levelsArr);
    free(partsDisplacementArr);
    return LATENT_SVM_OK;
}
