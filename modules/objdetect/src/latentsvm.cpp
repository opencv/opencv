#include "precomp.hpp"
#include "_latentsvm.h"
#include "_lsvm_matching.h"
#include "_lsvm_function.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifdef HAVE_TBB
#include <tbb/tbb.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#endif

void FeaturePyramid32(CvLSVMFeaturePyramid* H, int maxX, int maxY){
    CvLSVMFeatureMap *H32; 
    int i, j, k, l;
    int p  = H->pyramid[0]->numFeatures;

    for(i = 0; i < H->numLevels; i++){
        allocFeatureMapObject(&(H32), H->pyramid[i]->sizeX, H->pyramid[i]->sizeY, p + 1);
        for(j = 0; j < (H->pyramid[i]->sizeX * H->pyramid[i]->sizeY); j++){
            for(k = 0; k < p; k++){
                H32->map[j * (p + 1) + k] = H->pyramid[i]->map[j * p + k];
            }
            H32->map[j * (p + 1) + k] = 1.0f;
        }
        freeFeatureMapObject(&(H->pyramid[i]));
        H->pyramid[i] = H32;
    }
    for(l = 0; l < H->numLevels; l++){
        for(j = maxY + 1; j < (H->pyramid[l]->sizeY - maxY - 1); j++){
            for(i = maxX + 1; i < (H->pyramid[l]->sizeX - maxX - 1); i++){
                H->pyramid[l]->map[ (j * H->pyramid[l]->sizeX + i) * (p+1) + p] = 0.0f;
            }
        }
    }
}

CvLSVMFeaturePyramid* createPCA_FeaturePyramid(CvLSVMFeaturePyramid* H, CvLatentSvmDetector* detector, int maxX, int maxY){
    CvLSVMFeaturePyramid *H_PCA; 
    int i, j, k, l;
    int max_l = detector->pca_size;
    int p = H->pyramid[0]->numFeatures;

    allocFeaturePyramidObject(&H_PCA, H->numLevels);

    for(i = 0; i < H->numLevels; i++){
        allocFeatureMapObject(&(H_PCA->pyramid[i]), H->pyramid[i]->sizeX, H->pyramid[i]->sizeY, 6);
        for(j = 0; j < (H->pyramid[i]->sizeX * H->pyramid[i]->sizeY); j++){
            for(k = 0; k < 5; k++){
                for(l = 0; l < max_l; l++){
                    H_PCA->pyramid[i]->map[j * 6 + k] += 
                      detector->pca[k * max_l + l] * H->pyramid[i]->map[j * p + l];
                }
            }
            H_PCA->pyramid[i]->map[j * 6 + k] = 1.0f;
        }
    }
    for(l = 0; l < H->numLevels; l++){
        for(j = maxY + 1; j < (H->pyramid[l]->sizeY - maxY - 1); j++){
            for(i = maxX + 1; i < (H->pyramid[l]->sizeX - maxX - 1); i++){
                H_PCA->pyramid[l]->map[ (j * H->pyramid[l]->sizeX + i) * 6 + 5] = 0.0f;
            }
        }
    }
    
    return H_PCA;
}

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
    int i, j;
    float step, scale;
    step = powf( 2.0f, 1.0f / ((float)lambda) );

    //computeBorderSize(maxXBorder, maxYBorder, &bx, &by);
    
    for (i = 0; i < kPoints; i++)
    {
        // scaling factor for root filter
        scale = SIDE_LENGTH * powf(step, (float)(levels[i] - initialImageLevel));
        points[i].x = (int)((points[i].x - maxXBorder) * scale);
        points[i].y = (int)((points[i].y - maxYBorder) * scale);

        // scaling factor for part filters
        scale = SIDE_LENGTH * powf(step, (float)(levels[i] - lambda - initialImageLevel));
        for (j = 0; j < n; j++)
        {            
            partsDisplacement[i][j].x = (int)((partsDisplacement[i][j].x - 
                                               maxXBorder) * scale);
            partsDisplacement[i][j].y = (int)((partsDisplacement[i][j].y - 
                                               maxYBorder) * scale);
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
    bx = maxXBorder + 1; 
    by = maxYBorder + 1;
    for (level = 0; level < H->numLevels; level++)
    {
        addNullableBorder(H->pyramid[level], bx, by);
    }
    return H;
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
int estimateBoxes(CvPoint *points, int *levels, int kPoints, 
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
                             const CvLSVMFilterObject **all_F, int n,
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
                          const CvLSVMFeaturePyramid *H_PCA,
                          const CvLSVMFilterObject **all_F, int n,
                          float b, 
                          int maxXBorder, int maxYBorder, 
                          float scoreThreshold,
                          CvPoint **points, int **levels, int *kPoints, 
                          float **score, CvPoint ***partsDisplacement)
{
    int opResult = LATENT_SVM_OK;

  int i, j, k, path;
  int di, dj, ii;

    //int *map,jj, nomer;
    //FILE *dump;

  float p;
  float fine, pfine;
  float mpath;

  CvPoint *tmpPoints;
  int     *tmpLevels;
    float   **tmpAScore;

    int flag,flag2;

    CvPoint *PCAPoints;
  int     *PCALevels;
    float   **PCAAScore; 
  int      PCAkPoints;
    float    PCAScore;
    int tmpSize = 10;
    int tmpStep = 10;

    float *rootScoreForLevel;
    int maxX, maxY, maxPathX, maxPathY, step;
    int pathX, pathY;
    int ai;


    float **cashM;
    int   **maskM;
    int    sizeM;

    sizeM  = H_PCA->pyramid[0]->sizeX - maxXBorder + 1;
    sizeM *= H_PCA->pyramid[0]->sizeY - maxYBorder + 1;

    cashM = (float**)malloc(sizeof(float *) * n);
    maskM = (int **)malloc(sizeof(int   *) * n);
    for(ai = 0; ai < n; ai++){
        cashM[ai] = (float*)malloc(sizeof(float) * sizeM);
        maskM[ai] = (int *)malloc(sizeof(int) * (sizeM/(sizeof(int) * 8) + 1));
    }
    
    
    PCAPoints = (CvPoint*)malloc(sizeof(CvPoint) * tmpSize);
    PCALevels = (int*)malloc(sizeof(int)     * tmpSize);
    PCAAScore = (float **)malloc(sizeof(float *) * tmpSize);
    for(ai = 0; ai < tmpSize; ai++){
        PCAAScore[ai] = (float *)malloc(sizeof(float) * (n + 2));
    }

    PCAkPoints = 0;
    for(k = LAMBDA; k < H_PCA->numLevels; k++){
        maxX = H_PCA->pyramid[k]->sizeX - maxXBorder + 1;
        maxY = H_PCA->pyramid[k]->sizeY - maxYBorder + 1;
        maxPathX = H_PCA->pyramid[k - LAMBDA]->sizeX - maxXBorder + 1;
        maxPathY = H_PCA->pyramid[k - LAMBDA]->sizeY - maxYBorder + 1;
        rootScoreForLevel = (float *) malloc(sizeof(float) 
            * (maxX - (int)ceil(maxXBorder/2.0)) 
            * (maxY - (int)ceil(maxYBorder/2.0)));

        step = maxX - (int)ceil(maxXBorder/2.0);
        //dump = fopen("map_10.csv", "w");
        for(j = (int)ceil(maxYBorder/2.0) ; j < maxY; j++){
            for(i = (int)ceil(maxXBorder/2.0) ; i < maxX; i++){        
          rootScoreForLevel[(j - (int)ceil(maxYBorder/2.0)) * step + i - (int)ceil(maxXBorder/2.0)] 
                = calcM_PCA(k, i, j, H_PCA, all_F[0]);
        //         fprintf(dump, "%f;", rootScoreForLevel[j * maxX + i]);
            }
        //     fprintf(dump, "\n");
        }
        // fclose(dump);

        sizeM  = maxPathX * maxPathY;
        for(path = 0 ; path < n; path++){
            memset(maskM[path], 0, sizeof(int) * (sizeM/(sizeof(int) * 8) + 1));
        }
      
        for(j = (int)ceil(maxYBorder/2.0) ; j < maxY; j++){
            for(i = (int)ceil(maxXBorder/2.0) ; i < maxX; i++){
    //      PCAScore = calcM_PCA(k, i, j, H_PCA, all_F[0]);
                PCAScore = 
                    rootScoreForLevel[(j - (int)ceil(maxYBorder/2.0)) * step + i - (int)ceil(maxXBorder/2.0)];
                PCAScore += b;
                PCAAScore[PCAkPoints][0] = PCAScore - b;

                flag2=0;
          for(path = 1 ; (path <= n) && (!flag2); path++){
                    if(PCAScore > all_F[path - 1]->Deformation_PCA)
                    {
              p = F_MIN ;
                        pfine = 0.f;
                        //pathX = (i - maxXBorder - 1) * 2 + maxXBorder + 1 + all_F[path]->V.x;
                        //pathY = (j - maxYBorder - 1) * 2 + maxYBorder + 1 + all_F[path]->V.y; 
                        pathX = i * 2 - maxXBorder + all_F[path]->V.x;
                        pathY = j * 2 - maxYBorder + all_F[path]->V.y; 
                        flag = 1;
                        for(dj = max(0,        pathY - all_F[path]->deltaY); 
                dj < min(maxPathY, pathY + all_F[path]->deltaY); 
                dj++){
                            for(di = max(0,        pathX - all_F[path]->deltaX); 
                      di < min(maxPathX, pathX + all_F[path]->deltaX); 
                      di++){
                    //fine = calcFine(all_F[path], abs(pathX - di), abs(pathY - dj));
                                fine = calcFine(all_F[path], pathX - di, pathY - dj);
                                if((PCAScore - fine) > all_F[path - 1]->Hypothesis_PCA)
                                {
                                    flag = 0;
                      mpath = calcM_PCA_cash(k - LAMBDA, di, dj, H_PCA, all_F[path], cashM[path - 1], maskM[path - 1], maxPathX) - fine;
                      if( mpath > p){
                        p     = mpath;
                                        pfine = fine;
                                }
                                }
                            }
                        }
                        if(flag==0){
                            PCAAScore[PCAkPoints][path] = p;// + pfine;
                            PCAScore += p;// + pfine;                            
                        } else flag2 = 1;
            } 
                    else flag2 = 1;
          }
                if((PCAScore > all_F[n]->Hypothesis_PCA)&&(flag2==0)){
                 PCALevels[PCAkPoints]   = k;
            PCAPoints[PCAkPoints].x = i;
            PCAPoints[PCAkPoints].y = j;
                    PCAAScore[PCAkPoints][n + 1] = PCAScore;
            PCAkPoints ++;
            if(PCAkPoints >= tmpSize){
                        tmpPoints = (CvPoint*)malloc(sizeof(CvPoint) * (tmpSize + tmpStep));
                        tmpLevels = (int*)malloc(sizeof(int)     * (tmpSize + tmpStep));
                        tmpAScore = (float **)malloc(sizeof(float *) * (tmpSize + tmpStep));
                        for(ai = tmpSize; ai < tmpSize + tmpStep; ai++){
                            tmpAScore[ai] = (float *)malloc(sizeof(float) * (n + 2));
                        }
                        for(ii = 0; ii < PCAkPoints; ii++){
                            tmpLevels[ii]   = PCALevels[ii]  ;
                    tmpPoints[ii].x = PCAPoints[ii].x;
                    tmpPoints[ii].y = PCAPoints[ii].y;
                            tmpAScore[ii]   = PCAAScore[ii]  ;
                        }
                        free(PCALevels);
                        free(PCAPoints);
                        free(PCAAScore);
                        PCALevels = tmpLevels;
                        PCAPoints = tmpPoints;
                        PCAAScore = tmpAScore;
                        tmpSize += tmpStep;
            }
                }     
        }            
      }
        free (rootScoreForLevel);
    }

  (*points) = (CvPoint *)malloc(sizeof(CvPoint) * PCAkPoints);
  (*levels) = (int    *)malloc(sizeof(int    ) * PCAkPoints);
  (*score ) = (float  *)malloc(sizeof(float  ) * PCAkPoints);
  (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (PCAkPoints + 1));
  
  (*kPoints) = 0;
    if(PCAkPoints > 0)
        (*partsDisplacement)[(*kPoints)] = (CvPoint *)malloc(sizeof(CvPoint) * (n + 1));
    for(ii = 0; ii < PCAkPoints; ii++)
    {
      k = PCALevels[ii]  ;
      i = PCAPoints[ii].x;
      j = PCAPoints[ii].y;
        
        maxPathX = H_PCA->pyramid[k - LAMBDA]->sizeX - maxXBorder + 1;
        maxPathY = H_PCA->pyramid[k - LAMBDA]->sizeY - maxYBorder + 1;

      (*score )[(*kPoints)] = PCAAScore[ii][n + 1] + calcM(k, i, j, H, all_F[0]) - PCAAScore[ii][0];
        (*partsDisplacement)[(*kPoints)][0].x = i;
    (*partsDisplacement)[(*kPoints)][0].y = j;
      for(path = 1 ; path <= n; path++){
            if((*score )[(*kPoints)] < all_F[path - 1]->Deformation) break;
           // {
        p = F_MIN ;
            flag = 1;
            //pathX = (i - maxXBorder - 1) * 2 + maxXBorder + 1 + all_F[path]->V.x;
            //pathY = (j - maxYBorder - 1) * 2 + maxYBorder + 1 + all_F[path]->V.y; 
            pathX = i * 2 - maxXBorder + all_F[path]->V.x;
            pathY = j * 2 - maxYBorder + all_F[path]->V.y; 
            for(dj = max(0,        pathY - all_F[path]->deltaY); 
          dj < min(maxPathY, pathY + all_F[path]->deltaY); 
          dj++){
                for(di = max(0,        pathX - all_F[path]->deltaX); 
                di < min(maxPathX, pathX + all_F[path]->deltaX); 
                di++){
            //fine = calcFine(all_F[path], abs(pathX - di), abs(pathY - dj));
                    fine = calcFine(all_F[path], pathX - di, pathY - dj);
                    if(((*score )[(*kPoints)] - fine) > all_F[path - 1]->Hypothesis)
                    {
                        flag = 0;
                        mpath = calcM(k - LAMBDA, di, dj, H, all_F[path]) - fine;
              if(mpath > p){
                p = mpath;
                            pfine = fine;
                (*partsDisplacement)[(*kPoints)][path].x = di;
                (*partsDisplacement)[(*kPoints)][path].y = dj;
              }
            }
          }
        }
            if(flag == 0)
            (*score )[(*kPoints)] +=  p - PCAAScore[ii][path];// + pfine;
       // }
      }
      if((*score )[(*kPoints)] > scoreThreshold)
        {
        (*levels)[(*kPoints)]   = k;
        (*points)[(*kPoints)].x = i;
        (*points)[(*kPoints)].y = j;
        (*kPoints) ++;
            (*partsDisplacement)[(*kPoints)] = (CvPoint*) malloc(sizeof(CvPoint) * (n + 1));
      }
  }
    if((*kPoints) > 0){
        free((*partsDisplacement)[(*kPoints)]);
    }
    // Matching end

    free(PCAPoints);
    free(PCALevels);
    for(ai = 0; ai < tmpSize; ai++){
        free(PCAAScore[ai]);
    }
    free(PCAAScore);

    for(ai = 0; ai < n; ai++){
        free(cashM[ai]);
        free(maskM[ai]);
    }
    free(cashM);
    free(maskM);

    if (opResult != (LATENT_SVM_OK))
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
                           const CvLSVMFilterObject *filter, 
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
                           const CvLSVMFilterObject *filter, 
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

///*
//// Computation maximum filter size for each dimension
////
//// API
//// int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
//                        const int *kPartFilters, 
//                        unsigned int *maxXBorder, unsigned int *maxYBorder);
//// INPUT
//// filters           - a set of filters (at first root filter, then part filters 
//                       and etc. for all components)
//// kComponents       - number of components
//// kPartFilters      - number of part filters for each component
//// OUTPUT
//// maxXBorder        - maximum of filter size at the horizontal dimension
//// maxYBorder        - maximum of filter size at the vertical dimension
//// RESULT
//// Error status
//*/
//int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
//                     const int *kPartFilters, 
//                     unsigned int *maxXBorder, unsigned int *maxYBorder)
//{
//    int i, componentIndex;    
//    *maxXBorder = filters[0]->sizeX;
//    *maxYBorder = filters[0]->sizeY;
//    componentIndex = kPartFilters[0] + 1;
//    for (i = 1; i < kComponents; i++)
//    {
//        if ((unsigned)filters[componentIndex]->sizeX > *maxXBorder)
//        {
//            *maxXBorder = filters[componentIndex]->sizeX;
//        }
//        if ((unsigned)filters[componentIndex]->sizeY > *maxYBorder)
//        {
//            *maxYBorder = filters[componentIndex]->sizeY;
//        }
//        componentIndex += (kPartFilters[i] + 1);
//    }
//    return LATENT_SVM_OK;
//}


#ifdef HAVE_TBB

struct PathOfModel {
    int *componentIndex;
    const CvLSVMFeaturePyramid *H;
    const CvLSVMFeaturePyramid *H_PCA;
    const CvLSVMFilterObject **filters;
    const int *kPartFilters;
    const float *b;
    unsigned int maxXBorder, maxYBorder;
    CvPoint **pointsArr, **oppPointsArr, ***partsDisplacementArr;
    float **scoreArr;
    int *kPointsArr, **levelsArr;
    float scoreThreshold;
    CvPoint **oppPoints;
public:
    PathOfModel(
      int *_componentIndex,
    const CvLSVMFeaturePyramid *_H,
    const CvLSVMFeaturePyramid *_H_PCA,
    const CvLSVMFilterObject **_filters,
    const int *_kPartFilters,
    const float *_b,
    unsigned int _maxXBorder, unsigned int _maxYBorder,
    CvPoint **_pointsArr, CvPoint  **_oppPointsArr, CvPoint  ***_partsDisplacementArr,
    float **_scoreArr,
    int *_kPointsArr, int **_levelsArr,
    float _scoreThreshold,
    CvPoint **_oppPoints
    ):
    componentIndex(_componentIndex),
    H(_H),
    H_PCA(_H_PCA),
    filters(_filters),
    kPartFilters(_kPartFilters),
    b(_b),
    maxXBorder(_maxXBorder),
    maxYBorder(_maxYBorder),
    pointsArr(_pointsArr),
    oppPointsArr(_oppPointsArr),
    partsDisplacementArr(_partsDisplacementArr),
    scoreArr(_scoreArr),
    kPointsArr(_kPointsArr),
    levelsArr(_levelsArr),
    scoreThreshold(_scoreThreshold),
    oppPoints(_oppPoints)
    {}

    
    void operator()( const tbb::blocked_range<int>& range ) const {
        
        for( int i=range.begin(); i!=range.end(); ++i )
        {
          searchObjectThreshold(H, H_PCA, &(filters[componentIndex[i]]), kPartFilters[i],
            b[i], maxXBorder, maxYBorder, scoreThreshold, 
            &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]), 
            &(scoreArr[i]), &(partsDisplacementArr[i]));
          estimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i], 
            filters[componentIndex[i]]->sizeX, filters[componentIndex[i]]->sizeY, &(oppPointsArr[i]));
        }
    }
};

#endif
/*
// Computation root filters displacement and values of score function
//
// API
// int searchObjectThresholdSomeComponents(const featurePyramid *H,
                                           const CvLSVMFilterObject **filters, 
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
                                        const CvLSVMFeaturePyramid *H_PCA,
                                        const CvLSVMFilterObject **filters, 
                                        int kComponents, const int *kPartFilters,
                                        const float *b, float scoreThreshold,
                                        CvPoint **points, CvPoint **oppPoints,
                                        float **score, int *kPoints)
{
     int error = 0;
    int i, j, s, f, *componentIndex;
    unsigned int maxXBorder, maxYBorder;
    CvPoint **pointsArr, **oppPointsArr, ***partsDisplacementArr;
    float **scoreArr;
    int *kPointsArr, **levelsArr;
    int sum;
    
    // Allocation memory
    pointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
    oppPointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
    scoreArr = (float **)malloc(sizeof(float *) * kComponents);
    kPointsArr = (int *)malloc(sizeof(int) * kComponents);
    levelsArr = (int **)malloc(sizeof(int *) * kComponents);
    partsDisplacementArr = (CvPoint ***)malloc(sizeof(CvPoint **) * kComponents);
    componentIndex = (int *)malloc(sizeof(int) * kComponents);
    
    // Getting maximum filter dimensions
    error = getMaxFilterDims(filters, kComponents, kPartFilters, &maxXBorder, &maxYBorder);
    *kPoints = 0;
    sum = 0;
    componentIndex[0] = 0;
    for (i = 1; i < kComponents; i++)
    {
        componentIndex[i] = componentIndex[i - 1] + (kPartFilters[i - 1] + 1);
    }
    // For each component perform searching
//#pragma omp parallel for schedule(dynamic) reduction(+ : sum) 
#ifdef HAVE_TBB
    PathOfModel POM(
      componentIndex,
      H,
      H_PCA,
      filters,
      kPartFilters,
      b,
      maxXBorder,
      maxYBorder,
      pointsArr,
      oppPointsArr,
      partsDisplacementArr,
      scoreArr,
      kPointsArr,
      levelsArr,
      scoreThreshold,
      oppPoints);
    tbb::parallel_for( tbb::blocked_range<int>( 0, kComponents ), POM);
#else
    for (i = 0; i < kComponents; i++)
    {
        searchObjectThreshold(H, H_PCA, &(filters[componentIndex[i]]), kPartFilters[i],
            b[i], maxXBorder, maxYBorder, scoreThreshold, 
            &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]), 
            &(scoreArr[i]), &(partsDisplacementArr[i]));
        estimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i], 
            filters[componentIndex[i]]->sizeX, filters[componentIndex[i]]->sizeY, &(oppPointsArr[i]));
    }
#endif
    for (i = 0; i < kComponents; i++)
    {    
        //*kPoints += kPointsArr[i];
        sum += kPointsArr[i];
    } 
    *kPoints = sum;
    *points = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    *oppPoints = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    *score = (float *)malloc(sizeof(float) * (*kPoints));

    //file = fopen("point.txt", "w");
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
      //      fprintf(file, "%d %d %d %d %f\n", (*points)[j].x, (*points)[j].y,
      //          (*oppPoints)[j].x, (*oppPoints)[j].y, (*score)[j]);
        }
        s = f;
    }
    //fclose(file);

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
    free(componentIndex);
    return LATENT_SVM_OK;
}
