#include "precomp.hpp"
#include "_lsvm_matching.h"
#include <stdio.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

/*
// Function for convolution computation
//
// INPUT
// Fi                - filter object
// map               - feature map
// OUTPUT
// f                 - the convolution
// RESULT
// Error status
*/
int convolution(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *map, float *f)
{
    int n1, m1, n2, m2, p, size, diff1, diff2;
	int i1, i2, j1, j2, k;
	float tmp_f1, tmp_f2, tmp_f3, tmp_f4;
	float *pMap = NULL;
	float *pH = NULL;
	    
	n1 = map->sizeY;
	m1 = map->sizeX;
	n2 = Fi->sizeY;
	m2 = Fi->sizeX;
	p = map->p;

	diff1 = n1 - n2 + 1;
	diff2 = m1 - m2 + 1;
	size = diff1 * diff2;
	for (j1 = diff2 - 1; j1 >= 0; j1--)
	{
		
		for (i1 = diff1 - 1; i1 >= 0; i1--)
		{
			tmp_f1 = 0.0f;
			tmp_f2 = 0.0f;
			tmp_f3 = 0.0f;
			tmp_f4 = 0.0f;
			for (i2 = 0; i2 < n2; i2++)
			{
				for (j2 = 0; j2 < m2; j2++)
				{
					pMap = map->Map + (i1 + i2) * m1 * p + (j1 + j2) * p;//sm2
					pH = Fi->H + (i2 * m2 + j2) * p;//sm2
					for (k = 0; k < p/4; k++)
					{

						tmp_f1 += pMap[4*k]*pH[4*k];//sm2
						tmp_f2 += pMap[4*k+1]*pH[4*k+1];
						tmp_f3 += pMap[4*k+2]*pH[4*k+2];
						tmp_f4 += pMap[4*k+3]*pH[4*k+3];
					}
			
					if (p%4==1)
					{
						tmp_f1 += pH[p-1]*pMap[p-1];
					}
					else
					{
						if (p%4==2)
						{
							tmp_f1 += pH[p-2]*pMap[p-2] + pH[p-1]*pMap[p-1];
						}
						else 
						{
							if (p%4==3)
							{
								tmp_f1 += pH[p-3]*pMap[p-3] + pH[p-2]*pMap[p-2] + pH[p-1]*pMap[p-1];
							}
						}
					}
					
				}
			}
			f[i1 * diff2 + j1] = tmp_f1 + tmp_f2 + tmp_f3 + tmp_f4;//sm1
		}
	}
    return LATENT_SVM_OK;
}

/*
// Computation multiplication of FFT images
//
// API
// int fftImagesMulti(float *fftImage1, float *fftImage2, int numRows, int numColls, 
                      float *multi);
// INPUT
// fftImage1         - first fft image
// fftImage2         - second fft image
// (numRows, numColls) - image dimesions
// OUTPUT
// multi             - multiplication
// RESULT
// Error status
*/
int fftImagesMulti(float *fftImage1, float *fftImage2, int numRows, int numColls, 
                   float *multi)
{
    int i, index, size;
    size = numRows * numColls;
    for (i = 0; i < size; i++)
    {
        index = 2 * i;
        multi[index] = fftImage1[index] * fftImage2[index] - 
                       fftImage1[index + 1] * fftImage2[index + 1];
        multi[index + 1] = fftImage1[index] * fftImage2[index + 1] +
                           fftImage1[index + 1] * fftImage2[index];
    }
    return LATENT_SVM_OK;
}

/*
// Turnover filter matrix for the single feature
//
// API
// int rot2PI(float *filter, int dimX, int dimY, float *rot2PIFilter, 
              int p, int shift);
// INPUT
// filter            - filter weight matrix
// (dimX, dimY)      - dimension of filter matrix
// p                 - number of features
// shift             - number of feature (or channel)
// OUTPUT
// rot2PIFilter      - rotated matrix
// RESULT
// Error status
*/
int rot2PI(float *filter, int dimX, int dimY, float *rot2PIFilter, 
           int p, int shift)
{
    int i, size;
    size = dimX * dimY;
    for (i = 0; i < size; i++)
    {
        rot2PIFilter[i] = filter[(size - i - 1) * p + shift];
    }
    return LATENT_SVM_OK;
}

/*
// Addition nullable bars to the dimension of feature map (single feature)
//
// API
// int addNullableBars(float *rot2PIFilter, int dimX, int dimY, 
                       float *newFilter, int newDimX, int newDimY);
// INPUT
// rot2PIFilter      - filter matrix for the single feature that was rotated
// (dimX, dimY)      - dimension rot2PIFilter
// (newDimX, newDimY)- dimension of feature map for the single feature
// OUTPUT
// newFilter         - filter matrix with nullable bars
// RESULT
// Error status
*/
int addNullableBars(float *rot2PIFilter, int dimX, int dimY, 
                    float *newFilter, int newDimX, int newDimY)
{
    int size, i, j;
    size = newDimX * newDimY;
    for (i = 0; i < size; i++)
    {
        newFilter[2 * i] = 0.0;
        newFilter[2 * i + 1] = 0.0;
    }
    for (i = 0; i < dimY; i++)
    {
        for (j = 0; j < dimX; j++)
        {
            newFilter[2 * (i * newDimX + j)] = rot2PIFilter[i * dimX + j];
        }
    }
    return LATENT_SVM_OK;
}

/*
// Computation FFT image for filter object
//
// API
// int getFFTImageFilterObject(const CvLSVMFilterObject *filter, 
                               int mapDimX, int mapDimY,
                               fftImage **image);
// INPUT
// filter        - filter object
// (mapDimX, mapDimY)- dimension of feature map
// OUTPUT
// image         - fft image
// RESULT
// Error status
*/
int getFFTImageFilterObject(const CvLSVMFilterObject *filter, 
                            int mapDimX, int mapDimY,
                            CvLSVMFftImage **image)
{
    unsigned int i, mapSize, filterSize;
    int res;
    float *newFilter, *rot2PIFilter;    
    
    filterSize = filter->sizeX * filter->sizeY;
    mapSize = mapDimX * mapDimY;
    newFilter = (float *)malloc(sizeof(float) * (2 * mapSize));
    rot2PIFilter = (float *)malloc(sizeof(float) * filterSize);
    res = allocFFTImage(image, filter->p, mapDimX, mapDimY);
    if (res != LATENT_SVM_OK)
    {
        return res;
    }
    for (i = 0; i < filter->p; i++)
    {        
        rot2PI(filter->H, filter->sizeX, filter->sizeY, rot2PIFilter, filter->p, i);
        addNullableBars(rot2PIFilter, filter->sizeX, filter->sizeY, 
                        newFilter, mapDimX, mapDimY);
        fft2d(newFilter, (*image)->channels[i], mapDimY, mapDimX);
    }   
    free(newFilter);
    free(rot2PIFilter);
    return LATENT_SVM_OK;
}

/*
// Computation FFT image for feature map
//
// API
// int getFFTImageFeatureMap(const featureMap *map, fftImage **image);
// INPUT
// OUTPUT
// RESULT
// Error status
*/
int getFFTImageFeatureMap(const CvLSVMFeatureMap *map, CvLSVMFftImage **image)
{
    int i, j, size;
    float *buf;    
    allocFFTImage(image, map->p, map->sizeX, map->sizeY);
    size = map->sizeX * map->sizeY;
    buf = (float *)malloc(sizeof(float) * (2 * size));
    for (i = 0; i < map->p; i++)
    {
        for (j = 0; j < size; j++)
        {
            buf[2 * j] = map->Map[j * map->p + i];
            buf[2 * j + 1] = 0.0;
        }
        fft2d(buf, (*image)->channels[i], map->sizeY, map->sizeX);
    }
    free(buf);
    return LATENT_SVM_OK;
}

/*
// Function for convolution computation using FFT
// 
// API
// int convFFTConv2d(const fftImage *featMapImage, const fftImage *filterImage, 
                     int filterDimX, int filterDimY, float **conv);
// INPUT
// featMapImage      - feature map image
// filterImage       - filter image
// (filterDimX,filterDimY) - filter dimension
// OUTPUT
// conv              - the convolution
// RESULT
// Error status
*/
int convFFTConv2d(const CvLSVMFftImage *featMapImage, const CvLSVMFftImage *filterImage, 
                  int filterDimX, int filterDimY, float **conv)
{
    int i, j, size, diffX, diffY, index;
    float *imagesMult, *imagesMultRes, *fconv;
    size = 2 * featMapImage->dimX * featMapImage->dimY;
    imagesMult = (float *)malloc(sizeof(float) * size);
    imagesMultRes = (float *)malloc(sizeof(float) * size);
    fftImagesMulti(featMapImage->channels[0], filterImage->channels[0], 
            featMapImage->dimY, featMapImage->dimX, imagesMultRes);
    for (i = 1; (i < (int)featMapImage->p) && (i < (int)filterImage->p); i++)
    {
        fftImagesMulti(featMapImage->channels[i],filterImage->channels[i], 
            featMapImage->dimY, featMapImage->dimX, imagesMult);
        for (j = 0; j < size; j++)
        {
            imagesMultRes[j] += imagesMult[j];
        }
    }
    fconv = (float *)malloc(sizeof(float) * size);
    fftInverse2d(imagesMultRes, fconv, featMapImage->dimY, featMapImage->dimX);
    diffX = featMapImage->dimX - filterDimX + 1;
    diffY = featMapImage->dimY - filterDimY + 1;
    *conv = (float *)malloc(sizeof(float) * (diffX * diffY));
    for (i = 0; i < diffY; i++)
    {
        for (j = 0; j < diffX; j++)
        {
            index = (i + filterDimY - 1) * featMapImage->dimX + 
                    (j + filterDimX - 1);
            (*conv)[i * diffX + j] = fconv[2 * index];
        }
    }
    free(imagesMult);
    free(imagesMultRes);
    free(fconv);
    return LATENT_SVM_OK;
}

/*
// Computation objective function D according the original paper
//
// API
// int filterDispositionLevel(const CvLSVMFilterObject *Fi, const featurePyramid *H, 
                              int level, float **scoreFi, 
                              int **pointsX, int **pointsY);
// INPUT
// Fi                - filter object (weights and coefficients of penalty 
                       function that are used in this routine)
// H                 - feature pyramid
// level             - level number
// OUTPUT
// scoreFi           - values of distance transform on the level at all positions
// (pointsX, pointsY)- positions that correspond to the maximum value 
                       of distance transform at all grid nodes
// RESULT
// Error status
*/
int filterDispositionLevel(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *pyramid,
                           float **scoreFi, 
                           int **pointsX, int **pointsY)
{
    int n1, m1, n2, m2, p, size, diff1, diff2;
    float *f;    
    int i1, j1;
    int res;
    
    n1 = pyramid->sizeY;
    m1 = pyramid->sizeX;
    n2 = Fi->sizeY;
    m2 = Fi->sizeX;
    p = pyramid->p;
    (*scoreFi) = NULL;
    (*pointsX) = NULL;
    (*pointsY) = NULL;
    
    // Processing the situation when part filter goes 
    // beyond the boundaries of the block set
    if (n1 < n2 || m1 < m2)
    {
        return FILTER_OUT_OF_BOUNDARIES;
    } /* if (n1 < n2 || m1 < m2) */

    // Computation number of positions for the filter
    diff1 = n1 - n2 + 1;
    diff2 = m1 - m2 + 1;
    size = diff1 * diff2;

    // Allocation memory for additional array (must be free in this function)
    f = (float *)malloc(sizeof(float) * size);       
    // Allocation memory for arrays for saving decisions
    (*scoreFi) = (float *)malloc(sizeof(float) * size);
    (*pointsX) = (int *)malloc(sizeof(int) * size);
    (*pointsY) = (int *)malloc(sizeof(int) * size);

    // Consruction values of the array f 
    // (a dot product vectors of feature map and weights of the filter)
    res = convolution(Fi, pyramid, f); 
    if (res != LATENT_SVM_OK)
    {
        free(f);
        free(*scoreFi);
        free(*pointsX);
        free(*pointsY);
        return res;
    }

    // TODO: necessary to change
    for (i1 = 0; i1 < diff1; i1++)
    {
         for (j1 = 0; j1 < diff2; j1++)
         {
             f[i1 * diff2 + j1] *= (-1);
         }   
    }

    // Decision of the general distance transform task 
    DistanceTransformTwoDimensionalProblem(f, diff1, diff2, Fi->fineFunction, 
                                          (*scoreFi), (*pointsX), (*pointsY));

    // Release allocated memory
    free(f);
    return LATENT_SVM_OK;
}

/*
// Computation objective function D according the original paper using FFT
//
// API
// int filterDispositionLevelFFT(const CvLSVMFilterObject *Fi, const fftImage *featMapImage,
                                 float **scoreFi, 
                                 int **pointsX, int **pointsY);
// INPUT
// Fi                - filter object (weights and coefficients of penalty 
                       function that are used in this routine)
// featMapImage      - FFT image of feature map
// OUTPUT
// scoreFi           - values of distance transform on the level at all positions
// (pointsX, pointsY)- positions that correspond to the maximum value 
                       of distance transform at all grid nodes
// RESULT
// Error status
*/
int filterDispositionLevelFFT(const CvLSVMFilterObject *Fi, const CvLSVMFftImage *featMapImage,
                              float **scoreFi, 
                              int **pointsX, int **pointsY)
{
    int n1, m1, n2, m2, p, size, diff1, diff2;
    float *f;    
    int i1, j1;
    int res;
    CvLSVMFftImage *filterImage;
    
    n1 = featMapImage->dimY;
    m1 = featMapImage->dimX;
    n2 = Fi->sizeY;
    m2 = Fi->sizeX;
    p = featMapImage->p;
    (*scoreFi) = NULL;
    (*pointsX) = NULL;
    (*pointsY) = NULL;
    
    // Processing the situation when part filter goes 
    // beyond the boundaries of the block set
    if (n1 < n2 || m1 < m2)
    {
        return FILTER_OUT_OF_BOUNDARIES;
    } /* if (n1 < n2 || m1 < m2) */

    // Computation number of positions for the filter
    diff1 = n1 - n2 + 1;
    diff2 = m1 - m2 + 1;
    size = diff1 * diff2;

    // Allocation memory for arrays for saving decisions
    (*scoreFi) = (float *)malloc(sizeof(float) * size);
    (*pointsX) = (int *)malloc(sizeof(int) * size);
    (*pointsY) = (int *)malloc(sizeof(int) * size);

    // create filter image
    getFFTImageFilterObject(Fi, featMapImage->dimX, featMapImage->dimY, &filterImage);

    // Consruction values of the array f 
    // (a dot product vectors of feature map and weights of the filter)
    res = convFFTConv2d(featMapImage, filterImage, Fi->sizeX, Fi->sizeY, &f);
    if (res != LATENT_SVM_OK)
    {
        free(f);
        free(*scoreFi);
        free(*pointsX);
        free(*pointsY);
        return res;
    }

    // TODO: necessary to change
    for (i1 = 0; i1 < diff1; i1++)
    {
         for (j1 = 0; j1 < diff2; j1++)
         {
             f[i1 * diff2 + j1] *= (-1);
         }   
    }

    // Decision of the general distance transform task 
    DistanceTransformTwoDimensionalProblem(f, diff1, diff2, Fi->fineFunction, 
                                          (*scoreFi), (*pointsX), (*pointsY));

    // Release allocated memory
    free(f);
    freeFFTImage(&filterImage);
    return LATENT_SVM_OK;
}

/*
// Computation border size for feature map
//
// API
// int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by);
// INPUT
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// bx                - border size (X-direction)
// by                - border size (Y-direction)
// RESULT
// Error status
*/
int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by)
{
    *bx = (int)ceilf(((float) maxXBorder) / 2.0f + 1.0f);
    *by = (int)ceilf(((float) maxYBorder) / 2.0f + 1.0f);
    return LATENT_SVM_OK;
}

/*
// Addition nullable border to the feature map
//
// API
// int addNullableBorder(featureMap *map, int bx, int by);
// INPUT
// map               - feature map
// bx                - border size (X-direction)
// by                - border size (Y-direction)
// OUTPUT
// RESULT
// Error status
*/
int addNullableBorder(CvLSVMFeatureMap *map, int bx, int by)
{
    int sizeX, sizeY, i, j, k;
    float *new_map;
    sizeX = map->sizeX + 2 * bx;
    sizeY = map->sizeY + 2 * by;
    new_map = (float *)malloc(sizeof(float) * sizeX * sizeY * map->p);
    for (i = 0; i < sizeX * sizeY * map->p; i++)
    {
        new_map[i] = 0.0;
    }
    for (i = by; i < map->sizeY + by; i++)
    {
        for (j = bx; j < map->sizeX + bx; j++)
        {
            for (k = 0; k < map->p; k++)
            {
                new_map[(i * sizeX + j) * map->p + k] = 
                    map->Map[((i - by) * map->sizeX + j - bx) * map->p + k];
            }
        }
    }
    map->sizeX = sizeX;
    map->sizeY = sizeY;
    free(map->Map);
    map->Map = new_map;
    return LATENT_SVM_OK;
}

CvLSVMFeatureMap* featureMapBorderPartFilter(CvLSVMFeatureMap *map, 
                                       int maxXBorder, int maxYBorder)
{
    int bx, by;
    int sizeX, sizeY, i, j, k;
    CvLSVMFeatureMap *new_map;
    
    computeBorderSize(maxXBorder, maxYBorder, &bx, &by);
    sizeX = map->sizeX + 2 * bx;
    sizeY = map->sizeY + 2 * by;
    allocFeatureMapObject(&new_map, sizeX, sizeY, map->p, map->xp);
    for (i = 0; i < sizeX * sizeY * map->p; i++)
    {
        new_map->Map[i] = 0.0;
    }
    for (i = by; i < map->sizeY + by; i++)
    {
        for (j = bx; j < map->sizeX + bx; j++)
        {
            for (k = 0; k < map->p; k++)
            {
                new_map->Map[(i * sizeX + j) * map->p + k] = 
                    map->Map[((i - by) * map->sizeX + j - bx) * map->p + k];
            }
        }
    }
    return new_map;
}

/*
// Computation the maximum of the score function at the level
//
// API
// int maxFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n, 
                                    const featurePyramid *H, 
                                    int level, float b, 
                                    int maxXBorder, int maxYBorder,                                 
                                    float *score, CvPoint **points, int *kPoints,
                                    CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// level             - feature pyramid level for computation maximum score
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// score             - the maximum of the score function at the level
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int maxFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n, 
                                 const CvLSVMFeaturePyramid *H, 
                                 int level, float b, 
                                 int maxXBorder, int maxYBorder,                                 
                                 float *score, CvPoint **points, 
                                 int *kPoints, CvPoint ***partsDisplacement)
{
    int i, j, k, dimX, dimY, nF0, mF0, p;
    int diff1, diff2, index, last, partsLevel;
    CvLSVMFilterDisposition **disposition;
    float *f;
    float *scores;
    float sumScorePartDisposition, maxScore;
    int res;
    CvLSVMFeatureMap *map;
#ifdef FFT_CONV
    CvLSVMFftImage *rootFilterImage, *mapImage;
#else
#endif

    /*
    // DEBUG variables
    FILE *file;
    char *tmp;
    char buf[40] = "..\\Data\\score\\score", buf1[10] = ".csv";
    tmp = (char *)malloc(sizeof(char) * 80);
    itoa(level, tmp, 10);
    strcat(tmp, buf1);
    //*/

    // Feature map matrix dimension on the level
    dimX = H->pyramid[level]->sizeX;
    dimY = H->pyramid[level]->sizeY;

    // Number of features
    p = H->pyramid[level]->p;
        
    // Getting dimension of root filter
    nF0 = all_F[0]->sizeY;
    mF0 = all_F[0]->sizeX;
    // Processing the situation when root filter goes 
    // beyond the boundaries of the block set
    if (nF0 > dimY || mF0 > dimX)
    {
        return LATENT_SVM_FAILED_SUPERPOSITION;
    }
        
    diff1 = dimY - nF0 + 1;
    diff2 = dimX - mF0 + 1;   
   
    // Allocation memory for saving values of function D 
    // on the level for each part filter
    disposition = (CvLSVMFilterDisposition **)malloc(sizeof(CvLSVMFilterDisposition *) * n);
    for (i = 0; i < n; i++)
    {
        disposition[i] = (CvLSVMFilterDisposition *)malloc(sizeof(CvLSVMFilterDisposition));
    }  

    // Allocation memory for values of score function for each block on the level
    scores = (float *)malloc(sizeof(float) * (diff1 * diff2));
    
    // A dot product vectors of feature map and weights of root filter
#ifdef FFT_CONV
    getFFTImageFeatureMap(H->pyramid[level], &mapImage);
    getFFTImageFilterObject(all_F[0], H->pyramid[level]->sizeX, H->pyramid[level]->sizeY, &rootFilterImage);
    res = convFFTConv2d(mapImage, rootFilterImage, all_F[0]->sizeX, all_F[0]->sizeY, &f);
    freeFFTImage(&mapImage);
    freeFFTImage(&rootFilterImage);
#else
    // Allocation memory for saving a dot product vectors of feature map and 
    // weights of root filter
    f = (float *)malloc(sizeof(float) * (diff1 * diff2));
    // A dot product vectors of feature map and weights of root filter
    res = convolution(all_F[0], H->pyramid[level], f);
#endif
    if (res != LATENT_SVM_OK)
    {
        free(f);
        free(scores);
        for (i = 0; i < n; i++)
        {
            free(disposition[i]);
        }
        free(disposition);
        return res;
    }

    // Computation values of function D for each part filter 
    // on the level (level - LAMBDA)
    partsLevel = level - LAMBDA;
    // For feature map at the level 'partsLevel' add nullable border
    map = featureMapBorderPartFilter(H->pyramid[partsLevel], 
                                     maxXBorder, maxYBorder);
    
    // Computation the maximum of score function
    sumScorePartDisposition = 0.0;
#ifdef FFT_CONV
    getFFTImageFeatureMap(map, &mapImage);
    for (k = 1; k <= n; k++)
    {  
        filterDispositionLevelFFT(all_F[k], mapImage, 
                               &(disposition[k - 1]->score), 
                               &(disposition[k - 1]->x), 
                               &(disposition[k - 1]->y));
    }
    freeFFTImage(&mapImage);
#else
    for (k = 1; k <= n; k++)
    {        
        filterDispositionLevel(all_F[k], map, 
                               &(disposition[k - 1]->score), 
                               &(disposition[k - 1]->x), 
                               &(disposition[k - 1]->y));
    }
#endif
    scores[0] = f[0] - sumScorePartDisposition + b;
    maxScore = scores[0];
    (*kPoints) = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            sumScorePartDisposition = 0.0;
            for (k = 1; k <= n; k++)
            {                
                // This condition takes on a value true
                // when filter goes beyond the boundaries of block set
                if ((2 * i + all_F[k]->V.y < 
                            map->sizeY - all_F[k]->sizeY + 1) &&
                    (2 * j + all_F[k]->V.x < 
                            map->sizeX - all_F[k]->sizeX + 1))
                {
                    index = (2 * i + all_F[k]->V.y) * 
                                (map->sizeX - all_F[k]->sizeX + 1) + 
                            (2 * j + all_F[k]->V.x);
                    sumScorePartDisposition += disposition[k - 1]->score[index];
                } 
            }
            scores[i * diff2 + j] = f[i * diff2 + j] - sumScorePartDisposition + b;
            if (maxScore < scores[i * diff2 + j])
            {
                maxScore = scores[i * diff2 + j];
                (*kPoints) = 1;
            } 
            else if ((scores[i * diff2 + j] - maxScore) * 
                     (scores[i * diff2 + j] - maxScore) <= EPS)
            {
                (*kPoints)++;
            } /* if (maxScore < scores[i * diff2 + j]) */
        }
    }

    // Allocation memory for saving positions of root filter and part filters
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
    for (i = 0; i < (*kPoints); i++)
    {
        (*partsDisplacement)[i] = (CvPoint *)malloc(sizeof(CvPoint) * n);
    }

    /*// DEBUG
    strcat(buf, tmp);
    file = fopen(buf, "w+");
    //*/
    // Construction of the set of positions for root filter 
    // that correspond the maximum of score function on the level
    (*score) = maxScore;
    last = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            if ((scores[i * diff2 + j] - maxScore) * 
                (scores[i * diff2 + j] - maxScore) <= EPS)
            {
                (*points)[last].y = i;
                (*points)[last].x = j;
                for (k = 1; k <= n; k++)
                {                    
                    if ((2 * i + all_F[k]->V.y < 
                            map->sizeY - all_F[k]->sizeY + 1) &&
                        (2 * j + all_F[k]->V.x < 
                            map->sizeX - all_F[k]->sizeX + 1))
                    {
                        index = (2 * i + all_F[k]->V.y) * 
                                   (map->sizeX - all_F[k]->sizeX + 1) + 
                                (2 * j + all_F[k]->V.x);
                        (*partsDisplacement)[last][k - 1].x = 
                                              disposition[k - 1]->x[index];
                        (*partsDisplacement)[last][k - 1].y = 
                                              disposition[k - 1]->y[index];
                    } 
                }
                last++;
            } /* if ((scores[i * diff2 + j] - maxScore) * 
                     (scores[i * diff2 + j] - maxScore) <= EPS) */
            //fprintf(file, "%lf;", scores[i * diff2 + j]);
        }
        //fprintf(file, "\n");
    }
    //fclose(file);
    //free(tmp);
    
    // Release allocated memory
    for (i = 0; i < n ; i++)
    {
        free(disposition[i]->score);
        free(disposition[i]->x);
        free(disposition[i]->y);
        free(disposition[i]);
    }
    free(disposition);
    free(f);
    free(scores);
    freeFeatureMapObject(&map);
    return LATENT_SVM_OK;
}

/*
// Computation score function at the level that exceed threshold
//
// API
// int thresholdFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n, 
                                          const featurePyramid *H, 
                                          int level, float b, 
                                          int maxXBorder, int maxYBorder,
                                          float scoreThreshold,
                                          float **score, CvPoint **points, int *kPoints,
                                          CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// level             - feature pyramid level for computation maximum score
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// score             - score function at the level that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int thresholdFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n, 
                                       const CvLSVMFeaturePyramid *H, 
                                       int level, float b, 
                                       int maxXBorder, int maxYBorder,
                                       float scoreThreshold,
                                       float **score, CvPoint **points, int *kPoints,
                                       CvPoint ***partsDisplacement)
{
    int i, j, k, dimX, dimY, nF0, mF0, p;
    int diff1, diff2, index, last, partsLevel;
    CvLSVMFilterDisposition **disposition;
    float *f;
    float *scores;
    float sumScorePartDisposition;
    int res;
    CvLSVMFeatureMap *map;
#ifdef FFT_CONV
    CvLSVMFftImage *rootFilterImage, *mapImage;
#else
#endif
    /*
    // DEBUG variables
    FILE *file;
    char *tmp;
    char buf[40] = "..\\Data\\score\\score", buf1[10] = ".csv";
    tmp = (char *)malloc(sizeof(char) * 80);
    itoa(level, tmp, 10);
    strcat(tmp, buf1);
    //*/

    // Feature map matrix dimension on the level
    dimX = H->pyramid[level]->sizeX;
    dimY = H->pyramid[level]->sizeY;

    // Number of features
    p = H->pyramid[level]->p;
        
    // Getting dimension of root filter
    nF0 = all_F[0]->sizeY;
    mF0 = all_F[0]->sizeX;
    // Processing the situation when root filter goes 
    // beyond the boundaries of the block set
    if (nF0 > dimY || mF0 > dimX)
    {
        return LATENT_SVM_FAILED_SUPERPOSITION;
    }
        
    diff1 = dimY - nF0 + 1;
    diff2 = dimX - mF0 + 1;   
   
    // Allocation memory for saving values of function D 
    // on the level for each part filter
    disposition = (CvLSVMFilterDisposition **)malloc(sizeof(CvLSVMFilterDisposition *) * n);
    for (i = 0; i < n; i++)
    {
        disposition[i] = (CvLSVMFilterDisposition *)malloc(sizeof(CvLSVMFilterDisposition));
    }  

    // Allocation memory for values of score function for each block on the level
    scores = (float *)malloc(sizeof(float) * (diff1 * diff2));
    // A dot product vectors of feature map and weights of root filter
#ifdef FFT_CONV
    getFFTImageFeatureMap(H->pyramid[level], &mapImage);
    getFFTImageFilterObject(all_F[0], H->pyramid[level]->sizeX, H->pyramid[level]->sizeY, &rootFilterImage);
    res = convFFTConv2d(mapImage, rootFilterImage, all_F[0]->sizeX, all_F[0]->sizeY, &f);
    freeFFTImage(&mapImage);
    freeFFTImage(&rootFilterImage);
#else
    // Allocation memory for saving a dot product vectors of feature map and 
    // weights of root filter
    f = (float *)malloc(sizeof(float) * (diff1 * diff2));
    res = convolution(all_F[0], H->pyramid[level], f);
#endif
    if (res != LATENT_SVM_OK)
    {
        free(f);
        free(scores);
        for (i = 0; i < n; i++)
        {
            free(disposition[i]);
        }
        free(disposition);
        return res;
    }

    // Computation values of function D for each part filter 
    // on the level (level - LAMBDA)
    partsLevel = level - LAMBDA;
    // For feature map at the level 'partsLevel' add nullable border
    map = featureMapBorderPartFilter(H->pyramid[partsLevel], 
                                     maxXBorder, maxYBorder);
    
    // Computation the maximum of score function
    sumScorePartDisposition = 0.0;
#ifdef FFT_CONV
    getFFTImageFeatureMap(map, &mapImage);
    for (k = 1; k <= n; k++)
    {  
        filterDispositionLevelFFT(all_F[k], mapImage, 
                               &(disposition[k - 1]->score), 
                               &(disposition[k - 1]->x), 
                               &(disposition[k - 1]->y));
    }
    freeFFTImage(&mapImage);
#else
    for (k = 1; k <= n; k++)
    {        
        filterDispositionLevel(all_F[k], map, 
                               &(disposition[k - 1]->score), 
                               &(disposition[k - 1]->x), 
                               &(disposition[k - 1]->y));
    }
#endif
    (*kPoints) = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            sumScorePartDisposition = 0.0;
            for (k = 1; k <= n; k++)
            {                
                // This condition takes on a value true
                // when filter goes beyond the boundaries of block set
                if ((2 * i + all_F[k]->V.y < 
                            map->sizeY - all_F[k]->sizeY + 1) &&
                    (2 * j + all_F[k]->V.x < 
                            map->sizeX - all_F[k]->sizeX + 1))
                {
                    index = (2 * i + all_F[k]->V.y) * 
                                (map->sizeX - all_F[k]->sizeX + 1) + 
                            (2 * j + all_F[k]->V.x);
                    sumScorePartDisposition += disposition[k - 1]->score[index];
                } 
            }
            scores[i * diff2 + j] = f[i * diff2 + j] - sumScorePartDisposition + b;
            if (scores[i * diff2 + j] > scoreThreshold)                
            {
                (*kPoints)++;
            }
        }
    }

    // Allocation memory for saving positions of root filter and part filters
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
    for (i = 0; i < (*kPoints); i++)
    {
        (*partsDisplacement)[i] = (CvPoint *)malloc(sizeof(CvPoint) * n);
    }

    /*// DEBUG
    strcat(buf, tmp);
    file = fopen(buf, "w+");
    //*/
    // Construction of the set of positions for root filter 
    // that correspond score function on the level that exceed threshold
    (*score) = (float *)malloc(sizeof(float) * (*kPoints));
    last = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            if (scores[i * diff2 + j] > scoreThreshold) 
            {
                (*score)[last] = scores[i * diff2 + j];
                (*points)[last].y = i;
                (*points)[last].x = j;
                for (k = 1; k <= n; k++)
                {                    
                    if ((2 * i + all_F[k]->V.y < 
                            map->sizeY - all_F[k]->sizeY + 1) &&
                        (2 * j + all_F[k]->V.x < 
                            map->sizeX - all_F[k]->sizeX + 1))
                    {
                        index = (2 * i + all_F[k]->V.y) * 
                                   (map->sizeX - all_F[k]->sizeX + 1) + 
                                (2 * j + all_F[k]->V.x);
                        (*partsDisplacement)[last][k - 1].x = 
                                              disposition[k - 1]->x[index];
                        (*partsDisplacement)[last][k - 1].y = 
                                              disposition[k - 1]->y[index];
                    } 
                }
                last++;
            }
            //fprintf(file, "%lf;", scores[i * diff2 + j]);
        }
        //fprintf(file, "\n");
    }
    //fclose(file);
    //free(tmp);

    // Release allocated memory
    for (i = 0; i < n ; i++)
    {
        free(disposition[i]->score);
        free(disposition[i]->x);
        free(disposition[i]->y);
        free(disposition[i]);
    }
    free(disposition);
    free(f);
    free(scores);
    freeFeatureMapObject(&map);
    return LATENT_SVM_OK;
}

/*
// Computation the maximum of the score function
//
// API
// int maxFunctionalScore(const CvLSVMFilterObject **all_F, int n, 
                          const featurePyramid *H, float b, 
                          int maxXBorder, int maxYBorder,
                          float *score, 
                          CvPoint **points, int **levels, int *kPoints,
                          CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// score             - the maximum of the score function
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int maxFunctionalScore(const CvLSVMFilterObject **all_F, int n, 
                       const CvLSVMFeaturePyramid *H, float b, 
                       int maxXBorder, int maxYBorder,
                       float *score, 
                       CvPoint **points, int **levels, int *kPoints,
                       CvPoint ***partsDisplacement)
{
    int l, i, j, k, s, f, level, numLevels;
    float *tmpScore;
    CvPoint ***tmpPoints;
    CvPoint ****tmpPartsDisplacement;   
    int *tmpKPoints;
    float maxScore;
    int res;

    /* DEBUG
    FILE *file;
    //*/
    
    // Computation the number of levels for seaching object,
    // first lambda-levels are used for computation values
    // of score function for each position of root filter
    numLevels = H->countLevel - H->lambda;
    
    // Allocation memory for maximum value of score function for each level
    tmpScore = (float *)malloc(sizeof(float) * numLevels);        
    // Allocation memory for the set of points that corresponds 
    // to the maximum of score function
    tmpPoints = (CvPoint ***)malloc(sizeof(CvPoint **) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPoints[i] = (CvPoint **)malloc(sizeof(CvPoint *));
    }
    // Allocation memory for memory for saving parts displacement on each level
    tmpPartsDisplacement = (CvPoint ****)malloc(sizeof(CvPoint ***) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPartsDisplacement[i] = (CvPoint ***)malloc(sizeof(CvPoint **));
    }
    // Number of points that corresponds to the maximum 
    // of score function on each level
    tmpKPoints = (int *)malloc(sizeof(int) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpKPoints[i] = 0;
    }

    // Set current value of the maximum of score function
    res = maxFunctionalScoreFixedLevel(all_F, n, H, H->lambda, b, 
            maxXBorder, maxYBorder,
            &(tmpScore[0]), 
            tmpPoints[0], 
            &(tmpKPoints[0]), 
            tmpPartsDisplacement[0]);
    maxScore = tmpScore[0];
    (*kPoints) = tmpKPoints[0];

    // Computation maxima of score function on each level
    // and getting the maximum on all levels
    /* DEBUG: maxScore
    file = fopen("maxScore.csv", "w+");
    fprintf(file, "%i;%lf;\n", H->lambda, tmpScore[0]);    
    //*/
    for (l = H->lambda + 1; l < H->countLevel; l++)
    {        
        k = l - H->lambda;
        res = maxFunctionalScoreFixedLevel(all_F, n, H, l, b,
                                           maxXBorder, maxYBorder,
                                           &(tmpScore[k]), 
                                           tmpPoints[k], 
                                           &(tmpKPoints[k]), 
                                           tmpPartsDisplacement[k]);        
        //fprintf(file, "%i;%lf;\n", l, tmpScore[k]);    
        if (res != LATENT_SVM_OK)
        {
            continue;
        }
        if (maxScore < tmpScore[k])
        {
            maxScore = tmpScore[k];
            (*kPoints) = tmpKPoints[k];
        }
        else if ((maxScore - tmpScore[k]) * (maxScore - tmpScore[k]) <= EPS)
        {
            (*kPoints) += tmpKPoints[k];
        } /* if (maxScore < tmpScore[k]) else if (...)*/
    }
    //fclose(file);

    // Allocation memory for levels
    (*levels) = (int *)malloc(sizeof(int) * (*kPoints));
    // Allocation memory for the set of points
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));   
    // Allocation memory for parts displacement
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));

    // Filling the set of points, levels and parts displacement
    s = 0;
    f = 0;
    for (i = 0; i < numLevels; i++)
    {
        if ((tmpScore[i] - maxScore) * (tmpScore[i] - maxScore) <= EPS)
        {
            // Computation the number of level
            level = i + H->lambda; 

            // Addition a set of points
            f += tmpKPoints[i];
            for (j = s; j < f; j++)
            {
                (*levels)[j] = level;
                (*points)[j] = (*tmpPoints[i])[j - s];
                (*partsDisplacement)[j] = (*(tmpPartsDisplacement[i]))[j - s];
            }            
            s = f;
        } /* if ((tmpScore[i] - maxScore) * (tmpScore[i] - maxScore) <= EPS) */
    }
    (*score) = maxScore;    

    // Release allocated memory
    for (i = 0; i < numLevels; i++)
    {
        free(tmpPoints[i]);
        free(tmpPartsDisplacement[i]);
    }
    free(tmpPoints);
    free(tmpScore);
    free(tmpKPoints);
    
    return LATENT_SVM_OK;   
}

/*
// Computation score function that exceed threshold
//
// API
// int thresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n, 
                                const featurePyramid *H, 
                                float b, 
                                int maxXBorder, int maxYBorder,
                                float scoreThreshold,
                                float **score, 
                                CvPoint **points, int **levels, int *kPoints,
                                CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// score             - score function values that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int thresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n, 
                             const CvLSVMFeaturePyramid *H, 
                             float b, 
                             int maxXBorder, int maxYBorder,
                             float scoreThreshold,
                             float **score, 
                             CvPoint **points, int **levels, int *kPoints,
                             CvPoint ***partsDisplacement)
{
    int l, i, j, k, s, f, level, numLevels;
    float **tmpScore;
    CvPoint ***tmpPoints;
    CvPoint ****tmpPartsDisplacement;   
    int *tmpKPoints;
    int res;

    /* DEBUG
    FILE *file;
    //*/
    
    // Computation the number of levels for seaching object,
    // first lambda-levels are used for computation values
    // of score function for each position of root filter
    numLevels = H->countLevel - H->lambda;
    
    // Allocation memory for values of score function for each level
    // that exceed threshold
    tmpScore = (float **)malloc(sizeof(float*) * numLevels);        
    // Allocation memory for the set of points that corresponds 
    // to the maximum of score function
    tmpPoints = (CvPoint ***)malloc(sizeof(CvPoint **) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPoints[i] = (CvPoint **)malloc(sizeof(CvPoint *));
    }
    // Allocation memory for memory for saving parts displacement on each level
    tmpPartsDisplacement = (CvPoint ****)malloc(sizeof(CvPoint ***) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPartsDisplacement[i] = (CvPoint ***)malloc(sizeof(CvPoint **));
    }
    // Number of points that corresponds to the maximum 
    // of score function on each level
    tmpKPoints = (int *)malloc(sizeof(int) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpKPoints[i] = 0;
    }

    // Computation maxima of score function on each level
    // and getting the maximum on all levels
    /* DEBUG: maxScore
    file = fopen("maxScore.csv", "w+");
    fprintf(file, "%i;%lf;\n", H->lambda, tmpScore[0]);    
    //*/
    (*kPoints) = 0;
    for (l = H->lambda; l < H->countLevel; l++)
    {        
        k = l - H->lambda;
        //printf("Score at the level %i\n", l);
        res = thresholdFunctionalScoreFixedLevel(all_F, n, H, l, b, 
            maxXBorder, maxYBorder, scoreThreshold,
            &(tmpScore[k]), 
            tmpPoints[k], 
            &(tmpKPoints[k]), 
            tmpPartsDisplacement[k]);
        //fprintf(file, "%i;%lf;\n", l, tmpScore[k]);    
        if (res != LATENT_SVM_OK)
        {
            continue;
        }
        (*kPoints) += tmpKPoints[k];
    }
    //fclose(file);
    
    // Allocation memory for levels
    (*levels) = (int *)malloc(sizeof(int) * (*kPoints));
    // Allocation memory for the set of points
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));   
    // Allocation memory for parts displacement
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
    // Allocation memory for score function values
    (*score) = (float *)malloc(sizeof(float) * (*kPoints));

    // Filling the set of points, levels and parts displacement
    s = 0;
    f = 0;
    for (i = 0; i < numLevels; i++)
    {
        // Computation the number of level
        level = i + H->lambda; 

        // Addition a set of points
        f += tmpKPoints[i];
        for (j = s; j < f; j++)
        {
            (*levels)[j] = level;
            (*points)[j] = (*tmpPoints[i])[j - s];
            (*score)[j] = tmpScore[i][j - s];
            (*partsDisplacement)[j] = (*(tmpPartsDisplacement[i]))[j - s];
        }            
        s = f;
    }

    // Release allocated memory
    for (i = 0; i < numLevels; i++)
    {
        free(tmpPoints[i]);
        free(tmpPartsDisplacement[i]);
    }
    free(tmpPoints);
    free(tmpScore);
    free(tmpKPoints);
    free(tmpPartsDisplacement);
    
    return LATENT_SVM_OK;  
}

/*
// Creating schedule of pyramid levels processing 
//
// API
// int createSchedule(const featurePyramid *H, const filterObject **all_F,
                      const int n, const int bx, const int by,
                      const int threadsNum, int *kLevels, 
                      int **processingLevels)
// INPUT
// H                 - feature pyramid
// all_F             - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// bx                - size of nullable border (X direction)
// by                - size of nullable border (Y direction)
// threadsNum        - number of threads that will be created in TBB version
// OUTPUT
// kLevels           - array that contains number of levels processed 
                       by each thread
// processingLevels  - array that contains lists of levels processed 
                       by each thread
// RESULT
// Error status
*/
int createSchedule(const CvLSVMFeaturePyramid *H, const CvLSVMFilterObject **all_F,
                   const int n, const int bx, const int by,
                   const int threadsNum, int *kLevels, int **processingLevels)
{
    int rootFilterDim, sumPartFiltersDim, i, numLevels, dbx, dby, numDotProducts;
    int averNumDotProd, j, minValue, argMin, tmp, lambda, maxValue, k;
    int *dotProd, *weights, *disp;
    if (H == NULL || all_F == NULL)
    {
        return LATENT_SVM_TBB_SCHEDULE_CREATION_FAILED;
    }
    // Number of feature vectors in root filter
    rootFilterDim = all_F[0]->sizeX * all_F[0]->sizeY;
    // Number of feature vectors in all part filters
    sumPartFiltersDim = 0;
    for (i = 1; i <= n; i++)
    {
        sumPartFiltersDim += all_F[i]->sizeX * all_F[i]->sizeY;
    }
    // Number of levels which are used for computation of score function
    numLevels = H->countLevel - H->lambda;
    // Allocation memory for saving number of dot products that will be
    // computed for each level of feature pyramid
    dotProd = (int *)malloc(sizeof(int) * numLevels);
    // Size of nullable border that's used in computing convolution
    // of feature map with part filter
    dbx = 2 * bx;
    dby = 2 * by;
    // Total number of dot products for all levels
    numDotProducts = 0;
    lambda = H->lambda;
    for (i = 0; i < numLevels; i++)
    {
        dotProd[i] = H->pyramid[i + lambda]->sizeX * 
                     H->pyramid[i + lambda]->sizeY * rootFilterDim +
                     (H->pyramid[i]->sizeX + dbx) * 
                     (H->pyramid[i]->sizeY + dby) * sumPartFiltersDim;
        numDotProducts += dotProd[i];
    }
    // Average number of dot products that would be performed at the best
    averNumDotProd = numDotProducts / threadsNum;
    // Allocation memory for saving dot product number performed by each thread
    weights = (int *)malloc(sizeof(int) * threadsNum);
    // Allocation memory for saving dispertion
    disp = (int *)malloc(sizeof(int) * threadsNum);
    // At the first step we think of first threadsNum levels will be processed
    // by different threads
    for (i = 0; i < threadsNum; i++)
    {
        kLevels[i] = 1;
        weights[i] = dotProd[i];
        disp[i] = 0;
    }
    // Computation number of levels that will be processed by each thread
    for (i = threadsNum; i < numLevels; i++)
    {
        // Search number of thread that will process level number i
        for (j = 0; j < threadsNum; j++)
        {
            weights[j] += dotProd[i];
            minValue = weights[0];
            maxValue = weights[0];
            for (k = 1; k < threadsNum; k++)
            {
                minValue = min(minValue, weights[k]);
                maxValue = max(maxValue, weights[k]);
            }
            disp[j] = maxValue - minValue;
            weights[j] -= dotProd[i];
        }
        minValue = disp[0];
        argMin = 0;
        for (j = 1; j < threadsNum; j++)
        {
            if (disp[j] < minValue)
            {
                minValue = disp[j];
                argMin = j;
            }
        }
        // Addition new level
        kLevels[argMin]++;
        weights[argMin] += dotProd[i];
    }
    for (i = 0; i < threadsNum; i++)
    {
        // Allocation memory for saving list of levels for each level
        processingLevels[i] = (int *)malloc(sizeof(int) * kLevels[i]);
        // At the first step we think of first threadsNum levels will be processed
        // by different threads
        processingLevels[i][0] = lambda + i;
        kLevels[i] = 1;
        weights[i] = dotProd[i];
    }
    // Creating list of levels
    for (i = threadsNum; i < numLevels; i++)
    {
        for (j = 0; j < threadsNum; j++)
        {
            weights[j] += dotProd[i];
            minValue = weights[0];
            maxValue = weights[0];
            for (k = 1; k < threadsNum; k++)
            {
                minValue = min(minValue, weights[k]);
                maxValue = max(maxValue, weights[k]);
            }
            disp[j] = maxValue - minValue;
            weights[j] -= dotProd[i];
        }
        minValue = disp[0];
        argMin = 0;
        for (j = 1; j < threadsNum; j++)
        {
            if (disp[j] < minValue)
            {
                minValue = disp[j];
                argMin = j;
            }
        }
        processingLevels[argMin][kLevels[argMin]] = lambda + i;
        kLevels[argMin]++;
        weights[argMin] += dotProd[i];
    }
    // Release allocated memory
    free(weights);
    free(dotProd);
    free(disp);
    return LATENT_SVM_OK;
}

#ifdef HAVE_TBB
/*
// int tbbThresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n, 
                                   const CvLSVMFeaturePyramid *H, 
                                   const float b, 
                                   const int maxXBorder, const int maxYBorder,
                                   const float scoreThreshold,
                                   const int threadsNum,
                                   float **score, 
                                   CvPoint **points, int **levels, int *kPoints,
                                   CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// threadsNum        - number of threads that will be created using TBB version
// OUTPUT
// score             - score function values that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int tbbThresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n, 
                                const CvLSVMFeaturePyramid *H, 
                                const float b, 
                                const int maxXBorder, const int maxYBorder,
                                const float scoreThreshold,
                                const int threadsNum,
                                float **score, 
                                CvPoint **points, int **levels, int *kPoints,
                                CvPoint ***partsDisplacement)
{
    int i, j, s, f, level, numLevels;
    float **tmpScore;
    CvPoint ***tmpPoints;
    CvPoint ****tmpPartsDisplacement;   
    int *tmpKPoints;
    int res;

    int *kLevels, **procLevels;
    int bx, by;
    
    // Computation the number of levels for seaching object,
    // first lambda-levels are used for computation values
    // of score function for each position of root filter
    numLevels = H->countLevel - H->lambda;

    kLevels = (int *)malloc(sizeof(int) * threadsNum);
    procLevels = (int **)malloc(sizeof(int*) * threadsNum);
    computeBorderSize(maxXBorder, maxYBorder, &bx, &by);
    res = createSchedule(H, all_F, n, bx, by, threadsNum, kLevels, procLevels);
    if (res != LATENT_SVM_OK)
    {
        for (i = 0; i < threadsNum; i++)
        {
            if (procLevels[i] != NULL) 
            {
                free(procLevels[i]);
            }
        }
        free(procLevels);
        free(kLevels);
        return res;
    }
    
    // Allocation memory for values of score function for each level
    // that exceed threshold
    tmpScore = (float **)malloc(sizeof(float*) * numLevels);        
    // Allocation memory for the set of points that corresponds 
    // to the maximum of score function
    tmpPoints = (CvPoint ***)malloc(sizeof(CvPoint **) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPoints[i] = (CvPoint **)malloc(sizeof(CvPoint *));
    }
    // Allocation memory for memory for saving parts displacement on each level
    tmpPartsDisplacement = (CvPoint ****)malloc(sizeof(CvPoint ***) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPartsDisplacement[i] = (CvPoint ***)malloc(sizeof(CvPoint **));
    }
    // Number of points that corresponds to the maximum 
    // of score function on each level
    tmpKPoints = (int *)malloc(sizeof(int) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpKPoints[i] = 0;
    }

    // Computation maxima of score function on each level
    // and getting the maximum on all levels using TBB tasks
    tbbTasksThresholdFunctionalScore(all_F, n, H, b, maxXBorder, maxYBorder,
        scoreThreshold, kLevels, procLevels, 
        threadsNum, tmpScore, tmpPoints, 
        tmpKPoints, tmpPartsDisplacement);
    (*kPoints) = 0;
    for (i = 0; i < numLevels; i++)
    {
        (*kPoints) += tmpKPoints[i];
    }
        
    // Allocation memory for levels
    (*levels) = (int *)malloc(sizeof(int) * (*kPoints));
    // Allocation memory for the set of points
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));   
    // Allocation memory for parts displacement
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
    // Allocation memory for score function values
    (*score) = (float *)malloc(sizeof(float) * (*kPoints));

    // Filling the set of points, levels and parts displacement
    s = 0;
    f = 0;
    for (i = 0; i < numLevels; i++)
    {
        // Computation the number of level
        level = i + H->lambda; 

        // Addition a set of points
        f += tmpKPoints[i];
        for (j = s; j < f; j++)
        {
            (*levels)[j] = level;
            (*points)[j] = (*tmpPoints[i])[j - s];
            (*score)[j] = tmpScore[i][j - s];
            (*partsDisplacement)[j] = (*(tmpPartsDisplacement[i]))[j - s];
        }            
        s = f;
    }

    // Release allocated memory
    for (i = 0; i < numLevels; i++)
    {
        free(tmpPoints[i]);
        free(tmpPartsDisplacement[i]);
    }
    for (i = 0; i < threadsNum; i++)
    {
        free(procLevels[i]);
    }
    free(procLevels);
    free(kLevels);
    free(tmpPoints);
    free(tmpScore);
    free(tmpKPoints);
    free(tmpPartsDisplacement);

    return LATENT_SVM_OK;
}
#endif

void sort(int n, const float* x, int* indices)
{
	int i, j;
	for (i = 0; i < n; i++)
		for (j = i + 1; j < n; j++)
		{
			if (x[indices[j]] > x[indices[i]])
			{
				//float x_tmp = x[i];
				int index_tmp = indices[i];
				//x[i] = x[j];
				indices[i] = indices[j];
				//x[j] = x_tmp;
				indices[j] = index_tmp;
			}
		}
}

/*
// Perform non-maximum suppression algorithm (described in original paper)
// to remove "similar" bounding boxes
//
// API
// int nonMaximumSuppression(int numBoxes, const CvPoint *points, 
                             const CvPoint *oppositePoints, const float *score,
                             float overlapThreshold, 
                             int *numBoxesOut, CvPoint **pointsOut, 
                             CvPoint **oppositePointsOut, float **scoreOut);
// INPUT
// numBoxes          - number of bounding boxes
// points            - array of left top corner coordinates
// oppositePoints    - array of right bottom corner coordinates
// score             - array of detection scores
// overlapThreshold  - threshold: bounding box is removed if overlap part 
					   is greater than passed value
// OUTPUT
// numBoxesOut       - the number of bounding boxes algorithm returns
// pointsOut         - array of left top corner coordinates
// oppositePointsOut - array of right bottom corner coordinates
// scoreOut          - array of detection scores
// RESULT
// Error status
*/
int nonMaximumSuppression(int numBoxes, const CvPoint *points, 
                          const CvPoint *oppositePoints, const float *score,
                          float overlapThreshold, 
                          int *numBoxesOut, CvPoint **pointsOut, 
                          CvPoint **oppositePointsOut, float **scoreOut)
{
    int i, j, index;
	float* box_area = (float*)malloc(numBoxes * sizeof(float));
	int* indices = (int*)malloc(numBoxes * sizeof(int));
	int* is_suppressed = (int*)malloc(numBoxes * sizeof(int));
	
	for (i = 0; i < numBoxes; i++)
	{
		indices[i] = i;
		is_suppressed[i] = 0;
        box_area[i] = (float)( (oppositePoints[i].x - points[i].x + 1) * 
                                (oppositePoints[i].y - points[i].y + 1));
	}

	sort(numBoxes, score, indices);
	for (i = 0; i < numBoxes; i++)
	{
		if (!is_suppressed[indices[i]])
		{
			for (j = i + 1; j < numBoxes; j++)
			{
				if (!is_suppressed[indices[j]])
				{
                    int x1max = max(points[indices[i]].x, points[indices[j]].x);
                    int x2min = min(oppositePoints[indices[i]].x, oppositePoints[indices[j]].x);
                    int y1max = max(points[indices[i]].y, points[indices[j]].y);
                    int y2min = min(oppositePoints[indices[i]].y, oppositePoints[indices[j]].y);
					int overlapWidth = x2min - x1max + 1;
					int overlapHeight = y2min - y1max + 1;
					if (overlapWidth > 0 && overlapHeight > 0)
					{
						float overlapPart = (overlapWidth * overlapHeight) / box_area[indices[j]];
						if (overlapPart > overlapThreshold)
						{
							is_suppressed[indices[j]] = 1;
						}
					}
				}
			}
		}
	}

	*numBoxesOut = 0;
	for (i = 0; i < numBoxes; i++)
	{
		if (!is_suppressed[i]) (*numBoxesOut)++;
	}

    *pointsOut = (CvPoint *)malloc((*numBoxesOut) * sizeof(CvPoint));
    *oppositePointsOut = (CvPoint *)malloc((*numBoxesOut) * sizeof(CvPoint));
    *scoreOut = (float *)malloc((*numBoxesOut) * sizeof(float));
	index = 0;
	for (i = 0; i < numBoxes; i++)
	{
		if (!is_suppressed[indices[i]])
		{
            (*pointsOut)[index].x = points[indices[i]].x;
            (*pointsOut)[index].y = points[indices[i]].y;
            (*oppositePointsOut)[index].x = oppositePoints[indices[i]].x;
            (*oppositePointsOut)[index].y = oppositePoints[indices[i]].y;
			(*scoreOut)[index] = score[indices[i]];
			index++;
		}

	}

	free(indices);
	free(box_area);
	free(is_suppressed);

	return LATENT_SVM_OK;
}
