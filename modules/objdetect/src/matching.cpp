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
    new_map = (float *)malloc(sizeof(float) * sizeX * sizeY * map->numFeatures);
    for (i = 0; i < sizeX * sizeY * map->numFeatures; i++)
    {
        new_map[i] = 0.0;
    }
    for (i = by; i < map->sizeY + by; i++)
    {
        for (j = bx; j < map->sizeX + bx; j++)
        {
            for (k = 0; k < map->numFeatures; k++)
            {
                new_map[(i * sizeX + j) * map->numFeatures + k] =
                    map->map[((i - by) * map->sizeX + j - bx) * map->numFeatures + k];
            }
        }
    }
    map->sizeX = sizeX;
    map->sizeY = sizeY;
    free(map->map);
    map->map = new_map;
    return LATENT_SVM_OK;
}

/*
// Computation maximum filter size for each dimension
//
// API
// int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
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
        if (filters[componentIndex]->sizeX > *maxXBorder)
        {
            *maxXBorder = filters[componentIndex]->sizeX;
        }
        if (filters[componentIndex]->sizeY > *maxYBorder)
        {
            *maxYBorder = filters[componentIndex]->sizeY;
        }
        componentIndex += (kPartFilters[i] + 1);
    }
    return LATENT_SVM_OK;
}

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
