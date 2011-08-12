#include "precomp.hpp"
#include "_lsvm_routine.h"

int allocFilterObject(CvLSVMFilterObject **obj, const int sizeX,
                      const int sizeY, const int numFeatures) 
{
    int i;
    (*obj) = (CvLSVMFilterObject *)malloc(sizeof(CvLSVMFilterObject));
    (*obj)->sizeX           = sizeX;
    (*obj)->sizeY           = sizeY;
    (*obj)->numFeatures     = numFeatures;
    (*obj)->fineFunction[0] = 0.0f;
    (*obj)->fineFunction[1] = 0.0f;
    (*obj)->fineFunction[2] = 0.0f;
    (*obj)->fineFunction[3] = 0.0f;
    (*obj)->V.x         = 0;
    (*obj)->V.y         = 0;
    (*obj)->V.l         = 0;
    (*obj)->H = (float *) malloc(sizeof (float) * 
                                (sizeX * sizeY  * numFeatures));
    for(i = 0; i < sizeX * sizeY * numFeatures; i++)
    {
        (*obj)->H[i] = 0.0f;
    }
    return LATENT_SVM_OK;
}
int freeFilterObject (CvLSVMFilterObject **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->H);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeatureMapObject(CvLSVMFeatureMap **obj, const int sizeX, 
                          const int sizeY, const int numFeatures)
{
    int i;
    (*obj) = (CvLSVMFeatureMap *)malloc(sizeof(CvLSVMFeatureMap));
    (*obj)->sizeX       = sizeX;
    (*obj)->sizeY       = sizeY;
    (*obj)->numFeatures = numFeatures;
    (*obj)->map = (float *) malloc(sizeof (float) * 
                                  (sizeX * sizeY  * numFeatures));
    for(i = 0; i < sizeX * sizeY * numFeatures; i++)
    {
        (*obj)->map[i] = 0.0f;
    }
    return LATENT_SVM_OK;
}
int freeFeatureMapObject (CvLSVMFeatureMap **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->map);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeaturePyramidObject(CvLSVMFeaturePyramid **obj,
                              const int numLevels) 
{
    (*obj) = (CvLSVMFeaturePyramid *)malloc(sizeof(CvLSVMFeaturePyramid));
    (*obj)->numLevels = numLevels;
    (*obj)->pyramid    = (CvLSVMFeatureMap **)malloc(
                         sizeof(CvLSVMFeatureMap *) * numLevels);
    return LATENT_SVM_OK;
}

int freeFeaturePyramidObject (CvLSVMFeaturePyramid **obj)
{
    int i; 
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    for(i = 0; i < (*obj)->numLevels; i++)
    {
        freeFeatureMapObject(&((*obj)->pyramid[i]));
    }
    free((*obj)->pyramid);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFFTImage(CvLSVMFftImage **image, int numFeatures, int dimX, int dimY)
{
    int i, j, size;
    *image = (CvLSVMFftImage *)malloc(sizeof(CvLSVMFftImage));
    (*image)->numFeatures = numFeatures;
    (*image)->dimX         = dimX;
    (*image)->dimY         = dimY;
    (*image)->channels     = (float **)malloc(sizeof(float *) * numFeatures);
    size = 2 * dimX * dimY;
    for (i = 0; i < numFeatures; i++)
    {
        (*image)->channels[i] = (float *)malloc(sizeof(float) * size);
        for (j = 0; j < size; j++)
        {
            (*image)->channels[i][j] = 0.0f;
        }
    }
    return LATENT_SVM_OK;
}

int freeFFTImage(CvLSVMFftImage **image)
{
    int i;
    if (*image == NULL) return LATENT_SVM_OK;
    for (i = 0; i < (*image)->numFeatures; i++)
    {
        free((*image)->channels[i]);
        (*image)->channels[i] = NULL;
    }
    free((*image)->channels);
    (*image)->channels = NULL;
    return LATENT_SVM_OK;
}
