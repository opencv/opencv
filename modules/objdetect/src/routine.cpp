#include "precomp.hpp"
#include "_lsvm_routine.h"

int allocFilterObject(filterObject **obj, const int sizeX, const int sizeY, const int p, const int xp){
    int i;
    (*obj) = (filterObject *)malloc(sizeof(filterObject));
    (*obj)->sizeX = sizeX;
    (*obj)->sizeY = sizeY;
    (*obj)->p     = p    ;
    (*obj)->xp    = xp   ;
    (*obj)->fineFunction[0] = 0.0f;
    (*obj)->fineFunction[1] = 0.0f;
    (*obj)->fineFunction[2] = 0.0f;
    (*obj)->fineFunction[3] = 0.0f;
    (*obj)->V.x         = 0;
    (*obj)->V.y         = 0;
    (*obj)->V.l         = 0;
    (*obj)->H = (float *) malloc(sizeof (float) * (sizeX * sizeY * p));
    for(i = 0; i < sizeX * sizeY * p; i++){
        (*obj)->H[i] = 0.0f;
    }
    return LATENT_SVM_OK;
}
int freeFilterObject (filterObject **obj){
    if(*obj == NULL) return 0;
    free((*obj)->H);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeatureMapObject(featureMap **obj, const int sizeX, const int sizeY, const int p, const int xp){
    int i;
    (*obj) = (featureMap *)malloc(sizeof(featureMap));
    (*obj)->sizeX = sizeX;
    (*obj)->sizeY = sizeY;
    (*obj)->p     = p    ;
    (*obj)->xp    = xp   ;
    (*obj)->Map = (float *) malloc(sizeof (float) * (sizeX * sizeY * p));
    for(i = 0; i < sizeX * sizeY * p; i++){
        (*obj)->Map[i] = 0.0;
    }
    return LATENT_SVM_OK;
}
int freeFeatureMapObject (featureMap **obj){
    if(*obj == NULL) return 0;
    free((*obj)->Map);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeaturePyramidObject(featurePyramid **obj, const int lambda, const int countLevel){
    (*obj) = (featurePyramid *)malloc(sizeof(featurePyramid));
    (*obj)->countLevel = countLevel;
    (*obj)->pyramid    = (featureMap **)malloc(sizeof(featureMap *) * countLevel);
    (*obj)->lambda     = lambda;
    return LATENT_SVM_OK;
}

int freeFeaturePyramidObject (featurePyramid **obj){
    int i; 
    if(*obj == NULL) return 0;
    for(i = 0; i < (*obj)->countLevel; i++)
        freeFeatureMapObject(&((*obj)->pyramid[i]));
    free((*obj)->pyramid);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFFTImage(fftImage **image, int p, int dimX, int dimY)
{
    int i, j, size;
    *image = (fftImage *)malloc(sizeof(fftImage));
    (*image)->p = p;
    (*image)->dimX = dimX;
    (*image)->dimY = dimY;
    (*image)->channels = (float **)malloc(sizeof(float *) * p);
    size = 2 * dimX * dimY;
    for (i = 0; i < p; i++)
    {
        (*image)->channels[i] = (float *)malloc(sizeof(float) * size);
        for (j = 0; j < size; j++)
        {
            (*image)->channels[i][j] = 0.0;
        }
    }
    return LATENT_SVM_OK;
}

int freeFFTImage(fftImage **image)
{
    unsigned int i;
    if (*image == NULL) return LATENT_SVM_OK;
    for (i = 0; i < (*image)->p; i++)
    {
        free((*image)->channels[i]);
        (*image)->channels[i] = NULL;
    }
    free((*image)->channels);
    (*image)->channels = NULL;
    return LATENT_SVM_OK;
}