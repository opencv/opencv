#include "precomp.hpp"
#include "_lsvm_routine.h"

int allocFilterObject(CvLSVMFilterObject **obj, const int sizeX, const int sizeY, const int p, const int xp){
    int i;
    (*obj) = (CvLSVMFilterObject *)malloc(sizeof(CvLSVMFilterObject));
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
int freeFilterObject (CvLSVMFilterObject **obj){
    if(*obj == NULL) return 0;
    free((*obj)->H);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeatureMapObject(CvLSVMFeatureMap **obj, const int sizeX, const int sizeY, const int p, const int xp){
    int i;
    (*obj) = (CvLSVMFeatureMap *)malloc(sizeof(CvLSVMFeatureMap));
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
int freeFeatureMapObject (CvLSVMFeatureMap **obj){
    if(*obj == NULL) return 0;
    free((*obj)->Map);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeaturePyramidObject(CvLSVMFeaturePyramid **obj, const int lambda, const int countLevel){
    (*obj) = (CvLSVMFeaturePyramid *)malloc(sizeof(CvLSVMFeaturePyramid));
    (*obj)->countLevel = countLevel;
    (*obj)->pyramid    = (CvLSVMFeatureMap **)malloc(sizeof(CvLSVMFeatureMap *) * countLevel);
    (*obj)->lambda     = lambda;
    return LATENT_SVM_OK;
}

int freeFeaturePyramidObject (CvLSVMFeaturePyramid **obj){
    int i; 
    if(*obj == NULL) return 0;
    for(i = 0; i < (*obj)->countLevel; i++)
        freeFeatureMapObject(&((*obj)->pyramid[i]));
    free((*obj)->pyramid);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFFTImage(CvLSVMFftImage **image, int p, int dimX, int dimY)
{
    int i, j, size;
    *image = (CvLSVMFftImage *)malloc(sizeof(CvLSVMFftImage));
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

int freeFFTImage(CvLSVMFftImage **image)
{
    unsigned i;
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
