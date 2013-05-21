#include "precomp.hpp"
#include "_lsvmc_routine.h"
namespace cv
{
namespace lsvmcascade
{
int allocFilterObject(CvLSVMFilterObjectCaskad **obj, const int sizeX,
                      const int sizeY, const int numFeatures) 
{
    int i;
    (*obj) = (CvLSVMFilterObjectCaskad *)malloc(sizeof(CvLSVMFilterObjectCaskad));
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
int freeFilterObject (CvLSVMFilterObjectCaskad **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->H);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeatureMapObject(CvLSVMFeatureMapCaskad **obj, const int sizeX, 
                          const int sizeY, const int numFeatures)
{
    int i;
    (*obj) = (CvLSVMFeatureMapCaskad *)malloc(sizeof(CvLSVMFeatureMapCaskad));
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
int freeFeatureMapObject (CvLSVMFeatureMapCaskad **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->map);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeaturePyramidObject(CvLSVMFeaturePyramidCaskad **obj,
                              const int numLevels) 
{
    (*obj) = (CvLSVMFeaturePyramidCaskad *)malloc(sizeof(CvLSVMFeaturePyramidCaskad));
    (*obj)->numLevels = numLevels;
    (*obj)->pyramid    = (CvLSVMFeatureMapCaskad **)malloc(
                         sizeof(CvLSVMFeatureMapCaskad *) * numLevels);
    return LATENT_SVM_OK;
}

int freeFeaturePyramidObject (CvLSVMFeaturePyramidCaskad **obj)
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
}
}
