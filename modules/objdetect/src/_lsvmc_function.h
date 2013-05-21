#ifndef FUNCTION_SC
#define FUNCTION_SC

#include "_lsvmc_types.h"

namespace cv
{
namespace lsvmcascade
{

float calcM         (int k,int di,int dj, const CvLSVMFeaturePyramidCaskad * H, const CvLSVMFilterObjectCaskad *filter);
float calcM_PCA     (int k,int di,int dj, const CvLSVMFeaturePyramidCaskad * H, const CvLSVMFilterObjectCaskad *filter);
float calcM_PCA_cash(int k,int di,int dj, const CvLSVMFeaturePyramidCaskad * H, const CvLSVMFilterObjectCaskad *filter, float * cashM, int * maskM, int step);
float calcFine (const CvLSVMFilterObjectCaskad *filter, int di, int dj);
}
}
#endif