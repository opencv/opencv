#include "_lsvm_function.h"

float calcM    (int k,int di,int dj, const CvLSVMFeaturePyramid * H, const CvLSVMFilterObject *filter){
    unsigned int i, j;
    float m = 0.0f;
    for(j = dj; j < dj + filter->sizeY; j++){
        for(i = di * H->pyramid[k]->numFeatures; i < (di + filter->sizeX) * H->pyramid[k]->numFeatures; i++){
             m += H->pyramid[k]->map[(j * H->pyramid[k]->sizeX     ) * H->pyramid[k]->numFeatures + i] * 
                  filter ->H        [((j - dj) * filter->sizeX - di) * H->pyramid[k]->numFeatures + i];            
        }
    }
    return m;
}
float calcM_PCA(int k,int di,int dj, const CvLSVMFeaturePyramid * H, const CvLSVMFilterObject *filter){
    unsigned int i, j;
    float m = 0.0f;
    for(j = dj; j < dj + filter->sizeY; j++){
        for(i = di * H->pyramid[k]->numFeatures; i < (di + filter->sizeX) * H->pyramid[k]->numFeatures; i++){
            m += H->pyramid[k]->map[(j * H->pyramid[k]->sizeX     ) * H->pyramid[k]->numFeatures + i] * 
                 filter ->H_PCA    [((j - dj) * filter->sizeX - di) * H->pyramid[k]->numFeatures + i];
        }
    }

    return m;
}
float calcM_PCA_cash(int k,int di,int dj, const CvLSVMFeaturePyramid * H, const CvLSVMFilterObject *filter, float * cashM, int * maskM, int step){
    unsigned int i, j, mean;
    unsigned int n;
    float m = 0.0f;
    float tmp1, tmp2, tmp3, tmp4;
    float res;
    int pos;
    float *a, *b;

    pos = dj * step + di;

    if(!((maskM[pos / (sizeof(int) * 8)]) & (1 << pos % (sizeof(int) * 8))))
    {
        for(j = dj; j < dj + filter->sizeY; j++)
        {
            a = H->pyramid[k]->map + (j * H->pyramid[k]->sizeX) * H->pyramid[k]->numFeatures
              + di * H->pyramid[k]->numFeatures;
            b = filter ->H_PCA + (j - dj) * filter->sizeX * H->pyramid[k]->numFeatures;
            n = ((di + filter->sizeX) * H->pyramid[k]->numFeatures) - 
              (di * H->pyramid[k]->numFeatures);
            
            res = 0.0f;
            tmp1 = 0.0f; tmp2 = 0.0f; tmp3 = 0.0f; tmp4 = 0.0f;

            for (i = 0; i < (n >> 2); ++i)
            {
                tmp1 += a[4 * i + 0] * b[4 * i + 0];
                tmp2 += a[4 * i + 1] * b[4 * i + 1];
                tmp3 += a[4 * i + 2] * b[4 * i + 2];
                tmp4 += a[4 * i + 3] * b[4 * i + 3];
            }
            
            mean = (n >> 2) << 2;
            for (i = mean; i < n; ++i)
            {
                res += a[i] * b[i];
            }

            res += tmp1 + tmp2 + tmp3 + tmp4;

            m += res;
        }

        cashM[pos                    ]  = m;
        maskM[pos / (sizeof(int) * 8)] |= 1 << pos % (sizeof(int) * 8);
    }
    else
    {
        m = cashM[pos];
    }
    return m;
}
float calcFine (const CvLSVMFilterObject *filter, int di, int dj){
    return filter->fineFunction[0] * di      + filter->fineFunction[1] * dj + 
           filter->fineFunction[2] * di * di + filter->fineFunction[3] * dj * dj;
}