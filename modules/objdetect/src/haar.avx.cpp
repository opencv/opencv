/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* Haar features calculation */

#include "precomp.hpp"
#include "haar.hpp"

namespace cv_haar_avx
{

// AVX version icvEvalHidHaarClassifier.  Process 8 CvHidHaarClassifiers per call. Check AVX support before invocation!!
#if CV_HAAR_USE_AVX
double icvEvalHidHaarClassifierAVX(CvHidHaarClassifier* classifier,
    double variance_norm_factor, size_t p_offset)
{
    int  CV_DECL_ALIGNED(32) idxV[8] = { 0,0,0,0,0,0,0,0 };
    uchar flags[8] = { 0,0,0,0,0,0,0,0 };
    CvHidHaarTreeNode* nodes[8];
    double res = 0;
    uchar exitConditionFlag = 0;
    for (;;)
    {
        float CV_DECL_ALIGNED(32) tmp[8] = { 0,0,0,0,0,0,0,0 };
        nodes[0] = (classifier + 0)->node + idxV[0];
        nodes[1] = (classifier + 1)->node + idxV[1];
        nodes[2] = (classifier + 2)->node + idxV[2];
        nodes[3] = (classifier + 3)->node + idxV[3];
        nodes[4] = (classifier + 4)->node + idxV[4];
        nodes[5] = (classifier + 5)->node + idxV[5];
        nodes[6] = (classifier + 6)->node + idxV[6];
        nodes[7] = (classifier + 7)->node + idxV[7];

        __m256 t = _mm256_set1_ps(static_cast<float>(variance_norm_factor));

        t = _mm256_mul_ps(t, _mm256_set_ps(nodes[7]->threshold,
            nodes[6]->threshold,
            nodes[5]->threshold,
            nodes[4]->threshold,
            nodes[3]->threshold,
            nodes[2]->threshold,
            nodes[1]->threshold,
            nodes[0]->threshold));

        __m256 offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[0], p_offset),
            calc_sumf(nodes[6]->feature.rect[0], p_offset),
            calc_sumf(nodes[5]->feature.rect[0], p_offset),
            calc_sumf(nodes[4]->feature.rect[0], p_offset),
            calc_sumf(nodes[3]->feature.rect[0], p_offset),
            calc_sumf(nodes[2]->feature.rect[0], p_offset),
            calc_sumf(nodes[1]->feature.rect[0], p_offset),
            calc_sumf(nodes[0]->feature.rect[0], p_offset));

        __m256 weight = _mm256_set_ps(nodes[7]->feature.rect[0].weight,
            nodes[6]->feature.rect[0].weight,
            nodes[5]->feature.rect[0].weight,
            nodes[4]->feature.rect[0].weight,
            nodes[3]->feature.rect[0].weight,
            nodes[2]->feature.rect[0].weight,
            nodes[1]->feature.rect[0].weight,
            nodes[0]->feature.rect[0].weight);

        __m256 sum = _mm256_mul_ps(offset, weight);

        offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[1], p_offset),
            calc_sumf(nodes[6]->feature.rect[1], p_offset),
            calc_sumf(nodes[5]->feature.rect[1], p_offset),
            calc_sumf(nodes[4]->feature.rect[1], p_offset),
            calc_sumf(nodes[3]->feature.rect[1], p_offset),
            calc_sumf(nodes[2]->feature.rect[1], p_offset),
            calc_sumf(nodes[1]->feature.rect[1], p_offset),
            calc_sumf(nodes[0]->feature.rect[1], p_offset));

        weight = _mm256_set_ps(nodes[7]->feature.rect[1].weight,
            nodes[6]->feature.rect[1].weight,
            nodes[5]->feature.rect[1].weight,
            nodes[4]->feature.rect[1].weight,
            nodes[3]->feature.rect[1].weight,
            nodes[2]->feature.rect[1].weight,
            nodes[1]->feature.rect[1].weight,
            nodes[0]->feature.rect[1].weight);

        sum = _mm256_add_ps(sum, _mm256_mul_ps(offset, weight));

        if (nodes[0]->feature.rect[2].p0)
            tmp[0] = calc_sumf(nodes[0]->feature.rect[2], p_offset) * nodes[0]->feature.rect[2].weight;
        if (nodes[1]->feature.rect[2].p0)
            tmp[1] = calc_sumf(nodes[1]->feature.rect[2], p_offset) * nodes[1]->feature.rect[2].weight;
        if (nodes[2]->feature.rect[2].p0)
            tmp[2] = calc_sumf(nodes[2]->feature.rect[2], p_offset) * nodes[2]->feature.rect[2].weight;
        if (nodes[3]->feature.rect[2].p0)
            tmp[3] = calc_sumf(nodes[3]->feature.rect[2], p_offset) * nodes[3]->feature.rect[2].weight;
        if (nodes[4]->feature.rect[2].p0)
            tmp[4] = calc_sumf(nodes[4]->feature.rect[2], p_offset) * nodes[4]->feature.rect[2].weight;
        if (nodes[5]->feature.rect[2].p0)
            tmp[5] = calc_sumf(nodes[5]->feature.rect[2], p_offset) * nodes[5]->feature.rect[2].weight;
        if (nodes[6]->feature.rect[2].p0)
            tmp[6] = calc_sumf(nodes[6]->feature.rect[2], p_offset) * nodes[6]->feature.rect[2].weight;
        if (nodes[7]->feature.rect[2].p0)
            tmp[7] = calc_sumf(nodes[7]->feature.rect[2], p_offset) * nodes[7]->feature.rect[2].weight;

        sum = _mm256_add_ps(sum, _mm256_load_ps(tmp));

        __m256 left = _mm256_set_ps(static_cast<float>(nodes[7]->left), static_cast<float>(nodes[6]->left),
            static_cast<float>(nodes[5]->left), static_cast<float>(nodes[4]->left),
            static_cast<float>(nodes[3]->left), static_cast<float>(nodes[2]->left),
            static_cast<float>(nodes[1]->left), static_cast<float>(nodes[0]->left));
        __m256 right = _mm256_set_ps(static_cast<float>(nodes[7]->right), static_cast<float>(nodes[6]->right),
            static_cast<float>(nodes[5]->right), static_cast<float>(nodes[4]->right),
            static_cast<float>(nodes[3]->right), static_cast<float>(nodes[2]->right),
            static_cast<float>(nodes[1]->right), static_cast<float>(nodes[0]->right));

        _mm256_store_si256((__m256i*)idxV, _mm256_cvttps_epi32(_mm256_blendv_ps(right, left, _mm256_cmp_ps(sum, t, _CMP_LT_OQ))));

        for (int i = 0; i < 8; i++)
        {
            if (idxV[i] <= 0)
            {
                if (!flags[i])
                {
                    exitConditionFlag++;
                    flags[i] = 1;
                    res += (classifier + i)->alpha[-idxV[i]];
                }
                idxV[i] = 0;
            }
        }
        if (exitConditionFlag == 8)
            return res;
    }
}

double icvEvalHidHaarStumpClassifierAVX(CvHidHaarClassifier* classifier,
    double variance_norm_factor, size_t p_offset)
{
    float CV_DECL_ALIGNED(32) tmp[8] = { 0,0,0,0,0,0,0,0 };
    CvHidHaarTreeNode* nodes[8];

    nodes[0] = classifier[0].node;
    nodes[1] = classifier[1].node;
    nodes[2] = classifier[2].node;
    nodes[3] = classifier[3].node;
    nodes[4] = classifier[4].node;
    nodes[5] = classifier[5].node;
    nodes[6] = classifier[6].node;
    nodes[7] = classifier[7].node;

    __m256 t = _mm256_set1_ps(static_cast<float>(variance_norm_factor));

    t = _mm256_mul_ps(t, _mm256_set_ps(nodes[7]->threshold,
        nodes[6]->threshold,
        nodes[5]->threshold,
        nodes[4]->threshold,
        nodes[3]->threshold,
        nodes[2]->threshold,
        nodes[1]->threshold,
        nodes[0]->threshold));

    __m256 offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[0], p_offset),
        calc_sumf(nodes[6]->feature.rect[0], p_offset),
        calc_sumf(nodes[5]->feature.rect[0], p_offset),
        calc_sumf(nodes[4]->feature.rect[0], p_offset),
        calc_sumf(nodes[3]->feature.rect[0], p_offset),
        calc_sumf(nodes[2]->feature.rect[0], p_offset),
        calc_sumf(nodes[1]->feature.rect[0], p_offset),
        calc_sumf(nodes[0]->feature.rect[0], p_offset));

    __m256 weight = _mm256_set_ps(nodes[7]->feature.rect[0].weight,
        nodes[6]->feature.rect[0].weight,
        nodes[5]->feature.rect[0].weight,
        nodes[4]->feature.rect[0].weight,
        nodes[3]->feature.rect[0].weight,
        nodes[2]->feature.rect[0].weight,
        nodes[1]->feature.rect[0].weight,
        nodes[0]->feature.rect[0].weight);

    __m256 sum = _mm256_mul_ps(offset, weight);

    offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[1], p_offset),
        calc_sumf(nodes[6]->feature.rect[1], p_offset),
        calc_sumf(nodes[5]->feature.rect[1], p_offset),
        calc_sumf(nodes[4]->feature.rect[1], p_offset),
        calc_sumf(nodes[3]->feature.rect[1], p_offset),
        calc_sumf(nodes[2]->feature.rect[1], p_offset),
        calc_sumf(nodes[1]->feature.rect[1], p_offset),
        calc_sumf(nodes[0]->feature.rect[1], p_offset));

    weight = _mm256_set_ps(nodes[7]->feature.rect[1].weight,
        nodes[6]->feature.rect[1].weight,
        nodes[5]->feature.rect[1].weight,
        nodes[4]->feature.rect[1].weight,
        nodes[3]->feature.rect[1].weight,
        nodes[2]->feature.rect[1].weight,
        nodes[1]->feature.rect[1].weight,
        nodes[0]->feature.rect[1].weight);

    sum = _mm256_add_ps(sum, _mm256_mul_ps(offset, weight));

    if (nodes[0]->feature.rect[2].p0)
        tmp[0] = calc_sumf(nodes[0]->feature.rect[2], p_offset) * nodes[0]->feature.rect[2].weight;
    if (nodes[1]->feature.rect[2].p0)
        tmp[1] = calc_sumf(nodes[1]->feature.rect[2], p_offset) * nodes[1]->feature.rect[2].weight;
    if (nodes[2]->feature.rect[2].p0)
        tmp[2] = calc_sumf(nodes[2]->feature.rect[2], p_offset) * nodes[2]->feature.rect[2].weight;
    if (nodes[3]->feature.rect[2].p0)
        tmp[3] = calc_sumf(nodes[3]->feature.rect[2], p_offset) * nodes[3]->feature.rect[2].weight;
    if (nodes[4]->feature.rect[2].p0)
        tmp[4] = calc_sumf(nodes[4]->feature.rect[2], p_offset) * nodes[4]->feature.rect[2].weight;
    if (nodes[5]->feature.rect[2].p0)
        tmp[5] = calc_sumf(nodes[5]->feature.rect[2], p_offset) * nodes[5]->feature.rect[2].weight;
    if (nodes[6]->feature.rect[2].p0)
        tmp[6] = calc_sumf(nodes[6]->feature.rect[2], p_offset) * nodes[6]->feature.rect[2].weight;
    if (nodes[7]->feature.rect[2].p0)
        tmp[7] = calc_sumf(nodes[7]->feature.rect[2], p_offset) * nodes[7]->feature.rect[2].weight;

    sum = _mm256_add_ps(sum, _mm256_load_ps(tmp));

    __m256 alpha0 = _mm256_set_ps(classifier[7].alpha[0],
        classifier[6].alpha[0],
        classifier[5].alpha[0],
        classifier[4].alpha[0],
        classifier[3].alpha[0],
        classifier[2].alpha[0],
        classifier[1].alpha[0],
        classifier[0].alpha[0]);
    __m256 alpha1 = _mm256_set_ps(classifier[7].alpha[1],
        classifier[6].alpha[1],
        classifier[5].alpha[1],
        classifier[4].alpha[1],
        classifier[3].alpha[1],
        classifier[2].alpha[1],
        classifier[1].alpha[1],
        classifier[0].alpha[1]);

    __m256 outBuf = _mm256_blendv_ps(alpha0, alpha1, _mm256_cmp_ps(t, sum, _CMP_LE_OQ));
    outBuf = _mm256_hadd_ps(outBuf, outBuf);
    outBuf = _mm256_hadd_ps(outBuf, outBuf);
    _mm256_store_ps(tmp, outBuf);
    return (tmp[0] + tmp[4]);
}

double icvEvalHidHaarStumpClassifierTwoRectAVX(CvHidHaarClassifier* classifier,
    double variance_norm_factor, size_t p_offset)
{
    float CV_DECL_ALIGNED(32) buf[8];
    CvHidHaarTreeNode* nodes[8];
    nodes[0] = classifier[0].node;
    nodes[1] = classifier[1].node;
    nodes[2] = classifier[2].node;
    nodes[3] = classifier[3].node;
    nodes[4] = classifier[4].node;
    nodes[5] = classifier[5].node;
    nodes[6] = classifier[6].node;
    nodes[7] = classifier[7].node;

    __m256 t = _mm256_set1_ps(static_cast<float>(variance_norm_factor));
    t = _mm256_mul_ps(t, _mm256_set_ps(nodes[7]->threshold,
        nodes[6]->threshold,
        nodes[5]->threshold,
        nodes[4]->threshold,
        nodes[3]->threshold,
        nodes[2]->threshold,
        nodes[1]->threshold,
        nodes[0]->threshold));

    __m256 offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[0], p_offset),
        calc_sumf(nodes[6]->feature.rect[0], p_offset),
        calc_sumf(nodes[5]->feature.rect[0], p_offset),
        calc_sumf(nodes[4]->feature.rect[0], p_offset),
        calc_sumf(nodes[3]->feature.rect[0], p_offset),
        calc_sumf(nodes[2]->feature.rect[0], p_offset),
        calc_sumf(nodes[1]->feature.rect[0], p_offset),
        calc_sumf(nodes[0]->feature.rect[0], p_offset));

    __m256 weight = _mm256_set_ps(nodes[7]->feature.rect[0].weight,
        nodes[6]->feature.rect[0].weight,
        nodes[5]->feature.rect[0].weight,
        nodes[4]->feature.rect[0].weight,
        nodes[3]->feature.rect[0].weight,
        nodes[2]->feature.rect[0].weight,
        nodes[1]->feature.rect[0].weight,
        nodes[0]->feature.rect[0].weight);

    __m256 sum = _mm256_mul_ps(offset, weight);

    offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[1], p_offset),
        calc_sumf(nodes[6]->feature.rect[1], p_offset),
        calc_sumf(nodes[5]->feature.rect[1], p_offset),
        calc_sumf(nodes[4]->feature.rect[1], p_offset),
        calc_sumf(nodes[3]->feature.rect[1], p_offset),
        calc_sumf(nodes[2]->feature.rect[1], p_offset),
        calc_sumf(nodes[1]->feature.rect[1], p_offset),
        calc_sumf(nodes[0]->feature.rect[1], p_offset));

    weight = _mm256_set_ps(nodes[7]->feature.rect[1].weight,
        nodes[6]->feature.rect[1].weight,
        nodes[5]->feature.rect[1].weight,
        nodes[4]->feature.rect[1].weight,
        nodes[3]->feature.rect[1].weight,
        nodes[2]->feature.rect[1].weight,
        nodes[1]->feature.rect[1].weight,
        nodes[0]->feature.rect[1].weight);

    sum = _mm256_add_ps(sum, _mm256_mul_ps(offset, weight));

    __m256 alpha0 = _mm256_set_ps(classifier[7].alpha[0],
        classifier[6].alpha[0],
        classifier[5].alpha[0],
        classifier[4].alpha[0],
        classifier[3].alpha[0],
        classifier[2].alpha[0],
        classifier[1].alpha[0],
        classifier[0].alpha[0]);
    __m256 alpha1 = _mm256_set_ps(classifier[7].alpha[1],
        classifier[6].alpha[1],
        classifier[5].alpha[1],
        classifier[4].alpha[1],
        classifier[3].alpha[1],
        classifier[2].alpha[1],
        classifier[1].alpha[1],
        classifier[0].alpha[1]);

    _mm256_store_ps(buf, _mm256_blendv_ps(alpha0, alpha1, _mm256_cmp_ps(t, sum, _CMP_LE_OQ)));
    return (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]);
}

#endif //CV_HAAR_USE_AVX

}

/* End of file. */
