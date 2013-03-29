/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Wang Weiyan, wangweiyanster@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Wu Xinglong, wxl370@126.com
//    Wang Yao, bitwangyaoyao@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "precomp.hpp"
#include <stdio.h>

using namespace cv;
using namespace cv::ocl;

namespace cv
{
namespace ocl
{
///////////////////////////OpenCL kernel strings///////////////////////////
extern const char *haarobjectdetect;
extern const char *haarobjectdetectbackup;
extern const char *haarobjectdetect_scaled2;
}
}

/* these settings affect the quality of detection: change with care */
#define CV_ADJUST_FEATURES 1
#define CV_ADJUST_WEIGHTS  0

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
    struct
    {
        sumtype *p0, *p1, *p2, *p3;
        float weight;
    }
    rect[CV_HAAR_FEATURE_MAX];
}
CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
    CvHidHaarFeature feature;
    float threshold;
    int left;
    int right;
}
CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
    int count;
    //CvHaarFeature* orig_feature;
    CvHidHaarTreeNode *node;
    float *alpha;
}
CvHidHaarClassifier;


typedef struct CvHidHaarStageClassifier
{
    int  count;
    float threshold;
    CvHidHaarClassifier *classifier;
    int two_rects;

    struct CvHidHaarStageClassifier *next;
    struct CvHidHaarStageClassifier *child;
    struct CvHidHaarStageClassifier *parent;
}
CvHidHaarStageClassifier;


struct CvHidHaarClassifierCascade
{
    int  count;
    int  is_stump_based;
    int  has_tilted_features;
    int  is_tree;
    double inv_window_area;
    CvMat sum, sqsum, tilted;
    CvHidHaarStageClassifier *stage_classifier;
    sqsumtype *pq0, *pq1, *pq2, *pq3;
    sumtype *p0, *p1, *p2, *p3;

    void **ipp_stages;
};
typedef struct
{
    //int rows;
    //int ystep;
    int width_height;
    //int height;
    int grpnumperline_totalgrp;
    //int totalgrp;
    int imgoff;
    float factor;
} detect_piramid_info;
#ifdef WIN32
#define _ALIGNED_ON(_ALIGNMENT) __declspec(align(_ALIGNMENT))
typedef _ALIGNED_ON(128) struct  GpuHidHaarFeature
{
    _ALIGNED_ON(32) struct
    {
        _ALIGNED_ON(4)  int    p0 ;
        _ALIGNED_ON(4)  int    p1 ;
        _ALIGNED_ON(4)  int    p2 ;
        _ALIGNED_ON(4)  int    p3 ;
        _ALIGNED_ON(4)  float weight  ;
    }
    /*_ALIGNED_ON(32)*/ rect[CV_HAAR_FEATURE_MAX] ;
}
GpuHidHaarFeature;


typedef _ALIGNED_ON(128) struct  GpuHidHaarTreeNode
{
    _ALIGNED_ON(64) int p[CV_HAAR_FEATURE_MAX][4];
    //_ALIGNED_ON(16) int p1[CV_HAAR_FEATURE_MAX] ;
    //_ALIGNED_ON(16) int p2[CV_HAAR_FEATURE_MAX] ;
    //_ALIGNED_ON(16) int p3[CV_HAAR_FEATURE_MAX] ;
    /*_ALIGNED_ON(16)*/
    float weight[CV_HAAR_FEATURE_MAX] ;
    /*_ALIGNED_ON(4)*/
    float threshold ;
    _ALIGNED_ON(8) float alpha[2] ;
    _ALIGNED_ON(4) int left ;
    _ALIGNED_ON(4) int right ;
    // GpuHidHaarFeature feature __attribute__((aligned (128)));
}
GpuHidHaarTreeNode;


typedef  _ALIGNED_ON(32) struct  GpuHidHaarClassifier
{
    _ALIGNED_ON(4) int count;
    //CvHaarFeature* orig_feature;
    _ALIGNED_ON(8) GpuHidHaarTreeNode *node ;
    _ALIGNED_ON(8) float *alpha ;
}
GpuHidHaarClassifier;


typedef _ALIGNED_ON(64) struct   GpuHidHaarStageClassifier
{
    _ALIGNED_ON(4) int  count ;
    _ALIGNED_ON(4) float threshold ;
    _ALIGNED_ON(4) int two_rects ;
    _ALIGNED_ON(8) GpuHidHaarClassifier *classifier ;
    _ALIGNED_ON(8) struct GpuHidHaarStageClassifier *next;
    _ALIGNED_ON(8) struct GpuHidHaarStageClassifier *child ;
    _ALIGNED_ON(8) struct GpuHidHaarStageClassifier *parent ;
}
GpuHidHaarStageClassifier;


typedef _ALIGNED_ON(64) struct  GpuHidHaarClassifierCascade
{
    _ALIGNED_ON(4) int  count ;
    _ALIGNED_ON(4) int  is_stump_based ;
    _ALIGNED_ON(4) int  has_tilted_features ;
    _ALIGNED_ON(4) int  is_tree ;
    _ALIGNED_ON(4) int pq0 ;
    _ALIGNED_ON(4) int pq1 ;
    _ALIGNED_ON(4) int pq2 ;
    _ALIGNED_ON(4) int pq3 ;
    _ALIGNED_ON(4) int p0 ;
    _ALIGNED_ON(4) int p1 ;
    _ALIGNED_ON(4) int p2 ;
    _ALIGNED_ON(4) int p3 ;
    _ALIGNED_ON(4) float inv_window_area ;
    // GpuHidHaarStageClassifier* stage_classifier __attribute__((aligned (8)));
} GpuHidHaarClassifierCascade;
#else
#define _ALIGNED_ON(_ALIGNMENT) __attribute__((aligned(_ALIGNMENT) ))

typedef struct _ALIGNED_ON(128) GpuHidHaarFeature
{
    struct _ALIGNED_ON(32)
{
    int    p0 _ALIGNED_ON(4);
    int    p1 _ALIGNED_ON(4);
    int    p2 _ALIGNED_ON(4);
    int    p3 _ALIGNED_ON(4);
    float weight  _ALIGNED_ON(4);
}
rect[CV_HAAR_FEATURE_MAX] _ALIGNED_ON(32);
}
GpuHidHaarFeature;


typedef struct _ALIGNED_ON(128) GpuHidHaarTreeNode
{
    int p[CV_HAAR_FEATURE_MAX][4] _ALIGNED_ON(64);
    float weight[CV_HAAR_FEATURE_MAX];// _ALIGNED_ON(16);
    float threshold;// _ALIGNED_ON(4);
    float alpha[2] _ALIGNED_ON(8);
    int left _ALIGNED_ON(4);
    int right _ALIGNED_ON(4);
}
GpuHidHaarTreeNode;

typedef struct _ALIGNED_ON(32) GpuHidHaarClassifier
{
    int count _ALIGNED_ON(4);
    GpuHidHaarTreeNode *node _ALIGNED_ON(8);
    float *alpha _ALIGNED_ON(8);
}
GpuHidHaarClassifier;


typedef struct _ALIGNED_ON(64) GpuHidHaarStageClassifier
{
    int  count _ALIGNED_ON(4);
    float threshold _ALIGNED_ON(4);
    int two_rects _ALIGNED_ON(4);
    GpuHidHaarClassifier *classifier _ALIGNED_ON(8);
    struct GpuHidHaarStageClassifier *next _ALIGNED_ON(8);
    struct GpuHidHaarStageClassifier *child _ALIGNED_ON(8);
    struct GpuHidHaarStageClassifier *parent _ALIGNED_ON(8);
}
GpuHidHaarStageClassifier;


typedef struct _ALIGNED_ON(64) GpuHidHaarClassifierCascade
{
    int  count _ALIGNED_ON(4);
    int  is_stump_based _ALIGNED_ON(4);
    int  has_tilted_features _ALIGNED_ON(4);
    int  is_tree _ALIGNED_ON(4);
    int pq0 _ALIGNED_ON(4);
    int pq1 _ALIGNED_ON(4);
    int pq2 _ALIGNED_ON(4);
    int pq3 _ALIGNED_ON(4);
    int p0 _ALIGNED_ON(4);
    int p1 _ALIGNED_ON(4);
    int p2 _ALIGNED_ON(4);
    int p3 _ALIGNED_ON(4);
    float inv_window_area _ALIGNED_ON(4);
    // GpuHidHaarStageClassifier* stage_classifier __attribute__((aligned (8)));
} GpuHidHaarClassifierCascade;
#endif

const int icv_object_win_border = 1;
const float icv_stage_threshold_bias = 0.0001f;
double globaltime = 0;


// static CvHaarClassifierCascade * gpuCreateHaarClassifierCascade( int stage_count )
// {
//     CvHaarClassifierCascade *cascade = 0;

//     int block_size = sizeof(*cascade) + stage_count * sizeof(*cascade->stage_classifier);

//     if( stage_count <= 0 )
//         CV_Error( CV_StsOutOfRange, "Number of stages should be positive" );

//     cascade = (CvHaarClassifierCascade *)cvAlloc( block_size );
//     memset( cascade, 0, block_size );

//     cascade->stage_classifier = (CvHaarStageClassifier *)(cascade + 1);
//     cascade->flags = CV_HAAR_MAGIC_VAL;
//     cascade->count = stage_count;

//     return cascade;
// }

//static int globalcounter = 0;

// static void gpuReleaseHidHaarClassifierCascade( GpuHidHaarClassifierCascade **_cascade )
// {
//     if( _cascade && *_cascade )
//     {
//         cvFree( _cascade );
//     }
// }

/* create more efficient internal representation of haar classifier cascade */
static GpuHidHaarClassifierCascade * gpuCreateHidHaarClassifierCascade( CvHaarClassifierCascade *cascade, int *size, int *totalclassifier)
{
    GpuHidHaarClassifierCascade *out = 0;

    int i, j, k, l;
    int datasize;
    int total_classifiers = 0;
    int total_nodes = 0;
    char errorstr[100];

    GpuHidHaarStageClassifier *stage_classifier_ptr;
    GpuHidHaarClassifier *haar_classifier_ptr;
    GpuHidHaarTreeNode *haar_node_ptr;

    CvSize orig_window_size;
    int has_tilted_features = 0;

    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( cascade->hid_cascade )
        CV_Error( CV_StsError, "hid_cascade has been already created" );

    if( !cascade->stage_classifier )
        CV_Error( CV_StsNullPtr, "" );

    if( cascade->count <= 0 )
        CV_Error( CV_StsOutOfRange, "Negative number of cascade stages" );

    orig_window_size = cascade->orig_window_size;

    /* check input structure correctness and calculate total memory size needed for
    internal representation of the classifier cascade */
    for( i = 0; i < cascade->count; i++ )
    {
        CvHaarStageClassifier *stage_classifier = cascade->stage_classifier + i;

        if( !stage_classifier->classifier ||
                stage_classifier->count <= 0 )
        {
            sprintf( errorstr, "header of the stage classifier #%d is invalid "
                     "(has null pointers or non-positive classfier count)", i );
            CV_Error( CV_StsError, errorstr );
        }

        total_classifiers += stage_classifier->count;

        for( j = 0; j < stage_classifier->count; j++ )
        {
            CvHaarClassifier *classifier = stage_classifier->classifier + j;

            total_nodes += classifier->count;
            for( l = 0; l < classifier->count; l++ )
            {
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    if( classifier->haar_feature[l].rect[k].r.width )
                    {
                        CvRect r = classifier->haar_feature[l].rect[k].r;
                        int tilted = classifier->haar_feature[l].tilted;
                        has_tilted_features |= tilted != 0;
                        if( r.width < 0 || r.height < 0 || r.y < 0 ||
                                r.x + r.width > orig_window_size.width
                                ||
                                (!tilted &&
                                 (r.x < 0 || r.y + r.height > orig_window_size.height))
                                ||
                                (tilted && (r.x - r.height < 0 ||
                                            r.y + r.width + r.height > orig_window_size.height)))
                        {
                            sprintf( errorstr, "rectangle #%d of the classifier #%d of "
                                     "the stage classifier #%d is not inside "
                                     "the reference (original) cascade window", k, j, i );
                            CV_Error( CV_StsNullPtr, errorstr );
                        }
                    }
                }
            }
        }
    }

    // this is an upper boundary for the whole hidden cascade size
    datasize = sizeof(GpuHidHaarClassifierCascade)                   +
               sizeof(GpuHidHaarStageClassifier) * cascade->count    +
               sizeof(GpuHidHaarClassifier)      * total_classifiers +
               sizeof(GpuHidHaarTreeNode)        * total_nodes;

    *totalclassifier = total_classifiers;
    *size = datasize;
    out = (GpuHidHaarClassifierCascade *)cvAlloc( datasize );
    memset( out, 0, sizeof(*out) );

    /* init header */
    out->count = cascade->count;
    stage_classifier_ptr = (GpuHidHaarStageClassifier *)(out + 1);
    haar_classifier_ptr = (GpuHidHaarClassifier *)(stage_classifier_ptr + cascade->count);
    haar_node_ptr = (GpuHidHaarTreeNode *)(haar_classifier_ptr + total_classifiers);

    out->is_stump_based = 1;
    out->has_tilted_features = has_tilted_features;
    out->is_tree = 0;

    /* initialize internal representation */
    for( i = 0; i < cascade->count; i++ )
    {
        CvHaarStageClassifier *stage_classifier = cascade->stage_classifier + i;
        GpuHidHaarStageClassifier *hid_stage_classifier = stage_classifier_ptr + i;

        hid_stage_classifier->count = stage_classifier->count;
        hid_stage_classifier->threshold = stage_classifier->threshold - icv_stage_threshold_bias;
        hid_stage_classifier->classifier = haar_classifier_ptr;
        hid_stage_classifier->two_rects = 1;
        haar_classifier_ptr += stage_classifier->count;

        /*
        hid_stage_classifier->parent = (stage_classifier->parent == -1)
        ? NULL : stage_classifier_ptr + stage_classifier->parent;
        hid_stage_classifier->next = (stage_classifier->next == -1)
        ? NULL : stage_classifier_ptr + stage_classifier->next;
        hid_stage_classifier->child = (stage_classifier->child == -1)
        ? NULL : stage_classifier_ptr + stage_classifier->child;

        out->is_tree |= hid_stage_classifier->next != NULL;
        */

        for( j = 0; j < stage_classifier->count; j++ )
        {
            CvHaarClassifier *classifier         = stage_classifier->classifier + j;
            GpuHidHaarClassifier *hid_classifier = hid_stage_classifier->classifier + j;
            int node_count = classifier->count;

            //   float* alpha_ptr = (float*)(haar_node_ptr + node_count);
            float *alpha_ptr = &haar_node_ptr->alpha[0];

            hid_classifier->count = node_count;
            hid_classifier->node = haar_node_ptr;
            hid_classifier->alpha = alpha_ptr;

            for( l = 0; l < node_count; l++ )
            {
                GpuHidHaarTreeNode *node     = hid_classifier->node + l;
                CvHaarFeature      *feature = classifier->haar_feature + l;

                memset( node, -1, sizeof(*node) );
                node->threshold = classifier->threshold[l];
                node->left      = classifier->left[l];
                node->right     = classifier->right[l];

                if( fabs(feature->rect[2].weight) < DBL_EPSILON ||
                        feature->rect[2].r.width == 0 ||
                        feature->rect[2].r.height == 0 )
                {
                    node->p[2][0] = 0;
                    node->p[2][1] = 0;
                    node->p[2][2] = 0;
                    node->p[2][3] = 0;
                    node->weight[2] = 0;
                }
                //   memset( &(node->feature.rect[2]), 0, sizeof(node->feature.rect[2]) );
                else
                    hid_stage_classifier->two_rects = 0;
            }

            memcpy( alpha_ptr, classifier->alpha, (node_count + 1)*sizeof(alpha_ptr[0]));
            haar_node_ptr = haar_node_ptr + 1;
            // (GpuHidHaarTreeNode*)cvAlignPtr(alpha_ptr+node_count+1, sizeof(void*));
            //   (GpuHidHaarTreeNode*)(alpha_ptr+node_count+1);

            out->is_stump_based &= node_count == 1;
        }
    }

    cascade->hid_cascade = (CvHidHaarClassifierCascade *)out;
    assert( (char *)haar_node_ptr - (char *)out <= datasize );

    return out;
}


#define sum_elem_ptr(sum,row,col)  \
	((sumtype*)CV_MAT_ELEM_PTR_FAST((sum),(row),(col),sizeof(sumtype)))

#define sqsum_elem_ptr(sqsum,row,col)  \
	((sqsumtype*)CV_MAT_ELEM_PTR_FAST((sqsum),(row),(col),sizeof(sqsumtype)))

#define calc_sum(rect,offset) \
	((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])


static void gpuSetImagesForHaarClassifierCascade( CvHaarClassifierCascade *_cascade,
                                      /*   const CvArr* _sum,
                                      const CvArr* _sqsum,
                                      const CvArr* _tilted_sum,*/
                                      double scale,
                                      int step)
{
    //   CvMat sum_stub, *sum = (CvMat*)_sum;
    //   CvMat sqsum_stub, *sqsum = (CvMat*)_sqsum;
    //   CvMat tilted_stub, *tilted = (CvMat*)_tilted_sum;
    GpuHidHaarClassifierCascade *cascade;
    int coi0 = 0, coi1 = 0;
    int i;
    int datasize;
    int total;
    CvRect equRect;
    double weight_scale;
    GpuHidHaarStageClassifier *stage_classifier;

    if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
        CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( scale <= 0 )
        CV_Error( CV_StsOutOfRange, "Scale must be positive" );

    //   sum = cvGetMat( sum, &sum_stub, &coi0 );
    //   sqsum = cvGetMat( sqsum, &sqsum_stub, &coi1 );

    if( coi0 || coi1 )
        CV_Error( CV_BadCOI, "COI is not supported" );

    //   if( !CV_ARE_SIZES_EQ( sum, sqsum ))
    //       CV_Error( CV_StsUnmatchedSizes, "All integral images must have the same size" );

    //   if( CV_MAT_TYPE(sqsum->type) != CV_64FC1 ||
    //       CV_MAT_TYPE(sum->type) != CV_32SC1 )
    //       CV_Error( CV_StsUnsupportedFormat,
    //       "Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );

    if( !_cascade->hid_cascade )
        gpuCreateHidHaarClassifierCascade(_cascade, &datasize, &total);

    cascade = (GpuHidHaarClassifierCascade *) _cascade->hid_cascade;
    stage_classifier = (GpuHidHaarStageClassifier *) (cascade + 1);

    if( cascade->has_tilted_features )
    {
        //    tilted = cvGetMat( tilted, &tilted_stub, &coi1 );

        //    if( CV_MAT_TYPE(tilted->type) != CV_32SC1 )
        //        CV_Error( CV_StsUnsupportedFormat,
        //        "Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );

        //    if( sum->step != tilted->step )
        //        CV_Error( CV_StsUnmatchedSizes,
        //        "Sum and tilted_sum must have the same stride (step, widthStep)" );

        //    if( !CV_ARE_SIZES_EQ( sum, tilted ))
        //        CV_Error( CV_StsUnmatchedSizes, "All integral images must have the same size" );
        //  cascade->tilted = *tilted;
    }

    _cascade->scale = scale;
    _cascade->real_window_size.width = cvRound( _cascade->orig_window_size.width * scale );
    _cascade->real_window_size.height = cvRound( _cascade->orig_window_size.height * scale );

    //cascade->sum = *sum;
    //cascade->sqsum = *sqsum;

    equRect.x = equRect.y = cvRound(scale);
    equRect.width = cvRound((_cascade->orig_window_size.width - 2) * scale);
    equRect.height = cvRound((_cascade->orig_window_size.height - 2) * scale);
    weight_scale = 1. / (equRect.width * equRect.height);
    cascade->inv_window_area = weight_scale;

    //	cascade->pq0 = equRect.y * step + equRect.x;
    //	cascade->pq1 = equRect.y * step + equRect.x + equRect.width ;
    //	cascade->pq2 = (equRect.y + equRect.height)*step + equRect.x;
    //	cascade->pq3 = (equRect.y + equRect.height)*step + equRect.x + equRect.width ;

    cascade->pq0 = equRect.x;
    cascade->pq1 = equRect.y;
    cascade->pq2 = equRect.x + equRect.width;
    cascade->pq3 = equRect.y + equRect.height;

    cascade->p0 = equRect.x;
    cascade->p1 = equRect.y;
    cascade->p2 = equRect.x + equRect.width;
    cascade->p3 = equRect.y + equRect.height;


    /* init pointers in haar features according to real window size and
    given image pointers */
    for( i = 0; i < _cascade->count; i++ )
    {
        int j, k, l;
        for( j = 0; j < stage_classifier[i].count; j++ )
        {
            for( l = 0; l < stage_classifier[i].classifier[j].count; l++ )
            {
                CvHaarFeature *feature =
                    &_cascade->stage_classifier[i].classifier[j].haar_feature[l];
                /*  GpuHidHaarClassifier* classifier =
                cascade->stage_classifier[i].classifier + j; */
                //GpuHidHaarFeature* hidfeature =
                //    &cascade->stage_classifier[i].classifier[j].node[l].feature;
                GpuHidHaarTreeNode *hidnode = &stage_classifier[i].classifier[j].node[l];
                double sum0 = 0, area0 = 0;
                CvRect r[3];

                int base_w = -1, base_h = -1;
                int new_base_w = 0, new_base_h = 0;
                int kx, ky;
                int flagx = 0, flagy = 0;
                int x0 = 0, y0 = 0;
                int nr;

                /* align blocks */
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    //if( !hidfeature->rect[k].p0 )
                    //    break;
                    if(!hidnode->p[k][0])
                        break;
                    r[k] = feature->rect[k].r;
                    base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].width - 1) );
                    base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].x - r[0].x - 1) );
                    base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].height - 1) );
                    base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].y - r[0].y - 1) );
                }

                nr = k;
                base_w += 1;
                base_h += 1;
                if(base_w == 0)
                    base_w = 1;
                kx = r[0].width / base_w;
                if(base_h == 0)
                    base_h = 1;
                ky = r[0].height / base_h;

                if( kx <= 0 )
                {
                    flagx = 1;
                    new_base_w = cvRound( r[0].width * scale ) / kx;
                    x0 = cvRound( r[0].x * scale );
                }

                if( ky <= 0 )
                {
                    flagy = 1;
                    new_base_h = cvRound( r[0].height * scale ) / ky;
                    y0 = cvRound( r[0].y * scale );
                }

                for( k = 0; k < nr; k++ )
                {
                    CvRect tr;
                    double correction_ratio;

                    if( flagx )
                    {
                        tr.x = (r[k].x - r[0].x) * new_base_w / base_w + x0;
                        tr.width = r[k].width * new_base_w / base_w;
                    }
                    else
                    {
                        tr.x = cvRound( r[k].x * scale );
                        tr.width = cvRound( r[k].width * scale );
                    }

                    if( flagy )
                    {
                        tr.y = (r[k].y - r[0].y) * new_base_h / base_h + y0;
                        tr.height = r[k].height * new_base_h / base_h;
                    }
                    else
                    {
                        tr.y = cvRound( r[k].y * scale );
                        tr.height = cvRound( r[k].height * scale );
                    }

#if CV_ADJUST_WEIGHTS
                    {
                        // RAINER START
                        const float orig_feature_size =  (float)(feature->rect[k].r.width) * feature->rect[k].r.height;
                        const float orig_norm_size = (float)(_cascade->orig_window_size.width) * (_cascade->orig_window_size.height);
                        const float feature_size = float(tr.width * tr.height);
                        //const float normSize    = float(equRect.width*equRect.height);
                        float target_ratio = orig_feature_size / orig_norm_size;
                        //float isRatio = featureSize / normSize;
                        //correctionRatio = targetRatio / isRatio / normSize;
                        correction_ratio = target_ratio / feature_size;
                        // RAINER END
                    }
#else
                    correction_ratio = weight_scale * (!feature->tilted ? 1 : 0.5);
#endif

                    if( !feature->tilted )
                    {
                        /*     hidfeature->rect[k].p0 = tr.y * sum->cols + tr.x;
                        hidfeature->rect[k].p1 = tr.y * sum->cols + tr.x + tr.width;
                        hidfeature->rect[k].p2 = (tr.y + tr.height) * sum->cols + tr.x;
                        hidfeature->rect[k].p3 = (tr.y + tr.height) * sum->cols + tr.x + tr.width;
                        */
                        /*hidnode->p0[k] = tr.y * step + tr.x;
                        hidnode->p1[k] = tr.y * step + tr.x + tr.width;
                        hidnode->p2[k] = (tr.y + tr.height) * step + tr.x;
                        hidnode->p3[k] = (tr.y + tr.height) * step + tr.x + tr.width;*/
                        hidnode->p[k][0] = tr.x;
                        hidnode->p[k][1] = tr.y;
                        hidnode->p[k][2] = tr.x + tr.width;
                        hidnode->p[k][3] = tr.y + tr.height;
                    }
                    else
                    {
                        /*    hidfeature->rect[k].p2 = (tr.y + tr.width) * tilted->cols + tr.x + tr.width;
                        hidfeature->rect[k].p3 = (tr.y + tr.width + tr.height) * tilted->cols + tr.x + tr.width - tr.height;
                        hidfeature->rect[k].p0 = tr.y * tilted->cols + tr.x;
                        hidfeature->rect[k].p1 = (tr.y + tr.height) * tilted->cols + tr.x - tr.height;
                        */

                        hidnode->p[k][2] = (tr.y + tr.width) * step + tr.x + tr.width;
                        hidnode->p[k][3] = (tr.y + tr.width + tr.height) * step + tr.x + tr.width - tr.height;
                        hidnode->p[k][0] = tr.y * step + tr.x;
                        hidnode->p[k][1] = (tr.y + tr.height) * step + tr.x - tr.height;
                    }

                    //hidfeature->rect[k].weight = (float)(feature->rect[k].weight * correction_ratio);
                    hidnode->weight[k] = (float)(feature->rect[k].weight * correction_ratio);
                    if( k == 0 )
                        area0 = tr.width * tr.height;
                    else
                        //sum0 += hidfeature->rect[k].weight * tr.width * tr.height;
                        sum0 += hidnode->weight[k] * tr.width * tr.height;
                }

                // hidfeature->rect[0].weight = (float)(-sum0/area0);
                hidnode->weight[0] = (float)(-sum0 / area0);
            } /* l */
        } /* j */
    }
}

static void gpuSetHaarClassifierCascade( CvHaarClassifierCascade *_cascade
                             /*double scale=0.0,*/
                             /*int step*/)
{
    GpuHidHaarClassifierCascade *cascade;
    int i;
    int datasize;
    int total;
    CvRect equRect;
    double weight_scale;
    GpuHidHaarStageClassifier *stage_classifier;

    if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
        CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( !_cascade->hid_cascade )
        gpuCreateHidHaarClassifierCascade(_cascade, &datasize, &total);

    cascade = (GpuHidHaarClassifierCascade *) _cascade->hid_cascade;
    stage_classifier = (GpuHidHaarStageClassifier *) cascade + 1;

    _cascade->scale = 1.0;
    _cascade->real_window_size.width =  _cascade->orig_window_size.width ;
    _cascade->real_window_size.height = _cascade->orig_window_size.height;

    equRect.x = equRect.y = 1;
    equRect.width = _cascade->orig_window_size.width - 2;
    equRect.height = _cascade->orig_window_size.height - 2;
    weight_scale = 1;
    cascade->inv_window_area = weight_scale;

    cascade->p0 = equRect.x;
    cascade->p1 = equRect.y;
    cascade->p2 = equRect.height;
    cascade->p3 = equRect.width ;
    for( i = 0; i < _cascade->count; i++ )
    {
        int j, k, l;
        for( j = 0; j < stage_classifier[i].count; j++ )
        {
            for( l = 0; l < stage_classifier[i].classifier[j].count; l++ )
            {
                CvHaarFeature *feature =
                    &_cascade->stage_classifier[i].classifier[j].haar_feature[l];
                GpuHidHaarTreeNode *hidnode = &stage_classifier[i].classifier[j].node[l];
                CvRect r[3];


                int nr;

                /* align blocks */
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    if(!hidnode->p[k][0])
                        break;
                    r[k] = feature->rect[k].r;
                    // 					base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].width-1) );
                    // 					base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].x - r[0].x-1) );
                    // 					base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].height-1) );
                    // 					base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].y - r[0].y-1) );
                }

                nr = k;
                for( k = 0; k < nr; k++ )
                {
                    CvRect tr;
                    double correction_ratio;
                    tr.x = r[k].x;
                    tr.width = r[k].width;
                    tr.y = r[k].y ;
                    tr.height = r[k].height;
                    correction_ratio = weight_scale * (!feature->tilted ? 1 : 0.5);
                    hidnode->p[k][0] = tr.x;
                    hidnode->p[k][1] = tr.y;
                    hidnode->p[k][2] = tr.width;
                    hidnode->p[k][3] = tr.height;
                    hidnode->weight[k] = (float)(feature->rect[k].weight * correction_ratio);
                }
                //hidnode->weight[0]=(float)(-sum0/area0);
            } /* l */
        } /* j */
    }
}
CvSeq *cv::ocl::OclCascadeClassifier::oclHaarDetectObjects( oclMat &gimg, CvMemStorage *storage, double scaleFactor,
        int minNeighbors, int flags, CvSize minSize, CvSize maxSize)
{
    CvHaarClassifierCascade *cascade = oldCascade;

    //double alltime = (double)cvGetTickCount();
    //double t = (double)cvGetTickCount();
    const double GROUP_EPS = 0.2;
    oclMat gtemp, gsum1, gtilted1, gsqsum1, gnormImg, gsumcanny;
    CvSeq *result_seq = 0;
    cv::Ptr<CvMemStorage> temp_storage;

    cv::ConcurrentRectVector allCandidates;
    std::vector<cv::Rect> rectList;
    std::vector<int> rweights;
    double factor;
    int datasize=0;
    int totalclassifier=0;

    //void *out;
    GpuHidHaarClassifierCascade *gcascade;
    GpuHidHaarStageClassifier    *stage;
    GpuHidHaarClassifier         *classifier;
    GpuHidHaarTreeNode           *node;

    int *candidate;
    cl_int status;

    //    bool doCannyPruning = (flags & CV_HAAR_DO_CANNY_PRUNING) != 0;
    bool findBiggestObject = (flags & CV_HAAR_FIND_BIGGEST_OBJECT) != 0;
    //    bool roughSearch = (flags & CV_HAAR_DO_ROUGH_SEARCH) != 0;

    //double t = 0;
    if( maxSize.height == 0 || maxSize.width == 0 )
    {
        maxSize.height = gimg.rows;
        maxSize.width = gimg.cols;
    }

    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier cascade" );

    if( !storage )
        CV_Error( CV_StsNullPtr, "Null storage pointer" );

    if( CV_MAT_DEPTH(gimg.type()) != CV_8U )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );

    if( scaleFactor <= 1 )
        CV_Error( CV_StsOutOfRange, "scale factor must be > 1" );

    if( findBiggestObject )
        flags &= ~CV_HAAR_SCALE_IMAGE;

    //gtemp = oclMat( gimg.rows, gimg.cols, CV_8UC1);
    //gsum1 =  oclMat( gimg.rows + 1, gimg.cols + 1, CV_32SC1 );
    //gsqsum1 = oclMat( gimg.rows + 1, gimg.cols + 1, CV_32FC1 );

    if( !cascade->hid_cascade )
        /*out = (void *)*/gpuCreateHidHaarClassifierCascade(cascade, &datasize, &totalclassifier);
    if( cascade->hid_cascade->has_tilted_features )
        gtilted1 = oclMat( gimg.rows + 1, gimg.cols + 1, CV_32SC1 );

    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

    if( CV_MAT_CN(gimg.type()) > 1 )
    {
        cvtColor( gimg, gtemp, CV_BGR2GRAY );
        gimg = gtemp;
    }

    if( findBiggestObject )
        flags &= ~(CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING);
    //t = (double)cvGetTickCount() - t;
    //printf( "before if time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    if( gimg.cols < minSize.width || gimg.rows < minSize.height )
        CV_Error(CV_StsError, "Image too small");

    if( (flags & CV_HAAR_SCALE_IMAGE) )
    {
        CvSize winSize0 = cascade->orig_window_size;
        //float scalefactor = 1.1f;
        //float factor = 1.f;
        int totalheight = 0;
        int indexy = 0;
        CvSize sz;
        //t = (double)cvGetTickCount();
        std::vector<CvSize> sizev;
        std::vector<float> scalev;
        for(factor = 1.f;; factor *= scaleFactor)
        {
            CvSize winSize = { cvRound(winSize0.width * factor), cvRound(winSize0.height * factor) };
            sz.width     = cvRound( gimg.cols / factor ) + 1;
            sz.height    = cvRound( gimg.rows / factor ) + 1;
            CvSize sz1     = { sz.width - winSize0.width - 1,      sz.height - winSize0.height - 1 };

            if( sz1.width <= 0 || sz1.height <= 0 )
                break;
            if( winSize.width > maxSize.width || winSize.height > maxSize.height )
                break;
            if( winSize.width < minSize.width || winSize.height < minSize.height )
                continue;

            totalheight += sz.height;
            sizev.push_back(sz);
            scalev.push_back(factor);
        }
        //int flag = 0;

        oclMat gimg1(gimg.rows, gimg.cols, CV_8UC1);
        oclMat gsum(totalheight + 4, gimg.cols + 1, CV_32SC1);
        oclMat gsqsum(totalheight + 4, gimg.cols + 1, CV_32FC1);

        //cl_mem cascadebuffer;
        cl_mem stagebuffer;
        //cl_mem classifierbuffer;
        cl_mem nodebuffer;
        cl_mem candidatebuffer;
        cl_mem scaleinfobuffer;
        //cl_kernel kernel;
        //kernel = openCLGetKernelFromSource(gimg.clCxt, &haarobjectdetect, "gpuRunHaarClassifierCascade");
        cv::Rect roi, roi2;
        cv::Mat imgroi, imgroisq;
        cv::ocl::oclMat resizeroi, gimgroi, gimgroisq;
        int grp_per_CU = 12;

        size_t blocksize = 8;
        size_t localThreads[3] = { blocksize, blocksize , 1 };
        size_t globalThreads[3] = { grp_per_CU *((gsum.clCxt)->computeUnits()) *localThreads[0],
                                    localThreads[1], 1
                                  };
        int outputsz = 256 * globalThreads[0] / localThreads[0];
        int loopcount = sizev.size();
        detect_piramid_info *scaleinfo = (detect_piramid_info *)malloc(sizeof(detect_piramid_info) * loopcount);

        //t = (double)cvGetTickCount() - t;
        // printf( "pre time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        //int *it =scaleinfo;
        // t = (double)cvGetTickCount();

        for( int i = 0; i < loopcount; i++ )
        {
            sz = sizev[i];
            factor = scalev[i];
            roi = Rect(0, indexy, sz.width, sz.height);
            roi2 = Rect(0, 0, sz.width - 1, sz.height - 1);
            resizeroi = gimg1(roi2);
            gimgroi = gsum(roi);
            gimgroisq = gsqsum(roi);
            //scaleinfo[i].rows = gimgroi.rows;
            int width = gimgroi.cols - 1 - cascade->orig_window_size.width;
            int height = gimgroi.rows - 1 - cascade->orig_window_size.height;
            scaleinfo[i].width_height = (width << 16) | height;


            int grpnumperline = (width + localThreads[0] - 1) / localThreads[0];
            int totalgrp = ((height + localThreads[1] - 1) / localThreads[1]) * grpnumperline;
            //outputsz +=width*height;

            scaleinfo[i].grpnumperline_totalgrp = (grpnumperline << 16) | totalgrp;
            scaleinfo[i].imgoff = gimgroi.offset >> 2;
            scaleinfo[i].factor = factor;
            //printf("rows = %d,ystep = %d,width = %d,height = %d,grpnumperline = %d,totalgrp = %d,imgoff = %d,factor = %f\n",
            //	scaleinfo[i].rows,scaleinfo[i].ystep,scaleinfo[i].width,scaleinfo[i].height,scaleinfo[i].grpnumperline,
            //	scaleinfo[i].totalgrp,scaleinfo[i].imgoff,scaleinfo[i].factor);
            cv::ocl::resize(gimg, resizeroi, Size(sz.width - 1, sz.height - 1), 0, 0, INTER_LINEAR);
            //cv::imwrite("D:\\1.jpg",gimg1);
            cv::ocl::integral(resizeroi, gimgroi, gimgroisq);
            //cv::ocl::oclMat chk(sz.height,sz.width,CV_32SC1),chksq(sz.height,sz.width,CV_32FC1);
            //cv::ocl::integral(gimg1, chk, chksq);
            //double r = cv::norm(chk,gimgroi,NORM_INF);
            //if(r > std::numeric_limits<double>::epsilon())
            //{
            //	printf("failed");
            //}
            indexy += sz.height;
        }
        //int ystep = factor > 2 ? 1 : 2;
        // t = (double)cvGetTickCount() - t;
        //printf( "resize integral time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        //t = (double)cvGetTickCount();
        gcascade   = (GpuHidHaarClassifierCascade *)cascade->hid_cascade;
        stage      = (GpuHidHaarStageClassifier *)(gcascade + 1);
        classifier = (GpuHidHaarClassifier *)(stage + gcascade->count);
        node       = (GpuHidHaarTreeNode *)(classifier->node);

        //int m,n;
        //m = (gsum.cols - 1 - cascade->orig_window_size.width  + ystep - 1)/ystep;
        //n = (gsum.rows - 1 - cascade->orig_window_size.height + ystep - 1)/ystep;
        //int counter = m*n;

        int nodenum = (datasize - sizeof(GpuHidHaarClassifierCascade) -
                       sizeof(GpuHidHaarStageClassifier) * gcascade->count - sizeof(GpuHidHaarClassifier) * totalclassifier) / sizeof(GpuHidHaarTreeNode);
        //if(flag == 0){
        candidate = (int *)malloc(4 * sizeof(int) * outputsz);
        //memset((char*)candidate,0,4*sizeof(int)*outputsz);
        gpuSetImagesForHaarClassifierCascade( cascade,/* &sum1, &sqsum1, _tilted,*/ 1., gsum.step / 4 );

        //cascadebuffer = clCreateBuffer(gsum.clCxt->clContext,CL_MEM_READ_ONLY,sizeof(GpuHidHaarClassifierCascade),NULL,&status);
        //openCLVerifyCall(status);
        //openCLSafeCall(clEnqueueWriteBuffer(gsum.clCxt->clCmdQueue,cascadebuffer,1,0,sizeof(GpuHidHaarClassifierCascade),gcascade,0,NULL,NULL));

        stagebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(GpuHidHaarStageClassifier) * gcascade->count);
        //openCLVerifyCall(status);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), stagebuffer, 1, 0, sizeof(GpuHidHaarStageClassifier)*gcascade->count, stage, 0, NULL, NULL));

        //classifierbuffer = clCreateBuffer(gsum.clCxt->clContext,CL_MEM_READ_ONLY,sizeof(GpuHidHaarClassifier)*totalclassifier,NULL,&status);
        //status = clEnqueueWriteBuffer(gsum.clCxt->clCmdQueue,classifierbuffer,1,0,sizeof(GpuHidHaarClassifier)*totalclassifier,classifier,0,NULL,NULL);

        nodebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, nodenum * sizeof(GpuHidHaarTreeNode));
        //openCLVerifyCall(status);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), nodebuffer, 1, 0,
                                            nodenum * sizeof(GpuHidHaarTreeNode),
                                            node, 0, NULL, NULL));
        candidatebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_WRITE_ONLY, 4 * sizeof(int) * outputsz);
        //openCLVerifyCall(status);
        scaleinfobuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(detect_piramid_info) * loopcount);
        //openCLVerifyCall(status);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), scaleinfobuffer, 1, 0, sizeof(detect_piramid_info)*loopcount, scaleinfo, 0, NULL, NULL));
        //flag  = 1;
        //}

        //t = (double)cvGetTickCount() - t;
        //printf( "update time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

        //size_t globalThreads[3] = { counter+blocksize*blocksize-counter%(blocksize*blocksize),1,1};
        //t = (double)cvGetTickCount();
        int startstage = 0;
        int endstage = gcascade->count;
        int startnode = 0;
        int pixelstep = gsum.step / 4;
        int splitstage = 3;
        int splitnode = stage[0].count + stage[1].count + stage[2].count;
        cl_int4 p, pq;
        p.s[0] = gcascade->p0;
        p.s[1] = gcascade->p1;
        p.s[2] = gcascade->p2;
        p.s[3] = gcascade->p3;
        pq.s[0] = gcascade->pq0;
        pq.s[1] = gcascade->pq1;
        pq.s[2] = gcascade->pq2;
        pq.s[3] = gcascade->pq3;
        float correction = gcascade->inv_window_area;

        //int grpnumperline = ((m + localThreads[0] - 1) / localThreads[0]);
        //int totalgrp = ((n + localThreads[1] - 1) / localThreads[1])*grpnumperline;
        //   openCLVerifyKernel(gsum.clCxt, kernel, &blocksize, globalThreads, localThreads);
        //openCLSafeCall(clSetKernelArg(kernel,argcount++,sizeof(cl_mem),(void*)&cascadebuffer));

        std::vector<std::pair<size_t, const void *> > args;
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&stagebuffer ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&scaleinfobuffer ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&nodebuffer ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&gsum.data ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&gsqsum.data ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&candidatebuffer ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&pixelstep ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&loopcount ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&startstage ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&splitstage ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&endstage ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&startnode ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&splitnode ));
        args.push_back ( std::make_pair(sizeof(cl_int4) , (void *)&p ));
        args.push_back ( std::make_pair(sizeof(cl_int4) , (void *)&pq ));
        args.push_back ( std::make_pair(sizeof(cl_float) , (void *)&correction ));

        openCLExecuteKernel(gsum.clCxt, &haarobjectdetect, "gpuRunHaarClassifierCascade", globalThreads, localThreads, args, -1, -1);
        //t = (double)cvGetTickCount() - t;
        //printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        //t = (double)cvGetTickCount();
        //openCLSafeCall(clEnqueueReadBuffer(gsum.clCxt->impl->clCmdQueue, candidatebuffer, 1, 0, 4 * sizeof(int)*outputsz, candidate, 0, NULL, NULL));
        openCLReadBuffer( gsum.clCxt, candidatebuffer, candidate, 4 * sizeof(int)*outputsz );

        for(int i = 0; i < outputsz; i++)
            if(candidate[4 * i + 2] != 0)
                allCandidates.push_back(Rect(candidate[4 * i], candidate[4 * i + 1], candidate[4 * i + 2], candidate[4 * i + 3]));
        // t = (double)cvGetTickCount() - t;
        //printf( "post time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        //t = (double)cvGetTickCount();
        free(scaleinfo);
        free(candidate);
        //openCLSafeCall(clReleaseMemObject(cascadebuffer));
        openCLSafeCall(clReleaseMemObject(stagebuffer));
        openCLSafeCall(clReleaseMemObject(scaleinfobuffer));
        openCLSafeCall(clReleaseMemObject(nodebuffer));
        openCLSafeCall(clReleaseMemObject(candidatebuffer));
        // openCLSafeCall(clReleaseKernel(kernel));
        //t = (double)cvGetTickCount() - t;
        //printf( "release time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    }
    else
    {
        CvSize winsize0 = cascade->orig_window_size;
        int n_factors = 0;
        oclMat gsum;
        oclMat gsqsum;
        cv::ocl::integral(gimg, gsum, gsqsum);
        CvSize sz;
        std::vector<CvSize> sizev;
        std::vector<float> scalev;
        gpuSetHaarClassifierCascade(cascade);
        gcascade   = (GpuHidHaarClassifierCascade *)cascade->hid_cascade;
        stage      = (GpuHidHaarStageClassifier *)(gcascade + 1);
        classifier = (GpuHidHaarClassifier *)(stage + gcascade->count);
        node       = (GpuHidHaarTreeNode *)(classifier->node);
        cl_mem stagebuffer;
        //cl_mem classifierbuffer;
        cl_mem nodebuffer;
        cl_mem candidatebuffer;
        cl_mem scaleinfobuffer;
        cl_mem pbuffer;
        cl_mem correctionbuffer;
        for( n_factors = 0, factor = 1;
                cvRound(factor * winsize0.width) < gimg.cols - 10 &&
                cvRound(factor * winsize0.height) < gimg.rows - 10;
                n_factors++, factor *= scaleFactor )
        {
            CvSize winSize = { cvRound( winsize0.width * factor ),
                               cvRound( winsize0.height * factor )
                             };
            if( winSize.width < minSize.width || winSize.height < minSize.height )
            {
                continue;
            }
            sizev.push_back(winSize);
            scalev.push_back(factor);
        }
        int loopcount = scalev.size();
        if(loopcount == 0)
        {
            loopcount = 1;
            n_factors = 1;
            sizev.push_back(minSize);
            scalev.push_back( std::min(cvRound(minSize.width / winsize0.width), cvRound(minSize.height / winsize0.height)) );

        }
        detect_piramid_info *scaleinfo = (detect_piramid_info *)malloc(sizeof(detect_piramid_info) * loopcount);
        cl_int4 *p = (cl_int4 *)malloc(sizeof(cl_int4) * loopcount);
        float *correction = (float *)malloc(sizeof(float) * loopcount);
        int grp_per_CU = 12;
        size_t blocksize = 8;
        size_t localThreads[3] = { blocksize, blocksize , 1 };
        size_t globalThreads[3] = { grp_per_CU *gsum.clCxt->computeUnits() *localThreads[0],
                                    localThreads[1], 1
                                  };
        int outputsz = 256 * globalThreads[0] / localThreads[0];
        int nodenum = (datasize - sizeof(GpuHidHaarClassifierCascade) -
                       sizeof(GpuHidHaarStageClassifier) * gcascade->count - sizeof(GpuHidHaarClassifier) * totalclassifier) / sizeof(GpuHidHaarTreeNode);
        nodebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY,
                                        nodenum * sizeof(GpuHidHaarTreeNode));
        //openCLVerifyCall(status);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), nodebuffer, 1, 0,
                                            nodenum * sizeof(GpuHidHaarTreeNode),
                                            node, 0, NULL, NULL));
        cl_mem newnodebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_WRITE,
                               loopcount * nodenum * sizeof(GpuHidHaarTreeNode));
        int startstage = 0;
        int endstage = gcascade->count;
        //cl_kernel kernel;
        //kernel = openCLGetKernelFromSource(gsum.clCxt, &haarobjectdetect_scaled2, "gpuRunHaarClassifierCascade_scaled2");
        //cl_kernel kernel2 = openCLGetKernelFromSource(gimg.clCxt, &haarobjectdetect_scaled2, "gpuscaleclassifier");
        for(int i = 0; i < loopcount; i++)
        {
            sz = sizev[i];
            factor = scalev[i];
            int ystep = cvRound(std::max(2., factor));
            int equRect_x = (int)(factor * gcascade->p0 + 0.5);
            int equRect_y = (int)(factor * gcascade->p1 + 0.5);
            int equRect_w = (int)(factor * gcascade->p3 + 0.5);
            int equRect_h = (int)(factor * gcascade->p2 + 0.5);
            p[i].s[0] = equRect_x;
            p[i].s[1] = equRect_y;
            p[i].s[2] = equRect_x + equRect_w;
            p[i].s[3] = equRect_y + equRect_h;
            correction[i] = 1. / (equRect_w * equRect_h);
            int width = (gsum.cols - 1 - sz.width  + ystep - 1) / ystep;
            int height = (gsum.rows - 1 - sz.height + ystep - 1) / ystep;
            int grpnumperline = (width + localThreads[0] - 1) / localThreads[0];
            int totalgrp = ((height + localThreads[1] - 1) / localThreads[1]) * grpnumperline;
            //outputsz +=width*height;
            scaleinfo[i].width_height = (width << 16) | height;
            scaleinfo[i].grpnumperline_totalgrp = (grpnumperline << 16) | totalgrp;
            scaleinfo[i].imgoff = 0;
            scaleinfo[i].factor = factor;
            int startnodenum = nodenum * i;
            float factor2 = (float)factor;

            std::vector<std::pair<size_t, const void *> > args1;
            args1.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&nodebuffer ));
            args1.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&newnodebuffer ));
            args1.push_back ( std::make_pair(sizeof(cl_float) , (void *)&factor2 ));
            args1.push_back ( std::make_pair(sizeof(cl_float) , (void *)&correction[i] ));
            args1.push_back ( std::make_pair(sizeof(cl_int) , (void *)&startnodenum ));

            size_t globalThreads2[3] = {nodenum, 1, 1};

            openCLExecuteKernel(gsum.clCxt, &haarobjectdetect_scaled2, "gpuscaleclassifier", globalThreads2, NULL/*localThreads2*/, args1, -1, -1);

            //clEnqueueNDRangeKernel(gsum.clCxt->impl->clCmdQueue, kernel2, 1, NULL, globalThreads2, 0, 0, NULL, NULL);
            //clFinish(gsum.clCxt->impl->clCmdQueue);
        }
        //clReleaseKernel(kernel2);
        int step = gsum.step / 4;
        int startnode = 0;
        int splitstage = 3;
        int splitnode = stage[0].count + stage[1].count + stage[2].count;
        stagebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(GpuHidHaarStageClassifier) * gcascade->count);
        //openCLVerifyCall(status);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), stagebuffer, 1, 0, sizeof(GpuHidHaarStageClassifier)*gcascade->count, stage, 0, NULL, NULL));
        candidatebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 4 * sizeof(int) * outputsz);
        //openCLVerifyCall(status);
        scaleinfobuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(detect_piramid_info) * loopcount);
        //openCLVerifyCall(status);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), scaleinfobuffer, 1, 0, sizeof(detect_piramid_info)*loopcount, scaleinfo, 0, NULL, NULL));
        pbuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(cl_int4) * loopcount);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), pbuffer, 1, 0, sizeof(cl_int4)*loopcount, p, 0, NULL, NULL));
        correctionbuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(cl_float) * loopcount);
        openCLSafeCall(clEnqueueWriteBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), correctionbuffer, 1, 0, sizeof(cl_float)*loopcount, correction, 0, NULL, NULL));
        //int argcount = 0;

        std::vector<std::pair<size_t, const void *> > args;
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&stagebuffer ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&scaleinfobuffer ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&newnodebuffer ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&gsum.data ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&gsqsum.data ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&candidatebuffer ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&step ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&loopcount ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&startstage ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&splitstage ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&endstage ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&startnode ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&splitnode ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&pbuffer ));
        args.push_back ( std::make_pair(sizeof(cl_mem) , (void *)&correctionbuffer ));
        args.push_back ( std::make_pair(sizeof(cl_int) , (void *)&nodenum ));


        openCLExecuteKernel(gsum.clCxt, &haarobjectdetect_scaled2, "gpuRunHaarClassifierCascade_scaled2", globalThreads, localThreads, args, -1, -1);

        //openCLSafeCall(clEnqueueReadBuffer(gsum.clCxt->clCmdQueue,candidatebuffer,1,0,4*sizeof(int)*outputsz,candidate,0,NULL,NULL));
        candidate = (int *)clEnqueueMapBuffer((cl_command_queue)gsum.clCxt->oclCommandQueue(), candidatebuffer, 1, CL_MAP_READ, 0, 4 * sizeof(int), 0, 0, 0, &status);

        for(int i = 0; i < outputsz; i++)
        {
            if(candidate[4 * i + 2] != 0)
                allCandidates.push_back(Rect(candidate[4 * i], candidate[4 * i + 1], candidate[4 * i + 2], candidate[4 * i + 3]));
        }

        free(scaleinfo);
        free(p);
        free(correction);
        clEnqueueUnmapMemObject((cl_command_queue)gsum.clCxt->oclCommandQueue(), candidatebuffer, candidate, 0, 0, 0);
        openCLSafeCall(clReleaseMemObject(stagebuffer));
        openCLSafeCall(clReleaseMemObject(scaleinfobuffer));
        openCLSafeCall(clReleaseMemObject(nodebuffer));
        openCLSafeCall(clReleaseMemObject(newnodebuffer));
        openCLSafeCall(clReleaseMemObject(candidatebuffer));
        openCLSafeCall(clReleaseMemObject(pbuffer));
        openCLSafeCall(clReleaseMemObject(correctionbuffer));
    }
    //t = (double)cvGetTickCount() ;
    cvFree(&cascade->hid_cascade);
    //    printf("%d\n",globalcounter);
    rectList.resize(allCandidates.size());
    if(!allCandidates.empty())
        std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

    //cout << "count = " << rectList.size()<< endl;

    if( minNeighbors != 0 || findBiggestObject )
        groupRectangles(rectList, rweights, std::max(minNeighbors, 1), GROUP_EPS);
    else
        rweights.resize(rectList.size(), 0);


    if( findBiggestObject && rectList.size() )
    {
        CvAvgComp result_comp = {{0, 0, 0, 0}, 0};

        for( size_t i = 0; i < rectList.size(); i++ )
        {
            cv::Rect r = rectList[i];
            if( r.area() > cv::Rect(result_comp.rect).area() )
            {
                result_comp.rect = r;
                result_comp.neighbors = rweights[i];
            }
        }
        cvSeqPush( result_seq, &result_comp );
    }
    else
    {
        for( size_t i = 0; i < rectList.size(); i++ )
        {
            CvAvgComp c;
            c.rect = rectList[i];
            c.neighbors = rweights[i];
            cvSeqPush( result_seq, &c );
        }
    }
    //t = (double)cvGetTickCount() - t;
    //printf( "get face time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    //alltime = (double)cvGetTickCount() - alltime;
    //printf( "all time = %g ms\n", alltime/((double)cvGetTickFrequency()*1000.) );
    return result_seq;
}


#ifndef _MAX_PATH
#define _MAX_PATH 1024
#endif


/****************************************************************************************\
*                                  Persistence functions                                 *
\****************************************************************************************/

/* field names */

#define ICV_HAAR_SIZE_NAME            "size"
#define ICV_HAAR_STAGES_NAME          "stages"
#define ICV_HAAR_TREES_NAME             "trees"
#define ICV_HAAR_FEATURE_NAME             "feature"
#define ICV_HAAR_RECTS_NAME                 "rects"
#define ICV_HAAR_TILTED_NAME                "tilted"
#define ICV_HAAR_THRESHOLD_NAME           "threshold"
#define ICV_HAAR_LEFT_NODE_NAME           "left_node"
#define ICV_HAAR_LEFT_VAL_NAME            "left_val"
#define ICV_HAAR_RIGHT_NODE_NAME          "right_node"
#define ICV_HAAR_RIGHT_VAL_NAME           "right_val"
#define ICV_HAAR_STAGE_THRESHOLD_NAME   "stage_threshold"
#define ICV_HAAR_PARENT_NAME            "parent"
#define ICV_HAAR_NEXT_NAME              "next"

static int gpuRunHaarClassifierCascade( /*const CvHaarClassifierCascade *_cascade, CvPoint pt, int start_stage */)
{
    return 1;
}

namespace cv
{
namespace ocl
{

struct gpuHaarDetectObjects_ScaleImage_Invoker
{
    gpuHaarDetectObjects_ScaleImage_Invoker( const CvHaarClassifierCascade *_cascade,
            int _stripSize, double _factor,
            const Mat &_sum1, const Mat &_sqsum1, Mat *_norm1,
            Mat *_mask1, Rect _equRect, ConcurrentRectVector &_vec )
    {
        cascade = _cascade;
        stripSize = _stripSize;
        factor = _factor;
        sum1 = _sum1;
        sqsum1 = _sqsum1;
        norm1 = _norm1;
        mask1 = _mask1;
        equRect = _equRect;
        vec = &_vec;
    }

    void operator()( const BlockedRange &range ) const
    {
        Size winSize0 = cascade->orig_window_size;
        Size winSize(cvRound(winSize0.width * factor), cvRound(winSize0.height * factor));
        int y1 = range.begin() * stripSize, y2 = std::min(range.end() * stripSize, sum1.rows - 1 - winSize0.height);
        Size ssz(sum1.cols - 1 - winSize0.width, y2 - y1);
        int x, y, ystep = factor > 2 ? 1 : 2;

        for( y = y1; y < y2; y += ystep )
            for( x = 0; x < ssz.width; x += ystep )
            {
                if( gpuRunHaarClassifierCascade( /*cascade, cvPoint(x, y), 0*/ ) > 0 )
                    vec->push_back(Rect(cvRound(x * factor), cvRound(y * factor),
                                        winSize.width, winSize.height));
            }
    }

    const CvHaarClassifierCascade *cascade;
    int stripSize;
    double factor;
    Mat sum1, sqsum1, *norm1, *mask1;
    Rect equRect;
    ConcurrentRectVector *vec;
};


struct gpuHaarDetectObjects_ScaleCascade_Invoker
{
    gpuHaarDetectObjects_ScaleCascade_Invoker( const CvHaarClassifierCascade *_cascade,
            Size _winsize, const Range &_xrange, double _ystep,
            size_t _sumstep, const int **_p, const int **_pq,
            ConcurrentRectVector &_vec )
    {
        cascade = _cascade;
        winsize = _winsize;
        xrange = _xrange;
        ystep = _ystep;
        sumstep = _sumstep;
        p = _p;
        pq = _pq;
        vec = &_vec;
    }

    void operator()( const BlockedRange &range ) const
    {
        int iy, startY = range.begin(), endY = range.end();
        const int *p0 = p[0], *p1 = p[1], *p2 = p[2], *p3 = p[3];
        const int *pq0 = pq[0], *pq1 = pq[1], *pq2 = pq[2], *pq3 = pq[3];
        bool doCannyPruning = p0 != 0;
        int sstep = (int)(sumstep / sizeof(p0[0]));

        for( iy = startY; iy < endY; iy++ )
        {
            int ix, y = cvRound(iy * ystep), ixstep = 1;
            for( ix = xrange.start; ix < xrange.end; ix += ixstep )
            {
                int x = cvRound(ix * ystep); // it should really be ystep, not ixstep

                if( doCannyPruning )
                {
                    int offset = y * sstep + x;
                    int s = p0[offset] - p1[offset] - p2[offset] + p3[offset];
                    int sq = pq0[offset] - pq1[offset] - pq2[offset] + pq3[offset];
                    if( s < 100 || sq < 20 )
                    {
                        ixstep = 2;
                        continue;
                    }
                }

                int result = gpuRunHaarClassifierCascade(/* cascade, cvPoint(x, y), 0 */);
                if( result > 0 )
                    vec->push_back(Rect(x, y, winsize.width, winsize.height));
                ixstep = result != 0 ? 1 : 2;
            }
        }
    }

    const CvHaarClassifierCascade *cascade;
    double ystep;
    size_t sumstep;
    Size winsize;
    Range xrange;
    const int **p;
    const int **pq;
    ConcurrentRectVector *vec;
};

}
}
