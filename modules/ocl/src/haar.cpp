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
//    Sen Liu, swjtuls1987@126.com
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

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
    int width_height;
    int grpnumperline_totalgrp;
    int imgoff;
    float factor;
} detect_piramid_info;
#ifdef _MSC_VER
#define _ALIGNED_ON(_ALIGNMENT) __declspec(align(_ALIGNMENT))

typedef _ALIGNED_ON(128) struct  GpuHidHaarTreeNode
{
    _ALIGNED_ON(64) int p[CV_HAAR_FEATURE_MAX][4];
    float weight[CV_HAAR_FEATURE_MAX] ;
    float threshold ;
    _ALIGNED_ON(16) float alpha[3] ;
    _ALIGNED_ON(4) int left ;
    _ALIGNED_ON(4) int right ;
}
GpuHidHaarTreeNode;


typedef  _ALIGNED_ON(32) struct  GpuHidHaarClassifier
{
    _ALIGNED_ON(4) int count;
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
} GpuHidHaarClassifierCascade;
#else
#define _ALIGNED_ON(_ALIGNMENT) __attribute__((aligned(_ALIGNMENT) ))

typedef struct _ALIGNED_ON(128) GpuHidHaarTreeNode
{
    int p[CV_HAAR_FEATURE_MAX][4] _ALIGNED_ON(64);
    float weight[CV_HAAR_FEATURE_MAX];// _ALIGNED_ON(16);
    float threshold;// _ALIGNED_ON(4);
    float alpha[3] _ALIGNED_ON(16);
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
} GpuHidHaarClassifierCascade;
#endif

const int icv_object_win_border = 1;
const float icv_stage_threshold_bias = 0.0001f;
double globaltime = 0;

/* create more efficient internal representation of haar classifier cascade */
static GpuHidHaarClassifierCascade * gpuCreateHidHaarClassifierCascade( CvHaarClassifierCascade *cascade, int *size, int *totalclassifier)
{
    GpuHidHaarClassifierCascade *out = 0;

    int i, j, k, l;
    int datasize;
    int total_classifiers = 0;
    int total_nodes = 0;
    char errorstr[256];

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

        for( j = 0; j < stage_classifier->count; j++ )
        {
            CvHaarClassifier *classifier         = stage_classifier->classifier + j;
            GpuHidHaarClassifier *hid_classifier = hid_stage_classifier->classifier + j;
            int node_count = classifier->count;

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
                else
                    hid_stage_classifier->two_rects = 0;

                memcpy( node->alpha, classifier->alpha, (node_count + 1)*sizeof(alpha_ptr[0]));
                haar_node_ptr = haar_node_ptr + 1;
            }
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
                                      double scale,
                                      int step)
{
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

    if( coi0 || coi1 )
        CV_Error( CV_BadCOI, "COI is not supported" );

    if( !_cascade->hid_cascade )
        gpuCreateHidHaarClassifierCascade(_cascade, &datasize, &total);

    cascade = (GpuHidHaarClassifierCascade *) _cascade->hid_cascade;
    stage_classifier = (GpuHidHaarStageClassifier *) (cascade + 1);

    _cascade->scale = scale;
    _cascade->real_window_size.width = cvRound( _cascade->orig_window_size.width * scale );
    _cascade->real_window_size.height = cvRound( _cascade->orig_window_size.height * scale );

    equRect.x = equRect.y = cvRound(scale);
    equRect.width = cvRound((_cascade->orig_window_size.width - 2) * scale);
    equRect.height = cvRound((_cascade->orig_window_size.height - 2) * scale);
    weight_scale = 1. / (equRect.width * equRect.height);
    cascade->inv_window_area = weight_scale;

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
                        hidnode->p[k][0] = tr.x;
                        hidnode->p[k][1] = tr.y;
                        hidnode->p[k][2] = tr.x + tr.width;
                        hidnode->p[k][3] = tr.y + tr.height;
                    }
                    else
                    {
                        hidnode->p[k][2] = (tr.y + tr.width) * step + tr.x + tr.width;
                        hidnode->p[k][3] = (tr.y + tr.width + tr.height) * step + tr.x + tr.width - tr.height;
                        hidnode->p[k][0] = tr.y * step + tr.x;
                        hidnode->p[k][1] = (tr.y + tr.height) * step + tr.x - tr.height;
                    }
                    hidnode->weight[k] = (float)(feature->rect[k].weight * correction_ratio);
                    if( k == 0 )
                        area0 = tr.width * tr.height;
                    else
                        sum0 += hidnode->weight[k] * tr.width * tr.height;
                }
                hidnode->weight[0] = (float)(-sum0 / area0);
            } /* l */
        } /* j */
    }
}

static void gpuSetHaarClassifierCascade( CvHaarClassifierCascade *_cascade)
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
        int j, l;
        for( j = 0; j < stage_classifier[i].count; j++ )
        {
            for( l = 0; l < stage_classifier[i].classifier[j].count; l++ )
            {
                const CvHaarFeature *feature =
                    &_cascade->stage_classifier[i].classifier[j].haar_feature[l];
                GpuHidHaarTreeNode *hidnode = &stage_classifier[i].classifier[j].node[l];

                for( int k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    const CvRect tr = feature->rect[k].r;
                    if (tr.width == 0)
                        break;
                    double correction_ratio = weight_scale * (!feature->tilted ? 1 : 0.5);
                    hidnode->p[k][0] = tr.x;
                    hidnode->p[k][1] = tr.y;
                    hidnode->p[k][2] = tr.width;
                    hidnode->p[k][3] = tr.height;
                    hidnode->weight[k] = (float)(feature->rect[k].weight * correction_ratio);
                }
            } /* l */
        } /* j */
    }
}

CvSeq *cv::ocl::OclCascadeClassifier::oclHaarDetectObjects( oclMat &gimg, CvMemStorage *storage, double scaleFactor,
        int minNeighbors, int flags, CvSize minSize, CvSize maxSize)
{
    CvHaarClassifierCascade *cascade = oldCascade;

    const double GROUP_EPS = 0.2;
    CvSeq *result_seq = 0;

    cv::ConcurrentRectVector allCandidates;
    std::vector<cv::Rect> rectList;
    std::vector<int> rweights;
    double factor;
    int datasize=0;
    int totalclassifier=0;

    GpuHidHaarClassifierCascade *gcascade;
    GpuHidHaarStageClassifier    *stage;
    GpuHidHaarClassifier         *classifier;
    GpuHidHaarTreeNode           *node;

    int *candidate;
    cl_int status;

    bool findBiggestObject = (flags & CV_HAAR_FIND_BIGGEST_OBJECT) != 0;

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

    if( !cascade->hid_cascade )
        gpuCreateHidHaarClassifierCascade(cascade, &datasize, &totalclassifier);

    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

    if( CV_MAT_CN(gimg.type()) > 1 )
    {
        oclMat gtemp;
        cvtColor( gimg, gtemp, CV_BGR2GRAY );
        gimg = gtemp;
    }

    if( findBiggestObject )
        flags &= ~(CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING);

    if( gimg.cols < minSize.width || gimg.rows < minSize.height )
        CV_Error(CV_StsError, "Image too small");

    cl_command_queue qu = getClCommandQueue(Context::getContext());
    if( (flags & CV_HAAR_SCALE_IMAGE) )
    {
        CvSize winSize0 = cascade->orig_window_size;
        int totalheight = 0;
        int indexy = 0;
        CvSize sz;
        vector<CvSize> sizev;
        vector<float> scalev;
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

        oclMat gimg1(gimg.rows, gimg.cols, CV_8UC1);
        oclMat gsum(totalheight + 4, gimg.cols + 1, CV_32SC1);
        oclMat gsqsum(totalheight + 4, gimg.cols + 1, CV_32FC1);

        cl_mem stagebuffer;
        cl_mem nodebuffer;
        cl_mem candidatebuffer;
        cl_mem scaleinfobuffer;
        cv::Rect roi, roi2;
        cv::Mat imgroi, imgroisq;
        cv::ocl::oclMat resizeroi, gimgroi, gimgroisq;
        int grp_per_CU = 12;

        size_t blocksize = 8;
        size_t localThreads[3] = { blocksize, blocksize , 1 };
        size_t globalThreads[3] = { grp_per_CU *(gsum.clCxt->getDeviceInfo().maxComputeUnits) *localThreads[0],
                                    localThreads[1], 1
                                  };
        int outputsz = 256 * globalThreads[0] / localThreads[0];
        int loopcount = sizev.size();
        detect_piramid_info *scaleinfo = (detect_piramid_info *)malloc(sizeof(detect_piramid_info) * loopcount);

        for( int i = 0; i < loopcount; i++ )
        {
            sz = sizev[i];
            factor = scalev[i];
            roi = Rect(0, indexy, sz.width, sz.height);
            roi2 = Rect(0, 0, sz.width - 1, sz.height - 1);
            resizeroi = gimg1(roi2);
            gimgroi = gsum(roi);
            gimgroisq = gsqsum(roi);
            int width = gimgroi.cols - 1 - cascade->orig_window_size.width;
            int height = gimgroi.rows - 1 - cascade->orig_window_size.height;
            scaleinfo[i].width_height = (width << 16) | height;


            int grpnumperline = (width + localThreads[0] - 1) / localThreads[0];
            int totalgrp = ((height + localThreads[1] - 1) / localThreads[1]) * grpnumperline;

            scaleinfo[i].grpnumperline_totalgrp = (grpnumperline << 16) | totalgrp;
            scaleinfo[i].imgoff = gimgroi.offset >> 2;
            scaleinfo[i].factor = factor;
            cv::ocl::resize(gimg, resizeroi, Size(sz.width - 1, sz.height - 1), 0, 0, INTER_LINEAR);
            cv::ocl::integral(resizeroi, gimgroi, gimgroisq);
            indexy += sz.height;
        }

        gcascade   = (GpuHidHaarClassifierCascade *)cascade->hid_cascade;
        stage      = (GpuHidHaarStageClassifier *)(gcascade + 1);
        classifier = (GpuHidHaarClassifier *)(stage + gcascade->count);
        node       = (GpuHidHaarTreeNode *)(classifier->node);

        int nodenum = (datasize - sizeof(GpuHidHaarClassifierCascade) -
                       sizeof(GpuHidHaarStageClassifier) * gcascade->count - sizeof(GpuHidHaarClassifier) * totalclassifier) / sizeof(GpuHidHaarTreeNode);

        candidate = (int *)malloc(4 * sizeof(int) * outputsz);

        gpuSetImagesForHaarClassifierCascade( cascade, 1., gsum.step / 4 );

        stagebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(GpuHidHaarStageClassifier) * gcascade->count);
        openCLSafeCall(clEnqueueWriteBuffer(qu, stagebuffer, 1, 0, sizeof(GpuHidHaarStageClassifier)*gcascade->count, stage, 0, NULL, NULL));

        nodebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, nodenum * sizeof(GpuHidHaarTreeNode));

        openCLSafeCall(clEnqueueWriteBuffer(qu, nodebuffer, 1, 0, nodenum * sizeof(GpuHidHaarTreeNode),
                                            node, 0, NULL, NULL));
        candidatebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_WRITE_ONLY, 4 * sizeof(int) * outputsz);

        scaleinfobuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(detect_piramid_info) * loopcount);
        openCLSafeCall(clEnqueueWriteBuffer(qu, scaleinfobuffer, 1, 0, sizeof(detect_piramid_info)*loopcount, scaleinfo, 0, NULL, NULL));

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

        vector<pair<size_t, const void *> > args;
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&stagebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&scaleinfobuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&nodebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsqsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&candidatebuffer ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&pixelstep ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&loopcount ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&splitstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&endstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startnode ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&splitnode ));
        args.push_back ( make_pair(sizeof(cl_int4) , (void *)&p ));
        args.push_back ( make_pair(sizeof(cl_int4) , (void *)&pq ));
        args.push_back ( make_pair(sizeof(cl_float) , (void *)&correction ));

        if(gcascade->is_stump_based && gsum.clCxt->supportsFeature(FEATURE_CL_INTEL_DEVICE))
        {
            //setup local group size for "pixel step" = 1
            localThreads[0] = 16;
            localThreads[1] = 32;
            localThreads[2] = 1;

            //calc maximal number of workgroups
            int WGNumX = 1+(sizev[0].width /(localThreads[0]));
            int WGNumY = 1+(sizev[0].height/(localThreads[1]));
            int WGNumZ = loopcount;
            int WGNumTotal = 0; //accurate number of non-empty workgroups
            int WGNumSampled = 0; //accurate number of workgroups processed only 1/4 part of all pixels. it is made for large images with scale <= 2
            oclMat      oclWGInfo(1,sizeof(cl_int4) * WGNumX*WGNumY*WGNumZ,CV_8U);
            {
                cl_int4*    pWGInfo = (cl_int4*)clEnqueueMapBuffer(getClCommandQueue(oclWGInfo.clCxt),(cl_mem)oclWGInfo.datastart,true,CL_MAP_WRITE, 0, oclWGInfo.step, 0,0,0,&status);
                openCLVerifyCall(status);
                for(int z=0;z<WGNumZ;++z)
                {
                    int     Width  = (scaleinfo[z].width_height >> 16)&0xFFFF;
                    int     Height = (scaleinfo[z].width_height >> 0 )& 0xFFFF;
                    for(int y=0;y<WGNumY;++y)
                    {
                        int     gy = y*localThreads[1];
                        if(gy>=Height)
                            continue; // no data to process
                        for(int x=0;x<WGNumX;++x)
                        {
                            int     gx = x*localThreads[0];
                            if(gx>=Width)
                                continue; // no data to process

                            if(scaleinfo[z].factor<=2)
                            {
                                WGNumSampled++;
                            }
                            // save no-empty workgroup info into array
                            pWGInfo[WGNumTotal].s[0] = scaleinfo[z].width_height;
                            pWGInfo[WGNumTotal].s[1] = (gx << 16) | gy;
                            pWGInfo[WGNumTotal].s[2] = scaleinfo[z].imgoff;
                            memcpy(&(pWGInfo[WGNumTotal].s[3]),&(scaleinfo[z].factor),sizeof(float));
                            WGNumTotal++;
                        }
                    }
                }
                openCLSafeCall(clEnqueueUnmapMemObject(getClCommandQueue(oclWGInfo.clCxt),(cl_mem)oclWGInfo.datastart,pWGInfo,0,0,0));
                pWGInfo = NULL;
            }

#define NODE_SIZE 12
            // pack node info to have less memory loads on the device side
            oclMat  oclNodesPK(1,sizeof(cl_int) * NODE_SIZE * nodenum,CV_8U);
            {
                cl_int  localStatus;
                cl_int* pNodesPK = (cl_int*)clEnqueueMapBuffer(getClCommandQueue(oclNodesPK.clCxt),(cl_mem)oclNodesPK.datastart,true,CL_MAP_WRITE, 0, oclNodesPK.step, 0,0,0,&localStatus);
                openCLVerifyCall(localStatus);
                //use known local data stride to precalulate indexes
                int DATA_SIZE_X = (localThreads[0]+cascade->orig_window_size.width);
                // check that maximal value is less than maximal unsigned short
                assert(DATA_SIZE_X*cascade->orig_window_size.height+cascade->orig_window_size.width < (int)USHRT_MAX);
                for(int i = 0;i<nodenum;++i)
                {//process each node from classifier
                    struct NodePK
                    {
                        unsigned short  slm_index[3][4];
                        float           weight[3];
                        float           threshold;
                        float           alpha[2];
                    };
                    struct NodePK * pOut = (struct NodePK *)(pNodesPK + NODE_SIZE*i);
                    for(int k=0;k<3;++k)
                    {// calc 4 short indexes in shared local mem for each rectangle instead of 2 (x,y) pair.
                        int* lp = &(node[i].p[k][0]);
                        pOut->slm_index[k][0] = (unsigned short)(lp[1]*DATA_SIZE_X+lp[0]);
                        pOut->slm_index[k][1] = (unsigned short)(lp[1]*DATA_SIZE_X+lp[2]);
                        pOut->slm_index[k][2] = (unsigned short)(lp[3]*DATA_SIZE_X+lp[0]);
                        pOut->slm_index[k][3] = (unsigned short)(lp[3]*DATA_SIZE_X+lp[2]);
                    }
                    //store used float point values for each node
                    pOut->weight[0] = node[i].weight[0];
                    pOut->weight[1] = node[i].weight[1];
                    pOut->weight[2] = node[i].weight[2];
                    pOut->threshold = node[i].threshold;
                    pOut->alpha[0] = node[i].alpha[0];
                    pOut->alpha[1] = node[i].alpha[1];
                }
                openCLSafeCall(clEnqueueUnmapMemObject(getClCommandQueue(oclNodesPK.clCxt),(cl_mem)oclNodesPK.datastart,pNodesPK,0,0,0));
                pNodesPK = NULL;
            }
            // add 2 additional buffers (WGinfo and packed nodes) as 2 last args
            args.push_back ( make_pair(sizeof(cl_mem) , (void *)&oclNodesPK.datastart ));
            args.push_back ( make_pair(sizeof(cl_mem) , (void *)&oclWGInfo.datastart ));

            //form build options for kernel
            string  options = "-D PACKED_CLASSIFIER";
            options += format(" -D NODE_SIZE=%d",NODE_SIZE);
            options += format(" -D WND_SIZE_X=%d",cascade->orig_window_size.width);
            options += format(" -D WND_SIZE_Y=%d",cascade->orig_window_size.height);
            options += format(" -D STUMP_BASED=%d",gcascade->is_stump_based);
            options += format(" -D SPLITNODE=%d",splitnode);
            options += format(" -D SPLITSTAGE=%d",splitstage);
            options += format(" -D OUTPUTSZ=%d",outputsz);

            // init candiate global count by 0
            int pattern = 0;
            openCLSafeCall(clEnqueueWriteBuffer(qu, candidatebuffer, 1, 0, 1 * sizeof(pattern),&pattern, 0, NULL, NULL));

            if(WGNumTotal>WGNumSampled)
            {// small images and each pixel is processed
                // setup global sizes to have linear array of workgroups with WGNum size
                int     pstep = 1;
                size_t  LS[3]={localThreads[0]/pstep,localThreads[1]/pstep,1};
                globalThreads[0] = LS[0]*(WGNumTotal-WGNumSampled);
                globalThreads[1] = LS[1];
                globalThreads[2] = 1;
                string options1 = options;
                options1 += format(" -D PIXEL_STEP=%d",pstep);
                options1 += format(" -D WGSTART=%d",WGNumSampled);
                options1 += format(" -D LSx=%d",LS[0]);
                options1 += format(" -D LSy=%d",LS[1]);
                // execute face detector
                openCLExecuteKernel(gsum.clCxt, &haarobjectdetect, "gpuRunHaarClassifierCascadePacked", globalThreads, LS, args, -1, -1, options1.c_str());
            }
            if(WGNumSampled>0)
            {// large images each 4th pixel is processed
                // setup global sizes to have linear array of workgroups with WGNum size
                int     pstep = 2;
                size_t  LS[3]={localThreads[0]/pstep,localThreads[1]/pstep,1};
                globalThreads[0] = LS[0]*WGNumSampled;
                globalThreads[1] = LS[1];
                globalThreads[2] = 1;
                string options2 = options;
                options2 += format(" -D PIXEL_STEP=%d",pstep);
                options2 += format(" -D WGSTART=%d",0);
                options2 += format(" -D LSx=%d",LS[0]);
                options2 += format(" -D LSy=%d",LS[1]);
                // execute face detector
                openCLExecuteKernel(gsum.clCxt, &haarobjectdetect, "gpuRunHaarClassifierCascadePacked", globalThreads, LS, args, -1, -1, options2.c_str());
            }
            //read candidate buffer back and put it into host list
            openCLReadBuffer( gsum.clCxt, candidatebuffer, candidate, 4 * sizeof(int)*outputsz );
            assert(candidate[0]<outputsz);
            //printf("candidate[0]=%d\n",candidate[0]);
            for(int i = 1; i <= candidate[0]; i++)
            {
                allCandidates.push_back(Rect(candidate[4 * i], candidate[4 * i + 1],candidate[4 * i + 2], candidate[4 * i + 3]));
            }
        }
        else
        {
            const char * build_options = gcascade->is_stump_based ? "-D STUMP_BASED=1" : "-D STUMP_BASED=0";

            openCLExecuteKernel(gsum.clCxt, &haarobjectdetect, "gpuRunHaarClassifierCascade", globalThreads, localThreads, args, -1, -1, build_options);

            openCLReadBuffer( gsum.clCxt, candidatebuffer, candidate, 4 * sizeof(int)*outputsz );

            for(int i = 0; i < outputsz; i++)
                if(candidate[4 * i + 2] != 0)
                    allCandidates.push_back(Rect(candidate[4 * i], candidate[4 * i + 1],
                    candidate[4 * i + 2], candidate[4 * i + 3]));
        }

        free(scaleinfo);
        free(candidate);
        openCLSafeCall(clReleaseMemObject(stagebuffer));
        openCLSafeCall(clReleaseMemObject(scaleinfobuffer));
        openCLSafeCall(clReleaseMemObject(nodebuffer));
        openCLSafeCall(clReleaseMemObject(candidatebuffer));

    }
    else
    {
        CvSize winsize0 = cascade->orig_window_size;
        int n_factors = 0;
        oclMat gsum;
        oclMat gsqsum;
        cv::ocl::integral(gimg, gsum, gsqsum);
        CvSize sz;
        vector<CvSize> sizev;
        vector<float> scalev;
        gpuSetHaarClassifierCascade(cascade);
        gcascade   = (GpuHidHaarClassifierCascade *)cascade->hid_cascade;
        stage      = (GpuHidHaarStageClassifier *)(gcascade + 1);
        classifier = (GpuHidHaarClassifier *)(stage + gcascade->count);
        node       = (GpuHidHaarTreeNode *)(classifier->node);
        cl_mem stagebuffer;
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
        size_t globalThreads[3] = { grp_per_CU *gsum.clCxt->getDeviceInfo().maxComputeUnits *localThreads[0],
                                    localThreads[1], 1 };
        int outputsz = 256 * globalThreads[0] / localThreads[0];
        int nodenum = (datasize - sizeof(GpuHidHaarClassifierCascade) -
                       sizeof(GpuHidHaarStageClassifier) * gcascade->count - sizeof(GpuHidHaarClassifier) * totalclassifier) / sizeof(GpuHidHaarTreeNode);
        nodebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY,
                                        nodenum * sizeof(GpuHidHaarTreeNode));
        openCLSafeCall(clEnqueueWriteBuffer(qu, nodebuffer, 1, 0,
                                            nodenum * sizeof(GpuHidHaarTreeNode),
                                            node, 0, NULL, NULL));
        cl_mem newnodebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_WRITE,
                               loopcount * nodenum * sizeof(GpuHidHaarTreeNode));
        int startstage = 0;
        int endstage = gcascade->count;
        for(int i = 0; i < loopcount; i++)
        {
            sz = sizev[i];
            factor = scalev[i];
            double ystep = std::max(2., factor);
            int equRect_x = cvRound(factor * gcascade->p0);
            int equRect_y = cvRound(factor * gcascade->p1);
            int equRect_w = cvRound(factor * gcascade->p3);
            int equRect_h = cvRound(factor * gcascade->p2);
            p[i].s[0] = equRect_x;
            p[i].s[1] = equRect_y;
            p[i].s[2] = equRect_x + equRect_w;
            p[i].s[3] = equRect_y + equRect_h;
            correction[i] = 1. / (equRect_w * equRect_h);
            int width = (gsum.cols - 1 - sz.width  + ystep - 1) / ystep;
            int height = (gsum.rows - 1 - sz.height + ystep - 1) / ystep;
            int grpnumperline = (width + localThreads[0] - 1) / localThreads[0];
            int totalgrp = ((height + localThreads[1] - 1) / localThreads[1]) * grpnumperline;

            scaleinfo[i].width_height = (width << 16) | height;
            scaleinfo[i].grpnumperline_totalgrp = (grpnumperline << 16) | totalgrp;
            scaleinfo[i].imgoff = 0;
            scaleinfo[i].factor = factor;
            int startnodenum = nodenum * i;
            float factor2 = (float)factor;

            vector<pair<size_t, const void *> > args1;
            args1.push_back ( make_pair(sizeof(cl_mem) , (void *)&nodebuffer ));
            args1.push_back ( make_pair(sizeof(cl_mem) , (void *)&newnodebuffer ));
            args1.push_back ( make_pair(sizeof(cl_float) , (void *)&factor2 ));
            args1.push_back ( make_pair(sizeof(cl_float) , (void *)&correction[i] ));
            args1.push_back ( make_pair(sizeof(cl_int) , (void *)&startnodenum ));

            size_t globalThreads2[3] = {(size_t)nodenum, 1, 1};
            openCLExecuteKernel(gsum.clCxt, &haarobjectdetect_scaled2, "gpuscaleclassifier", globalThreads2, NULL/*localThreads2*/, args1, -1, -1);
        }

        int step = gsum.step / 4;
        int startnode = 0;
        int splitstage = 3;
        stagebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(GpuHidHaarStageClassifier) * gcascade->count);
        openCLSafeCall(clEnqueueWriteBuffer(qu, stagebuffer, 1, 0, sizeof(GpuHidHaarStageClassifier)*gcascade->count, stage, 0, NULL, NULL));
        candidatebuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 4 * sizeof(int) * outputsz);
        scaleinfobuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(detect_piramid_info) * loopcount);
        openCLSafeCall(clEnqueueWriteBuffer(qu, scaleinfobuffer, 1, 0, sizeof(detect_piramid_info)*loopcount, scaleinfo, 0, NULL, NULL));
        pbuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(cl_int4) * loopcount);
        openCLSafeCall(clEnqueueWriteBuffer(qu, pbuffer, 1, 0, sizeof(cl_int4)*loopcount, p, 0, NULL, NULL));
        correctionbuffer = openCLCreateBuffer(gsum.clCxt, CL_MEM_READ_ONLY, sizeof(cl_float) * loopcount);
        openCLSafeCall(clEnqueueWriteBuffer(qu, correctionbuffer, 1, 0, sizeof(cl_float)*loopcount, correction, 0, NULL, NULL));

        vector<pair<size_t, const void *> > args;
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&stagebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&scaleinfobuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&newnodebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsqsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&candidatebuffer ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&gsum.rows ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&gsum.cols ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&step ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&loopcount ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&splitstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&endstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startnode ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&pbuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&correctionbuffer ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&nodenum ));
        const char * build_options = gcascade->is_stump_based ? "-D STUMP_BASED=1" : "-D STUMP_BASED=0";
        openCLExecuteKernel(gsum.clCxt, &haarobjectdetect_scaled2, "gpuRunHaarClassifierCascade_scaled2", globalThreads, localThreads, args, -1, -1, build_options);

        candidate = (int *)clEnqueueMapBuffer(qu, candidatebuffer, 1, CL_MAP_READ, 0, 4 * sizeof(int) * outputsz, 0, 0, 0, &status);

        for(int i = 0; i < outputsz; i++)
        {
            if(candidate[4 * i + 2] != 0)
                allCandidates.push_back(Rect(candidate[4 * i], candidate[4 * i + 1], candidate[4 * i + 2], candidate[4 * i + 3]));
        }

        free(scaleinfo);
        free(p);
        free(correction);
        clEnqueueUnmapMemObject(qu, candidatebuffer, candidate, 0, 0, 0);
        openCLSafeCall(clReleaseMemObject(stagebuffer));
        openCLSafeCall(clReleaseMemObject(scaleinfobuffer));
        openCLSafeCall(clReleaseMemObject(nodebuffer));
        openCLSafeCall(clReleaseMemObject(newnodebuffer));
        openCLSafeCall(clReleaseMemObject(candidatebuffer));
        openCLSafeCall(clReleaseMemObject(pbuffer));
        openCLSafeCall(clReleaseMemObject(correctionbuffer));
    }

    cvFree(&cascade->hid_cascade);
    rectList.resize(allCandidates.size());
    if(!allCandidates.empty())
        std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

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

    return result_seq;
}


struct getRect
{
    Rect operator()(const CvAvgComp &e) const
    {
        return e.rect;
    }
};

void cv::ocl::OclCascadeClassifier::detectMultiScale(oclMat &gimg, CV_OUT std::vector<cv::Rect>& faces,
                                                        double scaleFactor, int minNeighbors, int flags,
                                                        Size minSize, Size maxSize)
{
    CvSeq* _objects;
    MemStorage storage(cvCreateMemStorage(0));
    _objects = oclHaarDetectObjects(gimg, storage, scaleFactor, minNeighbors, flags, minSize, maxSize);
    vector<CvAvgComp> vecAvgComp;
    Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
    faces.resize(vecAvgComp.size());
    std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());
}

struct OclBuffers
{
    cl_mem stagebuffer;
    cl_mem nodebuffer;
    cl_mem candidatebuffer;
    cl_mem scaleinfobuffer;
    cl_mem pbuffer;
    cl_mem correctionbuffer;
    cl_mem newnodebuffer;
};


void cv::ocl::OclCascadeClassifierBuf::detectMultiScale(oclMat &gimg, CV_OUT std::vector<cv::Rect>& faces,
                                                        double scaleFactor, int minNeighbors, int flags,
                                                        Size minSize, Size maxSize)
{
    int blocksize = 8;
    int grp_per_CU = 12;
    size_t localThreads[3] = { (size_t)blocksize, (size_t)blocksize, 1 };
    size_t globalThreads[3] = { grp_per_CU * cv::ocl::Context::getContext()->getDeviceInfo().maxComputeUnits *localThreads[0],
        localThreads[1],
        1 };
    int outputsz = 256 * globalThreads[0] / localThreads[0];

    Init(gimg.rows, gimg.cols, scaleFactor, flags, outputsz, localThreads, minSize, maxSize);

    const double GROUP_EPS = 0.2;

    cv::ConcurrentRectVector allCandidates;
    std::vector<cv::Rect> rectList;
    std::vector<int> rweights;

    CvHaarClassifierCascade      *cascade = oldCascade;
    GpuHidHaarClassifierCascade  *gcascade;
    GpuHidHaarStageClassifier    *stage;

    if( CV_MAT_DEPTH(gimg.type()) != CV_8U )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );

    if( CV_MAT_CN(gimg.type()) > 1 )
    {
        oclMat gtemp;
        cvtColor( gimg, gtemp, CV_BGR2GRAY );
        gimg = gtemp;
    }

    int *candidate;
    cl_command_queue qu = getClCommandQueue(Context::getContext());
    if( (flags & CV_HAAR_SCALE_IMAGE) )
    {
        int indexy = 0;
        CvSize sz;

        cv::Rect roi, roi2;
        cv::ocl::oclMat resizeroi, gimgroi, gimgroisq;

        for( int i = 0; i < m_loopcount; i++ )
        {
            sz = sizev[i];
            roi = Rect(0, indexy, sz.width, sz.height);
            roi2 = Rect(0, 0, sz.width - 1, sz.height - 1);
            resizeroi = gimg1(roi2);
            gimgroi = gsum(roi);
            gimgroisq = gsqsum(roi);

            cv::ocl::resize(gimg, resizeroi, Size(sz.width - 1, sz.height - 1), 0, 0, INTER_LINEAR);
            cv::ocl::integral(resizeroi, gimgroi, gimgroisq);
            indexy += sz.height;
        }

        gcascade   = (GpuHidHaarClassifierCascade *)(cascade->hid_cascade);
        stage      = (GpuHidHaarStageClassifier *)(gcascade + 1);

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

        vector<pair<size_t, const void *> > args;
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->stagebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->scaleinfobuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->nodebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsqsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->candidatebuffer ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&pixelstep ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&m_loopcount ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&splitstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&endstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startnode ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&splitnode ));
        args.push_back ( make_pair(sizeof(cl_int4) , (void *)&p ));
        args.push_back ( make_pair(sizeof(cl_int4) , (void *)&pq ));
        args.push_back ( make_pair(sizeof(cl_float) , (void *)&correction ));

        const char * build_options = gcascade->is_stump_based ? "-D STUMP_BASED=1" : "-D STUMP_BASED=0";

        openCLExecuteKernel(gsum.clCxt, &haarobjectdetect, "gpuRunHaarClassifierCascade", globalThreads, localThreads, args, -1, -1, build_options);

        candidate = (int *)malloc(4 * sizeof(int) * outputsz);
        memset(candidate, 0, 4 * sizeof(int) * outputsz);

        openCLReadBuffer( gsum.clCxt, ((OclBuffers *)buffers)->candidatebuffer, candidate, 4 * sizeof(int)*outputsz );

        for(int i = 0; i < outputsz; i++)
        {
            if(candidate[4 * i + 2] != 0)
            {
                allCandidates.push_back(Rect(candidate[4 * i], candidate[4 * i + 1],
                candidate[4 * i + 2], candidate[4 * i + 3]));
            }
        }
        free((void *)candidate);
        candidate = NULL;
    }
    else
    {
        cv::ocl::integral(gimg, gsum, gsqsum);

        gcascade   = (GpuHidHaarClassifierCascade *)cascade->hid_cascade;

        int step = gsum.step / 4;
        int startnode = 0;
        int splitstage = 3;

        int startstage = 0;
        int endstage = gcascade->count;

        vector<pair<size_t, const void *> > args;
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->stagebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->scaleinfobuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->newnodebuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&gsqsum.data ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->candidatebuffer ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&gsum.rows ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&gsum.cols ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&step ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&m_loopcount ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&splitstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&endstage ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&startnode ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->pbuffer ));
        args.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->correctionbuffer ));
        args.push_back ( make_pair(sizeof(cl_int) , (void *)&m_nodenum ));

        const char * build_options = gcascade->is_stump_based ? "-D STUMP_BASED=1" : "-D STUMP_BASED=0";
        openCLExecuteKernel(gsum.clCxt, &haarobjectdetect_scaled2, "gpuRunHaarClassifierCascade_scaled2", globalThreads, localThreads, args, -1, -1, build_options);

        candidate = (int *)clEnqueueMapBuffer(qu, ((OclBuffers *)buffers)->candidatebuffer, 1, CL_MAP_READ, 0, 4 * sizeof(int) * outputsz, 0, 0, 0, NULL);

        for(int i = 0; i < outputsz; i++)
        {
            if(candidate[4 * i + 2] != 0)
                allCandidates.push_back(Rect(candidate[4 * i], candidate[4 * i + 1],
                candidate[4 * i + 2], candidate[4 * i + 3]));
        }
        clEnqueueUnmapMemObject(qu, ((OclBuffers *)buffers)->candidatebuffer, candidate, 0, 0, 0);
    }
    rectList.resize(allCandidates.size());
    if(!allCandidates.empty())
        std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

    if( minNeighbors != 0 || findBiggestObject )
        groupRectangles(rectList, rweights, std::max(minNeighbors, 1), GROUP_EPS);
    else
        rweights.resize(rectList.size(), 0);

    GenResult(faces, rectList, rweights);
}

void cv::ocl::OclCascadeClassifierBuf::Init(const int rows, const int cols,
    double scaleFactor, int flags,
    const int outputsz, const size_t localThreads[],
    CvSize minSize, CvSize maxSize)
{
    if(initialized)
    {
        return; // we only allow one time initialization
    }
    CvHaarClassifierCascade      *cascade = oldCascade;

    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier cascade" );

    if( scaleFactor <= 1 )
        CV_Error( CV_StsOutOfRange, "scale factor must be > 1" );

    if( cols < minSize.width || rows < minSize.height )
        CV_Error(CV_StsError, "Image too small");

    int datasize=0;
    int totalclassifier=0;

    if( !cascade->hid_cascade )
    {
        gpuCreateHidHaarClassifierCascade(cascade, &datasize, &totalclassifier);
    }

    if( maxSize.height == 0 || maxSize.width == 0 )
    {
        maxSize.height = rows;
        maxSize.width = cols;
    }

    findBiggestObject = (flags & CV_HAAR_FIND_BIGGEST_OBJECT) != 0;
    if( findBiggestObject )
        flags &= ~(CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING);

    CreateBaseBufs(datasize, totalclassifier, flags, outputsz);
    CreateFactorRelatedBufs(rows, cols, flags, scaleFactor, localThreads, minSize, maxSize);

    m_scaleFactor = scaleFactor;
    m_rows = rows;
    m_cols = cols;
    m_flags = flags;
    m_minSize = minSize;
    m_maxSize = maxSize;

    // initialize nodes
    GpuHidHaarClassifierCascade  *gcascade;
    GpuHidHaarStageClassifier    *stage;
    GpuHidHaarClassifier         *classifier;
    GpuHidHaarTreeNode           *node;
    cl_command_queue qu = getClCommandQueue(Context::getContext());
    if( (flags & CV_HAAR_SCALE_IMAGE) )
    {
        gcascade   = (GpuHidHaarClassifierCascade *)(cascade->hid_cascade);
        stage      = (GpuHidHaarStageClassifier *)(gcascade + 1);
        classifier = (GpuHidHaarClassifier *)(stage + gcascade->count);
        node       = (GpuHidHaarTreeNode *)(classifier->node);

        gpuSetImagesForHaarClassifierCascade( cascade, 1., gsum.step / 4 );

        openCLSafeCall(clEnqueueWriteBuffer(qu, ((OclBuffers *)buffers)->stagebuffer, 1, 0,
            sizeof(GpuHidHaarStageClassifier) * gcascade->count,
            stage, 0, NULL, NULL));

        openCLSafeCall(clEnqueueWriteBuffer(qu, ((OclBuffers *)buffers)->nodebuffer, 1, 0,
                                            m_nodenum * sizeof(GpuHidHaarTreeNode),
                                            node, 0, NULL, NULL));
    }
    else
    {
        gpuSetHaarClassifierCascade(cascade);

        gcascade   = (GpuHidHaarClassifierCascade *)cascade->hid_cascade;
        stage      = (GpuHidHaarStageClassifier *)(gcascade + 1);
        classifier = (GpuHidHaarClassifier *)(stage + gcascade->count);
        node       = (GpuHidHaarTreeNode *)(classifier->node);

        openCLSafeCall(clEnqueueWriteBuffer(qu, ((OclBuffers *)buffers)->nodebuffer, 1, 0,
            m_nodenum * sizeof(GpuHidHaarTreeNode),
            node, 0, NULL, NULL));

        cl_int4 *p = (cl_int4 *)malloc(sizeof(cl_int4) * m_loopcount);
        float *correction = (float *)malloc(sizeof(float) * m_loopcount);
        double factor;
        for(int i = 0; i < m_loopcount; i++)
        {
            factor = scalev[i];
            int equRect_x = (int)(factor * gcascade->p0 + 0.5);
            int equRect_y = (int)(factor * gcascade->p1 + 0.5);
            int equRect_w = (int)(factor * gcascade->p3 + 0.5);
            int equRect_h = (int)(factor * gcascade->p2 + 0.5);
            p[i].s[0] = equRect_x;
            p[i].s[1] = equRect_y;
            p[i].s[2] = equRect_x + equRect_w;
            p[i].s[3] = equRect_y + equRect_h;
            correction[i] = 1. / (equRect_w * equRect_h);
            int startnodenum = m_nodenum * i;
            float factor2 = (float)factor;

            vector<pair<size_t, const void *> > args1;
            args1.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->nodebuffer ));
            args1.push_back ( make_pair(sizeof(cl_mem) , (void *)&((OclBuffers *)buffers)->newnodebuffer ));
            args1.push_back ( make_pair(sizeof(cl_float) , (void *)&factor2 ));
            args1.push_back ( make_pair(sizeof(cl_float) , (void *)&correction[i] ));
            args1.push_back ( make_pair(sizeof(cl_int) , (void *)&startnodenum ));

            size_t globalThreads2[3] = {(size_t)m_nodenum, 1, 1};

            openCLExecuteKernel(Context::getContext(), &haarobjectdetect_scaled2, "gpuscaleclassifier", globalThreads2, NULL/*localThreads2*/, args1, -1, -1);
        }
        openCLSafeCall(clEnqueueWriteBuffer(qu, ((OclBuffers *)buffers)->stagebuffer, 1, 0, sizeof(GpuHidHaarStageClassifier)*gcascade->count, stage, 0, NULL, NULL));
        openCLSafeCall(clEnqueueWriteBuffer(qu, ((OclBuffers *)buffers)->pbuffer, 1, 0, sizeof(cl_int4)*m_loopcount, p, 0, NULL, NULL));
        openCLSafeCall(clEnqueueWriteBuffer(qu, ((OclBuffers *)buffers)->correctionbuffer, 1, 0, sizeof(cl_float)*m_loopcount, correction, 0, NULL, NULL));

        free(p);
        free(correction);
    }
    initialized = true;
}

void cv::ocl::OclCascadeClassifierBuf::CreateBaseBufs(const int datasize, const int totalclassifier,
                                                      const int flags, const int outputsz)
{
    if (!initialized)
    {
        buffers = malloc(sizeof(OclBuffers));

        size_t tempSize =
            sizeof(GpuHidHaarStageClassifier) * ((GpuHidHaarClassifierCascade *)oldCascade->hid_cascade)->count;
        m_nodenum = (datasize - sizeof(GpuHidHaarClassifierCascade) - tempSize - sizeof(GpuHidHaarClassifier) * totalclassifier)
            / sizeof(GpuHidHaarTreeNode);

        ((OclBuffers *)buffers)->stagebuffer     = openCLCreateBuffer(cv::ocl::Context::getContext(), CL_MEM_READ_ONLY,  tempSize);
        ((OclBuffers *)buffers)->nodebuffer      = openCLCreateBuffer(cv::ocl::Context::getContext(), CL_MEM_READ_ONLY,  m_nodenum * sizeof(GpuHidHaarTreeNode));
    }

    if (initialized
        && ((m_flags & CV_HAAR_SCALE_IMAGE) ^ (flags & CV_HAAR_SCALE_IMAGE)))
    {
        openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->candidatebuffer));
    }

    if (flags & CV_HAAR_SCALE_IMAGE)
    {
        ((OclBuffers *)buffers)->candidatebuffer = openCLCreateBuffer(cv::ocl::Context::getContext(),
                                                        CL_MEM_WRITE_ONLY,
                                                        4 * sizeof(int) * outputsz);
    }
    else
    {
        ((OclBuffers *)buffers)->candidatebuffer = openCLCreateBuffer(cv::ocl::Context::getContext(),
                                                        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                                        4 * sizeof(int) * outputsz);
    }
}

void cv::ocl::OclCascadeClassifierBuf::CreateFactorRelatedBufs(
    const int rows, const int cols, const int flags,
    const double scaleFactor, const size_t localThreads[],
    CvSize minSize, CvSize maxSize)
{
    if (initialized)
    {
        if ((m_flags & CV_HAAR_SCALE_IMAGE) && !(flags & CV_HAAR_SCALE_IMAGE))
        {
            gimg1.release();
            gsum.release();
            gsqsum.release();
        }
        else if (!(m_flags & CV_HAAR_SCALE_IMAGE) && (flags & CV_HAAR_SCALE_IMAGE))
        {
            openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->newnodebuffer));
            openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->correctionbuffer));
            openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->pbuffer));
        }
        else if ((m_flags & CV_HAAR_SCALE_IMAGE) && (flags & CV_HAAR_SCALE_IMAGE))
        {
            if (fabs(m_scaleFactor - scaleFactor) < 1e-6
                && (rows == m_rows && cols == m_cols)
                && (minSize.width == m_minSize.width)
                && (minSize.height == m_minSize.height)
                && (maxSize.width == m_maxSize.width)
                && (maxSize.height == m_maxSize.height))
            {
                return;
            }
        }
        else
        {
            if (fabs(m_scaleFactor - scaleFactor) < 1e-6
                && (rows == m_rows && cols == m_cols)
                && (minSize.width == m_minSize.width)
                && (minSize.height == m_minSize.height)
                && (maxSize.width == m_maxSize.width)
                && (maxSize.height == m_maxSize.height))
            {
                return;
            }
            else
            {
                openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->newnodebuffer));
                openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->correctionbuffer));
                openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->pbuffer));
            }
        }
    }

    int loopcount;
    int indexy = 0;
    int totalheight = 0;
    double factor;
    Rect roi;
    CvSize sz;
    CvSize winSize0 = oldCascade->orig_window_size;
    detect_piramid_info *scaleinfo;
    cl_command_queue qu = getClCommandQueue(Context::getContext());
    if (flags & CV_HAAR_SCALE_IMAGE)
    {
        for(factor = 1.f;; factor *= scaleFactor)
        {
            CvSize winSize = { cvRound(winSize0.width * factor), cvRound(winSize0.height * factor) };
            sz.width     = cvRound( cols / factor ) + 1;
            sz.height    = cvRound( rows / factor ) + 1;
            CvSize sz1     = { sz.width - winSize0.width - 1,      sz.height - winSize0.height - 1 };

            if( sz1.width <= 0 || sz1.height <= 0 )
                break;
            if( winSize.width > maxSize.width || winSize.height > maxSize.height )
                break;
            if( winSize.width < minSize.width || winSize.height < minSize.height )
                continue;

            totalheight += sz.height;
            sizev.push_back(sz);
            scalev.push_back(static_cast<float>(factor));
        }

        loopcount = sizev.size();
        gimg1.create(rows, cols, CV_8UC1);
        gsum.create(totalheight + 4, cols + 1, CV_32SC1);
        gsqsum.create(totalheight + 4, cols + 1, CV_32FC1);

        scaleinfo = (detect_piramid_info *)malloc(sizeof(detect_piramid_info) * loopcount);
        for( int i = 0; i < loopcount; i++ )
        {
            sz = sizev[i];
            roi = Rect(0, indexy, sz.width, sz.height);
            int width = sz.width - 1 - oldCascade->orig_window_size.width;
            int height = sz.height - 1 - oldCascade->orig_window_size.height;
            int grpnumperline = (width + localThreads[0] - 1) / localThreads[0];
            int totalgrp = ((height + localThreads[1] - 1) / localThreads[1]) * grpnumperline;

            ((detect_piramid_info *)scaleinfo)[i].width_height = (width << 16) | height;
            ((detect_piramid_info *)scaleinfo)[i].grpnumperline_totalgrp = (grpnumperline << 16) | totalgrp;
            ((detect_piramid_info *)scaleinfo)[i].imgoff = gsum(roi).offset >> 2;
            ((detect_piramid_info *)scaleinfo)[i].factor = scalev[i];

            indexy += sz.height;
        }
    }
    else
    {
        for(factor = 1;
            cvRound(factor * winSize0.width) < cols - 10 && cvRound(factor * winSize0.height) < rows - 10;
            factor *= scaleFactor)
        {
            CvSize winSize = { cvRound( winSize0.width * factor ), cvRound( winSize0.height * factor ) };
            if( winSize.width < minSize.width || winSize.height < minSize.height )
            {
                continue;
            }
            sizev.push_back(winSize);
            scalev.push_back(factor);
        }

        loopcount = scalev.size();
        if(loopcount == 0)
        {
            loopcount = 1;
            sizev.push_back(minSize);
            scalev.push_back( std::min(cvRound(minSize.width / winSize0.width), cvRound(minSize.height / winSize0.height)) );
        }

        ((OclBuffers *)buffers)->pbuffer = openCLCreateBuffer(cv::ocl::Context::getContext(), CL_MEM_READ_ONLY,
            sizeof(cl_int4) * loopcount);
        ((OclBuffers *)buffers)->correctionbuffer = openCLCreateBuffer(cv::ocl::Context::getContext(), CL_MEM_READ_ONLY,
            sizeof(cl_float) * loopcount);
        ((OclBuffers *)buffers)->newnodebuffer = openCLCreateBuffer(cv::ocl::Context::getContext(), CL_MEM_READ_WRITE,
            loopcount * m_nodenum * sizeof(GpuHidHaarTreeNode));

        scaleinfo = (detect_piramid_info *)malloc(sizeof(detect_piramid_info) * loopcount);
        for( int i = 0; i < loopcount; i++ )
        {
            sz = sizev[i];
            factor = scalev[i];
            double ystep = cv::max(2.,factor);
            int width = cvRound((cols - 1 - sz.width  + ystep - 1) / ystep);
            int height = cvRound((rows - 1 - sz.height + ystep - 1) / ystep);
            int grpnumperline = (width + localThreads[0] - 1) / localThreads[0];
            int totalgrp = ((height + localThreads[1] - 1) / localThreads[1]) * grpnumperline;

            ((detect_piramid_info *)scaleinfo)[i].width_height = (width << 16) | height;
            ((detect_piramid_info *)scaleinfo)[i].grpnumperline_totalgrp = (grpnumperline << 16) | totalgrp;
            ((detect_piramid_info *)scaleinfo)[i].imgoff = 0;
            ((detect_piramid_info *)scaleinfo)[i].factor = factor;
        }
    }

    if (loopcount != m_loopcount)
    {
        if (initialized)
        {
            openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->scaleinfobuffer));
        }
        ((OclBuffers *)buffers)->scaleinfobuffer = openCLCreateBuffer(cv::ocl::Context::getContext(), CL_MEM_READ_ONLY, sizeof(detect_piramid_info) * loopcount);
    }

    openCLSafeCall(clEnqueueWriteBuffer(qu, ((OclBuffers *)buffers)->scaleinfobuffer, 1, 0,
        sizeof(detect_piramid_info)*loopcount,
        scaleinfo, 0, NULL, NULL));
    free(scaleinfo);

    m_loopcount = loopcount;
}

void cv::ocl::OclCascadeClassifierBuf::GenResult(CV_OUT std::vector<cv::Rect>& faces,
                                                 const std::vector<cv::Rect> &rectList,
                                                 const std::vector<int> &rweights)
{
    MemStorage tempStorage(cvCreateMemStorage(0));
    CvSeq *result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), tempStorage );

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

    vector<CvAvgComp> vecAvgComp;
    Seq<CvAvgComp>(result_seq).copyTo(vecAvgComp);
    faces.resize(vecAvgComp.size());
    std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());
}

void cv::ocl::OclCascadeClassifierBuf::release()
{
    if(initialized)
    {
        openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->stagebuffer));
        openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->scaleinfobuffer));
        openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->nodebuffer));
        openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->candidatebuffer));

        if( (m_flags & CV_HAAR_SCALE_IMAGE) )
        {
            cvFree(&oldCascade->hid_cascade);
        }
        else
        {
            openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->newnodebuffer));
            openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->correctionbuffer));
            openCLSafeCall(clReleaseMemObject(((OclBuffers *)buffers)->pbuffer));
        }

        free(buffers);
        buffers = NULL;
        initialized = false;
    }
}

#ifndef _MAX_PATH
#define _MAX_PATH 1024
#endif
