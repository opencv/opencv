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

/*
 * cvhaarclassifier.cpp
 *
 * haar classifiers (stump, CART, stage, cascade)
 */

#include "_cvhaartraining.h"


CvIntHaarClassifier* icvCreateCARTHaarClassifier( int count )
{
    CvCARTHaarClassifier* cart;
    size_t datasize;

    datasize = sizeof( *cart ) +
        ( sizeof( int ) +
          sizeof( CvTHaarFeature ) + sizeof( CvFastHaarFeature ) +
          sizeof( float ) + sizeof( int ) + sizeof( int ) ) * count +
        sizeof( float ) * (count + 1);

    cart = (CvCARTHaarClassifier*) cvAlloc( datasize );
    memset( cart, 0, datasize );

    cart->feature = (CvTHaarFeature*) (cart + 1);
    cart->fastfeature = (CvFastHaarFeature*) (cart->feature + count);
    cart->threshold = (float*) (cart->fastfeature + count);
    cart->left = (int*) (cart->threshold + count);
    cart->right = (int*) (cart->left + count);
    cart->val = (float*) (cart->right + count);
    cart->compidx = (int*) (cart->val + count + 1 );
    cart->count = count;
    cart->eval = icvEvalCARTHaarClassifier;
    cart->save = icvSaveCARTHaarClassifier;
    cart->release = icvReleaseHaarClassifier;

    return (CvIntHaarClassifier*) cart;
}


void icvReleaseHaarClassifier( CvIntHaarClassifier** classifier )
{
    cvFree( classifier );
    *classifier = NULL;
}


void icvInitCARTHaarClassifier( CvCARTHaarClassifier* carthaar, CvCARTClassifier* cart,
                                CvIntHaarFeatures* intHaarFeatures )
{
    int i;

    for( i = 0; i < cart->count; i++ )
    {
        carthaar->feature[i] = intHaarFeatures->feature[cart->compidx[i]];
        carthaar->fastfeature[i] = intHaarFeatures->fastfeature[cart->compidx[i]];
        carthaar->threshold[i] = cart->threshold[i];
        carthaar->left[i] = cart->left[i];
        carthaar->right[i] = cart->right[i];
        carthaar->val[i] = cart->val[i];
        carthaar->compidx[i] = cart->compidx[i];
    }
    carthaar->count = cart->count;
    carthaar->val[cart->count] = cart->val[cart->count];
}


float icvEvalCARTHaarClassifier( CvIntHaarClassifier* classifier,
                                 sum_type* sum, sum_type* tilted, float normfactor )
{
    int idx = 0;

    do
    {
        if( cvEvalFastHaarFeature(
                ((CvCARTHaarClassifier*) classifier)->fastfeature + idx, sum, tilted )
              < (((CvCARTHaarClassifier*) classifier)->threshold[idx] * normfactor) )
        {
            idx = ((CvCARTHaarClassifier*) classifier)->left[idx];
        }
        else
        {
            idx = ((CvCARTHaarClassifier*) classifier)->right[idx];
        }
    } while( idx > 0 );

    return ((CvCARTHaarClassifier*) classifier)->val[-idx];
}


CvIntHaarClassifier* icvCreateStageHaarClassifier( int count, float threshold )
{
    CvStageHaarClassifier* stage;
    size_t datasize;

    datasize = sizeof( *stage ) + sizeof( CvIntHaarClassifier* ) * count;
    stage = (CvStageHaarClassifier*) cvAlloc( datasize );
    memset( stage, 0, datasize );

    stage->count = count;
    stage->threshold = threshold;
    stage->classifier = (CvIntHaarClassifier**) (stage + 1);

    stage->eval = icvEvalStageHaarClassifier;
    stage->save = icvSaveStageHaarClassifier;
    stage->release = icvReleaseStageHaarClassifier;

    return (CvIntHaarClassifier*) stage;
}


void icvReleaseStageHaarClassifier( CvIntHaarClassifier** classifier )
{
    int i;

    for( i = 0; i < ((CvStageHaarClassifier*) *classifier)->count; i++ )
    {
        if( ((CvStageHaarClassifier*) *classifier)->classifier[i] != NULL )
        {
            ((CvStageHaarClassifier*) *classifier)->classifier[i]->release(
                &(((CvStageHaarClassifier*) *classifier)->classifier[i]) );
        }
    }

    cvFree( classifier );
    *classifier = NULL;
}


float icvEvalStageHaarClassifier( CvIntHaarClassifier* classifier,
                                  sum_type* sum, sum_type* tilted, float normfactor )
{
    int i;
    float stage_sum;

    stage_sum = 0.0F;
    for( i = 0; i < ((CvStageHaarClassifier*) classifier)->count; i++ )
    {
        stage_sum +=
            ((CvStageHaarClassifier*) classifier)->classifier[i]->eval(
                ((CvStageHaarClassifier*) classifier)->classifier[i],
                sum, tilted, normfactor );
    }

    return stage_sum;
}


CvIntHaarClassifier* icvCreateCascadeHaarClassifier( int count )
{
    CvCascadeHaarClassifier* ptr;
    size_t datasize;

    datasize = sizeof( *ptr ) + sizeof( CvIntHaarClassifier* ) * count;
    ptr = (CvCascadeHaarClassifier*) cvAlloc( datasize );
    memset( ptr, 0, datasize );

    ptr->count = count;
    ptr->classifier = (CvIntHaarClassifier**) (ptr + 1);

    ptr->eval = icvEvalCascadeHaarClassifier;
    ptr->save = NULL;
    ptr->release = icvReleaseCascadeHaarClassifier;

    return (CvIntHaarClassifier*) ptr;
}


void icvReleaseCascadeHaarClassifier( CvIntHaarClassifier** classifier )
{
    int i;

    for( i = 0; i < ((CvCascadeHaarClassifier*) *classifier)->count; i++ )
    {
        if( ((CvCascadeHaarClassifier*) *classifier)->classifier[i] != NULL )
        {
            ((CvCascadeHaarClassifier*) *classifier)->classifier[i]->release(
                &(((CvCascadeHaarClassifier*) *classifier)->classifier[i]) );
        }
    }

    cvFree( classifier );
    *classifier = NULL;
}


float icvEvalCascadeHaarClassifier( CvIntHaarClassifier* classifier,
                                    sum_type* sum, sum_type* tilted, float normfactor )
{
    int i;

    for( i = 0; i < ((CvCascadeHaarClassifier*) classifier)->count; i++ )
    {
        if( ((CvCascadeHaarClassifier*) classifier)->classifier[i]->eval(
                    ((CvCascadeHaarClassifier*) classifier)->classifier[i],
                    sum, tilted, normfactor )
            < ( ((CvStageHaarClassifier*)
                    ((CvCascadeHaarClassifier*) classifier)->classifier[i])->threshold
                            - CV_THRESHOLD_EPS) )
        {
            return 0.0;
        }
    }

    return 1.0;
}


void icvSaveHaarFeature( CvTHaarFeature* feature, FILE* file )
{
    fprintf( file, "%d\n", ( ( feature->rect[2].weight == 0.0F ) ? 2 : 3) );
    fprintf( file, "%d %d %d %d %d %d\n",
        feature->rect[0].r.x,
        feature->rect[0].r.y,
        feature->rect[0].r.width,
        feature->rect[0].r.height,
        0,
        (int) (feature->rect[0].weight) );
    fprintf( file, "%d %d %d %d %d %d\n",
        feature->rect[1].r.x,
        feature->rect[1].r.y,
        feature->rect[1].r.width,
        feature->rect[1].r.height,
        0,
        (int) (feature->rect[1].weight) );
    if( feature->rect[2].weight != 0.0F )
    {
        fprintf( file, "%d %d %d %d %d %d\n",
            feature->rect[2].r.x,
            feature->rect[2].r.y,
            feature->rect[2].r.width,
            feature->rect[2].r.height,
            0,
            (int) (feature->rect[2].weight) );
    }
    fprintf( file, "%s\n", &(feature->desc[0]) );
}


void icvLoadHaarFeature( CvTHaarFeature* feature, FILE* file )
{
    int nrect;
    int j;
    int tmp;
    int weight;

    nrect = 0;
    int values_read = fscanf( file, "%d", &nrect );
    CV_Assert(values_read == 1);

    assert( nrect <= CV_HAAR_FEATURE_MAX );

    for( j = 0; j < nrect; j++ )
    {
        values_read = fscanf( file, "%d %d %d %d %d %d",
            &(feature->rect[j].r.x),
            &(feature->rect[j].r.y),
            &(feature->rect[j].r.width),
            &(feature->rect[j].r.height),
            &tmp, &weight );
        CV_Assert(values_read == 6);
        feature->rect[j].weight = (float) weight;
    }
    for( j = nrect; j < CV_HAAR_FEATURE_MAX; j++ )
    {
        feature->rect[j].r.x = 0;
        feature->rect[j].r.y = 0;
        feature->rect[j].r.width = 0;
        feature->rect[j].r.height = 0;
        feature->rect[j].weight = 0.0f;
    }
    values_read = fscanf( file, "%s", &(feature->desc[0]) );
    CV_Assert(values_read == 1);
    feature->tilted = ( feature->desc[0] == 't' );
}


void icvSaveCARTHaarClassifier( CvIntHaarClassifier* classifier, FILE* file )
{
    int i;
    int count;

    count = ((CvCARTHaarClassifier*) classifier)->count;
    fprintf( file, "%d\n", count );
    for( i = 0; i < count; i++ )
    {
        icvSaveHaarFeature( &(((CvCARTHaarClassifier*) classifier)->feature[i]), file );
        fprintf( file, "%e %d %d\n",
            ((CvCARTHaarClassifier*) classifier)->threshold[i],
            ((CvCARTHaarClassifier*) classifier)->left[i],
            ((CvCARTHaarClassifier*) classifier)->right[i] );
    }
    for( i = 0; i <= count; i++ )
    {
        fprintf( file, "%e ", ((CvCARTHaarClassifier*) classifier)->val[i] );
    }
    fprintf( file, "\n" );
}


CvIntHaarClassifier* icvLoadCARTHaarClassifier( FILE* file, int step )
{
    CvCARTHaarClassifier* ptr;
    int i;
    int count;

    ptr = NULL;
    int values_read = fscanf( file, "%d", &count );
    CV_Assert(values_read == 1);

    if( count > 0 )
    {
        ptr = (CvCARTHaarClassifier*) icvCreateCARTHaarClassifier( count );
        for( i = 0; i < count; i++ )
        {
            icvLoadHaarFeature( &(ptr->feature[i]), file );
            values_read = fscanf( file, "%f %d %d", &(ptr->threshold[i]), &(ptr->left[i]),
                                      &(ptr->right[i]) );
            CV_Assert(values_read == 3);
        }
        for( i = 0; i <= count; i++ )
        {
            values_read = fscanf( file, "%f", &(ptr->val[i]) );
            CV_Assert(values_read == 1);
        }
        icvConvertToFastHaarFeature( ptr->feature, ptr->fastfeature, ptr->count, step );
    }

    return (CvIntHaarClassifier*) ptr;
}


void icvSaveStageHaarClassifier( CvIntHaarClassifier* classifier, FILE* file )
{
    int count;
    int i;
    float threshold;

    count = ((CvStageHaarClassifier*) classifier)->count;
    fprintf( file, "%d\n", count );
    for( i = 0; i < count; i++ )
    {
        ((CvStageHaarClassifier*) classifier)->classifier[i]->save(
            ((CvStageHaarClassifier*) classifier)->classifier[i], file );
    }

    threshold = ((CvStageHaarClassifier*) classifier)->threshold;

    /* to be compatible with the previous implementation */
    /* threshold = 2.0F * ((CvStageHaarClassifier*) classifier)->threshold - count; */

    fprintf( file, "%e\n", threshold );
}



static CvIntHaarClassifier* icvLoadCARTStageHaarClassifierF( FILE* file, int step )
{
    CvStageHaarClassifier* ptr = NULL;

    //CV_FUNCNAME( "icvLoadCARTStageHaarClassifierF" );

    __BEGIN__;

    if( file != NULL )
    {
        int count;
        int i;
        float threshold;

        count = 0;
        int values_read = fscanf( file, "%d", &count );
        CV_Assert(values_read == 1);
        if( count > 0 )
        {
            ptr = (CvStageHaarClassifier*) icvCreateStageHaarClassifier( count, 0.0F );
            for( i = 0; i < count; i++ )
            {
                ptr->classifier[i] = icvLoadCARTHaarClassifier( file, step );
            }

            values_read = fscanf( file, "%f", &threshold );
            CV_Assert(values_read == 1);

            ptr->threshold = threshold;
            /* to be compatible with the previous implementation */
            /* ptr->threshold = 0.5F * (threshold + count); */
        }
        if( feof( file ) )
        {
            ptr->release( (CvIntHaarClassifier**) &ptr );
            ptr = NULL;
        }
    }

    __END__;

    return (CvIntHaarClassifier*) ptr;
}


CvIntHaarClassifier* icvLoadCARTStageHaarClassifier( const char* filename, int step )
{
    CvIntHaarClassifier* ptr = NULL;

    CV_FUNCNAME( "icvLoadCARTStageHaarClassifier" );

    __BEGIN__;

    FILE* file;

    file = fopen( filename, "r" );
    if( file )
    {
        CV_CALL( ptr = icvLoadCARTStageHaarClassifierF( file, step ) );
        fclose( file );
    }

    __END__;

    return ptr;
}

/* tree cascade classifier */

/* evaluates a tree cascade classifier */

float icvEvalTreeCascadeClassifier( CvIntHaarClassifier* classifier,
                                    sum_type* sum, sum_type* tilted, float normfactor )
{
    CvTreeCascadeNode* ptr;

    ptr = ((CvTreeCascadeClassifier*) classifier)->root;

    while( ptr )
    {
        if( ptr->stage->eval( (CvIntHaarClassifier*) ptr->stage,
                              sum, tilted, normfactor )
                >= ptr->stage->threshold - CV_THRESHOLD_EPS )
        {
            ptr = ptr->child;
        }
        else
        {
            while( ptr && ptr->next == NULL ) ptr = ptr->parent;
            if( ptr == NULL ) return 0.0F;
            ptr = ptr->next;
        }
    }

    return 1.0F;
}

/* sets path int the tree form the root to the leaf node */

void icvSetLeafNode( CvTreeCascadeClassifier* tcc, CvTreeCascadeNode* leaf )
{
    CV_FUNCNAME( "icvSetLeafNode" );

    __BEGIN__;

    CvTreeCascadeNode* ptr;

    ptr = NULL;
    while( leaf )
    {
        leaf->child_eval = ptr;
        ptr = leaf;
        leaf = leaf->parent;
    }

    leaf = tcc->root;
    while( leaf && leaf != ptr ) leaf = leaf->next;
    if( leaf != ptr )
        CV_ERROR( CV_StsError, "Invalid tcc or leaf node." );

    tcc->root_eval = ptr;

    __END__;
}

/* evaluates a tree cascade classifier. used in filtering */

float icvEvalTreeCascadeClassifierFilter( CvIntHaarClassifier* classifier, sum_type* sum,
                                          sum_type* tilted, float normfactor )
{
    CvTreeCascadeNode* ptr;
    //CvTreeCascadeClassifier* tree;

    //tree = (CvTreeCascadeClassifier*) classifier;



    ptr = ((CvTreeCascadeClassifier*) classifier)->root_eval;
    while( ptr )
    {
        if( ptr->stage->eval( (CvIntHaarClassifier*) ptr->stage,
                              sum, tilted, normfactor )
                < ptr->stage->threshold - CV_THRESHOLD_EPS )
        {
            return 0.0F;
        }
        ptr = ptr->child_eval;
    }

    return 1.0F;
}

/* creates tree cascade node */

CvTreeCascadeNode* icvCreateTreeCascadeNode()
{
    CvTreeCascadeNode* ptr = NULL;

    CV_FUNCNAME( "icvCreateTreeCascadeNode" );

    __BEGIN__;
    size_t data_size;

    data_size = sizeof( *ptr );
    CV_CALL( ptr = (CvTreeCascadeNode*) cvAlloc( data_size ) );
    memset( ptr, 0, data_size );

    __END__;

    return ptr;
}

/* releases all tree cascade nodes accessible via links */

void icvReleaseTreeCascadeNodes( CvTreeCascadeNode** node )
{
    //CV_FUNCNAME( "icvReleaseTreeCascadeNodes" );

    __BEGIN__;

    if( node && *node )
    {
        CvTreeCascadeNode* ptr;
        CvTreeCascadeNode* ptr_;

        ptr = *node;

        while( ptr )
        {
            while( ptr->child ) ptr = ptr->child;

            if( ptr->stage ) ptr->stage->release( (CvIntHaarClassifier**) &ptr->stage );
            ptr_ = ptr;

            while( ptr && ptr->next == NULL ) ptr = ptr->parent;
            if( ptr ) ptr = ptr->next;

            cvFree( &ptr_ );
        }
    }

    __END__;
}


/* releases tree cascade classifier */

void icvReleaseTreeCascadeClassifier( CvIntHaarClassifier** classifier )
{
    if( classifier && *classifier )
    {
        icvReleaseTreeCascadeNodes( &((CvTreeCascadeClassifier*) *classifier)->root );
        cvFree( classifier );
        *classifier = NULL;
    }
}


void icvPrintTreeCascade( CvTreeCascadeNode* root )
{
    //CV_FUNCNAME( "icvPrintTreeCascade" );

    __BEGIN__;

    CvTreeCascadeNode* node;
    CvTreeCascadeNode* n;
    char buf0[256];
    char buf[256];
    int level;
    int i;
    int max_level;

    node = root;
    level = max_level = 0;
    while( node )
    {
        while( node->child ) { node = node->child; level++; }
        if( level > max_level ) { max_level = level; }
        while( node && !node->next ) { node = node->parent; level--; }
        if( node ) node = node->next;
    }

    printf( "\nTree Classifier\n" );
    printf( "Stage\n" );
    for( i = 0; i <= max_level; i++ ) printf( "+---" );
    printf( "+\n" );
    for( i = 0; i <= max_level; i++ ) printf( "|%3d", i );
    printf( "|\n" );
    for( i = 0; i <= max_level; i++ ) printf( "+---" );
    printf( "+\n\n" );

    node = root;

    buf[0] = 0;
    while( node )
    {
        sprintf( buf + strlen( buf ), "%3d", node->idx );
        while( node->child )
        {
            node = node->child;
            sprintf( buf + strlen( buf ),
                ((node->idx < 10) ? "---%d" : ((node->idx < 100) ? "--%d" : "-%d")),
                node->idx );
        }
        printf( " %s\n", buf );

        while( node && !node->next ) { node = node->parent; }
        if( node )
        {
            node = node->next;

            n = node->parent;
            buf[0] = 0;
            while( n )
            {
                if( n->next )
                    sprintf( buf0, "  | %s", buf );
                else
                    sprintf( buf0, "    %s", buf );
                strcpy( buf, buf0 );
                n = n->parent;
            }
            printf( " %s  |\n", buf );
        }
    }
    printf( "\n" );
    fflush( stdout );

    __END__;
}



CvIntHaarClassifier* icvLoadTreeCascadeClassifier( const char* filename, int step,
                                                   int* splits )
{
    CvTreeCascadeClassifier* ptr = NULL;
    CvTreeCascadeNode** nodes = NULL;

    CV_FUNCNAME( "icvLoadTreeCascadeClassifier" );

    __BEGIN__;

    size_t data_size;
    CvStageHaarClassifier* stage;
    char stage_name[PATH_MAX];
    char* suffix;
    int i, num;
    FILE* f;
    int result, parent=0, next=0;
    int stub;

    if( !splits ) splits = &stub;

    *splits = 0;

    data_size = sizeof( *ptr );

    CV_CALL( ptr = (CvTreeCascadeClassifier*) cvAlloc( data_size ) );
    memset( ptr, 0, data_size );

    ptr->eval = icvEvalTreeCascadeClassifier;
    ptr->release = icvReleaseTreeCascadeClassifier;

    sprintf( stage_name, "%s/", filename );
    suffix = stage_name + strlen( stage_name );

    for( i = 0; ; i++ )
    {
        sprintf( suffix, "%d/%s", i, CV_STAGE_CART_FILE_NAME );
        f = fopen( stage_name, "r" );
        if( !f ) break;
        fclose( f );
    }
    num = i;

    if( num < 1 ) EXIT;

    data_size = sizeof( *nodes ) * num;
    CV_CALL( nodes = (CvTreeCascadeNode**) cvAlloc( data_size ) );

    for( i = 0; i < num; i++ )
    {
        sprintf( suffix, "%d/%s", i, CV_STAGE_CART_FILE_NAME );
        f = fopen( stage_name, "r" );
        CV_CALL( stage = (CvStageHaarClassifier*)
            icvLoadCARTStageHaarClassifierF( f, step ) );

        result = ( f && stage ) ? fscanf( f, "%d%d", &parent, &next ) : 0;
        if( f ) fclose( f );

        if( result != 2 )
        {
            num = i;
            break;
        }

        printf( "Stage %d loaded\n", i );

        if( parent >= i || (next != -1 && next != i + 1) )
            CV_ERROR( CV_StsError, "Invalid tree links" );

        CV_CALL( nodes[i] = icvCreateTreeCascadeNode() );
        nodes[i]->stage = stage;
        nodes[i]->idx = i;
        nodes[i]->parent = (parent != -1 ) ? nodes[parent] : NULL;
        nodes[i]->next = ( next != -1 ) ? nodes[i] : NULL;
        nodes[i]->child = NULL;
    }
    for( i = 0; i < num; i++ )
    {
        if( nodes[i]->next )
        {
            (*splits)++;
            nodes[i]->next = nodes[i+1];
        }
        if( nodes[i]->parent && nodes[i]->parent->child == NULL )
        {
            nodes[i]->parent->child = nodes[i];
        }
    }
    ptr->root = nodes[0];
    ptr->next_idx = num;

    __END__;

    cvFree( &nodes );

    return (CvIntHaarClassifier*) ptr;
}


CvTreeCascadeNode* icvFindDeepestLeaves( CvTreeCascadeClassifier* tcc )
{
    CvTreeCascadeNode* leaves;

    //CV_FUNCNAME( "icvFindDeepestLeaves" );

    __BEGIN__;

    int level, cur_level;
    CvTreeCascadeNode* ptr;
    CvTreeCascadeNode* last;

    leaves = last = NULL;

    ptr = tcc->root;
    level = -1;
    cur_level = 0;

    /* find leaves with maximal level */
    while( ptr )
    {
        if( ptr->child ) { ptr = ptr->child; cur_level++; }
        else
        {
            if( cur_level == level )
            {
                last->next_same_level = ptr;
                ptr->next_same_level = NULL;
                last = ptr;
            }
            if( cur_level > level )
            {
                level = cur_level;
                leaves = last = ptr;
                ptr->next_same_level = NULL;
            }
            while( ptr && ptr->next == NULL ) { ptr = ptr->parent; cur_level--; }
            if( ptr ) ptr = ptr->next;
        }
    }

    __END__;

    return leaves;
}

/* End of file. */
