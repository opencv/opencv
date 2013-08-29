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

#include "precomp.hpp"

CvBGCodeBookModel* cvCreateBGCodeBookModel()
{
    CvBGCodeBookModel* model = (CvBGCodeBookModel*)cvAlloc( sizeof(*model) );
    memset( model, 0, sizeof(*model) );
    model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 10;
    model->modMin[0] = 3;
    model->modMax[0] = 10;
    model->modMin[1] = model->modMin[2] = 1;
    model->modMax[1] = model->modMax[2] = 1;
    model->storage = cvCreateMemStorage();

    return model;
}

void cvReleaseBGCodeBookModel( CvBGCodeBookModel** model )
{
    if( model && *model )
    {
        cvReleaseMemStorage( &(*model)->storage );
        memset( *model, 0, sizeof(**model) );
        cvFree( model );
    }
}

static uchar satTab8u[768];
#undef SAT_8U
#define SAT_8U(x) satTab8u[(x) + 255]

static void icvInitSatTab()
{
    static int initialized = 0;
    if( !initialized )
    {
        for( int i = 0; i < 768; i++ )
        {
            int v = i - 255;
            satTab8u[i] = (uchar)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
        initialized = 1;
    }
}


void cvBGCodeBookUpdate( CvBGCodeBookModel* model, const CvArr* _image,
                         CvRect roi, const CvArr* _mask )
{
    CV_FUNCNAME( "cvBGCodeBookUpdate" );

    __BEGIN__;

    CvMat stub, *image = cvGetMat( _image, &stub );
    CvMat mstub, *mask = _mask ? cvGetMat( _mask, &mstub ) : 0;
    int i, x, y, T;
    int nblocks;
    uchar cb0, cb1, cb2;
    CvBGCodeBookElem* freeList;

    CV_ASSERT( model && CV_MAT_TYPE(image->type) == CV_8UC3 &&
        (!mask || (CV_IS_MASK_ARR(mask) && CV_ARE_SIZES_EQ(image, mask))) );

    if( roi.x == 0 && roi.y == 0 && roi.width == 0 && roi.height == 0 )
    {
        roi.width = image->cols;
        roi.height = image->rows;
    }
    else
        CV_ASSERT( (unsigned)roi.x < (unsigned)image->cols &&
                   (unsigned)roi.y < (unsigned)image->rows &&
                   roi.width >= 0 && roi.height >= 0 &&
                   roi.x + roi.width <= image->cols &&
                   roi.y + roi.height <= image->rows );

    if( image->cols != model->size.width || image->rows != model->size.height )
    {
        cvClearMemStorage( model->storage );
        model->freeList = 0;
        cvFree( &model->cbmap );
        int bufSz = image->cols*image->rows*sizeof(model->cbmap[0]);
        model->cbmap = (CvBGCodeBookElem**)cvAlloc(bufSz);
        memset( model->cbmap, 0, bufSz );
        model->size = cvSize(image->cols, image->rows);
    }

    icvInitSatTab();

    cb0 = model->cbBounds[0];
    cb1 = model->cbBounds[1];
    cb2 = model->cbBounds[2];

    T = ++model->t;
    freeList = model->freeList;
    nblocks = (int)((model->storage->block_size - sizeof(CvMemBlock))/sizeof(*freeList));
    nblocks = MIN( nblocks, 1024 );
    CV_ASSERT( nblocks > 0 );

    for( y = 0; y < roi.height; y++ )
    {
        const uchar* p = image->data.ptr + image->step*(y + roi.y) + roi.x*3;
        const uchar* m = mask ? mask->data.ptr + mask->step*(y + roi.y) + roi.x : 0;
        CvBGCodeBookElem** cb = model->cbmap + image->cols*(y + roi.y) + roi.x;

        for( x = 0; x < roi.width; x++, p += 3, cb++ )
        {
            CvBGCodeBookElem *e, *found = 0;
            uchar p0, p1, p2, l0, l1, l2, h0, h1, h2;
            int negRun;

            if( m && m[x] == 0 )
                continue;

            p0 = p[0]; p1 = p[1]; p2 = p[2];
            l0 = SAT_8U(p0 - cb0); l1 = SAT_8U(p1 - cb1); l2 = SAT_8U(p2 - cb2);
            h0 = SAT_8U(p0 + cb0); h1 = SAT_8U(p1 + cb1); h2 = SAT_8U(p2 + cb2);

            for( e = *cb; e != 0; e = e->next )
            {
                if( e->learnMin[0] <= p0 && p0 <= e->learnMax[0] &&
                    e->learnMin[1] <= p1 && p1 <= e->learnMax[1] &&
                    e->learnMin[2] <= p2 && p2 <= e->learnMax[2] )
                {
                    e->tLastUpdate = T;
                    e->boxMin[0] = MIN(e->boxMin[0], p0);
                    e->boxMax[0] = MAX(e->boxMax[0], p0);
                    e->boxMin[1] = MIN(e->boxMin[1], p1);
                    e->boxMax[1] = MAX(e->boxMax[1], p1);
                    e->boxMin[2] = MIN(e->boxMin[2], p2);
                    e->boxMax[2] = MAX(e->boxMax[2], p2);

                    // no need to use SAT_8U for updated learnMin[i] & learnMax[i] here,
                    // as the bounding li & hi are already within 0..255.
                    if( e->learnMin[0] > l0 ) e->learnMin[0]--;
                    if( e->learnMax[0] < h0 ) e->learnMax[0]++;
                    if( e->learnMin[1] > l1 ) e->learnMin[1]--;
                    if( e->learnMax[1] < h1 ) e->learnMax[1]++;
                    if( e->learnMin[2] > l2 ) e->learnMin[2]--;
                    if( e->learnMax[2] < h2 ) e->learnMax[2]++;

                    found = e;
                    break;
                }
                negRun = T - e->tLastUpdate;
                e->stale = MAX( e->stale, negRun );
            }

            for( ; e != 0; e = e->next )
            {
                negRun = T - e->tLastUpdate;
                e->stale = MAX( e->stale, negRun );
            }

            if( !found )
            {
                if( !freeList )
                {
                    freeList = (CvBGCodeBookElem*)cvMemStorageAlloc(model->storage,
                        nblocks*sizeof(*freeList));
                    for( i = 0; i < nblocks-1; i++ )
                        freeList[i].next = &freeList[i+1];
                    freeList[nblocks-1].next = 0;
                }
                e = freeList;
                freeList = freeList->next;

                e->learnMin[0] = l0; e->learnMax[0] = h0;
                e->learnMin[1] = l1; e->learnMax[1] = h1;
                e->learnMin[2] = l2; e->learnMax[2] = h2;
                e->boxMin[0] = e->boxMax[0] = p0;
                e->boxMin[1] = e->boxMax[1] = p1;
                e->boxMin[2] = e->boxMax[2] = p2;
                e->tLastUpdate = T;
                e->stale = 0;
                e->next = *cb;
                *cb = e;
            }
        }
    }

    model->freeList = freeList;

    __END__;
}


int cvBGCodeBookDiff( const CvBGCodeBookModel* model, const CvArr* _image,
                      CvArr* _fgmask, CvRect roi )
{
    int maskCount = -1;

    CV_FUNCNAME( "cvBGCodeBookDiff" );

    __BEGIN__;

    CvMat stub, *image = cvGetMat( _image, &stub );
    CvMat mstub, *mask = cvGetMat( _fgmask, &mstub );
    int x, y;
    uchar m0, m1, m2, M0, M1, M2;

    CV_ASSERT( model && CV_MAT_TYPE(image->type) == CV_8UC3 &&
        image->cols == model->size.width && image->rows == model->size.height &&
        CV_IS_MASK_ARR(mask) && CV_ARE_SIZES_EQ(image, mask) );

    if( roi.x == 0 && roi.y == 0 && roi.width == 0 && roi.height == 0 )
    {
        roi.width = image->cols;
        roi.height = image->rows;
    }
    else
        CV_ASSERT( (unsigned)roi.x < (unsigned)image->cols &&
                   (unsigned)roi.y < (unsigned)image->rows &&
                   roi.width >= 0 && roi.height >= 0 &&
                   roi.x + roi.width <= image->cols &&
                   roi.y + roi.height <= image->rows );

    m0 = model->modMin[0]; M0 = model->modMax[0];
    m1 = model->modMin[1]; M1 = model->modMax[1];
    m2 = model->modMin[2]; M2 = model->modMax[2];

    maskCount = roi.height*roi.width;
    for( y = 0; y < roi.height; y++ )
    {
        const uchar* p = image->data.ptr + image->step*(y + roi.y) + roi.x*3;
        uchar* m = mask->data.ptr + mask->step*(y + roi.y) + roi.x;
        CvBGCodeBookElem** cb = model->cbmap + image->cols*(y + roi.y) + roi.x;

        for( x = 0; x < roi.width; x++, p += 3, cb++ )
        {
            CvBGCodeBookElem *e;
            uchar p0 = p[0], p1 = p[1], p2 = p[2];
            int l0 = p0 + m0, l1 = p1 + m1, l2 = p2 + m2;
            int h0 = p0 - M0, h1 = p1 - M1, h2 = p2 - M2;
            m[x] = (uchar)255;

            for( e = *cb; e != 0; e = e->next )
            {
                if( e->boxMin[0] <= l0 && h0 <= e->boxMax[0] &&
                    e->boxMin[1] <= l1 && h1 <= e->boxMax[1] &&
                    e->boxMin[2] <= l2 && h2 <= e->boxMax[2] )
                {
                    m[x] = 0;
                    maskCount--;
                    break;
                }
            }
        }
    }

    __END__;

    return maskCount;
}

void cvBGCodeBookClearStale( CvBGCodeBookModel* model, int staleThresh,
                             CvRect roi, const CvArr* _mask )
{
    CV_FUNCNAME( "cvBGCodeBookClearStale" );

    __BEGIN__;

    CvMat mstub, *mask = _mask ? cvGetMat( _mask, &mstub ) : 0;
    int x, y, T;
    CvBGCodeBookElem* freeList;

    CV_ASSERT( model && (!mask || (CV_IS_MASK_ARR(mask) &&
        mask->cols == model->size.width && mask->rows == model->size.height)) );

    if( roi.x == 0 && roi.y == 0 && roi.width == 0 && roi.height == 0 )
    {
        roi.width = model->size.width;
        roi.height = model->size.height;
    }
    else
        CV_ASSERT( (unsigned)roi.x < (unsigned)mask->cols &&
                   (unsigned)roi.y < (unsigned)mask->rows &&
                   roi.width >= 0 && roi.height >= 0 &&
                   roi.x + roi.width <= mask->cols &&
                   roi.y + roi.height <= mask->rows );

    icvInitSatTab();
    freeList = model->freeList;
    T = model->t;

    for( y = 0; y < roi.height; y++ )
    {
        const uchar* m = mask ? mask->data.ptr + mask->step*(y + roi.y) + roi.x : 0;
        CvBGCodeBookElem** cb = model->cbmap + model->size.width*(y + roi.y) + roi.x;

        for( x = 0; x < roi.width; x++, cb++ )
        {
            CvBGCodeBookElem *e, first, *prev = &first;

            if( m && m[x] == 0 )
                continue;

            for( first.next = e = *cb; e != 0; e = prev->next )
            {
                if( e->stale > staleThresh )
                {
                    prev->next = e->next;
                    e->next = freeList;
                    freeList = e;
                }
                else
                {
                    e->stale = 0;
                    e->tLastUpdate = T;
                    prev = e;
                }
            }

            *cb = first.next;
        }
    }

    model->freeList = freeList;

    __END__;
}

/* End of file. */
