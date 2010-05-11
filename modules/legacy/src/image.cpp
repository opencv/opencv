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

/* ////////////////////////////////////////////////////////////////////
//
//  C++ classes for image and matrices
//
// */

#include "precomp.hpp"
#include "opencv2/highgui/highgui_c.h"

/////////////////////////////// CvImage implementation //////////////////////////////////

static bool
icvIsXmlOrYaml( const char* filename )
{
    const char* suffix = strrchr( filename, '.' );
    return suffix &&
        (strcmp( suffix, ".xml" ) == 0 ||
        strcmp( suffix, ".Xml" ) == 0 ||
        strcmp( suffix, ".XML" ) == 0 ||
        strcmp( suffix, ".yml" ) == 0 ||
        strcmp( suffix, ".Yml" ) == 0 ||
        strcmp( suffix, ".YML" ) == 0 ||
        strcmp( suffix, ".yaml" ) == 0 ||
        strcmp( suffix, ".Yaml" ) == 0 ||
        strcmp( suffix, ".YAML" ) == 0);
}


static IplImage*
icvRetrieveImage( void* obj )
{
    IplImage* img = 0;

    if( CV_IS_IMAGE(obj) )
        img = (IplImage*)obj;
    else if( CV_IS_MAT(obj) )
    {
        CvMat* m = (CvMat*)obj;
        img = cvCreateImageHeader( cvSize(m->cols,m->rows),
                        CV_MAT_DEPTH(m->type), CV_MAT_CN(m->type) );
        cvSetData( img, m->data.ptr, m->step );
        img->imageDataOrigin = (char*)m->refcount;
        m->data.ptr = 0; m->step = 0;
        cvReleaseMat( &m );
    }
    else if( obj )
    {
        cvRelease( &obj );
        CV_Error( CV_StsUnsupportedFormat, "The object is neither an image, nor a matrix" );
    }

    return img;
}


bool CvImage::load( const char* filename, const char* imgname, int color )
{
    IplImage* img = 0;

    if( icvIsXmlOrYaml(filename) )
    {
        img = icvRetrieveImage(cvLoad(filename,0,imgname));
        if( (img->nChannels > 1) != (color == 0) )
            CV_Error( CV_StsNotImplemented,
            "RGB<->Grayscale conversion is not implemented for images stored in XML/YAML" );
        /*{
            IplImage* temp_img = 0;
            temp_img = cvCreateImage( cvGetSize(img), img->depth, color > 0 ? 3 : 1 ));
            cvCvtColor( img, temp_img, color > 0 ? CV_GRAY2BGR : CV_BGR2GRAY );
            cvReleaseImage( &img );
            img = temp_img;
        }*/
    }
    else
        img = cvLoadImage( filename, color );

    attach( img );
    return img != 0;
}


bool CvImage::read( CvFileStorage* fs, const char* mapname, const char* imgname )
{
    void* obj = 0;
    IplImage* img = 0;

    if( mapname )
    {
        CvFileNode* mapnode = cvGetFileNodeByName( fs, 0, mapname );
        if( !mapnode )
            obj = cvReadByName( fs, mapnode, imgname );
    }
    else
        obj = cvReadByName( fs, 0, imgname );

    img = icvRetrieveImage(obj);
    attach( img );
    return img != 0;
}


bool CvImage::read( CvFileStorage* fs, const char* seqname, int idx )
{
    void* obj = 0;
    IplImage* img = 0;
    CvFileNode* seqnode = seqname ?
        cvGetFileNodeByName( fs, 0, seqname ) : cvGetRootFileNode(fs,0);

    if( seqnode && CV_NODE_IS_SEQ(seqnode->tag) )
        obj = cvRead( fs, (CvFileNode*)cvGetSeqElem( seqnode->data.seq, idx ));
    img = icvRetrieveImage(obj);
    attach( img );
    return img != 0;
}


void CvImage::save( const char* filename, const char* imgname, const int* params )
{
    if( !image )
        return;
    if( icvIsXmlOrYaml( filename ) )
        cvSave( filename, image, imgname );
    else
        cvSaveImage( filename, image, params );
}


void CvImage::write( CvFileStorage* fs, const char* imgname )
{
    if( image )
        cvWrite( fs, imgname, image );
}


void CvImage::show( const char* window_name )
{
    if( image )
        cvShowImage( window_name, image );
}


/////////////////////////////// CvMatrix implementation //////////////////////////////////

CvMatrix::CvMatrix( int rows, int cols, int type, CvMemStorage* storage, bool alloc_data )
{
    if( storage )
    {
        matrix = (CvMat*)cvMemStorageAlloc( storage, sizeof(*matrix) );
        cvInitMatHeader( matrix, rows, cols, type, alloc_data ?
            cvMemStorageAlloc( storage, rows*cols*CV_ELEM_SIZE(type) ) : 0 );
    }
    else
        matrix = 0;
}

static CvMat*
icvRetrieveMatrix( void* obj )
{
    CvMat* m = 0;

    if( CV_IS_MAT(obj) )
        m = (CvMat*)obj;
    else if( CV_IS_IMAGE(obj) )
    {
        IplImage* img = (IplImage*)obj;
        CvMat hdr, *src = cvGetMat( img, &hdr );
        m = cvCreateMat( src->rows, src->cols, src->type );
        cvCopy( src, m );
        cvReleaseImage( &img );
    }
    else if( obj )
    {
        cvRelease( &obj );
        CV_Error( CV_StsUnsupportedFormat, "The object is neither an image, nor a matrix" );
    }

    return m;
}


bool CvMatrix::load( const char* filename, const char* matname, int color )
{
    CvMat* m = 0;
    if( icvIsXmlOrYaml(filename) )
    {
        m = icvRetrieveMatrix(cvLoad(filename,0,matname));

        if( (CV_MAT_CN(m->type) > 1) != (color == 0) )
            CV_Error( CV_StsNotImplemented,
            "RGB<->Grayscale conversion is not implemented for matrices stored in XML/YAML" );
        /*{
            CvMat* temp_mat;
            temp_mat = cvCreateMat( m->rows, m->cols,
                CV_MAKETYPE(CV_MAT_DEPTH(m->type), color > 0 ? 3 : 1 )));
            cvCvtColor( m, temp_mat, color > 0 ? CV_GRAY2BGR : CV_BGR2GRAY );
            cvReleaseMat( &m );
            m = temp_mat;
        }*/
    }
    else
        m = cvLoadImageM( filename, color );

    set( m, false );
    return m != 0;
}


bool CvMatrix::read( CvFileStorage* fs, const char* mapname, const char* matname )
{
    void* obj = 0;
    CvMat* m = 0;

    if( mapname )
    {
        CvFileNode* mapnode = cvGetFileNodeByName( fs, 0, mapname );
        if( !mapnode )
            obj = cvReadByName( fs, mapnode, matname );
    }
    else
        obj = cvReadByName( fs, 0, matname );

    m = icvRetrieveMatrix(obj);
    set( m, false );
    return m != 0;
}


bool CvMatrix::read( CvFileStorage* fs, const char* seqname, int idx )
{
    void* obj = 0;
    CvMat* m = 0;
    CvFileNode* seqnode = seqname ?
        cvGetFileNodeByName( fs, 0, seqname ) : cvGetRootFileNode(fs,0);

    if( seqnode && CV_NODE_IS_SEQ(seqnode->tag) )
        obj = cvRead( fs, (CvFileNode*)cvGetSeqElem( seqnode->data.seq, idx ));
    m = icvRetrieveMatrix(obj);
    set( m, false );
    return m != 0;
}


void CvMatrix::save( const char* filename, const char* matname, const int* params )
{
    if( !matrix )
        return;
    if( icvIsXmlOrYaml( filename ) )
        cvSave( filename, matrix, matname );
    else
        cvSaveImage( filename, matrix, params );
}


void CvMatrix::write( CvFileStorage* fs, const char* matname )
{
    if( matrix )
        cvWrite( fs, matname, matrix );
}


void CvMatrix::show( const char* window_name )
{
    if( matrix )
        cvShowImage( window_name, matrix );
}


/* End of file. */

