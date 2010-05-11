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

#include "cvtest.h"

#if 0

#include "aeigenobjects.inc"

#define __8U   8
#define __32F 32
#define MAXDIFF 1.01
#define RELDIFF 1.0e-4

typedef struct _UserData  /* User data structure for callback mode */
{
    void*  addr1;  /* Array of objects ROI start addresses */
    void*  addr2;
    int    step1;  /* Step in bytes */
    int    step2;
    CvSize size1;  /* ROI or full size */
    CvSize size2;
} UserData;

/* Testing parameters */
static char FuncName[]  =
"cvCalcCovarMatrixEx, cvCalcEigenObjects, cvCalcDecompCoeff, cvEigenDecomposite, cvEigenProjection";
static char TestName[]  = "Eigen objects functions group test";
static char TestClass[] = "Algorithm";
static int obj_number, obj_width, obj_height;
static double rel_bufSize;

/*-----------------------------=--=-=== Callback functions ===-=--=---------------------*/

int read_callback_8u( int ind, void* buf, void* userData)
{
    int i, j, k = 0;
    UserData* data = (UserData*)userData;
    uchar* start = ((uchar**)(data->addr1))[ind];
    uchar* buff = (uchar*)buf;

    if( ind<0 ) return CV_BADFACTOR_ERR;
    if( buf==NULL || userData==NULL ) return CV_NULLPTR_ERR;

    for( i=0; i<data->size1.height; i++, start+=data->step1 )
        for( j=0; j<data->size1.width; j++, k++ )
            buff[k] = start[j];
    return CV_NO_ERR;
}
/*----------------------*/
int read_callback_32f( int ind, void* buf, void* userData)
{
    int i, j, k = 0;
    UserData* data = (UserData*)userData;
    float* start = ((float**)(data->addr2))[ind];
    float* buff = (float*)buf;

    if( ind<0 ) return CV_BADFACTOR_ERR;
    if( buf==NULL || userData==NULL ) return CV_NULLPTR_ERR;

    for( i=0; i<data->size2.height; i++, start+=data->step2/4 )
        for( j=0; j<data->size2.width; j++, k++ )
            buff[k] = start[j];
    return CV_NO_ERR;
}
/*========================*/
int write_callback_8u( int ind, void* buf, void* userData)
{
    int i, j, k = 0;
    UserData* data = (UserData*)userData;
    uchar* start = ((uchar**)(data->addr1))[ind];
    uchar* buff = (uchar*)buf;

    if( ind<0 ) return CV_BADFACTOR_ERR;
    if( buf==NULL || userData==NULL ) return CV_NULLPTR_ERR;

    for( i=0; i<data->size1.height; i++, start+=data->step1 )
        for( j=0; j<data->size1.width; j++, k++ )
            start[j] = buff[k];
    return CV_NO_ERR;
}
/*----------------------*/
int write_callback_32f( int ind, void* buf, void* userData)
{
    int i, j, k = 0;
    UserData* data = (UserData*)userData;
    float* start = ((float**)(data->addr2))[ind];
    float* buff = (float*)buf;

    if( ind<0 ) return CV_BADFACTOR_ERR;
    if( buf==NULL || userData==NULL ) return CV_NULLPTR_ERR;

    for( i=0; i<data->size2.height; i++, start+=data->step2/4 )
        for( j=0; j<data->size2.width; j++, k++ )
            start[j] = buff[k];
    return CV_NO_ERR;
}

/*##########################################=-- Test body --=###########################*/
static int fmaEigenObjects( void )
{
    int n, n4, i, j, ie, m1, rep = 0, roi, roi4, bufSize;
    int roix=0, roiy=0, sizex, sizey, step, step4, step44;
    int err0, err1, err2, err3, err4, err5, err6, err7, err=0;
    uchar *pro, *pro0, *object;
    uchar** objs;
    float *covMatr, *covMatr0, *avg, *avg0, *eigVal, *eigVal0, *coeffs, *coeffs0,
          covMatrMax, coeffm, singleCoeff0;
    float **eigObjs, **eigObjs0;
    IplImage **Objs, **EigObjs, **EigObjs0, *Pro, *Pro0, *Object, *Avg, *Avg0;
    double eps0, amax=0, singleCoeff, p;
    AtsRandState state;
    CvSize size;
    int  r;
    CvTermCriteria limit;
    UserData userData;
    int (*read_callback)( int ind, void* buf, void* userData)=
                 read_callback_8u;
    int (*read2_callback)( int ind, void* buf, void* userData)=
                 read_callback_32f;
    int (*write_callback)( int ind, void* buf, void* userData)=
                 write_callback_32f;
    CvInput* u_r = (CvInput*)&read_callback;
    CvInput* u_r2= (CvInput*)&read2_callback;
    CvInput* u_w = (CvInput*)&write_callback;
    void* read_    = (u_r)->data;
    void* read_2   = (u_r2)->data;
    void* write_   = (u_w)->data;

    /* Reading test parameters */
    trsiRead( &obj_width,    "100", "width of objects" );
    trsiRead( &obj_height,   "100", "height of objects" );
    trsiRead( &obj_number,    "11", "number of objects" );
    trsdRead( &rel_bufSize, "0.09", "relative i/o buffer size" );

    if( rel_bufSize < 0.0 ) rel_bufSize = 0.0;
    m1  = obj_number - 1;
    eps0= 1.0e-27;
    n = obj_width * obj_height;
    sizex = obj_width,  sizey = obj_height;

    Objs     = (IplImage**)cvAlloc(sizeof(IplImage*) * obj_number );
    EigObjs  = (IplImage**)cvAlloc(sizeof(IplImage*) * m1         );
    EigObjs0 = (IplImage**)cvAlloc(sizeof(IplImage*) * m1         );

    objs     = (uchar**)cvAlloc(sizeof(uchar*) * obj_number     );
    eigObjs  = (float**)cvAlloc(sizeof(float*) * m1             );
    eigObjs0 = (float**)cvAlloc(sizeof(float*) * m1             );
    covMatr  = (float*) cvAlloc(sizeof(float)  * obj_number * obj_number );
    covMatr0 = (float*) cvAlloc(sizeof(float)  * obj_number * obj_number );
    coeffs   = (float*) cvAlloc(sizeof(float*) * m1             );
    coeffs0  = (float*) cvAlloc(sizeof(float*) * m1             );
    eigVal   = (float*) cvAlloc(sizeof(float)  * obj_number     );
    eigVal0  = (float*) cvAlloc(sizeof(float)  * obj_number     );

    size.width = obj_width;  size.height = obj_height;
    atsRandInit( &state, 0, 255, 13 );

    Avg  = cvCreateImage( size, IPL_DEPTH_32F, 1 );
    cvSetImageROI( Avg, cvRect(0, 0, Avg->width, Avg->height) );
    Avg0 = cvCreateImage( size, IPL_DEPTH_32F, 1 );
    cvSetImageROI( Avg0, cvRect(0, 0, Avg0->width, Avg0->height) );
    avg  = (float*)Avg->imageData;
    avg0 = (float*)Avg0->imageData;
    Pro  = cvCreateImage( size, IPL_DEPTH_8U, 1 );
    cvSetImageROI( Pro, cvRect(0, 0, Pro->width, Pro->height) );
    Pro0 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
    cvSetImageROI( Pro0, cvRect(0, 0, Pro0->width, Pro0->height) );
    pro  = (uchar*)Pro->imageData;
    pro0 = (uchar*)Pro0->imageData;
    Object = cvCreateImage( size, IPL_DEPTH_8U, 1 );
    cvSetImageROI( Object, cvRect(0, 0, Object->width, Object->height) );
    object = (uchar*)Object->imageData;

    step = Pro->widthStep;  step4 = Avg->widthStep;  step44 = step4/4;
    n = step*obj_height;   n4= step44*obj_height;
    atsbRand8u ( &state, object, n );

    for( i=0; i<obj_number; i++ )
    {
        Objs[i]     = cvCreateImage( size, IPL_DEPTH_8U, 1 );
        cvSetImageROI( Objs[i], cvRect(0, 0, Objs[i]->width, Objs[i]->height) );
        objs[i] = (uchar*)Objs[i]->imageData;
        atsbRand8u ( &state, objs[i], n );
        if( i < m1 )
        {
            EigObjs[i]  = cvCreateImage( size, IPL_DEPTH_32F, 1 );
            cvSetImageROI( EigObjs[i], cvRect(0, 0, EigObjs[i]->width, EigObjs[i]->height) );
            EigObjs0[i] = cvCreateImage( size, IPL_DEPTH_32F, 1 );
            cvSetImageROI( EigObjs0[i], cvRect(0, 0, EigObjs0[i]->width, EigObjs0[i]->height) );
        }
    }

    limit.type = CV_TERMCRIT_ITER;  limit.max_iter = m1;  limit.epsilon = 1;//(float)eps0;

    bufSize = (int)(4*n*obj_number*rel_bufSize);
trsWrite(TW_RUN|TW_CON, "\n i/o buffer size : %10d bytes\n", bufSize );

trsWrite(TW_RUN|TW_CON, "\n ROI unsupported\n" );

/* User data fill */
    userData.addr1 = (void*)objs;
    userData.addr2 = (void*)eigObjs;
    userData.step1 = step;
    userData.step2 = step4;


repeat:
    roi  = roiy*step    + roix;
    roi4 = roiy*step44 + roix;

    Avg->roi->xOffset    = roix;         Avg->roi->yOffset    = roiy;
    Avg->roi->height     = size.height;  Avg->roi->width      = size.width;
    Avg0->roi->xOffset   = roix;         Avg0->roi->yOffset   = roiy;
    Avg0->roi->height    = size.height;  Avg0->roi->width     = size.width;
    Pro->roi->xOffset    = roix;         Pro->roi->yOffset    = roiy;
    Pro->roi->height     = size.height;  Pro->roi->width      = size.width;
    Pro0->roi->xOffset   = roix;         Pro0->roi->yOffset   = roiy;
    Pro0->roi->height    = size.height;  Pro0->roi->width     = size.width;
    Object->roi->xOffset = roix;         Object->roi->yOffset = roiy;
    Object->roi->height  = size.height;  Object->roi->width   = size.width;

    for( i=0; i<obj_number; i++ )
    {
        Objs[i]->roi->xOffset = roix;         Objs[i]->roi->yOffset = roiy;
        Objs[i]->roi->height  = size.height;  Objs[i]->roi->width   = size.width;
        objs[i] = (uchar*)Objs[i]->imageData + roi;
        if( i < m1 )
        {
            EigObjs[i]->roi->xOffset  = roix;        EigObjs[i]->roi->yOffset  = roiy;
            EigObjs[i]->roi->height   = size.height; EigObjs[i]->roi->width    = size.width;
            EigObjs0[i]->roi->xOffset = roix;        EigObjs0[i]->roi->yOffset = roiy;
            EigObjs0[i]->roi->height  = size.height; EigObjs0[i]->roi->width   = size.width;
            eigObjs[i]  = (float*)EigObjs[i]->imageData  + roi4;
            eigObjs0[i] = (float*)EigObjs0[i]->imageData + roi4;
        }
    }

    userData.size1 = userData.size2 = size;

/* =================================== Test functions run ============================= */

    r = _cvCalcEigenObjects_8u32fR_q( obj_number, objs, step, eigObjs0, step4,
                                    size, eigVal0, avg0+roi4, step4, &m1, &eps0 );

    r = _cvEigenDecomposite_8u32fR_q( object+roi, step, m1, eigObjs0, step4,
                                    avg0+roi4, step4, size, coeffs0 );

    r = _cvEigenProjection_8u32fR_q( m1, eigObjs0, step4, coeffs0, avg0+roi4, step4,
                                   pro0+roi, step, size );

    r = _cvCalcCovarMatrix_8u32fR_q( obj_number, objs, step, avg0+roi4, step4,
                                     size, covMatr0 );

    singleCoeff0 = _cvCalcDecompCoeff_8u32fR_q( object+roi, step, eigObjs0[0], step4,
                                                avg0+roi4, step4, size );

    covMatrMax = 0.f;
    for( i=0; i<obj_number*obj_number; i++ )
        if( covMatrMax < (float)fabs( covMatr[i] ) )
            covMatrMax = (float)fabs( covMatr[i] );

    amax = 0;
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
            {
                int ij = i*obj_width + j;
                float e = eigObjs0[ie][ij];
                if( amax < fabs(e) ) amax = fabs(e);
            }

    coeffm = 0.f;
    for( i=0; i<m1; i++ )
        if( coeffm < (float)fabs(coeffs0[i]) ) coeffm = (float)fabs(coeffs0[i]);

/*- - - - - - - - - - - - - - - - - - - - - without callbacks - - - - - - - - - - - - - */
    for( i=0; i<obj_number*obj_number; i++ ) covMatr[i] = covMatr0[i];
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ ) pro[i*step + j] = pro0[i*step + j];
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ ) avg[i*step44 + j] = avg0[i*step44 + j];
    for( i=0; i<m1;   i++ ) { coeffs[i] = coeffs0[i];  eigVal[i] = eigVal0[i]; }
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
                eigObjs[ie][i*step44+j] = eigObjs0[ie][i*step44+j];

    err1 = err2 = err3 = err4 = err5 = err6 = err7 = 0;

    cvCalcCovarMatrixEx( obj_number,
                         (void*)Objs,
                         CV_EIGOBJ_NO_CALLBACK,
                         bufSize,
                         NULL,
                         (void*)&userData,
                         Avg,
                         covMatr );

    cvCalcEigenObjects ( obj_number,
                         (void*)Objs,
                         (void*)EigObjs,
                         CV_EIGOBJ_NO_CALLBACK,
                         bufSize,
                         (void*)&userData,
                         &limit,
                         Avg,
                         eigVal );

    singleCoeff = cvCalcDecompCoeff( Object, EigObjs[0], Avg );
    if( fabs( (singleCoeff - singleCoeff0)/singleCoeff0 ) > RELDIFF ) err7++;

    cvEigenDecomposite( Object,
                        m1,
                        (void*)EigObjs,
                        CV_EIGOBJ_NO_CALLBACK,
                        (void*)&userData,
                        Avg,
                        coeffs );
    cvEigenProjection ( (void*)EigObjs,
                        m1,
                        CV_EIGOBJ_NO_CALLBACK,
                        (void*)&userData,
                        coeffs,
                        Avg,
                        Pro );

/*  Covariance matrix comparision */
    for( i=0; i<obj_number*obj_number; i++ )
        if( fabs(covMatr[i] - covMatr0[i]) > RELDIFF*fabs(covMatrMax) ) err6++;

/*  Averaged object comparision */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ )
        {
            int ij = i*step44 + j;
            if( fabs( (avg+roi)[ij] - (avg0+roi)[ij] ) > MAXDIFF ) err1++;
        }

/*  Eigen objects comparision */
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
            {
                int ij = i*step44 + j;
                float e0 = (eigObjs0[ie])[ij],  e = (eigObjs[ie])[ij];
                if( fabs( (e-e0)/amax ) > RELDIFF ) err2++;
            }

/*  Eigen values comparision */
    for( i=0; i<m1; i++ )
    {
        double e0 = eigVal0[i], e = eigVal[i];
        if(e0)
            if( fabs( (e-e0)/e0 ) > RELDIFF ) err3++;
    }

/*  Decomposition coefficients comparision */
    for( i=0; i<m1; i++ )
        if(coeffs0[i])
            if( fabs( (coeffs[i] - coeffs0[i])/coeffm ) > RELDIFF ) err4++;

/*  Projection comparision */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ )
        {
            int ij = i*step + j;
            if( fabs( (double)((pro+roi)[ij] - (pro0+roi)[ij]) ) > MAXDIFF ) err5++;
        }

    err0 = 0;
    p = 100.f*err6/(float)(obj_number*obj_number);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Covar. matrix - %d errors (%7.3f %% );\n", err6, p );
        err0 += err6;
    }
    p = 100.f*err1/(float)(size.height*size.width);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Averaged obj. - %d errors (%7.3f %% );\n", err1, p );
        err0 += err1;
    }
    p = 100.f*err3/(float)(m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen values  - %d errors (%7.3f %% );\n", err3, p );
        err0 += err3;
    }
    p = 100.f*err2/(float)(size.height*size.width*m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen objects - %d errors (%7.3f %% );\n", err2, p );
        err0 += err2;
    }
    p = 100.f*err4/(float)(m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Decomp.coeffs - %d errors (%7.3f %% );\n", err4, p );
        err0 += err4;
    }
    if( ((float)err7)/m1 > 0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Single dec.c. - %d errors        ;\n", err7);
        err0 += err7;
    }
    p = 100.f*err5/(float)(size.height*size.width);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Projection    - %d errors (%7.3f %% );\n", err5, p );
        err0 += err5;
    }
    trsWrite(TW_RUN|TW_CON, "     without callbacks :  %8d  errors;\n", err0 );

    err += err0;

/*- - - - - - - - - - - - - - - - - - - - - input callback - - - - - - - - - - - - - */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ ) pro[i*step + j] = pro0[i*step + j];
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ ) avg[i*step44 + j] = avg0[i*step44 + j];
    for( i=0; i<m1;   i++ ) { coeffs[i] = coeffs0[i];  eigVal[i] = eigVal0[i]; }
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
                eigObjs[ie][i*step44+j] = eigObjs0[ie][i*step44+j];

    err1 = err2 = err3 = err4 = err5 = err6 = err7 = 0;

    cvCalcEigenObjects ( obj_number,
                         read_,
                         (void*)EigObjs,
                         CV_EIGOBJ_INPUT_CALLBACK,
                         bufSize,
                         (void*)&userData,
                         &limit,
                         Avg,
                         eigVal );

    cvEigenDecomposite( Object,
                        m1,
                        read_2,
                        CV_EIGOBJ_INPUT_CALLBACK,
                        (void*)&userData,
                        Avg,
                        coeffs );

    cvEigenProjection ( read_2,
                        m1,
                        CV_EIGOBJ_INPUT_CALLBACK,
                        (void*)&userData,
                        coeffs,
                        Avg,
                        Pro );

/*  Averaged object comparision */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ )
        {
            int ij = i*step44 + j;
            if( fabs( (avg+roi)[ij] - (avg0+roi)[ij] ) > MAXDIFF ) err1++;
        }

/*  Eigen objects comparision */
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
            {
                int ij = i*step44 + j;
                float e0 = (eigObjs0[ie])[ij],  e = (eigObjs[ie])[ij];
                    if( fabs( (e-e0)/amax ) > RELDIFF ) err2++;
            }

/*  Eigen values comparision */
    for( i=0; i<m1; i++ )
    {
        double e0 = eigVal0[i], e = eigVal[i];
        if(e0)
            if( fabs( (e-e0)/e0 ) > RELDIFF ) err3++;
    }

/*  Projection comparision */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ )
        {
            int ij = i*step + j;
            if( fabs( (double)((pro+roi)[ij] - (pro0+roi)[ij]) ) > MAXDIFF ) err5++;
        }

/*  Decomposition coefficients comparision */
    for( i=0; i<m1; i++ )
        if(coeffs0[i])
            if( fabs( (coeffs[i] - coeffs0[i])/coeffm ) > RELDIFF ) err4++;

    err0 = 0;
    p = 100.f*err1/(float)(size.height*size.width);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Averaged obj. - %d errors (%7.3f %% );\n", err1, p );
        err0 += err1;
    }
    p = 100.f*err3/(float)(m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen values  - %d errors (%7.3f %% );\n", err3, p );
        err0 += err3;
    }
    p = 100.f*err2/(float)(size.height*size.width*m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen objects - %d errors (%7.3f %% );\n", err2, p );
        err0 += err2;
    }
    p = 100.f*err4/(float)(m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Decomp.coeffs - %d errors (%7.3f %% );\n", err4, p );
        err0 += err4;
    }
    p = 100.f*err5/(float)(size.height*size.width);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Projection    - %d errors (%7.3f %% );\n", err5, p );
        err0 += err5;
    }
    trsWrite(TW_RUN|TW_CON, "        input callback :  %8d  errors;\n", err0 );

    err += err0;

/*- - - - - - - - - - - - - - - - - - - - - output callback - - - - - - - - - - - - - */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ ) avg[i*step44 + j] = avg0[i*step44 + j];
    for( i=0; i<m1;   i++ ) eigVal[i] = eigVal0[i];
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
                eigObjs[ie][i*step44+j] = eigObjs0[ie][i*step44+j];

    err1 = err2 = err3 = err4 = err5 = 0;

    cvCalcEigenObjects ( obj_number,
                         (void*)Objs,
                         write_,
                         CV_EIGOBJ_OUTPUT_CALLBACK,
                         bufSize,
                         (void*)&userData,
                         &limit,
                         Avg,
                         eigVal );

/*  Averaged object comparision */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ )
        {
            int ij = i*step44 + j;
            if( fabs( (avg+roi)[ij] - (avg0+roi)[ij] ) > MAXDIFF ) err1++;
        }

/*  Eigen objects comparision */
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
            {
                int ij = i*step44 + j;
                float e0 = (eigObjs0[ie])[ij],  e = (eigObjs[ie])[ij];
                    if( fabs( (e-e0)/amax ) > RELDIFF ) err2++;
            }

/*  Eigen values comparision */
    for( i=0; i<m1; i++ )
    {
        double e0 = eigVal0[i], e = eigVal[i];
        if(e0)
            if( fabs( (e-e0)/e0 ) > RELDIFF ) err3++;
    }

    err0 = 0;
    p = 100.f*err1/(float)(size.height*size.width);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Averaged obj. - %d errors (%7.3f %% );\n", err1, p );
        err0 += err1;
    }
    p = 100.f*err3/(float)(m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen values  - %d errors (%7.3f %% );\n", err3, p );
        err0 += err3;
    }
    p = 100.f*err2/(float)(size.height*size.width*m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen objects - %d errors (%7.3f %% );\n", err2, p );
        err0 += err2;
    }
    trsWrite(TW_RUN|TW_CON, "       output callback :  %8d  errors;\n", err0 );

    err += err0;

/*- - - - - - - - - - - - - - - - - - - - - both callbacks - - - - - - - - - - - - - */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ ) avg[i*step44 + j] = avg0[i*step44 + j];
    for( i=0; i<m1;   i++ ) eigVal[i] = eigVal0[i];
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
                eigObjs[ie][i*step44+j] = eigObjs0[ie][i*step44+j];

    err1 = err2 = err3 = err4 = err5 = 0;

    cvCalcEigenObjects ( obj_number,
                         read_,
                         write_,
                         CV_EIGOBJ_INPUT_CALLBACK | CV_EIGOBJ_OUTPUT_CALLBACK,
                         bufSize,
                         (void*)&userData,
                         &limit,
                         Avg,
                         eigVal );

/*  Averaged object comparision */
    for( i=0; i<size.height; i++ )
        for( j=0; j<size.width; j++ )
        {
            int ij = i*step44 + j;
            if( fabs( (avg+roi)[ij] - (avg0+roi)[ij] ) > MAXDIFF ) err1++;
        }

/*  Eigen objects comparision */
    for( ie=0; ie<m1; ie++ )
        for( i=0; i<size.height; i++ )
            for( j=0; j<size.width; j++ )
            {
                int ij = i*step44 + j;
                float e0 = (eigObjs0[ie])[ij],  e = (eigObjs[ie])[ij];
                    if( fabs( (e-e0)/amax ) > RELDIFF ) err2++;
            }

/*  Eigen values comparision */
    for( i=0; i<m1; i++ )
    {
        double e0 = eigVal0[i], e = eigVal[i];
        if(e0)
            if( fabs( (e-e0)/e0 ) > RELDIFF ) err3++;
    }

    err0 = 0;
    p = 100.f*err1/(float)(size.height*size.width);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Averaged obj. - %d errors (%7.3f %% );\n", err1, p );
        err0 += err1;
    }
    p = 100.f*err3/(float)(m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen values  - %d errors (%7.3f %% );\n", err3, p );
        err0 += err3;
    }
    p = 100.f*err2/(float)(size.height*size.width*m1);
    if( p>0.1 )
    {
        trsWrite(TW_RUN|TW_CON, "         Eigen objects - %d errors (%7.3f %% );\n", err2, p );
        err0 += err2;
    }
    trsWrite(TW_RUN|TW_CON, "        both callbacks :  %8d  errors;\n", err0 );

    err += err0;


/*================================-- test with ROI --===================================*/

    if(!rep)
    {
        roix  = (int)(0.157f*obj_width);
        roiy  = (int)(0.131f*obj_height);
        sizex = (int)(0.611f*obj_width);
        sizey = (int)(0.737f*obj_height);
        roi   = roiy*obj_width + roix;

trsWrite(TW_RUN|TW_CON, "\n ROI   supported\n" );
        rep++;
        size.width = sizex;  size.height = sizey;

        goto repeat;
    }

/*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ free memory ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/
    cvReleaseImage( &Avg    );
    cvReleaseImage( &Avg0   );
    cvReleaseImage( &Pro    );
    cvReleaseImage( &Pro0   );
    cvReleaseImage( &Object );
    for( i=0; i<obj_number; i++ )
    {
        cvReleaseImage( &Objs[i] );
        if( i < m1 )
        {
            cvReleaseImage( &EigObjs[i]  );
            cvReleaseImage( &EigObjs0[i] );
        }
    }

    cvFree( &objs     );
    cvFree( &eigObjs  );
    cvFree( &eigObjs0 );
    cvFree( &coeffs   );
    cvFree( &coeffs0  );
    cvFree( &eigVal   );
    cvFree( &eigVal0  );
    cvFree( &Objs     );
    cvFree( &EigObjs  );
    cvFree( &EigObjs0 );
    cvFree( &covMatr  );
    cvFree( &covMatr0 );

trsWrite(TW_RUN|TW_CON, "\n Errors number: %d\n", err );

    if(err) return trsResult( TRS_FAIL, "Algorithm test has passed. %d errors.", err );
    else    return trsResult( TRS_OK, "Algorithm test has passed successfully" );
    
} /*fma*/

/*------------------------------------------- Initialize function ------------------------ */
void InitAEigenObjects( void )
{
   /* Registering test function */
    trsReg( FuncName, TestName, TestClass, fmaEigenObjects );
} /* InitAEigenObjects */

#endif

/*  End of file  */
