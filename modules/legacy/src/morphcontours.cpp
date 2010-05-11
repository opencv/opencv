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

#include "precomp.hpp"

#define PATH_TO_E       1
#define PATH_TO_SE      2
#define PATH_TO_S       3

#define K_S         2
#define E_S         2
#define C_S         .01
#define K_Z         5000
#define K_NM        50000
#define K_B         40
#define NULL_EDGE   0.001f
#define inf         DBL_MAX

typedef struct __CvWork
{
    double w_east;
    double w_southeast;
    double w_south;
    char path_e;
    char path_se;
    char path_s;
}_CvWork;


double _cvBendingWork(  CvPoint2D32f* B0,
                        CvPoint2D32f* F0,
                        CvPoint2D32f* B1,
                        CvPoint2D32f* F1/*,
                        CvPoint* K */);

double _cvStretchingWork(CvPoint2D32f* P1,
                         CvPoint2D32f* P2);

void _cvWorkEast     (int i, int j, _CvWork** W, CvPoint2D32f* edges1, CvPoint2D32f* edges2);
void _cvWorkSouthEast(int i, int j, _CvWork** W, CvPoint2D32f* edges1, CvPoint2D32f* edges2);
void _cvWorkSouth    (int i, int j, _CvWork** W, CvPoint2D32f* edges1, CvPoint2D32f* edges2);

static CvPoint2D32f null_edge = {0,0};

double _cvStretchingWork(CvPoint2D32f* P1,
                         CvPoint2D32f* P2)
{
    double L1,L2, L_min, dL;

    L1 = sqrt( (double)P1->x*P1->x + P1->y*P1->y);
    L2 = sqrt( (double)P2->x*P2->x + P2->y*P2->y);
    
    L_min = MIN(L1, L2);
    dL = fabs( L1 - L2 );

    return K_S * pow( dL, E_S ) / ( L_min + C_S*dL );
}


////////////////////////////////////////////////////////////////////////////////////
double _cvBendingWork(  CvPoint2D32f* B0,
                        CvPoint2D32f* F0,
                        CvPoint2D32f* B1,
                        CvPoint2D32f* F1/*,
                        CvPoint* K*/)
{
    CvPoint2D32f Q( CvPoint2D32f q0, CvPoint2D32f q1, CvPoint2D32f q2, double t );
    double angle( CvPoint2D32f A, CvPoint2D32f B );

    CvPoint2D32f Q0, Q1, Q2;
    CvPoint2D32f Q1_nm = { 0, 0 }, Q2_nm = { 0, 0 };
    double d0, d1, d2, des, t_zero;
    double k_zero, k_nonmon;
    CvPoint2D32f center;
    double check01, check02;
    char check_origin;
    double d_angle, d_nm_angle;
/*
    if( (B0->x==0) && (B0->y==0) )
    {
        if( (F0->x==0) && (F0->y==0) )
        {
            B1->x = -B1->x;
            B1->y = -B1->y;

            d_angle = acos( (B1->x*F1->x + B1->y*F1->y)/sqrt( (B1->x*B1->x + B1->y*B1->y)*(F1->x*F1->x + F1->y*F1->y) ) );
            d_angle = CV_PI - d_angle;

            B1->x = -B1->x;
            B1->y = -B1->y;

            //return d_angle*K_B;
            return 100;
        }
        K->x = -K->x;
        K->y = -K->y;
        B1->x = -B1->x;
        B1->y = -B1->y;

        d_angle = acos( (B1->x*F1->x + B1->y*F1->y)/sqrt( (B1->x*B1->x + B1->y*B1->y)*(F1->x*F1->x + F1->y*F1->y) ) );
        d_angle = d_angle - acos( (F0->x*K->x + F0->y*K->y)/sqrt( (F0->x*F0->x + F0->y*F0->y)*(K->x*K->x + K->y*K->y) ) );
        d_angle = d_angle - CV_PI*0.5;
        d_angle = fabs(d_angle);

        
        K->x = -K->x;
        K->y = -K->y;
        B1->x = -B1->x;
        B1->y = -B1->y;

        //return d_angle*K_B;
        return 100;
    }


    if( (F0->x==0) && (F0->y==0) )
        {
            K->x = -K->x;
            K->y = -K->y;
            B1->x = -B1->x;
            B1->y = -B1->y;

            d_angle = acos( (B1->x*F1->x + B1->y*F1->y)/sqrt( (B1->x*B1->x + B1->y*B1->y)*(F1->x*F1->x + F1->y*F1->y) ) );
            d_angle = d_angle - acos( (B0->x*K->x + B0->y*K->y)/sqrt( (B0->x*B0->x + B0->y*B0->y)*(K->x*K->x + K->y*K->y) ) );
            d_angle = d_angle - CV_PI*0.5;
            d_angle = fabs(d_angle);

            K->x = -K->x;
            K->y = -K->y;
            B1->x = -B1->x;
            B1->y = -B1->y;

            //return d_angle*K_B;
            return 100;
        }
///////////////

    if( (B1->x==0) && (B1->y==0) )
    {
        if( (F1->x==0) && (F1->y==0) )
        {
            B0->x = -B0->x;
            B0->y = -B0->y;

            d_angle = acos( (B0->x*F0->x + B0->y*F0->y)/sqrt( (B0->x*B0->x + B0->y*B0->y)*(F0->x*F0->x + F0->y*F0->y) ) );
            d_angle = CV_PI - d_angle;

            B0->x = -B0->x;
            B0->y = -B0->y;

            //return d_angle*K_B;
            return 100;
        }
        K->x = -K->x;
        K->y = -K->y;
        B0->x = -B0->x;
        B0->y = -B0->y;

        d_angle = acos( (B0->x*F0->x + B0->y*F0->y)/sqrt( (B0->x*B0->x + B0->y*B0->y)*(F0->x*F0->x + F0->y*F0->y) ) );
        d_angle = d_angle - acos( (F1->x*K->x + F1->y*K->y)/sqrt( (F1->x*F1->x + F1->y*F1->y)*(K->x*K->x + K->y*K->y) ) );
        d_angle = d_angle - CV_PI*0.5;
        d_angle = fabs(d_angle);

        K->x = -K->x;
        K->y = -K->y;
        B0->x = -B0->x;
        B0->y = -B0->y;

        //return d_angle*K_B;
        return 100;
    }


    if( (F1->x==0) && (F1->y==0) )
        {
            K->x = -K->x;
            K->y = -K->y;
            B0->x = -B0->x;
            B0->y = -B0->y;

            d_angle = acos( (B0->x*F0->x + B0->y*F0->y)/sqrt( (B0->x*B0->x + B0->y*B0->y)*(F0->x*F0->x + F0->y*F0->y) ) );
            d_angle = d_angle - acos( (B1->x*K->x + B1->y*K->y)/sqrt( (B1->x*B1->x + B1->y*B1->y)*(K->x*K->x + K->y*K->y) ) );
            d_angle = d_angle - CV_PI*0.5;
            d_angle = fabs(d_angle);

            K->x  = -K->x;
            K->y  = -K->y;
            B0->x = -B0->x;
            B0->y = -B0->y;

            //return d_angle*K_B;
            return 100;
        }

*/

/*
    B0->x = -B0->x;
    B0->y = -B0->y;
    B1->x = -B1->x;
    B1->y = -B1->y;
*/
    Q0.x = F0->x * (-B0->x) + F0->y * (-B0->y);
    Q0.y = F0->x * (-B0->y) - F0->y * (-B0->x);

    Q1.x = 0.5f*( (F1->x * (-B0->x) + F1->y * (-B0->y)) + (F0->x * (-B1->x) + F0->y * (-B1->y)) );
    Q1.y = 0.5f*( (F1->x * (-B0->y) - F1->y * (-B0->x)) + (F0->x * (-B1->y) - F0->y * (-B1->x)) );

    Q2.x = F1->x * (-B1->x) + F1->y * (-B1->y);
    Q2.y = F1->x * (-B1->y) - F1->y * (-B1->x);

    d0 = Q0.x * Q1.y - Q0.y * Q1.x;
    d1 = 0.5f*(Q0.x * Q2.y - Q0.y * Q2.x);
    d2 = Q1.x * Q2.y - Q1.y * Q2.x;

    // Check angles goes to zero
    des = Q1.y*Q1.y - Q0.y*Q2.y;

    k_zero = 0;

    if( des >= 0 )
    {
        t_zero = ( Q0.y - Q1.y + sqrt(des) )/( Q0.y - 2*Q1.y + Q2.y );

        if( (0 < t_zero) && (t_zero < 1) && ( Q(Q0, Q1, Q2, t_zero).x > 0 ) )
        {
            k_zero = inf;
        }

        t_zero = ( Q0.y - Q1.y - sqrt(des) )/( Q0.y - 2*Q1.y + Q2.y );

        if( (0 < t_zero) && (t_zero < 1) && ( Q(Q0, Q1, Q2, t_zero).x > 0 ) )
        {
            k_zero = inf;
        }
    }

    // Check nonmonotonic
    des = d1*d1 - d0*d2;

    k_nonmon = 0;

    if( des >= 0 )
    {
        t_zero = ( d0 - d1 - sqrt(des) )/( d0 - 2*d1 + d2 );

        if( (0 < t_zero) && (t_zero < 1) )
        {
            k_nonmon = 1;
            Q1_nm = Q(Q0, Q1, Q2, t_zero);
        }

        t_zero = ( d0 - d1 + sqrt(des) )/( d0 - 2*d1 + d2 );

        if( (0 < t_zero) && (t_zero < 1) )
        {
            k_nonmon += 2;
            Q2_nm = Q(Q0, Q1, Q2, t_zero);
        }
    }

    // Finde origin lie in Q0Q1Q2
    check_origin = 1;

    center.x = (Q0.x + Q1.x + Q2.x)/3;
    center.y = (Q0.y + Q1.y + Q2.y)/3;

    check01 = (center.x - Q0.x)*(Q1.y - Q0.y) + (center.y - Q0.y)*(Q1.x - Q0.x);
    check02 = (-Q0.x)*(Q1.y - Q0.y) + (-Q0.y)*(Q1.x - Q0.x);
    if( check01*check02 > 0 )
    {
        check01 = (center.x - Q1.x)*(Q2.y - Q1.y) + (center.y - Q1.y)*(Q2.x - Q1.x);
        check02 = (-Q1.x)*(Q2.y - Q1.y) + (-Q1.y)*(Q2.x - Q1.x);
        if( check01*check02 > 0 )
        {
            check01 = (center.x - Q2.x)*(Q0.y - Q2.y) + (center.y - Q2.y)*(Q0.x - Q2.x);
            check02 = (-Q2.x)*(Q0.y - Q2.y) + (-Q2.y)*(Q0.x - Q2.x);
            if( check01*check02 > 0 )
            {
                check_origin = 0;
            }
        }
    }

    // Calculate angle
    d_nm_angle = 0;
    d_angle = angle(Q0,Q2);
    if( k_nonmon == 0 )
    {
        if( check_origin == 0 )
        {
        }
        else
        {
            d_angle = 2*CV_PI - d_angle;
        }
    }
    else
    {
        if( k_nonmon == 1 )
        {
            d_nm_angle = angle(Q0,Q1_nm);
            if(d_nm_angle > d_angle)
            {
                d_nm_angle = d_nm_angle - d_angle;
            }
        }

        if( k_nonmon == 2 )
        {
            d_nm_angle = angle(Q0,Q2_nm);
            if(d_nm_angle > d_angle)
            {
                d_nm_angle = d_nm_angle - d_angle;
            }
        }

        if( k_nonmon == 3 )
        {
            d_nm_angle = angle(Q0,Q1_nm);
            if(d_nm_angle > d_angle)
            {
                d_nm_angle = d_nm_angle - d_angle;
                d_nm_angle = d_nm_angle + angle(Q0, Q2_nm);
            }
            else
            {
                d_nm_angle = d_nm_angle + angle(Q2,Q2_nm);
            }
        }
    }
/*
    B0->x = -B0->x;
    B0->y = -B0->y;
    B1->x = -B1->x;
    B1->y = -B1->y;
*/
    return d_angle*K_B + d_nm_angle*K_NM + k_zero*K_Z;
    //return 0;
}


/////////////////////////////////////////////////////////////////////////////////
void _cvWorkEast(int i, int j, _CvWork** W, CvPoint2D32f* edges1, CvPoint2D32f* edges2)
{
    double w1,w2;
    CvPoint2D32f small_edge;

    //W[i,j].w_east
    w1 = W[i-1][j].w_east /*+ _cvBendingWork(   &edges1[i-2],
                                            &edges1[i-1],
                                            &null_edge ,
                                            &null_edge,
                                            NULL)*/;

    small_edge.x = NULL_EDGE*edges1[i-1].x;
    small_edge.y = NULL_EDGE*edges1[i-1].y;

    w2 = W[i-1][j].w_southeast + _cvBendingWork(&edges1[i-2],
                                                &edges1[i-1],
                                                &edges2[j-1],
                                                /*&null_edge*/&small_edge/*,
                                                &edges2[j]*/);

    if(w1<w2)
    {
        W[i][j].w_east = w1 + _cvStretchingWork( &edges1[i-1], &null_edge );
        W[i][j].path_e = PATH_TO_E;
    }
    else
    {
        W[i][j].w_east = w2 + _cvStretchingWork( &edges1[i-1], &null_edge );
        W[i][j].path_e = PATH_TO_SE;
    }
}





////////////////////////////////////////////////////////////////////////////////////
void _cvWorkSouthEast(int i, int j, _CvWork** W, CvPoint2D32f* edges1, CvPoint2D32f* edges2)
{
    double w1,w2,w3;
    CvPoint2D32f small_edge;

    //W[i,j].w_southeast
    small_edge.x = NULL_EDGE*edges1[i-2].x;
    small_edge.y = NULL_EDGE*edges1[i-2].y;

    w1 = W[i-1][j-1].w_east + _cvBendingWork(&edges1[i-2],
                                            &edges1[i-1],                                           
                                            /*&null_edge*/&small_edge,
                                            &edges2[j-1]/*,
                                            &edges2[j-2]*/);

    w2 = W[i-1][j-1].w_southeast + _cvBendingWork(  &edges1[i-2],
                                                    &edges1[i-1],
                                                    &edges2[j-2],
                                                    &edges2[j-1]/*,
                                                    NULL*/);

    small_edge.x = NULL_EDGE*edges2[j-2].x;
    small_edge.y = NULL_EDGE*edges2[j-2].y;

    w3 = W[i-1][j-1].w_south + _cvBendingWork(  /*&null_edge*/&small_edge,
                                                &edges1[i-1],                                           
                                                &edges2[j-2],
                                                &edges2[j-1]/*,
                                                &edges1[i-2]*/);

    if( w1<w2 )
    {
        if(w1<w3)
        {
            W[i][j].w_southeast = w1 + _cvStretchingWork( &edges1[i-1], &edges2[j-1] );
            W[i][j].path_se = PATH_TO_E;
        }
        else
        {
            W[i][j].w_southeast = w3 + _cvStretchingWork( &edges1[i-1], &edges2[j-1] );
            W[i][j].path_se = 3;
        }
    }
    else
    {
        if( w2<w3)
        {
            W[i][j].w_southeast = w2 + _cvStretchingWork( &edges1[i-1], &edges2[j-1] );
            W[i][j].path_se = PATH_TO_SE;
        }
        else
        {
            W[i][j].w_southeast = w3 + _cvStretchingWork( &edges1[i-1], &edges2[j-1] );
            W[i][j].path_se = 3;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////
void _cvWorkSouth(int i, int j, _CvWork** W, CvPoint2D32f* edges1, CvPoint2D32f* edges2)
{
    double w1,w2;
    CvPoint2D32f small_edge;

    //W[i,j].w_south

    small_edge.x = NULL_EDGE*edges2[j-1].x;
    small_edge.y = NULL_EDGE*edges2[j-1].y;

    w1 = W[i][j-1].w_southeast + _cvBendingWork(&edges1[i-1],
                                                /*&null_edge*/&small_edge,
                                                &edges2[j-2],
                                                &edges2[j-1]/*,
                                                &edges1[i]*/);

    w2 = W[i][j-1].w_south /*+ _cvBendingWork(  &null_edge ,
                                            &null_edge,
                                            &edges2[j-2],
                                            &edges2[j-1],
                                            NULL)*/;

    if( w1<w2 )
    {
        W[i][j].w_south = w1 + _cvStretchingWork( &null_edge, &edges2[j-1] );
        W[i][j].path_s = PATH_TO_SE;
    }
    else
    {
        W[i][j].w_south = w2 + _cvStretchingWork( &null_edge, &edges2[j-1] );
        W[i][j].path_s = 3;
    }
}

//===================================================
CvPoint2D32f Q(CvPoint2D32f q0,CvPoint2D32f q1,CvPoint2D32f q2,double t)
{
    CvPoint2D32f q;

    q.x = (float)(q0.x*(1-t)*(1-t) + 2*q1.x*t*(1-t) + q2.x*t*t);
    q.y = (float)(q0.y*(1-t)*(1-t) + 2*q1.y*t*(1-t) + q2.y*t*t);

    return q;       
}

double angle(CvPoint2D32f A, CvPoint2D32f B)
{
    return acos( (A.x*B.x + A.y*B.y)/sqrt( (double)(A.x*A.x + A.y*A.y)*(B.x*B.x + B.y*B.y) ) );
}

/***************************************************************************************\
*
*   This function compute intermediate polygon between contour1 and contour2
*
*   Correspondence between points of contours specify by corr
*
*   param = [0,1];  0 correspondence to contour1, 1 - contour2
*
\***************************************************************************************/
CvSeq* icvBlendContours(CvSeq* contour1, 
                        CvSeq* contour2,
                        CvSeq* corr,
                        double param,
                        CvMemStorage* storage)
{
    int j;
    
    CvSeqWriter writer01;
    CvSeqReader reader01;

    int Ni,Nj;              // size of contours
    int i;                  // counter

    CvPoint* point1;        // array of first contour point
    CvPoint* point2;        // array of second contour point

    CvPoint point_output;   // intermediate storage of ouput point

    int corr_point;

    // Create output sequence.
    CvSeq* output = cvCreateSeq(0,                      
                                sizeof(CvSeq),
                                sizeof(CvPoint),
                                storage );

    // Find size of contours.
    Ni = contour1->total + 1;
    Nj = contour2->total + 1;

    point1 = (CvPoint* )malloc( Ni*sizeof(CvPoint) );
    point2 = (CvPoint* )malloc( Nj*sizeof(CvPoint) );

    // Initialize arrays of point 
    cvCvtSeqToArray( contour1, point1, CV_WHOLE_SEQ );
    cvCvtSeqToArray( contour2, point2, CV_WHOLE_SEQ );

    // First and last point mast be equal.
    point1[Ni-1] = point1[0];
    point2[Nj-1] = point2[0];

    // Initializes process of writing to sequence.
    cvStartAppendToSeq( output, &writer01);

    i = Ni-1; //correspondence to points of contour1
    for( ; corr; corr = corr->h_next )
    {       
        //Initializes process of sequential reading from sequence
        cvStartReadSeq( corr, &reader01, 0 );

        for(j=0; j < corr->total; j++)
        {
            // Read element from sequence.
            CV_READ_SEQ_ELEM( corr_point, reader01 );

            // Compute point of intermediate polygon.
            point_output.x = cvRound(point1[i].x + param*( point2[corr_point].x - point1[i].x ));
            point_output.y = cvRound(point1[i].y + param*( point2[corr_point].y - point1[i].y ));
            
            // Write element to sequence.
            CV_WRITE_SEQ_ELEM( point_output, writer01 );
        }
        i--;
    }
    // Updates sequence header.
    cvFlushSeqWriter( &writer01 );
    
    return output;
}

/**************************************************************************************************
*
*
*
*
*
*
*
*
*
*
**************************************************************************************************/


void icvCalcContoursCorrespondence(CvSeq* contour1, 
                                   CvSeq* contour2, 
                                   CvSeq** corr, 
                                   CvMemStorage* storage)
{
    int i,j;                    // counter of cycles
    int Ni,Nj;                  // size of contours
    _CvWork** W;                // graph for search minimum of work

    CvPoint* point1;            // array of first contour point
    CvPoint* point2;            // array of second contour point
    CvPoint2D32f* edges1;       // array of first contour edge
    CvPoint2D32f* edges2;       // array of second contour edge

    //CvPoint null_edge = {0,0};    //
    CvPoint2D32f small_edge;
    //double inf;                   // infinity

    CvSeq* corr01;
    CvSeqWriter writer;

    char path;                  //

    // Find size of contours
    Ni = contour1->total + 1;
    Nj = contour2->total + 1;

    // Create arrays
    W = (_CvWork**)malloc(sizeof(_CvWork*)*Ni);
    for(i=0; i<Ni; i++)
    {
        W[i] = (_CvWork*)malloc(sizeof(_CvWork)*Nj);
    }

    point1 = (CvPoint* )malloc( Ni*sizeof(CvPoint) );
    point2 = (CvPoint* )malloc( Nj*sizeof(CvPoint) );
    edges1 = (CvPoint2D32f* )malloc( (Ni-1)*sizeof(CvPoint2D32f) );
    edges2 = (CvPoint2D32f* )malloc( (Nj-1)*sizeof(CvPoint2D32f) );

    // Initialize arrays of point 
    cvCvtSeqToArray( contour1, point1, CV_WHOLE_SEQ );
    cvCvtSeqToArray( contour2, point2, CV_WHOLE_SEQ );

    point1[Ni-1] = point1[0];
    point2[Nj-1] = point2[0];

    for(i=0;i<Ni-1;i++)
    {
        edges1[i].x = (float)( point1[i+1].x - point1[i].x );
        edges1[i].y = (float)( point1[i+1].y - point1[i].y );
    };

    for(i=0;i<Nj-1;i++)
    {
        edges2[i].x = (float)( point2[i+1].x - point2[i].x );
        edges2[i].y = (float)( point2[i+1].y - point2[i].y );
    };

    // Find infinity constant 
    //inf=1;
/////////////

//Find min path in graph

/////////////
    W[0][0].w_east      = 0;
    W[0][0].w_south     = 0;
    W[0][0].w_southeast = 0;

    W[1][1].w_southeast = _cvStretchingWork( &edges1[0], &edges2[0] );
    W[1][1].w_east = inf;
    W[1][1].w_south = inf;
    W[1][1].path_se = PATH_TO_SE;

    W[0][1].w_south =  _cvStretchingWork( &null_edge, &edges2[0] );
    W[0][1].path_s = 3;
    W[1][0].w_east =  _cvStretchingWork( &edges2[0], &null_edge );
    W[1][0].path_e = PATH_TO_E;

    for( i=1; i<Ni; i++ )
    {
        W[i][0].w_south     = inf;
        W[i][0].w_southeast = inf;
    }

    for(j=1; j<Nj; j++)
    {
        W[0][j].w_east      = inf;
        W[0][j].w_southeast = inf;
    }

    for(i=2; i<Ni; i++)
    {
        j=0;/////////
        W[i][j].w_east = W[i-1][j].w_east;
        W[i][j].w_east = W[i][j].w_east /*+ 
            _cvBendingWork( &edges1[i-2], &edges1[i-1], &null_edge, &null_edge, NULL )*/;
        W[i][j].w_east = W[i][j].w_east + _cvStretchingWork( &edges2[i-1], &null_edge );
        W[i][j].path_e = PATH_TO_E;
        
        j=1;//////////
        W[i][j].w_south = inf;

        _cvWorkEast (i, j, W, edges1, edges2);

        W[i][j].w_southeast = W[i-1][j-1].w_east;
        W[i][j].w_southeast = W[i][j].w_southeast + _cvStretchingWork( &edges1[i-1], &edges2[j-1] );

        small_edge.x = NULL_EDGE*edges1[i-2].x;
        small_edge.y = NULL_EDGE*edges1[i-2].y;

        W[i][j].w_southeast = W[i][j].w_southeast + 
            _cvBendingWork( &edges1[i-2], &edges1[i-1], /*&null_edge*/&small_edge, &edges2[j-1]/*, &edges2[Nj-2]*/);

        W[i][j].path_se = PATH_TO_E;
    }

    for(j=2; j<Nj; j++)
    {       
        i=0;//////////
        W[i][j].w_south = W[i][j-1].w_south;
        W[i][j].w_south = W[i][j].w_south + _cvStretchingWork( &null_edge, &edges2[j-1] );
        W[i][j].w_south = W[i][j].w_south /*+ 
            _cvBendingWork( &null_edge, &null_edge, &edges2[j-2], &edges2[j-1], NULL )*/;
        W[i][j].path_s = 3;

        i=1;///////////
        W[i][j].w_east= inf;

        _cvWorkSouth(i, j, W, edges1, edges2);

        W[i][j].w_southeast = W[i-1][j-1].w_south;
        W[i][j].w_southeast = W[i][j].w_southeast + _cvStretchingWork( &edges1[i-1], &edges2[j-1] );

        small_edge.x = NULL_EDGE*edges2[j-2].x;
        small_edge.y = NULL_EDGE*edges2[j-2].y;

        W[i][j].w_southeast = W[i][j].w_southeast + 
            _cvBendingWork( /*&null_edge*/&small_edge, &edges1[i-1], &edges2[j-2], &edges2[j-1]/*, &edges1[Ni-2]*/);
        W[i][j].path_se = 3;
    }

    for(i=2; i<Ni; i++)
        for(j=2; j<Nj; j++)
        {
            _cvWorkEast     (i, j, W, edges1, edges2);
            _cvWorkSouthEast(i, j, W, edges1, edges2);
            _cvWorkSouth    (i, j, W, edges1, edges2);
        }

    i=Ni-1;j=Nj-1;

    *corr = cvCreateSeq(0,                    
                        sizeof(CvSeq),        
                        sizeof(int),
                        storage );

    corr01 = *corr;
    cvStartAppendToSeq( corr01, &writer );
    if( W[i][j].w_east > W[i][j].w_southeast )
        {
            if( W[i][j].w_southeast > W[i][j].w_south )
            {
                path = 3;
            }
            else
            {
                path = PATH_TO_SE;
            }
        }
        else
        {
            if( W[i][j].w_east < W[i][j].w_south )
            {
                path = PATH_TO_E;
            }
            else
            {
                path = 3;
            }
        }
    do
    {
        CV_WRITE_SEQ_ELEM( j, writer );

        switch( path ) 
        {
        case PATH_TO_E:
            path = W[i][j].path_e;
            i--;
            cvFlushSeqWriter( &writer );
            corr01->h_next = cvCreateSeq(   0,                    
                                            sizeof(CvSeq),        
                                            sizeof(int),
                                            storage );
            corr01 = corr01->h_next;
            cvStartAppendToSeq( corr01, &writer );
            break;
        
        case PATH_TO_SE:
            path = W[i][j].path_se;
            j--; i--;
            cvFlushSeqWriter( &writer );
            corr01->h_next = cvCreateSeq(   0,                    
                                            sizeof(CvSeq),        
                                            sizeof(int),
                                            storage );
            corr01 = corr01->h_next;
            cvStartAppendToSeq( corr01, &writer );
            break;

        case 3:
            path = W[i][j].path_s;
            j--;
            break;
        }

    } while( (i>=0) && (j>=0) );
    cvFlushSeqWriter( &writer );

    // Free memory
    for(i=1;i<Ni;i++)
    {
        free(W[i]);
    }
    free(W);
    free(point1);
    free(point2);
    free(edges1);
    free(edges2);
}

