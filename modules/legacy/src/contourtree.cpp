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

#define _CV_BINTREE_LIST()                                          \
struct _CvTrianAttr* prev_v;   /* pointer to the parent  element on the previous level of the tree  */    \
struct _CvTrianAttr* next_v1;   /* pointer to the child  element on the next level of the tree  */        \
struct _CvTrianAttr* next_v2;   /* pointer to the child  element on the next level of the tree  */        

typedef struct _CvTrianAttr
{
    CvPoint pt;    /* Coordinates x and y of the vertex  which don't lie on the base line LMIAT  */
    char sign;             /*  sign of the triangle   */
    double area;       /*   area of the triangle    */
    double r1;   /*  The ratio of the height of triangle to the base of the triangle  */
    double r2;  /*   The ratio of the projection of the left side of the triangle on the base to the base */
    _CV_BINTREE_LIST()    /* structure double list   */
}
_CvTrianAttr;

#define CV_MATCH_CHECK( status, cvFun )                                    \
  {                                                                        \
    status = cvFun;                                                        \
    if( status != CV_OK )                                                  \
     goto M_END;                                                           \
  }

static CvStatus
icvCalcTriAttr( const CvSeq * contour, CvPoint t2, CvPoint t1, int n1,
                CvPoint t3, int n3, double *s, double *s_c,
                double *h, double *a, double *b );

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCreateContourTree
//    Purpose:
//    Create binary tree representation for the contour 
//    Context:
//    Parameters:
//      contour - pointer to input contour object.
//      storage - pointer to the current storage block
//      tree   -  output pointer to the binary tree representation 
//      threshold - threshold for the binary tree building 
//
//F*/
static CvStatus
icvCreateContourTree( const CvSeq * contour, CvMemStorage * storage,
                      CvContourTree ** tree, double threshold )
{
    CvPoint *pt_p;              /*  pointer to previos points   */
    CvPoint *pt_n;              /*  pointer to next points      */
    CvPoint *pt1, *pt2;         /*  pointer to current points   */

    CvPoint t, tp1, tp2, tp3, tn1, tn2, tn3;
    int lpt, flag, i, j, i_tree, j_1, j_3, i_buf;
    double s, sp1, sp2, sn1, sn2, s_c, sp1_c, sp2_c, sn1_c, sn2_c, h, hp1, hp2, hn1, hn2,
        a, ap1, ap2, an1, an2, b, bp1, bp2, bn1, bn2;
    double a_s_c, a_sp1_c;

    _CvTrianAttr **ptr_p, **ptr_n, **ptr1, **ptr2;      /*  pointers to pointers of triangles  */
    _CvTrianAttr *cur_adr;

    int *num_p, *num_n, *num1, *num2;   /*   numbers of input contour points   */
    int nm, nmp1, nmp2, nmp3, nmn1, nmn2, nmn3;
    int seq_flags = 1, i_end, prev_null, prev2_null;
    double koef = 1.5;
    double eps = 1.e-7;
    double e;
    CvStatus status;
    int hearder_size;
    _CvTrianAttr tree_one, tree_two, *tree_end, *tree_root;

    CvSeqWriter writer;

    assert( contour != NULL && contour->total >= 4 );
    status = CV_OK;

    if( contour == NULL )
        return CV_NULLPTR_ERR;
    if( contour->total < 4 )
        return CV_BADSIZE_ERR;

    if( !CV_IS_SEQ_POINT_SET( contour ))
        return CV_BADFLAG_ERR;


/*   Convert Sequence to array    */
    lpt = contour->total;
    pt_p = pt_n = NULL;
    num_p = num_n = NULL;
    ptr_p = ptr_n = ptr1 = ptr2 = NULL;
    tree_end = NULL;

    pt_p = (CvPoint *) cvAlloc( lpt * sizeof( CvPoint ));
    pt_n = (CvPoint *) cvAlloc( lpt * sizeof( CvPoint ));

    num_p = (int *) cvAlloc( lpt * sizeof( int ));
    num_n = (int *) cvAlloc( lpt * sizeof( int ));

    hearder_size = sizeof( CvContourTree );
    seq_flags = CV_SEQ_POLYGON_TREE;
    cvStartWriteSeq( seq_flags, hearder_size, sizeof( _CvTrianAttr ), storage, &writer );

    ptr_p = (_CvTrianAttr **) cvAlloc( lpt * sizeof( _CvTrianAttr * ));
    ptr_n = (_CvTrianAttr **) cvAlloc( lpt * sizeof( _CvTrianAttr * ));

    memset( ptr_p, 0, lpt * sizeof( _CvTrianAttr * ));
    memset( ptr_n, 0, lpt * sizeof( _CvTrianAttr * ));

    if( pt_p == NULL || pt_n == NULL )
        return CV_OUTOFMEM_ERR;
    if( ptr_p == NULL || ptr_n == NULL )
        return CV_OUTOFMEM_ERR;

/*     write fild for the binary tree root   */
/*  start_writer = writer;   */

    tree_one.pt.x = tree_one.pt.y = 0;
    tree_one.sign = 0;
    tree_one.area = 0;
    tree_one.r1 = tree_one.r2 = 0;
    tree_one.next_v1 = tree_one.next_v2 = tree_one.prev_v = NULL;

    CV_WRITE_SEQ_ELEM( tree_one, writer );
    tree_root = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

    if( cvCvtSeqToArray( contour, (char *) pt_p ) == (char *) contour )
        return CV_BADPOINT_ERR;

    for( i = 0; i < lpt; i++ )
        num_p[i] = i;

    i = lpt;
    flag = 0;
    i_tree = 0;
    e = 20.;                    /*  initial threshold value   */
    ptr1 = ptr_p;
    ptr2 = ptr_n;
    pt1 = pt_p;
    pt2 = pt_n;
    num1 = num_p;
    num2 = num_n;
/*  binary tree constraction    */
    while( i > 4 )
    {
        if( flag == 0 )
        {
            ptr1 = ptr_p;
            ptr2 = ptr_n;
            pt1 = pt_p;
            pt2 = pt_n;
            num1 = num_p;
            num2 = num_n;
            flag = 1;
        }
        else
        {
            ptr1 = ptr_n;
            ptr2 = ptr_p;
            pt1 = pt_n;
            pt2 = pt_p;
            num1 = num_n;
            num2 = num_p;
            flag = 0;
        }
        t = pt1[0];
        nm = num1[0];
        tp1 = pt1[i - 1];
        nmp1 = num1[i - 1];
        tp2 = pt1[i - 2];
        nmp2 = num1[i - 2];
        tp3 = pt1[i - 3];
        nmp3 = num1[i - 3];
        tn1 = pt1[1];
        nmn1 = num1[1];
        tn2 = pt1[2];
        nmn2 = num1[2];

        i_buf = 0;
        i_end = -1;
        CV_MATCH_CHECK( status,
                        icvCalcTriAttr( contour, t, tp1, nmp1, tn1, nmn1, &s, &s_c, &h, &a,
                                        &b ));
        CV_MATCH_CHECK( status,
                        icvCalcTriAttr( contour, tp1, tp2, nmp2, t, nm, &sp1, &sp1_c, &hp1,
                                        &ap1, &bp1 ));
        CV_MATCH_CHECK( status,
                        icvCalcTriAttr( contour, tp2, tp3, nmp3, tp1, nmp1, &sp2, &sp2_c, &hp2,
                                        &ap2, &bp2 ));
        CV_MATCH_CHECK( status,
                        icvCalcTriAttr( contour, tn1, t, nm, tn2, nmn2, &sn1, &sn1_c, &hn1,
                                        &an1, &bn1 ));


        j_3 = 3;
        prev_null = prev2_null = 0;
        for( j = 0; j < i; j++ )
        {
            tn3 = pt1[j_3];
            nmn3 = num1[j_3];
            if( j == 0 )
                j_1 = i - 1;
            else
                j_1 = j - 1;

            CV_MATCH_CHECK( status, icvCalcTriAttr( contour, tn2, tn1, nmn1, tn3, nmn3,
                                                    &sn2, &sn2_c, &hn2, &an2, &bn2 ));

            if( (s_c < sp1_c && s_c < sp2_c && s_c <= sn1_c && s_c <= sn2_c && s_c < e) ||
                (((s_c == sp1_c && s_c <= sp2_c) || (s_c == sp2_c && s_c <= sp1_c)) &&
                s_c <= sn1_c && s_c <= sn2_c && s_c < e && j > 1 && prev2_null == 0) ||
                (s_c < eps && j > 0 && prev_null == 0) )
            {
                prev_null = prev2_null = 1;
                if( s_c < threshold )
                {
                    if( ptr1[j_1] == NULL && ptr1[j] == NULL )
                    {
                        if( i_buf > 0 )
                            ptr2[i_buf - 1] = NULL;
                        else
                            i_end = 0;
                    }
                    else
                    {
/*   form next vertex  */
                        tree_one.pt = t;
                        tree_one.sign = (char) (CV_SIGN( s ));
                        tree_one.r1 = h / a;
                        tree_one.r2 = b / a;
                        tree_one.area = fabs( s );
                        tree_one.next_v1 = ptr1[j_1];
                        tree_one.next_v2 = ptr1[j];

                        CV_WRITE_SEQ_ELEM( tree_one, writer );
                        cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

                        if( ptr1[j_1] != NULL )
                            ptr1[j_1]->prev_v = cur_adr;
                        if( ptr1[j] != NULL )
                            ptr1[j]->prev_v = cur_adr;

                        if( i_buf > 0 )
                            ptr2[i_buf - 1] = cur_adr;
                        else
                        {
                            tree_end = (_CvTrianAttr *) writer.ptr;
                            i_end = 1;
                        }
                        i_tree++;
                    }
                }
                else
/*   form next vertex    */
                {
                    tree_one.pt = t;
                    tree_one.sign = (char) (CV_SIGN( s ));
                    tree_one.area = fabs( s );
                    tree_one.r1 = h / a;
                    tree_one.r2 = b / a;
                    tree_one.next_v1 = ptr1[j_1];
                    tree_one.next_v2 = ptr1[j];

                    CV_WRITE_SEQ_ELEM( tree_one, writer );
                    cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

                    if( ptr1[j_1] != NULL )
                        ptr1[j_1]->prev_v = cur_adr;
                    if( ptr1[j] != NULL )
                        ptr1[j]->prev_v = cur_adr;

                    if( i_buf > 0 )
                        ptr2[i_buf - 1] = cur_adr;
                    else
                    {
                        tree_end = cur_adr;
                        i_end = 1;
                    }
                    i_tree++;
                }
            }
            else
/*   the current triangle is'not LMIAT    */
            {
                prev_null = 0;
                switch (prev2_null)
                {
                case 0:
                    break;
                case 1:
                    {
                        prev2_null = 2;
                        break;
                    }
                case 2:
                    {
                        prev2_null = 0;
                        break;
                    }
                }
                if( j != i - 1 || i_end == -1 )
                    ptr2[i_buf] = ptr1[j];
                else if( i_end == 0 )
                    ptr2[i_buf] = NULL;
                else
                    ptr2[i_buf] = tree_end;
                pt2[i_buf] = t;
                num2[i_buf] = num1[j];
                i_buf++;
            }
/*    go to next vertex    */
            tp3 = tp2;
            tp2 = tp1;
            tp1 = t;
            t = tn1;
            tn1 = tn2;
            tn2 = tn3;
            nmp3 = nmp2;
            nmp2 = nmp1;
            nmp1 = nm;
            nm = nmn1;
            nmn1 = nmn2;
            nmn2 = nmn3;

            sp2 = sp1;
            sp1 = s;
            s = sn1;
            sn1 = sn2;
            sp2_c = sp1_c;
            sp1_c = s_c;
            s_c = sn1_c;
            sn1_c = sn2_c;

            ap2 = ap1;
            ap1 = a;
            a = an1;
            an1 = an2;
            bp2 = bp1;
            bp1 = b;
            b = bn1;
            bn1 = bn2;
            hp2 = hp1;
            hp1 = h;
            h = hn1;
            hn1 = hn2;
            j_3++;
            if( j_3 >= i )
                j_3 = 0;
        }

        i = i_buf;
        e = e * koef;
    }

/*  constract tree root  */
    if( i != 4 )
        return CV_BADFACTOR_ERR;

    t = pt2[0];
    tn1 = pt2[1];
    tn2 = pt2[2];
    tp1 = pt2[3];
    nm = num2[0];
    nmn1 = num2[1];
    nmn2 = num2[2];
    nmp1 = num2[3];
/*   first pair of the triangles   */
    CV_MATCH_CHECK( status,
                    icvCalcTriAttr( contour, t, tp1, nmp1, tn1, nmn1, &s, &s_c, &h, &a, &b ));
    CV_MATCH_CHECK( status,
                    icvCalcTriAttr( contour, tn2, tn1, nmn1, tp1, nmp1, &sn2, &sn2_c, &hn2,
                                    &an2, &bn2 ));
/*   second pair of the triangles   */
    CV_MATCH_CHECK( status,
                    icvCalcTriAttr( contour, tn1, t, nm, tn2, nmn2, &sn1, &sn1_c, &hn1, &an1,
                                    &bn1 ));
    CV_MATCH_CHECK( status,
                    icvCalcTriAttr( contour, tp1, tn2, nmn2, t, nm, &sp1, &sp1_c, &hp1, &ap1,
                                    &bp1 ));

    a_s_c = fabs( s_c - sn2_c );
    a_sp1_c = fabs( sp1_c - sn1_c );

    if( a_s_c > a_sp1_c )
/*   form child vertexs for the root     */
    {
        tree_one.pt = t;
        tree_one.sign = (char) (CV_SIGN( s ));
        tree_one.area = fabs( s );
        tree_one.r1 = h / a;
        tree_one.r2 = b / a;
        tree_one.next_v1 = ptr2[3];
        tree_one.next_v2 = ptr2[0];

        tree_two.pt = tn2;
        tree_two.sign = (char) (CV_SIGN( sn2 ));
        tree_two.area = fabs( sn2 );
        tree_two.r1 = hn2 / an2;
        tree_two.r2 = bn2 / an2;
        tree_two.next_v1 = ptr2[1];
        tree_two.next_v2 = ptr2[2];

        CV_WRITE_SEQ_ELEM( tree_one, writer );
        cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

        if( s_c > sn2_c )
        {
            if( ptr2[3] != NULL )
                ptr2[3]->prev_v = cur_adr;
            if( ptr2[0] != NULL )
                ptr2[0]->prev_v = cur_adr;
            ptr1[0] = cur_adr;

            i_tree++;

            CV_WRITE_SEQ_ELEM( tree_two, writer );
            cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

            if( ptr2[1] != NULL )
                ptr2[1]->prev_v = cur_adr;
            if( ptr2[2] != NULL )
                ptr2[2]->prev_v = cur_adr;
            ptr1[1] = cur_adr;

            i_tree++;

            pt1[0] = tp1;
            pt1[1] = tn1;
        }
        else
        {
            CV_WRITE_SEQ_ELEM( tree_two, writer );
            cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

            if( ptr2[1] != NULL )
                ptr2[1]->prev_v = cur_adr;
            if( ptr2[2] != NULL )
                ptr2[2]->prev_v = cur_adr;
            ptr1[0] = cur_adr;

            i_tree++;

            CV_WRITE_SEQ_ELEM( tree_one, writer );
            cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

            if( ptr2[3] != NULL )
                ptr2[3]->prev_v = cur_adr;
            if( ptr2[0] != NULL )
                ptr2[0]->prev_v = cur_adr;
            ptr1[1] = cur_adr;

            i_tree++;

            pt1[0] = tn1;
            pt1[1] = tp1;
        }
    }
    else
    {
        tree_one.pt = tp1;
        tree_one.sign = (char) (CV_SIGN( sp1 ));
        tree_one.area = fabs( sp1 );
        tree_one.r1 = hp1 / ap1;
        tree_one.r2 = bp1 / ap1;
        tree_one.next_v1 = ptr2[2];
        tree_one.next_v2 = ptr2[3];

        tree_two.pt = tn1;
        tree_two.sign = (char) (CV_SIGN( sn1 ));
        tree_two.area = fabs( sn1 );
        tree_two.r1 = hn1 / an1;
        tree_two.r2 = bn1 / an1;
        tree_two.next_v1 = ptr2[0];
        tree_two.next_v2 = ptr2[1];

        CV_WRITE_SEQ_ELEM( tree_one, writer );
        cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

        if( sp1_c > sn1_c )
        {
            if( ptr2[2] != NULL )
                ptr2[2]->prev_v = cur_adr;
            if( ptr2[3] != NULL )
                ptr2[3]->prev_v = cur_adr;
            ptr1[0] = cur_adr;

            i_tree++;

            CV_WRITE_SEQ_ELEM( tree_two, writer );
            cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

            if( ptr2[0] != NULL )
                ptr2[0]->prev_v = cur_adr;
            if( ptr2[1] != NULL )
                ptr2[1]->prev_v = cur_adr;
            ptr1[1] = cur_adr;

            i_tree++;

            pt1[0] = tn2;
            pt1[1] = t;
        }
        else
        {
            CV_WRITE_SEQ_ELEM( tree_two, writer );
            cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

            if( ptr2[0] != NULL )
                ptr2[0]->prev_v = cur_adr;
            if( ptr2[1] != NULL )
                ptr2[1]->prev_v = cur_adr;
            ptr1[0] = cur_adr;

            i_tree++;

            CV_WRITE_SEQ_ELEM( tree_one, writer );
            cur_adr = (_CvTrianAttr *) (writer.ptr - writer.seq->elem_size);

            if( ptr2[2] != NULL )
                ptr2[2]->prev_v = cur_adr;
            if( ptr2[3] != NULL )
                ptr2[3]->prev_v = cur_adr;
            ptr1[1] = cur_adr;

            i_tree++;

            pt1[0] = t;
            pt1[1] = tn2;

        }
    }

/*    form root   */
    s = cvContourArea( contour );

    tree_root->pt = pt1[1];
    tree_root->sign = 0;
    tree_root->area = fabs( s );
    tree_root->r1 = 0;
    tree_root->r2 = 0;
    tree_root->next_v1 = ptr1[0];
    tree_root->next_v2 = ptr1[1];
    tree_root->prev_v = NULL;

    ptr1[0]->prev_v = (_CvTrianAttr *) tree_root;
    ptr1[1]->prev_v = (_CvTrianAttr *) tree_root;

/*     write binary tree root   */
/*    CV_WRITE_SEQ_ELEM (tree_one, start_writer);   */
    i_tree++;
/*  create Sequence hearder     */
    *((CvSeq **) tree) = cvEndWriteSeq( &writer );
/*   write points for the main segment into sequence header   */
    (*tree)->p1 = pt1[0];

  M_END:

    cvFree( &ptr_n );
    cvFree( &ptr_p );
    cvFree( &num_n );
    cvFree( &num_p );
    cvFree( &pt_n );
    cvFree( &pt_p );

    return status;
}

/****************************************************************************************\

 triangle attributes calculations 

\****************************************************************************************/
static CvStatus
icvCalcTriAttr( const CvSeq * contour, CvPoint t2, CvPoint t1, int n1,
                CvPoint t3, int n3, double *s, double *s_c,
                double *h, double *a, double *b )
{
    double x13, y13, x12, y12, l_base, nx, ny, qq;
    double eps = 1.e-5;

    x13 = t3.x - t1.x;
    y13 = t3.y - t1.y;
    x12 = t2.x - t1.x;
    y12 = t2.y - t1.y;
    qq = x13 * x13 + y13 * y13;
    l_base = cvSqrt( (float) (qq) );
    if( l_base > eps )
    {
        nx = y13 / l_base;
        ny = -x13 / l_base;

        *h = nx * x12 + ny * y12;

        *s = (*h) * l_base / 2.;

        *b = nx * y12 - ny * x12;

        *a = l_base;
/*   calculate interceptive area   */
        *s_c = cvContourArea( contour, cvSlice(n1, n3+1));
    }
    else
    {
        *h = 0;
        *s = 0;
        *s_c = 0;
        *b = 0;
        *a = 0;
    }

    return CV_OK;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvCreateContourTree
//    Purpose:
//    Create binary tree representation for the contour 
//    Context:
//    Parameters:
//      contour - pointer to input contour object.
//      storage - pointer to the current storage block
//      tree   -  output pointer to the binary tree representation 
//      threshold - threshold for the binary tree building 
//
//F*/
CV_IMPL CvContourTree*
cvCreateContourTree( const CvSeq* contour, CvMemStorage* storage, double threshold )
{
    CvContourTree* tree = 0;
    
    IPPI_CALL( icvCreateContourTree( contour, storage, &tree, threshold ));

    return tree;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvContourFromContourTree
//    Purpose:
//    reconstracts contour from binary tree representation  
//    Context:
//    Parameters:
//      tree   -  pointer to the input binary tree representation 
//      storage - pointer to the current storage block
//      contour - pointer to output contour object.
//      criteria - criteria for the definition threshold value
//                 for the contour reconstracting (level or precision)
//F*/
CV_IMPL CvSeq*
cvContourFromContourTree( const CvContourTree*  tree,
                          CvMemStorage*  storage,
                          CvTermCriteria  criteria )
{
    CvSeq* contour = 0;
    cv::AutoBuffer<_CvTrianAttr*> ptr_buf;     /*  pointer to the pointer's buffer  */
    cv::AutoBuffer<int> level_buf;
    int i_buf;

    int lpt;
    double area_all;
    double threshold;
    int cur_level;
    int level;
    int seq_flags;
    char log_iter, log_eps;
    int out_hearder_size;
    _CvTrianAttr *tree_one = 0, tree_root;  /*  current vertex  */

    CvSeqReader reader;
    CvSeqWriter writer;

    if( !tree )
        CV_Error( CV_StsNullPtr, "" );

    if( !CV_IS_SEQ_POLYGON_TREE( tree ))
        CV_Error( CV_StsBadArg, "" );

    criteria = cvCheckTermCriteria( criteria, 0., 100 );

    lpt = tree->total;
    ptr_buf = NULL;
    level_buf = NULL;
    i_buf = 0;
    cur_level = 0;
    log_iter = (char) (criteria.type == CV_TERMCRIT_ITER ||
                       (criteria.type == CV_TERMCRIT_ITER + CV_TERMCRIT_EPS));
    log_eps = (char) (criteria.type == CV_TERMCRIT_EPS ||
                      (criteria.type == CV_TERMCRIT_ITER + CV_TERMCRIT_EPS));

    cvStartReadSeq( (CvSeq *) tree, &reader, 0 );

    out_hearder_size = sizeof( CvContour );

    seq_flags = CV_SEQ_POLYGON;
    cvStartWriteSeq( seq_flags, out_hearder_size, sizeof( CvPoint ), storage, &writer );

    ptr_buf.allocate(lpt);
    if( log_iter )
        level_buf.allocate(lpt);

    memset( ptr_buf, 0, lpt * sizeof( _CvTrianAttr * ));

/*     write the first tree root's point as a start point of the result contour  */
    CV_WRITE_SEQ_ELEM( tree->p1, writer );
/*     write the second tree root"s point into buffer    */

/*     read the root of the tree   */
    CV_READ_SEQ_ELEM( tree_root, reader );

    tree_one = &tree_root;
    area_all = tree_one->area;

    if( log_eps )
        threshold = criteria.epsilon * area_all;
    else
        threshold = 10 * area_all;

    if( log_iter )
        level = criteria.max_iter;
    else
        level = -1;

/*  contour from binary tree constraction    */
    while( i_buf >= 0 )
    {
        if( tree_one != NULL && (cur_level <= level || tree_one->area >= threshold) )
/*   go to left sub tree for the vertex and save pointer to the right vertex   */
/*   into the buffer     */
        {
            ptr_buf[i_buf] = tree_one;
            if( log_iter )
            {
                level_buf[i_buf] = cur_level;
                cur_level++;
            }
            i_buf++;
            tree_one = tree_one->next_v1;
        }
        else
        {
            i_buf--;
            if( i_buf >= 0 )
            {
                CvPoint pt = ptr_buf[i_buf]->pt;
                CV_WRITE_SEQ_ELEM( pt, writer );
                tree_one = ptr_buf[i_buf]->next_v2;
                if( log_iter )
                {
                    cur_level = level_buf[i_buf] + 1;
                }
            }
        }
    }

    contour = cvEndWriteSeq( &writer );
    cvBoundingRect( contour, 1 );

    return contour;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvMatchContourTrees
//    Purpose:
//      Calculates matching of the two contour trees
//    Context:
//    Parameters:
//      tree1 - pointer to the first input contour tree object.
//      tree2 - pointer to the second input contour tree object.
//      method - method for the matching calculation
//      (now CV_CONTOUR_TREES_MATCH_I1 only  )
//      threshold - threshold for the contour trees matching 
//      result - output calculated measure 
//F*/
CV_IMPL  double
cvMatchContourTrees( const CvContourTree* tree1, const CvContourTree* tree2,
                     int method, double threshold )
{
    cv::AutoBuffer<_CvTrianAttr*> buf;
    _CvTrianAttr **ptr_p1 = 0, **ptr_p2 = 0;    /*pointers to the pointer's buffer */
    _CvTrianAttr **ptr_n1 = 0, **ptr_n2 = 0;    /*pointers to the pointer's buffer */
    _CvTrianAttr **ptr11, **ptr12, **ptr21, **ptr22;

    int lpt1, lpt2, lpt, flag, flag_n, i, j, ibuf, ibuf1;
    double match_v, d12, area1, area2, r11, r12, r21, r22, w1, w2;
    double eps = 1.e-5;
    char s1, s2;
    _CvTrianAttr tree_1, tree_2;        /*current vertex 1 and 2 tree */
    CvSeqReader reader1, reader2;

    if( !tree1 || !tree2 )
        CV_Error( CV_StsNullPtr, "" );

    if( method != CV_CONTOUR_TREES_MATCH_I1 )
        CV_Error( CV_StsBadArg, "Unknown/unsupported comparison method" );

    if( !CV_IS_SEQ_POLYGON_TREE( tree1 ))
        CV_Error( CV_StsBadArg, "The first argument is not a valid contour tree" );

    if( !CV_IS_SEQ_POLYGON_TREE( tree2 ))
        CV_Error( CV_StsBadArg, "The second argument is not a valid contour tree" );

    lpt1 = tree1->total;
    lpt2 = tree2->total;
    lpt = lpt1 > lpt2 ? lpt1 : lpt2;

    ptr_p1 = ptr_n1 = ptr_p2 = ptr_n2 = NULL;
    buf.allocate(lpt*4);
    ptr_p1 = buf;
    ptr_p2 = ptr_p1 + lpt;
    ptr_n1 = ptr_p2 + lpt;
    ptr_n2 = ptr_n1 + lpt;

    cvStartReadSeq( (CvSeq *) tree1, &reader1, 0 );
    cvStartReadSeq( (CvSeq *) tree2, &reader2, 0 );

/*read the root of the first and second tree*/
    CV_READ_SEQ_ELEM( tree_1, reader1 );
    CV_READ_SEQ_ELEM( tree_2, reader2 );

/*write to buffer pointers to root's childs vertexs*/
    ptr_p1[0] = tree_1.next_v1;
    ptr_p1[1] = tree_1.next_v2;
    ptr_p2[0] = tree_2.next_v1;
    ptr_p2[1] = tree_2.next_v2;
    i = 2;
    match_v = 0.;
    area1 = tree_1.area;
    area2 = tree_2.area;

    if( area1 < eps || area2 < eps || lpt < 4 )
        CV_Error( CV_StsBadSize, "" );

    r11 = r12 = r21 = r22 = w1 = w2 = d12 = 0;
    flag = 0;
    s1 = s2 = 0;
    do
    {
        if( flag == 0 )
        {
            ptr11 = ptr_p1;
            ptr12 = ptr_n1;
            ptr21 = ptr_p2;
            ptr22 = ptr_n2;
            flag = 1;
        }
        else
        {
            ptr11 = ptr_n1;
            ptr12 = ptr_p1;
            ptr21 = ptr_n2;
            ptr22 = ptr_p2;
            flag = 0;
        }
        ibuf = 0;
        for( j = 0; j < i; j++ )
        {
            flag_n = 0;
            if( ptr11[j] != NULL )
            {
                r11 = ptr11[j]->r1;
                r12 = ptr11[j]->r2;
                flag_n = 1;
                w1 = ptr11[j]->area / area1;
                s1 = ptr11[j]->sign;
            }
            else
            {
                r11 = r21 = 0;
            }
            if( ptr21[j] != NULL )
            {
                r21 = ptr21[j]->r1;
                r22 = ptr21[j]->r2;
                flag_n = 1;
                w2 = ptr21[j]->area / area2;
                s2 = ptr21[j]->sign;
            }
            else
            {
                r21 = r22 = 0;
            }
            if( flag_n != 0 )
/* calculate node distance */
            {
                switch (method)
                {
                case 1:
                    {
                        double t0, t1;
                        if( s1 != s2 )
                        {
                            t0 = fabs( r11 * w1 + r21 * w2 );
                            t1 = fabs( r12 * w1 + r22 * w2 );
                        }
                        else
                        {
                            t0 = fabs( r11 * w1 - r21 * w2 );
                            t1 = fabs( r12 * w1 - r22 * w2 );
                        }
                        d12 = t0 + t1;
                        break;
                    }
                }
                match_v += d12;
                ibuf1 = ibuf + 1;
/*write to buffer the pointer to child vertexes*/
                if( ptr11[j] != NULL )
                {
                    ptr12[ibuf] = ptr11[j]->next_v1;
                    ptr12[ibuf1] = ptr11[j]->next_v2;
                }
                else
                {
                    ptr12[ibuf] = NULL;
                    ptr12[ibuf1] = NULL;
                }
                if( ptr21[j] != NULL )
                {
                    ptr22[ibuf] = ptr21[j]->next_v1;
                    ptr22[ibuf1] = ptr21[j]->next_v2;
                }
                else
                {
                    ptr22[ibuf] = NULL;
                    ptr22[ibuf1] = NULL;
                }
                ibuf += 2;
            }
        }
        i = ibuf;
    }
    while( i > 0 && match_v < threshold );

    return match_v;
}
