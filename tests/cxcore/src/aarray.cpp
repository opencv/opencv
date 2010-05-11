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

//////////////////////////////////////////////////////////////////////////////////////////
////////////////// tests for arithmetic, logic and statistical functions /////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "cxcoretest.h"
#include <map>
#include <vector>

using namespace cv;
using namespace std;

class CV_ArrayOpTest : public CvTest
{
public:
    CV_ArrayOpTest();
    ~CV_ArrayOpTest();    
protected:
    void run(int);    
};


CV_ArrayOpTest::CV_ArrayOpTest() : CvTest( "nd-array-ops", "?" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}
CV_ArrayOpTest::~CV_ArrayOpTest() {}

static string idx2string(const int* idx, int dims)
{
    char buf[256];
    char* ptr = buf;
    for( int k = 0; k < dims; k++ )
    {
        sprintf(ptr, "%4d ", idx[k]);
        ptr += strlen(ptr);
    }
    ptr[-1] = '\0';
    return string(buf);
}

static const int* string2idx(const string& s, int* idx, int dims)
{
    const char* ptr = s.c_str();
    for( int k = 0; k < dims; k++ )
    {
        int n = 0;
        sscanf(ptr, "%d%n", idx + k, &n);
        ptr += n;
    }
    return idx;
}

static double getValue(SparseMat& M, const int* idx, RNG& rng)
{
    int d = M.dims();
    size_t hv = 0, *phv = 0;
    if( (unsigned)rng % 2 )
    {
        hv = d == 2 ? M.hash(idx[0], idx[1]) :
        d == 3 ? M.hash(idx[0], idx[1], idx[2]) : M.hash(idx);
        phv = &hv;
    }
    
    const uchar* ptr = d == 2 ? M.ptr(idx[0], idx[1], false, phv) :
                       d == 3 ? M.ptr(idx[0], idx[1], idx[2], false, phv) :
                       M.ptr(idx, false, phv);
    return !ptr ? 0 : M.type() == CV_32F ? *(float*)ptr : M.type() == CV_64F ? *(double*)ptr : 0;
}

static double getValue(const CvSparseMat* M, const int* idx)
{
    int type = 0;
    const uchar* ptr = cvPtrND(M, idx, &type, 0);
    return !ptr ? 0 : type == CV_32F ? *(float*)ptr : type == CV_64F ? *(double*)ptr : 0;
}

static void eraseValue(SparseMat& M, const int* idx, RNG& rng)
{
    int d = M.dims();
    size_t hv = 0, *phv = 0;
    if( (unsigned)rng % 2 )
    {
        hv = d == 2 ? M.hash(idx[0], idx[1]) :
             d == 3 ? M.hash(idx[0], idx[1], idx[2]) : M.hash(idx);
        phv = &hv;
    }
    
    if( d == 2 )
        M.erase(idx[0], idx[1], phv);
    else if( d == 3 )
        M.erase(idx[0], idx[1], idx[2], phv);
    else
        M.erase(idx, phv);
}

static void eraseValue(CvSparseMat* M, const int* idx)
{
    cvClearND(M, idx);
}

static void setValue(SparseMat& M, const int* idx, double value, RNG& rng)
{
    int d = M.dims();
    size_t hv = 0, *phv = 0;
    if( (unsigned)rng % 2 )
    {
        hv = d == 2 ? M.hash(idx[0], idx[1]) :
             d == 3 ? M.hash(idx[0], idx[1], idx[2]) : M.hash(idx);
        phv = &hv;
    }
    
    uchar* ptr = d == 2 ? M.ptr(idx[0], idx[1], true, phv) :
                 d == 3 ? M.ptr(idx[0], idx[1], idx[2], true, phv) :
                 M.ptr(idx, true, phv);
    if( M.type() == CV_32F )
        *(float*)ptr = (float)value;
    else if( M.type() == CV_64F )
        *(double*)ptr = value;
    else
        CV_Error(CV_StsUnsupportedFormat, "");
}

void CV_ArrayOpTest::run( int /* start_from */)
{
    int errcount = 0;
    
    // dense matrix operations
    {
    int sz3[] = {5, 10, 15};
    MatND A(3, sz3, CV_32F), B(3, sz3, CV_16SC4);
    CvMatND matA = A, matB = B;
    RNG rng;
    rng.fill(A, CV_RAND_UNI, Scalar::all(-10), Scalar::all(10));
    rng.fill(B, CV_RAND_UNI, Scalar::all(-10), Scalar::all(10));
    
    int idx0[] = {3,4,5}, idx1[] = {0, 9, 7};
    float val0 = 130;
    Scalar val1(-1000, 30, 3, 8);
    cvSetRealND(&matA, idx0, val0);
    cvSetReal3D(&matA, idx1[0], idx1[1], idx1[2], -val0);
    cvSetND(&matB, idx0, val1);
    cvSet3D(&matB, idx1[0], idx1[1], idx1[2], -val1);
    Ptr<CvMatND> matC = cvCloneMatND(&matB);
    
    if( A.at<float>(idx0[0], idx0[1], idx0[2]) != val0 ||
        A.at<float>(idx1[0], idx1[1], idx1[2]) != -val0 ||
        cvGetReal3D(&matA, idx0[0], idx0[1], idx0[2]) != val0 ||
        cvGetRealND(&matA, idx1) != -val0 ||
        
        Scalar(B.at<Vec4s>(idx0[0], idx0[1], idx0[2])) != val1 ||
        Scalar(B.at<Vec4s>(idx1[0], idx1[1], idx1[2])) != -val1 ||
        Scalar(cvGet3D(matC, idx0[0], idx0[1], idx0[2])) != val1 ||
        Scalar(cvGetND(matC, idx1)) != -val1 )
    {
        ts->printf(CvTS::LOG, "one of cvSetReal3D, cvSetRealND, cvSet3D, cvSetND "
                              "or the corresponding *Get* functions is not correct\n");
        errcount++;
    }
    }
    
    RNG rng;
    const int MAX_DIM = 5, MAX_DIM_SZ = 10;
    // sparse matrix operations
    for( int si = 0; si < 10; si++ )
    {
        int depth = (unsigned)rng % 2 == 0 ? CV_32F : CV_64F;
        int dims = ((unsigned)rng % MAX_DIM) + 1;
        int i, k, size[MAX_DIM]={0}, idx[MAX_DIM]={0};
        vector<string> all_idxs;
        vector<double> all_vals;
        vector<double> all_vals2;
        string sidx, min_sidx, max_sidx;
        double min_val=0, max_val=0;
        
        int p = 1;
        for( k = 0; k < dims; k++ )
        {
            size[k] = ((unsigned)rng % MAX_DIM_SZ) + 1;
            p *= size[k];
        }
        SparseMat M( dims, size, depth );
        map<string, double> M0;
        
        int nz0 = (unsigned)rng % max(p/5,10);
        nz0 = min(max(nz0, 1), p);
        all_vals.resize(nz0);
        all_vals2.resize(nz0);
        Mat_<double> _all_vals(all_vals), _all_vals2(all_vals2);
        rng.fill(_all_vals, CV_RAND_UNI, Scalar(-1000), Scalar(1000));
        if( depth == CV_32F )
        {
            Mat _all_vals_f;
            _all_vals.convertTo(_all_vals_f, CV_32F);
            _all_vals_f.convertTo(_all_vals, CV_64F);
        }
        _all_vals.convertTo(_all_vals2, _all_vals2.type(), 2);
        if( depth == CV_32F )
        {
            Mat _all_vals2_f;
            _all_vals2.convertTo(_all_vals2_f, CV_32F);
            _all_vals2_f.convertTo(_all_vals2, CV_64F);
        }

        minMaxLoc(_all_vals, &min_val, &max_val);
        double _norm0 = norm(_all_vals, CV_C);
        double _norm1 = norm(_all_vals, CV_L1);
        double _norm2 = norm(_all_vals, CV_L2);
        
        for( i = 0; i < nz0; i++ )
        {
            for(;;)
            {
                for( k = 0; k < dims; k++ )
                    idx[k] = (unsigned)rng % size[k];
                sidx = idx2string(idx, dims);
                if( M0.count(sidx) == 0 )
                    break;
            }
            all_idxs.push_back(sidx);
            M0[sidx] = all_vals[i];
            if( all_vals[i] == min_val )
                min_sidx = sidx;
            if( all_vals[i] == max_val )
                max_sidx = sidx;
            setValue(M, idx, all_vals[i], rng);
            double v = getValue(M, idx, rng);
            if( v != all_vals[i] )
            {
                ts->printf(CvTS::LOG, "%d. immediately after SparseMat[%s]=%.20g the current value is %.20g\n",
                           i, sidx.c_str(), all_vals[i], v);
                errcount++;
                break;
            }
        }
        
        Ptr<CvSparseMat> M2 = (CvSparseMat*)M;
        MatND Md;
        M.copyTo(Md);
        SparseMat M3; SparseMat(Md).convertTo(M3, Md.type(), 2);
        
        int nz1 = M.nzcount(), nz2 = M3.nzcount();
        double norm0 = norm(M, CV_C);
        double norm1 = norm(M, CV_L1);
        double norm2 = norm(M, CV_L2);
        double eps = depth == CV_32F ? FLT_EPSILON*100 : DBL_EPSILON*1000;
        
        if( nz1 != nz0 || nz2 != nz0)
        {
            errcount++;
            ts->printf(CvTS::LOG, "%d: The number of non-zero elements before/after converting to/from dense matrix is not correct: %d/%d (while it should be %d)\n",
                       si, nz1, nz2, nz0 );
            break;
        }
        
        if( fabs(norm0 - _norm0) > fabs(_norm0)*eps ||
            fabs(norm1 - _norm1) > fabs(_norm1)*eps ||
            fabs(norm2 - _norm2) > fabs(_norm2)*eps )
        {
            errcount++;
            ts->printf(CvTS::LOG, "%d: The norms are different: %.20g/%.20g/%.20g vs %.20g/%.20g/%.20g\n",
                       si, norm0, norm1, norm2, _norm0, _norm1, _norm2 );
            break;
        }
        
        int n = (unsigned)rng % max(p/5,10);
        n = min(max(n, 1), p) + nz0;
        
        for( i = 0; i < n; i++ )
        {
            double val1, val2, val3, val0;
            if(i < nz0)
            {
                sidx = all_idxs[i];
                string2idx(sidx, idx, dims);
                val0 = all_vals[i];
            }
            else
            {
                for( k = 0; k < dims; k++ )
                    idx[k] = (unsigned)rng % size[k];
                sidx = idx2string(idx, dims);
                val0 = M0[sidx];
            }
            val1 = getValue(M, idx, rng);
            val2 = getValue(M2, idx);
            val3 = getValue(M3, idx, rng);
            
            if( val1 != val0 || val2 != val0 || fabs(val3 - val0*2) > fabs(val0*2)*FLT_EPSILON )
            {
                errcount++;
                ts->printf(CvTS::LOG, "SparseMat M[%s] = %g/%g/%g (while it should be %g)\n", sidx.c_str(), val1, val2, val3, val0 );
                break;
            }
        }
        
        for( i = 0; i < n; i++ )
        {
            double val1, val2;
            if(i < nz0)
            {
                sidx = all_idxs[i];
                string2idx(sidx, idx, dims);
            }
            else
            {
                for( k = 0; k < dims; k++ )
                    idx[k] = (unsigned)rng % size[k];
                sidx = idx2string(idx, dims);
            }
            eraseValue(M, idx, rng);
            eraseValue(M2, idx);
            val1 = getValue(M, idx, rng);
            val2 = getValue(M2, idx);
            if( val1 != 0 || val2 != 0 )
            {
                errcount++;
                ts->printf(CvTS::LOG, "SparseMat: after deleting M[%s], it is =%g/%g (while it should be 0)\n", sidx.c_str(), val1, val2 );
                break;
            }    
        }
        
        int nz = M.nzcount();
        if( nz != 0 )
        {
            errcount++;
            ts->printf(CvTS::LOG, "The number of non-zero elements after removing all the elements = %d (while it should be 0)\n", nz );
            break;
        }
        
        int idx1[MAX_DIM], idx2[MAX_DIM];
        double val1 = 0, val2 = 0;
        M3 = SparseMat(Md);
        minMaxLoc(M3, &val1, &val2, idx1, idx2);
        string s1 = idx2string(idx1, dims), s2 = idx2string(idx2, dims);
        if( val1 != min_val || val2 != max_val || s1 != min_sidx || s2 != max_sidx )
        {
            errcount++;
            ts->printf(CvTS::LOG, "%d. Sparse: The value and positions of minimum/maximum elements are different from the reference values and positions:\n\t"
                    "(%g, %g, %s, %s) vs (%g, %g, %s, %s)\n", si, val1, val2, s1.c_str(), s2.c_str(),
                       min_val, max_val, min_sidx.c_str(), max_sidx.c_str());
            break;
        }
        
        minMaxLoc(Md, &val1, &val2, idx1, idx2);
        s1 = idx2string(idx1, dims), s2 = idx2string(idx2, dims);
        if( (min_val < 0 && (val1 != min_val || s1 != min_sidx)) ||
            (max_val > 0 && (val2 != max_val || s2 != max_sidx)) )
        {
            errcount++;
            ts->printf(CvTS::LOG, "%d. Dense: The value and positions of minimum/maximum elements are different from the reference values and positions:\n\t"
                    "(%g, %g, %s, %s) vs (%g, %g, %s, %s)\n", si, val1, val2, s1.c_str(), s2.c_str(),
                       min_val, max_val, min_sidx.c_str(), max_sidx.c_str());
            break;
        }
    }
    
    ts->set_failed_test_info(errcount == 0 ? CvTS::OK : CvTS::FAIL_INVALID_OUTPUT);
}

CV_ArrayOpTest cv_ArrayOp_test;

