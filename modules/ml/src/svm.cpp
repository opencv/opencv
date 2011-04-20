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

/****************************************************************************************\
                                COPYRIGHT NOTICE
                                ----------------

  The code has been derived from libsvm library (version 2.6)
  (http://www.csie.ntu.edu.tw/~cjlin/libsvm).

  Here is the orignal copyright:
------------------------------------------------------------------------------------------
    Copyright (c) 2000-2003 Chih-Chung Chang and Chih-Jen Lin
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither name of copyright holders nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\****************************************************************************************/

using namespace cv;

#define CV_SVM_MIN_CACHE_SIZE  (40 << 20)  /* 40Mb */

#include <stdarg.h>
#include <ctype.h>

#if _MSC_VER >= 1200
#pragma warning( disable: 4514 ) /* unreferenced inline functions */
#endif

#if 1
typedef float Qfloat;
#define QFLOAT_TYPE CV_32F
#else
typedef double Qfloat;
#define QFLOAT_TYPE CV_64F
#endif

// Param Grid
bool CvParamGrid::check() const
{
    bool ok = false;

    CV_FUNCNAME( "CvParamGrid::check" );
    __BEGIN__;

    if( min_val > max_val )
        CV_ERROR( CV_StsBadArg, "Lower bound of the grid must be less then the upper one" );
    if( min_val < DBL_EPSILON )
        CV_ERROR( CV_StsBadArg, "Lower bound of the grid must be positive" );
    if( step < 1. + FLT_EPSILON )
        CV_ERROR( CV_StsBadArg, "Grid step must greater then 1" );

    ok = true;

    __END__;

    return ok;
}

CvParamGrid CvSVM::get_default_grid( int param_id )
{
    CvParamGrid grid;
    if( param_id == CvSVM::C )
    {
        grid.min_val = 0.1;
        grid.max_val = 500;
        grid.step = 5; // total iterations = 5
    }
    else if( param_id == CvSVM::GAMMA )
    {
        grid.min_val = 1e-5;
        grid.max_val = 0.6;
        grid.step = 15; // total iterations = 4
    }
    else if( param_id == CvSVM::P )
    {
        grid.min_val = 0.01;
        grid.max_val = 100;
        grid.step = 7; // total iterations = 4
    }
    else if( param_id == CvSVM::NU )
    {
        grid.min_val = 0.01;
        grid.max_val = 0.2;
        grid.step = 3; // total iterations = 3
    }
    else if( param_id == CvSVM::COEF )
    {
        grid.min_val = 0.1;
        grid.max_val = 300;
        grid.step = 14; // total iterations = 3
    }
    else if( param_id == CvSVM::DEGREE )
    {
        grid.min_val = 0.01;
        grid.max_val = 4;
        grid.step = 7; // total iterations = 3
    }
    else
        cvError( CV_StsBadArg, "CvSVM::get_default_grid", "Invalid type of parameter "
            "(use one of CvSVM::C, CvSVM::GAMMA et al.)", __FILE__, __LINE__ );
    return grid;
}

// SVM training parameters
CvSVMParams::CvSVMParams() :
    svm_type(CvSVM::C_SVC), kernel_type(CvSVM::RBF), degree(0),
    gamma(1), coef0(0), C(1), nu(0), p(0), class_weights(0)
{
    term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
}


CvSVMParams::CvSVMParams( int _svm_type, int _kernel_type,
    double _degree, double _gamma, double _coef0,
    double _Con, double _nu, double _p,
    CvMat* _class_weights, CvTermCriteria _term_crit ) :
    svm_type(_svm_type), kernel_type(_kernel_type),
    degree(_degree), gamma(_gamma), coef0(_coef0),
    C(_Con), nu(_nu), p(_p), class_weights(_class_weights), term_crit(_term_crit)
{
}


/////////////////////////////////////// SVM kernel ///////////////////////////////////////

CvSVMKernel::CvSVMKernel()
{
    clear();
}


void CvSVMKernel::clear()
{
    params = 0;
    calc_func = 0;
}


CvSVMKernel::~CvSVMKernel()
{
}


CvSVMKernel::CvSVMKernel( const CvSVMParams* _params, Calc _calc_func )
{
    clear();
    create( _params, _calc_func );
}


bool CvSVMKernel::create( const CvSVMParams* _params, Calc _calc_func )
{
    clear();
    params = _params;
    calc_func = _calc_func;

    if( !calc_func )
        calc_func = params->kernel_type == CvSVM::RBF ? &CvSVMKernel::calc_rbf :
                    params->kernel_type == CvSVM::POLY ? &CvSVMKernel::calc_poly :
                    params->kernel_type == CvSVM::SIGMOID ? &CvSVMKernel::calc_sigmoid :
                    &CvSVMKernel::calc_linear;

    return true;
}


void CvSVMKernel::calc_non_rbf_base( int vcount, int var_count, const float** vecs,
                                     const float* another, Qfloat* results,
                                     double alpha, double beta )
{
    int j, k;
    for( j = 0; j < vcount; j++ )
    {
        const float* sample = vecs[j];
        double s = 0;
        for( k = 0; k <= var_count - 4; k += 4 )
            s += sample[k]*another[k] + sample[k+1]*another[k+1] +
                 sample[k+2]*another[k+2] + sample[k+3]*another[k+3];
        for( ; k < var_count; k++ )
            s += sample[k]*another[k];
        results[j] = (Qfloat)(s*alpha + beta);
    }
}


void CvSVMKernel::calc_linear( int vcount, int var_count, const float** vecs,
                               const float* another, Qfloat* results )
{
    calc_non_rbf_base( vcount, var_count, vecs, another, results, 1, 0 );
}


void CvSVMKernel::calc_poly( int vcount, int var_count, const float** vecs,
                             const float* another, Qfloat* results )
{
    CvMat R = cvMat( 1, vcount, QFLOAT_TYPE, results );
    calc_non_rbf_base( vcount, var_count, vecs, another, results, params->gamma, params->coef0 );
    if( vcount > 0 )
        cvPow( &R, &R, params->degree );
}


void CvSVMKernel::calc_sigmoid( int vcount, int var_count, const float** vecs,
                                const float* another, Qfloat* results )
{
    int j;
    calc_non_rbf_base( vcount, var_count, vecs, another, results,
                       -2*params->gamma, -2*params->coef0 );
    // TODO: speedup this
    for( j = 0; j < vcount; j++ )
    {
        Qfloat t = results[j];
        double e = exp(-fabs(t));
        if( t > 0 )
            results[j] = (Qfloat)((1. - e)/(1. + e));
        else
            results[j] = (Qfloat)((e - 1.)/(e + 1.));
    }
}


void CvSVMKernel::calc_rbf( int vcount, int var_count, const float** vecs,
                            const float* another, Qfloat* results )
{
    CvMat R = cvMat( 1, vcount, QFLOAT_TYPE, results );
    double gamma = -params->gamma;
    int j, k;

    for( j = 0; j < vcount; j++ )
    {
        const float* sample = vecs[j];
        double s = 0;

        for( k = 0; k <= var_count - 4; k += 4 )
        {
            double t0 = sample[k] - another[k];
            double t1 = sample[k+1] - another[k+1];

            s += t0*t0 + t1*t1;

            t0 = sample[k+2] - another[k+2];
            t1 = sample[k+3] - another[k+3];

            s += t0*t0 + t1*t1;
        }

        for( ; k < var_count; k++ )
        {
            double t0 = sample[k] - another[k];
            s += t0*t0;
        }
        results[j] = (Qfloat)(s*gamma);
    }

    if( vcount > 0 )
        cvExp( &R, &R );
}


void CvSVMKernel::calc( int vcount, int var_count, const float** vecs,
                        const float* another, Qfloat* results )
{
    const Qfloat max_val = (Qfloat)(FLT_MAX*1e-3);
    int j;
    (this->*calc_func)( vcount, var_count, vecs, another, results );
    for( j = 0; j < vcount; j++ )
    {
        if( results[j] > max_val )
            results[j] = max_val;
    }
}


// Generalized SMO+SVMlight algorithm
// Solves:
//
//  min [0.5(\alpha^T Q \alpha) + b^T \alpha]
//
//      y^T \alpha = \delta
//      y_i = +1 or -1
//      0 <= alpha_i <= Cp for y_i = 1
//      0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//  Q, b, y, Cp, Cn, and an initial feasible point \alpha
//  l is the size of vectors and matrices
//  eps is the stopping criterion
//
// solution will be put in \alpha, objective value will be put in obj
//

void CvSVMSolver::clear()
{
    G = 0;
    alpha = 0;
    y = 0;
    b = 0;
    buf[0] = buf[1] = 0;
    cvReleaseMemStorage( &storage );
    kernel = 0;
    select_working_set_func = 0;
    calc_rho_func = 0;

    rows = 0;
    samples = 0;
    get_row_func = 0;
}


CvSVMSolver::CvSVMSolver()
{
    storage = 0;
    clear();
}


CvSVMSolver::~CvSVMSolver()
{
    clear();
}


CvSVMSolver::CvSVMSolver( int _sample_count, int _var_count, const float** _samples, schar* _y,
                int _alpha_count, double* _alpha, double _Cp, double _Cn,
                CvMemStorage* _storage, CvSVMKernel* _kernel, GetRow _get_row,
                SelectWorkingSet _select_working_set, CalcRho _calc_rho )
{
    storage = 0;
    create( _sample_count, _var_count, _samples, _y, _alpha_count, _alpha, _Cp, _Cn,
            _storage, _kernel, _get_row, _select_working_set, _calc_rho );
}


bool CvSVMSolver::create( int _sample_count, int _var_count, const float** _samples, schar* _y,
                int _alpha_count, double* _alpha, double _Cp, double _Cn,
                CvMemStorage* _storage, CvSVMKernel* _kernel, GetRow _get_row,
                SelectWorkingSet _select_working_set, CalcRho _calc_rho )
{
    bool ok = false;
    int i, svm_type;

    CV_FUNCNAME( "CvSVMSolver::create" );

    __BEGIN__;

    int rows_hdr_size;

    clear();

    sample_count = _sample_count;
    var_count = _var_count;
    samples = _samples;
    y = _y;
    alpha_count = _alpha_count;
    alpha = _alpha;
    kernel = _kernel;

    C[0] = _Cn;
    C[1] = _Cp;
    eps = kernel->params->term_crit.epsilon;
    max_iter = kernel->params->term_crit.max_iter;
    storage = cvCreateChildMemStorage( _storage );

    b = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(b[0]));
    alpha_status = (schar*)cvMemStorageAlloc( storage, alpha_count*sizeof(alpha_status[0]));
    G = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(G[0]));
    for( i = 0; i < 2; i++ )
        buf[i] = (Qfloat*)cvMemStorageAlloc( storage, sample_count*2*sizeof(buf[i][0]) );
    svm_type = kernel->params->svm_type;

    select_working_set_func = _select_working_set;
    if( !select_working_set_func )
        select_working_set_func = svm_type == CvSVM::NU_SVC || svm_type == CvSVM::NU_SVR ?
        &CvSVMSolver::select_working_set_nu_svm : &CvSVMSolver::select_working_set;

    calc_rho_func = _calc_rho;
    if( !calc_rho_func )
        calc_rho_func = svm_type == CvSVM::NU_SVC || svm_type == CvSVM::NU_SVR ?
            &CvSVMSolver::calc_rho_nu_svm : &CvSVMSolver::calc_rho;

    get_row_func = _get_row;
    if( !get_row_func )
        get_row_func = params->svm_type == CvSVM::EPS_SVR ||
                       params->svm_type == CvSVM::NU_SVR ? &CvSVMSolver::get_row_svr :
                       params->svm_type == CvSVM::C_SVC ||
                       params->svm_type == CvSVM::NU_SVC ? &CvSVMSolver::get_row_svc :
                       &CvSVMSolver::get_row_one_class;

    cache_line_size = sample_count*sizeof(Qfloat);
    // cache size = max(num_of_samples^2*sizeof(Qfloat)*0.25, 64Kb)
    // (assuming that for large training sets ~25% of Q matrix is used)
    cache_size = MAX( cache_line_size*sample_count/4, CV_SVM_MIN_CACHE_SIZE );

    // the size of Q matrix row headers
    rows_hdr_size = sample_count*sizeof(rows[0]);
    if( rows_hdr_size > storage->block_size )
        CV_ERROR( CV_StsOutOfRange, "Too small storage block size" );

    lru_list.prev = lru_list.next = &lru_list;
    rows = (CvSVMKernelRow*)cvMemStorageAlloc( storage, rows_hdr_size );
    memset( rows, 0, rows_hdr_size );

    ok = true;

    __END__;

    return ok;
}


float* CvSVMSolver::get_row_base( int i, bool* _existed )
{
    int i1 = i < sample_count ? i : i - sample_count;
    CvSVMKernelRow* row = rows + i1;
    bool existed = row->data != 0;
    Qfloat* data;

    if( existed || cache_size <= 0 )
    {
        CvSVMKernelRow* del_row = existed ? row : lru_list.prev;
        data = del_row->data;
        assert( data != 0 );

        // delete row from the LRU list
        del_row->data = 0;
        del_row->prev->next = del_row->next;
        del_row->next->prev = del_row->prev;
    }
    else
    {
        data = (Qfloat*)cvMemStorageAlloc( storage, cache_line_size );
        cache_size -= cache_line_size;
    }

    // insert row into the LRU list
    row->data = data;
    row->prev = &lru_list;
    row->next = lru_list.next;
    row->prev->next = row->next->prev = row;

    if( !existed )
    {
        kernel->calc( sample_count, var_count, samples, samples[i1], row->data );
    }

    if( _existed )
        *_existed = existed;

    return row->data;
}


float* CvSVMSolver::get_row_svc( int i, float* row, float*, bool existed )
{
    if( !existed )
    {
        const schar* _y = y;
        int j, len = sample_count;
        assert( _y && i < sample_count );

        if( _y[i] > 0 )
        {
            for( j = 0; j < len; j++ )
                row[j] = _y[j]*row[j];
        }
        else
        {
            for( j = 0; j < len; j++ )
                row[j] = -_y[j]*row[j];
        }
    }
    return row;
}


float* CvSVMSolver::get_row_one_class( int, float* row, float*, bool )
{
    return row;
}


float* CvSVMSolver::get_row_svr( int i, float* row, float* dst, bool )
{
    int j, len = sample_count;
    Qfloat* dst_pos = dst;
    Qfloat* dst_neg = dst + len;
    if( i >= len )
    {
        Qfloat* temp;
        CV_SWAP( dst_pos, dst_neg, temp );
    }

    for( j = 0; j < len; j++ )
    {
        Qfloat t = row[j];
        dst_pos[j] = t;
        dst_neg[j] = -t;
    }
    return dst;
}



float* CvSVMSolver::get_row( int i, float* dst )
{
    bool existed = false;
    float* row = get_row_base( i, &existed );
    return (this->*get_row_func)( i, row, dst, existed );
}


#undef is_upper_bound
#define is_upper_bound(i) (alpha_status[i] > 0)

#undef is_lower_bound
#define is_lower_bound(i) (alpha_status[i] < 0)

#undef is_free
#define is_free(i) (alpha_status[i] == 0)

#undef get_C
#define get_C(i) (C[y[i]>0])

#undef update_alpha_status
#define update_alpha_status(i) \
    alpha_status[i] = (schar)(alpha[i] >= get_C(i) ? 1 : alpha[i] <= 0 ? -1 : 0)

#undef reconstruct_gradient
#define reconstruct_gradient() /* empty for now */


bool CvSVMSolver::solve_generic( CvSVMSolutionInfo& si )
{
    int iter = 0;
    int i, j, k;

    // 1. initialize gradient and alpha status
    for( i = 0; i < alpha_count; i++ )
    {
        update_alpha_status(i);
        G[i] = b[i];
        if( fabs(G[i]) > 1e200 )
            return false;
    }

    for( i = 0; i < alpha_count; i++ )
    {
        if( !is_lower_bound(i) )
        {
            const Qfloat *Q_i = get_row( i, buf[0] );
            double alpha_i = alpha[i];

            for( j = 0; j < alpha_count; j++ )
                G[j] += alpha_i*Q_i[j];
        }
    }

    // 2. optimization loop
    for(;;)
    {
        const Qfloat *Q_i, *Q_j;
        double C_i, C_j;
        double old_alpha_i, old_alpha_j, alpha_i, alpha_j;
        double delta_alpha_i, delta_alpha_j;

#ifdef _DEBUG
        for( i = 0; i < alpha_count; i++ )
        {
            if( fabs(G[i]) > 1e+300 )
                return false;

            if( fabs(alpha[i]) > 1e16 )
                return false;
        }
#endif

        if( (this->*select_working_set_func)( i, j ) != 0 || iter++ >= max_iter )
            break;

        Q_i = get_row( i, buf[0] );
        Q_j = get_row( j, buf[1] );

        C_i = get_C(i);
        C_j = get_C(j);

        alpha_i = old_alpha_i = alpha[i];
        alpha_j = old_alpha_j = alpha[j];

        if( y[i] != y[j] )
        {
            double denom = Q_i[i]+Q_j[j]+2*Q_i[j];
            double delta = (-G[i]-G[j])/MAX(fabs(denom),FLT_EPSILON);
            double diff = alpha_i - alpha_j;
            alpha_i += delta;
            alpha_j += delta;

            if( diff > 0 && alpha_j < 0 )
            {
                alpha_j = 0;
                alpha_i = diff;
            }
            else if( diff <= 0 && alpha_i < 0 )
            {
                alpha_i = 0;
                alpha_j = -diff;
            }

            if( diff > C_i - C_j && alpha_i > C_i )
            {
                alpha_i = C_i;
                alpha_j = C_i - diff;
            }
            else if( diff <= C_i - C_j && alpha_j > C_j )
            {
                alpha_j = C_j;
                alpha_i = C_j + diff;
            }
        }
        else
        {
            double denom = Q_i[i]+Q_j[j]-2*Q_i[j];
            double delta = (G[i]-G[j])/MAX(fabs(denom),FLT_EPSILON);
            double sum = alpha_i + alpha_j;
            alpha_i -= delta;
            alpha_j += delta;

            if( sum > C_i && alpha_i > C_i )
            {
                alpha_i = C_i;
                alpha_j = sum - C_i;
            }
            else if( sum <= C_i && alpha_j < 0)
            {
                alpha_j = 0;
                alpha_i = sum;
            }

            if( sum > C_j && alpha_j > C_j )
            {
                alpha_j = C_j;
                alpha_i = sum - C_j;
            }
            else if( sum <= C_j && alpha_i < 0 )
            {
                alpha_i = 0;
                alpha_j = sum;
            }
        }

        // update alpha
        alpha[i] = alpha_i;
        alpha[j] = alpha_j;
        update_alpha_status(i);
        update_alpha_status(j);

        // update G
        delta_alpha_i = alpha_i - old_alpha_i;
        delta_alpha_j = alpha_j - old_alpha_j;

        for( k = 0; k < alpha_count; k++ )
            G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
    }

    // calculate rho
    (this->*calc_rho_func)( si.rho, si.r );

    // calculate objective value
    for( i = 0, si.obj = 0; i < alpha_count; i++ )
        si.obj += alpha[i] * (G[i] + b[i]);

    si.obj *= 0.5;

    si.upper_bound_p = C[1];
    si.upper_bound_n = C[0];

    return true;
}


// return 1 if already optimal, return 0 otherwise
bool
CvSVMSolver::select_working_set( int& out_i, int& out_j )
{
    // return i,j which maximize -grad(f)^T d , under constraint
    // if alpha_i == C, d != +1
    // if alpha_i == 0, d != -1
    double Gmax1 = -DBL_MAX;        // max { -grad(f)_i * d | y_i*d = +1 }
    int Gmax1_idx = -1;

    double Gmax2 = -DBL_MAX;        // max { -grad(f)_i * d | y_i*d = -1 }
    int Gmax2_idx = -1;

    int i;

    for( i = 0; i < alpha_count; i++ )
    {
        double t;

        if( y[i] > 0 )    // y = +1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax1 )  // d = +1
            {
                Gmax1 = t;
                Gmax1_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax2 )  // d = -1
            {
                Gmax2 = t;
                Gmax2_idx = i;
            }
        }
        else        // y = -1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax2 )  // d = +1
            {
                Gmax2 = t;
                Gmax2_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax1 )  // d = -1
            {
                Gmax1 = t;
                Gmax1_idx = i;
            }
        }
    }

    out_i = Gmax1_idx;
    out_j = Gmax2_idx;

    return Gmax1 + Gmax2 < eps;
}


void
CvSVMSolver::calc_rho( double& rho, double& r )
{
    int i, nr_free = 0;
    double ub = DBL_MAX, lb = -DBL_MAX, sum_free = 0;

    for( i = 0; i < alpha_count; i++ )
    {
        double yG = y[i]*G[i];

        if( is_lower_bound(i) )
        {
            if( y[i] > 0 )
                ub = MIN(ub,yG);
            else
                lb = MAX(lb,yG);
        }
        else if( is_upper_bound(i) )
        {
            if( y[i] < 0)
                ub = MIN(ub,yG);
            else
                lb = MAX(lb,yG);
        }
        else
        {
            ++nr_free;
            sum_free += yG;
        }
    }

    rho = nr_free > 0 ? sum_free/nr_free : (ub + lb)*0.5;
    r = 0;
}


bool
CvSVMSolver::select_working_set_nu_svm( int& out_i, int& out_j )
{
    // return i,j which maximize -grad(f)^T d , under constraint
    // if alpha_i == C, d != +1
    // if alpha_i == 0, d != -1
    double Gmax1 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = +1, d = +1 }
    int Gmax1_idx = -1;

    double Gmax2 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = +1, d = -1 }
    int Gmax2_idx = -1;

    double Gmax3 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = -1, d = +1 }
    int Gmax3_idx = -1;

    double Gmax4 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = -1, d = -1 }
    int Gmax4_idx = -1;

    int i;

    for( i = 0; i < alpha_count; i++ )
    {
        double t;

        if( y[i] > 0 )    // y == +1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax1 )  // d = +1
            {
                Gmax1 = t;
                Gmax1_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax2 )  // d = -1
            {
                Gmax2 = t;
                Gmax2_idx = i;
            }
        }
        else        // y == -1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax3 )  // d = +1
            {
                Gmax3 = t;
                Gmax3_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax4 )  // d = -1
            {
                Gmax4 = t;
                Gmax4_idx = i;
            }
        }
    }

    if( MAX(Gmax1 + Gmax2, Gmax3 + Gmax4) < eps )
        return 1;

    if( Gmax1 + Gmax2 > Gmax3 + Gmax4 )
    {
        out_i = Gmax1_idx;
        out_j = Gmax2_idx;
    }
    else
    {
        out_i = Gmax3_idx;
        out_j = Gmax4_idx;
    }
    return 0;
}


void
CvSVMSolver::calc_rho_nu_svm( double& rho, double& r )
{
    int nr_free1 = 0, nr_free2 = 0;
    double ub1 = DBL_MAX, ub2 = DBL_MAX;
    double lb1 = -DBL_MAX, lb2 = -DBL_MAX;
    double sum_free1 = 0, sum_free2 = 0;
    double r1, r2;

    int i;

    for( i = 0; i < alpha_count; i++ )
    {
        double G_i = G[i];
        if( y[i] > 0 )
        {
            if( is_lower_bound(i) )
                ub1 = MIN( ub1, G_i );
            else if( is_upper_bound(i) )
                lb1 = MAX( lb1, G_i );
            else
            {
                ++nr_free1;
                sum_free1 += G_i;
            }
        }
        else
        {
            if( is_lower_bound(i) )
                ub2 = MIN( ub2, G_i );
            else if( is_upper_bound(i) )
                lb2 = MAX( lb2, G_i );
            else
            {
                ++nr_free2;
                sum_free2 += G_i;
            }
        }
    }

    r1 = nr_free1 > 0 ? sum_free1/nr_free1 : (ub1 + lb1)*0.5;
    r2 = nr_free2 > 0 ? sum_free2/nr_free2 : (ub2 + lb2)*0.5;

    rho = (r1 - r2)*0.5;
    r = (r1 + r2)*0.5;
}


/*
///////////////////////// construct and solve various formulations ///////////////////////
*/

bool CvSVMSolver::solve_c_svc( int _sample_count, int _var_count, const float** _samples, schar* _y,
                               double _Cp, double _Cn, CvMemStorage* _storage,
                               CvSVMKernel* _kernel, double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;

    if( !create( _sample_count, _var_count, _samples, _y, _sample_count,
                 _alpha, _Cp, _Cn, _storage, _kernel, &CvSVMSolver::get_row_svc,
                 &CvSVMSolver::select_working_set, &CvSVMSolver::calc_rho ))
        return false;

    for( i = 0; i < sample_count; i++ )
    {
        alpha[i] = 0;
        b[i] = -1;
    }

    if( !solve_generic( _si ))
        return false;

    for( i = 0; i < sample_count; i++ )
        alpha[i] *= y[i];

    return true;
}


bool CvSVMSolver::solve_nu_svc( int _sample_count, int _var_count, const float** _samples, schar* _y,
                                CvMemStorage* _storage, CvSVMKernel* _kernel,
                                double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;
    double sum_pos, sum_neg, inv_r;

    if( !create( _sample_count, _var_count, _samples, _y, _sample_count,
                 _alpha, 1., 1., _storage, _kernel, &CvSVMSolver::get_row_svc,
                 &CvSVMSolver::select_working_set_nu_svm, &CvSVMSolver::calc_rho_nu_svm ))
        return false;

    sum_pos = kernel->params->nu * sample_count * 0.5;
    sum_neg = kernel->params->nu * sample_count * 0.5;

    for( i = 0; i < sample_count; i++ )
    {
        if( y[i] > 0 )
        {
            alpha[i] = MIN(1.0, sum_pos);
            sum_pos -= alpha[i];
        }
        else
        {
            alpha[i] = MIN(1.0, sum_neg);
            sum_neg -= alpha[i];
        }
        b[i] = 0;
    }

    if( !solve_generic( _si ))
        return false;

    inv_r = 1./_si.r;

    for( i = 0; i < sample_count; i++ )
        alpha[i] *= y[i]*inv_r;

    _si.rho *= inv_r;
    _si.obj *= (inv_r*inv_r);
    _si.upper_bound_p = inv_r;
    _si.upper_bound_n = inv_r;

    return true;
}


bool CvSVMSolver::solve_one_class( int _sample_count, int _var_count, const float** _samples,
                                   CvMemStorage* _storage, CvSVMKernel* _kernel,
                                   double* _alpha, CvSVMSolutionInfo& _si )
{
    int i, n;
    double nu = _kernel->params->nu;

    if( !create( _sample_count, _var_count, _samples, 0, _sample_count,
                 _alpha, 1., 1., _storage, _kernel, &CvSVMSolver::get_row_one_class,
                 &CvSVMSolver::select_working_set, &CvSVMSolver::calc_rho ))
        return false;

    y = (schar*)cvMemStorageAlloc( storage, sample_count*sizeof(y[0]) );
    n = cvRound( nu*sample_count );

    for( i = 0; i < sample_count; i++ )
    {
        y[i] = 1;
        b[i] = 0;
        alpha[i] = i < n ? 1 : 0;
    }

    if( n < sample_count )
        alpha[n] = nu * sample_count - n;
    else
        alpha[n-1] = nu * sample_count - (n-1);

    return solve_generic(_si);
}


bool CvSVMSolver::solve_eps_svr( int _sample_count, int _var_count, const float** _samples,
                                 const float* _y, CvMemStorage* _storage,
                                 CvSVMKernel* _kernel, double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;
    double p = _kernel->params->p, C = _kernel->params->C;

    if( !create( _sample_count, _var_count, _samples, 0,
                 _sample_count*2, 0, C, C, _storage, _kernel, &CvSVMSolver::get_row_svr,
                 &CvSVMSolver::select_working_set, &CvSVMSolver::calc_rho ))
        return false;

    y = (schar*)cvMemStorageAlloc( storage, sample_count*2*sizeof(y[0]) );
    alpha = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(alpha[0]) );

    for( i = 0; i < sample_count; i++ )
    {
        alpha[i] = 0;
        b[i] = p - _y[i];
        y[i] = 1;

        alpha[i+sample_count] = 0;
        b[i+sample_count] = p + _y[i];
        y[i+sample_count] = -1;
    }

    if( !solve_generic( _si ))
        return false;

    for( i = 0; i < sample_count; i++ )
        _alpha[i] = alpha[i] - alpha[i+sample_count];

    return true;
}


bool CvSVMSolver::solve_nu_svr( int _sample_count, int _var_count, const float** _samples,
                                const float* _y, CvMemStorage* _storage,
                                CvSVMKernel* _kernel, double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;
    double C = _kernel->params->C, sum;

    if( !create( _sample_count, _var_count, _samples, 0,
                 _sample_count*2, 0, 1., 1., _storage, _kernel, &CvSVMSolver::get_row_svr,
                 &CvSVMSolver::select_working_set_nu_svm, &CvSVMSolver::calc_rho_nu_svm ))
        return false;

    y = (schar*)cvMemStorageAlloc( storage, sample_count*2*sizeof(y[0]) );
    alpha = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(alpha[0]) );
    sum = C * _kernel->params->nu * sample_count * 0.5;

    for( i = 0; i < sample_count; i++ )
    {
        alpha[i] = alpha[i + sample_count] = MIN(sum, C);
        sum -= alpha[i];

        b[i] = -_y[i];
        y[i] = 1;

        b[i + sample_count] = _y[i];
        y[i + sample_count] = -1;
    }

    if( !solve_generic( _si ))
        return false;

    for( i = 0; i < sample_count; i++ )
        _alpha[i] = alpha[i] - alpha[i+sample_count];

    return true;
}


//////////////////////////////////////////////////////////////////////////////////////////

CvSVM::CvSVM()
{
    decision_func = 0;
    class_labels = 0;
    class_weights = 0;
    storage = 0;
    var_idx = 0;
    kernel = 0;
    solver = 0;
    default_model_name = "my_svm";

    clear();
}


CvSVM::~CvSVM()
{
    clear();
}


void CvSVM::clear()
{
    cvFree( &decision_func );
    cvReleaseMat( &class_labels );
    cvReleaseMat( &class_weights );
    cvReleaseMemStorage( &storage );
    cvReleaseMat( &var_idx );
    delete kernel;
    delete solver;
    kernel = 0;
    solver = 0;
    var_all = 0;
    sv = 0;
    sv_total = 0;
}


CvSVM::CvSVM( const CvMat* _train_data, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx, CvSVMParams _params )
{
    decision_func = 0;
    class_labels = 0;
    class_weights = 0;
    storage = 0;
    var_idx = 0;
    kernel = 0;
    solver = 0;
    default_model_name = "my_svm";

    train( _train_data, _responses, _var_idx, _sample_idx, _params );
}


int CvSVM::get_support_vector_count() const
{
    return sv_total;
}


const float* CvSVM::get_support_vector(int i) const
{
    return sv && (unsigned)i < (unsigned)sv_total ? sv[i] : 0;
}


bool CvSVM::set_params( const CvSVMParams& _params )
{
    bool ok = false;

    CV_FUNCNAME( "CvSVM::set_params" );

    __BEGIN__;

    int kernel_type, svm_type;

    params = _params;

    kernel_type = params.kernel_type;
    svm_type = params.svm_type;

    if( kernel_type != LINEAR && kernel_type != POLY &&
        kernel_type != SIGMOID && kernel_type != RBF )
        CV_ERROR( CV_StsBadArg, "Unknown/unsupported kernel type" );

    if( kernel_type == LINEAR )
        params.gamma = 1;
    else if( params.gamma <= 0 )
        CV_ERROR( CV_StsOutOfRange, "gamma parameter of the kernel must be positive" );

    if( kernel_type != SIGMOID && kernel_type != POLY )
        params.coef0 = 0;
    else if( params.coef0 < 0 )
        CV_ERROR( CV_StsOutOfRange, "The kernel parameter <coef0> must be positive or zero" );

    if( kernel_type != POLY )
        params.degree = 0;
    else if( params.degree <= 0 )
        CV_ERROR( CV_StsOutOfRange, "The kernel parameter <degree> must be positive" );

    if( svm_type != C_SVC && svm_type != NU_SVC &&
        svm_type != ONE_CLASS && svm_type != EPS_SVR &&
        svm_type != NU_SVR )
        CV_ERROR( CV_StsBadArg, "Unknown/unsupported SVM type" );

    if( svm_type == ONE_CLASS || svm_type == NU_SVC )
        params.C = 0;
    else if( params.C <= 0 )
        CV_ERROR( CV_StsOutOfRange, "The parameter C must be positive" );

    if( svm_type == C_SVC || svm_type == EPS_SVR )
        params.nu = 0;
    else if( params.nu <= 0 || params.nu >= 1 )
        CV_ERROR( CV_StsOutOfRange, "The parameter nu must be between 0 and 1" );

    if( svm_type != EPS_SVR )
        params.p = 0;
    else if( params.p <= 0 )
        CV_ERROR( CV_StsOutOfRange, "The parameter p must be positive" );

    if( svm_type != C_SVC )
        params.class_weights = 0;

    params.term_crit = cvCheckTermCriteria( params.term_crit, DBL_EPSILON, INT_MAX );
    params.term_crit.epsilon = MAX( params.term_crit.epsilon, DBL_EPSILON );
    ok = true;

    __END__;

    return ok;
}



void CvSVM::create_kernel()
{
    kernel = new CvSVMKernel(&params,0);
}


void CvSVM::create_solver( )
{
    solver = new CvSVMSolver;
}


// switching function
bool CvSVM::train1( int sample_count, int var_count, const float** samples,
                    const void* _responses, double Cp, double Cn,
                    CvMemStorage* _storage, double* alpha, double& rho )
{
    bool ok = false;

    //CV_FUNCNAME( "CvSVM::train1" );

    __BEGIN__;

    CvSVMSolutionInfo si;
    int svm_type = params.svm_type;

    si.rho = 0;

    ok = svm_type == C_SVC ? solver->solve_c_svc( sample_count, var_count, samples, (schar*)_responses,
                                                  Cp, Cn, _storage, kernel, alpha, si ) :
         svm_type == NU_SVC ? solver->solve_nu_svc( sample_count, var_count, samples, (schar*)_responses,
                                                    _storage, kernel, alpha, si ) :
         svm_type == ONE_CLASS ? solver->solve_one_class( sample_count, var_count, samples,
                                                          _storage, kernel, alpha, si ) :
         svm_type == EPS_SVR ? solver->solve_eps_svr( sample_count, var_count, samples, (float*)_responses,
                                                      _storage, kernel, alpha, si ) :
         svm_type == NU_SVR ? solver->solve_nu_svr( sample_count, var_count, samples, (float*)_responses,
                                                    _storage, kernel, alpha, si ) : false;

    rho = si.rho;

    __END__;

    return ok;
}


bool CvSVM::do_train( int svm_type, int sample_count, int var_count, const float** samples,
                    const CvMat* responses, CvMemStorage* temp_storage, double* alpha )
{
    bool ok = false;

    CV_FUNCNAME( "CvSVM::do_train" );

    __BEGIN__;

    CvSVMDecisionFunc* df = 0;
    const int sample_size = var_count*sizeof(samples[0][0]);
    int i, j, k;

    cvClearMemStorage( storage );

    if( svm_type == ONE_CLASS || svm_type == EPS_SVR || svm_type == NU_SVR )
    {
        int sv_count = 0;

        CV_CALL( decision_func = df =
            (CvSVMDecisionFunc*)cvAlloc( sizeof(df[0]) ));

        df->rho = 0;
        if( !train1( sample_count, var_count, samples, svm_type == ONE_CLASS ? 0 :
            responses->data.i, 0, 0, temp_storage, alpha, df->rho ))
            EXIT;

        for( i = 0; i < sample_count; i++ )
            sv_count += fabs(alpha[i]) > 0;

        sv_total = df->sv_count = sv_count;
        CV_CALL( df->alpha = (double*)cvMemStorageAlloc( storage, sv_count*sizeof(df->alpha[0])) );
        CV_CALL( sv = (float**)cvMemStorageAlloc( storage, sv_count*sizeof(sv[0])));

        for( i = k = 0; i < sample_count; i++ )
        {
            if( fabs(alpha[i]) > 0 )
            {
                CV_CALL( sv[k] = (float*)cvMemStorageAlloc( storage, sample_size ));
                memcpy( sv[k], samples[i], sample_size );
                df->alpha[k++] = alpha[i];
            }
        }
    }
    else
    {
        int class_count = class_labels->cols;
        int* sv_tab = 0;
        const float** temp_samples = 0;
        int* class_ranges = 0;
        schar* temp_y = 0;
        assert( svm_type == CvSVM::C_SVC || svm_type == CvSVM::NU_SVC );

        if( svm_type == CvSVM::C_SVC && params.class_weights )
        {
            const CvMat* cw = params.class_weights;

            if( !CV_IS_MAT(cw) || (cw->cols != 1 && cw->rows != 1) ||
                cw->rows + cw->cols - 1 != class_count ||
                (CV_MAT_TYPE(cw->type) != CV_32FC1 && CV_MAT_TYPE(cw->type) != CV_64FC1) )
                CV_ERROR( CV_StsBadArg, "params.class_weights must be 1d floating-point vector "
                    "containing as many elements as the number of classes" );

            CV_CALL( class_weights = cvCreateMat( cw->rows, cw->cols, CV_64F ));
            CV_CALL( cvConvert( cw, class_weights ));
            CV_CALL( cvScale( class_weights, class_weights, params.C ));
        }

        CV_CALL( decision_func = df = (CvSVMDecisionFunc*)cvAlloc(
            (class_count*(class_count-1)/2)*sizeof(df[0])));

        CV_CALL( sv_tab = (int*)cvMemStorageAlloc( temp_storage, sample_count*sizeof(sv_tab[0]) ));
        memset( sv_tab, 0, sample_count*sizeof(sv_tab[0]) );
        CV_CALL( class_ranges = (int*)cvMemStorageAlloc( temp_storage,
                            (class_count + 1)*sizeof(class_ranges[0])));
        CV_CALL( temp_samples = (const float**)cvMemStorageAlloc( temp_storage,
                            sample_count*sizeof(temp_samples[0])));
        CV_CALL( temp_y = (schar*)cvMemStorageAlloc( temp_storage, sample_count));

        class_ranges[class_count] = 0;
        cvSortSamplesByClasses( samples, responses, class_ranges, 0 );
        //check that while cross-validation there were the samples from all the classes
        if( class_ranges[class_count] <= 0 )
            CV_ERROR( CV_StsBadArg, "While cross-validation one or more of the classes have "
            "been fell out of the sample. Try to enlarge <CvSVMParams::k_fold>" );

        if( svm_type == NU_SVC )
        {
            // check if nu is feasible
            for(i = 0; i < class_count; i++ )
            {
                int ci = class_ranges[i+1] - class_ranges[i];
                for( j = i+1; j< class_count; j++ )
                {
                    int cj = class_ranges[j+1] - class_ranges[j];
                    if( params.nu*(ci + cj)*0.5 > MIN( ci, cj ) )
                    {
                        // !!!TODO!!! add some diagnostic
                        EXIT; // exit immediately; will release the model and return NULL pointer
                    }
                }
            }
        }

        // train n*(n-1)/2 classifiers
        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                int si = class_ranges[i], ci = class_ranges[i+1] - si;
                int sj = class_ranges[j], cj = class_ranges[j+1] - sj;
                double Cp = params.C, Cn = Cp;
                int k1 = 0, sv_count = 0;

                for( k = 0; k < ci; k++ )
                {
                    temp_samples[k] = samples[si + k];
                    temp_y[k] = 1;
                }

                for( k = 0; k < cj; k++ )
                {
                    temp_samples[ci + k] = samples[sj + k];
                    temp_y[ci + k] = -1;
                }

                if( class_weights )
                {
                    Cp = class_weights->data.db[i];
                    Cn = class_weights->data.db[j];
                }

                if( !train1( ci + cj, var_count, temp_samples, temp_y,
                             Cp, Cn, temp_storage, alpha, df->rho ))
                    EXIT;

                for( k = 0; k < ci + cj; k++ )
                    sv_count += fabs(alpha[k]) > 0;

                df->sv_count = sv_count;

                CV_CALL( df->alpha = (double*)cvMemStorageAlloc( temp_storage,
                                                sv_count*sizeof(df->alpha[0])));
                CV_CALL( df->sv_index = (int*)cvMemStorageAlloc( temp_storage,
                                                sv_count*sizeof(df->sv_index[0])));

                for( k = 0; k < ci; k++ )
                {
                    if( fabs(alpha[k]) > 0 )
                    {
                        sv_tab[si + k] = 1;
                        df->sv_index[k1] = si + k;
                        df->alpha[k1++] = alpha[k];
                    }
                }

                for( k = 0; k < cj; k++ )
                {
                    if( fabs(alpha[ci + k]) > 0 )
                    {
                        sv_tab[sj + k] = 1;
                        df->sv_index[k1] = sj + k;
                        df->alpha[k1++] = alpha[ci + k];
                    }
                }
            }
        }

        // allocate support vectors and initialize sv_tab
        for( i = 0, k = 0; i < sample_count; i++ )
        {
            if( sv_tab[i] )
                sv_tab[i] = ++k;
        }

        sv_total = k;
        CV_CALL( sv = (float**)cvMemStorageAlloc( storage, sv_total*sizeof(sv[0])));

        for( i = 0, k = 0; i < sample_count; i++ )
        {
            if( sv_tab[i] )
            {
                CV_CALL( sv[k] = (float*)cvMemStorageAlloc( storage, sample_size ));
                memcpy( sv[k], samples[i], sample_size );
                k++;
            }
        }

        df = (CvSVMDecisionFunc*)decision_func;

        // set sv pointers
        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                for( k = 0; k < df->sv_count; k++ )
                {
                    df->sv_index[k] = sv_tab[df->sv_index[k]]-1;
                    assert( (unsigned)df->sv_index[k] < (unsigned)sv_total );
                }
            }
        }
    }

    ok = true;

    __END__;

    return ok;
}

bool CvSVM::train( const CvMat* _train_data, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx, CvSVMParams _params )
{
    bool ok = false;
    CvMat* responses = 0;
    CvMemStorage* temp_storage = 0;
    const float** samples = 0;

    CV_FUNCNAME( "CvSVM::train" );

    __BEGIN__;

    int svm_type, sample_count, var_count, sample_size;
    int block_size = 1 << 16;
    double* alpha;

    clear();
    CV_CALL( set_params( _params ));

    svm_type = _params.svm_type;

    /* Prepare training data and related parameters */
    CV_CALL( cvPrepareTrainData( "CvSVM::train", _train_data, CV_ROW_SAMPLE,
                                 svm_type != CvSVM::ONE_CLASS ? _responses : 0,
                                 svm_type == CvSVM::C_SVC ||
                                 svm_type == CvSVM::NU_SVC ? CV_VAR_CATEGORICAL :
                                 CV_VAR_ORDERED, _var_idx, _sample_idx,
                                 false, &samples, &sample_count, &var_count, &var_all,
                                 &responses, &class_labels, &var_idx ));


    sample_size = var_count*sizeof(samples[0][0]);

    // make the storage block size large enough to fit all
    // the temporary vectors and output support vectors.
    block_size = MAX( block_size, sample_count*(int)sizeof(CvSVMKernelRow));
    block_size = MAX( block_size, sample_count*2*(int)sizeof(double) + 1024 );
    block_size = MAX( block_size, sample_size*2 + 1024 );

    CV_CALL( storage = cvCreateMemStorage(block_size + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));
    CV_CALL( temp_storage = cvCreateChildMemStorage(storage));
    CV_CALL( alpha = (double*)cvMemStorageAlloc(temp_storage, sample_count*sizeof(double)));

    create_kernel();
    create_solver();

    if( !do_train( svm_type, sample_count, var_count, samples, responses, temp_storage, alpha ))
        EXIT;

    ok = true; // model has been trained succesfully

    __END__;

    delete solver;
    solver = 0;
    cvReleaseMemStorage( &temp_storage );
    cvReleaseMat( &responses );
    cvFree( &samples );

    if( cvGetErrStatus() < 0 || !ok )
        clear();

    return ok;
}

struct indexedratio 
{
    double val;
    int ind;
    int count_smallest, count_biggest;
    void eval() { val = (double) count_smallest/(count_smallest+count_biggest); }
};

static int CV_CDECL
icvCmpIndexedratio( const void* a, const void* b )
{
    return ((const indexedratio*)a)->val < ((const indexedratio*)b)->val ? -1
    : ((const indexedratio*)a)->val > ((const indexedratio*)b)->val ? 1
    : 0;
}

bool CvSVM::train_auto( const CvMat* _train_data, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx, CvSVMParams _params, int k_fold,
    CvParamGrid C_grid, CvParamGrid gamma_grid, CvParamGrid p_grid,
    CvParamGrid nu_grid, CvParamGrid coef_grid, CvParamGrid degree_grid,
    bool balanced)
{
    bool ok = false;
    CvMat* responses = 0;
    CvMat* responses_local = 0;
    CvMemStorage* temp_storage = 0;
    const float** samples = 0;
    const float** samples_local = 0;

    CV_FUNCNAME( "CvSVM::train_auto" );
    __BEGIN__;

    int svm_type, sample_count, var_count, sample_size;
    int block_size = 1 << 16;
    double* alpha;
    int i, k;
    RNG* rng = &theRNG();

    // all steps are logarithmic and must be > 1
    double degree_step = 10, g_step = 10, coef_step = 10, C_step = 10, nu_step = 10, p_step = 10;
    double gamma = 0, C = 0, degree = 0, coef = 0, p = 0, nu = 0;
    double best_degree = 0, best_gamma = 0, best_coef = 0, best_C = 0, best_nu = 0, best_p = 0;
    float min_error = FLT_MAX, error;

    if( _params.svm_type == CvSVM::ONE_CLASS )
    {
        if(!train( _train_data, _responses, _var_idx, _sample_idx, _params ))
            EXIT;
        return true;
    }

    clear();

    if( k_fold < 2 )
        CV_ERROR( CV_StsBadArg, "Parameter <k_fold> must be > 1" );

    CV_CALL(set_params( _params ));
    svm_type = _params.svm_type;

    // All the parameters except, possibly, <coef0> are positive.
    // <coef0> is nonnegative
    if( C_grid.step <= 1 )
    {
        C_grid.min_val = C_grid.max_val = params.C;
        C_grid.step = 10;
    }
    else
        CV_CALL(C_grid.check());

    if( gamma_grid.step <= 1 )
    {
        gamma_grid.min_val = gamma_grid.max_val = params.gamma;
        gamma_grid.step = 10;
    }
    else
        CV_CALL(gamma_grid.check());

    if( p_grid.step <= 1 )
    {
        p_grid.min_val = p_grid.max_val = params.p;
        p_grid.step = 10;
    }
    else
        CV_CALL(p_grid.check());

    if( nu_grid.step <= 1 )
    {
        nu_grid.min_val = nu_grid.max_val = params.nu;
        nu_grid.step = 10;
    }
    else
        CV_CALL(nu_grid.check());

    if( coef_grid.step <= 1 )
    {
        coef_grid.min_val = coef_grid.max_val = params.coef0;
        coef_grid.step = 10;
    }
    else
        CV_CALL(coef_grid.check());

    if( degree_grid.step <= 1 )
    {
        degree_grid.min_val = degree_grid.max_val = params.degree;
        degree_grid.step = 10;
    }
    else
        CV_CALL(degree_grid.check());

    // these parameters are not used:
    if( params.kernel_type != CvSVM::POLY )
        degree_grid.min_val = degree_grid.max_val = params.degree;
    if( params.kernel_type == CvSVM::LINEAR )
        gamma_grid.min_val = gamma_grid.max_val = params.gamma;
    if( params.kernel_type != CvSVM::POLY && params.kernel_type != CvSVM::SIGMOID )
        coef_grid.min_val = coef_grid.max_val = params.coef0;
    if( svm_type == CvSVM::NU_SVC || svm_type == CvSVM::ONE_CLASS )
        C_grid.min_val = C_grid.max_val = params.C;
    if( svm_type == CvSVM::C_SVC || svm_type == CvSVM::EPS_SVR )
        nu_grid.min_val = nu_grid.max_val = params.nu;
    if( svm_type != CvSVM::EPS_SVR )
        p_grid.min_val = p_grid.max_val = params.p;

    CV_ASSERT( g_step > 1 && degree_step > 1 && coef_step > 1);
    CV_ASSERT( p_step > 1 && C_step > 1 && nu_step > 1 );

    /* Prepare training data and related parameters */
    CV_CALL(cvPrepareTrainData( "CvSVM::train_auto", _train_data, CV_ROW_SAMPLE,
                                 svm_type != CvSVM::ONE_CLASS ? _responses : 0,
                                 svm_type == CvSVM::C_SVC ||
                                 svm_type == CvSVM::NU_SVC ? CV_VAR_CATEGORICAL :
                                 CV_VAR_ORDERED, _var_idx, _sample_idx,
                                 false, &samples, &sample_count, &var_count, &var_all,
                                 &responses, &class_labels, &var_idx ));

    sample_size = var_count*sizeof(samples[0][0]);

    // make the storage block size large enough to fit all
    // the temporary vectors and output support vectors.
    block_size = MAX( block_size, sample_count*(int)sizeof(CvSVMKernelRow));
    block_size = MAX( block_size, sample_count*2*(int)sizeof(double) + 1024 );
    block_size = MAX( block_size, sample_size*2 + 1024 );

    CV_CALL(storage = cvCreateMemStorage(block_size));
    CV_CALL(temp_storage = cvCreateChildMemStorage(storage));
    CV_CALL(alpha = (double*)cvMemStorageAlloc(temp_storage, sample_count*sizeof(double)));

    create_kernel();
    create_solver();

    {
    const int testset_size = sample_count/k_fold;
    const int trainset_size = sample_count - testset_size;
    const int last_testset_size = sample_count - testset_size*(k_fold-1);
    const int last_trainset_size = sample_count - last_testset_size;
    const bool is_regression = (svm_type == EPS_SVR) || (svm_type == NU_SVR);

    size_t resp_elem_size = CV_ELEM_SIZE(responses->type);
    size_t size = 2*last_trainset_size*sizeof(samples[0]);

    samples_local = (const float**) cvAlloc( size );
    memset( samples_local, 0, size );

    responses_local = cvCreateMat( 1, trainset_size, CV_MAT_TYPE(responses->type) );
    cvZero( responses_local );

    // randomly permute samples and responses
    for( i = 0; i < sample_count; i++ )
    {
        int i1 = (*rng)(sample_count);
        int i2 = (*rng)(sample_count);
        const float* temp;
        float t;
        int y;

        CV_SWAP( samples[i1], samples[i2], temp );
        if( is_regression )
            CV_SWAP( responses->data.fl[i1], responses->data.fl[i2], t );
        else
            CV_SWAP( responses->data.i[i1], responses->data.i[i2], y );
    }
        
    if (!is_regression && class_labels->cols==2 && balanced)
    {
        // count class samples
        int num_0=0,num_1=0;
        for (i=0; i<sample_count; ++i)
        {
            if (responses->data.i[i]==class_labels->data.i[0])
                ++num_0;
            else
                ++num_1;
        }
        
        int label_smallest_class;
        int label_biggest_class;
        if (num_0 < num_1)
        {
            label_biggest_class = class_labels->data.i[1];
            label_smallest_class = class_labels->data.i[0]; 
        }
        else
        {
            label_biggest_class = class_labels->data.i[0];
            label_smallest_class = class_labels->data.i[1];
            int y;
            CV_SWAP(num_0,num_1,y);
        }
        const double class_ratio = (double) num_0/sample_count;
        // calculate class ratio of each fold
        indexedratio *ratios=0;
        ratios = (indexedratio*) cvAlloc(k_fold*sizeof(*ratios));
        for (int k=0, i_begin=0; k<k_fold; ++k, i_begin+=testset_size)
        {
            int count0=0;
            int count1=0;
            int i_end = i_begin + (k<k_fold-1 ? testset_size : last_testset_size);
            for (int i=i_begin; i<i_end; ++i)
            {
                if (responses->data.i[i]==label_smallest_class)
                    ++count0;
                else
                    ++count1;
            }
            ratios[k].ind = k;
            ratios[k].count_smallest = count0;
            ratios[k].count_biggest = count1;
            ratios[k].eval();
        }
        // initial distance
        qsort(ratios, k_fold, sizeof(ratios[0]), icvCmpIndexedratio);
        double old_dist = 0.0;
        for (int k=0; k<k_fold; ++k)
            old_dist += abs(ratios[k].val-class_ratio);
        double new_dist = 1.0;
        // iterate to make the folds more balanced
        while (new_dist > 0.0)
        {
            if (ratios[0].count_biggest==0 || ratios[k_fold-1].count_smallest==0)
                break; // we are not able to swap samples anymore
            // what if we swap the samples, calculate the new distance
            ratios[0].count_smallest++;
            ratios[0].count_biggest--;
            ratios[0].eval();
            ratios[k_fold-1].count_smallest--;
            ratios[k_fold-1].count_biggest++;
            ratios[k_fold-1].eval();
            qsort(ratios, k_fold, sizeof(ratios[0]), icvCmpIndexedratio);
            new_dist = 0.0;
            for (int k=0; k<k_fold; ++k)
                new_dist += abs(ratios[k].val-class_ratio);
            if (new_dist < old_dist)
            {
                // swapping really improves, so swap the samples
                // index of the biggest_class sample from the minimum ratio fold
                int i1 = ratios[0].ind * testset_size;
                for ( ; i1<sample_count; ++i1)
                {
                    if (responses->data.i[i1]==label_biggest_class)
                        break;
                }
                // index of the smallest_class sample from the maximum ratio fold
                int i2 = ratios[k_fold-1].ind * testset_size;
                for ( ; i2<sample_count; ++i2)
                {
                    if (responses->data.i[i2]==label_smallest_class)
                        break;
                }
                // swap
                const float* temp;
                int y;
                CV_SWAP( samples[i1], samples[i2], temp );
                CV_SWAP( responses->data.i[i1], responses->data.i[i2], y );
                old_dist = new_dist;
            }
            else
                break; // does not improve, so break the loop
        }
        cvFree(&ratios);
    }

    int* cls_lbls = class_labels ? class_labels->data.i : 0;
    C = C_grid.min_val;
    do
    {
      params.C = C;
      gamma = gamma_grid.min_val;
      do
      {
        params.gamma = gamma;
        p = p_grid.min_val;
        do
        {
          params.p = p;
          nu = nu_grid.min_val;
          do
          {
            params.nu = nu;
            coef = coef_grid.min_val;
            do
            {
              params.coef0 = coef;
              degree = degree_grid.min_val;
              do
              {
                params.degree = degree;

                float** test_samples_ptr = (float**)samples;
                uchar* true_resp = responses->data.ptr;
                int test_size = testset_size;
                int train_size = trainset_size;

                error = 0;
                for( k = 0; k < k_fold; k++ )
                {
                    memcpy( samples_local, samples, sizeof(samples[0])*test_size*k );
                    memcpy( samples_local + test_size*k, test_samples_ptr + test_size,
                        sizeof(samples[0])*(sample_count - testset_size*(k+1)) );

                    memcpy( responses_local->data.ptr, responses->data.ptr, resp_elem_size*test_size*k );
                    memcpy( responses_local->data.ptr + resp_elem_size*test_size*k,
                        true_resp + resp_elem_size*test_size,
                        resp_elem_size*(sample_count - testset_size*(k+1)) );

                    if( k == k_fold - 1 )
                    {
                        test_size = last_testset_size;
                        train_size = last_trainset_size;
                        responses_local->cols = last_trainset_size;
                    }

                    // Train SVM on <train_size> samples
                    if( !do_train( svm_type, train_size, var_count,
                        (const float**)samples_local, responses_local, temp_storage, alpha ) )
                        EXIT;

                    // Compute test set error on <test_size> samples
                    for( i = 0; i < test_size; i++, true_resp += resp_elem_size, test_samples_ptr++ )
                    {
                        float resp = predict( *test_samples_ptr, var_count );
                        error += is_regression ? powf( resp - *(float*)true_resp, 2 )
                            : ((int)resp != cls_lbls[*(int*)true_resp]);
                    }
                }
                if( min_error > error )
                {
                    min_error   = error;
                    best_degree = degree;
                    best_gamma  = gamma;
                    best_coef   = coef;
                    best_C      = C;
                    best_nu     = nu;
                    best_p      = p;
                }
                degree *= degree_grid.step;
              }
              while( degree < degree_grid.max_val );
              coef *= coef_grid.step;
            }
            while( coef < coef_grid.max_val );
            nu *= nu_grid.step;
          }
          while( nu < nu_grid.max_val );
          p *= p_grid.step;
        }
        while( p < p_grid.max_val );
        gamma *= gamma_grid.step;
      }
      while( gamma < gamma_grid.max_val );
      C *= C_grid.step;
    }
    while( C < C_grid.max_val );
    }

    min_error /= (float) sample_count;

    params.C      = best_C;
    params.nu     = best_nu;
    params.p      = best_p;
    params.gamma  = best_gamma;
    params.degree = best_degree;
    params.coef0  = best_coef;

    CV_CALL(ok = do_train( svm_type, sample_count, var_count, samples, responses, temp_storage, alpha ));

    __END__;

    delete solver;
    solver = 0;
    cvReleaseMemStorage( &temp_storage );
    cvReleaseMat( &responses );
    cvReleaseMat( &responses_local );
    cvFree( &samples );
    cvFree( &samples_local );

    if( cvGetErrStatus() < 0 || !ok )
        clear();

    return ok;
}

float CvSVM::predict( const float* row_sample, int row_len, bool returnDFVal ) const
{
    assert( kernel );
    assert( row_sample );

    int var_count = get_var_count();
    assert( row_len == var_count );

    int class_count = class_labels ? class_labels->cols :
                  params.svm_type == ONE_CLASS ? 1 : 0;

    float result = 0;
    cv::AutoBuffer<float> _buffer(sv_total + (class_count+1)*2);
    float* buffer = _buffer;

    if( params.svm_type == EPS_SVR ||
        params.svm_type == NU_SVR ||
        params.svm_type == ONE_CLASS )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int i, sv_count = df->sv_count;
        double sum = -df->rho;

        kernel->calc( sv_count, var_count, (const float**)sv, row_sample, buffer );
        for( i = 0; i < sv_count; i++ )
            sum += buffer[i]*df->alpha[i];

        result = params.svm_type == ONE_CLASS ? (float)(sum > 0) : (float)sum;
    }
    else if( params.svm_type == C_SVC ||
             params.svm_type == NU_SVC )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int* vote = (int*)(buffer + sv_total);
        int i, j, k;

        memset( vote, 0, class_count*sizeof(vote[0]));
        kernel->calc( sv_total, var_count, (const float**)sv, row_sample, buffer );
        double sum = 0.;

        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                sum = -df->rho;
                int sv_count = df->sv_count;
                for( k = 0; k < sv_count; k++ )
                    sum += df->alpha[k]*buffer[df->sv_index[k]];

                vote[sum > 0 ? i : j]++;
            }
        }

        for( i = 1, k = 0; i < class_count; i++ )
        {
            if( vote[i] > vote[k] )
                k = i;
        }
        result = returnDFVal && class_count == 2 ? (float)sum : (float)(class_labels->data.i[k]);
    }
    else
        CV_Error( CV_StsBadArg, "INTERNAL ERROR: Unknown SVM type, "
                                "the SVM structure is probably corrupted" );

    return result;
}

float CvSVM::predict( const CvMat* sample, bool returnDFVal ) const
{
    float result = 0;
    float* row_sample = 0;

    CV_FUNCNAME( "CvSVM::predict" );

    __BEGIN__;

    int class_count;
    
    if( !kernel )
        CV_ERROR( CV_StsBadArg, "The SVM should be trained first" );

    class_count = class_labels ? class_labels->cols :
                  params.svm_type == ONE_CLASS ? 1 : 0;

    CV_CALL( cvPreparePredictData( sample, var_all, var_idx,
                                   class_count, 0, &row_sample ));
    result = predict( row_sample, get_var_count(), returnDFVal );
  
    __END__;

    if( sample && (!CV_IS_MAT(sample) || sample->data.fl != row_sample) )
        cvFree( &row_sample );

    return result;
}

struct predict_body_svm {
    predict_body_svm(const CvSVM* _pointer, float* _result, const CvMat* _samples, CvMat* _results)
    {
        pointer = _pointer;
        result = _result;
        samples = _samples;
        results = _results;
    }
    
    const CvSVM* pointer;
    float* result;
    const CvMat* samples;
    CvMat* results;
  
    void operator()( const cv::BlockedRange& range ) const
    {
        for(int i = range.begin(); i < range.end(); i++ )
        {
            CvMat sample;
            cvGetRow( samples, &sample, i );
            int r = (int)pointer->predict(&sample);
            if (results)
                results->data.fl[i] = r;
            if (i == 0)
                *result = r;
	}
    }
};

float CvSVM::predict(const CvMat* samples, CV_OUT CvMat* results) const
{
    float result = 0;
    cv::parallel_for(cv::BlockedRange(0, samples->rows), 
		     predict_body_svm(this, &result, samples, results)
    );
    return result;
}


CvSVM::CvSVM( const Mat& _train_data, const Mat& _responses,
              const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params )
{
    decision_func = 0;
    class_labels = 0;
    class_weights = 0;
    storage = 0;
    var_idx = 0;
    kernel = 0;
    solver = 0;
    default_model_name = "my_svm";
    
    train( _train_data, _responses, _var_idx, _sample_idx, _params );
}

bool CvSVM::train( const Mat& _train_data, const Mat& _responses,
                  const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params )
{
    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx, sidx = _sample_idx;
    return train(&tdata, &responses, vidx.data.ptr ? &vidx : 0, sidx.data.ptr ? &sidx : 0, _params);
}


bool CvSVM::train_auto( const Mat& _train_data, const Mat& _responses,
                       const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params, int k_fold,
                       CvParamGrid C_grid, CvParamGrid gamma_grid, CvParamGrid p_grid,
                       CvParamGrid nu_grid, CvParamGrid coef_grid, CvParamGrid degree_grid, bool balanced )
{
    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx, sidx = _sample_idx;
    return train_auto(&tdata, &responses, vidx.data.ptr ? &vidx : 0,
                      sidx.data.ptr ? &sidx : 0, _params, k_fold, C_grid, gamma_grid, p_grid,
                      nu_grid, coef_grid, degree_grid, balanced);
}

float CvSVM::predict( const Mat& _sample, bool returnDFVal ) const
{
    CvMat sample = _sample; 
    return predict(&sample, returnDFVal);
}


void CvSVM::write_params( CvFileStorage* fs ) const
{
    //CV_FUNCNAME( "CvSVM::write_params" );

    __BEGIN__;

    int svm_type = params.svm_type;
    int kernel_type = params.kernel_type;

    const char* svm_type_str =
        svm_type == CvSVM::C_SVC ? "C_SVC" :
        svm_type == CvSVM::NU_SVC ? "NU_SVC" :
        svm_type == CvSVM::ONE_CLASS ? "ONE_CLASS" :
        svm_type == CvSVM::EPS_SVR ? "EPS_SVR" :
        svm_type == CvSVM::NU_SVR ? "NU_SVR" : 0;
    const char* kernel_type_str =
        kernel_type == CvSVM::LINEAR ? "LINEAR" :
        kernel_type == CvSVM::POLY ? "POLY" :
        kernel_type == CvSVM::RBF ? "RBF" :
        kernel_type == CvSVM::SIGMOID ? "SIGMOID" : 0;

    if( svm_type_str )
        cvWriteString( fs, "svm_type", svm_type_str );
    else
        cvWriteInt( fs, "svm_type", svm_type );

    // save kernel
    cvStartWriteStruct( fs, "kernel", CV_NODE_MAP + CV_NODE_FLOW );

    if( kernel_type_str )
        cvWriteString( fs, "type", kernel_type_str );
    else
        cvWriteInt( fs, "type", kernel_type );

    if( kernel_type == CvSVM::POLY || !kernel_type_str )
        cvWriteReal( fs, "degree", params.degree );

    if( kernel_type != CvSVM::LINEAR || !kernel_type_str )
        cvWriteReal( fs, "gamma", params.gamma );

    if( kernel_type == CvSVM::POLY || kernel_type == CvSVM::SIGMOID || !kernel_type_str )
        cvWriteReal( fs, "coef0", params.coef0 );

    cvEndWriteStruct(fs);

    if( svm_type == CvSVM::C_SVC || svm_type == CvSVM::EPS_SVR ||
        svm_type == CvSVM::NU_SVR || !svm_type_str )
        cvWriteReal( fs, "C", params.C );

    if( svm_type == CvSVM::NU_SVC || svm_type == CvSVM::ONE_CLASS ||
        svm_type == CvSVM::NU_SVR || !svm_type_str )
        cvWriteReal( fs, "nu", params.nu );

    if( svm_type == CvSVM::EPS_SVR || !svm_type_str )
        cvWriteReal( fs, "p", params.p );

    cvStartWriteStruct( fs, "term_criteria", CV_NODE_MAP + CV_NODE_FLOW );
    if( params.term_crit.type & CV_TERMCRIT_EPS )
        cvWriteReal( fs, "epsilon", params.term_crit.epsilon );
    if( params.term_crit.type & CV_TERMCRIT_ITER )
        cvWriteInt( fs, "iterations", params.term_crit.max_iter );
    cvEndWriteStruct( fs );

    __END__;
}


void CvSVM::write( CvFileStorage* fs, const char* name ) const
{
    CV_FUNCNAME( "CvSVM::write" );

    __BEGIN__;

    int i, var_count = get_var_count(), df_count, class_count;
    const CvSVMDecisionFunc* df = decision_func;

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_SVM );

    write_params( fs );

    cvWriteInt( fs, "var_all", var_all );
    cvWriteInt( fs, "var_count", var_count );

    class_count = class_labels ? class_labels->cols :
                  params.svm_type == CvSVM::ONE_CLASS ? 1 : 0;

    if( class_count )
    {
        cvWriteInt( fs, "class_count", class_count );

        if( class_labels )
            cvWrite( fs, "class_labels", class_labels );

        if( class_weights )
            cvWrite( fs, "class_weights", class_weights );
    }

    if( var_idx )
        cvWrite( fs, "var_idx", var_idx );

    // write the joint collection of support vectors
    cvWriteInt( fs, "sv_total", sv_total );
    cvStartWriteStruct( fs, "support_vectors", CV_NODE_SEQ );
    for( i = 0; i < sv_total; i++ )
    {
        cvStartWriteStruct( fs, 0, CV_NODE_SEQ + CV_NODE_FLOW );
        cvWriteRawData( fs, sv[i], var_count, "f" );
        cvEndWriteStruct( fs );
    }

    cvEndWriteStruct( fs );

    // write decision functions
    df_count = class_count > 1 ? class_count*(class_count-1)/2 : 1;
    df = decision_func;

    cvStartWriteStruct( fs, "decision_functions", CV_NODE_SEQ );
    for( i = 0; i < df_count; i++ )
    {
        int sv_count = df[i].sv_count;
        cvStartWriteStruct( fs, 0, CV_NODE_MAP );
        cvWriteInt( fs, "sv_count", sv_count );
        cvWriteReal( fs, "rho", df[i].rho );
        cvStartWriteStruct( fs, "alpha", CV_NODE_SEQ+CV_NODE_FLOW );
        cvWriteRawData( fs, df[i].alpha, df[i].sv_count, "d" );
        cvEndWriteStruct( fs );
        if( class_count > 1 )
        {
            cvStartWriteStruct( fs, "index", CV_NODE_SEQ+CV_NODE_FLOW );
            cvWriteRawData( fs, df[i].sv_index, df[i].sv_count, "i" );
            cvEndWriteStruct( fs );
        }
        else
            CV_ASSERT( sv_count == sv_total );
        cvEndWriteStruct( fs );
    }
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );

    __END__;
}


void CvSVM::read_params( CvFileStorage* fs, CvFileNode* svm_node )
{
    CV_FUNCNAME( "CvSVM::read_params" );

    __BEGIN__;

    int svm_type, kernel_type;
    CvSVMParams _params;

    CvFileNode* tmp_node = cvGetFileNodeByName( fs, svm_node, "svm_type" );
    CvFileNode* kernel_node;
    if( !tmp_node )
        CV_ERROR( CV_StsBadArg, "svm_type tag is not found" );

    if( CV_NODE_TYPE(tmp_node->tag) == CV_NODE_INT )
        svm_type = cvReadInt( tmp_node, -1 );
    else
    {
        const char* svm_type_str = cvReadString( tmp_node, "" );
        svm_type =
            strcmp( svm_type_str, "C_SVC" ) == 0 ? CvSVM::C_SVC :
            strcmp( svm_type_str, "NU_SVC" ) == 0 ? CvSVM::NU_SVC :
            strcmp( svm_type_str, "ONE_CLASS" ) == 0 ? CvSVM::ONE_CLASS :
            strcmp( svm_type_str, "EPS_SVR" ) == 0 ? CvSVM::EPS_SVR :
            strcmp( svm_type_str, "NU_SVR" ) == 0 ? CvSVM::NU_SVR : -1;

        if( svm_type < 0 )
            CV_ERROR( CV_StsParseError, "Missing of invalid SVM type" );
    }

    kernel_node = cvGetFileNodeByName( fs, svm_node, "kernel" );
    if( !kernel_node )
        CV_ERROR( CV_StsParseError, "SVM kernel tag is not found" );

    tmp_node = cvGetFileNodeByName( fs, kernel_node, "type" );
    if( !tmp_node )
        CV_ERROR( CV_StsParseError, "SVM kernel type tag is not found" );

    if( CV_NODE_TYPE(tmp_node->tag) == CV_NODE_INT )
        kernel_type = cvReadInt( tmp_node, -1 );
    else
    {
        const char* kernel_type_str = cvReadString( tmp_node, "" );
        kernel_type =
            strcmp( kernel_type_str, "LINEAR" ) == 0 ? CvSVM::LINEAR :
            strcmp( kernel_type_str, "POLY" ) == 0 ? CvSVM::POLY :
            strcmp( kernel_type_str, "RBF" ) == 0 ? CvSVM::RBF :
            strcmp( kernel_type_str, "SIGMOID" ) == 0 ? CvSVM::SIGMOID : -1;

        if( kernel_type < 0 )
            CV_ERROR( CV_StsParseError, "Missing of invalid SVM kernel type" );
    }

    _params.svm_type = svm_type;
    _params.kernel_type = kernel_type;
    _params.degree = cvReadRealByName( fs, kernel_node, "degree", 0 );
    _params.gamma = cvReadRealByName( fs, kernel_node, "gamma", 0 );
    _params.coef0 = cvReadRealByName( fs, kernel_node, "coef0", 0 );

    _params.C = cvReadRealByName( fs, svm_node, "C", 0 );
    _params.nu = cvReadRealByName( fs, svm_node, "nu", 0 );
    _params.p = cvReadRealByName( fs, svm_node, "p", 0 );
    _params.class_weights = 0;

    tmp_node = cvGetFileNodeByName( fs, svm_node, "term_criteria" );
    if( tmp_node )
    {
        _params.term_crit.epsilon = cvReadRealByName( fs, tmp_node, "epsilon", -1. );
        _params.term_crit.max_iter = cvReadIntByName( fs, tmp_node, "iterations", -1 );
        _params.term_crit.type = (_params.term_crit.epsilon >= 0 ? CV_TERMCRIT_EPS : 0) +
                               (_params.term_crit.max_iter >= 0 ? CV_TERMCRIT_ITER : 0);
    }
    else
        _params.term_crit = cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, FLT_EPSILON );

    set_params( _params );

    __END__;
}


void CvSVM::read( CvFileStorage* fs, CvFileNode* svm_node )
{
    const double not_found_dbl = DBL_MAX;

    CV_FUNCNAME( "CvSVM::read" );

    __BEGIN__;

    int i, var_count, df_count, class_count;
    int block_size = 1 << 16, sv_size;
    CvFileNode *sv_node, *df_node;
    CvSVMDecisionFunc* df;
    CvSeqReader reader;

    if( !svm_node )
        CV_ERROR( CV_StsParseError, "The requested element is not found" );

    clear();

    // read SVM parameters
    read_params( fs, svm_node );

    // and top-level data
    sv_total = cvReadIntByName( fs, svm_node, "sv_total", -1 );
    var_all = cvReadIntByName( fs, svm_node, "var_all", -1 );
    var_count = cvReadIntByName( fs, svm_node, "var_count", var_all );
    class_count = cvReadIntByName( fs, svm_node, "class_count", 0 );

    if( sv_total <= 0 || var_all <= 0 || var_count <= 0 || var_count > var_all || class_count < 0 )
        CV_ERROR( CV_StsParseError, "SVM model data is invalid, check sv_count, var_* and class_count tags" );

    CV_CALL( class_labels = (CvMat*)cvReadByName( fs, svm_node, "class_labels" ));
    CV_CALL( class_weights = (CvMat*)cvReadByName( fs, svm_node, "class_weights" ));
    CV_CALL( var_idx = (CvMat*)cvReadByName( fs, svm_node, "var_idx" ));

    if( class_count > 1 && (!class_labels ||
        !CV_IS_MAT(class_labels) || class_labels->cols != class_count))
        CV_ERROR( CV_StsParseError, "Array of class labels is missing or invalid" );

    if( var_count < var_all && (!var_idx || !CV_IS_MAT(var_idx) || var_idx->cols != var_count) )
        CV_ERROR( CV_StsParseError, "var_idx array is missing or invalid" );

    // read support vectors
    sv_node = cvGetFileNodeByName( fs, svm_node, "support_vectors" );
    if( !sv_node || !CV_NODE_IS_SEQ(sv_node->tag))
        CV_ERROR( CV_StsParseError, "Missing or invalid sequence of support vectors" );

    block_size = MAX( block_size, sv_total*(int)sizeof(CvSVMKernelRow));
    block_size = MAX( block_size, sv_total*2*(int)sizeof(double));
    block_size = MAX( block_size, var_all*(int)sizeof(double));
    CV_CALL( storage = cvCreateMemStorage( block_size ));
    CV_CALL( sv = (float**)cvMemStorageAlloc( storage,
                                sv_total*sizeof(sv[0]) ));

    CV_CALL( cvStartReadSeq( sv_node->data.seq, &reader, 0 ));
    sv_size = var_count*sizeof(sv[0][0]);

    for( i = 0; i < sv_total; i++ )
    {
        CvFileNode* sv_elem = (CvFileNode*)reader.ptr;
        CV_ASSERT( var_count == 1 || (CV_NODE_IS_SEQ(sv_elem->tag) &&
                   sv_elem->data.seq->total == var_count) );

        CV_CALL( sv[i] = (float*)cvMemStorageAlloc( storage, sv_size ));
        CV_CALL( cvReadRawData( fs, sv_elem, sv[i], "f" ));
        CV_NEXT_SEQ_ELEM( sv_node->data.seq->elem_size, reader );
    }

    // read decision functions
    df_count = class_count > 1 ? class_count*(class_count-1)/2 : 1;
    df_node = cvGetFileNodeByName( fs, svm_node, "decision_functions" );
    if( !df_node || !CV_NODE_IS_SEQ(df_node->tag) ||
        df_node->data.seq->total != df_count )
        CV_ERROR( CV_StsParseError, "decision_functions is missing or is not a collection "
                  "or has a wrong number of elements" );

    CV_CALL( df = decision_func = (CvSVMDecisionFunc*)cvAlloc( df_count*sizeof(df[0]) ));
    cvStartReadSeq( df_node->data.seq, &reader, 0 );

    for( i = 0; i < df_count; i++ )
    {
        CvFileNode* df_elem = (CvFileNode*)reader.ptr;
        CvFileNode* alpha_node = cvGetFileNodeByName( fs, df_elem, "alpha" );

        int sv_count = cvReadIntByName( fs, df_elem, "sv_count", -1 );
        if( sv_count <= 0 )
            CV_ERROR( CV_StsParseError, "sv_count is missing or non-positive" );
        df[i].sv_count = sv_count;

        df[i].rho = cvReadRealByName( fs, df_elem, "rho", not_found_dbl );
        if( fabs(df[i].rho - not_found_dbl) < DBL_EPSILON )
            CV_ERROR( CV_StsParseError, "rho is missing" );

        if( !alpha_node )
            CV_ERROR( CV_StsParseError, "alpha is missing in the decision function" );

        CV_CALL( df[i].alpha = (double*)cvMemStorageAlloc( storage,
                                        sv_count*sizeof(df[i].alpha[0])));
        CV_ASSERT( sv_count == 1 || (CV_NODE_IS_SEQ(alpha_node->tag) &&
                   alpha_node->data.seq->total == sv_count) );
        CV_CALL( cvReadRawData( fs, alpha_node, df[i].alpha, "d" ));

        if( class_count > 1 )
        {
            CvFileNode* index_node = cvGetFileNodeByName( fs, df_elem, "index" );
            if( !index_node )
                CV_ERROR( CV_StsParseError, "index is missing in the decision function" );
            CV_CALL( df[i].sv_index = (int*)cvMemStorageAlloc( storage,
                                            sv_count*sizeof(df[i].sv_index[0])));
            CV_ASSERT( sv_count == 1 || (CV_NODE_IS_SEQ(index_node->tag) &&
                   index_node->data.seq->total == sv_count) );
            CV_CALL( cvReadRawData( fs, index_node, df[i].sv_index, "i" ));
        }
        else
            df[i].sv_index = 0;

        CV_NEXT_SEQ_ELEM( df_node->data.seq->elem_size, reader );
    }

    create_kernel();

    __END__;
}

#if 0

static void*
icvCloneSVM( const void* _src )
{
    CvSVMModel* dst = 0;

    CV_FUNCNAME( "icvCloneSVM" );

    __BEGIN__;

    const CvSVMModel* src = (const CvSVMModel*)_src;
    int var_count, class_count;
    int i, sv_total, df_count;
    int sv_size;

    if( !CV_IS_SVM(src) )
        CV_ERROR( !src ? CV_StsNullPtr : CV_StsBadArg, "Input pointer is NULL or invalid" );

    // 0. create initial CvSVMModel structure
    CV_CALL( dst = icvCreateSVM() );
    dst->params = src->params;
    dst->params.weight_labels = 0;
    dst->params.weights = 0;

    dst->var_all = src->var_all;
    if( src->class_labels )
        dst->class_labels = cvCloneMat( src->class_labels );
    if( src->class_weights )
        dst->class_weights = cvCloneMat( src->class_weights );
    if( src->comp_idx )
        dst->comp_idx = cvCloneMat( src->comp_idx );

    var_count = src->comp_idx ? src->comp_idx->cols : src->var_all;
    class_count = src->class_labels ? src->class_labels->cols :
                  src->params.svm_type == CvSVM::ONE_CLASS ? 1 : 0;
    sv_total = dst->sv_total = src->sv_total;
    CV_CALL( dst->storage = cvCreateMemStorage( src->storage->block_size ));
    CV_CALL( dst->sv = (float**)cvMemStorageAlloc( dst->storage,
                                    sv_total*sizeof(dst->sv[0]) ));

    sv_size = var_count*sizeof(dst->sv[0][0]);

    for( i = 0; i < sv_total; i++ )
    {
        CV_CALL( dst->sv[i] = (float*)cvMemStorageAlloc( dst->storage, sv_size ));
        memcpy( dst->sv[i], src->sv[i], sv_size );
    }

    df_count = class_count > 1 ? class_count*(class_count-1)/2 : 1;

    CV_CALL( dst->decision_func = cvAlloc( df_count*sizeof(CvSVMDecisionFunc) ));

    for( i = 0; i < df_count; i++ )
    {
        const CvSVMDecisionFunc *sdf =
            (const CvSVMDecisionFunc*)src->decision_func+i;
        CvSVMDecisionFunc *ddf =
            (CvSVMDecisionFunc*)dst->decision_func+i;
        int sv_count = sdf->sv_count;
        ddf->sv_count = sv_count;
        ddf->rho = sdf->rho;
        CV_CALL( ddf->alpha = (double*)cvMemStorageAlloc( dst->storage,
                                        sv_count*sizeof(ddf->alpha[0])));
        memcpy( ddf->alpha, sdf->alpha, sv_count*sizeof(ddf->alpha[0]));

        if( class_count > 1 )
        {
            CV_CALL( ddf->sv_index = (int*)cvMemStorageAlloc( dst->storage,
                                                sv_count*sizeof(ddf->sv_index[0])));
            memcpy( ddf->sv_index, sdf->sv_index, sv_count*sizeof(ddf->sv_index[0]));
        }
        else
            ddf->sv_index = 0;
    }

    __END__;

    if( cvGetErrStatus() < 0 && dst )
        icvReleaseSVM( &dst );

    return dst;
}

static int icvRegisterSVMType()
{
    CvTypeInfo info;
    memset( &info, 0, sizeof(info) );

    info.flags = 0;
    info.header_size = sizeof( info );
    info.is_instance = icvIsSVM;
    info.release = (CvReleaseFunc)icvReleaseSVM;
    info.read = icvReadSVM;
    info.write = icvWriteSVM;
    info.clone = icvCloneSVM;
    info.type_name = CV_TYPE_NAME_ML_SVM;
    cvRegisterType( &info );

    return 1;
}


static int svm = icvRegisterSVMType();

/* The function trains SVM model with optimal parameters, obtained by using cross-validation.
The parameters to be estimated should be indicated by setting theirs values to FLT_MAX.
The optimal parameters are saved in <model_params> */
CV_IMPL CvStatModel*
cvTrainSVM_CrossValidation( const CvMat* train_data, int tflag,
            const CvMat* responses,
            CvStatModelParams* model_params,
            const CvStatModelParams* cross_valid_params,
            const CvMat* comp_idx,
            const CvMat* sample_idx,
            const CvParamGrid* degree_grid,
            const CvParamGrid* gamma_grid,
            const CvParamGrid* coef_grid,
            const CvParamGrid* C_grid,
            const CvParamGrid* nu_grid,
            const CvParamGrid* p_grid )
{
    CvStatModel* svm = 0;

    CV_FUNCNAME("cvTainSVMCrossValidation");
    __BEGIN__;

    double degree_step = 7,
	       g_step      = 15,
		   coef_step   = 14,
		   C_step      = 20,
		   nu_step     = 5,
		   p_step      = 7; // all steps must be > 1
    double degree_begin = 0.01, degree_end = 2;
    double g_begin      = 1e-5, g_end      = 0.5;
    double coef_begin   = 0.1,  coef_end   = 300;
    double C_begin      = 0.1,  C_end      = 6000;
    double nu_begin     = 0.01,  nu_end    = 0.4;
    double p_begin      = 0.01, p_end      = 100;

    double rate = 0, gamma = 0, C = 0, degree = 0, coef = 0, p = 0, nu = 0;

	double best_rate    = 0;
    double best_degree  = degree_begin;
    double best_gamma   = g_begin;
    double best_coef    = coef_begin;
	double best_C       = C_begin;
	double best_nu      = nu_begin;
    double best_p       = p_begin;

    CvSVMModelParams svm_params, *psvm_params;
    CvCrossValidationParams* cv_params = (CvCrossValidationParams*)cross_valid_params;
    int svm_type, kernel;
    int is_regression;

    if( !model_params )
        CV_ERROR( CV_StsBadArg, "" );
    if( !cv_params )
        CV_ERROR( CV_StsBadArg, "" );

    svm_params = *(CvSVMModelParams*)model_params;
    psvm_params = (CvSVMModelParams*)model_params;
    svm_type = svm_params.svm_type;
    kernel = svm_params.kernel_type;

    svm_params.degree = svm_params.degree > 0 ? svm_params.degree : 1;
    svm_params.gamma = svm_params.gamma > 0 ? svm_params.gamma : 1;
    svm_params.coef0 = svm_params.coef0 > 0 ? svm_params.coef0 : 1e-6;
    svm_params.C = svm_params.C > 0 ? svm_params.C : 1;
    svm_params.nu = svm_params.nu > 0 ? svm_params.nu : 1;
    svm_params.p = svm_params.p > 0 ? svm_params.p : 1;

    if( degree_grid )
    {
        if( !(degree_grid->max_val == 0 && degree_grid->min_val == 0 &&
              degree_grid->step == 0) )
        {
            if( degree_grid->min_val > degree_grid->max_val )
                CV_ERROR( CV_StsBadArg,
                "low bound of grid should be less then the upper one");
            if( degree_grid->step <= 1 )
                CV_ERROR( CV_StsBadArg, "grid step should be greater 1" );
            degree_begin = degree_grid->min_val;
            degree_end   = degree_grid->max_val;
            degree_step  = degree_grid->step;
        }
    }
    else
        degree_begin = degree_end = svm_params.degree;

    if( gamma_grid )
    {
        if( !(gamma_grid->max_val == 0 && gamma_grid->min_val == 0 &&
              gamma_grid->step == 0) )
        {
            if( gamma_grid->min_val > gamma_grid->max_val )
                CV_ERROR( CV_StsBadArg,
                "low bound of grid should be less then the upper one");
            if( gamma_grid->step <= 1 )
                CV_ERROR( CV_StsBadArg, "grid step should be greater 1" );
            g_begin = gamma_grid->min_val;
            g_end   = gamma_grid->max_val;
            g_step  = gamma_grid->step;
        }
    }
    else
        g_begin = g_end = svm_params.gamma;

    if( coef_grid )
    {
        if( !(coef_grid->max_val == 0 && coef_grid->min_val == 0 &&
              coef_grid->step == 0) )
        {
            if( coef_grid->min_val > coef_grid->max_val )
                CV_ERROR( CV_StsBadArg,
                "low bound of grid should be less then the upper one");
            if( coef_grid->step <= 1 )
                CV_ERROR( CV_StsBadArg, "grid step should be greater 1" );
            coef_begin = coef_grid->min_val;
            coef_end   = coef_grid->max_val;
            coef_step  = coef_grid->step;
        }
    }
    else
        coef_begin = coef_end = svm_params.coef0;

    if( C_grid )
    {
        if( !(C_grid->max_val == 0 && C_grid->min_val == 0 && C_grid->step == 0))
        {
            if( C_grid->min_val > C_grid->max_val )
                CV_ERROR( CV_StsBadArg,
                "low bound of grid should be less then the upper one");
            if( C_grid->step <= 1 )
                CV_ERROR( CV_StsBadArg, "grid step should be greater 1" );
            C_begin = C_grid->min_val;
            C_end   = C_grid->max_val;
            C_step  = C_grid->step;
        }
    }
    else
        C_begin = C_end = svm_params.C;

    if( nu_grid )
    {
        if(!(nu_grid->max_val == 0 && nu_grid->min_val == 0 && nu_grid->step==0))
        {
            if( nu_grid->min_val > nu_grid->max_val )
                CV_ERROR( CV_StsBadArg,
                "low bound of grid should be less then the upper one");
            if( nu_grid->step <= 1 )
                CV_ERROR( CV_StsBadArg, "grid step should be greater 1" );
            nu_begin = nu_grid->min_val;
            nu_end   = nu_grid->max_val;
            nu_step  = nu_grid->step;
        }
    }
    else
        nu_begin = nu_end = svm_params.nu;

    if( p_grid )
    {
        if( !(p_grid->max_val == 0 && p_grid->min_val == 0 && p_grid->step == 0))
        {
            if( p_grid->min_val > p_grid->max_val )
                CV_ERROR( CV_StsBadArg,
                "low bound of grid should be less then the upper one");
            if( p_grid->step <= 1 )
                CV_ERROR( CV_StsBadArg, "grid step should be greater 1" );
            p_begin = p_grid->min_val;
            p_end   = p_grid->max_val;
            p_step  = p_grid->step;
        }
    }
    else
        p_begin = p_end = svm_params.p;

    // these parameters are not used:
    if( kernel != CvSVM::POLY )
        degree_begin = degree_end = svm_params.degree;

   if( kernel == CvSVM::LINEAR )
        g_begin = g_end = svm_params.gamma;

    if( kernel != CvSVM::POLY && kernel != CvSVM::SIGMOID )
        coef_begin = coef_end = svm_params.coef0;

    if( svm_type == CvSVM::NU_SVC || svm_type == CvSVM::ONE_CLASS )
        C_begin = C_end = svm_params.C;

    if( svm_type == CvSVM::C_SVC || svm_type == CvSVM::EPS_SVR )
        nu_begin = nu_end = svm_params.nu;

    if( svm_type != CvSVM::EPS_SVR )
        p_begin = p_end = svm_params.p;

    is_regression = cv_params->is_regression;
    best_rate = is_regression ? FLT_MAX : 0;

    assert( g_step > 1 && degree_step > 1 && coef_step > 1);
    assert( p_step > 1 && C_step > 1 && nu_step > 1 );

    for( degree = degree_begin; degree <= degree_end; degree *= degree_step )
    {
      svm_params.degree = degree;
      //printf("degree = %.3f\n", degree );
      for( gamma= g_begin; gamma <= g_end; gamma *= g_step )
      {
        svm_params.gamma = gamma;
        //printf("   gamma = %.3f\n", gamma );
        for( coef = coef_begin; coef <= coef_end; coef *= coef_step )
        {
          svm_params.coef0 = coef;
          //printf("      coef = %.3f\n", coef );
          for( C = C_begin; C <= C_end; C *= C_step )
          {
            svm_params.C = C;
            //printf("         C = %.3f\n", C );
            for( nu = nu_begin; nu <= nu_end; nu *= nu_step )
            {
              svm_params.nu = nu;
              //printf("            nu = %.3f\n", nu );
              for( p = p_begin; p <= p_end; p *= p_step )
              {
                int well;
                svm_params.p = p;
                //printf("               p = %.3f\n", p );

                CV_CALL(rate = cvCrossValidation( train_data, tflag, responses, &cvTrainSVM,
                    cross_valid_params, (CvStatModelParams*)&svm_params, comp_idx, sample_idx ));

                well =  rate > best_rate && !is_regression || rate < best_rate && is_regression;
                if( well || (rate == best_rate && C < best_C) )
                {
                    best_rate   = rate;
                    best_degree = degree;
                    best_gamma  = gamma;
                    best_coef   = coef;
                    best_C      = C;
                    best_nu     = nu;
                    best_p      = p;
                }
                //printf("                  rate = %.2f\n", rate );
              }
            }
          }
        }
      }
    }
    //printf("The best:\nrate = %.2f%% degree = %f gamma = %f coef = %f c = %f nu = %f p = %f\n",
      //  best_rate, best_degree, best_gamma, best_coef, best_C, best_nu, best_p );

    psvm_params->C      = best_C;
    psvm_params->nu     = best_nu;
    psvm_params->p      = best_p;
    psvm_params->gamma  = best_gamma;
    psvm_params->degree = best_degree;
    psvm_params->coef0  = best_coef;

    CV_CALL(svm = cvTrainSVM( train_data, tflag, responses, model_params, comp_idx, sample_idx ));

    __END__;

    return svm;
}

#endif

/* End of file. */

