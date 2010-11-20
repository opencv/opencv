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
*                          K-Nearest Neighbors Classifier                                *
\****************************************************************************************/

// k Nearest Neighbors
CvKNearest::CvKNearest()
{
    samples = 0;
    clear();
}


CvKNearest::~CvKNearest()
{
    clear();
}


CvKNearest::CvKNearest( const CvMat* _train_data, const CvMat* _responses,
                        const CvMat* _sample_idx, bool _is_regression, int _max_k )
{
    samples = 0;
    train( _train_data, _responses, _sample_idx, _is_regression, _max_k, false );
}


void CvKNearest::clear()
{
    while( samples )
    {
        CvVectors* next_samples = samples->next;
        cvFree( &samples->data.fl );
        cvFree( &samples );
        samples = next_samples;
    }
    var_count = 0;
    total = 0;
    max_k = 0;
}


int CvKNearest::get_max_k() const { return max_k; }

int CvKNearest::get_var_count() const { return var_count; }

bool CvKNearest::is_regression() const { return regression; }

int CvKNearest::get_sample_count() const { return total; }

bool CvKNearest::train( const CvMat* _train_data, const CvMat* _responses,
                        const CvMat* _sample_idx, bool _is_regression,
                        int _max_k, bool _update_base )
{
    bool ok = false;
    CvMat* responses = 0;

    CV_FUNCNAME( "CvKNearest::train" );

    __BEGIN__;

    CvVectors* _samples;
    float** _data;
    int _count, _dims, _dims_all, _rsize;

    if( !_update_base )
        clear();

    // Prepare training data and related parameters.
    // Treat categorical responses as ordered - to prevent class label compression and
    // to enable entering new classes in the updates
    CV_CALL( cvPrepareTrainData( "CvKNearest::train", _train_data, CV_ROW_SAMPLE,
        _responses, CV_VAR_ORDERED, 0, _sample_idx, true, (const float***)&_data,
        &_count, &_dims, &_dims_all, &responses, 0, 0 ));

    if( _update_base && _dims != var_count )
        CV_ERROR( CV_StsBadArg, "The newly added data have different dimensionality" );

    if( !_update_base )
    {
        if( _max_k < 1 )
            CV_ERROR( CV_StsOutOfRange, "max_k must be a positive number" );

        regression = _is_regression;
        var_count = _dims;
        max_k = _max_k;
    }

    _rsize = _count*sizeof(float);
    CV_CALL( _samples = (CvVectors*)cvAlloc( sizeof(*_samples) + _rsize ));
    _samples->next = samples;
    _samples->type = CV_32F;
    _samples->data.fl = _data;
    _samples->count = _count;
    total += _count;

    samples = _samples;
    memcpy( _samples + 1, responses->data.fl, _rsize );

    ok = true;

    __END__;

    return ok;
}



void CvKNearest::find_neighbors_direct( const CvMat* _samples, int k, int start, int end,
                    float* neighbor_responses, const float** neighbors, float* dist ) const
{
    int i, j, count = end - start, k1 = 0, k2 = 0, d = var_count;
    CvVectors* s = samples;

    for( ; s != 0; s = s->next )
    {
        int n = s->count;
        for( j = 0; j < n; j++ )
        {
            for( i = 0; i < count; i++ )
            {
                double sum = 0;
                Cv32suf si;
                const float* v = s->data.fl[j];
                const float* u = (float*)(_samples->data.ptr + _samples->step*(start + i));
                Cv32suf* dd = (Cv32suf*)(dist + i*k);
                float* nr;
                const float** nn;
                int t, ii, ii1;

                for( t = 0; t <= d - 4; t += 4 )
                {
                    double t0 = u[t] - v[t], t1 = u[t+1] - v[t+1];
                    double t2 = u[t+2] - v[t+2], t3 = u[t+3] - v[t+3];
                    sum += t0*t0 + t1*t1 + t2*t2 + t3*t3;
                }

                for( ; t < d; t++ )
                {
                    double t0 = u[t] - v[t];
                    sum += t0*t0;
                }

                si.f = (float)sum;
                for( ii = k1-1; ii >= 0; ii-- )
                    if( si.i > dd[ii].i )
                        break;
                if( ii >= k-1 )
                    continue;

                nr = neighbor_responses + i*k;
                nn = neighbors ? neighbors + (start + i)*k : 0;
                for( ii1 = k2 - 1; ii1 > ii; ii1-- )
                {
                    dd[ii1+1].i = dd[ii1].i;
                    nr[ii1+1] = nr[ii1];
                    if( nn ) nn[ii1+1] = nn[ii1];
                }
                dd[ii+1].i = si.i;
                nr[ii+1] = ((float*)(s + 1))[j];
                if( nn )
                    nn[ii+1] = v;
            }
            k1 = MIN( k1+1, k );
            k2 = MIN( k1, k-1 );
        }
    }
}


float CvKNearest::write_results( int k, int k1, int start, int end,
    const float* neighbor_responses, const float* dist,
    CvMat* _results, CvMat* _neighbor_responses,
    CvMat* _dist, Cv32suf* sort_buf ) const
{
    float result = 0.f;
    int i, j, j1, count = end - start;
    double inv_scale = 1./k1;
    int rstep = _results && !CV_IS_MAT_CONT(_results->type) ? _results->step/sizeof(result) : 1;

    for( i = 0; i < count; i++ )
    {
        const Cv32suf* nr = (const Cv32suf*)(neighbor_responses + i*k);
        float* dst;
        float r;
        if( _results || start+i == 0 )
        {
            if( regression )
            {
                double s = 0;
                for( j = 0; j < k1; j++ )
                    s += nr[j].f;
                r = (float)(s*inv_scale);
            }
            else
            {
                int prev_start = 0, best_count = 0, cur_count;
                Cv32suf best_val;

                for( j = 0; j < k1; j++ )
                    sort_buf[j].i = nr[j].i;

                for( j = k1-1; j > 0; j-- )
                {
                    bool swap_fl = false;
                    for( j1 = 0; j1 < j; j1++ )
                        if( sort_buf[j1].i > sort_buf[j1+1].i )
                        {
                            int t;
                            CV_SWAP( sort_buf[j1].i, sort_buf[j1+1].i, t );
                            swap_fl = true;
                        }
                    if( !swap_fl )
                        break;
                }

                best_val.i = 0;
                for( j = 1; j <= k1; j++ )
                    if( j == k1 || sort_buf[j].i != sort_buf[j-1].i )
                    {
                        cur_count = j - prev_start;
                        if( best_count < cur_count )
                        {
                            best_count = cur_count;
                            best_val.i = sort_buf[j-1].i;
                        }
                        prev_start = j;
                    }
                r = best_val.f;
            }

            if( start+i == 0 )
                result = r;

            if( _results )
                _results->data.fl[(start + i)*rstep] = r;
        }

        if( _neighbor_responses )
        {
            dst = (float*)(_neighbor_responses->data.ptr +
                (start + i)*_neighbor_responses->step);
            for( j = 0; j < k1; j++ )
                dst[j] = nr[j].f;
            for( ; j < k; j++ )
                dst[j] = 0.f;
        }

        if( _dist )
        {
            dst = (float*)(_dist->data.ptr + (start + i)*_dist->step);
            for( j = 0; j < k1; j++ )
                dst[j] = dist[j + i*k];
            for( ; j < k; j++ )
                dst[j] = 0.f;
        }
    }

    return result;
}



float CvKNearest::find_nearest( const CvMat* _samples, int k, CvMat* _results,
    const float** _neighbors, CvMat* _neighbor_responses, CvMat* _dist ) const
{
    float result = 0.f;
    bool local_alloc = false;
    float* buf = 0;
    const int max_blk_count = 128, max_buf_sz = 1 << 12;

    CV_FUNCNAME( "CvKNearest::find_nearest" );

    __BEGIN__;

    int i, count, count_scale, blk_count0, blk_count = 0, buf_sz, k1;

    if( !samples )
        CV_ERROR( CV_StsError, "The search tree must be constructed first using train method" );

    if( !CV_IS_MAT(_samples) ||
        CV_MAT_TYPE(_samples->type) != CV_32FC1 ||
        _samples->cols != var_count )
        CV_ERROR( CV_StsBadArg, "Input samples must be floating-point matrix (<num_samples>x<var_count>)" );

    if( _results && (!CV_IS_MAT(_results) ||
        (_results->cols != 1 && _results->rows != 1) ||
        _results->cols + _results->rows - 1 != _samples->rows) )
        CV_ERROR( CV_StsBadArg,
        "The results must be 1d vector containing as much elements as the number of samples" );

    if( _results && CV_MAT_TYPE(_results->type) != CV_32FC1 &&
        (CV_MAT_TYPE(_results->type) != CV_32SC1 || regression))
        CV_ERROR( CV_StsUnsupportedFormat,
        "The results must be floating-point or integer (in case of classification) vector" );

    if( k < 1 || k > max_k )
        CV_ERROR( CV_StsOutOfRange, "k must be within 1..max_k range" );

    if( _neighbor_responses )
    {
        if( !CV_IS_MAT(_neighbor_responses) || CV_MAT_TYPE(_neighbor_responses->type) != CV_32FC1 ||
            _neighbor_responses->rows != _samples->rows || _neighbor_responses->cols != k )
            CV_ERROR( CV_StsBadArg,
            "The neighbor responses (if present) must be floating-point matrix of <num_samples> x <k> size" );
    }

    if( _dist )
    {
        if( !CV_IS_MAT(_dist) || CV_MAT_TYPE(_dist->type) != CV_32FC1 ||
            _dist->rows != _samples->rows || _dist->cols != k )
            CV_ERROR( CV_StsBadArg,
            "The distances from the neighbors (if present) must be floating-point matrix of <num_samples> x <k> size" );
    }

    count = _samples->rows;
    count_scale = k*2*sizeof(float);
    blk_count0 = MIN( count, max_blk_count );
    buf_sz = MIN( blk_count0 * count_scale, max_buf_sz );
    blk_count0 = MAX( buf_sz/count_scale, 1 );
    blk_count0 += blk_count0 % 2;
    blk_count0 = MIN( blk_count0, count );
    buf_sz = blk_count0 * count_scale + k*sizeof(float);
    k1 = get_sample_count();
    k1 = MIN( k1, k );

    if( buf_sz <= CV_MAX_LOCAL_SIZE )
    {
        buf = (float*)cvStackAlloc( buf_sz );
        local_alloc = true;
    }
    else
        CV_CALL( buf = (float*)cvAlloc( buf_sz ));

    for( i = 0; i < count; i += blk_count )
    {
        blk_count = MIN( count - i, blk_count0 );
        float* neighbor_responses = buf;
        float* dist = buf + blk_count*k;
        Cv32suf* sort_buf = (Cv32suf*)(dist + blk_count*k);

        find_neighbors_direct( _samples, k, i, i + blk_count,
                    neighbor_responses, _neighbors, dist );

        float r = write_results( k, k1, i, i + blk_count, neighbor_responses, dist,
                                 _results, _neighbor_responses, _dist, sort_buf );
        if( i == 0 )
            result = r;
    }

    __END__;

    if( !local_alloc )
        cvFree( &buf );

    return result;
}


using namespace cv;

CvKNearest::CvKNearest( const Mat& _train_data, const Mat& _responses,
                       const Mat& _sample_idx, bool _is_regression, int _max_k )
{
    samples = 0;
    train(_train_data, _responses, _sample_idx, _is_regression, _max_k, false );
}

bool CvKNearest::train( const Mat& _train_data, const Mat& _responses,
                        const Mat& _sample_idx, bool _is_regression,
                        int _max_k, bool _update_base )
{
    CvMat tdata = _train_data, responses = _responses, sidx = _sample_idx;
    
    return train(&tdata, &responses, sidx.data.ptr ? &sidx : 0, _is_regression, _max_k, _update_base );
}


float CvKNearest::find_nearest( const Mat& _samples, int k, Mat* _results,
                                const float** _neighbors, Mat* _neighbor_responses,
                                Mat* _dist ) const
{
    CvMat s = _samples, results, *presults = 0, nresponses, *pnresponses = 0, dist, *pdist = 0;
    
    if( _results )
    {
        if(!(_results->data && (_results->type() == CV_32F ||
            (_results->type() == CV_32S && regression)) &&
             (_results->cols == 1 || _results->rows == 1) &&
             _results->cols + _results->rows - 1 == _samples.rows) )
            _results->create(_samples.rows, 1, CV_32F);
        presults = &(results = *_results);
    }
    
    if( _neighbor_responses )
    {
        if(!(_neighbor_responses->data && _neighbor_responses->type() == CV_32F &&
             _neighbor_responses->cols == k && _neighbor_responses->rows == _samples.rows) )
            _neighbor_responses->create(_samples.rows, k, CV_32F);
        pnresponses = &(nresponses = *_neighbor_responses);
    }
    
    if( _dist )
    {
        if(!(_dist->data && _dist->type() == CV_32F &&
             _dist->cols == k && _dist->rows == _samples.rows) )
            _dist->create(_samples.rows, k, CV_32F);
        pdist = &(dist = *_dist);
    }
    
    return find_nearest(&s, k, presults, _neighbors, pnresponses, pdist );
}


float CvKNearest::find_nearest( const cv::Mat& samples, int k, CV_OUT cv::Mat& results,
                                CV_OUT cv::Mat& neighborResponses, CV_OUT cv::Mat& dists) const
{
    return find_nearest(samples, k, &results, 0, &neighborResponses, &dists);
}

/* End of file */

