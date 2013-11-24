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

#define LN2PI 1.837877f
#define BIG_FLT 1.e+10f


#define _CV_ERGODIC 1
#define _CV_CAUSAL 2

#define _CV_LAST_STATE 1
#define _CV_BEST_STATE 2


//*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: _cvCreateObsInfo
//    Purpose: The function allocates memory for CvImgObsInfo structure
//             and its inner stuff
//    Context:
//    Parameters: obs_info - addres of pointer to CvImgObsInfo structure
//                num_hor_obs - number of horizontal observation vectors
//                num_ver_obs - number of horizontal observation vectors
//                obs_size - length of observation vector
//
//    Returns: error status
//
//    Notes:
//F*/
static CvStatus CV_STDCALL icvCreateObsInfo(  CvImgObsInfo** obs_info,
                                           CvSize num_obs, int obs_size )
{
    int total = num_obs.height * num_obs.width;

    CvImgObsInfo* obs = (CvImgObsInfo*)cvAlloc( sizeof( CvImgObsInfo) );

    obs->obs_x = num_obs.width;
    obs->obs_y = num_obs.height;

    obs->obs = (float*)cvAlloc( total * obs_size * sizeof(float) );

    obs->state = (int*)cvAlloc( 2 * total * sizeof(int) );
    obs->mix = (int*)cvAlloc( total * sizeof(int) );

    obs->obs_size = obs_size;

    obs_info[0] = obs;

    return CV_NO_ERR;
}

static CvStatus CV_STDCALL icvReleaseObsInfo( CvImgObsInfo** p_obs_info )
{
    CvImgObsInfo* obs_info = p_obs_info[0];

    cvFree( &(obs_info->obs) );
    cvFree( &(obs_info->mix) );
    cvFree( &(obs_info->state) );
    cvFree( &(obs_info) );

    p_obs_info[0] = NULL;

    return CV_NO_ERR;
}


//*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCreate2DHMM
//    Purpose: The function allocates memory for 2-dimensional embedded HMM model
//             and its inner stuff
//    Context:
//    Parameters: hmm - addres of pointer to CvEHMM structure
//                state_number - array of hmm sizes (size of array == state_number[0]+1 )
//                num_mix - number of gaussian mixtures in low-level HMM states
//                          size of array is defined by previous array values
//                obs_size - length of observation vectors
//
//    Returns: error status
//
//    Notes: state_number[0] - number of states in external HMM.
//           state_number[i] - number of states in embedded HMM
//
//           example for face recognition: state_number = { 5 3 6 6 6 3 },
//                                         length of num_mix array = 3+6+6+6+3 = 24//
//
//F*/
static CvStatus CV_STDCALL icvCreate2DHMM( CvEHMM** this_hmm,
                                         int* state_number, int* num_mix, int obs_size )
{
    int i;
    int real_states = 0;

    CvEHMMState* all_states;
    CvEHMM* hmm;
    int total_mix = 0;
    float* pointers;

    //compute total number of states of all level in 2d EHMM
    for( i = 1; i <= state_number[0]; i++ )
    {
        real_states += state_number[i];
    }

    /* allocate memory for all hmms (from all levels) */
    hmm = (CvEHMM*)cvAlloc( (state_number[0] + 1) * sizeof(CvEHMM) );

    /* set number of superstates */
    hmm[0].num_states = state_number[0];
    hmm[0].level = 1;

    /* allocate memory for all states */
    all_states = (CvEHMMState *)cvAlloc( real_states * sizeof( CvEHMMState ) );

    /* assign number of mixtures */
    for( i = 0; i < real_states; i++ )
    {
        all_states[i].num_mix = num_mix[i];
    }

    /* compute size of inner of all real states */
    for( i = 0; i < real_states; i++ )
    {
        total_mix += num_mix[i];
    }
    /* allocate memory for states stuff */
    pointers = (float*)cvAlloc( total_mix * (2/*for mu invvar */ * obs_size +
                                 2/*for weight and log_var_val*/ ) * sizeof( float) );

    /* organize memory */
    for( i = 0; i < real_states; i++ )
    {
        all_states[i].mu      = pointers; pointers += num_mix[i] * obs_size;
        all_states[i].inv_var = pointers; pointers += num_mix[i] * obs_size;

        all_states[i].log_var_val = pointers; pointers += num_mix[i];
        all_states[i].weight      = pointers; pointers += num_mix[i];
    }

    /* set pointer to embedded hmm array */
    hmm->u.ehmm = hmm + 1;

    for( i = 0; i < hmm[0].num_states; i++ )
    {
        hmm[i+1].u.state = all_states;
        all_states += state_number[i+1];
        hmm[i+1].num_states = state_number[i+1];
    }

    for( i = 0; i <= state_number[0]; i++ )
    {
        hmm[i].transP = icvCreateMatrix_32f( hmm[i].num_states, hmm[i].num_states );
        hmm[i].obsProb = NULL;
        hmm[i].level = i ? 0 : 1;
    }

    /* if all ok - return pointer */
    *this_hmm = hmm;
    return CV_NO_ERR;
}

static CvStatus CV_STDCALL icvRelease2DHMM( CvEHMM** phmm )
{
    CvEHMM* hmm = phmm[0];
    int i;
    for( i = 0; i < hmm[0].num_states + 1; i++ )
    {
        icvDeleteMatrix( hmm[i].transP );
    }

    if (hmm->obsProb != NULL)
    {
        int* tmp = ((int*)(hmm->obsProb)) - 3;
        cvFree( &(tmp)  );
    }

    cvFree( &(hmm->u.ehmm->u.state->mu) );
    cvFree( &(hmm->u.ehmm->u.state) );


    /* free hmm structures */
    cvFree( phmm );

    phmm[0] = NULL;

    return CV_NO_ERR;
}

/* distance between 2 vectors */
static float icvSquareDistance( CvVect32f v1, CvVect32f v2, int len )
{
    int i;
    double dist0 = 0;
    double dist1 = 0;

    for( i = 0; i <= len - 4; i += 4 )
    {
        double t0 = v1[i] - v2[i];
        double t1 = v1[i+1] - v2[i+1];
        dist0 += t0*t0;
        dist1 += t1*t1;

        t0 = v1[i+2] - v2[i+2];
        t1 = v1[i+3] - v2[i+3];
        dist0 += t0*t0;
        dist1 += t1*t1;
    }

    for( ; i < len; i++ )
    {
        double t0 = v1[i] - v2[i];
        dist0 += t0*t0;
    }

    return (float)(dist0 + dist1);
}

/*can be used in CHMM & DHMM */
static CvStatus CV_STDCALL
icvUniformImgSegm(  CvImgObsInfo* obs_info, CvEHMM* hmm )
{
#if 1
    /* implementation is very bad */
    int  i, j, counter = 0;
    CvEHMMState* first_state;
    float inv_x = 1.f/obs_info->obs_x;
    float inv_y = 1.f/obs_info->obs_y;

    /* check arguments */
    if ( !obs_info || !hmm ) return CV_NULLPTR_ERR;

    first_state = hmm->u.ehmm->u.state;

    for (i = 0; i < obs_info->obs_y; i++)
    {
        //bad line (division )
        int superstate = (int)((i * hmm->num_states)*inv_y);/* /obs_info->obs_y; */

        int index = (int)(hmm->u.ehmm[superstate].u.state - first_state);

        for (j = 0; j < obs_info->obs_x; j++, counter++)
        {
            int state = (int)((j * hmm->u.ehmm[superstate].num_states)* inv_x); /* / obs_info->obs_x; */

            obs_info->state[2 * counter] = superstate;
            obs_info->state[2 * counter + 1] = state + index;
        }
    }
#else
    //this is not ready yet

    int i,j,k,m;
    CvEHMMState* first_state = hmm->u.ehmm->u.state;

    /* check bad arguments */
    if ( hmm->num_states > obs_info->obs_y ) return CV_BADSIZE_ERR;

    //compute vertical subdivision
    float row_per_state = (float)obs_info->obs_y / hmm->num_states;
    float col_per_state[1024]; /* maximum 1024 superstates */

    //for every horizontal band compute subdivision
    for( i = 0; i < hmm->num_states; i++ )
    {
        CvEHMM* ehmm = &(hmm->u.ehmm[i]);
        col_per_state[i] = (float)obs_info->obs_x / ehmm->num_states;
    }

    //compute state bounds
    int ss_bound[1024];
    for( i = 0; i < hmm->num_states - 1; i++ )
    {
        ss_bound[i] = floor( row_per_state * ( i+1 ) );
    }
    ss_bound[hmm->num_states - 1] = obs_info->obs_y;

    //work inside every superstate

    int row = 0;

    for( i = 0; i < hmm->num_states; i++ )
    {
        CvEHMM* ehmm = &(hmm->u.ehmm[i]);
        int index = ehmm->u.state - first_state;

        //calc distribution in superstate
        int es_bound[1024];
        for( j = 0; j < ehmm->num_states - 1; j++ )
        {
            es_bound[j] = floor( col_per_state[i] * ( j+1 ) );
        }
        es_bound[ehmm->num_states - 1] = obs_info->obs_x;

        //assign states to first row of superstate
        int col = 0;
        for( j = 0; j < ehmm->num_states; j++ )
        {
            for( k = col; k < es_bound[j]; k++, col++ )
            {
                obs_info->state[row * obs_info->obs_x + 2 * k] = i;
                obs_info->state[row * obs_info->obs_x + 2 * k + 1] = j + index;
            }
            col = es_bound[j];
        }

        //copy the same to other rows of superstate
        for( m = row; m < ss_bound[i]; m++ )
        {
            memcpy( &(obs_info->state[m * obs_info->obs_x * 2]),
                    &(obs_info->state[row * obs_info->obs_x * 2]), obs_info->obs_x * 2 * sizeof(int) );
        }

        row = ss_bound[i];
    }

#endif

    return CV_NO_ERR;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: InitMixSegm
//    Purpose: The function implements the mixture segmentation of the states of the
//             embedded HMM
//    Context: used with the Viterbi training of the embedded HMM
//             Function uses K-Means algorithm for clustering
//
//    Parameters:  obs_info_array - array of pointers to image observations
//                 num_img - length of above array
//                 hmm - pointer to HMM structure
//
//    Returns: error status
//
//    Notes:
//F*/
static CvStatus CV_STDCALL
icvInitMixSegm( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm )
{
    int  k, i, j;
    int* num_samples; /* number of observations in every state */
    int* counter;     /* array of counters for every state */

    int**  a_class;   /* for every state - characteristic array */

    CvVect32f** samples; /* for every state - pointer to observation vectors */
    int***  samples_mix;   /* for every state - array of pointers to vectors mixtures */

    CvTermCriteria criteria = cvTermCriteria( CV_TERMCRIT_EPS|CV_TERMCRIT_ITER,
                                              1000,    /* iter */
                                              0.01f ); /* eps  */

    int total = 0;

    CvEHMMState* first_state = hmm->u.ehmm->u.state;

    for( i = 0 ; i < hmm->num_states; i++ )
    {
        total += hmm->u.ehmm[i].num_states;
    }

    /* for every state integer is allocated - number of vectors in state */
    num_samples = (int*)cvAlloc( total * sizeof(int) );

    /* integer counter is allocated for every state */
    counter = (int*)cvAlloc( total * sizeof(int) );

    samples = (CvVect32f**)cvAlloc( total * sizeof(CvVect32f*) );
    samples_mix = (int***)cvAlloc( total * sizeof(int**) );

    /* clear */
    memset( num_samples, 0 , total*sizeof(int) );
    memset( counter, 0 , total*sizeof(int) );


    /* for every state the number of vectors which belong to it is computed (smth. like histogram) */
    for (k = 0; k < num_img; k++)
    {
        CvImgObsInfo* obs = obs_info_array[k];
        int count = 0;

        for (i = 0; i < obs->obs_y; i++)
        {
            for (j = 0; j < obs->obs_x; j++, count++)
            {
                int state = obs->state[ 2 * count + 1];
                num_samples[state] += 1;
            }
        }
    }

    /* for every state int* is allocated */
    a_class = (int**)cvAlloc( total*sizeof(int*) );

    for (i = 0; i < total; i++)
    {
        a_class[i] = (int*)cvAlloc( num_samples[i] * sizeof(int) );
        samples[i] = (CvVect32f*)cvAlloc( num_samples[i] * sizeof(CvVect32f) );
        samples_mix[i] = (int**)cvAlloc( num_samples[i] * sizeof(int*) );
    }

    /* for every state vectors which belong to state are gathered */
    for (k = 0; k < num_img; k++)
    {
        CvImgObsInfo* obs = obs_info_array[k];
        int num_obs = ( obs->obs_x ) * ( obs->obs_y );
        float* vector = obs->obs;

        for (i = 0; i < num_obs; i++, vector+=obs->obs_size )
        {
            int state = obs->state[2*i+1];

            samples[state][counter[state]] = vector;
            samples_mix[state][counter[state]] = &(obs->mix[i]);
            counter[state]++;
        }
    }

    /* clear counters */
    memset( counter, 0, total*sizeof(int) );

    /* do the actual clustering using the K Means algorithm */
    for (i = 0; i < total; i++)
    {
        if ( first_state[i].num_mix == 1)
        {
            for (k = 0; k < num_samples[i]; k++)
            {
                /* all vectors belong to one mixture */
                a_class[i][k] = 0;
            }
        }
        else if( num_samples[i] )
        {
            /* clusterize vectors  */
            cvKMeans( first_state[i].num_mix, samples[i], num_samples[i],
                      obs_info_array[0]->obs_size, criteria, a_class[i] );
        }
    }

    /* for every vector number of mixture is assigned */
    for( i = 0; i < total; i++ )
    {
        for (j = 0; j < num_samples[i]; j++)
        {
            samples_mix[i][j][0] = a_class[i][j];
        }
    }

    for (i = 0; i < total; i++)
    {
        cvFree( &(a_class[i]) );
        cvFree( &(samples[i]) );
        cvFree( &(samples_mix[i]) );
    }

    cvFree( &a_class );
    cvFree( &samples );
    cvFree( &samples_mix );
    cvFree( &counter );
    cvFree( &num_samples );

    return CV_NO_ERR;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: ComputeUniModeGauss
//    Purpose: The function computes the Gaussian pdf for a sample vector
//    Context:
//    Parameters:  obsVeq - pointer to the sample vector
//                 mu - pointer to the mean vector of the Gaussian pdf
//                 var - pointer to the variance vector of the Gaussian pdf
//                 VecSize - the size of sample vector
//
//    Returns: the pdf of the sample vector given the specified Gaussian
//
//    Notes:
//F*/
/*static float icvComputeUniModeGauss(CvVect32f vect, CvVect32f mu,
                              CvVect32f inv_var, float log_var_val, int vect_size)
{
    int n;
    double tmp;
    double prob;

    prob = -log_var_val;

    for (n = 0; n < vect_size; n++)
    {
        tmp = (vect[n] - mu[n]) * inv_var[n];
        prob = prob - tmp * tmp;
   }
   //prob *= 0.5f;

   return (float)prob;
}*/

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: ComputeGaussMixture
//    Purpose: The function computes the mixture Gaussian pdf of a sample vector.
//    Context:
//    Parameters:  obsVeq - pointer to the sample vector
//                 mu  - two-dimensional pointer to the mean vector of the Gaussian pdf;
//                       the first dimension is indexed over the number of mixtures and
//                       the second dimension is indexed along the size of the mean vector
//                 var - two-dimensional pointer to the variance vector of the Gaussian pdf;
//                       the first dimension is indexed over the number of mixtures and
//                       the second dimension is indexed along the size of the variance vector
//                 VecSize - the size of sample vector
//                 weight - pointer to the wights of the Gaussian mixture
//                 NumMix - the number of Gaussian mixtures
//
//    Returns: the pdf of the sample vector given the specified Gaussian mixture.
//
//    Notes:
//F*/
/* Calculate probability of observation at state in logarithmic scale*/
/*static float
icvComputeGaussMixture( CvVect32f vect, float* mu,
                        float* inv_var, float* log_var_val,
                        int vect_size, float* weight, int num_mix )
{
    double prob, l_prob;

    prob = 0.0f;

    if (num_mix == 1)
    {
        return icvComputeUniModeGauss( vect, mu, inv_var, log_var_val[0], vect_size);
    }
    else
    {
        int m;
        for (m = 0; m < num_mix; m++)
        {
            if ( weight[m] > 0.0)
            {
                l_prob = icvComputeUniModeGauss(vect, mu + m*vect_size,
                                                        inv_var + m * vect_size,
                                                        log_var_val[m],
                                                        vect_size);

                prob = prob + weight[m]*exp((double)l_prob);
            }
        }
        prob = log(prob);
    }
    return (float)prob;
}*/


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: EstimateObsProb
//    Purpose: The function computes the probability of every observation in every state
//    Context:
//    Parameters:  obs_info - observations
//                 hmm      - hmm
//    Returns: error status
//
//    Notes:
//F*/
static CvStatus CV_STDCALL icvEstimateObsProb( CvImgObsInfo* obs_info, CvEHMM* hmm )
{
    int i, j;
    int total_states = 0;

    /* check if matrix exist and check current size
       if not sufficient - realloc */
    int status = 0; /* 1 - not allocated, 2 - allocated but small size,
                       3 - size is enough, but distribution is bad, 0 - all ok */

    for( j = 0; j < hmm->num_states; j++ )
    {
       total_states += hmm->u.ehmm[j].num_states;
    }

    if ( hmm->obsProb == NULL )
    {
        /* allocare memory */
        int need_size = ( obs_info->obs_x * obs_info->obs_y * total_states * sizeof(float) +
                          obs_info->obs_y * hmm->num_states * sizeof( CvMatr32f) );

        int* buffer = (int*)cvAlloc( need_size + 3 * sizeof(int) );
        buffer[0] = need_size;
        buffer[1] = obs_info->obs_y;
        buffer[2] = obs_info->obs_x;
        hmm->obsProb = (float**) (buffer + 3);
        status = 3;

    }
    else
    {
        /* check current size */
        int* total= (int*)(((int*)(hmm->obsProb)) - 3);
        int need_size = ( obs_info->obs_x * obs_info->obs_y * total_states * sizeof(float) +
                          obs_info->obs_y * hmm->num_states * sizeof( CvMatr32f/*(float*)*/ ) );

        assert( sizeof(float*) == sizeof(int) );

        if ( need_size > (*total) )
        {
            int* buffer = ((int*)(hmm->obsProb)) - 3;
            cvFree( &buffer);
            buffer = (int*)cvAlloc( need_size + 3 * sizeof(int));
            buffer[0] = need_size;
            buffer[1] = obs_info->obs_y;
            buffer[2] = obs_info->obs_x;

            hmm->obsProb = (float**)(buffer + 3);

            status = 3;
        }
    }
    if (!status)
    {
        int* obsx = ((int*)(hmm->obsProb)) - 1;
        int* obsy = ((int*)(hmm->obsProb)) - 2;

        assert( (*obsx > 0) && (*obsy > 0) );

        /* is good distribution? */
        if ( (obs_info->obs_x > (*obsx) ) || (obs_info->obs_y > (*obsy) ) )
            status = 3;
    }

    /* if bad status - do reallocation actions */
    assert( (status == 0) || (status == 3) );

    if ( status )
    {
        float** tmp = hmm->obsProb;
        float*  tmpf;

        /* distribute pointers of ehmm->obsProb */
        for( i = 0; i < hmm->num_states; i++ )
        {
            hmm->u.ehmm[i].obsProb = tmp;
            tmp += obs_info->obs_y;
        }

        tmpf = (float*)tmp;

        /* distribute pointers of ehmm->obsProb[j] */
        for( i = 0; i < hmm->num_states; i++ )
        {
            CvEHMM* ehmm = &( hmm->u.ehmm[i] );

            for( j = 0; j < obs_info->obs_y; j++ )
            {
                ehmm->obsProb[j] = tmpf;
                tmpf += ehmm->num_states * obs_info->obs_x;
            }
        }
    }/* end of pointer distribution */

#if 1
    {
#define MAX_BUF_SIZE  1200
        float  local_log_mix_prob[MAX_BUF_SIZE];
        double local_mix_prob[MAX_BUF_SIZE];
        int    vect_size = obs_info->obs_size;
        CvStatus res = CV_NO_ERR;

        float*  log_mix_prob = local_log_mix_prob;
        double* mix_prob = local_mix_prob;

        int  max_size = 0;
        int  obs_x = obs_info->obs_x;

        /* calculate temporary buffer size */
        for( i = 0; i < hmm->num_states; i++ )
        {
            CvEHMM* ehmm = &(hmm->u.ehmm[i]);
            CvEHMMState* state = ehmm->u.state;

            int max_mix = 0;
            for( j = 0; j < ehmm->num_states; j++ )
            {
                int t = state[j].num_mix;
                if( max_mix < t ) max_mix = t;
            }
            max_mix *= ehmm->num_states;
            if( max_size < max_mix ) max_size = max_mix;
        }

        max_size *= obs_x * vect_size;

        /* allocate buffer */
        if( max_size > MAX_BUF_SIZE )
        {
            log_mix_prob = (float*)cvAlloc( max_size*(sizeof(float) + sizeof(double)));
            if( !log_mix_prob ) return CV_OUTOFMEM_ERR;
            mix_prob = (double*)(log_mix_prob + max_size);
        }

        memset( log_mix_prob, 0, max_size*sizeof(float));

        /*****************computing probabilities***********************/

        /* loop through external states */
        for( i = 0; i < hmm->num_states; i++ )
        {
            CvEHMM* ehmm = &(hmm->u.ehmm[i]);
            CvEHMMState* state = ehmm->u.state;

            int max_mix = 0;
            int n_states = ehmm->num_states;

            /* determine maximal number of mixtures (again) */
            for( j = 0; j < ehmm->num_states; j++ )
            {
                int t = state[j].num_mix;
                if( max_mix < t ) max_mix = t;
            }

            /* loop through rows of the observation matrix */
            for( j = 0; j < obs_info->obs_y; j++ )
            {
                int  m, n;

                float* obs = obs_info->obs + j * obs_x * vect_size;
                float* log_mp = max_mix > 1 ? log_mix_prob : ehmm->obsProb[j];
                double* mp = mix_prob;

                /* several passes are done below */

                /* 1. calculate logarithms of probabilities for each mixture */

                /* loop through mixtures */
                for( m = 0; m < max_mix; m++ )
                {
                    /* set pointer to first observation in the line */
                    float* vect = obs;

                    /* cycles through obseravtions in the line */
                    for( n = 0; n < obs_x; n++, vect += vect_size, log_mp += n_states )
                    {
                        int k, l;
                        for( l = 0; l < n_states; l++ )
                        {
                            if( state[l].num_mix > m )
                            {
                                float* mu = state[l].mu + m*vect_size;
                                float* inv_var = state[l].inv_var + m*vect_size;
                                double prob = -state[l].log_var_val[m];
                                for( k = 0; k < vect_size; k++ )
                                {
                                    double t = (vect[k] - mu[k])*inv_var[k];
                                    prob -= t*t;
                                }
                                log_mp[l] = MAX( (float)prob, -500 );
                            }
                        }
                    }
                }

                /* skip the rest if there is a single mixture */
                if( max_mix == 1 ) continue;

                /* 2. calculate exponent of log_mix_prob
                      (i.e. probability for each mixture) */
                cvbFastExp( log_mix_prob, mix_prob, max_mix * obs_x * n_states );

                /* 3. sum all mixtures with weights */
                /* 3a. first mixture - simply scale by weight */
                for( n = 0; n < obs_x; n++, mp += n_states )
                {
                    int l;
                    for( l = 0; l < n_states; l++ )
                    {
                        mp[l] *= state[l].weight[0];
                    }
                }

                /* 3b. add other mixtures */
                for( m = 1; m < max_mix; m++ )
                {
                    int ofs = -m*obs_x*n_states;
                    for( n = 0; n < obs_x; n++, mp += n_states )
                    {
                        int l;
                        for( l = 0; l < n_states; l++ )
                        {
                            if( m < state[l].num_mix )
                            {
                                mp[l + ofs] += mp[l] * state[l].weight[m];
                            }
                        }
                    }
                }

                /* 4. Put logarithms of summary probabilities to the destination matrix */
                cvbFastLog( mix_prob, ehmm->obsProb[j], obs_x * n_states );
            }
        }

        if( log_mix_prob != local_log_mix_prob ) cvFree( &log_mix_prob );
        return res;
#undef MAX_BUF_SIZE
    }
#else
    for( i = 0; i < hmm->num_states; i++ )
    {
        CvEHMM* ehmm = &(hmm->u.ehmm[i]);
        CvEHMMState* state = ehmm->u.state;

        for( j = 0; j < obs_info->obs_y; j++ )
        {
            int k,m;

            int obs_index = j * obs_info->obs_x;

            float* B = ehmm->obsProb[j];

            /* cycles through obs and states */
            for( k = 0; k < obs_info->obs_x; k++ )
            {
                CvVect32f vect = (obs_info->obs) + (obs_index + k) * vect_size;

                float* matr_line = B + k * ehmm->num_states;

                for( m = 0; m < ehmm->num_states; m++ )
                {
                    matr_line[m] = icvComputeGaussMixture( vect, state[m].mu, state[m].inv_var,
                                                             state[m].log_var_val, vect_size, state[m].weight,
                                                             state[m].num_mix );
                }
            }
        }
    }
#endif
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: EstimateTransProb
//    Purpose: The function calculates the state and super state transition probabilities
//             of the model given the images,
//             the state segmentation and the input parameters
//    Context:
//    Parameters: obs_info_array - array of pointers to image observations
//                num_img - length of above array
//                hmm - pointer to HMM structure
//    Returns: void
//
//    Notes:
//F*/
static CvStatus CV_STDCALL
icvEstimateTransProb( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm )
{
    int  i, j, k;

    CvEHMMState* first_state = hmm->u.ehmm->u.state;
    /* as a counter we will use transP matrix */

    /* initialization */

    /* clear transP */
    icvSetZero_32f( hmm->transP, hmm->num_states, hmm->num_states );
    for (i = 0; i < hmm->num_states; i++ )
    {
        icvSetZero_32f( hmm->u.ehmm[i].transP , hmm->u.ehmm[i].num_states, hmm->u.ehmm[i].num_states );
    }

    /* compute the counters */
    for (i = 0; i < num_img; i++)
    {
        int counter = 0;
        CvImgObsInfo* info = obs_info_array[i];

        for (j = 0; j < info->obs_y; j++)
        {
            for (k = 0; k < info->obs_x; k++, counter++)
            {
                /* compute how many transitions from state to state
                   occured both in horizontal and vertical direction */
                int superstate, state;
                int nextsuperstate, nextstate;
                int begin_ind;

                superstate = info->state[2 * counter];
                begin_ind = (int)(hmm->u.ehmm[superstate].u.state - first_state);
                state = info->state[ 2 * counter + 1] - begin_ind;

                if (j < info->obs_y - 1)
                {
                    int transP_size = hmm->num_states;

                    nextsuperstate = info->state[ 2*(counter + info->obs_x) ];

                    hmm->transP[superstate * transP_size + nextsuperstate] += 1;
                }

                if (k < info->obs_x - 1)
                {
                    int transP_size = hmm->u.ehmm[superstate].num_states;

                    nextstate = info->state[2*(counter+1) + 1] - begin_ind;
                    hmm->u.ehmm[superstate].transP[ state * transP_size + nextstate] += 1;
                }
            }
        }
    }
    /* estimate superstate matrix */
    for( i = 0; i < hmm->num_states; i++)
    {
        float total = 0;
        float inv_total;
        for( j = 0; j < hmm->num_states; j++)
        {
            total += hmm->transP[i * hmm->num_states + j];
        }
        //assert( total );

        inv_total = total ? 1.f/total : 0;

        for( j = 0; j < hmm->num_states; j++)
        {
            hmm->transP[i * hmm->num_states + j] =
                hmm->transP[i * hmm->num_states + j] ?
                (float)log( hmm->transP[i * hmm->num_states + j] * inv_total ) : -BIG_FLT;
        }
    }

    /* estimate other matrices */
    for( k = 0; k < hmm->num_states; k++ )
    {
        CvEHMM* ehmm = &(hmm->u.ehmm[k]);

        for( i = 0; i < ehmm->num_states; i++)
        {
            float total = 0;
            float inv_total;
            for( j = 0; j < ehmm->num_states; j++)
            {
                total += ehmm->transP[i*ehmm->num_states + j];
            }
            //assert( total );
            inv_total = total ? 1.f/total :  0;

            for( j = 0; j < ehmm->num_states; j++)
            {
                ehmm->transP[i * ehmm->num_states + j] =
                    (ehmm->transP[i * ehmm->num_states + j]) ?
                    (float)log( ehmm->transP[i * ehmm->num_states + j] * inv_total) : -BIG_FLT ;
            }
        }
    }
    return CV_NO_ERR;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: MixSegmL2
//    Purpose: The function implements the mixture segmentation of the states of the
//             embedded HMM
//    Context: used with the Viterbi training of the embedded HMM
//
//    Parameters:
//             obs_info_array
//             num_img
//             hmm
//    Returns: void
//
//    Notes:
//F*/
static CvStatus CV_STDCALL
icvMixSegmL2( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm )
{
    int     k, i, j, m;

    CvEHMMState* state = hmm->u.ehmm[0].u.state;


    for (k = 0; k < num_img; k++)
    {
        int counter = 0;
        CvImgObsInfo* info = obs_info_array[k];

        for (i = 0; i < info->obs_y; i++)
        {
            for (j = 0; j < info->obs_x; j++, counter++)
            {
                int e_state = info->state[2 * counter + 1];
                float min_dist;

                min_dist = icvSquareDistance((info->obs) + (counter * info->obs_size),
                                               state[e_state].mu, info->obs_size);
                info->mix[counter] = 0;

                for (m = 1; m < state[e_state].num_mix; m++)
                {
                    float dist=icvSquareDistance( (info->obs) + (counter * info->obs_size),
                                                    state[e_state].mu + m * info->obs_size,
                                                    info->obs_size);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        /* assign mixture with smallest distance */
                        info->mix[counter] = m;
                    }
                }
            }
        }
    }
    return CV_NO_ERR;
}

/*
CvStatus icvMixSegmProb(CvImgObsInfo* obs_info, int num_img, CvEHMM* hmm )
{
    int     k, i, j, m;

    CvEHMMState* state = hmm->ehmm[0].state_info;


    for (k = 0; k < num_img; k++)
    {
        int counter = 0;
        CvImgObsInfo* info = obs_info + k;

        for (i = 0; i < info->obs_y; i++)
        {
            for (j = 0; j < info->obs_x; j++, counter++)
            {
                int e_state = info->in_state[counter];
                float max_prob;

                max_prob = icvComputeUniModeGauss( info->obs[counter], state[e_state].mu[0],
                                                    state[e_state].inv_var[0],
                                                    state[e_state].log_var[0],
                                                    info->obs_size );
                info->mix[counter] = 0;

                for (m = 1; m < state[e_state].num_mix; m++)
                {
                    float prob=icvComputeUniModeGauss(info->obs[counter], state[e_state].mu[m],
                                                       state[e_state].inv_var[m],
                                                       state[e_state].log_var[m],
                                                       info->obs_size);
                    if (prob > max_prob)
                    {
                        max_prob = prob;
                        // assign mixture with greatest probability.
                        info->mix[counter] = m;
                    }
                }
            }
        }
    }

    return CV_NO_ERR;
}
*/
static CvStatus CV_STDCALL
icvViterbiSegmentation( int num_states, int /*num_obs*/, CvMatr32f transP,
                        CvMatr32f B, int start_obs, int prob_type,
                        int** q, int min_num_obs, int max_num_obs,
                        float* prob )
{
    // memory allocation
    int i, j, last_obs;
    int m_HMMType = _CV_ERGODIC; /* _CV_CAUSAL or _CV_ERGODIC */

    int m_ProbType   = prob_type; /* _CV_LAST_STATE or _CV_BEST_STATE */

    int m_minNumObs  = min_num_obs; /*??*/
    int m_maxNumObs  = max_num_obs; /*??*/

    int m_numStates  = num_states;

    float* m_pi = (float*)cvAlloc( num_states* sizeof(float) );
    CvMatr32f m_a = transP;

    // offset brobability matrix to starting observation
    CvMatr32f m_b = B + start_obs * num_states;
    //so m_xl will not be used more

    //m_xl = start_obs;

    /*     if (muDur != NULL){
    m_d = new int[m_numStates];
    m_l = new double[m_numStates];
    for (i = 0; i < m_numStates; i++){
    m_l[i] = muDur[i];
    }
    }
    else{
    m_d = NULL;
    m_l = NULL;
    }
    */

    CvMatr32f m_Gamma = icvCreateMatrix_32f( num_states, m_maxNumObs );
    int* m_csi = (int*)cvAlloc( num_states * m_maxNumObs * sizeof(int) );

    //stores maximal result for every ending observation */
    CvVect32f   m_MaxGamma = prob;


//    assert( m_xl + max_num_obs <= num_obs );

    /*??m_q          = new int*[m_maxNumObs - m_minNumObs];
      ??for (i = 0; i < m_maxNumObs - m_minNumObs; i++)
      ??     m_q[i] = new int[m_minNumObs + i + 1];
    */

    /******************************************************************/
    /*    Viterbi initialization                                      */
    /* set initial state probabilities, in logarithmic scale */
    for (i = 0; i < m_numStates; i++)
    {
        m_pi[i] = -BIG_FLT;
    }
    m_pi[0] = 0.0f;

    for  (i = 0; i < num_states; i++)
    {
        m_Gamma[0 * num_states + i] = m_pi[i] + m_b[0 * num_states + i];
        m_csi[0 * num_states + i] = 0;
    }

    /******************************************************************/
    /*    Viterbi recursion                                           */

    if ( m_HMMType == _CV_CAUSAL ) //causal model
    {
        int t;

        for (t = 1 ; t < m_maxNumObs; t++)
        {
            // evaluate self-to-self transition for state 0
            m_Gamma[t * num_states + 0] = m_Gamma[(t-1) * num_states + 0] + m_a[0];
            m_csi[t * num_states + 0] = 0;

            for (j = 1; j < num_states; j++)
            {
                float self = m_Gamma[ (t-1) * num_states + j] + m_a[ j * num_states + j];
                float prev = m_Gamma[ (t-1) * num_states +(j-1)] + m_a[ (j-1) * num_states + j];

                if ( prev > self )
                {
                    m_csi[t * num_states + j] = j-1;
                    m_Gamma[t * num_states + j] = prev;
                }
                else
                {
                    m_csi[t * num_states + j] = j;
                    m_Gamma[t * num_states + j] = self;
                }

                m_Gamma[t * num_states + j] = m_Gamma[t * num_states + j] + m_b[t * num_states + j];
            }
        }
    }
    else if ( m_HMMType == _CV_ERGODIC ) //ergodic model
    {
        int t;
        for (t = 1 ; t < m_maxNumObs; t++)
        {
            for (j = 0; j < num_states; j++)
            {
                m_Gamma[ t*num_states + j] = m_Gamma[(t-1) * num_states + 0] + m_a[0*num_states+j];
                m_csi[t *num_states + j] = 0;

                for (i = 1; i < num_states; i++)
                {
                    float currGamma = m_Gamma[(t-1) *num_states + i] + m_a[i *num_states + j];
                    if (currGamma > m_Gamma[t *num_states + j])
                    {
                        m_Gamma[t * num_states + j] = currGamma;
                        m_csi[t * num_states + j] = i;
                    }
                }
                m_Gamma[t *num_states + j] = m_Gamma[t *num_states + j] + m_b[t * num_states + j];
            }
        }
    }

    for( last_obs = m_minNumObs-1, i = 0; last_obs < m_maxNumObs; last_obs++, i++ )
    {
        int t;

        /******************************************************************/
        /*    Viterbi termination                                         */

        if ( m_ProbType == _CV_LAST_STATE )
        {
            m_MaxGamma[i] = m_Gamma[last_obs * num_states + num_states - 1];
            q[i][last_obs] = num_states - 1;
        }
        else if( m_ProbType == _CV_BEST_STATE )
        {
            int k;
            q[i][last_obs] = 0;
            m_MaxGamma[i] = m_Gamma[last_obs * num_states + 0];

            for(k = 1; k < num_states; k++)
            {
                if ( m_Gamma[last_obs * num_states + k] > m_MaxGamma[i] )
                {
                    m_MaxGamma[i] = m_Gamma[last_obs * num_states + k];
                    q[i][last_obs] = k;
                }
            }
        }

        /******************************************************************/
        /*    Viterbi backtracking                                        */
        for  (t = last_obs-1; t >= 0; t--)
        {
            q[i][t] = m_csi[(t+1) * num_states + q[i][t+1] ];
        }
    }

    /* memory free */
    cvFree( &m_pi );
    cvFree( &m_csi );
    icvDeleteMatrix( m_Gamma );

    return CV_NO_ERR;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvEViterbi
//    Purpose: The function calculates the embedded Viterbi algorithm
//             for 1 image
//    Context:
//    Parameters:
//             obs_info - observations
//             hmm      - HMM
//
//    Returns: the Embedded Viterbi probability (float)
//             and do state segmentation of observations
//
//    Notes:
//F*/
static float CV_STDCALL icvEViterbi( CvImgObsInfo* obs_info, CvEHMM* hmm )
{
    int    i, j, counter;
    float  log_likelihood;

    float inv_obs_x = 1.f / obs_info->obs_x;

    CvEHMMState* first_state = hmm->u.ehmm->u.state;

    /* memory allocation for superB */
    CvMatr32f superB = icvCreateMatrix_32f(hmm->num_states, obs_info->obs_y );

    /* memory allocation for q */
    int*** q = (int***)cvAlloc( hmm->num_states * sizeof(int**) );
    int* super_q = (int*)cvAlloc( obs_info->obs_y * sizeof(int) );

    for (i = 0; i < hmm->num_states; i++)
    {
        q[i] = (int**)cvAlloc( obs_info->obs_y * sizeof(int*) );

        for (j = 0; j < obs_info->obs_y ; j++)
        {
            q[i][j] = (int*)cvAlloc( obs_info->obs_x * sizeof(int) );
        }
    }

    /* start Viterbi segmentation */
    for (i = 0; i < hmm->num_states; i++)
    {
        CvEHMM* ehmm = &(hmm->u.ehmm[i]);

        for (j = 0; j < obs_info->obs_y; j++)
        {
            float max_gamma;

            /* 1D HMM Viterbi segmentation */
            icvViterbiSegmentation( ehmm->num_states, obs_info->obs_x,
                ehmm->transP, ehmm->obsProb[j], 0,
                _CV_LAST_STATE, &q[i][j], obs_info->obs_x,
                obs_info->obs_x, &max_gamma);

            superB[j * hmm->num_states + i] = max_gamma * inv_obs_x;
        }
    }

    /* perform global Viterbi segmentation (i.e. process higher-level HMM) */

    icvViterbiSegmentation( hmm->num_states, obs_info->obs_y,
                             hmm->transP, superB, 0,
                             _CV_LAST_STATE, &super_q, obs_info->obs_y,
                             obs_info->obs_y, &log_likelihood );

    log_likelihood /= obs_info->obs_y ;


    counter = 0;
    /* assign new state to observation vectors */
    for (i = 0; i < obs_info->obs_y; i++)
    {
        for (j = 0; j < obs_info->obs_x; j++, counter++)
        {
            int superstate = super_q[i];
            int state = (int)(hmm->u.ehmm[superstate].u.state - first_state);

            obs_info->state[2 * counter] = superstate;
            obs_info->state[2 * counter + 1] = state + q[superstate][i][j];
        }
    }

    /* memory deallocation for superB */
    icvDeleteMatrix( superB );

    /*memory deallocation for q */
    for (i = 0; i < hmm->num_states; i++)
    {
        for (j = 0; j < obs_info->obs_y ; j++)
        {
            cvFree( &q[i][j] );
        }
        cvFree( &q[i] );
    }

    cvFree( &q );
    cvFree( &super_q );

    return log_likelihood;
}

static CvStatus CV_STDCALL
icvEstimateHMMStateParams( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm )
{
    /* compute gamma, weights, means, vars */
    int k, i, j, m;
    int total = 0;
    int vect_len = obs_info_array[0]->obs_size;

    float start_log_var_val = LN2PI * vect_len;

    CvVect32f tmp_vect = icvCreateVector_32f( vect_len );

    CvEHMMState* first_state = hmm->u.ehmm[0].u.state;

    assert( sizeof(float) == sizeof(int) );

    for(i = 0; i < hmm->num_states; i++ )
    {
        total+= hmm->u.ehmm[i].num_states;
    }

    /***************Gamma***********************/
    /* initialize gamma */
    for( i = 0; i < total; i++ )
    {
        for (m = 0; m < first_state[i].num_mix; m++)
        {
            ((int*)(first_state[i].weight))[m] = 0;
        }
    }

    /* maybe gamma must be computed in mixsegm process ?? */

    /* compute gamma */
    for (k = 0; k < num_img; k++)
    {
        CvImgObsInfo* info = obs_info_array[k];
        int num_obs = info->obs_y * info->obs_x;

        for (i = 0; i < num_obs; i++)
        {
            int state, mixture;
            state = info->state[2*i + 1];
            mixture = info->mix[i];
            /* computes gamma - number of observations corresponding
               to every mixture of every state */
            ((int*)(first_state[state].weight))[mixture] += 1;
        }
    }
    /***************Mean and Var***********************/
    /* compute means and variances of every item */
    /* initially variance placed to inv_var */
    /* zero mean and variance */
    for (i = 0; i < total; i++)
    {
        memset( (void*)first_state[i].mu, 0, first_state[i].num_mix * vect_len *
                                                                         sizeof(float) );
        memset( (void*)first_state[i].inv_var, 0, first_state[i].num_mix * vect_len *
                                                                         sizeof(float) );
    }

    /* compute sums */
    for (i = 0; i < num_img; i++)
    {
        CvImgObsInfo* info = obs_info_array[i];
        int total_obs = info->obs_x * info->obs_y;

        float* vector = info->obs;

        for (j = 0; j < total_obs; j++, vector+=vect_len )
        {
            int state = info->state[2 * j + 1];
            int mixture = info->mix[j];

            CvVect32f mean  = first_state[state].mu + mixture * vect_len;
            CvVect32f mean2 = first_state[state].inv_var + mixture * vect_len;

            icvAddVector_32f( mean, vector, mean, vect_len );
            for( k = 0; k < vect_len; k++ )
                mean2[k] += vector[k]*vector[k];
        }
    }

    /*compute the means and variances */
    /* assume gamma already computed */
    for (i = 0; i < total; i++)
    {
        CvEHMMState* state = &(first_state[i]);

        for (m = 0; m < state->num_mix; m++)
        {
            CvVect32f mu  = state->mu + m * vect_len;
            CvVect32f invar = state->inv_var + m * vect_len;

            if ( ((int*)state->weight)[m] > 1)
            {
                float inv_gamma = 1.f/((int*)(state->weight))[m];

                icvScaleVector_32f( mu, mu, vect_len, inv_gamma);
                icvScaleVector_32f( invar, invar, vect_len, inv_gamma);
            }

            icvMulVectors_32f(mu, mu, tmp_vect, vect_len);
            icvSubVector_32f( invar, tmp_vect, invar, vect_len);

            /* low bound of variance - 100 (Ara's experimental result) */
            for( k = 0; k < vect_len; k++ )
            {
                invar[k] = (invar[k] > 100.f) ? invar[k] : 100.f;
            }

            /* compute log_var */
            state->log_var_val[m] = start_log_var_val;
            for( k = 0; k < vect_len; k++ )
            {
                state->log_var_val[m] += (float)log( invar[k] );
            }

            /* SMOLI 27.10.2000 */
            state->log_var_val[m] *= 0.5;


            /* compute inv_var = 1/sqrt(2*variance) */
            icvScaleVector_32f(invar, invar, vect_len, 2.f );
            cvbInvSqrt( invar, invar, vect_len );
        }
    }

    /***************Weights***********************/
    /* normilize gammas - i.e. compute mixture weights */

    //compute weights
    for (i = 0; i < total; i++)
    {
        int gamma_total = 0;
        float norm;

        for (m = 0; m < first_state[i].num_mix; m++)
        {
            gamma_total += ((int*)(first_state[i].weight))[m];
        }

        norm = gamma_total ? (1.f/(float)gamma_total) : 0.f;

        for (m = 0; m < first_state[i].num_mix; m++)
        {
            first_state[i].weight[m] = ((int*)(first_state[i].weight))[m] * norm;
        }
    }

    icvDeleteVector( tmp_vect);
    return CV_NO_ERR;
}

/*
CvStatus icvLightingCorrection8uC1R( uchar* img, CvSize roi, int src_step )
{
    int i, j;
    int width = roi.width;
    int height = roi.height;

    float x1, x2, y1, y2;
    int f[3] = {0, 0, 0};
    float a[3] = {0, 0, 0};

    float h1;
    float h2;

    float c1,c2;

    float min = FLT_MAX;
    float max = -FLT_MAX;
    float correction;

    float* float_img = icvAlloc( width * height * sizeof(float) );

    x1 = width * (width + 1) / 2.0f; // Sum (1, ... , width)
    x2 = width * (width + 1 ) * (2 * width + 1) / 6.0f; // Sum (1^2, ... , width^2)
    y1 = height * (height + 1)/2.0f; // Sum (1, ... , width)
    y2 = height * (height + 1 ) * (2 * height + 1) / 6.0f; // Sum (1^2, ... , width^2)


    // extract grayvalues
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            f[2] = f[2] + j * img[i*src_step + j];
            f[1] = f[1] + i * img[i*src_step + j];
            f[0] = f[0] +     img[i*src_step + j];
        }
    }

    h1 = (float)f[0] * (float)x1 / (float)width;
    h2 = (float)f[0] * (float)y1 / (float)height;

    a[2] = ((float)f[2] - h1) / (float)(x2*height - x1*x1*height/(float)width);
    a[1] = ((float)f[1] - h2) / (float)(y2*width - y1*y1*width/(float)height);
    a[0] = (float)f[0]/(float)(width*height) - (float)y1*a[1]/(float)height -
        (float)x1*a[2]/(float)width;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {

            correction = a[0] + a[1]*(float)i + a[2]*(float)j;

            float_img[i*width + j] = img[i*src_step + j] - correction;

            if (float_img[i*width + j] < min) min = float_img[i*width+j];
            if (float_img[i*width + j] > max) max = float_img[i*width+j];
        }
    }

    //rescaling to the range 0:255
    c2 = 0;
    if (max == min)
        c2 = 255.0f;
    else
        c2 = 255.0f/(float)(max - min);

    c1 = (-(float)min)*c2;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            int value = (int)floor(c2*float_img[i*width + j] + c1);
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            img[i*src_step + j] = (uchar)value;
        }
    }

    cvFree( &float_img );
    return CV_NO_ERR;
}


CvStatus icvLightingCorrection( icvImage* img )
{
    CvSize roi;
    if ( img->type != IPL_DEPTH_8U || img->channels != 1 )
    return CV_BADFACTOR_ERR;

    roi = _cvSize( img->roi.width, img->roi.height );

    return _cvLightingCorrection8uC1R( img->data + img->roi.y * img->step + img->roi.x,
                                        roi, img->step );

}

*/

CV_IMPL CvEHMM*
cvCreate2DHMM( int *state_number, int *num_mix, int obs_size )
{
    CvEHMM* hmm = 0;

    IPPI_CALL( icvCreate2DHMM( &hmm, state_number, num_mix, obs_size ));

    return hmm;
}

CV_IMPL void
cvRelease2DHMM( CvEHMM ** hmm )
{
    IPPI_CALL( icvRelease2DHMM( hmm ));
}

CV_IMPL CvImgObsInfo*
cvCreateObsInfo( CvSize num_obs, int obs_size )
{
    CvImgObsInfo *obs_info = 0;

    IPPI_CALL( icvCreateObsInfo( &obs_info, num_obs, obs_size ));

    return obs_info;
}

CV_IMPL void
cvReleaseObsInfo( CvImgObsInfo ** obs_info )
{
    IPPI_CALL( icvReleaseObsInfo( obs_info ));
}


CV_IMPL void
cvUniformImgSegm( CvImgObsInfo * obs_info, CvEHMM * hmm )
{
    IPPI_CALL( icvUniformImgSegm( obs_info, hmm ));
}

CV_IMPL void
cvInitMixSegm( CvImgObsInfo ** obs_info_array, int num_img, CvEHMM * hmm )
{
    IPPI_CALL( icvInitMixSegm( obs_info_array, num_img, hmm ));
}

CV_IMPL void
cvEstimateHMMStateParams( CvImgObsInfo ** obs_info_array, int num_img, CvEHMM * hmm )
{
    IPPI_CALL( icvEstimateHMMStateParams( obs_info_array, num_img, hmm ));
}

CV_IMPL void
cvEstimateTransProb( CvImgObsInfo ** obs_info_array, int num_img, CvEHMM * hmm )
{
    IPPI_CALL( icvEstimateTransProb( obs_info_array, num_img, hmm ));
}

CV_IMPL void
cvEstimateObsProb( CvImgObsInfo * obs_info, CvEHMM * hmm )
{
    IPPI_CALL( icvEstimateObsProb( obs_info, hmm ));
}

CV_IMPL float
cvEViterbi( CvImgObsInfo * obs_info, CvEHMM * hmm )
{
    if( (obs_info == NULL) || (hmm == NULL) )
        CV_Error( CV_BadDataPtr, "Null pointer." );

    return icvEViterbi( obs_info, hmm );
}

CV_IMPL void
cvMixSegmL2( CvImgObsInfo ** obs_info_array, int num_img, CvEHMM * hmm )
{
    IPPI_CALL( icvMixSegmL2( obs_info_array, num_img, hmm ));
}

/* End of file */
