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

#if 0

#define LN2PI 1.837877f
#define BIG_FLT 1.e+10f


#define _CV_ERGODIC 1
#define _CV_CAUSAL 2

#define _CV_LAST_STATE 1
#define _CV_BEST_STATE 2

//*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvForward1DHMM
//    Purpose: The function performs baum-welsh algorithm
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
#if 0
CvStatus icvForward1DHMM( int num_states, int num_obs, CvMatr64d A,
                          CvMatr64d B,
                          double* scales)
{
    // assume that observation and transition
    // probabilities already computed
    int m_HMMType  = _CV_CAUSAL;
    double* m_pi = icvAlloc( num_states* sizeof( double) );

    /* alpha is matrix
       rows throuhg states
       columns through time
    */
    double* alpha = icvAlloc( num_states*num_obs * sizeof( double ) );

    /* All calculations will be in non-logarithmic domain */

    /* Initialization */
    /* set initial state probabilities */
    m_pi[0] = 1;
    for (i = 1; i < num_states; i++)
    {
        m_pi[i] = 0.0;
    }

    for  (i = 0; i < num_states; i++)
    {
        alpha[i] = m_pi[i] * m_b[ i];
    }

    /******************************************************************/
    /*   Induction                                                    */

    if ( m_HMMType == _CV_ERGODIC )
    {
        int t;
        for (t = 1 ; t < num_obs; t++)
        {
            for (j = 0; j < num_states; j++)
            {
               double sum = 0.0;
               int i;

                for (i = 0; i < num_states; i++)
                {
                     sum += alpha[(t - 1) * num_states + i] * A[i * num_states + j];
                }

                alpha[(t - 1) * num_states + j] = sum * B[t * num_states + j];

                /* add computed alpha to scale factor */
                sum_alpha += alpha[(t - 1) * num_states + j];
            }

            double scale = 1/sum_alpha;

            /* scale alpha */
            for (j = 0; j < num_states; j++)
            {
                alpha[(t - 1) * num_states + j] *= scale;
            }

            scales[t] = scale;

        }
    }

#endif



//*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCreateObsInfo
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
/*CvStatus icvCreateObsInfo( CvImgObsInfo** obs_info,
                              CvSize num_obs, int obs_size )
{
    int total = num_obs.height * num_obs.width;

    CvImgObsInfo* obs = (CvImgObsInfo*)icvAlloc( sizeof( CvImgObsInfo) );

    obs->obs_x = num_obs.width;
    obs->obs_y = num_obs.height;

    obs->obs = (float*)icvAlloc( total * obs_size * sizeof(float) );

    obs->state = (int*)icvAlloc( 2 * total * sizeof(int) );
    obs->mix = (int*)icvAlloc( total * sizeof(int) );

    obs->obs_size = obs_size;

    obs_info[0] = obs;

    return CV_NO_ERR;
}*/

/*CvStatus icvReleaseObsInfo( CvImgObsInfo** p_obs_info )
{
    CvImgObsInfo* obs_info = p_obs_info[0];

    icvFree( &(obs_info->obs) );
    icvFree( &(obs_info->mix) );
    icvFree( &(obs_info->state) );
    icvFree( &(obs_info) );

    p_obs_info[0] = NULL;

    return CV_NO_ERR;
} */


//*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCreate1DHMM
//    Purpose: The function allocates memory for 1-dimensional HMM
//             and its inner stuff
//    Context:
//    Parameters: hmm - addres of pointer to CvEHMM structure
//                state_number - number of states in HMM
//                num_mix - number of gaussian mixtures in HMM states
//                          size of array is defined by previous parameter
//                obs_size - length of observation vectors
//
//    Returns: error status
//    Notes:
//F*/
CvStatus icvCreate1DHMM( CvEHMM** this_hmm,
                         int state_number, int* num_mix, int obs_size )
{
    int i;
    int real_states = state_number;

    CvEHMMState* all_states;
    CvEHMM* hmm;
    int total_mix = 0;
    float* pointers;

    /* allocate memory for hmm */
    hmm = (CvEHMM*)icvAlloc( sizeof(CvEHMM) );

    /* set number of superstates */
    hmm->num_states = state_number;
    hmm->level = 0;

    /* allocate memory for all states */
    all_states = (CvEHMMState *)icvAlloc( real_states * sizeof( CvEHMMState ) );

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
    pointers = (float*)icvAlloc( total_mix * (2/*for mu invvar */ * obs_size +
                                 2/*for weight and log_var_val*/ ) * sizeof( float) );

    /* organize memory */
    for( i = 0; i < real_states; i++ )
    {
        all_states[i].mu      = pointers; pointers += num_mix[i] * obs_size;
        all_states[i].inv_var = pointers; pointers += num_mix[i] * obs_size;

        all_states[i].log_var_val = pointers; pointers += num_mix[i];
        all_states[i].weight      = pointers; pointers += num_mix[i];
    }
    hmm->u.state = all_states;

    hmm->transP = icvCreateMatrix_32f( hmm->num_states, hmm->num_states );
    hmm->obsProb = NULL;

    /* if all ok - return pointer */
    *this_hmm = hmm;
    return CV_NO_ERR;
}

CvStatus icvRelease1DHMM( CvEHMM** phmm )
{
    CvEHMM* hmm = phmm[0];
    icvDeleteMatrix( hmm->transP );

    if (hmm->obsProb != NULL)
    {
        int* tmp = ((int*)(hmm->obsProb)) - 3;
        icvFree( &(tmp)  );
    }

    icvFree( &(hmm->u.state->mu) );
    icvFree( &(hmm->u.state) );

    phmm[0] = NULL;

    return CV_NO_ERR;
}

/*can be used in CHMM & DHMM */
CvStatus icvUniform1DSegm( Cv1DObsInfo* obs_info, CvEHMM* hmm )
{
    /* implementation is very bad */
    int  i;
    CvEHMMState* first_state;

    /* check arguments */
    if ( !obs_info || !hmm ) return CV_NULLPTR_ERR;

    first_state = hmm->u.state;

    for (i = 0; i < obs_info->obs_x; i++)
    {
        //bad line (division )
        int state = (i * hmm->num_states)/obs_info->obs_x;
        obs_info->state[i] = state;
    }
    return CV_NO_ERR;
}



/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: InitMixSegm
//    Purpose: The function implements the mixture segmentation of the states of the embedded HMM
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
CvStatus icvInit1DMixSegm(Cv1DObsInfo** obs_info_array, int num_img, CvEHMM* hmm)
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

    int total = hmm->num_states;
    CvEHMMState* first_state = hmm->u.state;

    /* for every state integer is allocated - number of vectors in state */
    num_samples = (int*)icvAlloc( total * sizeof(int) );

    /* integer counter is allocated for every state */
    counter = (int*)icvAlloc( total * sizeof(int) );

    samples = (CvVect32f**)icvAlloc( total * sizeof(CvVect32f*) );
    samples_mix = (int***)icvAlloc( total * sizeof(int**) );

    /* clear */
    memset( num_samples, 0 , total*sizeof(int) );
    memset( counter, 0 , total*sizeof(int) );


    /* for every state the number of vectors which belong to it is computed (smth. like histogram) */
    for (k = 0; k < num_img; k++)
    {
        CvImgObsInfo* obs = obs_info_array[k];

        for (i = 0; i < obs->obs_x; i++)
        {
            int state = obs->state[ i ];
            num_samples[state] += 1;
        }
    }

    /* for every state int* is allocated */
    a_class = (int**)icvAlloc( total*sizeof(int*) );

    for (i = 0; i < total; i++)
    {
        a_class[i] = (int*)icvAlloc( num_samples[i] * sizeof(int) );
        samples[i] = (CvVect32f*)icvAlloc( num_samples[i] * sizeof(CvVect32f) );
        samples_mix[i] = (int**)icvAlloc( num_samples[i] * sizeof(int*) );
    }

    /* for every state vectors which belong to state are gathered */
    for (k = 0; k < num_img; k++)
    {
        CvImgObsInfo* obs = obs_info_array[k];
        int num_obs = obs->obs_x;
        float* vector = obs->obs;

        for (i = 0; i < num_obs; i++, vector+=obs->obs_size )
        {
            int state = obs->state[i];

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
            icvKMeans( first_state[i].num_mix, samples[i], num_samples[i],
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
        icvFree( &(a_class[i]) );
        icvFree( &(samples[i]) );
        icvFree( &(samples_mix[i]) );
    }

    icvFree( &a_class );
    icvFree( &samples );
    icvFree( &samples_mix );
    icvFree( &counter );
    icvFree( &num_samples );


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
/*float icvComputeUniModeGauss(CvVect32f vect, CvVect32f mu,
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
/*float icvComputeGaussMixture( CvVect32f vect, float* mu,
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
}
*/

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
CvStatus icvEstimate1DObsProb(CvImgObsInfo* obs_info, CvEHMM* hmm )
{
    int j;
    int total_states = 0;

    /* check if matrix exist and check current size
       if not sufficient - realloc */
    int status = 0; /* 1 - not allocated, 2 - allocated but small size,
                       3 - size is enough, but distribution is bad, 0 - all ok */

    /*for( j = 0; j < hmm->num_states; j++ )
    {
       total_states += hmm->u.ehmm[j].num_states;
    }*/
    total_states = hmm->num_states;

    if ( hmm->obsProb == NULL )
    {
        /* allocare memory */
        int need_size = ( obs_info->obs_x /* * obs_info->obs_y*/ * total_states * sizeof(float) /* +
                          obs_info->obs_y * hmm->num_states * sizeof( CvMatr32f) */);

        int* buffer = (int*)icvAlloc( need_size + 3 * sizeof(int) );
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
        int need_size = ( obs_info->obs_x /* * obs_info->obs_y*/ * total_states * sizeof(float) /* +
                           obs_info->obs_y * hmm->num_states * sizeof( CvMatr32f(float*)  )*/ );

        assert( sizeof(float*) == sizeof(int) );

        if ( need_size > (*total) )
        {
            int* buffer = ((int*)(hmm->obsProb)) - 3;
            icvFree( &buffer);
            buffer = (int*)icvAlloc( need_size + 3);
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
        //int* obsy = ((int*)(hmm->obsProb)) - 2;

        assert( /*(*obsy > 0) &&*/ (*obsx > 0) );

        /* is good distribution? */
        if ( (obs_info->obs_x > (*obsx) ) /* || (obs_info->obs_y > (*obsy) ) */ )
            status = 3;
    }

    assert( (status == 0) || (status == 3) );
    /* if bad status - do reallocation actions */
    if ( status )
    {
        float** tmp = hmm->obsProb;
        //float*  tmpf;

        /* distribute pointers of ehmm->obsProb */
/*        for( i = 0; i < hmm->num_states; i++ )
        {
            hmm->u.ehmm[i].obsProb = tmp;
            tmp += obs_info->obs_y;
        }
*/
        //tmpf = (float*)tmp;

        /* distribute pointers of ehmm->obsProb[j] */
/*      for( i = 0; i < hmm->num_states; i++ )
        {
            CvEHMM* ehmm = &( hmm->u.ehmm[i] );

            for( j = 0; j < obs_info->obs_y; j++ )
            {
                ehmm->obsProb[j] = tmpf;
                tmpf += ehmm->num_states * obs_info->obs_x;
            }
        }
*/
        hmm->obsProb = tmp;

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
        //for( i = 0; i < hmm->num_states; i++ )
        //{
        //    CvEHMM* ehmm = &(hmm->u.ehmm[i]);
            CvEHMMState* state = hmm->u.state;

            int max_mix = 0;
            for( j = 0; j < hmm->num_states; j++ )
            {
                int t = state[j].num_mix;
                if( max_mix < t ) max_mix = t;
            }
            max_mix *= hmm->num_states;
            /*if( max_size < max_mix )*/ max_size = max_mix;
        //}

        max_size *= obs_x * vect_size;

        /* allocate buffer */
        if( max_size > MAX_BUF_SIZE )
        {
            log_mix_prob = (float*)icvAlloc( max_size*(sizeof(float) + sizeof(double)));
            if( !log_mix_prob ) return CV_OUTOFMEM_ERR;
            mix_prob = (double*)(log_mix_prob + max_size);
        }

        memset( log_mix_prob, 0, max_size*sizeof(float));

        /*****************computing probabilities***********************/

        /* loop through external states */
        //for( i = 0; i < hmm->num_states; i++ )
        {
        //    CvEHMM* ehmm = &(hmm->u.ehmm[i]);
            CvEHMMState* state = hmm->u.state;

            int max_mix = 0;
            int n_states = hmm->num_states;

            /* determine maximal number of mixtures (again) */
            for( j = 0; j < hmm->num_states; j++ )
            {
                int t = state[j].num_mix;
                if( max_mix < t ) max_mix = t;
            }

            /* loop through rows of the observation matrix */
            //for( j = 0; j < obs_info->obs_y; j++ )
            {
                int  m, n;

                float* obs = obs_info->obs;/* + j * obs_x * vect_size; */
                float* log_mp = max_mix > 1 ? log_mix_prob : (float*)(hmm->obsProb);
                double* mp = mix_prob;

                /* several passes are done below */

                /* 1. calculate logarithms of probabilities for each mixture */

                /* loop through mixtures */
    /*  !!!! */     for( m = 0; m < max_mix; m++ )
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
                if( max_mix != 1 )
                {
                    /* 2. calculate exponent of log_mix_prob
                          (i.e. probability for each mixture) */
                    res = icvbExp_32f64f( log_mix_prob, mix_prob,
                                            max_mix * obs_x * n_states );
                    if( res < 0 ) goto processing_exit;

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
                    res = icvbLog_64f32f( mix_prob, (float*)(hmm->obsProb),//[j],
                                            obs_x * n_states );
                    if( res < 0 ) goto processing_exit;
                }
            }
        }

processing_exit:

        if( log_mix_prob != local_log_mix_prob ) icvFree( &log_mix_prob );
        return res;
#undef MAX_BUF_SIZE
    }
#else
/*    for( i = 0; i < hmm->num_states; i++ )
    {
        CvEHMM* ehmm = &(hmm->u.ehmm[i]);
        CvEHMMState* state = ehmm->u.state;

        for( j = 0; j < obs_info->obs_y; j++ )
        {
            int k,m;

            int obs_index = j * obs_info->obs_x;

            float* B = ehmm->obsProb[j];

            // cycles through obs and states
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
*/
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
CvStatus icvEstimate1DTransProb( Cv1DObsInfo** obs_info_array,
                                 int num_seq,
                                 CvEHMM* hmm )
{
    int    i, j, k;

    /* as a counter we will use transP matrix */

    /* initialization */

    /* clear transP */
    icvSetZero_32f( hmm->transP, hmm->num_states, hmm->num_states );


    /* compute the counters */
    for (i = 0; i < num_seq; i++)
    {
        int counter = 0;
        Cv1DObsInfo* info = obs_info_array[i];

        for (k = 0; k < info->obs_x; k++, counter++)
        {
            /* compute how many transitions from state to state
               occured */
            int state;
            int nextstate;

            state = info->state[counter];

            if (k < info->obs_x - 1)
            {
                int transP_size = hmm->num_states;

                nextstate = info->state[counter+1];
                hmm->transP[ state * transP_size + nextstate] += 1;
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

    return CV_NO_ERR;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: MixSegmL2
//    Purpose: The function implements the mixture segmentation of the states of the embedded HMM
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
CvStatus icv1DMixSegmL2(CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm )
{
    int     k, i, m;

    CvEHMMState* state = hmm->u.state;

    for (k = 0; k < num_img; k++)
    {
        //int counter = 0;
        CvImgObsInfo* info = obs_info_array[k];

        for (i = 0; i < info->obs_x; i++)
        {
            int e_state = info->state[i];
            float min_dist;

            min_dist = icvSquareDistance((info->obs) + (i * info->obs_size),
                                               state[e_state].mu, info->obs_size);
            info->mix[i] = 0;

            for (m = 1; m < state[e_state].num_mix; m++)
            {
                float dist=icvSquareDistance( (info->obs) + (i * info->obs_size),
                                               state[e_state].mu + m * info->obs_size,
                                               info->obs_size);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    /* assign mixture with smallest distance */
                    info->mix[i] = m;
                }
            }
        }
    }
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
float icvViterbi(Cv1DObsInfo* obs_info, CvEHMM* hmm)
{
    int    i, counter;
    float  log_likelihood;

    //CvEHMMState* first_state = hmm->u.state;

    /* memory allocation for superB */
    /*CvMatr32f superB = picvCreateMatrix_32f(hmm->num_states, obs_info->obs_x );*/

    /* memory allocation for q */
    int* super_q = (int*)icvAlloc( obs_info->obs_x * sizeof(int) );

    /* perform Viterbi segmentation (process 1D HMM) */
    icvViterbiSegmentation( hmm->num_states, obs_info->obs_x,
                            hmm->transP, (float*)(hmm->obsProb), 0,
                            _CV_LAST_STATE, &super_q, obs_info->obs_x,
                             obs_info->obs_x, &log_likelihood );

    log_likelihood /= obs_info->obs_x ;

    counter = 0;
    /* assign new state to observation vectors */
    for (i = 0; i < obs_info->obs_x; i++)
    {
         int state = super_q[i];
         obs_info->state[i] = state;
    }

    /* memory deallocation for superB */
    /*picvDeleteMatrix( superB );*/
    icvFree( &super_q );

    return log_likelihood;
}

CvStatus icvEstimate1DHMMStateParams(CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm)

{
    /* compute gamma, weights, means, vars */
    int k, i, j, m;
    int counter = 0;
    int total = 0;
    int vect_len = obs_info_array[0]->obs_size;

    float start_log_var_val = LN2PI * vect_len;

    CvVect32f tmp_vect = icvCreateVector_32f( vect_len );

    CvEHMMState* first_state = hmm->u.state;

    assert( sizeof(float) == sizeof(int) );

    total+= hmm->num_states;

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
    counter = 0;
    for (k = 0; k < num_img; k++)
    {
        CvImgObsInfo* info = obs_info_array[k];
        int num_obs = info->obs_y * info->obs_x;

        for (i = 0; i < num_obs; i++)
        {
            int state, mixture;
            state = info->state[i];
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
        int total_obs = info->obs_x;// * info->obs_y;

        float* vector = info->obs;

        for (j = 0; j < total_obs; j++, vector+=vect_len )
        {
            int state = info->state[j];
            int mixture = info->mix[j];

            CvVect32f mean  = first_state[state].mu + mixture * vect_len;
            CvVect32f mean2 = first_state[state].inv_var + mixture * vect_len;

            icvAddVector_32f( mean, vector, mean, vect_len );
            icvAddSquare_32f_C1IR( vector, vect_len * sizeof(float),
                                    mean2, vect_len * sizeof(float), cvSize(vect_len, 1) );
        }
    }

    /*compute the means and variances */
    /* assume gamma already computed */
    counter = 0;
    for (i = 0; i < total; i++)
    {
        CvEHMMState* state = &(first_state[i]);

        for (m = 0; m < state->num_mix; m++)
        {
            int k;
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

            /* low bound of variance - 0.01 (Ara's experimental result) */
            for( k = 0; k < vect_len; k++ )
            {
                invar[k] = (invar[k] > 0.01f) ? invar[k] : 0.01f;
            }

            /* compute log_var */
            state->log_var_val[m] = start_log_var_val;
            for( k = 0; k < vect_len; k++ )
            {
                state->log_var_val[m] += (float)log( invar[k] );
            }

            state->log_var_val[m] *= 0.5;

            /* compute inv_var = 1/sqrt(2*variance) */
            icvScaleVector_32f(invar, invar, vect_len, 2.f );
            icvbInvSqrt_32f(invar, invar, vect_len );
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





#endif

