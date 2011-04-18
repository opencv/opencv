/* Original code has been submitted by Liu Liu. Here is the copyright.
----------------------------------------------------------------------------------
 * An OpenCV Implementation of SURF
 * Further Information Refer to "SURF: Speed-Up Robust Feature"
 * Author: Liu Liu
 * liuliu.1987+opencv@gmail.com
 *
 * There are still serveral lacks for this experimental implementation:
 * 1.The interpolation of sub-pixel mentioned in article was not implemented yet;
 * 2.A comparision with original libSurf.so shows that the hessian detector is not a 100% match to their implementation;
 * 3.Due to above reasons, I recommanded the original one for study and reuse;
 *
 * However, the speed of this implementation is something comparable to original one.
 *
 * Copyright© 2008, Liu Liu All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 * 	Redistributions of source code must retain the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer.
 * 	Redistributions in binary form must reproduce the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer in the documentation and/or other materials
 * 	provided with the distribution.
 * 	The name of Contributor may not be used to endorse or
 * 	promote products derived from this software without
 * 	specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 */

/* 
   The following changes have been made, comparing to the original contribution:
   1. A lot of small optimizations, less memory allocations, got rid of global buffers
   2. Reversed order of cvGetQuadrangleSubPix and cvResize calls; probably less accurate, but much faster
   3. The descriptor computing part (which is most expensive) is threaded using OpenMP
   (subpixel-accurate keypoint localization and scale estimation are still TBD)
*/

/*
KeyPoint position and scale interpolation has been implemented as described in
the Brown and Lowe paper cited by the SURF paper.

The sampling step along the x and y axes of the image for the determinant of the
Hessian is now the same for each layer in an octave. While this increases the
computation time, it ensures that a true 3x3x3 neighbourhood exists, with
samples calculated at the same position in the layers above and below. This
results in improved maxima detection and non-maxima suppression, and I think it
is consistent with the description in the SURF paper.

The wavelet size sampling interval has also been made consistent. The wavelet
size at the first layer of the first octave is now 9 instead of 7. Along with
regular position sampling steps, this makes location and scale interpolation
easy. I think this is consistent with the SURF paper and original
implementation.

The scaling of the wavelet parameters has been fixed to ensure that the patterns
are symmetric around the centre. Previously the truncation caused by integer
division in the scaling ratio caused a bias towards the top left of the wavelet,
resulting in inconsistent keypoint positions.

The matrices for the determinant and trace of the Hessian are now reused in each
octave.

The extraction of the patch of pixels surrounding a keypoint used to build a
descriptor has been simplified.

KeyPoint descriptor normalisation has been changed from normalising each 4x4 
cell (resulting in a descriptor of magnitude 16) to normalising the entire 
descriptor to magnitude 1.

The default number of octaves has been increased from 3 to 4 to match the
original SURF binary default. The increase in computation time is minimal since
the higher octaves are sampled sparsely.

The default number of layers per octave has been reduced from 3 to 2, to prevent
redundant calculation of similar sizes in consecutive octaves.  This decreases 
computation time. The number of features extracted may be less, however the 
additional features were mostly redundant.

The radius of the circle of gradient samples used to assign an orientation has
been increased from 4 to 6 to match the description in the SURF paper. This is 
now defined by ORI_RADIUS, and could be made into a parameter.

The size of the sliding window used in orientation assignment has been reduced
from 120 to 60 degrees to match the description in the SURF paper. This is now
defined by ORI_WIN, and could be made into a parameter.

Other options like  HAAR_SIZE0, HAAR_SIZE_INC, SAMPLE_STEP0, ORI_SEARCH_INC, 
ORI_SIGMA and DESC_SIGMA have been separated from the code and documented. 
These could also be made into parameters.

Modifications by Ian Mahon

*/
#include "precomp.hpp"

CvSURFParams cvSURFParams(double threshold, int extended)
{
    CvSURFParams params;
    params.hessianThreshold = threshold;
    params.extended = extended;
    params.nOctaves = 4;
    params.nOctaveLayers = 2;
    return params;
}

struct CvSurfHF
{
    int p0, p1, p2, p3;
    float w;
};

CV_INLINE float
icvCalcHaarPattern( const int* origin, const CvSurfHF* f, int n )
{
    double d = 0;
    for( int k = 0; k < n; k++ )
        d += (origin[f[k].p0] + origin[f[k].p3] - origin[f[k].p1] - origin[f[k].p2])*f[k].w;
    return (float)d;
}

static void
icvResizeHaarPattern( const int src[][5], CvSurfHF* dst, int n, int oldSize, int newSize, int widthStep )
{
    float ratio = (float)newSize/oldSize;
    for( int k = 0; k < n; k++ )
    {
        int dx1 = cvRound( ratio*src[k][0] );
        int dy1 = cvRound( ratio*src[k][1] );
        int dx2 = cvRound( ratio*src[k][2] );
        int dy2 = cvRound( ratio*src[k][3] );
        dst[k].p0 = dy1*widthStep + dx1;
        dst[k].p1 = dy2*widthStep + dx1;
        dst[k].p2 = dy1*widthStep + dx2;
        dst[k].p3 = dy2*widthStep + dx2;
        dst[k].w = src[k][4]/((float)(dx2-dx1)*(dy2-dy1));
    }
}

/*
 * Calculate the determinant and trace of the Hessian for a layer of the
 * scale-space pyramid
 */
CV_INLINE void
icvCalcLayerDetAndTrace( const CvMat* sum, int size, int sampleStep, CvMat *det, CvMat *trace )
{
    const int NX=3, NY=3, NXY=4;
    const int dx_s[NX][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
    const int dy_s[NY][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
    const int dxy_s[NXY][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };

    CvSurfHF Dx[NX], Dy[NY], Dxy[NXY];
    double dx = 0, dy = 0, dxy = 0;
    int i, j, samples_i, samples_j, margin;
    int *sum_ptr;
    float *det_ptr, *trace_ptr;

    if( size>sum->rows-1 || size>sum->cols-1 )
       return;

    icvResizeHaarPattern( dx_s , Dx , NX , 9, size, sum->cols );
    icvResizeHaarPattern( dy_s , Dy , NY , 9, size, sum->cols );
    icvResizeHaarPattern( dxy_s, Dxy, NXY, 9, size, sum->cols );

    /* The integral image 'sum' is one pixel bigger than the source image */
    samples_i = 1+(sum->rows-1-size)/sampleStep;
    samples_j = 1+(sum->cols-1-size)/sampleStep;

    /* Ignore pixels where some of the kernel is outside the image */
    margin = (size/2)/sampleStep;

    for( i = 0; i < samples_i; i++ )
    {
        sum_ptr = sum->data.i + (i*sampleStep)*sum->cols;
        det_ptr = det->data.fl + (i+margin)*det->cols + margin;
        trace_ptr = trace->data.fl + (i+margin)*trace->cols + margin;
        for( j=0; j<samples_j; j++ )
        {
            dx  = icvCalcHaarPattern( sum_ptr, Dx , 3 );
            dy  = icvCalcHaarPattern( sum_ptr, Dy , 3 );
            dxy = icvCalcHaarPattern( sum_ptr, Dxy, 4 );
            sum_ptr += sampleStep;
            *det_ptr++ = (float)(dx*dy - 0.81*dxy*dxy);
            *trace_ptr++ = (float)(dx + dy);
        }
    }
}


/*
 * Maxima location interpolation as described in "Invariant Features from
 * Interest Point Groups" by Matthew Brown and David Lowe. This is performed by
 * fitting a 3D quadratic to a set of neighbouring samples.
 * 
 * The gradient vector and Hessian matrix at the initial keypoint location are 
 * approximated using central differences. The linear system Ax = b is then
 * solved, where A is the Hessian, b is the negative gradient, and x is the 
 * offset of the interpolated maxima coordinates from the initial estimate.
 * This is equivalent to an iteration of Netwon's optimisation algorithm.
 *
 * N9 contains the samples in the 3x3x3 neighbourhood of the maxima
 * dx is the sampling step in x
 * dy is the sampling step in y
 * ds is the sampling step in size
 * point contains the keypoint coordinates and scale to be modified
 *
 * Return value is 1 if interpolation was successful, 0 on failure.
 */
CV_INLINE int
icvInterpolateKeypoint( float N9[3][9], int dx, int dy, int ds, CvSURFPoint *point )
{
    int solve_ok;
    float A[9], x[3], b[3];
    CvMat matA = cvMat(3, 3, CV_32F, A);
    CvMat _x   = cvMat(3, 1, CV_32F, x);
    CvMat _b   = cvMat(3, 1, CV_32F, b);

    b[0] = -(N9[1][5]-N9[1][3])/2;  /* Negative 1st deriv with respect to x */
    b[1] = -(N9[1][7]-N9[1][1])/2;  /* Negative 1st deriv with respect to y */
    b[2] = -(N9[2][4]-N9[0][4])/2;  /* Negative 1st deriv with respect to s */

    A[0] = N9[1][3]-2*N9[1][4]+N9[1][5];            /* 2nd deriv x, x */
    A[1] = (N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4; /* 2nd deriv x, y */
    A[2] = (N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4; /* 2nd deriv x, s */
    A[3] = A[1];                                    /* 2nd deriv y, x */
    A[4] = N9[1][1]-2*N9[1][4]+N9[1][7];            /* 2nd deriv y, y */
    A[5] = (N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4; /* 2nd deriv y, s */
    A[6] = A[2];                                    /* 2nd deriv s, x */
    A[7] = A[5];                                    /* 2nd deriv s, y */
    A[8] = N9[0][4]-2*N9[1][4]+N9[2][4];            /* 2nd deriv s, s */

    solve_ok = cvSolve( &matA, &_b, &_x );
    if( solve_ok )
    {
        if (x[0] > 1 || x[0] <-1 || x[1] > 1 || x[1] <-1 || x[2] > 1 || x[2] <-1 )
            solve_ok = 0;
        else
        {
            point->pt.x += x[0]*dx;
            point->pt.y += x[1]*dy;
            point->size = cvRound( point->size + x[2]*ds );
        }
    }
    return solve_ok;
}

/*
 * Find the maxima in the determinant of the Hessian in a layer of the 
 * scale-space pyramid
 */ 
CV_INLINE void
icvFindMaximaInLayer( const CvMat *sum, const CvMat* mask_sum, const CvSURFParams* params,
                      CvMat **dets, CvMat **traces, const int *sizes, 
                      int layer, int sampleStep, CvSeq* points )
{
    /* Wavelet Data */
    const int NM=1;
    const int dm[NM][5] = { {0, 0, 9, 9, 1} };

    CvSurfHF Dm;
    int i, j, size, margin, layer_rows, layer_cols;
    float *det_ptr, *trace_ptr;

    size = sizes[layer];

    /* The integral image 'sum' is one pixel bigger than the source image */
    layer_rows = (sum->rows-1)/sampleStep;
    layer_cols = (sum->cols-1)/sampleStep;

    /* Ignore pixels without a 3x3x3 neighbourhood in the layer above */
    margin = (sizes[layer+1]/2)/sampleStep+1; 

    if( mask_sum )
       icvResizeHaarPattern( dm, &Dm, NM, 9, size, mask_sum->cols );

    for( i = margin; i < layer_rows-margin; i++ )
    {
        det_ptr = dets[layer]->data.fl + i*dets[layer]->cols;
        trace_ptr = traces[layer]->data.fl + i*traces[layer]->cols;
        for( j = margin; j < layer_cols-margin; j++ )
        {
            float val0 = det_ptr[j];
            if( val0 > params->hessianThreshold )
            {
                /* Coordinates for the start of the wavelet in the sum image. There   
                   is some integer division involved, so don't try to simplify this
                   (cancel out sampleStep) without checking the result is the same */
                int sum_i = sampleStep*(i-(size/2)/sampleStep);
                int sum_j = sampleStep*(j-(size/2)/sampleStep);

                /* The 3x3x3 neighbouring samples around the maxima. 
                   The maxima is included at N9[1][4] */
                int c = dets[layer]->cols;
                const float *det1 = dets[layer-1]->data.fl + i*c + j;
                const float *det2 = dets[layer]->data.fl   + i*c + j;
                const float *det3 = dets[layer+1]->data.fl + i*c + j;
                float N9[3][9] = { { det1[-c-1], det1[-c], det1[-c+1],
                                     det1[-1]  , det1[0] , det1[1],
                                     det1[c-1] , det1[c] , det1[c+1]  },
                                   { det2[-c-1], det2[-c], det2[-c+1],
                                     det2[-1]  , det2[0] , det2[1],
                                     det2[c-1] , det2[c] , det2[c+1]  },
                                   { det3[-c-1], det3[-c], det3[-c+1],
                                     det3[-1]  , det3[0] , det3[1],
                                     det3[c-1] , det3[c] , det3[c+1]  } };

                /* Check the mask - why not just check the mask at the center of the wavelet? */
                if( mask_sum )
                {
                    const int* mask_ptr = mask_sum->data.i +  mask_sum->cols*sum_i + sum_j;
                    float mval = icvCalcHaarPattern( mask_ptr, &Dm, 1 );
                    if( mval < 0.5 )
                        continue;
                }

                /* Non-maxima suppression. val0 is at N9[1][4]*/
                if( val0 > N9[0][0] && val0 > N9[0][1] && val0 > N9[0][2] &&
                    val0 > N9[0][3] && val0 > N9[0][4] && val0 > N9[0][5] &&
                    val0 > N9[0][6] && val0 > N9[0][7] && val0 > N9[0][8] &&
                    val0 > N9[1][0] && val0 > N9[1][1] && val0 > N9[1][2] &&
                    val0 > N9[1][3]                    && val0 > N9[1][5] &&
                    val0 > N9[1][6] && val0 > N9[1][7] && val0 > N9[1][8] &&
                    val0 > N9[2][0] && val0 > N9[2][1] && val0 > N9[2][2] &&
                    val0 > N9[2][3] && val0 > N9[2][4] && val0 > N9[2][5] &&
                    val0 > N9[2][6] && val0 > N9[2][7] && val0 > N9[2][8] )
                {
                    /* Calculate the wavelet center coordinates for the maxima */
                    double center_i = sum_i + (double)(size-1)/2;
                    double center_j = sum_j + (double)(size-1)/2;

                    CvSURFPoint point = cvSURFPoint( cvPoint2D32f(center_j,center_i),
                                                     CV_SIGN(trace_ptr[j]), sizes[layer], 0, val0 );

                    /* Interpolate maxima location within the 3x3x3 neighbourhood  */
                    int ds = size-sizes[layer-1];
                    int interp_ok = icvInterpolateKeypoint( N9, sampleStep, sampleStep, ds, &point );

                    /* Sometimes the interpolation step gives a negative size etc. */
                    if( interp_ok  )
                    {
                        /*printf( "KeyPoint %f %f %d\n", point.pt.x, point.pt.y, point.size );*/
                    #ifdef HAVE_TBB
                        static tbb::mutex m;
                        tbb::mutex::scoped_lock lock(m);
                    #endif
                        cvSeqPush( points, &point );
                    }
                }
            }
        }
    }
}


namespace cv
{
/* Multi-threaded construction of the scale-space pyramid */
struct SURFBuildInvoker
{
    SURFBuildInvoker( const CvMat *_sum, const int *_sizes, const int *_sampleSteps,
                      CvMat** _dets, CvMat** _traces )
    {
        sum = _sum;
        sizes = _sizes;
        sampleSteps = _sampleSteps;
        dets = _dets;
        traces = _traces;
    }

    void operator()(const BlockedRange& range) const
    {
        for( int i=range.begin(); i<range.end(); i++ )
            icvCalcLayerDetAndTrace( sum, sizes[i], sampleSteps[i], dets[i], traces[i] );
    }

    const CvMat *sum;
    const int *sizes;
    const int *sampleSteps;
    CvMat** dets;
    CvMat** traces;
};

/* Multi-threaded search of the scale-space pyramid for keypoints */
struct SURFFindInvoker
{
    SURFFindInvoker( const CvMat *_sum, const CvMat *_mask_sum, const CvSURFParams* _params,
                     CvMat** _dets, CvMat** _traces,  const int *_sizes,
                     const int *_sampleSteps, const int *_middleIndices, CvSeq* _points )

    {
       sum = _sum;
       mask_sum = _mask_sum;
       params = _params;
       dets = _dets;
       traces = _traces;
       sizes = _sizes;
       sampleSteps = _sampleSteps;
       middleIndices = _middleIndices;
       points = _points;
    }

    void operator()(const BlockedRange& range) const
    {
        for( int i=range.begin(); i<range.end(); i++ )
        {
            int layer = middleIndices[i];
            icvFindMaximaInLayer( sum, mask_sum, params, dets, traces, sizes, layer, 
                                  sampleSteps[layer], points );
        }
    }

    const CvMat *sum;
    const CvMat *mask_sum;
    const CvSURFParams* params;
    CvMat** dets;
    CvMat** traces;
    const int *sizes;
    const int *sampleSteps;
    const int *middleIndices;
    CvSeq* points;
};

} // namespace cv



/* Wavelet size at first layer of first octave. */ 
const int HAAR_SIZE0 = 9;

/* Wavelet size increment between layers. This should be an even number,
 such that the wavelet sizes in an octave are either all even or all odd.
 This ensures that when looking for the neighbours of a sample, the layers
 above and below are aligned correctly. */
const int HAAR_SIZE_INC = 6;


static CvSeq* icvFastHessianDetector( const CvMat* sum, const CvMat* mask_sum,
    CvMemStorage* storage, const CvSURFParams* params )
{
    CvSeq* points = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFPoint), storage );

    /* Sampling step along image x and y axes at first octave. This is doubled
       for each additional octave. WARNING: Increasing this improves speed,
       however keypoint extraction becomes unreliable. */
    const int SAMPLE_STEP0 = 1;

    int nTotalLayers = (params->nOctaveLayers+2)*params->nOctaves;
    int nMiddleLayers = params->nOctaveLayers*params->nOctaves;

    cv::AutoBuffer<CvMat*> dets(nTotalLayers);
    cv::AutoBuffer<CvMat*> traces(nTotalLayers);
    cv::AutoBuffer<int> sizes(nTotalLayers);
    cv::AutoBuffer<int> sampleSteps(nTotalLayers);
    cv::AutoBuffer<int> middleIndices(nMiddleLayers);
    int octave, layer, step, index, middleIndex;

    /* Allocate space and calculate properties of each layer */
    index = 0;
    middleIndex = 0;
    step = SAMPLE_STEP0;
    for( octave=0; octave<params->nOctaves; octave++ )
    {
        for( layer=0; layer<params->nOctaveLayers+2; layer++ )
        {
            /* The integral image sum is one pixel bigger than the source image*/
            dets[index]   = cvCreateMat( (sum->rows-1)/step, (sum->cols-1)/step, CV_32FC1 );
            traces[index] = cvCreateMat( (sum->rows-1)/step, (sum->cols-1)/step, CV_32FC1 );
            sizes[index] = (HAAR_SIZE0+HAAR_SIZE_INC*layer)<<octave;
            sampleSteps[index] = step;

            if( layer!=0 && layer!=params->nOctaveLayers+1 )
               middleIndices[middleIndex++] = index;
            index++;
        }
        step*=2;
    }

    /* Calculate hessian determinant and trace samples in each layer*/
    cv::parallel_for( cv::BlockedRange(0, nTotalLayers),
                      cv::SURFBuildInvoker(sum,sizes,sampleSteps,dets,traces) );

    /* Find maxima in the determinant of the hessian */
    cv::parallel_for( cv::BlockedRange(0, nMiddleLayers),
                      cv::SURFFindInvoker(sum,mask_sum,params,dets,traces,sizes,
                                          sampleSteps,middleIndices,points) );

    /* Clean-up */
    for( layer = 0; layer < nTotalLayers; layer++ )
    {
        cvReleaseMat( &dets[layer] );
        cvReleaseMat( &traces[layer] );
    }

    return points;
}


namespace cv
{

/* Methods to free data allocated in SURFInvoker constructor */
template<> inline void Ptr<float>::delete_obj()   { cvFree(&obj); }
template<> inline void Ptr<CvPoint>::delete_obj() { cvFree(&obj); }

struct SURFInvoker
{
    enum { ORI_RADIUS = 6, ORI_WIN = 60, PATCH_SZ = 20 };

    static const int   ORI_SEARCH_INC;
    static const float ORI_SIGMA;
    static const float DESC_SIGMA;

    SURFInvoker( const CvSURFParams* _params,
                 CvSeq* _keypoints, CvSeq* _descriptors,
                 const CvMat* _img, const CvMat* _sum )
    {
        params = _params;
        keypoints = _keypoints;
        descriptors = _descriptors;
        img = _img;
        sum = _sum;

        /* Simple bound for number of grid points in circle of radius ORI_RADIUS */
        const int nOriSampleBound = (2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);

        /* Allocate arrays */
        apt = (CvPoint*)cvAlloc(nOriSampleBound*sizeof(CvPoint));
        aptw = (float*)cvAlloc(nOriSampleBound*sizeof(float));
        DW = (float*)cvAlloc(PATCH_SZ*PATCH_SZ*sizeof(float));

        /* Coordinates and weights of samples used to calculate orientation */
        cv::Mat G_ori = cv::getGaussianKernel( 2*ORI_RADIUS+1, ORI_SIGMA, CV_32F );
        nOriSamples = 0;
        for( int i = -ORI_RADIUS; i <= ORI_RADIUS; i++ )
        {
            for( int j = -ORI_RADIUS; j <= ORI_RADIUS; j++ )
            {
                if( i*i + j*j <= ORI_RADIUS*ORI_RADIUS )
                {
                    apt[nOriSamples] = cvPoint(i,j);
                    aptw[nOriSamples++] = G_ori.at<float>(i+ORI_RADIUS,0) * G_ori.at<float>(j+ORI_RADIUS,0);
                }
            }
        }
        CV_Assert( nOriSamples <= nOriSampleBound );

        /* Gaussian used to weight descriptor samples */
        cv::Mat G_desc = cv::getGaussianKernel( PATCH_SZ, DESC_SIGMA, CV_32F );
        for( int i = 0; i < PATCH_SZ; i++ )
        {
            for( int j = 0; j < PATCH_SZ; j++ )
                DW[i*PATCH_SZ+j] = G_desc.at<float>(i,0) * G_desc.at<float>(j,0);
        }
    }

    void operator()(const BlockedRange& range) const
    {
        /* X and Y gradient wavelet data */
        const int NX=2, NY=2;
        const int dx_s[NX][5] = {{0, 0, 2, 4, -1}, {2, 0, 4, 4, 1}};
        const int dy_s[NY][5] = {{0, 0, 4, 2, 1}, {0, 2, 4, 4, -1}};

        const int descriptor_size = params->extended ? 128 : 64;
        /* Optimisation is better using nOriSampleBound than nOriSamples for 
         array lengths.  Maybe because it is a constant known at compile time */
        const int nOriSampleBound =(2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);

        float X[nOriSampleBound], Y[nOriSampleBound], angle[nOriSampleBound];
        uchar PATCH[PATCH_SZ+1][PATCH_SZ+1];
        float DX[PATCH_SZ][PATCH_SZ], DY[PATCH_SZ][PATCH_SZ];
        CvMat matX = cvMat(1, nOriSampleBound, CV_32F, X);
        CvMat matY = cvMat(1, nOriSampleBound, CV_32F, Y);
        CvMat _angle = cvMat(1, nOriSampleBound, CV_32F, angle);
        CvMat _patch = cvMat(PATCH_SZ+1, PATCH_SZ+1, CV_8U, PATCH);

        int k, k1 = range.begin(), k2 = range.end();
        int maxSize = 0;
        for( k = k1; k < k2; k++ )
        {
            maxSize = std::max(maxSize, ((CvSURFPoint*)cvGetSeqElem( keypoints, k ))->size);
        }
        maxSize = cvCeil((PATCH_SZ+1)*maxSize*1.2f/9.0f);
        Ptr<CvMat> winbuf = cvCreateMat( 1, maxSize > 0 ? maxSize*maxSize : 1, CV_8U );
        for( k = k1; k < k2; k++ )
        {
            const int* sum_ptr = sum->data.i;
            int sum_cols = sum->cols;
            int i, j, kk, x, y, nangle;
            float* vec;
            CvSurfHF dx_t[NX], dy_t[NY];
            CvSURFPoint* kp = (CvSURFPoint*)cvGetSeqElem( keypoints, k );
            int size = kp->size;
            CvPoint2D32f center = kp->pt;
            /* The sampling intervals and wavelet sized for selecting an orientation
             and building the keypoint descriptor are defined relative to 's' */
            float s = (float)size*1.2f/9.0f;
            /* To find the dominant orientation, the gradients in x and y are
             sampled in a circle of radius 6s using wavelets of size 4s.
             We ensure the gradient wavelet size is even to ensure the 
             wavelet pattern is balanced and symmetric around its center */
            int grad_wav_size = 2*cvRound( 2*s );
            if ( sum->rows < grad_wav_size || sum->cols < grad_wav_size )
            {
                /* when grad_wav_size is too big,
                 * the sampling of gradient will be meaningless
                 * mark keypoint for deletion. */
                kp->size = -1;
                continue;
            }
            icvResizeHaarPattern( dx_s, dx_t, NX, 4, grad_wav_size, sum->cols );
            icvResizeHaarPattern( dy_s, dy_t, NY, 4, grad_wav_size, sum->cols );
            for( kk = 0, nangle = 0; kk < nOriSamples; kk++ )
            {
                const int* ptr;
                float vx, vy;
                x = cvRound( center.x + apt[kk].x*s - (float)(grad_wav_size-1)/2 );
                y = cvRound( center.y + apt[kk].y*s - (float)(grad_wav_size-1)/2 );
                if( (unsigned)y >= (unsigned)(sum->rows - grad_wav_size) ||
                   (unsigned)x >= (unsigned)(sum->cols - grad_wav_size) )
                    continue;
                ptr = sum_ptr + x + y*sum_cols;
                vx = icvCalcHaarPattern( ptr, dx_t, 2 );
                vy = icvCalcHaarPattern( ptr, dy_t, 2 );
                X[nangle] = vx*aptw[kk]; Y[nangle] = vy*aptw[kk];
                nangle++;
            }
            if ( nangle == 0 )
            {
                /* No gradient could be sampled because the keypoint is too
                 * near too one or more of the sides of the image. As we
                 * therefore cannot find a dominant direction, we skip this
                 * keypoint and mark it for later deletion from the sequence. */
                kp->size = -1;
                continue;
            }
            matX.cols = matY.cols = _angle.cols = nangle;
            cvCartToPolar( &matX, &matY, 0, &_angle, 1 );

            float bestx = 0, besty = 0, descriptor_mod = 0;
            for( i = 0; i < 360; i += ORI_SEARCH_INC )
            {
                float sumx = 0, sumy = 0, temp_mod;
                for( j = 0; j < nangle; j++ )
                {
                    int d = std::abs(cvRound(angle[j]) - i);
                    if( d < ORI_WIN/2 || d > 360-ORI_WIN/2 )
                    {
                        sumx += X[j];
                        sumy += Y[j];
                    }
                }
                temp_mod = sumx*sumx + sumy*sumy;
                if( temp_mod > descriptor_mod )
                {
                    descriptor_mod = temp_mod;
                    bestx = sumx;
                    besty = sumy;
                }
            }
            float descriptor_dir = cvFastArctan( besty, bestx );
            kp->dir = descriptor_dir;
            if( !descriptors )
                continue;
            descriptor_dir *= (float)(CV_PI/180);
            /* Extract a window of pixels around the keypoint of size 20s */
            int win_size = (int)((PATCH_SZ+1)*s);
            CV_Assert( winbuf->cols >= win_size*win_size );
            CvMat win = cvMat(win_size, win_size, CV_8U, winbuf->data.ptr);
            float sin_dir = sin(descriptor_dir);
            float cos_dir = cos(descriptor_dir) ;

            /* Subpixel interpolation version (slower). Subpixel not required since
             the pixels will all get averaged when we scale down to 20 pixels */
            /*
             float w[] = { cos_dir, sin_dir, center.x,
             -sin_dir, cos_dir , center.y };
             CvMat W = cvMat(2, 3, CV_32F, w);
             cvGetQuadrangleSubPix( img, &win, &W );
             */

            /* Nearest neighbour version (faster) */
            float win_offset = -(float)(win_size-1)/2;
            float start_x = center.x + win_offset*cos_dir + win_offset*sin_dir;
            float start_y = center.y - win_offset*sin_dir + win_offset*cos_dir;
            uchar* WIN = win.data.ptr;
            for( i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir )
            {
                float pixel_x = start_x;
                float pixel_y = start_y;
                for( j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir )
                {
                    int x = std::min(std::max(cvRound(pixel_x), 0), img->cols-1);
                    int y = std::min(std::max(cvRound(pixel_y), 0), img->rows-1);
                    WIN[i*win_size + j] = img->data.ptr[y*img->step + x];
                }
            }

            /* Scale the window to size PATCH_SZ so each pixel's size is s. This
             makes calculating the gradients with wavelets of size 2s easy */
            cvResize( &win, &_patch, CV_INTER_AREA );

            /* Calculate gradients in x and y with wavelets of size 2s */
            for( i = 0; i < PATCH_SZ; i++ )
                for( j = 0; j < PATCH_SZ; j++ )
                {
                    float dw = DW[i*PATCH_SZ + j];
                    float vx = (PATCH[i][j+1] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i+1][j])*dw;
                    float vy = (PATCH[i+1][j] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i][j+1])*dw;
                    DX[i][j] = vx;
                    DY[i][j] = vy;
                }

            /* Construct the descriptor */
            vec = (float*)cvGetSeqElem( descriptors, k );
            for( kk = 0; kk < (int)(descriptors->elem_size/sizeof(vec[0])); kk++ )
                vec[kk] = 0;
            double square_mag = 0;
            if( params->extended )
            {
                /* 128-bin descriptor */
                for( i = 0; i < 4; i++ )
                    for( j = 0; j < 4; j++ )
                    {
                        for( y = i*5; y < i*5+5; y++ )
                        {
                            for( x = j*5; x < j*5+5; x++ )
                            {
                                float tx = DX[y][x], ty = DY[y][x];
                                if( ty >= 0 )
                                {
                                    vec[0] += tx;
                                    vec[1] += (float)fabs(tx);
                                } else {
                                    vec[2] += tx;
                                    vec[3] += (float)fabs(tx);
                                }
                                if ( tx >= 0 )
                                {
                                    vec[4] += ty;
                                    vec[5] += (float)fabs(ty);
                                } else {
                                    vec[6] += ty;
                                    vec[7] += (float)fabs(ty);
                                }
                            }
                        }
                        for( kk = 0; kk < 8; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec += 8;
                    }
            }
            else
            {
                /* 64-bin descriptor */
                for( i = 0; i < 4; i++ )
                    for( j = 0; j < 4; j++ )
                    {
                        for( y = i*5; y < i*5+5; y++ )
                        {
                            for( x = j*5; x < j*5+5; x++ )
                            {
                                float tx = DX[y][x], ty = DY[y][x];
                                vec[0] += tx; vec[1] += ty;
                                vec[2] += (float)fabs(tx); vec[3] += (float)fabs(ty);
                            }
                        }
                        for( kk = 0; kk < 4; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec+=4;
                    }
            }

            /* unit vector is essential for contrast invariance */
            vec = (float*)cvGetSeqElem( descriptors, k );
            double scale = 1./(sqrt(square_mag) + DBL_EPSILON);
            for( kk = 0; kk < descriptor_size; kk++ )
                vec[kk] = (float)(vec[kk]*scale);
        }
    }

    /* Parameters */
    const CvSURFParams* params;
    const CvMat* img;
    const CvMat* sum;
    CvSeq* keypoints;
    CvSeq* descriptors;

    /* Pre-calculated values */
    int nOriSamples;
    cv::Ptr<CvPoint> apt;
    cv::Ptr<float> aptw;
    cv::Ptr<float> DW;
};

const int   SURFInvoker::ORI_SEARCH_INC = 5;
const float SURFInvoker::ORI_SIGMA      = 2.5f;
const float SURFInvoker::DESC_SIGMA     = 3.3f;
}


CV_IMPL void
cvExtractSURF( const CvArr* _img, const CvArr* _mask,
               CvSeq** _keypoints, CvSeq** _descriptors,
               CvMemStorage* storage, CvSURFParams params,
               int useProvidedKeyPts)
{
    CvMat *sum = 0, *mask1 = 0, *mask_sum = 0;

    if( _keypoints && !useProvidedKeyPts ) // If useProvidedKeyPts!=0 we'll use current contents of "*_keypoints"
        *_keypoints = 0;
    if( _descriptors )
        *_descriptors = 0;

    CvSeq *keypoints, *descriptors = 0;
    CvMat imghdr, *img = cvGetMat(_img, &imghdr);
    CvMat maskhdr, *mask = _mask ? cvGetMat(_mask, &maskhdr) : 0;

    int descriptor_size = params.extended ? 128 : 64;
    const int descriptor_data_type = CV_32F;
    int i, N;

    CV_Assert(img != 0);
    CV_Assert(CV_MAT_TYPE(img->type) == CV_8UC1);
    CV_Assert(mask == 0 || (CV_ARE_SIZES_EQ(img,mask) && CV_MAT_TYPE(mask->type) == CV_8UC1));
    CV_Assert(storage != 0);
    CV_Assert(params.hessianThreshold >= 0);
    CV_Assert(params.nOctaves > 0);
    CV_Assert(params.nOctaveLayers > 0);

    sum = cvCreateMat( img->rows+1, img->cols+1, CV_32SC1 );
    cvIntegral( img, sum );

    // Compute keypoints only if we are not asked for evaluating the descriptors are some given locations:
    if (!useProvidedKeyPts)
    {
        if( mask )
        {
            mask1 = cvCreateMat( img->height, img->width, CV_8UC1 );
            mask_sum = cvCreateMat( img->height+1, img->width+1, CV_32SC1 );
            cvMinS( mask, 1, mask1 );
            cvIntegral( mask1, mask_sum );
        }
        keypoints = icvFastHessianDetector( sum, mask_sum, storage, &params );
    }
    else
    {
        CV_Assert(useProvidedKeyPts && (_keypoints != 0) && (*_keypoints != 0));
        keypoints = *_keypoints;
    }

    N = keypoints->total;
    if( _descriptors )
    {
        descriptors = cvCreateSeq( 0, sizeof(CvSeq),
            descriptor_size*CV_ELEM_SIZE(descriptor_data_type), storage );
        cvSeqPushMulti( descriptors, 0, N );
    }


    if ( N > 0 )
        cv::parallel_for(cv::BlockedRange(0, N),
                     cv::SURFInvoker(&params, keypoints, descriptors, img, sum) );


    /* remove keypoints that were marked for deletion */
    for ( i = 0; i < N; i++ )
    {
        CvSURFPoint* kp = (CvSURFPoint*)cvGetSeqElem( keypoints, i );
        if ( kp->size == -1 )
        {
            cvSeqRemove( keypoints, i );
            if ( _descriptors )
                cvSeqRemove( descriptors, i );
            i--;
            N--;
        }
    }

    if( _keypoints && !useProvidedKeyPts )
        *_keypoints = keypoints;
    if( _descriptors )
        *_descriptors = descriptors;

    cvReleaseMat( &sum );
    if (mask1) cvReleaseMat( &mask1 );
    if (mask_sum) cvReleaseMat( &mask_sum );
}


namespace cv
{

SURF::SURF()
{
    hessianThreshold = 100;
    extended = 1;
    nOctaves = 4;
    nOctaveLayers = 2;
}

SURF::SURF(double _threshold, int _nOctaves, int _nOctaveLayers, bool _extended)
{
    hessianThreshold = _threshold;
    extended = _extended;
    nOctaves = _nOctaves;
    nOctaveLayers = _nOctaveLayers;
}

int SURF::descriptorSize() const { return extended ? 128 : 64; }


static int getPointOctave(const CvSURFPoint& kpt, const CvSURFParams& params)
{
    int octave = 0, layer = 0, best_octave = 0;
    float min_diff = FLT_MAX;
    for( octave = 1; octave < params.nOctaves; octave++ )
        for( layer = 0; layer < params.nOctaveLayers; layer++ )
        {
            float diff = std::abs(kpt.size - (float)((HAAR_SIZE0 + HAAR_SIZE_INC*layer) << octave));
            if( min_diff > diff )
            {
                min_diff = diff;
                best_octave = octave;
                if( min_diff == 0 )
                    return best_octave;
            }
        }
    return best_octave;
}


void SURF::operator()(const Mat& img, const Mat& mask,
                      vector<KeyPoint>& keypoints) const
{
    CvMat _img = img, _mask, *pmask = 0;
    if( mask.data )
        pmask = &(_mask = mask);
    MemStorage storage(cvCreateMemStorage(0));
    Seq<CvSURFPoint> kp;
    cvExtractSURF(&_img, pmask, &kp.seq, 0, storage, *(const CvSURFParams*)this, 0);
    Seq<CvSURFPoint>::iterator it = kp.begin();
    size_t i, n = kp.size();
    keypoints.resize(n);
    for( i = 0; i < n; i++, ++it )
    {
        const CvSURFPoint& kpt = *it;
        keypoints[i] = KeyPoint(kpt.pt, (float)kpt.size, kpt.dir,
                                kpt.hessian, getPointOctave(kpt, *this));
    }
}

void SURF::operator()(const Mat& img, const Mat& mask,
                vector<KeyPoint>& keypoints,
                vector<float>& descriptors,
                bool useProvidedKeypoints) const
{
    CvMat _img = img, _mask, *pmask = 0;
    if( mask.data )
        pmask = &(_mask = mask);
    MemStorage storage(cvCreateMemStorage(0));
    Seq<CvSURFPoint> kp;
    CvSeq* d = 0;
    size_t i, n;
    if( useProvidedKeypoints )
    {
        kp = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFPoint), storage);
        n = keypoints.size();
        for( i = 0; i < n; i++ )
        {
            const KeyPoint& kpt = keypoints[i];
            kp.push_back(cvSURFPoint(kpt.pt, 1, cvRound(kpt.size), kpt.angle, kpt.response));
        }
    }

    cvExtractSURF(&_img, pmask, &kp.seq, &d, storage,
        *(const CvSURFParams*)this, useProvidedKeypoints);

    // input keypoints can be filtered in cvExtractSURF()
    if( !useProvidedKeypoints || (useProvidedKeypoints && keypoints.size() != kp.size()) )
    {
        Seq<CvSURFPoint>::iterator it = kp.begin();
        size_t i, n = kp.size();
        keypoints.resize(n);
        for( i = 0; i < n; i++, ++it )
        {
            const CvSURFPoint& kpt = *it;
            keypoints[i] = KeyPoint(kpt.pt, (float)kpt.size, kpt.dir,
                                    kpt.hessian, getPointOctave(kpt, *this),
                                    kpt.laplacian);
        }
    }
    descriptors.resize(d ? d->total*d->elem_size/sizeof(float) : 0);
    if(descriptors.size() != 0)
        cvCvtSeqToArray(d, &descriptors[0]);
}

}
