Motion Analysis and Object Tracking
===================================

.. highlight:: c



.. index:: CalcGlobalOrientation

.. _CalcGlobalOrientation:

CalcGlobalOrientation
---------------------

`id=0.848432169537 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CalcGlobalOrientation>`__




.. cfunction:: double cvCalcGlobalOrientation(  const CvArr* orientation, const CvArr* mask, const CvArr* mhi, double timestamp, double duration )

    Calculates the global motion orientation of some selected region.





    
    :param orientation: Motion gradient orientation image; calculated by the function  :ref:`CalcMotionGradient` 
    
    
    :param mask: Mask image. It may be a conjunction of a valid gradient mask, obtained with  :ref:`CalcMotionGradient`  and the mask of the region, whose direction needs to be calculated 
    
    
    :param mhi: Motion history image 
    
    
    :param timestamp: Current time in milliseconds or other units, it is better to store time passed to  :ref:`UpdateMotionHistory`  before and reuse it here, because running  :ref:`UpdateMotionHistory`  and  :ref:`CalcMotionGradient`  on large images may take some time 
    
    
    :param duration: Maximal duration of motion track in milliseconds, the same as  :ref:`UpdateMotionHistory` 
    
    
    
The function calculates the general
motion direction in the selected region and returns the angle between
0 degrees  and 360 degrees . At first the function builds the orientation histogram
and finds the basic orientation as a coordinate of the histogram
maximum. After that the function calculates the shift relative to the
basic orientation as a weighted sum of all of the orientation vectors: the more
recent the motion, the greater the weight. The resultant angle is
a circular sum of the basic orientation and the shift.


.. index:: CalcMotionGradient

.. _CalcMotionGradient:

CalcMotionGradient
------------------

`id=0.691063668639 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CalcMotionGradient>`__




.. cfunction:: void cvCalcMotionGradient(  const CvArr* mhi, CvArr* mask, CvArr* orientation, double delta1, double delta2, int apertureSize=3 )

    Calculates the gradient orientation of a motion history image.





    
    :param mhi: Motion history image 
    
    
    :param mask: Mask image; marks pixels where the motion gradient data is correct; output parameter 
    
    
    :param orientation: Motion gradient orientation image; contains angles from 0 to ~360 degrees  
    
    
    :param delta1: See below 
    
    
    :param delta2: See below 
    
    
    :param apertureSize: Aperture size of derivative operators used by the function: CV _ SCHARR, 1, 3, 5 or 7 (see  :ref:`Sobel` ) 
    
    
    
The function calculates the derivatives 
:math:`Dx`
and 
:math:`Dy`
of 
``mhi``
and then calculates gradient orientation as:



.. math::

    \texttt{orientation} (x,y)= \arctan{\frac{Dy(x,y)}{Dx(x,y)}} 


where both 
:math:`Dx(x,y)`
and 
:math:`Dy(x,y)`
signs are taken into account (as in the 
:ref:`CartToPolar`
function). After that 
``mask``
is filled to indicate where the orientation is valid (see the 
``delta1``
and 
``delta2``
description).

The function finds the minimum (
:math:`m(x,y)`
) and maximum (
:math:`M(x,y)`
) mhi values over each pixel 
:math:`(x,y)`
neighborhood and assumes the gradient is valid only if


.. math::

    \min ( \texttt{delta1} ,  \texttt{delta2} )  \le M(x,y)-m(x,y)  \le \max ( \texttt{delta1} , \texttt{delta2} ). 



.. index:: CalcOpticalFlowBM

.. _CalcOpticalFlowBM:

CalcOpticalFlowBM
-----------------

`id=0.754519759158 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CalcOpticalFlowBM>`__




.. cfunction:: void cvCalcOpticalFlowBM(  const CvArr* prev, const CvArr* curr, CvSize blockSize, CvSize shiftSize, CvSize max_range, int usePrevious, CvArr* velx, CvArr* vely )

    Calculates the optical flow for two images by using the block matching method.





    
    :param prev: First image, 8-bit, single-channel 
    
    
    :param curr: Second image, 8-bit, single-channel 
    
    
    :param blockSize: Size of basic blocks that are compared 
    
    
    :param shiftSize: Block coordinate increments 
    
    
    :param max_range: Size of the scanned neighborhood in pixels around the block 
    
    
    :param usePrevious: Uses the previous (input) velocity field 
    
    
    :param velx: Horizontal component of the optical flow of  
        
        .. math::
        
            \left \lfloor   \frac{\texttt{prev->width} - \texttt{blockSize.width}}{\texttt{shiftSize.width}}   \right \rfloor \times \left \lfloor   \frac{\texttt{prev->height} - \texttt{blockSize.height}}{\texttt{shiftSize.height}}   \right \rfloor 
        
        size, 32-bit floating-point, single-channel 
    
    
    :param vely: Vertical component of the optical flow of the same size  ``velx`` , 32-bit floating-point, single-channel 
    
    
    
The function calculates the optical
flow for overlapped blocks 
:math:`\texttt{blockSize.width} \times \texttt{blockSize.height}`
pixels each, thus the velocity
fields are smaller than the original images. For every block in 
``prev``
the functions tries to find a similar block in
``curr``
in some neighborhood of the original block or shifted by (velx(x0,y0),vely(x0,y0)) block as has been calculated by previous
function call (if 
``usePrevious=1``
)


.. index:: CalcOpticalFlowHS

.. _CalcOpticalFlowHS:

CalcOpticalFlowHS
-----------------

`id=0.152735471909 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CalcOpticalFlowHS>`__




.. cfunction:: void cvCalcOpticalFlowHS(  const CvArr* prev, const CvArr* curr, int usePrevious, CvArr* velx, CvArr* vely, double lambda, CvTermCriteria criteria )

    Calculates the optical flow for two images.





    
    :param prev: First image, 8-bit, single-channel 
    
    
    :param curr: Second image, 8-bit, single-channel 
    
    
    :param usePrevious: Uses the previous (input) velocity field 
    
    
    :param velx: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    
    :param vely: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    
    :param lambda: Lagrangian multiplier 
    
    
    :param criteria: Criteria of termination of velocity computing 
    
    
    
The function computes the flow for every pixel of the first input image using the Horn and Schunck algorithm
Horn81
.


.. index:: CalcOpticalFlowLK

.. _CalcOpticalFlowLK:

CalcOpticalFlowLK
-----------------

`id=0.853253276574 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CalcOpticalFlowLK>`__




.. cfunction:: void cvCalcOpticalFlowLK(  const CvArr* prev, const CvArr* curr, CvSize winSize, CvArr* velx, CvArr* vely )

    Calculates the optical flow for two images.





    
    :param prev: First image, 8-bit, single-channel 
    
    
    :param curr: Second image, 8-bit, single-channel 
    
    
    :param winSize: Size of the averaging window used for grouping pixels 
    
    
    :param velx: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    
    :param vely: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    
    
The function computes the flow for every pixel of the first input image using the Lucas and Kanade algorithm
Lucas81
.


.. index:: CalcOpticalFlowPyrLK

.. _CalcOpticalFlowPyrLK:

CalcOpticalFlowPyrLK
--------------------

`id=0.47107753089 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CalcOpticalFlowPyrLK>`__




.. cfunction:: void cvCalcOpticalFlowPyrLK(  const CvArr* prev, const CvArr* curr, CvArr* prevPyr, CvArr* currPyr, const CvPoint2D32f* prevFeatures, CvPoint2D32f* currFeatures, int count, CvSize winSize, int level, char* status, float* track_error, CvTermCriteria criteria, int flags )

    Calculates the optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.





    
    :param prev: First frame, at time  ``t`` 
    
    
    :param curr: Second frame, at time  ``t + dt``   
    
    
    :param prevPyr: Buffer for the pyramid for the first frame. If the pointer is not  ``NULL``  , the buffer must have a sufficient size to store the pyramid from level  ``1``  to level  ``level``  ; the total size of  ``(image_width+8)*image_height/3``  bytes is sufficient 
    
    
    :param currPyr: Similar to  ``prevPyr`` , used for the second frame 
    
    
    :param prevFeatures: Array of points for which the flow needs to be found 
    
    
    :param currFeatures: Array of 2D points containing the calculated new positions of the input features in the second image 
    
    
    :param count: Number of feature points 
    
    
    :param winSize: Size of the search window of each pyramid level 
    
    
    :param level: Maximal pyramid level number. If  ``0``  , pyramids are not used (single level), if  ``1``  , two levels are used, etc 
    
    
    :param status: Array. Every element of the array is set to  ``1``  if the flow for the corresponding feature has been found,  ``0``  otherwise 
    
    
    :param track_error: Array of double numbers containing the difference between patches around the original and moved points. Optional parameter; can be  ``NULL`` 
    
    
    :param criteria: Specifies when the iteration process of finding the flow for each point on each pyramid level should be stopped 
    
    
    :param flags: Miscellaneous flags: 
        
                
            * **CV_LKFLOWPyr_A_READY** pyramid for the first frame is precalculated before the call 
            
               
            * **CV_LKFLOWPyr_B_READY**  pyramid for the second frame is precalculated before the call 
            
               
            * **CV_LKFLOW_INITIAL_GUESSES** array B contains initial coordinates of features before the function call 
            
            
    
    
    
The function implements the sparse iterative version of the Lucas-Kanade optical flow in pyramids
Bouguet00
. It calculates the coordinates of the feature points on the current video
frame given their coordinates on the previous frame. The function finds
the coordinates with sub-pixel accuracy.

Both parameters 
``prevPyr``
and 
``currPyr``
comply with the
following rules: if the image pointer is 0, the function allocates the
buffer internally, calculates the pyramid, and releases the buffer after
processing. Otherwise, the function calculates the pyramid and stores
it in the buffer unless the flag 
``CV_LKFLOWPyr_A[B]_READY``
is set. The image should be large enough to fit the Gaussian pyramid
data. After the function call both pyramids are calculated and the
readiness flag for the corresponding image can be set in the next call
(i.e., typically, for all the image pairs except the very first one
``CV_LKFLOWPyr_A_READY``
is set).



.. index:: CamShift

.. _CamShift:

CamShift
--------

`id=0.583105572641 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CamShift>`__




.. cfunction:: int cvCamShift(  const CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp, CvBox2D* box=NULL )

    Finds the object center, size, and orientation.





    
    :param prob_image: Back projection of object histogram (see  :ref:`CalcBackProject` ) 
    
    
    :param window: Initial search window 
    
    
    :param criteria: Criteria applied to determine when the window search should be finished 
    
    
    :param comp: Resultant structure that contains the converged search window coordinates ( ``comp->rect``  field) and the sum of all of the pixels inside the window ( ``comp->area``  field) 
    
    
    :param box: Circumscribed box for the object. If not  ``NULL`` , it contains object size and orientation 
    
    
    
The function implements the CAMSHIFT object tracking algrorithm
Bradski98
.
First, it finds an object center using 
:ref:`MeanShift`
and, after that, calculates the object size and orientation. The function returns number of iterations made within 
:ref:`MeanShift`
.

The 
``CamShiftTracker``
class declared in cv.hpp implements the color object tracker that uses the function.


CvConDensation
--------------


ConDenstation state.




::


    
        typedef struct CvConDensation
        {
            int MP;     //Dimension of measurement vector
            int DP;     // Dimension of state vector
            float* DynamMatr;       // Matrix of the linear Dynamics system
            float* State;           // Vector of State
            int SamplesNum;         // Number of the Samples
            float** flSamples;      // array of the Sample Vectors
            float** flNewSamples;   // temporary array of the Sample Vectors
            float* flConfidence;    // Confidence for each Sample
            float* flCumulative;    // Cumulative confidence
            float* Temp;            // Temporary vector
            float* RandomSample;    // RandomVector to update sample set
            CvRandState* RandS;     // Array of structures to generate random vectors
        } CvConDensation;
    
    

..

The structure 
``CvConDensation``
stores the CONditional DENSity propagATION tracker state. The information about the algorithm can be found at 
http://www.dai.ed.ac.uk/CVonline/LOCAL\_COPIES/ISARD1/condensation.html
.


.. index:: CreateConDensation

.. _CreateConDensation:

CreateConDensation
------------------

`id=0.31878352255 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CreateConDensation>`__




.. cfunction:: CvConDensation* cvCreateConDensation(  int dynam_params, int measure_params, int sample_count )

    Allocates the ConDensation filter structure.





    
    :param dynam_params: Dimension of the state vector 
    
    
    :param measure_params: Dimension of the measurement vector 
    
    
    :param sample_count: Number of samples 
    
    
    
The function creates a 
``CvConDensation``
structure and returns a pointer to the structure.


.. index:: ConDensInitSampleSet

.. _ConDensInitSampleSet:

ConDensInitSampleSet
--------------------

`id=0.386398764636 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/ConDensInitSampleSet>`__




.. cfunction:: void cvConDensInitSampleSet( CvConDensation* condens,  CvMat* lower_bound,  CvMat* upper_bound )

    Initializes the sample set for the ConDensation algorithm.





    
    :param condens: Pointer to a structure to be initialized 
    
    
    :param lower_bound: Vector of the lower boundary for each dimension 
    
    
    :param upper_bound: Vector of the upper boundary for each dimension 
    
    
    
The function fills the samples arrays in the structure 
``condens``
with values within the specified ranges.

.. index:: CvKalman

.. _CvKalman:

CvKalman
--------

`id=0.625509453461 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CvKalman>`__

.. ctype:: CvKalman



Kalman filter state.




::


    
    typedef struct CvKalman
    {
        int MP;                     /* number of measurement vector dimensions */
        int DP;                     /* number of state vector dimensions */
        int CP;                     /* number of control vector dimensions */
    
        /* backward compatibility fields */
    #if 1
        float* PosterState;         /* =state_pre->data.fl */
        float* PriorState;          /* =state_post->data.fl */
        float* DynamMatr;           /* =transition_matrix->data.fl */
        float* MeasurementMatr;     /* =measurement_matrix->data.fl */
        float* MNCovariance;        /* =measurement_noise_cov->data.fl */
        float* PNCovariance;        /* =process_noise_cov->data.fl */
        float* KalmGainMatr;        /* =gain->data.fl */
        float* PriorErrorCovariance;/* =error_cov_pre->data.fl */
        float* PosterErrorCovariance;/* =error_cov_post->data.fl */
        float* Temp1;               /* temp1->data.fl */
        float* Temp2;               /* temp2->data.fl */
    #endif
    
        CvMat* state_pre;           /* predicted state (x'(k)):
                                        x(k)=A*x(k-1)+B*u(k) */
        CvMat* state_post;          /* corrected state (x(k)):
                                        x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) */
        CvMat* transition_matrix;   /* state transition matrix (A) */
        CvMat* control_matrix;      /* control matrix (B)
                                       (it is not used if there is no control)*/
        CvMat* measurement_matrix;  /* measurement matrix (H) */
        CvMat* process_noise_cov;   /* process noise covariance matrix (Q) */
        CvMat* measurement_noise_cov; /* measurement noise covariance matrix (R) */
        CvMat* error_cov_pre;       /* priori error estimate covariance matrix (P'(k)):
                                        P'(k)=A*P(k-1)*At + Q*/
        CvMat* gain;                /* Kalman gain matrix (K(k)):
                                        K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)*/
        CvMat* error_cov_post;      /* posteriori error estimate covariance matrix (P(k)):
                                        P(k)=(I-K(k)*H)*P'(k) */
        CvMat* temp1;               /* temporary matrices */
        CvMat* temp2;
        CvMat* temp3;
        CvMat* temp4;
        CvMat* temp5;
    }
    CvKalman;
    

..

The structure 
``CvKalman``
is used to keep the Kalman filter
state. It is created by the 
:ref:`CreateKalman`
function, updated
by the 
:ref:`KalmanPredict`
and 
:ref:`KalmanCorrect`
functions
and released by the 
:ref:`ReleaseKalman`
function
. Normally, the
structure is used for the standard Kalman filter (notation and the
formulas below are borrowed from the excellent Kalman tutorial
Welch95
)



.. math::

    \begin{array}{l} x_k=A  \cdot x_{k-1}+B  \cdot u_k+w_k \\ z_k=H  \cdot x_k+v_k \end{array} 


where:



.. math::

    \begin{array}{l l} x_k \; (x_{k-1})&  \text{state of the system at the moment \emph{k} (\emph{k-1})} \\ z_k &  \text{measurement of the system state at the moment \emph{k}} \\ u_k &  \text{external control applied at the moment \emph{k}} \end{array} 


:math:`w_k`
and 
:math:`v_k`
are normally-distributed process and measurement noise, respectively:



.. math::

    \begin{array}{l} p(w)  \sim N(0,Q) \\ p(v)  \sim N(0,R) \end{array} 


that is,

:math:`Q`
process noise covariance matrix, constant or variable,

:math:`R`
measurement noise covariance matrix, constant or variable

In the case of the standard Kalman filter, all of the matrices: A, B, H, Q and R are initialized once after the 
:ref:`CvKalman`
structure is allocated via 
:ref:`CreateKalman`
. However, the same structure and the same functions may be used to simulate the extended Kalman filter by linearizing the extended Kalman filter equation in the current system state neighborhood, in this case A, B, H (and, probably, Q and R) should be updated on every step.


.. index:: CreateKalman

.. _CreateKalman:

CreateKalman
------------

`id=0.495816671145 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/CreateKalman>`__




.. cfunction:: CvKalman* cvCreateKalman(  int dynam_params, int measure_params, int control_params=0 )

    Allocates the Kalman filter structure.





    
    :param dynam_params: dimensionality of the state vector 
    
    
    :param measure_params: dimensionality of the measurement vector 
    
    
    :param control_params: dimensionality of the control vector 
    
    
    
The function allocates 
:ref:`CvKalman`
and all its matrices and initializes them somehow.



.. index:: KalmanCorrect

.. _KalmanCorrect:

KalmanCorrect
-------------

`id=0.247263362016 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/KalmanCorrect>`__




.. cfunction:: const CvMat* cvKalmanCorrect( CvKalman* kalman, const CvMat* measurement )

    Adjusts the model state.





    
    :param kalman: Pointer to the structure to be updated 
    
    
    :param measurement: CvMat containing the measurement vector 
    
    
    
The function adjusts the stochastic model state on the basis of the given measurement of the model state:



.. math::

    \begin{array}{l} K_k=P'_k  \cdot H^T  \cdot (H  \cdot P'_k  \cdot H^T+R)^{-1} \\ x_k=x'_k+K_k  \cdot (z_k-H  \cdot x'_k) \\ P_k=(I-K_k  \cdot H)  \cdot P'_k \end{array} 


where


.. table::

    ===========  ===============================================
    :math:`z_k`  given measurement ( ``mesurement`` parameter) \
    ===========  ===============================================
    :math:`K_k`  Kalman "gain" matrix. \                        
    ===========  ===============================================

The function stores the adjusted state at 
``kalman->state_post``
and returns it on output.

Example. Using Kalman filter to track a rotating point



::


    
    #include "cv.h"
    #include "highgui.h"
    #include <math.h>
    
    int main(int argc, char** argv)
    {
        /* A matrix data */
        const float A[] = { 1, 1, 0, 1 };
    
        IplImage* img = cvCreateImage( cvSize(500,500), 8, 3 );
        CvKalman* kalman = cvCreateKalman( 2, 1, 0 );
        /* state is (phi, delta_phi) - angle and angle increment */
        CvMat* state = cvCreateMat( 2, 1, CV_32FC1 );
        CvMat* process_noise = cvCreateMat( 2, 1, CV_32FC1 );
        /* only phi (angle) is measured */
        CvMat* measurement = cvCreateMat( 1, 1, CV_32FC1 );
        CvRandState rng;
        int code = -1;
    
        cvRandInit( &rng, 0, 1, -1, CV_RAND_UNI );
    
        cvZero( measurement );
        cvNamedWindow( "Kalman", 1 );
    
        for(;;)
        {
            cvRandSetRange( &rng, 0, 0.1, 0 );
            rng.disttype = CV_RAND_NORMAL;
    
            cvRand( &rng, state );
    
            memcpy( kalman->transition_matrix->data.fl, A, sizeof(A));
            cvSetIdentity( kalman->measurement_matrix, cvRealScalar(1) );
            cvSetIdentity( kalman->process_noise_cov, cvRealScalar(1e-5) );
            cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(1e-1) );
            cvSetIdentity( kalman->error_cov_post, cvRealScalar(1));
            /* choose random initial state */
            cvRand( &rng, kalman->state_post );
    
            rng.disttype = CV_RAND_NORMAL;
    
            for(;;)
            {
                #define calc_point(angle)                                      \
                    cvPoint( cvRound(img->width/2 + img->width/3*cos(angle)),  \
                             cvRound(img->height/2 - img->width/3*sin(angle)))
    
                float state_angle = state->data.fl[0];
                CvPoint state_pt = calc_point(state_angle);
    
                /* predict point position */
                const CvMat* prediction = cvKalmanPredict( kalman, 0 );
                float predict_angle = prediction->data.fl[0];
                CvPoint predict_pt = calc_point(predict_angle);
                float measurement_angle;
                CvPoint measurement_pt;
    
                cvRandSetRange( &rng,
                                0,
                                sqrt(kalman->measurement_noise_cov->data.fl[0]),
                                0 );
                cvRand( &rng, measurement );
    
                /* generate measurement */
                cvMatMulAdd( kalman->measurement_matrix, state, measurement, measurement );
    
                measurement_angle = measurement->data.fl[0];
                measurement_pt = calc_point(measurement_angle);
    
                /* plot points */
                #define draw_cross( center, color, d )                        \
                    cvLine( img, cvPoint( center.x - d, center.y - d ),       \
                                 cvPoint( center.x + d, center.y + d ),       \
                                 color, 1, 0 );                               \
                    cvLine( img, cvPoint( center.x + d, center.y - d ),       \
                                 cvPoint( center.x - d, center.y + d ),       \
                                 color, 1, 0 )
    
                cvZero( img );
                draw_cross( state_pt, CV_RGB(255,255,255), 3 );
                draw_cross( measurement_pt, CV_RGB(255,0,0), 3 );
                draw_cross( predict_pt, CV_RGB(0,255,0), 3 );
                cvLine( img, state_pt, predict_pt, CV_RGB(255,255,0), 3, 0 );
    
                /* adjust Kalman filter state */
                cvKalmanCorrect( kalman, measurement );
    
                cvRandSetRange( &rng,
                                0,
                                sqrt(kalman->process_noise_cov->data.fl[0]),
                                0 );
                cvRand( &rng, process_noise );
                cvMatMulAdd( kalman->transition_matrix,
                             state,
                             process_noise,
                             state );
    
                cvShowImage( "Kalman", img );
                code = cvWaitKey( 100 );
    
                if( code > 0 ) /* break current simulation by pressing a key */
                    break;
            }
            if( code == 27 ) /* exit by ESCAPE */
                break;
        }
    
        return 0;
    }
    

..


.. index:: KalmanPredict

.. _KalmanPredict:

KalmanPredict
-------------

`id=0.406145730558 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/KalmanPredict>`__




.. cfunction:: const CvMat* cvKalmanPredict(  CvKalman* kalman,  const CvMat* control=NULL)

    Estimates the subsequent model state.





    
    :param kalman: Kalman filter state 
    
    
    :param control: Control vector  :math:`u_k` , should be NULL iff there is no external control ( ``control_params``  =0) 
    
    
    
The function estimates the subsequent stochastic model state by its current state and stores it at 
``kalman->state_pre``
:



.. math::

    \begin{array}{l} x'_k=A x_{k-1} + B u_k \\ P'_k=A P_{k-1} A^T + Q \end{array} 


where


.. table::

    ===============  ====================================================================================================================================================================
    :math:`x'_k`     is predicted state  ``kalman->state_pre`` , \                                                                                                                       
    ===============  ====================================================================================================================================================================
    :math:`x_{k-1}`  is corrected state on the previous step  ``kalman->state_post`` (should be initialized somehow in the beginning, zero vector by default), \                         
    :math:`u_k`      is external control ( ``control`` parameter), \                                                                                                                     
    :math:`P'_k`     is priori error covariance matrix  ``kalman->error_cov_pre`` \                                                                                                      
    :math:`P_{k-1}`  is posteriori error covariance matrix on the previous step  ``kalman->error_cov_post`` (should be initialized somehow in the beginning, identity matrix by default),
    ===============  ====================================================================================================================================================================

The function returns the estimated state.


KalmanUpdateByMeasurement
-------------------------


Synonym for 
:ref:`KalmanCorrect`

KalmanUpdateByTime
------------------


Synonym for 
:ref:`KalmanPredict`

.. index:: MeanShift

.. _MeanShift:

MeanShift
---------

`id=0.377464124859 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/MeanShift>`__




.. cfunction:: int cvMeanShift(  const CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp )

    Finds the object center on back projection.





    
    :param prob_image: Back projection of the object histogram (see  :ref:`CalcBackProject` ) 
    
    
    :param window: Initial search window 
    
    
    :param criteria: Criteria applied to determine when the window search should be finished 
    
    
    :param comp: Resultant structure that contains the converged search window coordinates ( ``comp->rect``  field) and the sum of all of the pixels inside the window ( ``comp->area``  field) 
    
    
    
The function iterates to find the object center
given its back projection and initial position of search window. The
iterations are made until the search window center moves by less than
the given value and/or until the function has done the maximum number
of iterations. The function returns the number of iterations made.


.. index:: ReleaseConDensation

.. _ReleaseConDensation:

ReleaseConDensation
-------------------

`id=0.860558456819 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/ReleaseConDensation>`__




.. cfunction:: void cvReleaseConDensation( CvConDensation** condens )

    Deallocates the ConDensation filter structure.





    
    :param condens: Pointer to the pointer to the structure to be released 
    
    
    
The function releases the structure 
``condens``
) and frees all memory previously allocated for the structure.


.. index:: ReleaseKalman

.. _ReleaseKalman:

ReleaseKalman
-------------

`id=0.202454950979 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/ReleaseKalman>`__




.. cfunction:: void cvReleaseKalman(  CvKalman** kalman )

    Deallocates the Kalman filter structure.





    
    :param kalman: double pointer to the Kalman filter structure 
    
    
    
The function releases the structure 
:ref:`CvKalman`
and all of the underlying matrices.


.. index:: SegmentMotion

.. _SegmentMotion:

SegmentMotion
-------------

`id=0.604815881374 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/SegmentMotion>`__




.. cfunction:: CvSeq* cvSegmentMotion(  const CvArr* mhi, CvArr* seg_mask, CvMemStorage* storage, double timestamp, double seg_thresh )

    Segments a whole motion into separate moving parts.





    
    :param mhi: Motion history image 
    
    
    :param seg_mask: Image where the mask found should be stored, single-channel, 32-bit floating-point 
    
    
    :param storage: Memory storage that will contain a sequence of motion connected components 
    
    
    :param timestamp: Current time in milliseconds or other units 
    
    
    :param seg_thresh: Segmentation threshold; recommended to be equal to the interval between motion history "steps" or greater 
    
    
    
The function finds all of the motion segments and
marks them in 
``seg_mask``
with individual values (1,2,...). It
also returns a sequence of 
:ref:`CvConnectedComp`
structures, one for each motion component. After that the
motion direction for every component can be calculated with
:ref:`CalcGlobalOrientation`
using the extracted mask of the particular
component 
:ref:`Cmp`
.


.. index:: SnakeImage

.. _SnakeImage:

SnakeImage
----------

`id=0.376286588765 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/SnakeImage>`__




.. cfunction:: void cvSnakeImage(  const IplImage* image, CvPoint* points, int length, float* alpha, float* beta, float* gamma, int coeff_usage, CvSize win, CvTermCriteria criteria, int calc_gradient=1 )

    Changes the contour position to minimize its energy.





    
    :param image: The source image or external energy field 
    
    
    :param points: Contour points (snake) 
    
    
    :param length: Number of points in the contour 
    
    
    :param alpha: Weight[s] of continuity energy, single float or
        array of  ``length``  floats, one for each contour point 
    
    
    :param beta: Weight[s] of curvature energy, similar to  ``alpha`` 
    
    
    :param gamma: Weight[s] of image energy, similar to  ``alpha`` 
    
    
    :param coeff_usage: Different uses of the previous three parameters: 
        
                
            * **CV_VALUE** indicates that each of  ``alpha, beta, gamma``  is a pointer to a single value to be used for all points; 
            
               
            * **CV_ARRAY** indicates that each of  ``alpha, beta, gamma``  is a pointer to an array of coefficients different for all the points of the snake. All the arrays must have the size equal to the contour size. 
            
            
    
    
    :param win: Size of neighborhood of every point used to search the minimum, both  ``win.width``  and  ``win.height``  must be odd 
    
    
    :param criteria: Termination criteria 
    
    
    :param calc_gradient: Gradient flag; if not 0, the function calculates the gradient magnitude for every image pixel and consideres it as the energy field, otherwise the input image itself is considered 
    
    
    
The function updates the snake in order to minimize its
total energy that is a sum of internal energy that depends on the contour
shape (the smoother contour is, the smaller internal energy is) and
external energy that depends on the energy field and reaches minimum at
the local energy extremums that correspond to the image edges in the case
of using an image gradient.

The parameter 
``criteria.epsilon``
is used to define the minimal
number of points that must be moved during any iteration to keep the
iteration process running.

If at some iteration the number of moved points is less
than 
``criteria.epsilon``
or the function performed
``criteria.max_iter``
iterations, the function terminates.


.. index:: UpdateMotionHistory

.. _UpdateMotionHistory:

UpdateMotionHistory
-------------------

`id=0.131540988983 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/video/UpdateMotionHistory>`__




.. cfunction:: void cvUpdateMotionHistory(  const CvArr* silhouette, CvArr* mhi, double timestamp, double duration )

    Updates the motion history image by a moving silhouette.





    
    :param silhouette: Silhouette mask that has non-zero pixels where the motion occurs 
    
    
    :param mhi: Motion history image, that is updated by the function (single-channel, 32-bit floating-point) 
    
    
    :param timestamp: Current time in milliseconds or other units 
    
    
    :param duration: Maximal duration of the motion track in the same units as  ``timestamp`` 
    
    
    
The function updates the motion history image as following:



.. math::

    \texttt{mhi} (x,y)= \forkthree{\texttt{timestamp}}{if $\texttt{silhouette}(x,y) \ne 0$}{0}{if $\texttt{silhouette}(x,y) = 0$ and $\texttt{mhi} < (\texttt{timestamp} - \texttt{duration})$}{\texttt{mhi}(x,y)}{otherwise} 


That is, MHI pixels where motion occurs are set to the current timestamp, while the pixels where motion happened far ago are cleared.

