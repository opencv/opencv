Motion Analysis and Object Tracking
===================================

.. highlight:: python



.. index:: CalcGlobalOrientation

.. _CalcGlobalOrientation:

CalcGlobalOrientation
---------------------

`id=0.671861796406 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CalcGlobalOrientation>`__


.. function:: CalcGlobalOrientation(orientation,mask,mhi,timestamp,duration)-> float

    Calculates the global motion orientation of some selected region.





    
    :param orientation: Motion gradient orientation image; calculated by the function  :ref:`CalcMotionGradient` 
    
    :type orientation: :class:`CvArr`
    
    
    :param mask: Mask image. It may be a conjunction of a valid gradient mask, obtained with  :ref:`CalcMotionGradient`  and the mask of the region, whose direction needs to be calculated 
    
    :type mask: :class:`CvArr`
    
    
    :param mhi: Motion history image 
    
    :type mhi: :class:`CvArr`
    
    
    :param timestamp: Current time in milliseconds or other units, it is better to store time passed to  :ref:`UpdateMotionHistory`  before and reuse it here, because running  :ref:`UpdateMotionHistory`  and  :ref:`CalcMotionGradient`  on large images may take some time 
    
    :type timestamp: float
    
    
    :param duration: Maximal duration of motion track in milliseconds, the same as  :ref:`UpdateMotionHistory` 
    
    :type duration: float
    
    
    
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

`id=0.734160644258 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CalcMotionGradient>`__


.. function:: CalcMotionGradient(mhi,mask,orientation,delta1,delta2,apertureSize=3)-> None

    Calculates the gradient orientation of a motion history image.





    
    :param mhi: Motion history image 
    
    :type mhi: :class:`CvArr`
    
    
    :param mask: Mask image; marks pixels where the motion gradient data is correct; output parameter 
    
    :type mask: :class:`CvArr`
    
    
    :param orientation: Motion gradient orientation image; contains angles from 0 to ~360 degrees  
    
    :type orientation: :class:`CvArr`
    
    
    :param delta1: See below 
    
    :type delta1: float
    
    
    :param delta2: See below 
    
    :type delta2: float
    
    
    :param apertureSize: Aperture size of derivative operators used by the function: CV _ SCHARR, 1, 3, 5 or 7 (see  :ref:`Sobel` ) 
    
    :type apertureSize: int
    
    
    
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

`id=0.167052327583 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CalcOpticalFlowBM>`__


.. function:: CalcOpticalFlowBM(prev,curr,blockSize,shiftSize,max_range,usePrevious,velx,vely)-> None

    Calculates the optical flow for two images by using the block matching method.





    
    :param prev: First image, 8-bit, single-channel 
    
    :type prev: :class:`CvArr`
    
    
    :param curr: Second image, 8-bit, single-channel 
    
    :type curr: :class:`CvArr`
    
    
    :param blockSize: Size of basic blocks that are compared 
    
    :type blockSize: :class:`CvSize`
    
    
    :param shiftSize: Block coordinate increments 
    
    :type shiftSize: :class:`CvSize`
    
    
    :param max_range: Size of the scanned neighborhood in pixels around the block 
    
    :type max_range: :class:`CvSize`
    
    
    :param usePrevious: Uses the previous (input) velocity field 
    
    :type usePrevious: int
    
    
    :param velx: Horizontal component of the optical flow of  
        
        .. math::
        
            \left \lfloor   \frac{\texttt{prev->width} - \texttt{blockSize.width}}{\texttt{shiftSize.width}}   \right \rfloor \times \left \lfloor   \frac{\texttt{prev->height} - \texttt{blockSize.height}}{\texttt{shiftSize.height}}   \right \rfloor 
        
        size, 32-bit floating-point, single-channel 
    
    :type velx: :class:`CvArr`
    
    
    :param vely: Vertical component of the optical flow of the same size  ``velx`` , 32-bit floating-point, single-channel 
    
    :type vely: :class:`CvArr`
    
    
    
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

`id=0.932788904949 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CalcOpticalFlowHS>`__


.. function:: CalcOpticalFlowHS(prev,curr,usePrevious,velx,vely,lambda,criteria)-> None

    Calculates the optical flow for two images.





    
    :param prev: First image, 8-bit, single-channel 
    
    :type prev: :class:`CvArr`
    
    
    :param curr: Second image, 8-bit, single-channel 
    
    :type curr: :class:`CvArr`
    
    
    :param usePrevious: Uses the previous (input) velocity field 
    
    :type usePrevious: int
    
    
    :param velx: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    :type velx: :class:`CvArr`
    
    
    :param vely: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    :type vely: :class:`CvArr`
    
    
    :param lambda: Lagrangian multiplier 
    
    :type lambda: float
    
    
    :param criteria: Criteria of termination of velocity computing 
    
    :type criteria: :class:`CvTermCriteria`
    
    
    
The function computes the flow for every pixel of the first input image using the Horn and Schunck algorithm
Horn81
.


.. index:: CalcOpticalFlowLK

.. _CalcOpticalFlowLK:

CalcOpticalFlowLK
-----------------

`id=0.849649850841 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CalcOpticalFlowLK>`__


.. function:: CalcOpticalFlowLK(prev,curr,winSize,velx,vely)-> None

    Calculates the optical flow for two images.





    
    :param prev: First image, 8-bit, single-channel 
    
    :type prev: :class:`CvArr`
    
    
    :param curr: Second image, 8-bit, single-channel 
    
    :type curr: :class:`CvArr`
    
    
    :param winSize: Size of the averaging window used for grouping pixels 
    
    :type winSize: :class:`CvSize`
    
    
    :param velx: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    :type velx: :class:`CvArr`
    
    
    :param vely: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel 
    
    :type vely: :class:`CvArr`
    
    
    
The function computes the flow for every pixel of the first input image using the Lucas and Kanade algorithm
Lucas81
.


.. index:: CalcOpticalFlowPyrLK

.. _CalcOpticalFlowPyrLK:

CalcOpticalFlowPyrLK
--------------------

`id=0.333066207955 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CalcOpticalFlowPyrLK>`__


.. function:: CalcOpticalFlowPyrLK(  prev, curr, prevPyr, currPyr, prevFeatures, winSize, level, criteria, flags, guesses = None) -> (currFeatures, status, track_error)

    Calculates the optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.





    
    :param prev: First frame, at time  ``t`` 
    
    :type prev: :class:`CvArr`
    
    
    :param curr: Second frame, at time  ``t + dt``   
    
    :type curr: :class:`CvArr`
    
    
    :param prevPyr: Buffer for the pyramid for the first frame. If the pointer is not  ``NULL``  , the buffer must have a sufficient size to store the pyramid from level  ``1``  to level  ``level``  ; the total size of  ``(image_width+8)*image_height/3``  bytes is sufficient 
    
    :type prevPyr: :class:`CvArr`
    
    
    :param currPyr: Similar to  ``prevPyr`` , used for the second frame 
    
    :type currPyr: :class:`CvArr`
    
    
    :param prevFeatures: Array of points for which the flow needs to be found 
    
    :type prevFeatures: :class:`CvPoint2D32f`
    
    
    :param currFeatures: Array of 2D points containing the calculated new positions of the input features in the second image 
    
    :type currFeatures: :class:`CvPoint2D32f`
    
    
    :param winSize: Size of the search window of each pyramid level 
    
    :type winSize: :class:`CvSize`
    
    
    :param level: Maximal pyramid level number. If  ``0``  , pyramids are not used (single level), if  ``1``  , two levels are used, etc 
    
    :type level: int
    
    
    :param status: Array. Every element of the array is set to  ``1``  if the flow for the corresponding feature has been found,  ``0``  otherwise 
    
    :type status: str
    
    
    :param track_error: Array of double numbers containing the difference between patches around the original and moved points. Optional parameter; can be  ``NULL`` 
    
    :type track_error: float
    
    
    :param criteria: Specifies when the iteration process of finding the flow for each point on each pyramid level should be stopped 
    
    :type criteria: :class:`CvTermCriteria`
    
    
    :param flags: Miscellaneous flags: 
        
                
            * **CV_LKFLOWPyr_A_READY** pyramid for the first frame is precalculated before the call 
            
               
            * **CV_LKFLOWPyr_B_READY**  pyramid for the second frame is precalculated before the call 
            
               
            
    
    :type flags: int
    
    
    :param guesses: optional array of estimated coordinates of features in second frame, with same length as  ``prevFeatures`` 
    
    :type guesses: :class:`CvPoint2D32f`
    
    
    
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

`id=0.228709757227 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CamShift>`__


.. function:: CamShift(prob_image,window,criteria)-> (int, comp, box)

    Finds the object center, size, and orientation.





    
    :param prob_image: Back projection of object histogram (see  :ref:`CalcBackProject` ) 
    
    :type prob_image: :class:`CvArr`
    
    
    :param window: Initial search window 
    
    :type window: :class:`CvRect`
    
    
    :param criteria: Criteria applied to determine when the window search should be finished 
    
    :type criteria: :class:`CvTermCriteria`
    
    
    :param comp: Resultant structure that contains the converged search window coordinates ( ``comp->rect``  field) and the sum of all of the pixels inside the window ( ``comp->area``  field) 
    
    :type comp: :class:`CvConnectedComp`
    
    
    :param box: Circumscribed box for the object. 
    
    :type box: :class:`CvBox2D`
    
    
    
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


.. index:: CvKalman

.. _CvKalman:

CvKalman
--------

`id=0.911390647458 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CvKalman>`__

.. class:: CvKalman



Kalman filter state.



    
    
    .. attribute:: MP
    
    
    
        number of measurement vector dimensions 
    
    
    
    .. attribute:: DP
    
    
    
        number of state vector dimensions 
    
    
    
    .. attribute:: CP
    
    
    
        number of control vector dimensions 
    
    
    
    .. attribute:: state_pre
    
    
    
        predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k) 
    
    
    
    .. attribute:: state_post
    
    
    
        corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) 
    
    
    
    .. attribute:: transition_matrix
    
    
    
        state transition matrix (A) 
    
    
    
    .. attribute:: control_matrix
    
    
    
        control matrix (B) (it is not used if there is no control) 
    
    
    
    .. attribute:: measurement_matrix
    
    
    
        measurement matrix (H) 
    
    
    
    .. attribute:: process_noise_cov
    
    
    
        process noise covariance matrix (Q) 
    
    
    
    .. attribute:: measurement_noise_cov
    
    
    
        measurement noise covariance matrix (R) 
    
    
    
    .. attribute:: error_cov_pre
    
    
    
        priori error estimate covariance matrix (P'(k)):  P'(k)=A*P(k-1)*At + Q 
    
    
    
    .. attribute:: gain
    
    
    
        Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R) 
    
    
    
    .. attribute:: error_cov_post
    
    
    
        posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k) 
    
    
    
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

`id=0.636220879554 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/CreateKalman>`__


.. function:: CreateKalman(dynam_params, measure_params, control_params=0) -> CvKalman

    Allocates the Kalman filter structure.





    
    :param dynam_params: dimensionality of the state vector 
    
    :type dynam_params: int
    
    
    :param measure_params: dimensionality of the measurement vector 
    
    :type measure_params: int
    
    
    :param control_params: dimensionality of the control vector 
    
    :type control_params: int
    
    
    
The function allocates 
:ref:`CvKalman`
and all its matrices and initializes them somehow.



.. index:: KalmanCorrect

.. _KalmanCorrect:

KalmanCorrect
-------------

`id=0.175175296579 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/KalmanCorrect>`__


.. function:: KalmanCorrect(kalman, measurement) -> cvmat

    Adjusts the model state.





    
    :param kalman: Kalman filter object returned by  :ref:`CreateKalman` 
    
    :type kalman: :class:`CvKalman`
    
    
    :param measurement: CvMat containing the measurement vector 
    
    :type measurement: :class:`CvMat`
    
    
    
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


.. index:: KalmanPredict

.. _KalmanPredict:

KalmanPredict
-------------

`id=0.930945319496 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/KalmanPredict>`__


.. function:: KalmanPredict(kalman, control=None) -> cvmat

    Estimates the subsequent model state.





    
    :param kalman: Kalman filter object returned by  :ref:`CreateKalman` 
    
    :type kalman: :class:`CvKalman`
    
    
    :param control: Control vector  :math:`u_k` , should be NULL iff there is no external control ( ``control_params``  =0) 
    
    :type control: :class:`CvMat`
    
    
    
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

`id=0.555115149553 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/MeanShift>`__


.. function:: MeanShift(prob_image,window,criteria)-> comp

    Finds the object center on back projection.





    
    :param prob_image: Back projection of the object histogram (see  :ref:`CalcBackProject` ) 
    
    :type prob_image: :class:`CvArr`
    
    
    :param window: Initial search window 
    
    :type window: :class:`CvRect`
    
    
    :param criteria: Criteria applied to determine when the window search should be finished 
    
    :type criteria: :class:`CvTermCriteria`
    
    
    :param comp: Resultant structure that contains the converged search window coordinates ( ``comp->rect``  field) and the sum of all of the pixels inside the window ( ``comp->area``  field) 
    
    :type comp: :class:`CvConnectedComp`
    
    
    
The function iterates to find the object center
given its back projection and initial position of search window. The
iterations are made until the search window center moves by less than
the given value and/or until the function has done the maximum number
of iterations. The function returns the number of iterations made.


.. index:: SegmentMotion

.. _SegmentMotion:

SegmentMotion
-------------

`id=0.698315173881 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/SegmentMotion>`__


.. function:: SegmentMotion(mhi,seg_mask,storage,timestamp,seg_thresh)-> None

    Segments a whole motion into separate moving parts.





    
    :param mhi: Motion history image 
    
    :type mhi: :class:`CvArr`
    
    
    :param seg_mask: Image where the mask found should be stored, single-channel, 32-bit floating-point 
    
    :type seg_mask: :class:`CvArr`
    
    
    :param storage: Memory storage that will contain a sequence of motion connected components 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param timestamp: Current time in milliseconds or other units 
    
    :type timestamp: float
    
    
    :param seg_thresh: Segmentation threshold; recommended to be equal to the interval between motion history "steps" or greater 
    
    :type seg_thresh: float
    
    
    
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

`id=0.218492276516 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/SnakeImage>`__


.. function:: SnakeImage(image,points,alpha,beta,gamma,win,criteria,calc_gradient=1)-> new_points

    Changes the contour position to minimize its energy.





    
    :param image: The source image or external energy field 
    
    :type image: :class:`IplImage`
    
    
    :param points: Contour points (snake) 
    
    :type points: :class:`CvPoints`
    
    
    :param alpha: Weight[s] of continuity energy, single float or
        a list of floats, one for each contour point 
    
    :type alpha: sequence of float
    
    
    :param beta: Weight[s] of curvature energy, similar to  ``alpha`` 
    
    :type beta: sequence of float
    
    
    :param gamma: Weight[s] of image energy, similar to  ``alpha`` 
    
    :type gamma: sequence of float
    
    
    :param win: Size of neighborhood of every point used to search the minimum, both  ``win.width``  and  ``win.height``  must be odd 
    
    :type win: :class:`CvSize`
    
    
    :param criteria: Termination criteria 
    
    :type criteria: :class:`CvTermCriteria`
    
    
    :param calc_gradient: Gradient flag; if not 0, the function calculates the gradient magnitude for every image pixel and consideres it as the energy field, otherwise the input image itself is considered 
    
    :type calc_gradient: int
    
    
    
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

The function returns the updated list of points.

.. index:: UpdateMotionHistory

.. _UpdateMotionHistory:

UpdateMotionHistory
-------------------

`id=0.316306086975 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/video/UpdateMotionHistory>`__


.. function:: UpdateMotionHistory(silhouette,mhi,timestamp,duration)-> None

    Updates the motion history image by a moving silhouette.





    
    :param silhouette: Silhouette mask that has non-zero pixels where the motion occurs 
    
    :type silhouette: :class:`CvArr`
    
    
    :param mhi: Motion history image, that is updated by the function (single-channel, 32-bit floating-point) 
    
    :type mhi: :class:`CvArr`
    
    
    :param timestamp: Current time in milliseconds or other units 
    
    :type timestamp: float
    
    
    :param duration: Maximal duration of the motion track in the same units as  ``timestamp`` 
    
    :type duration: float
    
    
    
The function updates the motion history image as following:



.. math::

    \texttt{mhi} (x,y)= \forkthree{\texttt{timestamp}}{if $\texttt{silhouette}(x,y) \ne 0$}{0}{if $\texttt{silhouette}(x,y) = 0$ and $\texttt{mhi} < (\texttt{timestamp} - \texttt{duration})$}{\texttt{mhi}(x,y)}{otherwise} 


That is, MHI pixels where motion occurs are set to the current timestamp, while the pixels where motion happened far ago are cleared.

