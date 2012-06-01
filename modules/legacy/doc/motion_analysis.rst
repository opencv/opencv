Motion Analysis
===============

.. highlight:: cpp


CalcOpticalFlowBM
-----------------
Calculates the optical flow for two images by using the block matching method.

.. ocv:cfunction:: void cvCalcOpticalFlowBM( const CvArr* prev, const CvArr* curr, CvSize block_size, CvSize shift_size, CvSize max_range, int use_previous, CvArr* velx, CvArr* vely )

.. ocv:pyoldfunction:: cv.CalcOpticalFlowBM(prev, curr, blockSize, shiftSize, max_range, usePrevious, velx, vely)-> None

        :param prev: First image, 8-bit, single-channel

        :param curr: Second image, 8-bit, single-channel

        :param block_size: Size of basic blocks that are compared

        :param shift_size: Block coordinate increments

        :param max_range: Size of the scanned neighborhood in pixels around the block

        :param use_previous: Flag that specifies whether to use the input velocity as initial approximations or not.

        :param velx: Horizontal component of the optical flow of

            .. math::

                \left \lfloor   \frac{\texttt{prev->width} - \texttt{block\_size.width}}{\texttt{shift\_size.width}}   \right \rfloor \times \left \lfloor   \frac{\texttt{prev->height} - \texttt{block\_size.height}}{\texttt{shift\_size.height}}   \right \rfloor

            size, 32-bit floating-point, single-channel

        :param vely: Vertical component of the optical flow of the same size  ``velx`` , 32-bit floating-point, single-channel


The function calculates the optical flow for overlapped blocks ``block_size.width x block_size.height`` pixels each, thus the velocity fields are smaller than the original images. For every block in  ``prev``
the functions tries to find a similar block in ``curr`` in some neighborhood of the original block or shifted by ``(velx(x0,y0), vely(x0,y0))`` block as has been calculated by previous function call (if ``use_previous=1``)


CalcOpticalFlowHS
-----------------
Calculates the optical flow for two images using Horn-Schunck algorithm.

.. ocv:cfunction:: void cvCalcOpticalFlowHS(const CvArr* prev, const CvArr* curr, int use_previous, CvArr* velx, CvArr* vely, double lambda, CvTermCriteria criteria)

.. ocv:pyoldfunction:: cv.CalcOpticalFlowHS(prev, curr, usePrevious, velx, vely, lambda, criteria)-> None

    :param prev: First image, 8-bit, single-channel

    :param curr: Second image, 8-bit, single-channel

    :param use_previous: Flag that specifies whether to use the input velocity as initial approximations or not.

    :param velx: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param vely: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param lambda: Smoothness weight. The larger it is, the smoother optical flow map you get.

    :param criteria: Criteria of termination of velocity computing

The function computes the flow for every pixel of the first input image using the Horn and Schunck algorithm [Horn81]_. The function is obsolete. To track sparse features, use :ocv:func:`calcOpticalFlowPyrLK`. To track all the pixels, use :ocv:func:`calcOpticalFlowFarneback`.


CalcOpticalFlowLK
-----------------

Calculates the optical flow for two images using Lucas-Kanade algorithm.

.. ocv:cfunction:: void cvCalcOpticalFlowLK( const CvArr* prev, const CvArr* curr, CvSize win_size, CvArr* velx, CvArr* vely )

.. ocv:pyoldfunction:: cv.CalcOpticalFlowLK(prev, curr, winSize, velx, vely)-> None

    :param prev: First image, 8-bit, single-channel

    :param curr: Second image, 8-bit, single-channel

    :param win_size: Size of the averaging window used for grouping pixels

    :param velx: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param vely: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

The function computes the flow for every pixel of the first input image using the Lucas and Kanade algorithm [Lucas81]_. The function is obsolete. To track sparse features, use :ocv:func:`calcOpticalFlowPyrLK`. To track all the pixels, use :ocv:func:`calcOpticalFlowFarneback`.


