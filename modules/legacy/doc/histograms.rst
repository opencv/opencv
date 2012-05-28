Histograms
==========

.. highlight:: cpp



CalcPGH
-------
Calculates a pair-wise geometrical histogram for a contour.

.. ocv:cfunction:: void cvCalcPGH( const CvSeq* contour, CvHistogram* hist )

    :param contour: Input contour. Currently, only integer point coordinates are allowed.

    :param hist: Calculated histogram. It must be two-dimensional.

The function calculates a 2D pair-wise geometrical histogram (PGH), described in [Iivarinen97]_ for the contour. The algorithm considers every pair of contour
edges. The angle between the edges and the minimum/maximum distances
are determined for every pair. To do this, each of the edges in turn
is taken as the base, while the function loops through all the other
edges. When the base edge and any other edge are considered, the minimum
and maximum distances from the points on the non-base edge and line of
the base edge are selected. The angle between the edges defines the row
of the histogram in which all the bins that correspond to the distance
between the calculated minimum and maximum distances are incremented
(that is, the histogram is transposed relatively to the definition in the original paper). The histogram can be used for contour matching.


.. [Iivarinen97] Jukka Iivarinen, Markus Peura, Jaakko Srel, and Ari Visa. *Comparison of Combined Shape Descriptors for Irregular Objects*, 8th British Machine Vision Conference, BMVC'97. http://www.cis.hut.fi/research/IA/paper/publications/bmvc97/bmvc97.html


QueryHistValue*D
----------------
Queries the value of the histogram bin.

.. ocv:cfunction:: float cvQueryHistValue_1D(CvHistogram hist, int idx0)
.. ocv:cfunction:: float cvQueryHistValue_2D(CvHistogram hist, int idx0, int idx1)
.. ocv:cfunction:: float cvQueryHistValue_3D(CvHistogram hist, int idx0, int idx1, int idx2)
.. ocv:cfunction:: float cvQueryHistValue_nD(CvHistogram hist, const int* idx)

.. ocv:pyoldfunction:: cv.QueryHistValue_1D(hist, idx0) -> float
.. ocv:pyoldfunction:: cv.QueryHistValue_2D(hist, idx0, idx1) -> float
.. ocv:pyoldfunction:: cv.QueryHistValue_3D(hist, idx0, idx1, idx2) -> float
.. ocv:pyoldfunction:: cv.QueryHistValue_nD(hist, idx) -> float

    :param hist: Histogram.

    :param idx0: 0-th index.

    :param idx1: 1-st index.

    :param idx2: 2-nd index.

    :param idx: Array of indices.

The macros return the value of the specified bin of the 1D, 2D, 3D, or N-D histogram. In case of a sparse histogram, the function returns 0. If the bin is not present in the histogram, no new bin is created.

GetHistValue\_?D
----------------
Returns a pointer to the histogram bin.

.. ocv:cfunction:: float cvGetHistValue_1D(CvHistogram hist, int idx0)

.. ocv:cfunction:: float cvGetHistValue_2D(CvHistogram hist, int idx0, int idx1)

.. ocv:cfunction:: float cvGetHistValue_3D(CvHistogram hist, int idx0, int idx1, int idx2)

.. ocv:cfunction:: float cvGetHistValue_nD(CvHistogram hist, int idx)

    :param hist: Histogram.

    :param idx0: 0-th index.

    :param idx1: 1-st index.

    :param idx2: 2-nd index.

    :param idx: Array of indices.

::

    #define cvGetHistValue_1D( hist, idx0 )
        ((float*)(cvPtr1D( (hist)->bins, (idx0), 0 ))
    #define cvGetHistValue_2D( hist, idx0, idx1 )
        ((float*)(cvPtr2D( (hist)->bins, (idx0), (idx1), 0 )))
    #define cvGetHistValue_3D( hist, idx0, idx1, idx2 )
        ((float*)(cvPtr3D( (hist)->bins, (idx0), (idx1), (idx2), 0 )))
    #define cvGetHistValue_nD( hist, idx )
        ((float*)(cvPtrND( (hist)->bins, (idx), 0 )))

..

The macros ``GetHistValue`` return a pointer to the specified bin of the 1D, 2D, 3D, or N-D histogram. In case of a sparse histogram, the function creates a new bin and sets it to 0, unless it exists already.

