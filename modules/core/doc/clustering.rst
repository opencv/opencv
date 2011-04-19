Clustering
==========

.. highlight:: cpp

.. index:: kmeans

.. _kmeans:

kmeans
------

.. c:function:: double kmeans( const Mat& samples, int clusterCount, Mat& labels, TermCriteria termcrit, int attempts,               int flags, Mat* centers )

    Finds centers of clusters and groups input samples around the clusters.

    :param samples: Floating-point matrix of input samples, one row per sample.

    :param clusterCount: Number of clusters to split the set by.

    :param labels: Input/output integer array that stores the cluster indices for every sample.

    :param termcrit: Flag to specify the maximum number of iterations and/or the desired accuracy. The accuracy is specified as ``termcrit.epsilon``. As soon as each of the cluster centers moves by less than ``termcrit.epsilon`` on some iteration, the algorithm stops.

    :param attempts: Flag to specify how many times the algorithm is executed using different initial labelings. The algorithm returns the labels that yield the best compactness (see the last function parameter).

    :param flags: Flag that can take the following values:

            * **KMEANS_RANDOM_CENTERS** Select random initial centers in each attempt.

            * **KMEANS_PP_CENTERS** Use ``kmeans++`` center initialization by Arthur and Vassilvitskii.

            * **KMEANS_USE_INITIAL_LABELS** During the first (and possibly the only) attempt, use the user-supplied labels instead of computing them from the initial centers. For the second and further attempts, use the random or semi-random centers (use one of  ``KMEANS_*_CENTERS``  flag to specify the exact method).

    :param centers: Output matrix of the cluster centers, one row per each cluster center.

The function ``kmeans`` implements a k-means algorithm that finds the
centers of ``clusterCount`` clusters and groups the input samples
around the clusters. On output,
:math:`\texttt{labels}_i` contains a 0-based cluster index for
the sample stored in the
:math:`i^{th}` row of the ``samples`` matrix.

The function returns the compactness measure, which is computed as

.. math::

    \sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2

after every attempt. The best (minimum) value is chosen and the
corresponding labels and the compactness value are returned by the function.
Basically, you can use only the core of the function, set the number of
attempts to 1, initialize labels each time using a custom algorithm, pass them with the
( ``flags`` = ``KMEANS_USE_INITIAL_LABELS`` ) flag, and then choose the best (most-compact) clustering.

.. index:: partition

partition
-------------
.. c:function:: template<typename _Tp, class _EqPredicate> int

.. c:function:: partition( const vector<_Tp>& vec, vector<int>& labels,               _EqPredicate predicate=_EqPredicate())

    Splits an element set into equivalency classes.

    :param vec: Set of elements stored as a vector.

    :param labels: Output vector of labels. It contains as many elements as  ``vec`` . Each label  ``labels[i]``  is a 0-based cluster index of  ``vec[i]`` .   
	
	:param predicate: Equivalence predicate (pointer to a boolean function of two arguments or an instance of the class that has the method  ``bool operator()(const _Tp& a, const _Tp& b)`` ). The predicate returns ``true`` when the elements are certainly in the same class, and returns ``false`` if they may or may not be in the same class.

The generic function ``partition`` implements an
:math:`O(N^2)` algorithm for
splitting a set of
:math:`N` elements into one or more equivalency classes, as described in
http://en.wikipedia.org/wiki/Disjoint-set_data_structure
. The function
returns the number of equivalency classes.

