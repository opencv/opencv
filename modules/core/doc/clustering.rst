Clustering
==========

.. highlight:: cpp

.. index:: kmeans

kmeans
----------
.. c:function:: double kmeans( const Mat\& samples, int clusterCount, Mat\& labels,               TermCriteria termcrit, int attempts,               int flags, Mat* centers )

    Finds the centers of clusters and groups the input samples around the clusters.

    :param samples: Floating-point matrix of input samples, one row per sample

    :param clusterCount: The number of clusters to split the set by

    :param labels: The input/output integer array that will store the cluster indices for every sample

    :param termcrit: Specifies maximum number of iterations and/or accuracy (distance the centers can move by between subsequent iterations)

    :param attempts: How many times the algorithm is executed using different initial labelings. The algorithm returns the labels that yield the best compactness (see the last function parameter)

    :param flags: It can take the following values:

            * **KMEANS_RANDOM_CENTERS** Random initial centers are selected in each attempt

            * **KMEANS_PP_CENTERS** Use kmeans++ center initialization by Arthur and Vassilvitskii

            * **KMEANS_USE_INITIAL_LABELS** During the first (and possibly the only) attempt, the
                function uses the user-supplied labels instaed of computing them from the initial centers. For the second and further attempts, the function will use the random or semi-random centers (use one of  ``KMEANS_*_CENTERS``  flag to specify the exact method)

    :param centers: The output matrix of the cluster centers, one row per each cluster center

The function ``kmeans`` implements a k-means algorithm that finds the
centers of ``clusterCount`` clusters and groups the input samples
around the clusters. On output,
:math:`\texttt{labels}_i` contains a 0-based cluster index for
the sample stored in the
:math:`i^{th}` row of the ``samples`` matrix.

The function returns the compactness measure, which is computed as

.. math::

    \sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2

after every attempt; the best (minimum) value is chosen and the
corresponding labels and the compactness value are returned by the function.
Basically, the user can use only the core of the function, set the number of
attempts to 1, initialize labels each time using some custom algorithm and pass them with
( ``flags`` = ``KMEANS_USE_INITIAL_LABELS`` ) flag, and then choose the best (most-compact) clustering.

.. index:: partition

partition
-------------
.. c:function:: template<typename _Tp, class _EqPredicate> int

.. c:function:: partition( const vector<_Tp>\& vec, vector<int>\& labels,               _EqPredicate predicate=_EqPredicate())

    Splits an element set into equivalency classes.

    :param vec: The set of elements stored as a vector

    :param labels: The output vector of labels; will contain as many elements as  ``vec`` . Each label  ``labels[i]``  is 0-based cluster index of  ``vec[i]``     :param predicate: The equivalence predicate (i.e. pointer to a boolean function of two arguments or an instance of the class that has the method  ``bool operator()(const _Tp& a, const _Tp& b)`` . The predicate returns true when the elements are certainly if the same class, and false if they may or may not be in the same class

The generic function ``partition`` implements an
:math:`O(N^2)` algorithm for
splitting a set of
:math:`N` elements into one or more equivalency classes, as described in
http://en.wikipedia.org/wiki/Disjoint-set_data_structure
. The function
returns the number of equivalency classes.

