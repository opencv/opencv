Clustering
==========

.. highlight:: python



.. index:: KMeans2

.. _KMeans2:

KMeans2
-------




.. function:: KMeans2(samples,nclusters,labels,termcrit)-> None

    Splits set of vectors by a given number of clusters.





    
    :param samples: Floating-point matrix of input samples, one row per sample 
    
    :type samples: :class:`CvArr`
    
    
    :param nclusters: Number of clusters to split the set by 
    
    :type nclusters: int
    
    
    :param labels: Output integer vector storing cluster indices for every sample 
    
    :type labels: :class:`CvArr`
    
    
    :param termcrit: Specifies maximum number of iterations and/or accuracy (distance the centers can move by between subsequent iterations) 
    
    :type termcrit: :class:`CvTermCriteria`
    
    
    
The function 
``cvKMeans2``
implements a k-means algorithm that finds the
centers of 
``nclusters``
clusters and groups the input samples
around the clusters. On output, 
:math:`\texttt{labels}_i`
contains a cluster index for
samples stored in the i-th row of the 
``samples``
matrix.

