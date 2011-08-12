Clustering
==========

.. highlight:: cpp

flann::hierarchicalClustering<ET,DT>
--------------------------------------------
Clusters features using hierarchical k-means algorithm.

.. ocv:function:: int flann::hierarchicalClustering<ET,DT>(const Mat& features, Mat& centers,                                      const KMeansIndexParams& params)

    :param features: The points to be clustered. The matrix must have elements of type ET. 
    
    :param centers: The centers of the clusters obtained. The matrix must have type DT. The number of rows in this matrix represents the number of clusters desired, however, because of the way the cut in the hierarchical tree is chosen, the number of clusters computed will be the highest number of the form  ``(branching-1)*k+1``  that's lower than the number of clusters desired, where  ``branching``  is the tree's branching factor (see description of the KMeansIndexParams). 
    
    :param params: Parameters used in the construction of the hierarchical k-means tree 
    
The method clusters the given feature vectors by constructing a hierarchical k-means tree and choosing a cut in the tree that minimizes the cluster's variance. It returns the number of clusters found.
