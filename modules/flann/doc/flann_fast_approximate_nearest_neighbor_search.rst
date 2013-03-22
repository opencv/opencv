Fast Approximate Nearest Neighbor Search
========================================

.. highlight:: cpp


This section documents OpenCV's interface to the FLANN library. FLANN (Fast Library for Approximate Nearest Neighbors) is a library that contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features. More information about FLANN can be found in [Muja2009]_ .

.. [Muja2009] Marius Muja, David G. Lowe. Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration, 2009

flann::Index\_
-----------------

.. ocv:class:: flann::Index_

The FLANN nearest neighbor index class. This class is templated with the type of elements for which the index is built.


flann::Index_<T>::Index\_
--------------------------
Constructs a nearest neighbor search index for a given dataset.

.. ocv:function:: flann::Index_<T>::Index_(const Mat& features, const IndexParams& params)

    :param features:  Matrix of containing the features(points) to index. The size of the matrix is ``num_features x feature_dimensionality`` and the data type of the elements in the matrix must coincide with the type of the index.

    :param params: Structure containing the index parameters. The type of index that will be constructed depends on the type of this parameter. See the description.

The method constructs a fast search structure from a set of features using the specified algorithm with specified parameters, as defined by ``params``. ``params`` is a reference to one of the following class ``IndexParams`` descendants:

    *

       **LinearIndexParams** When passing an object of this type, the index will perform a linear, brute-force search. ::

            struct LinearIndexParams : public IndexParams
            {
            };

       ..

    *

       **KDTreeIndexParams** When passing an object of this type the index constructed will consist of a set of randomized kd-trees which will be searched in parallel. ::

            struct KDTreeIndexParams : public IndexParams
            {
                KDTreeIndexParams( int trees = 4 );
            };

       ..

            * **trees** The number of parallel kd-trees to use. Good values are in the range [1..16]

    *

       **KMeansIndexParams** When passing an object of this type the index constructed will be a hierarchical k-means tree. ::

            struct KMeansIndexParams : public IndexParams
            {
                KMeansIndexParams(
                    int branching = 32,
                    int iterations = 11,
                    flann_centers_init_t centers_init = CENTERS_RANDOM,
                    float cb_index = 0.2 );
            };

       ..

           * **branching**  The branching factor to use for the hierarchical k-means tree

           * **iterations**  The maximum number of iterations to use in the k-means clustering stage when building the k-means tree. A value of -1 used here means that the k-means clustering should be iterated until convergence

           * **centers_init** The algorithm to use for selecting the initial centers when performing a k-means clustering step. The possible values are  ``CENTERS_RANDOM``  (picks the initial cluster centers randomly),  ``CENTERS_GONZALES``  (picks the initial centers using Gonzales' algorithm) and  ``CENTERS_KMEANSPP``  (picks the initial centers using the algorithm suggested in  arthur_kmeanspp_2007 )

           * **cb_index** This parameter (cluster boundary index) influences the way exploration is performed in the hierarchical kmeans tree. When  ``cb_index``  is zero the next kmeans domain to be explored is chosen to be the one with the closest center. A value greater then zero also takes into account the size of the domain.

    *
       **CompositeIndexParams** When using a parameters object of this type the index created combines the randomized kd-trees  and the hierarchical k-means tree. ::

            struct CompositeIndexParams : public IndexParams
            {
                CompositeIndexParams(
                    int trees = 4,
                    int branching = 32,
                    int iterations = 11,
                    flann_centers_init_t centers_init = CENTERS_RANDOM,
                    float cb_index = 0.2 );
            };

    *
       **LshIndexParams** When using a parameters object of this type the index created uses multi-probe LSH (by ``Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search`` by Qin Lv, William Josephson, Zhe Wang, Moses Charikar, Kai Li., Proceedings of the 33rd International Conference on Very Large Data Bases (VLDB). Vienna, Austria. September 2007) ::

            struct LshIndexParams : public IndexParams
            {
                LshIndexParams(
                    unsigned int table_number,
                    unsigned int key_size,
                    unsigned int multi_probe_level );
            };

       ..

           * **table_number**  the number of hash tables to use (between 10 and 30 usually).


           * **key_size**  the size of the hash key in bits (between 10 and 20 usually).


           * **multi_probe_level**  the number of bits to shift to check for neighboring buckets (0 is regular LSH, 2 is recommended).

    *
       **AutotunedIndexParams** When passing an object of this type the index created is automatically tuned to offer  the best performance, by choosing the optimal index type (randomized kd-trees, hierarchical kmeans, linear) and parameters for the dataset provided. ::

            struct AutotunedIndexParams : public IndexParams
            {
                AutotunedIndexParams(
                    float target_precision = 0.9,
                    float build_weight = 0.01,
                    float memory_weight = 0,
                    float sample_fraction = 0.1 );
            };

       ..

           * **target_precision**  Is a number between 0 and 1 specifying the percentage of the approximate nearest-neighbor searches that return the exact nearest-neighbor. Using a higher value for this parameter gives more accurate results, but the search takes longer. The optimum value usually depends on the application.


           * **build_weight**  Specifies the importance of the index build time raported to the nearest-neighbor search time. In some applications it's acceptable for the index build step to take a long time if the subsequent searches in the index can be performed very fast. In other applications it's required that the index be build as fast as possible even if that leads to slightly longer search times.


           * **memory_weight** Is used to specify the tradeoff between time (index build time and search time) and memory used by the index. A value less than 1 gives more importance to the time spent and a value greater than 1 gives more importance to the memory usage.


           * **sample_fraction** Is a number between 0 and 1 indicating what fraction of the dataset to use in the automatic parameter configuration algorithm. Running the algorithm on the full dataset gives the most accurate results, but for very large datasets can take longer than desired. In such case using just a fraction of the data helps speeding up this algorithm while still giving good approximations of the optimum parameters.

    *
       **SavedIndexParams** This object type is used for loading a previously saved index from the disk. ::

            struct SavedIndexParams : public IndexParams
            {
                SavedIndexParams( String filename );
            };


       ..

          * **filename**  The filename in which the index was saved.


flann::Index_<T>::knnSearch
----------------------------
Performs a K-nearest neighbor search for a given query point using the index.

.. ocv:function:: void flann::Index_<T>::knnSearch(const vector<T>& query, vector<int>& indices, vector<float>& dists, int knn, const SearchParams& params)

.. ocv:function:: void flann::Index_<T>::knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& params)

    :param query: The query point

    :param indices: Vector that will contain the indices of the K-nearest neighbors found. It must have at least knn size.

    :param dists: Vector that will contain the distances to the K-nearest neighbors found. It must have at least knn size.

    :param knn: Number of nearest neighbors to search for.

    :param params:

                Search parameters ::

                      struct SearchParams {
                              SearchParams(int checks = 32);
                      };

                ..

                    * **checks**  The number of times the tree(s) in the index should be recursively traversed. A higher value for this parameter would give better search precision, but also take more time. If automatic configuration was used when the index was created, the number of checks required to achieve the specified precision was also computed, in which case this parameter is ignored.


flann::Index_<T>::radiusSearch
--------------------------------------
Performs a radius nearest neighbor search for a given query point.

.. ocv:function:: int flann::Index_<T>::radiusSearch(const vector<T>& query, vector<int>& indices, vector<float>& dists, float radius, const SearchParams& params)

.. ocv:function:: int flann::Index_<T>::radiusSearch(const Mat& query, Mat& indices, Mat& dists, float radius, const SearchParams& params)

    :param query: The query point

    :param indices: Vector that will contain the indices of the points found within the search radius in decreasing order of the distance to the query point. If the number of neighbors in the search radius is bigger than the size of this vector, the ones that don't fit in the vector are ignored.

    :param dists: Vector that will contain the distances to the points found within the search radius

    :param radius: The search radius

    :param params: Search parameters


flann::Index_<T>::save
------------------------------
Saves the index to a file.

.. ocv:function:: void flann::Index_<T>::save(String filename)

    :param filename: The file to save the index to


flann::Index_<T>::getIndexParameters
--------------------------------------------
Returns the index parameters.

.. ocv:function:: const IndexParams* flann::Index_<T>::getIndexParameters()

The method is useful in the case of auto-tuned indices, when the parameters are chosen during the index construction. Then, the method can be used to retrieve the actual parameter values.
