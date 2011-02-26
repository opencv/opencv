Object Categorization
=====================

.. highlight:: cpp

Some approaches based on local 2D features and used to object categorization
are described in this section.

.. index:: BOWTrainer

.. _BOWTrainer:

BOWTrainer
----------
.. ctype:: BOWTrainer

Abstract base class for training ''bag of visual words'' vocabulary from a set of descriptors.
See e.g. ''Visual Categorization with Bags of Keypoints'' of Gabriella Csurka, Christopher R. Dance,
Lixin Fan, Jutta Willamowski, Cedric Bray, 2004. ::

    class BOWTrainer
    {
    public:
        BOWTrainer(){}
        virtual ~BOWTrainer(){}

        void add( const Mat& descriptors );
        const vector<Mat>& getDescriptors() const;
        int descripotorsCount() const;

        virtual void clear();

        virtual Mat cluster() const = 0;
        virtual Mat cluster( const Mat& descriptors ) const = 0;

    protected:
        ...
    };
..

.. index:: BOWTrainer::add

cv::BOWTrainer::add
------------------- ````
.. cfunction:: void BOWTrainer::add( const Mat\& descriptors )

    Add descriptors to training set. The training set will be clustered using clustermethod to construct vocabulary.

    :param descriptors: Descriptors to add to training set. Each row of  ``descriptors``                                                 matrix is a one descriptor.

.. index:: BOWTrainer::getDescriptors

cv::BOWTrainer::getDescriptors
------------------------------
.. cfunction:: const vector<Mat>\& BOWTrainer::getDescriptors() const

    Returns training set of descriptors.

.. index:: BOWTrainer::descripotorsCount

cv::BOWTrainer::descripotorsCount
---------------------------------
.. cfunction:: const vector<Mat>\& BOWTrainer::descripotorsCount() const

    Returns count of all descriptors stored in the training set.

.. index:: BOWTrainer::cluster

cv::BOWTrainer::cluster
-----------------------
.. cfunction:: Mat BOWTrainer::cluster() const

    Cluster train descriptors. Vocabulary consists from cluster centers. So this method
returns vocabulary. In first method variant the stored in object train descriptors will be
clustered, in second variant -- input descriptors will be clustered.

.. cfunction:: Mat BOWTrainer::cluster( const Mat\& descriptors ) const

    :param descriptors: Descriptors to cluster. Each row of  ``descriptors``                                                 matrix is a one descriptor. Descriptors will not be added
                                                to the inner train descriptor set.

.. index:: BOWKMeansTrainer

.. _BOWKMeansTrainer:

BOWKMeansTrainer
----------------
.. ctype:: BOWKMeansTrainer

:func:`kmeans` based class to train visual vocabulary using the ''bag of visual words'' approach. ::

    class BOWKMeansTrainer : public BOWTrainer
    {
    public:
        BOWKMeansTrainer( int clusterCount, const TermCriteria& termcrit=TermCriteria(),
                          int attempts=3, int flags=KMEANS_PP_CENTERS );
        virtual ~BOWKMeansTrainer(){}

        // Returns trained vocabulary (i.e. cluster centers).
        virtual Mat cluster() const;
        virtual Mat cluster( const Mat& descriptors ) const;

    protected:
        ...
    };
..

To gain an understanding of constructor parameters see
:func:`kmeans` function
arguments.

.. index:: BOWImgDescriptorExtractor

.. _BOWImgDescriptorExtractor:

BOWImgDescriptorExtractor
-------------------------
.. ctype:: BOWImgDescriptorExtractor

Class to compute image descriptor using ''bad of visual words''. In few,
 such computing consists from the following steps:
 1. Compute descriptors for given image and it's keypoints set,
\
2. Find nearest visual words from vocabulary for each keypoint descriptor,
\
3. Image descriptor is a normalized histogram of vocabulary words encountered in the image. I.e.
 ``i`` -bin of the histogram is a frequency of ``i`` -word of vocabulary in the given image. ::

    class BOWImgDescriptorExtractor
    {
    public:
        BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor,
                                   const Ptr<DescriptorMatcher>& dmatcher );
        virtual ~BOWImgDescriptorExtractor(){}

        void setVocabulary( const Mat& vocabulary );
        const Mat& getVocabulary() const;
        void compute( const Mat& image, vector<KeyPoint>& keypoints,
                      Mat& imgDescriptor,
                      vector<vector<int> >* pointIdxsOfClusters=0,
                      Mat* descriptors=0 );
        int descriptorSize() const;
        int descriptorType() const;

    protected:
        ...
    };
..

.. index:: BOWImgDescriptorExtractor::BOWImgDescriptorExtractor

cv::BOWImgDescriptorExtractor::BOWImgDescriptorExtractor
--------------------------------------------------------
.. cfunction:: BOWImgDescriptorExtractor::BOWImgDescriptorExtractor(           const Ptr<DescriptorExtractor>\& dextractor,          const Ptr<DescriptorMatcher>\& dmatcher )

    Constructor.

    :param dextractor: Descriptor extractor that will be used to compute descriptors
                                           for input image and it's keypoints.

    :param dmatcher: Descriptor matcher that will be used to find nearest word of trained vocabulary to
                                         each keupoints descriptor of the image.

.. index:: BOWImgDescriptorExtractor::setVocabulary

cv::BOWImgDescriptorExtractor::setVocabulary
--------------------------------------------
.. cfunction:: void BOWImgDescriptorExtractor::setVocabulary( const Mat\& vocabulary )

    Method to set visual vocabulary.

    :param vocabulary: Vocabulary (can be trained using inheritor of  :func:`BOWTrainer` ).
                                           Each row of vocabulary is a one visual word (cluster center).

.. index:: BOWImgDescriptorExtractor::getVocabulary

cv::BOWImgDescriptorExtractor::getVocabulary
--------------------------------------------
.. cfunction:: const Mat\& BOWImgDescriptorExtractor::getVocabulary() const

    Returns set vocabulary.

.. index:: BOWImgDescriptorExtractor::compute

cv::BOWImgDescriptorExtractor::compute
--------------------------------------
.. cfunction:: void BOWImgDescriptorExtractor::compute( const Mat\& image,           vector<KeyPoint>\& keypoints, Mat\& imgDescriptor,           vector<vector<int> >* pointIdxsOfClusters=0,           Mat* descriptors=0 )

    Compute image descriptor using set visual vocabulary.

    :param image: The image. Image descriptor will be computed for this.

    :param keypoints: Keypoints detected in the input image.

    :param imgDescriptor: This is output, i.e. computed image descriptor.

    :param pointIdxsOfClusters: Indices of keypoints which belong to the cluster, i.e. ``pointIdxsOfClusters[i]``  is keypoint indices which belong
                                to the  ``i-`` cluster (word of vocabulary) (returned if it is not 0.)

    :param descriptors: Descriptors of the image keypoints (returned if it is not 0.)

.. index:: BOWImgDescriptorExtractor::descriptorSize

cv::BOWImgDescriptorExtractor::descriptorSize
---------------------------------------------
.. cfunction:: int BOWImgDescriptorExtractor::descriptorSize() const

    Returns image discriptor size, if vocabulary was set, and 0 otherwise.

.. index:: BOWImgDescriptorExtractor::descriptorType

cv::BOWImgDescriptorExtractor::descriptorType
---------------------------------------------
.. cfunction:: int BOWImgDescriptorExtractor::descriptorType() const

    Returns image descriptor type.

