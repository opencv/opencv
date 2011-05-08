Object Categorization
=====================

.. highlight:: cpp

This section describes some approaches based on local 2D features and used to categorize objects.

.. index:: BOWTrainer

.. _BOWTrainer:

BOWTrainer
----------
.. c:type:: BOWTrainer

Abstract base class for training the ''bag of visual words'' vocabulary from a set of descriptors.
For details, see, for example, *Visual Categorization with Bags of Keypoints* by Gabriella Csurka, Christopher R. Dance,
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


.. index:: BOWTrainer::add

BOWTrainer::add
-------------------
.. c:function:: void BOWTrainer::add( const Mat\& descriptors )

    Adds descriptors to a training set. The training set ise clustered using ``clustermethod`` to construct the vocabulary.

    :param descriptors: Descriptors to add to a training set. Each row of  the ``descriptors``  matrix is a descriptor.

.. index:: BOWTrainer::getDescriptors

BOWTrainer::getDescriptors
------------------------------
.. c:function:: const vector<Mat>\& BOWTrainer::getDescriptors() const

    Returns a training set of descriptors.

.. index:: BOWTrainer::descripotorsCount

BOWTrainer::descripotorsCount
---------------------------------
.. c:function:: const vector<Mat>\& BOWTrainer::descripotorsCount() const

    Returns the count of all descriptors stored in the training set.

.. index:: BOWTrainer::cluster

BOWTrainer::cluster
-----------------------
.. c:function:: Mat BOWTrainer::cluster() const

    Clusters train descriptors. The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first variant of the method, train descriptors stored in the object are clustered. In the second variant, input descriptors are clustered.

.. c:function:: Mat BOWTrainer::cluster( const Mat\& descriptors ) const

    :param descriptors: Descriptors to cluster. Each row of  the ``descriptors``  matrix is a descriptor. Descriptors are not added to the inner train descriptor set.

.. index:: BOWKMeansTrainer

.. _BOWKMeansTrainer:

BOWKMeansTrainer
----------------
.. c:type:: BOWKMeansTrainer

:ref:`kmeans` -based class to train visual vocabulary using the ''bag of visual words'' approach ::

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


To understand constructor parameters, see
:ref:`kmeans` function
arguments.

.. index:: BOWImgDescriptorExtractor

.. _BOWImgDescriptorExtractor:

BOWImgDescriptorExtractor
-------------------------
.. c:type:: BOWImgDescriptorExtractor

Class to compute an image descriptor using the ''bag of visual words''. Such a computation consists of the following steps:

    #. Compute descriptors for a given image and its keypoints set.
    #. Find the nearest visual words from the vocabulary for each keypoint descriptor.
    #. Image descriptor is a normalized histogram of vocabulary words encountered in the image. This means that the ``i`` -th bin of the histogram is a frequency of ``i`` -th word of the vocabulary in the given image.??this is not a step ::

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


.. index:: BOWImgDescriptorExtractor::BOWImgDescriptorExtractor

BOWImgDescriptorExtractor::BOWImgDescriptorExtractor
--------------------------------------------------------
.. c:function:: BOWImgDescriptorExtractor::BOWImgDescriptorExtractor(           const Ptr<DescriptorExtractor>\& dextractor,          const Ptr<DescriptorMatcher>\& dmatcher )

    Constructs ??.

    :param dextractor: Descriptor extractor that is used to compute descriptors for an input image and its keypoints.

    :param dmatcher: Descriptor matcher that is used to find the nearest word of the trained vocabulary for each keypoint descriptor of the image.

.. index:: BOWImgDescriptorExtractor::setVocabulary

BOWImgDescriptorExtractor::setVocabulary
--------------------------------------------
.. c:function:: void BOWImgDescriptorExtractor::setVocabulary( const Mat\& vocabulary )

    Sets a visual vocabulary.

    :param vocabulary: Vocabulary (can be trained using the inheritor of  :ref:`BOWTrainer` ). Each row of the vocabulary is a visual word (cluster center).

.. index:: BOWImgDescriptorExtractor::getVocabulary

BOWImgDescriptorExtractor::getVocabulary
--------------------------------------------
.. c:function:: const Mat\& BOWImgDescriptorExtractor::getVocabulary() const

    Returns the set vocabulary.

.. index:: BOWImgDescriptorExtractor::compute

BOWImgDescriptorExtractor::compute
--------------------------------------
.. c:function:: void BOWImgDescriptorExtractor::compute( const Mat\& image,           vector<KeyPoint>\& keypoints, Mat\& imgDescriptor,           vector<vector<int> >* pointIdxsOfClusters=0,           Mat* descriptors=0 )

    Computes an image descriptor using the set visual vocabulary.

    :param image: Image. Image descriptor is computed for this.

    :param keypoints: Keypoints detected in the input image.

    :param imgDescriptor: Output computed image descriptor.

    :param pointIdxsOfClusters: Indices of keypoints that belong to the cluster. This means that ``pointIdxsOfClusters[i]``  are keypoint indices that belong to the  ``i`` -th cluster (word of vocabulary) returned if it is non-zero.

    :param descriptors: Descriptors of the image keypoints  that are returned if they are non-zero.

.. index:: BOWImgDescriptorExtractor::descriptorSize

BOWImgDescriptorExtractor::descriptorSize
---------------------------------------------
.. c:function:: int BOWImgDescriptorExtractor::descriptorSize() const

    Returns an image discriptor size if the vocabulary is set. Otherwise, it returns 0.

.. index:: BOWImgDescriptorExtractor::descriptorType

BOWImgDescriptorExtractor::descriptorType
---------------------------------------------
.. c:function:: int BOWImgDescriptorExtractor::descriptorType() const

    Returns an image descriptor type.

