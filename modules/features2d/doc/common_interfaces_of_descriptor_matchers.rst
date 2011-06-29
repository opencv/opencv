Common Interfaces of Descriptor Matchers
========================================

.. highlight:: cpp

Matchers of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. This section is devoted to matching descriptors
that are represented as vectors in a multidimensional space. All objects that implement ``vector``
descriptor matchers inherit the
:ocv:class:`DescriptorMatcher` interface.

DMatch
------
.. ocv:class:: DMatch

Class for matching keypoint descriptors: query descriptor index,
train descriptor index, train image index, and distance between descriptors. ::

    struct DMatch
    {
        DMatch() : queryIdx(-1), trainIdx(-1), imgIdx(-1),
                   distance(std::numeric_limits<float>::max()) {}
        DMatch( int _queryIdx, int _trainIdx, float _distance ) :
                queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1),
                distance(_distance) {}
        DMatch( int _queryIdx, int _trainIdx, int _imgIdx, float _distance ) :
                queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx),
                distance(_distance) {}

        int queryIdx; // query descriptor index
        int trainIdx; // train descriptor index
        int imgIdx;   // train image index

        float distance;

        // less is better
        bool operator<( const DMatch &m ) const;
    };


DescriptorMatcher
-----------------
.. ocv:class:: DescriptorMatcher

Abstract base class for matching keypoint descriptors. It has two groups
of match methods: for matching descriptors of an image with another image or
with an image set. ::

    class DescriptorMatcher
    {
    public:
        virtual ~DescriptorMatcher();

        virtual void add( const vector<Mat>& descriptors );

        const vector<Mat>& getTrainDescriptors() const;
        virtual void clear();
        bool empty() const;
        virtual bool isMaskSupported() const = 0;

        virtual void train();

        /*
         * Group of methods to match descriptors from an image pair.
         */
        void match( const Mat& queryDescriptors, const Mat& trainDescriptors,
                    vector<DMatch>& matches, const Mat& mask=Mat() ) const;
        void knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                       vector<vector<DMatch> >& matches, int k,
                       const Mat& mask=Mat(), bool compactResult=false ) const;
        void radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                          vector<vector<DMatch> >& matches, float maxDistance,
                          const Mat& mask=Mat(), bool compactResult=false ) const;
        /*
         * Group of methods to match descriptors from one image to an image set.
         */
        void match( const Mat& queryDescriptors, vector<DMatch>& matches,
                    const vector<Mat>& masks=vector<Mat>() );
        void knnMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches,
                       int k, const vector<Mat>& masks=vector<Mat>(),
                       bool compactResult=false );
        void radiusMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches,
                          float maxDistance, const vector<Mat>& masks=vector<Mat>(),
                          bool compactResult=false );

        virtual void read( const FileNode& );
        virtual void write( FileStorage& ) const;

        virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

        static Ptr<DescriptorMatcher> create( const string& descriptorMatcherType );

    protected:
        vector<Mat> trainDescCollection;
        ...
    };


DescriptorMatcher::add
--------------------------
Adds descriptors to train a descriptor collection. If the collection ``trainDescCollectionis`` is not empty, the new descriptors are added to existing train descriptors.

.. ocv:function:: void DescriptorMatcher::add( const vector<Mat>& descriptors )

    :param descriptors: Descriptors to add. Each  ``descriptors[i]``  is a set of descriptors from the same train image.


DescriptorMatcher::getTrainDescriptors
------------------------------------------
Returns a constant link to the train descriptor collection ``trainDescCollection`` .

.. ocv:function:: const vector<Mat>& DescriptorMatcher::getTrainDescriptors() const

    



DescriptorMatcher::clear
----------------------------
Clears the train descriptor collection.

.. ocv:function:: void DescriptorMatcher::clear()



DescriptorMatcher::empty
----------------------------
Returns true if there are no train descriptors in the collection.

.. ocv:function:: bool DescriptorMatcher::empty() const



DescriptorMatcher::isMaskSupported
--------------------------------------
Returns true if the descriptor matcher supports masking permissible matches.

.. ocv:function:: bool DescriptorMatcher::isMaskSupported()



DescriptorMatcher::train
----------------------------
Trains a descriptor matcher

.. ocv:function:: void DescriptorMatcher::train()

Trains a descriptor matcher (for example, the flann index). In all methods to match, the method ``train()`` is run every time before matching. Some descriptor matchers (for example, ``BruteForceMatcher``) have an empty implementation of this method. Other matchers really train their inner structures (for example, ``FlannBasedMatcher`` trains ``flann::Index`` ).



DescriptorMatcher::match
----------------------------
Finds the best match for each descriptor from a query set.

.. ocv:function:: void DescriptorMatcher::match( const Mat& queryDescriptors, const Mat& trainDescriptors, vector<DMatch>& matches, const Mat& mask=Mat() ) const

.. ocv:function:: void DescriptorMatcher::match( const Mat& queryDescriptors, vector<DMatch>& matches, const vector<Mat>& masks=vector<Mat>() )

    :param queryDescriptors: Query set of descriptors.

    :param trainDescriptors: Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.

    :param matches: Matches. If a query descriptor is masked out in  ``mask`` , no match is added for this descriptor. So, ``matches``  size may be smaller than the query descriptors count.

    :param mask: Mask specifying permissible matches between an input query and train matrices of descriptors.

    :param masks: Set of masks. Each  ``masks[i]``  specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image ``trainDescCollection[i]``.

In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by ``DescriptorMatcher::add`` is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, ``queryDescriptors[i]`` can be matched with ``trainDescriptors[j]`` only if ``mask.at<uchar>(i,j)`` is non-zero. 



DescriptorMatcher::knnMatch
-------------------------------
Finds the k best matches for each descriptor from a query set.

.. ocv:function:: void DescriptorMatcher::knnMatch( const Mat& queryDescriptors,       const Mat& trainDescriptors,       vector<vector<DMatch> >& matches,       int k, const Mat& mask=Mat(),       bool compactResult=false ) const

.. ocv:function:: void DescriptorMatcher::knnMatch( const Mat& queryDescriptors,           vector<vector<DMatch> >& matches, int k,      const vector<Mat>& masks=vector<Mat>(),       bool compactResult=false )

    :param queryDescriptors: Query set of descriptors.

    :param trainDescriptors: Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.

    :param mask: Mask specifying permissible matches between an input query and train matrices of descriptors.

    :param masks: Set of masks. Each  ``masks[i]``  specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image ``trainDescCollection[i]``.

    :param matches: Matches. Each  ``matches[i]``  is k or less matches for the same query descriptor.

    :param k: Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.

    :param compactResult: Parameter used when the mask (or masks) is not empty. If  ``compactResult``  is false, the  ``matches``  vector has the same size as  ``queryDescriptors``  rows. If  ``compactResult``  is true, the  ``matches``  vector does not contain matches for fully masked-out query descriptors.

These extended variants of :ocv:func:`DescriptorMatcher::match` methods find several best matches for each query descriptor. The matches are returned in the distance increasing order. See :ocv:func:`DescriptorMatcher::match` for the details about query and train descriptors. 



DescriptorMatcher::radiusMatch
----------------------------------
For each query descriptor, finds the training descriptors not farther than the specified distance.

.. ocv:function:: void DescriptorMatcher::radiusMatch( const Mat& queryDescriptors,           const Mat& trainDescriptors,           vector<vector<DMatch> >& matches,           float maxDistance, const Mat& mask=Mat(),           bool compactResult=false ) const

.. ocv:function:: void DescriptorMatcher::radiusMatch( const Mat& queryDescriptors,           vector<vector<DMatch> >& matches,           float maxDistance,      const vector<Mat>& masks=vector<Mat>(),       bool compactResult=false )

    :param queryDescriptors: Query set of descriptors.

    :param trainDescriptors: Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.

    :param mask: Mask specifying permissible matches between an input query and train matrices of descriptors.

    :param masks: Set of masks. Each  ``masks[i]``  specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image ``trainDescCollection[i]``.

    :param matches: Found matches.

    :param compactResult: Parameter used when the mask (or masks) is not empty. If  ``compactResult``  is false, the  ``matches``  vector has the same size as  ``queryDescriptors``  rows. If  ``compactResult``  is true, the  ``matches``  vector does not contain matches for fully masked-out query descriptors.

    :param maxDistance: Threshold for the distance between matched descriptors.
    
For each query descriptor, the methods find such training descriptors that the distance between the query descriptor and the training descriptor is equal or smaller than ``maxDistance``. Found matches are returned in the distance increasing order.



DescriptorMatcher::clone
----------------------------
Clones the matcher.

.. ocv:function:: Ptr<DescriptorMatcher> DescriptorMatcher::clone( bool emptyTrainData ) const

    :param emptyTrainData: If ``emptyTrainData`` is false, the method creates a deep copy of the object, that is, copies both parameters and train data. If ``emptyTrainData`` is true, the method creates an object copy with the current parameters but with empty train data.



DescriptorMatcher::create
-----------------------------
Creates a descriptor matcher of a given type with the default parameters (using default constructor).

.. ocv:function:: Ptr<DescriptorMatcher> DescriptorMatcher::create( const string& descriptorMatcherType )

    :param descriptorMatcherType: Descriptor matcher type. Now the following matcher types are supported:

        * 
            ``BruteForce`` (it uses ``L2`` )
        * 
            ``BruteForce-L1``
        * 
            ``BruteForce-Hamming``
        * 
            ``BruteForce-HammingLUT``
        * 
            ``FlannBased``





BruteForceMatcher
-----------------
.. ocv:class:: BruteForceMatcher

Brute-force descriptor matcher. For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches of descriptor sets. ::

    template<class Distance>
    class BruteForceMatcher : public DescriptorMatcher
    {
    public:
        BruteForceMatcher( Distance d = Distance() );
        virtual ~BruteForceMatcher();

        virtual bool isMaskSupported() const;
        virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;
    protected:
        ...
    }


For efficiency, ``BruteForceMatcher`` is used as a template parameterized with the distance type. For float descriptors, ``L2<float>`` is a common choice. The following distances are supported: ::

    template<typename T>
    struct Accumulator
    {
        typedef T Type;
    };

    template<> struct Accumulator<unsigned char>  { typedef unsigned int Type; };
    template<> struct Accumulator<unsigned short> { typedef unsigned int Type; };
    template<> struct Accumulator<char>   { typedef int Type; };
    template<> struct Accumulator<short>  { typedef int Type; };

    /*
     * Squared Euclidean distance functor
     */
    template<class T>
    struct L2
    {
        typedef T ValueType;
        typedef typename Accumulator<T>::Type ResultType;

        ResultType operator()( const T* a, const T* b, int size ) const;
    };

    /*
     * Manhattan distance (city block distance) functor
     */
    template<class T>
    struct CV_EXPORTS L1
    {
        typedef T ValueType;
        typedef typename Accumulator<T>::Type ResultType;

        ResultType operator()( const T* a, const T* b, int size ) const;
        ...
    };

    /*
     * Hamming distance functor
     */
    struct HammingLUT
    {
        typedef unsigned char ValueType;
        typedef int ResultType;

        ResultType operator()( const unsigned char* a, const unsigned char* b,
                               int size ) const;
        ...
    };

    struct Hamming
    {
        typedef unsigned char ValueType;
        typedef int ResultType;

        ResultType operator()( const unsigned char* a, const unsigned char* b,
                               int size ) const;
        ...
    };






FlannBasedMatcher
-----------------
.. ocv:class:: FlannBasedMatcher

Flann-based descriptor matcher. This matcher trains :ocv:func:`flann::Index` on a train descriptor collection and calls its nearest search methods to find the best matches. So, this matcher may be faster when matching a large train collection than the brute force matcher. ``FlannBasedMatcher`` does not support masking permissible matches of descriptor sets because ``flann::Index`` does not support this. ::

    class FlannBasedMatcher : public DescriptorMatcher
    {
    public:
        FlannBasedMatcher(
          const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams(),
          const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams() );

        virtual void add( const vector<Mat>& descriptors );
        virtual void clear();

        virtual void train();
        virtual bool isMaskSupported() const;

        virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;
    protected:
        ...
    };

..

