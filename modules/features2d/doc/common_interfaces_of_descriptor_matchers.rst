Common Interfaces of Descriptor Matchers
========================================

.. highlight:: cpp


Matchers of keypoint descriptors in OpenCV have wrappers with common interface that enables to switch easily 
between different algorithms solving the same problem. This section is devoted to matching descriptors 
that are represented as vectors in a multidimensional space. All objects that implement ''vector'' 
descriptor matchers inherit 
:func:`DescriptorMatcher`
interface.


.. index:: DMatch

.. _DMatch:

DMatch
------

`id=0.193402930617 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DMatch>`__

.. ctype:: DMatch



Match between two keypoint descriptors: query descriptor index, 
train descriptor index, train image index and distance between descriptors.




::


    
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
    

..


.. index:: DescriptorMatcher

.. _DescriptorMatcher:

DescriptorMatcher
-----------------

`id=0.0185035556985 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher>`__

.. ctype:: DescriptorMatcher



Abstract base class for matching keypoint descriptors. It has two groups 
of match methods: for matching descriptors of one image with other image or
with image set.




::


    
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
         * Group of methods to match descriptors from image pair.
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
         * Group of methods to match descriptors from one image to image set.
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
    

..


.. index:: DescriptorMatcher::add


cv::DescriptorMatcher::add
--------------------------

`id=0.549221986718 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3Aadd>`__


````


.. cfunction:: void add( const vector<Mat>\& descriptors )

    Add descriptors to train descriptor collection. If collection trainDescCollectionis not empty
the new descriptors are added to existing train descriptors.





    
    :param descriptors: Descriptors to add. Each  ``descriptors[i]``  is a set of descriptors 
                            from the same (one) train image. 
    
    
    

.. index:: DescriptorMatcher::getTrainDescriptors


cv::DescriptorMatcher::getTrainDescriptors
------------------------------------------

`id=0.354691082433 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3AgetTrainDescriptors>`__


````


.. cfunction:: const vector<Mat>\& getTrainDescriptors() const

    Returns constant link to the train descriptor collection (i.e. trainDescCollection).




.. index:: DescriptorMatcher::clear


cv::DescriptorMatcher::clear
----------------------------

`id=0.776403699262 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3Aclear>`__




.. cfunction:: void DescriptorMatcher::clear()

    Clear train descriptor collection.




.. index:: DescriptorMatcher::empty


cv::DescriptorMatcher::empty
----------------------------

`id=0.186730120991 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3Aempty>`__




.. cfunction:: bool DescriptorMatcher::empty() const

    Return true if there are not train descriptors in collection.




.. index:: DescriptorMatcher::isMaskSupported


cv::DescriptorMatcher::isMaskSupported
--------------------------------------

`id=0.4880242426 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3AisMaskSupported>`__




.. cfunction:: bool DescriptorMatcher::isMaskSupported()

    Returns true if descriptor matcher supports masking permissible matches.




.. index:: DescriptorMatcher::train


cv::DescriptorMatcher::train
----------------------------

`id=0.708209257367 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3Atrain>`__




.. cfunction:: void DescriptorMatcher::train()

    Train descriptor matcher (e.g. train flann index).  In all methods to match the method train() 
is run every time before matching. Some descriptor matchers (e.g. BruteForceMatcher) have empty 
implementation of this method, other matchers realy train their inner structures (e.g. FlannBasedMatcher 
trains flann::Index)




.. index:: DescriptorMatcher::match


cv::DescriptorMatcher::match
----------------------------

`id=0.803878673329 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3Amatch>`__


````
````
````
````


.. cfunction:: void DescriptorMatcher::match( const Mat\& queryDescriptors,                           const Mat\& trainDescriptors,               vector<DMatch>\& matches,              const Mat\& mask=Mat() ) const

    Find the best match for each descriptor from a query set with train descriptors.
Supposed that the query descriptors are of keypoints detected on the same query image. 
In first variant of this method train descriptors are set as input argument and 
supposed that they are of keypoints detected on the same train image. In second variant 
of the method train descriptors collection that was set using addmethod is used.
Optional mask (or masks) can be set to describe which descriptors can be matched. queryDescriptors[i]can be matched with trainDescriptors[j]only if mask.at<uchar>(i,j)is non-zero.





.. cfunction:: void DescriptorMatcher::match( const Mat\& queryDescriptors,                                   vector<DMatch>\& matches,                  const vector<Mat>\& masks=vector<Mat>() )





    
    :param queryDescriptors: Query set of descriptors. 
    
    
    :param trainDescriptors: Train set of descriptors. This will not be added to train descriptors collection 
                                           stored in class object. 
    
    
    :param matches: Matches. If some query descriptor masked out in  ``mask``  no match will be added for this descriptor.
                                        So  ``matches``  size may be less query descriptors count. 
    
    
    :param mask: Mask specifying permissible matches between input query and train matrices of descriptors. 
    
    
    :param masks: The set of masks. Each  ``masks[i]``  specifies permissible matches between input query descriptors
                      and stored train descriptors from i-th image (i.e.  ``trainDescCollection[i])`` . 
    
    
    

.. index:: DescriptorMatcher::knnMatch


cv::DescriptorMatcher::knnMatch
-------------------------------

`id=0.510078848403 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3AknnMatch>`__


:func:`DescriptorMatcher::match`


.. cfunction:: void DescriptorMatcher::knnMatch( const Mat\& queryDescriptors,       const Mat\& trainDescriptors,       vector<vector<DMatch> >\& matches,       int k, const Mat\& mask=Mat(),       bool compactResult=false ) const

    Find the k best matches for each descriptor from a query set with train descriptors. 
Found k (or less if not possible) matches are returned in distance increasing order. 
Details about query and train descriptors see in .





.. cfunction:: void DescriptorMatcher::knnMatch( const Mat\& queryDescriptors,           vector<vector<DMatch> >\& matches, int k,      const vector<Mat>\& masks=vector<Mat>(),       bool compactResult=false )





    
    :param queryDescriptors, trainDescriptors, mask, masks: See in  :func:`DescriptorMatcher::match` . 
    
    
    :param matches: Mathes. Each  ``matches[i]``  is k or less matches for the same query descriptor. 
    
    
    :param k: Count of best matches will be found per each query descriptor (or less if it's not possible). 
    
    
    :param compactResult: It's used when mask (or masks) is not empty. If  ``compactResult``  is false  ``matches``  vector will have the same size as  ``queryDescriptors``  rows. If  ``compactResult``  
        is true  ``matches``  vector will not contain matches for fully masked out query descriptors. 
    
    
    

.. index:: DescriptorMatcher::radiusMatch


cv::DescriptorMatcher::radiusMatch
----------------------------------

`id=0.763278154174 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3AradiusMatch>`__


:func:`DescriptorMatcher::match`


.. cfunction:: void DescriptorMatcher::radiusMatch( const Mat\& queryDescriptors,           const Mat\& trainDescriptors,           vector<vector<DMatch> >\& matches,           float maxDistance, const Mat\& mask=Mat(),           bool compactResult=false ) const

    Find the best matches for each query descriptor which have distance less than given threshold. 
Found matches are returned in distance increasing order. Details about query and train 
descriptors see in .





.. cfunction:: void DescriptorMatcher::radiusMatch( const Mat\& queryDescriptors,           vector<vector<DMatch> >\& matches,           float maxDistance,      const vector<Mat>\& masks=vector<Mat>(),       bool compactResult=false )





    
    :param queryDescriptors, trainDescriptors, mask, masks: See in  :func:`DescriptorMatcher::match` . 
    
    
    :param matches, compactResult: See in  :func:`DescriptorMatcher::knnMatch` . 
    
    
    :param maxDistance: The threshold to found match distances. 
    
    
    

.. index:: DescriptorMatcher::clone


cv::DescriptorMatcher::clone
----------------------------

`id=0.743679534249 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3Aclone>`__




.. cfunction:: Ptr<DescriptorMatcher> \\DescriptorMatcher::clone( bool emptyTrainData ) const

    Clone the matcher.





    
    :param emptyTrainData: If emptyTrainData is false the method create deep copy of the object, i.e. copies
             both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
             but with empty train data.. 
    
    
    

.. index:: DescriptorMatcher::create


cv::DescriptorMatcher::create
-----------------------------

`id=0.681869512138 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/DescriptorMatcher%3A%3Acreate>`__


:func:`DescriptorMatcher`


.. cfunction:: Ptr<DescriptorMatcher> DescriptorMatcher::create( const string\& descriptorMatcherType )

    Descriptor matcher factory that creates of 
given type with default parameters (rather using default constructor).





    
    :param descriptorMatcherType: Descriptor matcher type. 
    
    
    
Now the following matcher types are supported: 
``"BruteForce"``
(it uses 
``L2``
), 
``"BruteForce-L1"``
, 
``"BruteForce-Hamming"``
, 
``"BruteForce-HammingLUT"``
, 
``"FlannBased"``
.


.. index:: BruteForceMatcher

.. _BruteForceMatcher:

BruteForceMatcher
-----------------

`id=0.47821275438 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/BruteForceMatcher>`__

.. ctype:: BruteForceMatcher



Brute-force descriptor matcher. For each descriptor in the first set, this matcher finds the closest
descriptor in the second set by trying each one. This descriptor matcher supports masking 
permissible matches between descriptor sets.




::


    
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
    

..

For efficiency, BruteForceMatcher is templated on the distance metric.
For float descriptors, a common choice would be 
``L2<float>``
. Class of supported distances are:




::


    
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
     * Hamming distance (city block distance) functor
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
    

..


.. index:: FlannBasedMatcher

.. _FlannBasedMatcher:

FlannBasedMatcher
-----------------

`id=0.721140850904 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/features2d/FlannBasedMatcher>`__

.. ctype:: FlannBasedMatcher



Flann based descriptor matcher. This matcher trains 
:func:`flann::Index`
on 
train descriptor collection and calls it's nearest search methods to find best matches. 
So this matcher may be faster in cases of matching to large train collection than 
brute force matcher. 
``FlannBasedMatcher``
does not support masking permissible 
matches between descriptor sets, because 
:func:`flann::Index`
does not 
support this.




::


    
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

