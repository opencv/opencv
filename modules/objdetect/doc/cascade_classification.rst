Cascade Classification
======================

.. highlight:: cpp



.. index:: FeatureEvaluator

.. _FeatureEvaluator:

FeatureEvaluator
----------------

`id=0.360131889668 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator>`__

.. ctype:: FeatureEvaluator



Base class for computing feature values in cascade classifiers.




::


    
    class CV_EXPORTS FeatureEvaluator
    {
    public:    
        enum { HAAR = 0, LBP = 1 }; // supported feature types 
        virtual ~FeatureEvaluator(); // destructor
        virtual bool read(const FileNode& node);
        virtual Ptr<FeatureEvaluator> clone() const;
        virtual int getFeatureType() const;
        
        virtual bool setImage(const Mat& img, Size origWinSize);
        virtual bool setWindow(Point p);
    
        virtual double calcOrd(int featureIdx) const;
        virtual int calcCat(int featureIdx) const;
    
        static Ptr<FeatureEvaluator> create(int type);
    };
    

..


.. index:: FeatureEvaluator::read


cv::FeatureEvaluator::read
--------------------------

`id=0.201865718724 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3Aread>`__




.. cfunction:: bool FeatureEvaluator::read(const FileNode\& node)

    Reads parameters of the features from a FileStorage node.





    
    :param node: File node from which the feature parameters are read. 
    
    
    

.. index:: FeatureEvaluator::clone


cv::FeatureEvaluator::clone
---------------------------

`id=0.296896128079 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3Aclone>`__




.. cfunction:: Ptr<FeatureEvaluator> FeatureEvaluator::clone() const

    Returns a full copy of the feature evaluator.




.. index:: FeatureEvaluator::getFeatureType


cv::FeatureEvaluator::getFeatureType
------------------------------------

`id=0.0597446379803 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3AgetFeatureType>`__




.. cfunction:: int FeatureEvaluator::getFeatureType() const

    Returns the feature type (HAAR or LBP for now).




.. index:: FeatureEvaluator::setImage


cv::FeatureEvaluator::setImage
------------------------------

`id=0.203782054077 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3AsetImage>`__




.. cfunction:: bool FeatureEvaluator::setImage(const Mat\& img, Size origWinSize)

    Sets the image in which to compute the features.





    
    :param img: Matrix of type   ``CV_8UC1``  containing the image in which to compute the features. 
    
    
    :param origWinSize: Size of training images. 
    
    
    

.. index:: FeatureEvaluator::setWindow


cv::FeatureEvaluator::setWindow
-------------------------------

`id=0.403436827824 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3AsetWindow>`__


:func:`CascadeClassifier::runAt`


.. cfunction:: bool FeatureEvaluator::setWindow(Point p)

    Sets window in the current image in which the features will be computed (called by ).





    
    :param p: The upper left point of window in which the features will be computed. Size of the window is equal to size of training images. 
    
    
    

.. index:: FeatureEvaluator::calcOrd


cv::FeatureEvaluator::calcOrd
-----------------------------

`id=0.549815479033 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3AcalcOrd>`__




.. cfunction:: double FeatureEvaluator::calcOrd(int featureIdx) const

    Computes value of an ordered (numerical) feature.





    
    :param featureIdx: Index of feature whose value will be computed. 
    
    
    
Returns computed value of ordered feature.


.. index:: FeatureEvaluator::calcCat


cv::FeatureEvaluator::calcCat
-----------------------------

`id=0.581631081759 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3AcalcCat>`__




.. cfunction:: int FeatureEvaluator::calcCat(int featureIdx) const

    Computes value of a categorical feature.





    
    :param featureIdx: Index of feature whose value will be computed. 
    
    
    
Returns computed label of categorical feature, i.e. value from [0,... (number of categories - 1)].


.. index:: FeatureEvaluator::create


cv::FeatureEvaluator::create
----------------------------

`id=0.415170878436 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/FeatureEvaluator%3A%3Acreate>`__




.. cfunction:: static Ptr<FeatureEvaluator> FeatureEvaluator::create(int type)

    Constructs feature evaluator.





    
    :param type: Type of features evaluated by cascade (HAAR or LBP for now). 
    
    
    

.. index:: CascadeClassifier

.. _CascadeClassifier:

CascadeClassifier
-----------------

`id=0.173067043388 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier>`__

.. ctype:: CascadeClassifier



The cascade classifier class for object detection.




::


    
    class CascadeClassifier
    {
    public:
            // structure for storing tree node
        struct CV_EXPORTS DTreeNode 
        {
            int featureIdx; // feature index on which is a split
            float threshold; // split threshold of ordered features only
            int left; // left child index in the tree nodes array
            int right; // right child index in the tree nodes array
        };
        
        // structure for storing desision tree
        struct CV_EXPORTS DTree 
        {
            int nodeCount; // nodes count
        };
        
        // structure for storing cascade stage (BOOST only for now)
        struct CV_EXPORTS Stage
        {
            int first; // first tree index in tree array
            int ntrees; // number of trees
            float threshold; // treshold of stage sum
        };
        
        enum { BOOST = 0 }; // supported stage types
        
        // mode of detection (see parameter flags in function HaarDetectObjects)
        enum { DO_CANNY_PRUNING = CV_HAAR_DO_CANNY_PRUNING,
               SCALE_IMAGE = CV_HAAR_SCALE_IMAGE,
               FIND_BIGGEST_OBJECT = CV_HAAR_FIND_BIGGEST_OBJECT,
               DO_ROUGH_SEARCH = CV_HAAR_DO_ROUGH_SEARCH }; 
    
        CascadeClassifier(); // default constructor
        CascadeClassifier(const string& filename);
        ~CascadeClassifier(); // destructor
        
        bool empty() const;
        bool load(const string& filename);
        bool read(const FileNode& node);
    
        void detectMultiScale( const Mat& image, vector<Rect>& objects, 
                               double scaleFactor=1.1, int minNeighbors=3, 
                                                       int flags=0, Size minSize=Size());
        
        bool setImage( Ptr<FeatureEvaluator>&, const Mat& );
        int runAt( Ptr<FeatureEvaluator>&, Point );
    
        bool is_stump_based; // true, if the trees are stumps
    
        int stageType; // stage type (BOOST only for now)
        int featureType; // feature type (HAAR or LBP for now)
        int ncategories; // number of categories (for categorical features only) 
        Size origWinSize; // size of training images
        
        vector<Stage> stages; // vector of stages (BOOST for now)
        vector<DTree> classifiers; // vector of decision trees
        vector<DTreeNode> nodes; // vector of tree nodes
        vector<float> leaves; // vector of leaf values
        vector<int> subsets; // subsets of split by categorical feature
    
        Ptr<FeatureEvaluator> feval; // pointer to feature evaluator
        Ptr<CvHaarClassifierCascade> oldCascade; // pointer to old cascade
    };
    

..


.. index:: CascadeClassifier::CascadeClassifier


cv::CascadeClassifier::CascadeClassifier
----------------------------------------

`id=0.751407128029 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier%3A%3ACascadeClassifier>`__




.. cfunction:: CascadeClassifier::CascadeClassifier(const string\& filename)

    Loads the classifier from file.





    
    :param filename: Name of file from which classifier will be load. 
    
    
    

.. index:: CascadeClassifier::empty


cv::CascadeClassifier::empty
----------------------------

`id=0.907371026536 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier%3A%3Aempty>`__




.. cfunction:: bool CascadeClassifier::empty() const

    Checks if the classifier has been loaded or not.




.. index:: CascadeClassifier::load


cv::CascadeClassifier::load
---------------------------

`id=0.689328093704 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier%3A%3Aload>`__




.. cfunction:: bool CascadeClassifier::load(const string\& filename)

    Loads the classifier from file. The previous content is destroyed.





    
    :param filename: Name of file from which classifier will be load. File may contain as old haar classifier (trained by haartraining application) or new cascade classifier (trained traincascade application). 
    
    
    

.. index:: CascadeClassifier::read


cv::CascadeClassifier::read
---------------------------

`id=0.21698114693 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier%3A%3Aread>`__




.. cfunction:: bool CascadeClassifier::read(const FileNode\& node)

    Reads the classifier from a FileStorage node. File may contain a new cascade classifier (trained traincascade application) only.




.. index:: CascadeClassifier::detectMultiScale


cv::CascadeClassifier::detectMultiScale
---------------------------------------

`id=0.0317051017457 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier%3A%3AdetectMultiScale>`__




.. cfunction:: void CascadeClassifier::detectMultiScale( const Mat\& image,                            vector<Rect>\& objects,                            double scaleFactor=1.1,                            int minNeighbors=3, int flags=0,                            Size minSize=Size())

    Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.





    
    :param image: Matrix of type   ``CV_8U``  containing the image in which to detect objects. 
    
    
    :param objects: Vector of rectangles such that each rectangle contains the detected object. 
    
    
    :param scaleFactor: Specifies how much the image size is reduced at each image scale. 
    
    
    :param minNeighbors: Speficifes how many neighbors should each candiate rectangle have to retain it. 
    
    
    :param flags: This parameter is not used for new cascade and have the same meaning for old cascade as in function cvHaarDetectObjects. 
    
    
    :param minSize: The minimum possible object size. Objects smaller than that are ignored. 
    
    
    

.. index:: CascadeClassifier::setImage


cv::CascadeClassifier::setImage
-------------------------------

`id=0.632605719384 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier%3A%3AsetImage>`__




.. cfunction:: bool CascadeClassifier::setImage( Ptr<FeatureEvaluator>\& feval, const Mat\& image )

    Sets the image for detection (called by detectMultiScale at each image level).





    
    :param feval: Pointer to feature evaluator which is used for computing features. 
    
    
    :param image: Matrix of type   ``CV_8UC1``  containing the image in which to compute the features. 
    
    
    

.. index:: CascadeClassifier::runAt


cv::CascadeClassifier::runAt
----------------------------

`id=0.159942031477 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/CascadeClassifier%3A%3ArunAt>`__




.. cfunction:: int CascadeClassifier::runAt( Ptr<FeatureEvaluator>\& feval, Point pt )

    Runs the detector at the specified point (the image that the detector is working with should be set by setImage).





    
    :param feval: Feature evaluator which is used for computing features. 
    
    
    :param pt: The upper left point of window in which the features will be computed. Size of the window is equal to size of training images. 
    
    
    
Returns:
1 - if cascade classifier detects object in the given location.
-si - otherwise. si is an index of stage which first predicted that given window is a background image.


.. index:: groupRectangles


cv::groupRectangles
-------------------

`id=0.226659440065 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/objdetect/groupRectangles>`__




.. cfunction:: void groupRectangles(vector<Rect>\& rectList,                     int groupThreshold, double eps=0.2)

    Groups the object candidate rectangles





    
    :param rectList: The input/output vector of rectangles. On output there will be retained and grouped rectangles 
    
    
    :param groupThreshold: The minimum possible number of rectangles, minus 1, in a group of rectangles to retain it. 
    
    
    :param eps: The relative difference between sides of the rectangles to merge them into a group 
    
    
    
The function is a wrapper for a generic function 
:func:`partition`
. It clusters all the input rectangles using the rectangle equivalence criteria, that combines rectangles that have similar sizes and similar locations (the similarity is defined by 
``eps``
). When 
``eps=0``
, no clustering is done at all. If 
:math:`\texttt{eps}\rightarrow +\inf`
, all the rectangles will be put in one cluster. Then, the small clusters, containing less than or equal to 
``groupThreshold``
rectangles, will be rejected. In each other cluster the average rectangle will be computed and put into the output rectangle list.  
