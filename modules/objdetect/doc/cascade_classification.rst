Cascade Classification
======================

.. highlight:: cpp

FeatureEvaluator
----------------
.. ocv:class:: FeatureEvaluator

Base class for computing feature values in cascade classifiers. ::

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


FeatureEvaluator::read
--------------------------
Reads parameters of features from the ``FileStorage`` node.

.. ocv:function:: bool FeatureEvaluator::read(const FileNode& node)

    :param node: File node from which the feature parameters are read.



FeatureEvaluator::clone
---------------------------
Returns a full copy of the feature evaluator.

.. ocv:function:: Ptr<FeatureEvaluator> FeatureEvaluator::clone() const



FeatureEvaluator::getFeatureType
------------------------------------
Returns the feature type (``HAAR`` or ``LBP`` for now).

.. ocv:function:: int FeatureEvaluator::getFeatureType() const


FeatureEvaluator::setImage
------------------------------
Assigns an image to feature evaluator.

.. ocv:function:: bool FeatureEvaluator::setImage(const Mat& img, Size origWinSize)

    :param img: Matrix of the type   ``CV_8UC1``  containing an image where the features are computed.

    :param origWinSize: Size of training images.

The method assigns an image, where the features will be computed, to the feature evaluator.



FeatureEvaluator::setWindow
-------------------------------
Assigns a window in the current image where the features will be computed.

.. ocv:function:: bool FeatureEvaluator::setWindow(Point p)

    :param p: Upper left point of the window where the features are computed. Size of the window is equal to the size of training images.

FeatureEvaluator::calcOrd
-----------------------------
Computes the value of an ordered (numerical) feature.

.. ocv:function:: double FeatureEvaluator::calcOrd(int featureIdx) const

    :param featureIdx: Index of the feature whose value is computed.

The function returns the computed value of an ordered feature.



FeatureEvaluator::calcCat
-----------------------------
Computes the value of a categorical feature.

.. ocv:function:: int FeatureEvaluator::calcCat(int featureIdx) const

    :param featureIdx: Index of the feature whose value is computed.

The function returns the computed label of a categorical feature, which is the value from [0,... (number of categories - 1)].


FeatureEvaluator::create
----------------------------
Constructs the feature evaluator.

.. ocv:function:: static Ptr<FeatureEvaluator> FeatureEvaluator::create(int type)

    :param type: Type of features evaluated by cascade (``HAAR`` or ``LBP`` for now).


CascadeClassifier
-----------------
.. ocv:class:: CascadeClassifier

Cascade classifier class for object detection. ::

    class CascadeClassifier
    {
    public:
            // structure for storing a tree node
        struct CV_EXPORTS DTreeNode
        {
            int featureIdx; // index of the feature on which we perform the split
            float threshold; // split threshold of ordered features only
            int left; // left child index in the tree nodes array
            int right; // right child index in the tree nodes array
        };

        // structure for storing a decision tree
        struct CV_EXPORTS DTree
        {
            int nodeCount; // nodes count
        };

        // structure for storing a cascade stage (BOOST only for now)
        struct CV_EXPORTS Stage
        {
            int first; // first tree index in tree array
            int ntrees; // number of trees
            float threshold; // threshold of stage sum
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




CascadeClassifier::CascadeClassifier
----------------------------------------
Loads a classifier from a file.

.. ocv:function:: CascadeClassifier::CascadeClassifier(const string& filename)

    :param filename: Name of the file from which the classifier is loaded.



CascadeClassifier::empty
----------------------------
Checks whether the classifier has been loaded.

.. ocv:function:: bool CascadeClassifier::empty() const


.. ocv:pyfunction:: cv2.CascadeClassifier.empty() -> retval

CascadeClassifier::load
---------------------------
Loads a classifier from a file.

.. ocv:function:: bool CascadeClassifier::load(const string& filename)

.. ocv:pyfunction:: cv2.CascadeClassifier.load(filename) -> retval

    :param filename: Name of the file from which the classifier is loaded. The file may contain an old HAAR classifier trained by the haartraining application or a new cascade classifier trained by the traincascade application.



CascadeClassifier::read
---------------------------
Reads a classifier from a FileStorage node. 

.. ocv:function:: bool CascadeClassifier::read(const FileNode& node)

.. note:: The file may contain a new cascade classifier (trained traincascade application) only.


CascadeClassifier::detectMultiScale
---------------------------------------
Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

.. ocv:function:: void CascadeClassifier::detectMultiScale( const Mat& image,                            vector<Rect>& objects,                            double scaleFactor=1.1,                            int minNeighbors=3, int flags=0,                            Size minSize=Size())

.. ocv:pyfunction:: cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects
.. ocv:pyfunction:: cv2.CascadeClassifier.detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) -> objects

    :param image: Matrix of the type   ``CV_8U``  containing an image where objects are detected.

    :param objects: Vector of rectangles where each rectangle contains the detected object.

    :param scaleFactor: Parameter specifying how much the image size is reduced at each image scale.

    :param minNeighbors: Parameter specifying how many neighbors each candiate rectangle should have to retain it.

    :param flags: Parameter with the same meaning for an old cascade as in the function ``cvHaarDetectObjects``. It is not used for a new cascade.

    :param minSize: Minimum possible object size. Objects smaller than that are ignored.



CascadeClassifier::setImage
-------------------------------
Sets an image for detection that is called by ``detectMultiScale`` at each image level.

.. ocv:function:: bool CascadeClassifier::setImage( Ptr<FeatureEvaluator>& feval, const Mat& image )

    :param feval: Pointer to the feature evaluator used for computing features.

    :param image: Matrix of the type   ``CV_8UC1``  containing an image where the features are computed.



CascadeClassifier::runAt
----------------------------
Runs the detector at the specified point. Use ``setImage`` to set the image for the detector to work with.

.. ocv:function:: int CascadeClassifier::runAt( Ptr<FeatureEvaluator>& feval, Point pt )

    :param feval: Feature evaluator used for computing features.

    :param pt: Upper left point of the window where the features are computed. Size of the window is equal to the size of training images.

The function returns 1 if the cascade classifier detects an object in the given location.
Otherwise, it returns negated index of the stage at which the candidate has been rejected.



groupRectangles
-------------------
Groups the object candidate rectangles.

.. ocv:function:: void groupRectangles(vector<Rect>& rectList,                     int groupThreshold, double eps=0.2)

.. ocv:pyfunction:: cv2.groupRectangles(rectList, groupThreshold[, eps]) -> None
.. ocv:pyfunction:: cv2.groupRectangles(rectList, groupThreshold[, eps]) -> weights
.. ocv:pyfunction:: cv2.groupRectangles(rectList, groupThreshold, eps, weights, levelWeights) -> None

    :param rectList: Input/output vector of rectangles. Output vector includes retained and grouped rectangles.

    :param groupThreshold: Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.

    :param eps: Relative difference between sides of the rectangles to merge them into a group.

The function is a wrapper for the generic function
:ocv:func:`partition` . It clusters all the input rectangles using the rectangle equivalence criteria that combines rectangles with similar sizes and similar locations. The similarity is defined by ``eps``. When ``eps=0`` , no clustering is done at all. If
:math:`\texttt{eps}\rightarrow +\inf` , all the rectangles are put in one cluster. Then, the small clusters containing less than or equal to ``groupThreshold`` rectangles are rejected. In each other cluster, the average rectangle is computed and put into the output rectangle list.
