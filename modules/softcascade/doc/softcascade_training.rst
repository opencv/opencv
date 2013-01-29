Soft Cascade Training
=======================

.. highlight:: cpp

Soft Cascade Detector Training
--------------------------------------------


SoftCascadeOctave
-----------------
.. ocv:class:: SoftCascadeOctave

Public interface for soft cascade training algorithm

    class CV_EXPORTS SoftCascadeOctave : public Algorithm
    {
    public:

        enum {
            // Direct backward pruning. (Cha Zhang and Paul Viola)
            DBP = 1,
            // Multiple instance pruning. (Cha Zhang and Paul Viola)
            MIP = 2,
            // Originally proposed by L. Bourdev and J. Brandt
            HEURISTIC = 4 };

        virtual ~SoftCascadeOctave();
        static cv::Ptr<SoftCascadeOctave> create(cv::Rect boundingBox, int npositives, int nnegatives, int logScale, int shrinkage);

        virtual bool train(const Dataset* dataset, const FeaturePool* pool, int weaks, int treeDepth) = 0;
        virtual void setRejectThresholds(OutputArray thresholds) = 0;
        virtual void write( cv::FileStorage &fs, const FeaturePool* pool, InputArray thresholds) const = 0;
        virtual void write( CvFileStorage* fs, string name) const = 0;

    };



SoftCascadeOctave::~SoftCascadeOctave
---------------------------------------
Destructor for SoftCascadeOctave.

.. ocv:function:: SoftCascadeOctave::~SoftCascadeOctave()


SoftCascadeOctave::train
------------------------

.. ocv:function:: bool SoftCascadeOctave::train(const Dataset* dataset, const FeaturePool* pool, int weaks, int treeDepth)

    :param dataset an object that allows communicate for training set.

    :param pool an object that presents feature pool.

    :param weaks a number of weak trees should be trained.

    :param treeDepth a depth of resulting weak trees.



SoftCascadeOctave::setRejectThresholds
--------------------------------------

.. ocv:function:: void SoftCascadeOctave::setRejectThresholds(OutputArray thresholds)

    :param thresholds an output array of resulted rejection vector. Have same size as number of trained stages.


SoftCascadeOctave::write
------------------------

.. ocv:function:: write SoftCascadeOctave::train(cv::FileStorage &fs, const FeaturePool* pool, InputArray thresholds) const
.. ocv:function:: write SoftCascadeOctave::train( CvFileStorage* fs, string name) const

    :param fs an output file storage to store trained detector.

    :param pool an object that presents feature pool.

    :param dataset a rejection vector that should be included in detector xml file.

    :param name a name of root node for trained detector.
