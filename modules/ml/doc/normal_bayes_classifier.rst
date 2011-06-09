.. _Bayes Classifier:

Normal Bayes Classifier
=======================

This is a simple classification model assuming that feature vectors from each class are normally distributed (though, not necessarily independently distributed). So, the whole data distribution function is assumed to be a Gaussian mixture, one component per  class. Using the training data the algorithm estimates mean vectors and covariance matrices for every class, and then it uses them for prediction.

[Fukunaga90] K. Fukunaga. *Introduction to Statistical Pattern Recognition*. second ed., New York: Academic Press, 1990.

.. index:: CvNormalBayesClassifier

CvNormalBayesClassifier
-----------------------
.. c:type:: CvNormalBayesClassifier

Bayes classifier for normally distributed data ::

    class CvNormalBayesClassifier : public CvStatModel
    {
    public:
        CvNormalBayesClassifier();
        virtual ~CvNormalBayesClassifier();

        CvNormalBayesClassifier( const Mat& _train_data, const Mat& _responses,
            const Mat& _var_idx=Mat(), const Mat& _sample_idx=Mat() );

        virtual bool train( const Mat& _train_data, const Mat& _responses,
            const Mat& _var_idx=Mat(), const Mat& _sample_idx=Mat(), bool update=false );

        virtual float predict( const Mat& _samples, Mat* results=0 ) const;
        virtual void clear();

        virtual void save( const char* filename, const char* name=0 );
        virtual void load( const char* filename, const char* name=0 );

        virtual void write( CvFileStorage* storage, const char* name );
        virtual void read( CvFileStorage* storage, CvFileNode* node );
    protected:
        ...
    };


.. index:: CvNormalBayesClassifier::train

.. _CvNormalBayesClassifier::train:

CvNormalBayesClassifier::train
------------------------------
.. cpp:function:: bool CvNormalBayesClassifier::train(  const Mat& _train_data,  const Mat& _responses,                 const Mat& _var_idx =Mat(),  const Mat& _sample_idx=Mat(),  bool update=false )

    Trains the model.

The method trains the Normal Bayes classifier. It follows the conventions of the generic ``train`` "method" with the following limitations: 

* Only ``CV_ROW_SAMPLE`` data layout is supported.
* Input variables are all ordered.
* Output variable is categorical , which means that elements of ``_responses`` must be integer numbers, though the vector may have the ``CV_32FC1`` type.
* Missing measurements are not supported.

In addition, there is an ``update`` flag that identifies whether the model should be trained from scratch ( ``update=false`` ) or should be updated using the new training data ( ``update=true`` ).

.. index:: CvNormalBayesClassifier::predict

.. _CvNormalBayesClassifier::predict:

CvNormalBayesClassifier::predict
--------------------------------
.. cpp:function:: float CvNormalBayesClassifier::predict(  const Mat& samples,  Mat* results=0 ) const

    Predicts the response for sample(s).

The method ``predict`` estimates the most probable classes for input vectors. Input vectors (one or more) are stored as rows of the matrix ``samples`` . In case of multiple input vectors, there should be one output vector ``results`` . The predicted class for a single input vector is returned by the method.

