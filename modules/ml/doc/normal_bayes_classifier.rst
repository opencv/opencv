.. _Bayes Classifier:

Normal Bayes Classifier
=======================

.. highlight:: cpp

This simple classification model assumes that feature vectors from each class are normally distributed (though, not necessarily independently distributed). So, the whole data distribution function is assumed to be a Gaussian mixture, one component per  class. Using the training data the algorithm estimates mean vectors and covariance matrices for every class, and then it uses them for prediction.

.. [Fukunaga90] K. Fukunaga. *Introduction to Statistical Pattern Recognition*. second ed., New York: Academic Press, 1990.

CvNormalBayesClassifier
-----------------------
.. ocv:class:: CvNormalBayesClassifier : public CvStatModel

Bayes classifier for normally distributed data.

CvNormalBayesClassifier::CvNormalBayesClassifier
------------------------------------------------
Default and training constructors.

.. ocv:function:: CvNormalBayesClassifier::CvNormalBayesClassifier()

.. ocv:function:: CvNormalBayesClassifier::CvNormalBayesClassifier( const Mat& trainData, const Mat& responses, const Mat& varIdx=Mat(), const Mat& sampleIdx=Mat() )

.. ocv:function:: CvNormalBayesClassifier::CvNormalBayesClassifier( const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, const CvMat* sampleIdx=0 )

.. ocv:pyfunction:: cv2.NormalBayesClassifier([trainData, responses[, varIdx[, sampleIdx]]]) -> <NormalBayesClassifier object>

The constructors follow conventions of :ocv:func:`CvStatModel::CvStatModel`. See :ocv:func:`CvStatModel::train` for parameters descriptions.

CvNormalBayesClassifier::train
------------------------------
Trains the model.

.. ocv:function:: bool CvNormalBayesClassifier::train( const Mat& trainData, const Mat& responses, const Mat& varIdx = Mat(), const Mat& sampleIdx=Mat(), bool update=false )

.. ocv:function:: bool CvNormalBayesClassifier::train( const CvMat* trainData, const CvMat* responses, const CvMat* varIdx = 0, const CvMat* sampleIdx=0, bool update=false )

.. ocv:pyfunction:: cv2.NormalBayesClassifier.train(trainData, responses[, varIdx[, sampleIdx[, update]]]) -> retval

    :param update: Identifies whether the model should be trained from scratch (``update=false``) or should be updated using the new training data (``update=true``).

The method trains the Normal Bayes classifier. It follows the conventions of the generic :ocv:func:`CvStatModel::train` approach with the following limitations:

* Only ``CV_ROW_SAMPLE`` data layout is supported.
* Input variables are all ordered.
* Output variable is categorical , which means that elements of ``responses`` must be integer numbers, though the vector may have the ``CV_32FC1`` type.
* Missing measurements are not supported.

CvNormalBayesClassifier::predict
--------------------------------
Predicts the response for sample(s).

.. ocv:function:: float CvNormalBayesClassifier::predict(  const Mat& samples,  Mat* results=0 ) const

.. ocv:function:: float CvNormalBayesClassifier::predict( const CvMat* samples, CvMat* results=0 ) const

.. ocv:pyfunction:: cv2.NormalBayesClassifier.predict(samples) -> retval, results

The method estimates the most probable classes for input vectors. Input vectors (one or more) are stored as rows of the matrix ``samples``. In case of multiple input vectors, there should be one output vector ``results``. The predicted class for a single input vector is returned by the method.

The function is parallelized with the TBB library.
