.. _Bayes Classifier:

Normal Bayes Classifier
=======================

.. highlight:: cpp

This simple classification model assumes that feature vectors from each class are normally distributed (though, not necessarily independently distributed). So, the whole data distribution function is assumed to be a Gaussian mixture, one component per  class. Using the training data the algorithm estimates mean vectors and covariance matrices for every class, and then it uses them for prediction.

.. [Fukunaga90] K. Fukunaga. *Introduction to Statistical Pattern Recognition*. second ed., New York: Academic Press, 1990.

NormalBayesClassifier
-----------------------
.. ocv:class:: NormalBayesClassifier : public StatModel

Bayes classifier for normally distributed data.

NormalBayesClassifier::create
-----------------------------
Creates empty model

.. ocv:function:: Ptr<NormalBayesClassifier> NormalBayesClassifier::create(const NormalBayesClassifier::Params& params=Params())

    :param params: The model parameters. There is none so far, the structure is used as a placeholder for possible extensions.

Use ``StatModel::train`` to train the model, ``StatModel::train<NormalBayesClassifier>(traindata, params)`` to create and train the model, ``StatModel::load<NormalBayesClassifier>(filename)`` to load the pre-trained model.

NormalBayesClassifier::predictProb
----------------------------------
Predicts the response for sample(s).

.. ocv:function:: float NormalBayesClassifier::predictProb( InputArray inputs, OutputArray outputs, OutputArray outputProbs, int flags=0 ) const

The method estimates the most probable classes for input vectors. Input vectors (one or more) are stored as rows of the matrix ``inputs``. In case of multiple input vectors, there should be one output vector ``outputs``. The predicted class for a single input vector is returned by the method. The vector ``outputProbs`` contains the output probabilities corresponding to each element of ``result``.
