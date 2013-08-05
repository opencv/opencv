Logistic Regression
===================

.. highlight:: cpp

ML implements logistic regression, which is a probabilistic classification technique. Logistic Regression is a binary classification algorithm which is closely related to Support Vector Machines (SVM).
Like SVM, Logistic Regression can be extended to work on multi-class classification problems like digit recognition (i.e. recognizing digitis like 0,1 2, 3,... from the given images).
This version of Logistic Regression supports both binary and multi-class classifications (for multi-class it creates a multiple 2-class classifiers).
In order to train the logistic regression classifier, Batch Gradient Descent and Mini-Batch Gradient Descent algorithms are used (see [BatchDesWiki]_).
Logistic Regression is a discriminative classifier (see [LogRegTomMitch]_ for more details). Logistic Regression is implemented as a C++ class in ``CvLR``.


In Logistic Regression, we try to optimize the training paramater
:math:`\theta`
such that the hypothesis
:math:`0 \leq h_\theta(x) \leq 1` is acheived.
We have
:math:`h_\theta(x) = g(h_\theta(x))`
and
:math:`g(z) = \frac{1}{1+e^{-z}}`
as the logistic or sigmoid function.
The term "Logistic" in Logistic Regression refers to this function.
For given data of a binary classification problem of classes 0 and 1,
one can determine that the given data instance belongs to class 1 if
:math:`h_\theta(x) \geq 0.5`
or class 0 if
:math:`h_\theta(x) < 0.5`
.

In Logistic Regression, choosing the right parameters is of utmost importance for reducing the training error and ensuring high training accuracy.
``CvLR_TrainParams`` is the structure that defines parameters that are required to train a Logistic Regression classifier.
The learning rate is determined by ``CvLR_TrainParams.alpha``. It determines how faster we approach the solution.
It is a positive real number. Optimization algorithms like Batch Gradient Descent and Mini-Batch Gradient Descent are supported in ``CvLR``.
It is important that we mention the number of iterations these optimization algorithms have to run.
The number of iterations are mentioned by ``CvLR_TrainParams.num_iters``.
The number of iterations can be thought as number of steps taken and learning rate specifies if it is a long step or a short step. These two parameters define how fast we arrive at a possible solution.
In order to compensate for overfitting regularization is performed, which can be enabled by setting ``CvLR_TrainParams.regularized`` to a positive integer (greater than zero).
One can specify what kind of regularization has to be performed by setting ``CvLR_TrainParams.norm`` to ``CvLR::REG_L1`` or ``CvLR::REG_L2`` values.
``CvLR`` provides a choice of 2 training methods with Batch Gradient Descent or the Mini-Batch Gradient Descent. To specify this, set ``CvLR_TrainParams.train_method`` to either ``CvLR::BATCH`` or ``CvLR::MINI_BATCH``.
If ``CvLR_TrainParams`` is set to ``CvLR::MINI_BATCH``, the size of the mini batch has to be to a postive integer using ``CvLR_TrainParams.minibatchsize``.

A sample set of training parameters for the Logistic Regression classifier can be initialized as follows:
::
    CvLR_TrainParams params;
    params.alpha = 0.5;
    params.num_iters = 10000;
    params.norm = CvLR::REG_L2;
    params.regularized = 1;
    params.train_method = CvLR::MINI_BATCH;
    params.minibatchsize = 10;

.. [LogRegWiki] http://en.wikipedia.org/wiki/Logistic_regression. Wikipedia article about the Logistic Regression algorithm.

.. [RenMalik2003] Learning a Classification Model for Segmentation. Proc. CVPR, Nice, France (2003).

.. [LogRegTomMitch] http://www.cs.cmu.edu/~tom/NewChapters.html. "Generative and Discriminative Classifiers: Naive Bayes and Logistic Regression" in Machine Learning, Tom Mitchell.
.. [BatchDesWiki] http://en.wikipedia.org/wiki/Gradient_descent_optimization. Wikipedia article about Gradient Descent based optimization.

CvLR_TrainParams
----------------
.. ocv:struct:: CvLR_TrainParams

  Parameters of the Logistic Regression training algorithm. You can initialize the structure using a constructor or declaring the variable and initializing the the individual parameters.

  The training parameters for Logistic Regression:

  .. ocv:member:: double alpha

    The learning rate of the optimization algorithm. The higher the value, faster the rate and vice versa. If the value is too high, the learning algorithm may overshoot the optimal parameters and result in lower training accuracy. If the value is too low, the learning algorithm converges towards the optimal parameters very slowly. The value must a be a positive real number. You can experiment with different values with small increments as in 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, ... and select the learning rate with less training error.

  .. ocv:member:: int num_iters

    The number of iterations required for the learing algorithm (Gradient Descent or Mini Batch Gradient Descent). It has to be a positive integer. You can try different number of iterations like in 100, 1000, 2000, 3000, 5000, 10000, .. so on.

  .. ocv:member:: int norm

    The type of normalization applied. It takes value ``CvLR::L1`` or ``CvLR::L2``.

  .. ocv:member:: int regularized

    It should be set to postive integer (greater than zero) in order to enable regularization.

  .. ocv:member:: int train_method

    The kind of training method used to train the classifier. It should be set to either ``CvLR::BATCH`` or ``CvLR::MINI_BATCH``.

  .. ocv:member:: int minibatchsize

    If the training method is set to CvLR::MINI_BATCH, it has to be set to positive integer. It can range from 1 to number of training samples.


CvLR_TrainParams::CvLR_TrainParams
----------------------------------
The constructors.

.. ocv:function:: CvLR_TrainParams::CvLR_TrainParams()

.. ocv:function:: CvLR_TrainParams::CvLR_TrainParams(double alpha, int num_iters, int norm, int regularized, int train_method, int minbatchsize)

    :param alpha: Specifies the learning rate.

    :param num_iters: Specifies the number of iterations.

    :param norm: Specifies the kind of regularization to be applied. ``CvLR::REG_L1`` or ``CvLR::REG_L2``. To use this, set ``CvLR_TrainParams.regularized`` to a integer greater than zero.

    :param: regularized: To enable or disable regularization. Set to positive integer (greater than zero) to enable and to 0 to disable.

    :param: train_method: Specifies the kind of training method used. It should be set to either ``CvLR::BATCH`` or ``CvLR::MINI_BATCH``. If using ``CvLR::MINI_BATCH``, set ``CvLR_TrainParams.minibatchsize`` to a positive integer.

    :param: minibatchsize: Specifies the number of training samples taken in each step of Mini-Batch Gradient Descent.

By initializing this structure, one can set all the parameters required for Logistic Regression classifier.

CvLR
----
.. ocv:class:: CvLR : public CvStatModel

Implements Logistic Regression classifier.

CvLR::CvLR
----------
The constructors.

.. ocv:function:: CvLR::CvLR()

.. ocv:function:: CvLR::CvLR(const cv::Mat& data, const cv::Mat& labels, const CvLR_TrainParams& params)

    :param data: The data variable of type ``CV_32F``. Each data instance has to be arranged per across different rows.

    :param labels: The data variable of type ``CV_32F``. Each label instance has to be arranged across differnet rows.

    :param params: The training parameters for the classifier of type ``CVLR_TrainParams``.

The constructor with parameters allows to create a Logistic Regression object intialized with given data and trains it.

CvLR::train
-----------
Trains the Logistic Regression classifier and returns true if successful.

.. ocv:function:: bool CvLR::train(const cv::Mat& data, const cv::Mat& labels)

    :param data: The data variable of type ``CV_32F``. Each data instance has to be arranged per across different rows.

    :param labels: The data variable of type ``CV_32F``. Each label instance has to be arranged across differnet rows.


CvLR::predict
-------------
Predicts responses for input samples and returns a float type.

.. ocv:function:: float CvLR::predict(const Mat& data)

    :param data: The data variable should be a row matrix and of type ``CV_32F``.

.. ocv:function:: float CvLR::predict( const Mat& data, Mat& predicted_labels )

    :param data: The input data for the prediction algorithm. The ``data`` variable should be of type ``CV_32F``.

    :param predicted_labels: Predicted labels as a column matrix and of type ``CV_32S``.

The function ``CvLR::predict(const Mat& data)`` returns the label of single data variable. It should be used if data contains only 1 row.


CvLR::get_learnt_mat()
----------------------
This function returns the trained paramters arranged across rows. For a two class classifcation problem, it returns a row matrix.

.. ocv:function:: cv::Mat CvLR::get_learnt_mat()

It returns learnt paramters of the Logistic Regression as a matrix of type ``CV_32F``.
