Logistic Regression
===================

.. highlight:: cpp

ML implements logistic regression, which is a probabilistic classification technique. Logistic Regression is a binary classification algorithm which is closely related to Support Vector Machines (SVM).
Like SVM, Logistic Regression can be extended to work on multi-class classification problems like digit recognition (i.e. recognizing digitis like 0,1 2, 3,... from the given images).
This version of Logistic Regression supports both binary and multi-class classifications (for multi-class it creates a multiple 2-class classifiers).
In order to train the logistic regression classifier, Batch Gradient Descent and Mini-Batch Gradient Descent algorithms are used (see [BatchDesWiki]_).
Logistic Regression is a discriminative classifier (see [LogRegTomMitch]_ for more details). Logistic Regression is implemented as a C++ class in ``LogisticRegression``.


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
``LogisticRegressionParams`` is the structure that defines parameters that are required to train a Logistic Regression classifier.
The learning rate is determined by ``LogisticRegressionParams.alpha``. It determines how faster we approach the solution.
It is a positive real number. Optimization algorithms like Batch Gradient Descent and Mini-Batch Gradient Descent are supported in ``LogisticRegression``.
It is important that we mention the number of iterations these optimization algorithms have to run.
The number of iterations are mentioned by ``LogisticRegressionParams.num_iters``.
The number of iterations can be thought as number of steps taken and learning rate specifies if it is a long step or a short step. These two parameters define how fast we arrive at a possible solution.
In order to compensate for overfitting regularization is performed, which can be enabled by setting ``LogisticRegressionParams.regularized`` to a positive integer (greater than zero).
One can specify what kind of regularization has to be performed by setting ``LogisticRegressionParams.norm`` to ``LogisticRegression::REG_L1`` or ``LogisticRegression::REG_L2`` values.
``LogisticRegression`` provides a choice of 2 training methods with Batch Gradient Descent or the Mini-Batch Gradient Descent. To specify this, set ``LogisticRegressionParams.train_method`` to either ``LogisticRegression::BATCH`` or ``LogisticRegression::MINI_BATCH``.
If ``LogisticRegressionParams`` is set to ``LogisticRegression::MINI_BATCH``, the size of the mini batch has to be to a postive integer using ``LogisticRegressionParams.mini_batch_size``.

A sample set of training parameters for the Logistic Regression classifier can be initialized as follows:
::
    LogisticRegressionParams params;
    params.alpha = 0.5;
    params.num_iters = 10000;
    params.norm = LogisticRegression::REG_L2;
    params.regularized = 1;
    params.train_method = LogisticRegression::MINI_BATCH;
    params.mini_batch_size = 10;

.. [LogRegWiki] http://en.wikipedia.org/wiki/Logistic_regression. Wikipedia article about the Logistic Regression algorithm.

.. [RenMalik2003] Learning a Classification Model for Segmentation. Proc. CVPR, Nice, France (2003).

.. [LogRegTomMitch] http://www.cs.cmu.edu/~tom/NewChapters.html. "Generative and Discriminative Classifiers: Naive Bayes and Logistic Regression" in Machine Learning, Tom Mitchell.
.. [BatchDesWiki] http://en.wikipedia.org/wiki/Gradient_descent_optimization. Wikipedia article about Gradient Descent based optimization.

LogisticRegressionParams
------------------------
.. ocv:struct:: LogisticRegressionParams

  Parameters of the Logistic Regression training algorithm. You can initialize the structure using a constructor or declaring the variable and initializing the the individual parameters.

  The training parameters for Logistic Regression:

  .. ocv:member:: double alpha

    The learning rate of the optimization algorithm. The higher the value, faster the rate and vice versa. If the value is too high, the learning algorithm may overshoot the optimal parameters and result in lower training accuracy. If the value is too low, the learning algorithm converges towards the optimal parameters very slowly. The value must a be a positive real number. You can experiment with different values with small increments as in 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, ... and select the learning rate with less training error.

  .. ocv:member:: int num_iters

    The number of iterations required for the learing algorithm (Gradient Descent or Mini Batch Gradient Descent). It has to be a positive integer. You can try different number of iterations like in 100, 1000, 2000, 3000, 5000, 10000, .. so on.

  .. ocv:member:: int norm

    The type of normalization applied. It takes value ``LogisticRegression::L1`` or ``LogisticRegression::L2``.

  .. ocv:member:: int regularized

    It should be set to postive integer (greater than zero) in order to enable regularization.

  .. ocv:member:: int train_method

    The kind of training method used to train the classifier. It should be set to either ``LogisticRegression::BATCH`` or ``LogisticRegression::MINI_BATCH``.

  .. ocv:member:: int mini_batch_size

    If the training method is set to LogisticRegression::MINI_BATCH, it has to be set to positive integer. It can range from 1 to number of training samples.

  .. ocv:member:: cv::TermCriteria term_crit

    Sets termination criteria for training algorithm.

LogisticRegressionParams::LogisticRegressionParams
--------------------------------------------------
The constructors.

.. ocv:function:: LogisticRegressionParams::LogisticRegressionParams()

.. ocv:function:: LogisticRegressionParams::LogisticRegressionParams(double learning_rate, int iters, int train_method, int normlization, int reg, int mini_batch_size)

    :param learning_rate: Specifies the learning rate.

    :param iters: Specifies the number of iterations.

    :param: train_method: Specifies the kind of training method used. It should be set to either ``LogisticRegression::BATCH`` or ``LogisticRegression::MINI_BATCH``. If using ``LogisticRegression::MINI_BATCH``, set ``LogisticRegressionParams.mini_batch_size`` to a positive integer.

    :param normalization: Specifies the kind of regularization to be applied. ``LogisticRegression::REG_L1`` or ``LogisticRegression::REG_L2`` (L1 norm or L2 norm). To use this, set ``LogisticRegressionParams.regularized`` to a integer greater than zero.

    :param: reg: To enable or disable regularization. Set to positive integer (greater than zero) to enable and to 0 to disable.

    :param: mini_batch_size: Specifies the number of training samples taken in each step of Mini-Batch Gradient Descent. Will only be used if using ``LogisticRegression::MINI_BATCH`` training algorithm.


The full constructor initializes corresponding members. The default constructor creates an object with dummy parameters.

::

    LogisticRegressionParams::LogisticRegressionParams()
    {
        term_crit = cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 0.001);
        alpha = 0.001;
        num_iters = 1000;
        norm = LogisticRegression::REG_L2;
        regularized = 1;
        train_method = LogisticRegression::BATCH;
        mini_batch_size = 1;
    }

By initializing this structure, one can set all the parameters required for Logistic Regression classifier.

LogisticRegression
------------------
.. ocv:class:: LogisticRegression

Implements Logistic Regression classifier.

LogisticRegression::LogisticRegression
--------------------------------------
The constructors.

.. ocv:function:: LogisticRegression::LogisticRegression( const LogisticRegressionParams& params)

    :param params: The training parameters for the classifier of type ``LogisticRegressionParams``.

.. ocv:function:: LogisticRegression::LogisticRegression(cv::InputArray data_ip, cv::InputArray labels_ip, const LogisticRegressionParams& params)

    :param data: The data variable of type ``CV_32F``. Each data instance has to be arranged per across different rows.

    :param labels_ip: The data variable of type ``CV_32F``. Each label instance has to be arranged across different rows.

    :param params: The training parameters for the classifier of type  ``LogisticRegressionParams``.

The constructor with parameters allows to create a Logistic Regression object intialized with given data and trains it.

LogisticRegression::train
-------------------------
Trains the Logistic Regression classifier and returns true if successful.

.. ocv:function:: bool LogisticRegression::train(cv::InputArray data_ip, cv::InputArray label_ip)

    :param data_ip: An InputArray variable of type ``CV_32F``. Each data instance has to be arranged per across different rows.

    :param labels_ip: An InputArray variable of type ``CV_32F``. Each label instance has to be arranged across differnet rows.


LogisticRegression::predict
---------------------------
Predicts responses for input samples and returns a float type.

.. ocv:function:: void LogisticRegression::predict( cv::InputArray data, cv::OutputArray predicted_labels ) const

    :param data: The input data for the prediction algorithm. The ``data`` variable should be of type ``CV_32F``.

    :param predicted_labels: Predicted labels as a column matrix and of type ``CV_32S``.


LogisticRegression::get_learnt_thetas
-------------------------------------
This function returns the trained paramters arranged across rows. For a two class classifcation problem, it returns a row matrix.

.. ocv:function:: cv::Mat LogisticRegression::get_learnt_thetas() const;

It returns learnt paramters of the Logistic Regression as a matrix of type ``CV_32F``.

LogisticRegression::read
------------------------
This function reads the trained LogisticRegression clasifier from disk.

.. ocv:function:: void LogisticRegression::read(const FileNode& fn)

LogisticRegression::write
------------------------
This function writes the trained LogisticRegression clasifier to disk.

.. ocv:function:: void LogisticRegression::write(FileStorage& fs) const;
