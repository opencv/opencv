K-Nearest Neighbors
===================

.. highlight:: cpp

The algorithm caches all training samples and predicts the response for a new sample by analyzing a certain number (**K**) of the nearest neighbors of the sample using voting, calculating weighted sum, and so on. The method is sometimes referred to as "learning by example" because for prediction it looks for the feature vector with a known response that is closest to the given vector.

KNearest
----------
.. ocv:class:: KNearest : public StatModel

The class implements K-Nearest Neighbors model as described in the beginning of this section.

.. note::

   * (Python) An example of digit recognition using KNearest can be found at opencv_source/samples/python2/digits.py
   * (Python) An example of grid search digit recognition using KNearest can be found at opencv_source/samples/python2/digits_adjust.py
   * (Python) An example of video digit recognition using KNearest can be found at opencv_source/samples/python2/digits_video.py

KNearest::create
----------------------
Creates the empty model

.. ocv:function:: Ptr<KNearest> KNearest::create(const Params& params=Params())

    :param params: The model parameters: default number of neighbors to use in predict method (in ``KNearest::findNearest`` this number must be passed explicitly) and the flag on whether classification or regression model should be trained.

The static method creates empty KNearest classifier. It should be then trained using ``train`` method (see ``StatModel::train``). Alternatively, you can load boost model from file using ``StatModel::load<KNearest>(filename)``.


KNearest::findNearest
------------------------
Finds the neighbors and predicts responses for input vectors.

.. ocv:function:: float KNearest::findNearest( InputArray samples, int k, OutputArray results, OutputArray neighborResponses=noArray(), OutputArray dist=noArray() ) const

    :param samples: Input samples stored by rows. It is a single-precision floating-point matrix of ``<number_of_samples> * k`` size.

    :param k: Number of used nearest neighbors. Should be greater than 1.

    :param results: Vector with results of prediction (regression or classification) for each input sample. It is a single-precision floating-point vector with ``<number_of_samples>`` elements.

    :param neighborResponses: Optional output values for corresponding neighbors. It is a single-precision floating-point matrix of ``<number_of_samples> * k`` size.

    :param dist: Optional output distances from the input vectors to the corresponding neighbors. It is a single-precision floating-point matrix of ``<number_of_samples> * k`` size.

For each input vector (a row of the matrix ``samples``), the method finds the ``k`` nearest neighbors.  In case of regression, the predicted result is a mean value of the particular vector's neighbor responses. In case of classification, the class is determined by voting.

For each input vector, the neighbors are sorted by their distances to the vector.

In case of C++ interface you can use output pointers to empty matrices and the function will allocate memory itself.

If only a single input vector is passed, all output matrices are optional and the predicted value is returned by the method.

The function is parallelized with the TBB library.

KNearest::getDefaultK
---------------------
Returns the default number of neighbors

.. ocv:function:: int KNearest::getDefaultK() const

The function returns the default number of neighbors that is used in a simpler ``predict`` method, not ``findNearest``.

KNearest::setDefaultK
---------------------
Returns the default number of neighbors

.. ocv:function:: void KNearest::setDefaultK(int k)

The function sets the default number of neighbors that is used in a simpler ``predict`` method, not ``findNearest``.
